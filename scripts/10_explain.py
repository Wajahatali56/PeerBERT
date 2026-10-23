"""
10_explain.py
Stage 9 - explainability for a trained PeerBERT on REAL manifest rows (no SHAP/LIME):
  * attention from an eager-attention copy (last layer, mean over layers, rollout)
  * Integrated Gradients (Captum) on the token embeddings, signed, with convergence delta
  * faithfulness: comprehensiveness / sufficiency at k = 10%, 20%, vs a random ranking
  * attention-vs-IG rank agreement (Spearman)
  * global view: tokens with the largest mean IG attribution per explained class

Examples come from 05_evaluate.py predictions of one seed on one split (default test_baseline):
for every (true class, correct/incorrect) group, up to --n-per-group rows are drawn with a fixed
seed. The explained class is the model's PREDICTED class. Reviews are tokenised exactly as in
training (configs/train.yaml max_length / truncation).
Default seed = the seed with the median VALIDATION Macro-F1.

Outputs: outputs/results/<ds>/xai/<tag>/seed<k>/
  explanations.jsonl (restart-safe, one row per example), summary.json, top_tokens.csv,
  examples.html

  python scripts/10_explain.py --dataset intellabs
  python scripts/10_explain.py --dataset semdetect --n-per-group 10 --n-steps 32
"""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=("intellabs", "semdetect"), required=True)
    ap.add_argument("--tag", default="final")
    ap.add_argument("--seed", type=int, default=None, help="model seed (default: median validation Macro-F1)")
    ap.add_argument("--split", default="test_baseline")
    ap.add_argument("--n-per-group", type=int, default=25)
    ap.add_argument("--n-steps", type=int, default=50, help="Integrated Gradients steps")
    ap.add_argument("--internal-batch-size", type=int, default=4, help="IG steps per backward pass (halved on out-of-memory)")
    ap.add_argument("--ks", type=float, nargs="+", default=[0.1, 0.2])
    ap.add_argument("--min-token-count", type=int, default=5, help="for the global top-token table")
    ap.add_argument("--sanity-n", type=int, default=15,
                    help="reviews for the weight-randomisation sanity check (0 = skip)")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--config", default=None)
    ap.add_argument("--train-config", default=None)
    return ap.parse_args()


def reference_seed(model_dir):
    from src.data.common import load_json
    runs = sorted((load_json(d / "done.json")["best_val_macro_f1"], int(d.name[4:]))
                  for d in model_dir.glob("seed*") if (d / "done.json").exists())
    return runs[(len(runs) - 1) // 2][1] if runs else None


def select_examples(pred, n_per_group, seed):
    """Up to n rows per (true class, correct/incorrect) group, fixed random seed."""
    import pandas as pd
    pred = pred.assign(correct=(pred["pred"] == pred["label"]))
    parts = [g.sample(min(n_per_group, len(g)), random_state=seed)
             for _, g in pred.groupby(["label", "correct"], sort=True)]
    return pd.concat(parts).reset_index(drop=True) if parts else pred.iloc[:0]


def main():
    args = parse()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import numpy as np
    import pandas as pd
    import torch

    from src.data.common import dataset_dir_name, load_config, manifest_dir, resolve_path, save_json
    from src.evaluation.predict import load_predictions
    from src.models.dataset import ReviewDataset
    from src.models.modernbert_classifier import load_trained
    from src.models.trainer import CLASS_NAMES, load_train_config, model_root
    from src.xai.attention import attention_scores, spearman
    from src.xai.faithfulness import faithfulness
    from src.xai.integrated_gradients import display_tokens, ig_attributions_safe, normalise, special_token_mask
    from src.xai.sanity import randomization_check, randomized_copy
    from src.xai.visualize import page

    cfg = load_config(args.config)
    tcfg = load_train_config(args.train_config)
    ds = args.dataset
    names = CLASS_NAMES[ds]
    mroot = model_root(cfg, ds) / args.tag
    seed = args.seed if args.seed is not None else reference_seed(mroot)
    if seed is None:
        raise SystemExit(f"no finished model under {mroot}; run 03/04 first")
    res_root = resolve_path(cfg, "results_dir", "outputs/results") / dataset_dir_name(cfg, ds)
    pred_p = res_root / "peerbert" / args.tag / f"seed{seed}" / f"{args.split}_predictions.csv"
    if not pred_p.exists():
        raise SystemExit(f"{pred_p} not found; run 05_evaluate.py first")
    manifest = pd.read_parquet(manifest_dir(cfg, ds) / f"{args.split}.parquet")
    pred = load_predictions(pred_p)
    chosen = select_examples(pred, args.n_per_group, int(cfg["project"]["seed"]))
    chosen = chosen.merge(manifest[["row_id", "text"]], on="row_id", how="left")
    if chosen["text"].isna().any():
        raise SystemExit("prediction rows not found in the manifest (manifests changed after evaluation?)")

    out = res_root / "xai" / args.tag / f"seed{seed}"
    out.mkdir(parents=True, exist_ok=True)
    jl = out / "explanations.jsonl"
    done = set()
    if jl.exists():
        done = {json.loads(line)["row_id"] for line in jl.read_text(encoding="utf-8").splitlines()}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, tok = load_trained(mroot / f"seed{seed}" / "best", eager_attention=True)
    model.to(device).float().eval()
    m = tcfg["model"]
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    todo = chosen[~chosen["row_id"].isin(done)]
    ds_tok = ReviewDataset(todo["text"].tolist(), None, tok, m["max_length"], m.get("truncation", "head"),
                           m.get("head_tokens", 256))
    print(f"[xai] {ds} seed {seed}: {len(chosen)} examples ({len(done)} already done), device={device}")

    with jl.open("a", encoding="utf-8") as f:
        for i, r in enumerate(todo.itertuples(index=False)):
            ids = ds_tok.input_ids[i]
            special = special_token_mask(tok, ids)
            target = int(r.pred)
            attn = attention_scores(model, ids, device)
            ig, delta, p, p0 = ig_attributions_safe(model, ids, special, target, device, pad_id, args.n_steps,
                                               args.internal_batch_size)
            content = [j for j, s in enumerate(special) if not s]
            faith = faithfulness(model, ids, special, {"integrated_gradients": ig, "attention_last": attn["last_layer"],
                                                       "attention_rollout": attn["rollout"]},
                                 target, device, tuple(args.ks), seed=i)
            att_show = np.where(special, 0.0, attn["rollout"])
            ex = {
                "row_id": r.row_id, "label": int(r.label), "label_name": names[int(r.label)],
                "pred": target, "pred_name": names[target], "target_name": names[target],
                "correct": bool(r.pred == r.label), "p_target": p, "p_baseline": p0, "ig_delta": delta,
                "truncated": bool(ds_tok.raw_lengths[i] > m["max_length"]), "n_tokens": len(ids),
                "tokens": display_tokens(tok, ids), "token_ids": [int(t) for t in ids], "special": special,
                "ig": ig.tolist(), "ig_norm": normalise(np.where(special, 0.0, ig)).tolist(),
                "attention": {k: v.tolist() for k, v in attn.items() if k in ("last_layer", "mean_layers", "rollout")},
                "attention_norm": normalise(att_show).tolist(), "attention_kind": "rollout",
                "attention_pooling": attn["pooling"],
                "spearman_abs_ig_vs_attention": {
                    k: spearman(np.abs(ig)[content], np.asarray(attn[k])[content])
                    for k in ("last_layer", "mean_layers", "rollout")},
                "faithfulness": faith,
            }
            f.write(json.dumps(ex) + "\n")
            f.flush()
            if (i + 1) % 10 == 0:
                print(f"[xai] {i + 1}/{len(todo)}")

    exs = [json.loads(line) for line in jl.read_text(encoding="utf-8").splitlines()]
    exs = [e for e in exs if e["row_id"] in set(chosen["row_id"])]

    def mean_of(get):
        vals = [v for v in (get(e) for e in exs) if v is not None]
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals)} if vals else None

    methods = ["integrated_gradients", "attention_last", "attention_rollout", "random"]
    faith_summary = {meth: {str(k): {metric: mean_of(lambda e, meth=meth, k=k, metric=metric:
                                                     e["faithfulness"][meth][str(k)][metric])
                                     for metric in ("comprehensiveness", "sufficiency")}
                            for k in args.ks} for meth in methods}
    rows = []
    for e in exs:
        for t, a, s in zip(e["tokens"], e["ig"], e["special"]):
            if not s and t.strip():
                rows.append({"explained_class": e["target_name"], "token": t.strip().lower(), "ig": a})
    tok_df = pd.DataFrame(rows)
    if len(tok_df):
        top = (tok_df.groupby(["explained_class", "token"])["ig"].agg(["mean", "count"]).reset_index()
               .query("count >= @args.min_token_count")
               .sort_values(["explained_class", "mean"], ascending=[True, False]))
        top = top.groupby("explained_class").head(30)
    else:
        top = pd.DataFrame(columns=["explained_class", "token", "mean", "count"])
    top.to_csv(out / "top_tokens.csv", index=False)

    sanity = None
    if args.sanity_n and exs:
        rnd = randomized_copy(mroot / f"seed{seed}" / "best", seed=0)
        rows_s = randomization_check(rnd, exs[:args.sanity_n], device, pad_id, args.n_steps, args.internal_batch_size)
        del rnd
        vals = lambda k: [r[k] for r in rows_s if r[k] is not None]  # noqa: E731
        sanity = {"n": len(rows_s), "per_example": rows_s,
                  "spearman_abs": {"mean": float(np.mean(vals("spearman_abs"))) if vals("spearman_abs") else None},
                  "spearman_signed": {"mean": float(np.mean(vals("spearman_signed"))) if vals("spearman_signed") else None},
                  "reading": "Spearman between trained-model and randomly-initialised-model attributions; "
                             "values near 0 mean the attributions depend on the learned weights"}
        print(f"[xai] weight-randomisation check: mean Spearman |IG| = {sanity['spearman_abs']['mean']}")
    summary = {
        "dataset": ds, "tag": args.tag, "seed": seed, "split": args.split, "n_examples": len(exs),
        "sanity_randomization": sanity,
        "groups": ({f"{k[0]}|{'correct' if k[1] else 'incorrect'}": int(v) for k, v in
                    pd.DataFrame(exs).groupby(["label_name", "correct"]).size().items()} if exs else {}),
        "ig": {"n_steps": args.n_steps, "baseline": "pad tokens (special tokens kept)",
               "target": "predicted class probability", "mean_abs_delta": mean_of(lambda e: e["ig_delta"])},
        "attention_pooling": exs[0]["attention_pooling"] if exs else None,
        "spearman_abs_ig_vs_attention": {k: mean_of(lambda e, k=k: e["spearman_abs_ig_vs_attention"][k])
                                         for k in ("last_layer", "mean_layers", "rollout")},
        "faithfulness": faith_summary,
        "truncated_examples": int(sum(e["truncated"] for e in exs)),
        "notes": ["comprehensiveness: higher = more faithful; sufficiency: lower = more faithful",
                  "an explanation is informative only if it beats the random ranking",
                  "attention is not a faithful explanation by itself (Jain & Wallace, 2019)"],
    }
    save_json(summary, out / "summary.json")
    (out / "examples.html").write_text(page(f"PeerBERT explanations - {ds} seed {seed}", exs[:60]), encoding="utf-8")
    print(f"[xai] {out / 'summary.json'}\n[xai] {out / 'examples.html'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
