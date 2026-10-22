"""
05_evaluate.py
Evaluate trained PeerBERT checkpoints on test, test_baseline and (IntelLabs, optional) extended.

Per seed:
- IntelLabs (binary): metrics at threshold 0.5 and at the VALIDATION-calibrated threshold,
  per-generator and per-conference-year breakdowns, paper-level cluster bootstrap CIs.
- Sem-Detect: native 3-class metrics AND a human/ai binary view (rewrite rows excluded;
  score = P(ai) / (P(human)+P(ai))), with the binary threshold calibrated on the
  validation binary view.
- Error analysis (FP/FN tables, per-class / conference-year breakdowns) on test.
Across seeds: mean +/- std. Comparison rows (type "trained classifier") are merged into
outputs/results/<ds>/comparison_table.{csv,json}.

  python scripts/05_evaluate.py --dataset intellabs
  python scripts/05_evaluate.py --dataset semdetect --seeds 42 --n-boot 200
Predictions are cached per seed/split; rerunning skips finished predictions.
"""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=("intellabs", "semdetect"), required=True)
    ap.add_argument("--tag", default="final")
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--splits", nargs="+", default=["test", "test_baseline", "extended"])
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--config", default=None)
    ap.add_argument("--train-config", default=None)
    return ap.parse_args()


def evaluate_split(pred, dataset, val_thr, n_boot, seed, n_words=None):
    """pred: prediction frame for one split -> metrics dict."""
    from src.evaluation.aggregate import by_generator, by_group, by_length_bin
    from src.evaluation.metrics import binary_metrics, multiclass_metrics
    from src.evaluation.predict import binary_view
    from src.evaluation.stats import auc_fn, cluster_bootstrap_ci, macro_f1_fn

    out = {}
    if dataset == "semdetect":
        names = ("human", "ai", "rewrite")
        out["native_3class"] = multiclass_metrics(pred["label"], pred[[f"proba_{c}" for c in names]].values, names)
        b = binary_view(pred)
    else:
        b = pred
    out["binary@0.5"] = binary_metrics(b["label"], b["score"], 0.5)
    out["binary@val_threshold"] = binary_metrics(b["label"], b["score"], val_thr)
    out["binary_note"] = ("human/ai rows only; rewrite excluded; score = P(ai)/(P(human)+P(ai))"
                          if dataset == "semdetect" else "positive class = AI")
    out["by_conf_year@0.5"] = by_group(b, "conf_year")
    if n_words is not None:
        out["by_length_tercile@0.5"] = by_length_bin(b, b["row_id"].map(n_words).values)
    if dataset == "intellabs":
        out["by_generator@0.5"] = by_generator(b)
    if n_boot:
        out["cluster_bootstrap_ci"] = {
            "roc_auc": cluster_bootstrap_ci(b["label"], b["score"], b["paper_key"], auc_fn, n_boot, seed=seed),
            "macro_f1@0.5": cluster_bootstrap_ci(b["label"], b["score"], b["paper_key"], macro_f1_fn(0.5),
                                                 n_boot, seed=seed),
        }
    return out


def main():
    args = parse()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import pandas as pd

    from src.data.common import dataset_dir_name, load_config, load_json, manifest_dir, resolve_path, save_json
    from src.evaluation.aggregate import (
        comparison_row, merge_comparison_rows, save_comparison_table, seed_summary,
    )
    from src.evaluation.error_analysis import save_error_analysis, text_lookup
    from src.evaluation.metrics import calibrate_threshold
    from src.evaluation.predict import binary_view, load_predictions, prediction_frame, save_predictions
    from src.models.trainer import CLASS_NAMES, load_train_config, model_root, predict_proba

    cfg = load_config(args.config)
    tcfg = load_train_config(args.train_config)
    ds = args.dataset
    names = CLASS_NAMES[ds]
    seeds = args.seeds or tcfg["training"]["seeds"]
    mdir = manifest_dir(cfg, ds)
    res_root = resolve_path(cfg, "results_dir", "outputs/results") / dataset_dir_name(cfg, ds)
    out_root = res_root / "peerbert" / args.tag
    manifests = {s: pd.read_parquet(mdir / f"{s}.parquet") for s in args.splits if (mdir / f"{s}.parquet").exists()}
    missing = [s for s in args.splits if s not in manifests]
    if missing:
        print(f"[eval] skipping missing manifests: {missing}")

    per_seed, rows = {}, []
    for seed in seeds:
        run_dir = model_root(cfg, ds) / args.tag / f"seed{seed}"
        if not (run_dir / "done.json").exists():
            print(f"[eval] {run_dir} has no finished model; skipping seed {seed}")
            continue
        info = load_json(run_dir / "run_info.json")
        val_pred = pd.read_csv(run_dir / "val_predictions.csv")
        if ds == "semdetect":
            vb = binary_view(val_pred)
            val_thr = calibrate_threshold(vb["label"], vb["score"], tcfg["threshold"]["objective"])
        else:
            val_thr = info["val_threshold"]
        seed_out = {"val_threshold": val_thr, "model_dir": info["best_dir"], "splits": {}}
        for split, man in manifests.items():
            pp = out_root / f"seed{seed}" / f"{split}_predictions.csv"
            if pp.exists():
                pred = load_predictions(pp)
                trunc = load_json(pp.with_suffix(".truncation.json"))
            else:
                proba, trunc = predict_proba(run_dir / "best", man["text"].tolist(), tcfg)
                pred = prediction_frame(man, proba=proba, class_names=names)
                save_predictions(pred, pp)
                save_json(trunc, pp.with_suffix(".truncation.json"))
            n_words = pd.Series(man["text"].str.split().str.len().values, index=man["row_id"].values)
            m = evaluate_split(pred, ds, val_thr, args.n_boot if split != "test_baseline" else 0, seed, n_words)
            m["truncation"] = trunc
            seed_out["splits"][split] = m
            print(f"[eval] seed {seed} {split}: binary@0.5 macro_f1={m['binary@0.5']['macro_f1']} "
                  f"roc_auc={m['binary@0.5']['roc_auc']}"
                  + (f" | 3-class macro_f1={m['native_3class']['macro_f1']}" if "native_3class" in m else ""))
            if split == "test":
                b = binary_view(pred) if ds == "semdetect" else pred
                paths, breakdown = save_error_analysis(b, out_root / f"seed{seed}", "test_binary", text_lookup(man))
                seed_out["error_analysis"] = {"tables": paths, "breakdown": breakdown}
                if ds == "semdetect":
                    p3, bd3 = save_error_analysis(pred, out_root / f"seed{seed}", "test_3class", text_lookup(man))
                    seed_out["error_analysis_3class"] = {"tables": p3, "breakdown": bd3}
            for kind, key in (("0.5", "binary@0.5"), ("val_calibrated", "binary@val_threshold")):
                rows.append(comparison_row(f"PeerBERT[{args.tag}] seed{seed}", "trained classifier", split, kind,
                                           m[key], seed=seed))
        per_seed[seed] = seed_out
        save_json(seed_out, out_root / f"seed{seed}" / "metrics.json")

    if not per_seed:
        print("[eval] nothing evaluated")
        return 1
    summary = {"dataset": ds, "tag": args.tag, "seeds": list(per_seed), "per_split": {}}
    for split in manifests:
        block = {}
        for key in ("binary@0.5", "binary@val_threshold"):
            block[key] = seed_summary([per_seed[s]["splits"][split][key] for s in per_seed])
        if ds == "semdetect":
            block["native_3class"] = seed_summary([per_seed[s]["splits"][split]["native_3class"] for s in per_seed],
                                                  keys=("accuracy", "macro_f1", "weighted_f1", "macro_ovr_roc_auc"))
        summary["per_split"][split] = block
    summary["note"] = "mean +/- std over seeds (ddof=1); claim improvements only with all seeds evaluated"
    save_json(summary, out_root / "metrics_summary.json")
    table = merge_comparison_rows(res_root / "comparison_table.json", rows)
    save_comparison_table(table, res_root)
    print(f"[eval] summary: {out_root / 'metrics_summary.json'}\n[eval] table: {res_root / 'comparison_table.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
