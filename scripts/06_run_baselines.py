"""
06_run_baselines.py
Run the 8 baselines on the test_baseline subset (nested inside the model test set).

- Validation sample (whole papers, common.val_rows) -> threshold calibration ONLY.
- Train sample (whole papers, common.train_rows) -> fits GLTR / MFD heads ONLY.
- Zero-shot / pretrained detectors are never trained; nothing is tuned on test.
- Sem-Detect: human/ai rows only; rewrite rows -> descriptive "fraction flagged as AI".
- One shared token-statistics LM pass feeds LogRank, GLTR, DetectLLM-LRR, MFD and
  Fast-DetectGPT (analytic, same reference/scoring model).
Outputs: outputs/results/<ds>/baselines/<name>/{scores_*.parquet, metrics.json,
predictions_test_baseline.csv} and rows merged into outputs/results/<ds>/comparison_table.*

  python scripts/06_run_baselines.py --dataset intellabs
  python scripts/06_run_baselines.py --dataset semdetect --only logrank radar
Restart-safe: finished detectors are skipped (use --force to recompute).
"""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TOKEN_STATS_DETECTORS = ("logrank", "gltr", "detectllm_lrr", "mfd", "fast_detectgpt")
PROBABILITY_OUTPUT = ("gltr", "mfd", "radar")


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=("intellabs", "semdetect"), required=True)
    ap.add_argument("--only", nargs="+", default=None, help="subset of baselines.yaml run list")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--baselines-config", default=None)
    return ap.parse_args()


def main():
    args = parse()
    import yaml
    bc_path = Path(args.baselines_config) if args.baselines_config else ROOT / "configs" / "baselines.yaml"
    bcfg = yaml.safe_load(bc_path.read_text())
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu if args.gpu is not None else bcfg["common"].get("gpu_id", 0))

    import numpy as np
    import pandas as pd

    from src.baselines import registry
    from src.baselines.statistical import detectllm_lrr, gltr, logrank, mfd
    from src.baselines.statistical.token_stats import compute_token_stats, free, load_causal_lm
    from src.baselines.zero_shot import fast_detectgpt
    from src.data.common import (
        cache_dir, dataset_dir_name, load_config, load_json, manifest_dir, resolve_path, save_json, set_seed,
    )
    from src.data.splits import cap_by_papers
    from src.evaluation.aggregate import comparison_row, merge_comparison_rows, save_comparison_table
    from src.evaluation.metrics import binary_metrics, calibrate_threshold
    from src.evaluation.predict import prediction_frame, save_predictions

    cfg = load_config(args.config)
    ds = args.dataset
    seed = int(cfg["project"]["seed"])
    set_seed(seed)
    com = bcfg["common"]
    names = args.only or bcfg["run"]
    for n in names:
        registry.get(n)

    mdir = manifest_dir(cfg, ds)
    train = pd.read_parquet(mdir / "train.parquet")
    val = pd.read_parquet(mdir / "val.parquet")
    test_b = pd.read_parquet(mdir / "test_baseline.parquet")
    binary = lambda d: d[d["label"].isin([0, 1])].reset_index(drop=True)  # noqa: E731
    sets = {
        "train": cap_by_papers(binary(train), com["train_rows"], seed).reset_index(drop=True),
        "val": cap_by_papers(binary(val), com["val_rows"], seed).reset_index(drop=True),
        "test_baseline": binary(test_b),
    }
    rewrite = test_b[test_b["label"] == 2].reset_index(drop=True)
    if len(rewrite):
        sets["rewrite"] = rewrite
    print(f"[baselines] {ds}: " + ", ".join(f"{k}={len(v)}" for k, v in sets.items()))

    res_root = resolve_path(cfg, "results_dir", "outputs/results") / dataset_dir_name(cfg, ds)
    out_root = res_root / "baselines"
    bcache = cache_dir(cfg, ds) / "baselines"

    def done(name):
        return (out_root / name / "metrics.json").exists() and not args.force

    # ------------------------------------------------ shared token statistics
    stats = {}
    need_stats = [n for n in names if n in TOKEN_STATS_DETECTORS and not done(n)]
    if need_stats:
        ts = bcfg["token_stats"]
        loaded = load_causal_lm(ts["model"], com["dtype"])
        for split, frame in sets.items():
            if split == "train" and not ({"gltr", "mfd"} & set(need_stats)):
                continue
            stats[split] = compute_token_stats(frame, ts["model"], bcache, com["max_tokens"], com["batch_size"],
                                               com["dtype"], tuple(bcfg["gltr"]["rank_buckets"]), loaded=loaded)
        free(loaded[0])

    # ------------------------------------------------ per-detector scores
    def scores_for(name):
        """-> dict split -> score array, extra info."""
        info = {}
        eval_splits = [s for s in sets if s != "train"]
        if name == "logrank":
            return {s: logrank.score(stats[s]) for s in eval_splits}, info
        if name == "detectllm_lrr":
            return {s: detectllm_lrr.score(stats[s]) for s in eval_splits}, {"epsilon": detectllm_lrr.EPS}
        if name == "fast_detectgpt":
            fc = bcfg["fast_detectgpt"]
            if fc["reference_model"] != "same" or fc["scoring_model"] != "same":
                raise NotImplementedError("use score_two_models() for a separate reference model")
            return ({s: fast_detectgpt.score_from_stats(stats[s]) for s in eval_splits},
                    {"reference_model": bcfg["token_stats"]["model"], "scoring_model": bcfg["token_stats"]["model"]})
        if name in ("gltr", "mfd"):
            y = sets["train"]["label"].values
            if name == "gltr":
                clf = gltr.fit(stats["train"], y, seed)
                return {s: gltr.score(clf, stats[s]) for s in eval_splits}, {"train_rows": len(y)}
            mc = bcfg["mfd"]
            clf = mfd.fit(stats["train"], y, mc["head"], mc.get("mlp_hidden", (16, 16, 16)), seed)
            return ({s: mfd.score(clf, stats[s]) for s in eval_splits},
                    {"train_rows": len(y), "head": mc["head"], "label": "MFD" if mc["head"] == "logreg" else "MFD-M"})
        if name == "binoculars":
            from src.baselines.zero_shot import binoculars
            bc = bcfg["binoculars"]
            o, p = ((bc["paper_observer"], bc["paper_performer"]) if bc["use_paper_pair"]
                    else (bc["observer"], bc["performer"]))
            obs, perf, tok, dev = binoculars.load_pair(o, p, com["dtype"])
            out = {s: binoculars.score(sets[s]["text"].tolist(), obs, perf, tok, dev, com["max_tokens"],
                                       bc.get("batch_size", 2)) for s in eval_splits}
            free(obs, perf)
            return out, {"observer": o, "performer": p, "paper_pair": bool(bc["use_paper_pair"])}
        if name == "radar":
            from src.baselines.neural_detectors import radar
            rc = bcfg["radar"]
            model, tok, dev = radar.load(rc["model"])
            probs = {s: radar.probabilities(model, tok, sets[s]["text"].tolist(), dev, rc["max_length"],
                                            rc.get("batch_size", 16)) for s in eval_splits}
            free(model)
            idx, decision = radar.orient(probs["val"], sets["val"]["label"].values)
            return {s: p[:, idx] for s, p in probs.items()}, {"orientation": decision, "model": rc["model"]}
        if name == "dnagpt":
            from src.baselines.zero_shot import dnagpt
            dc = bcfg["dnagpt"]
            model, tok, dev = load_causal_lm(dc["regenerator"], com["dtype"])
            out, w = {}, {}
            for s in eval_splits:
                out[s], w[s] = dnagpt.run(sets[s], model, tok, dev, dc, bcache / f"dnagpt_{ds}_{s}.jsonl", seed)
            free(model)
            info = {"regenerator": dc["regenerator"], "k": dc["k"], "gamma": dc["gamma"]}
            if dc.get("wscore"):
                info["wscore_auroc_val"] = binary_metrics(sets["val"]["label"], np.nan_to_num(w["val"]))["roc_auc"]
            return out, info
        raise KeyError(name)

    rows = []
    for name in names:
        meta = registry.get(name)
        odir = out_root / name
        if done(name):
            print(f"[baselines] {name}: done (metrics.json exists); skipping")
            m = load_json(odir / "metrics.json")
        else:
            print(f"[baselines] {name} ...")
            scores, info = scores_for(name)
            odir.mkdir(parents=True, exist_ok=True)
            for s, sc in scores.items():
                if not np.isfinite(sc).all():
                    raise RuntimeError(f"{name}: non-finite scores on {s}")
                pd.DataFrame({"row_id": sets[s]["row_id"], "score": sc}).to_parquet(
                    odir / f"scores_{s}.parquet", index=False)
            thr = calibrate_threshold(sets["val"]["label"], scores["val"], com["threshold_objective"])
            m = {"baseline": name, "display": meta["display"], "type": meta["type"], "registry": meta,
                 "info": info, "val_threshold": thr, "n_val": len(sets["val"]),
                 "val@val_threshold": binary_metrics(sets["val"]["label"], scores["val"], thr),
                 "test_baseline@val_threshold": binary_metrics(sets["test_baseline"]["label"],
                                                               scores["test_baseline"], thr)}
            if name in PROBABILITY_OUTPUT:
                m["test_baseline@0.5"] = binary_metrics(sets["test_baseline"]["label"], scores["test_baseline"], 0.5)
            if "rewrite" in scores:
                m["rewrite_fraction_flagged_ai@val_threshold"] = {
                    "n_rewrite": len(scores["rewrite"]),
                    "fraction": float((scores["rewrite"] >= thr).mean()),
                    "note": "descriptive only; rewrite is not part of any classification metric"}
            pred = prediction_frame(sets["test_baseline"], score=scores["test_baseline"], threshold=thr)
            save_predictions(pred, odir / "predictions_test_baseline.csv")
            save_json(m, odir / "metrics.json")
            mt = m["test_baseline@val_threshold"]
            print(f"[baselines] {name}: test_baseline roc_auc={mt['roc_auc']} macro_f1@val_thr={mt['macro_f1']}")
        rows.append(comparison_row(m["display"], m["type"], "test_baseline", "val_calibrated",
                                   m["test_baseline@val_threshold"]))
        if "test_baseline@0.5" in m:
            rows.append(comparison_row(m["display"], m["type"], "test_baseline", "0.5", m["test_baseline@0.5"]))

    table = merge_comparison_rows(res_root / "comparison_table.json", rows)
    save_comparison_table(table, res_root)
    save_json({"dataset": ds, "baselines": names, "sample_sizes": {k: len(v) for k, v in sets.items()},
               "config": bcfg}, out_root / "run_summary.json")
    print(f"[baselines] table: {res_root / 'comparison_table.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
