"""
08_cross_dataset.py
Cross-dataset transfer, Human vs AI only, NO retraining.

  IntelLabs model (binary)  -> Sem-Detect test human/ai rows (rewrite excluded)
  Sem-Detect model (3-class)-> IntelLabs test rows, scored with the binary view
                               P(ai) / (P(human) + P(ai))
Thresholds: 0.5 and the SOURCE validation-calibrated threshold (never the target's).

Known mismatches (documented in the output):
- generators: IntelLabs AI reviews come from 5 named LLMs; Sem-Detect "ai" reviews come
  from its own generation pipeline (models not taken from its metadata here);
- label definitions: Sem-Detect has a separate "rewrite" (LLM-refined human) class that
  has no IntelLabs counterpart and is excluded;
- venues/years and review templates differ between the datasets.

  python scripts/08_cross_dataset.py --source intellabs
  python scripts/08_cross_dataset.py --source semdetect --seeds 42
"""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MISMATCH = [
    "Generators differ: IntelLabs AI = claude/gemini/gpt-4o/llama/qwen; Sem-Detect AI from its own pipeline.",
    "Sem-Detect 'rewrite' (LLM-refined human) has no IntelLabs counterpart and is excluded.",
    "Conference-years and review templates differ between the datasets.",
    "Thresholds come from the SOURCE validation set; nothing is tuned on the target.",
]


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=("intellabs", "semdetect"), required=True)
    ap.add_argument("--tag", default="final")
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--target-split", default="test")
    ap.add_argument("--max-target-rows", type=int, default=None, help="whole-paper cap on the target rows")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--config", default=None)
    ap.add_argument("--train-config", default=None)
    return ap.parse_args()


def main():
    args = parse()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import pandas as pd

    from src.data.common import dataset_dir_name, load_config, load_json, manifest_dir, resolve_path, save_json
    from src.data.splits import cap_by_papers
    from src.evaluation.aggregate import seed_summary
    from src.evaluation.metrics import binary_metrics, calibrate_threshold
    from src.evaluation.predict import binary_view, prediction_frame, save_predictions
    from src.models.trainer import CLASS_NAMES, load_train_config, model_root, predict_proba

    cfg = load_config(args.config)
    tcfg = load_train_config(args.train_config)
    src = args.source
    tgt = "semdetect" if src == "intellabs" else "intellabs"
    seeds = args.seeds or tcfg["training"]["seeds"]
    target = pd.read_parquet(manifest_dir(cfg, tgt) / f"{args.target_split}.parquet")
    target = target[target["label"].isin([0, 1])].reset_index(drop=True)
    if args.max_target_rows:
        target = cap_by_papers(target, args.max_target_rows, int(cfg["project"]["seed"])).reset_index(drop=True)
    out = resolve_path(cfg, "results_dir", "outputs/results") / dataset_dir_name(cfg, tgt) / f"cross_from_{src}"

    per_seed = {}
    for seed in seeds:
        run_dir = model_root(cfg, src) / args.tag / f"seed{seed}"
        if not (run_dir / "done.json").exists():
            print(f"[cross] no finished source model at {run_dir}; skipping")
            continue
        vp = pd.read_csv(run_dir / "val_predictions.csv")
        if src == "semdetect":
            vb = binary_view(vp)
            src_thr = calibrate_threshold(vb["label"], vb["score"], tcfg["threshold"]["objective"])
        else:
            src_thr = load_json(run_dir / "run_info.json")["val_threshold"]
        pp = out / f"seed{seed}_predictions.csv"
        if pp.exists():
            pred = pd.read_csv(pp, dtype={"row_id": str})
        else:
            proba, _ = predict_proba(run_dir / "best", target["text"].tolist(), tcfg)
            pred = prediction_frame(target, proba=proba, class_names=CLASS_NAMES[src])
            if src == "semdetect":
                pred = binary_view(pred)
            save_predictions(pred, pp)
        per_seed[seed] = {"source_val_threshold": src_thr,
                          "binary@0.5": binary_metrics(pred["label"], pred["score"], 0.5),
                          "binary@source_val_threshold": binary_metrics(pred["label"], pred["score"], src_thr)}
        print(f"[cross] {src}->{tgt} seed {seed}: roc_auc={per_seed[seed]['binary@0.5']['roc_auc']} "
              f"macro_f1@src_thr={per_seed[seed]['binary@source_val_threshold']['macro_f1']}")
    if not per_seed:
        return 1
    summary = {"source": src, "target": tgt, "target_split": args.target_split, "n_target": len(target),
               "target_label_counts": target["label"].value_counts().to_dict(), "seeds": list(per_seed),
               "per_seed": per_seed,
               "mean_std@0.5": seed_summary([v["binary@0.5"] for v in per_seed.values()]),
               "mean_std@source_val_threshold": seed_summary([v["binary@source_val_threshold"]
                                                              for v in per_seed.values()]),
               "mismatch_notes": MISMATCH}
    save_json(summary, out / "metrics.json")
    print(f"[cross] {out / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
