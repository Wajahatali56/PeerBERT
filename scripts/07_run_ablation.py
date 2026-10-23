"""
07_run_ablation.py
Loss / regularisation ablation for PeerBERT. The variant for the final models is SELECTED ON
VALIDATION Macro-F1 ONLY; test numbers are reported next to it for transparency and never used
for selection.

Variants (configs/train.yaml -> ablation.variants):
  unweighted CE | class-weighted CE | label smoothing 0.05 | focal (gamma=2, weighted)
  R-Drop alpha in {0, 0.5, 1.0} (alpha 0 = ce_weighted) | dropout off
  CoCoDet-style cost-sensitive margin loss on a cosine head

Protocol:
  * one seed per variant (ablation.seed), same train/validation/test rows for every variant;
  * a reduced compute budget is allowed (--max-train-rows / --max-val-rows / --epochs) and is
    recorded in ablation.json - disclose it in the paper;
  * selection rule fixed in advance: the pre-declared default variant (ablation.default_variant)
    is kept unless another variant beats it on validation Macro-F1 by at least ablation.min_gain;
  * the test sample (--test-rows whole-paper rows, identical for all variants) is evaluated at the
    validation-calibrated threshold (binary) or argmax (3-class);
  * checkpoints are deleted after evaluation (disk limit); logs and validation predictions stay.

Then train the final models with:  scripts/03_train_intellabs.py --variant auto  (and 04 for Sem-Detect).

  python scripts/07_run_ablation.py --dataset intellabs --max-train-rows 6000 --max-val-rows 2500 --epochs 2
  python scripts/07_run_ablation.py --dataset semdetect --variants ce_weighted rdrop_1.0
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=("intellabs", "semdetect"), required=True)
    ap.add_argument("--variants", nargs="+", default=None)
    ap.add_argument("--max-train-rows", type=int, default=None, help="whole-paper compute cap (disclose it)")
    ap.add_argument("--max-val-rows", type=int, default=None, help="whole-paper cap on validation rows")
    ap.add_argument("--test-rows", type=int, default=4000, help="whole-paper test sample, 0 = skip test reporting")
    ap.add_argument("--min-gain", type=float, default=None, help="override ablation.min_gain")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--keep-checkpoints", action="store_true")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--model", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--train-config", default=None)
    return ap.parse_args()


def select_variant(rows, default_variant, min_gain):
    """Fixed-in-advance rule. rows: dicts with 'variant' and 'val_macro_f1'. Returns (name, explanation)."""
    best = max(rows, key=lambda r: (r["val_macro_f1"], -rows.index(r)))
    default = next((r for r in rows if r["variant"] == default_variant), None)
    if default is None:
        return best["variant"], f"default variant '{default_variant}' was not run: highest validation Macro-F1"
    gain = best["val_macro_f1"] - default["val_macro_f1"]
    if best["variant"] == default_variant or gain < min_gain:
        return default_variant, (f"kept the pre-declared default '{default_variant}': best variant "
                                 f"'{best['variant']}' beats it by {gain:.4f} < min_gain {min_gain}")
    return best["variant"], (f"'{best['variant']}' beats the default '{default_variant}' by {gain:.4f} "
                             f">= min_gain {min_gain}")


def main():
    args = parse()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import numpy as np
    import pandas as pd

    from src.data.common import (
        dataset_dir_name, load_config, load_json, manifest_dir, resolve_path, save_json,
    )
    from src.data.splits import cap_by_papers
    from src.evaluation.metrics import binary_metrics, multiclass_metrics
    from src.models.trainer import (
        CLASS_NAMES, load_train_config, model_root, predict_proba, resolve_variant, train_from_manifests,
    )

    ds = args.dataset
    cfg = load_config(args.config)
    base = load_train_config(args.train_config)
    if args.epochs:
        base["training"]["epochs"] = args.epochs
    if args.model:
        base["model"]["name"] = args.model
    abl = base["ablation"]
    seed = int(abl["seed"])
    all_names = [v["name"] for v in abl["variants"]]
    names = args.variants or all_names
    unknown = set(names) - set(all_names)
    if unknown:
        raise SystemExit(f"unknown variants: {sorted(unknown)}")
    min_gain = args.min_gain if args.min_gain is not None else float(abl.get("min_gain", 0.005))
    names_c = CLASS_NAMES[ds]

    test = None
    if args.test_rows:
        full = pd.read_parquet(manifest_dir(cfg, ds) / "test.parquet")
        test = cap_by_papers(full, args.test_rows, int(cfg["project"]["seed"])).reset_index(drop=True)

    rows = []
    for name in names:
        tcfg, _ = resolve_variant(cfg, ds, base, name)
        train_from_manifests(ds, cfg, tcfg, [seed], f"ablation/{name}", args.max_train_rows,
                             args.max_val_rows, name)
        run_dir = model_root(cfg, ds) / "ablation" / name / f"seed{seed}"
        done = load_json(run_dir / "done.json")
        result_p = run_dir / "ablation_result.json"
        result = load_json(result_p) if result_p.exists() else None
        if result is None or result.get("run_key") != done["run_key"] or result.get("test_rows") != (
                0 if test is None else len(test)):
            result = {"variant": name, "run_key": done["run_key"], "loss": tcfg["loss"]["type"],
                      "rdrop_alpha": tcfg["loss"].get("rdrop_alpha", 0.0),
                      "dropout_on": tcfg["model"].get("dropout_on", True), "head": tcfg["model"].get("head", "linear"),
                      "seed": seed, "val_macro_f1": done["best_val_macro_f1"], "epochs_run": done["epochs_run"],
                      "minutes": done["minutes"], "test_rows": 0 if test is None else len(test),
                      "test_macro_f1": None, "test_roc_auc": None}
            if test is not None and not (run_dir / "best").exists():
                print(f"[ablation] {name}: checkpoint already deleted, test metrics unavailable "
                      f"(rerun with --keep-checkpoints and a fresh folder to recompute)")
            elif test is not None:
                proba, _ = predict_proba(run_dir / "best", test["text"].tolist(), tcfg)
                y = test["label"].values
                if len(names_c) == 2:
                    m = binary_metrics(y, proba[:, 1], done["val_threshold"])
                    result.update(test_macro_f1=m["macro_f1"], test_roc_auc=m["roc_auc"])
                else:
                    m = multiclass_metrics(y, proba, names_c)
                    keep = y <= 1
                    b = binary_metrics(y[keep], proba[keep, 1] / (proba[keep, 0] + proba[keep, 1]).clip(1e-12))
                    result.update(test_macro_f1=m["macro_f1"], test_roc_auc=b["roc_auc"])
            save_json(result, result_p)
        rows.append(result)
        if not args.keep_checkpoints and (run_dir / "best").exists():
            shutil.rmtree(run_dir / "best")                        # disk limit; predictions stay
        print(f"[ablation] {name}: val Macro-F1 {result['val_macro_f1']:.4f} | test Macro-F1 "
              f"{result['test_macro_f1']} | {result['minutes']} min")

    default_variant = abl.get("default_variant")
    selected, rule = select_variant(rows, default_variant, min_gain)
    df = pd.DataFrame(rows).sort_values("val_macro_f1", ascending=False)
    out = resolve_path(cfg, "results_dir", "outputs/results") / dataset_dir_name(cfg, ds)
    out.mkdir(parents=True, exist_ok=True)
    df.drop(columns=["run_key"]).to_csv(out / "ablation.csv", index=False)
    save_json({
        "dataset": ds, "selected_variant": selected, "selection_rule": rule,
        "default_variant": default_variant, "min_gain": min_gain,
        "selection_metric": "validation Macro-F1 (single seed; native 3-class for Sem-Detect)",
        "complete": set(names) == set(all_names),
        "budget": {"seed": seed, "max_train_rows": args.max_train_rows, "max_val_rows": args.max_val_rows,
                   "epochs": base["training"]["epochs"], "max_length": base["model"]["max_length"],
                   "test_sample_rows": 0 if test is None else len(test)},
        "rows": [{k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in r.items() if k != "run_key"}
                 for r in df.to_dict(orient="records")],
        "note": ("single-seed ablation: differences below seed noise are not evidence; test columns are "
                 "reported for transparency and were not used for selection"),
    }, out / "ablation.json")
    print(df.drop(columns=["run_key"]).to_string(index=False))
    print(f"[ablation] selected: {selected} ({rule})\n[ablation] {out / 'ablation.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
