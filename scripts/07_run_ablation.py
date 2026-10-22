"""
07_run_ablation.py
Loss / regularisation ablations for PeerBERT, selected on VALIDATION Macro-F1 only.

Variants (configs/train.yaml -> ablation.variants):
  unweighted CE | class-weighted CE | label smoothing 0.05 | focal (gamma=2, weighted)
  R-Drop alpha in {0, 0.5, 1.0} (alpha 0 = ce_weighted) | dropout off
Each variant is trained with ablation.seed into outputs/models/<ds>/ablation/<variant>/.
Test-set numbers of the selected variant come from 05_evaluate.py --tag ablation/<variant>;
they are never used for selection. Repeat the chosen configuration with 3 seeds
(03/04) before claiming an improvement.

  python scripts/07_run_ablation.py --dataset intellabs
  python scripts/07_run_ablation.py --dataset semdetect --variants ce_weighted rdrop_1.0 --max-train-rows 8000
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
    ap.add_argument("--variants", nargs="+", default=None)
    ap.add_argument("--max-train-rows", type=int, default=None, help="whole-paper compute cap (disclose it)")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--model", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--train-config", default=None)
    return ap.parse_args()


def main():
    args = parse()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import pandas as pd

    from src.data.common import dataset_dir_name, load_config, resolve_path, save_json
    from src.models.trainer import deep_merge, load_train_config, train_from_manifests

    cfg = load_config(args.config)
    base = load_train_config(args.train_config)
    if args.epochs:
        base["training"]["epochs"] = args.epochs
    if args.model:
        base["model"]["name"] = args.model
    abl = base["ablation"]
    variants = [v for v in abl["variants"] if not args.variants or v["name"] in args.variants]
    unknown = set(args.variants or []) - {v["name"] for v in abl["variants"]}
    if unknown:
        raise SystemExit(f"unknown variants: {sorted(unknown)}")

    rows = []
    for v in variants:
        tcfg = deep_merge(base, {k: v[k] for k in ("loss", "model", "training") if k in v})
        if tcfg["loss"].get("rdrop_alpha", 0) > 0 and not tcfg["model"].get("dropout_on", True):
            raise SystemExit(f"{v['name']}: R-Drop requires dropout")
        s = train_from_manifests(args.dataset, cfg, tcfg, [int(abl["seed"])], f"ablation/{v['name']}",
                                 args.max_train_rows)
        r = s["runs"][0]
        rows.append({"variant": v["name"], "loss": tcfg["loss"]["type"], "rdrop_alpha": tcfg["loss"]["rdrop_alpha"],
                     "dropout_on": tcfg["model"].get("dropout_on", True), "seed": r["seed"],
                     "val_macro_f1": r["best_val_macro_f1"], "epochs_run": r["epochs_run"], "minutes": r["minutes"]})

    df = pd.DataFrame(rows).sort_values("val_macro_f1", ascending=False)
    out = resolve_path(cfg, "results_dir", "outputs/results") / dataset_dir_name(cfg, args.dataset)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "ablation.csv", index=False)
    save_json({"dataset": args.dataset, "selection": "validation Macro-F1 (single seed)",
               "selected_variant": df.iloc[0]["variant"] if len(df) else None,
               "train_row_cap": args.max_train_rows, "rows": df.to_dict(orient="records"),
               "note": "single-seed ablation; confirm the selected configuration with 3 seeds"},
              out / "ablation.json")
    print(df.to_string(index=False))
    print(f"[ablation] {out / 'ablation.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
