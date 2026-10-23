"""
Command-line entry shared by scripts/03_train_intellabs.py and scripts/04_train_semdetect.py.

CUDA_VISIBLE_DEVICES is set from --gpu (or train.yaml gpu_id) before torch is imported, so each
process trains on exactly one GPU and two processes can use the two Kaggle GPUs side by side.
"""

import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def parse(description):
    ap = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--tag", default="final", help="output sub-folder under outputs/models/<ds>/")
    ap.add_argument("--variant", default=None,
                    help="train.yaml ablation variant to train; 'auto' = the variant selected on validation "
                         "by scripts/07_run_ablation.py; default = base configuration")
    ap.add_argument("--gpu", type=int, default=None, help="GPU index (overrides train.yaml gpu_id)")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--model", default=None, help="override model name/path (e.g. a local copy)")
    ap.add_argument("--max-train-rows", type=int, default=None, help="whole-paper cap (disclose it)")
    ap.add_argument("--max-val-rows", type=int, default=None)
    ap.add_argument("--smoke", action="store_true", help="400 train / 200 val rows, 1 epoch, tag=smoke")
    ap.add_argument("--config", default=None)
    ap.add_argument("--train-config", default=None)
    return ap.parse_args()


def main(dataset, description=""):
    args = parse(description)
    import yaml
    tc_path = Path(args.train_config) if args.train_config else ROOT / "configs" / "train.yaml"
    gpu = args.gpu if args.gpu is not None else yaml.safe_load(tc_path.read_text())["training"].get("gpu_id", 0)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)            # must be set before torch is imported

    from src.data.common import load_config
    from src.models.trainer import load_train_config, resolve_variant, train_from_manifests

    over = {}
    if args.epochs or args.smoke:
        over.setdefault("training", {})["epochs"] = 1 if args.smoke else args.epochs
    if args.model:
        over.setdefault("model", {})["name"] = args.model
    cfg = load_config(args.config)
    tcfg, variant = resolve_variant(cfg, dataset, load_train_config(tc_path, over), args.variant)
    seeds = args.seeds or tcfg["training"]["seeds"]
    if args.smoke:
        seeds = seeds[:1]
    print(f"[train] {dataset}: variant={variant} seeds={list(seeds)} loss={tcfg['loss']['type']} "
          f"rdrop_alpha={tcfg['loss'].get('rdrop_alpha')} head={tcfg['model'].get('head', 'linear')}")
    train_from_manifests(dataset, cfg, tcfg, seeds, "smoke" if args.smoke else args.tag,
                         400 if args.smoke else args.max_train_rows, 200 if args.smoke else args.max_val_rows,
                         variant)
