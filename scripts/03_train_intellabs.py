"""
03_train_intellabs.py
Train PeerBERT (ModernBERT-base) on the intellabs manifests. Training is blocked unless
outputs/audit/<ds>/audit_status.json exists and is not FAIL (scripts/02_prepare_splits.py).

  python scripts/03_train_intellabs.py                          # all seeds in configs/train.yaml
  python scripts/03_train_intellabs.py --seeds 42               # one seed
  python scripts/03_train_intellabs.py --smoke                  # tiny whole-paper subset, 1 epoch (pipeline check)
Restart-safe: rerun the same command to resume (last_state.pt) or skip finished seeds (done.json).
"""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DATASET = "intellabs"


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--tag", default="final", help="output sub-folder under outputs/models/<ds>/")
    ap.add_argument("--gpu", type=int, default=None, help="GPU index (overrides train.yaml gpu_id)")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--model", default=None, help="override model name/path (e.g. local copy)")
    ap.add_argument("--max-train-rows", type=int, default=None)
    ap.add_argument("--max-val-rows", type=int, default=None)
    ap.add_argument("--smoke", action="store_true", help="400 train / 200 val rows, 1 epoch, tag=smoke")
    ap.add_argument("--config", default=None)
    ap.add_argument("--train-config", default=None)
    return ap.parse_args()


def main():
    args = parse()
    import yaml
    tc_path = Path(args.train_config) if args.train_config else ROOT / "configs" / "train.yaml"
    gpu = args.gpu if args.gpu is not None else (yaml.safe_load(tc_path.read_text())["training"].get("gpu_id", 0))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)   # must be set before torch is imported

    from src.data.common import load_config
    from src.models.trainer import load_train_config, train_from_manifests

    over = {}
    if args.epochs or args.smoke:
        over.setdefault("training", {})["epochs"] = 1 if args.smoke else args.epochs
    if args.model:
        over.setdefault("model", {})["name"] = args.model
    tcfg = load_train_config(tc_path, over)
    cfg = load_config(args.config)
    seeds = args.seeds or tcfg["training"]["seeds"]
    if args.smoke:
        seeds = seeds[:1]
    train_from_manifests(DATASET, cfg, tcfg, seeds, "smoke" if args.smoke else args.tag,
                         400 if args.smoke else args.max_train_rows, 200 if args.smoke else args.max_val_rows)


if __name__ == "__main__":
    main()
