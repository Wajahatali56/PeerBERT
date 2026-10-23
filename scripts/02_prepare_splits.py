"""
02_prepare_splits.py
Load both datasets, split by paper, run the leakage audit on the FULL partitions,
then apply whole-paper compute caps and write manifests.

Outputs (per dataset folder: intellabs | sem_detect):
  outputs/audit/<ds>/leakage_audit.json, audit_status.json
  outputs/manifests/<ds>/{train,val,test,test_baseline[,extended,extended_baseline]}.parquet
  outputs/manifests/<ds>/exclusions.parquet, summary.json, loader_stats.json

Run from the project root:
  python scripts/02_prepare_splits.py                       # both datasets
  python scripts/02_prepare_splits.py --dataset intellabs
  python scripts/02_prepare_splits.py --audit-only          # audit, no manifests
  python scripts/02_prepare_splits.py --verify-only         # re-check saved manifests (offline)
Exit codes: 0 ok, 2 audit FAIL (no manifests written), 3 manifest verification FAIL.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src.data.common import (  # noqa: E402
    ExclusionLog, audit_dir, config_hash, load_config, load_json, manifest_dir, save_json, set_seed,
)
from src.data.intellabs_loader import load_subset  # noqa: E402
from src.data.leakage_audit import print_summary, run_audit, save_audit  # noqa: E402
from src.data.sem_detect_loader import load_semdetect  # noqa: E402
from src.data.splits import cap_by_papers, check_invariants, split_by_paper  # noqa: E402

DATASETS = ("intellabs", "semdetect")


def binary_rows(df):
    """Rows usable by binary baselines (Sem-Detect: human/ai only, rewrite excluded)."""
    return df["label"].isin([0, 1])


def full_partitions_intellabs(cfg, seed, use_cache):
    icfg = cfg["data"]["intellabs"]
    log, stats = ExclusionLog(), {}
    pool, stats["calibration"], l1 = load_subset(cfg, icfg["train_source_subset"], use_cache=use_cache)
    test, stats["test"], l2 = load_subset(cfg, icfg["test_subset"], use_cache=use_cache)
    log.extend(l1)
    log.extend(l2)
    train, val = split_by_paper(pool, icfg["val_fraction_papers"], seed)
    parts = {"train": train, "val": val, "test": test}
    if icfg.get("use_extended"):
        # Extended is robustness-only and its schemas were never inspected: if it cannot be loaded
        # (unknown columns, a failed download) the main experiment must not be blocked.
        try:
            ext, stats["extended"], l3 = load_subset(cfg, icfg["extended_subset"], use_cache=use_cache)
        except Exception as e:                                    # noqa: BLE001
            stats["extended_skipped"] = f"{type(e).__name__}: {str(e)[:3000]}"
            print(f"[intellabs] WARNING: extended subset skipped ({type(e).__name__}); the main experiment is "
                  f"unaffected.\n  {str(e)[:1200]}")
        else:
            log.extend(l3)
            seen = set(train["paper_key"]) | set(val["paper_key"])
            overlap = ext["paper_key"].isin(seen)
            log.add("intellabs", "extended", ext.loc[overlap, "row_id"], "paper_in_train_or_val",
                    "extended rows whose paper appears in train/val are excluded from extended evaluation")
            stats["extended_rows_excluded_for_train_val_overlap"] = int(overlap.sum())
            parts["extended"] = ext.loc[~overlap].copy()
    caps = {"train": icfg.get("train_max_rows"), "val": icfg.get("val_max_rows"),
            "test": icfg.get("model_eval_max_rows"), "test_baseline": icfg.get("baseline_eval_max_rows"),
            "extended": icfg.get("extended_max_rows"),
            "extended_baseline": icfg.get("extended_baseline_eval_max_rows")}
    return parts, caps, stats, log


def full_partitions_semdetect(cfg, seed, use_cache):
    scfg = cfg["data"]["semdetect"]
    df, stats, log = load_semdetect(cfg, use_cache=use_cache)
    train_pool = df[df["partition"] == "train"]
    test = df[df["partition"] == "test"].copy()          # official test, never deduplicated
    train, val = split_by_paper(train_pool, scfg["val_fraction_papers"], seed)
    caps = {"train": scfg.get("train_max_rows"), "val": scfg.get("val_max_rows"),
            "test": scfg.get("model_eval_max_rows"), "test_baseline": scfg.get("baseline_eval_max_rows")}
    return {"train": train, "val": val, "test": test}, caps, {"all": stats}, log


def apply_caps(parts, caps, seed, dataset, log):
    out = {}
    for name in ("train", "val", "test", "extended"):
        if name not in parts:
            continue
        capped = cap_by_papers(parts[name], caps.get(name), seed)
        dropped = parts[name].loc[~parts[name]["row_id"].isin(capped["row_id"])]
        log.add(dataset, name, dropped["row_id"], "compute_cap", f"whole-paper cap {caps.get(name)}")
        out[name] = capped
    # Baseline subset nested inside the (capped) model test set; for Sem-Detect only
    # human/ai rows count toward the cap (rewrite rows of chosen papers are kept for diagnostics).
    mask = binary_rows(out["test"]) if dataset == "semdetect" else None
    out["test_baseline"] = cap_by_papers(out["test"], caps.get("test_baseline"), seed, count_mask=mask)
    if "extended" in out:             # baselines are also compared on a subset nested inside extended
        out["extended_baseline"] = cap_by_papers(out["extended"], caps.get("extended_baseline"), seed)
    return out


def describe(parts):
    return {n: {"rows": len(df), "papers": int(df["paper_key"].nunique()),
                "label_counts": {str(k): int(v) for k, v in df["label"].value_counts().sort_index().items()}}
            for n, df in parts.items()}


def slim_stats(stats):
    return {k: ({kk: vv for kk, vv in v.items() if kk != "file_stats"} if isinstance(v, dict) else v)
            for k, v in stats.items()}


def prepare(dataset, cfg, audit_only, use_cache):
    seed = int(cfg["project"]["seed"])
    set_seed(seed)
    builder = full_partitions_intellabs if dataset == "intellabs" else full_partitions_semdetect
    parts, caps, stats, log = builder(cfg, seed, use_cache)

    print(f"\n[{dataset}] full partitions: {describe(parts)}")
    report = run_audit(parts, dataset, cfg.get("audit", {}), seed=seed,
                       extra_info={"loader": slim_stats(stats), "full_partitions": describe(parts)})
    adir = audit_dir(cfg, dataset)
    save_audit(report, adir)
    print_summary(report)
    print(f"  report: {adir / 'leakage_audit.json'}")
    if report["status"] == "FAIL":
        print(f"[{dataset}] audit FAILED: manifests NOT written. Fix the data issue; "
              f"never edit the official test split to pass.")
        return 2
    if audit_only:
        return 0

    capped = apply_caps(parts, caps, seed, dataset, log)
    mdir = manifest_dir(cfg, dataset)
    mdir.mkdir(parents=True, exist_ok=True)
    for name, df in capped.items():
        df.reset_index(drop=True).to_parquet(mdir / f"{name}.parquet", index=False)
    log.frame().to_parquet(mdir / "exclusions.parquet", index=False)
    save_json(stats, mdir / "loader_stats.json")
    summary = {
        "dataset": dataset, "seed": seed, "caps": caps,
        "config_hash": config_hash(cfg["data"][dataset]), "audit_status": report["status"],
        "full_partitions": describe(parts), "manifests": describe(capped),
        "exclusions": log.summary(),
        "notes": [
            "Audit ran on full partitions before caps; caps keep whole papers.",
            "test_baseline is nested inside test; all methods are compared on it.",
        ] + (["Sem-Detect test_baseline cap counts human/ai rows only; rewrite rows are "
              "kept for the descriptive 'fraction flagged as AI' diagnostic."]
             if dataset == "semdetect" else []),
    }
    save_json(summary, mdir / "summary.json")
    print(f"[{dataset}] manifests: {describe(capped)}\n  written to {mdir}")
    return verify(dataset, cfg)


def verify(dataset, cfg):
    mdir = manifest_dir(cfg, dataset)
    sp = mdir / "summary.json"
    if not sp.exists():
        print(f"[{dataset}] no manifests at {mdir}")
        return 3
    summary = load_json(sp)
    parts = {p.stem: pd.read_parquet(p) for p in sorted(mdir.glob("*.parquet")) if p.stem != "exclusions"}
    mask = binary_rows if dataset == "semdetect" else None
    fails = check_invariants(parts, summary["caps"], baseline_count_mask=mask)
    status_p = audit_dir(cfg, dataset) / "audit_status.json"
    audit_status = load_json(status_p)["status"] if status_p.exists() else "MISSING"
    if audit_status in ("FAIL", "MISSING"):
        fails.append(f"audit status is {audit_status}")
    print(f"[{dataset}] verify: {'PASS' if not fails else 'FAIL'} "
          f"(audit {audit_status}; manifests {describe(parts)})")
    for f in fails:
        print(f"  - {f}")
    return 0 if not fails else 3


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=DATASETS + ("all",), default="all")
    ap.add_argument("--audit-only", action="store_true", help="run the audit; write no manifests")
    ap.add_argument("--verify-only", action="store_true", help="re-check saved manifests; no network")
    ap.add_argument("--no-cache", action="store_true", help="ignore the parquet cache and reload")
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)
    if args.audit_only and args.verify_only:
        ap.error("--audit-only and --verify-only are exclusive")

    cfg = load_config(args.config)
    codes = []
    for ds in (DATASETS if args.dataset == "all" else (args.dataset,)):
        if args.verify_only:
            codes.append(verify(ds, cfg))
        else:
            codes.append(prepare(ds, cfg, args.audit_only, not args.no_cache))
    return max(codes)


if __name__ == "__main__":
    sys.exit(main())
