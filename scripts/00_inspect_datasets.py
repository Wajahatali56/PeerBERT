
"""
00_inspect_datasets.py
Read-only dataset inspection for AI-generated peer-review detection.

Default: IntelLabs only (calibration, test, extended).
Optional: --include-semdetect

Run from project root:
    python scripts/00_inspect_datasets.py
    python scripts/00_inspect_datasets.py --include-semdetect
    python scripts/00_inspect_datasets.py --skip-intellabs --include-semdetect

Only writes outputs/audit/inspection_report.json.
No training, dataset modification, or model generation.
"""

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
EXPECTED_SUBSETS = ("calibration", "test", "extended")
EXPECTED_LABELS = {0, 1}

FNAME_RE = re.compile(
    r"^(?P<conf>[A-Za-z]+)(?P<year>\d{4})\."
    r"(?P<subset>[A-Za-z]+)\.(?P<gen>.+)\.csv$"
)


def load_cfg():
    config_path = ROOT / "configs" / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration not found: {config_path}")

    with config_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_work_dir(cfg):
    return Path(
        os.environ.get("PRAD_WORK_DIR", cfg.get("paths", {}).get("work_dir", str(ROOT)))
    )


def jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if pd.isna(obj):
        return None
    return obj


def normalize_label(value):
    """Normalize common CSV representations without silently remapping labels."""
    if pd.isna(value):
        return None
    value = str(value).strip()
    if value in {"0", "0.0"}:
        return 0
    if value in {"1", "1.0"}:
        return 1
    return value


def analyze_intel_file(df, filename):
    """Summarize one IntelLabs CSV and flag unexpected data."""
    report = {
        "file": filename,
        "rows": int(len(df)),
        "columns": list(df.columns),
        "missing_required_columns": [
            col for col in ("label", "review") if col not in df.columns
        ],
    }

    if "label" in df.columns:
        labels = df["label"].map(normalize_label)
        report["label_counts"] = {
            str(k): int(v)
            for k, v in labels.value_counts(dropna=False).items()
        }
        observed = set(labels.dropna().tolist())
        report["observed_labels"] = sorted(observed, key=str)
        report["labels_match_expected_binary_values"] = observed.issubset(
            EXPECTED_LABELS
        )
        report["missing_labels"] = int(labels.isna().sum())

    if "review" in df.columns:
        text = df["review"].fillna("").astype(str).str.strip()
        lengths = text.str.len()
        report["empty_review_count"] = int(text.eq("").sum())
        report["review_length_chars"] = {
            "mean": float(lengths.mean()) if len(lengths) else 0.0,
            "median": float(lengths.median()) if len(lengths) else 0.0,
            "max": int(lengths.max()) if len(lengths) else 0,
        }

        if "label" in df.columns:
            temp = pd.DataFrame({"label": df["label"].map(normalize_label),
                                 "length": lengths})
            report["mean_review_length_by_label"] = {
                str(k): float(v)
                for k, v in temp.groupby("label", dropna=False)["length"].mean().items()
            }

    for col in ("PaperId", "review-id", "custom_id"):
        if col in df.columns:
            ids = df[col]
            non_null = ids.dropna()
            report[f"{col}_stats"] = {
                "non_null": int(ids.notna().sum()),
                "unique_non_null": int(non_null.nunique()),
                "duplicate_non_null_rows": int(non_null.duplicated().sum()),
                "dtype": str(ids.dtype),
            }

    if "review-id" in df.columns and "label" in df.columns:
        valid = df[["review-id", "label"]].dropna(subset=["review-id"]).copy()
        valid["label"] = valid["label"].map(normalize_label)
        label_sets = valid.groupby("review-id")["label"].agg(
            lambda values: tuple(sorted(set(values), key=str))
        )
        report["review_id_label_patterns"] = {
            str(k): int(v)
            for k, v in label_sets.value_counts().head(10).items()
        }
        report["review_ids_associated_with_multiple_labels"] = int(
            (label_sets.map(len) > 1).sum()
        )

    return report


def inspect_intellabs(cfg):
    from huggingface_hub import hf_hub_download, list_repo_files

    ic = cfg.get("data", {}).get("intellabs", {})
    repo_id = ic.get("repo_id", "IntelLabs/AI-Peer-Review-Detection-Benchmark")

    print(f"\n=== Inspecting IntelLabs: {repo_id} ===")

    files = [
        f for f in list_repo_files(repo_id, repo_type="dataset")
        if f.lower().endswith(".csv")
    ]

    # Index files by official subset, generator and conference/year.
    indexed = defaultdict(list)
    layout = defaultdict(lambda: defaultdict(int))
    for path in files:
        parts = path.split("/")
        if len(parts) < 3:
            continue

        subset, generator = parts[0], parts[1]
        match = FNAME_RE.match(Path(path).name)
        if not match:
            continue

        info = match.groupdict()
        if info["subset"].lower() != subset.lower():
            continue

        conf_year = f'{info["conf"]}{info["year"]}'
        indexed[(subset, conf_year)].append({
            "path": path,
            "generator": generator,
            "conf_year": conf_year,
        })
        layout[subset][generator] += 1

    report = {
        "repo_id": repo_id,
        "expected_label_mapping": {
            "0": "human-written",
            "1": "AI-generated",
        },
        "label_mapping_source": "Dataset card supplied for this experiment",
        "subset_file_counts": {
            subset: dict(generators)
            for subset, generators in layout.items()
        },
        "subsets_found": sorted(layout.keys()),
        "subsets_missing": [
            s for s in EXPECTED_SUBSETS if s not in layout
        ],
        "files": [],
        "subset_summaries": {},
        "overlap_audits": {},
        "errors": [],
    }

    # Use configured conference-years when supplied; otherwise inspect all.
    inspection_cfg = cfg.get("inspection", {})
    configured_years = inspection_cfg.get("sample_conf_years")
    wanted_years = set(map(str, configured_years)) if configured_years else None

    selected = []
    for (subset, conf_year), entries in sorted(indexed.items()):
        if subset not in EXPECTED_SUBSETS:
            continue
        if wanted_years and conf_year not in wanted_years:
            continue
        selected.extend(entries)

    cache = {}

    def fetch(path):
        if path not in cache:
            local_path = hf_hub_download(
                repo_id=repo_id,
                filename=path,
                repo_type="dataset",
            )
            cache[path] = pd.read_csv(local_path, low_memory=False)
        return cache[path]

    # Inspect each selected file, including all generator folders.
    subset_frames = defaultdict(list)
    human_review_ids = defaultdict(set)
    human_review_texts = defaultdict(set)
    paper_ids = defaultdict(set)

    for entry in selected:
        path = entry["path"]
        subset = path.split("/")[0]
        generator = entry["generator"]
        conf_year = entry["conf_year"]

        try:
            df = fetch(path)
            item = analyze_intel_file(df, path)
            item.update({
                "subset": subset,
                "generator_folder": generator,
                "conference_year": conf_year,
            })
            report["files"].append(item)
            subset_frames[subset].append(df)

            print(
                f"[{subset}] {path}: rows={len(df)}, "
                f"labels={item.get('label_counts', 'missing')}"
            )

            # Track papers and human reviews to detect repeated samples.
            if "PaperId" in df.columns:
                paper_ids[(subset, conf_year)].update(
                    df["PaperId"].dropna().astype(str)
                )

            if "label" in df.columns:
                labels = df["label"].map(normalize_label)
                humans = df.loc[labels.eq(0)]

                if "review-id" in humans.columns:
                    human_review_ids[(subset, conf_year)].update(
                        humans["review-id"].dropna().astype(str)
                    )

                # Text fingerprints help find repeated human reviews even
                # when review IDs are missing or inconsistent.
                if "review" in humans.columns:
                    texts = (
                        humans["review"].dropna().astype(str).str.strip()
                    )
                    human_review_texts[(subset, conf_year)].update(
                        texts[texts.ne("")].tolist()
                    )

        except Exception as exc:
            error = {"file": path, "error": repr(exc)}
            report["errors"].append(error)
            print(f"ERROR reading {path}: {exc!r}")

    # Summarize rows and labels by subset. These totals include repeated
    # records from generator-specific files, so they are raw file totals.
    for subset, frames in subset_frames.items():
        total_rows = sum(len(df) for df in frames)
        label_counts = defaultdict(int)
        missing_review_column = 0
        missing_review_rows = 0

        for df in frames:
            if "label" in df.columns:
                for label, count in df["label"].map(normalize_label).value_counts(
                    dropna=False
                ).items():
                    label_counts[str(label)] += int(count)

            if "review" not in df.columns:
                missing_review_column += 1
            else:
                missing_review_rows += int(
                    df["review"].isna().sum()
                    + df["review"].fillna("").astype(str).str.strip().eq("").sum()
                )

        report["subset_summaries"][subset] = {
            "files_inspected": len(frames),
            "raw_rows_across_files": int(total_rows),
            "label_counts_across_files": dict(label_counts),
            "files_missing_review_column": missing_review_column,
            "missing_or_empty_review_cells": missing_review_rows,
            "note": "Raw totals may count repeated human reviews in multiple generator files.",
        }

    # Compare available subsets for each conference/year.
    overlap = {}
    all_years = sorted({cy for _, cy in indexed})
    for cy in all_years:
        if wanted_years and cy not in wanted_years:
            continue

        subset_results = {}
        for subset in EXPECTED_SUBSETS:
            ids = human_review_ids.get((subset, cy), set())
            texts = human_review_texts.get((subset, cy), set())
            papers = paper_ids.get((subset, cy), set())
            subset_results[subset] = {
                "unique_human_review_ids": len(ids),
                "unique_human_review_texts": len(texts),
                "unique_paper_ids": len(papers),
            }

        for i, subset_a in enumerate(EXPECTED_SUBSETS):
            for subset_b in EXPECTED_SUBSETS[i + 1:]:
                ids_a = human_review_ids.get((subset_a, cy), set())
                ids_b = human_review_ids.get((subset_b, cy), set())
                texts_a = human_review_texts.get((subset_a, cy), set())
                texts_b = human_review_texts.get((subset_b, cy), set())
                papers_a = paper_ids.get((subset_a, cy), set())
                papers_b = paper_ids.get((subset_b, cy), set())

                subset_results[f"{subset_a}_vs_{subset_b}"] = {
                    "shared_human_review_ids": len(ids_a & ids_b),
                    "shared_human_review_texts": len(texts_a & texts_b),
                    "shared_paper_ids": len(papers_a & papers_b),
                }

        overlap[cy] = subset_results

    report["overlap_audits"] = overlap

    report["inspection_scope"] = {
        "conference_year_filter": sorted(wanted_years) if wanted_years else "all",
        "subsets": list(EXPECTED_SUBSETS),
        "files_selected": len(selected),
        "files_successfully_inspected": len(report["files"]),
        "failed_files": len(report["errors"]),
    }

    print("\n=== IntelLabs subset summaries ===")
    print(json.dumps(jsonable(report["subset_summaries"]), indent=2))
    print("\n=== Inspection scope ===")
    print(json.dumps(report["inspection_scope"], indent=2))

    return report


# ----------------------------------------------------------------------------
# Optional Sem-Detect inspection (not needed for the current binary stage)
# ----------------------------------------------------------------------------
def analyze_semdetect(df):
    required = {"class", "split", "paper_id", "clean_review"}
    missing = sorted(required - set(df.columns))

    report = {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "missing_required_columns": missing,
    }

    if "class" not in df.columns or "split" not in df.columns:
        return report

    report["class_counts"] = {
        str(k): int(v)
        for k, v in df["class"].value_counts(dropna=False).items()
    }
    report["split_counts"] = {
        str(k): int(v)
        for k, v in df["split"].value_counts(dropna=False).items()
    }
    report["class_by_split"] = {
        str(split): {
            str(cls): int(count)
            for cls, count in counts.items()
        }
        for split, counts in pd.crosstab(df["split"], df["class"]).iterrows()
    }

    if "paper_id" in df.columns:
        tr = df.loc[df["split"] == "train"]
        te = df.loc[df["split"] == "test"]
        report["paper_id_overlap_train_test"] = len(
            set(tr["paper_id"].dropna()) & set(te["paper_id"].dropna())
        )

    if "clean_review" in df.columns:
        text = df["clean_review"].fillna("").astype(str).str.strip()
        report["empty_review_count"] = int(text.eq("").sum())

    return report


def inspect_semdetect(cfg):
    from huggingface_hub import hf_hub_download, list_repo_files

    sc = cfg.get("data", {}).get("semdetect", {})
    repo_id = sc.get("repo_id", "Sem-Detect/ML_Conferences-Peer-Reviews")
    config_name = sc.get("config_name", "reviews")

    files = [
        f for f in list_repo_files(repo_id, repo_type="dataset")
        if f.startswith(f"{config_name}/") and f.endswith(".parquet")
    ]

    if not files:
        raise FileNotFoundError(
            f"No Parquet files found under {config_name}/ in {repo_id}"
        )

    frames = []
    for path in files:
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=path,
            repo_type="dataset",
        )
        frames.append(pd.read_parquet(local_path))

    df = pd.concat(frames, ignore_index=True)
    report = {
        "repo_id": repo_id,
        "config": config_name,
        "files": files,
        "analysis": analyze_semdetect(df),
    }
    print("\n=== Optional Sem-Detect inspection ===")
    print(json.dumps(jsonable(report["analysis"]), indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-intellabs",
        action="store_true",
        help="Skip IntelLabs inspection.",
    )
    parser.add_argument(
        "--include-semdetect",
        action="store_true",
        help="Also inspect Sem-Detect; not required for the current stage.",
    )
    args = parser.parse_args()

    if args.skip_intellabs and not args.include_semdetect:
        parser.error("Nothing to inspect. Enable --include-semdetect.")

    cfg = load_cfg()
    report = {}

    if not args.skip_intellabs:
        report["intellabs"] = inspect_intellabs(cfg)

    if args.include_semdetect:
        report["semdetect"] = inspect_semdetect(cfg)

    paths_cfg = cfg.get("paths", {})
    audit_dir_name = paths_cfg.get("audit_dir", "outputs/audit")
    audit_dir = Path(audit_dir_name)
    if not audit_dir.is_absolute():
        audit_dir = get_work_dir(cfg) / audit_dir

    audit_dir.mkdir(parents=True, exist_ok=True)
    output_path = audit_dir / "inspection_report.json"

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(jsonable(report), f, indent=2, ensure_ascii=False)

    print(f"\nInspection report saved to: {output_path}")


if __name__ == "__main__":
    main()
