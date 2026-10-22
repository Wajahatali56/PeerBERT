"""
Sem-Detect/ML_Conferences-Peer-Reviews loader (3-class: human=0, ai=1, rewrite=2).

- Config `reviews`, original Parquet files read with huggingface_hub + pandas.
- The official `split` column (train/test) is respected. Validation is cut later
  from train papers only (src/data/splits.py).
- paper_id is NOT globally unique: paper_key = conference_year_paper_id;
  submission_id is kept to cross-check overlap in the audit.
- og_review_id is metadata only (not a pairing key), never used for grouping.
- Only invalid rows are dropped (missing class/split/paper key, empty text), each
  logged. The test split is NOT deduplicated; duplicate texts are only flagged.
- Nothing about the dataset is hard-coded: row, class and split counts are read from
  the real files and saved in the load stats.
"""

import pandas as pd

from .common import (
    ExclusionLog, UNIFIED_COLUMNS, cache_dir, clean_text, config_hash,
    load_json, save_json, text_hash,
)

DATASET = "semdetect"
SPLITS = ("train", "test")


def hf_frame_loader(repo_id, config_name):
    from huggingface_hub import hf_hub_download, list_repo_files
    files = sorted(
        f for f in list_repo_files(repo_id, repo_type="dataset")
        if f.startswith(f"{config_name}/") and f.endswith(".parquet")
    )
    if not files:
        raise FileNotFoundError(f"No Parquet files under {config_name}/ in {repo_id}")
    frames = [pd.read_parquet(hf_hub_download(repo_id=repo_id, filename=f, repo_type="dataset"))
              for f in files]
    return pd.concat(frames, ignore_index=True), files


def _norm(series):
    return series.map(clean_text).str.lower()


def observed_counts(raw):
    """Counts read from the real files (reported, never compared with assumed numbers)."""
    return {"rows": len(raw), "columns": list(raw.columns),
            "class_counts": _norm(raw["class"]).value_counts().to_dict(),
            "split_counts": _norm(raw["split"]).value_counts().to_dict()}


def build_frame(raw, scfg):
    """Raw Sem-Detect frame -> (unified frame, stats, ExclusionLog)."""
    missing = [c for c in scfg["required_columns"] if c not in raw.columns]
    if missing:
        raise ValueError(f"Sem-Detect: missing required columns {missing}; got {list(raw.columns)}")
    label_map = {str(k).lower(): int(v) for k, v in scfg["label_map"].items()}
    raw = raw.reset_index(drop=True).copy()
    optional_missing = [c for c in scfg.get("optional_columns", []) if c not in raw.columns]
    for c in optional_missing:
        raw[c] = ""          # absent in the real files: kept empty and reported, never invented

    cls = _norm(raw["class"])
    split = _norm(raw["split"])
    unknown_cls = sorted(set(cls[cls != ""]) - set(label_map))
    unknown_split = sorted(set(split[split != ""]) - set(SPLITS))
    if unknown_cls or unknown_split:
        raise ValueError(f"Sem-Detect: unknown class values {unknown_cls} / split values {unknown_split}")

    review_id = raw["review_id"].map(clean_text)
    ids_unique = bool(review_id.ne("").all() and not review_id.duplicated().any())
    row_id = ("sd:" + review_id) if ids_unique else ("sd:row" + raw.index.astype(str))

    conference = raw["conference"].map(clean_text)
    year = raw["year"].map(clean_text).str.replace(r"\.0$", "", regex=True)
    paper_id = raw["paper_id"].map(clean_text)
    text = raw[scfg.get("text_field", "clean_review")].map(clean_text)
    label = cls.map(label_map)

    df = pd.DataFrame({
        "row_id": row_id,
        "dataset": DATASET,
        "partition": split,
        "text": text,
        "label": label,
        "label_name": cls,
        "paper_key": conference + "_" + year + "_" + paper_id,
        "paper_id_raw": paper_id,
        "conference": conference,
        "year": pd.to_numeric(year, errors="coerce"),
        "conf_year": conference + year,
        "generator": "unspecified",
        "review_id": review_id,
        "source_file": "reviews",
        "submission_id": raw["submission_id"].map(clean_text),
        "og_review_id": raw["og_review_id"].map(clean_text),   # metadata only
        "text_hash": text.map(text_hash),
    })
    df.loc[df["label"] == 0, "generator"] = "human"

    log = ExclusionLog()
    invalid = {
        "missing_class": label.isna(),
        "missing_split": split.eq(""),
        "missing_paper_key": conference.eq("") | year.eq("") | paper_id.eq(""),
        "empty_text": text.eq(""),
    }
    drop = pd.Series(False, index=df.index)
    for reason, mask in invalid.items():
        new = mask & ~drop                      # each dropped row logged once (first reason)
        for part, ids in df.loc[new].groupby("partition")["row_id"]:
            log.add(DATASET, part or "unknown", ids, reason)
        drop |= mask
    df = df.loc[~drop].reset_index(drop=True)
    df["label"] = df["label"].astype(int)

    dup_within = {s: int(g["text_hash"].duplicated(keep=False).sum()) for s, g in df.groupby("partition")}
    tr, te = df[df["partition"] == "train"], df[df["partition"] == "test"]
    stats = {
        "rows_raw": len(raw), "rows": len(df), "optional_columns_missing": optional_missing,
        "row_id_scheme": "review_id" if ids_unique else "row_index (review_id not unique)",
        "dropped": {k: int(v.sum()) for k, v in invalid.items()},
        "class_by_split": {s: g["label_name"].value_counts().to_dict() for s, g in df.groupby("partition")},
        "duplicate_text_rows_within_split (flagged, kept)": dup_within,
        "test_rows_with_text_in_train (flagged, kept)": int(te["text_hash"].isin(tr["text_hash"]).sum()),
        "naive_paper_id_overlap_train_test": len(set(tr["paper_id_raw"]) & set(te["paper_id_raw"])),
        "paper_key_overlap_train_test": len(set(tr["paper_key"]) & set(te["paper_key"])),
        "submission_id_overlap_train_test": len(
            set(tr["submission_id"]) - {""} & set(te["submission_id"]) - {""}),
        "og_review_id_distinct": int(df["og_review_id"].nunique()),
    }
    return df, stats, log


def load_semdetect(cfg, frame_loader=None, use_cache=True):
    """Load the whole config -> (unified frame with partition train/test, stats, ExclusionLog)."""
    scfg = cfg["data"]["semdetect"]
    key = config_hash({k: scfg.get(k) for k in ("repo_id", "config_name", "label_map", "text_field")})
    cdir = cache_dir(cfg, DATASET)
    data_p, excl_p, stats_p = cdir / "all.parquet", cdir / "all.exclusions.parquet", cdir / "all.stats.json"
    if use_cache and data_p.exists() and stats_p.exists():
        stats = load_json(stats_p)
        if stats.get("config_hash") == key:
            print(f"[semdetect] using cache {data_p}")
            log = ExclusionLog()
            if excl_p.exists():
                log._parts.append(pd.read_parquet(excl_p))
            return pd.read_parquet(data_p), stats, log

    loader = frame_loader or hf_frame_loader
    raw, files = loader(scfg["repo_id"], scfg.get("config_name", "reviews"))
    observed = observed_counts(raw)
    df, stats, log = build_frame(raw, scfg)
    stats = {"config_hash": key, "files": files, "observed_raw": observed, **stats}
    print(f"[semdetect] observed in the real files: rows={observed['rows']} "
          f"classes={observed['class_counts']} splits={observed['split_counts']}")
    if stats["optional_columns_missing"]:
        print(f"[semdetect] optional columns not present: {stats['optional_columns_missing']}")
    print(f"[semdetect] {len(df)} rows kept of {len(raw)}; class_by_split={stats['class_by_split']}")

    cdir.mkdir(parents=True, exist_ok=True)
    keep = UNIFIED_COLUMNS + ["submission_id", "og_review_id", "text_hash"]
    df[keep].to_parquet(data_p, index=False)
    log.frame().to_parquet(excl_p, index=False)
    save_json(stats, stats_p)
    return df, stats, log
