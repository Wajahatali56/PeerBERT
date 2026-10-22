"""
IntelLabs/AI-Peer-Review-Detection-Benchmark loader (binary: 0 = human, 1 = AI).

Facts this loader enforces (from outputs/audit/inspection_report.json):
- CSV files at <subset>/<generator>/<Conf><Year>.<subset>.<model>.csv.
  Subsets are never merged. Folders are AI generators only: labels come from
  the `label` column, never from the folder.
- label 0 <-> custom_id starting with "human"; checked on every row of every file.
- Each review-id has exactly one human and one AI row per file.
- Human rows are replicated in every generator file of a subset: deduplicated by
  (conf_year, review-id), first copy kept (files sorted by conf_year, generator),
  every dropped copy logged, copies whose text differs counted.
- PaperId is unique only within a conference-year: paper_key = Conf_Year_PaperId.
- Review text columns differ per conference-year: text = allow-listed fields
  (config order) joined with a blank line.
- Fail-late schema guard: all schema/label problems from all files are collected
  and raised once.

The HF `datasets` auto-loader fails on this repo (schema mismatches), so files are
fetched one by one with huggingface_hub and read with pandas.
"""

import re
from collections import Counter

import numpy as np
import pandas as pd

from .common import (
    ExclusionLog, UNIFIED_COLUMNS, cache_dir, clean_text, config_hash,
    load_json, normalize_label, save_json, text_hash,
)

DATASET = "intellabs"
SUBSETS = ("calibration", "test", "extended")
FNAME_RE = re.compile(
    r"^(?P<conf>[A-Za-z]+)(?P<year>\d{4})\.(?P<subset>[A-Za-z]+)\.(?P<model>.+)\.csv$"
)
CONF_YEAR_IN_ID = re.compile(r"(iclr|neurips)[_\-.]?(\d{4})", re.IGNORECASE)
KNOWN_CONFS = ("iclr", "neurips")


class SchemaError(RuntimeError):
    """Raised once with the full list of problems across all files."""


# ---------------------------------------------------------------- hub access
def hf_lister(repo_id):
    from huggingface_hub import list_repo_files
    return list_repo_files(repo_id, repo_type="dataset")


def hf_fetcher(repo_id, path):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=repo_id, filename=path, repo_type="dataset")


def parse_path(path):
    """Return file info dict, or None if the path is not a benchmark CSV."""
    parts = path.split("/")
    if len(parts) != 3 or not path.lower().endswith(".csv"):
        return None
    subset, generator, name = parts
    m = FNAME_RE.match(name)
    if not m or m["subset"].lower() != subset.lower():
        return None
    return {
        "path": path, "subset": subset, "generator": generator,
        "conference": m["conf"], "year": int(m["year"]),
        "conf_year": f'{m["conf"]}{m["year"]}', "model": m["model"],
    }


def list_subset_files(repo_id, subset, lister=None):
    infos = [parse_path(p) for p in (lister or hf_lister)(repo_id)]
    infos = [i for i in infos if i and i["subset"] == subset]
    return sorted(infos, key=lambda i: (i["conf_year"], i["generator"], i["path"]))


# ---------------------------------------------------------------- schema
def classify_columns(df, icfg):
    allow = list(icfg["text_fields_allow"])
    never = set(icfg["text_fields_never"])
    prefixes = tuple(icfg.get("text_field_never_prefixes", []))
    min_chars = float(icfg.get("unknown_narrative_min_mean_chars", 150))

    text_fields = [c for c in allow if c in df.columns]
    prefixed, unknown_narrative, unknown_other = [], [], []
    for c in df.columns:
        if c in allow or c in never:
            continue
        if prefixes and c.startswith(prefixes):
            prefixed.append(c)
            continue
        mean_len = df[c].astype(str).str.len().mean() if len(df) else 0.0
        (unknown_narrative if mean_len > min_chars else unknown_other).append(
            {"column": c, "mean_chars": round(float(mean_len), 1)}
        )
    return {
        "text_fields": text_fields, "prefixed_excluded": prefixed,
        "unknown_narrative": unknown_narrative, "unknown_other": unknown_other,
    }


def build_text(df, fields):
    """Join non-empty allow-listed fields with a blank line; strip only."""
    if not fields:
        return pd.Series([""] * len(df), index=df.index, dtype=object)
    cols = [df[f].map(clean_text) for f in fields]
    out = []
    for vals in zip(*cols):
        out.append("\n\n".join(v for v in vals if v))
    return pd.Series(out, index=df.index, dtype=object)


# ---------------------------------------------------------------- checks
def _conference_value_problems(values, conf, year):
    """Count `conference` values naming a different conference or year."""
    bad, unparsed = 0, 0
    for v in values:
        s = re.sub(r"[^a-z0-9]", "", str(v).lower())
        named = [k for k in KNOWN_CONFS if k in s]
        years = re.findall(r"(20\d{2})", s)
        if not named and not years:
            unparsed += 1
        elif (named and conf.lower() not in named) or (years and str(year) not in years):
            bad += 1
    return bad, unparsed


def _custom_id_conf_problems(custom_ids, conf, year):
    bad, unparsed = 0, 0
    for cid in custom_ids:
        m = CONF_YEAR_IN_ID.search(str(cid))
        if not m:
            unparsed += 1
        elif m[1].lower() != conf.lower() or m[2] != str(year):
            bad += 1
    return bad, unparsed


def load_file(info, local_path, icfg):
    """Read one CSV -> (unified frame, stats, problems). Never raises on content."""
    df = pd.read_csv(local_path, dtype=str, keep_default_na=False, na_values=[])
    path, problems = info["path"], []
    required = ["label", "custom_id", "PaperId", "review-id"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        return None, {"file": path, "rows": len(df)}, [f"{path}: missing required columns {missing}"]

    schema = classify_columns(df, icfg)
    if not schema["text_fields"]:
        problems.append(f"{path}: no allow-listed text field among columns {list(df.columns)}")
    for u in schema["unknown_narrative"]:
        problems.append(
            f"{path}: unrecognized narrative-looking column '{u['column']}' "
            f"(mean {u['mean_chars']} chars) - add it to text_fields_allow or text_fields_never"
        )

    labels = df["label"].map(normalize_label)
    bad_labels = labels[~labels.isin([0, 1])]
    if len(bad_labels):
        problems.append(f"{path}: {len(bad_labels)} rows with label outside {{0,1}}: "
                        f"{sorted(map(str, bad_labels.unique()))[:5]}")
    is_human_id = df["custom_id"].str.strip().str.lower().str.startswith("human")
    mismatch = int(((labels == 0) & ~is_human_id).sum() + ((labels == 1) & is_human_id).sum())
    if mismatch:
        problems.append(f"{path}: {mismatch} rows where label disagrees with custom_id prefix")

    pairs = labels.groupby(df["review-id"]).agg(lambda s: tuple(sorted(s.tolist())))
    bad_pairs = int((pairs != (0, 1)).sum())
    if bad_pairs:
        problems.append(f"{path}: {bad_pairs} review-ids without exactly one human + one AI row")

    pid_numeric = pd.to_numeric(df["PaperId"], errors="coerce")
    if pid_numeric.isna().any():
        problems.append(f"{path}: {int(pid_numeric.isna().sum())} non-integer PaperId values")

    conf, year = info["conference"], info["year"]
    stats_conf = {}
    if "conference" in df.columns:
        bad, unparsed = _conference_value_problems(df["conference"], conf, year)
        stats_conf["conference_column_unparsed"] = unparsed
        if bad:
            problems.append(f"{path}: {bad} rows whose `conference` disagrees with filename {conf}{year}")
    bad, unparsed = _custom_id_conf_problems(df["custom_id"], conf, year)
    stats_conf["custom_id_without_conf_year"] = unparsed
    if bad:
        problems.append(f"{path}: {bad} custom_ids naming a different conference-year than {conf}{year}")

    text = build_text(df, schema["text_fields"])
    lab = labels.where(labels.isin([0, 1]))
    out = pd.DataFrame({
        "row_id": f"{info['subset']}:{info['conf_year']}:" + df["custom_id"].str.strip(),
        "dataset": DATASET,
        "partition": info["subset"],
        "text": text,
        "label": lab,
        "label_name": lab.map({0: "human", 1: "ai"}),
        "paper_key": f"{conf}_{year}_" + df["PaperId"].str.strip(),
        "paper_id_raw": df["PaperId"].str.strip(),
        "conference": conf,
        "year": year,
        "conf_year": info["conf_year"],
        "generator": np.where(lab == 0, "human", info["generator"]),
        "review_id": df["review-id"].str.strip(),
        "source_file": path,
        "generator_model": np.where(lab == 0, "human", info["model"]),
        "text_hash": text.map(text_hash),
    })

    never_lengths = {}
    for c in icfg["text_fields_never"]:
        if c in df.columns and c not in required:
            lengths = df[c].astype(str).str.len()
            never_lengths[c] = {str(k): round(float(v), 1) for k, v in lengths.groupby(lab).mean().items()}

    stats = {
        "file": path, "rows": len(df), "generator": info["generator"], "model": info["model"],
        "conf_year": info["conf_year"], "columns": list(df.columns),
        "label_counts": {str(k): int(v) for k, v in lab.value_counts().items()},
        "text_fields": schema["text_fields"], "prefixed_excluded": schema["prefixed_excluded"],
        "unknown_other_columns": schema["unknown_other"],
        "empty_text_rows": int((text == "").sum()),
        "mean_text_chars_by_label": {
            str(k): round(float(v), 1) for k, v in text.str.len().groupby(lab).mean().items()
        },
        "never_field_mean_chars_by_label": never_lengths,
        **stats_conf,
    }
    return out, stats, problems


# ---------------------------------------------------------------- subset load
def _dedupe_humans(df, log, subset):
    human = df["label"] == 0
    key = df["conf_year"] + "|" + df["review_id"]
    dup = human & key.duplicated(keep="first")
    first_hash = df.loc[human & ~dup].set_index(key[human & ~dup])["text_hash"]
    first_paper = df.loc[human & ~dup].set_index(key[human & ~dup])["paper_key"]
    dropped = df.loc[dup]
    differs = dropped["text_hash"].values != first_hash.loc[key[dup]].values
    paper_differs = dropped["paper_key"].values != first_paper.loc[key[dup]].values
    log.add(DATASET, subset, dropped.loc[~differs, "row_id"], "human_duplicate_copy", "text_identical")
    log.add(DATASET, subset, dropped.loc[differs, "row_id"], "human_duplicate_copy", "text_differs_from_first_copy")
    stats = {
        "human_rows_before_dedupe": int(human.sum()),
        "human_duplicate_copies_dropped": int(dup.sum()),
        "human_copies_with_different_text": int(differs.sum()),
        "human_copies_with_different_paper": int(paper_differs.sum()),
    }
    return df.loc[~dup].copy(), stats


def load_subset(cfg, subset, lister=None, fetcher=None, use_cache=True):
    """
    Load one official subset -> (unified frame, stats, ExclusionLog).
    Restart-safe: result cached as parquet, keyed by the loader-relevant config.
    """
    if subset not in SUBSETS:
        raise ValueError(f"Unknown IntelLabs subset '{subset}'")
    icfg = cfg["data"]["intellabs"]
    repo_id = icfg["repo_id"]
    key = config_hash({k: icfg.get(k) for k in (
        "repo_id", "text_fields_allow", "text_fields_never",
        "text_field_never_prefixes", "unknown_narrative_min_mean_chars")})

    cdir = cache_dir(cfg, DATASET)
    data_p, excl_p, stats_p = (cdir / f"{subset}.parquet", cdir / f"{subset}.exclusions.parquet",
                               cdir / f"{subset}.stats.json")
    if use_cache and data_p.exists() and stats_p.exists():
        stats = load_json(stats_p)
        if stats.get("config_hash") == key:
            print(f"[intellabs] {subset}: using cache {data_p}")
            log = ExclusionLog()
            if excl_p.exists():
                log._parts.append(pd.read_parquet(excl_p))
            return pd.read_parquet(data_p), stats, log

    files = list_subset_files(repo_id, subset, lister)
    if not files:
        raise FileNotFoundError(f"No CSV files found for subset '{subset}' in {repo_id}")
    print(f"[intellabs] {subset}: {len(files)} files")

    frames, file_stats, problems = [], [], []
    for info in files:
        local = (fetcher or hf_fetcher)(repo_id, info["path"])
        out, st, pr = load_file(info, local, icfg)
        file_stats.append(st)
        problems.extend(pr)
        if out is not None:
            frames.append(out)
    if problems:
        raise SchemaError(
            f"IntelLabs/{subset}: {len(problems)} problem(s) across {len(files)} files:\n  - "
            + "\n  - ".join(problems)
        )

    df = pd.concat(frames, ignore_index=True)
    df["label"] = df["label"].astype(int)
    log = ExclusionLog()
    df, dedupe_stats = _dedupe_humans(df, log, subset)

    empty = df["text"] == ""
    log.add(DATASET, subset, df.loc[empty, "row_id"], "empty_text")
    df = df.loc[~empty].reset_index(drop=True)

    dup_ids = df["row_id"].duplicated(keep=False)
    if dup_ids.any():
        raise SchemaError(f"IntelLabs/{subset}: {int(dup_ids.sum())} duplicate row_ids after dedupe, "
                          f"e.g. {df.loc[dup_ids, 'row_id'].head(5).tolist()}")

    counts = df["label"].value_counts()
    n_h, n_ai = int(counts.get(0, 0)), int(counts.get(1, 0))
    stats = {
        "config_hash": key, "subset": subset, "files": len(files), "rows": len(df),
        "label_counts": {"0": n_h, "1": n_ai},
        "human_to_ai_ratio": f"1:{n_ai / n_h:.2f}" if n_h else "n/a",
        "generators": sorted(df.loc[df["label"] == 1, "generator"].unique().tolist()),
        "conf_years": sorted(df["conf_year"].unique().tolist()),
        **dedupe_stats,
        "empty_text_dropped": int(empty.sum()),
        "schemas": dict(Counter("|".join(s.get("text_fields", [])) for s in file_stats)),
        "files_with_prefixed_fields": [s["file"] for s in file_stats if s.get("prefixed_excluded")],
        "file_stats": file_stats,
    }
    print(f"[intellabs] {subset}: {len(df)} rows after dedupe, human:AI = {stats['human_to_ai_ratio']}, "
          f"{dedupe_stats['human_duplicate_copies_dropped']} human copies dropped "
          f"({dedupe_stats['human_copies_with_different_text']} with different text)")

    cdir.mkdir(parents=True, exist_ok=True)
    df[UNIFIED_COLUMNS + ["generator_model", "text_hash"]].to_parquet(data_p, index=False)
    log.frame().to_parquet(excl_p, index=False)
    save_json(stats, stats_p)
    return df, stats, log


def per_generator_view(df, generator):
    """Human rows of the conference-years `generator` covers + that generator's AI rows."""
    ai = df[(df["label"] == 1) & (df["generator"] == generator)]
    human = df[(df["label"] == 0) & df["conf_year"].isin(ai["conf_year"].unique())]
    return pd.concat([human, ai])
