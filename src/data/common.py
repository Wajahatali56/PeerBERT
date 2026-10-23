"""
Shared helpers for the data layer: config, paths, seeding, labels, text,
exclusion logging and JSON output.

Every loader returns rows in the UNIFIED_COLUMNS schema so that splitting,
auditing, training and evaluation never depend on dataset-specific columns.
"""

import contextlib
import hashlib
import json
import os
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]

UNIFIED_COLUMNS = [
    "row_id",        # unique within a dataset
    "dataset",       # intellabs | semdetect
    "partition",     # official source: calibration/test/extended or train/test
    "text",
    "label",         # int
    "label_name",
    "paper_key",     # conference_year_paperid (composite, globally unique)
    "paper_id_raw",  # naive key, information only
    "conference",
    "year",
    "conf_year",
    "generator",     # AI generator for IntelLabs AI rows, "human" for human rows
    "review_id",
    "source_file",
]


# ---------------------------------------------------------------- config/paths
def load_config(path=None):
    path = Path(path) if path else ROOT / "configs" / "config.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Configuration not found: {path}")
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_work_dir(cfg):
    """PRAD_WORK_DIR overrides paths.work_dir; relative paths resolve from the repo root."""
    wd = Path(os.environ.get("PRAD_WORK_DIR", cfg.get("paths", {}).get("work_dir", ".")))
    return wd if wd.is_absolute() else (ROOT / wd).resolve()


def resolve_path(cfg, key, default):
    p = Path(cfg.get("paths", {}).get(key, default))
    return p if p.is_absolute() else get_work_dir(cfg) / p


def dataset_dir_name(cfg, dataset):
    """Config key -> folder name (semdetect -> sem_detect)."""
    dirs = cfg.get("dataset_dirs", {"intellabs": "intellabs", "semdetect": "sem_detect"})
    if dataset not in dirs:
        raise KeyError(f"Unknown dataset '{dataset}'. Known: {sorted(dirs)}")
    return dirs[dataset]


def manifest_dir(cfg, dataset):
    return resolve_path(cfg, "manifest_dir", "outputs/manifests") / dataset_dir_name(cfg, dataset)


def audit_dir(cfg, dataset):
    return resolve_path(cfg, "audit_dir", "outputs/audit") / dataset_dir_name(cfg, dataset)


def cache_dir(cfg, dataset):
    return resolve_path(cfg, "cache_dir", "outputs/cache") / dataset_dir_name(cfg, dataset)


def config_hash(obj):
    blob = json.dumps(jsonable(obj), sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ---------------------------------------------------------------- labels/text
def normalize_label(value):
    """Normalize CSV representations of 0/1 without silently remapping anything else."""
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return None
    value = str(value).strip()
    if value in {"0", "0.0"}:
        return 0
    if value in {"1", "1.0"}:
        return 1
    return value


def clean_text(value):
    """Minimal preprocessing: strip only. Missing values become ''."""
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return ""
    return str(value).strip()


_WS = re.compile(r"\s+")


def text_hash(text):
    """Hash for exact-duplicate detection (whitespace-collapsed, case kept)."""
    return hashlib.md5(_WS.sub(" ", text).strip().encode("utf-8")).hexdigest()


def is_text_dtype(series):
    return pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)


# ---------------------------------------------------------------- logging/IO
class ExclusionLog:
    """Every dropped row is recorded with a reason. Saved next to the manifests."""

    COLUMNS = ["dataset", "partition", "row_id", "reason", "detail"]

    def __init__(self):
        self._parts = []

    def add(self, dataset, partition, row_ids, reason, detail=""):
        row_ids = list(row_ids)
        if not row_ids:
            return
        self._parts.append(pd.DataFrame({
            "dataset": dataset, "partition": partition, "row_id": row_ids,
            "reason": reason, "detail": detail,
        }))

    def extend(self, other):
        self._parts.extend(other._parts)

    def frame(self):
        if not self._parts:
            return pd.DataFrame(columns=self.COLUMNS)
        return pd.concat(self._parts, ignore_index=True)[self.COLUMNS]

    def summary(self):
        f = self.frame()
        if f.empty:
            return {}
        return {f"{p}:{r}": int(n) for (p, r), n in f.groupby(["partition", "reason"]).size().items()}


@contextlib.contextmanager
def file_lock(path):
    """
    Exclusive lock around a read-modify-write of a shared file, so that two scripts running on
    different GPUs cannot overwrite each other's rows (Linux/macOS; a no-op where fcntl is missing).
    """
    path = Path(str(path) + ".lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        try:
            import fcntl
            fcntl.flock(fh, fcntl.LOCK_EX)
        except ImportError:
            pass
        yield


def jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, float):
        return None if not np.isfinite(obj) else obj
    if isinstance(obj, np.ndarray):
        return jsonable(obj.tolist())
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    if obj is not None and not isinstance(obj, (str, int, bool)) and pd.isna(obj):
        return None
    return obj


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(jsonable(obj), f, indent=2, ensure_ascii=False)
    return path


def load_json(path):
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def ensure_unified(df):
    missing = [c for c in UNIFIED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Frame is missing unified columns: {missing}")
    return df[UNIFIED_COLUMNS + [c for c in df.columns if c not in UNIFIED_COLUMNS]]
