"""
Breakdowns and tables.

- by_group: binary metrics per value of a column (conf_year, ...).
- by_generator (IntelLabs): for generator g, AI rows of g plus the human rows of the
  conference-years g covers (human rows are shared across generators).
- seed_summary: mean +/- std over seeds (sample std, ddof=1).
- comparison table: one row per method with its type
  (trained classifier / pretrained detector / zero-shot detector) -> CSV + JSON.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.common import load_json, save_json

from .metrics import binary_metrics, headline

METHOD_TYPES = ("trained classifier", "pretrained detector", "zero-shot detector")


def by_group(pred, col, threshold=0.5, min_neg_events=10):
    out = {}
    for g, sub in pred.groupby(col):
        out[str(g)] = headline(binary_metrics(sub["label"], sub["score"], threshold, min_neg_events))
    return out


def by_generator(pred, threshold=0.5, min_neg_events=10):
    out = {}
    ai = pred[pred["label"] == 1]
    for g in sorted(ai["generator"].unique()):
        a = ai[ai["generator"] == g]
        h = pred[(pred["label"] == 0) & pred["conf_year"].isin(a["conf_year"].unique())]
        out[g] = headline(binary_metrics(pd.concat([h, a])["label"], pd.concat([h, a])["score"],
                                         threshold, min_neg_events))
    return out


def seed_summary(metric_dicts, keys=("accuracy", "macro_f1", "f1", "roc_auc", "pr_auc")):
    out = {"n_seeds": len(metric_dicts)}
    for k in keys:
        vals = [m.get(k) for m in metric_dicts if m.get(k) is not None]
        if vals:
            out[k] = {"mean": float(np.mean(vals)),
                      "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                      "values": vals}
    return out


def comparison_row(method, method_type, split, threshold_kind, metrics, **extra):
    if method_type not in METHOD_TYPES:
        raise ValueError(f"method_type must be one of {METHOD_TYPES}")
    return {"method": method, "type": method_type, "split": split,
            "threshold_kind": threshold_kind, **headline(metrics), **extra}


def save_comparison_table(rows, out_dir, name="comparison_table"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    if not df.empty:
        order = {t: i for i, t in enumerate(METHOD_TYPES)}
        df = df.sort_values(["split", "threshold_kind", "type", "method"],
                            key=lambda s: s.map(order) if s.name == "type" else s).reset_index(drop=True)
    df.to_csv(out_dir / f"{name}.csv", index=False)
    save_json(df.to_dict(orient="records"), out_dir / f"{name}.json")
    return df


def merge_comparison_rows(path_json, new_rows, key=("method", "split", "threshold_kind")):
    """Upsert rows into an existing comparison_table.json (restart-safe incremental runs)."""
    path_json = Path(path_json)
    old = load_json(path_json) if path_json.exists() else []
    new_keys = {tuple(r[k] for k in key) for r in new_rows}
    return [r for r in old if tuple(r.get(k) for k in key) not in new_keys] + list(new_rows)
