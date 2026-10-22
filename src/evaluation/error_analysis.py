"""
Error analysis on a prediction frame (binary: label/pred/score; multiclass: label/pred/proba_*).

- fp_fn_tables: false positives / false negatives with ids, label, prediction and
  confidence, most confident errors first.
- error_breakdown: error rate per class and per conference-year (and generator).
- Text is attached (truncated) when a text column/lookup is given, so the tables
  can be read without opening the manifests.
"""

from pathlib import Path

import numpy as np
import pandas as pd


def _confidence(pred):
    pcols = [c for c in pred.columns if c.startswith("proba_")]
    if pcols and "score" not in pred.columns:
        return pred[pcols].max(axis=1)
    return np.where(pred["pred"] == 1, pred["score"], 1 - pred["score"]) \
        if pred["score"].between(0, 1).all() else pred["score"].abs()


def fp_fn_tables(pred, texts=None, top_k=50, text_chars=400):
    p = pred.copy()
    p["confidence"] = _confidence(p)
    if texts is not None:
        p["text"] = p["row_id"].map(texts).fillna("").str.slice(0, text_chars)
    cols = [c for c in ("row_id", "paper_key", "conf_year", "generator", "label", "label_name",
                        "pred", "score", "confidence", "text") if c in p.columns]
    wrong = p[p["pred"] != p["label"]].sort_values("confidence", ascending=False)
    if set(p["label"].unique()) <= {0, 1}:
        fp = wrong[(wrong["label"] == 0) & (wrong["pred"] == 1)][cols].head(top_k)
        fn = wrong[(wrong["label"] == 1) & (wrong["pred"] == 0)][cols].head(top_k)
        return {"false_positives": fp, "false_negatives": fn}
    return {"misclassified": wrong[cols].head(top_k)}


def error_breakdown(pred, cols=("label_name", "conf_year", "generator")):
    out = {}
    err = (pred["pred"] != pred["label"]).astype(int)
    for c in cols:
        if c in pred.columns:
            g = err.groupby(pred[c])
            out[c] = {str(k): {"n": int(n), "errors": int(e), "error_rate": round(e / n, 6) if n else None}
                      for k, n, e in zip(g.size().index, g.size().values, g.sum().values)}
    return out


def save_error_analysis(pred, out_dir, prefix, texts=None, top_k=50):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, table in fp_fn_tables(pred, texts, top_k).items():
        p = out_dir / f"{prefix}_{name}.csv"
        table.to_csv(p, index=False)
        paths[name] = str(p)
    return paths, error_breakdown(pred)


def text_lookup(manifest):
    return pd.Series(manifest["text"].values, index=manifest["row_id"].values)
