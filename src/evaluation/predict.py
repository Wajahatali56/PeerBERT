"""
Prediction frames: one row per example with id, label, probabilities and metadata.
Saved as CSV under outputs/results/<dataset>/.

Columns: row_id, label, pred, score (= P(AI) or a 'larger = more AI' detector score),
         proba_<class> (multiclass), paper_key, conf_year, generator, label_name.

Sem-Detect binary view of the 3-class model (for comparison with binary baselines):
rows with label in {human, ai} only, score = P(ai) / (P(human) + P(ai)).
Thresholding that score at 0.5 equals argmax over the two classes.
"""

from pathlib import Path

import numpy as np
import pandas as pd

META_COLS = ["row_id", "label", "label_name", "paper_key", "conf_year", "generator"]


def softmax(logits):
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def prediction_frame(meta, proba=None, score=None, class_names=("human", "ai"), threshold=0.5):
    """
    meta: frame with META_COLS (extra columns ignored).
    proba: (n, K) probabilities (model). score: (n,) detector score (baselines).
    """
    out = meta[[c for c in META_COLS if c in meta.columns]].reset_index(drop=True).copy()
    if proba is not None:
        P = np.asarray(proba, dtype=float)
        for i, name in enumerate(class_names):
            out[f"proba_{name}"] = P[:, i]
        out["pred"] = P.argmax(1)
        if len(class_names) == 2:
            out["score"] = P[:, 1]
            out["pred"] = (out["score"] >= threshold).astype(int)
    elif score is not None:
        out["score"] = np.asarray(score, dtype=float)
        out["pred"] = (out["score"] >= threshold).astype(int)
    else:
        raise ValueError("give proba or score")
    return out


def binary_view(pred3, human="human", ai="ai", threshold=0.5):
    """3-class prediction frame -> human/ai-only binary frame (rewrite rows dropped)."""
    keep = pred3["label"].isin([0, 1])
    b = pred3.loc[keep].copy()
    ph, pa = b[f"proba_{human}"], b[f"proba_{ai}"]
    b["score"] = pa / (ph + pa).clip(lower=1e-12)
    b["pred"] = (b["score"] >= threshold).astype(int)
    return b


def save_predictions(df, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def load_predictions(path):
    return pd.read_csv(path, dtype={"row_id": str, "paper_key": str})


def run_inference(model, loader, device, amp_dtype=None):
    """Logits for every batch of a DataLoader yielding dicts (input_ids, attention_mask, ...)."""
    import torch
    model.eval()
    chunks = []
    with torch.inference_mode():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items() if k in ("input_ids", "attention_mask")}
            if amp_dtype is not None and device.type == "cuda":
                with torch.autocast("cuda", dtype=amp_dtype):
                    logits = model(**batch)
            else:
                logits = model(**batch)
            chunks.append(logits.float().cpu().numpy())
    return np.concatenate(chunks) if chunks else np.zeros((0, 0))
