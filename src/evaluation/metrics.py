"""
Classification metrics. Positive class = AI (label 1).

binary_metrics:     Accuracy, Precision, Recall, F1 (AI), Macro-F1, Weighted-F1,
                    ROC-AUC, PR-AUC, confusion matrix, per-class metrics, AI TPR/FPR,
                    TPR at 1/5/10% FPR (reported only when n_neg supports it).
multiclass_metrics: Accuracy, Macro/Weighted-F1, per-class metrics, confusion
                    matrix, one-vs-rest macro ROC-AUC (Sem-Detect native 3-class).
calibrate_threshold: chosen on VALIDATION scores only.

Edge cases: single-class inputs give None for ranking metrics (with a note);
divisions by zero give 0.0 (sklearn zero_division=0).
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    precision_recall_fscore_support, roc_auc_score, roc_curve,
)

FPR_TARGETS = (0.01, 0.05, 0.10)
BINARY_NAMES = ("human", "ai")


def _r(x):
    return None if x is None else round(float(x), 6)


def tpr_at_fpr(y_true, score, targets=FPR_TARGETS, min_neg_events=10):
    """
    Highest TPR with FPR <= target. `supported` is False when n_neg * target < min_neg_events
    (fewer than min_neg_events negatives would define the tail); the value is then None.
    """
    y_true = np.asarray(y_true).astype(int)
    n_neg = int((y_true == 0).sum())
    out = {}
    have_both = len(np.unique(y_true)) == 2
    fpr, tpr, thr = roc_curve(y_true, score) if have_both else (None, None, None)
    for t in targets:
        key = f"{int(round(t * 100))}%"
        supported = have_both and n_neg * t >= min_neg_events
        if not supported:
            out[key] = {"tpr": None, "threshold": None, "supported": False, "n_neg": n_neg}
            continue
        ok = fpr <= t
        i = int(np.argmax(np.where(ok, tpr, -1)))
        out[key] = {"tpr": _r(tpr[i]), "threshold": _r(thr[i]), "supported": True, "n_neg": n_neg}
    return out


def binary_metrics(y_true, score, threshold=0.5, min_neg_events=10):
    """y_true in {0,1}; score = P(AI) or any 'larger = more AI' score; pred = score >= threshold."""
    y = np.asarray(y_true).astype(int)
    s = np.asarray(score, dtype=float)
    if len(y) != len(s):
        raise ValueError("y_true and score lengths differ")
    if not np.isfinite(s).all():
        raise ValueError(f"{int((~np.isfinite(s)).sum())} non-finite scores")
    pred = (s >= threshold).astype(int)
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    p, r, f, sup = precision_recall_fscore_support(y, pred, labels=[0, 1], zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    both = n_pos > 0 and n_neg > 0
    return {
        "n": len(y), "n_pos": n_pos, "n_neg": n_neg, "threshold": _r(threshold),
        "accuracy": _r(accuracy_score(y, pred)) if len(y) else None,
        "precision": _r(p[1]), "recall": _r(r[1]), "f1": _r(f[1]),
        "macro_f1": _r(f.mean()),
        "weighted_f1": _r(f1_score(y, pred, labels=[0, 1], average="weighted", zero_division=0)),
        "roc_auc": _r(roc_auc_score(y, s)) if both else None,
        "pr_auc": _r(average_precision_score(y, s)) if both else None,
        "confusion_matrix": {"labels": list(BINARY_NAMES), "matrix": [[int(tn), int(fp)], [int(fn), int(tp)]]},
        "per_class": {BINARY_NAMES[k]: {"precision": _r(p[k]), "recall": _r(r[k]), "f1": _r(f[k]),
                                        "support": int(sup[k])} for k in (0, 1)},
        "ai_tpr": _r(tp / n_pos) if n_pos else None,
        "ai_fpr": _r(fp / n_neg) if n_neg else None,
        "tpr_at_fpr": tpr_at_fpr(y, s, min_neg_events=min_neg_events),
        "notes": [] if both else ["single class present: ROC-AUC/PR-AUC/TPR@FPR undefined"],
    }


def multiclass_metrics(y_true, proba, class_names):
    y = np.asarray(y_true).astype(int)
    P = np.asarray(proba, dtype=float)
    k = len(class_names)
    if P.shape != (len(y), k):
        raise ValueError(f"proba shape {P.shape} != ({len(y)}, {k})")
    pred = P.argmax(1)
    labels = list(range(k))
    p, r, f, sup = precision_recall_fscore_support(y, pred, labels=labels, zero_division=0)
    present = np.unique(y)
    auc, notes = None, []
    if len(present) == k:
        auc = roc_auc_score(y, P, multi_class="ovr", average="macro", labels=labels)
    else:
        notes.append(f"classes present {present.tolist()}: macro OvR ROC-AUC undefined")
    return {
        "n": len(y), "class_counts": {class_names[i]: int(sup[i]) for i in labels},
        "accuracy": _r(accuracy_score(y, pred)) if len(y) else None,
        "macro_f1": _r(f.mean()),
        "weighted_f1": _r(f1_score(y, pred, labels=labels, average="weighted", zero_division=0)),
        "macro_ovr_roc_auc": _r(auc),
        "confusion_matrix": {"labels": list(class_names),
                             "matrix": confusion_matrix(y, pred, labels=labels).tolist()},
        "per_class": {class_names[i]: {"precision": _r(p[i]), "recall": _r(r[i]), "f1": _r(f[i]),
                                       "support": int(sup[i])} for i in labels},
        "notes": notes,
    }


def calibrate_threshold(y_val, score_val, objective="macro_f1"):
    """
    Threshold chosen on validation only. objective: 'macro_f1' (default) or 'youden'.
    Candidates are midpoints between sorted unique scores (plus the extremes).
    """
    y = np.asarray(y_val).astype(int)
    s = np.asarray(score_val, dtype=float)
    if len(np.unique(y)) < 2:
        return 0.5
    u = np.unique(s)
    cands = np.concatenate([[u[0] - 1e-12], (u[:-1] + u[1:]) / 2, [u[-1] + 1e-12]])
    if len(cands) > 2000:                                   # quantile grid keeps it fast
        cands = np.unique(np.quantile(s, np.linspace(0, 1, 2001)))
    best_t, best_v = 0.5, -np.inf
    for t in cands:
        pred = (s >= t).astype(int)
        if objective == "youden":
            tp = ((pred == 1) & (y == 1)).sum() / max((y == 1).sum(), 1)
            fp = ((pred == 1) & (y == 0)).sum() / max((y == 0).sum(), 1)
            v = tp - fp
        else:
            v = f1_score(y, pred, average="macro", zero_division=0)
        if v > best_v:
            best_t, best_v = float(t), v
    return best_t


def headline(m):
    """Flat row of the main numbers for comparison tables."""
    row = {k: m.get(k) for k in ("n", "n_pos", "n_neg", "threshold", "accuracy", "precision", "recall",
                                 "f1", "macro_f1", "weighted_f1", "roc_auc", "pr_auc", "ai_tpr", "ai_fpr")}
    for k, v in (m.get("tpr_at_fpr") or {}).items():
        row[f"tpr@{k}fpr"] = v["tpr"]
    return row
