"""
Statistics for comparing detectors.

- cluster_bootstrap_ci: percentile CI where PAPERS (not reviews) are resampled with
  replacement, because reviews of one paper are correlated.
- paired_cluster_bootstrap: difference of a metric between two methods scored on the
  SAME examples; same paper resample for both methods; two-sided bootstrap p-value.
- mcnemar_exact: paired test on correct/incorrect decisions (exact binomial).
- holm: Holm-Bonferroni adjustment for a family of comparisons.

Assumptions and limits (state them in the thesis):
- papers are treated as independent clusters; dependence across papers (same
  reviewer, same venue, same generator) is not modelled;
- percentile bootstrap CIs can be too narrow with few clusters (< ~30 papers);
- McNemar ignores paper clustering (reviews treated as independent) and is
  therefore reported next to, not instead of, the paired cluster bootstrap;
- bootstrap p-values are bounded below by 1/(n_boot+1).
"""

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score


def auc_fn(y, s):
    return roc_auc_score(y, s) if len(np.unique(y)) == 2 else np.nan


def macro_f1_fn(threshold=0.5):
    def f(y, s):
        return f1_score(y, (np.asarray(s) >= threshold).astype(int), average="macro",
                        labels=[0, 1], zero_division=0)
    return f


def _cluster_index(clusters):
    clusters = np.asarray(clusters)
    _, inv = np.unique(clusters, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    bounds = np.flatnonzero(np.diff(inv[order])) + 1
    return np.split(order, bounds)


def _resample(groups, rng):
    pick = rng.integers(0, len(groups), len(groups))
    return np.concatenate([groups[i] for i in pick])


def cluster_bootstrap_ci(y, s, clusters, metric_fn=auc_fn, n_boot=1000, alpha=0.05, seed=0):
    y, s = np.asarray(y), np.asarray(s, dtype=float)
    groups = _cluster_index(clusters)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = _resample(groups, rng)
        boots.append(metric_fn(y[idx], s[idx]))
    boots = np.asarray(boots, dtype=float)
    ok = boots[np.isfinite(boots)]
    est = metric_fn(y, s)
    return {
        "estimate": float(est) if np.isfinite(est) else None,
        "ci_low": float(np.quantile(ok, alpha / 2)) if len(ok) else None,
        "ci_high": float(np.quantile(ok, 1 - alpha / 2)) if len(ok) else None,
        "alpha": alpha, "n_boot": n_boot, "valid_boot": int(len(ok)), "n_clusters": len(groups),
    }


def paired_cluster_bootstrap(y, s_a, s_b, clusters, metric_fn=auc_fn, n_boot=1000, alpha=0.05, seed=0):
    """Metric(a) - Metric(b) on shared examples. Returns estimate, CI and two-sided p-value."""
    y = np.asarray(y)
    s_a, s_b = np.asarray(s_a, dtype=float), np.asarray(s_b, dtype=float)
    if not (len(y) == len(s_a) == len(s_b)):
        raise ValueError("paired test needs the same examples for both methods")
    groups = _cluster_index(clusters)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        idx = _resample(groups, rng)
        diffs.append(metric_fn(y[idx], s_a[idx]) - metric_fn(y[idx], s_b[idx]))
    d = np.asarray(diffs, dtype=float)
    d = d[np.isfinite(d)]
    est = metric_fn(y, s_a) - metric_fn(y, s_b)
    if len(d) == 0:
        return {"diff": None, "ci_low": None, "ci_high": None, "p_value": None, "n_boot": n_boot}
    p = 2 * min((np.sum(d <= 0) + 1) / (len(d) + 1), (np.sum(d >= 0) + 1) / (len(d) + 1))
    return {
        "diff": float(est), "ci_low": float(np.quantile(d, alpha / 2)),
        "ci_high": float(np.quantile(d, 1 - alpha / 2)), "p_value": float(min(p, 1.0)),
        "n_boot": n_boot, "valid_boot": int(len(d)), "n_clusters": len(groups),
    }


def mcnemar_exact(y, pred_a, pred_b):
    from scipy.stats import binomtest
    y, a, b = map(np.asarray, (y, pred_a, pred_b))
    ca, cb = a == y, b == y
    n01 = int((ca & ~cb).sum())     # a right, b wrong
    n10 = int((~ca & cb).sum())     # a wrong, b right
    n = n01 + n10
    p = 1.0 if n == 0 else float(binomtest(n01, n, 0.5).pvalue)
    return {"a_right_b_wrong": n01, "a_wrong_b_right": n10, "p_value": p}


def holm(pvalues):
    """dict name -> p  ->  dict name -> Holm-adjusted p (monotone, capped at 1)."""
    items = [(k, v) for k, v in pvalues.items() if v is not None]
    items.sort(key=lambda kv: kv[1])
    m, running, out = len(items), 0.0, {}
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    for k, v in pvalues.items():
        out.setdefault(k, None)
    return out
