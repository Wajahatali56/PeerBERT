"""
Paper-level splitting and caps. Individual reviews are never split: every
function selects or assigns WHOLE papers (group_col, default paper_key).

- split_by_paper: train/val split of a train pool, stratified by conference-year.
- cap_by_papers: random whole-paper subset whose (counted) rows fit a budget,
  allocated proportionally per conference-year. Used for train/val/test caps and
  for the baseline subset, which is drawn from the (capped) model test set so it
  is nested inside it.
- check_invariants: re-checkable conditions on saved manifests (--verify-only).
"""

import numpy as np
import pandas as pd


def _papers_by_stratum(df, group_col, strata_col):
    strata = df[strata_col] if strata_col and strata_col in df.columns else pd.Series("all", index=df.index)
    pairs = pd.DataFrame({"g": df[group_col], "s": strata}).drop_duplicates()
    if pairs["g"].duplicated().any():
        raise ValueError(f"papers span several {strata_col} values: "
                         f"{pairs.loc[pairs['g'].duplicated(), 'g'].head(5).tolist()}")
    first = pairs
    return {s: sorted(g["g"].tolist()) for s, g in first.groupby("s", sort=True)}


def split_by_paper(df, val_fraction, seed, group_col="paper_key", strata_col="conf_year"):
    """Return (train, val). About val_fraction of the papers of each stratum go to val."""
    if not 0 < val_fraction < 1:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")
    rng = np.random.default_rng(seed)
    val_papers = set()
    for _, papers in _papers_by_stratum(df, group_col, strata_col).items():
        papers = list(papers)
        rng.shuffle(papers)
        n_val = int(round(val_fraction * len(papers)))
        if len(papers) > 1:
            n_val = min(max(n_val, 1), len(papers) - 1)
        val_papers.update(papers[:n_val])
    in_val = df[group_col].isin(val_papers)
    return df.loc[~in_val].copy(), df.loc[in_val].copy()


def cap_by_papers(df, max_rows, seed, group_col="paper_key", strata_col="conf_year", count_mask=None):
    """
    Whole-paper subset with at most max_rows counted rows (all rows if max_rows is None
    or the frame already fits). count_mask (bool Series) selects which rows count
    toward the budget; uncounted rows of chosen papers are kept too.
    """
    counted = count_mask if count_mask is not None else pd.Series(True, index=df.index)
    total = int(counted.sum())
    if max_rows is None or total <= max_rows:
        return df.copy()
    rng = np.random.default_rng(seed)
    sizes = counted.groupby(df[group_col]).sum()
    strata = _papers_by_stratum(df, group_col, strata_col)
    stratum_rows = {s: int(sizes.loc[p].sum()) for s, p in strata.items()}

    chosen, used, leftovers = set(), 0, []
    for s, papers in strata.items():
        budget = int(np.floor(max_rows * stratum_rows[s] / total))
        papers = list(papers)
        rng.shuffle(papers)
        got = 0
        for p in papers:
            n = int(sizes[p])
            if got + n <= budget:
                chosen.add(p)
                got += n
            else:
                leftovers.append(p)
        used += got
    rng.shuffle(leftovers)                       # fill the rounding remainder
    for p in leftovers:
        n = int(sizes[p])
        if used + n <= max_rows:
            chosen.add(p)
            used += n
    return df.loc[df[group_col].isin(chosen)].copy()


def paper_overlaps(parts, group_col="paper_key"):
    """{"a|b": n_shared_papers} for every pair of partitions."""
    names = list(parts)
    out = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            out[f"{a}|{b}"] = len(set(parts[a][group_col]) & set(parts[b][group_col]))
    return out


def check_invariants(parts, caps, disjoint=("train", "val", "test"), baseline_count_mask=None):
    """
    Conditions every saved manifest set must satisfy. Returns a list of failures (empty = PASS).
    parts: name -> frame; caps: name -> max_rows or None.
    """
    fails = []
    for name, df in parts.items():
        if df["row_id"].duplicated().any():
            fails.append(f"{name}: duplicate row_id")
        if df["label"].isna().any():
            fails.append(f"{name}: missing labels")
        if (df["text"].fillna("") == "").any():
            fails.append(f"{name}: empty text")
    present = [d for d in disjoint if d in parts]
    for pair, n in paper_overlaps({d: parts[d] for d in present}).items():
        if n:
            fails.append(f"paper overlap {pair}: {n}")
    for name in ("train", "val"):
        if name in parts and "extended" in parts:
            n = len(set(parts[name]["paper_key"]) & set(parts["extended"]["paper_key"]))
            if n:
                fails.append(f"paper overlap {name}|extended: {n} (should have been excluded)")
    if "test_baseline" in parts and "test" in parts:
        if not set(parts["test_baseline"]["row_id"]) <= set(parts["test"]["row_id"]):
            fails.append("test_baseline is not nested inside test")
    if "extended_baseline" in parts and "extended" in parts:
        if not set(parts["extended_baseline"]["row_id"]) <= set(parts["extended"]["row_id"]):
            fails.append("extended_baseline is not nested inside extended")
    for name, cap in caps.items():
        if cap is None or name not in parts:
            continue
        df = parts[name]
        n = int(baseline_count_mask(df).sum()) if (name == "test_baseline" and baseline_count_mask) else len(df)
        if n > cap:
            fails.append(f"{name}: {n} counted rows exceed cap {cap}")
    return fails
