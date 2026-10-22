"""
Leakage audit. Runs on the FULL partitions (before caps) and blocks training on FAIL.

Checks (status PASS / WARN / FAIL / INFO):
  label_values           labels outside the dataset's label set            -> FAIL
  duplicate_row_ids      within or across partitions                       -> FAIL
  paper_overlap          composite paper_key across train/val/test         -> FAIL
                         naive key (paper_id_raw)                           -> INFO
                         submission_id (Sem-Detect cross-check)             -> FAIL
                         extended vs train/val (must be 0 after exclusion)  -> FAIL
  review_pair_integrity  IntelLabs: one human per review, never split       -> FAIL/WARN
  exact_duplicate_text   within partition -> WARN; eval rows whose text is in
                         train: fraction > configured threshold -> FAIL, >0 -> WARN
  near_duplicate_text    TF-IDF cosine on a sample                          -> INFO
  distributions          class / conference-year / generator                -> INFO
  length_only_auroc      text length as the only feature (overall and per
                         IntelLabs generator); >= threshold                 -> WARN
"""

from datetime import datetime, timezone

import pandas as pd

from .common import save_json, text_hash

LABEL_SETS = {"intellabs": {0, 1}, "semdetect": {0, 1, 2}}
TRAIN_SIDE = ("train",)
EVAL_SIDE = ("val", "test", "extended")


def _check(name, status, **details):
    return {"check": name, "status": status, **details}


def _hashes(df):
    return df["text_hash"] if "text_hash" in df.columns else df["text"].map(text_hash)


def check_labels(parts, dataset):
    allowed = LABEL_SETS[dataset]
    bad = {n: sorted(set(df["label"].dropna().astype(int)) - allowed) for n, df in parts.items()}
    bad = {n: v for n, v in bad.items() if v}
    return _check("label_values", "FAIL" if bad else "PASS", allowed=sorted(allowed), unexpected=bad)


def check_row_ids(parts):
    within = {n: int(df["row_id"].duplicated().sum()) for n, df in parts.items()}
    names = list(parts)
    across = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            n = len(set(parts[a]["row_id"]) & set(parts[b]["row_id"]))
            if n:
                across[f"{a}|{b}"] = n
    fail = any(within.values()) or bool(across)
    return _check("duplicate_row_ids", "FAIL" if fail else "PASS", within=within, across=across)


def check_paper_overlap(parts, dataset):
    core = [n for n in ("train", "val", "test") if n in parts]
    out = {"composite": {}, "naive_paper_id (info)": {}, "submission_id": {}, "extended": {}}
    fail = False
    for i, a in enumerate(core):
        for b in core[i + 1:]:
            pa, pb = parts[a], parts[b]
            n = len(set(pa["paper_key"]) & set(pb["paper_key"]))
            out["composite"][f"{a}|{b}"] = n
            out["naive_paper_id (info)"][f"{a}|{b}"] = len(set(pa["paper_id_raw"]) & set(pb["paper_id_raw"]))
            fail |= n > 0
            if "submission_id" in pa.columns and "submission_id" in pb.columns:
                m = len((set(pa["submission_id"]) - {""}) & (set(pb["submission_id"]) - {""}))
                out["submission_id"][f"{a}|{b}"] = m
                fail |= m > 0
    if "extended" in parts:
        for a in core:
            n = len(set(parts[a]["paper_key"]) & set(parts["extended"]["paper_key"]))
            out["extended"][f"{a}|extended"] = n
            if a in ("train", "val"):
                fail |= n > 0
    return _check("paper_overlap", "FAIL" if fail else "PASS", overlaps=out)


def check_review_pairs(parts, dataset):
    if dataset != "intellabs":
        return _check("review_pair_integrity", "INFO",
                      note="not applicable: Sem-Detect has no human<->AI pairing key (og_review_id is metadata)")
    all_rows = pd.concat([df.assign(_part=n) for n, df in parts.items()], ignore_index=True)
    key = all_rows["conf_year"] + "|" + all_rows["review_id"]
    # Partition names are unique per source subset; a review may only live in one of
    # the partitions derived from the same subset (train/val come from calibration).
    derived = all_rows["partition"] + "|" + key
    split_reviews = int((all_rows.groupby(derived)["_part"].nunique() > 1).sum())
    humans = all_rows["label"].eq(0).groupby(derived).sum()
    ais = all_rows["label"].eq(1).groupby(derived).sum()
    multi_human = int((humans > 1).sum())
    orphan_ai = int(((humans == 0) & (ais > 0)).sum())
    status = "FAIL" if (split_reviews or multi_human) else ("WARN" if orphan_ai else "PASS")
    return _check("review_pair_integrity", status,
                  reviews_split_across_partitions=split_reviews,
                  reviews_with_multiple_human_rows=multi_human,
                  ai_reviews_without_human_row=orphan_ai,
                  max_ai_rows_per_review=int(ais.max()) if len(ais) else 0)


def check_exact_duplicates(parts, fail_fraction):
    hashes = {n: _hashes(df) for n, df in parts.items()}
    within = {n: int(h.duplicated(keep=False).sum()) for n, h in hashes.items()}
    cross, status = {}, ("WARN" if any(within.values()) else "PASS")
    train_h = set().union(*(set(hashes[n]) for n in TRAIN_SIDE if n in hashes))
    pairs = [(t, e) for t in TRAIN_SIDE for e in EVAL_SIDE if t in hashes and e in hashes]
    if "val" in hashes and "test" in hashes:
        pairs.append(("val", "test"))
    for a, b in pairs:
        ref = train_h if a in TRAIN_SIDE else set(hashes[a])
        n = int(hashes[b].isin(ref).sum())
        frac = n / max(len(hashes[b]), 1)
        cross[f"{a}|{b}"] = {"rows": n, "fraction_of_" + b: round(frac, 6)}
        if frac > fail_fraction:
            status = "FAIL"
        elif n and status != "FAIL":
            status = "WARN"
    return _check("exact_duplicate_text", status, within_partition_rows=within, cross_partition=cross,
                  fail_fraction=fail_fraction)


def check_near_duplicates(parts, sample_size, threshold, seed):
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError:
        return _check("near_duplicate_text", "INFO", skipped="scikit-learn not installed")
    if "train" not in parts:
        return _check("near_duplicate_text", "INFO", skipped="no train partition")
    sample = {n: df.sample(min(sample_size, len(df)), random_state=seed)["text"].tolist()
              for n, df in parts.items() if len(df)}
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    vec.fit([t for texts in sample.values() for t in texts])
    xt = vec.transform(sample["train"])
    out = {}
    for n in EVAL_SIDE:
        if n not in sample:
            continue
        sims = (vec.transform(sample[n]) @ xt.T).max(axis=1).toarray().ravel()
        out[f"train|{n}"] = {"sampled": len(sims), f"cosine>={threshold}": int((sims >= threshold).sum()),
                             "max_cosine": round(float(sims.max()), 4) if len(sims) else None}
    return _check("near_duplicate_text", "INFO", sample_size=sample_size, results=out)


def check_distributions(parts):
    out = {}
    for n, df in parts.items():
        out[n] = {
            "rows": len(df), "papers": int(df["paper_key"].nunique()),
            "label": df["label_name"].value_counts().to_dict(),
            "conf_year": df["conf_year"].value_counts().sort_index().to_dict(),
            "generator": df["generator"].value_counts().to_dict(),
        }
    return _check("distributions", "INFO", partitions=out)


def _length_auc(df, pos, neg):
    from sklearn.metrics import roc_auc_score
    sub = df[df["label"].isin([pos, neg])]
    if sub["label"].nunique() < 2:
        return None
    auc = roc_auc_score((sub["label"] == pos).astype(int), sub["text"].str.len())
    return {"auroc": round(float(auc), 4), "max(auroc,1-auroc)": round(float(max(auc, 1 - auc)), 4),
            "n_pos": int((sub["label"] == pos).sum()), "n_neg": int((sub["label"] == neg).sum())}


def check_length_auroc(parts, dataset, warn_at):
    tasks = {"ai_vs_human": (1, 0)}
    if dataset == "semdetect":
        tasks["rewrite_vs_human"] = (2, 0)
    out, worst = {}, 0.0
    for n, df in parts.items():
        res = {t: _length_auc(df, p, q) for t, (p, q) in tasks.items()}
        if dataset == "intellabs":
            per_gen = {}
            for g in sorted(df.loc[df["label"] == 1, "generator"].unique()):
                ai = df[(df["label"] == 1) & (df["generator"] == g)]
                human = df[(df["label"] == 0) & df["conf_year"].isin(ai["conf_year"].unique())]
                per_gen[g] = _length_auc(pd.concat([human, ai]), 1, 0)
            res["per_generator"] = per_gen
        out[n] = res
    for res in out.values():
        for v in list(res.values()) + list(res.get("per_generator", {}).values()):
            if isinstance(v, dict) and "max(auroc,1-auroc)" in v:
                worst = max(worst, v["max(auroc,1-auroc)"])
    return _check("length_only_auroc", "WARN" if worst >= warn_at else "PASS",
                  warn_at=warn_at, worst=worst, results=out,
                  note="shortcut risk: a length-only detector; WARN does not block training")


def run_audit(parts, dataset, audit_cfg, seed=42, extra_info=None):
    """parts: partition name -> unified frame (full, uncapped). Returns the report dict."""
    checks = [
        check_labels(parts, dataset),
        check_row_ids(parts),
        check_paper_overlap(parts, dataset),
        check_review_pairs(parts, dataset),
        check_exact_duplicates(parts, float(audit_cfg.get("exact_dup_cross_fail_fraction", 0.001))),
        check_near_duplicates(parts, int(audit_cfg.get("near_dup_sample_size", 2000)),
                              float(audit_cfg.get("near_dup_cosine", 0.9)), seed),
        check_distributions(parts),
        check_length_auroc(parts, dataset, float(audit_cfg.get("length_auroc_warn", 0.75))),
    ]
    statuses = [c["status"] for c in checks]
    overall = "FAIL" if "FAIL" in statuses else ("WARN" if "WARN" in statuses else "PASS")
    return {
        "dataset": dataset, "status": overall,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "summary": {c["check"]: c["status"] for c in checks},
        "checks": checks, "info": extra_info or {},
    }


def save_audit(report, out_dir):
    save_json(report, out_dir / "leakage_audit.json")
    save_json({k: report[k] for k in ("dataset", "status", "created_utc", "summary")},
              out_dir / "audit_status.json")


def print_summary(report):
    print(f"\n=== Leakage audit [{report['dataset']}]: {report['status']} ===")
    for c in report["checks"]:
        print(f"  {c['status']:<5} {c['check']}")


def assert_audit_passed(out_dir):
    """Called by training scripts: refuses to train without a non-FAIL audit."""
    from .common import load_json
    p = out_dir / "audit_status.json"
    if not p.exists():
        raise RuntimeError(f"No leakage audit at {p}. Run scripts/02_prepare_splits.py first.")
    status = load_json(p)["status"]
    if status == "FAIL":
        raise RuntimeError(f"Leakage audit FAILED ({p}); training is blocked.")
    return status
