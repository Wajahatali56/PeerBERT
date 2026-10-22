"""Stage 1 tests on toy data: loaders, schema guard, splits, caps, audit, prepare script."""

import importlib.util

import pandas as pd
import pytest

from conftest import ROOT, intel_frame, semdetect_raw
from src.data.intellabs_loader import SchemaError, build_text, load_subset
from src.data.leakage_audit import assert_audit_passed, run_audit
from src.data.sem_detect_loader import build_frame, check_expected
from src.data.splits import cap_by_papers, check_invariants, paper_overlaps, split_by_paper


def _load(cfg, hub, subset="calibration"):
    return load_subset(cfg, subset, lister=hub.lister, fetcher=hub.fetcher, use_cache=False)


# ---------------------------------------------------------------- IntelLabs
def test_human_rows_deduplicated_and_logged(cfg, hub):
    for gen in ("claude", "gpt4o"):
        hub.add("calibration", gen, "ICLR", 2021, intel_frame("ICLR", 2021, gen))
    df, stats, log = _load(cfg, hub)
    assert (df["label"] == 0).sum() == 12            # 6 papers x 2 reviews, kept once
    assert (df["label"] == 1).sum() == 24            # one AI row per review per generator
    assert stats["human_duplicate_copies_dropped"] == 12
    assert stats["human_copies_with_different_text"] == 0
    assert stats["human_to_ai_ratio"] == "1:2.00"
    assert (log.frame()["reason"] == "human_duplicate_copy").sum() == 12
    assert df["row_id"].is_unique
    assert set(df.loc[df.label == 0, "generator"]) == {"human"}


def test_human_copy_with_different_text_is_counted(cfg, hub):
    hub.add("calibration", "claude", "ICLR", 2021, intel_frame("ICLR", 2021, "claude"))
    hub.add("calibration", "gpt4o", "ICLR", 2021,
            intel_frame("ICLR", 2021, "gpt4o", human_text=lambda p, r: f"edited {p} {r}"))
    _, stats, log = _load(cfg, hub)
    assert stats["human_copies_with_different_text"] == 12
    assert (log.frame()["detail"] == "text_differs_from_first_copy").sum() == 12


def test_text_built_from_allow_list_and_leaky_fields_excluded(cfg, hub):
    hub.add("test", "claude", "ICLR", 2024, intel_frame("ICLR", 2024, "claude", schema="ssq"))
    df, stats, _ = _load(cfg, hub, "test")
    t = df.iloc[0]["text"]
    assert t.startswith("Summary:") and "\n\nStrengths" in t and "\n\nWeaknesses" in t
    assert "xxx" not in t                                # code_of_conduct never used
    assert not t.endswith("\n")                          # empty `questions` skipped
    st = stats["file_stats"][0]
    assert st["never_field_mean_chars_by_label"]["code_of_conduct"] == {"0": 168.0, "1": 3.0}


def test_build_text_order_follows_config():
    df = pd.DataFrame({"b": ["B"], "a": ["A"]})
    assert build_text(df, ["a", "b"]).iloc[0] == "A\n\nB"


def test_prefixed_subfields_excluded_not_fatal(cfg, hub):
    extra = lambda label: {"strengths.strength_1": "long flattened strength text " * 20}  # noqa: E731
    hub.add("test", "claude", "NeurIPS", 2024, intel_frame("NeurIPS", 2024, "claude", schema="ssq", extra=extra))
    df, stats, _ = _load(cfg, hub, "test")
    assert "long flattened" not in df.iloc[0]["text"]
    assert stats["files_with_prefixed_fields"]


def test_schema_guard_collects_all_problems_then_raises(cfg, hub):
    no_text = intel_frame("ICLR", 2022, "claude").drop(columns=["review"])
    unknown = intel_frame("ICLR", 2021, "claude", extra=lambda l: {"mystery_essay": "y" * 400})
    hub.add("calibration", "claude", "ICLR", 2022, no_text)
    hub.add("calibration", "claude", "ICLR", 2021, unknown)
    with pytest.raises(SchemaError) as e:
        _load(cfg, hub)
    msg = str(e.value)
    assert "ICLR2022" in msg and "no allow-listed text field" in msg
    assert "ICLR2021" in msg and "mystery_essay" in msg


def test_label_custom_id_mismatch_raises(cfg, hub):
    bad = intel_frame("ICLR", 2021, "claude")
    bad.loc[0, "label"] = 1                                # human custom_id but label 1
    hub.add("calibration", "claude", "ICLR", 2021, bad)
    with pytest.raises(SchemaError, match="label disagrees with custom_id"):
        _load(cfg, hub)


def test_conference_column_mismatch_raises(cfg, hub):
    bad = intel_frame("ICLR", 2021, "claude")
    bad["conference"] = "NeurIPS2021"
    hub.add("calibration", "claude", "ICLR", 2021, bad)
    with pytest.raises(SchemaError, match="disagrees with filename"):
        _load(cfg, hub)


def test_paper_key_is_composite(cfg, hub):
    hub.add("calibration", "claude", "ICLR", 2021, intel_frame("ICLR", 2021, "claude"))
    hub.add("calibration", "claude", "ICLR", 2022, intel_frame("ICLR", 2022, "claude"))
    df, _, _ = _load(cfg, hub)
    assert df["paper_id_raw"].nunique() == 6                # same raw ids in both years
    assert df["paper_key"].nunique() == 12
    assert df["paper_key"].str.match(r"^ICLR_202[12]_\d+$").all()


# ---------------------------------------------------------------- Sem-Detect
def test_semdetect_frame(cfg):
    raw = semdetect_raw()
    raw.loc[0, "clean_review"] = "   "                      # invalid -> dropped and logged
    dup = raw[raw.split == "test"].iloc[[0]].assign(review_id="dupid")
    raw = pd.concat([raw, dup], ignore_index=True)           # duplicate test text: flagged, kept
    df, stats, log = build_frame(raw, cfg["data"]["semdetect"])
    assert len(df) == len(raw) - 1
    assert log.frame()["reason"].tolist() == ["empty_text"]
    assert stats["duplicate_text_rows_within_split (flagged, kept)"]["test"] == 2
    assert set(df["label"]) == {0, 1, 2}
    assert stats["paper_key_overlap_train_test"] == 0
    assert stats["naive_paper_id_overlap_train_test"] == 0
    assert df["paper_key"].str.match(r"^ICLR_202[34]_\d+$").all()


def test_semdetect_naive_key_collision_vs_composite(cfg):
    raw = semdetect_raw()
    test = raw["split"] == "test"
    raw.loc[test, "paper_id"] = raw.loc[test, "paper_id"] - 100      # reuse train ids...
    raw.loc[test, "conference"] = "NeurIPS"                          # ...in another venue
    _, stats, _ = build_frame(raw, cfg["data"]["semdetect"])
    assert stats["naive_paper_id_overlap_train_test"] > 0
    assert stats["paper_key_overlap_train_test"] == 0


def test_semdetect_unknown_class_raises(cfg):
    raw = semdetect_raw()
    raw.loc[0, "class"] = "mixed"
    with pytest.raises(ValueError, match="unknown class"):
        build_frame(raw, cfg["data"]["semdetect"])


def test_semdetect_expected_facts_reported(cfg):
    scfg = dict(cfg["data"]["semdetect"], expected={"rows": 22165, "class_counts": {"human": 3065}})
    w = check_expected(semdetect_raw(), scfg)
    assert any("rows" in x for x in w) and any("class=human" in x for x in w)


# ---------------------------------------------------------------- splits/caps
def _toy_pool(n_papers=40):
    rows = []
    for p in range(n_papers):
        for r in range(1 + p % 4):
            rows.append({"paper_key": f"C_{2021 + p % 2}_{p}", "conf_year": f"C{2021 + p % 2}",
                         "row_id": f"{p}-{r}", "label": r % 2, "text": "t"})
    return pd.DataFrame(rows)


def test_split_by_paper_is_disjoint_and_deterministic():
    df = _toy_pool()
    tr, va = split_by_paper(df, 0.15, seed=42)
    tr2, va2 = split_by_paper(df, 0.15, seed=42)
    assert paper_overlaps({"train": tr, "val": va}) == {"train|val": 0}
    assert len(tr) + len(va) == len(df)
    assert set(va["paper_key"]) == set(va2["paper_key"])
    assert set(va["conf_year"]) == {"C2021", "C2022"}        # stratified


def test_cap_keeps_whole_papers_and_nests():
    df = _toy_pool()
    capped = cap_by_papers(df, 30, seed=1)
    assert 0 < len(capped) <= 30
    full_sizes = df.groupby("paper_key").size()
    got = capped.groupby("paper_key").size()
    assert (got.values == full_sizes.loc[got.index].values).all()
    base = cap_by_papers(capped, 10, seed=1)
    assert set(base["row_id"]) <= set(capped["row_id"])
    assert cap_by_papers(df, None, seed=1).shape == df.shape


def test_cap_count_mask_counts_only_selected_rows():
    df = _toy_pool()
    mask = df["label"] == 0
    capped = cap_by_papers(df, 10, seed=3, count_mask=mask)
    assert (capped["label"] == 0).sum() <= 10
    assert (capped["label"] == 1).sum() > 0


def test_invariants_detect_problems():
    df = _toy_pool()
    tr, va = split_by_paper(df, 0.2, seed=0)
    assert check_invariants({"train": tr, "val": va}, {}) == []
    fails = check_invariants({"train": tr, "val": pd.concat([va, tr.iloc[:1]])}, {"train": 1})
    assert any("paper overlap" in f for f in fails)
    assert any("exceed cap" in f for f in fails)


# ---------------------------------------------------------------- audit
def _parts(cfg, hub):
    for gen in ("claude", "gpt4o"):
        hub.add("calibration", gen, "ICLR", 2021, intel_frame("ICLR", 2021, gen, n_papers=10))
        hub.add("test", gen, "ICLR", 2021, intel_frame("ICLR", 2021, gen, n_papers=5, pid_offset=500))
    pool, _, _ = _load(cfg, hub)
    test, _, _ = _load(cfg, hub, "test")
    tr, va = split_by_paper(pool, 0.2, seed=0)
    return {"train": tr, "val": va, "test": test}


def test_audit_passes_clean_data(cfg, hub):
    rep = run_audit(_parts(cfg, hub), "intellabs", cfg["audit"])
    assert rep["status"] in ("PASS", "WARN")
    assert rep["summary"]["paper_overlap"] == "PASS"
    assert rep["summary"]["review_pair_integrity"] == "PASS"
    assert rep["summary"]["exact_duplicate_text"] == "PASS"
    assert "per_generator" in rep["checks"][-1]["results"]["test"]


def test_length_only_auroc_flags_length_shortcut(cfg):
    from src.data.leakage_audit import check_length_auroc
    df = pd.DataFrame({"label": [0] * 5 + [1] * 5, "text": ["a" * 10] * 5 + ["b" * 99] * 5,
                       "generator": ["human"] * 5 + ["gpt4o"] * 5, "conf_year": "ICLR2021"})
    res = check_length_auroc({"test": df}, "intellabs", 0.75)
    assert res["status"] == "WARN" and res["worst"] == 1.0
    assert res["results"]["test"]["per_generator"]["gpt4o"]["auroc"] == 1.0


def test_audit_fails_on_paper_overlap_and_duplicates(cfg, hub):
    parts = _parts(cfg, hub)
    leaked = parts["train"].iloc[:2].copy()
    leaked["row_id"] = leaked["row_id"] + "-copy"
    parts["test"] = pd.concat([parts["test"], leaked], ignore_index=True)
    rep = run_audit(parts, "intellabs", cfg["audit"])
    assert rep["status"] == "FAIL"
    assert rep["summary"]["paper_overlap"] == "FAIL"
    assert rep["summary"]["exact_duplicate_text"] == "FAIL"


def test_audit_gate(tmp_path):
    with pytest.raises(RuntimeError, match="No leakage audit"):
        assert_audit_passed(tmp_path)
    (tmp_path / "audit_status.json").write_text('{"status": "FAIL"}')
    with pytest.raises(RuntimeError, match="blocked"):
        assert_audit_passed(tmp_path)


# ---------------------------------------------------------------- end to end
def _prepare_module():
    spec = importlib.util.spec_from_file_location("prep", ROOT / "scripts" / "02_prepare_splits.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_prepare_and_verify_end_to_end(cfg, hub, monkeypatch, tmp_path):
    import yaml
    for gen in ("claude", "gpt4o"):
        for year, off in ((2021, 0), (2022, 0)):
            hub.add("calibration", gen, "ICLR", year, intel_frame("ICLR", year, gen, n_papers=12))
            hub.add("test", gen, "ICLR", year, intel_frame("ICLR", year, gen, n_papers=8, pid_offset=500))
    cfg["data"]["intellabs"].update(train_max_rows=40, model_eval_max_rows=50, baseline_eval_max_rows=12)
    cfg["data"]["semdetect"].update(baseline_eval_max_rows=6)
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    mod = _prepare_module()
    real_load = mod.load_subset
    monkeypatch.setattr(mod, "load_subset", lambda c, s, use_cache=True: real_load(
        c, s, lister=hub.lister, fetcher=hub.fetcher, use_cache=use_cache))
    real_sd = mod.load_semdetect
    monkeypatch.setattr(mod, "load_semdetect", lambda c, use_cache=True: real_sd(
        c, frame_loader=lambda r, n: (semdetect_raw(20, 10), ["reviews/toy.parquet"]), use_cache=use_cache))

    assert mod.main(["--config", str(cfg_path), "--audit-only"]) == 0
    assert mod.main(["--config", str(cfg_path)]) == 0
    assert mod.main(["--config", str(cfg_path), "--verify-only"]) == 0

    from src.data.common import manifest_dir
    mdir = manifest_dir(cfg, "semdetect")
    assert mdir.name == "sem_detect"
    base = pd.read_parquet(mdir / "test_baseline.parquet")
    assert base["label"].isin([0, 1]).sum() <= 6 and (base["label"] == 2).any()
    excl = pd.read_parquet(manifest_dir(cfg, "intellabs") / "exclusions.parquet")
    assert {"human_duplicate_copy", "compute_cap"} <= set(excl["reason"])
    train = pd.read_parquet(manifest_dir(cfg, "intellabs") / "train.parquet")
    assert len(train) <= 40

    # tamper: a test paper leaks into train -> verify must fail
    test = pd.read_parquet(manifest_dir(cfg, "intellabs") / "test.parquet")
    pd.concat([train, test.iloc[:1].assign(row_id="x")]).to_parquet(
        manifest_dir(cfg, "intellabs") / "train.parquet", index=False)
    assert mod.main(["--config", str(cfg_path), "--verify-only", "--dataset", "intellabs"]) == 3
