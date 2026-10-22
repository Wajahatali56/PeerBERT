"""Label mapping tests: labels come from data columns, never from folders or guesses."""

import pandas as pd

from conftest import intel_frame, semdetect_raw
from src.data.common import normalize_label
from src.data.intellabs_loader import parse_path
from src.data.sem_detect_loader import build_frame


def test_normalize_label_does_not_remap_unknown_values():
    assert normalize_label("0") == 0 and normalize_label("1.0") == 1 and normalize_label(1) == 1
    assert normalize_label("2") == "2"
    assert normalize_label(None) is None and normalize_label(float("nan")) is None


def test_config_label_maps(cfg):
    assert cfg["data"]["semdetect"]["label_map"] == {"human": 0, "ai": 1, "rewrite": 2}
    assert cfg["data"]["intellabs"]["ai_label_value"] == 1


def test_intellabs_labels_from_column_not_folder(cfg, hub):
    from src.data.intellabs_loader import load_subset
    hub.add("calibration", "claude", "ICLR", 2021, intel_frame("ICLR", 2021, "claude"))
    df, _, _ = load_subset(cfg, "calibration", lister=hub.lister, fetcher=hub.fetcher, use_cache=False)
    assert set(df["label"]) == {0, 1}
    assert (df.loc[df["label"] == 0, "row_id"].str.contains(":human-")).all()
    assert set(df.loc[df["label"] == 1, "generator"]) == {"claude"}
    assert dict(zip(df["label"], df["label_name"])) == {0: "human", 1: "ai"}


def test_parse_path():
    info = parse_path("test/gpt4o/NeurIPS2024.test.gpt-4o.csv")
    assert info["conf_year"] == "NeurIPS2024" and info["year"] == 2024 and info["model"] == "gpt-4o"
    assert parse_path("test/gpt4o/NeurIPS2024.calibration.gpt-4o.csv") is None   # subset mismatch
    assert parse_path("README.md") is None


def test_semdetect_classes_map_case_insensitively(cfg):
    raw = semdetect_raw()
    raw["class"] = raw["class"].str.upper()
    df, _, _ = build_frame(raw, cfg["data"]["semdetect"])
    assert dict(zip(df["label_name"], df["label"])) == {"human": 0, "ai": 1, "rewrite": 2}
    assert set(df.loc[df["label"] == 0, "generator"]) == {"human"}


def test_semdetect_og_review_id_is_metadata_only(cfg):
    df, stats, _ = build_frame(semdetect_raw(), cfg["data"]["semdetect"])
    assert "og_review_id" in df.columns and stats["og_review_id_distinct"] == 3
    assert isinstance(df, pd.DataFrame)
