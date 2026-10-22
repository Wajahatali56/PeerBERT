"""Stage 2 tests: metrics, thresholds, predictions, aggregation, statistics, errors."""

import numpy as np
import pandas as pd
import pytest

from src.evaluation.aggregate import (
    by_generator, by_length_bin, comparison_row, save_comparison_table, seed_summary,
)
from src.evaluation.error_analysis import error_breakdown, fp_fn_tables
from src.evaluation.metrics import binary_metrics, calibrate_threshold, multiclass_metrics, tpr_at_fpr
from src.evaluation.predict import binary_view, prediction_frame, softmax
from src.evaluation.stats import (
    auc_fn, cluster_bootstrap_ci, holm, mcnemar_exact, paired_cluster_bootstrap,
)


def test_binary_metrics_known_values():
    y = [0, 0, 0, 1, 1, 1]
    s = [0.1, 0.6, 0.2, 0.9, 0.4, 0.8]
    m = binary_metrics(y, s)
    assert m["confusion_matrix"]["matrix"] == [[2, 1], [1, 2]]
    assert m["accuracy"] == pytest.approx(4 / 6, abs=1e-6)
    assert m["precision"] == pytest.approx(2 / 3, abs=1e-6) and m["recall"] == pytest.approx(2 / 3, abs=1e-6)
    assert m["ai_tpr"] == pytest.approx(2 / 3, abs=1e-6) and m["ai_fpr"] == pytest.approx(1 / 3, abs=1e-6)
    assert m["roc_auc"] == pytest.approx(8 / 9, abs=1e-6)
    assert m["per_class"]["human"]["support"] == 3


def test_binary_metrics_single_class_and_zero_division():
    m = binary_metrics([1, 1, 1], [0.2, 0.3, 0.1])
    assert m["roc_auc"] is None and m["pr_auc"] is None and m["notes"]
    assert m["precision"] == 0.0 and m["ai_fpr"] is None
    with pytest.raises(ValueError):
        binary_metrics([0, 1], [0.1, np.nan])


def test_tpr_at_fpr_support_rule():
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(200), np.ones(200)]
    s = np.r_[rng.normal(0, 1, 200), rng.normal(2, 1, 200)]
    r = tpr_at_fpr(y, s, min_neg_events=10)
    assert r["1%"]["supported"] is False and r["1%"]["tpr"] is None       # 200*0.01 = 2 < 10
    assert r["5%"]["supported"] and r["10%"]["supported"]
    assert r["5%"]["tpr"] <= r["10%"]["tpr"]
    perfect = tpr_at_fpr(np.r_[np.zeros(1000), np.ones(10)], np.r_[np.zeros(1000), np.ones(10)])
    assert perfect["1%"]["tpr"] == 1.0


def test_multiclass_metrics():
    y = [0, 1, 2, 2]
    P = [[.8, .1, .1], [.1, .8, .1], [.1, .1, .8], [.6, .2, .2]]
    m = multiclass_metrics(y, P, ("human", "ai", "rewrite"))
    assert m["accuracy"] == 0.75 and m["confusion_matrix"]["matrix"][2] == [1, 0, 1]
    assert m["macro_ovr_roc_auc"] is not None
    m2 = multiclass_metrics([0, 0], [[.9, .05, .05], [.2, .7, .1]], ("human", "ai", "rewrite"))
    assert m2["macro_ovr_roc_auc"] is None and m2["notes"]


def test_calibrate_threshold_uses_given_scores_only():
    y = np.array([0, 0, 0, 1, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.35, 0.4, 0.5])
    t = calibrate_threshold(y, s)
    assert 0.3 < t <= 0.35
    assert binary_metrics(y, s, threshold=t)["macro_f1"] == 1.0
    assert calibrate_threshold([1, 1], [0.2, 0.3]) == 0.5


def test_prediction_frames_and_binary_view():
    meta = pd.DataFrame({"row_id": list("abcd"), "label": [0, 1, 2, 1], "label_name": ["human", "ai", "rewrite", "ai"],
                         "paper_key": ["p1", "p1", "p2", "p2"], "conf_year": "C", "generator": "x"})
    P = softmax(np.array([[2., 0, 0], [0, 2., 0], [0, 0, 2.], [1., 0.5, 3.]]))
    p3 = prediction_frame(meta, proba=P, class_names=("human", "ai", "rewrite"))
    assert p3["pred"].tolist() == [0, 1, 2, 2]
    b = binary_view(p3)
    assert len(b) == 3 and set(b["label"]) == {0, 1}
    assert b.loc[b.row_id == "d", "pred"].item() == 0          # human 1.0 > ai 0.5 when rewrite removed
    pb = prediction_frame(meta.iloc[:2], score=[0.2, 0.7])
    assert pb["pred"].tolist() == [0, 1]


def test_by_generator_shares_humans():
    pred = pd.DataFrame({
        "label": [0, 0, 1, 1, 1], "score": [.1, .2, .9, .8, .3],
        "generator": ["human", "human", "a", "b", "b"], "conf_year": ["X", "Y", "X", "Y", "Y"],
    })
    r = by_generator(pred)
    assert r["a"]["n"] == 2 and r["b"]["n"] == 3


def test_length_bins_cover_every_row():
    pred = pd.DataFrame({"label": [0, 1] * 6, "score": np.linspace(0, 1, 12)})
    r = by_length_bin(pred, np.arange(100, 1300, 100))
    assert len(r) == 3 and sum(v["n"] for v in r.values()) == 12


def test_seed_summary_and_table(tmp_path):
    s = seed_summary([{"macro_f1": .8}, {"macro_f1": .9}, {"macro_f1": .85}])
    assert s["macro_f1"]["mean"] == pytest.approx(.85) and s["macro_f1"]["std"] == pytest.approx(.05)
    m = binary_metrics([0, 1], [.2, .8])
    rows = [comparison_row("LogRank", "zero-shot detector", "test_baseline", "0.5", m),
            comparison_row("PeerBERT", "trained classifier", "test_baseline", "0.5", m)]
    df = save_comparison_table(rows, tmp_path)
    assert df["type"].tolist()[0] == "trained classifier"
    assert (tmp_path / "comparison_table.csv").exists() and (tmp_path / "comparison_table.json").exists()
    with pytest.raises(ValueError):
        comparison_row("X", "magic", "t", "0.5", m)


def test_cluster_bootstrap_and_paired_test():
    rng = np.random.default_rng(1)
    n = 400
    clusters = np.repeat(np.arange(100), 4)
    y = rng.integers(0, 2, n)
    good = y + rng.normal(0, 0.5, n)
    bad = y + rng.normal(0, 3.0, n)
    ci = cluster_bootstrap_ci(y, good, clusters, auc_fn, n_boot=200)
    assert ci["ci_low"] <= ci["estimate"] <= ci["ci_high"] and ci["n_clusters"] == 100
    d = paired_cluster_bootstrap(y, good, bad, clusters, auc_fn, n_boot=200)
    assert d["diff"] > 0 and d["ci_low"] > 0 and d["p_value"] < 0.05
    same = paired_cluster_bootstrap(y, good, good, clusters, auc_fn, n_boot=50)
    assert same["diff"] == 0 and same["p_value"] == 1.0


def test_mcnemar_and_holm():
    y = np.ones(30)
    r = mcnemar_exact(y, np.ones(30), np.r_[np.zeros(15), np.ones(15)])
    assert r["a_right_b_wrong"] == 15 and r["p_value"] < 0.001
    adj = holm({"a": 0.01, "b": 0.04, "c": 0.03, "d": None})
    assert adj["a"] == pytest.approx(0.03) and adj["c"] == pytest.approx(0.06)
    assert adj["b"] == pytest.approx(0.06) and adj["d"] is None


def test_error_analysis():
    pred = pd.DataFrame({"row_id": list("abcd"), "label": [0, 0, 1, 1], "pred": [1, 0, 0, 1],
                         "score": [.9, .1, .2, .8], "conf_year": ["X", "X", "Y", "Y"],
                         "label_name": ["human", "human", "ai", "ai"]})
    t = fp_fn_tables(pred, texts=pd.Series({"a": "text a"}))
    assert t["false_positives"]["row_id"].tolist() == ["a"] and t["false_positives"]["text"].item() == "text a"
    assert t["false_negatives"]["confidence"].item() == pytest.approx(0.8)
    b = error_breakdown(pred)
    assert b["conf_year"]["X"]["error_rate"] == 0.5
