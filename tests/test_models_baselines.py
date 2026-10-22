"""
Stage 3-8 tests with TINY RANDOM models built offline (no hub, CPU): dataset/truncation,
losses, trainer (R-Drop, resume/skip), token statistics vs brute force, detectors, and an
end-to-end smoke run of scripts 03 -> 05 -> 06 -> 08 -> 09 on toy manifests.
These check code paths only; no number here is a thesis result.
"""

import json
import math
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from conftest import ROOT  # noqa: E402

WORDS = ("the paper proposes a method novel results strong weak experiments clarity baseline "
         "we think authors should improve related work missing ablation table figure overall "
         "accept reject review model data training evaluation human ai rewrite").split()


def _tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import PreTrainedTokenizerFast
    specials = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    vocab = {t: i for i, t in enumerate(specials + WORDS)}
    tk = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    tk.post_processor = processors.TemplateProcessing(single="[CLS] $A [SEP]",
                                                      special_tokens=[("[CLS]", 2), ("[SEP]", 3)])
    return PreTrainedTokenizerFast(tokenizer_object=tk, pad_token="[PAD]", unk_token="[UNK]", cls_token="[CLS]",
                                   sep_token="[SEP]", mask_token="[MASK]", eos_token="[SEP]")


@pytest.fixture(scope="session")
def tiny(tmp_path_factory):
    from transformers import (AutoModelForCausalLM, AutoModelForSequenceClassification, GPT2Config,
                              ModernBertConfig)
    root = tmp_path_factory.mktemp("tiny")
    tok = _tokenizer()
    v = len(tok)
    torch.manual_seed(0)
    mb = ModernBertConfig(vocab_size=v, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                          num_attention_heads=2, max_position_embeddings=512, local_attention=16,
                          global_attn_every_n_layers=1, pad_token_id=0, cls_token_id=2, sep_token_id=3,
                          bos_token_id=2, eos_token_id=3, num_labels=2)
    AutoModelForSequenceClassification.from_config(mb).save_pretrained(root / "modernbert")
    tok.save_pretrained(root / "modernbert")
    g = GPT2Config(vocab_size=v, n_embd=32, n_layer=2, n_head=2, n_positions=512, bos_token_id=3,
                   eos_token_id=3, pad_token_id=0)
    for name in ("gpt", "gpt_b"):
        torch.manual_seed(1 if name == "gpt" else 2)
        AutoModelForCausalLM.from_config(g).save_pretrained(root / name)
        tok.save_pretrained(root / name)
    return {"modernbert": str(root / "modernbert"), "gpt": str(root / "gpt"), "gpt_b": str(root / "gpt_b"),
            "tok": tok}


def _texts(n, seed=0, ai=False):
    rng = np.random.default_rng(seed)
    pool = WORDS[:20] if ai else WORDS[10:]
    return [" ".join(rng.choice(pool, rng.integers(8, 40))) for _ in range(n)]


# ---------------------------------------------------------------- dataset / losses
def test_dataset_keeps_special_tokens_and_truncates(tiny):
    from src.models.dataset import ReviewDataset, special_template
    tok = tiny["tok"]
    assert special_template(tok) == ([2], [3])
    long = " ".join(WORDS * 3)
    ds = ReviewDataset([long, "the paper"], [1, 0], tok, max_length=12)
    assert ds.input_ids[0][0] == 2 and ds.input_ids[0][-1] == 3 and len(ds.input_ids[0]) == 12
    assert ds.input_ids[1] == [2, tok.convert_tokens_to_ids("the"), tok.convert_tokens_to_ids("paper"), 3]
    ht = ReviewDataset([long], [1], tok, max_length=12, truncation="head_tail", head_tokens=4)
    ids = tok(long, add_special_tokens=False)["input_ids"]
    assert ht.input_ids[0] == [2] + ids[:4] + ids[-6:] + [3]
    st = ds.truncation_stats()
    assert st["truncated_fraction"] == 0.5 and st["tokens_max"] == len(ids) + 2
    b = ds.collate([ds[0], ds[1]])
    assert b["input_ids"].shape == (2, 12) and b["attention_mask"][1].sum() == 4


def test_losses():
    from src.models.losses import PeerBERTLoss, class_weights, classification_loss, focal_loss, rdrop_kl
    w = class_weights([0, 1, 1, 1], 2)
    assert torch.allclose(w, torch.tensor([2.0, 2 / 3]))
    with pytest.raises(ValueError):
        class_weights([1, 1], 2)
    logits, y = torch.randn(6, 3), torch.tensor([0, 1, 2, 0, 1, 2])
    assert rdrop_kl(logits, logits).item() == pytest.approx(0.0, abs=1e-6)
    l2 = torch.randn(6, 3)
    assert rdrop_kl(logits, l2).item() == pytest.approx(rdrop_kl(l2, logits).item(), rel=1e-5)
    w3 = torch.tensor([1.0, 2.0, 0.5])
    assert focal_loss(logits, y, 0.0, w3).item() == pytest.approx(
        classification_loss(logits, y, "weighted_ce", w3).item(), rel=1e-5)
    loss = PeerBERTLoss("weighted_ce", w3, rdrop_alpha=1.0)
    total, parts = loss(logits, y, l2)
    assert total.item() == pytest.approx(parts["cls"] + parts["cons"], rel=1e-5)
    with pytest.raises(ValueError):
        loss(logits, y)


# ---------------------------------------------------------------- trainer
def _frames(n=24):
    rows = []
    for i in range(n):
        lab = i % 2
        rows.append({"row_id": f"r{i}", "text": _texts(1, i, ai=bool(lab))[0], "label": lab,
                     "label_name": ["human", "ai"][lab], "paper_key": f"P_{i // 2}", "conf_year": "C2021",
                     "generator": "x" if lab else "human"})
    return pd.DataFrame(rows)


def test_train_one_rdrop_resume_and_predict(tiny, tmp_path):
    from src.models.trainer import load_train_config, predict_proba, train_one
    tcfg = load_train_config(overrides={
        "model": {"name": tiny["modernbert"], "max_length": 32},
        "training": {"epochs": 2, "batch_size": 4, "grad_accum": 2, "eval_batch_size": 8, "num_workers": 0,
                     "early_stopping_patience": 5},
        "loss": {"type": "weighted_ce", "rdrop_alpha": 1.0}})
    df = _frames()
    info = train_one(df.iloc[:16], df.iloc[16:], tcfg, "intellabs", tmp_path / "run", seed=7)
    assert info["rdrop_passes_differ"] is True
    assert info["model_info"]["dropout_before"]["attention_dropout"] == 0.0
    assert info["model_info"]["dropout_after"]["attention_dropout"] == 0.1
    assert info["epochs_run"] == 2 and info["val_threshold"] is not None
    assert (tmp_path / "run" / "best" / "config.json").exists()
    assert not (tmp_path / "run" / "last_state.pt").exists()
    assert len(json.loads((tmp_path / "run" / "history.json").read_text())) == 2
    again = train_one(df.iloc[:16], df.iloc[16:], tcfg, "intellabs", tmp_path / "run", seed=7)
    assert again["epochs_run"] == 2                       # skipped via done.json
    proba, stats = predict_proba(tmp_path / "run" / "best", df["text"].tolist(), tcfg)
    assert proba.shape == (24, 2) and np.allclose(proba.sum(1), 1) and stats["n"] == 24


def test_rdrop_refused_without_dropout(tiny, tmp_path):
    from src.models.trainer import load_train_config, train_one
    tcfg = load_train_config(overrides={
        "model": {"name": tiny["modernbert"], "max_length": 32, "dropout_on": False},
        "training": {"epochs": 1, "batch_size": 4, "grad_accum": 1, "num_workers": 0},
        "loss": {"rdrop_alpha": 0.5}})
    df = _frames(8)
    with pytest.raises(RuntimeError, match="R-Drop"):
        train_one(df.iloc[:4], df.iloc[4:], tcfg, "intellabs", tmp_path / "r", seed=1)


# ---------------------------------------------------------------- token statistics / detectors
def test_token_stats_match_brute_force(tiny, tmp_path):
    from src.baselines.statistical import detectllm_lrr, logrank
    from src.baselines.statistical.token_stats import batch_stats, compute_token_stats, load_causal_lm
    from src.baselines.zero_shot import fast_detectgpt
    model, tok, dev = load_causal_lm(tiny["gpt"])
    texts = _texts(3, 5)
    st = batch_stats(model, tok, texts, 64, dev)
    ids = tok(texts[1], return_tensors="pt", truncation=True, max_length=64)["input_ids"]
    with torch.no_grad():
        lg = model(ids).logits[0, :-1].float()
    tgt = ids[0, 1:]
    lp = torch.log_softmax(lg, -1)
    ranks = [int((torch.argsort(lg[t], descending=True) == tgt[t]).nonzero()) + 1 for t in range(len(tgt))]
    assert st["mean_ll"][1] == pytest.approx(float(lp.gather(1, tgt[:, None]).mean()), rel=1e-4)
    assert st["mean_logrank"][1] == pytest.approx(float(np.mean(np.log(ranks))), rel=1e-4)
    assert st["gltr_top10"][1] == pytest.approx(np.mean(np.array(ranks) <= 10), rel=1e-6)
    frame = pd.DataFrame({"row_id": ["a", "b", "c"], "text": texts})
    stats = compute_token_stats(frame, tiny["gpt"], tmp_path, 64, 2, loaded=(model, tok, dev))
    again = compute_token_stats(frame.iloc[[2, 0]], tiny["gpt"], tmp_path, 64, 2, loaded=(model, tok, dev))
    assert again["row_id"].tolist() == ["c", "a"]
    assert again["mean_ll"].tolist() == stats.set_index("row_id").loc[["c", "a"], "mean_ll"].tolist()
    assert np.isfinite(logrank.score(stats)).all() and np.isfinite(detectllm_lrr.score(stats)).all()
    two = fast_detectgpt.score_two_models(texts, model, model, tok, dev, 64)
    assert np.allclose(fast_detectgpt.score_from_stats(stats), two, rtol=1e-4, atol=1e-4)


def test_binoculars_formula(tiny):
    from src.baselines.zero_shot import binoculars
    obs, perf, tok, dev = binoculars.load_pair(tiny["gpt"], tiny["gpt_b"], "float32")
    texts = _texts(2, 9)
    b, ppl, xppl = binoculars.batch_scores(obs, perf, tok, texts[:1], dev, 64)
    ids = tok(texts[0], return_tensors="pt")["input_ids"]
    with torch.no_grad():
        lo, lpf = obs(ids).logits[0, :-1], perf(ids).logits[0, :-1]
    nll = torch.nn.functional.cross_entropy(lpf, ids[0, 1:]).item()
    x = (-(lo.softmax(-1) * lpf.log_softmax(-1)).sum(-1)).mean().item()
    assert ppl[0] == pytest.approx(nll, rel=1e-4) and xppl[0] == pytest.approx(x, rel=1e-4)
    s = binoculars.score(texts, obs, perf, tok, dev, 64, 1)
    assert s[0] == pytest.approx(-nll / x, rel=1e-4)


def test_dnagpt_bscore_and_run(tiny, tmp_path):
    from src.baselines.statistical.token_stats import load_causal_lm
    from src.baselines.zero_shot import dnagpt
    y0 = "a b c d e f g h"
    assert dnagpt.bscore(y0, [y0], 4, 25) == pytest.approx(
        sum(n * math.log(n) * (8 - n + 1) / (8 * (8 - n + 1)) for n in range(4, 9)))
    assert dnagpt.bscore(y0, ["x y z w v u t s"]) == 0.0
    model, tok, dev = load_causal_lm(tiny["gpt"])
    cfg = {"gamma": 0.5, "k": 2, "max_prompt_tokens": 32, "max_new_tokens": 8, "temperature": 0.7,
           "ngram_min": 4, "ngram_max": 25, "wscore": True, "batch_size": 2}
    frame = pd.DataFrame({"row_id": ["a", "b", "c"], "text": _texts(3, 3)})
    b, w = dnagpt.run(frame, model, tok, dev, cfg, tmp_path / "regen.jsonl")
    assert len(b) == 3 and np.isfinite(b).all() and len(w) == 3
    assert len((tmp_path / "regen.jsonl").read_text().splitlines()) == 3
    b2, _ = dnagpt.run(frame, model, tok, dev, dict(cfg, wscore=False), tmp_path / "regen.jsonl")
    assert np.allclose(b, b2)                                   # served from cache


def test_radar_orientation_on_validation():
    from src.baselines.neural_detectors import radar
    y = np.array([0, 0, 1, 1])
    p = np.array([[.2, .8], [.3, .7], [.9, .1], [.8, .2]])     # column 0 tracks AI
    idx, d = radar.orient(p, y)
    assert idx == 0 and d["val_auroc_per_output"] == [1.0, 0.0]
    assert radar.orient(p[:, ::-1], y)[0] == 1


# ---------------------------------------------------------------- XAI
def test_xai_attention_ig_faithfulness(tiny):
    from src.models.modernbert_classifier import load_trained
    from src.xai.attention import attention_scores, spearman
    from src.xai.faithfulness import faithfulness
    from src.xai.integrated_gradients import ig_attributions, special_token_mask
    model, tok = load_trained(tiny["modernbert"], eager_attention=True)
    model.float().eval()
    dev = torch.device("cpu")
    ids = tok("the paper proposes a novel method zzunknownzz strong results")["input_ids"]
    special = special_token_mask(tok, ids)
    assert special[0] and special[-1] and not any(special[1:-1])
    assert tok.unk_token_id in ids[1:-1]                    # [UNK] content is still attributed
    att = attention_scores(model, ids, dev)
    assert len(att["last_layer"]) == len(ids) and att["pooling"] == model.hf.config.classifier_pooling
    assert att["rollout"].sum() == pytest.approx(1.0, abs=1e-4) or att["pooling"] == "mean"
    ig, delta, p, p0 = ig_attributions(model, ids, special, 1, dev, tok.pad_token_id, n_steps=300,
                                       internal_batch_size=50)
    assert len(ig) == len(ids)
    assert ig.sum() == pytest.approx(p - p0, abs=max(5e-3, 2 * delta))      # completeness
    f = faithfulness(model, ids, special, {"ig": ig, "att": att["rollout"]}, 1, dev, (0.2,))
    assert set(f) == {"p_full", "ig", "att", "random"}
    assert set(f["ig"]["0.2"]) == {"comprehensiveness", "sufficiency"}
    assert spearman([1, 2, 3, 4], [1, 2, 3, 5]) == pytest.approx(1.0)


# ---------------------------------------------------------------- end to end (toy manifests)
def _toy_manifests(work, cfg):
    from src.data.common import audit_dir, manifest_dir, save_json
    rng = np.random.default_rng(0)
    for ds, k in (("intellabs", 2), ("semdetect", 3)):
        md = manifest_dir(cfg, ds)
        md.mkdir(parents=True, exist_ok=True)
        names = ["human", "ai", "rewrite"][:k]
        for split, n_pap, off in (("train", 12, 0), ("val", 6, 100), ("test", 6, 200)):
            rows = []
            for p in range(n_pap):
                for lab in range(k):
                    rows.append({"row_id": f"{ds}-{split}-{p}-{lab}", "dataset": ds, "partition": split,
                                 "text": _texts(1, int(rng.integers(1e6)), ai=lab > 0)[0], "label": lab,
                                 "label_name": names[lab], "paper_key": f"C_2021_{off + p}",
                                 "paper_id_raw": str(off + p), "conference": "C", "year": 2021,
                                 "conf_year": "C2021", "generator": "human" if lab == 0 else "g",
                                 "review_id": f"{p}-{lab}", "source_file": "toy"})
            pd.DataFrame(rows).to_parquet(md / f"{split}.parquet", index=False)
        test = pd.read_parquet(md / "test.parquet")
        test[test["paper_key"].isin(test["paper_key"].unique()[:4])].to_parquet(md / "test_baseline.parquet",
                                                                               index=False)
        save_json({"caps": {}}, md / "summary.json")
        save_json({"status": "WARN", "dataset": ds}, audit_dir(cfg, ds) / "audit_status.json")


def test_pipeline_scripts_end_to_end(tiny, tmp_path):
    from src.data.common import load_config
    work = tmp_path / "work"
    env = dict(os.environ, PRAD_WORK_DIR=str(work), PYTHONPATH=str(ROOT))
    os.environ["PRAD_WORK_DIR"] = str(work)
    try:
        cfg = load_config()
        _toy_manifests(work, cfg)
    finally:
        os.environ.pop("PRAD_WORK_DIR")
    tc = yaml.safe_load((ROOT / "configs" / "train.yaml").read_text())
    tc["model"].update(name=tiny["modernbert"], max_length=48)
    tc["training"].update(seeds=[1, 2], epochs=1, batch_size=4, grad_accum=1, eval_batch_size=8, num_workers=0)
    tc["loss"]["rdrop_alpha"] = 0.0
    tc_p = tmp_path / "train.yaml"
    tc_p.write_text(yaml.safe_dump(tc))
    bc = yaml.safe_load((ROOT / "configs" / "baselines.yaml").read_text())
    bc["common"].update(dtype="float32", max_tokens=48, batch_size=4, val_rows=100, train_rows=100)
    bc["token_stats"]["model"] = tiny["gpt"]
    bc["binoculars"].update(observer=tiny["gpt"], performer=tiny["gpt_b"])
    bc["dnagpt"].update(regenerator=tiny["gpt"], k=2, max_new_tokens=8, max_prompt_tokens=32)
    bc_p = tmp_path / "baselines.yaml"
    bc_p.write_text(yaml.safe_dump(bc))

    def run(*a):
        r = subprocess.run([sys.executable, *map(str, a)], cwd=ROOT, env=env, capture_output=True, text=True)
        assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
        return r.stdout

    S = ROOT / "scripts"
    run(S / "03_train_intellabs.py", "--train-config", tc_p)
    run(S / "04_train_semdetect.py", "--train-config", tc_p, "--seeds", "1")
    run(S / "05_evaluate.py", "--dataset", "intellabs", "--train-config", tc_p, "--n-boot", "20")
    run(S / "05_evaluate.py", "--dataset", "semdetect", "--train-config", tc_p, "--seeds", "1", "--n-boot", "20")
    run(S / "06_run_baselines.py", "--dataset", "intellabs", "--baselines-config", bc_p, "--only",
        "logrank", "gltr", "detectllm_lrr", "mfd", "fast_detectgpt", "binoculars", "dnagpt")
    run(S / "06_run_baselines.py", "--dataset", "semdetect", "--baselines-config", bc_p, "--only", "logrank", "mfd")
    run(S / "08_cross_dataset.py", "--source", "intellabs", "--train-config", tc_p)
    run(S / "08_cross_dataset.py", "--source", "semdetect", "--train-config", tc_p, "--seeds", "1")
    out = run(S / "07_run_ablation.py", "--dataset", "intellabs", "--train-config", tc_p, "--variants",
              "ce_unweighted", "focal_weighted")
    assert "ce_unweighted" in out
    run(S / "09_build_report.py", "--n-boot", "20")
    run(S / "10_explain.py", "--dataset", "intellabs", "--train-config", tc_p, "--n-per-group", "2", "--n-steps", "8")
    run(S / "10_explain.py", "--dataset", "semdetect", "--train-config", tc_p, "--n-per-group", "1", "--n-steps", "8",
        "--min-token-count", "1")

    res = work / "outputs" / "results"
    t = pd.read_csv(res / "intellabs" / "comparison_table.csv")
    assert {"trained classifier", "zero-shot detector"} <= set(t["type"])
    assert {"LogRank", "GLTR", "DetectLLM-LRR", "MFD", "Fast-DetectGPT", "Binoculars", "DNA-GPT"} <= set(t["method"])
    sd = json.loads((res / "sem_detect" / "baselines" / "logrank" / "metrics.json").read_text())
    assert sd["test_baseline@val_threshold"]["n"] == 8                  # 4 papers x human/ai only
    assert sd["rewrite_fraction_flagged_ai@val_threshold"]["n_rewrite"] == 4
    ms = json.loads((res / "sem_detect" / "peerbert" / "final" / "metrics_summary.json").read_text())
    assert "native_3class" in ms["per_split"]["test"]
    assert (res / "report" / "intellabs_paired_tests.csv").exists()
    report = (res / "report" / "report.md").read_text()
    assert "Leakage audit: WARN" in report
    cross = json.loads((res / "sem_detect" / "cross_from_intellabs" / "metrics.json").read_text())
    assert cross["n_target"] == 12 and cross["mismatch_notes"]
    for dd in ("intellabs", "sem_detect"):
        xs = next((res / dd / "xai" / "final").glob("seed*/summary.json"))
        x = json.loads(xs.read_text())
        assert x["n_examples"] > 0 and "random" in x["faithfulness"]
        assert (xs.parent / "examples.html").exists() and (xs.parent / "top_tokens.csv").exists()
        lines = (xs.parent / "explanations.jsonl").read_text().splitlines()
        assert len(lines) == x["n_examples"]
    run(S / "09_build_report.py", "--n-boot", "20")
    assert "### Explainability" in (res / "report" / "report.md").read_text()
