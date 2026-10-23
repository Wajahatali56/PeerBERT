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


def _tokenizer(words=WORDS):
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import PreTrainedTokenizerFast
    specials = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    vocab = {t: i for i, t in enumerate(specials + list(words))}
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


def test_csm_margins_follow_cocodet():
    from src.models.losses import classification_loss, csm_logits
    cos = torch.tensor([[0.5, 0.2, 0.1], [0.1, 0.6, 0.3], [0.2, 0.1, 0.7]])
    y = torch.tensor([0, 1, 2])
    z = csm_logits(cos, y, scale=10.0, m_base=0.25, m_cost=0.25)
    expected = torch.tensor([[0.25, 0.45, 0.1],     # true human: -m_base; competing ai: +m_cost
                             [0.35, 0.35, 0.3],     # true ai: competing human +m_cost
                             [0.2, 0.1, 0.45]]) * 10  # true rewrite: no cost margin
    assert torch.allclose(z, expected)
    loss = classification_loss(cos * 10, y, "csm", csm={"scale": 10.0})
    assert loss.item() == pytest.approx(torch.nn.functional.cross_entropy(expected, y).item(), rel=1e-5)


def test_cosine_head_roundtrip(tiny, tmp_path):
    from src.models.modernbert_classifier import CosineHead, build_model, load_trained
    model, tok, info = build_model(tiny["modernbert"], 3, ("human", "ai", "rewrite"), head="cosine", cosine_scale=20)
    assert isinstance(model.hf.classifier, CosineHead) and info["head"] == "cosine"
    model.eval()
    ids = tok(["the paper proposes a method", "weak experiments"], padding=True, return_tensors="pt")
    with torch.no_grad():
        a = model(ids["input_ids"], ids["attention_mask"])
    assert a.abs().max() <= 20 + 1e-4                     # |scale * cos| <= scale
    model.save(tmp_path / "m", tok)
    loaded, _ = load_trained(tmp_path / "m")
    loaded.eval()
    with torch.no_grad():
        b = loaded(ids["input_ids"], ids["attention_mask"])
    assert isinstance(loaded.hf.classifier, CosineHead) and torch.allclose(a, b, atol=1e-5)


def test_binoculars_tokenizer_check(tiny):
    from src.baselines.zero_shot.binoculars import check_tokenizers
    check_tokenizers(tiny["tok"], tiny["tok"], ("the paper", "weak experiments"))
    other = _tokenizer(WORDS[::-1])                        # same words, different token ids
    with pytest.raises(ValueError, match="tokenise text differently"):
        check_tokenizers(tiny["tok"], other, ("the paper",))


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


def test_stale_runs_are_refused(tiny, tmp_path):
    import copy
    from src.models.trainer import load_train_config, train_one
    tcfg = load_train_config(overrides={
        "model": {"name": tiny["modernbert"], "max_length": 32},
        "training": {"epochs": 1, "batch_size": 4, "grad_accum": 1, "eval_batch_size": 8, "num_workers": 0},
        "loss": {"type": "ce", "rdrop_alpha": 0.0}})
    df = _frames()
    first = train_one(df.iloc[:16], df.iloc[16:], tcfg, "intellabs", tmp_path / "r", seed=3)
    assert train_one(df.iloc[:16], df.iloc[16:], tcfg, "intellabs", tmp_path / "r", seed=3)["run_key"] == first["run_key"]
    changed = copy.deepcopy(tcfg)
    changed["training"]["lr"] = 1e-4
    with pytest.raises(RuntimeError, match="different configuration or different data"):
        train_one(df.iloc[:16], df.iloc[16:], changed, "intellabs", tmp_path / "r", seed=3)
    with pytest.raises(RuntimeError, match="different configuration or different data"):
        train_one(df.iloc[:12], df.iloc[16:], tcfg, "intellabs", tmp_path / "r", seed=3)      # other rows
    speed_only = copy.deepcopy(tcfg)
    speed_only["training"].update(num_workers=2, gpu_id=1, eval_batch_size=4)
    train_one(df.iloc[:16], df.iloc[16:], speed_only, "intellabs", tmp_path / "r", seed=3)     # still reusable


def test_amp_dtype_uses_bf16_only_with_native_support(monkeypatch):
    from src.models import trainer
    cuda = torch.device("cuda")
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda *a, **k: True)       # what recent PyTorch says on a T4
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (7, 5))
    assert trainer.amp_dtype("auto", cuda) == torch.float16                           # T4: fp16 + GradScaler
    assert trainer.amp_dtype("bf16", cuda) == torch.float16
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (8, 0))
    assert trainer.amp_dtype("auto", cuda) == torch.bfloat16                          # A100/L4
    assert trainer.amp_dtype("off", cuda) is None and trainer.amp_dtype("auto", torch.device("cpu")) is None


def test_ig_retries_with_smaller_internal_batch_on_oom(monkeypatch):
    from src.xai import integrated_gradients as igm
    calls = []

    def fake(model, ids, special, target, device, pad_id, n_steps, ibs):
        calls.append(ibs)
        if ibs > 1:
            raise torch.cuda.OutOfMemoryError("out of memory")
        return "ok"

    monkeypatch.setattr(igm, "ig_attributions", fake)
    assert igm.ig_attributions_safe(None, [1], [False], 0, None, 0, 10, 8) == "ok"
    assert calls == [8, 4, 2, 1]


def test_seed_summary_covers_seeds_trained_by_separate_processes(tiny, tmp_path):
    from src.data.common import load_config
    from src.models.trainer import load_train_config, model_root, train_from_manifests
    os.environ["PRAD_WORK_DIR"] = str(tmp_path / "work")
    try:
        cfg = load_config()
        _toy_manifests(tmp_path / "work", cfg)
        tcfg = load_train_config(overrides={
            "model": {"name": tiny["modernbert"], "max_length": 32},
            "training": {"epochs": 1, "batch_size": 4, "grad_accum": 1, "eval_batch_size": 8, "num_workers": 0},
            "loss": {"type": "ce", "rdrop_alpha": 0.0}})
        train_from_manifests("intellabs", cfg, tcfg, [1], "final")           # what GPU 0 would run
        train_from_manifests("intellabs", cfg, tcfg, [2], "final")           # what GPU 1 would run
        s = json.loads((model_root(cfg, "intellabs") / "final" / "seed_summary.json").read_text())
    finally:
        os.environ.pop("PRAD_WORK_DIR")
    assert s["seeds"] == [1, 2] and [r["seed"] for r in s["runs"]] == [1, 2]
    assert s["best_val_macro_f1"]["macro_f1"]["values"] and len(s["best_val_macro_f1"]["macro_f1"]["values"]) == 2


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


def test_weight_randomisation_check(tiny):
    from src.models.modernbert_classifier import load_trained
    from src.xai.integrated_gradients import ig_attributions, special_token_mask
    from src.xai.sanity import randomization_check, randomized_copy
    model, tok = load_trained(tiny["modernbert"], eager_attention=True)
    model.float().eval()
    dev = torch.device("cpu")
    ids = tok("the paper proposes a novel method with strong results and weak experiments")["input_ids"]
    special = special_token_mask(tok, ids)
    ig, _, _, _ = ig_attributions(model, ids, special, 1, dev, tok.pad_token_id, n_steps=20)
    ex = {"row_id": "r1", "token_ids": ids, "special": special, "pred": 1, "ig": ig.tolist()}
    rnd = randomized_copy(tiny["modernbert"], seed=123)
    assert not torch.equal(next(iter(rnd.parameters())), next(iter(model.parameters())))     # really different weights
    rows = randomization_check(rnd, [ex, dict(ex, row_id="r2")], dev, tok.pad_token_id, n_steps=20)
    assert [r["row_id"] for r in rows] == ["r1", "r2"] and rows[0]["n_tokens"] == len(ids) - 2
    for r in rows:
        assert r["spearman_abs"] is None or -1 <= r["spearman_abs"] <= 1
        assert r["delta_random_model"] >= 0


# ---------------------------------------------------------------- end to end (toy manifests)
def _toy_manifests(work, cfg):
    from src.data.common import audit_dir, manifest_dir, save_json
    rng = np.random.default_rng(0)
    for ds, k in (("intellabs", 2), ("semdetect", 3)):
        md = manifest_dir(cfg, ds)
        md.mkdir(parents=True, exist_ok=True)
        names = ["human", "ai", "rewrite"][:k]
        splits = [("train", 12, 0), ("val", 6, 100), ("test", 6, 200)] + (
            [("extended", 6, 300)] if ds == "intellabs" else [])
        for split, n_pap, off in splits:
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
        if ds == "intellabs":
            ext = pd.read_parquet(md / "extended.parquet")
            ext[ext["paper_key"].isin(ext["paper_key"].unique()[:4])].to_parquet(md / "extended_baseline.parquet",
                                                                                 index=False)
        desc = {n: {"rows": len(f), "papers": int(f["paper_key"].nunique()),
                    "label_counts": {str(k): int(v) for k, v in f["label"].value_counts().sort_index().items()}}
                for n, f in ((n, pd.read_parquet(md / f"{n}.parquet"))
                             for n in ("train", "val", "test", "test_baseline") + (
                                 ("extended", "extended_baseline") if ds == "intellabs" else ()))}
        save_json({"dataset": ds, "seed": 42, "caps": {"test_baseline": 600}, "audit_status": "WARN",
                   "full_partitions": {n: desc[n] for n in ("train", "val", "test")}, "manifests": desc,
                   "exclusions": {}}, md / "summary.json")
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
    # protocol order: ablation (validation selection) first, then the final models from the selected variant
    out = run(S / "07_run_ablation.py", "--dataset", "intellabs", "--train-config", tc_p, "--variants",
              "ce_unweighted", "focal_weighted", "--test-rows", "12")
    assert "ce_unweighted" in out and "selected:" in out
    run(S / "07_run_ablation.py", "--dataset", "semdetect", "--train-config", tc_p, "--variants", "csm_cosine",
        "--test-rows", "12")
    run(S / "03_train_intellabs.py", "--train-config", tc_p, "--variant", "auto")
    run(S / "04_train_semdetect.py", "--train-config", tc_p, "--seeds", "1", "--variant", "auto")   # cosine head
    run(S / "05_evaluate.py", "--dataset", "intellabs", "--train-config", tc_p, "--n-boot", "20")
    run(S / "05_evaluate.py", "--dataset", "semdetect", "--train-config", tc_p, "--seeds", "1", "--n-boot", "20")
    run(S / "06_run_baselines.py", "--dataset", "intellabs", "--baselines-config", bc_p, "--only",
        "logrank", "gltr", "detectllm_lrr", "mfd", "fast_detectgpt", "binoculars", "dnagpt")
    run(S / "06_run_baselines.py", "--dataset", "semdetect", "--baselines-config", bc_p, "--only", "logrank", "mfd")
    run(S / "08_cross_dataset.py", "--source", "intellabs", "--train-config", tc_p)
    run(S / "08_cross_dataset.py", "--source", "semdetect", "--train-config", tc_p, "--seeds", "1")
    run(S / "10_explain.py", "--dataset", "intellabs", "--train-config", tc_p, "--n-per-group", "2", "--n-steps", "8",
        "--sanity-n", "3")
    run(S / "10_explain.py", "--dataset", "semdetect", "--train-config", tc_p, "--n-per-group", "1", "--n-steps", "8",
        "--min-token-count", "1", "--sanity-n", "2")
    run(S / "09_build_report.py", "--n-boot", "20")
    run(S / "11_experiment_card.py")

    res = work / "outputs" / "results"
    t = pd.read_csv(res / "intellabs" / "comparison_table.csv")
    assert {"trained classifier", "zero-shot detector"} <= set(t["type"])
    assert {"LogRank", "GLTR", "DetectLLM-LRR", "MFD", "Fast-DetectGPT", "Binoculars", "DNA-GPT"} <= set(t["method"])
    assert {"test_baseline", "extended_baseline"} <= set(t["split"])                       # baselines scored on extended too
    assert (res / "intellabs" / "baselines" / "logrank" / "predictions_extended_baseline.csv").exists()
    ms_i = json.loads((res / "intellabs" / "peerbert" / "final" / "metrics_summary.json").read_text())
    assert {"test", "test_baseline", "extended", "extended_baseline"} <= set(ms_i["per_split"])
    assert (res / "report" / "intellabs_comparison_extended.csv").exists()
    sd = json.loads((res / "sem_detect" / "baselines" / "logrank" / "metrics.json").read_text())
    assert sd["test_baseline@val_threshold"]["n"] == 8                  # 4 papers x human/ai only
    assert sd["rewrite_fraction_flagged_ai@val_threshold"]["n_rewrite"] == 4
    m42 = json.loads(next((res / "intellabs" / "peerbert" / "final").glob("seed*/metrics.json")).read_text())
    assert m42["splits"]["test"]["by_length_tercile@0.5"]
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
        assert x["sanity_randomization"]["n"] > 0 and len(x["sanity_randomization"]["per_example"]) > 0
        assert (xs.parent / "examples.html").exists() and (xs.parent / "top_tokens.csv").exists()
        lines = (xs.parent / "explanations.jsonl").read_text().splitlines()
        assert len(lines) == x["n_examples"]
    assert "### Explainability" in (res / "report" / "report.md").read_text()

    # selection protocol: validation-only choice, recorded budget, checkpoints pruned, finals use the variant
    ab = json.loads((res / "intellabs" / "ablation.json").read_text())
    assert ab["selected_variant"] in ("ce_unweighted", "focal_weighted") and ab["selection_rule"]
    assert ab["budget"]["test_sample_rows"] > 0 and all("test_macro_f1" in r for r in ab["rows"])
    assert not list((work / "outputs" / "models" / "intellabs" / "ablation").glob("*/seed*/best"))
    fin = json.loads((work / "outputs" / "models" / "intellabs" / "final" / "seed_summary.json").read_text())
    assert fin["variant"] == ab["selected_variant"]
    sd_fin = json.loads((work / "outputs" / "models" / "sem_detect" / "final" / "seed_summary.json").read_text())
    assert sd_fin["variant"] == "csm_cosine"
    assert json.loads((res / "sem_detect" / "peerbert" / "final" / "metrics_summary.json").read_text())["variant"] == "csm_cosine"
    card = (res / "report" / "experiment_card.md").read_text()
    assert "Deviations and limitations" in card and "csm_cosine" in card and "Code hash" in card
    assert json.loads((res / "report" / "experiment_card.json").read_text())["datasets"]["intellabs"]["model"]["variant"]
