"""
Shared token-statistics pass with one causal LM (one forward pass per review).

Per review (over the scored positions t = 2..T; the first token has no prediction):
  n_tokens
  mean_ll          mean log p(x_t | x_<t)
  mean_logrank     mean log(rank_t), rank 1-indexed (DetectLLM baselines/rank.py)
  mean_logrank_sq  mean log(rank_t)^2          (MFD "LLM-Deviation")
  mean_entropy     mean of -sum_v p log p      (DetectLLM baselines/entropy.py)
  gltr_*           fractions of tokens with rank <= 10, 11-100, 101-1000, > 1000
  fdg_*            sums for the analytic Fast-DetectGPT criterion with the same model as
                   reference and scorer: sum log p(x_t), sum E_p[log p], sum Var_p[log p]

Results are cached per (model, max_tokens) in outputs/cache/baselines/ and only missing
row_ids are computed, so the pass is restart-safe and shared by all detectors.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

STAT_COLS = ["n_tokens", "mean_ll", "mean_logrank", "mean_logrank_sq", "mean_entropy",
             "gltr_top10", "gltr_10_100", "gltr_100_1000", "gltr_over_1000",
             "fdg_sum_ll", "fdg_sum_mean_ref", "fdg_sum_var_ref"]


def slug(name):
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")


def load_causal_lm(name, dtype="float16", device=None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    td = getattr(torch, dtype) if device.type == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=td).to(device).eval()
    return model, tok, device


def free(*objs):
    import gc
    import torch
    for o in objs:
        del o
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def batch_stats(model, tok, texts, max_tokens, device, buckets=(10, 100, 1000)):
    import torch
    import torch.nn.functional as F
    enc = tok(list(texts), return_tensors="pt", padding=True, truncation=True, max_length=max_tokens).to(device)
    with torch.inference_mode():
        logits = model(**enc).logits[:, :-1].float()
    labels = enc["input_ids"][:, 1:]
    mask = enc["attention_mask"][:, 1:].float()
    vocab = logits.shape[-1]
    labels = labels.clamp(max=vocab - 1)
    logp = F.log_softmax(logits, -1)
    p = logp.exp()
    tok_lp = logp.gather(-1, labels[..., None]).squeeze(-1)
    tok_logit = logits.gather(-1, labels[..., None])
    rank = (logits > tok_logit).sum(-1).float() + 1.0           # 1-indexed
    logrank = rank.log()
    ent = -(p * logp).sum(-1)
    mean_ref = (p * logp).sum(-1)                                # = -entropy
    var_ref = (p * logp.square()).sum(-1) - mean_ref.square()
    n = mask.sum(-1)
    safe = n.clamp(min=1)

    def mean(x):
        return (x * mask).sum(-1) / safe

    b1, b2, b3 = buckets
    out = {
        "n_tokens": n, "mean_ll": mean(tok_lp), "mean_logrank": mean(logrank),
        "mean_logrank_sq": mean(logrank.square()), "mean_entropy": mean(ent),
        "gltr_top10": mean((rank <= b1).float()), "gltr_10_100": mean(((rank > b1) & (rank <= b2)).float()),
        "gltr_100_1000": mean(((rank > b2) & (rank <= b3)).float()), "gltr_over_1000": mean((rank > b3).float()),
        "fdg_sum_ll": (tok_lp * mask).sum(-1), "fdg_sum_mean_ref": (mean_ref * mask).sum(-1),
        "fdg_sum_var_ref": (var_ref * mask).sum(-1),
    }
    res = {k: v.cpu().numpy() for k, v in out.items()}
    for k in STAT_COLS:
        if k != "n_tokens":
            res[k] = np.where(res["n_tokens"] > 0, res[k], np.nan)
    return res


def compute_token_stats(frame, model_name, cache_root, max_tokens=512, batch_size=4, dtype="float16",
                        buckets=(10, 100, 1000), loaded=None, flush_every=50):
    """
    frame: rows with row_id, text. Returns frame [row_id] + STAT_COLS aligned to `frame`.
    loaded: optional (model, tok, device) to reuse an already loaded LM.
    """
    cache_root = Path(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_p = cache_root / f"token_stats_{slug(model_name)}_{max_tokens}.parquet"
    cached = pd.read_parquet(cache_p) if cache_p.exists() else pd.DataFrame(columns=["row_id"] + STAT_COLS)
    todo = frame.loc[~frame["row_id"].isin(cached["row_id"]), ["row_id", "text"]].drop_duplicates("row_id")
    if len(todo):
        model, tok, device = loaded or load_causal_lm(model_name, dtype)
        order = todo["text"].str.len().sort_values().index        # similar lengths per batch
        todo = todo.loc[order]
        new, batches = [], 0
        for i in range(0, len(todo), batch_size):
            chunk = todo.iloc[i:i + batch_size]
            st = batch_stats(model, tok, chunk["text"].tolist(), max_tokens, device, buckets)
            new.append(pd.DataFrame({"row_id": chunk["row_id"].values, **st}))
            batches += 1
            if batches % flush_every == 0:
                cached = pd.concat([cached, *new], ignore_index=True)
                cached.to_parquet(cache_p, index=False)
                new = []
                print(f"[token_stats] {model_name}: {len(cached)} cached")
        cached = pd.concat([cached, *new], ignore_index=True)
        cached.to_parquet(cache_p, index=False)
        if loaded is None:
            free(model)
    stats = frame[["row_id"]].merge(cached.drop_duplicates("row_id"), on="row_id", how="left")
    if stats[STAT_COLS].isna().all(axis=1).any():
        raise RuntimeError("token stats missing for some rows after computation")
    return stats
