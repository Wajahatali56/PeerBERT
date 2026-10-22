"""
DNA-GPT (Yang et al., ICLR 2024): divergent n-gram analysis.
1. Truncate each review at gamma (characters): prefix X, original continuation Y0.
2. Regenerate K continuations S_k of X with a LOCAL LM (deviation: the paper regenerates
   with the suspected source LLM; disclosed).
3. BScore = (1/K) sum_k sum_{n=n0..N} f(n) |G_n(S_k) & G_n(Y0)| / (|S_k| * |G_n(Y0)|),
   f(n) = n log n, n0 = 4, N = 25, G_n = set of word n-grams, |S_k| = words in S_k.
   Larger = more overlap = more likely machine-generated. Zero-shot.
Notes (disclosed): the official demo script (DNA-GPT-dist.py) differs from the paper
formula (n from 1, index-based weights, different normalisation, 30 regenerations); the
paper formula above is used. Optional WScore (white-box): mean_k of the per-token mean
log-likelihood difference log p(Y0|X) - log p(S_k|X) under the regenerator.
Regenerations are cached per row_id (restart-safe).
"""

import json
import math
import re
from pathlib import Path

import numpy as np

WORD = re.compile(r"\w+", re.UNICODE)


def words(text):
    return WORD.findall(text.lower())


def ngrams(tokens, n):
    return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def bscore(original, regenerations, n_min=4, n_max=25):
    y0 = words(original)
    total, used = 0.0, 0
    for s in regenerations:
        sk = words(s)
        if not sk:
            continue
        acc = 0.0
        for n in range(n_min, n_max + 1):
            g0 = ngrams(y0, n)
            if not g0:
                break
            acc += n * math.log(n) * len(ngrams(sk, n) & g0) / (len(sk) * len(g0))
        total += acc
        used += 1
    return total / used if used else 0.0


def split_text(text, gamma=0.5):
    cut = int(gamma * len(text))
    return text[:cut], text[cut:]


def regenerate(model, tok, prefixes, device, k=5, max_prompt_tokens=384, max_new_tokens=200,
               temperature=0.7, seed=0):
    import torch
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    enc = tok(list(prefixes), return_tensors="pt", padding=True, truncation=True,
              max_length=max_prompt_tokens).to(device)
    torch.manual_seed(seed)
    with torch.inference_mode():
        out = model.generate(**enc, do_sample=True, temperature=temperature, top_p=1.0, top_k=0,
                             max_new_tokens=max_new_tokens, num_return_sequences=k,
                             pad_token_id=tok.pad_token_id)
    gen = out[:, enc["input_ids"].shape[1]:]
    texts = tok.batch_decode(gen, skip_special_tokens=True)
    tok.padding_side = "right"
    return [texts[i * k:(i + 1) * k] for i in range(len(prefixes))]


def mean_token_ll(model, tok, prefix, continuation, device, max_tokens=768):
    import torch
    import torch.nn.functional as F
    p_ids = tok(prefix, add_special_tokens=False)["input_ids"]
    c_ids = tok(continuation, add_special_tokens=False)["input_ids"]
    ids = (p_ids + c_ids)[-max_tokens:]
    n_c = min(len(c_ids), len(ids) - 1)
    if n_c <= 0:
        return float("nan")
    x = torch.tensor([ids], device=device)
    with torch.inference_mode():
        lp = F.log_softmax(model(x).logits[0, :-1].float(), -1)
    tgt = x[0, 1:]
    ll = lp.gather(-1, tgt[:, None]).squeeze(-1)[-n_c:]
    return float(ll.mean())


def run(frame, model, tok, device, cfg, cache_path, seed=0):
    """frame: row_id, text. Returns (bscore array, wscore array or None)."""
    cache_path = Path(cache_path)
    cache = {}
    if cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            cache[r["row_id"]] = r["regens"]
    todo = [(rid, t) for rid, t in zip(frame["row_id"], frame["text"]) if rid not in cache]
    bs = int(cfg.get("batch_size", 2))
    with cache_path.open("a", encoding="utf-8") as f:
        for i in range(0, len(todo), bs):
            chunk = todo[i:i + bs]
            prefixes = [split_text(t, cfg["gamma"])[0] for _, t in chunk]
            regens = regenerate(model, tok, prefixes, device, cfg["k"], cfg["max_prompt_tokens"],
                                cfg["max_new_tokens"], cfg["temperature"], seed + i)
            for (rid, _), r in zip(chunk, regens):
                cache[rid] = r
                f.write(json.dumps({"row_id": rid, "regens": r}) + "\n")
            if (i // bs) % 25 == 0:
                print(f"[dnagpt] regenerated {min(i + bs, len(todo))}/{len(todo)}")
    b, w = [], []
    for rid, t in zip(frame["row_id"], frame["text"]):
        prefix, y0 = split_text(t, cfg["gamma"])
        b.append(bscore(y0, cache[rid], cfg["ngram_min"], cfg["ngram_max"]))
        if cfg.get("wscore"):
            ll0 = mean_token_ll(model, tok, prefix, y0, device)
            w.append(float(np.nanmean([ll0 - mean_token_ll(model, tok, prefix, s, device) for s in cache[rid]])))
    return np.asarray(b), (np.asarray(w) if cfg.get("wscore") else None)
