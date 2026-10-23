"""
Binoculars (Hans et al., ICML 2024). Verified against the official repo (ahans30/Binoculars,
binoculars/detector.py + metrics.py):
  ppl   = perplexity(encodings, performer_logits)          mean token NLL under the PERFORMER
  x_ppl = entropy(observer_logits, performer_logits, ...)  mean over tokens of
          -sum_v softmax(observer)_v * log_softmax(performer)_v
  binoculars score = ppl / x_ppl ;  LOWER = more likely AI.
This module returns -score so that larger = more likely AI (as for all detectors here).
Official pair: observer tiiuae/falcon-7b, performer tiiuae/falcon-7b-instruct. Published
thresholds are NOT reused: the threshold is calibrated on validation. Zero-shot.
"""

import numpy as np


PROBES = ("The paper proposes a novel method and evaluates it on three benchmarks.",
          "Weaknesses: the ablation study is missing, and Table 2 lacks error bars.",
          "Questions for the authors: how does the approach scale to 10^6 tokens?")


def check_tokenizers(tok_o, tok_p, texts=PROBES):
    """
    The two models must tokenise text identically (the official code also requires this).
    Instruct variants may add chat tokens, so the check compares the actual token ids
    produced for real text instead of the full vocabularies.
    """
    for t in texts:
        a = tok_o(t, add_special_tokens=False)["input_ids"]
        b = tok_p(t, add_special_tokens=False)["input_ids"]
        if a != b:
            raise ValueError("Binoculars observer and performer tokenise text differently; "
                             "choose a base/instruct pair from the same model family")


def load_pair(observer, performer, dtype="float16"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from src.baselines.statistical.token_stats import dtype_kwarg
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    td = getattr(torch, dtype) if device.type == "cuda" else torch.float32
    tok_o, tok_p = AutoTokenizer.from_pretrained(observer), AutoTokenizer.from_pretrained(performer)
    check_tokenizers(tok_o, tok_p)
    if tok_o.pad_token is None:
        tok_o.pad_token = tok_o.eos_token
    tok_o.padding_side = "right"
    try:
        obs = AutoModelForCausalLM.from_pretrained(observer, **dtype_kwarg(td)).to(device).eval()
        perf = AutoModelForCausalLM.from_pretrained(performer, **dtype_kwarg(td)).to(device).eval()
    except torch.cuda.OutOfMemoryError as e:
        raise RuntimeError(
            f"CUDA out of memory while loading the Binoculars pair ({observer}, {performer}). Two 7B models "
            f"(the paper's Falcon pair) need about 29 GB of GPU memory in fp16; choose a smaller base/instruct "
            f"pair from the same family in configs/baselines.yaml (binoculars.observer / performer).") from e
    return obs, perf, tok_o, device


def batch_scores(obs, perf, tok, texts, device, max_tokens=512):
    """Returns (binoculars_score, ppl, x_ppl) arrays for one batch."""
    import torch
    import torch.nn.functional as F
    enc = tok(list(texts), return_tensors="pt", padding=True, truncation=True, max_length=max_tokens).to(device)
    with torch.inference_mode():
        lo = obs(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).logits[:, :-1].float()
        lp = perf(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).logits[:, :-1].float()
    v = min(lo.shape[-1], lp.shape[-1])          # padded embedding sizes can differ by a few rows
    lo, lp = lo[..., :v], lp[..., :v]
    labels = enc["input_ids"][:, 1:].clamp(max=v - 1)
    mask = enc["attention_mask"][:, 1:].float()
    n = mask.sum(-1).clamp(min=1)
    nll = F.cross_entropy(lp.transpose(1, 2), labels, reduction="none")
    ppl = (nll * mask).sum(-1) / n
    x = -(F.softmax(lo, -1) * F.log_softmax(lp, -1)).sum(-1)
    x_ppl = (x * mask).sum(-1) / n
    return (ppl / x_ppl).cpu().numpy(), ppl.cpu().numpy(), x_ppl.cpu().numpy()


def score(texts, obs, perf, tok, device, max_tokens=512, batch_size=2):
    order = np.argsort([len(t) for t in texts], kind="stable")
    out = np.empty(len(texts))
    for i in range(0, len(texts), batch_size):
        idx = order[i:i + batch_size]
        b, _, _ = batch_scores(obs, perf, tok, [texts[j] for j in idx], device, max_tokens)
        out[idx] = b
    return -out
