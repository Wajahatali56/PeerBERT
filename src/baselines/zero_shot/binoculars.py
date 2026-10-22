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


def load_pair(observer, performer, dtype="float16"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    td = getattr(torch, dtype) if device.type == "cuda" else torch.float32
    tok_o, tok_p = AutoTokenizer.from_pretrained(observer), AutoTokenizer.from_pretrained(performer)
    if tok_o.get_vocab() != tok_p.get_vocab():
        raise ValueError(f"Binoculars needs identical tokenizers: {observer} vs {performer}")
    if tok_o.pad_token is None:
        tok_o.pad_token = tok_o.eos_token
    tok_o.padding_side = "right"
    obs = AutoModelForCausalLM.from_pretrained(observer, torch_dtype=td).to(device).eval()
    perf = AutoModelForCausalLM.from_pretrained(performer, torch_dtype=td).to(device).eval()
    return obs, perf, tok_o, device


def batch_scores(obs, perf, tok, texts, device, max_tokens=512):
    """Returns (binoculars_score, ppl, x_ppl) arrays for one batch."""
    import torch
    import torch.nn.functional as F
    enc = tok(list(texts), return_tensors="pt", padding=True, truncation=True, max_length=max_tokens).to(device)
    with torch.inference_mode():
        lo = obs(**enc).logits[:, :-1].float()
        lp = perf(**enc).logits[:, :-1].float()
    labels = enc["input_ids"][:, 1:]
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
