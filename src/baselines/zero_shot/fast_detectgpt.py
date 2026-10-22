"""
Fast-DetectGPT (Bao et al., ICLR 2024), ANALYTIC conditional probability curvature.
Verified against the official repo (baoguangsheng/fast-detect-gpt,
scripts/fast_detect_gpt.py::get_sampling_discrepancy_analytic):
  lp(v)    = log p_score(v | x_<t)
  mean_t   = sum_v p_ref(v | x_<t) lp(v)
  var_t    = sum_v p_ref(v | x_<t) lp(v)^2 - mean_t^2
  score    = (sum_t lp(x_t) - sum_t mean_t) / sqrt(sum_t var_t)
Larger = more likely machine-generated. Zero-shot. The old sampling approximation is not used.
Default: reference = scoring = the shared token-stats LM (white-box setting); the surrogate
model is disclosed. A different reference model can be used with score_two_models().
"""

import numpy as np


def score_from_stats(stats):
    num = stats["fdg_sum_ll"].to_numpy(dtype=float) - stats["fdg_sum_mean_ref"].to_numpy(dtype=float)
    return num / np.sqrt(np.clip(stats["fdg_sum_var_ref"].to_numpy(dtype=float), 1e-12, None))


def score_two_models(texts, ref, score_m, tok, device, max_tokens=512, batch_size=2):
    """Reference and scoring models must share the tokenizer `tok`."""
    import torch
    import torch.nn.functional as F
    out = []
    for i in range(0, len(texts), batch_size):
        enc = tok(list(texts[i:i + batch_size]), return_tensors="pt", padding=True, truncation=True,
                  max_length=max_tokens).to(device)
        with torch.inference_mode():
            lr = ref(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).logits[:, :-1].float()
            ls = score_m(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).logits[:, :-1].float()
        v = min(lr.shape[-1], ls.shape[-1])
        lr, ls = lr[..., :v], ls[..., :v]
        labels = enc["input_ids"][:, 1:].clamp(max=v - 1)
        mask = enc["attention_mask"][:, 1:].float()
        lp = F.log_softmax(ls, -1)
        pref = F.softmax(lr, -1)
        ll = lp.gather(-1, labels[..., None]).squeeze(-1)
        mean = (pref * lp).sum(-1)
        var = (pref * lp.square()).sum(-1) - mean.square()
        s = ((ll - mean) * mask).sum(-1) / (var * mask).sum(-1).clamp(min=1e-12).sqrt()
        out.append(s.cpu().numpy())
    return np.concatenate(out)
