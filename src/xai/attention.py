"""
Attention-based explanations from an EAGER-attention copy of a trained PeerBERT
(sdpa/flash kernels do not return attention weights).

For one example (no padding) three token-importance vectors are returned:
  last_layer  attention FROM the pooled position TO each token in the last layer, mean over heads
  mean_layers the same, averaged over all layers
  rollout     attention rollout (Abnar & Zuidema, ACL 2020): per layer 0.5*A + 0.5*I
              (residual), rows renormalised, multiplied across layers
"Pooled position": ModernBERT's classifier pools with `classifier_pooling` = "cls" (row 0)
or "mean" (mean over all rows); the config value is read, not assumed.
Attention weights are not faithful explanations by themselves (Jain & Wallace, 2019);
they are reported next to Integrated Gradients and checked with faithfulness tests.
"""

import numpy as np
import torch


def pooling_rows(config, n):
    mode = getattr(config, "classifier_pooling", "cls")
    return (slice(0, 1), mode) if mode == "cls" else (slice(0, n), mode)


@torch.no_grad()
def attention_scores(model, input_ids, device):
    """input_ids: list[int] (one example). Returns dict of np arrays of length len(input_ids)."""
    ids = torch.tensor([input_ids], device=device)
    mask = torch.ones_like(ids)
    out = model.hf(input_ids=ids, attention_mask=mask, output_attentions=True)
    if not out.attentions or out.attentions[0] is None:
        raise RuntimeError("model returned no attentions; load it with eager_attention=True")
    att = torch.stack([a[0].float().mean(0) for a in out.attentions])     # (layers, n, n)
    n = att.shape[-1]
    rows, mode = pooling_rows(model.hf.config, n)
    last = att[-1, rows].mean(0)
    mean_layers = att[:, rows].mean(1).mean(0)
    eye = torch.eye(n, device=att.device)
    roll = eye
    for a in att:
        a = 0.5 * a + 0.5 * eye
        a = a / a.sum(-1, keepdim=True)
        roll = a @ roll
    rollout = roll[rows].mean(0)
    to_np = lambda x: x.detach().cpu().numpy().astype(float)  # noqa: E731
    return {"last_layer": to_np(last), "mean_layers": to_np(mean_layers), "rollout": to_np(rollout),
            "pooling": mode, "n_layers": int(att.shape[0])}


def spearman(a, b):
    """Rank correlation between two importance vectors (ties averaged)."""
    from scipy.stats import spearmanr
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return None
    return float(spearmanr(a, b).statistic)
