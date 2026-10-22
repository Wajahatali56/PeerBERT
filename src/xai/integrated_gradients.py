"""
Integrated Gradients (Sundararajan et al., ICML 2017) with Captum's LayerIntegratedGradients
on the token-embedding layer of a trained PeerBERT.

- Target: the class being explained (default: the model's predicted class).
- Output explained: softmax probability of the target class.
- Baseline: the same sequence with every non-special token replaced by the pad token
  (special tokens such as [CLS]/[SEP] are kept), so only the review's words are attributed.
- Token attribution = sum over the embedding dimension (signed: > 0 pushes towards the
  target class). The completeness gap |sum(attr) - (F(x) - F(baseline))| is returned as
  `delta` so convergence can be reported (raise n_steps if it is large).
"""

import numpy as np
import torch


def _forward(model):
    def f(input_ids, attention_mask, target):
        probs = torch.softmax(model(input_ids, attention_mask).float(), -1)
        return probs[:, target]
    return f


def ig_attributions(model, input_ids, special_mask, target, device, pad_id, n_steps=50, internal_batch_size=8):
    """
    input_ids: list[int]; special_mask: list[bool] (True = special token kept in the baseline).
    Returns (attributions np.array[len], delta float, p_target float, p_baseline float).
    """
    from captum.attr import LayerIntegratedGradients

    model.eval()
    ids = torch.tensor([input_ids], device=device)
    keep = torch.tensor([special_mask], device=device)
    base = torch.where(keep, ids, torch.full_like(ids, pad_id))
    mask = torch.ones_like(ids)
    fwd = _forward(model)
    lig = LayerIntegratedGradients(lambda i, m: fwd(i, m, target), model.hf.get_input_embeddings())
    attr, delta = lig.attribute(inputs=ids, baselines=base, additional_forward_args=(mask,), n_steps=n_steps,
                                internal_batch_size=internal_batch_size, return_convergence_delta=True)
    with torch.no_grad():
        p = float(fwd(ids, mask, target)[0])
        p0 = float(fwd(base, mask, target)[0])
    tok_attr = attr.sum(-1)[0].detach().float().cpu().numpy().astype(float)
    return tok_attr, float(delta.abs().item()), p, p0


def special_token_mask(tokenizer, input_ids):
    """
    True at the template positions the tokenizer adds (e.g. [CLS] first, [SEP] last), read from
    the tokenizer. Content tokens are never marked, even [UNK], so they are always attributed.
    """
    from src.models.dataset import special_template
    prefix, suffix = special_template(tokenizer)
    n = len(input_ids)
    return [i < len(prefix) or i >= n - len(suffix) for i in range(n)]


def display_tokens(tokenizer, input_ids):
    """Human-readable token strings (BPE markers removed; leading space kept as a word start)."""
    return [tokenizer.decode([i], clean_up_tokenization_spaces=False) for i in input_ids]


def normalise(x):
    x = np.asarray(x, float)
    m = np.abs(x).max()
    return x / m if m > 0 else x
