"""
Faithfulness of token-importance explanations (ERASER metrics; DeYoung et al., ACL 2020).

For an importance vector over the NON-special tokens and a fraction k:
  comprehensiveness(k) = p(x) - p(x without its top-k% tokens)   (higher = more faithful)
  sufficiency(k)       = p(x) - p(only the top-k% tokens)         (lower  = more faithful)
p = probability of the explained class; special tokens are always kept; tokens are
deleted (not masked). A seeded RANDOM ranking is evaluated the same way as a reference:
an explanation is only informative if it beats random.
"""

import numpy as np
import torch


@torch.no_grad()
def prob_of(model, ids_list, target, device):
    out = []
    for ids in ids_list:
        x = torch.tensor([ids], device=device)
        out.append(float(torch.softmax(model(x, torch.ones_like(x)).float(), -1)[0, target]))
    return np.array(out)


def _subsets(input_ids, special_mask, ranking, k):
    content = [i for i, s in enumerate(special_mask) if not s]
    n_top = max(1, int(round(k * len(content)))) if content else 0
    top = set(ranking[:n_top])
    without = [t for i, t in enumerate(input_ids) if special_mask[i] or i not in top]
    only = [t for i, t in enumerate(input_ids) if special_mask[i] or i in top]
    return without, only


def faithfulness(model, input_ids, special_mask, importances, target, device, ks=(0.1, 0.2), seed=0):
    """
    importances: dict name -> importance array over all positions (larger = more important for
    the target class). Returns {name: {k: {comprehensiveness, sufficiency}}} incl. "random".
    """
    content = np.array([i for i, s in enumerate(special_mask) if not s])
    p_full = prob_of(model, [input_ids], target, device)[0]
    rng = np.random.default_rng(seed)
    rankings = {name: content[np.argsort(-np.asarray(imp)[content], kind="stable")].tolist()
                for name, imp in importances.items()}
    rankings["random"] = rng.permutation(content).tolist()
    out = {"p_full": float(p_full)}
    for name, ranking in rankings.items():
        res = {}
        for k in ks:
            without, only = _subsets(input_ids, special_mask, ranking, k)
            p_wo, p_only = prob_of(model, [without, only], target, device)
            res[str(k)] = {"comprehensiveness": float(p_full - p_wo), "sufficiency": float(p_full - p_only)}
        out[name] = res
    return out
