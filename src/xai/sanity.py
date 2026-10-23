"""
Parameter-randomisation sanity check for attributions (Adebayo et al., NeurIPS 2018).

An explanation method that really reflects what a trained model learned must change when the
model's weights are replaced by random ones. The same Integrated Gradients procedure is run on a
randomly initialised copy of the architecture and compared with the trained model's attributions
for the same review and the same explained class: LOW rank correlation is the desired outcome.
A high correlation would mean the attributions mostly reflect the input or the architecture,
not the learned decision.
"""

import numpy as np
import torch

from .attention import spearman
from .integrated_gradients import ig_attributions_safe


def randomized_copy(model_dir, seed=0):
    """Same architecture and configuration as the saved model, freshly initialised weights."""
    from transformers import AutoConfig, AutoModelForSequenceClassification

    from src.models.modernbert_classifier import PeerBERT
    config = AutoConfig.from_pretrained(model_dir)
    if hasattr(config, "reference_compile"):
        config.reference_compile = False
    torch.manual_seed(seed)
    return PeerBERT(AutoModelForSequenceClassification.from_config(config))


def randomization_check(random_model, examples, device, pad_id, n_steps=50, internal_batch_size=4):
    """
    examples: dicts saved by scripts/10_explain.py (token_ids, special, pred, ig, row_id).
    Returns one dict per example with Spearman(|IG_trained|, |IG_random|) and the signed version,
    computed over content (non-special) tokens only.
    """
    random_model.to(device).eval()
    out = []
    for e in examples:
        ig_r, delta, _, _ = ig_attributions_safe(random_model, e["token_ids"], e["special"], e["pred"], device, pad_id,
                                                 n_steps, internal_batch_size)
        content = [i for i, s in enumerate(e["special"]) if not s]
        trained = np.asarray(e["ig"], float)[content]
        rand = np.asarray(ig_r, float)[content]
        out.append({"row_id": e["row_id"], "n_tokens": len(content),
                    "spearman_abs": spearman(np.abs(trained), np.abs(rand)),
                    "spearman_signed": spearman(trained, rand), "delta_random_model": delta})
    return out
