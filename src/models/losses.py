"""
PeerBERT loss:  L = L_cls + alpha * L_cons

L_cls  classification loss, one of
       ce               plain cross-entropy (unweighted)
       weighted_ce      cross-entropy with w_c = N / (K * n_c), counts from the TRAIN split only
       label_smoothing  weighted cross-entropy with label smoothing (default 0.05)
       focal            weighted focal loss, gamma = 2 (Lin et al., 2017)
L_cons R-Drop consistency (Liang et al., NeurIPS 2021): symmetric KL between the
       predictive distributions of two stochastic (dropout) passes of the same batch.
       With R-Drop on, L_cls is the mean of the two passes' losses, as in R-Drop.
"""

import numpy as np
import torch
import torch.nn.functional as F

LOSS_TYPES = ("ce", "weighted_ce", "label_smoothing", "focal")


def class_weights(train_labels, num_classes):
    """w_c = N / (K * n_c) from training labels. Raises if a class is absent from train."""
    y = np.asarray(train_labels).astype(int)
    counts = np.bincount(y, minlength=num_classes)
    if (counts == 0).any():
        raise ValueError(f"class(es) {np.flatnonzero(counts == 0).tolist()} absent from the training split")
    return torch.tensor(len(y) / (num_classes * counts), dtype=torch.float32)


def focal_loss(logits, target, gamma=2.0, weight=None):
    logp = F.log_softmax(logits.float(), dim=-1)
    logp_t = logp.gather(1, target[:, None]).squeeze(1)
    loss = -((1 - logp_t.exp()) ** gamma) * logp_t
    if weight is None:
        return loss.mean()
    w = weight.to(logits.device)[target]
    return (w * loss).sum() / w.sum()           # same normalisation as weighted F.cross_entropy


def classification_loss(logits, target, kind="weighted_ce", weight=None, smoothing=0.05, gamma=2.0):
    logits = logits.float()
    w = weight.to(logits.device) if weight is not None else None
    if kind == "ce":
        return F.cross_entropy(logits, target)
    if kind == "weighted_ce":
        return F.cross_entropy(logits, target, weight=w)
    if kind == "label_smoothing":
        return F.cross_entropy(logits, target, weight=w, label_smoothing=smoothing)
    if kind == "focal":
        return focal_loss(logits, target, gamma, w)
    raise ValueError(f"unknown loss '{kind}', expected one of {LOSS_TYPES}")


def rdrop_kl(logits1, logits2):
    """Symmetric KL: 0.5 * (KL(p1||p2) + KL(p2||p1)), averaged over the batch."""
    lp1, lp2 = F.log_softmax(logits1.float(), -1), F.log_softmax(logits2.float(), -1)
    kl12 = F.kl_div(lp2, lp1, log_target=True, reduction="batchmean")   # KL(p1 || p2)
    kl21 = F.kl_div(lp1, lp2, log_target=True, reduction="batchmean")   # KL(p2 || p1)
    return 0.5 * (kl12 + kl21)


class PeerBERTLoss:
    def __init__(self, kind="weighted_ce", weight=None, rdrop_alpha=0.0, smoothing=0.05, gamma=2.0):
        if kind not in LOSS_TYPES:
            raise ValueError(f"unknown loss '{kind}'")
        if kind == "ce":
            weight = None
        self.kind, self.weight, self.alpha = kind, weight, float(rdrop_alpha)
        self.smoothing, self.gamma = smoothing, gamma

    @property
    def uses_rdrop(self):
        return self.alpha > 0

    def __call__(self, logits1, target, logits2=None):
        """Returns (total, parts dict). logits2 is required when R-Drop is on."""
        cls1 = classification_loss(logits1, target, self.kind, self.weight, self.smoothing, self.gamma)
        if not self.uses_rdrop:
            return cls1, {"cls": float(cls1.detach()), "cons": 0.0}
        if logits2 is None:
            raise ValueError("R-Drop needs two stochastic passes")
        cls2 = classification_loss(logits2, target, self.kind, self.weight, self.smoothing, self.gamma)
        cls = 0.5 * (cls1 + cls2)
        cons = rdrop_kl(logits1, logits2)
        return cls + self.alpha * cons, {"cls": float(cls.detach()), "cons": float(cons.detach())}
