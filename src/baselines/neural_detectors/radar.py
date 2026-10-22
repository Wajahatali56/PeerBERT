"""
RADAR (Hu et al., NeurIPS 2023), checkpoint TrustSafeAI/RADAR-Vicuna-7B: a RoBERTa-based
sequence classifier released by the authors. Pretrained detector: no training on our data.

Label order: the model card does not document it and the official demo could not be
fetched in this session -> UNVERIFIED. `ai_index` (config) is the assumed index of the
AI-generated class; its orientation is checked on the VALIDATION sample only: if
AUROC(val) < 0.5 the other index is used, and the decision is recorded and disclosed.
"""

import numpy as np


def load(model_name):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device).eval()
    return model, tok, device


def probabilities(model, tok, texts, device, max_length=512, batch_size=16):
    import torch
    order = np.argsort([len(t) for t in texts], kind="stable")
    out = None
    for i in range(0, len(texts), batch_size):
        idx = order[i:i + batch_size]
        enc = tok([texts[j] for j in idx], return_tensors="pt", padding=True, truncation=True,
                  max_length=max_length).to(device)
        with torch.inference_mode():
            p = torch.softmax(model(**enc).logits.float(), -1).cpu().numpy()
        if out is None:
            out = np.empty((len(texts), p.shape[1]))
        out[idx] = p
    return out


def orient(val_proba, y_val, ai_index=0):
    """Pick the AI column on validation only. Returns (index, decision dict)."""
    from sklearn.metrics import roc_auc_score
    if val_proba.shape[1] != 2:
        raise ValueError(f"RADAR expected 2 outputs, got {val_proba.shape[1]}")
    auc = roc_auc_score(y_val, val_proba[:, ai_index])
    idx = ai_index if auc >= 0.5 else 1 - ai_index
    return idx, {"configured_ai_index": ai_index, "val_auroc_configured_index": float(auc),
                 "used_ai_index": idx, "flipped_on_validation": idx != ai_index,
                 "status": "UNVERIFIED label order; orientation decided on validation only"}
