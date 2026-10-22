"""
PeerBERT = ModernBERT-base + sequence-classification head (2 classes for IntelLabs,
3 classes for Sem-Detect), loaded with AutoModelForSequenceClassification.

Version-robust choices (transformers changes quickly; features are detected, not assumed):
- reference_compile=False is set when the config has that field (avoids torch.compile
  inside ModernBERT on Kaggle/Colab GPUs).
- Dropout: ModernBERT ships with dropout 0.0. Every *_dropout field present in the
  config is reported, and set to `dropout` when dropout_on is true. R-Drop requires
  non-zero dropout; the trainer refuses R-Drop otherwise.
- attn_implementation is left to the library default (sdpa when available). XAI later
  uses an eager-attention copy.
"""

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

HEAD_FILE = "peerbert_head.json"

DROPOUT_FIELDS = ("embedding_dropout", "attention_dropout", "mlp_dropout", "classifier_dropout",
                  "hidden_dropout_prob", "attention_probs_dropout_prob")


def dropout_values(config):
    return {f: getattr(config, f) for f in DROPOUT_FIELDS if getattr(config, f, None) is not None}


class CosineHead(nn.Module):
    """logits = scale * cos(features, class weight vectors) (large-margin cosine classifier)."""

    def __init__(self, in_features, num_labels, scale=30.0):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_labels, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.scale = float(scale)

    def forward(self, x):
        return self.scale * F.linear(F.normalize(x, dim=-1), F.normalize(self.weight, dim=-1))


def use_cosine_head(hf, scale):
    old = hf.classifier
    head = CosineHead(old.in_features, old.out_features, scale).to(old.weight.device)
    with torch.no_grad():
        head.weight.copy_(old.weight)
    hf.classifier = head
    return hf


class PeerBERT(nn.Module):
    """Thin wrapper: forward(input_ids, attention_mask) -> logits."""

    def __init__(self, hf_model):
        super().__init__()
        self.hf = hf_model

    @property
    def config(self):
        return self.hf.config

    def forward(self, input_ids, attention_mask=None):
        return self.hf(input_ids=input_ids, attention_mask=attention_mask).logits

    def save(self, out_dir, tokenizer=None):
        self.hf.save_pretrained(out_dir)
        head = self.hf.classifier
        meta = ({"head": "cosine", "scale": head.scale} if isinstance(head, CosineHead) else {"head": "linear"})
        (Path(out_dir) / HEAD_FILE).write_text(json.dumps(meta))
        if tokenizer is not None:
            tokenizer.save_pretrained(out_dir)


def build_model(name_or_path, num_labels, class_names, dropout_on=True, dropout=0.1, head="linear",
                cosine_scale=30.0):
    """Returns (PeerBERT, tokenizer, info dict with verified dropout before/after)."""
    from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

    config = AutoConfig.from_pretrained(name_or_path)
    before = dropout_values(config)
    if hasattr(config, "reference_compile"):
        config.reference_compile = False
    if dropout_on:
        for f in before:
            setattr(config, f, float(dropout))
    config.num_labels = num_labels
    config.id2label = {i: n for i, n in enumerate(class_names)}
    config.label2id = {n: i for i, n in enumerate(class_names)}
    # The classification head is always new; a checkpoint that already has a head of another
    # size (e.g. 2 vs 3 labels) is re-initialised instead of raising.
    hf = AutoModelForSequenceClassification.from_pretrained(name_or_path, config=config,
                                                            ignore_mismatched_sizes=True)
    if head == "cosine":
        use_cosine_head(hf, cosine_scale)
    elif head != "linear":
        raise ValueError(f"unknown head '{head}' (linear | cosine)")
    tok = AutoTokenizer.from_pretrained(name_or_path)
    info = {"model": name_or_path, "head": head, "model_type": config.model_type, "dropout_before": before,
            "dropout_after": dropout_values(hf.config),
            "reference_compile": getattr(hf.config, "reference_compile", "n/a"),
            "num_parameters": sum(p.numel() for p in hf.parameters())}
    return PeerBERT(hf), tok, info


def load_trained(model_dir, eager_attention=False):
    """Load a saved checkpoint for inference (eager_attention=True for attention XAI)."""
    from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

    config = AutoConfig.from_pretrained(model_dir)
    if hasattr(config, "reference_compile"):
        config.reference_compile = False
    kwargs = {"attn_implementation": "eager"} if eager_attention else {}
    hf = AutoModelForSequenceClassification.from_pretrained(model_dir, config=config, **kwargs)
    meta_p = Path(model_dir) / HEAD_FILE
    meta = json.loads(meta_p.read_text()) if meta_p.exists() else {"head": "linear"}
    if meta["head"] == "cosine":
        use_cosine_head(hf, meta["scale"])     # classifier.weight was loaded into the Linear first
    return PeerBERT(hf), AutoTokenizer.from_pretrained(model_dir)


def has_active_dropout(model):
    return any(isinstance(m, nn.Dropout) and m.p > 0 for m in model.modules()) or any(
        v and v > 0 for v in dropout_values(model.config).values())


def passes_differ(model, batch):
    """Two train-mode passes on the same batch must differ when dropout is active."""
    was = model.training
    model.train()
    with torch.no_grad():
        a = model(batch["input_ids"], batch["attention_mask"])
        b = model(batch["input_ids"], batch["attention_mask"])
    model.train(was)
    return not torch.allclose(a.float(), b.float())
