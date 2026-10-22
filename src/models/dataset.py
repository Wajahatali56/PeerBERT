"""
Tokenized review dataset with truncation reporting.

- Text is used as stored in the manifests (strip-only preprocessing upstream).
- truncation="head": first max_length tokens (incl. special tokens).
  truncation="head_tail": first head_tokens + last remaining tokens of long reviews.
- truncation_stats(): fraction of truncated examples and token-length summary,
  saved with every training/evaluation run.
"""

import numpy as np
import torch
from torch.utils.data import Dataset


def special_template(tokenizer):
    """
    (prefix_ids, suffix_ids) the tokenizer adds around a single sequence, read from the
    tokenizer itself (e.g. [CLS] ... [SEP] for ModernBERT). Fast tokenizers do not
    implement build_inputs_with_special_tokens, so this is detected instead of assumed.
    """
    probe = tokenizer("review text", add_special_tokens=False)["input_ids"]
    full = tokenizer("review text", add_special_tokens=True)["input_ids"]
    for i in range(len(full) - len(probe) + 1):
        if full[i:i + len(probe)] == probe:
            return full[:i], full[i + len(probe):]
    raise ValueError("could not locate the special-token template of this tokenizer")


class ReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=1024, truncation="head", head_tokens=256):
        if truncation not in ("head", "head_tail"):
            raise ValueError(f"truncation must be head or head_tail, got {truncation}")
        self.labels = None if labels is None else np.asarray(labels).astype(int)
        prefix, suffix = special_template(tokenizer)
        n_special = len(prefix) + len(suffix)
        budget = max_length - n_special
        if budget <= 0:
            raise ValueError("max_length too small for special tokens")
        enc = tokenizer(list(texts), add_special_tokens=False, truncation=False)["input_ids"]
        self.raw_lengths = np.array([len(e) + n_special for e in enc])
        self.input_ids = []
        for ids in enc:
            if len(ids) > budget:
                if truncation == "head_tail":
                    h = min(head_tokens, budget)
                    ids = ids[:h] + (ids[len(ids) - (budget - h):] if budget > h else [])
                else:
                    ids = ids[:budget]
            self.input_ids.append(prefix + ids + suffix)
        self.max_length = max_length
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i):
        item = {"input_ids": self.input_ids[i]}
        if self.labels is not None:
            item["labels"] = int(self.labels[i])
        return item

    def collate(self, batch):
        width = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), width), self.pad_id, dtype=torch.long)
        mask = torch.zeros((len(batch), width), dtype=torch.long)
        for j, b in enumerate(batch):
            n = len(b["input_ids"])
            ids[j, :n] = torch.tensor(b["input_ids"], dtype=torch.long)
            mask[j, :n] = 1
        out = {"input_ids": ids, "attention_mask": mask}
        if "labels" in batch[0]:
            out["labels"] = torch.tensor([b["labels"] for b in batch], dtype=torch.long)
        return out

    def truncation_stats(self):
        L = self.raw_lengths
        if len(L) == 0:
            return {"n": 0}
        return {
            "n": int(len(L)), "max_length": self.max_length,
            "truncated_fraction": round(float((L > self.max_length).mean()), 6),
            "tokens_mean": round(float(L.mean()), 1), "tokens_median": float(np.median(L)),
            "tokens_p95": float(np.quantile(L, 0.95)), "tokens_max": int(L.max()),
        }
