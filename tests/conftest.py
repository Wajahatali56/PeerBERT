"""Toy-data fixtures. No network: hub access is replaced by local files."""

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.common import load_config  # noqa: E402

MODELS = {"claude": "claude-sonnet-3.5v2", "gpt4o": "gpt-4o"}


def intel_frame(conf, year, gen, n_papers=6, reviews_per_paper=2, schema="review",
                pid_offset=0, human_text=None, extra=None):
    rows = []
    for p in range(n_papers):
        pid = 1000 + pid_offset + p
        for r in range(reviews_per_paper):
            rid = f"{conf}{year}-{pid}-r{r}"
            h = human_text(pid, r) if human_text else f"Human review of {conf}{year} paper {pid} number {r}. " * 3
            a = f"AI {gen} review of {conf}{year} paper {pid} number {r}, quite verbose. " * (2 + r)
            for label, cid, body in ((0, f"human-{rid}", h), (1, f"{MODELS[gen]}-batch-{rid}", a)):
                row = {"label": label, "custom_id": cid, "PaperId": pid, "review-id": rid,
                       "conference": f"{conf}{year}", "rating": 6, "confidence": 4,
                       "code_of_conduct": "x" * (168 if label == 0 else 3)}
                if schema == "review":
                    row["review"] = body
                else:   # summary / strengths / weaknesses / questions schema
                    row.update(summary=f"Summary: {body}", strengths=f"Strengths {pid}",
                               weaknesses=f"Weaknesses {pid}", questions="")
                if extra:
                    row.update(extra(label))
                rows.append(row)
    return pd.DataFrame(rows)


class FakeHub:
    """Writes toy CSVs under tmp and exposes lister/fetcher callables."""

    def __init__(self, root):
        self.root = Path(root)
        self.paths = []

    def add(self, subset, gen, conf, year, df):
        rel = f"{subset}/{gen}/{conf}{year}.{subset}.{MODELS[gen]}.csv"
        p = self.root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(p, index=False)
        self.paths.append(rel)
        return rel

    def lister(self, repo_id):
        return list(self.paths)

    def fetcher(self, repo_id, path):
        return str(self.root / path)


def semdetect_raw(n_train_papers=8, n_test_papers=4):
    rows, i = [], 0
    for split, n, off in (("train", n_train_papers, 0), ("test", n_test_papers, 100)):
        for p in range(n):
            pid = off + p
            for cls in ("human", "ai", "rewrite"):
                rows.append({
                    "paper_id": pid, "submission_id": f"sub{split}{pid}", "conference": "ICLR",
                    "year": 2023 if p % 2 else 2024, "paper_title": f"T{pid}",
                    "review_id": f"rv{i}", "og_review_id": f"og{i % 3}", "author": "a",
                    "class": cls, "rating": 5,
                    "clean_review": f"{cls} review for {split} paper {pid} " * (1 + (cls == 'ai')),
                    "claim_extraction": "", "split": split,
                })
                i += 1
    return pd.DataFrame(rows)


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    monkeypatch.setenv("PRAD_WORK_DIR", str(tmp_path / "work"))
    return load_config()


@pytest.fixture
def hub(tmp_path):
    return FakeHub(tmp_path / "hub")
