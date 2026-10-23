"""The generated notebook must carry exactly the repository code and contain only valid Python cells."""

import ast
import json
import subprocess
import sys

from conftest import ROOT


def _build(tmp_path):
    out = tmp_path / "nb.ipynb"
    subprocess.run([sys.executable, str(ROOT / "notebooks" / "make_full_notebook.py"), str(out)], check=True,
                   capture_output=True, cwd=ROOT)
    return json.loads(out.read_text())


def test_notebook_embeds_repository_code_and_parses(tmp_path):
    nb = _build(tmp_path)
    embedded = {}
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        if src.startswith("%%writefile "):
            first, _, body = src.partition("\n")
            embedded[first.split()[1]] = body
            continue
        py = "\n".join((l[:len(l) - len(l.lstrip())] + "pass") if l.strip().startswith(("!", "%")) else l
                       for l in src.splitlines())
        ast.parse(py)                                          # raises on a syntax error in the notebook's own cells
    repo = {}
    for d in ("configs", "src", "scripts"):
        for p in sorted((ROOT / d).rglob("*")):
            if p.is_file() and p.suffix in (".py", ".yaml") and "__pycache__" not in str(p) and p.read_text().strip():
                repo[p.relative_to(ROOT).as_posix()] = p.read_text(encoding="utf-8")
    assert set(embedded) == set(repo)
    for rel, text in repo.items():
        assert embedded[rel].rstrip("\n") == text.rstrip("\n"), f"{rel} differs between notebook and repository"
    ids = [c["id"] for c in nb["cells"]]
    assert len(ids) == len(set(ids))


def test_notebook_covers_every_experiment_and_kaggle_features(tmp_path):
    nb = _build(tmp_path)
    text = "\n".join("".join(c["source"]) for c in nb["cells"])
    for needle in ("Experiment 1", "Experiment 2", "Experiment 3", "Experiment 4", "Experiment 5", "Experiment 6",
                   "Experiment 7", "Experiment 8", "07_run_ablation.py", "--variant {FINAL_VARIANT}", "05_evaluate.py",
                   "06_run_baselines.py", "08_cross_dataset.py", "10_explain.py", "11_experiment_card.py",
                   "restore_outputs", "/kaggle/working", "run_jobs", "Budget(", "PeerBERT_results_bundle.zip",
                   "sanity_randomization", "cluster_bootstrap_ci"):
        assert needle in text, needle
    # protocol order: ablation must come before the final training
    assert text.index("07_run_ablation.py --dataset") < text.index("--variant {FINAL_VARIANT}")
