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


# ---------------------------------------------------------------- IntelLabs-only notebook
def _build_intellabs(tmp_path):
    out = tmp_path / "intellabs.ipynb"
    subprocess.run([sys.executable, str(ROOT / "notebooks" / "make_intellabs_notebook.py"), str(out)], check=True,
                   capture_output=True, cwd=ROOT)
    return json.loads(out.read_text())


def test_intellabs_notebook_has_exactly_the_repository_code_and_valid_cells(tmp_path):
    nb = _build_intellabs(tmp_path)
    embedded = {}
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        if src.startswith("%%writefile "):
            first, _, body = src.partition("\n")
            embedded[first.split()[1]] = body
            continue
        py = "\n".join((l[:len(l) - len(l.lstrip())] + "pass") if l.strip().startswith(("!", "%")) else l
                       for l in src.splitlines())
        ast.parse(py)
    for rel, body in embedded.items():
        text = (ROOT / rel).read_text(encoding="utf-8")
        if rel == "configs/train.yaml":
            text = text[:text.index("# Ablations")]                  # the ablation recipe is left out on purpose
        assert body.rstrip("\n") == text.rstrip("\n"), f"{rel} differs between notebook and repository"
    names = set(embedded)
    assert not any(n.startswith("src/xai/") for n in names)                              # no explainability
    assert {"scripts/02_prepare_splits.py", "scripts/03_train_intellabs.py", "scripts/05_evaluate.py",
            "scripts/06_run_baselines.py", "scripts/09_build_report.py"} <= names
    assert not names & {"scripts/04_train_semdetect.py", "scripts/07_run_ablation.py", "scripts/08_cross_dataset.py",
                        "scripts/10_explain.py"}                                         # no second dataset, ablation, XAI
    assert "ablation" not in embedded["configs/train.yaml"]


def test_intellabs_notebook_is_one_dataset_and_in_the_right_order(tmp_path):
    nb = _build_intellabs(tmp_path)
    text = "\n".join("".join(c["source"]) for c in nb["cells"] if not "".join(c["source"]).startswith("%%writefile"))
    for needle in ("IntelLabs/AI-Peer-Review-Detection-Benchmark", "`0` = **Human**, `1` = **AI**", "calibration", "extended",
                   "--dataset intellabs", "03_train_intellabs.py", "05_evaluate.py", "06_run_baselines.py",
                   "extended_baseline", "PeerBERT_intellabs_results.zip", "restore_outputs", "run_jobs"):
        assert needle in text, needle
    for banned in ("semdetect", "Sem-Detect", "10_explain", "Integrated Gradients", "cross_dataset", "07_run_ablation"):
        assert banned not in text, banned
    order = [text.index(k) for k in ("## 2. Data", "## 3. Train", "## 4. Evaluate", "## 5. Compare", "## 6. Report")]
    assert order == sorted(order)
    assert text.index("03_train_intellabs.py --smoke") < text.index("--variant base --seeds")
