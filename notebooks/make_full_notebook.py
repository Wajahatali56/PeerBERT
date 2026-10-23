"""
Builds the single, self-contained experiment notebook (Kaggle first, also runs on Colab):

    python notebooks/make_full_notebook.py [output.ipynb]

The notebook writes the project code (configs/, src/, scripts/) into the runtime, so no repository
clone is needed, then runs every experiment on the original datasets with the original models and
shows tables, figures and data-driven analysis after each experiment. It is generated from the
repository files, so it always carries exactly the tested code. Re-run this script after changing
anything under src/, scripts/ or configs/.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "notebooks" / "PeerBERT_full_experiments.ipynb"
CODE_DIRS = ("configs", "src", "scripts")


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)}


def code(text, hide=False):
    meta = {"_kg_hide-input": True, "_kg_hide-output": True, "jupyter": {"source_hidden": True},
            "cellView": "form"} if hide else {}
    return {"cell_type": "code", "execution_count": None, "metadata": meta, "outputs": [],
            "source": text.strip("\n").splitlines(True)}


def project_files():
    files = []
    for d in CODE_DIRS:
        for p in sorted((ROOT / d).rglob("*")):
            rel = p.relative_to(ROOT).as_posix()
            if p.is_file() and p.suffix in (".py", ".yaml") and "__pycache__" not in rel and p.read_text().strip():
                files.append((rel, p.read_text(encoding="utf-8")))
    return files


cells = []
A = cells.append

# =============================================================================== title
A(md(r"""
# PeerBERT: Explainable Classification of AI-Generated Peer Reviews

One self-contained notebook: it writes the whole project into the runtime, runs **every experiment on the
original datasets with the original models**, and shows tables, figures and a short data-driven analysis
after each experiment.

| # | Experiment | Shows |
|---|---|---|
| 1 | Data, paper-level splits, leakage audit | dataset statistics, excluded rows and why, audit verdict, length-shortcut check |
| 2 | Ablation and variant selection (validation only) | loss / R-Drop / dropout / margin-loss variants, the selected configuration |
| 3 | PeerBERT training (3 seeds per dataset) | training curves, run table |
| 4 | PeerBERT evaluation | metrics with 95% CIs, TPR at low FPR, confusion matrices, ROC/PR/calibration, per generator / venue / length, errors, robustness |
| 5 | Eight baselines on the shared test subset | comparison with CIs, ROC overlay, per-generator heatmap, paired significance tests |
| 6 | Cross-dataset transfer | in-domain vs. transfer performance |
| 7 | Explainability | faithfulness vs. random, weight-randomisation sanity check, IG convergence, top tokens, case studies |
| 8 | Report | tables (CSV/LaTeX), figures (PNG/PDF), experiment card for the paper, results bundle |

**Datasets:** [IntelLabs/AI-Peer-Review-Detection-Benchmark](https://huggingface.co/datasets/IntelLabs/AI-Peer-Review-Detection-Benchmark)
(human vs AI) and [Sem-Detect/ML_Conferences-Peer-Reviews](https://huggingface.co/datasets/Sem-Detect/ML_Conferences-Peer-Reviews)
(human / ai / rewrite). **Model:** [answerdotai/ModernBERT-base](https://huggingface.co/answerdotai/ModernBERT-base).

## Protocol at a glance (what makes the results defensible)
- **No synthetic data.** Labels come from the data columns. Every dropped row is logged with a reason.
- **Splits by paper only.** Train/validation are separated by paper; the official test splits are untouched. Whole-paper compute caps are recorded.
- **Leakage audit before any training.** Paper overlap, review pairs, exact/near duplicates, length-only shortcut. A FAIL blocks training.
- **Selection on validation only.** Variant, early stopping and every decision threshold use validation data. Test data is never used to choose anything.
- **Pre-declared selection rule.** The default configuration is kept unless another variant wins on validation by a margin fixed in advance.
- **Seeds and uncertainty.** 3 seeds (mean ± std), paper-level cluster-bootstrap 95% CIs, paired tests on the same reviews with Holm correction.
- **Fair baselines.** The same test subset for all methods, thresholds calibrated on the same validation data, every deviation from the original method listed.
- **Explanations are tested, not just shown.** Faithfulness against a random ranking, plus a weight-randomisation sanity check.
- **Everything is recorded.** The experiment card (Experiment 8) lists exact settings, library versions, a code hash and all limitations.

## How to run on Kaggle
1. **File -> Import notebook** and upload this file. In the right panel (**Session options**): **Accelerator: GPU T4 x2**, **Internet: On**.
   If your account shows **Persistence**, set it to *Files only*.
2. Press **Run All** (interactive) or **Save Version -> Save & Run All** (unattended, up to 12 hours).
3. The full study needs more than one 12-hour session. Run it in **stages** (Settings cell, `STAGE = "1"` ... `"4"`). Before a later stage,
   add the previous version's *output* as a notebook input (**Add Input -> Notebooks**); the setup cell restores everything automatically.
   Each stage is restart-safe: finished work is skipped and interrupted training resumes from its last epoch.

On Google Colab the same notebook works (outputs go to Google Drive); Kaggle's 2 GPUs let the two datasets train at the same time.
"""))

# =============================================================================== settings
A(md("## 0. Settings"))
A(code(r"""
# ===================== SETTINGS (edit here) =====================================================
STAGE = "all"        # "all", or one stage per session:  "1" data+ablation | "2" train+evaluate | "3" baselines | "4" cross+xai+report
PRESETS = {
    "all": ["prepare", "smoke", "ablation", "train", "evaluate", "baselines", "cross", "xai", "report"],
    "1":   ["prepare", "smoke", "ablation"],
    "2":   ["train", "evaluate"],
    "3":   ["baselines"],
    "4":   ["cross", "xai", "report"],
}
ALL_STEPS = PRESETS["all"]
RUN = {step: step in PRESETS[STAGE] for step in ALL_STEPS}     # you can also switch single steps here

DATASETS = ["intellabs", "semdetect"]
SEEDS = None           # None = the three seeds in configs/train.yaml (42, 43, 44). Use [42] for a fast first pass.
N_BOOT = 1000          # bootstrap resamples (confidence intervals, paired tests)
USE_EXTENDED = True    # IntelLabs 'extended' subset: optional robustness test (skipped automatically if it cannot be loaded)
XAI_PER_GROUP = 25     # explained reviews per (true class x correct/incorrect) group
FINAL_VARIANT = "auto" # "auto" = the variant selected by Experiment 2; "base" = the default configuration (skips the need for the ablation)

# Whole-paper compute caps (None = no cap). They are recorded in the experiment card.
CAPS = {
    "intellabs": {"train_max_rows": 24000, "model_eval_max_rows": 30000, "baseline_eval_max_rows": 600},
    "semdetect": {"train_max_rows": None,  "model_eval_max_rows": None,  "baseline_eval_max_rows": 600},
}
# Ablation runs at a reduced budget (one seed per variant); the selected variant is then retrained at full budget.
ABLATION = {"train_rows": 6000, "val_rows": 2500, "epochs": 2, "test_rows": 4000}

TIME_BUDGET_H = 11.0   # soft limit per session: a step whose estimate would overrun it is skipped. Simply run the notebook
                       # again in a new session (with the previous output added as input): finished work is not repeated.
# Planning estimates in hours for 2 GPUs (rough assumptions, not measurements; replaced by the projection
# computed from the smoke run when available):
EST_H = {"prepare": 0.6, "smoke": 0.3, "ablation": 3.5, "train": 6.0, "evaluate": 0.8, "baselines": 4.5,
         "cross": 0.5, "xai": 0.8, "report": 0.2}
print("steps that will run:", [k for k, v in RUN.items() if v])
"""))

# =============================================================================== setup
A(md(r"""
### Environment
Paths, caches, optional restore of an earlier session, dependency check. Models and datasets are downloaded from the
Hugging Face Hub; add an `HF_TOKEN` secret (Kaggle: *Add-ons -> Secrets*) for faster, unthrottled downloads.
"""))
A(code(r"""
import os, sys, subprocess, time, json, shutil, urllib.request
from pathlib import Path

ON_KAGGLE = os.path.exists("/kaggle/working")
if ON_KAGGLE:
    WORK = "/kaggle/working/PeerBERT_outputs"
    TEMP = "/kaggle/temp" if os.path.isdir("/kaggle/temp") else "/tmp"
else:
    try:
        from google.colab import drive
        drive.mount("/content/drive")
        WORK = "/content/drive/MyDrive/PeerBERT_outputs"
    except ImportError:
        WORK = os.path.abspath("PeerBERT_outputs")
    TEMP = "/tmp"
CODE = os.path.join(TEMP, "PeerBERT_code")
os.environ["PRAD_WORK_DIR"] = WORK                          # all outputs of the pipeline go here
os.environ["HF_HOME"] = os.path.join(TEMP, "hf_cache")      # model/data downloads: not counted as output
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"            # keep the notebook output readable
for d in (WORK, CODE, os.environ["HF_HOME"]):
    os.makedirs(d, exist_ok=True)
print("platform :", "Kaggle" if ON_KAGGLE else "Colab/other")
print("code     :", CODE)
print("outputs  :", WORK)
print("HF cache :", os.environ["HF_HOME"])

try:
    if ON_KAGGLE:
        from kaggle_secrets import UserSecretsClient
        os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
    else:
        from google.colab import userdata
        os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
    print("HF_TOKEN : loaded")
except Exception:
    print("HF_TOKEN : not set (anonymous downloads)")
"""))
A(code(r"""
from importlib import metadata
from packaging.version import Version

def ensure(pkg, minimum=None):
    # install/upgrade only when missing or too old, to leave the platform's tested stack alone
    try:
        ok = minimum is None or Version(metadata.version(pkg)) >= Version(minimum)
    except metadata.PackageNotFoundError:
        ok = False
    if not ok:
        spec = f"{pkg}>={minimum}" if minimum else pkg
        print("installing", spec)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", spec], check=True)

for pkg, minimum in [("transformers", "4.48.0"), ("huggingface_hub", "0.22"), ("accelerate", "0.26"), ("captum", "0.7.0"),
                     ("pyarrow", None), ("scikit-learn", "1.3"), ("scipy", "1.10"), ("matplotlib", "3.7"), ("pyyaml", None),
                     ("packaging", None)]:
    ensure(pkg, minimum)
import torch, transformers, sklearn, pandas, numpy
print("torch", torch.__version__, "| transformers", transformers.__version__, "| sklearn", sklearn.__version__,
      "| pandas", pandas.__version__, "| numpy", numpy.__version__)
"""))

# =============================================================================== code
A(md(r"""
## Project code
The next cells write the project (configs, `src/`, `scripts/`) into the runtime. The code is collapsed by default; expand a
cell to read it. **To change a training or baseline setting, edit `configs/train.yaml` / `configs/baselines.yaml` here and re-run
that cell** before the experiment (the experiment card records the settings that were really used).
"""))
A(code(r"""
for d in ["configs", "scripts", "src/data", "src/models", "src/evaluation", "src/xai", "src/baselines/statistical",
          "src/baselines/zero_shot", "src/baselines/neural_detectors"]:
    os.makedirs(os.path.join(CODE, d), exist_ok=True)
%cd {CODE}
"""))
for rel, text in project_files():
    A(code(f"%%writefile {rel}\n{text}", hide=not rel.startswith("configs/")))
A(code(r"""
for d, _, _ in os.walk(os.path.join(CODE, "src")):
    open(os.path.join(d, "__init__.py"), "a").close()
if CODE not in sys.path:
    sys.path.insert(0, CODE)
print("project code written to", CODE)
"""))

A(code(r"""
# Apply the settings above to the config files (the printed values are what the run uses)
import yaml
cp = Path(CODE) / "configs" / "config.yaml"
c = yaml.safe_load(cp.read_text())
c["data"]["intellabs"]["use_extended"] = bool(USE_EXTENDED)
for ds_name, caps in CAPS.items():
    c["data"][ds_name].update(caps)
cp.write_text(yaml.safe_dump(c, sort_keys=False))
print("IntelLabs: use_extended =", c["data"]["intellabs"]["use_extended"], "| caps:", CAPS["intellabs"])
print("Sem-Detect caps:", CAPS["semdetect"])
tc = yaml.safe_load((Path(CODE) / "configs" / "train.yaml").read_text())
print("training:", {k: tc["training"][k] for k in ("epochs", "lr", "batch_size", "grad_accum", "amp")},
      "| max_length:", tc["model"]["max_length"], "| default loss:", tc["loss"]["type"], "rdrop", tc["loss"]["rdrop_alpha"])
"""))

A(code(r"""
from src.notebook_utils import Budget, disk_report, gpu_count, restore_outputs, run_cmd, run_jobs

BUDGET = Budget(TIME_BUDGET_H)
N_GPUS = gpu_count()
PY = f'"{sys.executable}"'
LOGS = Path(WORK) / "logs"

heavy = any(RUN[s] for s in ("prepare", "smoke", "ablation", "train", "baselines"))
if ON_KAGGLE:
    restored = restore_outputs(WORK)
    for src, n in restored:
        print(f"restored {n} files from {src}")
    if not restored:
        print("no earlier session outputs found in /kaggle/input (fine for a first run)")
if heavy:
    try:
        urllib.request.urlopen("https://huggingface.co", timeout=20)
        print("internet: OK (huggingface.co reachable)")
    except Exception as e:
        raise RuntimeError("huggingface.co is not reachable. On Kaggle enable Session options -> Internet.") from e
gpu_text = subprocess.run("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader", shell=True,
                          capture_output=True, text=True).stdout.strip()
print("GPUs:", gpu_text or "none")
if (RUN["smoke"] or RUN["ablation"] or RUN["train"]) and not torch.cuda.is_available():
    raise RuntimeError("Training needs a GPU. Select a GPU accelerator in the session options.")
if "P100" in gpu_text:
    print("note: a P100 works but is slower than a T4 for fp16 transformer training")
print(f"parallel plan: {N_GPUS} GPU(s) -> " + ("the two datasets run side by side" if N_GPUS > 1 else "datasets run one after another"))
disk_report(WORK)
"""))

# =============================================================================== helpers
A(md("### Helpers for results and plots"))
A(code(r"""
import glob, warnings
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown, HTML
warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.width", 220, "display.max_columns", 50, "display.precision", 4, "display.max_colwidth", 80)
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 300, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})

from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.calibration import calibration_curve
from src.evaluation.predict import binary_view
from src.evaluation.metrics import binary_metrics, calibrate_threshold
from src.evaluation.aggregate import by_generator
from src.evaluation.stats import auc_fn, cluster_bootstrap_ci, macro_f1_fn

OUT = Path(WORK) / "outputs"
RES = OUT / "results"
FIG = RES / "figures"
FIG.mkdir(parents=True, exist_ok=True)
DD = {"intellabs": "intellabs", "semdetect": "sem_detect"}
NAME = {"intellabs": "IntelLabs", "semdetect": "Sem-Detect"}
TAG = "final"
SEED_ARG = f"--seeds {' '.join(map(str, SEEDS))}" if SEEDS else ""
TYPE_COLOR = {"trained classifier": "#6B46C1", "zero-shot detector": "#2C7A7B", "pretrained detector": "#C05621"}


def jload(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def not_run(what):
    display(Markdown(f"> **{what}: no results yet.** Run the experiment cell above (or an earlier stage) first."))


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    plt.show()


def md_(text):
    display(Markdown(text))


def done_seeds(ds, tag=TAG):
    runs = []
    for d in sorted((OUT / "models" / DD[ds] / tag).glob("seed*")):
        info = jload(d / "done.json")
        if info:
            runs.append((info["best_val_macro_f1"], int(d.name[4:])))
    return runs


def ref_seed(ds):
    runs = sorted(done_seeds(ds))
    return runs[(len(runs) - 1) // 2][1] if runs else None       # median validation Macro-F1: chosen without test data


def peerbert_pred(ds, seed, split, binary=True):
    p = RES / DD[ds] / "peerbert" / TAG / f"seed{seed}" / f"{split}_predictions.csv"
    if seed is None or not p.exists():
        return None
    df = pd.read_csv(p, dtype={"row_id": str, "paper_key": str})
    return binary_view(df) if (binary and ds == "semdetect") else df


def ms(d, key):
    v = d.get(key) if d else None
    return f"{v['mean']:.4f} ± {v['std']:.4f}" if v else "–"


def fmt_ci(est, lo, hi):
    return "–" if est is None else f"{est:.4f} [{lo:.4f}, {hi:.4f}]"


def method_predictions(ds):
    # name -> (type, binary prediction frame on the shared test subset, validation-calibrated threshold)
    out = {}
    seed = ref_seed(ds)
    p = peerbert_pred(ds, seed, "test_baseline")
    if p is not None:
        vp = pd.read_csv(OUT / "models" / DD[ds] / TAG / f"seed{seed}" / "val_predictions.csv")
        vp = binary_view(vp) if ds == "semdetect" else vp
        out["PeerBERT"] = ("trained classifier", p, calibrate_threshold(vp["label"], vp["score"], "macro_f1"))
    for d in sorted((RES / DD[ds] / "baselines").glob("*/metrics.json")):
        m = jload(d)
        pr = pd.read_csv(d.parent / "predictions_test_baseline.csv", dtype={"row_id": str, "paper_key": str})
        out[m["display"]] = (m["type"], pr, m["val_threshold"])
    return out


def ci_table(ds, n_boot=None):
    # every method on the same reviews: point estimate + paper-level cluster-bootstrap 95% CI
    rows = []
    for name, (typ, p, thr) in method_predictions(ds).items():
        y, s, c = p["label"].values, p["score"].values, p["paper_key"].values
        if len(np.unique(y)) < 2:
            continue
        a = cluster_bootstrap_ci(y, s, c, auc_fn, n_boot or N_BOOT, seed=0)
        f = cluster_bootstrap_ci(y, s, c, macro_f1_fn(thr), n_boot or N_BOOT, seed=0)
        bm = binary_metrics(y, s, thr)
        t10 = bm["tpr_at_fpr"]["10%"]
        rows.append({"method": name, "type": typ, "n": len(p), "AI share": bm["n_pos"] / bm["n"],
                     "ROC-AUC": a["estimate"], "AUC lo": a["ci_low"], "AUC hi": a["ci_high"],
                     "Macro-F1": f["estimate"], "F1 lo": f["ci_low"], "F1 hi": f["ci_high"],
                     "PR-AUC": bm["pr_auc"], "TPR@10%FPR": t10["tpr"] if t10["supported"] else np.nan,
                     "FPR on human": bm["ai_fpr"], "threshold": thr})
    return pd.DataFrame(rows)


def progress():
    # which parts of the study are finished (outputs of earlier sessions count)
    from src.baselines.registry import BASELINES
    n_seeds = len(SEEDS) if SEEDS else len(yaml.safe_load((Path(CODE) / "configs" / "train.yaml").read_text())["training"]["seeds"])
    rows = []
    for ds in DATASETS:
        tgt = "semdetect" if ds == "intellabs" else "intellabs"
        au = jload(OUT / "audit" / DD[ds] / "audit_status.json")
        ab = jload(RES / DD[ds] / "ablation.json")
        ev = jload(RES / DD[ds] / "peerbert" / TAG / "metrics_summary.json")
        rows.append({"dataset": NAME[ds],
                     "1 data + audit": au["status"] if au else "-",
                     "2 ablation (selected)": ab["selected_variant"] if ab else "-",
                     "3 training": f"{len(done_seeds(ds))}/{n_seeds} seeds",
                     "4 evaluation": f"{len(ev['seeds'])} seeds" if ev else "-",
                     "5 baselines": f"{len(list((RES / DD[ds] / 'baselines').glob('*/metrics.json')))}/{len(BASELINES)}",
                     "6 cross-dataset": "done" if (RES / DD[tgt] / f"cross_from_{ds}" / "metrics.json").exists() else "-",
                     "7 explainability": "done" if any((RES / DD[ds] / "xai" / TAG).glob("seed*/summary.json")) else "-"})
    card = (RES / "report" / "experiment_card.md").exists()
    display(pd.DataFrame(rows).set_index("dataset"))
    print("8 report + experiment card:", "done" if card else "-")


def load_xai(ds):
    xs = sorted((RES / DD[ds] / "xai" / TAG).glob("seed*/explanations.jsonl"))
    if not xs:
        return None, None
    summ = jload(xs[0].parent / "summary.json")
    ids = None
    if summ:
        ids = summ["n_examples"]
    exs = [json.loads(line) for line in xs[0].read_text(encoding="utf-8").splitlines()]
    return exs, xs[0].parent
"""))

A(md("### Progress so far (including results restored from an earlier session)"))
A(code(r"""
progress()
"""))

# =============================================================================== 1 data
A(md(r"""
---
## Experiment 1: Data, paper-level splits and leakage audit
**Purpose.** Show that the evaluation data is clean and that no information can leak from training into evaluation.

**Protocol.** Both datasets are loaded from the Hub, every file is validated (label vs. `custom_id`, one human and one AI row per
review, schema), repeated human reviews are kept once, and train/validation are split **by paper**. The leakage audit runs on the
*full* partitions before any cap; a FAIL blocks training. If this cell stops with `SchemaError` or `audit FAILED`, read the message:
it lists exactly what was found in the data.
"""))
A(code(r"""
if RUN["prepare"] and BUDGET.allow("data preparation", EST_H["prepare"]):
    for ds in DATASETS:
        run_cmd(f"{PY} scripts/02_prepare_splits.py --dataset {ds}", CODE)
    run_cmd(f"{PY} scripts/02_prepare_splits.py --verify-only", CODE)
    disk_report(WORK)
"""))
A(md("### 1.1 Sizes of the partitions"))
A(code(r"""
rows = []
for ds in DATASETS:
    s = jload(OUT / "manifests" / DD[ds] / "summary.json")
    if not s:
        not_run(f"{NAME[ds]} manifests"); continue
    for part, d in s["full_partitions"].items():
        rows.append({"dataset": NAME[ds], "stage": "full (audited)", "partition": part, "rows": d["rows"], "papers": d["papers"],
                     **{f"label {k}": v for k, v in d["label_counts"].items()}})
    for part, d in s["manifests"].items():
        rows.append({"dataset": NAME[ds], "stage": "used (whole-paper caps)", "partition": part, "rows": d["rows"],
                     "papers": d["papers"], **{f"label {k}": v for k, v in d["label_counts"].items()}})
    ex = pd.Series(s["exclusions"], name="rows").rename_axis("partition:reason").reset_index()
    md_(f"**{NAME[ds]}: rows excluded from the data (every dropped row is logged in `exclusions.parquet`)**")
    display(ex if len(ex) else pd.DataFrame({"note": ["no rows excluded"]}))
if rows:
    md_("**Rows, papers and labels per partition** (labels: 0 human, 1 AI, 2 rewrite). Caps keep whole papers.")
    display(pd.DataFrame(rows).fillna(0))
"""))
A(code(r"""
fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
for ax, ds in zip(axes, DATASETS):
    p = OUT / "manifests" / DD[ds]
    if not (p / "train.parquet").exists():
        ax.set_axis_off(); ax.set_title(f"{NAME[ds]}: not run"); continue
    counts = {part: pd.read_parquet(p / f"{part}.parquet", columns=["label_name"])["label_name"].value_counts()
              for part in ("train", "val", "test")}
    pd.DataFrame(counts).T.plot.bar(ax=ax, rot=0)
    ax.set_title(f"{NAME[ds]}: class counts per split"); ax.set_ylabel("reviews")
save(fig, "data_class_counts")
fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
for ax, ds in zip(axes, DATASETS):
    p = OUT / "manifests" / DD[ds] / "train.parquet"
    if not p.exists():
        ax.set_axis_off(); continue
    df = pd.read_parquet(p, columns=["text", "label_name"])
    df["words"] = df["text"].str.split().str.len()
    for name, g in df.groupby("label_name"):
        ax.hist(g["words"].clip(upper=g["words"].quantile(0.99)), bins=50, alpha=0.5, label=name, density=True)
    ax.set_title(f"{NAME[ds]}: review length (train)"); ax.set_xlabel("words"); ax.legend()
save(fig, "data_review_length")
"""))
A(md(r"""
### 1.2 Data handling that a reviewer will ask about
- **Human reviews repeated across generator files** (IntelLabs) are kept once; how many copies were dropped, and whether any copy differed in text, is counted.
- **Fields not used as text.** Scores, flags and metadata are excluded, and so is `code_of_conduct`, whose length differed strongly by class in the dataset inspection.
  The table recomputes this from the loaded files: each excluded field's mean length per class, averaged over files (a large ratio means the field alone would reveal the label).
- **Sem-Detect** values are read from the real files; nothing about it is assumed.
"""))
A(code(r"""
ls = jload(OUT / "manifests" / DD["intellabs"] / "loader_stats.json")
if ls:
    rows = []
    for sub in ("calibration", "test", "extended"):
        s = ls.get(sub)
        if s:
            rows.append({"subset": sub, "files": s["files"], "rows after dedupe": s["rows"], "human:AI": s["human_to_ai_ratio"],
                         "human rows before dedupe": s["human_rows_before_dedupe"],
                         "copies dropped": s["human_duplicate_copies_dropped"],
                         "copies with different text": s["human_copies_with_different_text"],
                         "conference-years": len(s["conf_years"]), "generators": ", ".join(s["generators"])})
    md_("**IntelLabs: deduplication of human reviews and coverage**")
    display(pd.DataFrame(rows))
    if ls.get("extended_skipped"):
        md_(f"> The optional *extended* subset was **skipped**: `{ls['extended_skipped'][:400]}` The main experiment is unaffected.")
    acc = {}
    for sub in ("calibration", "test"):
        for f in ls.get(sub, {}).get("file_stats", []):
            for field, byl in f.get("never_field_mean_chars_by_label", {}).items():
                for lab, v in byl.items():
                    acc.setdefault(field, {}).setdefault(lab, []).append(v)
    if acc:
        t = pd.DataFrame({fld: {("human" if k == "0" else "AI"): np.mean(v) for k, v in d.items()} for fld, d in acc.items()}).T
        t["ratio"] = t.max(axis=1) / t.min(axis=1).clip(lower=1e-9)
        md_("**Excluded fields: mean length (characters) per class** (a large ratio means the field alone would leak the label)")
        display(t.sort_values("ratio", ascending=False))
    layouts = pd.Series(ls["calibration"]["schemas"]).rename_axis("text fields used (joined in this order)").reset_index(name="files")
    md_("**Review layouts found in the calibration files** (text is built from the allow-listed fields present in each file)")
    display(layouts)
sd = jload(OUT / "manifests" / DD["semdetect"] / "loader_stats.json")
if sd:
    a = sd["all"]
    md_("**Sem-Detect: values observed in the files**")
    display(pd.DataFrame(a["class_by_split"]).fillna(0).astype(int))
    st = pd.Series({"rows (raw)": a["observed_raw"]["rows"], "columns": ", ".join(a["observed_raw"]["columns"]),
                    "naive paper_id overlap train/test": a["naive_paper_id_overlap_train_test"],
                    "composite paper_key overlap train/test": a["paper_key_overlap_train_test"],
                    "submission_id overlap train/test": a["submission_id_overlap_train_test"],
                    "optional columns missing": a["optional_columns_missing"],
                    "dropped invalid rows": a["dropped"]}, name="value")
    display(st.to_frame())
"""))
A(md("### 1.3 Leakage audit"))
A(code(r"""
for ds in DATASETS:
    a = jload(OUT / "audit" / DD[ds] / "leakage_audit.json")
    if not a:
        not_run(f"{NAME[ds]} audit"); continue
    md_(f"**{NAME[ds]}: overall audit status `{a['status']}`** (PASS/WARN allow training, FAIL blocks it)")
    display(pd.DataFrame([{"check": c["check"], "status": c["status"]} for c in a["checks"]]))
    chk = {c["check"]: c for c in a["checks"]}
    ov = chk["paper_overlap"]["overlaps"]
    display(pd.DataFrame({"composite paper_key": ov["composite"], "naive paper_id (info only)": ov["naive_paper_id (info)"]}))
    lr = []
    for part, res in chk["length_only_auroc"]["results"].items():
        for task, v in res.items():
            if task == "per_generator":
                lr += [{"partition": part, "task": f"ai vs human [{g}]", **vv} for g, vv in v.items() if vv]
            elif v:
                lr.append({"partition": part, "task": task, **v})
    md_("Length-only AUROC: how well review length **alone** separates the classes (0.5 = no shortcut; WARN-level, not a blocker).")
    display(pd.DataFrame(lr))
"""))
A(code(r"""
for ds in DATASETS:
    a = jload(OUT / "audit" / DD[ds] / "leakage_audit.json")
    if not a:
        continue
    chk = {c["check"]: c for c in a["checks"]}
    failed = [c for c, v in chk.items() if v["status"] == "FAIL"]
    overlap = chk["paper_overlap"]["overlaps"]["composite"]
    worst = chk["length_only_auroc"]["worst"]
    t = [f"**Analysis ({NAME[ds]}).** Audit verdict **{a['status']}**" +
         (f" (failed: {', '.join(failed)}; training stays blocked until the data issue is fixed)." if failed else ".")]
    t.append("No paper is shared between train, validation and test." if not any(overlap.values()) else
             "Papers shared between splits: " + ", ".join(f"{k} {v}" for k, v in overlap.items() if v) + ".")
    t.append(f"The strongest length-only separation is max(AUROC, 1-AUROC) = {worst:.3f}. " + (
        "Review length alone is a strong signal in this data, so Experiment 4 reports results per length tercile to check that "
        "PeerBERT does not simply rely on it." if worst >= 0.75 else "Review length alone separates the classes only weakly."))
    md_(" ".join(t))
"""))

# =============================================================================== 2 ablation
A(md(r"""
---
## Experiment 2: Ablation and selection of the training variant
**Purpose.** Decide the training configuration **on validation data only** and measure what each component contributes.

**Protocol.**
- Variants: unweighted CE, class-weighted CE, label smoothing 0.05, focal loss (γ=2), R-Drop α ∈ {0.5, 1.0}, dropout off, and the cost-sensitive margin loss with a cosine head (an idea taken from CoCoDet, ACL 2026).
- One seed per variant, the **same** train/validation/test rows for all variants, at a **reduced budget** (see Settings; recorded in the experiment card).
- **Selection rule fixed in advance:** the default configuration (`rdrop_1.0`) is kept unless another variant beats it on validation Macro-F1 by at least `min_gain = 0.005`.
- Test Macro-F1 on a fixed whole-paper sample is **reported for transparency and never used for selection**.
- A single-seed ablation cannot establish small differences; the selected configuration is retrained with 3 seeds in Experiment 3.

First a short smoke run checks the whole GPU path on a tiny subset (and measures speed).
"""))
A(code(r"""
if RUN["smoke"] and BUDGET.allow("smoke run", EST_H["smoke"]):
    jobs = []
    if "intellabs" in DATASETS:
        jobs.append(("smoke_intellabs", f"{PY} scripts/03_train_intellabs.py --smoke --gpu {{gpu}}"))
    if "semdetect" in DATASETS:
        jobs.append(("smoke_semdetect", f"{PY} scripts/04_train_semdetect.py --smoke --gpu {{gpu}}"))
    run_jobs(jobs, CODE, LOGS, n_gpus=min(N_GPUS, len(jobs)))
"""))
A(code(r"""
# Projection of the training time from the measured smoke epoch (rows are treated as equally expensive: conservative)
import yaml
tc = yaml.safe_load((Path(CODE) / "configs" / "train.yaml").read_text())
n_seeds = len(SEEDS) if SEEDS else len(tc["training"]["seeds"])
proj = []
for ds in DATASETS:
    root = OUT / "models" / DD[ds] / "smoke" / "seed42"
    info, summ = jload(root / "run_info.json"), jload(OUT / "models" / DD[ds] / "smoke" / "seed_summary.json")
    hist = jload(root / "history.json")
    ms_ = jload(OUT / "manifests" / DD[ds] / "summary.json")
    if not (info and summ and hist and ms_):
        continue
    rate = (summ["train_rows"] + summ["val_rows"]) / hist[0]["seconds"]            # rows per second
    full_train = ms_["manifests"]["train"]["rows"]; full_val = ms_["manifests"]["val"]["rows"]
    h_final = n_seeds * tc["training"]["epochs"] * (full_train + full_val) / rate / 3600
    ab = ABLATION
    n_var = len(tc["ablation"]["variants"])
    h_abl = n_var * ab["epochs"] * (min(ab["train_rows"], full_train) + min(ab["val_rows"], full_val)) / rate / 3600
    proj.append({"dataset": NAME[ds], "measured rows/s": rate, "projected final training (h)": h_final,
                 "projected ablation (h)": h_abl, "smoke best val Macro-F1": info["best_val_macro_f1"],
                 "AMP": info["amp"], "peak GPU memory (GB)": info.get("peak_gpu_gb"),
                 "train reviews truncated %": 100 * info["truncation"]["train"]["truncated_fraction"]})
if proj:
    pj = pd.DataFrame(proj)
    md_("**Projected GPU time from the measured smoke speed** (per dataset, one GPU; conservative because validation rows are cheaper than training rows)")
    display(pj)
    if len(pj):
        EST_H["train"] = float(pj["projected final training (h)"].max())       # datasets run side by side
        EST_H["ablation"] = float(pj["projected ablation (h)"].max())
        print(f"updated planning estimates (2 GPUs): ablation {EST_H['ablation']:.1f} h, final training {EST_H['train']:.1f} h")
else:
    print("no smoke measurements yet; using the planning estimates from the Settings cell")
"""))
A(code(r"""
if RUN["ablation"] and BUDGET.allow("ablation", EST_H["ablation"]):
    a = ABLATION
    jobs = [(f"ablation_{ds}", f"{PY} scripts/07_run_ablation.py --dataset {ds} --gpu {{gpu}} --max-train-rows {a['train_rows']} "
                               f"--max-val-rows {a['val_rows']} --epochs {a['epochs']} --test-rows {a['test_rows']}")
            for ds in DATASETS]
    run_jobs(jobs, CODE, LOGS, n_gpus=min(N_GPUS, len(jobs)))
    disk_report(WORK)
"""))
A(code(r"""
fig, axes = plt.subplots(1, len(DATASETS), figsize=(7 * len(DATASETS), 4.2), squeeze=False)
shown = False
for ax, ds in zip(axes[0], DATASETS):
    a = jload(RES / DD[ds] / "ablation.json")
    if not a:
        ax.set_axis_off(); not_run(f"{NAME[ds]} ablation"); continue
    d = pd.DataFrame(a["rows"]).sort_values("val_macro_f1")
    y = np.arange(len(d))
    ax.barh(y + 0.2, d["val_macro_f1"], 0.38, label="validation Macro-F1 (selection)",
            color=["#D69E2E" if v == a["selected_variant"] else "#A0AEC0" for v in d["variant"]])
    if d["test_macro_f1"].notna().any():
        ax.barh(y - 0.2, d["test_macro_f1"].fillna(0), 0.38, label="test-sample Macro-F1 (reported, not used)", color="#2C7A7B", alpha=0.7)
    ax.set_yticks(y, d["variant"]); lo = np.nanmin(np.r_[d["val_macro_f1"], d["test_macro_f1"].fillna(1)])
    ax.set_xlim(max(0, lo - 0.05), 1.0); ax.set_xlabel("Macro-F1")
    ax.set_title(f"{NAME[ds]}: selected = {a['selected_variant']}"); shown = True
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
if shown:
    save(fig, "ablation")
else:
    plt.close(fig)
for ds in DATASETS:
    a = jload(RES / DD[ds] / "ablation.json")
    if not a:
        continue
    d = pd.DataFrame(a["rows"]).sort_values("val_macro_f1", ascending=False).reset_index(drop=True)
    md_(f"**{NAME[ds]}: ablation results** (budget: {a['budget']})")
    display(d[["variant", "loss", "rdrop_alpha", "dropout_on", "head", "val_macro_f1", "test_macro_f1", "test_roc_auc", "epochs_run", "minutes"]])
    spread = d["val_macro_f1"].max() - d["val_macro_f1"].min()
    cosine = d[d["variant"] == "csm_cosine"]
    t = [f"**Analysis ({NAME[ds]}).** Selected variant: **{a['selected_variant']}** - {a['selection_rule']}.",
         f"Validation Macro-F1 spans {spread:.4f} across the {len(d)} variants" +
         (" (a spread this small is within what a single seed can produce, so the ranking is weak evidence)." if spread < 2 * a["min_gain"] else "."),
         ("The ablation covers all variants." if a["complete"] else "Not all variants were run; the selection is among those that were.")]
    if len(cosine):
        t.append(f"The CoCoDet-style margin loss reached validation Macro-F1 {cosine.iloc[0]['val_macro_f1']:.4f}.")
    md_(" ".join(t))
"""))

# =============================================================================== 3 training
A(md(r"""
---
## Experiment 3: PeerBERT training
ModernBERT-base is fine-tuned separately on each dataset with the variant selected in Experiment 2, for **3 seeds**. The loss is
class-weighted cross-entropy (weights from the training split only) plus an R-Drop consistency term in the default variant.
Early stopping and the best checkpoint use validation Macro-F1; the decision threshold is calibrated on validation predictions.
IntelLabs and Sem-Detect train at the same time on the two GPUs. Each seed is restart-safe, and a run is reused only if its
configuration and data are unchanged (a changed setting raises an error instead of silently mixing results).
If you deliberately change a training setting after a run has started, delete `outputs/models/<dataset>/final` first.
If training stops with *CUDA out of memory*, lower `training.batch_size` (and raise `grad_accum`) in the `configs/train.yaml` cell, or set
`gradient_checkpointing: true`, then run the step again; the smoke run reports the peak GPU memory so you can see the headroom beforehand.
"""))
A(code(r"""
if RUN["train"] and BUDGET.allow("final training", EST_H["train"]):
    jobs = []
    if "intellabs" in DATASETS:
        jobs.append(("train_intellabs", f"{PY} scripts/03_train_intellabs.py --variant {FINAL_VARIANT} {SEED_ARG} --gpu {{gpu}}"))
    if "semdetect" in DATASETS:
        jobs.append(("train_semdetect", f"{PY} scripts/04_train_semdetect.py --variant {FINAL_VARIANT} {SEED_ARG} --gpu {{gpu}}"))
    run_jobs(jobs, CODE, LOGS, n_gpus=min(N_GPUS, len(jobs)))
    disk_report(WORK)
"""))
A(code(r"""
fig, axes = plt.subplots(2, len(DATASETS), figsize=(6 * len(DATASETS), 7), squeeze=False)
any_run = False
for col, ds in enumerate(DATASETS):
    for d in sorted((OUT / "models" / DD[ds] / TAG).glob("seed*")):
        h = jload(d / "history.json")
        if not h:
            continue
        any_run = True
        h = pd.DataFrame(h)
        axes[0][col].plot(h["epoch"], h["train_loss"], marker="o", label=d.name)
        axes[1][col].plot(h["epoch"], h["val_macro_f1"], marker="o", label=d.name)
    axes[0][col].set_title(f"{NAME[ds]}: training loss"); axes[1][col].set_title(f"{NAME[ds]}: validation Macro-F1")
    axes[1][col].set_xlabel("epoch")
    for r in (0, 1):
        axes[r][col].legend(); axes[r][col].xaxis.set_major_locator(plt.MaxNLocator(integer=True))
if any_run:
    save(fig, "train_curves")
else:
    plt.close(fig); not_run("Training")
rows = []
for ds in DATASETS:
    ss = jload(OUT / "models" / DD[ds] / TAG / "seed_summary.json")
    for d in sorted((OUT / "models" / DD[ds] / TAG).glob("seed*")):
        i = jload(d / "run_info.json")
        if i:
            rows.append({"dataset": NAME[ds], "variant": (ss or {}).get("variant"), "seed": i["seed"],
                         "best val Macro-F1": i["best_val_macro_f1"], "epochs": i["epochs_run"], "minutes": i["minutes"],
                         "AMP": i["amp"], "head": i["model_info"].get("head"),
                         "dropout (attn)": i["model_info"]["dropout_after"].get("attention_dropout"),
                         "R-Drop passes differ": i["rdrop_passes_differ"], "skipped non-finite": i["skipped_nonfinite_steps"],
                         "train truncated %": 100 * i["truncation"]["train"]["truncated_fraction"],
                         "median tokens": i["truncation"]["train"]["tokens_median"]})
if rows:
    display(pd.DataFrame(rows))
    t = pd.DataFrame(rows)
    for ds, g in t.groupby("dataset"):
        md_(f"**Analysis ({ds}).** Best validation Macro-F1 over {len(g)} seed(s): {g['best val Macro-F1'].mean():.4f} ± {g['best val Macro-F1'].std(ddof=1) if len(g) > 1 else 0:.4f}; "
            f"{int((g['skipped non-finite'] > 0).sum())} run(s) had skipped non-finite steps; "
            f"{g['train truncated %'].iloc[0]:.1f}% of training reviews exceed the maximum length and are truncated (reported, not hidden).")
else:
    not_run("Training")
"""))

# =============================================================================== 4 evaluation
A(md(r"""
---
## Experiment 4: PeerBERT evaluation
Evaluation on the official test split (whole-paper capped) and on the shared baseline subset. Thresholds: 0.5 and the threshold calibrated on **validation**.
Sem-Detect is reported natively (3-class) and as a human-vs-AI binary view. Confidence intervals resample **whole papers**.
Accuracy depends on the human:AI ratio of the split (the AI share is shown in the tables), so Macro-F1 and ROC-AUC are the primary metrics.
"""))
A(code(r"""
if RUN["evaluate"] and BUDGET.allow("evaluation", EST_H["evaluate"]):
    jobs = [(f"eval_{ds}", f"{PY} scripts/05_evaluate.py --dataset {ds} --n-boot {N_BOOT} {SEED_ARG} --gpu {{gpu}}") for ds in DATASETS]
    run_jobs(jobs, CODE, LOGS, n_gpus=min(N_GPUS, len(jobs)))
"""))
A(md("### 4.1 Main results"))
A(code(r"""
rows = []
for ds in DATASETS:
    s = jload(RES / DD[ds] / "peerbert" / TAG / "metrics_summary.json")
    if not s:
        not_run(f"{NAME[ds]} evaluation"); continue
    for split, block in s["per_split"].items():
        for kind in ("binary@0.5", "binary@val_threshold"):
            b = block[kind]
            rows.append({"dataset": NAME[ds], "split": split, "view": kind, "seeds": b["n_seeds"], "Accuracy": ms(b, "accuracy"),
                         "Macro-F1": ms(b, "macro_f1"), "F1 (AI)": ms(b, "f1"), "ROC-AUC": ms(b, "roc_auc"), "PR-AUC": ms(b, "pr_auc")})
        if "native_3class" in block:
            b = block["native_3class"]
            rows.append({"dataset": NAME[ds], "split": split, "view": "3-class (native)", "seeds": b["n_seeds"],
                         "Accuracy": ms(b, "accuracy"), "Macro-F1": ms(b, "macro_f1"), "ROC-AUC": ms(b, "macro_ovr_roc_auc")})
if rows:
    md_("**Mean ± std over seeds** (variant: " + ", ".join(f"{NAME[d]} = {(jload(RES / DD[d] / 'peerbert' / TAG / 'metrics_summary.json') or {}).get('variant')}" for d in DATASETS) + ")")
    display(pd.DataFrame(rows).fillna("–"))
rows = []
for ds in DATASETS:
    seed = ref_seed(ds)
    m = jload(RES / DD[ds] / "peerbert" / TAG / f"seed{seed}" / "metrics.json") if seed is not None else None
    if not m or "test" not in m["splits"]:
        continue
    t = m["splits"]["test"]; ci = t.get("cluster_bootstrap_ci", {}); b = t["binary@0.5"]
    rows.append({"dataset": NAME[ds], "seed": seed, "n": b["n"], "AI share": b["n_pos"] / b["n"],
                 "ROC-AUC [95% CI]": fmt_ci(ci["roc_auc"]["estimate"], ci["roc_auc"]["ci_low"], ci["roc_auc"]["ci_high"]) if ci else "–",
                 "Macro-F1 [95% CI]": fmt_ci(ci["macro_f1@0.5"]["estimate"], ci["macro_f1@0.5"]["ci_low"], ci["macro_f1@0.5"]["ci_high"]) if ci else "–",
                 **{f"TPR@{k} FPR": (f"{v['tpr']:.4f}" if v["supported"] else f"n/a (n_neg={v['n_neg']})") for k, v in b["tpr_at_fpr"].items()},
                 "FPR on human": b["ai_fpr"]})
if rows:
    md_("**Test split: paper-level cluster-bootstrap CIs and detection rates at low false-positive rates** (reference seed = median validation Macro-F1; TPR at a given FPR is shown only when enough human reviews exist to estimate it)")
    display(pd.DataFrame(rows))
"""))
A(code(r"""
fig, axes = plt.subplots(1, len(DATASETS), figsize=(6.5 * len(DATASETS), 3.8), squeeze=False)
shown = False
for ax, ds in zip(axes[0], DATASETS):
    s = jload(RES / DD[ds] / "peerbert" / TAG / "metrics_summary.json")
    if not s:
        ax.set_axis_off(); continue
    vals = {"Macro-F1": s["per_split"]["test"]["binary@val_threshold"]["macro_f1"], "ROC-AUC": s["per_split"]["test"]["binary@val_threshold"]["roc_auc"]}
    for j, (k, v) in enumerate(vals.items()):
        ax.bar(j, v["mean"], 0.55, yerr=v["std"], capsize=4, color="#6B46C1", alpha=0.35)
        ax.scatter(np.full(len(v["values"]), j), v["values"], c="#6B46C1", zorder=3, s=25)
    ax.set_xticks(range(len(vals)), list(vals)); ax.set_ylim(min(0.5, min(min(v["values"]) for v in vals.values()) - 0.05), 1.0)
    ax.set_title(f"{NAME[ds]}: test, one dot per seed"); shown = True
if shown:
    save(fig, "peerbert_seed_variability")
else:
    plt.close(fig)
"""))
A(md("### 4.2 Confusion matrices"))
A(code(r"""
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
k = 0
for ds in DATASETS:
    seed = ref_seed(ds)
    m = jload(RES / DD[ds] / "peerbert" / TAG / f"seed{seed}" / "metrics.json") if seed is not None else None
    if not m:
        continue
    for key, lab in [("binary@0.5", "binary")] + ([("native_3class", "3-class")] if ds == "semdetect" else []):
        if k >= 3:
            break
        cm = m["splits"]["test"][key]["confusion_matrix"]
        M = np.array(cm["matrix"]); Mn = M / M.sum(1, keepdims=True).clip(min=1)
        ax = axes[k]; k += 1
        ax.imshow(Mn, cmap="Blues", vmin=0, vmax=1)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, f"{M[i, j]}\n({Mn[i, j]:.1%})", ha="center", va="center", color="white" if Mn[i, j] > 0.5 else "black", fontsize=9)
        ax.set_xticks(range(len(cm["labels"])), cm["labels"]); ax.set_yticks(range(len(cm["labels"])), cm["labels"])
        ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(f"{NAME[ds]} ({lab}), seed {seed}")
for ax in axes[k:]:
    ax.set_axis_off()
if k:
    save(fig, "peerbert_confusion")
else:
    plt.close(fig); not_run("Evaluation")
"""))
A(md("### 4.3 Score distributions, ROC, precision-recall and calibration"))
A(code(r"""
fig, axes = plt.subplots(1, 4, figsize=(19, 4.2))
shown = False
for j, ds in enumerate(DATASETS):
    seed = ref_seed(ds)
    p = peerbert_pred(ds, seed, "test")
    if p is None or p["label"].nunique() < 2:
        continue
    shown = True
    for lab, g in p.groupby("label_name"):
        axes[0].hist(g["score"], bins=40, alpha=0.5, density=True, label=f"{NAME[ds]} {lab}")
    fpr, tpr, _ = roc_curve(p["label"], p["score"]); axes[1].plot(fpr, tpr, label=NAME[ds])
    pr, rc, _ = precision_recall_curve(p["label"], p["score"]); axes[2].plot(rc, pr, label=f"{NAME[ds]} (AI share {p['label'].mean():.2f})")
    fp, mp = calibration_curve(p["label"], p["score"], n_bins=10, strategy="quantile"); axes[3].plot(mp, fp, marker="o", label=NAME[ds])
axes[0].set_xlabel("P(AI)"); axes[0].set_ylabel("density"); axes[0].set_title("Score distribution (test)"); axes[0].set_yscale("log")
axes[1].plot([0, 1], [0, 1], ":", c="grey"); axes[1].set_xscale("log"); axes[1].set_xlim(1e-3, 1); axes[1].set_xlabel("false positive rate (log)")
axes[1].set_ylabel("true positive rate"); axes[1].set_title("ROC (test)")
axes[2].set_xlabel("recall"); axes[2].set_ylabel("precision"); axes[2].set_title("Precision-recall (test)")
axes[3].plot([0, 1], [0, 1], ":", c="grey"); axes[3].set_xlabel("predicted P(AI)"); axes[3].set_ylabel("observed AI rate"); axes[3].set_title("Calibration (test)")
for ax in axes:
    ax.legend(fontsize=7)
if shown:
    save(fig, "peerbert_scores_roc_pr_calibration")
else:
    plt.close(fig); not_run("Evaluation")
"""))
A(md("### 4.4 Where does PeerBERT work? Generators, venues and review length"))
A(code(r"""
def breakdown_plot(key, title, fname, datasets):
    fig, axes = plt.subplots(1, len(datasets), figsize=(6.5 * len(datasets), 3.8), squeeze=False)
    shown = False
    for ax, ds in zip(axes[0], datasets):
        seed = ref_seed(ds)
        m = jload(RES / DD[ds] / "peerbert" / TAG / f"seed{seed}" / "metrics.json") if seed is not None else None
        if not m or key not in m["splits"]["test"]:
            ax.set_axis_off(); continue
        d = pd.DataFrame(m["splits"]["test"][key]).T
        d[["macro_f1", "roc_auc"]].astype(float).plot.bar(ax=ax, rot=30, ylim=(0, 1.02))
        ax.set_title(f"{NAME[ds]}: {title}"); shown = True
        md_(f"**{NAME[ds]}: {title}** (subgroups are smaller than the full test set, so their metrics are noisier)")
        display(d[["n", "n_pos", "n_neg", "macro_f1", "roc_auc", "ai_fpr"]])
    if shown:
        save(fig, fname)
    else:
        plt.close(fig)

breakdown_plot("by_generator@0.5", "per AI generator", "peerbert_by_generator", ["intellabs"])
breakdown_plot("by_conf_year@0.5", "per venue-year", "peerbert_by_venue", DATASETS)
breakdown_plot("by_length_tercile@0.5", "per review-length tercile", "peerbert_by_length", DATASETS)
for ds in DATASETS:
    seed = ref_seed(ds)
    m = jload(RES / DD[ds] / "peerbert" / TAG / f"seed{seed}" / "metrics.json") if seed is not None else None
    if not m:
        continue
    t = m["splits"]["test"]
    lines = [f"**Analysis ({NAME[ds]}, seed {seed}).** Test Macro-F1 {t['binary@0.5']['macro_f1']:.4f}, ROC-AUC {t['binary@0.5']['roc_auc']:.4f} (human vs AI, threshold 0.5)."]
    lt = pd.DataFrame(t.get("by_length_tercile@0.5", {})).T
    if len(lt):
        sp = lt["macro_f1"].astype(float).max() - lt["macro_f1"].astype(float).min()
        lines.append(f"Macro-F1 varies by {sp:.4f} across review-length terciles (subgroup sizes n are in the table above)" + (": small, so no sign of a length shortcut in this view." if sp < 0.03 else ": performance depends on review length; this should be discussed, after checking that the subgroups are large enough for the difference to be more than noise."))
    if "by_generator@0.5" in t:
        g = pd.DataFrame(t["by_generator@0.5"]).T["macro_f1"].astype(float)
        if len(g) > 1:
            lines.append(f"Hardest generator: {g.idxmin()} (Macro-F1 {g.min():.4f}); easiest: {g.idxmax()} ({g.max():.4f}).")
    if "native_3class" in t:
        lines.append("3-class F1: " + ", ".join(f"{k} {v['f1']:.4f}" for k, v in t["native_3class"]["per_class"].items()) + ".")
    md_(" ".join(lines))
"""))
A(md("### 4.5 Robustness on the IntelLabs *extended* subset and error analysis"))
A(code(r"""
s = jload(RES / DD["intellabs"] / "peerbert" / TAG / "metrics_summary.json")
if s and "extended" in s["per_split"]:
    e, t = s["per_split"]["extended"]["binary@val_threshold"], s["per_split"]["test"]["binary@val_threshold"]
    md_("**Extended subset** (generators gpt4o and llama; papers that appear in train/validation were excluded; optional robustness test)")
    display(pd.DataFrame({"test": {k: ms(t, k) for k in ("macro_f1", "roc_auc", "f1")},
                          "extended": {k: ms(e, k) for k in ("macro_f1", "roc_auc", "f1")}}))
    md_(f"**Analysis.** Extended-subset Macro-F1 is {e['macro_f1']['mean']:.4f} versus {t['macro_f1']['mean']:.4f} on the test split "
        f"(difference {e['macro_f1']['mean'] - t['macro_f1']['mean']:+.4f}).")
else:
    md_("> Extended subset: not evaluated (disabled, skipped at loading, or evaluation not run yet).")
for ds in DATASETS:
    seed = ref_seed(ds)
    d = RES / DD[ds] / "peerbert" / TAG / f"seed{seed}"
    for name in ("test_binary_false_positives", "test_binary_false_negatives"):
        p = d / f"{name}.csv"
        if p.exists():
            t = pd.read_csv(p)
            md_(f"**{NAME[ds]}: {name.replace('_', ' ')}** (most confident mistakes; {len(t)} stored)")
            display(t.head(5))
"""))

# =============================================================================== 5 baselines
A(md(r"""
---
## Experiment 5: Baselines
Eight detectors on the **same shared test subset** as PeerBERT; all thresholds calibrated on the same validation sample.
- **Zero-shot detectors:** LogRank, DetectLLM-LRR, Fast-DetectGPT (analytic), Binoculars, DNA-GPT.
- **Feature-based trained classifiers:** GLTR, MFD (fit on a capped training sample only).
- **Pretrained detector:** RADAR.
- On Sem-Detect, baselines see human/ai reviews only; rewrite reviews are reported as the share flagged as AI.
- IntelLabs baselines run on one GPU and Sem-Detect baselines on the other.

The table at the end of this experiment lists, for every baseline, what was verified against the original method and every deviation.
"""))
A(code(r"""
if RUN["baselines"] and BUDGET.allow("baselines", EST_H["baselines"]):
    groups = ["logrank gltr detectllm_lrr mfd fast_detectgpt", "radar binoculars", "dnagpt"]
    jobs = []
    for g in groups:
        for ds in DATASETS:       # interleaved: dataset 0 -> GPU 0, dataset 1 -> GPU 1
            jobs.append((f"base_{ds}_{g.split()[0]}", f"{PY} scripts/06_run_baselines.py --dataset {ds} --only {g} --gpu {{gpu}}"))
    run_jobs(jobs, CODE, LOGS, n_gpus=min(N_GPUS, len(DATASETS)))
    disk_report(WORK)
"""))
A(md("### 5.1 All methods on the same reviews, with confidence intervals"))
A(code(r"""
CI = {}
for ds in DATASETS:
    t = ci_table(ds)
    if t.empty:
        not_run(f"{NAME[ds]} comparison"); continue
    CI[ds] = t
    show = t.copy()
    show["ROC-AUC [95% CI]"] = [fmt_ci(a, l, h) for a, l, h in zip(t["ROC-AUC"], t["AUC lo"], t["AUC hi"])]
    show["Macro-F1 [95% CI]"] = [fmt_ci(a, l, h) for a, l, h in zip(t["Macro-F1"], t["F1 lo"], t["F1 hi"])]
    md_(f"**{NAME[ds]}: shared test subset, validation-calibrated thresholds** (CIs resample whole papers; n = {int(t['n'].iloc[0])} reviews; "
        f"AI share {t['AI share'].iloc[0]:.2f}, which is the PR-AUC of a random scorer)")
    display(show[["type", "method", "ROC-AUC [95% CI]", "Macro-F1 [95% CI]", "PR-AUC", "TPR@10%FPR", "FPR on human"]]
            .sort_values(["type", "method"]).reset_index(drop=True))
"""))
A(code(r"""
fig, axes = plt.subplots(len(CI) or 1, 2, figsize=(13, 4.2 * (len(CI) or 1)), squeeze=False)
for r, (ds, t) in enumerate(CI.items()):
    for c, (col, lo, hi, lab) in enumerate((("ROC-AUC", "AUC lo", "AUC hi", "ROC-AUC"), ("Macro-F1", "F1 lo", "F1 hi", "Macro-F1 (val-calibrated threshold)"))):
        d = t.sort_values(col)
        ax = axes[r][c]
        ax.barh(d["method"], d[col], xerr=[d[col] - d[lo], d[hi] - d[col]], color=[TYPE_COLOR.get(x, "grey") for x in d["type"]], capsize=3, alpha=0.9)
        ax.set_xlim(0, 1.02); ax.set_xlabel(lab); ax.set_title(f"{NAME[ds]}: shared test subset (95% CI)")
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in TYPE_COLOR.values()]
if CI:
    fig.legend(handles, TYPE_COLOR.keys(), loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02))
    save(fig, "comparison_with_ci")
else:
    plt.close(fig)
"""))
A(code(r"""
fig, axes = plt.subplots(1, len(DATASETS), figsize=(6.5 * len(DATASETS), 5), squeeze=False)
shown = False
for ax, ds in zip(axes[0], DATASETS):
    for name, (typ, p, thr) in method_predictions(ds).items():
        if p["label"].nunique() == 2:
            fpr, tpr, _ = roc_curve(p["label"], p["score"])
            ax.plot(fpr, tpr, lw=2.6 if name == "PeerBERT" else 1.2, color=None if name != "PeerBERT" else "#6B46C1", label=name); shown = True
    ax.plot([0, 1], [0, 1], ":", c="grey"); ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate")
    ax.set_title(f"{NAME[ds]}: ROC on the shared test subset"); ax.legend(fontsize=7, loc="lower right")
if shown:
    save(fig, "comparison_roc")
else:
    plt.close(fig)
"""))
A(md("### 5.2 Which generators are hard for which detector?"))
A(code(r"""
mp = method_predictions("intellabs")
if mp:
    cols = {}
    for name, (typ, p, thr) in mp.items():
        g = by_generator(p)
        cols[name] = {gen: v["roc_auc"] for gen, v in g.items()}
    H = pd.DataFrame(cols).T
    if H.shape[1]:
        fig, ax = plt.subplots(figsize=(1.6 * H.shape[1] + 3, 0.5 * len(H) + 1.5))
        im = ax.imshow(H.astype(float).values, cmap="viridis", vmin=0.4, vmax=1.0, aspect="auto")
        ax.set_xticks(range(H.shape[1]), H.columns, rotation=30); ax.set_yticks(range(len(H)), H.index)
        for i in range(H.shape[0]):
            for j in range(H.shape[1]):
                v = H.iloc[i, j]
                ax.text(j, i, "–" if pd.isna(v) else f"{v:.2f}", ha="center", va="center", color="white" if (not pd.isna(v) and v < 0.75) else "black", fontsize=8)
        fig.colorbar(im, ax=ax, label="ROC-AUC"); ax.set_title("IntelLabs: ROC-AUC per AI generator (shared test subset; small subgroups)")
        save(fig, "comparison_by_generator")
        md_("Per-generator subsets of the shared test subset are small, so differences of a few points are not reliable; read the pattern across detectors, not single cells.")
else:
    not_run("Per-generator comparison")
"""))
A(md("### 5.3 Paired significance tests, and what each baseline is"))
A(code(r"""
# paired tests are computed by scripts/09_build_report.py; run it here so this section is self-contained
if any((RES / DD[d] / "baselines").exists() for d in DATASETS):
    try:
        run_cmd(f"{PY} scripts/09_build_report.py --n-boot {N_BOOT}", CODE)
    except RuntimeError as e:
        print("report step failed:", str(e)[:500])
for ds in DATASETS:
    p = RES / "report" / f"{ds}_paired_tests.csv"
    if p.exists():
        t = pd.read_csv(p)
        md_(f"**{NAME[ds]}: PeerBERT vs each baseline on the same reviews** (positive ROC-AUC difference = PeerBERT better; paper-level paired bootstrap, McNemar at the validation threshold, Holm-adjusted across baselines)")
        display(t)
        if "p_bootstrap_holm" in t:
            sig = t[t["p_bootstrap_holm"] < 0.05]
            better = sig[sig["auc_diff_peerbert_minus_baseline"] > 0]["baseline"].tolist()
            worse = sig[sig["auc_diff_peerbert_minus_baseline"] < 0]["baseline"].tolist()
            ns = t[~t["baseline"].isin(sig["baseline"])]["baseline"].tolist()
            md_(f"**Analysis ({NAME[ds]}).** PeerBERT has significantly higher ROC-AUC than: {', '.join(better) or 'none'}; significantly lower than: {', '.join(worse) or 'none'}; "
                f"no significant difference (alpha 0.05, Holm): {', '.join(ns) or 'none'}. With a {int(CI[ds]['n'].iloc[0]) if ds in CI else 'small'}-review subset, "
                f'a non-significant result means "not distinguishable here", not "equal".')
from src.baselines.registry import BASELINES
md_("**Baseline fidelity: what was verified against the original method and what deviates**")
display(pd.DataFrame([{"baseline": v["display"], "type": v["type"], "verified": "; ".join(v["verified"]) or "not verified",
                       "deviations": "; ".join(v["deviations"])} for v in BASELINES.values()]))
rows = []
for ds in DATASETS:
    for d in sorted((RES / DD[ds] / "baselines").glob("*/metrics.json")):
        m = jload(d)
        r = {"dataset": NAME[ds], "baseline": m["display"], "val threshold": m["val_threshold"]}
        if "orientation" in m["info"]:
            r["note"] = f"RADAR: output index {m['info']['orientation']['used_ai_index']} chosen as AI on validation (AUROC per output {np.round(m['info']['orientation']['val_auroc_per_output'], 3)})"
        if "rewrite_fraction_flagged_ai@val_threshold" in m:
            rw = m["rewrite_fraction_flagged_ai@val_threshold"]; r["rewrite flagged as AI"] = f"{rw['fraction']:.1%} of {rw['n_rewrite']}"
        rows.append(r)
if rows:
    md_("**Baseline details** (Sem-Detect rewrite reviews are only reported as the share flagged as AI)")
    display(pd.DataFrame(rows).fillna(""))
"""))

# =============================================================================== 6 cross
A(md(r"""
---
## Experiment 6: Cross-dataset transfer
Each PeerBERT model is applied to the **other** dataset **without retraining** (human vs AI only; Sem-Detect rewrite excluded). The decision threshold comes from the
**source** validation set. The datasets differ in generators, label definitions, venues and review templates, so a drop is expected; its size is the finding.
"""))
A(code(r"""
if RUN["cross"] and BUDGET.allow("cross-dataset transfer", EST_H["cross"]):
    jobs = [(f"cross_{src}", f"{PY} scripts/08_cross_dataset.py --source {src} {SEED_ARG} --gpu {{gpu}}") for src in DATASETS]
    run_jobs(jobs, CODE, LOGS, n_gpus=min(N_GPUS, len(jobs)))
"""))
A(code(r"""
rows = []
for src in DATASETS:
    tgt = "semdetect" if src == "intellabs" else "intellabs"
    s_in = jload(RES / DD[src] / "peerbert" / TAG / "metrics_summary.json")
    c = jload(RES / DD[tgt] / f"cross_from_{src}" / "metrics.json")
    if s_in:
        b = s_in["per_split"]["test"]["binary@val_threshold"]
        rows.append({"train on": NAME[src], "test on": NAME[src] + " (in-domain)", "Macro-F1": b["macro_f1"]["mean"], "ROC-AUC": b["roc_auc"]["mean"],
                     "Macro-F1 std": b["macro_f1"]["std"], "ROC-AUC std": b["roc_auc"]["std"]})
    if c:
        b = c["mean_std@source_val_threshold"]
        rows.append({"train on": NAME[src], "test on": NAME[tgt] + " (transfer)", "Macro-F1": b["macro_f1"]["mean"], "ROC-AUC": b["roc_auc"]["mean"],
                     "Macro-F1 std": b["macro_f1"]["std"], "ROC-AUC std": b["roc_auc"]["std"]})
if rows:
    d = pd.DataFrame(rows); display(d)
    fig, ax = plt.subplots(figsize=(8.5, 3.8)); x = np.arange(len(d))
    ax.bar(x - 0.18, d["Macro-F1"], 0.36, yerr=d["Macro-F1 std"], capsize=3, label="Macro-F1")
    ax.bar(x + 0.18, d["ROC-AUC"], 0.36, yerr=d["ROC-AUC std"], capsize=3, label="ROC-AUC")
    ax.set_xticks(x, [f"{a}\n-> {b}" for a, b in zip(d["train on"], d["test on"])], fontsize=8); ax.set_ylim(0, 1.05)
    ax.set_title("In-domain vs cross-dataset (mean ± std over seeds)"); ax.legend(); save(fig, "cross_dataset")
    for src in DATASETS:
        tgt = "semdetect" if src == "intellabs" else "intellabs"
        a, b = d[(d["train on"] == NAME[src]) & d["test on"].str.contains("in-domain")], d[(d["train on"] == NAME[src]) & d["test on"].str.contains("transfer")]
        if len(a) and len(b):
            dF, dA = b["Macro-F1"].iloc[0] - a["Macro-F1"].iloc[0], b["ROC-AUC"].iloc[0] - a["ROC-AUC"].iloc[0]
            verb = "drops" if dF < -0.005 else ("rises" if dF > 0.005 else "stays about the same")
            md_(f"**Analysis.** Trained on {NAME[src]}, Macro-F1 {verb}: {a['Macro-F1'].iloc[0]:.4f} in-domain -> {b['Macro-F1'].iloc[0]:.4f} on {NAME[tgt]} "
                f"({dF:+.4f}); ROC-AUC {a['ROC-AUC'].iloc[0]:.4f} -> {b['ROC-AUC'].iloc[0]:.4f} ({dA:+.4f}). "
                "The ROC-AUC change shows how much of the shift is in the ranking of reviews; the remainder of a Macro-F1 change comes from the decision threshold, "
                "which was fixed on the source validation set and not adapted to the target.")
    c = jload(RES / "sem_detect" / "cross_from_intellabs" / "metrics.json") or jload(RES / "intellabs" / "cross_from_semdetect" / "metrics.json")
    if c:
        md_("**Known mismatches between the datasets:**\n" + "\n".join(f"- {n}" for n in c["mismatch_notes"]))
else:
    not_run("Cross-dataset transfer")
"""))

# =============================================================================== 7 xai
A(md(r"""
---
## Experiment 7: Explainability
**What is explained.** Reviews of the shared test subset, balanced over (true class × correct/incorrect), explained for the model's **predicted** class.
- **Integrated Gradients** on the token embeddings (signed; baseline = pad tokens with special tokens kept; the completeness gap is reported).
- **Attention** from an eager-attention copy: last layer, mean over layers, rollout.

**How the explanations are tested.**
1. *Faithfulness* (ERASER): remove the top-k% tokens (**comprehensiveness**, higher is better) or keep only them (**sufficiency**, lower is better), compared with a random ranking.
2. *Weight-randomisation sanity check* (Adebayo et al., 2018): attributions of the trained model must differ from those of a randomly initialised copy.
3. *Convergence*: the Integrated Gradients completeness gap relative to the prediction change.
"""))
A(code(r"""
if RUN["xai"] and BUDGET.allow("explainability", EST_H["xai"]):
    jobs = [(f"xai_{ds}", f"{PY} scripts/10_explain.py --dataset {ds} --n-per-group {XAI_PER_GROUP} --gpu {{gpu}}") for ds in DATASETS]
    run_jobs(jobs, CODE, LOGS, n_gpus=min(N_GPUS, len(jobs)))
"""))
A(code(r"""
from scipy.stats import wilcoxon
XAI = {ds: load_xai(ds) for ds in DATASETS}
fig, axes = plt.subplots(len(DATASETS), 2, figsize=(13, 4 * len(DATASETS)), squeeze=False)
shown = False
for r, ds in enumerate(DATASETS):
    exs, folder = XAI[ds]
    if not exs:
        axes[r][0].set_axis_off(); axes[r][1].set_axis_off(); not_run(f"{NAME[ds]} explainability"); continue
    shown = True
    kmin = min(float(k) for k in exs[0]["faithfulness"]["random"])
    methods = ["integrated_gradients", "attention_last", "attention_rollout", "random"]
    data = {m: [e["faithfulness"][m][str(kmin)]["comprehensiveness"] for e in exs] for m in methods}
    suff = {m: [e["faithfulness"][m][str(kmin)]["sufficiency"] for e in exs] for m in methods}
    for col, series, ttl in ((0, data, "comprehensiveness (higher = more faithful)"), (1, suff, "sufficiency (lower = more faithful)")):
        axes[r][col].boxplot(list(series.values()), showmeans=True)
        axes[r][col].set_xticks(range(1, len(series) + 1), [m.replace("_", "\n") for m in series])
        axes[r][col].axhline(0, c="grey", lw=0.8); axes[r][col].set_title(f"{NAME[ds]}: {ttl} at k={kmin:.0%}")
    rows = []
    for m in methods[:-1]:
        d = np.array(data[m]) - np.array(data["random"])
        try:
            pval = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
        except ValueError:
            pval = np.nan
        ds_ = np.array(suff[m]) - np.array(suff["random"])
        try:
            pval_s = wilcoxon(ds_).pvalue if np.any(ds_ != 0) else 1.0
        except ValueError:
            pval_s = np.nan
        rows.append({"method": m, "n": len(d), "mean comprehensiveness": np.mean(data[m]), "random": np.mean(data["random"]),
                     "median diff vs random": np.median(d), "Wilcoxon p (comp.)": pval,
                     "mean sufficiency": np.mean(suff[m]), "Wilcoxon p (suff.)": pval_s})
    t = pd.DataFrame(rows)
    md_(f"**{NAME[ds]}: faithfulness vs. a random ranking at k = {kmin:.0%}** (paired Wilcoxon signed-rank over reviews; per-review values are not independent across reviews of the same paper)")
    display(t)
    good = t[(t["median diff vs random"] > 0) & (t["Wilcoxon p (comp.)"] < 0.05)]["method"].tolist()
    att_good = [m for m in good if m.startswith("attention")]
    att_txt = ("Attention also beats the random ranking here; it is still a heuristic rather than an explanation (Jain & Wallace, 2019), "
               "so Integrated Gradients remains the primary explanation method." if att_good else
               "Attention does not beat the random ranking here, consistent with the known caveat that attention weights are not explanations by themselves (Jain & Wallace, 2019).")
    md_(f"**Analysis ({NAME[ds]}).** Methods with significantly higher comprehensiveness than a random ranking at k = {kmin:.0%}: {', '.join(good) or 'none'}. " + att_txt)
if shown:
    save(fig, "xai_faithfulness")
else:
    plt.close(fig)
"""))
A(code(r"""
fig, axes = plt.subplots(len(DATASETS), 3, figsize=(15, 3.8 * len(DATASETS)), squeeze=False)
shown = False
for r, ds in enumerate(DATASETS):
    exs, folder = XAI[ds]
    if not exs:
        for a in axes[r]:
            a.set_axis_off()
        continue
    shown = True
    rel = np.array([e["ig_delta"] / max(abs(e["p_target"] - e["p_baseline"]), 1e-3) for e in exs])
    axes[r][0].hist(np.clip(rel, 0, np.quantile(rel, 0.99) if len(rel) > 10 else rel.max() + 1e-9), bins=30, color="#B83280", alpha=0.8)
    axes[r][0].set_title(f"{NAME[ds]}: IG completeness gap / prediction change"); axes[r][0].set_xlabel("relative gap")
    for k, color in (("last_layer", "#2C7A7B"), ("mean_layers", "#6B46C1"), ("rollout", "#C05621")):
        v = [e["spearman_abs_ig_vs_attention"][k] for e in exs if e["spearman_abs_ig_vs_attention"][k] is not None]
        axes[r][1].hist(v, bins=20, alpha=0.5, color=color, label=k)
    axes[r][1].set_title(f"{NAME[ds]}: Spearman(|IG|, attention)"); axes[r][1].legend(fontsize=7)
    sm = (jload(folder / "summary.json") or {}).get("sanity_randomization")
    if sm:
        v = [e["spearman_abs"] for e in sm["per_example"] if e["spearman_abs"] is not None]
        axes[r][2].hist(v, bins=15, color="#4A5568", range=(-1, 1)); axes[r][2].axvline(np.mean(v), c="red")
        axes[r][2].set_title(f"{NAME[ds]}: trained vs randomised model (Spearman |IG|)"); axes[r][2].set_xlim(-1, 1)
    else:
        axes[r][2].set_axis_off()
    s_ = jload(folder / "summary.json")
    md_(f"**{NAME[ds]}** ({len(exs)} reviews, {s_['ig']['n_steps']} IG steps). Median relative completeness gap: **{np.median(rel):.3f}** "
        f"(share above 0.05: {np.mean(rel > 0.05):.1%}; if it is large, raise `--n-steps`). Weight-randomisation check: mean Spearman(|IG| trained, |IG| randomised) = "
        f"**{(sm or {}).get('spearman_abs', {}).get('mean')}** on {(sm or {}).get('n')} reviews. Values near 0 mean the attributions depend on the learned weights; "
        "values near 1 would mean they mostly reflect the input or architecture.")
if shown:
    save(fig, "xai_convergence_agreement_sanity")
else:
    plt.close(fig)
"""))
A(code(r"""
for ds in DATASETS:
    xs = sorted((RES / DD[ds] / "xai" / TAG).glob("seed*/top_tokens.csv"))
    if not xs:
        continue
    t = pd.read_csv(xs[0])
    classes = list(t["explained_class"].unique())
    if not classes:
        continue
    fig, axes = plt.subplots(1, len(classes), figsize=(5 * len(classes), 4.5), squeeze=False)
    for ax, c in zip(axes[0], classes):
        d = t[t["explained_class"] == c].nlargest(15, "mean").iloc[::-1]
        ax.barh(d["token"], d["mean"], color="#B83280"); ax.set_title(f"{NAME[ds]}: tokens pushing to '{c}'")
        ax.set_xlabel("mean Integrated Gradients attribution")
    save(fig, f"xai_top_tokens_{ds}")
    md_("Top tokens are averages over the explained reviews: they describe what the model uses, and are **not** a proof that these words cause the classification. Check whether they look like content (e.g. domain terms) or boilerplate/style.")
for ds in DATASETS:
    xs = sorted((RES / DD[ds] / "xai" / TAG).glob("seed*/examples.html"))
    if xs:
        md_(f"**{NAME[ds]}: explained reviews** (red pushes towards the predicted class, blue against it). Full page: `{xs[0]}`")
        display(HTML(f"<div style='max-height:600px;overflow:auto;border:1px solid #ddd;padding:8px'>{xs[0].read_text()}</div>"))
"""))

# =============================================================================== 8 report
A(md(r"""
---
## Experiment 8: Report, experiment card and results bundle
Collects every result into thesis-ready tables (CSV + LaTeX), writes the **experiment card** (exact settings, data counts, library versions,
code hash, selection decisions, every deviation and limitation, all read from the files this run produced), and zips the results for download.
"""))
A(code(r"""
if RUN["report"] and BUDGET.allow("report", EST_H["report"]):
    run_cmd(f"{PY} scripts/09_build_report.py --n-boot {N_BOOT}", CODE)
    run_cmd(f"{PY} scripts/11_experiment_card.py", CODE)
card = RES / "report" / "experiment_card.md"
if card.exists():
    display(Markdown(card.read_text()))
else:
    not_run("Experiment card")
"""))
A(code(r"""
import zipfile
bundle = Path(WORK) / "PeerBERT_results_bundle.zip"
if RES.exists():
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as z:
        for root in (RES, OUT / "audit"):
            for p in root.rglob("*"):
                if p.is_file() and p.suffix in (".csv", ".json", ".png", ".pdf", ".md", ".tex", ".html", ".jsonl"):
                    z.write(p, p.relative_to(OUT))
        for ds in DATASETS:
            for name in ("summary.json", "loader_stats.json"):
                p = OUT / "manifests" / DD[ds] / name
                if p.exists():
                    z.write(p, p.relative_to(OUT))
    print(f"results bundle: {bundle} ({bundle.stat().st_size / 1e6:.1f} MB): tables, figures (PNG+PDF), metrics, experiment card")
print("figures :", FIG)
for f in sorted(FIG.glob("*.png")):
    print("   ", f.name)
print("tables  :", RES / "report")
disk_report(WORK)
print(f"session time used: {BUDGET.elapsed_h():.1f} h of the {TIME_BUDGET_H:.0f} h budget")
progress()
"""))
A(md(r"""
### Limitations to state in the paper
The experiment card lists the ones that follow from the actual settings. In general:
- Both datasets are English ML-conference reviews from a fixed set of LLM generators; no claim is made beyond them.
- Hyper-parameters other than the ablated loss components (learning rate, epochs, batch size, maximum length) were fixed in advance, not tuned.
- The ablation is single-seed and at a reduced budget; differences below seed noise are not evidence.
- Baselines see a shared subset (default 600 reviews) with a 512-token context, so their confidence intervals are wide; deviations of individual baselines are listed in the fidelity table.
- Reviews longer than the maximum length are truncated; the truncation rate is reported.
- Significance tests treat papers as independent clusters and do not model venue, reviewer or generator effects.
- Explanations describe the model's behaviour, not human-verifiable reasons; they are checked for faithfulness and sensitivity to the learned weights, which is necessary but not sufficient for causal claims.
"""))

for i, c in enumerate(cells):
    c["id"] = f"cell-{i:03d}"
nb = {"cells": cells,
      "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4", "toc_visible": True},
                   "kaggle": {"accelerator": "nvidiaTeslaT4", "isGpuEnabled": True, "isInternetEnabled": True},
                   "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"wrote {OUT} ({len(cells)} cells, {OUT.stat().st_size / 1e3:.0f} KB)")
