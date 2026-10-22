"""
Builds notebooks/PeerBERT_full_experiments.ipynb: one self-contained notebook that writes the
project code into the runtime (no git clone needed), runs every experiment on the original
datasets and models, and shows tables, figures and a short data-driven analysis after each one.

    python notebooks/make_full_notebook.py

Re-run it after changing any file under src/, scripts/ or configs/ so the notebook stays in sync.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "notebooks" / "PeerBERT_full_experiments.ipynb"
CODE_DIRS = ("configs", "src", "scripts")
SKIP = {"configs/inspection_report.json"}


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)}


def code(text, hidden=False):
    meta = {"cellView": "form", "jupyter": {"source_hidden": True}} if hidden else {}
    return {"cell_type": "code", "execution_count": None, "metadata": meta, "outputs": [],
            "source": text.strip("\n").splitlines(True)}


def project_files():
    files = []
    for d in CODE_DIRS:
        for p in sorted((ROOT / d).rglob("*")):
            rel = p.relative_to(ROOT).as_posix()
            if p.is_file() and p.suffix in (".py", ".yaml") and "__pycache__" not in rel and rel not in SKIP:
                files.append((rel, p.read_text(encoding="utf-8")))
    return files


cells = []
A = cells.append

# ============================================================================ intro
A(md(r"""
# PeerBERT: Explainable Classification of AI-Generated Peer Reviews

**This notebook contains the whole project.** It writes the code into the runtime, runs every
experiment on the **original datasets** with the **original models**, and shows the results
with figures and a short analysis after each experiment.

| # | Experiment | What it produces |
|---|---|---|
| 1 | Data loading, paper-level splits, leakage audit | dataset statistics, audit verdict, length-shortcut check |
| 2 | PeerBERT training (ModernBERT-base, 3 seeds per dataset) | training curves |
| 3 | PeerBERT evaluation | metrics with 95% CIs, confusion matrices, ROC/PR, per-generator, per-venue, length terciles, calibration, errors |
| 4 | Eight baselines on the shared test subset | comparison table, ROC overlay, TPR at low FPR, paired significance tests |
| 5 | Ablation (loss variants, R-Drop, dropout, CoCoDet-style margin loss) | validation Macro-F1 per variant |
| 6 | Cross-dataset transfer (no retraining) | in-domain vs. transfer performance |
| 7 | Explainability (Integrated Gradients + attention) | faithfulness vs. random, top tokens, highlighted reviews |
| 8 | Final report | all tables (CSV/LaTeX) and figures (PNG/PDF) |

Datasets: [IntelLabs/AI-Peer-Review-Detection-Benchmark](https://huggingface.co/datasets/IntelLabs/AI-Peer-Review-Detection-Benchmark)
(binary) and [Sem-Detect/ML_Conferences-Peer-Reviews](https://huggingface.co/datasets/Sem-Detect/ML_Conferences-Peer-Reviews)
(human / ai / rewrite). Model: [answerdotai/ModernBERT-base](https://huggingface.co/answerdotai/ModernBERT-base).

## How to run
1. **Runtime -> Change runtime type -> T4 GPU** (an A100/L4 is faster).
2. **Runtime -> Run all**, or run the cells one by one.
3. If Colab disconnects, reconnect and **Run all** again. Finished work is skipped, and training
   resumes from its last finished epoch, because everything is saved to Google Drive.

The full run takes many GPU hours (training 2 datasets x 3 seeds, plus 8 baselines). You can
switch experiments on and off in the **Settings** cell, and run them across several sessions.
"""))

# ============================================================================ setup
A(md("## 0. Settings"))
A(code(r"""
# ---- what to run ------------------------------------------------------------------------
RUN = {
    "prepare":   True,   # 1. load datasets, split by paper, leakage audit
    "smoke":     True,   # quick 1-epoch training check before the real training
    "train":     True,   # 2. PeerBERT, all seeds, both datasets
    "evaluate":  True,   # 3. PeerBERT evaluation
    "baselines": True,   # 4. eight baselines
    "ablation":  True,   # 5. ablation variants (single seed, validation selection)
    "cross":     True,   # 6. cross-dataset transfer
    "xai":       True,   # 7. explainability
}
DATASETS = ["intellabs", "semdetect"]
SEEDS = None          # None = seeds from configs/train.yaml ([42, 43, 44]); e.g. [42] for a faster first run
N_BOOT = 1000         # bootstrap resamples for confidence intervals
XAI_PER_GROUP = 25    # explained reviews per (true class, correct/incorrect) group
"""))

A(code(r"""
import os, sys, subprocess
ON_KAGGLE = os.path.exists("/kaggle/working")
if ON_KAGGLE:
    WORK = "/kaggle/working/PeerBERT_outputs"
else:
    try:
        from google.colab import drive
        drive.mount("/content/drive")
        WORK = "/content/drive/MyDrive/PeerBERT_outputs"
    except ImportError:                       # local Jupyter
        WORK = os.path.abspath("PeerBERT_outputs")
CODE = "/content/PeerBERT" if os.path.exists("/content") else os.path.abspath("PeerBERT_code")
os.makedirs(WORK, exist_ok=True)
os.makedirs(CODE, exist_ok=True)
os.environ["PRAD_WORK_DIR"] = WORK
print("code    ->", CODE)
print("outputs ->", WORK)
print(subprocess.run("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader", shell=True,
                     capture_output=True, text=True).stdout or "no GPU found: select a GPU runtime")
"""))

A(md(r"""
### Install dependencies
torch is preinstalled on Colab. Optionally, add a Hugging Face token as a Colab secret named
`HF_TOKEN` (key icon on the left) for faster downloads without rate limits.
"""))
A(code(r"""
!pip install -q "transformers>=4.48.0" "huggingface_hub>=0.22" "accelerate>=0.26" "captum>=0.7.0" pyarrow scikit-learn scipy matplotlib pyyaml packaging
try:
    from google.colab import userdata
    os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
    print("HF_TOKEN loaded")
except Exception:
    print("no HF_TOKEN secret: anonymous downloads")
import torch, transformers
print("torch", torch.__version__, "| transformers", transformers.__version__, "| CUDA", torch.cuda.is_available())
"""))

# ============================================================================ code
A(md(r"""
## Project code
The next cells write the project code (configs, `src/`, `scripts/`) into the runtime.

**To change a setting, edit the config cells** (`configs/config.yaml`, `configs/train.yaml`,
`configs/baselines.yaml`) and re-run them before the experiment.
"""))
A(code(r"""
for d in ["configs", "scripts", "src/data", "src/models", "src/evaluation", "src/xai", "src/baselines/statistical",
          "src/baselines/zero_shot", "src/baselines/neural_detectors"]:
    os.makedirs(os.path.join(CODE, d), exist_ok=True)
%cd {CODE}
"""))
for rel, text in project_files():
    if not text.strip():
        continue
    A(code(f"%%writefile {rel}\n{text}", hidden=not rel.startswith("configs/")))
A(code(r"""
# empty package markers
for d, _, _ in os.walk(os.path.join(CODE, "src")):
    open(os.path.join(d, "__init__.py"), "a").close()
sys.path.insert(0, CODE)
print("project code written to", CODE)
"""))

# ============================================================================ helpers
A(md("## Helpers for running steps and plotting"))
A(code(r"""
import json, glob, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown, HTML
warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.width", 200, "display.max_columns", 40, "display.precision", 4)
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 300, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})

from src.evaluation.predict import binary_view
from src.evaluation.metrics import binary_metrics

OUT = Path(WORK) / "outputs"
RES = OUT / "results"
FIG = RES / "figures"
FIG.mkdir(parents=True, exist_ok=True)
DD = {"intellabs": "intellabs", "semdetect": "sem_detect"}
NAME = {"intellabs": "IntelLabs", "semdetect": "Sem-Detect"}
TAG = "final"
SEED_ARG = f"--seeds {' '.join(map(str, SEEDS))}" if SEEDS else ""


def run(cmd):
    # Run a pipeline script, streaming its output; stops the notebook on failure.
    print("$", cmd, flush=True)
    p = subprocess.Popen(cmd, shell=True, cwd=CODE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, bufsize=1)
    for line in p.stdout:
        print(line, end="")
    if p.wait() != 0:
        raise RuntimeError(f"step failed (exit code {p.returncode}): {cmd}")


def jload(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def not_run(what):
    display(Markdown(f"> **{what}: no results yet.** Run the experiment cell above first."))


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    plt.show()


def done_seeds(ds, tag=TAG):
    root = OUT / "models" / DD[ds] / tag
    runs = []
    for d in sorted(root.glob("seed*")):
        info = jload(d / "done.json")
        if info:
            runs.append((info["best_val_macro_f1"], int(d.name[4:])))
    return runs


def ref_seed(ds):
    runs = sorted(done_seeds(ds))
    return runs[(len(runs) - 1) // 2][1] if runs else None   # median validation Macro-F1


def peerbert_pred(ds, seed, split, binary=True):
    p = RES / DD[ds] / "peerbert" / TAG / f"seed{seed}" / f"{split}_predictions.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p, dtype={"row_id": str, "paper_key": str})
    return binary_view(df) if (binary and ds == "semdetect") else df


def ms(d, key):
    v = d.get(key) if d else None
    return f"{v['mean']:.4f} ± {v['std']:.4f}" if v else "–"
"""))

# ============================================================================ 1 data
A(md(r"""
---
## Experiment 1: Data, paper-level splits and leakage audit
Loads both datasets from the Hub. It checks every file (labels, review pairs, schema), keeps
each repeated human review once, and splits **by paper**. It then runs the leakage audit,
which blocks training if it fails. If the cell stops with `SchemaError` or `audit FAILED`, read
the message: it lists exactly what was found in the data.
"""))
A(code(r"""
if RUN["prepare"]:
    run("python scripts/00_inspect_datasets.py --include-semdetect")
    for ds in DATASETS:
        run(f"python scripts/02_prepare_splits.py --dataset {ds}")
    run("python scripts/02_prepare_splits.py --verify-only")
"""))
A(md("### 1.1 Dataset and split statistics"))
A(code(r"""
rows = []
for ds in DATASETS:
    s = jload(OUT / "manifests" / DD[ds] / "summary.json")
    if not s:
        not_run(f"{NAME[ds]} manifests"); continue
    for part, d in s["full_partitions"].items():
        rows.append({"dataset": NAME[ds], "stage": "full (audited)", "partition": part, "rows": d["rows"],
                     "papers": d["papers"], **{f"label {k}": v for k, v in d["label_counts"].items()}})
    for part, d in s["manifests"].items():
        rows.append({"dataset": NAME[ds], "stage": "used (whole-paper caps)", "partition": part, "rows": d["rows"],
                     "papers": d["papers"], **{f"label {k}": v for k, v in d["label_counts"].items()}})
    ex = pd.Series(s["exclusions"], name="rows").rename_axis("partition:reason").reset_index()
    display(Markdown(f"**{NAME[ds]}: excluded rows (every dropped row is logged)**")); display(ex)
if rows:
    display(Markdown("**Rows, papers and labels per partition** (labels: 0 human, 1 AI, 2 rewrite)"))
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
"""))
A(code(r"""
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
A(md("### 1.2 Leakage audit"))
A(code(r"""
for ds in DATASETS:
    a = jload(OUT / "audit" / DD[ds] / "leakage_audit.json")
    if not a:
        not_run(f"{NAME[ds]} audit"); continue
    display(Markdown(f"**{NAME[ds]}: overall audit status: `{a['status']}`**"))
    display(pd.DataFrame([{"check": c["check"], "status": c["status"]} for c in a["checks"]]))
    chk = {c["check"]: c for c in a["checks"]}
    lr = []
    for part, res in chk["length_only_auroc"]["results"].items():
        for task, v in res.items():
            if task == "per_generator":
                for g, vv in v.items():
                    if vv: lr.append({"partition": part, "task": f"ai vs human [{g}]", **vv})
            elif v:
                lr.append({"partition": part, "task": task, **v})
    display(Markdown("Length-only AUROC (how well review length alone separates the classes; "
                     "0.5 = no shortcut). This is a WARN-level check, not a blocker."))
    display(pd.DataFrame(lr))
"""))
A(code(r"""
for ds in DATASETS:
    a = jload(OUT / "audit" / DD[ds] / "leakage_audit.json")
    if not a:
        continue
    lo = {c["check"]: c for c in a["checks"]}["length_only_auroc"]
    worst, overlap = lo["worst"], {c["check"]: c for c in a["checks"]}["paper_overlap"]["overlaps"]["composite"]
    failed = [c["check"] for c in a["checks"] if c["status"] == "FAIL"]
    txt = [f"**Analysis ({NAME[ds]}).** The audit verdict is **{a['status']}**"
           + (f" (failed checks: {', '.join(failed)}; training stays blocked until the data issue is fixed)."
              if failed else ".")]
    txt.append("No paper is shared between train, validation and test." if not any(overlap.values()) else
               "Shared papers between splits: " + ", ".join(f"{k} {v}" for k, v in overlap.items() if v) + ".")
    txt.append(f"The strongest length-only separation is max(AUROC, 1-AUROC) = {worst:.3f}.")
    txt.append("Review length alone is a strong signal here. This motivates the per-length-tercile evaluation "
               "in Experiment 3, which checks that PeerBERT does not rely on it." if worst >= 0.75 else
               "Review length alone separates the classes only weakly.")
    display(Markdown(" ".join(txt)))
"""))

# ============================================================================ 2 train
A(md(r"""
---
## Experiment 2: PeerBERT training
ModernBERT-base is fine-tuned separately on each dataset.
- **Loss:** class-weighted cross-entropy + R-Drop consistency.
- **Selection:** early stopping on validation Macro-F1; the best checkpoint is kept.
- **Seeds:** 3 per dataset, reported as mean ± std.
- **Restart-safe:** each finished seed is skipped on re-run, and an interrupted seed resumes from
  its last epoch.
"""))
A(code(r"""
if RUN["smoke"]:
    run("python scripts/03_train_intellabs.py --smoke")      # about 5 minutes: checks the GPU path end to end
if RUN["train"]:
    if "intellabs" in DATASETS:
        run(f"python scripts/03_train_intellabs.py {SEED_ARG}")
    if "semdetect" in DATASETS:
        run(f"python scripts/04_train_semdetect.py {SEED_ARG}")
"""))
A(code(r"""
fig, axes = plt.subplots(2, 2, figsize=(12, 7), squeeze=False)
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
        axes[r][col].legend()
        axes[r][col].xaxis.set_major_locator(plt.MaxNLocator(integer=True))
if any_run:
    save(fig, "train_curves")
else:
    plt.close(fig); not_run("Training")
"""))
A(code(r"""
rows = []
for ds in DATASETS:
    for d in sorted((OUT / "models" / DD[ds] / TAG).glob("seed*")):
        i = jload(d / "run_info.json")
        if i:
            rows.append({"dataset": NAME[ds], "seed": i["seed"], "best val Macro-F1": i["best_val_macro_f1"],
                         "epochs": i["epochs_run"], "minutes": i["minutes"], "AMP": i["amp"],
                         "dropout": i["model_info"]["dropout_after"].get("attention_dropout"),
                         "R-Drop passes differ": i["rdrop_passes_differ"],
                         "train truncated %": 100 * i["truncation"]["train"]["truncated_fraction"],
                         "median tokens": i["truncation"]["train"]["tokens_median"]})
if rows:
    display(pd.DataFrame(rows))
else:
    not_run("Training")
"""))

# ============================================================================ 3 eval
A(md(r"""
---
## Experiment 3: PeerBERT evaluation
Evaluated on the official test split (whole-paper capped) and on the shared baseline subset.
- **Thresholds:** 0.5, plus a threshold calibrated on validation.
- **Sem-Detect:** reported natively as 3-class and as a human-vs-AI binary view.
- **Confidence intervals:** 95%, resampling whole papers.
"""))
A(code(r"""
if RUN["evaluate"]:
    for ds in DATASETS:
        run(f"python scripts/05_evaluate.py --dataset {ds} --n-boot {N_BOOT} {SEED_ARG}")
"""))
A(md("### 3.1 Main results (mean ± std over seeds)"))
A(code(r"""
rows = []
for ds in DATASETS:
    s = jload(RES / DD[ds] / "peerbert" / TAG / "metrics_summary.json")
    if not s:
        not_run(f"{NAME[ds]} evaluation"); continue
    for split, block in s["per_split"].items():
        for kind in ("binary@0.5", "binary@val_threshold"):
            b = block[kind]
            rows.append({"dataset": NAME[ds], "split": split, "view": kind, "seeds": b["n_seeds"],
                         "Accuracy": ms(b, "accuracy"), "Macro-F1": ms(b, "macro_f1"), "F1 (AI)": ms(b, "f1"),
                         "ROC-AUC": ms(b, "roc_auc"), "PR-AUC": ms(b, "pr_auc")})
        if "native_3class" in block:
            b = block["native_3class"]
            rows.append({"dataset": NAME[ds], "split": split, "view": "3-class (native)", "seeds": b["n_seeds"],
                         "Accuracy": ms(b, "accuracy"), "Macro-F1": ms(b, "macro_f1"),
                         "ROC-AUC": ms(b, "macro_ovr_roc_auc")})
if rows:
    display(pd.DataFrame(rows).fillna("–"))
"""))
A(code(r"""
rows = []
for ds in DATASETS:
    seed = ref_seed(ds)
    m = jload(RES / DD[ds] / "peerbert" / TAG / f"seed{seed}" / "metrics.json") if seed is not None else None
    if not m or "test" not in m["splits"]:
        continue
    t = m["splits"]["test"]
    ci = t.get("cluster_bootstrap_ci", {})
    b = t["binary@0.5"]
    rows.append({"dataset": NAME[ds], "seed": seed, "n": b["n"],
                 "ROC-AUC [95% CI]": f"{ci['roc_auc']['estimate']:.4f} [{ci['roc_auc']['ci_low']:.4f}, {ci['roc_auc']['ci_high']:.4f}]" if ci else "–",
                 "Macro-F1 [95% CI]": f"{ci['macro_f1@0.5']['estimate']:.4f} [{ci['macro_f1@0.5']['ci_low']:.4f}, {ci['macro_f1@0.5']['ci_high']:.4f}]" if ci else "–",
                 **{f"TPR@{k} FPR": (f"{v['tpr']:.4f}" if v["supported"] else f"n/a (n_neg={v['n_neg']})")
                    for k, v in b["tpr_at_fpr"].items()},
                 "FPR on human": b["ai_fpr"]})
if rows:
    display(Markdown("**Test set: paper-level cluster-bootstrap CIs and detection rates at low false-positive rates** "
                     "(reference seed = median validation Macro-F1)"))
    display(pd.DataFrame(rows))
"""))
A(md("### 3.2 Confusion matrices"))
A(code(r"""
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
k = 0
for ds in DATASETS:
    seed = ref_seed(ds)
    m = jload(RES / DD[ds] / "peerbert" / TAG / f"seed{seed}" / "metrics.json") if seed is not None else None
    if not m:
        continue
    views = [("binary@0.5", "binary")] + ([("native_3class", "3-class")] if ds == "semdetect" else [])
    for key, lab in views:
        if k >= 3:
            break
        cm = m["splits"]["test"][key]["confusion_matrix"]
        M = np.array(cm["matrix"]); Mn = M / M.sum(1, keepdims=True).clip(min=1)
        ax = axes[k]; k += 1
        ax.imshow(Mn, cmap="Blues", vmin=0, vmax=1)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, f"{M[i, j]}\n({Mn[i, j]:.1%})", ha="center", va="center",
                        color="white" if Mn[i, j] > 0.5 else "black", fontsize=9)
        ax.set_xticks(range(len(cm["labels"])), cm["labels"]); ax.set_yticks(range(len(cm["labels"])), cm["labels"])
        ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(f"{NAME[ds]} ({lab}), seed {seed}")
for ax in axes[k:]:
    ax.set_axis_off()
if k:
    save(fig, "peerbert_confusion")
else:
    plt.close(fig); not_run("Evaluation")
"""))
A(md("### 3.3 ROC, precision-recall and calibration"))
A(code(r"""
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.calibration import calibration_curve
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
shown = False
for ds in DATASETS:
    seed = ref_seed(ds)
    p = peerbert_pred(ds, seed, "test") if seed is not None else None
    if p is None or p["label"].nunique() < 2:
        continue
    shown = True
    fpr, tpr, _ = roc_curve(p["label"], p["score"]); axes[0].plot(fpr, tpr, label=NAME[ds])
    pr, rc, _ = precision_recall_curve(p["label"], p["score"]); axes[1].plot(rc, pr, label=NAME[ds])
    fp, mp = calibration_curve(p["label"], p["score"], n_bins=10, strategy="quantile")
    axes[2].plot(mp, fp, marker="o", label=NAME[ds])
axes[0].plot([0, 1], [0, 1], ":", c="grey"); axes[0].set_xscale("log"); axes[0].set_xlim(1e-3, 1)
axes[0].set_xlabel("false positive rate (log)"); axes[0].set_ylabel("true positive rate"); axes[0].set_title("ROC (test)")
axes[1].set_xlabel("recall"); axes[1].set_ylabel("precision"); axes[1].set_title("Precision-recall (test)")
axes[2].plot([0, 1], [0, 1], ":", c="grey"); axes[2].set_xlabel("predicted P(AI)"); axes[2].set_ylabel("observed AI rate")
axes[2].set_title("Calibration (test)")
for ax in axes:
    ax.legend()
if shown:
    save(fig, "peerbert_roc_pr_calibration")
else:
    plt.close(fig); not_run("Evaluation")
"""))
A(md("### 3.4 Where does PeerBERT work? Generators, venues and review length"))
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
        display(Markdown(f"**{NAME[ds]}: {title}**")); display(d[["n", "n_pos", "n_neg", "macro_f1", "roc_auc", "ai_fpr"]])
    if shown:
        save(fig, fname)
    else:
        plt.close(fig)

breakdown_plot("by_generator@0.5", "per AI generator", "peerbert_by_generator", ["intellabs"])
breakdown_plot("by_conf_year@0.5", "per venue-year", "peerbert_by_venue", DATASETS)
breakdown_plot("by_length_tercile@0.5", "per review-length tercile", "peerbert_by_length", DATASETS)
"""))
A(code(r"""
for ds in DATASETS:
    seed = ref_seed(ds)
    m = jload(RES / DD[ds] / "peerbert" / TAG / f"seed{seed}" / "metrics.json") if seed is not None else None
    if not m:
        continue
    t = m["splits"]["test"]
    lines = [f"**Analysis ({NAME[ds]}, seed {seed}).** On the test set, PeerBERT reaches Macro-F1 "
             f"{t['binary@0.5']['macro_f1']:.4f} and ROC-AUC {t['binary@0.5']['roc_auc']:.4f} (human vs AI)."]
    lt = pd.DataFrame(t.get("by_length_tercile@0.5", {})).T
    if len(lt):
        spread = lt["macro_f1"].astype(float).max() - lt["macro_f1"].astype(float).min()
        lines.append(f"Across review-length terciles, Macro-F1 varies by {spread:.4f}"
                     + (": performance is stable across lengths." if spread < 0.03
                        else ": performance depends on review length, which is worth discussing."))
    if "by_generator@0.5" in t:
        g = pd.DataFrame(t["by_generator@0.5"]).T["macro_f1"].astype(float)
        lines.append(f"Hardest generator: {g.idxmin()} (Macro-F1 {g.min():.4f}); easiest: {g.idxmax()} ({g.max():.4f}).")
    if "native_3class" in t:
        pc = t["native_3class"]["per_class"]
        lines.append("3-class F1: " + ", ".join(f"{k} {v['f1']:.4f}" for k, v in pc.items()) + ".")
    display(Markdown(" ".join(lines)))
"""))
A(md("### 3.5 Error analysis (most confident mistakes)"))
A(code(r"""
for ds in DATASETS:
    seed = ref_seed(ds)
    d = RES / DD[ds] / "peerbert" / TAG / f"seed{seed}"
    for name in ("test_binary_false_positives", "test_binary_false_negatives"):
        p = d / f"{name}.csv"
        if p.exists():
            t = pd.read_csv(p)
            display(Markdown(f"**{NAME[ds]}: {name.replace('_', ' ')}** ({len(t)} shown)"))
            display(t.head(5))
"""))

# ============================================================================ 4 baselines
A(md(r"""
---
## Experiment 4: Baselines
Eight detectors run on the **same shared test subset** as PeerBERT. Their thresholds are
calibrated on validation data only.
- **Zero-shot detectors:** LogRank, DetectLLM-LRR, Fast-DetectGPT, Binoculars, DNA-GPT.
- **Feature-based trained classifiers:** GLTR, MFD (fit on the training split only).
- **Pretrained detector:** RADAR.
- **Sem-Detect:** baselines are scored on human/ai reviews only.

The groups are run one after another, and GPU memory is freed between them.
"""))
A(code(r"""
if RUN["baselines"]:
    groups = ["logrank gltr detectllm_lrr mfd fast_detectgpt", "radar binoculars", "dnagpt"]
    for ds in DATASETS:
        for g in groups:
            run(f"python scripts/06_run_baselines.py --dataset {ds} --only {g}")
"""))
A(code(r"""
def comparison(ds):
    p = RES / DD[ds] / "comparison_table.csv"
    if not p.exists():
        return None
    t = pd.read_csv(p)
    t = t[t["split"] == "test_baseline"]
    # one PeerBERT row: the reference seed
    seed = ref_seed(ds)
    keep = ~(t["method"].str.startswith("PeerBERT")) | (t["method"] == f"PeerBERT[{TAG}] seed{seed}")
    return t[keep]

for ds in DATASETS:
    t = comparison(ds)
    if t is None:
        not_run(f"{NAME[ds]} comparison"); continue
    cols = ["type", "method", "threshold_kind", "n", "accuracy", "macro_f1", "f1", "roc_auc", "pr_auc",
            "tpr@1%fpr", "tpr@5%fpr", "tpr@10%fpr", "ai_fpr"]
    v = t[t["threshold_kind"] == "val_calibrated"][[c for c in cols if c in t.columns]]
    display(Markdown(f"**{NAME[ds]}: all methods on the shared test subset (validation-calibrated thresholds)**"))
    display(v.sort_values(["type", "macro_f1"], ascending=[True, False]).reset_index(drop=True))
"""))
A(code(r"""
fig, axes = plt.subplots(1, len(DATASETS), figsize=(7 * len(DATASETS), 4.5), squeeze=False)
shown = False
colors = {"trained classifier": "#6B46C1", "zero-shot detector": "#2C7A7B", "pretrained detector": "#C05621"}
for ax, ds in zip(axes[0], DATASETS):
    t = comparison(ds)
    if t is None:
        ax.set_axis_off(); continue
    v = t[t["threshold_kind"] == "val_calibrated"].sort_values("macro_f1")
    ax.barh(v["method"].str.replace(f"PeerBERT[{TAG}]", "PeerBERT", regex=False), v["macro_f1"],
            color=[colors.get(x, "grey") for x in v["type"]])
    for y, (f1, auc) in enumerate(zip(v["macro_f1"], v["roc_auc"])):
        ax.text(f1 + 0.01, y, f"F1 {f1:.3f} | AUC {auc:.3f}", va="center", fontsize=8)
    ax.set_xlim(0, 1.25); ax.set_xlabel("Macro-F1 (validation-calibrated threshold)")
    ax.set_title(f"{NAME[ds]}: shared test subset"); shown = True
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors.values()]
fig.legend(handles, colors.keys(), loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.04))
if shown:
    save(fig, "baselines_macro_f1")
else:
    plt.close(fig)
"""))
A(code(r"""
fig, axes = plt.subplots(1, len(DATASETS), figsize=(6.5 * len(DATASETS), 5), squeeze=False)
shown = False
for ax, ds in zip(axes[0], DATASETS):
    seed = ref_seed(ds)
    p = peerbert_pred(ds, seed, "test_baseline") if seed is not None else None
    curves = [("PeerBERT", p)] if p is not None else []
    for d in sorted((RES / DD[ds] / "baselines").glob("*/predictions_test_baseline.csv")):
        curves.append((jload(d.parent / "metrics.json")["display"], pd.read_csv(d, dtype={"row_id": str})))
    for name, c in curves:
        if c["label"].nunique() == 2:
            fpr, tpr, _ = roc_curve(c["label"], c["score"])
            ax.plot(fpr, tpr, lw=2.4 if name == "PeerBERT" else 1.2, label=name); shown = True
    ax.plot([0, 1], [0, 1], ":", c="grey"); ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate")
    ax.set_title(f"{NAME[ds]}: ROC on the shared test subset"); ax.legend(fontsize=8, loc="lower right")
if shown:
    save(fig, "baselines_roc")
else:
    plt.close(fig)
"""))
A(code(r"""
rows = []
for ds in DATASETS:
    for d in sorted((RES / DD[ds] / "baselines").glob("*/metrics.json")):
        m = jload(d)
        r = {"dataset": NAME[ds], "baseline": m["display"], "type": m["type"], "val threshold": m["val_threshold"]}
        if "orientation" in m["info"]:
            r["note"] = f"RADAR AI output chosen on validation: index {m['info']['orientation']['used_ai_index']}"
        if "rewrite_fraction_flagged_ai@val_threshold" in m:
            rw = m["rewrite_fraction_flagged_ai@val_threshold"]
            r["rewrite flagged as AI"] = f"{rw['fraction']:.1%} of {rw['n_rewrite']}"
        rows.append(r)
if rows:
    display(Markdown("**Baseline details** (Sem-Detect rewrite rows are reported only as the share flagged as AI)"))
    display(pd.DataFrame(rows).fillna(""))
"""))

# ============================================================================ 5 ablation
A(md(r"""
---
## Experiment 5: Ablation
One seed per variant; variants are selected on **validation** Macro-F1 only. The variants are:
- unweighted CE, class-weighted CE, label smoothing, focal loss
- R-Drop α ∈ {0, 0.5, 1}, and dropout off
- the cost-sensitive margin loss with a cosine head (an idea taken from CoCoDet, ACL 2026)

Confirm the selected configuration with all 3 seeds before claiming an improvement.
"""))
A(code(r"""
if RUN["ablation"]:
    for ds in DATASETS:
        run(f"python scripts/07_run_ablation.py --dataset {ds}")
"""))
A(code(r"""
fig, axes = plt.subplots(1, len(DATASETS), figsize=(6.5 * len(DATASETS), 4), squeeze=False)
shown = False
for ax, ds in zip(axes[0], DATASETS):
    a = jload(RES / DD[ds] / "ablation.json")
    if not a:
        ax.set_axis_off(); not_run(f"{NAME[ds]} ablation"); continue
    d = pd.DataFrame(a["rows"]).sort_values("val_macro_f1")
    ax.barh(d["variant"], d["val_macro_f1"], color=["#D69E2E" if v == a["selected_variant"] else "#A0AEC0"
                                                    for v in d["variant"]])
    lo = d["val_macro_f1"].min()
    ax.set_xlim(max(0, lo - 0.05), 1.0); ax.set_xlabel("validation Macro-F1")
    ax.set_title(f"{NAME[ds]}: ablation (selected: {a['selected_variant']})"); shown = True
    display(Markdown(f"**{NAME[ds]}**")); display(d[::-1].reset_index(drop=True))
if shown:
    save(fig, "ablation")
else:
    plt.close(fig)
"""))

# ============================================================================ 6 cross
A(md(r"""
---
## Experiment 6: Cross-dataset transfer
Each PeerBERT model is applied to the **other** dataset without retraining (human vs AI only;
Sem-Detect rewrite excluded). Thresholds come from the source validation set. The datasets differ
in generators, label definitions and venues, so a drop is expected; its size is the finding.
"""))
A(code(r"""
if RUN["cross"]:
    for src in DATASETS:
        run(f"python scripts/08_cross_dataset.py --source {src} {SEED_ARG}")
"""))
A(code(r"""
rows = []
for src in DATASETS:
    tgt = "semdetect" if src == "intellabs" else "intellabs"
    s_in = jload(RES / DD[src] / "peerbert" / TAG / "metrics_summary.json")
    c = jload(RES / DD[tgt] / f"cross_from_{src}" / "metrics.json")
    if s_in:
        b = s_in["per_split"]["test"]["binary@val_threshold"]
        rows.append({"train on": NAME[src], "test on": NAME[src] + " (in-domain)",
                     "Macro-F1": b["macro_f1"]["mean"], "ROC-AUC": b["roc_auc"]["mean"]})
    if c:
        b = c["mean_std@source_val_threshold"]
        rows.append({"train on": NAME[src], "test on": NAME[tgt] + " (transfer)",
                     "Macro-F1": b["macro_f1"]["mean"], "ROC-AUC": b["roc_auc"]["mean"]})
if rows:
    d = pd.DataFrame(rows); display(d)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    x = np.arange(len(d)); ax.bar(x - 0.18, d["Macro-F1"], 0.36, label="Macro-F1"); ax.bar(x + 0.18, d["ROC-AUC"], 0.36, label="ROC-AUC")
    ax.set_xticks(x, [f"{a}\n-> {b}" for a, b in zip(d["train on"], d["test on"])], fontsize=8); ax.set_ylim(0, 1.05)
    ax.set_title("In-domain vs cross-dataset (mean over seeds)"); ax.legend()
    save(fig, "cross_dataset")
    c = jload(RES / "sem_detect" / "cross_from_intellabs" / "metrics.json") or jload(RES / "intellabs" / "cross_from_semdetect" / "metrics.json")
    if c:
        display(Markdown("**Known mismatches between the datasets:**\n" + "\n".join(f"- {n}" for n in c["mismatch_notes"])))
else:
    not_run("Cross-dataset transfer")
"""))

# ============================================================================ 7 xai
A(md(r"""
---
## Experiment 7: Explainability
For reviews from the shared test subset (balanced over true class × correct/incorrect), the
model's decision is explained with:
- **Integrated Gradients** on the token embeddings (signed, with a completeness check);
- **attention**: last layer, mean over layers, and rollout.

**Faithfulness** (ERASER): remove the top-k% tokens (comprehensiveness, higher is better) or keep
only them (sufficiency, lower is better). A random ranking is the reference; an explanation is
only useful if it beats random.
"""))
A(code(r"""
if RUN["xai"]:
    for ds in DATASETS:
        run(f"python scripts/10_explain.py --dataset {ds} --n-per-group {XAI_PER_GROUP}")
"""))
A(code(r"""
fig, axes = plt.subplots(len(DATASETS), 2, figsize=(12, 3.8 * len(DATASETS)), squeeze=False)
shown = False
for r, ds in enumerate(DATASETS):
    xs = sorted((RES / DD[ds] / "xai" / TAG).glob("seed*/summary.json"))
    if not xs:
        axes[r][0].set_axis_off(); axes[r][1].set_axis_off(); not_run(f"{NAME[ds]} XAI"); continue
    x = jload(xs[0]); shown = True
    rows = []
    for meth, byk in x["faithfulness"].items():
        for k, v in byk.items():
            rows.append({"method": meth, "k": float(k), "comprehensiveness": v["comprehensiveness"]["mean"],
                         "sufficiency": v["sufficiency"]["mean"]})
    f = pd.DataFrame(rows)
    for c, metric in enumerate(("comprehensiveness", "sufficiency")):
        f.pivot(index="method", columns="k", values=metric).plot.bar(ax=axes[r][c], rot=0)
        axes[r][c].set_title(f"{NAME[ds]}: {metric} ({'higher' if c == 0 else 'lower'} = more faithful)")
        axes[r][c].axhline(0, c="grey", lw=0.8)
    display(Markdown(f"**{NAME[ds]}** (seed {x['seed']}, {x['n_examples']} reviews; mean |IG delta| "
                     f"{x['ig']['mean_abs_delta']['mean']:.4f}; Spearman(|IG|, attention): " +
                     ", ".join(f"{k} {v['mean']:.3f}" for k, v in x["spearman_abs_ig_vs_attention"].items() if v) + ")"))
    display(f.sort_values(["k", "comprehensiveness"], ascending=[True, False]).reset_index(drop=True))
    kmin = f["k"].min()
    best = f[f["k"] == kmin].set_index("method")
    if "random" in best.index:
        beats = [m for m in best.index if m != "random" and best.loc[m, "comprehensiveness"] > best.loc["random", "comprehensiveness"]]
        display(Markdown(f"**Analysis.** At k = {kmin:.0%}, methods that beat the random ranking on "
                         f"comprehensiveness: {', '.join(beats) or 'none'}."))
if shown:
    save(fig, "xai_faithfulness")
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
"""))
A(code(r"""
for ds in DATASETS:
    xs = sorted((RES / DD[ds] / "xai" / TAG).glob("seed*/examples.html"))
    if xs:
        display(Markdown(f"**{NAME[ds]}: explained reviews** (red: pushes towards the predicted class, "
                         f"blue: against it). Full page: `{xs[0]}`"))
        html = xs[0].read_text()
        display(HTML(f"<div style='max-height:600px;overflow:auto;border:1px solid #ddd;padding:8px'>{html}</div>"))
"""))

# ============================================================================ 8 report
A(md(r"""
---
## 8. Final report and significance tests
This step collects every result into thesis-ready tables (CSV + LaTeX), plus paired tests of
PeerBERT against each baseline on the same reviews. The paired tests use a paper-level bootstrap
on ROC-AUC and McNemar, with Holm correction.
"""))
A(code(r"""
run(f"python scripts/09_build_report.py --n-boot {N_BOOT}")
for ds in DATASETS:
    p = RES / "report" / f"{ds}_paired_tests.csv"
    if p.exists():
        t = pd.read_csv(p)
        display(Markdown(f"**{NAME[ds]}: PeerBERT vs each baseline (positive difference = PeerBERT better)**"))
        display(t)
        sig = t.loc[t["p_bootstrap_holm"] < 0.05, "baseline"].tolist() if "p_bootstrap_holm" in t else []
        display(Markdown(f"Significant after Holm correction (α = 0.05): {', '.join(sig) or 'none'}."))
display(Markdown((RES / "report" / "report.md").read_text()))
print("\nFigures (PNG + PDF):", FIG)
for f in sorted(FIG.glob("*.png")):
    print("  ", f.name)
print("Tables (CSV + LaTeX):", RES / "report")
"""))

for i, c in enumerate(cells):
    c["id"] = f"cell-{i:03d}"
nb = {"cells": cells,
      "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4", "toc_visible": True},
                   "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"wrote {OUT} ({len(cells)} cells)")
