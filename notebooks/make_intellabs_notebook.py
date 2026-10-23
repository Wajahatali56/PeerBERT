"""
Builds the single-dataset notebook (IntelLabs only), Kaggle first, also runs on Colab:

    python notebooks/make_intellabs_notebook.py [output.ipynb]

IntelLabs/AI-Peer-Review-Detection-Benchmark, label 0 = Human, 1 = AI:
  calibration -> train + validation (split by paper)   test -> official test   extended -> robustness test
PeerBERT (ModernBERT-base, fine-tuned) is compared with eight baselines on the same reviews.
No explainability and no second dataset in this notebook (see make_full_notebook.py for the two-dataset one).
The notebook carries the tested repository code, so rebuild it after changing anything under src/, scripts/, configs/.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "notebooks" / "PeerBERT_intellabs.ipynb"
SCRIPTS = ("02_prepare_splits.py", "03_train_intellabs.py", "05_evaluate.py", "06_run_baselines.py",
           "09_build_report.py", "11_experiment_card.py")
SHOWN_CONFIGS = ("configs/train.yaml", "configs/baselines.yaml")          # editable; config.yaml stays collapsed


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)}


def code(text, hide=False):
    meta = {"_kg_hide-input": True, "_kg_hide-output": True, "jupyter": {"source_hidden": True},
            "cellView": "form"} if hide else {}
    return {"cell_type": "code", "execution_count": None, "metadata": meta, "outputs": [],
            "source": text.strip("\n").splitlines(True)}


def project_files():
    files = []
    for d in ("configs", "src", "scripts"):
        for p in sorted((ROOT / d).rglob("*")):
            rel = p.relative_to(ROOT).as_posix()
            if not p.is_file() or p.suffix not in (".py", ".yaml") or "__pycache__" in rel or not p.read_text().strip():
                continue
            if rel.startswith("src/xai/") or (rel.startswith("scripts/") and p.name not in SCRIPTS):
                continue                                                  # no explainability, no second-dataset scripts
            text = p.read_text(encoding="utf-8")
            if rel == "configs/train.yaml":                                # the ablation recipe is not used in this notebook
                text = text[:text.index("# Ablations")].rstrip() + "\n"
            files.append((rel, text))
    return files


cells = []
A = cells.append

# ================================================================== title
A(md(r"""
# Detecting AI-written peer reviews: fine-tuned ModernBERT-base on IntelLabs

This notebook fine-tunes **ModernBERT-base** to tell **human** peer reviews from **AI-written** ones, and compares it with **eight other detectors**.
It uses **one dataset only**: [IntelLabs/AI-Peer-Review-Detection-Benchmark](https://huggingface.co/datasets/IntelLabs/AI-Peer-Review-Detection-Benchmark).

**Labels:** `0` = **Human**, `1` = **AI**.

| IntelLabs subset | What we use it for |
|---|---|
| **calibration** | Training. It is split **by paper** into *train* (the model learns from it) and *validation* (early stopping and the decision threshold). |
| **test** | The official test set. Used once, for the final numbers. |
| **extended** | A second, harder test (more reviews, generators gpt-4o and llama only). **Never used for training or tuning.** Papers that also appear in train or validation are removed. |

## The steps
1. **Setup**: Kaggle/Colab, code, settings.
2. **Data**: load the three subsets, check them, split by paper, run the leakage audit.
3. **Train** ModernBERT-base (3 seeds).
4. **Evaluate** on `test` and `extended`.
5. **Compare** with eight baselines on the same reviews.
6. **Report**: tables, figures, a short methods record for the paper.

## Why you can trust the numbers
- Labels come from the data itself; every dropped row is logged with a reason.
- Train and validation never share a paper; test and extended never share a paper with them.
- The model, the decision threshold and the baselines' thresholds use **validation data only**. Test and extended data are never used to choose anything.
- Three training runs with different seeds (mean ± std) and 95% confidence intervals that resample **whole papers**.
- Every baseline is scored on exactly the same reviews as PeerBERT. Where a baseline differs from its original paper, it is listed.

## How to run on Kaggle
Import the notebook, set **Accelerator: GPU T4 x2** and **Internet: On**, then **Run All**. With two GPUs, two training seeds and two groups of baselines run side by side.
If the session stops, run it again: finished work is skipped and training resumes from its last epoch. To continue in a **new** session, add the previous
version's *output* as an input (*Add Input -> Notebooks*); the setup cell restores it.
"""))

# ================================================================== settings
A(md("## 1. Setup"))
A(md("### 1.1 Settings (the only cell you normally edit)"))
A(code(r"""
# Which steps to run (all True = the whole study)
RUN = {"prepare": True, "smoke": True, "train": True, "evaluate": True, "baselines": True, "report": True}

SEEDS = None          # None = three training runs (seeds 42, 43, 44). Use [42] for a quick first pass.
N_BOOT = 1000         # bootstrap resamples for the confidence intervals
USE_EXTENDED = True   # use the 'extended' subset as a robustness test

# Whole-paper size limits (None = no limit). They only keep the run inside Kaggle's time limit and are written into the report.
CAPS = {
    "train_max_rows": 24000,                  # reviews used to train (from calibration, after the train/validation split)
    "model_eval_max_rows": 30000,             # reviews of the test subset scored by PeerBERT
    "baseline_eval_max_rows": 600,            # reviews of the test subset on which ALL methods are compared
    "extended_max_rows": 30000,               # reviews of the extended subset scored by PeerBERT
    "extended_baseline_eval_max_rows": 600,   # reviews of the extended subset on which ALL methods are compared
}
TIME_BUDGET_H = 11.0  # a step that would not fit in the remaining session time is skipped (run the notebook again later)
# Rough planning estimates in hours (assumptions, not measurements; the smoke test below measures the real speed):
EST_H = {"prepare": 0.5, "smoke": 0.2, "train": 3.5, "evaluate": 0.8, "baselines": 2.0, "report": 0.2}
"""))
A(md(r"""
### 1.2 Environment
Folders, caches and package check. Models and data come from the Hugging Face Hub. Optional: add a secret named `HF_TOKEN`
(Kaggle: *Add-ons -> Secrets*) for faster downloads.
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
os.environ["PRAD_WORK_DIR"] = WORK                          # everything the study produces goes here
os.environ["HF_HOME"] = os.path.join(TEMP, "hf_cache")      # downloaded models/data (not counted as output)
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"            # keeps the output readable
for d in (WORK, CODE, os.environ["HF_HOME"]):
    os.makedirs(d, exist_ok=True)
print("platform :", "Kaggle" if ON_KAGGLE else "Colab/other")
print("outputs  :", WORK)
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
    # install or upgrade only when missing or too old, so the platform's tested packages stay untouched
    try:
        ok = minimum is None or Version(metadata.version(pkg)) >= Version(minimum)
    except metadata.PackageNotFoundError:
        ok = False
    if not ok:
        print("installing", f"{pkg}>={minimum}" if minimum else pkg)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", f"{pkg}>={minimum}" if minimum else pkg], check=True)

for pkg, minimum in [("transformers", "4.48.0"), ("huggingface_hub", "0.22"), ("accelerate", "0.26"), ("pyarrow", None),
                     ("scikit-learn", "1.3"), ("scipy", "1.10"), ("matplotlib", "3.7"), ("pyyaml", None), ("packaging", None)]:
    ensure(pkg, minimum)
import torch, transformers
print("torch", torch.__version__, "| transformers", transformers.__version__)
"""))
A(md(r"""
### 1.3 Project code
The next cells write the project code into the runtime (collapsed; expand one to read it). The two **config** cells below are shown on purpose:
`configs/train.yaml` holds the training recipe and `configs/baselines.yaml` the baseline settings. Edit them here, then run that cell again, before the step that uses them.
"""))
A(code(r"""
for d in ["configs", "scripts", "src/data", "src/models", "src/evaluation", "src/baselines/statistical",
          "src/baselines/zero_shot", "src/baselines/neural_detectors"]:
    os.makedirs(os.path.join(CODE, d), exist_ok=True)
%cd {CODE}
"""))
for rel, text in project_files():
    A(code(f"%%writefile {rel}\n{text}", hide=rel not in SHOWN_CONFIGS))
A(code(r"""
for d, _, _ in os.walk(os.path.join(CODE, "src")):
    open(os.path.join(d, "__init__.py"), "a").close()
if CODE not in sys.path:
    sys.path.insert(0, CODE)
import yaml
cp = Path(CODE) / "configs" / "config.yaml"                  # apply the settings above
c = yaml.safe_load(cp.read_text())
c["data"]["intellabs"].update(CAPS)
c["data"]["intellabs"]["use_extended"] = bool(USE_EXTENDED)
cp.write_text(yaml.safe_dump(c, sort_keys=False))
print("code written to", CODE)
print("IntelLabs settings:", {k: c["data"]["intellabs"][k] for k in [*CAPS, "use_extended"]})
"""))
A(code(r"""
from src.notebook_utils import Budget, disk_report, gpu_count, restore_outputs, run_cmd, run_jobs

BUDGET = Budget(TIME_BUDGET_H)
N_GPUS = gpu_count()
PY = f'"{sys.executable}"'
LOGS = Path(WORK) / "logs"
need_hub = any(RUN[s] for s in ("prepare", "smoke", "train", "baselines"))
if ON_KAGGLE:
    for src, n in restore_outputs(WORK):
        print(f"restored {n} files from an earlier session: {src}")
if need_hub:
    try:
        urllib.request.urlopen("https://huggingface.co", timeout=20)
        print("internet: OK")
    except Exception as e:
        raise RuntimeError("huggingface.co is not reachable. On Kaggle turn on Session options -> Internet.") from e
gpus = subprocess.run("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader", shell=True, capture_output=True, text=True).stdout.strip()
print("GPUs:", gpus or "none")
if (RUN["smoke"] or RUN["train"]) and not torch.cuda.is_available():
    raise RuntimeError("Training needs a GPU: select a GPU accelerator in the session options.")
print(f"{N_GPUS} GPU(s): " + ("two jobs run side by side" if N_GPUS > 1 else "jobs run one after another"))
disk_report(WORK)
"""))

# ================================================================== helpers
A(md("### 1.4 Helper functions for tables and plots"))
A(code(r"""
import glob, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.width", 200, "display.max_columns", 40, "display.precision", 4, "display.max_colwidth", 90)
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 300, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

from sklearn.metrics import roc_curve, precision_recall_curve
from src.evaluation.metrics import binary_metrics, calibrate_threshold
from src.evaluation.aggregate import by_generator
from src.evaluation.stats import auc_fn, cluster_bootstrap_ci, macro_f1_fn

OUT = Path(WORK) / "outputs"
RES = OUT / "results" / "intellabs"
FIG = OUT / "results" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
TAG = "final"
SEED_ARG = f"--seeds {' '.join(map(str, SEEDS))}" if SEEDS else ""
LABELS = {0: "Human", 1: "AI"}
TYPE_COLOR = {"trained classifier": "#6B46C1", "zero-shot detector": "#2C7A7B", "pretrained detector": "#C05621"}


def jload(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def md_(text):
    display(Markdown(text))


def not_run(what):
    md_(f"> **{what}: no results yet.** Run the step above first.")


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    plt.show()


def done_seeds():
    runs = []
    for d in sorted((OUT / "models" / "intellabs" / TAG).glob("seed*")):
        info = jload(d / "done.json") if d.is_dir() else None
        if info:
            runs.append((info["best_val_macro_f1"], int(d.name[4:])))
    return runs


def ref_seed():
    # the run with the median validation Macro-F1: chosen without looking at any test data
    runs = sorted(done_seeds())
    return runs[(len(runs) - 1) // 2][1] if runs else None


def peerbert_pred(seed, split):
    p = RES / "peerbert" / TAG / f"seed{seed}" / f"{split}_predictions.csv"
    if seed is None or not p.exists():
        return None
    return pd.read_csv(p, dtype={"row_id": str, "paper_key": str})


def ms(d, key):
    v = d.get(key) if d else None
    return f"{v['mean']:.4f} ± {v['std']:.4f}" if v else "–"


def fmt_ci(est, lo, hi):
    return "–" if est is None or pd.isna(est) else f"{est:.4f} [{lo:.4f}, {hi:.4f}]"


def method_predictions(split):
    # every method on the same reviews: name -> (type, predictions, validation-calibrated threshold)
    out, seed = {}, ref_seed()
    p = peerbert_pred(seed, split)
    if p is not None:
        vp = pd.read_csv(OUT / "models" / "intellabs" / TAG / f"seed{seed}" / "val_predictions.csv")
        out["PeerBERT"] = ("trained classifier", p, calibrate_threshold(vp["label"], vp["score"], "macro_f1"))
    for d in sorted((RES / "baselines").glob("*/metrics.json")):
        pp = d.parent / f"predictions_{split}.csv"
        if pp.exists():
            m = jload(d)
            out[m["display"]] = (m["type"], pd.read_csv(pp, dtype={"row_id": str, "paper_key": str}), m["val_threshold"])
    return out


def ci_table(split):
    rows = []
    for name, (typ, p, thr) in method_predictions(split).items():
        y, s, c = p["label"].values, p["score"].values, p["paper_key"].values
        if len(np.unique(y)) < 2:
            continue
        a = cluster_bootstrap_ci(y, s, c, auc_fn, N_BOOT, seed=0)
        f = cluster_bootstrap_ci(y, s, c, macro_f1_fn(thr), N_BOOT, seed=0)
        bm = binary_metrics(y, s, thr)
        t10 = bm["tpr_at_fpr"]["10%"]
        rows.append({"method": name, "type": typ, "n": len(p), "AI share": bm["n_pos"] / bm["n"],
                     "ROC-AUC": a["estimate"], "AUC lo": a["ci_low"], "AUC hi": a["ci_high"],
                     "Macro-F1": f["estimate"], "F1 lo": f["ci_low"], "F1 hi": f["ci_high"],
                     "PR-AUC": bm["pr_auc"], "TPR@10%FPR": t10["tpr"] if t10["supported"] else np.nan,
                     "human wrongly flagged": bm["ai_fpr"]})
    return pd.DataFrame(rows)


def progress():
    from src.baselines.registry import BASELINES
    n_seeds = len(SEEDS) if SEEDS else len(yaml.safe_load((Path(CODE) / "configs" / "train.yaml").read_text())["training"]["seeds"])
    au = jload(OUT / "audit" / "intellabs" / "audit_status.json")
    ev = jload(RES / "peerbert" / TAG / "metrics_summary.json")
    rows = {"2 data + audit": au["status"] if au else "-",
            "3 training": f"{len(done_seeds())}/{n_seeds} seeds",
            "4 evaluation": f"{len(ev['seeds'])} seeds" if ev else "-",
            "5 baselines": f"{len(list((RES / 'baselines').glob('*/metrics.json')))}/{len(BASELINES)}",
            "6 report + methods record": "done" if (OUT / "results" / "report" / "experiment_card.md").exists() else "-"}
    display(pd.Series(rows, name="status").to_frame())
"""))
A(md("### Progress so far (includes results restored from an earlier session)"))
A(code("progress()"))

# ================================================================== 2 data
A(md(r"""
---
## 2. Data
**What happens here.**
1. The three subsets are downloaded file by file. Every file is checked: the `label` column must agree with the review's id (human reviews carry a `human` id), and each review must appear once as human and once as AI.
2. The human reviews are copied into every generator's file; we keep each one **once**.
3. The review text is built from the review-text fields only. Scores, flags and other metadata are not used.
4. `calibration` is split **by paper** into train and validation. `test` stays as it is. `extended` loses every review whose paper is also in train or validation.
5. A **leakage audit** looks for shared papers, duplicate text and length shortcuts. If it fails, training is blocked.

If this cell stops with an error message, read it: it says exactly what was found in the data.
"""))
A(code(r"""
if RUN["prepare"] and BUDGET.allow("data preparation", EST_H["prepare"]):
    run_cmd(f"{PY} scripts/02_prepare_splits.py --dataset intellabs", CODE)
    run_cmd(f"{PY} scripts/02_prepare_splits.py --dataset intellabs --verify-only", CODE)
    disk_report(WORK)
"""))
A(md("### 2.1 What each part of the data is used for"))
A(code(r"""
ROLE = {"train": "fine-tune the model", "val": "early stopping + decision threshold", "test": "final test (official)",
        "test_baseline": "all methods compared here (inside test)", "extended": "robustness test (never trained on)",
        "extended_baseline": "all methods compared here (inside extended)"}
SOURCE = {"train": "calibration", "val": "calibration", "test": "test", "test_baseline": "test", "extended": "extended",
          "extended_baseline": "extended"}
summ = jload(OUT / "manifests" / "intellabs" / "summary.json")
if summ:
    rows = [{"part": n, "IntelLabs subset": SOURCE[n], "used for": ROLE[n], "reviews": d["rows"], "papers": d["papers"],
             "Human (0)": d["label_counts"].get("0", 0), "AI (1)": d["label_counts"].get("1", 0)}
            for n, d in summ["manifests"].items()]
    t = pd.DataFrame(rows)
    t["AI share"] = t["AI (1)"] / t["reviews"]
    display(t)
    md_("Reviews are counted **after** the size limits in the settings (whole papers only). `AI share` is the fraction of AI reviews: it matters for "
        "the metrics (see Step 4: accuracy depends on it, Macro-F1 and ROC-AUC much less).")
    ls = jload(OUT / "manifests" / "intellabs" / "loader_stats.json") or {}
    if "extended" not in summ["manifests"]:
        md_("> **The extended subset is not part of this run.** " + (f"Reason: `{ls['extended_skipped'][:600]}`" if ls.get("extended_skipped") else "It is switched off in the settings."))
else:
    not_run("Data preparation")
"""))
A(code(r"""
if summ:
    parts = [p for p in ("train", "val", "test", "extended") if p in summ["manifests"]]
    fig, axes = plt.subplots(1, 3, figsize=(17, 3.8))
    lab = pd.DataFrame({p: pd.read_parquet(OUT / "manifests" / "intellabs" / f"{p}.parquet", columns=["label"])["label"].map(LABELS).value_counts() for p in parts}).T.reindex(columns=["Human", "AI"])
    lab.plot.bar(ax=axes[0], rot=0, color=["#2C7A7B", "#C05621"]); axes[0].set_title("Human vs AI reviews per part"); axes[0].set_ylabel("reviews")
    gen = pd.DataFrame({p: pd.read_parquet(OUT / "manifests" / "intellabs" / f"{p}.parquet", columns=["label", "generator"]).query("label == 1")["generator"].value_counts() for p in parts}).fillna(0).T
    gen.plot.bar(stacked=True, ax=axes[1], rot=0); axes[1].set_title("Which LLM wrote the AI reviews"); axes[1].set_ylabel("AI reviews")
    tr = pd.read_parquet(OUT / "manifests" / "intellabs" / "train.parquet", columns=["text", "label"])
    tr["words"] = tr["text"].str.split().str.len()
    for k, g in tr.groupby("label"):
        axes[2].hist(g["words"].clip(upper=g["words"].quantile(0.99)), bins=50, alpha=0.5, density=True, label=LABELS[k])
    axes[2].set_title("Review length in train"); axes[2].set_xlabel("words"); axes[2].legend()
    save(fig, "data_overview")
"""))
A(md("### 2.2 Checks that the labels and the text are right"))
A(code(r"""
if summ:
    tr = pd.read_parquet(OUT / "manifests" / "intellabs" / "train.parquet", columns=["row_id", "label", "label_name"])
    cid = tr["row_id"].str.split(":", n=2).str[2].str.lower()       # row_id = subset:conference_year:custom_id
    chk = pd.DataFrame({"label": [0, 1], "meaning": ["Human", "AI"], "reviews": [int((tr.label == 0).sum()), int((tr.label == 1).sum())],
                        "id says 'human'": [f"{cid[tr.label == 0].str.startswith('human').mean():.1%}", f"{cid[tr.label == 1].str.startswith('human').mean():.1%}"]})
    md_("**Label check (train).** Label 0 reviews should all have a `human` id, and label 1 reviews none. The loader enforces this on every file and stops otherwise.")
    display(chk)
ls = jload(OUT / "manifests" / "intellabs" / "loader_stats.json")
if ls:
    rows = []
    for sub in ("calibration", "test", "extended"):
        s = ls.get(sub)
        if s:
            rows.append({"subset": sub, "files": s["files"], "reviews kept": s["rows"], "human : AI": s["human_to_ai_ratio"],
                         "human copies dropped": s["human_duplicate_copies_dropped"],
                         "copies whose text differed": s["human_copies_with_different_text"], "venue-years": len(s["conf_years"]),
                         "AI generators": ", ".join(s["generators"])})
    md_("**Human reviews appear in every generator's file; we keep each once.**")
    display(pd.DataFrame(rows))
    acc = {}
    for sub in ("calibration", "test"):
        for f in ls.get(sub, {}).get("file_stats", []):
            for field, byl in f.get("never_field_mean_chars_by_label", {}).items():
                for lab_, v in byl.items():
                    acc.setdefault(field, {}).setdefault(lab_, []).append(v)
    if acc:
        t = pd.DataFrame({fld: {("Human" if k == "0" else "AI"): np.mean(v) for k, v in d.items()} for fld, d in acc.items()}).T
        t["ratio"] = t.max(axis=1) / t.min(axis=1).clip(lower=1e-9)
        md_("**Fields that are NOT used as review text** (scores, flags, metadata). Mean length in characters per class: a large ratio would let the field alone reveal "
            "the label, which is one reason they are excluded (`code_of_conduct` is the example from the dataset inspection).")
        display(t.sort_values("ratio", ascending=False))
"""))
A(md("### 2.3 Leakage audit"))
A(code(r"""
a = jload(OUT / "audit" / "intellabs" / "leakage_audit.json")
if a:
    md_(f"**Audit result: `{a['status']}`** (PASS or WARN lets training start; FAIL blocks it)")
    display(pd.DataFrame([{"check": c["check"], "status": c["status"]} for c in a["checks"]]))
    chk = {c["check"]: c for c in a["checks"]}
    ov = chk["paper_overlap"]["overlaps"]
    display(pd.DataFrame({"papers shared (venue_year_paperid)": ov["composite"]}).rename_axis("between").T)
    lr = []
    for part, res in chk["length_only_auroc"]["results"].items():
        for task, v in res.items():
            if task == "per_generator":
                lr += [{"part": part, "comparison": f"AI vs Human [{g}]", **vv} for g, vv in v.items() if vv]
            elif v:
                lr.append({"part": part, "comparison": "AI vs Human", **v})
    md_("**Length shortcut check:** how well review length **alone** separates AI from Human (0.5 = not at all; this is a warning, not a blocker).")
    display(pd.DataFrame(lr))
    failed = [c for c, v in chk.items() if v["status"] == "FAIL"]
    worst = chk["length_only_auroc"]["worst"]
    t = [f"**Takeaway.** The audit is **{a['status']}**" + (f" (failed: {', '.join(failed)}; fix the data issue before training)." if failed else ".")]
    t.append("No paper is shared between train, validation and test." if not any(ov["composite"].values()) else "Shared papers: " + ", ".join(f"{k} {v}" for k, v in ov["composite"].items() if v) + ".")
    t.append(f"The strongest length-only signal is {worst:.3f} " + ("(high: length alone separates the classes well, so Step 4 also reports results per length group to show the model does not just use length)." if worst >= 0.75 else "(low: length alone is a weak signal)."))
    md_(" ".join(t))
else:
    not_run("Leakage audit")
"""))

# ================================================================== 3 train
A(md(r"""
---
## 3. Train ModernBERT-base
**The recipe in plain words.**
- **Model:** `answerdotai/ModernBERT-base` plus a small classification layer (2 outputs: Human, AI).
- **Loss:** cross-entropy where the rarer class counts more (class weights from the training data only), plus an *R-Drop* term that asks the model to give the same answer under two different random dropouts of the same review.
- **Training:** 3 epochs, learning rate 3e-5, effective batch 32, mixed precision. After each epoch the model is scored on **validation**; the best epoch is kept (early stopping).
- **Decision threshold:** chosen on validation predictions (it maximises Macro-F1), never on test.
- **Three seeds:** the whole training is repeated with different random seeds, so we can report mean ± std.

The settings are in `configs/train.yaml` above. The **smoke test** trains for one epoch on a tiny subset to check that everything works on this GPU, and measures speed and memory.
"""))
A(code(r"""
if RUN["smoke"] and BUDGET.allow("smoke test", EST_H["smoke"]):
    run_cmd(f"{PY} scripts/03_train_intellabs.py --smoke --gpu 0", CODE)
"""))
A(code(r"""
# What the smoke test measured, and what the full training should therefore take (rows treated as equally expensive: a cautious estimate)
tc = yaml.safe_load((Path(CODE) / "configs" / "train.yaml").read_text())
n_seeds = len(SEEDS) if SEEDS else len(tc["training"]["seeds"])
root = OUT / "models" / "intellabs" / "smoke"
info, hist, ss = jload(root / "seed42" / "run_info.json"), jload(root / "seed42" / "history.json"), jload(root / "seed_summary.json")
if info and hist and ss and summ:
    rate = (ss["train_rows"] + ss["val_rows"]) / hist[0]["seconds"]
    per_seed_h = tc["training"]["epochs"] * (summ["manifests"]["train"]["rows"] + summ["manifests"]["val"]["rows"]) / rate / 3600
    rounds = -(-n_seeds // max(1, min(N_GPUS, n_seeds)))
    display(pd.Series({"speed (reviews per second)": round(rate, 1), "peak GPU memory (GB)": info.get("peak_gpu_gb"), "precision": info["amp"],
                       "estimated hours per seed": round(per_seed_h, 2), f"estimated hours for {n_seeds} seeds on {N_GPUS} GPU(s)": round(per_seed_h * rounds, 2)}, name="smoke test").to_frame())
    EST_H["train"] = per_seed_h * rounds
else:
    print("no smoke-test measurements yet; using the planning estimate")
"""))
A(code(r"""
if RUN["train"] and BUDGET.allow("training", EST_H["train"]):
    seeds = SEEDS or tc["training"]["seeds"]
    jobs = [(f"train_seed{s}", f"{PY} scripts/03_train_intellabs.py --variant base --seeds {s} --gpu {{gpu}}") for s in seeds]
    run_jobs(jobs, CODE, LOGS, n_gpus=min(N_GPUS, len(jobs)))
    disk_report(WORK)
"""))
A(md("### 3.1 Training curves and runs"))
A(code(r"""
fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
rows, any_run = [], False
for d in sorted((OUT / "models" / "intellabs" / TAG).glob("seed*")):
    if not d.is_dir():
        continue
    h, i = jload(d / "history.json"), jload(d / "run_info.json")
    if h:
        any_run = True
        h = pd.DataFrame(h)
        axes[0].plot(h["epoch"], h["train_loss"], marker="o", label=d.name)
        axes[1].plot(h["epoch"], h["val_macro_f1"], marker="o", label=d.name)
    if i:
        rows.append({"seed": i["seed"], "best validation Macro-F1": i["best_val_macro_f1"], "epochs run": i["epochs_run"], "minutes": i["minutes"],
                     "precision": i["amp"], "peak GPU GB": i.get("peak_gpu_gb"), "reviews cut at max length %": 100 * i["truncation"]["train"]["truncated_fraction"],
                     "decision threshold": i["val_threshold"]})
axes[0].set_title("Training loss"); axes[1].set_title("Validation Macro-F1 (higher is better)")
for ax in axes:
    ax.set_xlabel("epoch"); ax.legend(); ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
if any_run:
    save(fig, "training_curves")
    t = pd.DataFrame(rows)
    display(t)
    md_(f"**Takeaway.** Best validation Macro-F1 over {len(t)} run(s): {t['best validation Macro-F1'].mean():.4f} ± {t['best validation Macro-F1'].std(ddof=1) if len(t) > 1 else 0:.4f}. "
        f"{t['reviews cut at max length %'].iloc[0]:.1f}% of training reviews are longer than the model's input limit and are cut at the end (this is reported, not hidden).")
else:
    plt.close(fig); not_run("Training")
"""))

# ================================================================== 4 evaluate
A(md(r"""
---
## 4. Evaluate PeerBERT on `test` and `extended`
Every trained run is scored on the **test** subset and the **extended** subset, with the decision threshold chosen on validation.

**How to read the metrics.** *AI* is the positive class.
- **ROC-AUC**: how well the model **ranks** AI reviews above human ones (1.0 = perfect, 0.5 = guessing). It does not depend on any threshold.
- **Macro-F1**: the F1 score averaged over the two classes, so the rarer class counts as much as the common one. It is the main number here.
- **Accuracy** can look good just because one class is more common, so it is shown but not used to judge.
- **TPR at 1/5/10% FPR**: the share of AI reviews caught while wrongly flagging only 1/5/10% of human reviews. It is shown only when there are enough human reviews to measure it.
- **95% CI**: confidence interval from resampling whole **papers** (reviews of one paper are not independent).
"""))
A(code(r"""
if RUN["evaluate"] and BUDGET.allow("evaluation", EST_H["evaluate"]):
    run_cmd(f"{PY} scripts/05_evaluate.py --dataset intellabs --n-boot {N_BOOT} {SEED_ARG} --gpu 0", CODE)
"""))
A(md("### 4.1 Headline results"))
A(code(r"""
s = jload(RES / "peerbert" / TAG / "metrics_summary.json")
SPLITS = [p for p in ("test", "extended") if s and p in s["per_split"]]
if s:
    rows = []
    for split in SPLITS:
        b = s["per_split"][split]["binary@val_threshold"]
        rows.append({"data": split, "training runs": b["n_seeds"], "Macro-F1": ms(b, "macro_f1"), "ROC-AUC": ms(b, "roc_auc"), "F1 (AI class)": ms(b, "f1"),
                     "Accuracy": ms(b, "accuracy"), "PR-AUC": ms(b, "pr_auc")})
    md_("**PeerBERT, mean ± std over the training runs** (decision threshold from validation)")
    display(pd.DataFrame(rows))
    seed = ref_seed()
    rows = []
    for split in SPLITS:
        m = jload(RES / "peerbert" / TAG / f"seed{seed}" / "metrics.json")["splits"][split]
        ci, b = m.get("cluster_bootstrap_ci", {}), m["binary@val_threshold"]
        rows.append({"data": split, "reviews": b["n"], "AI share": b["n_pos"] / b["n"],
                     "ROC-AUC [95% CI]": fmt_ci(ci["roc_auc"]["estimate"], ci["roc_auc"]["ci_low"], ci["roc_auc"]["ci_high"]) if ci else "–",
                     **{f"TPR@{k} FPR": (f"{v['tpr']:.3f}" if v["supported"] else "n/a (too few human reviews)") for k, v in b["tpr_at_fpr"].items()},
                     "human wrongly flagged": b["ai_fpr"]})
    md_(f"**One run (seed {seed}, the run with the median validation score) with paper-level confidence intervals**")
    display(pd.DataFrame(rows))
    if len(SPLITS) == 2:
        t, e = s["per_split"]["test"]["binary@val_threshold"], s["per_split"]["extended"]["binary@val_threshold"]
        d = e["macro_f1"]["mean"] - t["macro_f1"]["mean"]
        md_(f"**Takeaway.** On the official test, Macro-F1 is {t['macro_f1']['mean']:.4f} and ROC-AUC {t['roc_auc']['mean']:.4f}. On the harder extended subset, Macro-F1 is "
            f"{e['macro_f1']['mean']:.4f} ({d:+.4f}) and ROC-AUC {e['roc_auc']['mean']:.4f}. A drop on extended means the model is somewhat less reliable on reviews "
            "from other papers and other generators; no drop means it holds up.")
else:
    not_run("Evaluation")
"""))
A(md("### 4.2 Mistakes and scores"))
A(code(r"""
if s:
    fig, axes = plt.subplots(1, len(SPLITS), figsize=(5 * len(SPLITS), 4.2), squeeze=False)
    for ax, split in zip(axes[0], SPLITS):
        cm = jload(RES / "peerbert" / TAG / f"seed{seed}" / "metrics.json")["splits"][split]["binary@val_threshold"]["confusion_matrix"]
        M = np.array(cm["matrix"]); Mn = M / M.sum(1, keepdims=True).clip(min=1)
        ax.imshow(Mn, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{M[i, j]}\n({Mn[i, j]:.1%})", ha="center", va="center", color="white" if Mn[i, j] > 0.5 else "black")
        ax.set_xticks([0, 1], ["Human", "AI"]); ax.set_yticks([0, 1], ["Human", "AI"]); ax.set_xlabel("predicted"); ax.set_ylabel("true")
        ax.set_title(f"{split}: confusion matrix (rows add to 100%)")
    save(fig, "peerbert_confusion")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    for split in SPLITS:
        p = peerbert_pred(seed, split)
        fpr, tpr, _ = roc_curve(p["label"], p["score"]); axes[0].plot(fpr, tpr, label=split)
        pr, rc, _ = precision_recall_curve(p["label"], p["score"]); axes[1].plot(rc, pr, label=f"{split} (AI share {p['label'].mean():.2f})")
        for k, g in p.groupby("label"):
            axes[2].hist(g["score"], bins=40, alpha=0.4, density=True, label=f"{split} {LABELS[k]}")
    axes[0].plot([0, 1], [0, 1], ":", c="grey"); axes[0].set_xscale("log"); axes[0].set_xlim(1e-3, 1)
    axes[0].set_xlabel("human reviews wrongly flagged (log scale)"); axes[0].set_ylabel("AI reviews caught"); axes[0].set_title("ROC curve")
    axes[1].set_xlabel("recall"); axes[1].set_ylabel("precision"); axes[1].set_title("Precision-recall curve")
    axes[2].set_xlabel("model's probability that the review is AI"); axes[2].set_yscale("log"); axes[2].set_title("Score distribution")
    for ax in axes:
        ax.legend(fontsize=7)
    save(fig, "peerbert_roc_pr_scores")
"""))
A(md("### 4.3 Where does it work well, and where not?"))
A(code(r"""
def group_plot(key, title, fname):
    fig, axes = plt.subplots(1, len(SPLITS), figsize=(6.5 * len(SPLITS), 3.8), squeeze=False)
    for ax, split in zip(axes[0], SPLITS):
        m = jload(RES / "peerbert" / TAG / f"seed{seed}" / "metrics.json")["splits"][split]
        if key not in m:
            ax.set_axis_off(); continue
        d = pd.DataFrame(m[key]).T
        d[["macro_f1", "roc_auc"]].astype(float).plot.bar(ax=ax, rot=30, ylim=(0, 1.02))
        ax.set_title(f"{split}: {title}")
        md_(f"**{split}: {title}** (small groups give noisier numbers: look at `n`)")
        display(d[["n", "n_pos", "n_neg", "macro_f1", "roc_auc", "ai_fpr"]].rename(columns={"n_pos": "AI", "n_neg": "Human", "ai_fpr": "human wrongly flagged"}))
    save(fig, fname)

if s:
    group_plot("by_generator@val_threshold", "by the LLM that wrote the AI reviews", "peerbert_by_generator")
    group_plot("by_conf_year@val_threshold", "by venue and year", "peerbert_by_venue")
    group_plot("by_length_tercile@val_threshold", "by review length (three equal groups)", "peerbert_by_length")
    lines = []
    for split in SPLITS:
        m = jload(RES / "peerbert" / TAG / f"seed{seed}" / "metrics.json")["splits"][split]
        g = pd.DataFrame(m.get("by_generator@val_threshold", {})).T
        lt = pd.DataFrame(m.get("by_length_tercile@val_threshold", {})).T
        if len(g) > 1:
            gm = g["macro_f1"].astype(float)
            lines.append(f"{split}: hardest generator **{gm.idxmin()}** (Macro-F1 {gm.min():.3f}), easiest **{gm.idxmax()}** ({gm.max():.3f}).")
        if len(lt):
            sp = lt["macro_f1"].astype(float).max() - lt["macro_f1"].astype(float).min()
            lines.append(f"{split}: Macro-F1 differs by {sp:.3f} between the shortest and longest reviews" + (" (small: the model does not just use length)." if sp < 0.03 else " (check the group sizes before reading much into this)."))
    md_("**Takeaway.** " + " ".join(lines))
    for name in ("test_binary_false_positives", "test_binary_false_negatives"):
        p = RES / "peerbert" / TAG / f"seed{seed}" / f"{name}.csv"
        if p.exists():
            md_(f"**{name.replace('test_binary_', '').replace('_', ' ')}** (the most confident mistakes on test; the text is shortened)")
            display(pd.read_csv(p).head(5))
"""))

# ================================================================== 5 baselines
A(md(r"""
---
## 5. Compare with eight baselines
All methods are scored on **the same reviews** (a fixed subset of `test`, and a fixed subset of `extended`), and every method's decision threshold is chosen on the **same validation reviews**.

| Baseline | Type | What it looks at, in one line |
|---|---|---|
| **LogRank** | zero-shot | How highly a language model ranks each word of the review (AI text tends to use likely words). |
| **DetectLLM-LRR** | zero-shot | Ratio of the average log-likelihood to the average log-rank of the words. |
| **Fast-DetectGPT** | zero-shot | How much more likely the actual words are than alternative words the same model would choose. |
| **Binoculars** | zero-shot | Compares how two related language models see the text (perplexity vs. cross-perplexity). |
| **DNA-GPT** | zero-shot | Cuts the review in half, lets a language model rewrite the second half several times, and measures how much the rewrites overlap with the original. |
| **GLTR** | trained classifier | Shares of words that were the model's top-10 / top-100 / top-1000 guesses; a small classifier is fitted on training reviews. |
| **MFD** | trained classifier | Four statistics of the language model's word probabilities; a small classifier is fitted on training reviews. |
| **RADAR** | pretrained detector | A published detector trained by its authors on other data; no training here. |

*Zero-shot* means no training on this data (only the decision threshold comes from validation). *Trained classifier* means a small model fitted on a sample of the training reviews.
The baselines read at most 512 tokens per review and PeerBERT up to 1024. The table at the end of this step lists, for each baseline, what was checked against the original method and where it differs.
"""))
A(code(r"""
if RUN["baselines"] and BUDGET.allow("baselines", EST_H["baselines"]):
    # three groups of baselines; the longest one (DNA-GPT) gets a GPU of its own when two GPUs are available
    jobs = [("baselines_statistical", f"{PY} scripts/06_run_baselines.py --dataset intellabs --only logrank gltr detectllm_lrr mfd fast_detectgpt --gpu {{gpu}}"),
            ("baselines_dnagpt", f"{PY} scripts/06_run_baselines.py --dataset intellabs --only dnagpt --gpu {{gpu}}"),
            ("baselines_radar_binoculars", f"{PY} scripts/06_run_baselines.py --dataset intellabs --only radar binoculars --gpu {{gpu}}")]
    run_jobs(jobs, CODE, LOGS, n_gpus=min(N_GPUS, 2))
    disk_report(WORK)
"""))
A(md("### 5.1 PeerBERT vs the baselines"))
A(code(r"""
CI = {}
for split, title in (("test_baseline", "official test"), ("extended_baseline", "extended")):
    t = ci_table(split)
    if t.empty:
        continue
    CI[split] = t
    show = t.copy()
    show["ROC-AUC [95% CI]"] = [fmt_ci(a, l, h) for a, l, h in zip(t["ROC-AUC"], t["AUC lo"], t["AUC hi"])]
    show["Macro-F1 [95% CI]"] = [fmt_ci(a, l, h) for a, l, h in zip(t["Macro-F1"], t["F1 lo"], t["F1 hi"])]
    md_(f"**{title}: every method on the same {int(t['n'].iloc[0])} reviews** (AI share {t['AI share'].iloc[0]:.2f}; thresholds from validation; CIs resample whole papers)")
    display(show[["type", "method", "ROC-AUC [95% CI]", "Macro-F1 [95% CI]", "PR-AUC", "TPR@10%FPR", "human wrongly flagged"]].sort_values(["type", "method"]).reset_index(drop=True))
if not CI:
    not_run("Baseline comparison")
"""))
A(code(r"""
if CI:
    fig, axes = plt.subplots(len(CI), 2, figsize=(13, 4.2 * len(CI)), squeeze=False)
    for r, (split, t) in enumerate(CI.items()):
        for c, (col, lo, hi, lab) in enumerate((("ROC-AUC", "AUC lo", "AUC hi", "ROC-AUC"), ("Macro-F1", "F1 lo", "F1 hi", "Macro-F1"))):
            d = t.sort_values(col)
            axes[r][c].barh(d["method"], d[col], xerr=[d[col] - d[lo], d[hi] - d[col]], color=[TYPE_COLOR.get(x, "grey") for x in d["type"]], capsize=3)
            axes[r][c].set_xlim(0, 1.02); axes[r][c].set_xlabel(lab); axes[r][c].set_title(f"{split.replace('_baseline', '')}: {lab} with 95% CI")
    fig.legend([plt.Rectangle((0, 0), 1, 1, color=c) for c in TYPE_COLOR.values()], TYPE_COLOR.keys(), loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.0))
    save(fig, "comparison_with_ci")
    fig, axes = plt.subplots(1, len(CI), figsize=(6.5 * len(CI), 5), squeeze=False)
    for ax, split in zip(axes[0], CI):
        for name, (typ, p, thr) in method_predictions(split).items():
            if p["label"].nunique() == 2:
                fpr, tpr, _ = roc_curve(p["label"], p["score"])
                ax.plot(fpr, tpr, lw=2.6 if name == "PeerBERT" else 1.1, color="#6B46C1" if name == "PeerBERT" else None, label=name)
        ax.plot([0, 1], [0, 1], ":", c="grey"); ax.set_xlabel("human reviews wrongly flagged"); ax.set_ylabel("AI reviews caught")
        ax.set_title(f"{split.replace('_baseline', '')}: ROC curves"); ax.legend(fontsize=7, loc="lower right")
    save(fig, "comparison_roc")
"""))
A(md("### 5.2 Which AI generators are hard for which detector?"))
A(code(r"""
for split in CI:
    H = pd.DataFrame({name: {g: v["roc_auc"] for g, v in by_generator(p).items()} for name, (typ, p, thr) in method_predictions(split).items()}).T
    if H.shape[1]:
        fig, ax = plt.subplots(figsize=(1.5 * H.shape[1] + 3, 0.5 * len(H) + 1.6))
        im = ax.imshow(H.astype(float).values, cmap="viridis", vmin=0.4, vmax=1.0, aspect="auto")
        ax.set_xticks(range(H.shape[1]), H.columns, rotation=30); ax.set_yticks(range(len(H)), H.index)
        for i in range(H.shape[0]):
            for j in range(H.shape[1]):
                v = H.iloc[i, j]
                ax.text(j, i, "–" if pd.isna(v) else f"{v:.2f}", ha="center", va="center", color="white" if (not pd.isna(v) and v < 0.75) else "black", fontsize=8)
        fig.colorbar(im, ax=ax, label="ROC-AUC"); ax.set_title(f"{split.replace('_baseline', '')}: ROC-AUC per AI generator (small groups)")
        save(fig, f"comparison_by_generator_{split.replace('_baseline', '')}")
if CI:
    md_("Each cell uses only the reviews of one generator (plus the human reviews of the same venues), so groups are small: read the pattern across detectors rather than single cells.")
"""))
A(md(r"""
### 5.3 Is PeerBERT's advantage real, or could it be luck?
For each baseline, the two methods are compared **on the same reviews**: the difference in ROC-AUC with a 95% interval from resampling whole papers, and a McNemar test on the correct/wrong decisions.
Because eight baselines are compared at once, the p-values are corrected (Holm). *Not significant* means "cannot be told apart with this many reviews", not "equal".
"""))
A(code(r"""
if CI:
    run_cmd(f"{PY} scripts/09_build_report.py --datasets intellabs --n-boot {N_BOOT}", CODE)
for split, suffix, title in (("test_baseline", "", "official test"), ("extended_baseline", "_extended", "extended")):
    p = OUT / "results" / "report" / f"intellabs_paired_tests{suffix}.csv"
    if not p.exists():
        continue
    t = pd.read_csv(p)
    md_(f"**{title}: PeerBERT minus baseline** (ROC-AUC difference; positive = PeerBERT better)")
    display(t[["baseline", "type", "n_shared", "auc_diff_peerbert_minus_baseline", "ci_low", "ci_high", "p_bootstrap_holm", "p_mcnemar"]])
    if "p_bootstrap_holm" in t:
        sig = t[t["p_bootstrap_holm"] < 0.05]
        better = sig[sig["auc_diff_peerbert_minus_baseline"] > 0]["baseline"].tolist()
        worse = sig[sig["auc_diff_peerbert_minus_baseline"] < 0]["baseline"].tolist()
        rest = t[~t["baseline"].isin(sig["baseline"])]["baseline"].tolist()
        md_(f"**Takeaway ({title}).** PeerBERT is significantly better than: {', '.join(better) or 'none'}; significantly worse than: {', '.join(worse) or 'none'}; "
            f"not distinguishable (alpha 0.05, Holm-corrected): {', '.join(rest) or 'none'}.")
"""))
A(md("### 5.4 What each baseline is, and every difference from its original method"))
A(code(r"""
from src.baselines.registry import BASELINES
display(pd.DataFrame([{"baseline": v["display"], "type": v["type"], "checked against the original": "; ".join(v["verified"]) or "not checked",
                       "differences from the original": "; ".join(v["deviations"])} for v in BASELINES.values()]))
rows = []
for d in sorted((RES / "baselines").glob("*/metrics.json")):
    m = jload(d)
    if "orientation" in m["info"]:
        rows.append({"baseline": m["display"], "note": f"output {m['info']['orientation']['used_ai_index']} was chosen as the AI score on validation (validation AUROC per output: {np.round(m['info']['orientation']['val_auroc_per_output'], 3)})"})
if rows:
    md_("**RADAR:** its documentation does not say which output means AI, so the output with the higher validation AUROC is used (validation data only).")
    display(pd.DataFrame(rows))
"""))

# ================================================================== 6 report
A(md(r"""
---
## 6. Report and methods record
The next cell collects the tables (CSV and LaTeX), writes a **methods record** (exact settings, data counts, library versions, a code hash, and every limitation that follows from the settings actually used)
and zips the results for download.
"""))
A(code(r"""
if RUN["report"] and BUDGET.allow("report", EST_H["report"]):
    run_cmd(f"{PY} scripts/09_build_report.py --datasets intellabs --n-boot {N_BOOT}", CODE)
    run_cmd(f"{PY} scripts/11_experiment_card.py --datasets intellabs", CODE)
card = OUT / "results" / "report" / "experiment_card.md"
if card.exists():
    display(Markdown(card.read_text()))
else:
    not_run("Methods record")
"""))
A(code(r"""
import zipfile
RESULTS = OUT / "results"
bundle = Path(WORK) / "PeerBERT_intellabs_results.zip"
if RESULTS.exists():
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as z:
        for p in list(RESULTS.rglob("*")) + list((OUT / "audit").rglob("*")):
            if p.is_file() and p.suffix in (".csv", ".json", ".png", ".pdf", ".md", ".tex"):
                z.write(p, p.relative_to(OUT))
        p = OUT / "manifests" / "intellabs"
        for name in ("summary.json", "loader_stats.json"):
            if (p / name).exists():
                z.write(p / name, (p / name).relative_to(OUT))
    print(f"results bundle: {bundle} ({bundle.stat().st_size / 1e6:.1f} MB)  -> download it from the Output tab")
print("figures:", FIG)
print("tables :", RESULTS / "report")
disk_report(WORK)
print(f"session time used: {BUDGET.elapsed_h():.1f} h of {TIME_BUDGET_H:.0f} h")
progress()
"""))
A(md(r"""
### Limitations to state in the paper
- One dataset: English ML-conference reviews and a fixed set of LLM generators (claude, gemini, gpt-4o, llama, qwen in train/test; gpt-4o and llama in extended). No claim is made beyond them.
- Learning rate, number of epochs, batch size and maximum length were fixed in advance and not tuned. The loss is class-weighted cross-entropy plus R-Drop, chosen in advance, not by looking at test data.
- Training and test sizes are limited to whole papers (see Settings); the baseline comparison uses a smaller shared subset, so its confidence intervals are wide and the detection rate at 1% false positives cannot be measured.
- Reviews longer than the model's input limit are cut at the end; the share is reported.
- The significance tests treat papers as independent and do not model venue, reviewer or generator effects.
- Baselines differ from their original papers where listed in Step 5.4 (for example a smaller language-model pair for Binoculars, and a local language model for DNA-GPT).
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
