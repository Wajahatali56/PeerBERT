"""
09_build_report.py
Collect every result that EXISTS on disk into thesis tables. Nothing is estimated: a
missing artefact is reported as "NOT RUN".

Outputs: outputs/results/report/
  report.md                       what was run, audit status, main tables, notes
  <ds>_comparison.csv / .tex      all methods on test_baseline, grouped by type
  <ds>_paired_tests.csv           PeerBERT vs each baseline on the SAME test_baseline rows:
                                  paired cluster bootstrap (ROC-AUC) + McNemar (val-calibrated
                                  decisions), Holm-adjusted across baselines
  <ds>_roc.png                    ROC curves on test_baseline (if matplotlib is available)

PeerBERT reference seed for paired tests = the seed with the median VALIDATION Macro-F1
(selection on validation only).

  python scripts/09_build_report.py
  python scripts/09_build_report.py --n-boot 2000
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.data.common import (  # noqa: E402
    audit_dir, dataset_dir_name, load_config, load_json, manifest_dir, resolve_path, save_json,
)
from src.evaluation.metrics import calibrate_threshold  # noqa: E402
from src.evaluation.predict import binary_view, load_predictions  # noqa: E402
from src.evaluation.stats import auc_fn, holm, mcnemar_exact, paired_cluster_bootstrap  # noqa: E402

DATASETS = ("intellabs", "semdetect")
TABLE_COLS = ["type", "method", "threshold_kind", "n", "accuracy", "precision", "recall", "f1", "macro_f1",
              "roc_auc", "pr_auc", "tpr@1%fpr", "tpr@5%fpr", "tpr@10%fpr"]


def latex_table(df, caption, label):
    cols = [c for c in TABLE_COLS if c in df.columns]
    lines = ["\\begin{table}[t]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             "\\begin{tabular}{" + "l" * 3 + "r" * (len(cols) - 3) + "}", "\\toprule",
             " & ".join(c.replace("_", "\\_").replace("%", "\\%").replace("@", "@") for c in cols) + " \\\\",
             "\\midrule"]
    prev = None
    for _, r in df[cols].iterrows():
        if prev is not None and r["type"] != prev:
            lines.append("\\midrule")
        prev = r["type"]
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                cells.append("--" if np.isnan(v) else f"{v:.3f}")
            else:
                cells.append("--" if v is None else str(v).replace("_", "\\_").replace("%", "\\%"))
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def reference_seed(model_dir):
    """Seed with the median validation Macro-F1 among finished seeds."""
    runs = []
    for d in sorted(model_dir.glob("seed*")):
        if (d / "done.json").exists():
            runs.append((load_json(d / "done.json")["best_val_macro_f1"], int(d.name[4:])))
    if not runs:
        return None
    runs.sort()
    return runs[(len(runs) - 1) // 2][1]


def peerbert_test_baseline(cfg, ds, res_root, tag):
    model_dir = resolve_path(cfg, "model_dir", "outputs/models") / dataset_dir_name(cfg, ds) / tag
    seed = reference_seed(model_dir)
    if seed is None:
        return None, None, None
    p = res_root / "peerbert" / tag / f"seed{seed}" / "test_baseline_predictions.csv"
    if not p.exists():
        return seed, None, None
    pred = load_predictions(p)
    vp = pd.read_csv(model_dir / f"seed{seed}" / "val_predictions.csv")
    if ds == "semdetect":
        pred, vp = binary_view(pred), binary_view(vp)
    thr = calibrate_threshold(vp["label"], vp["score"])
    return seed, pred, thr


def paired_tests(cfg, ds, res_root, tag, n_boot):
    seed, pb, pb_thr = peerbert_test_baseline(cfg, ds, res_root, tag)
    if pb is None:
        return None, f"PeerBERT test_baseline predictions NOT RUN (reference seed: {seed})"
    rows, pvals = [], {}
    for mdir in sorted((res_root / "baselines").glob("*/")):
        pp = mdir / "predictions_test_baseline.csv"
        if not pp.exists():
            continue
        m = load_json(mdir / "metrics.json")
        bp = load_predictions(pp)
        j = pb.merge(bp[["row_id", "score", "pred"]], on="row_id", suffixes=("_pb", "_bl"))
        if len(j) != len(bp) or len(j) != len(pb):
            rows.append({"baseline": m["display"], "note": f"row mismatch: peerbert {len(pb)}, baseline {len(bp)}, "
                                                           f"shared {len(j)}"})
            continue
        boot = paired_cluster_bootstrap(j["label"], j["score_pb"], j["score_bl"], j["paper_key"], auc_fn, n_boot)
        mc = mcnemar_exact(j["label"], (j["score_pb"] >= pb_thr).astype(int), j["pred_bl"])
        pvals[m["display"]] = boot["p_value"]
        rows.append({"baseline": m["display"], "type": m["type"], "n_shared": len(j), "n_papers": boot.get("n_clusters"),
                     "auc_diff_peerbert_minus_baseline": boot["diff"], "ci_low": boot["ci_low"],
                     "ci_high": boot["ci_high"], "p_bootstrap": boot["p_value"], "p_mcnemar": mc["p_value"],
                     "peerbert_right_baseline_wrong": mc["a_right_b_wrong"],
                     "peerbert_wrong_baseline_right": mc["a_wrong_b_right"]})
    adj = holm(pvals)
    for r in rows:
        if r.get("baseline") in adj:
            r["p_bootstrap_holm"] = adj[r["baseline"]]
    return pd.DataFrame(rows), f"PeerBERT reference seed {seed} (median validation Macro-F1); n_boot={n_boot}"


def roc_figure(ds, res_root, tag, out_png):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve
    except ImportError:
        return False
    curves = []
    for mdir in sorted((res_root / "baselines").glob("*/")):
        pp = mdir / "predictions_test_baseline.csv"
        if pp.exists():
            curves.append((load_json(mdir / "metrics.json")["display"], load_predictions(pp)))
    pb = sorted((res_root / "peerbert" / tag).glob("seed*/test_baseline_predictions.csv"))
    if pb:
        p = load_predictions(pb[0])
        curves.insert(0, (f"PeerBERT ({pb[0].parent.name})", binary_view(p) if ds == "semdetect" else p))
    if not curves:
        return False
    fig, ax = plt.subplots(figsize=(5, 5))
    for name, p in curves:
        if p["label"].nunique() == 2:
            fpr, tpr, _ = roc_curve(p["label"], p["score"])
            ax.plot(fpr, tpr, label=name, lw=1.4)
    ax.plot([0, 1], [0, 1], ls=":", c="grey")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"{ds}: test_baseline")
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    return True


def status_of(path):
    return "present" if Path(path).exists() else "NOT RUN"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="final")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = resolve_path(cfg, "results_dir", "outputs/results") / "report"
    out.mkdir(parents=True, exist_ok=True)
    md = ["# Results report", "", "Generated by scripts/09_build_report.py from artefacts on disk. "
          "Anything marked NOT RUN was not executed.", ""]
    index = {}
    for ds in DATASETS:
        dd = dataset_dir_name(cfg, ds)
        res_root = resolve_path(cfg, "results_dir", "outputs/results") / dd
        md += [f"## {ds}", ""]
        a = audit_dir(cfg, ds) / "audit_status.json"
        md.append(f"- Leakage audit: {load_json(a)['status'] if a.exists() else 'NOT RUN'}")
        s = manifest_dir(cfg, ds) / "summary.json"
        if s.exists():
            for name, d in load_json(s).get("manifests", {}).items():
                md.append(f"- Manifest `{name}`: {d['rows']} rows, {d['papers']} papers, labels {d['label_counts']}")
        else:
            md.append("- Manifests: NOT RUN")
        arts = {
            "PeerBERT metrics": res_root / "peerbert" / args.tag / "metrics_summary.json",
            "Baselines": res_root / "baselines" / "run_summary.json",
            "Ablation": res_root / "ablation.json",
            "Cross-dataset (into this dataset)": next(iter(sorted(res_root.glob("cross_from_*/metrics.json"))),
                                                      res_root / "cross_from_x" / "metrics.json"),
        }
        md += [f"- {k}: {status_of(v)}" for k, v in arts.items()] + [""]

        ms = arts["PeerBERT metrics"]
        if ms.exists():
            summ = load_json(ms)
            md += ["### PeerBERT (mean +/- std over seeds)", "", "| split | metric | mean | std | n_seeds |",
                   "|---|---|---|---|---|"]
            for split, block in summ["per_split"].items():
                for kind, vals in block.items():
                    for metric in ("macro_f1", "roc_auc", "accuracy"):
                        if metric in vals:
                            md.append(f"| {split} | {kind} {metric} | {vals[metric]['mean']:.4f} | "
                                      f"{vals[metric]['std']:.4f} | {vals['n_seeds']} |")
            md.append("")

        ct = res_root / "comparison_table.json"
        if ct.exists():
            df = pd.DataFrame(load_json(ct))
            df = df[df["split"] == "test_baseline"]
            df.to_csv(out / f"{ds}_comparison.csv", index=False)
            (out / f"{ds}_comparison.tex").write_text(latex_table(
                df, f"{ds}: all methods on the shared test\\_baseline subset", f"tab:{ds}_comparison"))
            md += ["### Comparison on test_baseline", "", f"See `{ds}_comparison.csv` / `.tex` "
                   "(zero-shot detectors, pretrained detectors and trained classifiers are separate groups).", ""]
            pt, note = paired_tests(cfg, ds, res_root, args.tag, args.n_boot)
            if pt is not None:
                pt.to_csv(out / f"{ds}_paired_tests.csv", index=False)
                md += ["### Paired tests", "", note, f"See `{ds}_paired_tests.csv`. Assumptions: papers are "
                       "independent clusters; McNemar ignores clustering; Holm across baselines.", ""]
            else:
                md += [f"Paired tests: {note}", ""]
            if roc_figure(ds, res_root, args.tag, out / f"{ds}_roc.png"):
                md.append(f"![ROC]({ds}_roc.png)\n")
        else:
            md += ["Comparison table: NOT RUN", ""]
        if arts["Ablation"].exists():
            ab = load_json(arts["Ablation"])
            md += ["### Ablation (validation Macro-F1, single seed)", "", "| variant | val Macro-F1 |", "|---|---|"]
            md += [f"| {r['variant']} | {r['val_macro_f1']:.4f} |" for r in ab["rows"]] + [""]
        cr = arts["Cross-dataset (into this dataset)"]
        if cr.exists():
            c = load_json(cr)
            m = c["mean_std@source_val_threshold"]
            md += [f"### Cross-dataset {c['source']} -> {c['target']} (no retraining)", "",
                   f"Macro-F1 @ source-val threshold: {m.get('macro_f1', {}).get('mean')} "
                   f"+/- {m.get('macro_f1', {}).get('std')}; ROC-AUC {m.get('roc_auc', {}).get('mean')}", ""]
            md += [f"- {n}" for n in c["mismatch_notes"]] + [""]
        index[ds] = {k: str(v) for k, v in arts.items()}
    (out / "report.md").write_text("\n".join(md), encoding="utf-8")
    save_json(index, out / "artefacts.json")
    print(f"[report] {out / 'report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
