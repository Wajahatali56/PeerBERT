"""
11_experiment_card.py
Write a reproducibility record for the paper: outputs/results/report/experiment_card.{md,json}.

Everything is read from files the run itself produced (manifest summaries, loader statistics, run
infos, ablation selection, baseline registry) plus the library versions, GPU names and a hash of the
code and config files. Nothing is estimated or filled in by hand; a missing artefact is listed as
NOT RUN. The "Deviations and limitations" section is generated from the actual settings, so it
states what was really done (compute caps, single-seed ablation, truncation, baseline sample
sizes, deviations of individual baselines).

  python scripts/11_experiment_card.py
"""

import hashlib
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from src.baselines.registry import BASELINES  # noqa: E402
from src.data.common import (  # noqa: E402
    audit_dir, dataset_dir_name, load_config, load_json, manifest_dir, resolve_path, save_json,
)

DATASETS = ("intellabs", "semdetect")


def code_hash():
    h = hashlib.sha256()
    files = []
    for d in ("configs", "src", "scripts"):
        files += [p for p in sorted((ROOT / d).rglob("*")) if p.is_file() and p.suffix in (".py", ".yaml")]
    for p in files:
        h.update(p.relative_to(ROOT).as_posix().encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:16], len(files)


def environment():
    env = {"python": sys.version.split()[0], "platform": platform.platform()}
    for mod in ("torch", "transformers", "huggingface_hub", "numpy", "pandas", "scipy", "sklearn", "captum"):
        try:
            env[mod] = __import__(mod).__version__
        except Exception:                                             # noqa: BLE001
            env[mod] = "not installed"
    try:
        import torch
        env["cuda"] = torch.version.cuda
        env["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except Exception:                                                 # noqa: BLE001
        env["gpus"] = []
    return env


def pct(x):
    return "n/a" if x is None else f"{100 * x:.1f}%"


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    args = ap.parse_args()
    cfg = load_config()
    res_root = resolve_path(cfg, "results_dir", "outputs/results")
    out = res_root / "report"
    out.mkdir(parents=True, exist_ok=True)
    digest, n_files = code_hash()
    card = {"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "environment": environment(), "code_hash": digest, "code_files": n_files,
            "configs": {n: (ROOT / "configs" / n).read_text() for n in ("config.yaml", "train.yaml", "baselines.yaml")},
            "datasets": {}}
    md = ["# Experiment card", "",
          f"Generated {card['created_utc']} from the artefacts of this run. Code hash `{digest}` "
          f"({n_files} files under configs/, src/, scripts/).", "",
          "## Environment", "",
          "| item | value |", "|---|---|"] + [f"| {k} | {v} |" for k, v in card["environment"].items()] + [""]
    limits = []

    for ds in args.datasets:
        dd = dataset_dir_name(cfg, ds)
        d = {}
        card["datasets"][ds] = d
        md += [f"## {ds}", ""]
        a = load_json(audit_dir(cfg, ds) / "audit_status.json") if (audit_dir(cfg, ds) / "audit_status.json").exists() else None
        d["audit"] = a["status"] if a else "NOT RUN"
        md.append(f"- Leakage audit: **{d['audit']}**")
        sp = manifest_dir(cfg, ds) / "summary.json"
        if sp.exists():
            s = load_json(sp)
            d["manifests"], d["caps"], d["exclusions"] = s["manifests"], s["caps"], s["exclusions"]
            md.append(f"- Seed for splits: {s['seed']}. Whole-paper row caps: {s['caps']}")
            for name, v in s["manifests"].items():
                md.append(f"- `{name}`: {v['rows']} rows from {v['papers']} papers, labels {v['label_counts']}")
            md.append(f"- Excluded rows (all logged): {s['exclusions'] or 'none'}")
            capped = {k: v for k, v in s["caps"].items() if v}
            if capped:
                limits.append(f"{ds}: compute caps keep whole papers but not every row ({capped}); full partitions "
                              f"before caps: { {k: v['rows'] for k, v in s['full_partitions'].items()} }.")
        else:
            md.append("- Manifests: NOT RUN")
        ls = manifest_dir(cfg, ds) / "loader_stats.json"
        if ls.exists():
            l = load_json(ls)
            if ds == "intellabs":
                cal = l.get("calibration", {})
                md.append(f"- Human reviews repeated across generator files were kept once: "
                          f"{cal.get('human_duplicate_copies_dropped')} copies dropped, "
                          f"{cal.get('human_copies_with_different_text')} of them with different text; "
                          f"human:AI ratio {cal.get('human_to_ai_ratio')}")
                if l.get("extended_skipped"):
                    md.append(f"- Extended subset was skipped: {l['extended_skipped'][:300]}")
            else:
                ob = l.get("all", {}).get("observed_raw", {})
                md.append(f"- Observed in the files: {ob.get('rows')} rows, classes {ob.get('class_counts')}, "
                          f"splits {ob.get('split_counts')}")
        # models
        ms = resolve_path(cfg, "model_dir", "outputs/models") / dd / "final" / "seed_summary.json"
        if ms.exists():
            m = load_json(ms)
            d["model"] = {k: m[k] for k in ("variant", "seeds", "train_rows", "val_rows", "best_val_macro_f1")}
            md += ["", f"**PeerBERT**: variant `{m['variant']}`, seeds {m['seeds']}, {m['train_rows']} training rows, "
                       f"{m['val_rows']} validation rows.", ""]
            md += ["| seed | best val Macro-F1 | epochs | minutes | AMP | train truncated | median tokens |",
                   "|---|---|---|---|---|---|---|"]
            for r in m["runs"]:
                info = load_json(ms.parent / f"seed{r['seed']}" / "run_info.json")
                tr = info["truncation"]["train"]
                md.append(f"| {r['seed']} | {r['best_val_macro_f1']:.4f} | {r['epochs_run']} | {r['minutes']} | "
                          f"{info['amp']} | {pct(tr['truncated_fraction'])} | {tr['tokens_median']:.0f} |")
            first = load_json(ms.parent / f"seed{m['runs'][0]['seed']}" / "run_info.json")
            tc = first["train_config"]
            md += ["", f"Fixed hyper-parameters (not tuned): lr {tc['training']['lr']}, epochs {tc['training']['epochs']}, "
                       f"effective batch {tc['training']['batch_size'] * tc['training']['grad_accum']}, "
                       f"max length {tc['model']['max_length']} ({tc['model'].get('truncation', 'head')} truncation), "
                       f"weight decay {tc['training']['weight_decay']}, warm-up {tc['training']['warmup_ratio']}.", ""]
            limits.append(f"{ds}: learning rate, epochs, batch size and max length were fixed in advance and not tuned; "
                          f"{pct(first['truncation']['train']['truncated_fraction'])} of training reviews exceed "
                          f"{tc['model']['max_length']} tokens and are truncated.")
        else:
            md += ["", "PeerBERT: NOT RUN", ""]
        ap = res_root / dd / "ablation.json"
        if ap.exists():
            ab = load_json(ap)
            d["ablation"] = {k: ab[k] for k in ("selected_variant", "selection_rule", "budget", "complete")}
            md.append(f"- Ablation selection: **{ab['selected_variant']}**. Rule: {ab['selection_rule']}. "
                      f"Budget: {ab['budget']}")
            limits.append(f"{ds}: the ablation used a single seed and a reduced budget {ab['budget']}; "
                          f"the selected variant was then retrained at full budget with several seeds.")
        bp = res_root / dd / "baselines" / "run_summary.json"
        if bp.exists():
            b = load_json(bp)
            d["baselines"] = {"sample_sizes": b["sample_sizes"], "run": b["baselines"]}
            md.append(f"- Baselines run: {b['baselines']}; sample sizes {b['sample_sizes']}; "
                      f"LM context {b['config']['common']['max_tokens']} tokens")
            limits.append(f"{ds}: baselines were evaluated on a shared whole-paper subset of {b['sample_sizes'].get('test_baseline')} "
                          f"test reviews with {b['config']['common']['max_tokens']}-token context (PeerBERT sees up to "
                          f"{yaml.safe_load(card['configs']['train.yaml'])['model']['max_length']}); confidence intervals are "
                          f"correspondingly wide and TPR at 1% FPR is not estimable.")
        xp = next(iter(sorted((res_root / dd / "xai" / "final").glob("seed*/summary.json"))), None)
        if xp:
            x = load_json(xp)
            d["xai"] = {"n_examples": x["n_examples"], "ig": x["ig"], "sanity": (x.get("sanity_randomization") or {}).get("spearman_abs")}
            md.append(f"- Explainability: {x['n_examples']} reviews, IG {x['ig']['n_steps']} steps, baseline {x['ig']['baseline']}")
        md.append("")

    md += ["## Baseline fidelity", "", "| baseline | type | verified | deviations |", "|---|---|---|---|"]
    for k, v in BASELINES.items():
        md.append(f"| {v['display']} | {v['type']} | {'; '.join(v['verified']) or 'not verified'} | "
                  f"{'; '.join(v['deviations'])} |")
    card["baseline_registry"] = BASELINES
    md += ["", "## Deviations and limitations", ""]
    md += [f"- {t}" for t in limits] or ["- (no results yet)"]
    md += ["- Both datasets are English, ML-conference peer reviews from fixed LLM generators; results do not "
           "transfer to other domains or to generators not represented here without further evidence.",
           "- Human-AI label noise is not measurable here: Sem-Detect 'rewrite' and IntelLabs AI reviews come from "
           "the dataset authors' generation pipelines.",
           "- Differences between methods are reported with paper-level cluster-bootstrap intervals and paired tests; "
           "p-values assume papers are independent clusters (venue, reviewer and generator effects are not modelled)."]
    (out / "experiment_card.md").write_text("\n".join(md), encoding="utf-8")
    save_json(card, out / "experiment_card.json")
    print(f"[card] {out / 'experiment_card.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
