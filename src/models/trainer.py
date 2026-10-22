"""
Custom single-GPU PyTorch training loop for PeerBERT (no HF Trainer, no DataParallel).

- CUDA_VISIBLE_DEVICES must be set by the calling script BEFORE torch is imported
  (scripts 03/04/07 do this from --gpu / train.yaml gpu_id).
- AMP: bf16 when supported, else fp16 with GradScaler; "off" = fp32.
- Gradient accumulation, gradient clipping, linear warmup/decay.
- Non-finite loss guard: the batch is skipped (counted); too many in a row aborts.
- Validation every epoch; early stopping and best-checkpoint selection on val Macro-F1.
- Restart-safe: last_state.pt (resume at epoch boundaries) and done.json (skip finished runs).
- Saves: best/ (HF checkpoint + tokenizer), val_predictions.csv (best model),
  history.json, run_info.json (config, dropout check, truncation, AMP, timings).
"""

import copy
import gc
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from src.data.common import ROOT, jsonable, save_json, set_seed
from src.evaluation.metrics import binary_metrics, calibrate_threshold, multiclass_metrics
from src.evaluation.predict import prediction_frame, run_inference, softmax

from .dataset import ReviewDataset
from .losses import PeerBERTLoss, class_weights
from .modernbert_classifier import build_model, has_active_dropout, load_trained, passes_differ

CLASS_NAMES = {"intellabs": ("human", "ai"), "semdetect": ("human", "ai", "rewrite")}


# ---------------------------------------------------------------- config
def deep_merge(base, over):
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_train_config(path=None, overrides=None):
    path = Path(path) if path else ROOT / "configs" / "train.yaml"
    with path.open(encoding="utf-8") as f:
        return deep_merge(yaml.safe_load(f) or {}, overrides)


def amp_dtype(setting, device):
    if setting == "off" or device.type != "cuda":
        return None
    if setting in ("auto", "bf16") and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def make_scaler(enabled):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def free_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------- helpers
def _loader(ds, batch_size, shuffle, seed, workers):
    g = torch.Generator()
    g.manual_seed(seed)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=g,
                                       collate_fn=ds.collate, num_workers=workers,
                                       pin_memory=torch.cuda.is_available())


def _sorted_loader(ds, batch_size, workers):
    """Length-sorted order for fast inference; returns (loader, order)."""
    order = np.argsort([len(x) for x in ds.input_ids], kind="stable")
    sub = torch.utils.data.Subset(ds, order.tolist())
    return torch.utils.data.DataLoader(sub, batch_size=batch_size, shuffle=False, collate_fn=ds.collate,
                                       num_workers=workers), order


def predict_logits(model, ds, batch_size, device, dtype, workers=0):
    loader, order = _sorted_loader(ds, batch_size, workers)
    logits = run_inference(model, loader, device, dtype)
    out = np.empty_like(logits)
    out[order] = logits
    return out


def eval_metrics(y, proba, class_names):
    if len(class_names) == 2:
        return binary_metrics(y, proba[:, 1])
    return multiclass_metrics(y, proba, class_names)


def _optimizer(model, lr, wd):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim < 2 or "norm" in n.lower() or n.endswith("bias") else decay).append(p)
    return torch.optim.AdamW([{"params": decay, "weight_decay": wd},
                              {"params": no_decay, "weight_decay": 0.0}], lr=lr)


def _scheduler(opt, total_steps, warmup_ratio):
    warm = max(1, int(total_steps * warmup_ratio))

    def f(step):
        if step < warm:
            return (step + 1) / warm
        return max(0.0, (total_steps - step) / max(1, total_steps - warm))
    return torch.optim.lr_scheduler.LambdaLR(opt, f)


def _rng_state():
    return {"py": random.getstate(), "np": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def _set_rng_state(s):
    random.setstate(s["py"])
    np.random.set_state(s["np"])
    torch.set_rng_state(s["torch"])
    if s.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(s["cuda"])


# ---------------------------------------------------------------- training
def train_one(train_df, val_df, tcfg, dataset, out_dir, seed, resume=True):
    """Train one seed. Returns run_info dict (also saved as run_info.json)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "done.json").exists():
        print(f"[train] {out_dir} already finished; skipping (delete done.json to retrain)")
        return json.loads((out_dir / "done.json").read_text())

    mcfg, tr, lcfg = tcfg["model"], tcfg["training"], tcfg["loss"]
    class_names = CLASS_NAMES[dataset]
    k = len(class_names)
    set_seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = amp_dtype(tr.get("amp", "auto"), device)
    t0 = time.time()

    head = mcfg.get("head", "linear")
    if lcfg["type"] == "csm" and head != "cosine":
        raise ValueError("loss 'csm' needs the cosine head: set model.head: cosine")
    model, tok, model_info = build_model(mcfg["name"], k, class_names, mcfg.get("dropout_on", True),
                                         mcfg.get("dropout", 0.1), head, mcfg.get("cosine_scale", 30.0))
    if tr.get("gradient_checkpointing"):
        model.hf.gradient_checkpointing_enable()
        model_info["gradient_checkpointing"] = True
    model.to(device)
    ds_kw = dict(max_length=mcfg["max_length"], truncation=mcfg.get("truncation", "head"),
                 head_tokens=mcfg.get("head_tokens", 256))
    train_ds = ReviewDataset(train_df["text"].tolist(), train_df["label"].values, tok, **ds_kw)
    val_ds = ReviewDataset(val_df["text"].tolist(), val_df["label"].values, tok, **ds_kw)

    weights = class_weights(train_df["label"].values, k)
    loss_fn = PeerBERTLoss(lcfg["type"], weights, lcfg.get("rdrop_alpha", 0.0),
                           lcfg.get("label_smoothing", 0.05), lcfg.get("focal_gamma", 2.0),
                           {"scale": mcfg.get("cosine_scale", 30.0), "m_base": lcfg.get("csm_base_margin", 0.25),
                            "m_cost": lcfg.get("csm_cost_margin", 0.25)})
    if loss_fn.uses_rdrop and not has_active_dropout(model.hf):
        raise RuntimeError("R-Drop is on but the model has no active dropout (set model.dropout_on: true)")

    bs, accum = int(tr["batch_size"]), int(tr["grad_accum"])
    loader = _loader(train_ds, bs, True, seed, int(tr.get("num_workers", 0)))
    steps_per_epoch = math.ceil(len(loader) / accum)
    total_steps = steps_per_epoch * int(tr["epochs"])
    opt = _optimizer(model, float(tr["lr"]), float(tr["weight_decay"]))
    sched = _scheduler(opt, total_steps, float(tr["warmup_ratio"]))
    scaler = make_scaler(dtype == torch.float16)

    state = {"epoch_done": 0, "best": -1.0, "bad_epochs": 0, "history": [], "skipped_nonfinite": 0,
             "rdrop_passes_differ": None}
    state_p = out_dir / "last_state.pt"
    if resume and state_p.exists():
        ck = torch.load(state_p, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        state = ck["state"]
        _set_rng_state(ck["rng"])
        print(f"[train] resumed from {state_p} after epoch {state['epoch_done']}")

    print(f"[train] {dataset} seed={seed} device={device} amp={dtype} train={len(train_ds)} val={len(val_ds)} "
          f"loss={lcfg['type']} rdrop_alpha={loss_fn.alpha} dropout={model_info['dropout_after']}")
    bad_in_row = 0
    for epoch in range(state["epoch_done"] + 1, int(tr["epochs"]) + 1):
        model.train()
        loader = _loader(train_ds, bs, True, seed + epoch, int(tr.get("num_workers", 0)))
        run_loss, n_ok, te = 0.0, 0, time.time()
        opt.zero_grad(set_to_none=True)
        for i, batch in enumerate(loader):
            batch = {kk: v.to(device, non_blocking=True) for kk, v in batch.items()}
            if loss_fn.uses_rdrop and state["rdrop_passes_differ"] is None:
                state["rdrop_passes_differ"] = passes_differ(model, batch)
                if not state["rdrop_passes_differ"]:
                    raise RuntimeError("R-Drop: two train-mode passes are identical (dropout inactive)")
            try:
                with torch.autocast(device_type=device.type, dtype=dtype or torch.float32,
                                    enabled=dtype is not None):
                    l1 = model(batch["input_ids"], batch["attention_mask"])
                    l2 = model(batch["input_ids"], batch["attention_mask"]) if loss_fn.uses_rdrop else None
                loss, parts = loss_fn(l1, batch["labels"], l2)
            except torch.cuda.OutOfMemoryError as e:
                raise RuntimeError(
                    "CUDA out of memory. In configs/train.yaml lower training.batch_size (and raise "
                    "grad_accum to keep the effective batch), set training.gradient_checkpointing: true, "
                    "or reduce model.max_length. The run resumes from the last finished epoch.") from e
            if not torch.isfinite(loss):
                state["skipped_nonfinite"] += 1
                bad_in_row += 1
                opt.zero_grad(set_to_none=True)
                if bad_in_row >= int(tr.get("max_bad_steps", 20)):
                    raise RuntimeError(f"{bad_in_row} consecutive non-finite losses; aborting")
                continue
            bad_in_row = 0
            scaler.scale(loss / accum).backward()
            run_loss += float(loss.detach())
            n_ok += 1
            if (i + 1) % accum == 0 or (i + 1) == len(loader):
                scaler.unscale_(opt)
                gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(tr["max_grad_norm"]))
                if torch.isfinite(gnorm):
                    scaler.step(opt)
                else:
                    state["skipped_nonfinite"] += 1
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()

        val_logits = predict_logits(model, val_ds, int(tr["eval_batch_size"]), device, dtype)
        val_proba = softmax(val_logits)
        vm = eval_metrics(val_df["label"].values, val_proba, class_names)
        score = vm["macro_f1"]
        improved = score > state["best"]
        state["history"].append({"epoch": epoch, "train_loss": run_loss / max(n_ok, 1), "val_macro_f1": score,
                                 "val_accuracy": vm["accuracy"], "lr": sched.get_last_lr()[0],
                                 "seconds": round(time.time() - te, 1), "improved": improved})
        print(f"[train] epoch {epoch}: loss={run_loss / max(n_ok, 1):.4f} val_macro_f1={score:.4f}"
              f"{' *' if improved else ''}")
        if improved:
            state["best"], state["bad_epochs"] = score, 0
            model.save(out_dir / "best", tok)
            vp = prediction_frame(val_df, proba=val_proba, class_names=class_names)
            vp.to_csv(out_dir / "val_predictions.csv", index=False)
        else:
            state["bad_epochs"] += 1
        state["epoch_done"] = epoch
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "state": state, "rng": _rng_state()}, state_p)
        save_json(state["history"], out_dir / "history.json")
        if state["bad_epochs"] >= int(tr["early_stopping_patience"]):
            print(f"[train] early stopping after epoch {epoch}")
            break

    vp = pd.read_csv(out_dir / "val_predictions.csv")
    val_threshold = None
    if k == 2:
        val_threshold = calibrate_threshold(vp["label"], vp["score"], tcfg.get("threshold", {}).get("objective", "macro_f1"))
    info = {
        "dataset": dataset, "seed": seed, "best_val_macro_f1": state["best"], "epochs_run": state["epoch_done"],
        "val_threshold": val_threshold, "skipped_nonfinite_steps": state["skipped_nonfinite"],
        "rdrop_passes_differ": state["rdrop_passes_differ"], "amp": str(dtype), "device": str(device),
        "model_info": model_info, "class_weights": weights.tolist(), "train_config": tcfg,
        "truncation": {"train": train_ds.truncation_stats(), "val": val_ds.truncation_stats()},
        "minutes": round((time.time() - t0) / 60, 2), "best_dir": str(out_dir / "best"),
    }
    save_json(info, out_dir / "run_info.json")
    save_json(info, out_dir / "done.json")
    if state_p.exists():
        state_p.unlink()            # finished: the large optimizer state is no longer needed
    del model, opt, sched, scaler
    free_gpu()
    return jsonable(info)


def predict_proba(model_dir, texts, tcfg, eager_attention=False):
    """Probabilities from a saved checkpoint + truncation stats of these texts."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = amp_dtype(tcfg["training"].get("amp", "auto"), device)
    model, tok = load_trained(model_dir, eager_attention)
    model.to(device)
    m = tcfg["model"]
    ds = ReviewDataset(list(texts), None, tok, m["max_length"], m.get("truncation", "head"), m.get("head_tokens", 256))
    proba = softmax(predict_logits(model, ds, int(tcfg["training"]["eval_batch_size"]), device, dtype))
    stats = ds.truncation_stats()
    del model
    free_gpu()
    return proba, stats


# ---------------------------------------------------------------- script entry
def model_root(cfg, dataset):
    from src.data.common import dataset_dir_name, resolve_path
    return resolve_path(cfg, "model_dir", "outputs/models") / dataset_dir_name(cfg, dataset)


def train_from_manifests(dataset, cfg, tcfg, seeds, tag="final", max_train_rows=None, max_val_rows=None):
    """
    Audit gate -> manifests -> one run per seed -> seed summary.
    Models: outputs/models/<ds>/<tag>/seed<k>/. Smoke-test limits keep whole papers.
    """
    from src.data.common import audit_dir, manifest_dir
    from src.data.leakage_audit import assert_audit_passed
    from src.data.splits import cap_by_papers
    from src.evaluation.aggregate import seed_summary

    status = assert_audit_passed(audit_dir(cfg, dataset))
    mdir = manifest_dir(cfg, dataset)
    train = pd.read_parquet(mdir / "train.parquet")
    val = pd.read_parquet(mdir / "val.parquet")
    if max_train_rows:
        train = cap_by_papers(train, max_train_rows, seeds[0])
    if max_val_rows:
        val = cap_by_papers(val, max_val_rows, seeds[0])
    root = model_root(cfg, dataset) / tag
    runs = []
    for s in seeds:
        runs.append(train_one(train, val, tcfg, dataset, root / f"seed{s}", s))
    summary = {"dataset": dataset, "tag": tag, "audit_status": status, "seeds": list(seeds),
               "train_rows": len(train), "val_rows": len(val),
               "smoke_limits": {"max_train_rows": max_train_rows, "max_val_rows": max_val_rows},
               "best_val_macro_f1": seed_summary([{"macro_f1": r["best_val_macro_f1"]} for r in runs],
                                                 keys=("macro_f1",)),
               "runs": [{k: r[k] for k in ("seed", "best_val_macro_f1", "epochs_run", "val_threshold",
                                           "best_dir", "minutes")} for r in runs]}
    save_json(summary, root / "seed_summary.json")
    print(f"[train] {dataset}/{tag}: val Macro-F1 over seeds = {summary['best_val_macro_f1'].get('macro_f1')}")
    return summary
