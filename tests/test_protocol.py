"""Protocol tests: selection rule, optional extended subset, Sem-Detect file discovery, real schemas, notebook helpers."""

import importlib.util
import json
import os
import sys
import time

import pandas as pd
import pytest

from conftest import ROOT, intel_frame
from src.data.intellabs_loader import classify_columns


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), ROOT / "scripts" / name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- selection rule
def test_selection_keeps_default_unless_gain_is_large_enough():
    sel = _load_script("07_run_ablation.py").select_variant
    rows = [{"variant": "ce_weighted", "val_macro_f1": 0.900}, {"variant": "rdrop_1.0", "val_macro_f1": 0.902},
            {"variant": "focal", "val_macro_f1": 0.905}]
    name, why = sel(rows, "rdrop_1.0", 0.005)
    assert name == "rdrop_1.0" and "kept the pre-declared default" in why        # gain 0.003 < 0.005
    name, why = sel(rows, "rdrop_1.0", 0.002)
    assert name == "focal" and "beats the default" in why
    assert sel(rows, "not_run", 0.005)[0] == "focal"                              # default absent: plain argmax
    tie = [{"variant": "a", "val_macro_f1": 0.5}, {"variant": "b", "val_macro_f1": 0.5}]
    assert sel(tie, "zzz", 0.005)[0] == "a"                                       # ties: first listed


def test_default_variant_equals_base_configuration():
    import yaml
    c = yaml.safe_load((ROOT / "configs" / "train.yaml").read_text())
    v = next(x for x in c["ablation"]["variants"] if x["name"] == c["ablation"]["default_variant"])
    assert v["loss"]["type"] == c["loss"]["type"] and v["loss"]["rdrop_alpha"] == c["loss"]["rdrop_alpha"]
    assert "model" not in v and "training" not in v


# ---------------------------------------------------------------- variants
def test_resolve_variant(cfg):
    from src.data.common import save_json
    from src.models.trainer import load_train_config, resolve_variant
    t = load_train_config()
    assert resolve_variant(cfg, "intellabs", t, None) == (t, "base")
    t2, name = resolve_variant(cfg, "intellabs", t, "csm_cosine")
    assert name == "csm_cosine" and t2["loss"]["type"] == "csm" and t2["model"]["head"] == "cosine"
    assert t["loss"]["type"] != "csm"                                              # original untouched
    with pytest.raises(KeyError, match="unknown variant"):
        resolve_variant(cfg, "intellabs", t, "nope")
    with pytest.raises(FileNotFoundError, match="07_run_ablation"):
        resolve_variant(cfg, "intellabs", t, "auto")
    from src.data.common import dataset_dir_name, resolve_path
    p = resolve_path(cfg, "results_dir", "outputs/results") / dataset_dir_name(cfg, "intellabs") / "ablation.json"
    save_json({"selected_variant": "focal_weighted"}, p)
    assert resolve_variant(cfg, "intellabs", t, "auto")[1] == "focal_weighted"


# ---------------------------------------------------------------- optional extended subset
def test_extended_failure_does_not_block_the_main_experiment(cfg, hub, tmp_path, monkeypatch):
    import yaml
    for gen in ("claude", "gpt4o"):
        hub.add("calibration", gen, "ICLR", 2021, intel_frame("ICLR", 2021, gen, n_papers=10))
        hub.add("test", gen, "ICLR", 2021, intel_frame("ICLR", 2021, gen, n_papers=5, pid_offset=500))
    hub.add("extended", "gpt4o", "ICLR", 2021,
            intel_frame("ICLR", 2021, "gpt4o", pid_offset=900, extra=lambda l: {"unreviewed_essay": "y" * 400}))
    cfg["data"]["intellabs"]["use_extended"] = True
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(cfg))
    mod = _load_script("02_prepare_splits.py")
    real = mod.load_subset
    monkeypatch.setattr(mod, "load_subset", lambda c, s, use_cache=True: real(
        c, s, lister=hub.lister, fetcher=hub.fetcher, use_cache=use_cache))
    assert mod.main(["--config", str(p), "--dataset", "intellabs"]) == 0
    from src.data.common import manifest_dir
    md = manifest_dir(cfg, "intellabs")
    assert (md / "train.parquet").exists() and not (md / "extended.parquet").exists()
    stats = json.loads((md / "loader_stats.json").read_text())
    assert "unreviewed_essay" in stats["extended_skipped"]


# ---------------------------------------------------------------- Sem-Detect file discovery
def test_select_parquet_files():
    from src.data.sem_detect_loader import select_parquet_files
    files = ["README.md", "reviews/train-0.parquet", "reviews/test-0.parquet", "claims/all.parquet"]
    assert select_parquet_files(files, "reviews") == (["reviews/test-0.parquet", "reviews/train-0.parquet"],
                                                     "files under reviews/")
    got, how = select_parquet_files(["data/reviews-00000.parquet", "data/other.parquet"], "reviews")
    assert got == ["data/reviews-00000.parquet"] and "contains" in how
    got, how = select_parquet_files(["a.parquet", "b.parquet"], "reviews")
    assert got == ["a.parquet", "b.parquet"] and "all Parquet" in how
    with pytest.raises(FileNotFoundError, match="no Parquet files"):
        select_parquet_files(["README.md", "x.csv"], "reviews")


# ---------------------------------------------------------------- real column layouts
def test_every_column_layout_in_the_real_inspection_report_is_handled(cfg):
    rep = json.loads((ROOT / "outputs" / "audit" / "inspection_report.json").read_text())["intellabs"]
    icfg = cfg["data"]["intellabs"]
    layouts = {tuple(f["columns"]) for f in rep["files"]}
    assert len(layouts) >= 10
    for cols in layouts:
        df = pd.DataFrame({c: [("x " * 200) if c in icfg["text_fields_allow"] else "1"] for c in cols})
        r = classify_columns(df, icfg)
        assert r["text_fields"], f"no text field for {cols}"
        assert not r["unknown_narrative"] and not r["unknown_other"], (cols, r)
        assert "code_of_conduct" not in r["text_fields"] and "rating" not in r["text_fields"]


# ---------------------------------------------------------------- notebook helpers
def test_run_cmd_and_failure(tmp_path):
    from src.notebook_utils import run_cmd
    run_cmd(f'"{sys.executable}" -c "print(1+1)"', tmp_path)
    with pytest.raises(RuntimeError, match="step failed"):
        run_cmd(f'"{sys.executable}" -c "import sys; print(\'boom\'); sys.exit(3)"', tmp_path)


def test_run_jobs_uses_one_queue_per_gpu(tmp_path):
    from src.notebook_utils import run_jobs
    code = ("import os,time,sys;print('gpu',os.environ.get('CUDA_VISIBLE_DEVICES'));"
            "time.sleep(1.5);print('done')")
    script = tmp_path / "job.py"
    script.write_text(code)
    jobs = [("a", f'CUDA_VISIBLE_DEVICES={{gpu}} "{sys.executable}" {script}'),
            ("b", f'CUDA_VISIBLE_DEVICES={{gpu}} "{sys.executable}" {script}')]
    t0 = time.time()
    run_jobs(jobs, tmp_path, tmp_path / "logs", n_gpus=2, poll_s=0.2)
    parallel = time.time() - t0
    assert parallel < 2.9                                           # two 1.5 s jobs ran at the same time
    assert "gpu 0" in (tmp_path / "logs" / "a.log").read_text() and "gpu 1" in (tmp_path / "logs" / "b.log").read_text()
    t0 = time.time()
    run_jobs(jobs, tmp_path, tmp_path / "logs1", n_gpus=1, poll_s=0.2)
    assert time.time() - t0 > 2.9                                   # one GPU: sequential
    bad = [("ok", f'"{sys.executable}" -c "print(1)"'), ("bad", f'"{sys.executable}" -c "import sys;sys.exit(2)"')]
    with pytest.raises(RuntimeError, match="failed jobs: bad"):
        run_jobs(bad, tmp_path, tmp_path / "logs2", n_gpus=2, poll_s=0.2)


def test_budget_and_disk_and_restore(tmp_path, capsys):
    from src.notebook_utils import Budget, disk_report, restore_outputs
    clock = {"t": 0.0}
    b = Budget(10, clock=lambda: clock["t"])
    assert b.allow("step", 4.0)
    clock["t"] = 7 * 3600
    assert not b.allow("big step", 4.0) and b.allow("small step", 2.0)
    assert "SKIPPED" in capsys.readouterr().out

    work = tmp_path / "work" / "PeerBERT_outputs"
    prev = tmp_path / "input" / "my-notebook" / "PeerBERT_outputs" / "outputs" / "models"
    prev.mkdir(parents=True)
    (prev / "a.txt").write_text("old")
    work.mkdir(parents=True)
    (work / "outputs" / "models").mkdir(parents=True)
    (work / "outputs" / "models" / "a.txt").write_text("newer")
    future = time.time() + 100
    os.utime(work / "outputs" / "models" / "a.txt", (future, future))
    (prev / "b.txt").write_text("only in previous session")
    rep = restore_outputs(work, tmp_path / "input")
    assert len(rep) == 1 and rep[0][1] == 1                          # only b.txt was copied
    assert (work / "outputs" / "models" / "a.txt").read_text() == "newer"
    assert (work / "outputs" / "models" / "b.txt").exists()
    assert restore_outputs(work, tmp_path / "does-not-exist") == []
    assert disk_report(work) < 0.001


def test_progress_bars_are_filtered_from_notebook_output():
    from src.notebook_utils import clean_lines
    text = ("Downloading:  10%|#         | 1.0G/10G [00:03<00:30, 300MB/s]\rDownloading: 100%|##########| 10G/10G [00:40<00:00]\n"
            "[train] epoch 1: loss=0.7\n\nLoading weights: 50%|#####| 9/18 [00:00<00:00, 2308.05it/s]\n[train] done\n")
    assert clean_lines(text) == ["[train] epoch 1: loss=0.7", "[train] done"]
    assert clean_lines("a\rb\rfinal line\n") == ["final line"]


def test_run_jobs_prints_complete_lines_only_once(tmp_path, capsys):
    from src.notebook_utils import run_jobs
    script = tmp_path / "j.py"
    script.write_text("import sys,time\nprint('first',flush=True)\nsys.stdout.write('half ');sys.stdout.flush()\n"
                      "time.sleep(0.8)\nprint('line',flush=True)\nprint('50%|#####| 1/2 [00:01<00:00, 1.0it/s]',flush=True)\n")
    run_jobs([("j", f'"{sys.executable}" {script}')], tmp_path, tmp_path / "l", n_gpus=1, poll_s=0.2)
    out = capsys.readouterr().out
    assert "[j] first" in out and "[j] half line" in out and "it/s" not in out
    assert out.count("[j] first") == 1
