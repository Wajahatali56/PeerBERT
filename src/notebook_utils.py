"""
Helpers for notebooks/PeerBERT_full_experiments.ipynb (kept here so they are unit-tested).

- run_cmd:       run one pipeline script, stream its output, raise on failure.
- run_jobs:      run several scripts at once, one queue per GPU (Kaggle has 2 x T4: IntelLabs on GPU 0,
                 Sem-Detect on GPU 1). Each process sees exactly one GPU.
- Budget:        wall-clock guard so a session never starts a step it cannot finish.
- restore_outputs: copy the outputs of an earlier session (a Kaggle notebook version added as input).
- disk_report:   size of the output folder against Kaggle's 20 GB limit.
"""

import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

_PROGRESS = re.compile(r"\d+%\||it/s\]|s/it\]")       # tqdm / download progress bars


def clean_lines(text):
    """Split output into lines, keep the last carriage-return segment, drop progress-bar updates and blanks."""
    out = []
    for raw in text.split("\n"):
        line = raw.split("\r")[-1].rstrip()
        if line and not _PROGRESS.search(line):
            out.append(line)
    return out


# ---------------------------------------------------------------- running scripts
def _tail(path, n=40):
    try:
        lines = Path(path).read_text(errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(lines[-n:])


def run_cmd(cmd, cwd, env=None):
    """Run a shell command, echoing its output as it arrives. Raises RuntimeError on a non-zero exit."""
    print("$", cmd, flush=True)
    proc = subprocess.Popen(cmd, shell=True, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    last = []
    for chunk in proc.stdout:
        for line in clean_lines(chunk):
            print(line, flush=True)
            last = (last + [line])[-40:]
    if proc.wait() != 0:
        raise RuntimeError(f"step failed (exit code {proc.returncode}): {cmd}\n" + "\n".join(last[-15:]))


class _Job:
    def __init__(self, label, cmd, gpu, log_path):
        self.label, self.cmd, self.gpu, self.log = label, cmd, gpu, Path(log_path)
        self.proc = None
        self.offset = 0
        self.buf = ""
        self.code = None

    def start(self, cwd, env):
        self.log.parent.mkdir(parents=True, exist_ok=True)
        self.fh = open(self.log, "w")
        self.proc = subprocess.Popen(self.cmd, shell=True, cwd=cwd, env=env, stdout=self.fh,
                                     stderr=subprocess.STDOUT, text=True)

    def new_lines(self, final=False):
        """New complete lines of the log (progress bars filtered); a partial last line waits for its newline."""
        with open(self.log, "rb") as f:
            f.seek(self.offset)
            data = f.read()
        self.offset += len(data)
        self.buf += data.decode("utf-8", errors="replace")
        if final:
            text, self.buf = self.buf, ""
        else:
            head, sep, tail = self.buf.rpartition("\n")
            text, self.buf = (head, tail) if sep else ("", self.buf)
        return clean_lines(text)


def run_jobs(jobs, cwd, log_dir, n_gpus=1, poll_s=5.0, env=None):
    """
    jobs: list of (label, command); the command may contain "{gpu}" and is formatted with the GPU
    index it is assigned to (job i -> GPU i % n_gpus). Jobs of the same GPU run one after another,
    different GPUs run at the same time. Output is echoed with a "[label]" prefix; raises
    RuntimeError after all jobs ended if any failed.
    """
    n_gpus = max(1, int(n_gpus))
    queues = {g: [] for g in range(n_gpus)}
    for i, (label, cmd) in enumerate(jobs):
        g = i % n_gpus
        queues[g].append(_Job(label, cmd.format(gpu=g), g, Path(log_dir) / f"{label}.log"))
    env = dict(os.environ if env is None else env)
    running, finished = {}, []
    for g, q in queues.items():
        if q:
            job = q.pop(0)
            job.start(cwd, env)
            running[g] = job
            print(f"[{job.label}] started on GPU {g}: {job.cmd}", flush=True)
    while running:
        time.sleep(poll_s)
        for g, job in list(running.items()):
            for line in job.new_lines():
                print(f"[{job.label}] {line}", flush=True)
            code = job.proc.poll()
            if code is None:
                continue
            for line in job.new_lines(final=True):
                print(f"[{job.label}] {line}", flush=True)
            job.code = code
            job.fh.close()
            finished.append(job)
            print(f"[{job.label}] finished with exit code {code}", flush=True)
            del running[g]
            if queues[g]:
                nxt = queues[g].pop(0)
                nxt.start(cwd, env)
                running[g] = nxt
                print(f"[{nxt.label}] started on GPU {g}: {nxt.cmd}", flush=True)
    failed = [j for j in finished if j.code != 0]
    if failed:
        raise RuntimeError("failed jobs: " + ", ".join(j.label for j in failed) + "\n\n" + "\n\n".join(
            f"--- {j.label} (last lines of {j.log}) ---\n{_tail(j.log)}" for j in failed))


def gpu_count():
    """Number of visible CUDA devices without creating a CUDA context in the notebook process."""
    try:
        out = subprocess.run("nvidia-smi -L", shell=True, capture_output=True, text=True).stdout
        return max(1, sum(1 for line in out.splitlines() if line.startswith("GPU ")))
    except Exception:                                                 # noqa: BLE001
        return 1


# ---------------------------------------------------------------- time and disk
class Budget:
    """Soft wall-clock budget: skip a step whose estimate would overrun the session."""

    def __init__(self, hours, clock=time.time):
        self.hours, self.clock, self.t0 = float(hours), clock, clock()

    def elapsed_h(self):
        return (self.clock() - self.t0) / 3600

    def remaining_h(self):
        return self.hours - self.elapsed_h()

    def allow(self, what, estimate_h):
        ok = estimate_h <= self.remaining_h()
        print(f"[budget] {what}: estimate {estimate_h:.1f} h, {self.remaining_h():.1f} h of {self.hours:.1f} h left -> "
              f"{'run' if ok else 'SKIPPED (run this step in the next session)'}")
        return ok


def dir_size_gb(path):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total / 1e9


def disk_report(path, limit_gb=20.0):
    gb = dir_size_gb(path)
    flag = "  <- close to the limit: delete old checkpoints before continuing" if gb > 0.85 * limit_gb else ""
    print(f"[disk] {path}: {gb:.2f} GB of the {limit_gb:.0f} GB output limit{flag}")
    return gb


def restore_outputs(work_dir, input_root="/kaggle/input", name="PeerBERT_outputs", max_depth=4):
    """
    Copy results of an earlier session into work_dir. In Kaggle, add the previous notebook version's
    output as an input; its files appear under /kaggle/input/<slug>/<name>. Existing newer files are kept.
    Returns [(source, files_copied)].
    """
    root = Path(input_root)
    if not root.exists():
        return []
    found = []
    for dirpath, dirs, _ in os.walk(root):
        depth = len(Path(dirpath).relative_to(root).parts)
        if depth > max_depth:
            dirs[:] = []
            continue
        if Path(dirpath).name == name:
            found.append(Path(dirpath))
            dirs[:] = []
    report = []
    for src in sorted(found):
        copied = 0

        def copy_if_newer(s, d):
            nonlocal copied
            if not os.path.exists(d) or os.path.getmtime(s) > os.path.getmtime(d):
                shutil.copy2(s, d)
                copied += 1
            return d

        shutil.copytree(src, work_dir, dirs_exist_ok=True, copy_function=copy_if_newer)
        report.append((str(src), copied))
    return report


def python_executable():
    return sys.executable
