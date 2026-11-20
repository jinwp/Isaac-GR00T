#!/usr/bin/env python3
"""Experiment 2 task queue: one task per GPU, resumable across Slurm legs.

Tasks (all write into /coc/flash2/jpark984/gr00t_baseline/outputs/cf_exp2):

* ``probes/``  GR00T gain probes of existing checkpoints (20k run (v2), 30k run
  (recipe), v3_sd02, v3_sd02_cj; the 20k run again at 16 denoising steps) and of
  the Experiment 1 arms once their checkpoint-2000 exists.
* ``runs/snr_*``  signal-to-noise law sweep (my report, Idea 1): rotation spread,
  matched-exposure and fixed-step dilution, perception blur, gauge-only.
* ``runs/ret_*``  retention arms (ChatGPT report, Candidate 1, arms A-F plus the
  stronger H/H2 controls), continued from a common old model per seed.

A task is complete when its results file exists.  On the deadline the runner sends
SIGTERM (training saves and exits 3), then exits 3 so the sbatch resubmits.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


ROOT = Path("/coc/flash2/jpark984/gr00t_baseline")
REPO = ROOT / "Isaac-GR00T"
OUT = ROOT / "outputs/cf_exp2"
HERE = Path(__file__).resolve().parent
TRAIN = HERE / "train_small.py"
PROBE = HERE.parent / "probe_gr00t_gain.py"
SEEDS = (11, 22, 33)


@dataclass
class Task:
    name: str
    out_dir: Path
    argv: list[str]
    done_file: Path
    priority: int
    deps: list[str] = field(default_factory=list)
    requires: Path | None = None


def latest_checkpoint(run: Path) -> Path | None:
    best, best_step = None, -1
    for path in run.glob("checkpoint-*"):
        tail = path.name[len("checkpoint-"):]
        if tail.isdigit() and (path / "trainer_state.json").is_file() and int(tail) > best_step:
            best, best_step = path, int(tail)
    return best


def build_tasks() -> list[Task]:
    tasks: list[Task] = []
    py = sys.executable
    outputs = ROOT / "outputs"

    probes = [
        ("v2_20k", outputs / "halab_g1_47d_openhlm_b64_g256_lr1e4_v2/checkpoint-20000", 0.0, 0),
        ("v2_20k_steps16", outputs / "halab_g1_47d_openhlm_b64_g256_lr1e4_v2/checkpoint-20000", 0.0, 16),
        ("recipe_30k", outputs / "halab_g1_47d_openhlm_recipe_b64_g256_lr5e5_const_30k/checkpoint-30000", 0.0, 0),
        ("v3_sd02", latest_checkpoint(outputs / "halab_g1_47d_v3_sd02_b64_g256_lr1e4"), 0.0, 0),
        ("v3_sd02_cj", latest_checkpoint(outputs / "halab_g1_47d_v3_sd02_cj_b64_g256_lr1e4"), 0.0, 0),
        ("cf_a_control", outputs / "halab_cf_a_control_lr2e5_2k/checkpoint-2000", 0.08, 0),
        ("cf_b_rot", outputs / "halab_cf_b_rot_f050_s4_lr2e5_2k/checkpoint-2000", 0.08, 0),
        ("cf_c_dilute", outputs / "halab_cf_c_rot_f0125_s4_lr2e5_2k/checkpoint-2000", 0.08, 0),
    ]
    for i, (name, ckpt, crop, steps) in enumerate(probes):
        if ckpt is None:
            print(f"[queue] skip probe {name}: no complete checkpoint yet", flush=True)
            continue
        out = OUT / "probes" / name
        argv = [py, "-u", str(PROBE), "--checkpoint", str(ckpt), "--output", str(out / "probe.json"),
                "--crop-frac", str(crop)]
        if steps:
            argv += ["--denoising-steps", str(steps)]
        tasks.append(Task(f"probe_{name}", out, argv, out / "probe.json", priority=i,
                          requires=ckpt / "trainer_state.json"))

    snr_arms = {
        "snr_s0": ["--sigma-deg", "0"],
        "snr_s2": ["--sigma-deg", "2"],
        "snr_s4": ["--sigma-deg", "4"],
        "snr_s6": ["--sigma-deg", "6"],
        "snr_s4_f050_matched": ["--sigma-deg", "4", "--frac", "0.5", "--steps", "24000"],
        "snr_s4_f025_matched": ["--sigma-deg", "4", "--frac", "0.25", "--steps", "48000"],
        "snr_s4_f050_fixed": ["--sigma-deg", "4", "--frac", "0.5"],
        "snr_s4_f025_fixed": ["--sigma-deg", "4", "--frac", "0.25"],
        "snr_s4_blur2": ["--sigma-deg", "4", "--blur", "2"],
        "snr_s4_blur4": ["--sigma-deg", "4", "--blur", "4"],
        "snr_s0_gauge45": ["--sigma-deg", "0", "--gauge-deg", "45"],
    }
    for arm, extra in snr_arms.items():
        for seed in SEEDS:
            name = f"{arm}_s{seed}"
            out = OUT / "runs" / name
            argv = [py, "-u", str(TRAIN), "--run-dir", str(out), "--mode", "mix", "--seed", str(seed),
                    "--steps", "12000", "--lr", "3e-4", "--warmup", "500"] + extra
            tasks.append(Task(name, out, argv, out / "results.json", priority=200))

    cont = ["--steps", "5000", "--lr", "1e-4", "--warmup", "100", "--sigma-deg", "4", "--mode", "fixed",
            "--stream-offset", "100000"]
    ret_arms = {
        "ret_A": ["--n-r", "32", "--n-o", "96"],
        "ret_B": ["--n-r", "16", "--n-o", "48", "--n-e", "64"],
        "ret_C": ["--n-r", "32", "--n-o", "48", "--n-e", "48"],
        "ret_D": ["--n-r", "32", "--n-o", "48", "--n-e", "48", "--distill-nominal", "1.0"],
        "ret_E": ["--n-r", "32", "--n-o", "48", "--n-e", "48", "--distill-endpoints", "1.0"],
        "ret_F": ["--n-r", "32", "--n-o", "48", "--n-e", "48", "--distill-response", "1.0"],
        "ret_H": ["--n-r", "32", "--n-o", "96", "--n-xe", "48"],
        "ret_H2": ["--n-r", "32", "--n-o", "96", "--n-xo", "48"],
    }
    for seed in SEEDS:
        old = f"ret_old_s{seed}"
        old_dir = OUT / "runs" / old
        tasks.append(Task(old, old_dir,
                          [py, "-u", str(TRAIN), "--run-dir", str(old_dir), "--mode", "fixed", "--seed", str(seed),
                           "--steps", "10000", "--lr", "3e-4", "--warmup", "500", "--sigma-deg", "4",
                           "--n-r", "32", "--n-o", "96"],
                          old_dir / "results.json", priority=100))
        for arm, extra in ret_arms.items():
            name = f"{arm}_s{seed}"
            out = OUT / "runs" / name
            argv = [py, "-u", str(TRAIN), "--run-dir", str(out), "--seed", str(seed),
                    "--init-from", str(old_dir)] + cont + extra
            tasks.append(Task(name, out, argv, out / "results.json", priority=150, deps=[old]))
    return tasks


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--deadline-seconds", type=float, default=13500)
    ap.add_argument("--gpus", type=int, default=0, help="0 = count CUDA_VISIBLE_DEVICES")
    ap.add_argument("--min-start-seconds", type=float, default=420)
    ap.add_argument("--max-failures", type=int, default=2)
    args = ap.parse_args()

    t0 = time.time()
    visible = [g for g in os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3").split(",") if g.strip()]
    gpus = visible[: args.gpus] if args.gpus else visible
    tasks = build_tasks()
    by_name = {t.name: t for t in tasks}
    OUT.mkdir(parents=True, exist_ok=True)

    def done(t: Task) -> bool:
        return t.done_file.is_file()

    def failures(t: Task) -> int:
        return len(list(t.out_dir.glob("FAILED_*"))) if t.out_dir.is_dir() else 0

    def runnable(t: Task) -> bool:
        if done(t) or failures(t) >= args.max_failures:
            return False
        if t.requires is not None and not t.requires.is_file():
            return False
        return all(done(by_name[d]) for d in t.deps)

    def dead(t: Task) -> bool:
        """Failed too often, or depends on a task that did."""
        if failures(t) >= args.max_failures:
            return True
        return any(dead(by_name[d]) for d in t.deps)

    running: dict[str, tuple[Task, subprocess.Popen, float]] = {}
    started_names: set[str] = set()
    completed_now = 0
    stopping = False
    while True:
        for gpu in list(running):
            task, proc, start = running[gpu]
            code = proc.poll()
            if code is None:
                continue
            del running[gpu]
            if code == 0 and done(task):
                completed_now += 1
                print(f"[queue] done {task.name} on GPU {gpu} ({time.time() - start:.0f}s)", flush=True)
            elif code == 3:
                print(f"[queue] {task.name} interrupted (resumes next leg)", flush=True)
            else:
                task.out_dir.mkdir(parents=True, exist_ok=True)
                (task.out_dir / f"FAILED_{int(time.time())}").write_text(f"exit code {code}\n")
                print(f"[queue] FAILED {task.name} (exit {code}); see {task.out_dir}/stdout.log", flush=True)

        remaining = args.deadline_seconds - (time.time() - t0)
        if remaining < 300 and not stopping:
            stopping = True
            print("[queue] deadline: stopping running tasks", flush=True)
            for task, proc, _ in running.values():
                proc.send_signal(signal.SIGTERM)
        if stopping:
            if not running:
                break
            if remaining < 60:
                for task, proc, _ in running.values():
                    proc.kill()
                break
            time.sleep(5)
            continue

        pending = [t for t in tasks if not done(t) and failures(t) < args.max_failures]
        free = [g for g in gpus if g not in running]
        busy = {t.name for t, _, _ in running.values()}
        candidates = sorted((t for t in pending if t.name not in busy and runnable(t)), key=lambda t: t.priority)
        if remaining > args.min_start_seconds:
            for gpu, task in zip(free, candidates):
                task.out_dir.mkdir(parents=True, exist_ok=True)
                log = open(task.out_dir / "stdout.log", "a")
                log.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} job {os.environ.get('SLURM_JOB_ID', '?')} "
                          f"GPU {gpu}\n{' '.join(task.argv)}\n")
                log.flush()
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, PYTHONUNBUFFERED="1")
                proc = subprocess.Popen(task.argv, cwd=str(REPO), env=env, stdout=log, stderr=subprocess.STDOUT)
                running[gpu] = (task, proc, time.time())
                started_names.add(task.name)
                print(f"[queue] start {task.name} on GPU {gpu}", flush=True)
        if not running and not candidates:
            print(f"[queue] nothing runnable; {len(pending)} pending (waiting on checkpoints, "
                  f"failed dependencies or the deadline)", flush=True)
            break
        time.sleep(15)

    status = {t.name: ("done" if done(t) else f"failed x{failures(t)}" if failures(t) else "pending")
              for t in tasks}
    (OUT / "queue_status.json").write_text(json.dumps(status, indent=1) + "\n")
    n_done = sum(v == "done" for v in status.values())
    print(f"[queue] {n_done}/{len(tasks)} tasks done; {completed_now} completed this leg", flush=True)
    (OUT / "last_leg_completed.txt").write_text(f"{completed_now}\n")
    resumable = [t for t in tasks if not done(t) and not dead(t)
                 and (t.requires is None or t.requires.is_file())]
    return 3 if resumable else 0


if __name__ == "__main__":
    sys.exit(main())
