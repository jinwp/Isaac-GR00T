#!/usr/bin/env python3
"""Score SONIC physics-sim replays of ``pred`` and ``gt`` against the recording.

``halab_sonic_replay.py`` runs the predicted and the recorded trajectory through SONIC
separately, and SONIC's own ``tracking`` only compares each run with its *own* command.
This puts both on one scale: for every episode with ``physics_sim/{gt,pred}/`` it
compares the body joints the simulated robot actually reached with the recording
(``gt_actions.npy``)::

    kinematic   |pred command  - recording|   (no physics; what kinematic play measures)
    sim(gt)     |sim of gt     - recording|   floor: SONIC tracking a perfect trajectory
    sim(pred)   |sim of pred   - recording|   the policy's trajectory, executed physically
    pred-gt     |sim of pred   - sim of gt|   policy error with SONIC's own error cancelled

plus falls (hard resets), settle failures and SONIC's commanded-vs-simulated error.
numpy only; run locally on rsynced results or on the cluster::

    python halab_sonic_compare.py --root <kinematic output dir (any level)>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

DEG = 180.0 / np.pi
GROUPS = {"legs": slice(0, 12), "waist": slice(12, 15), "L_arm": slice(15, 22), "R_arm": slice(22, 29), "body": slice(0, 29)}


def sim_body_30hz(run_dir: Path, fps: float) -> tuple[np.ndarray | None, dict]:
    """Measured body joints (T, 29) of one SONIC run, trimmed to the motion and resampled to ``fps``.

    Same trims as ``sonic_runner.run_sonic_physics_sim``: drop everything before the band
    release, then the settle-off hold, keep ``n_stream_frames`` control ticks.
    """
    result = json.loads((run_dir / "result.json").read_text())
    cap = run_dir / "g1_debug_capture.npz"
    if not cap.exists():
        return None, result
    dbg = np.load(cap)
    if "body_q_measured" not in dbg.files:  # control loop never engaged: result["error"] says why
        return None, result
    meas = np.asarray(dbg["body_q_measured"], dtype=np.float32)
    release = int(dbg["release_tick_idx"][0]) if "release_tick_idx" in dbg.files else 0
    stream_fps = float(result.get("sonic_stream_fps", 50.0))
    off = int(round(float(result.get("settle_off_band_seconds", 0.0)) * stream_fps))
    n = int(result.get("n_stream_frames", meas.shape[0]))
    meas = meas[release + off : release + off + n]
    t = np.arange(int(np.floor(len(meas) / stream_fps * fps)))
    return meas[np.minimum(np.round(t * stream_fps / fps).astype(int), len(meas) - 1)], result


def mae(a: np.ndarray, b: np.ndarray, cols: slice) -> float:
    n = min(len(a), len(b))
    return float(np.abs(a[:n, cols] - b[:n, cols]).mean() * DEG) if n else float("nan")


def fall_info(result: dict) -> str:
    if result.get("error"):
        return "NO-RUN"
    if result.get("settle_failed"):
        return "SETTLE-FAIL"
    if result.get("fell"):
        return f"FELL@{result.get('first_hard_reset_s', float('nan')):.1f}s"
    return "ok"


def compare_split_layout(kin_root: Path, sonic_root: Path, fps: float) -> int:
    """``halab_sonic_replay.py --out-root`` layout: one table per episode plus the mean over episodes.

    Kinematic rows come from ``<kin_root>/<task>/<episode>/`` (anchored) and its
    ``state_feedback/``; sims from ``<sonic_root>/<task>/<episode>/{gt, anchored_pred,
    state_feedback_pred}/``.
    """

    eps = sorted(p.parent for p in sonic_root.rglob("gt") if p.is_dir() and (p / "result.json").exists())
    if not eps:
        raise SystemExit(f"no */gt/result.json under {sonic_root}")
    labels = ["kin anchored", "kin SF", "sim(gt)", "sim anchored", "sim SF", "anchored-gt", "SF-gt"]
    totals: dict[str, list[list[float]]] = {k: [] for k in labels}
    falls: list[str] = []
    for ep in eps:
        kin = kin_root / ep.relative_to(sonic_root)
        rec = np.load(kin / "gt_actions.npy")[:, :29]  # mt47: body29 first
        cmd = {"anchored": np.load(kin / "pred_actions.npy")[:, :29]}
        if (kin / "state_feedback" / "pred_actions.npy").exists():
            cmd["SF"] = np.load(kin / "state_feedback" / "pred_actions.npy")[:, :29]
        sims = {}
        print(f"\n== {ep.relative_to(sonic_root)}")
        for name, sub in (("gt", "gt"), ("anchored", "anchored_pred"), ("SF", "state_feedback_pred")):
            if not (ep / sub / "result.json").exists():
                continue
            body, res = sim_body_30hz(ep / sub, fps)
            trk = res.get("tracking", {})
            status = fall_info(res)
            print(f"  sim {name:8}: {status:12} SONIC cmd-vs-sim {trk.get('target_vs_sim_mae_rad', float('nan')) * DEG:.2f} deg  error={res.get('error') or '-'}")
            if status != "ok":
                falls.append(f"{ep.relative_to(sonic_root)} {name}: {status}")
            if body is not None:
                sims[name] = body
        if "gt" not in sims:
            print("  sim gt missing: cannot compare joints")
            continue
        n = min(len(v) for v in sims.values())
        rows = {
            "kin anchored": (cmd["anchored"], rec),
            "kin SF": (cmd.get("SF"), rec),
            "sim(gt)": (sims["gt"], rec),
            "sim anchored": (sims.get("anchored"), rec),
            "sim SF": (sims.get("SF"), rec),
            "anchored-gt": (sims.get("anchored"), sims["gt"]),
            "SF-gt": (sims.get("SF"), sims["gt"]),
        }
        print(f"  body joint MAE (deg) over the first {n / fps:.1f}s:")
        print(f"  {'':13}" + "".join(f"{g:>8}" for g in GROUPS))
        for label, (a, b) in rows.items():
            if a is None:
                continue
            vals = [mae(a[:n], b[:n], c) for c in GROUPS.values()]
            totals[label].append(vals)
            print(f"  {label:13}" + "".join(f"{v:8.2f}" for v in vals))
    print(f"\n== MEAN over episodes ({len(eps)} episodes)")
    print(f"  {'':13}" + "".join(f"{g:>8}" for g in GROUPS) + "   n")
    for label in labels:
        if totals[label]:
            print(f"  {label:13}" + "".join(f"{v:8.2f}" for v in np.mean(totals[label], axis=0)) + f"  {len(totals[label]):2d}")
    print(f"\nfalls / settle failures: {falls or 'none'}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", help="old layout: kinematic output dir with */physics_sim/{gt,pred}")
    parser.add_argument("--kinematic-root", help="new layout: kinematic split dir, e.g. outputs/checkpoint20k_kinematic/val")
    parser.add_argument("--sonic-root", help="new layout: matching SONIC dir, e.g. outputs/checkpoint20k_sonic/val")
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()

    if args.kinematic_root or args.sonic_root:
        if not (args.kinematic_root and args.sonic_root):
            raise SystemExit("--kinematic-root and --sonic-root go together")
        return compare_split_layout(Path(args.kinematic_root), Path(args.sonic_root), args.fps)
    if not args.root:
        raise SystemExit("give --root (old layout) or --kinematic-root + --sonic-root")
    eps = sorted(p.parent for p in Path(args.root).rglob("physics_sim") if (p / "gt").is_dir() and (p / "pred").is_dir())
    if not eps:
        raise SystemExit(f"no */physics_sim/{{gt,pred}} under {args.root}")
    for ep in eps:
        rec = np.load(ep / "gt_actions.npy")[:, :29]  # mt47: body29 first
        cmd = np.load(ep / "pred_actions.npy")[:, :29]
        sim_gt, res_gt = sim_body_30hz(ep / "physics_sim" / "gt", args.fps)
        sim_pred, res_pred = sim_body_30hz(ep / "physics_sim" / "pred", args.fps)
        print(f"\n== {ep}")
        for name, res in (("gt", res_gt), ("pred", res_pred)):
            trk = res.get("tracking", {})
            print(
                f"  sim({name}): {fall_info(res)}  resets={res.get('n_hard_resets')}  "
                f"SONIC cmd-vs-sim {trk.get('target_vs_sim_mae_rad', float('nan')) * DEG:.2f} deg  "
                f"error={res.get('error') or '-'}"
            )
        if sim_gt is None or sim_pred is None:
            print("  g1_debug_capture.npz missing: cannot compare joints")
            continue
        n = min(len(sim_gt), len(sim_pred))
        rec, cmd = rec[:n], cmd[:n]
        print(f"  body joint MAE (deg) over {n / args.fps:.1f}s:")
        print(f"  {'':10}" + "".join(f"{g:>8}" for g in GROUPS))
        rows = {
            "kinematic": (cmd, rec),
            "sim(gt)": (sim_gt, rec),
            "sim(pred)": (sim_pred, rec),
            "pred-gt": (sim_pred, sim_gt),
        }
        for label, (a, b) in rows.items():
            print(f"  {label:10}" + "".join(f"{mae(a, b, c):8.2f}" for c in GROUPS.values()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
