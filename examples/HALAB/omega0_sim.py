#!/usr/bin/env python3
"""SONIC physics sim of ``omega0_infer.py`` token streams, with the arm override off.

Every ``<episode>/{pred,gt}_latent92.npy`` under ``--root`` is streamed through OpenHLM
``sonic_runner.run_sonic_physics_sim`` (protocol v4 token stream, decoder-only binary,
hands, ``arm_override=False``) into ``<episode>/physics_sim/{pred,gt}/``:
``physics_sim_overlay.mp4`` (grey = dataset body, blue = simulated robot),
``physics_sim_rollout.mp4``, ``g1_debug_capture.npz`` and ``result.json``.

Then, per episode and as a mean over episodes, body-joint MAE (deg) against the recording
(``body47.npy`` = measured ``observation.state``), with ``halab_sonic_compare.py``'s trims:

    sim(gt)    |sim of the recorded token - recording|   (also no arm override: arms off in both)
    sim(pred)  |sim of the ω-0 token      - recording|
    pred-gt    |sim of the ω-0 token      - sim of the recorded token|

plus falls / settle failures.  Run from the Isaac-GR00T root with its venv (as
``halab_sonic_replay.py``), one SONIC per node (fixed ZMQ ports): wrap it in ``flock``.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

import numpy as np

RUNS = ("gt", "pred", "gt_armov", "pred_armov")


def _yaw_wxyz(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.unwrap(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def heading_metrics(run: Path, body47: np.ndarray, fps: float, settle_off_s: float, n: int) -> dict | None:
    """Heading drift and travel of one run: joint MAE ignores where the robot faces and walks.

    Simulated root from ``sim_qpos_log.npz`` (200 Hz, same release + settle-off trim as the
    joints), recorded heading from ``body47``'s pelvis quaternion; both relative to their first frame.
    """
    log = run / "sim_qpos_log.npz"
    if not log.exists():
        return None
    q = np.load(log)
    hz = 200.0
    start = int(q["release_tick_idx"][0]) + int(round(settle_off_s * hz))
    after = q["qpos"][start:]
    idx = np.round(np.arange(n) * hz / fps).astype(int)
    qpos = after[idx[idx < len(after)]]
    m = min(len(qpos), len(body47))
    if m < 2:
        return None
    sim_yaw, rec_yaw = _yaw_wxyz(qpos[:m, 3:7]), _yaw_wxyz(body47[:m, 43:47])
    err = np.degrees(np.abs(np.angle(np.exp(1j * ((sim_yaw - sim_yaw[0]) - (rec_yaw - rec_yaw[0]))))))  # wrapped to [0, 180]
    return {"yaw_err_mean_deg": float(err.mean()), "yaw_err_end_deg": float(err[-1]),
            "travel_m": float(np.linalg.norm(qpos[m - 1, :2] - qpos[0, :2]))}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", required=True, help="omega0_infer.py output dir (any level)")
    parser.add_argument("--which", choices=("pred", "gt", "both"), default="both")
    parser.add_argument("--out-root", default=None,
                        help="write runs + sim_metrics.json to <out-root>/<task>/<episode>/ instead of next to the inputs "
                             "(runs already under --root still enter the comparison)")
    parser.add_argument("--tasks", nargs="*", default=None, help="only task folders containing one of these substrings")
    parser.add_argument("--arm-override", action="store_true",
                        help="drive the arms with latent92's recorded (GMR) arm joints, as on the robot during collection; "
                             "runs go to physics_sim/<name>_armov/")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--openhlm", default="/coc/flash2/jpark984/gr00t_baseline/OpenHLM")
    parser.add_argument("--motion-tokenizers-src", default="/coc/flash2/dscho1234/Workspace/OpenHLM/thirdparty/MotionTokenizers/src",
                        help="our OpenHLM copy has an empty MotionTokenizers submodule; imported read-only for the video render")
    parser.add_argument("--samsung-deploy-root", default="/coc/flash2/jpark984/samsung_deploy")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--max-seconds", type=float, default=30.0, help="replay only the first N seconds (0 = all)")
    parser.add_argument("--settle-on-s", type=float, default=1.0)
    parser.add_argument("--settle-off-s", type=float, default=2.0)
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    openhlm = Path(args.openhlm)
    for extra in (here, openhlm / "src" / "openpi4OpenHLM" / "src", openhlm, Path(args.motion_tokenizers_src)):
        if str(extra) not in sys.path:
            sys.path.insert(0, str(extra))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    from openpi.ha.viz import sonic_runner
    from openpi.ha.viz.sonic_runner import SonicRunConfig, run_sonic_physics_sim

    import halab_sonic_compat
    from halab_sonic_compare import GROUPS, fall_info, mae, sim_body_30hz

    sonic_runner.render_physics_sim_video_from_g1_debug = halab_sonic_compat.render_physics_sim_video_from_g1_debug

    root = Path(args.samsung_deploy_root).resolve()
    if "dscho1234" in str(root):
        raise SystemExit(f"refusing to run SONIC inside {root}: use your own samsung_deploy copy")
    if not (root / ".venv_teleop" / "bin" / "python").exists():
        raise SystemExit(f"{root}/.venv_teleop is missing: create it first")
    cfg = SonicRunConfig(samsung_deploy_root=root, arm_override=args.arm_override)
    suffix = "_armov" if args.arm_override else ""

    eps = sorted(p.parent for p in Path(args.root).rglob("pred_latent92.npy") if "physics_sim" not in p.parts)
    if args.tasks:
        eps = [ep for ep in eps if any(t in ep.parent.name for t in args.tasks)]
    if not eps:
        raise SystemExit(f"no pred_latent92.npy under {args.root} (tasks {args.tasks})")
    which = ("pred", "gt") if args.which == "both" else (args.which,)
    totals: dict[str, list[list[float]]] = {}
    falls = []
    for ep in eps:
        body47 = np.load(ep / "body47.npy")
        out_ep = Path(args.out_root) / ep.relative_to(args.root) if args.out_root else ep
        for name in which:
            out_dir = out_ep / "physics_sim" / f"{name}{suffix}"
            if args.skip_existing and (out_dir / "result.json").exists():
                done = json.loads((out_dir / "result.json").read_text())
                if not done.get("error") and done.get("video_path"):
                    print(f"[sonic] skip (done): {out_dir}", flush=True)
                    continue
                print(f"[sonic] rerun (failed before): {out_dir}", flush=True)
            actions = np.load(ep / f"{name}_latent92.npy").astype(np.float32)
            body = body47
            if args.max_seconds:
                actions, body = actions[: int(args.max_seconds * args.fps)], body47[: int(args.max_seconds * args.fps)]
            print(f"[sonic] {ep} {name} {actions.shape} -> {out_dir}", flush=True)
            result = run_sonic_physics_sim(
                actions, out_dir=out_dir, motion_name=f"{ep.parent.name}_{ep.name}_omega0_{name}{suffix}"[:120], fps=args.fps,
                cfg=cfg, body47=body, settle_on_band_seconds=args.settle_on_s, settle_off_band_seconds=args.settle_off_s,
            )
            for key in ("warning", "video_error", "rollout_video_error"):
                if result.get(key):
                    print(f"[sonic] {name}: {key}: {result[key]}", flush=True)
            (out_dir / "result.json").write_text(json.dumps(result, indent=2, default=str))
            print(f"[sonic] {name}: video={result.get('video_path')} rollout_video={result.get('rollout_video_path')}", flush=True)

        sims, heads, rec = {}, {}, body47[:, :29]
        print(f"\n== {ep}")
        for name in RUNS:
            run = next((d / "physics_sim" / name for d in (out_ep, ep) if (d / "physics_sim" / name / "result.json").exists()), None)
            if run is None:
                continue
            body, res = sim_body_30hz(run, args.fps)
            status = fall_info(res)
            trk = res.get("tracking", {})
            print(f"  sim {name:10}: {status:12} SONIC cmd-vs-sim {trk.get('target_vs_sim_mae_rad', float('nan')) * 57.29578:.2f} deg"
                  f"  error={res.get('error') or '-'}")
            if status != "ok":
                falls.append(f"{ep.relative_to(args.root)} {name}: {status}")
            if body is not None:
                sims[name] = body
                heads[name] = heading_metrics(run, body47, args.fps, args.settle_off_s, len(body))
        if "gt" not in sims and "gt_armov" not in sims:
            print("  no finished gt sim: cannot compare")
            continue
        n = min(len(rec), *(len(v) for v in sims.values()))
        rows = {f"sim({k})": (v, rec) for k, v in sims.items()}
        for s in ("", "_armov"):
            if f"gt{s}" in sims and f"pred{s}" in sims:
                rows[f"pred{s}-gt{s}"] = (sims[f"pred{s}"], sims[f"gt{s}"])
        table = {}
        print(f"  body joint MAE (deg) over the first {n / args.fps:.1f}s"
              f"   | yaw: |sim - recording| mean / at end (deg), sim travel from start (m)")
        print(f"  {'':16}" + "".join(f"{g:>8}" for g in GROUPS) + f"{'yaw':>8}{'yaw end':>8}{'travel':>8}")
        for label, (a, b) in rows.items():
            vals = [mae(a[:n], b[:n], c) for c in GROUPS.values()]
            h = heads.get(label[4:-1]) if label.startswith("sim(") else None
            hv = [h["yaw_err_mean_deg"], h["yaw_err_end_deg"], h["travel_m"]] if h else [float("nan")] * 3
            totals.setdefault(label, []).append(vals + hv)
            table[label] = dict(zip(list(GROUPS) + ["yaw_err_mean_deg", "yaw_err_end_deg", "travel_m"], vals + hv))
            print(f"  {label:16}" + "".join(f"{v:8.2f}" for v in vals + hv))
        out_ep.mkdir(parents=True, exist_ok=True)
        (out_ep / "sim_metrics.json").write_text(json.dumps({"seconds": n / args.fps, "body_joint_mae_deg": table}, indent=2))

    print(f"\n== MEAN over episodes")
    print(f"  {'':16}" + "".join(f"{g:>8}" for g in GROUPS) + f"{'yaw':>8}{'yaw end':>8}{'travel':>8}   n")
    for label, vals in totals.items():
        if vals:
            print(f"  {label:16}" + "".join(f"{v:8.2f}" for v in np.nanmean(vals, axis=0)) + f"  {len(vals):2d}")
    print(f"\nfalls / settle failures: {falls or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
