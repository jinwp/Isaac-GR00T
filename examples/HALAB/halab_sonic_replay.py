#!/usr/bin/env python3
"""Replay ``halab_kinematic_play.py`` trajectories through the SONIC physics sim.

OpenHLM ``run_ha_ckpt_viz.py --sonic-physics-sim`` (``run_sonic_standalone``) without
its policy / config imports: every ``pred_actions_abs.npy`` / ``gt_actions_abs.npy``
(``(T, 47)`` MotionTokenizer order) under ``--root`` is streamed as a 47-D pose stream
to samsung_deploy's SONIC controller in a headless MuJoCo physics sim, through
OpenHLM's ``openpi.ha.viz.sonic_runner.run_sonic_physics_sim``.  samsung_deploy runs in
its own processes and venv (``.venv_teleop``); this process only orchestrates.

Per trajectory, under ``<episode>/physics_sim/<pred|gt>/``: ``physics_sim_overlay.mp4``
(commanded target vs what the simulated robot did), ``physics_sim_rollout.mp4`` and
``result.json`` (tracking metrics, warnings).

Needs a GPU node (SONIC runs on TensorRT).  TensorRT engines are cached inside the
samsung_deploy checkout per GPU architecture; see the README of this step.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", required=True, help="kinematic-play output dir (any level: run, split, task or episode)")
    parser.add_argument("--which", choices=("pred", "gt", "both"), default="both")
    parser.add_argument(
        "--top-only", action="store_true",
        help="only trajectories directly in --root (skip stride*/state_feedback* subfolders)",
    )
    parser.add_argument(
        "--out-root",
        help="write results under this dir instead of <episode>/physics_sim/: "
        "<out-root>/<task>/<episode>/{gt, anchored_pred, state_feedback*_pred}/ (mirrors --root; "
        "gt runs once per episode, from the anchored trajectory)",
    )
    parser.add_argument("--skip-existing", action="store_true", help="skip runs whose result.json already exists")
    parser.add_argument("--openhlm", default="/coc/flash2/dscho1234/Workspace/OpenHLM")
    # Your own copy: SONIC writes TensorRT engines and logs under this root.
    parser.add_argument("--samsung-deploy-root", default="/coc/flash2/jpark984/samsung_deploy")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--max-seconds", type=float, default=0, help="replay only the first N seconds (0 = all)")
    parser.add_argument("--settle-on-s", type=float, default=1.0)
    parser.add_argument("--settle-off-s", type=float, default=2.0)
    args = parser.parse_args()

    openhlm = Path(args.openhlm)
    for extra in (openhlm / "src" / "openpi4OpenHLM" / "src", openhlm, openhlm / "thirdparty" / "MotionTokenizers" / "src"):
        if str(extra) not in sys.path:
            sys.path.insert(0, str(extra))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    from openpi.ha.viz import sonic_runner
    from openpi.ha.viz.sonic_runner import SonicRunConfig, run_sonic_physics_sim

    import halab_sonic_compat

    # The overlay render runs in an isolated child that re-imports the target by name:
    # point sonic_runner at our kwarg-filtering wrapper (see halab_sonic_compat).
    sonic_runner.render_physics_sim_video_from_g1_debug = halab_sonic_compat.render_physics_sim_video_from_g1_debug

    root = Path(args.samsung_deploy_root).resolve()
    if "dscho1234" in str(root):
        raise SystemExit(f"refusing to run SONIC inside {root}: use your own samsung_deploy copy")
    if not (root / ".venv_teleop" / "bin" / "python").exists():
        raise SystemExit(f"{root}/.venv_teleop is missing: create it first")
    cfg = SonicRunConfig(samsung_deploy_root=root)
    names = {"pred": ["pred_actions_abs.npy"], "gt": ["gt_actions_abs.npy"]}.get(
        args.which, ["pred_actions_abs.npy", "gt_actions_abs.npy"]
    )
    in_root = Path(args.root)
    search = in_root.glob if args.top_only else in_root.rglob
    files = sorted(p for name in names for p in search(name) if "physics_sim" not in p.parts)
    if not files:
        raise SystemExit(f"no {names} under {args.root}")
    for npy in files:
        which = "pred" if npy.name.startswith("pred") else "gt"
        mode_dir = npy.parent
        if args.out_root:
            # anchored (replan) rows sit in episode_XXXXXX/, state feedback in episode_XXXXXX/state_feedback*/;
            # other variants (strideN/, stepsN/, qnorm/) are not part of this layout.
            if mode_dir.name.startswith("episode_"):
                ep_dir, mode = mode_dir, "anchored"
            elif mode_dir.name.startswith("state_feedback") and mode_dir.parent.name.startswith("episode_"):
                ep_dir, mode = mode_dir.parent, mode_dir.name
            else:
                continue
            if which == "gt" and mode != "anchored":
                continue  # same recording as the anchored gt
            out_dir = Path(args.out_root) / ep_dir.relative_to(in_root) / ("gt" if which == "gt" else f"{mode}_pred")
        else:
            out_dir = mode_dir / "physics_sim" / which
        if args.skip_existing and (out_dir / "result.json").exists():
            print(f"[sonic] skip (done): {out_dir}", flush=True)
            continue
        actions = np.load(npy).astype(np.float32)
        if actions.shape[1] != 47:
            print(f"skip {npy}: expected (T, 47), got {actions.shape}", flush=True)
            continue
        if args.max_seconds:
            actions = actions[: int(args.max_seconds * args.fps)]
        motion_name = f"{mode_dir.parent.name}_{mode_dir.name}_{which}"[:120]
        print(f"[sonic] {npy} {actions.shape} -> {out_dir}", flush=True)
        result = run_sonic_physics_sim(
            actions, out_dir=out_dir, motion_name=motion_name, fps=args.fps, cfg=cfg,
            settle_on_band_seconds=args.settle_on_s, settle_off_band_seconds=args.settle_off_s,
        )
        for key in ("warning", "video_error", "rollout_video_error"):
            if result.get(key):
                print(f"[sonic] {which}: {key}: {result[key]}", flush=True)
        (out_dir / "result.json").write_text(json.dumps(result, indent=2, default=str))
        print(f"[sonic] {which}: video={result.get('video_path')} rollout_video={result.get('rollout_video_path')}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
