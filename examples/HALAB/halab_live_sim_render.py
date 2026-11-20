#!/usr/bin/env python3
"""Render a live-sim run's ground-truth MuJoCo trajectory (``sim_qpos_log.npz``) to mp4.

The qpos the physics actually produced (root + every joint, logged by OpenHLM
``_sim_loop_with_qpos_log.py`` at the sim rate), drawn with the sim's own MJCF and a camera that
follows the pelvis.  No MotionTokenizers, no reference robot: what you see is the simulated G1.

    MUJOCO_GL=egl python halab_live_sim_render.py --qpos <out>/sim_qpos_log.npz \\
        --mjcf /coc/flash2/jpark984/samsung_deploy/mujoco_sim/assets/g1_inspire.xml --out <out>/sim.mp4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np


def open_writer(path: Path, fps: float, width: int, height: int):
    try:
        import imageio.v2 as imageio

        writer = imageio.get_writer(str(path), fps=fps, codec="libx264", quality=7)
        return lambda frame: writer.append_data(frame), writer.close
    except Exception:  # noqa: BLE001 - imageio / ffmpeg missing: fall back to OpenCV
        import cv2

        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        return lambda frame: writer.write(frame[:, :, ::-1]), writer.release


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--qpos", required=True)
    parser.add_argument("--mjcf", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--sim-hz", type=float, default=200.0)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--from-start", action="store_true", help="include the time on the elastic band")
    args = parser.parse_args()

    log = np.load(args.qpos)
    qpos = np.asarray(log["qpos"], dtype=np.float64)
    start = 0 if args.from_start or "release_tick_idx" not in log.files else int(log["release_tick_idx"][0])
    ticks = np.arange(start, len(qpos), args.sim_hz / args.fps).astype(int)

    model = mujoco.MjModel.from_xml_path(args.mjcf)
    if qpos.shape[1] != model.nq:
        raise SystemExit(f"qpos width {qpos.shape[1]} != model nq {model.nq}: wrong --mjcf")
    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), args.width)
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), args.height)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, args.height, args.width)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    camera.trackbodyid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    camera.distance, camera.elevation, camera.azimuth = 2.8, -15.0, 135.0

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    write, close = open_writer(out, args.fps, args.width, args.height)
    for t in ticks:
        data.qpos[:] = qpos[t]
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera)
        write(renderer.render())
    close()
    print(f"{out}: {len(ticks)} frames ({len(ticks) / args.fps:.1f} s from tick {start}, pelvis z min {qpos[start:, 2].min():.3f} m)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
