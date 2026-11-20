#!/usr/bin/env python3
"""Low-resolution frame cache + warp calibration for the counterfactual study.

For every episode of the 20k run's split (train + val) this stores, read-only from
the dataset:

* ``rows_state.npy``  (R, 76) float32: observation.state at 30 Hz, hands in radians
  (exactly the GR00T adapter's 76D state: q | dq | base quat wxyz | hand | camera).
* ``rows_action.npy`` (R, 47) float32: per-frame action rows in GR00T order
  [measured body_q 29 | measured body quat 4 | hand rad 12 | camera cmd 2];
  the target chunk of anchor t is rows t+1..t+32, edge-padded, as in the loader.
* ``frames.npy`` (N, H, W, 3) uint8: ego frames of anchors t = 0, s, 2s, ...
  (default every 6th frame, 192x108), with ``frames_ep``, ``frames_row``,
  ``frames_walk`` (locomotion in the chunk), ``frames_up`` (world up in camera
  coordinates from the floor-aligned camera pose), ``frames_up_valid``.
* ``meta.json``: episodes, offsets, split membership.
* ``calibration.json``: the warp-sign check.  For pairs of cached frames 0.2 s apart
  where the pelvis yaw changes 2-10 deg and waist/neck yaw do not move, it finds
  the scene-rotation angle that best maps frame t onto frame t+s.  With consistent
  conventions that angle is minus the IMU yaw change (slope -1), giving
  ``label_sign = +1``; slope +1 gives -1; anything else gives 0 and the training
  jobs refuse to use rotation labels.

CPU only.  Run from the Isaac-GR00T root with the GR00T venv.
"""

from __future__ import annotations

import argparse
import json
import math
from multiprocessing import get_context
from pathlib import Path
import sys
import time

import cv2
import numpy as np


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import halab_rotation_cf as rcf  # noqa: E402

ROOT = Path("/coc/flash2/jpark984/gr00t_baseline")
sys.path.insert(0, str(ROOT))
from data_loader.ha.hand_action import halab_hand_action_to_radians  # noqa: E402
from data_loader.ha.loader import LeRobotEpisodeLoader  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-parent", type=Path, default=ROOT / "datasets/ha-lab-dataset/halab")
    p.add_argument("--episode-split", type=Path,
                   default=ROOT / "outputs/halab_g1_47d_openhlm_b64_g256_lr1e4_v2/episode_split.json")
    p.add_argument("--out-dir", type=Path, default=ROOT / "outputs/cf_cache")
    p.add_argument("--stride", type=int, default=6)
    p.add_argument("--width", type=int, default=192)
    p.add_argument("--height", type=int, default=108)
    p.add_argument("--walk-threshold", type=float, default=0.15)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--calib-pairs", type=int, default=600)
    return p.parse_args()


def plan_episodes(args) -> list[dict]:
    split = json.loads(args.episode_split.read_text())
    membership = {k: "train" for k in split["train_keys"]} | {k: "val" for k in split["val_keys"]}
    by_task: dict[str, dict[int, object]] = {}
    plan = []
    for key in sorted(membership):
        task, ep = key.rsplit("/", 1)
        if task not in by_task:
            loader = LeRobotEpisodeLoader(args.data_parent / task)
            by_task[task] = {e.episode_index: e for e in loader.list_episodes()}
        info = by_task[task][int(ep)]
        plan.append({
            "key": key,
            "task": task,
            "episode_index": int(ep),
            "episode_id": info.episode_id,
            "length": int(info.length),
            "split": membership[key],
            "instruction": info.tasks[0] if info.tasks else task.replace("_", " "),
        })
    row_off = frame_off = 0
    for ep in plan:
        ep["row_offset"] = row_off
        ep["anchor_rows"] = list(range(0, ep["length"] - 1, args.stride))
        ep["frame_offset"] = frame_off
        ep["n_frames"] = len(ep["anchor_rows"])
        row_off += ep["length"]
        frame_off += ep["n_frames"]
    return plan


def _worker(job) -> dict:
    ep, args_d = job
    import pyarrow.parquet as pq

    root = Path(args_d["data_parent"]) / ep["task"]
    idx = ep["episode_index"]
    length = ep["length"]
    table = pq.read_table(
        root / "data" / "chunk-000" / f"episode_{idx:06d}.parquet",
        columns=["observation.state", "action.measured_state", "action.hand_action", "action.camera_action"],
    )
    obs = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
    measured = np.asarray(table["action.measured_state"].to_pylist(), dtype=np.float32)
    hand = np.asarray(table["action.hand_action"].to_pylist(), dtype=np.float32)[:, :12]
    cam = np.asarray(table["action.camera_action"].to_pylist(), dtype=np.float32)[:, :2]
    if len(obs) < length:
        raise RuntimeError(f"{ep['key']}: parquet has {len(obs)} rows < length {length}")
    obs, measured, hand, cam = obs[:length], measured[:length], hand[:length], cam[:length]
    state = np.concatenate(
        [obs[:, :58], obs[:, 58:62], halab_hand_action_to_radians(obs[:, 62:74]), obs[:, 74:76]], axis=1
    ).astype(np.float32)
    action = np.concatenate(
        [measured[:, :29], measured[:, 58:62], halab_hand_action_to_radians(hand), cam], axis=1
    ).astype(np.float32)

    extras = pq.read_table(
        root / "aligned_extras" / "chunk-000" / f"episode_{idx:06d}.parquet",
        columns=["frame_index", "bev_camera_pose_world"],
    )
    frame_index = np.asarray(extras["frame_index"].to_numpy(zero_copy_only=False), dtype=np.int64)[:length]
    poses = extras["bev_camera_pose_world"].to_pylist()[:length]

    rows = np.asarray(ep["anchor_rows"], dtype=np.int64)
    up = np.zeros((len(rows), 3), dtype=np.float32)
    up_valid = np.zeros(len(rows), dtype=np.uint8)
    walk = np.zeros(len(rows), dtype=np.uint8)
    for i, r in enumerate(rows):
        pose = poses[r]
        if pose is not None and len(pose) == 16:
            u, ok = rcf.up_axis_in_camera(np.asarray(pose, dtype=np.float64))
        else:
            u, ok = rcf.FALLBACK_UP, False
        up[i], up_valid[i] = u, int(ok)
        chunk_rows = np.clip(np.arange(r + 1, r + 33), 0, length - 1)
        walk[i] = int(rcf.is_walking(action[chunk_rows, :29], args_d["walk_threshold"]))

    w, h = args_d["width"], args_d["height"]
    frames = np.zeros((len(rows), h, w, 3), dtype=np.uint8)
    want = {}
    for i, r in enumerate(rows):
        want.setdefault(int(frame_index[r]), []).append(i)
    video = root / "videos" / "chunk-000" / "observation.images.left" / f"episode_{idx:06d}.mp4"
    cap = cv2.VideoCapture(str(video))
    last_needed = max(want) if want else -1
    filled = np.zeros(len(rows), dtype=bool)
    pos = 0
    while pos <= last_needed:
        if not cap.grab():
            break
        if pos in want:
            ok, bgr = cap.retrieve()
            if ok:
                small = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)
                rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                for i in want[pos]:
                    frames[i] = rgb
                    filled[i] = True
        pos += 1
    cap.release()
    missing = int((~filled).sum())
    if missing and filled.any():
        # Rare decode gaps: copy the nearest earlier decoded frame (or the first one).
        good = np.where(filled)[0]
        for i in np.where(~filled)[0]:
            earlier = good[good < i]
            frames[i] = frames[earlier[-1] if len(earlier) else good[0]]

    out_dir = Path(args_d["out_dir"])
    rs = np.load(out_dir / "rows_state.npy", mmap_mode="r+")
    ra = np.load(out_dir / "rows_action.npy", mmap_mode="r+")
    rs[ep["row_offset"] : ep["row_offset"] + length] = state
    ra[ep["row_offset"] : ep["row_offset"] + length] = action
    rs.flush(), ra.flush()
    fr = np.load(out_dir / "frames.npy", mmap_mode="r+")
    fr[ep["frame_offset"] : ep["frame_offset"] + len(rows)] = frames
    fr.flush()
    sl = slice(ep["frame_offset"], ep["frame_offset"] + len(rows))
    for name, arr in (("frames_up", up), ("frames_up_valid", up_valid), ("frames_walk", walk)):
        m = np.load(out_dir / f"{name}.npy", mmap_mode="r+")
        m[sl] = arr
        m.flush()
    return {"key": ep["key"], "missing_frames": int(missing), "walk_frac": float(walk.mean()),
            "up_valid_frac": float(up_valid.mean())}


def calibrate(out_dir: Path, plan: list[dict], stride: int, n_pairs: int) -> dict:
    """Fit the scene-rotation angle that maps cached frame t to t+stride on turning pairs."""

    frames = np.load(out_dir / "frames.npy", mmap_mode="r")
    up = np.load(out_dir / "frames_up.npy")
    up_valid = np.load(out_dir / "frames_up_valid.npy")
    rows_state = np.load(out_dir / "rows_state.npy", mmap_mode="r")
    # Neck (dynamixel) state units are not documented: treat values beyond 2*pi as degrees.
    neck_to_rad = math.pi / 180.0 if float(np.nanmax(np.abs(rows_state[::97, 74:76]))) > 6.5 else 1.0
    cands = []
    for e_i, ep in enumerate(plan):
        base = ep["row_offset"]
        for j in range(ep["n_frames"] - 1):
            r0, r1 = ep["anchor_rows"][j], ep["anchor_rows"][j + 1]
            s0, s1 = rows_state[base + r0], rows_state[base + r1]
            dyaw = float(rcf.wrap_angle(rcf.quat_yaw(s1[58:62]) - rcf.quat_yaw(s0[58:62])))
            dwaist = abs(float(s1[12] - s0[12]))
            dneck = float(np.max(np.abs(s1[74:76] - s0[74:76]))) * neck_to_rad
            if math.radians(2) <= abs(dyaw) <= math.radians(10) and dwaist < 0.02 and dneck < 0.02:
                cands.append((ep["frame_offset"] + j, dyaw))
    if not cands:
        return {"label_sign": 0, "n_pairs": 0, "reason": "no turning pairs found"}
    pick = np.linspace(0, len(cands) - 1, min(n_pairs, len(cands))).round().astype(int)
    grid = np.radians(np.arange(-12.0, 12.0001, 0.25))
    h, w = frames.shape[1:3]
    my, mx = h // 6, w // 6

    def best_angle(f0, f1, u):
        g0 = cv2.cvtColor(f0, cv2.COLOR_RGB2GRAY).astype(np.float32)
        g1 = cv2.cvtColor(f1, cv2.COLOR_RGB2GRAY).astype(np.float32)
        errs = []
        for a in grid:
            warped = cv2.warpPerspective(g0, rcf.scene_rotation_homography(w, h, u, float(a)), (w, h),
                                         flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            errs.append(float(np.abs(warped[my:-my, mx:-mx] - g1[my:-my, mx:-mx]).mean()))
        k = int(np.argmin(errs))
        return float(grid[k]), k

    rows = []
    for i in pick:
        f, dyaw = cands[i]
        a_pose, k_pose = best_angle(frames[f], frames[f + 1], up[f].astype(np.float64))
        a_fb, k_fb = best_angle(frames[f], frames[f + 1], rcf.FALLBACK_UP)
        rows.append((dyaw, a_pose, k_pose, a_fb, k_fb, int(up_valid[f])))
    arr = np.asarray(rows, dtype=np.float64)
    interior = (arr[:, 2] > 0) & (arr[:, 2] < len(grid) - 1)
    x, y_pose, y_fb = arr[interior, 0], arr[interior, 1], arr[interior, 3]
    slope = float((x * y_pose).sum() / max((x * x).sum(), 1e-12))
    slope_fb = float((x * y_fb).sum() / max((x * x).sum(), 1e-12))
    resid = float(np.degrees(np.median(np.abs(y_pose - slope * x))))
    corr = float(np.corrcoef(x, y_pose)[0, 1]) if len(x) > 2 else float("nan")
    if -1.6 < slope < -0.4 and corr < -0.5:
        sign = 1
    elif 0.4 < slope < 1.6 and corr > 0.5:
        sign = -1
    else:
        sign = 0
    pitch = np.degrees(np.arcsin(np.clip(-up[up_valid > 0][:, 2], -1, 1))) if up_valid.any() else np.array([np.nan])
    return {
        "label_sign": sign,
        "slope_angle_vs_imu_yaw": slope,
        "slope_with_camera_vertical_axis": slope_fb,
        "corr": corr,
        "median_abs_residual_deg": resid,
        "n_pairs": int(interior.sum()),
        "n_candidates": len(cands),
        "up_valid_fraction": float(up_valid.mean()),
        "camera_pitch_deg_p10_p50_p90": [float(np.percentile(pitch, q)) for q in (10, 50, 90)],
        "rule": "scene-rotation angle mapping frame t to t+stride vs pelvis yaw change; slope ~ -1 => label_sign +1",
    }


def main() -> None:
    args = parse_args()
    t0 = time.time()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plan = plan_episodes(args)
    n_rows = sum(ep["length"] for ep in plan)
    n_frames = sum(ep["n_frames"] for ep in plan)
    print(f"{len(plan)} episodes, {n_rows} rows, {n_frames} cached frames "
          f"({n_frames * args.height * args.width * 3 / 1e9:.1f} GB)", flush=True)
    fmt = np.lib.format
    fmt.open_memmap(args.out_dir / "rows_state.npy", mode="w+", dtype=np.float32, shape=(n_rows, 76)).flush()
    fmt.open_memmap(args.out_dir / "rows_action.npy", mode="w+", dtype=np.float32, shape=(n_rows, 47)).flush()
    fmt.open_memmap(args.out_dir / "frames.npy", mode="w+", dtype=np.uint8,
                    shape=(n_frames, args.height, args.width, 3)).flush()
    fmt.open_memmap(args.out_dir / "frames_up.npy", mode="w+", dtype=np.float32, shape=(n_frames, 3)).flush()
    fmt.open_memmap(args.out_dir / "frames_up_valid.npy", mode="w+", dtype=np.uint8, shape=(n_frames,)).flush()
    fmt.open_memmap(args.out_dir / "frames_walk.npy", mode="w+", dtype=np.uint8, shape=(n_frames,)).flush()
    frames_ep = np.concatenate([np.full(ep["n_frames"], i, dtype=np.int32) for i, ep in enumerate(plan)])
    frames_row = np.concatenate([np.asarray(ep["anchor_rows"], dtype=np.int32) for ep in plan])
    np.save(args.out_dir / "frames_ep.npy", frames_ep)
    np.save(args.out_dir / "frames_row.npy", frames_row)

    args_d = {"data_parent": str(args.data_parent), "out_dir": str(args.out_dir), "width": args.width,
              "height": args.height, "walk_threshold": args.walk_threshold}
    reports = []
    with get_context("spawn").Pool(args.workers) as pool:
        for i, rep in enumerate(pool.imap_unordered(_worker, [(ep, args_d) for ep in plan])):
            reports.append(rep)
            if (i + 1) % 25 == 0 or i + 1 == len(plan):
                print(f"cached {i + 1}/{len(plan)} episodes ({time.time() - t0:.0f}s)", flush=True)
    missing = sum(r["missing_frames"] for r in reports)
    meta = {
        "stride": args.stride,
        "width": args.width,
        "height": args.height,
        "walk_threshold": args.walk_threshold,
        "n_rows": n_rows,
        "n_frames": n_frames,
        "missing_frames": missing,
        "episode_split": str(args.episode_split),
        "episodes": [{k: v for k, v in ep.items() if k != "anchor_rows"} for ep in plan],
        "reports": sorted(reports, key=lambda r: r["key"]),
    }
    (args.out_dir / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(f"cache written: missing frames {missing}; walk fraction "
          f"{np.mean([r['walk_frac'] for r in reports]):.3f}", flush=True)
    calib = calibrate(args.out_dir, plan, args.stride, args.calib_pairs)
    (args.out_dir / "calibration.json").write_text(json.dumps(calib, indent=2) + "\n")
    print("calibration:", json.dumps(calib, indent=2), flush=True)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
