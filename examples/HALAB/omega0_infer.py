#!/usr/bin/env python3
"""Zero-shot ω-0 (HF ``keycharon/Omega-zero``) on HA-Lab episodes: open-loop token prediction.

For the selected episodes of the GR00T run's split, at anchors ``0, R, 2R, ...`` the
checkpoint sees the *recorded* ZED left image (cropped to 640x360 like ω-0's camera
service), the instruction and the 45-D Sonic state built from ``observation.state``
exactly as ω-0's deploy client does (``omega_real.transforms.state.sonic_state``:
29 joints, 5 Inspire openness channels per hand, yaw-removed pelvis rot6d).  It returns
25 rows of ``[64 SONIC token | L trigger | R trigger]`` (30 Hz, rows = t+1 ... t+25);
rows ``0 .. R-1`` of every chunk are stitched into one timeline.  No RTC prefix: there
is no inference latency offline, and delay 0 is inside ω-0's training range (U{0..7}).

Per episode this writes ``pred66.npy`` / ``gt_token.npy``, the ``latent92`` rows the
OpenHLM SONIC sim streams (``pred_latent92.npy`` / ``gt_latent92.npy``), the dataset
body ``body47.npy`` (MT order, for the overlay and the joint comparison) and
``metrics.json`` (token MAE vs. a hold baseline, trigger vs. recorded hand closure).

latent92 rows ``[64 token | 14 arm q | 12 hand rad | 2 camera]``:
* pred: ω-0 token; hand = every finger ``openness = 1 - trigger`` (ω-0: trigger 1 = closed,
  Inspire openness 1 = open), through the same ramp as ``action.hand_action``;
* gt: recorded ``action.latent_state``; hand = recorded ``action.hand_action``;
* both: arm q / camera from the recording.  They are NOT driven in the sim (run it with
  the arm override off); the arms then follow SONIC's own decoding of the token.

This is open-loop and teacher-forced on the inputs: the model never sees what its own
tokens would have done.

``--home-root``: the same on ω-0's own released recordings (ω-HOME ``<task>/<episode>/``
``state_action.hdf5`` + ``ego.mp4``, first ``--episodes-per-task`` episodes of every
``--tasks`` folder that has its ego video), written to ``<out>/home/``: a check of this
deploy path on in-distribution data.  There, gt hands come from the recorded triggers like
pred's, arm q = ``rt_lowstate`` arms, camera = 0.

    <OMEGA-0>/.venv/bin/python omega0_infer.py --checkpoint <Omega-zero> \\
        --external-models <external> --episode-split <run>/episode_split.json --output-dir <out>
    <OMEGA-0>/.venv/bin/python omega0_infer.py --checkpoint <Omega-zero> \\
        --external-models <external> --home-root <omega-HOME> --tasks clean_bed pick_garbage --output-dir <out>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time
from types import SimpleNamespace

import numpy as np

# observation.state (76) = [29 q | 29 dq | 4 quat wxyz | 12 Inspire openness | 2 camera].
Q, QUAT, HAND, CAM = slice(0, 29), slice(58, 62), slice(62, 74), slice(74, 76)
ARM = slice(15, 29)  # MuJoCo order: left arm 15-21, right arm 22-28

# data_loader/ha/hand_action.py (MotionTokenizers ramps): Inspire order per side
# [pinky, ring, middle, index, thumb_bend, thumb_rotate] -> native [thumb_yaw, thumb_pitch,
# index, middle, ring, pinky], angle = rest + (1 - openness) * (grip - rest).
_PERM = np.concatenate([np.array([5, 4, 3, 2, 1, 0]), np.array([5, 4, 3, 2, 1, 0]) + 6])
_REST = np.tile([1.0, 0.20, 0.26, 0.26, 0.26, 0.26], 2)
_GRIP = np.tile([1.3, 0.45, 1.30, 1.30, 1.30, 1.30], 2)


def openness_to_radians(hand12_inspire: np.ndarray) -> np.ndarray:
    native = np.take(np.asarray(hand12_inspire, np.float64), _PERM, axis=-1)
    return (_REST + (1.0 - np.clip(native, 0.0, 1.0)) * (_GRIP - _REST)).astype(np.float32)


def select_episodes(split_path: Path, split: str, tasks: list[str] | None, per_task: int, seed: int) -> list[str]:
    """``halab_kinematic_play.select_episodes``: same seed -> the same episodes as the GR00T evals."""
    keys = json.loads(split_path.read_text())[f"{split}_keys"]
    by_task: dict[str, list[str]] = {}
    for key in keys:
        by_task.setdefault(key.rsplit("/", 1)[0], []).append(key)
    rng = random.Random(seed)
    picked = []
    for task in sorted(by_task):
        if tasks and task not in tasks:
            continue
        eps = sorted(by_task[task], key=lambda k: int(k.rsplit("/", 1)[1]))
        rng.shuffle(eps)
        picked += eps[:per_task]
    return picked


def load_episode(data_parent: Path, key: str) -> dict:
    import pandas as pd

    task, index = key.rsplit("/", 1)
    root = data_parent / task
    index = int(index)
    instruction = task.replace("_", " ")
    for line in (root / "meta" / "episodes.jsonl").read_text().splitlines():
        row = json.loads(line) if line.strip() else None
        if row and int(row["episode_index"]) == index and row.get("tasks"):
            instruction = row["tasks"][0]
    df = pd.read_parquet(root / "data" / "chunk-000" / f"episode_{index:06d}.parquet",
                         columns=["observation.state", "action.latent_state", "action.hand_action", "action.camera_action"])
    col = lambda name: np.stack(df[name].to_numpy()).astype(np.float32)
    return {"task": task, "name": f"episode_{index:06d}", "instruction": instruction,
            "video": root / "videos" / "chunk-000" / "observation.images.left" / f"episode_{index:06d}.mp4",
            "obs": col("observation.state"), "token": col("action.latent_state"),
            "hand_action": col("action.hand_action"), "camera_action": col("action.camera_action")}


# ω-HOME task folder -> instruction (paper Table 1 wording, lowercased as in finetune.yaml).
HOME_INSTRUCTIONS = {
    "pick_and_place_apple": "pick an apple and place into a basket",
    "arrange_fruit_in_the_closet": "arrange an apple on a shelf",
    "put_clothes_into_bucket": "pick clothes from bed and throw into basket",
    "put_towel_into_washing_machine": "move towel from basket to washing machine",
    "wipe_table": "wipe the table",
    "mop_floor": "mop the floor",
    "pick_garbage": "pick trash from different heights into handheld bin",
    "put_apple_and_close_drawer": "pick apple from table, throw into drawer, close drawer with knee",
    "clean_bed": "clean the bed",  # real/configs/test.yaml
    "pick_clothes_from_washing_machine": "take clothes out of the washing machine",
    "retrieve_from_the_upper_fridge": "retrieve a drink from the fridge",
}


def home_episodes(home_root: Path, tasks: list[str] | None, per_task: int) -> list[Path]:
    picked = []
    for task in tasks or sorted(HOME_INSTRUCTIONS):
        eps = sorted(p.parent for p in (home_root / task).glob("*/ego.mp4") if (p.parent / "state_action.hdf5").exists())
        if not eps:
            print(f"[home] {task}: no episode with both ego.mp4 and state_action.hdf5", flush=True)
        picked += eps[:per_task]
    return picked


def load_home_episode(ep: Path) -> dict:
    """ω-HOME episode in ``load_episode``'s layout (observation.state 76 = HA-Lab order)."""
    import h5py

    with h5py.File(ep / "state_action.hdf5", "r") as f:
        token = f["sonic/token_state"][()].astype(np.float32)
        trig = np.concatenate([f["action/left_trigger"][()], f["action/right_trigger"][()]], axis=1).astype(np.float32)
        low = f["state/rt_lowstate"][()].astype(np.float32)  # [29 q | quat wxyz | gyro | acc]
        hands = np.concatenate([f["state/inspire_left_state"][()][:, :6], f["state/inspire_right_state"][()][:, :6]], axis=1)
    n = min(len(token), len(trig), len(low), len(hands))
    obs = np.zeros((n, 76), np.float32)
    obs[:, Q], obs[:, QUAT], obs[:, HAND] = low[:n, :29], low[:n, 29:33], hands[:n] / 1000.0  # Inspire 1000 = open
    trig = np.clip(trig[:n], 0.0, 1.0)
    return {"task": ep.parent.name, "name": ep.name, "instruction": HOME_INSTRUCTIONS.get(ep.parent.name, ep.parent.name.replace("_", " ")),
            "video": ep / "ego.mp4", "obs": obs, "token": token[:n],
            "hand_action": np.concatenate([np.repeat(1.0 - trig[:, :1], 6, 1), np.repeat(1.0 - trig[:, 1:], 6, 1)], axis=1),
            "camera_action": np.zeros((n, 2), np.float32)}


def anchor_frames(video: Path, anchors: set[int], crop: tuple[int, int, int, int]) -> dict[int, np.ndarray]:
    import av

    x, y, w, h = crop
    frames = {}
    with av.open(str(video)) as container:
        for i, frame in enumerate(container.decode(video=0)):
            if i > max(anchors):
                break
            if i in anchors:
                img = frame.to_ndarray(format="rgb24")
                if img.shape[0] >= y + h and img.shape[1] >= x + w:
                    img = img[y : y + h, x : x + w]
                frames[i] = np.ascontiguousarray(img)
    return frames


def sample(value: np.ndarray, seq: int) -> SimpleNamespace:
    return SimpleNamespace(value=value, valid=True, stream="halab", sequence=seq, received_ns=seq)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="HF keycharon/Omega-zero snapshot dir")
    parser.add_argument("--external-models", required=True, help="dir with t5-base/ and stats/")
    parser.add_argument("--data-parent", default="/coc/flash2/jpark984/gr00t_baseline/datasets/ha-lab-dataset/halab")
    parser.add_argument("--episode-split", help="GR00T run episode_split.json (HA-Lab episodes)")
    parser.add_argument("--home-root", help="ω-HOME download instead of HA-Lab: <task>/<episode>/{state_action.hdf5,ego.mp4}")
    parser.add_argument("--split", choices=("val", "train"), default="val")
    parser.add_argument("--tasks", nargs="*")
    parser.add_argument("--episodes-per-task", type=int, default=1)
    parser.add_argument("--episode-keys", nargs="*", help="explicit 'task/episode_index' keys")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--replan", type=int, default=8, help="rows executed per chunk (ω-0 deploy: 8)")
    parser.add_argument("--max-seconds", type=float, default=0, help="cap each episode (0 = whole episode)")
    parser.add_argument("--crop", type=int, nargs=4, default=(0, 0, 640, 360), metavar=("X", "Y", "W", "H"))
    parser.add_argument("--attn", default="sdpa", help="Qwen attention: sdpa | flash_attention_2")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    from omega.inference.model import WAMInference
    from omega_real.transforms.state import sonic_state

    if args.home_root:
        keys, split = home_episodes(Path(args.home_root), args.tasks, args.episodes_per_task), "home"
        if not keys:
            raise SystemExit(f"no ω-HOME episode with ego.mp4 under {args.home_root} for {args.tasks}")
    elif args.episode_split:
        keys = args.episode_keys or select_episodes(Path(args.episode_split), args.split, args.tasks, args.episodes_per_task, args.seed)
        split = args.split
    else:
        raise SystemExit("give --episode-split (HA-Lab) or --home-root (ω-HOME)")
    print(f"{len(keys)} episode(s): {[str(k) for k in keys]}", flush=True)
    t0 = time.time()
    policy = WAMInference.load(checkpoint=args.checkpoint, external_models=args.external_models,
                               device=args.device, attn_implementation=args.attn)
    policy.pre["idle_reset_seconds"] = float("inf")  # offline: never reset between anchors of one episode
    horizon = int(policy.cfg["action_chunk_size"])
    print(f"loaded ω-0 in {time.time() - t0:.0f}s: chunk {horizon}, action {policy.dimension}, "
          f"action norm {policy.metadata['normalization']['action']['mode']}, "
          f"state norm {policy.metadata['normalization'].get('state', {}).get('mode')}", flush=True)
    replan = int(args.replan)
    if not 1 <= replan <= horizon:
        raise SystemExit(f"--replan must be in 1..{horizon}")

    for key in keys:
        ep = load_home_episode(key) if args.home_root else load_episode(Path(args.data_parent), key)
        key = f"{ep['task']}/{ep['name']}"
        n = len(ep["obs"])
        if args.max_seconds:
            n = min(n, int(args.max_seconds * 30))
        anchors = list(range(0, n, replan))
        frames = anchor_frames(ep["video"], set(anchors), tuple(args.crop))
        anchors = [a for a in anchors if a in frames]
        n = min(n, anchors[-1] + replan + 1)
        obs = ep["obs"][:n]
        quat0 = obs[0, QUAT]
        policy.reset(key)
        pred = np.zeros((n, policy.dimension), np.float32)
        infer_s = []
        for a in anchors:
            o = obs[a]
            state = sonic_state({"joint_pos": o[Q], "base_quat": o[QUAT], "left_hand": o[HAND][:6], "right_hand": o[HAND][6:]},
                                hand_scale=1.0, reference_quat=quat0)
            request = SimpleNamespace(
                view="ego", session_id=key, context={},
                observation=SimpleNamespace(instruction=ep["instruction"],
                                            samples={"image": sample(frames[a], a), "state": sample(state, a)}),
            )
            t1 = time.time()
            chunk = policy.predict(request)  # (25, 66), physical units, rows = t+1 ...
            infer_s.append(time.time() - t1)
            if a == 0:
                pred[0] = chunk[0]
            rows = min(replan, n - 1 - a)
            pred[a + 1 : a + 1 + rows] = chunk[:rows]
        gt_token = ep["token"][:n]
        hold = np.zeros_like(gt_token)  # baseline: the anchor's recorded token held for the chunk
        for a in anchors:
            hold[a : a + replan + 1] = gt_token[a]

        arm, cam = obs[:, ARM], ep["camera_action"][:n]
        trig = np.clip(pred[:, 64:66], 0.0, 1.0)
        pred_open = np.concatenate([np.repeat(1.0 - trig[:, :1], 6, 1), np.repeat(1.0 - trig[:, 1:], 6, 1)], axis=1)
        pred_l92 = np.concatenate([pred[:, :64], arm, openness_to_radians(pred_open), cam], axis=1)
        gt_l92 = np.concatenate([gt_token, arm, openness_to_radians(ep["hand_action"][:n]), cam], axis=1)
        body47 = np.concatenate([obs[:, Q], openness_to_radians(obs[:, HAND]), obs[:, CAM], obs[:, QUAT]], axis=1)

        ep_dir = Path(args.output_dir) / split / ep["task"] / ep["name"]
        ep_dir.mkdir(parents=True, exist_ok=True)
        for name, arr in (("pred66", pred), ("gt_token", gt_token), ("pred_latent92", pred_l92),
                          ("gt_latent92", gt_l92), ("body47", body47)):
            np.save(ep_dir / f"{name}.npy", arr.astype(np.float32))
        # Recorded hand closure per side (1 = closed): mean over the four fingers' openness.
        closure = 1.0 - np.stack([ep["hand_action"][:n, 0:4].mean(1), ep["hand_action"][:n, 6:10].mean(1)], 1)
        # SONIC tokens are FSQ levels: distance of the predictions to the recorded level set.
        levels = np.unique(np.round(ep["token"], 4))
        offgrid = np.abs(pred[:, :64, None] - levels[None, None, :]).min(-1) if len(levels) <= 64 else np.full(1, np.nan)
        corr = [float(np.corrcoef(trig[:, i], closure[:, i])[0, 1]) if trig[:, i].std() > 1e-6 and closure[:, i].std() > 1e-6
                else float("nan") for i in range(2)]
        metrics = {
            "episode_key": key, "instruction": ep["instruction"], "n_frames": int(n), "n_chunks": len(anchors),
            "replan_rows": replan, "chunk_rows": horizon,
            "token_mae": float(np.abs(pred[:, :64] - gt_token).mean()),
            "token_mae_hold_baseline": float(np.abs(hold - gt_token).mean()),
            "token_mae_zero_baseline": float(np.abs(gt_token).mean()),
            "token_offgrid_mean": float(offgrid.mean()), "token_levels_in_recording": int(len(levels)),
            "token_pred_range": [float(pred[:, :64].min()), float(pred[:, :64].max())],
            "token_gt_range": [float(gt_token.min()), float(gt_token.max())],
            "trigger_mean_lr": trig.mean(0).tolist(), "recorded_closure_mean_lr": closure.mean(0).tolist(),
            "trigger_vs_closure_corr_lr": corr, "infer_s_mean": float(np.mean(infer_s)),
            "checkpoint": str(args.checkpoint),
        }
        (ep_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        shown = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in metrics.items()
                 if k.startswith(("token_mae", "token_offgrid", "trigger_vs"))}
        print(f"[infer] {ep_dir}: {json.dumps(shown)} ({n} frames, {np.mean(infer_s):.2f}s/chunk)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
