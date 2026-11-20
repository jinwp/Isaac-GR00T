#!/usr/bin/env python3
"""Kinematic play of a fine-tuned GR00T HA-Lab checkpoint (OpenHLM ``run_ha_ckpt_viz.py`` ``flow/`` twin).

For the selected episodes of the run's train / val split this does what OpenHLM's
strided open-loop replay does:

1. ``infer``: at anchors ``0, 32, 64, ...`` of an episode the checkpoint sees the
   *recorded* ego image, 76D state and instruction and predicts the next 32 actions;
   the first 32 rows of every chunk are stitched into one timeline and compared with
   the recorded actions.  Writes, per episode, ``pred_actions.npy`` / ``gt_actions.npy``
   ``(T, 47)`` in physical units and MotionTokenizer order ``[body29 | hand12 |
   camera2 | quat_wxyz4]`` (OpenHLM's format, plus ``*_abs`` copies), the same
   ``metrics.json`` keys as OpenHLM and ``action_plot_gt_vs_pred.png``.
2. ``render``: ``kinematic_gt_vs_pred.mp4`` (grey = recorded, blue = predicted) with
   the lab's MotionTokenizers renderer and G1 + Inspire-hand MJCF, exactly as OpenHLM's
   ``render_kinematic_videos``: offscreen MuJoCo on EGL, one subprocess per video
   (EGL aborts after CUDA inference in the same process), NVIDIA first, Mesa fallback.

This is open-loop and teacher-forced: the robot never acts on its own predictions.
Closed-loop behavior needs the SONIC physics sim.

Run on a GPU node from the Isaac-GR00T root::

    python examples/HALAB/halab_kinematic_play.py \\
        --checkpoint /coc/flash2/jpark984/gr00t_baseline/outputs/<run>/checkpoint-20000 \\
        --data-parent /coc/flash2/jpark984/gr00t_baseline/datasets/ha-lab-dataset/halab \\
        --mt-src /coc/flash2/dscho1234/Workspace/OpenHLM/thirdparty/MotionTokenizers/src
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np


# Our 47D order -> MotionTokenizer e1 order [body29 | hand12 | camera2 | quat4].
OURS_TO_MT = np.r_[0:29, 33:45, 45:47, 29:33]
ACTION_KEYS = ("body_q", "base_quat", "inspire_hand", "camera")
MESA_VENDOR_JSON = "/usr/share/glvnd/egl_vendor.d/50_mesa.json"


def mt_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    """OpenHLM ``run_ha_ckpt_viz._metrics`` for the 47-D MotionTokenizer layout."""

    n = min(len(pred), len(gt))
    err = np.abs(pred[:n] - gt[:n])
    return {
        "n_frames": int(n),
        "mae_all": float(err.mean()),
        "mae_camera": float(err[:, 41:43].mean()),
        "mae_hands": float(err[:, 29:41].mean()),
        "mae_body": float(err[:, :29].mean()),
        "mae_quat": float(err[:, 43:47].mean()),
    }


MT_GROUPS = {"body": slice(0, 29), "hands": slice(29, 41), "camera": slice(41, 43), "quat": slice(43, 47)}


def reference_metrics(pred: np.ndarray, gt: np.ndarray, stride: int) -> dict[str, float]:
    """Hold baseline and jitter, so ``mae_*`` can be read against something.

    ``hold_mae_<group>``: repeat the state at each anchor for the whole kept chunk (gt row
    ``a-1`` is the state at anchor ``a``, since action row k is the state at t+1+k); a
    policy that predicts motion should beat it.  ``jitter_<group>``: mean |frame-to-frame
    change| of the prediction inside chunks (no chunk boundary crossed) divided by the
    same for the recording; 1 = as smooth as the data, >1 = shaking.
    ``boundary_jump_body``: mean |pred[a] - pred[a-1]| at chunk boundaries over the
    recording's per-frame change.
    """

    n = min(len(pred), len(gt))
    pred, gt = pred[:n], gt[:n]
    hold = gt.copy()
    for a in range(0, n, stride):
        hold[a : a + stride] = gt[a - 1] if a > 0 else gt[0]
    inside = np.array([i for i in range(1, n) if i % stride != 0], dtype=int)
    bounds = np.arange(stride, n, stride)
    dp, dg = np.abs(np.diff(pred, axis=0)), np.abs(np.diff(gt, axis=0))
    out: dict[str, float] = {}
    for name, cols in MT_GROUPS.items():
        out[f"hold_mae_{name}"] = float(np.abs(hold[:, cols] - gt[:, cols]).mean())
        if len(inside):
            out[f"jitter_{name}"] = float(dp[inside - 1, cols].mean() / max(dg[inside - 1, cols].mean(), 1e-9))
    if len(bounds):
        out["boundary_jump_body"] = float(dp[bounds - 1, :29].mean() / max(dg[:, :29].mean(), 1e-9))
    return out


def predicted_state(rows: np.ndarray, q_before: np.ndarray) -> dict[str, np.ndarray]:
    """76D state at the next anchor from the kept rows of a predicted chunk (our 47D order).

    Action row k is the measured state at t+1+k, so the last kept row is the state at
    the next anchor.  dq is the finite difference of the last two rows at 30 Hz.
    """

    last = rows[-1]
    q_prev = rows[-2, 0:29] if len(rows) >= 2 else q_before
    quat = last[29:33] / max(float(np.linalg.norm(last[29:33])), 1e-8)
    state = {
        "body_q": last[0:29],
        "body_dq": (last[0:29] - q_prev) * 30.0,
        "base_quat": quat,
        "inspire_hand": last[33:45],
        "camera": last[45:47],
    }
    return {k: np.asarray(v, np.float32)[None] for k, v in state.items()}


def plot_actions(pred: np.ndarray, gt: np.ndarray, out_path: Path, title: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    names = (
        [f"body{i}" for i in range(29)] + [f"hand{i}" for i in range(12)]
        + ["cam_yaw", "cam_pitch", "quat_w", "quat_x", "quat_y", "quat_z"]
    )
    fig, axes = plt.subplots(8, 6, figsize=(24, 22), sharex=True)
    for d, ax in enumerate(axes.flat):
        if d >= pred.shape[1]:
            ax.axis("off")
            continue
        ax.plot(gt[:, d], color="0.4", lw=1.0, label="gt")
        ax.plot(pred[:, d], color="tab:blue", lw=0.8, label="pred")
        ax.set_title(names[d], fontsize=8)
        ax.tick_params(labelsize=6)
    axes.flat[0].legend(fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=80)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Stage 1: inference (GPU, torch, GR00T)
# ---------------------------------------------------------------------------


def select_episodes(split_path: Path, split: str, tasks: list[str] | None, per_task: int, seed: int) -> list[str]:
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


def run_inference(args: argparse.Namespace) -> list[Path]:
    import torch

    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here))
    import importlib.util

    spec = importlib.util.spec_from_file_location("halab_47d_config", here / "halab_47d_config.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # registers the 47D modality config

    from halab_gr00t_adapter import HalabGr00tAdapter, build_halab_dataset
    from gr00t.data.types import EmbodimentTag, MessageType, VLAStepData
    from gr00t.policy.gr00t_policy import Gr00tPolicy, _rec_to_dtype

    checkpoint = Path(args.checkpoint).resolve()
    run_dir = checkpoint.parent
    split_path = Path(args.episode_split) if args.episode_split else run_dir / "episode_split.json"
    keys = args.episode_keys or select_episodes(split_path, args.split, args.tasks, args.episodes_per_task, args.seed)
    if not keys:
        raise SystemExit(f"no episodes selected from {split_path} (split={args.split}, tasks={args.tasks})")
    print(f"{len(keys)} episode(s): {keys}", flush=True)

    tag = EmbodimentTag.NEW_EMBODIMENT
    policy = Gr00tPolicy(tag, str(checkpoint), device=args.device)
    processor, model = policy.processor, policy.model
    default_steps = int(model.action_head.num_inference_timesteps)
    if args.denoising_steps:
        model.action_head.num_inference_timesteps = int(args.denoising_steps)  # as gr00t/eval/open_loop_eval.py
    print(f"ODE denoising steps: {model.action_head.num_inference_timesteps} (checkpoint: {default_steps})", flush=True)
    dataset = build_halab_dataset(data_parent=args.data_parent, episode_keys=keys, load_images=True)
    episodes = [m for m in dataset.episode_list]
    ranges = dataset.episode_anchor_ranges()
    if len(episodes) != len(ranges):
        raise RuntimeError("episode metadata and anchor ranges disagree")

    out_root = Path(args.output_dir) if args.output_dir else run_dir.parent / f"{run_dir.name}_kinematic" / checkpoint.name
    stride = int(args.stride)
    if not 1 <= stride <= 32:
        raise SystemExit("--stride must be in 1..32 (the chunk has 32 rows)")
    feedback = args.rollout == "state-feedback"
    written = []
    for meta, (start, stop) in zip(episodes, ranges, strict=True):
        indices = list(range(start, stop))
        if args.max_seconds:
            indices = indices[: int(args.max_seconds * 30)]
        anchors = list(range(0, len(indices), stride))
        ep_dir = out_root / args.split / meta["task_key"] / f"episode_{meta['episode_index']:06d}"
        if feedback:
            ep_dir = ep_dir / ("state_feedback" if args.feedback_dq == "finite-diff" else f"state_feedback_dq_{args.feedback_dq}")
        if stride != 32:
            ep_dir = ep_dir / f"stride{stride}"
        if args.denoising_steps and args.denoising_steps != default_steps:
            ep_dir = ep_dir / f"steps{args.denoising_steps}"
        if args.normalize_quat:
            ep_dir = ep_dir / "qnorm"
        ep_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        pred_chunks, gt_chunks, infer_s = [], [], []
        next_state = None  # state-feedback: the state the previous chunk predicted for this anchor
        # Feedback chains anchor to anchor, so it runs one anchor per forward pass.
        for b in range(0, len(anchors), 1 if feedback else args.batch_size):
            batch_anchors = anchors[b : b + (1 if feedback else args.batch_size)]
            processed, states, gts = [], [], []
            for a in batch_anchors:
                step = HalabGr00tAdapter.to_vla_step(dataset[indices[a]])
                gts.append(np.concatenate([step.actions[k] for k in ACTION_KEYS], axis=-1))
                if next_state is not None:
                    if args.feedback_dq == "recorded":
                        next_state = dict(next_state, body_dq=step.states["body_dq"])
                    elif args.feedback_dq == "zero":
                        next_state = dict(next_state, body_dq=np.zeros_like(step.states["body_dq"]))
                    step.states.update(next_state)  # recorded image + language, predicted state
                states.append(step.states)
                obs = VLAStepData(images=step.images, states=step.states, actions={}, text=step.text, embodiment=tag)
                processed.append(processor([{"type": MessageType.EPISODE_STEP.value, "content": obs}]))
            collated = _rec_to_dtype(policy.collate_fn(processed), dtype=torch.bfloat16)
            t1 = time.time()
            with torch.inference_mode():
                normalized = model.get_action(**collated)["action_pred"].float().cpu().numpy()
            infer_s.append((time.time() - t1) / len(batch_anchors))
            batched_states = {k: np.stack([s[k] for s in states]) for k in states[0]}
            decoded = processor.decode_action(normalized, tag, batched_states)
            pred = np.concatenate([decoded[k] for k in ACTION_KEYS], axis=-1).astype(np.float32)  # (B, 32, 47)
            if args.normalize_quat:  # unit base quaternion (cols 29:33 in our order); also feeds state-feedback
                pred[..., 29:33] /= np.maximum(np.linalg.norm(pred[..., 29:33], axis=-1, keepdims=True), 1e-8)
            for i, a in enumerate(batch_anchors):
                keep = min(stride, len(indices) - a)
                pred_chunks.append(pred[i, :keep])
                gt_chunks.append(gts[i][:keep])
            if feedback:
                next_state = predicted_state(pred[0, :keep], states[0]["body_q"][-1])
        pred_mt = np.concatenate(pred_chunks)[:, OURS_TO_MT]
        gt_mt = np.concatenate(gt_chunks)[:, OURS_TO_MT].astype(np.float32)
        for name, arr in (("pred_actions", pred_mt), ("gt_actions", gt_mt)):
            np.save(ep_dir / f"{name}.npy", arr)
            np.save(ep_dir / f"{name}_abs.npy", arr)  # absolute layout: identical (OpenHLM convention)
        (ep_dir / "action_layout.txt").write_text("mt47\n")
        metrics = mt_metrics(pred_mt, gt_mt)
        metrics.update(reference_metrics(pred_mt, gt_mt, stride))
        metrics.update(
            {"n_chunks": len(anchors), "chunk_stride": stride, "rollout": args.rollout,
             "denoising_steps": int(model.action_head.num_inference_timesteps), "normalize_quat": bool(args.normalize_quat),
             "feedback_dq": args.feedback_dq if feedback else None, "infer_s_mean": float(np.mean(infer_s)),
             "checkpoint": str(checkpoint), "episode_key": f"{meta['task_key']}/{meta['episode_index']}",
             "instruction": meta["instruction"]}
        )
        (ep_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        plot_actions(pred_mt, gt_mt, ep_dir / "action_plot_gt_vs_pred.png", f"{meta['task_key']} ep{meta['episode_index']}")
        shown = {k: round(v, 4) for k, v in metrics.items() if k.startswith(("mae", "hold_mae", "jitter", "boundary"))}
        print(f"[infer] {ep_dir}: {json.dumps(shown)} "
              f"({len(pred_mt)} frames, {time.time() - t0:.0f}s)", flush=True)
        written.append(ep_dir)
    return written


# ---------------------------------------------------------------------------
# Stage 2: render (CPU process, MuJoCo EGL, MotionTokenizers)
# ---------------------------------------------------------------------------


def render_episode(ep_dir: Path, mt_src: str, fps: int, width: int, height: int) -> Path:
    """OpenHLM ``kinematic_play.render_kinematic_videos`` for absolute mt47 actions."""

    sys.path.insert(0, mt_src)
    from motion_tokenizers.data.input_extractor import input_vector_to_qpos_qvel
    from motion_tokenizers.replay.mujoco_replay import render_overlay_video
    from motion_tokenizers.schema import InputFeatureSpec

    spec = InputFeatureSpec(include_hand_vel=False, include_body_vel=False, include_camera=True, include_delta=False)
    pred = np.load(ep_dir / "pred_actions.npy").astype(np.float32)
    gt = np.load(ep_dir / "gt_actions.npy").astype(np.float32)
    pred_q, _ = input_vector_to_qpos_qvel(pred, spec=spec, root_xyz=None, integrated=True)
    gt_q, _ = input_vector_to_qpos_qvel(gt, spec=spec, root_xyz=None, integrated=True)
    t = min(len(pred_q), len(gt_q))
    camera_slice = getattr(spec, "camera_slice", None)
    use_camera = int(getattr(spec, "camera_width", 0)) == 2 and camera_slice is not None
    out = ep_dir / "kinematic_gt_vs_pred.mp4"
    options = {
        "fps": fps, "width": width, "height": height, "mode": "mesh", "track_root": True,
        "camera_yaw_pitch_orig": gt[:t, camera_slice] if use_camera else None,
        "camera_yaw_pitch_recon": pred[:t, camera_slice] if use_camera else None,
    }
    # MotionTokenizers checkouts differ: older render_overlay_video has no neck-camera
    # (camera_yaw_pitch_*) or track_root arguments.  Pass only what this one accepts.
    import inspect

    accepted = inspect.signature(render_overlay_video).parameters
    dropped = sorted(k for k in options if k not in accepted)
    if dropped:
        print(f"render_overlay_video does not take {dropped}; rendering without them", flush=True)
    render_overlay_video(
        np.asarray(gt_q, np.float32)[:t], np.asarray(pred_q, np.float32)[:t], out,
        **{k: v for k, v in options.items() if k in accepted},
    )
    return out


def render_isolated(ep_dir: Path, args: argparse.Namespace) -> bool:
    """One subprocess per video; NVIDIA EGL first, then Mesa (OpenHLM ``viz/isolated.py``)."""

    vendors = [""] if os.path.exists("/dev/nvidiactl") else []
    if os.path.exists(MESA_VENDOR_JSON):
        vendors.append(MESA_VENDOR_JSON)
    for vendor in vendors or [""]:
        env = dict(os.environ, MUJOCO_GL="egl")
        env.pop("__EGL_VENDOR_LIBRARY_FILENAMES", None)
        if vendor:
            env["__EGL_VENDOR_LIBRARY_FILENAMES"] = vendor
        cmd = [sys.executable, str(Path(__file__).resolve()), "--render-dir", str(ep_dir), "--mt-src", args.mt_src,
               "--fps", str(args.fps), "--width", str(args.width), "--height", str(args.height)]
        t0 = time.time()
        result = subprocess.run(cmd, env=env, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"[render] {ep_dir / 'kinematic_gt_vs_pred.mp4'} ({vendor or 'NVIDIA EGL'}, {time.time() - t0:.0f}s)", flush=True)
            return True
        tail = (result.stderr or result.stdout).strip().splitlines()[-5:]
        print(f"[render] failed with {vendor or 'NVIDIA EGL'} (exit {result.returncode}): {' | '.join(tail)}", flush=True)
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", help="checkpoint dir, e.g. <run>/checkpoint-20000")
    parser.add_argument("--data-parent", default="/coc/flash2/jpark984/gr00t_baseline/datasets/ha-lab-dataset/halab")
    parser.add_argument("--mt-src", default="/coc/flash2/dscho1234/Workspace/OpenHLM/thirdparty/MotionTokenizers/src")
    parser.add_argument("--split", choices=("val", "train"), default="val")
    parser.add_argument("--episode-split", help="default: <run>/episode_split.json")
    parser.add_argument("--tasks", nargs="*", help="task keys (folder names); default: every task")
    parser.add_argument("--episodes-per-task", type=int, default=1)
    parser.add_argument("--episode-keys", nargs="*", help="explicit 'task/episode_index' keys instead of a selection")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-seconds", type=float, default=0, help="cap each episode (0 = whole episode)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--feedback-dq", choices=("finite-diff", "recorded", "zero"), default="finite-diff",
        help="state-feedback: body_dq from the predicted rows (x30 Hz), the recording, or zeros",
    )
    parser.add_argument(
        "--rollout", choices=("replan", "state-feedback"), default="replan",
        help="replan: every chunk starts from the recorded state (OpenHLM flow/). state-feedback: "
        "the state comes from the previous prediction (recorded images, no reset to the recording)",
    )
    parser.add_argument(
        "--stride", type=int, default=32,
        help="replan every N frames and keep only the first N rows of each chunk "
        "(OpenHLM --action-chunk-stride; 32 = the whole chunk)",
    )
    parser.add_argument(
        "--denoising-steps", type=int, default=0,
        help="flow-matching ODE steps at inference (0 = the checkpoint's, 4); results go to a stepsN/ subfolder",
    )
    parser.add_argument(
        "--normalize-quat", action="store_true",
        help="renormalize the predicted base quaternion to unit norm; results go to a qnorm/ subfolder",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", help="default: <outputs>/<run>_kinematic/<checkpoint>")
    parser.add_argument("--stage", choices=("all", "infer", "render"), default="all")
    parser.add_argument("--render-root", help="--stage render: render every episode dir with pred/gt npy under this dir")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=576)
    parser.add_argument("--render-dir", help=argparse.SUPPRESS)  # internal: one render in this process
    args = parser.parse_args()

    if args.render_dir:
        render_episode(Path(args.render_dir), args.mt_src, args.fps, args.width, args.height)
        return 0
    if args.stage == "render":
        if not args.render_root:
            raise SystemExit("--stage render needs --render-root")
        ep_dirs = sorted(p.parent for p in Path(args.render_root).rglob("pred_actions.npy"))
    else:
        if not args.checkpoint:
            raise SystemExit("--checkpoint is required")
        ep_dirs = run_inference(args)
        if args.stage == "infer":
            return 0
    ok = sum(render_isolated(d, args) for d in ep_dirs)
    print(f"done: {ok}/{len(ep_dirs)} video(s)", flush=True)
    return 0 if ok == len(ep_dirs) else 1


if __name__ == "__main__":
    sys.exit(main())
