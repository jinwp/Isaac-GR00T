#!/usr/bin/env python3
"""Visual heading gain G_v and state-yaw gain G_p of a GR00T HA-Lab checkpoint.

Inference only.  On locomotion anchors of the validation episodes, the recorded
frame is replaced by an exact scene-rotation counterfactual (``delta`` about world
up) with the recorded state, or the state yaw is offset by ``alpha`` with the
recorded frame.  The same flow noise is used in every condition, so differences
come from the input alone.

    G_v(k) = sum label_sign*delta*dyaw_k / sum delta^2     (1 = follows the scene; 0 = ignores vision)
    G_p(k) = sum alpha*dyaw_k / sum alpha^2                (1 = gauge-equivariant in IMU yaw)

Also reports, on nominal anchors, the slope of predicted vs recorded yaw change over
the chunk (the 20k run's playback slope was 0.58).  Writes one JSON.

Run from the Isaac-GR00T root with the GR00T venv::

    python examples/HALAB_CF/probe_gr00t_gain.py --checkpoint <run>/checkpoint-N --output <dir>/probe.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import sys
import time

import numpy as np


HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "HALAB"
for _path in (str(HERE), str(BASE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import halab_rotation_cf as rcf  # noqa: E402

ROOT = Path("/coc/flash2/jpark984/gr00t_baseline")
ACTION_KEYS = ("body_q", "base_quat", "inspire_hand", "camera")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--data-parent", type=Path, default=ROOT / "datasets/ha-lab-dataset/halab")
    p.add_argument("--episode-split", type=Path,
                   default=ROOT / "outputs/halab_g1_47d_openhlm_b64_g256_lr1e4_v2/episode_split.json")
    p.add_argument("--split", choices=("val", "train"), default="val")
    p.add_argument("--calibration", type=Path, default=ROOT / "outputs/cf_cache/calibration.json")
    p.add_argument("--stride", type=int, default=6)
    p.add_argument("--gain-anchors", type=int, default=512)
    p.add_argument("--nominal-anchors", type=int, default=1024)
    p.add_argument("--deltas-deg", default="-6,-3,3,6")
    p.add_argument("--state-yaw-deg", type=float, default=10.0)
    p.add_argument("--seeds", type=int, default=4)
    p.add_argument("--denoising-steps", type=int, default=0, help="0 = the checkpoint's default (4).")
    p.add_argument("--crop-frac", type=float, default=0.0,
                   help="0 for checkpoints trained on full frames; 0.08 for the Experiment 1 arms.")
    p.add_argument("--walk-threshold", type=float, default=0.15)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def walking_rows(parquet_path: Path, length: int, threshold: float) -> np.ndarray:
    """Per anchor row: does the action chunk rows [t+1, t+32] contain locomotion?"""

    import pyarrow.parquet as pq

    table = pq.read_table(parquet_path, columns=["action.measured_state"])
    measured = np.asarray(table["action.measured_state"].to_pylist(), dtype=np.float32)
    legs = measured[:, :12]
    out = np.zeros(length, dtype=bool)
    for t in range(length - 1):
        rows = np.clip(np.arange(t + 1, t + 33), 0, length - 1)
        out[t] = float(np.max(np.ptp(legs[rows], axis=0))) > threshold
    return out


def evenly(items: list[int], n: int) -> list[int]:
    if len(items) <= n:
        return list(items)
    idx = np.linspace(0, len(items) - 1, n).round().astype(int)
    return [items[i] for i in idx]


def main() -> None:
    args = parse_args()
    import torch

    spec = importlib.util.spec_from_file_location("halab_47d_config", BASE / "halab_47d_config.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # registers the 47D modality config

    from gr00t.data.types import EmbodimentTag, MessageType, VLAStepData
    from gr00t.policy.gr00t_policy import Gr00tPolicy, _rec_to_dtype
    from halab_gr00t_adapter import HalabGr00tAdapter, build_halab_dataset

    label_sign = 1
    if args.calibration.is_file():
        label_sign = int(json.loads(args.calibration.read_text())["label_sign"])
    if label_sign not in (1, -1):
        print(f"WARNING: calibration label_sign={label_sign}; reporting G_v with sign +1", flush=True)
        label_sign = 1

    split = json.loads(args.episode_split.read_text())
    keys = split[f"{args.split}_keys"]
    meta_ds = build_halab_dataset(data_parent=args.data_parent, episode_keys=keys, load_images=False)
    gain_pool, nominal_pool = [], []
    for meta, (start, stop) in zip(meta_ds.episode_list, meta_ds.episode_anchor_ranges(), strict=True):
        length = int(meta["len_episode"])
        parquet = Path(meta["dataset_root"]) / "data" / "chunk-000" / f"episode_{int(meta['episode_index']):06d}.parquet"
        walk = walking_rows(parquet, length, args.walk_threshold)
        for row in range(0, stop - start, args.stride):
            nominal_pool.append(start + row)
            if walk[row]:
                gain_pool.append(start + row)
    gain_idx = evenly(gain_pool, args.gain_anchors)
    nominal_idx = evenly(nominal_pool, args.nominal_anchors)
    print(f"{args.split}: {len(keys)} episodes, {len(nominal_pool)} anchors, {len(gain_pool)} locomotion; "
          f"probing {len(gain_idx)} gain / {len(nominal_idx)} nominal anchors", flush=True)
    dataset = build_halab_dataset(data_parent=args.data_parent, episode_keys=keys, load_images=True)
    if len(dataset) != len(meta_ds):
        raise RuntimeError("image and metadata datasets disagree on anchor count")

    tag = EmbodimentTag.NEW_EMBODIMENT
    policy = Gr00tPolicy(tag, str(args.checkpoint.resolve()), device=args.device)
    processor, model = policy.processor, policy.model
    if hasattr(processor, "eval"):
        processor.eval()
    default_steps = int(model.action_head.num_inference_timesteps)
    if args.denoising_steps:
        model.action_head.num_inference_timesteps = int(args.denoising_steps)
    steps_used = int(model.action_head.num_inference_timesteps)

    deltas = [math.radians(float(d)) for d in args.deltas_deg.split(",") if d.strip()]
    alpha = math.radians(args.state_yaw_deg)
    conditions = [("ref", 0.0, 0.0)] + [(f"delta{math.degrees(d):+.0f}", d, 0.0) for d in deltas] + [
        (f"alpha{math.degrees(a):+.0f}", 0.0, a) for a in (alpha, -alpha)
    ]

    def predict(steps, ups, delta: float, yaw_offset: float, seeds: list[int]):
        processed, states = [], []
        for step, up in zip(steps, ups):
            frame = step.images["ego_view"][-1]
            if delta != 0.0:
                frame = rcf.warp_scene_rotation(frame, up, delta)
            frame = rcf.center_crop_resize(frame, args.crop_frac)
            st = {k: np.array(v, copy=True) for k, v in step.states.items()}
            if yaw_offset != 0.0:
                st["base_quat"] = rcf.rotate_quat_yaw(st["base_quat"], yaw_offset).astype(np.float32)
            states.append(st)
            obs = VLAStepData(images={"ego_view": [frame]}, states=st, actions={}, text=step.text, embodiment=tag)
            processed.append(processor([{"type": MessageType.EPISODE_STEP.value, "content": obs}]))
        collated = _rec_to_dtype(policy.collate_fn(processed), dtype=torch.bfloat16)
        batched_states = {k: np.stack([s[k] for s in states]) for k in states[0]}
        out = []
        for seed in seeds:
            torch.manual_seed(1000 + seed)
            torch.cuda.manual_seed_all(1000 + seed)
            with torch.inference_mode():
                normalized = model.get_action(**collated)["action_pred"].float().cpu().numpy()
            decoded = processor.decode_action(normalized, tag, batched_states)
            out.append({k: np.asarray(decoded[k], dtype=np.float64) for k in ("body_q", "base_quat")})
        return out

    t0 = time.time()
    seeds = list(range(args.seeds))
    num_v = np.zeros(32)
    num_v_pos = np.zeros(32)
    num_v_neg = np.zeros(32)
    den_v = den_v_pos = den_v_neg = 0.0
    num_p = np.zeros(32)
    den_p = 0.0
    offtarget_sum, offtarget_n = 0.0, 0
    per_anchor_gv = []
    for b in range(0, len(gain_idx), args.batch_size):
        batch = gain_idx[b : b + args.batch_size]
        samples = [dataset[i] for i in batch]
        steps = [HalabGr00tAdapter.to_vla_step(s) for s in samples]
        ups = [rcf.up_axis_in_camera(s["observation"]["camera_pose"][-1].numpy())[0] for s in samples]
        preds = {name: predict(steps, ups, d, a, seeds) for name, d, a in conditions}
        ref_yaw = [rcf.quat_yaw(p["base_quat"]) for p in preds["ref"]]
        anchor_num = np.zeros(len(batch))
        anchor_den = 0.0
        for name, d, a in conditions[1:]:
            for s, p in enumerate(preds[name]):
                dyaw = rcf.wrap_angle(rcf.quat_yaw(p["base_quat"]) - ref_yaw[s])  # (B, 32)
                if d != 0.0:
                    x = label_sign * d
                    num_v += (x * dyaw).sum(axis=0)
                    den_v += len(batch) * d * d
                    if d > 0:
                        num_v_pos += (x * dyaw).sum(axis=0)
                        den_v_pos += len(batch) * d * d
                    else:
                        num_v_neg += (x * dyaw).sum(axis=0)
                        den_v_neg += len(batch) * d * d
                    anchor_num += x * dyaw[:, 31]
                    anchor_den += d * d
                    dbody = np.abs(p["body_q"] - preds["ref"][s]["body_q"]).mean(axis=(1, 2))
                    offtarget_sum += float((dbody / abs(d)).sum())
                    offtarget_n += len(batch)
                else:
                    num_p += (a * dyaw).sum(axis=0)
                    den_p += len(batch) * a * a
        per_anchor_gv.extend((anchor_num / anchor_den).tolist())
        print(f"gain batch {b // args.batch_size + 1}/{math.ceil(len(gain_idx) / args.batch_size)} "
              f"G_v(31) so far {num_v[31] / max(den_v, 1e-12):.3f} ({time.time() - t0:.0f}s)", flush=True)

    pred_d, gt_d, yaw_err0, yaw_err31 = [], [], [], []
    for b in range(0, len(nominal_idx), args.batch_size):
        batch = nominal_idx[b : b + args.batch_size]
        samples = [dataset[i] for i in batch]
        steps = [HalabGr00tAdapter.to_vla_step(s) for s in samples]
        ups = [rcf.FALLBACK_UP for _ in samples]
        p = predict(steps, ups, 0.0, 0.0, [0])[0]
        yaw_pred = rcf.quat_yaw(p["base_quat"])
        for i, step in enumerate(steps):
            valid = np.asarray(step.metadata["action_valid_mask"]) > 0
            if not valid[31]:
                continue
            yaw_state = float(rcf.quat_yaw(step.states["base_quat"][-1]))
            yaw_gt = rcf.quat_yaw(step.actions["base_quat"])
            pred_d.append(float(rcf.wrap_angle(yaw_pred[i, 31] - yaw_state)))
            gt_d.append(float(rcf.wrap_angle(yaw_gt[31] - yaw_state)))
            yaw_err0.append(abs(float(rcf.wrap_angle(yaw_pred[i, 0] - yaw_gt[0]))))
            yaw_err31.append(abs(float(rcf.wrap_angle(yaw_pred[i, 31] - yaw_gt[31]))))
    pred_d, gt_d = np.asarray(pred_d), np.asarray(gt_d)

    gv = num_v / max(den_v, 1e-12)
    gp = num_p / max(den_p, 1e-12)
    rng = np.random.default_rng(0)
    pa = np.asarray(per_anchor_gv)
    boots = [float(pa[rng.integers(0, len(pa), len(pa))].mean()) for _ in range(2000)] if len(pa) else [float("nan")]
    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "split": args.split,
        "label_sign": label_sign,
        "crop_frac": args.crop_frac,
        "denoising_steps": steps_used,
        "checkpoint_default_steps": default_steps,
        "seeds": args.seeds,
        "deltas_deg": [math.degrees(d) for d in deltas],
        "state_yaw_deg": args.state_yaw_deg,
        "n_gain_anchors": len(gain_idx),
        "n_nominal_used": int(len(gt_d)),
        "G_v_row": gv.tolist(),
        "G_v_0": float(gv[0]),
        "G_v_15": float(gv[15]),
        "G_v_31": float(gv[31]),
        "G_v_31_ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
        "G_v_31_pos": float(num_v_pos[31] / max(den_v_pos, 1e-12)),
        "G_v_31_neg": float(num_v_neg[31] / max(den_v_neg, 1e-12)),
        "G_p_row": gp.tolist(),
        "G_p_0": float(gp[0]),
        "G_p_31": float(gp[31]),
        "offtarget_body_rad_per_rad": offtarget_sum / max(offtarget_n, 1),
        "turn_slope_origin": float((gt_d * pred_d).sum() / max((gt_d**2).sum(), 1e-12)),
        "turn_slope_ols": float(np.polyfit(gt_d, pred_d, 1)[0]) if len(gt_d) > 2 else float("nan"),
        "turn_corr": float(np.corrcoef(gt_d, pred_d)[0, 1]) if len(gt_d) > 2 else float("nan"),
        "yaw_mae_row0_deg": float(np.degrees(np.mean(yaw_err0))) if yaw_err0 else float("nan"),
        "yaw_mae_row31_deg": float(np.degrees(np.mean(yaw_err31))) if yaw_err31 else float("nan"),
        "seconds": time.time() - t0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    shown = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in result.items() if not k.endswith("_row")}
    print(json.dumps(shown, indent=2), flush=True)


if __name__ == "__main__":
    main()
