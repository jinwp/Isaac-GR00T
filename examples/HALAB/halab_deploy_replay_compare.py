#!/usr/bin/env python3
"""Data-replay deploy test vs offline GR00T, anchor by anchor (same image, different joints / instruction).

A deploy run with ``--policy-arg image_source=<episode.mp4>`` feeds the policy the recorded episode's image at
each anchor while the joint state comes from the live robot.  For every anchor ``a`` of one recorded deploy
episode (the test dataset written by the episode recorder) this compares, on the rows the robot executes
(``[delay, delay + replan)`` of each 50-row chunk):

* ``A``  the recorded actions of the data episode (rows a+1 ...);
* ``B``  the model on the data image + the data state + the training instruction;
* ``B'`` the same with the deploy run's instruction;
* ``C``  what the model actually returned on the robot (``chunks/*.npz`` ``aux``, decoded);
* ``D``  the model on the data image + the robot's measured state + the deploy instruction, with the deploy
  run's RTC prefix: should reproduce ``C`` (else the deploy path differs from this one);
* ``E``  ``D`` without the prefix, and ``E'`` with the training instruction.

``B`` vs ``A``: is the model right on its own data.  ``C`` vs ``D``: is the deploy path faithful.  ``B`` vs
``E``: what the robot's joint state alone changes.  ``B`` vs ``B'``: what the instruction alone changes.
``I1``, ``I2``, ... (``--extra-instruction``): ``B`` with other sentences, e.g. other tasks' training sentences, to see
which task the deploy sentence behaves like.
``S:<group>`` (``--swap-groups``): ``B`` with only that part of the state taken from the robot (legs, waist, arms,
wrists, dq, base_quat, hands, camera), to see which part of ``E - B`` it carries.
Every variant of one anchor uses the same noise seed, so their differences come from the inputs; ``B2``
(``B`` with another seed) is the sampling noise floor.

The robot state is the recorded ``observation.state`` converted to the training units as the deploy client
does (``producer._proprio``: joint-sign flip, wrist scale); the conversion is checked against the first
chunk, which starts at the measured pose.  Also writes MT47 trajectories for
``halab_kinematic_play.py --stage render --render-root <out>/render`` (grey = first, blue = second):
``state_data_vs_robot`` (the poses the two runs were in), ``gt_vs_deploy`` and ``gt_vs_offline``.

    python examples/HALAB/halab_deploy_replay_compare.py \\
        --checkpoint <run>/checkpoint-50000 --test-root <gr00t_test> --test-episode 0 \\
        --data-key pick_puppy_doll_place_black_basket/99 --out <outputs>/deploy_compare
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import halab_gr00t_server as srv  # noqa: E402

# Our 47D order -> MotionTokenizer e1 order [body29 | hand12 | camera2 | quat_wxyz4].
OURS_TO_MT = np.r_[0:29, 33:45, 45:47, 29:33]
GROUPS = {  # body_q columns
    "legs": list(range(0, 12)),
    "waist": [12, 13, 14],
    "L_arm": list(range(15, 22)),
    "R_arm": list(range(22, 29)),
    "wrists": [19, 20, 21, 26, 27, 28],
}
STATE_FROM_SPLIT = {  # model state group <- _split_observation_state key (deploy/gr00t.py STATE_KEYS)
    "body_q": "measured_joint_q",
    "body_dq": "measured_joint_dq",
    "base_quat": "body_quat",
    "inspire_hand": "hand_action",
    "camera": "camera_action",
}
WRIST_MOTORS = (20, 21, 27, 28)
# A right-only Inspire service records the left hand as zeros (closed); the deploy client feeds the policy this
# open hand instead (samsung_deploy robot_frame/left_hand_override.py OPEN_LEFT_Q, hardware order, 1 = open).
OPEN_LEFT_Q = np.array([0.997, 1.0, 1.0, 1.0, 0.996, 0.984], dtype=np.float32)
LEFT_HAND = slice(62, 68)  # observation.state left Inspire block
DEG = 180.0 / np.pi


def yaw(q: np.ndarray) -> np.ndarray:
    w, x, y, z = np.moveaxis(np.asarray(q, np.float64), -1, 0)
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def group_mae_deg(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """Mean abs difference of two (..., 47) arrays in our order: body groups and camera in deg, hands in rad."""
    out = {g: float(np.abs(a[..., c] - b[..., c]).mean() * DEG) for g, c in GROUPS.items()}
    out["body"] = float(np.abs(a[..., :29] - b[..., :29]).mean() * DEG)
    out["hands_rad"] = float(np.abs(a[..., 33:45] - b[..., 33:45]).mean())
    out["camera"] = float(np.abs(a[..., 45:47] - b[..., 45:47]).mean() * DEG)
    d = np.angle(np.exp(1j * (yaw(a[..., 29:33]) - yaw(b[..., 29:33]))))
    out["root_yaw"] = float(np.abs(d).mean() * DEG)
    return out


def state_vec(states: dict[str, np.ndarray]) -> np.ndarray:
    """Model state groups -> one row in our action order [q 29 | quat 4 | hand 12 | cam 2] (for MAE / render)."""
    return np.concatenate([states[k][-1] for k in ("body_q", "base_quat", "inspire_hand", "camera")]).astype(np.float32)


def convert_robot_state(raw: np.ndarray, mode: str, flip: tuple[int, ...], scale: float) -> np.ndarray:
    """Recorded observation.state (76) -> the policy's units, as ``producer._proprio`` does after the recorder."""
    s = np.array(raw, dtype=np.float32, copy=True)
    if not s[..., LEFT_HAND].any():
        s[..., LEFT_HAND] = OPEN_LEFT_Q  # as frame_client.open_left_hand_if_unread
    if mode in ("flip", "flip+wrist"):
        for j in flip:
            s[..., j] *= -1.0
            s[..., 29 + j] *= -1.0
    if mode in ("wrist", "flip+wrist"):
        for m in WRIST_MOTORS:
            s[..., m] *= scale
            s[..., 29 + m] *= scale * 4.0  # reported dq is 1/4 of d(reported q)/dt (openpi.ha.joint_units)
    return s


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--test-root", required=True, help="recorded deploy dataset (meta/, data/, raw_streams/)")
    p.add_argument("--test-episode", type=int, required=True)
    p.add_argument("--data-key", required=True, help="'<task>/<episode_index>' of the replayed episode in --data-parent")
    p.add_argument("--data-parent", default="/coc/flash2/jpark984/gr00t_baseline/datasets/ha-lab-dataset-38tasks/halab")
    p.add_argument("--out", required=True)
    p.add_argument("--state-units", choices=("auto", "none", "flip", "wrist", "flip+wrist"), default="auto",
                   help="conversion of the recorded observation.state (auto: the one matching the first chunk)")
    p.add_argument("--rtc-max-delay", type=int, default=10)
    p.add_argument("--denoising-steps", type=int, default=4)
    p.add_argument("--swap-groups", action="store_true", help="also run S:<group> (one state part from the robot)")
    p.add_argument("--extra-instruction", action="append", default=[],
                   help="also run B with this sentence (repeatable): variants I1, I2, ...")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    test_root = Path(args.test_root)
    episodes = [json.loads(line) for line in (test_root / "meta" / "episodes.jsonl").read_text().splitlines() if line]
    meta = next(e for e in episodes if int(e["episode_index"]) == args.test_episode)
    run_dir = test_root / "raw_streams" / meta["episode_id"] / "vla_deploy"
    summary = json.loads((run_dir / "run_summary.json").read_text())
    deploy_text = str(summary["task"])
    delay = int(summary.get("inference_delay", 5))
    replan = int(summary.get("replan_interval", 20))
    flip = tuple(int(j) for j in summary.get("joint_sign_flip") or ())
    scale = float(summary.get("wrist_scale") or 1.0)
    start_ns = int(json.loads((run_dir / "recorded_episode.json").read_text())["start_ns"])
    replans = {int(r["anchor_step"]): r for r in map(json.loads, (run_dir / "replans.jsonl").read_text().splitlines())}
    chunks = {}
    for f in sorted((run_dir / "chunks").glob("chunk_*.npz")):
        z = np.load(f)
        chunks[int(z["anchor_step"][0])] = {"sent_mt": z["chunk"], "aux": z["aux"]}

    import pyarrow.parquet as pq

    table = pq.read_table(test_root / "data" / f"chunk-{args.test_episode // 1000:03d}" / f"episode_{args.test_episode:06d}.parquet",
                          columns=["observation.state", "timestamp"]).to_pydict()
    robot_raw = np.asarray(table["observation.state"], dtype=np.float32)
    robot_t = np.asarray(table["timestamp"], dtype=np.float64)

    model = srv.Gr00tHalabModel(args.checkpoint, args.device, args.denoising_steps, args.rtc_max_delay)
    os.environ["HALAB_ACTION_HORIZON"] = str(model.horizon)  # read when the adapter is imported
    import torch
    from halab_gr00t_adapter import HalabGr00tAdapter, build_halab_dataset
    from data_loader.ha.vla_episode_io import _split_observation_state

    dataset = build_halab_dataset(data_parent=args.data_parent, episode_keys=[args.data_key], load_images=True)
    (start, stop), = dataset.episode_anchor_ranges()
    n_data = stop - start

    def robot_row(anchor: int) -> int:
        rec = replans.get(anchor)
        if rec and rec.get("stamps", {}).get("frame_stamp_ns"):
            t = (int(rec["stamps"]["frame_stamp_ns"]) - start_ns) / 1e9
            return int(np.argmin(np.abs(robot_t - t)))
        return int(min(anchor, len(robot_raw) - 1))

    def robot_states(anchor: int, mode: str) -> dict[str, np.ndarray]:
        split = _split_observation_state(convert_robot_state(robot_raw[robot_row(anchor)], mode, flip, scale), include_body_vel=True)
        return {k: np.asarray(split[v], np.float32)[None] for k, v in STATE_FROM_SPLIT.items()}

    # The first chunk is eased from the measured pose: its row 0 tells which conversion the recorded state needs.
    a0 = min(chunks)
    row0 = chunks[a0]["sent_mt"][0]
    check = {}
    for mode in ("none", "flip", "wrist", "flip+wrist"):
        q = robot_states(a0, mode)["body_q"][0]
        check[mode] = float(np.abs(q[[12, 20, 21, 27, 28]] - row0[[12, 20, 21, 27, 28]]).mean() * DEG)
    mode = min(check, key=check.get) if args.state_units == "auto" else args.state_units
    print(f"[units] first chunk row 0 vs recorded state (waist yaw + wrists, deg): {check} -> using '{mode}'", flush=True)

    def infer(image, states, text, seed, prefix=None):
        torch.manual_seed(seed)
        actions, norm = model.infer(image, states, text, prefix)
        return actions, norm

    def decode(norm: np.ndarray, states: dict[str, np.ndarray]) -> np.ndarray:
        dec = model.processor.decode_action(norm[None].astype(np.float32), model.tag, {k: v[None] for k, v in states.items()})
        return np.concatenate([dec[k] for k in srv.ACTION_KEYS], axis=-1)[0].astype(np.float32)

    rows = slice(delay, delay + replan)  # what the robot executed of each chunk
    anchors = [a for a in sorted(chunks) if a < n_data]
    extra = {f"I{j + 1}": text for j, text in enumerate(args.extra_instruction)}
    # state part -> (state key, columns); q and dq of the same joints move together
    swaps = {}
    if args.swap_groups:
        for g in ("legs", "waist", "arms", "wrists"):
            cols = GROUPS[g] if g != "arms" else GROUPS["L_arm"] + GROUPS["R_arm"]
            swaps[f"S:{g}"] = [("body_q", cols), ("body_dq", cols)]
        swaps.update({"S:dq": [("body_dq", None)], "S:base_quat": [("base_quat", None)],
                      "S:hands": [("inspire_hand", None)], "S:camera": [("camera", None)]})
    names = ("A", "B", "B'", "B2", "C", "D", "E", "E'") + tuple(extra) + tuple(swaps)
    per_anchor, traj = [], {n: [] for n in names + ("state_data", "state_robot")}
    prev_aux = None
    for i, a in enumerate(anchors):
        step = HalabGr00tAdapter.to_vla_step(dataset[start + a])
        image = np.asarray(step.images["ego_view"])[-1]
        data_states = {k: np.asarray(v, np.float32) for k, v in step.states.items()}
        train_text = str(step.text)
        rob = robot_states(a, mode)
        gt = np.concatenate([step.actions[k] for k in srv.ACTION_KEYS], axis=-1).astype(np.float32)
        prefix = None
        if prev_aux is not None:
            off = a - anchors[i - 1]
            prefix = prev_aux[off : off + delay]
        out = {
            "A": gt,
            "B": infer(image, data_states, train_text, a)[0],
            "B'": infer(image, data_states, deploy_text, a)[0],
            "B2": infer(image, data_states, train_text, a + 100000)[0],
            "C": decode(chunks[a]["aux"][: model.horizon], rob),
            "D": infer(image, rob, deploy_text, a, prefix)[0],
            "E": infer(image, rob, deploy_text, a)[0],
            "E'": infer(image, rob, train_text, a)[0],
        }
        for name, text in extra.items():
            out[name] = infer(image, data_states, text, a)[0]
        for name, parts in swaps.items():
            mixed = {k: v.copy() for k, v in data_states.items()}
            for key, cols in parts:
                if cols is None:
                    mixed[key] = rob[key].copy()
                else:
                    mixed[key][..., cols] = rob[key][..., cols]
            out[name] = infer(image, mixed, train_text, a)[0]
        prev_aux = chunks[a]["aux"]
        n = min(len(v) for v in out.values())
        r = slice(rows.start, min(rows.stop, n))
        rec = {"anchor": a, "robot_row": robot_row(a),
               "state_gap": group_mae_deg(state_vec(rob)[None], state_vec(data_states)[None])}
        pairs = [("B", "A"), ("B2", "B"), ("C", "A"), ("C", "D"), ("D", "C"), ("E", "B"), ("B'", "B"), ("E'", "B"), ("C", "B")]
        pairs += [(name, ref) for name in extra for ref in ("B", "A", "B'")]
        pairs += [(name, ref) for name in swaps for ref in ("B", "E'")]
        for x, y in pairs:
            rec[f"{x}-{y}"] = group_mae_deg(out[x][r], out[y][r])
        per_anchor.append(rec)
        for k, v in out.items():
            traj[k].append(v[r])
        traj["state_data"].append(np.repeat(state_vec(data_states)[None], r.stop - r.start, 0))
        traj["state_robot"].append(np.repeat(state_vec(rob)[None], r.stop - r.start, 0))
        if i % 10 == 0:
            shown = ", ".join(f"{k} {rec[k]['body']:.2f}" for k in ("B-A", "C-A", "D-C", "E-B", "B'-B"))
            print(f"[{i + 1}/{len(anchors)}] anchor {a}: {shown} deg body, state gap {rec['state_gap']['body']:.2f}", flush=True)

    out_dir = Path(args.out) / f"test{args.test_episode}_{args.data_key.replace('/', '_ep')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    keys = [k for k in per_anchor[0] if k not in ("anchor", "robot_row")]
    mean = {k: {g: float(np.mean([r[k][g] for r in per_anchor])) for g in per_anchor[0][k]} for k in keys}
    info = {"test_episode": args.test_episode, "episode_id": meta["episode_id"], "data_key": args.data_key,
            "deploy_instruction": deploy_text, "training_instruction": train_text, "instruction_matches": deploy_text == train_text,
            "image_source": summary.get("policy_args", {}).get("image_source"), "n_anchors": len(anchors),
            "anchors_after_video_end": len([a for a in chunks if a >= n_data]), "data_frames": n_data,
            "rows_compared": [rows.start, rows.stop], "state_units": mode, "state_units_check_deg": check,
            "waist_yaw_offset_deg": summary.get("waist_yaw_offset_deg"), "extra_instructions": extra, "mean_mae": mean}
    (out_dir / "summary.json").write_text(json.dumps(info, indent=2))
    (out_dir / "per_anchor.json").write_text(json.dumps(per_anchor))
    stitched = {k: np.concatenate(v).astype(np.float32) for k, v in traj.items()}
    np.savez(out_dir / "traj_ours47.npz", **stitched)
    for name, (g, pr) in {"state_data_vs_robot": ("state_data", "state_robot"),
                           "gt_vs_deploy": ("A", "C"), "gt_vs_offline": ("A", "B")}.items():
        d = out_dir / "render" / name
        d.mkdir(parents=True, exist_ok=True)
        for fname, key in (("gt_actions.npy", g), ("pred_actions.npy", pr)):
            mt = stitched[key][:, OURS_TO_MT]
            mt[:, 43:47] /= np.maximum(np.linalg.norm(mt[:, 43:47], axis=1, keepdims=True), 1e-8)
            np.save(d / fname, mt)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        x = [r["anchor"] / 30.0 for r in per_anchor]
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        for k in ("B-A", "C-A", "E-B", "B'-B", "D-C", "B2-B"):
            axes[0].plot(x, [r[k]["body"] for r in per_anchor], label=k)
        axes[0].set_ylabel("body MAE (deg)")
        axes[0].legend(ncol=6, fontsize=8)
        for g in ("legs", "waist", "L_arm", "R_arm", "wrists"):
            axes[1].plot(x, [r["state_gap"][g] for r in per_anchor], label=g)
        axes[1].set_ylabel("robot vs data state (deg)")
        axes[1].legend(ncol=5, fontsize=8)
        for g in ("hands_rad",):
            for k in ("B-A", "C-A", "E-B"):
                axes[2].plot(x, [r[k][g] for r in per_anchor], label=f"{k} {g}")
        axes[2].set_ylabel("hands (rad)")
        axes[2].set_xlabel("anchor (s)")
        axes[2].legend(ncol=3, fontsize=8)
        fig.suptitle(f"test ep{args.test_episode} ({meta['episode_id']}) vs {args.data_key}")
        fig.tight_layout()
        fig.savefig(out_dir / "compare.png", dpi=110)
    except Exception as exc:  # noqa: BLE001 - the numbers are written already
        print(f"[plot] skipped: {exc}", flush=True)

    print(json.dumps({k: {g: round(v, 2) for g, v in m.items()} for k, m in mean.items()}, indent=1), flush=True)
    print(f"done: {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
