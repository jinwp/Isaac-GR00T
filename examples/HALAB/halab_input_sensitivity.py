#!/usr/bin/env python3
"""How much a GR00T HA-Lab checkpoint reacts to the image vs the joint state (copycat check).

At anchors ``t`` of recorded episodes the model sees the recorded image and state at ``t``
(``base``); then one input is replaced by the one from ``t + shift``:

* ``image``: image at ``t + shift``, state at ``t`` -- what the image alone changes;
* ``black``: a black image, state at ``t`` -- what removing the image changes;
* ``state``: image at ``t``, state at ``t + shift`` -- what the state alone changes;
* ``noise``: ``base`` with another noise seed -- the sampling floor.

Each change is the mean |difference| of the predicted chunk (body in deg, hands in rad), and is
also divided by how much the recorded actions really differ between ``t`` and ``t + shift``
(``ratio``): a model that reads the scene moves its output with the image (``image`` ratio near
1), a copycat follows the state (``state`` ratio near 1, ``image`` near 0).  All variants of one
anchor share one noise seed, so the differences come from the inputs.

    python examples/HALAB/halab_input_sensitivity.py --checkpoints <run>/keep-10000 <run>/checkpoint-50000 \\
        --data-parent <halab root> --episode-keys task/3 task/36 ... --out <outputs>/input_sensitivity_50k.json
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

DEG = 180.0 / np.pi
VARIANTS = ("image", "black", "state", "noise")


def diffs(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Mean |a - b| of two (H, 47) chunks in our order: body (deg), hands (rad)."""
    return float(np.abs(a[:, :29] - b[:, :29]).mean() * DEG), float(np.abs(a[:, 33:45] - b[:, 33:45]).mean())


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoints", nargs="+", required=True)
    p.add_argument("--data-parent", required=True)
    p.add_argument("--episode-keys", nargs="+", required=True, help="'<task>/<episode_index>' keys")
    p.add_argument("--stride", type=int, default=60, help="frames between anchors")
    p.add_argument("--shift", type=int, default=90, help="frames between t and the swapped-in input (3 s)")
    p.add_argument("--denoising-steps", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", required=True, help="summary JSON")
    args = p.parse_args()

    import torch

    horizons = {srv.checkpoint_horizon(c) for c in args.checkpoints}
    if len(horizons) != 1:
        raise SystemExit(f"checkpoints have different chunk lengths {sorted(horizons)}: run them separately")
    os.environ["HALAB_ACTION_HORIZON"] = str(horizons.pop())  # read when the adapter is imported
    from halab_gr00t_adapter import HalabGr00tAdapter, build_halab_dataset

    dataset = build_halab_dataset(data_parent=args.data_parent, episode_keys=args.episode_keys, load_images=True)
    pairs = []  # (t step, t + shift step)
    for start, stop in dataset.episode_anchor_ranges():
        for t in range(0, stop - start - args.shift, args.stride):
            pairs.append((start + t, start + t + args.shift))
    print(f"{len(pairs)} anchors from {len(args.episode_keys)} episode(s)", flush=True)
    steps = {}

    def step(i: int):
        if i not in steps:
            steps[i] = HalabGr00tAdapter.to_vla_step(dataset[i])
        return steps[i]

    from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS

    results = {}
    for ckpt in args.checkpoints:
        MODALITY_CONFIGS.pop("new_embodiment", None)  # each model load registers the 47D config again
        model = srv.Gr00tHalabModel(ckpt, args.device, args.denoising_steps)

        def infer(image, states, text, seed):
            torch.manual_seed(seed)
            return model.infer(image, states, text)[0]

        sums = {v: [] for v in VARIANTS}
        ratios = {v: [] for v in VARIANTS}
        for n, (i, j) in enumerate(pairs):
            s0, s1 = step(i), step(j)
            img0, img1 = np.asarray(s0.images["ego_view"])[-1], np.asarray(s1.images["ego_view"])[-1]
            st0 = {k: np.asarray(v, np.float32) for k, v in s0.states.items()}
            st1 = {k: np.asarray(v, np.float32) for k, v in s1.states.items()}
            base = infer(img0, st0, s0.text, n)
            out = {
                "image": infer(img1, st0, s0.text, n),
                "black": infer(np.zeros_like(img0), st0, s0.text, n),
                "state": infer(img0, st1, s0.text, n),
                "noise": infer(img0, st0, s0.text, n + 100000),
            }
            gt0 = np.concatenate([s0.actions[k] for k in srv.ACTION_KEYS], axis=-1)
            gt1 = np.concatenate([s1.actions[k] for k in srv.ACTION_KEYS], axis=-1)
            true_body, true_hand = diffs(gt1, gt0)
            for v, o in out.items():
                body, hand = diffs(o, base)
                sums[v].append((body, hand))
                ratios[v].append((body / max(true_body, 1e-6), hand / max(true_hand, 1e-3) if true_hand > 0.05 else np.nan))
            if n % 50 == 0:
                print(f"[{Path(ckpt).name}] {n + 1}/{len(pairs)}", flush=True)
        res = {}
        for v in VARIANTS:
            a, r = np.array(sums[v]), np.array(ratios[v])
            res[v] = {"body_deg": float(a[:, 0].mean()), "hands_rad": float(a[:, 1].mean()),
                      "body_ratio": float(np.mean(r[:, 0])), "hands_ratio": float(np.nanmean(r[:, 1])) if np.isfinite(r[:, 1]).any() else None}
        results[str(ckpt)] = res
        print(f"== {ckpt}", flush=True)
        for v in VARIANTS:
            x = res[v]
            hr = "n/a" if x["hands_ratio"] is None else f"{x['hands_ratio']:.2f}"
            print(f"   {v:6} body {x['body_deg']:6.2f} deg (ratio {x['body_ratio']:.2f})   hands {x['hands_rad']:.3f} rad (ratio {hr})", flush=True)
        del model
        torch.cuda.empty_cache()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"episode_keys": args.episode_keys, "stride": args.stride, "shift": args.shift,
                                          "n_anchors": len(pairs), "results": results}, indent=2))
    print(f"done: {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
