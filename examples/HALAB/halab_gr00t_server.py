#!/usr/bin/env python3
"""Serve a GR00T N1.7 HA-Lab checkpoint over ZMQ for samsung_deploy's ``--policy gr00t``.

The model runs here, in the Isaac-GR00T environment; the deploy client (``vla_deploy/policy/gr00t.py``,
OpenHLM / samsung_deploy environment) only needs ``zmq`` and ``numpy``.  Inference is the exact
path of ``halab_kinematic_play.py``: ``Gr00tPolicy`` processor -> ``get_action`` (bf16) ->
``decode_action``.

Protocol (REQ/REP, two frames: JSON header, ``np.savez`` payload; never pickle):

* ``{"cmd": "info"}`` -> header with chunk length, fps, action keys, denoising steps and the
  measured inference latency;
* ``{"cmd": "infer", "text": <instruction>}`` + arrays ``image`` (H, W, 3) uint8 and the five
  state groups ``body_q, body_dq, base_quat, inspire_hand, camera`` (1, D) float32 (what
  ``_split_observation_state`` returns) -> header ``{"infer_s"}`` + array ``actions`` (H, 47) in
  our order ``[body_q 29 | base_quat 4 | inspire_hand 12 (rad) | camera 2]``, H the
  checkpoint's chunk length (32 for v1-v4, read from its processor_config.json), and
  ``actions_norm`` (H, 47), the same rows as the model outputs them (normalized).

With ``--rtc-max-delay N`` (checkpoints trained with training-time RTC, N as in training) an
``infer`` may also carry ``prefix`` (d, 47): the rows committed by the previous chunk, taken
from its ``actions_norm``; they are pinned as in ``halab_rtc.py``.

    python examples/HALAB/halab_gr00t_server.py \\
        --checkpoint /coc/flash2/jpark984/gr00t_baseline/outputs/<v2 run>/checkpoint-20000 --port 5590
"""

from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import zmq

STATE_KEYS = ("body_q", "body_dq", "base_quat", "inspire_hand", "camera")
STATE_DIMS = (29, 29, 4, 12, 2)
ACTION_KEYS = ("body_q", "base_quat", "inspire_hand", "camera")
ACTION_DIM = 47


def pack(header: dict, arrays: dict[str, np.ndarray] | None = None) -> list[bytes]:
    buf = io.BytesIO()
    np.savez(buf, **(arrays or {}))
    return [json.dumps(header).encode(), buf.getvalue()]


def unpack(frames: list[bytes]) -> tuple[dict, dict[str, np.ndarray]]:
    header = json.loads(frames[0])
    arrays = dict(np.load(io.BytesIO(frames[1]), allow_pickle=False)) if len(frames) > 1 else {}
    return header, arrays


def checkpoint_horizon(checkpoint: str) -> int:
    """Rows per chunk the checkpoint was trained with (its saved new_embodiment modality config)."""

    kwargs = json.loads((Path(checkpoint) / "processor_config.json").read_text())["processor_kwargs"]
    return len(kwargs["modality_configs"]["new_embodiment"]["action"]["delta_indices"])


class Gr00tHalabModel:
    def __init__(self, checkpoint: str, device: str, denoising_steps: int, rtc_max_delay: int = 0) -> None:
        import torch

        self.horizon = checkpoint_horizon(checkpoint)
        os.environ["HALAB_ACTION_HORIZON"] = str(self.horizon)  # read by halab_47d_config below
        here = Path(__file__).resolve().parent
        sys.path.insert(0, str(here))
        import importlib.util

        spec = importlib.util.spec_from_file_location("halab_47d_config", here / "halab_47d_config.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # registers the 47D modality config

        self.rtc_max_delay, self.rtc_options = int(rtc_max_delay), None
        if self.rtc_max_delay:
            import halab_rtc

            halab_rtc.install(max_delay=self.rtc_max_delay)
            self.rtc_options = halab_rtc.rtc_options

        from gr00t.data.types import EmbodimentTag, MessageType, VLAStepData
        from gr00t.policy.gr00t_policy import Gr00tPolicy, _rec_to_dtype

        self.torch, self.message_type, self.step_type, self.to_dtype = torch, MessageType, VLAStepData, _rec_to_dtype
        self.tag = EmbodimentTag.NEW_EMBODIMENT
        self.checkpoint = str(Path(checkpoint).resolve())
        self.policy = Gr00tPolicy(self.tag, self.checkpoint, device=device)
        self.processor, self.model = self.policy.processor, self.policy.model
        self.processor.eval()
        default_steps = int(self.model.action_head.num_inference_timesteps)
        if denoising_steps:
            self.model.action_head.num_inference_timesteps = int(denoising_steps)
        self.denoising_steps = int(self.model.action_head.num_inference_timesteps)
        print(f"[gr00t_server] {self.checkpoint}: ODE steps {self.denoising_steps} (checkpoint {default_steps})", flush=True)

    def infer(
        self, image: np.ndarray, states: dict[str, np.ndarray], text: str, prefix: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        torch = self.torch
        options = None
        if prefix is not None:
            if self.rtc_options is None:
                raise ValueError("got a prefix, but the server runs without --rtc-max-delay")
            if prefix.ndim != 2 or prefix.shape[1] != ACTION_DIM or not 1 <= prefix.shape[0] < self.horizon:
                raise ValueError(f"prefix must be (d, {ACTION_DIM}) with 1 <= d < {self.horizon}, got {prefix.shape}")
            target = torch.zeros((1, self.horizon, self.model.action_head.action_dim))  # padded dims stay 0, as in training
            target[0, : prefix.shape[0], :ACTION_DIM] = torch.from_numpy(np.asarray(prefix, dtype=np.float32))
            options = self.rtc_options(target, prefix.shape[0])
        obs = self.step_type(images={"ego_view": [image]}, states=states, actions={}, text=text, embodiment=self.tag)
        processed = [self.processor([{"type": self.message_type.EPISODE_STEP.value, "content": obs}])]
        collated = self.to_dtype(self.policy.collate_fn(processed), dtype=torch.bfloat16)
        with torch.inference_mode():
            normalized = self.model.get_action(**collated, options=options)["action_pred"].float().cpu().numpy()
        decoded = self.processor.decode_action(normalized, self.tag, {k: v[None] for k, v in states.items()})
        actions = np.concatenate([decoded[k] for k in ACTION_KEYS], axis=-1)[0].astype(np.float32)
        return actions, normalized[0, :, :ACTION_DIM].astype(np.float32)


def check_request(arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    image = arrays["image"]
    if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
        raise ValueError(f"image must be (H, W, 3) uint8, got {image.shape} {image.dtype}")
    states = {}
    for key, dim in zip(STATE_KEYS, STATE_DIMS, strict=True):
        value = np.asarray(arrays[key], dtype=np.float32)
        if value.shape != (1, dim):
            raise ValueError(f"state {key} must be (1, {dim}), got {value.shape}")
        states[key] = value
    return image, states


def measure_latency(model: Gr00tHalabModel, runs: int) -> float:
    image = np.zeros((376, 672, 3), dtype=np.uint8)
    states = {k: np.zeros((1, d), dtype=np.float32) for k, d in zip(STATE_KEYS, STATE_DIMS, strict=True)}
    states["base_quat"][0, 0] = 1.0
    model.infer(image, states, "warm up")  # first call pays CUDA / kernel setup
    if model.rtc_max_delay:
        model.infer(image, states, "warm up", np.zeros((1, ACTION_DIM), dtype=np.float32))  # fails here, not on the robot
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        model.infer(image, states, "warm up")
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--port", type=int, default=5590)
    parser.add_argument("--bind", default="127.0.0.1", help="interface to listen on (0.0.0.0 for another machine)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--denoising-steps", type=int, default=4, help="ODE steps (4 = what every eval used)")
    parser.add_argument("--latency-runs", type=int, default=10)
    parser.add_argument(
        "--rtc-max-delay", type=int, default=0,
        help="the training run's --rtc-max-delay (10 for v4 / v5 / halab38): enables the RTC prefix; 0 = off",
    )
    args = parser.parse_args()

    model = Gr00tHalabModel(args.checkpoint, args.device, args.denoising_steps, args.rtc_max_delay)
    latency = measure_latency(model, args.latency_runs)
    print(f"[gr00t_server] inference latency (median of {args.latency_runs}): {latency * 1000:.0f} ms", flush=True)
    info = {
        "checkpoint": model.checkpoint,
        "chunk_len": model.horizon,
        "source_fps": 30.0,
        "action_keys": list(ACTION_KEYS),
        "state_keys": list(STATE_KEYS),
        "denoising_steps": model.denoising_steps,
        "infer_s": latency,
        "rtc_max_delay": model.rtc_max_delay,
    }

    sock = zmq.Context.instance().socket(zmq.REP)
    sock.bind(f"tcp://{args.bind}:{args.port}")
    print(f"[gr00t_server] listening on tcp://{args.bind}:{args.port}", flush=True)
    n = 0
    while True:
        header, arrays = unpack(sock.recv_multipart())
        try:
            if header.get("cmd") == "info":
                sock.send_multipart(pack(info))
                continue
            if header.get("cmd") != "infer":
                raise ValueError(f"unknown cmd {header.get('cmd')!r}")
            image, states = check_request(arrays)
            t0 = time.perf_counter()
            actions, actions_norm = model.infer(image, states, str(header["text"]), arrays.get("prefix"))
            took = time.perf_counter() - t0
            sock.send_multipart(pack({"infer_s": took}, {"actions": actions, "actions_norm": actions_norm}))
            n += 1
            if n == 1 or n % 100 == 0:
                print(f"[gr00t_server] request {n}: {took * 1000:.0f} ms, text {header['text']!r}", flush=True)
        except Exception as exc:  # noqa: BLE001 - the client must get an answer or its REQ socket wedges
            sock.send_multipart(pack({"error": f"{type(exc).__name__}: {exc}"}))
            print(f"[gr00t_server] error: {exc}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
