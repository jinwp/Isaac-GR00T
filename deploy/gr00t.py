"""GR00T N1.7 HA-Lab baseline behind the shared policy interface (model served by Isaac-GR00T).

    # Isaac-GR00T venv, same machine:
    python examples/HALAB/halab_gr00t_server.py --checkpoint <v2 run>/checkpoint-20000 --port 5590
    # deploy env:
    run_vla_sonic.py --policy gr00t --policy-arg server=tcp://127.0.0.1:5590 --task "<training instruction>" \\
        --rtc-inference-delay D --rtc-replan-interval D

The checkpoint: 76-D state ``[q 29 | dq 29 | quat 4 | hand 12 | cam 2]`` (``_split_observation_state``,
the training loader's own split, hands in radians), one ego image (the raw 376x672 ZED frame), the
instruction verbatim; H absolute rows at 30 Hz of ``[q 29 | quat 4 | hand 12 (rad) | cam 2]``
(H = 32 for v1-v4, the server reports it).
RTC prefix: a server started with ``--rtc-max-delay N`` (training-time RTC checkpoints) reports it and
``supports_prefix`` turns on; ``chunk_norm`` is the model's normalized chunk, which comes back as the next
prefix. Otherwise ``supports_prefix=False`` like ``openhlm_pytorch``.

What happens to the chunk here, none of it a retrain:

* **padding to ``chunk_len`` (default: the model's chunk, at least 40)**: the last predicted row is
  held. SONIC needs 46 buffered 50 Hz frames, i.e. ``chunk_len >= delay + replan + 29`` rows; 32 rows
  leave ``delay + replan <= 3``.
  Only rows ``[delay, delay + replan)`` of a chunk are ever executed (all inside the 32 predicted
  rows when ``delay + replan <= 32``); the held rows only fill SONIC's far lookahead, and are
  executed only when a replan is late -- the robot then stops at the last predicted pose;
* **splice cross-fade** (replans without a prefix only): successive chunks disagree at the splice row
  ``anchor + delay``. The first ``splice_blend_rows`` executed rows are faded from the previous
  sent chunk into the new one (smoothstep; the root quaternion nlerped);
* **head pitch x ``pitch_scale`` (2.0)**: the camera action is the Pico command, which the Orin
  scaled by 2 in pitch at recording time (HA-Lab data: measured / commanded pitch 1.82-1.98, yaw
  0.92-1.00); the deploy head target is in measured units;
* as ``openhlm_pytorch``: head rate limit (``head_max_step``), first chunk eased from the
  measured body pose (``start_blend_rows``), optional arm offset compensation (``arm_offset_comp``).
"""

from __future__ import annotations

import io
import json
import time
from typing import Any

import numpy as np

from data_loader.ha.vla_episode_io import _split_observation_state

from ..mt import layout as mt_layout, mt_dim
from .base import PolicyError, PolicySpec, PredictRequest, PredictResult, VlaPolicy
from .openhlm_pytorch import blend_start_rows, normalize_quat_rows, rate_limit_columns

DEFAULT_CHUNK_LEN = 40  # rows sent when the model's chunk is shorter (32-row checkpoints are held to it)
SOURCE_FPS = 30.0
# Our action order [q 0:29 | quat 29:33 | hand 33:45 | cam 45:47] -> MT [q | hand | cam | quat].
OURS_TO_MT = np.r_[0:29, 33:45, 45:47, 29:33]
STATE_KEYS = {  # model state group <- _split_observation_state key
    "body_q": "measured_joint_q",
    "body_dq": "measured_joint_dq",
    "base_quat": "body_quat",
    "inspire_hand": "hand_action",
    "camera": "camera_action",
}
_MT = mt_layout(False)  # joint_q 0:29 | hand 29:41 | camera 41:43 | quat 43:47
MT_CAMERA, MT_QUAT = _MT["camera"], _MT["quat"]
MT_ARMS = slice(15, 29)
ARM_OFFSET_MODES = ("none", "command")


def _pack(header: dict, arrays: dict[str, np.ndarray] | None = None) -> list[bytes]:
    buf = io.BytesIO()
    np.savez(buf, **(arrays or {}))
    return [json.dumps(header).encode(), buf.getvalue()]


def _unpack(frames: list[bytes]) -> tuple[dict, dict[str, np.ndarray]]:
    header = json.loads(frames[0])
    arrays = dict(np.load(io.BytesIO(frames[1]), allow_pickle=False)) if len(frames) > 1 else {}
    return header, arrays


def crossfade_rows(prev: np.ndarray, new: np.ndarray) -> np.ndarray:
    """Rows going from ``prev`` (row 0 almost all ``prev``) to ``new`` (smoothstep); quaternion nlerped."""
    n = prev.shape[0]
    t = (np.arange(n, dtype=np.float32) + 1.0) / float(n + 1)
    w = (t * t * (3.0 - 2.0 * t))[:, None]
    out = prev + w * (new - prev)
    q0, q1 = prev[:, MT_QUAT], new[:, MT_QUAT]
    q1 = np.where(np.sum(q0 * q1, axis=1, keepdims=True) < 0.0, -q1, q1)  # same hemisphere
    out[:, MT_QUAT] = normalize_quat_rows(q0 + w * (q1 - q0))
    return out.astype(np.float32)


class Gr00tHalabPolicy(VlaPolicy):
    def __init__(
        self,
        *,
        language: str,
        server: str,
        chunk_len: int | str | None = None,
        pitch_scale: float | str = 2.0,
        yaw_scale: float | str = 1.0,
        splice_blend_rows: int | str = 4,
        head_max_step: float | str = 0.10,
        start_blend_rows: int | str = 15,
        arm_offset_comp: str = "none",
        arm_resync_rad: float | str = 0.35,
        arm_offset_tau_s: float | str = 3.0,
        arm_offset_max_rad: float | str = 0.12,
        arm_offset_lag_rows: int | str = 6,
        arm_offset_static_rad: float | str = 0.03,
        timeout_ms: int | str = 5000,
        joint_units: str = "legacy",  # mjcf_v1 for runs on the 38-task data; legacy = raw units of the 12-task runs
        client: Any = None,  # frame-server client; unused (the producer fetches frames)
        **_ignored: object,
    ) -> None:
        import zmq

        self.language = str(language)
        self.pitch_scale, self.yaw_scale = float(pitch_scale), float(yaw_scale)
        self.splice_blend_rows = int(splice_blend_rows)
        self.head_max_step = float(head_max_step)
        self.start_blend_rows = int(start_blend_rows)
        # arm_offset_comp=command (openhlm_pytorch): EMA of (sent - measured) arm on quasi-static replans, added to the
        # sent arm rows, so the arm PD's steady error does not become a drift of a prefix-less absolute policy.
        self.arm_offset_comp = str(arm_offset_comp)
        if self.arm_offset_comp not in ARM_OFFSET_MODES:
            raise PolicyError(f"arm_offset_comp must be one of {ARM_OFFSET_MODES}, got {arm_offset_comp!r}")
        self.arm_resync_rad, self.arm_offset_tau_s = float(arm_resync_rad), float(arm_offset_tau_s)
        self.arm_offset_max_rad, self.arm_offset_lag_rows = float(arm_offset_max_rad), int(arm_offset_lag_rows)
        self.arm_offset_static_rad = float(arm_offset_static_rad)

        self._zmq, self._server, self._timeout_ms = zmq, str(server), int(timeout_ms)
        self._sock = None
        info, _ = self._request({"cmd": "info"})
        if float(info["source_fps"]) != SOURCE_FPS:
            raise PolicyError(f"server serves rows at {info['source_fps']} Hz, expected {SOURCE_FPS:g}")
        self.model_chunk = int(info["chunk_len"])  # 32 (v1-v4) or the checkpoint's longer horizon
        self.chunk_len = int(chunk_len) if chunk_len not in (None, "") else max(self.model_chunk, DEFAULT_CHUNK_LEN)
        if self.chunk_len < self.model_chunk:
            raise PolicyError(f"chunk_len must be >= the model's {self.model_chunk} rows, got {self.chunk_len}")
        if joint_units not in ("legacy", "mjcf_v1"):
            raise PolicyError(f"joint_units must be legacy or mjcf_v1, got {joint_units!r}")
        self.metadata = {"joint_units": str(joint_units), "server": dict(info)}
        self.rtc_max_delay = int(info.get("rtc_max_delay", 0))
        delay = int(np.ceil((float(info["infer_s"]) + 0.03) * SOURCE_FPS))  # + ~30 ms for transport and the frame fetch
        fed = self.chunk_len - 29  # delay + replan that still keeps SONIC's 46 frames fed
        print(
            f"[gr00t] {info['checkpoint']}: ODE steps {info['denoising_steps']}, inference {float(info['infer_s']) * 1000:.0f} ms "
            f"-> --rtc-inference-delay {delay} --rtc-replan-interval {delay} (chunk_len {self.chunk_len} keeps SONIC fed for "
            f"delay + replan <= {fed}{'' if 2 * delay <= fed else '; NOT met: pass --policy-arg chunk_len=' + str(2 * delay + 29)}); "
            f"RTC prefix {'delay <= ' + str(self.rtc_max_delay - 1) if self.rtc_max_delay >= 2 else 'off'}; "
            f"pitch x{self.pitch_scale:g}, splice blend {self.splice_blend_rows}, arm_offset_comp {self.arm_offset_comp}; "
            f"instruction {self.language!r} (must be a training instruction, verbatim)",
            flush=True,
        )
        self.spec = PolicySpec(
            name="gr00t",
            chunk_len=self.chunk_len,
            action_dim=mt_dim(False),
            source_fps=SOURCE_FPS,
            include_body_vel=False,
            supports_prefix=self.rtc_max_delay >= 2,
            max_prefix_delay=max(self.rtc_max_delay - 1, 0),  # training drew the delay from U{0..max_delay-1}
            num_obs_frames=1,
            description=f"GR00T N1.7 HA-Lab {info['checkpoint']}: {self.model_chunk} rows held to {self.chunk_len}, pitch x{self.pitch_scale:g}",
            action_layout="mt47",
        )
        self.reset()

    # -- transport ----------------------------------------------------------
    def _connect(self) -> None:
        if self._sock is not None:
            self._sock.close(linger=0)
        self._sock = self._zmq.Context.instance().socket(self._zmq.REQ)
        self._sock.setsockopt(self._zmq.RCVTIMEO, self._timeout_ms)
        self._sock.setsockopt(self._zmq.LINGER, 0)
        self._sock.connect(self._server)

    def _request(self, header: dict, arrays: dict[str, np.ndarray] | None = None) -> tuple[dict, dict[str, np.ndarray]]:
        if self._sock is None:
            self._connect()
        try:
            self._sock.send_multipart(_pack(header, arrays))
            reply, out = _unpack(self._sock.recv_multipart())
        except self._zmq.error.Again as exc:
            self._connect()  # a REQ socket without its reply cannot send again
            raise PolicyError(f"GR00T server {self._server} did not answer within {self._timeout_ms} ms") from exc
        if "error" in reply:
            raise PolicyError(f"GR00T server: {reply['error']}")
        return reply, out

    # -- per-replan pieces --------------------------------------------------
    def reset(self) -> None:
        self._n_predict = 0
        self._last_chunk: np.ndarray | None = None  # the chunk as sent (MT, chunk_len rows)
        self._last_anchor: int | None = None
        self._arm_offset = np.zeros(MT_ARMS.stop - MT_ARMS.start, dtype=np.float32)
        self._arm_gap = 0.0

    def _update_arm_offset(self, measured_mt: np.ndarray, anchor: int) -> None:
        self._arm_gap = 0.0
        if self.arm_offset_comp == "none" or self._last_chunk is None or self._last_anchor is None:
            return
        prev, delta = self._last_chunk, anchor - self._last_anchor
        row = int(np.clip(delta, 0, prev.shape[0] - 1))
        measured = measured_mt[MT_ARMS]
        self._arm_gap = float(np.abs(prev[row, MT_ARMS] - measured).max())
        if not np.isfinite(self._arm_gap) or self._arm_gap > self.arm_resync_rad:
            return
        row_est = int(np.clip(row - self.arm_offset_lag_rows, 0, prev.shape[0] - 1))
        moved = float(np.abs(prev[row, MT_ARMS] - prev[0, MT_ARMS]).max())
        if moved <= self.arm_offset_static_rad:
            dt = max(float(delta), 1.0) / SOURCE_FPS
            alpha = dt / (self.arm_offset_tau_s + dt) if self.arm_offset_tau_s > 0 else 1.0
            diff = prev[row_est, MT_ARMS] - measured
            self._arm_offset = np.clip(
                self._arm_offset + alpha * (diff - self._arm_offset), -self.arm_offset_max_rad, self.arm_offset_max_rad
            ).astype(np.float32)

    def _splice_ref(self, anchor: int, delay: int) -> np.ndarray | None:
        """Rows of the previous sent chunk from this chunk's splice row (``anchor + delay``) on."""
        if self._last_chunk is None or self._last_anchor is None:
            return None
        k = anchor + delay - self._last_anchor
        if not 0 <= k < self._last_chunk.shape[0]:
            return None
        return self._last_chunk[k:]

    def predict(self, request: PredictRequest) -> PredictResult:
        started = time.perf_counter()
        obs = request.obs
        anchor = int(obs.anchor_step)
        d = int(getattr(self, "rtc_inference_delay", request.delay))  # set by the producer: the real splice offset
        measured_mt = np.asarray(obs.proprio_mt, dtype=np.float32).reshape(-1)
        self._update_arm_offset(measured_mt, anchor)

        split = _split_observation_state(np.asarray(obs.proprio_state, dtype=np.float32), include_body_vel=True)
        arrays = {key: np.asarray(split[src], dtype=np.float32)[None, :] for key, src in STATE_KEYS.items()}
        image = np.ascontiguousarray(obs.image)
        if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
            raise PolicyError(f"expected an HxWx3 uint8 image, got {image.shape} {image.dtype}")
        arrays["image"] = image
        if request.prefix is not None and request.delay > 0:
            prefix = np.asarray(request.prefix, dtype=np.float32)
            if prefix.shape != (int(request.delay), 47):
                raise PolicyError(f"prefix must be ({request.delay}, 47), got {prefix.shape}")
            arrays["prefix"] = prefix
        reply, out = self._request({"cmd": "infer", "text": self.language}, arrays)
        actions = np.asarray(out["actions"], dtype=np.float32)
        if actions.shape != (self.model_chunk, 47):
            raise PolicyError(f"server returned actions {actions.shape}, expected ({self.model_chunk}, 47)")

        chunk = actions[:, OURS_TO_MT]
        chunk[:, MT_QUAT] = normalize_quat_rows(chunk[:, MT_QUAT])
        chunk[:, MT_CAMERA.start] *= self.yaw_scale
        chunk[:, MT_CAMERA.start + 1] *= self.pitch_scale
        if self.arm_offset_comp == "command":
            chunk[:, MT_ARMS] += self._arm_offset[None, :]
        chunk = np.concatenate([chunk, np.repeat(chunk[-1:], self.chunk_len - self.model_chunk, axis=0)])
        chunk_norm = None
        if self.spec.supports_prefix:
            norm = np.asarray(out["actions_norm"], dtype=np.float32)
            chunk_norm = np.concatenate([norm, np.repeat(norm[-1:], self.chunk_len - self.model_chunk, axis=0)])

        first = self._n_predict == 0
        self._n_predict += 1
        prev = self._splice_ref(anchor, d)
        n_fade = 0
        if prev is not None and self.splice_blend_rows > 0 and request.prefix is None:
            n_fade = min(self.splice_blend_rows, prev.shape[0], self.chunk_len - d)
            chunk[d : d + n_fade] = crossfade_rows(prev[:n_fade], chunk[d : d + n_fade])
        if self.head_max_step > 0:
            if prev is not None:
                ref = prev[0, MT_CAMERA]
            elif obs.commanded_camera is not None and np.size(obs.commanded_camera) >= 2:
                ref = np.asarray(obs.commanded_camera, dtype=np.float32).reshape(-1, 2)[-1]
            else:
                ref = measured_mt[MT_CAMERA]
            chunk[d:] = rate_limit_columns(chunk[d:], ref, self.head_max_step, MT_CAMERA)
        if first and self.start_blend_rows > 0:
            chunk = blend_start_rows(chunk, measured_mt, self.start_blend_rows)
        self._last_chunk, self._last_anchor = chunk.copy(), anchor

        info: dict[str, Any] = {
            "server_infer_s": round(float(reply["infer_s"]), 4),
            "splice_fade_rows": int(n_fade),
            "prefix_rows": 0 if request.prefix is None else int(request.delay),
        }
        if self.arm_offset_comp != "none":
            info["arm_gap"] = round(self._arm_gap, 4)
            info["arm_offset_max"] = round(float(np.abs(self._arm_offset).max()), 4)
        timings = {"total_s": time.perf_counter() - started, "infer_s": float(reply["infer_s"])}
        return self.validate(PredictResult(chunk_abs=chunk, chunk_norm=chunk_norm, timings=timings, info=info))

    def close(self) -> None:
        if self._sock is not None:
            self._sock.close(linger=0)
            self._sock = None
