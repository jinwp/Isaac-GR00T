"""MotionTokenizers-compatible frame-to-frame delta helpers for VLA actions.

Matches ``third_party/MotionTokenizers`` ``include_delta`` REPLACE semantics:
plain joint diffs, wrap-aware camera diffs, relative quaternion deltas.
Velocities are left unchanged by callers (already a rate).
"""

from __future__ import annotations

import numpy as np


def plain_diff(x: np.ndarray) -> np.ndarray:
    """Frame-to-frame plain diff; row 0 = 0."""
    x = np.asarray(x)
    d = np.zeros_like(x, dtype=np.float32)
    if x.shape[0] > 1:
        d[1:] = (x[1:] - x[:-1]).astype(np.float32, copy=False)
    return d


def wrapped_angle_diff(x: np.ndarray) -> np.ndarray:
    """Frame-to-frame angle diff mapped to (−π, π]; row 0 = 0."""
    x = np.asarray(x, dtype=np.float64)
    d = np.zeros_like(x, dtype=np.float64)
    if x.shape[0] > 1:
        d[1:] = x[1:] - x[:-1]
    return ((d + np.pi) % (2.0 * np.pi) - np.pi).astype(np.float32)


def relative_quat_wxyz(q: np.ndarray) -> np.ndarray:
    """Per-frame relative rotation ``q_{t−1}⁻¹ ⊗ q_t`` (wxyz); row 0 = identity."""
    q = np.asarray(q, dtype=np.float64)
    q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    out = np.zeros_like(q, dtype=np.float64)
    out[0] = [1.0, 0.0, 0.0, 0.0]
    if q.shape[0] <= 1:
        return out.astype(np.float32)
    w0, x0, y0, z0 = q[:-1, 0], q[:-1, 1], q[:-1, 2], q[:-1, 3]
    w1, x1, y1, z1 = q[1:, 0], q[1:, 1], q[1:, 2], q[1:, 3]
    rel = np.stack(
        [
            w0 * w1 + x0 * x1 + y0 * y1 + z0 * z1,
            w0 * x1 - x0 * w1 - y0 * z1 + z0 * y1,
            w0 * y1 + x0 * z1 - y0 * w1 - z0 * x1,
            w0 * z1 - x0 * y1 + y0 * x1 - z0 * w1,
        ],
        axis=1,
    )
    rel = np.where(rel[:, :1] < 0.0, -rel, rel)
    out[1:] = rel
    return out.astype(np.float32)


def apply_delta_to_action_arrays(
    *,
    measured_joint_q: np.ndarray,
    hand_action: np.ndarray,
    camera_action: np.ndarray,
    body_quat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """REPLACE absolute action channels with MT-style deltas (vel unchanged)."""
    return (
        plain_diff(measured_joint_q),
        plain_diff(hand_action),
        wrapped_angle_diff(camera_action),
        relative_quat_wxyz(body_quat),
    )


def _quat_inv_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    out = np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)
    n = float(np.linalg.norm(out))
    return out / max(n, 1e-12)


def _quat_mul_wxyz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = np.asarray(a, dtype=np.float64).reshape(4)
    bw, bx, by, bz = np.asarray(b, dtype=np.float64).reshape(4)
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=np.float64,
    )


def preceding_abs_from_obs_and_first_delta(
    abs_obs: np.ndarray,
    first_delta: np.ndarray,
    mt_spec: object,
) -> np.ndarray:
    """Recover abs[t-1] from abs[t] and REPLACE-delta row Δ_t (HA-Lab proprio @ t)."""
    abs_obs = np.asarray(abs_obs, dtype=np.float32).reshape(-1)
    first_delta = np.asarray(first_delta, dtype=np.float32).reshape(-1)
    if abs_obs.shape != first_delta.shape:
        raise ValueError(
            f"abs_obs dim {abs_obs.shape} != first_delta dim {first_delta.shape}"
        )
    out = abs_obs.copy()
    pos_sl = mt_spec.pos_slice  # type: ignore[attr-defined]
    out[pos_sl] = abs_obs[pos_sl] - first_delta[pos_sl]
    if int(getattr(mt_spec, "camera_width", 0)) > 0:
        cam_sl = mt_spec.camera_slice  # type: ignore[attr-defined]
        out[cam_sl] = abs_obs[cam_sl] - first_delta[cam_sl]
    quat_sl = mt_spec.root_quat_slice  # type: ignore[attr-defined]
    out[quat_sl] = _quat_mul_wxyz(
        abs_obs[quat_sl], _quat_inv_wxyz(first_delta[quat_sl])
    ).astype(np.float32)
    return out
