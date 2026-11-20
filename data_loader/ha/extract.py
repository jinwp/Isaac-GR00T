from __future__ import annotations

from typing import Any

import numpy as np

from .constants import (
    DYNAMIXEL_DIM,
    G1_MOTOR_DIM,
    INSPIRE_FORCE_LEFT,
    INSPIRE_FORCE_RIGHT,
    INSPIRE_LEFT_JOINTS,
    INSPIRE_RIGHT_JOINTS,
    INSPIRE_Q_DIM,
    MEASURED_ACTION_DIM,
    POSE_DIM,
    PROPRIO_SLICES,
)


def _vec(values: Any, size: int, default: float = 0.0) -> np.ndarray:
    out = np.full(size, default, dtype=np.float32)
    if values is None:
        return out
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    n = min(size, arr.shape[0])
    out[:n] = arr[:n]
    return out


def _optional(record: dict[str, Any] | None) -> dict[str, Any]:
    return record if isinstance(record, dict) else {}


def stamp_ns_of(record: dict[str, Any] | None) -> int | None:
    if not record:
        return None
    if "stamp_ns" in record:
        return int(record["stamp_ns"])
    if "timestamp_ns" in record:
        return int(record["timestamp_ns"])
    return None


def times_sec(stamps: list[int], start_ns: int) -> list[float]:
    return [(int(s) - int(start_ns)) * 1e-9 for s in stamps]


def dedupe_by_stamp(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep last sample per stamp_ns (inspire_state relay can duplicate)."""
    by_stamp: dict[int, dict[str, Any]] = {}
    for record in records:
        stamp = stamp_ns_of(record)
        if stamp is None:
            continue
        by_stamp[int(stamp)] = record
    return [by_stamp[k] for k in sorted(by_stamp)]


def extract_g1_fields(record: dict[str, Any] | None) -> dict[str, np.ndarray]:
    p = _optional(record)
    names = p.get("motor_names") or []
    out: dict[str, np.ndarray] = {
        "motor_q": _vec(p.get("motor_q"), G1_MOTOR_DIM),
        "motor_dq": _vec(p.get("motor_dq"), G1_MOTOR_DIM),
        "motor_tau_est": _vec(p.get("motor_tau_est"), G1_MOTOR_DIM),
        "base_quat_wxyz": _vec(p.get("base_quat_wxyz"), 4),
        "base_angular_velocity": _vec(p.get("base_angular_velocity"), 3),
        "base_linear_acceleration": _vec(p.get("base_linear_acceleration"), 3),
        "torso_quat_wxyz": _vec(p.get("torso_quat_wxyz"), 4),
        "torso_angular_velocity": _vec(p.get("torso_angular_velocity"), 3),
        "torso_linear_acceleration": _vec(p.get("torso_linear_acceleration"), 3),
    }
    if names:
        out["_motor_names"] = np.array(names, dtype=object)
    return out


def extract_inspire_fields(record: dict[str, Any] | None) -> dict[str, np.ndarray]:
    p = _optional(record)
    left_q = _vec(p.get("left_q"), 6)
    right_q = _vec(p.get("right_q"), 6)
    out = {
        "left_q": left_q,
        "right_q": right_q,
        "inspire_q": np.concatenate([left_q, right_q]),
        "left_force_raw": _vec(p.get("left_force_raw"), 6),
        "right_force_raw": _vec(p.get("right_force_raw"), 6),
        "left_force_N": _vec(p.get("left_force_N") or p.get("left_force"), 6),
        "right_force_N": _vec(p.get("right_force_N") or p.get("right_force"), 6),
    }
    return out


def extract_dynamixel_fields(record: dict[str, Any] | None) -> dict[str, np.ndarray]:
    p = _optional(record)
    if p.get("yaw_rad_measured") is not None and p.get("pitch_rad_measured") is not None:
        state = _vec([p["yaw_rad_measured"], p["pitch_rad_measured"]], DYNAMIXEL_DIM)
    else:
        state = _vec(p.get("yaw_pitch") or p.get("head_yaw_pitch") or p.get("position"), DYNAMIXEL_DIM)
    return {"yaw_pitch": state}


def extract_imu_fields(record: dict[str, Any] | None) -> dict[str, np.ndarray]:
    p = _optional(record)
    return {
        "angular_velocity": _vec(p.get("angular_velocity"), 3),
        "linear_acceleration": _vec(p.get("linear_acceleration"), 3),
        "orientation": _vec(p.get("orientation"), 4),
    }


def extract_pose_fields(record: dict[str, Any] | None) -> dict[str, np.ndarray]:
    p = _optional(record)
    return {
        "position": _vec(p.get("position"), 3),
        "orientation_xyzw": _vec(p.get("orientation_xyzw"), 4),
        "pose7": np.concatenate([_vec(p.get("position"), 3), _vec(p.get("orientation_xyzw"), 4)]),
    }


def extract_inspire_action(record: dict[str, Any] | None) -> dict[str, np.ndarray]:
    p = _optional(record)
    left = _vec(p.get("left_hand_desired"), 6)
    right = _vec(p.get("right_hand_desired"), 6)
    return {"left_hand_desired": left, "right_hand_desired": right, "inspire_action": np.concatenate([left, right])}


def extract_dynamixel_action(record: dict[str, Any] | None) -> dict[str, np.ndarray]:
    p = _optional(record)
    return {"head_yaw_pitch": _vec(p.get("head_yaw_pitch"), DYNAMIXEL_DIM)}


def extract_measured_action(record: dict[str, Any] | None) -> np.ndarray:
    p = _optional(record)
    return np.concatenate(
        [
            _vec(p.get("body_q"), G1_MOTOR_DIM),
            _vec(p.get("body_dq"), G1_MOTOR_DIM),
            _vec(p.get("body_quat_w"), 4),
            _vec(p.get("base_ang_vel"), 3),
        ],
        axis=0,
    ).astype(np.float32, copy=False)


def extract_proprio_vector(obs: dict[str, Any]) -> np.ndarray:
    g1 = extract_g1_fields(obs.get("g1"))["motor_q"]
    inspire = extract_inspire_fields(obs.get("inspire_state"))["inspire_q"]
    dynamixel = extract_dynamixel_fields(obs.get("dynamixel_state"))["yaw_pitch"]
    return np.concatenate([g1, inspire, dynamixel], axis=0)


def split_proprio(proprio: np.ndarray) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for name, (lo, hi) in PROPRIO_SLICES.items():
        out[name] = proprio[..., lo:hi] if proprio.ndim > 1 else proprio[lo:hi]
    return out


def measured_action_labels() -> list[str]:
    return (
        [f"body_q_{i}" for i in range(G1_MOTOR_DIM)]
        + [f"body_dq_{i}" for i in range(G1_MOTOR_DIM)]
        + [f"body_quat_wxyz_{i}" for i in range(4)]
        + [f"base_ang_vel_{i}" for i in range(MEASURED_ACTION_DIM - 2 * G1_MOTOR_DIM - 4)]
    )


def inspire_joint_labels() -> list[str]:
    return INSPIRE_LEFT_JOINTS + INSPIRE_RIGHT_JOINTS


def inspire_force_labels() -> list[str]:
    return INSPIRE_FORCE_LEFT + INSPIRE_FORCE_RIGHT


def pose_labels() -> list[str]:
    return ["px", "py", "pz", "qx", "qy", "qz", "qw"]
