"""Per-episode aligned arrays for VLA training (30 Hz left-master grid)."""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .constants import G1_MOTOR_DIM
from .extract import (
    extract_dynamixel_action,
    extract_inspire_action,
    extract_measured_action,
    stamp_ns_of,
)
from .loader import LeRobotEpisodeLoader
from .types import EpisodeData

_VLA_RAW_STREAMS = ("g1_debug", "inspire_action", "dynamixel_action", "boxer_output", "bev_output", "left")


def _infer_grid_shape(flat_len: int) -> tuple[int, int]:
    side = int(round(float(flat_len) ** 0.5))
    if side * side == flat_len:
        return side, side
    return 1, flat_len


def _reshape_map_rows(flat: np.ndarray) -> np.ndarray:
    flat = np.asarray(flat, dtype=np.float32)
    if flat.ndim == 1:
        h, w = _infer_grid_shape(int(flat.shape[0]))
        return flat.reshape(h, w)
    if flat.ndim == 2:
        return flat
    raise ValueError(f"expected 1D/2D map, got shape {flat.shape}")


def _pose_row_to_matrix(row: Any) -> np.ndarray:
    arr = np.asarray(row, dtype=np.float32).reshape(-1)
    if arr.size == 16:
        return arr.reshape(4, 4).copy()
    raise ValueError(f"bev_camera_pose_world must be 16 floats (4x4), got {arr.size}")


def _stream_index(records: list[dict[str, Any]]) -> tuple[list[int], list[dict[str, Any]]]:
    ordered = [
        rec
        for rec in sorted(records, key=lambda r: int(stamp_ns_of(r) or 0))
        if stamp_ns_of(rec) is not None
    ]
    return [int(stamp_ns_of(r)) for r in ordered], ordered


def _latest(
    stream_stamps: list[int],
    records: list[dict[str, Any]],
    master: int,
) -> dict[str, Any] | None:
    if not stream_stamps:
        return None
    idx = bisect.bisect_right(stream_stamps, int(master)) - 1
    if idx < 0:
        return None
    return records[idx]


def _split_measured(measured: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    measured = np.asarray(measured, dtype=np.float32).reshape(-1)
    q = measured[:G1_MOTOR_DIM]
    dq = measured[G1_MOTOR_DIM : 2 * G1_MOTOR_DIM]
    quat = measured[2 * G1_MOTOR_DIM : 2 * G1_MOTOR_DIM + 4]
    return q, dq, quat


@dataclass
class EpisodeAlignedStore:
    """Compact 30 Hz tables for one episode (aligned_extras + causal action joins)."""

    episode_index: int
    episode_id: str
    language: str
    length: int
    timestamp_ns: np.ndarray
    frame_index: np.ndarray
    camera_pose_world: np.ndarray
    global_uncertainty: np.ndarray
    local_uncertainty: np.ndarray
    global_grid_shape: tuple[int, int]
    local_grid_shape: tuple[int, int]
    boxer_raw_index: np.ndarray
    bev_raw_index: np.ndarray
    search_target_global: np.ndarray
    search_target_local: np.ndarray
    is_bos: np.ndarray
    target_local_clipped: np.ndarray
    measured_joint_q: np.ndarray
    measured_joint_dq: np.ndarray
    body_quat: np.ndarray
    latent_state: np.ndarray
    hand_action: np.ndarray
    camera_action: np.ndarray
    video_path: Path | None
    raw_dir: Path | None
    image_width: int = 672
    image_height: int = 376
    boxer_output: list[dict[str, Any]] = field(default_factory=list)
    bev_output: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_episode(cls, episode: EpisodeData, *, language: str = "") -> EpisodeAlignedStore:
        extras = episode.aligned_extras
        if not extras or "timestamp_ns" not in extras:
            raise ValueError(f"episode {episode.info.episode_id}: missing aligned_extras")

        t_len = int(extras["num_rows"])
        timestamp_ns = np.asarray(extras["timestamp_ns"], dtype=np.int64).reshape(-1)
        frame_index = np.asarray(extras.get("frame_index", np.arange(t_len)), dtype=np.int64).reshape(-1)

        global_flat = np.asarray(extras["bev_global_uncertainty"], dtype=np.float32)
        local_flat = np.asarray(extras["bev_local_uncertainty"], dtype=np.float32)
        gh, gw = _infer_grid_shape(int(global_flat.shape[1]))
        lh, lw = _infer_grid_shape(int(local_flat.shape[1]))
        global_maps = global_flat.reshape(t_len, gh, gw)
        local_maps = local_flat.reshape(t_len, lh, lw)

        poses = np.stack([_pose_row_to_matrix(row) for row in extras["bev_camera_pose_world"]], axis=0)

        if _has_compact_action_columns(episode):
            q, dq, quat, latent, hand, camera = _build_action_arrays_from_parquet(episode)
        else:
            q, dq, quat, latent, hand, camera = _build_action_arrays(episode, timestamp_ns.tolist())

        lang = language
        if not lang and episode.info.tasks:
            lang = str(episode.info.tasks[0])

        image_width, image_height = 672, 376
        left_records = episode.raw_streams.get("left") or []
        if left_records:
            image_width = int(left_records[0].get("width", image_width))
            image_height = int(left_records[0].get("height", image_height))

        return cls(
            episode_index=int(episode.info.episode_index),
            episode_id=str(episode.info.episode_id),
            language=lang,
            length=t_len,
            timestamp_ns=timestamp_ns,
            frame_index=frame_index,
            camera_pose_world=poses.astype(np.float32),
            global_uncertainty=global_maps,
            local_uncertainty=local_maps,
            global_grid_shape=(gh, gw),
            local_grid_shape=(lh, lw),
            boxer_raw_index=np.asarray(extras["boxer_raw_index"], dtype=np.int32).reshape(-1),
            bev_raw_index=np.asarray(extras["bev_raw_index"], dtype=np.int32).reshape(-1),
            search_target_global=np.stack(
                [
                    np.asarray(extras["bev_target_ix"], dtype=np.int32).reshape(-1),
                    np.asarray(extras["bev_target_iy"], dtype=np.int32).reshape(-1),
                ],
                axis=1,
            ),
            search_target_local=np.stack(
                [
                    np.asarray(extras["bev_target_local_ix"], dtype=np.int32).reshape(-1),
                    np.asarray(extras["bev_target_local_iy"], dtype=np.int32).reshape(-1),
                ],
                axis=1,
            ),
            is_bos=np.asarray(extras["bev_is_bos"], dtype=np.bool_).reshape(-1),
            target_local_clipped=np.asarray(extras["bev_target_local_clipped"], dtype=np.bool_).reshape(-1),
            measured_joint_q=q,
            measured_joint_dq=dq,
            body_quat=quat,
            latent_state=latent,
            hand_action=hand,
            camera_action=camera,
            video_path=episode.video_path,
            raw_dir=episode.raw_dir,
            image_width=image_width,
            image_height=image_height,
            boxer_output=list(episode.raw_streams.get("boxer_output") or []),
            bev_output=list(episode.raw_streams.get("bev_output") or []),
        )


def _build_action_arrays(
    episode: EpisodeData,
    master_stamps: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    indices = {
        stream: _stream_index(episode.raw_streams.get(stream) or [])
        for stream in _VLA_RAW_STREAMS
    }

    q_rows: list[np.ndarray] = []
    dq_rows: list[np.ndarray] = []
    quat_rows: list[np.ndarray] = []
    latent_rows: list[np.ndarray] = []
    hand_rows: list[np.ndarray] = []
    camera_rows: list[np.ndarray] = []

    for master in master_stamps:
        debug = _latest(*indices["g1_debug"], master) or {}
        hand = extract_inspire_action(_latest(*indices["inspire_action"], master))
        cam = extract_dynamixel_action(_latest(*indices["dynamixel_action"], master))
        measured = extract_measured_action(debug.get("measured"))
        q, dq, quat = _split_measured(measured)

        latent = np.zeros(64, dtype=np.float32)
        latent_state = debug.get("latent_state") or {}
        if isinstance(latent_state, dict) and latent_state.get("token_state") is not None:
            arr = np.asarray(latent_state["token_state"], dtype=np.float32).reshape(-1)
            latent[: min(64, arr.shape[0])] = arr[: min(64, arr.shape[0])]

        q_rows.append(q)
        dq_rows.append(dq)
        quat_rows.append(quat)
        latent_rows.append(latent)
        hand_rows.append(hand["inspire_action"])
        camera_rows.append(cam["head_yaw_pitch"])

    return (
        np.stack(q_rows, axis=0),
        np.stack(dq_rows, axis=0),
        np.stack(quat_rows, axis=0),
        np.stack(latent_rows, axis=0),
        np.stack(hand_rows, axis=0),
        np.stack(camera_rows, axis=0),
    )


def _has_compact_action_columns(episode: EpisodeData) -> bool:
    return all(
        key in episode.parquet
        for key in (
            "action.measured_state",
            "action.latent_state",
            "action.hand_action",
            "action.camera_action",
        )
    )


def _build_action_arrays_from_parquet(
    episode: EpisodeData,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    measured = np.asarray(episode.parquet["action.measured_state"], dtype=np.float32)
    q = measured[:, :G1_MOTOR_DIM]
    dq = measured[:, G1_MOTOR_DIM : 2 * G1_MOTOR_DIM]
    quat = measured[:, 2 * G1_MOTOR_DIM : 2 * G1_MOTOR_DIM + 4]
    latent = np.asarray(episode.parquet["action.latent_state"], dtype=np.float32)
    hand = np.asarray(episode.parquet["action.hand_action"], dtype=np.float32)
    camera = np.asarray(episode.parquet["action.camera_action"], dtype=np.float32)
    return q, dq, quat, latent, hand, camera


def load_episode_store(
    loader: LeRobotEpisodeLoader,
    episode_index: int,
) -> EpisodeAlignedStore:
    ep = loader.load_episode(
        episode_index=episode_index,
        load_parquet=True,
        load_frames=False,
        load_raw=False,
        load_aligned_extras=True,
    )
    if ep.raw_dir is None or not ep.raw_dir.is_dir():
        raise FileNotFoundError(f"episode {episode_index}: raw_streams dir missing")
    for name in _VLA_RAW_STREAMS:
        ep.raw_streams[name] = loader._read_raw_stream(ep.raw_dir, name, dedupe_inspire=False)
    language = ep.info.tasks[0] if ep.info.tasks else ""
    return EpisodeAlignedStore.from_episode(ep, language=language)
