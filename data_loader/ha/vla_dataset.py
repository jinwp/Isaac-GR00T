"""PyTorch Dataset for Samsung LeRobot VLA training (30 Hz aligned samples)."""

from __future__ import annotations

import os
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from data_loader.action_deltas import relative_quat_wxyz, wrapped_angle_diff

from .episode_review import filter_boxer_record
from .loader import LeRobotEpisodeLoader
from .types import EpisodeInfo
from .vla_episode_io import (
    EpisodeIOCache,
    EpisodeIOHandle,
    EpisodePaths,
    stack_aligned_maps,
    stack_poses,
    uncertainty_maps_from_aligned_rows,
)
from .vla_boxer_payload import BOXER_OBJECTS_COLUMN
from .vla_task_registry import resolve_dataset_roots

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover - optional dependency for inspection-only use
    torch = None  # type: ignore[assignment]
    Dataset = object  # type: ignore[misc,assignment]


@dataclass(frozen=True)
class EpisodeSampleContext:
    """Per-sample view passed to ``others_builders`` (on-the-fly, no full-episode arrays)."""

    handle: EpisodeIOHandle
    obs_indices: list[int]
    aligned_obs: dict[str, Any]


OthersBuilder = Callable[[EpisodeSampleContext], dict[str, Any]]


@dataclass(frozen=True)
class _TaskBundle:
    task_key: str
    root: Path
    loader: LeRobotEpisodeLoader
    episodes: list[EpisodeInfo]


def _pad_index(idx: int, length: int) -> int:
    if idx < 0:
        return 0
    if idx >= length:
        return length - 1
    return idx


def _parse_boxer_object(obj: dict[str, Any], *, load_crop: bool) -> dict[str, Any]:
    relative_crop_path = str(obj.get("crop_png_path") or "")
    readable_crop_path = str(obj.get("crop_png_abspath") or relative_crop_path)
    out: dict[str, Any] = {
        "track_id": int(obj.get("track_id", -1)),
        "track_state": str(obj.get("track_state") or ""),
        "language_label": str(obj.get("language_label") or ""),
        "T_world_object": np.asarray(obj.get("T_world_object"), dtype=np.float32).reshape(4, 4)
        if obj.get("T_world_object") is not None
        else np.full((4, 4), np.nan, dtype=np.float32),
        "bbox_size_xyz": np.asarray(obj.get("bbox_size_xyz"), dtype=np.float32).reshape(3)
        if obj.get("bbox_size_xyz") is not None
        else np.full(3, np.nan, dtype=np.float32),
        "bbox_center_world": np.asarray(obj.get("bbox_center_world"), dtype=np.float32).reshape(3)
        if obj.get("bbox_center_world") is not None
        else np.full(3, np.nan, dtype=np.float32),
        "global_ij": (
            int(obj.get("global_ix", -1)),
            int(obj.get("global_iy", -1)),
        ),
        "local_ij": (
            int(obj.get("local_ix", -1)),
            int(obj.get("local_iy", -1)),
        ),
        "crop_png_path": relative_crop_path,
        "crop_image": None,
    }
    if load_crop and readable_crop_path:
        img = cv2.imread(readable_crop_path, cv2.IMREAD_COLOR)
        if img is not None:
            out["crop_image"] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return out


@dataclass
class LeRobotVLADataset(Dataset if torch is not None else object):  # type: ignore[misc]
    """
    Episode-aligned VLA samples on the 30 Hz left-master grid.

    Supports a single task (``dataset_root``) or many tasks under ``data_parent``
    (auto-discovers new task folders with ``meta/info.json`` + ``meta/episodes.jsonl``).

    Flat index spans all tasks × episodes × anchor frames (HERawDataset-style).
    Use ``DataLoader(..., shuffle=True)`` for random task/episode/anchor batches.

    Data is loaded **on-the-fly** per sample (EgoDex / HERaw style): no full-episode
    RAM preload. A small LRU (``io_cache_size``) may retain jsonl byte-offset indices
    and image dimensions only — not aligned arrays or action tables.
    """

    dataset_root: str | Path | None = None
    data_parent: str | Path | None = None
    dataset_roots: list[str | Path] | None = None
    task_names: list[str] | None = None
    episodes: list[int] | None = None
    # Optional allowlist of ``\"{task_key}/{episode_index}\"`` keys (train/val split).
    episode_keys: list[str] | None = None
    num_past_frames: int = 0
    action_chunk_size: int = 32
    action_start_offset: int = 1
    upsample_rate: int = 1
    load_images: bool = True
    load_boxer_crops: bool = False
    boxer_object_policy: str = "active_inactive_dedup"
    include_body_vel: bool = True
    use_delta_actions: bool = False
    io_cache_size: int = 2
    video_cache_size: int = 16
    # Keep only the first N chronological anchors per episode (overfit / smoke).
    max_anchors_per_episode: int | None = None
    bev_uncertainty_source: str = "zarr"
    others_builders: list[OthersBuilder] = field(default_factory=list)
    # Required: downstream opt-in is keyed on ``meta.dataset_name``, not on the
    # loader class, so a silent default would make HA stage-1 fall back to the
    # subtask/polar dummies instead of the real annotation parquets.
    dataset_name: str | None = None

    def __post_init__(self) -> None:
        if torch is None:
            raise ImportError("LeRobotVLADataset requires PyTorch")
        if not self.dataset_name:
            raise ValueError(
                "LeRobotVLADataset requires dataset_name (e.g. 'halab'): it is "
                "reported as meta.dataset_name and selects the HA stage-1 "
                "annotation source (HaCustomRepackTransform.annotation_datasets)"
            )
        if self.action_chunk_size < 1:
            raise ValueError("action_chunk_size must be >= 1")
        if self.action_start_offset < 0:
            raise ValueError("action_start_offset must be >= 0")
        if self.num_past_frames < 0:
            raise ValueError("num_past_frames must be >= 0")
        if self.video_cache_size < 1:
            raise ValueError("video_cache_size must be >= 1")
        if self.max_anchors_per_episode is not None and int(self.max_anchors_per_episode) < 1:
            raise ValueError(
                f"max_anchors_per_episode must be >= 1 when set, got {self.max_anchors_per_episode}"
            )
        self.max_anchors_per_episode = (
            None
            if self.max_anchors_per_episode is None
            else int(self.max_anchors_per_episode)
        )
        self.include_body_vel = bool(self.include_body_vel)
        self.use_delta_actions = bool(self.use_delta_actions)
        if self.boxer_object_policy not in (
            "recorded",
            "active",
            "active_inactive_dedup",
            "active_memory",
        ):
            raise ValueError(f"unsupported boxer_object_policy={self.boxer_object_policy!r}")
        if self.bev_uncertainty_source not in ("zarr", "aligned_extras"):
            raise ValueError(
                f"unsupported bev_uncertainty_source={self.bev_uncertainty_source!r}; "
                "expected 'zarr' or 'aligned_extras'"
            )
        roots = resolve_dataset_roots(
            data_parent=Path(self.data_parent) if self.data_parent else None,
            dataset_root=Path(self.dataset_root) if self.dataset_root else None,
            dataset_roots=[Path(p) for p in self.dataset_roots] if self.dataset_roots else None,
            task_names=self.task_names,
        )

        self._tasks: list[_TaskBundle] = []
        wanted_eps = {int(i) for i in self.episodes} if self.episodes is not None else None
        wanted_keys = (
            {str(k) for k in self.episode_keys} if self.episode_keys is not None else None
        )

        for root in roots:
            loader = LeRobotEpisodeLoader(root)
            eps = loader.list_episodes()
            if wanted_eps is not None:
                eps = [ep for ep in eps if ep.episode_index in wanted_eps]
            if wanted_keys is not None:
                eps = [
                    ep
                    for ep in eps
                    if f"{root.name}/{int(ep.episode_index)}" in wanted_keys
                ]
            if not eps:
                continue
            self._tasks.append(
                _TaskBundle(
                    task_key=root.name,
                    root=root,
                    loader=loader,
                    episodes=eps,
                )
            )

        if not self._tasks:
            raise ValueError("no episodes found for the requested task(s)")

        self._anchors_by_episode: list[list[int]] = []
        self._anchor_starts: list[int] = []
        self._episode_lengths: list[int] = []
        self._episode_metas: list[dict[str, Any]] = []

        self._anchors_flat: list[tuple[int, int, int]] = []
        offset = 0
        for task_idx, task in enumerate(self._tasks):
            for ep in task.episodes:
                length = int(ep.length)
                if length <= 0:
                    raise ValueError(
                        f"episode {ep.episode_index} in {task.task_key}: "
                        "missing or zero length in meta/episodes.jsonl"
                    )
                self._episode_lengths.append(length)
                instruction = ep.tasks[0] if ep.tasks else task.task_key.replace("_", " ")
                self._episode_metas.append(
                    {
                        "task_index": task_idx,
                        "task_key": task.task_key,
                        "dataset_root": str(task.root),
                        "episode_index": ep.episode_index,
                        "episode_id": ep.episode_id,
                        "len_episode": length,
                        "instruction": instruction,
                    }
                )
                min_anchor = self.num_past_frames
                max_anchor_exclusive = length - self.action_start_offset
                anchors = list(range(min_anchor, max_anchor_exclusive, self.upsample_rate))
                if self.max_anchors_per_episode is not None:
                    anchors = anchors[: self.max_anchors_per_episode]
                self._anchors_by_episode.append(anchors)
                self._anchor_starts.append(offset)
                for anchor in anchors:
                    self._anchors_flat.append((task_idx, ep.episode_index, anchor))
                offset += len(anchors)

        if self.max_anchors_per_episode is not None:
            print(
                f"[LeRobotVLADataset] max_anchors_per_episode="
                f"{self.max_anchors_per_episode}: "
                f"{len(self._episode_metas)} episode(s) → "
                f"{len(self._anchors_flat)} sample(s)",
                flush=True,
            )

        self._video_caps: OrderedDict[tuple[int, int], Any] = OrderedDict()
        self._video_next_frame: dict[tuple[int, int], int] = {}
        self._video_cap_pid: int | None = None
        self._io_cache = EpisodeIOCache(max_episodes=self.io_cache_size)

    def __len__(self) -> int:
        return len(self._anchors_flat)

    @property
    def task_list(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for task_idx, task in enumerate(self._tasks):
            n_samples = sum(1 for ti, _, _ in self._anchors_flat if ti == task_idx)
            out.append(
                {
                    "task_index": task_idx,
                    "task_key": task.task_key,
                    "dataset_root": str(task.root),
                    "num_episodes": len(task.episodes),
                    "num_samples": n_samples,
                }
            )
        return out

    @property
    def episode_list(self) -> list[dict[str, Any]]:
        return list(self._episode_metas)

    def _get_io_handle(self, task_idx: int, episode_index: int) -> EpisodeIOHandle:
        meta = next(
            m
            for m in self._episode_metas
            if m["task_index"] == task_idx and m["episode_index"] == episode_index
        )
        task = self._tasks[task_idx]
        key = (int(task_idx), int(episode_index))

        def factory() -> EpisodeIOHandle:
            paths = EpisodePaths.from_task(
                task.root,
                episode_index,
                meta["episode_id"],
                meta["instruction"],
                meta["len_episode"],
            )
            return EpisodeIOHandle(paths=paths)

        return self._io_cache.get(key, factory)

    def _locate(self, idx: int) -> tuple[int, int, int]:
        return self._anchors_flat[int(idx)]

    def _flat_episode_id(self, task_idx: int, local_episode_index: int) -> int:
        for flat_id, meta in enumerate(self._episode_metas):
            if meta["task_index"] == task_idx and meta["episode_index"] == local_episode_index:
                return flat_id
        raise KeyError(f"episode not found task={task_idx} ep={local_episode_index}")

    def episode_anchor_ranges(self) -> list[tuple[int, int]]:
        """``[(start, stop_exclusive), ...]`` flat index ranges, one per episode.

        Anchors of one episode are contiguous and time-ordered in the flat index,
        which is what an episode-grouped batch sampler needs.
        """
        ranges: list[tuple[int, int]] = []
        for start, anchors in zip(self._anchor_starts, self._anchors_by_episode):
            if anchors:
                ranges.append((int(start), int(start) + len(anchors)))
        return ranges

    def episode_flat_index_bounds(self, flat_episode_id: int) -> tuple[int, int]:
        if not self._anchors_flat:
            raise ValueError("No samples in dataset")
        flat_episode_id = int(flat_episode_id) % len(self._episode_metas)
        start = self._anchor_starts[flat_episode_id]
        length = len(self._anchors_by_episode[flat_episode_id])
        if length == 0:
            raise ValueError(f"flat episode {flat_episode_id} has no valid anchors")
        return start, start + length - 1

    def _release_video_caps(self) -> None:
        for cap in self._video_caps.values():
            try:
                cap.release()
            except Exception:
                pass
        self._video_caps.clear()
        self._video_next_frame.clear()
        self._video_cap_pid = None

    def _ensure_video_caps_for_process(self) -> None:
        pid = os.getpid()
        if self._video_cap_pid != pid:
            self._release_video_caps()
            self._video_cap_pid = pid

    def _obs_frame_indices(self, anchor: int) -> list[int]:
        past = [anchor - (self.num_past_frames - t) for t in range(self.num_past_frames)]
        return past + [anchor]

    def _read_image(
        self,
        handle: EpisodeIOHandle,
        mp4_frame_index: int,
        cap_key: tuple[int, int],
    ) -> np.ndarray:
        if not self.load_images or handle.paths.video_path is None:
            return np.zeros((1, 1, 3), dtype=np.uint8)
        self._ensure_video_caps_for_process()
        cap = self._video_caps.get(cap_key)
        if cap is None:
            while len(self._video_caps) >= self.video_cache_size:
                stale_key, stale_cap = self._video_caps.popitem(last=False)
                stale_cap.release()
                self._video_next_frame.pop(stale_key, None)
            cap = cv2.VideoCapture(str(handle.paths.video_path))
            self._video_caps[cap_key] = cap
        else:
            self._video_caps.move_to_end(cap_key)
        frame_index = int(mp4_frame_index)
        if self._video_next_frame.get(cap_key) != frame_index:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, bgr = cap.read()
        if not ok:
            self._video_next_frame.pop(cap_key, None)
            raise RuntimeError(
                f"failed to read video frame {frame_index} from {handle.paths.video_path}"
            )
        self._video_next_frame[cap_key] = frame_index + 1
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def _action_frame_indices(self, obs_indices: list[int], length: int) -> list[int]:
        # Action / MT chunks are built from the latest obs (anchor) only.
        # Past frames are history for image/proprio, not extra action starts.
        frames: set[int] = set()
        for frame_idx in [obs_indices[-1]]:
            start = _pad_index(frame_idx + self.action_start_offset, length)
            for t in range(self.action_chunk_size):
                frames.add(_pad_index(start + t, length))
        if self.use_delta_actions:
            # Need previous absolute frame for each action row (MT REPLACE deltas).
            for frame in list(frames):
                if frame > 0:
                    frames.add(frame - 1)
        return sorted(frames)

    def _deltaize_action_by_frame(
        self,
        action_by_frame: dict[int, dict[str, np.ndarray]],
        chunk_frames: list[int],
    ) -> dict[int, dict[str, np.ndarray]]:
        """Convert absolute action rows to MT-style REPLACE deltas for chunk frames."""
        out: dict[int, dict[str, np.ndarray]] = {}
        for frame in chunk_frames:
            cur = action_by_frame[frame]
            if frame <= 0:
                q = np.zeros_like(cur["measured_joint_q"])
                hand = np.zeros_like(cur["hand_action"])
                cam = np.zeros_like(cur["camera_action"])
                quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            else:
                prev = action_by_frame[frame - 1]
                q = (cur["measured_joint_q"] - prev["measured_joint_q"]).astype(np.float32)
                hand = (cur["hand_action"] - prev["hand_action"]).astype(np.float32)
                cam = wrapped_angle_diff(
                    np.stack([prev["camera_action"], cur["camera_action"]], axis=0)
                )[1]
                quat = relative_quat_wxyz(
                    np.stack([prev["body_quat"], cur["body_quat"]], axis=0)
                )[1]
            row = {
                "measured_joint_q": q,
                "body_quat": quat,
                "latent_state": cur["latent_state"],
                "hand_action": hand,
                "camera_action": cam,
            }
            if self.include_body_vel:
                row["measured_joint_dq"] = cur["measured_joint_dq"]
            out[frame] = row
        return out

    def _prepare_action_by_frame(
        self,
        action_by_frame: dict[int, dict[str, np.ndarray]],
        obs_indices: list[int],
        length: int,
    ) -> dict[int, dict[str, np.ndarray]]:
        chunk_frames: set[int] = set()
        for frame_idx in [obs_indices[-1]]:
            start = _pad_index(frame_idx + self.action_start_offset, length)
            for t in range(self.action_chunk_size):
                chunk_frames.add(_pad_index(start + t, length))
        frames = sorted(chunk_frames)
        if self.use_delta_actions:
            return self._deltaize_action_by_frame(action_by_frame, frames)
        if self.include_body_vel:
            return {f: action_by_frame[f] for f in frames}
        out: dict[int, dict[str, np.ndarray]] = {}
        for frame in frames:
            row = dict(action_by_frame[frame])
            row.pop("measured_joint_dq", None)
            out[frame] = row
        return out

    def _build_action_chunks(
        self,
        handle: EpisodeIOHandle,
        obs_indices: list[int],
        length: int,
        action_by_frame: dict[int, dict[str, np.ndarray]],
    ) -> tuple[list[np.ndarray], ...]:
        q_list: list[np.ndarray] = []
        dq_list: list[np.ndarray] = []
        quat_list: list[np.ndarray] = []
        latent_list: list[np.ndarray] = []
        hand_list: list[np.ndarray] = []
        camera_list: list[np.ndarray] = []
        mask_list: list[np.ndarray] = []

        prepared = self._prepare_action_by_frame(action_by_frame, obs_indices, length)
        # One action chunk for the anchor only → leading dim 1 after stack.
        for frame_idx in [obs_indices[-1]]:
            start = _pad_index(frame_idx + self.action_start_offset, length)
            chunk_frames = [_pad_index(start + t, length) for t in range(self.action_chunk_size)]
            mask = np.array([0 <= start + t < length for t in range(self.action_chunk_size)], dtype=np.float32)
            q_list.append(np.stack([prepared[f]["measured_joint_q"] for f in chunk_frames], axis=0))
            if self.include_body_vel:
                dq_list.append(np.stack([prepared[f]["measured_joint_dq"] for f in chunk_frames], axis=0))
            quat_list.append(np.stack([prepared[f]["body_quat"] for f in chunk_frames], axis=0))
            latent_list.append(np.stack([prepared[f]["latent_state"] for f in chunk_frames], axis=0))
            hand_list.append(np.stack([prepared[f]["hand_action"] for f in chunk_frames], axis=0))
            camera_list.append(np.stack([prepared[f]["camera_action"] for f in chunk_frames], axis=0))
            mask_list.append(mask)

        return q_list, dq_list, quat_list, latent_list, hand_list, camera_list, mask_list

    def _action_dict_from_chunks(
        self,
        action_q: list[np.ndarray],
        action_dq: list[np.ndarray],
        action_quat: list[np.ndarray],
        action_latent: list[np.ndarray],
        action_hand: list[np.ndarray],
        action_camera: list[np.ndarray],
    ) -> dict[str, Any]:
        from data_loader.nymeria.nymeria_motion_action import pack_motion_tokenizer_input_from_dict

        out: dict[str, Any] = {
            "measured_joint_q": torch.from_numpy(np.stack(action_q, axis=0)),
            "body_quat": torch.from_numpy(np.stack(action_quat, axis=0)),
            "latent_state": torch.from_numpy(np.stack(action_latent, axis=0)),
            "hand_action": torch.from_numpy(np.stack(action_hand, axis=0)),
            "camera_action": torch.from_numpy(np.stack(action_camera, axis=0)),
        }
        channels: dict[str, np.ndarray] = {
            "measured_joint_q": np.stack(action_q, axis=0),
            "body_quat": np.stack(action_quat, axis=0),
            "hand_action": np.stack(action_hand, axis=0),
            "camera_action": np.stack(action_camera, axis=0),
        }
        if self.include_body_vel:
            dq = np.stack(action_dq, axis=0)
            out["measured_joint_dq"] = torch.from_numpy(dq)
            channels["measured_joint_dq"] = dq
        out["motion_tokenizer_input"] = torch.from_numpy(
            pack_motion_tokenizer_input_from_dict(
                channels, include_body_vel=self.include_body_vel
            )
        )
        return out

    def _build_proprio_mt_chunks(
        self,
        handle: EpisodeIOHandle,
        obs_indices: list[int],
        length: int,
    ) -> np.ndarray:
        """Pack ``observation.state`` into MT layout over ``[t : t+chunk)`` for the anchor.

        Shape ``(1, action_chunk_size, mt_dim)`` — past obs frames are history only.
        Window starts at observation time ``t`` (not ``t + action_start_offset``);
        edge indices are clamped. When ``use_delta_actions``, applies the same MT
        REPLACE deltas as action MT (measured-state deltas, not a copy of action).
        Per-key proprio history channels stay absolute.
        """
        from data_loader.nymeria.nymeria_motion_action import pack_motion_tokenizer_input_from_dict

        anchor_indices = [int(obs_indices[-1])]
        chunk_frames: set[int] = set()
        for frame_idx in anchor_indices:
            start = _pad_index(int(frame_idx), length)
            for t in range(self.action_chunk_size):
                chunk_frames.add(_pad_index(start + t, length))
        frames_to_read = set(chunk_frames)
        if self.use_delta_actions:
            for frame in list(chunk_frames):
                if frame > 0:
                    frames_to_read.add(frame - 1)
        frames = sorted(frames_to_read)
        stacked = handle.read_proprio_rows(frames, include_body_vel=self.include_body_vel)

        # TODO(HALAB-MT-CHECKPOINT-MIGRATION — DO NOT REMOVE THIS WARNING):
        # This is a TEMPORARY compatibility override for MotionTokenizers
        # checkpoint generation 1396937 (exp2x2_ver2). Those checkpoints were
        # trained with q/dq/quat/hands from observation.state, but with the
        # camera channel from action.camera_action instead of the measured
        # observation.state[74:76] head values.
        #
        # WHEN THE MOTION TOKENIZER IS RETRAINED ON ALL 76 observation.state
        # VALUES, DELETE THIS OVERRIDE so read_proprio_rows()'s measured camera
        # values flow through unchanged. Keeping it would silently violate the
        # new checkpoint's input contract.
        action_rows = handle.read_action_rows(frames)
        stacked["camera_action"] = np.stack(
            [action_rows[frame]["camera_action"] for frame in frames], axis=0
        )

        by_frame: dict[int, dict[str, np.ndarray]] = {
            f: {k: stacked[k][i] for k in stacked} for i, f in enumerate(frames)
        }
        emit_frames = sorted(chunk_frames)
        if self.use_delta_actions:
            by_frame = self._deltaize_action_by_frame(by_frame, emit_frames)

        chunks: list[np.ndarray] = []
        for frame_idx in anchor_indices:
            start = _pad_index(int(frame_idx), length)
            idxs = [_pad_index(start + t, length) for t in range(self.action_chunk_size)]
            channels: dict[str, np.ndarray] = {
                "measured_joint_q": np.stack(
                    [by_frame[f]["measured_joint_q"] for f in idxs], axis=0
                ),
                "body_quat": np.stack([by_frame[f]["body_quat"] for f in idxs], axis=0),
                "hand_action": np.stack([by_frame[f]["hand_action"] for f in idxs], axis=0),
                "camera_action": np.stack(
                    [by_frame[f]["camera_action"] for f in idxs], axis=0
                ),
            }
            if self.include_body_vel:
                channels["measured_joint_dq"] = np.stack(
                    [by_frame[f]["measured_joint_dq"] for f in idxs], axis=0
                )
            chunks.append(
                pack_motion_tokenizer_input_from_dict(
                    channels, include_body_vel=self.include_body_vel
                )
            )
        return np.stack(chunks, axis=0)

    def _proprio_with_mt_input(
        self,
        proprio_np: dict[str, np.ndarray],
        proprio_mt_chunks: np.ndarray,
    ) -> dict[str, Any]:
        """Absolute proprio @ t plus chunked MT from ``observation.state`` (delta if e3)."""
        out = {key: torch.from_numpy(value) for key, value in proprio_np.items()}
        out["motion_tokenizer_input"] = torch.from_numpy(proprio_mt_chunks)
        return out

    def _build_boxer_objects(
        self,
        handle: EpisodeIOHandle,
        aligned_obs: dict[str, Any],
        obs_step: int,
    ) -> list[dict[str, Any]]:
        if self.boxer_object_policy == "active_inactive_dedup" and BOXER_OBJECTS_COLUMN in aligned_obs:
            return [
                _parse_boxer_object(obj, load_crop=self.load_boxer_crops)
                for obj in aligned_obs[BOXER_OBJECTS_COLUMN][obs_step]
            ]
        if self.boxer_object_policy == "active_inactive_dedup":
            raise RuntimeError(
                "[LeRobotVLADataset] aligned_extras is missing "
                f"{BOXER_OBJECTS_COLUMN!r}; jsonl fallback is disabled for "
                "active_inactive_dedup. Run "
                "data_loader/ha/migration/migrate_vla_boxer_aligned_extras.py "
                f"(episode={handle.paths.raw_dir})."
            )
        # jsonl path retained for non-parquet policies (recorded / active / active_memory).
        boxer_raw_index = int(aligned_obs["boxer_raw_index"][obs_step])
        boxer_record = handle.boxer_record_at_aligned_row(boxer_raw_index)
        if boxer_record is None:
            return []
        pose_matrix = stack_poses([aligned_obs["bev_camera_pose_world"][obs_step]])[0]
        pose_record = handle.pose_record_from_matrix(pose_matrix)
        filtered = filter_boxer_record(
            boxer_record,
            pose_record,
            object_policy=self.boxer_object_policy,
            image_width=handle.image_width,
            image_height=handle.image_height,
        )
        objects = filtered.get("objects") or []
        return [_parse_boxer_object(obj, load_crop=self.load_boxer_crops) for obj in objects]

    def _build_others(
        self,
        handle: EpisodeIOHandle,
        obs_indices: list[int],
        aligned_obs: dict[str, Any],
    ) -> dict[str, Any]:
        boxer_steps = [
            self._build_boxer_objects(handle, aligned_obs, step) for step in range(len(obs_indices))
        ]
        global_ij = np.stack(
            [
                aligned_obs["bev_target_ix"].astype(np.int32),
                aligned_obs["bev_target_iy"].astype(np.int32),
            ],
            axis=1,
        )
        local_ij = np.stack(
            [
                aligned_obs["bev_target_local_ix"].astype(np.int32),
                aligned_obs["bev_target_local_iy"].astype(np.int32),
            ],
            axis=1,
        )
        others: dict[str, Any] = {
            "search_target": {
                "global_ij": global_ij.copy(),
                "local_ij": local_ij.copy(),
                "is_bos": aligned_obs["bev_is_bos"].astype(np.bool_).copy(),
                "target_local_clipped": aligned_obs["bev_target_local_clipped"].astype(np.bool_).copy(),
            },
            "boxer": {
                "objects": boxer_steps,
                "num_objects": np.array([len(objs) for objs in boxer_steps], dtype=np.int32),
            },
            "extra": {},
        }
        ctx = EpisodeSampleContext(handle=handle, obs_indices=obs_indices, aligned_obs=aligned_obs)
        for builder in self.others_builders:
            others["extra"].update(builder(ctx))
        return others

    def _load_uncertainty_maps(
        self,
        handle: EpisodeIOHandle,
        aligned_obs: dict[str, Any],
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.bev_uncertainty_source == "aligned_extras":
            missing = [
                name
                for name in ("bev_global_uncertainty", "bev_local_uncertainty")
                if name not in aligned_obs
            ]
            if missing:
                raise KeyError(
                    f"aligned_extras missing {missing}; use bev_uncertainty_source='zarr' or migrate dataset"
                )
            global_uncertainty, local_uncertainty, gshape, lshape = uncertainty_maps_from_aligned_rows(
                aligned_obs["bev_global_uncertainty"],
                aligned_obs["bev_local_uncertainty"],
            )
            handle.global_grid_shape = gshape
            handle.local_grid_shape = lshape
            return global_uncertainty, local_uncertainty

        global_rows, local_rows = handle.read_uncertainty_maps(
            [int(i) for i in aligned_obs["bev_raw_index"]]
        )
        return (
            stack_aligned_maps(global_rows, handle.global_grid_shape),
            stack_aligned_maps(local_rows, handle.local_grid_shape),
        )

    def __getitem__(self, idx: int) -> dict[str, Any]:
        task_idx, episode_index, anchor = self._locate(idx)
        task = self._tasks[task_idx]
        handle = self._get_io_handle(task_idx, episode_index)
        length = int(handle.paths.length)
        obs_indices = self._obs_frame_indices(anchor)
        cap_key = (task_idx, episode_index)

        obs_mask = np.array(
            [0 <= frame_idx < length for frame_idx in obs_indices],
            dtype=np.float32,
        )

        action_frames = self._action_frame_indices(obs_indices, length)
        all_rows = sorted(set(obs_indices + action_frames))
        want_boxer = self.boxer_object_policy == "active_inactive_dedup"
        aligned = handle.read_aligned_rows(
            all_rows,
            include_boxer_objects=want_boxer,
            boxer_row_indices=obs_indices if want_boxer else None,
            include_bev_maps=(self.bev_uncertainty_source == "aligned_extras"),
        )
        row_pos = {frame_idx: pos for pos, frame_idx in enumerate(all_rows)}
        obs_row_idx = [row_pos[i] for i in obs_indices]

        aligned_obs: dict[str, Any] = {}
        for name, col in aligned.items():
            if name in (BOXER_OBJECTS_COLUMN,):
                # Already sliced to obs_indices inside read_aligned_rows.
                aligned_obs[name] = col
            elif isinstance(col, np.ndarray):
                aligned_obs[name] = col[obs_row_idx]
            else:
                aligned_obs[name] = [col[i] for i in obs_row_idx]

        stamps = aligned["timestamp_ns"]
        action_by_frame = handle.read_action_rows(
            action_frames,
            fallback_stamps={frame: int(stamps[row_pos[frame]]) for frame in action_frames},
        )

        images = [
            self._read_image(handle, int(aligned_obs["frame_index"][pos]), cap_key)
            for pos in range(len(obs_indices))
        ]
        image_tensor = torch.from_numpy(np.stack(images, axis=0)).permute(0, 3, 1, 2).float() / 255.0

        action_q, action_dq, action_quat, action_latent, action_hand, action_camera, action_masks = (
            self._build_action_chunks(handle, obs_indices, length, action_by_frame)
        )
        action_chunk_valid = np.stack(action_masks, axis=0)
        flat_episode_id = self._flat_episode_id(task_idx, episode_index)

        camera_pose = stack_poses(aligned_obs["bev_camera_pose_world"])
        global_uncertainty, local_uncertainty = self._load_uncertainty_maps(handle, aligned_obs)
        from data_loader.nymeria.nymeria_motion_action import pack_proprioceptive_state

        proprio_np = handle.read_proprio_rows(
            obs_indices, include_body_vel=self.include_body_vel
        )
        proprio_mt = self._build_proprio_mt_chunks(handle, obs_indices, length)
        proprioceptive_state = pack_proprioceptive_state(
            proprio_np, include_body_vel=self.include_body_vel
        )

        return {
            "observation": {
                "image": image_tensor,
                "camera_pose": torch.from_numpy(camera_pose),
                "global_uncertainty": torch.from_numpy(global_uncertainty),
                "local_uncertainty": torch.from_numpy(local_uncertainty),
                "proprioceptive": self._proprio_with_mt_input(proprio_np, proprio_mt),
                "proprioceptive_state": torch.from_numpy(proprioceptive_state),
            },
            "action": self._action_dict_from_chunks(
                action_q, action_dq, action_quat, action_latent, action_hand, action_camera
            ),
            "language": handle.paths.language,
            "others": self._build_others(handle, obs_indices, aligned_obs),
            "meta": {
                "task_index": task_idx,
                "task_key": task.task_key,
                "dataset_root": str(task.root),
                "episode_index": episode_index,
                "flat_episode_id": flat_episode_id,
                "episode_id": handle.paths.episode_id,
                "frame_id": anchor,
                "timestamp_ns": int(aligned["timestamp_ns"][row_pos[anchor]]),
                "annotation_path": str(
                    task.root
                    / "preprocess"
                    / "annotation"
                    / f"episode_{int(episode_index):06d}.parquet"
                ),
                "annotation_frame": int(aligned["frame_index"][row_pos[anchor]]),
                "obs_frame_indices": obs_indices,
                "obs_mask": torch.from_numpy(obs_mask),
                "action_chunk_valid_mask": torch.from_numpy(action_chunk_valid),
                "boxer_object_policy": self.boxer_object_policy,
                "bev_uncertainty_source": self.bev_uncertainty_source,
                "dataset_name": self.dataset_name,
                "global_grid_shape": handle.global_grid_shape,
                "local_grid_shape": handle.local_grid_shape,
                "mt_input_sources": "halab_obs_state_and_action",
            },
        }

    def profile_sample(self, idx: int) -> tuple[dict[str, Any], dict[str, float]]:
        """Return one sample plus coarse per-stage timings in milliseconds."""
        timings: dict[str, float] = {}

        def mark(name: str, t0: float) -> float:
            now = time.perf_counter()
            timings[name] = (now - t0) * 1000.0
            return now

        t = time.perf_counter()
        task_idx, episode_index, anchor = self._locate(idx)
        task = self._tasks[task_idx]
        handle = self._get_io_handle(task_idx, episode_index)
        length = int(handle.paths.length)
        obs_indices = self._obs_frame_indices(anchor)
        cap_key = (task_idx, episode_index)
        t = mark("locate_handle", t)

        obs_mask = np.array(
            [0 <= frame_idx < length for frame_idx in obs_indices],
            dtype=np.float32,
        )
        action_frames = self._action_frame_indices(obs_indices, length)
        all_rows = sorted(set(obs_indices + action_frames))
        t = mark("index_math", t)

        want_boxer = self.boxer_object_policy == "active_inactive_dedup"
        aligned = handle.read_aligned_rows(
            all_rows,
            include_boxer_objects=want_boxer,
            boxer_row_indices=obs_indices if want_boxer else None,
            include_bev_maps=(self.bev_uncertainty_source == "aligned_extras"),
        )
        row_pos = {frame_idx: pos for pos, frame_idx in enumerate(all_rows)}
        obs_row_idx = [row_pos[i] for i in obs_indices]
        aligned_obs: dict[str, Any] = {}
        for name, col in aligned.items():
            if name in (BOXER_OBJECTS_COLUMN,):
                aligned_obs[name] = col
            elif isinstance(col, np.ndarray):
                aligned_obs[name] = col[obs_row_idx]
            else:
                aligned_obs[name] = [col[i] for i in obs_row_idx]
        t = mark("aligned_extras_read", t)

        stamps = aligned["timestamp_ns"]
        action_by_frame = handle.read_action_rows(
            action_frames,
            fallback_stamps={frame: int(stamps[row_pos[frame]]) for frame in action_frames},
        )
        t = mark("action_read", t)

        images = [
            self._read_image(handle, int(aligned_obs["frame_index"][pos]), cap_key)
            for pos in range(len(obs_indices))
        ]
        image_tensor = torch.from_numpy(np.stack(images, axis=0)).permute(0, 3, 1, 2).float() / 255.0
        t = mark("image_read_tensor", t)

        action_q, action_dq, action_quat, action_latent, action_hand, action_camera, action_masks = (
            self._build_action_chunks(handle, obs_indices, length, action_by_frame)
        )
        action_chunk_valid = np.stack(action_masks, axis=0)
        t = mark("action_chunk_stack", t)

        camera_pose = stack_poses(aligned_obs["bev_camera_pose_world"])
        if self.bev_uncertainty_source == "zarr":
            global_uncertainty, local_uncertainty = self._load_uncertainty_maps(handle, aligned_obs)
            t = mark("uncertainty_zarr_read", t)
        else:
            global_uncertainty, local_uncertainty = self._load_uncertainty_maps(handle, aligned_obs)
            t = mark("uncertainty_aligned_extras_read", t)
        t = mark("obs_map_stack", t)

        others = self._build_others(handle, obs_indices, aligned_obs)
        t = mark("others_boxer", t)

        from data_loader.nymeria.nymeria_motion_action import pack_proprioceptive_state

        proprio_np = handle.read_proprio_rows(
            obs_indices, include_body_vel=self.include_body_vel
        )
        proprio_mt = self._build_proprio_mt_chunks(handle, obs_indices, length)
        proprioceptive_state = pack_proprioceptive_state(
            proprio_np, include_body_vel=self.include_body_vel
        )
        t = mark("proprio_read", t)

        flat_episode_id = self._flat_episode_id(task_idx, episode_index)
        sample = {
            "observation": {
                "image": image_tensor,
                "camera_pose": torch.from_numpy(camera_pose),
                "global_uncertainty": torch.from_numpy(global_uncertainty),
                "local_uncertainty": torch.from_numpy(local_uncertainty),
                "proprioceptive": self._proprio_with_mt_input(proprio_np, proprio_mt),
                "proprioceptive_state": torch.from_numpy(proprioceptive_state),
            },
            "action": self._action_dict_from_chunks(
                action_q, action_dq, action_quat, action_latent, action_hand, action_camera
            ),
            "language": handle.paths.language,
            "others": others,
            "meta": {
                "task_index": task_idx,
                "task_key": task.task_key,
                "dataset_root": str(task.root),
                "episode_index": episode_index,
                "flat_episode_id": flat_episode_id,
                "episode_id": handle.paths.episode_id,
                "frame_id": anchor,
                "timestamp_ns": int(aligned["timestamp_ns"][row_pos[anchor]]),
                "annotation_path": str(
                    task.root
                    / "preprocess"
                    / "annotation"
                    / f"episode_{int(episode_index):06d}.parquet"
                ),
                "annotation_frame": int(aligned["frame_index"][row_pos[anchor]]),
                "obs_frame_indices": obs_indices,
                "obs_mask": torch.from_numpy(obs_mask),
                "action_chunk_valid_mask": torch.from_numpy(action_chunk_valid),
                "boxer_object_policy": self.boxer_object_policy,
                "bev_uncertainty_source": self.bev_uncertainty_source,
                "dataset_name": self.dataset_name,
                "global_grid_shape": handle.global_grid_shape,
                "local_grid_shape": handle.local_grid_shape,
                "mt_input_sources": "halab_obs_state_and_action",
            },
        }
        t = mark("torch_meta_assembly", t)
        timings["total"] = sum(timings.values())
        return sample, timings

    def __del__(self) -> None:
        caps = getattr(self, "_video_caps", None)
        if not caps:
            return
        for cap in caps.values():
            try:
                cap.release()
            except Exception:
                pass


def describe_vla_sample(sample: dict[str, Any]) -> str:
    """Human-readable shape summary for debugging."""
    obs = sample["observation"]
    act = sample["action"]
    meta = sample["meta"]
    prop = obs.get("proprioceptive")
    prop_lines: list[str]
    if isinstance(prop, dict):
        prop_lines = [f"observation.proprioceptive.{k}: {tuple(v.shape)}" for k, v in prop.items()]
    elif prop is not None:
        prop_lines = [f"observation.proprioceptive: {tuple(prop.shape)}"]
    else:
        prop_lines = ["observation.proprioceptive: <missing>"]
    lines = [
        f"language: {sample['language']!r}",
        f"task: {meta.get('task_key')} ep={meta.get('episode_index')} frame={meta.get('frame_id')}",
        f"observation.image: {tuple(obs['image'].shape)}",
        f"observation.camera_pose: {tuple(obs['camera_pose'].shape)}",
        f"observation.global_uncertainty: {tuple(obs['global_uncertainty'].shape)}",
        f"observation.local_uncertainty: {tuple(obs['local_uncertainty'].shape)}",
        *prop_lines,
        (
            f"observation.proprioceptive_state: {tuple(obs['proprioceptive_state'].shape)}"
            if "proprioceptive_state" in obs
            else "observation.proprioceptive_state: <missing>"
        ),
        f"action.measured_joint_q: {tuple(act['measured_joint_q'].shape)}",
        (
            f"action.measured_joint_dq: {tuple(act['measured_joint_dq'].shape)}"
            if "measured_joint_dq" in act
            else "action.measured_joint_dq: <omitted>"
        ),
        f"action.body_quat: {tuple(act['body_quat'].shape)}",
        f"action.latent_state: {tuple(act['latent_state'].shape)}",
        f"action.hand_action: {tuple(act['hand_action'].shape)}",
        f"action.camera_action: {tuple(act['camera_action'].shape)}",
        (
            f"action.motion_tokenizer_input: {tuple(act['motion_tokenizer_input'].shape)}"
            if "motion_tokenizer_input" in act
            else "action.motion_tokenizer_input: <missing>"
        ),
        f"meta.action_chunk_valid_mask: {tuple(meta['action_chunk_valid_mask'].shape)}",
        f"others.search_target.global_ij: {sample['others']['search_target']['global_ij'].shape}",
        f"others.boxer.num_objects: {sample['others']['boxer']['num_objects'].shape} "
        f"objects[0]={len(sample['others']['boxer']['objects'][0])}",
        f"grid={meta['global_grid_shape']}/{meta['local_grid_shape']}",
    ]
    return "\n".join(lines)
