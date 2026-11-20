"""On-the-fly episode I/O for VLA training (no full-episode RAM preload)."""

from __future__ import annotations

import bisect
import json
import os
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .constants import (
    G1_BASE_QUAT_DIM,
    G1_MOTOR_DIM,
    INSPIRE_Q_DIM,
    PROPRIO_DIM,
)
from .extract import (
    extract_dynamixel_action,
    extract_inspire_action,
    extract_measured_action,
    stamp_ns_of,
)
from .hand_action import halab_hand_action_to_radians
from .vla_boxer_payload import BOXER_OBJECTS_COLUMN, deserialize_boxer_objects

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover
    pa = None  # type: ignore[assignment]
    pq = None  # type: ignore[assignment]

try:
    import zarr
except ImportError:  # pragma: no cover
    zarr = None  # type: ignore[assignment]

_ACTION_STREAMS = ("g1_debug", "inspire_action", "dynamixel_action")
_ALIGNED_SCALAR_COLS = (
    "timestamp_ns",
    "frame_index",
    "bev_raw_index",
    "boxer_raw_index",
    "bev_is_bos",
    "bev_target_ix",
    "bev_target_iy",
    "bev_target_local_ix",
    "bev_target_local_iy",
    "bev_target_local_clipped",
)

_ALIGNED_MAP_COLS = (
    "bev_camera_pose_world",
)

_ALIGNED_UNCERTAINTY_COLS = (
    "bev_global_uncertainty",
    "bev_local_uncertainty",
)

_ALIGNED_OBJECT_COLS = (
    BOXER_OBJECTS_COLUMN,
)


def _infer_grid_shape(flat_len: int) -> tuple[int, int]:
    side = int(round(float(flat_len) ** 0.5))
    if side * side == flat_len:
        return side, side
    return 1, flat_len


def _pose_row_to_matrix(row: Any) -> np.ndarray:
    arr = np.asarray(row, dtype=np.float32).reshape(-1)
    if arr.size != 16:
        raise ValueError(f"bev_camera_pose_world must be 16 floats, got {arr.size}")
    return arr.reshape(4, 4).copy()


def _read_zarr_rows(array: Any, indices: list[int]) -> dict[int, np.ndarray]:
    """Batch a contiguous run and deduplicate sparse row reads."""
    unique = sorted({int(idx) for idx in indices if int(idx) >= 0})
    if not unique:
        return {}
    if len(unique) == 1:
        return {unique[0]: np.asarray(array[unique[0]], dtype=np.float32)}
    start, stop = unique[0], unique[-1] + 1
    if stop - start == len(unique):
        batch = np.asarray(array[start:stop], dtype=np.float32)
        return {idx: batch[idx - start] for idx in unique}
    return {idx: np.asarray(array[idx], dtype=np.float32) for idx in unique}


def _row_group_bounds(pf: Any) -> list[tuple[int, int, int]]:
    """Return [(rg_index, start_row, num_rows), ...] for a ParquetFile."""
    out: list[tuple[int, int, int]] = []
    start = 0
    meta = pf.metadata
    for rg in range(meta.num_row_groups):
        n = int(meta.row_group(rg).num_rows)
        out.append((rg, start, n))
        start += n
    return out


def _locate_row(bounds: list[tuple[int, int, int]], row_idx: int) -> tuple[int, int]:
    for rg, start, n in bounds:
        if start <= row_idx < start + n:
            return rg, row_idx - start
    raise IndexError(f"row {row_idx} out of range for parquet row groups")


def _read_parquet_columns_at_rows(
    path: Path,
    columns: list[str],
    *,
    row_indices: list[int],
    boxer_row_indices: list[int] | None,
    raw_dir: Path,
) -> dict[str, Any]:
    """Read only the row groups touched by the requested rows (critical for RG=32 pilots)."""
    if pq is None or pa is None:
        raise ImportError("pyarrow is required to read aligned_extras")
    pf = pq.ParquetFile(path)
    bounds = _row_group_bounds(pf)
    scalar_rows = [int(r) for r in row_indices]
    boxer_rows = [int(r) for r in boxer_row_indices] if boxer_row_indices is not None else []
    needed_rgs = sorted(
        {
            _locate_row(bounds, r)[0]
            for r in set(scalar_rows + boxer_rows)
        }
    )
    if not needed_rgs:
        return {name: [] for name in columns}

    tables = [pf.read_row_group(rg, columns=columns) for rg in needed_rgs]
    table = pa.concat_tables(tables) if len(tables) > 1 else tables[0]
    # Map global row -> offset within concatenated partial table.
    rg_to_offset: dict[int, int] = {}
    offset = 0
    for rg in needed_rgs:
        rg_to_offset[rg] = offset
        offset += int(pf.metadata.row_group(rg).num_rows)

    def local_indices(rows: list[int]) -> list[int]:
        locs: list[int] = []
        for row in rows:
            rg, local = _locate_row(bounds, row)
            locs.append(rg_to_offset[rg] + local)
        return locs

    scalar_take = pa.array(local_indices(scalar_rows), type=pa.int64())
    boxer_take = (
        pa.array(local_indices(boxer_rows), type=pa.int64())
        if boxer_row_indices is not None
        else scalar_take
    )

    out: dict[str, Any] = {}
    for name in columns:
        col = table[name]
        if name in _ALIGNED_OBJECT_COLS:
            taken = col.take(boxer_take)
            out[name] = [deserialize_boxer_objects(raw_dir, row.as_py()) for row in taken]
            continue
        taken = col.take(scalar_take)
        if name in _ALIGNED_MAP_COLS or name in _ALIGNED_UNCERTAINTY_COLS:
            out[name] = [np.asarray(row.as_py(), dtype=np.float32) for row in taken]
        else:
            out[name] = taken.to_numpy(zero_copy_only=False)
    return out


def _split_measured(measured: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    measured = np.asarray(measured, dtype=np.float32).reshape(-1)
    q = measured[:G1_MOTOR_DIM]
    dq = measured[G1_MOTOR_DIM : 2 * G1_MOTOR_DIM]
    quat = measured[2 * G1_MOTOR_DIM : 2 * G1_MOTOR_DIM + 4]
    return q, dq, quat


def _split_observation_state(
    obs_state: np.ndarray,
    *,
    include_body_vel: bool = True,
) -> dict[str, np.ndarray]:
    """Parse ``observation.state`` into action-layout proprio keys (MT concat order).

    Layout (76): ``[29 q | 29 dq | 4 base_quat | 12 hand_openness | 2 cam]``.
    Hand openness is converted to radians (same remap as ``action.hand_action``).
    """
    vec = np.asarray(obs_state, dtype=np.float32).reshape(-1)
    if vec.shape[0] < PROPRIO_DIM:
        raise ValueError(
            f"observation.state dim {vec.shape[0]} < {PROPRIO_DIM} "
            "(expected layout-2 q|dq|quat|hand|cam)"
        )
    q_end = G1_MOTOR_DIM
    dq_end = q_end + G1_MOTOR_DIM
    quat_end = dq_end + G1_BASE_QUAT_DIM
    hand_end = quat_end + INSPIRE_Q_DIM
    out: dict[str, np.ndarray] = {
        "measured_joint_q": vec[:q_end].copy(),
        "body_quat": vec[dq_end:quat_end].copy(),
        "hand_action": halab_hand_action_to_radians(vec[quat_end:hand_end]),
        "camera_action": vec[hand_end:hand_end + 2].copy(),
        "latent_state": np.zeros(64, dtype=np.float32),
    }
    if include_body_vel:
        out["measured_joint_dq"] = vec[q_end:dq_end].copy()
    return out


def _attach_boxer_crop_paths(raw_dir: Path, record: dict[str, Any]) -> dict[str, Any]:
    out = dict(record)
    objects = out.get("objects") or []
    if not isinstance(objects, list):
        return out
    new_objects: list[Any] = []
    for obj in objects:
        if not isinstance(obj, dict):
            new_objects.append(obj)
            continue
        new_obj = dict(obj)
        rel = new_obj.get("crop_png_path")
        if rel and "crop_png_abspath" not in new_obj:
            new_obj["crop_png_abspath"] = str(raw_dir / str(rel))
        new_objects.append(new_obj)
    out["objects"] = new_objects
    return out


class JsonlStampIndex:
    """Byte offsets + stamp_ns index for one JSONL file (records loaded one at a time)."""

    __slots__ = ("path", "offsets", "stamps")

    def __init__(self, path: Path) -> None:
        self.path = path
        self.offsets: list[int] = []
        self.stamps: list[int] = []
        if not path.exists():
            return
        with path.open("rb") as fh:
            while True:
                offset = fh.tell()
                line = fh.readline()
                if not line:
                    break
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                stamp = stamp_ns_of(record)
                if stamp is None:
                    continue
                self.offsets.append(offset)
                self.stamps.append(int(stamp))

    def __len__(self) -> int:
        return len(self.offsets)

    def read_record(self, line_index: int) -> dict[str, Any] | None:
        if line_index < 0 or line_index >= len(self.offsets):
            return None
        with self.path.open("rb") as fh:
            fh.seek(self.offsets[line_index])
            line = fh.readline()
        if not line:
            return None
        return json.loads(line)

    def latest_at_or_before(self, master_stamp: int) -> dict[str, Any] | None:
        if not self.stamps:
            return None
        idx = bisect.bisect_right(self.stamps, int(master_stamp)) - 1
        if idx < 0:
            return None
        return self.read_record(idx)


@dataclass
class EpisodePaths:
    task_root: Path
    episode_index: int
    episode_id: str
    language: str
    length: int
    aligned_extras_path: Path
    data_parquet_path: Path
    video_path: Path | None
    raw_dir: Path

    @classmethod
    def from_task(
        cls,
        task_root: Path,
        episode_index: int,
        episode_id: str,
        language: str,
        length: int,
    ) -> EpisodePaths:
        video = task_root / "videos" / "chunk-000" / "observation.images.left" / f"episode_{episode_index:06d}.mp4"
        return cls(
            task_root=task_root,
            episode_index=int(episode_index),
            episode_id=str(episode_id),
            language=str(language),
            length=int(length),
            aligned_extras_path=(
                task_root / "aligned_extras" / "chunk-000" / f"episode_{episode_index:06d}.parquet"
            ),
            data_parquet_path=task_root / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet",
            video_path=video if video.exists() else None,
            raw_dir=task_root / "raw_streams" / episode_id,
        )


@dataclass
class EpisodeIOHandle:
    """Lightweight per-episode handles (paths + jsonl indices only, no bulk arrays)."""

    paths: EpisodePaths
    image_width: int = 672
    image_height: int = 376
    global_grid_shape: tuple[int, int] = (60, 60)
    local_grid_shape: tuple[int, int] = (14, 14)
    _action_indices: dict[str, JsonlStampIndex] = field(default_factory=dict)
    _boxer_index: JsonlStampIndex | None = None
    _bev_zarr_root: Any | None = None
    _bev_global_arr: Any | None = None
    _bev_local_arr: Any | None = None
    _grid_shapes_set: bool = False
    _has_compact_action_columns: bool | None = None

    @property
    def bev_zarr_path(self) -> Path:
        return self.paths.raw_dir / "bev_output.zarr"

    def _ensure_image_size(self) -> None:
        left_path = self.paths.raw_dir / "left.jsonl"
        if not left_path.exists():
            return
        with left_path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                self.image_width = int(rec.get("width", self.image_width))
                self.image_height = int(rec.get("height", self.image_height))
                return

    def _action_index(self, stream: str) -> JsonlStampIndex:
        if stream not in self._action_indices:
            self._action_indices[stream] = JsonlStampIndex(self.paths.raw_dir / f"{stream}.jsonl")
        return self._action_indices[stream]

    @property
    def boxer_index(self) -> JsonlStampIndex:
        if self._boxer_index is None:
            self._boxer_index = JsonlStampIndex(self.paths.raw_dir / "boxer_output.jsonl")
        return self._boxer_index

    def read_aligned_rows(
        self,
        row_indices: list[int],
        *,
        include_boxer_objects: bool = False,
        boxer_row_indices: list[int] | None = None,
        include_bev_maps: bool = False,
    ) -> dict[str, Any]:
        if pq is None:
            raise ImportError("pyarrow is required to read aligned_extras")
        if not self.paths.aligned_extras_path.exists():
            raise FileNotFoundError(f"missing aligned_extras: {self.paths.aligned_extras_path}")
        schema = pq.read_schema(self.paths.aligned_extras_path)
        cols = list(_ALIGNED_SCALAR_COLS + _ALIGNED_MAP_COLS)
        boxer_cols = (
            [name for name in _ALIGNED_OBJECT_COLS if name in schema.names]
            if include_boxer_objects
            else []
        )
        cols.extend(boxer_cols)
        if include_bev_maps:
            cols.extend(name for name in _ALIGNED_UNCERTAINTY_COLS if name in schema.names)
        boxer_rows = boxer_row_indices if boxer_row_indices is not None else row_indices
        return _read_parquet_columns_at_rows(
            self.paths.aligned_extras_path,
            cols,
            row_indices=row_indices,
            boxer_row_indices=boxer_rows if include_boxer_objects else None,
            raw_dir=self.paths.raw_dir,
        )

    def read_aligned_boxer_objects(self, row_indices: list[int]) -> list[list[dict[str, Any]]] | None:
        if pq is None:
            raise ImportError("pyarrow is required to read aligned_extras")
        if not self.paths.aligned_extras_path.exists():
            raise FileNotFoundError(f"missing aligned_extras: {self.paths.aligned_extras_path}")
        schema = pq.read_schema(self.paths.aligned_extras_path)
        if BOXER_OBJECTS_COLUMN not in schema.names:
            return None
        out = _read_parquet_columns_at_rows(
            self.paths.aligned_extras_path,
            [BOXER_OBJECTS_COLUMN],
            row_indices=row_indices,
            boxer_row_indices=row_indices,
            raw_dir=self.paths.raw_dir,
        )
        return out[BOXER_OBJECTS_COLUMN]

    def _bev_zarr(self) -> Any:
        if self._bev_zarr_root is not None:
            return self._bev_zarr_root
        if zarr is None:
            raise ImportError("zarr is required to read BEV uncertainty maps")
        zarr_path = self.bev_zarr_path
        if not zarr_path.exists():
            raise FileNotFoundError(f"missing BEV zarr: {zarr_path}")
        self._bev_zarr_root = zarr.open(str(zarr_path), mode="r")
        return self._bev_zarr_root

    def _bev_uncertainty_arrays(self) -> tuple[Any, Any]:
        if self._bev_global_arr is not None and self._bev_local_arr is not None:
            return self._bev_global_arr, self._bev_local_arr
        root = self._bev_zarr()
        if "global_uncertainty" not in root or "local_uncertainty" not in root:
            raise KeyError(f"{self.bev_zarr_path} missing uncertainty arrays")
        self._bev_global_arr = root["global_uncertainty"]
        self._bev_local_arr = root["local_uncertainty"]
        return self._bev_global_arr, self._bev_local_arr

    def read_uncertainty_maps(self, bev_raw_indices: list[int]) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Read causally selected BEV uncertainty maps from raw_streams/bev_output.zarr only."""
        global_arr, local_arr = self._bev_uncertainty_arrays()
        if not self._grid_shapes_set:
            self.global_grid_shape = (int(global_arr.shape[1]), int(global_arr.shape[2]))
            self.local_grid_shape = (int(local_arr.shape[1]), int(local_arr.shape[2]))
            self._grid_shapes_set = True

        zero_global = np.zeros(self.global_grid_shape, dtype=np.float32)
        zero_local = np.zeros(self.local_grid_shape, dtype=np.float32)
        global_by_index = _read_zarr_rows(global_arr, bev_raw_indices)
        local_by_index = _read_zarr_rows(local_arr, bev_raw_indices)
        global_rows: list[np.ndarray] = []
        local_rows: list[np.ndarray] = []
        for raw_idx in bev_raw_indices:
            idx = int(raw_idx)
            if idx < 0:
                global_rows.append(zero_global.copy())
                local_rows.append(zero_local.copy())
                continue
            global_rows.append(global_by_index[idx])
            local_rows.append(local_by_index[idx])
        return global_rows, local_rows

    def action_vector_at_stamp(self, stamp_ns: int) -> dict[str, np.ndarray]:
        debug = self._action_index("g1_debug").latest_at_or_before(stamp_ns) or {}
        hand = extract_inspire_action(self._action_index("inspire_action").latest_at_or_before(stamp_ns))
        cam = extract_dynamixel_action(self._action_index("dynamixel_action").latest_at_or_before(stamp_ns))
        measured = extract_measured_action(debug.get("measured"))
        q, dq, quat = _split_measured(measured)
        latent = np.zeros(64, dtype=np.float32)
        latent_state = debug.get("latent_state") or {}
        if isinstance(latent_state, dict) and latent_state.get("token_state") is not None:
            arr = np.asarray(latent_state["token_state"], dtype=np.float32).reshape(-1)
            latent[: min(64, arr.shape[0])] = arr[: min(64, arr.shape[0])]
        return {
            "measured_joint_q": q,
            "measured_joint_dq": dq,
            "body_quat": quat,
            "latent_state": latent,
            "hand_action": halab_hand_action_to_radians(hand["inspire_action"]),
            "camera_action": cam["head_yaw_pitch"],
        }

    def _has_action_parquet_columns(self) -> bool:
        if self._has_compact_action_columns is not None:
            return self._has_compact_action_columns
        required = (
            "action.measured_state",
            "action.latent_state",
            "action.hand_action",
            "action.camera_action",
        )
        if pq is None or not self.paths.data_parquet_path.exists():
            self._has_compact_action_columns = False
            return False
        schema = pq.read_schema(self.paths.data_parquet_path)
        self._has_compact_action_columns = all(name in schema.names for name in required)
        return self._has_compact_action_columns

    def read_action_rows(
        self,
        row_indices: list[int],
        *,
        fallback_stamps: dict[int, int] | None = None,
    ) -> dict[int, dict[str, np.ndarray]]:
        """Read aligned action rows from compact parquet, falling back to raw jsonl for old datasets."""
        required = (
            "action.measured_state",
            "action.latent_state",
            "action.hand_action",
            "action.camera_action",
        )
        if self._has_action_parquet_columns():
            table = pq.read_table(self.paths.data_parquet_path, columns=list(required))
            idx_arr = pa.array(row_indices, type=pa.int64())
            measured_rows = table["action.measured_state"].take(idx_arr).to_pylist()
            latent_rows = table["action.latent_state"].take(idx_arr).to_pylist()
            hand_rows = table["action.hand_action"].take(idx_arr).to_pylist()
            camera_rows = table["action.camera_action"].take(idx_arr).to_pylist()
            out: dict[int, dict[str, np.ndarray]] = {}
            for pos, row_idx in enumerate(row_indices):
                q, dq, quat = _split_measured(np.asarray(measured_rows[pos], dtype=np.float32))
                out[int(row_idx)] = {
                    "measured_joint_q": q,
                    "measured_joint_dq": dq,
                    "body_quat": quat,
                    "latent_state": np.asarray(latent_rows[pos], dtype=np.float32).reshape(-1)[:64],
                    "hand_action": halab_hand_action_to_radians(
                        np.asarray(hand_rows[pos], dtype=np.float32).reshape(-1)[:12]
                    ),
                    "camera_action": np.asarray(camera_rows[pos], dtype=np.float32).reshape(-1)[:2],
                }
            return out

        if fallback_stamps is None:
            raise FileNotFoundError(
                f"{self.paths.data_parquet_path} is missing aligned action columns and no fallback stamps were provided"
            )
        return {
            int(row_idx): self.action_vector_at_stamp(int(fallback_stamps[int(row_idx)]))
            for row_idx in row_indices
        }

    def read_proprio_rows(
        self,
        row_indices: list[int],
        *,
        include_body_vel: bool = True,
    ) -> dict[str, np.ndarray]:
        """Stack absolute proprio from ``observation.state`` at ``row_indices``.

        Returns action-layout keys with shape ``(T, dim)`` each, matching the
        MotionTokenizers / HA-custom concat order ``[q | (dq?) | quat | hand | cam]``.
        """
        if pq is None:
            raise ImportError("pyarrow is required to read observation.state proprio")
        if not self.paths.data_parquet_path.exists():
            raise FileNotFoundError(self.paths.data_parquet_path)
        table = pq.read_table(self.paths.data_parquet_path, columns=["observation.state"])
        idx_arr = pa.array(row_indices, type=pa.int64())
        rows = table["observation.state"].take(idx_arr).to_pylist()
        frames = [
            _split_observation_state(row, include_body_vel=include_body_vel) for row in rows
        ]
        keys = frames[0].keys()
        return {k: np.stack([fd[k] for fd in frames], axis=0) for k in keys}

    def boxer_record_at_aligned_row(self, boxer_raw_index: int) -> dict[str, Any] | None:
        if boxer_raw_index < 0:
            return None
        record = self.boxer_index.read_record(int(boxer_raw_index))
        if record is None:
            return None
        return _attach_boxer_crop_paths(self.paths.raw_dir, record)

    def pose_record_from_matrix(self, pose: np.ndarray) -> dict[str, Any]:
        return {"camera_pose_world": pose.reshape(4, 4).astype(np.float64).tolist()}


class EpisodeIOCache:
    """LRU cache of lightweight EpisodeIOHandle (jsonl indices only, not sample tensors)."""

    def __init__(self, max_episodes: int = 2) -> None:
        self.max_episodes = max(0, int(max_episodes))
        self._handles: OrderedDict[tuple[int, int], EpisodeIOHandle] = OrderedDict()
        self._pid = os.getpid()

    def get(self, key: tuple[int, int], factory: Any) -> EpisodeIOHandle:
        pid = os.getpid()
        if pid != self._pid:
            self._handles.clear()
            self._pid = pid
        if key in self._handles:
            self._handles.move_to_end(key)
            return self._handles[key]
        handle: EpisodeIOHandle = factory()
        handle._ensure_image_size()
        if self.max_episodes <= 0:
            return handle
        self._handles[key] = handle
        self._handles.move_to_end(key)
        while len(self._handles) > self.max_episodes:
            self._handles.popitem(last=False)
        return handle


def stack_aligned_maps(rows: list[np.ndarray], grid_shape: tuple[int, int]) -> np.ndarray:
    h, w = grid_shape
    return np.stack([np.asarray(r, dtype=np.float32).reshape(-1).reshape(h, w) for r in rows], axis=0)


def uncertainty_maps_from_aligned_rows(
    global_rows: list[Any],
    local_rows: list[Any],
) -> tuple[np.ndarray, np.ndarray, tuple[int, int], tuple[int, int]]:
    """Reshape flat aligned_extras BEV map rows into (T, H, W) arrays."""
    global_flat = [np.asarray(r, dtype=np.float32).reshape(-1) for r in global_rows]
    local_flat = [np.asarray(r, dtype=np.float32).reshape(-1) for r in local_rows]
    if not global_flat or not local_flat:
        raise ValueError("aligned_extras BEV uncertainty rows are empty")
    gh, gw = _infer_grid_shape(int(global_flat[0].size))
    lh, lw = _infer_grid_shape(int(local_flat[0].size))
    return (
        stack_aligned_maps(global_flat, (gh, gw)),
        stack_aligned_maps(local_flat, (lh, lw)),
        (gh, gw),
        (lh, lw),
    )


def stack_poses(pose_rows: list[np.ndarray]) -> np.ndarray:
    return np.stack([_pose_row_to_matrix(r) for r in pose_rows], axis=0).astype(np.float32)
