from __future__ import annotations

import bisect
import json
from pathlib import Path
from typing import Any

import numpy as np

from .constants import G1_MOTOR_DIM, RAW_STREAM_NAMES
from .extract import (
    dedupe_by_stamp,
    extract_dynamixel_action,
    extract_dynamixel_fields,
    extract_g1_fields,
    extract_imu_fields,
    extract_inspire_action,
    extract_inspire_fields,
    extract_measured_action,
    extract_pose_fields,
    extract_proprio_vector,
    measured_action_labels,
    stamp_ns_of,
    times_sec,
)
from .jsonl_io import read_jsonl
from .types import EpisodeData, EpisodeInfo, TimeSeries

try:
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover
    pq = None

try:
    import zarr
except ImportError:  # pragma: no cover
    zarr = None


def _sample_video_frame_means(video_path: Path, indices: list[int]) -> dict[int, float]:
    if not video_path.exists() or not indices:
        return {}
    try:
        import cv2
    except ImportError:
        return {}
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {}
    out: dict[int, float] = {}
    try:
        for idx in sorted(set(int(i) for i in indices if i >= 0)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            out[idx] = float(frame.mean())
    finally:
        cap.release()
    return out


def _sample_depth_zarr_means(raw_dir: Path | None, indices: list[int]) -> dict[int, float]:
    if raw_dir is None or not indices:
        return {}
    zarr_path = raw_dir / "depth.zarr"
    if not zarr_path.exists():
        return {}
    try:
        import zarr
    except ImportError:
        return {}
    arr = zarr.open(str(zarr_path), mode="r")
    out: dict[int, float] = {}
    n = int(arr.shape[0])
    for idx in sorted(set(int(i) for i in indices if 0 <= int(i) < n)):
        depth = np.asarray(arr[idx])
        valid = depth[np.isfinite(depth)]
        out[idx] = float(valid.mean()) if valid.size else float("nan")
    return out


def _sample_depth_png_means(
    raw_dir: Path | None,
    records: list[dict[str, Any]],
    indices: list[int],
) -> dict[int, float]:
    if raw_dir is None or not indices:
        return {}
    try:
        import cv2
    except ImportError:
        return {}
    out: dict[int, float] = {}
    for idx in sorted(set(int(i) for i in indices if 0 <= int(i) < len(records))):
        record = records[idx]
        png_path = record.get("png_path")
        if not png_path:
            continue
        img = cv2.imread(str(raw_dir / str(png_path)), cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        scale = float(record.get("depth_scale_m_per_uint", 1.0))
        depth_m = img.astype(np.float32) * scale
        valid = depth_m[np.isfinite(depth_m) & (img > 0)]
        out[idx] = float(valid.mean()) if valid.size else float("nan")
    return out


class LeRobotEpisodeLoader:
    """Load LeRobot v2.1 episodes written by episode_recorder (g1_zed_bridge)."""

    def __init__(self, dataset_root: str | Path):
        self.root = Path(dataset_root).expanduser().resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(f"Dataset root not found: {self.root}")
        self.meta_dir = self.root / "meta"
        self._info = self._read_info()

    def _read_info(self) -> dict[str, Any]:
        path = self.meta_dir / "info.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing meta/info.json under {self.root}")
        return json.loads(path.read_text(encoding="utf-8"))

    @property
    def fps(self) -> float:
        return float(self._info.get("fps", 30.0))

    @property
    def info(self) -> dict[str, Any]:
        return dict(self._info)

    def list_episodes(self) -> list[EpisodeInfo]:
        path = self.meta_dir / "episodes.jsonl"
        if not path.exists():
            return []
        episodes: list[EpisodeInfo] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            episodes.append(
                EpisodeInfo(
                    episode_index=int(row["episode_index"]),
                    episode_id=str(row["episode_id"]),
                    length=int(row["length"]),
                    duration_s=float(row.get("duration_s", 0.0)),
                    tasks=list(row.get("tasks", [])),
                    collection_mode=str(row.get("collection_mode", "teleop")),
                    teleoperator=str(row.get("teleoperator", "")),
                )
            )
        return episodes

    def resolve_episode(
        self,
        *,
        episode_index: int | None = None,
        episode_id: str | None = None,
        path: str | Path | None = None,
    ) -> EpisodeInfo:
        episodes = self.list_episodes()
        if not episodes:
            raise ValueError(f"No episodes in {self.meta_dir / 'episodes.jsonl'}")

        if path is not None:
            p = Path(path).expanduser().resolve()
            if p.is_dir() and p.name.startswith("episode_") and (p.parent.name == "raw_streams" or p.parent == self.root / "raw_streams"):
                episode_id = p.name
                for ep in episodes:
                    if ep.episode_id == episode_id:
                        return ep
                return EpisodeInfo(
                    episode_index=0,
                    episode_id=episode_id,
                    length=0,
                    duration_s=0.0,
                    tasks=[],
                    collection_mode="teleop",
                    teleoperator="",
                )
            if p.is_dir() and (p / "meta" / "episodes.jsonl").exists():
                loader = LeRobotEpisodeLoader(p)
                eps = loader.list_episodes()
                if not eps:
                    raise ValueError(f"No episodes under {p}")
                return eps[0]
            raise ValueError(f"Cannot resolve episode from path: {path}")

        if episode_id is not None:
            for ep in episodes:
                if ep.episode_id == episode_id:
                    return ep
            raise KeyError(f"episode_id not found: {episode_id}")

        idx = 0 if episode_index is None else int(episode_index)
        for ep in episodes:
            if ep.episode_index == idx:
                return ep
        raise KeyError(f"episode_index not found: {idx}")

    def load_episode(
        self,
        *,
        episode_index: int | None = None,
        episode_id: str | None = None,
        path: str | Path | None = None,
        load_parquet: bool = True,
        load_frames: bool = True,
        load_raw: bool = True,
        load_aligned_extras: bool = True,
        dedupe_inspire: bool = True,
    ) -> EpisodeData:
        ep_info = self.resolve_episode(episode_index=episode_index, episode_id=episode_id, path=path)
        idx = ep_info.episode_index
        eid = ep_info.episode_id

        frames_path = self.root / "frames" / f"{eid}.jsonl"
        raw_dir = self.root / "raw_streams" / eid
        if path is not None:
            p = Path(path).expanduser().resolve()
            if p.is_dir() and p.name == eid:
                raw_dir = p
        parquet_path = self.root / "data" / "chunk-000" / f"episode_{idx:06d}.parquet"
        aligned_extras_path = self.root / "aligned_extras" / "chunk-000" / f"episode_{idx:06d}.parquet"
        video_path = self.root / "videos" / "chunk-000" / "observation.images.left" / f"episode_{idx:06d}.mp4"
        if not video_path.exists():
            video_path = None

        frames = read_jsonl(frames_path) if load_frames else []
        start_ns = int(frames[0]["timestamp_ns"]) if frames else 0
        if not frames and load_raw:
            g1_path = raw_dir / "g1.jsonl"
            g1_records = self._read_raw_stream(raw_dir, "g1", dedupe_inspire=dedupe_inspire)
            if g1_records:
                start_ns = int(g1_records[0]["stamp_ns"])

        raw_streams: dict[str, list[dict[str, Any]]] = {}
        if load_raw and raw_dir.is_dir():
            for name in RAW_STREAM_NAMES:
                raw_streams[name] = self._read_raw_stream(raw_dir, name, dedupe_inspire=dedupe_inspire)

        parquet: dict[str, Any] = {}
        if load_parquet and parquet_path.exists():
            parquet = self._read_parquet(parquet_path)
        aligned_extras: dict[str, Any] = {}
        if load_aligned_extras and aligned_extras_path.exists():
            aligned_extras = self._read_aligned_extras(aligned_extras_path)

        motor_names: list[str] = []
        g1_records = raw_streams.get("g1") or []
        if g1_records and g1_records[0].get("motor_names"):
            motor_names = list(g1_records[0]["motor_names"])
        elif frames:
            g1_obs = (frames[0].get("observation") or {}).get("g1") or {}
            motor_names = list(g1_obs.get("motor_names") or [])

        return EpisodeData(
            root=self.root,
            info=ep_info,
            start_ns=start_ns,
            fps=self.fps,
            parquet=parquet,
            frames=frames,
            raw_streams=raw_streams,
            aligned_extras=aligned_extras,
            video_path=video_path,
            frames_path=frames_path if frames_path.exists() else None,
            raw_dir=raw_dir if raw_dir.is_dir() else None,
            parquet_path=parquet_path if parquet_path.exists() else None,
            aligned_extras_path=aligned_extras_path if aligned_extras_path.exists() else None,
            motor_names=motor_names,
        )

    def _read_raw_stream(self, raw_dir: Path, name: str, *, dedupe_inspire: bool) -> list[dict[str, Any]]:
        records = read_jsonl(raw_dir / f"{name}.jsonl")
        if not records:
            return records
        records = self._hydrate_zarr_fields(raw_dir, records)
        if name == "bev_phase":
            records = self._hydrate_phase_zarr_fields(raw_dir, records)
        if name == "boxer_output":
            records = self._attach_boxer_crop_paths(raw_dir, records)
        if name == "inspire_state" and dedupe_inspire:
            records = dedupe_by_stamp(records)
        return records

    def _hydrate_phase_zarr_fields(self, raw_dir: Path, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if zarr is None:
            return records
        hydrated: list[dict[str, Any]] = []
        for record in records:
            zarr_path = record.get("phase_zarr_path")
            zarr_fields = record.get("phase_zarr_fields")
            if not zarr_path or not isinstance(zarr_fields, dict):
                hydrated.append(record)
                continue
            root_path = raw_dir / str(zarr_path)
            if not root_path.exists():
                hydrated.append(record)
                continue
            try:
                root = zarr.open(str(root_path), mode="r")
            except Exception as exc:
                raise RuntimeError(f"Failed to open BEV phase sidecar: {root_path}") from exc
            out = dict(record)
            phase = dict(out.get("phase") or {})
            for field, meta in zarr_fields.items():
                path = str(meta.get("path", field)) if isinstance(meta, dict) else str(field)
                if path in root:
                    phase[field] = np.asarray(root[path]).tolist()
            out["phase"] = phase
            hydrated.append(out)
        return hydrated

    def _hydrate_zarr_fields(self, raw_dir: Path, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        first = records[0] if records else {}
        zarr_path = first.get("zarr_path")
        zarr_fields = first.get("zarr_fields")
        if not zarr_path or not isinstance(zarr_fields, dict):
            return records
        if zarr is None:
            return records
        root_path = raw_dir / str(zarr_path)
        if not root_path.exists():
            return records
        try:
            root = zarr.open(str(root_path), mode="r")
        except Exception as exc:
            raise RuntimeError(
                f"Failed to open {root_path}. This dataset may have been written as Zarr v3, "
                "but the VLA environment uses Zarr v2. Re-record the episode with the updated "
                "writer or convert this sidecar to Zarr v2."
            ) from exc
        arrays = {field: root[field] for field in zarr_fields if field in root}
        if not arrays:
            return records
        hydrated: list[dict[str, Any]] = []
        for idx, record in enumerate(records):
            out = dict(record)
            zidx = int(out.get("zarr_index", idx))
            for field, arr in arrays.items():
                out[field] = np.asarray(arr[zidx]).tolist()
            hydrated.append(out)
        return hydrated

    @staticmethod
    def _attach_boxer_crop_paths(raw_dir: Path, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        hydrated: list[dict[str, Any]] = []
        for record in records:
            out = dict(record)
            objects = out.get("objects") or []
            if not isinstance(objects, list):
                hydrated.append(out)
                continue
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
            hydrated.append(out)
        return hydrated

    def _read_aligned_extras(self, path: Path) -> dict[str, Any]:
        if pq is None:
            raise ImportError("pyarrow is required to read aligned extras parquet")
        table = pq.read_table(path)
        data: dict[str, Any] = {"columns": table.column_names, "num_rows": table.num_rows}
        scalar_bev_columns = {
            "bev_floor_ready",
            "bev_raw_index",
            "bev_stamp_ns",
            "bev_is_bos",
            "bev_target_ix",
            "bev_target_iy",
            "bev_target_local_ix",
            "bev_target_local_iy",
            "bev_target_local_clipped",
        }
        for name in table.column_names:
            col = table[name]
            if name.startswith("bev_") and name not in scalar_bev_columns:
                data[name] = self._list_column_to_padded_array(col.to_pylist(), dtype=np.float32)
            else:
                data[name] = np.asarray(col.to_numpy(zero_copy_only=False))
        return data

    @staticmethod
    def _list_column_to_padded_array(rows: list[Any], *, dtype: Any) -> np.ndarray:
        width = 0
        flat_rows: list[np.ndarray] = []
        for row in rows:
            arr = np.asarray(row if row is not None else [], dtype=dtype).reshape(-1)
            flat_rows.append(arr)
            width = max(width, int(arr.shape[0]))
        out = np.full((len(flat_rows), width), np.nan, dtype=dtype)
        for idx, arr in enumerate(flat_rows):
            if arr.size:
                out[idx, : arr.size] = arr
        return out

    def _read_parquet(self, path: Path) -> dict[str, Any]:
        if pq is None:
            raise ImportError("pyarrow is required to read parquet. Install: apt install python3-pyarrow")
        table = pq.read_table(path)
        data: dict[str, Any] = {"columns": table.column_names, "num_rows": table.num_rows}
        for name in table.column_names:
            col = table[name]
            if name in (
                "observation.state",
                "action.latent_state",
                "action.measured_state",
                "action.hand_action",
                "action.camera_action",
            ):
                rows = col.to_pylist()
                data[name] = np.asarray(rows, dtype=np.float32)
            else:
                data[name] = np.asarray(col.to_numpy(zero_copy_only=False))
        return data

    def aligned_timeseries(self, episode: EpisodeData) -> dict[str, TimeSeries]:
        """30 Hz aligned samples from compact aligned_extras or legacy frames/*.jsonl."""
        if not episode.frames:
            return self._aligned_timeseries_from_compact(episode)
        stamps = [int(f["timestamp_ns"]) for f in episode.frames]
        t_sec = times_sec(stamps, episode.start_ns)
        out: dict[str, TimeSeries] = {}

        def add_matrix(name: str, matrices: list[np.ndarray], labels: list[str], source: str) -> None:
            if not matrices:
                return
            arr = np.stack(matrices, axis=0)
            out[name] = TimeSeries(name=name, stamp_ns=stamps, t_sec=t_sec, values=arr.tolist(), source=source)
            out[f"{name}__labels"] = TimeSeries(
                name=f"{name}__labels",
                stamp_ns=[],
                t_sec=[],
                values=labels,
                source=source,
            )

        g1_q, g1_dq, g1_tau = [], [], []
        inspire_q, inspire_l, inspire_r = [], [], []
        inspire_lf, inspire_rf = [], []
        dyn_state, dyn_cmd = [], []
        pose7, latent, measured_state = [], [], []
        hand_cmd_l, hand_cmd_r = [], []

        for frame in episode.frames:
            obs = frame.get("observation") or {}
            act = frame.get("action") or {}
            g1 = extract_g1_fields(obs.get("g1"))
            ins = extract_inspire_fields(obs.get("inspire_state"))
            dyn = extract_dynamixel_fields(obs.get("dynamixel_state"))
            pose = extract_pose_fields(obs.get("camera_pose"))
            hand = extract_inspire_action(act.get("hand") or act.get("inspire_action"))
            cam = extract_dynamixel_action(act.get("camera") or act.get("dynamixel_action"))

            g1_q.append(g1["motor_q"])
            g1_dq.append(g1["motor_dq"])
            g1_tau.append(g1["motor_tau_est"])
            inspire_q.append(ins["inspire_q"])
            inspire_l.append(ins["left_q"])
            inspire_r.append(ins["right_q"])
            inspire_lf.append(ins["left_force_raw"])
            inspire_rf.append(ins["right_force_raw"])
            dyn_state.append(dyn["yaw_pitch"])
            pose7.append(pose["pose7"])
            hand_cmd_l.append(hand["left_hand_desired"])
            hand_cmd_r.append(hand["right_hand_desired"])
            dyn_cmd.append(cam["head_yaw_pitch"])
            latent_vec = np.zeros(64, dtype=np.float32)
            latent_state = (act.get("latent_state") or {})
            if isinstance(latent_state, dict) and latent_state.get("token_state") is not None:
                arr = np.asarray(latent_state["token_state"], dtype=np.float32).reshape(-1)
                latent_vec[: min(64, arr.shape[0])] = arr[: min(64, arr.shape[0])]
            latent.append(latent_vec)
            measured_state.append(extract_measured_action(act.get("measured")))

        labels_g1 = episode.motor_names or [f"motor_{i}" for i in range(g1_q[0].shape[0])]
        add_matrix("aligned/g1/motor_q", g1_q, labels_g1, "frames")
        add_matrix("aligned/g1/motor_dq", g1_dq, labels_g1, "frames")
        add_matrix("aligned/g1/motor_tau_est", g1_tau, labels_g1, "frames")
        from .extract import inspire_joint_labels, inspire_force_labels, pose_labels

        add_matrix("aligned/inspire/q", inspire_q, inspire_joint_labels(), "frames")
        add_matrix("aligned/inspire/left_q", inspire_l, inspire_joint_labels()[:6], "frames")
        add_matrix("aligned/inspire/right_q", inspire_r, inspire_joint_labels()[6:], "frames")
        add_matrix("aligned/inspire/left_force_raw", inspire_lf, inspire_force_labels()[:6], "frames")
        add_matrix("aligned/inspire/right_force_raw", inspire_rf, inspire_force_labels()[6:], "frames")
        add_matrix("aligned/dynamixel/state", dyn_state, ["yaw", "pitch"], "frames")
        add_matrix("aligned/dynamixel/cmd", dyn_cmd, ["yaw_cmd", "pitch_cmd"], "frames")
        add_matrix("aligned/camera_pose", pose7, pose_labels(), "frames")
        add_matrix("aligned/hand_cmd/left", hand_cmd_l, inspire_joint_labels()[:6], "frames")
        add_matrix("aligned/hand_cmd/right", hand_cmd_r, inspire_joint_labels()[6:], "frames")
        add_matrix("aligned/action/latent_state", latent, [f"token_{i}" for i in range(64)], "frames")
        add_matrix("aligned/action/measured_state", measured_state, measured_action_labels(), "frames")

        proprio = [extract_proprio_vector(f.get("observation") or {}) for f in episode.frames]
        add_matrix("aligned/parquet/proprio", proprio, labels_g1 + inspire_joint_labels() + ["dyn_yaw", "dyn_pitch"], "frames")
        return out

    def _aligned_timeseries_from_compact(self, episode: EpisodeData) -> dict[str, TimeSeries]:
        extras = episode.aligned_extras
        if not extras or "timestamp_ns" not in extras:
            return {}
        stamps = [int(s) for s in np.asarray(extras["timestamp_ns"]).reshape(-1).tolist()]
        t_sec = times_sec(stamps, episode.start_ns)
        out: dict[str, TimeSeries] = {}

        def add_matrix(name: str, matrices: list[np.ndarray], labels: list[str], source: str) -> None:
            if not matrices:
                return
            arr = np.stack(matrices, axis=0)
            if arr.shape[0] != len(stamps):
                return
            out[name] = TimeSeries(name=name, stamp_ns=stamps, t_sec=t_sec, values=arr.tolist(), source=source)
            out[f"{name}__labels"] = TimeSeries(
                name=f"{name}__labels",
                stamp_ns=[],
                t_sec=[],
                values=labels,
                source=source,
            )

        def stream_index(stream: str) -> tuple[list[int], list[dict[str, Any]]]:
            records = [
                record
                for record in sorted(episode.raw_streams.get(stream) or [], key=lambda r: int(stamp_ns_of(r) or 0))
                if stamp_ns_of(record) is not None
            ]
            return [int(stamp_ns_of(r)) for r in records], records

        indices = {stream: stream_index(stream) for stream in episode.raw_streams.keys()}

        def latest(stream: str, master: int) -> dict[str, Any] | None:
            stream_stamps, records = indices.get(stream, ([], []))
            if not stream_stamps:
                return None
            idx = bisect.bisect_right(stream_stamps, int(master)) - 1
            if idx < 0:
                return None
            return records[idx]

        from .extract import inspire_joint_labels, inspire_force_labels, pose_labels

        labels_g1 = episode.motor_names or [f"motor_{i}" for i in range(G1_MOTOR_DIM)]
        g1_q, g1_dq, g1_tau = [], [], []
        inspire_q, inspire_l, inspire_r = [], [], []
        inspire_lf, inspire_rf = [], []
        dyn_state, dyn_cmd = [], []
        pose7, latent, measured_state = [], [], []
        hand_cmd_l, hand_cmd_r = [], []

        for master in stamps:
            g1 = extract_g1_fields(latest("g1", master))
            ins = extract_inspire_fields(latest("inspire_state", master))
            dyn = extract_dynamixel_fields(latest("dynamixel_state", master))
            pose = extract_pose_fields(latest("camera_pose", master))
            hand = extract_inspire_action(latest("inspire_action", master))
            cam = extract_dynamixel_action(latest("dynamixel_action", master))
            debug = latest("g1_debug", master) or {}

            g1_q.append(g1["motor_q"])
            g1_dq.append(g1["motor_dq"])
            g1_tau.append(g1["motor_tau_est"])
            inspire_q.append(ins["inspire_q"])
            inspire_l.append(ins["left_q"])
            inspire_r.append(ins["right_q"])
            inspire_lf.append(ins["left_force_raw"])
            inspire_rf.append(ins["right_force_raw"])
            dyn_state.append(dyn["yaw_pitch"])
            pose7.append(pose["pose7"])
            hand_cmd_l.append(hand["left_hand_desired"])
            hand_cmd_r.append(hand["right_hand_desired"])
            dyn_cmd.append(cam["head_yaw_pitch"])

            latent_vec = np.zeros(64, dtype=np.float32)
            latent_state = debug.get("latent_state") or {}
            if isinstance(latent_state, dict) and latent_state.get("token_state") is not None:
                arr = np.asarray(latent_state["token_state"], dtype=np.float32).reshape(-1)
                latent_vec[: min(64, arr.shape[0])] = arr[: min(64, arr.shape[0])]
            latent.append(latent_vec)
            measured_state.append(extract_measured_action(debug.get("measured")))

        source = "aligned_extras+raw_streams"
        add_matrix("aligned/g1/motor_q", g1_q, labels_g1, source)
        add_matrix("aligned/g1/motor_dq", g1_dq, labels_g1, source)
        add_matrix("aligned/g1/motor_tau_est", g1_tau, labels_g1, source)
        add_matrix("aligned/inspire/q", inspire_q, inspire_joint_labels(), source)
        add_matrix("aligned/inspire/left_q", inspire_l, inspire_joint_labels()[:6], source)
        add_matrix("aligned/inspire/right_q", inspire_r, inspire_joint_labels()[6:], source)
        add_matrix("aligned/inspire/left_force_raw", inspire_lf, inspire_force_labels()[:6], source)
        add_matrix("aligned/inspire/right_force_raw", inspire_rf, inspire_force_labels()[6:], source)
        add_matrix("aligned/dynamixel/state", dyn_state, ["yaw", "pitch"], source)
        add_matrix("aligned/dynamixel/cmd", dyn_cmd, ["yaw_cmd", "pitch_cmd"], source)
        add_matrix("aligned/camera_pose", pose7, pose_labels(), source)
        add_matrix("aligned/hand_cmd/left", hand_cmd_l, inspire_joint_labels()[:6], source)
        add_matrix("aligned/hand_cmd/right", hand_cmd_r, inspire_joint_labels()[6:], source)
        add_matrix("aligned/action/latent_state", latent, [f"token_{i}" for i in range(64)], source)
        add_matrix("aligned/action/measured_state", measured_state, measured_action_labels(), source)
        proprio = [np.concatenate([q, ins, dyn], axis=0) for q, ins, dyn in zip(g1_q, inspire_q, dyn_state, strict=False)]
        add_matrix("aligned/parquet/proprio", proprio, labels_g1 + inspire_joint_labels() + ["dyn_yaw", "dyn_pitch"], source)
        return out

    def raw_timeseries(self, episode: EpisodeData, stream: str) -> dict[str, TimeSeries]:
        records = episode.raw_streams.get(stream) or []
        if not records:
            return {}
        stamps = [int(stamp_ns_of(r)) for r in records if stamp_ns_of(r) is not None]
        if not stamps:
            return {}
        t_sec = times_sec(stamps, episode.start_ns)
        out: dict[str, TimeSeries] = {}

        def stack_field(name: str, vectors: list[np.ndarray], labels: list[str]) -> None:
            out[name] = TimeSeries(
                name=f"raw/{stream}/{name}",
                stamp_ns=stamps,
                t_sec=t_sec,
                values=np.stack(vectors, axis=0).tolist(),
                source=f"raw_streams/{stream}.jsonl",
            )
            out[f"raw/{stream}/{name}__labels"] = TimeSeries(
                name=f"raw/{stream}/{name}__labels",
                stamp_ns=[],
                t_sec=[],
                values=labels,
                source=stream,
            )

        if stream == "g1":
            fields = [extract_g1_fields(r) for r in records]
            labels = list(records[0].get("motor_names") or [f"motor_{i}" for i in range(G1_MOTOR_DIM)])
            for key in ("motor_q", "motor_dq", "motor_tau_est"):
                stack_field(key, [f[key] for f in fields], labels)
            for key in ("base_quat_wxyz", "base_angular_velocity", "base_linear_acceleration"):
                stack_field(key, [f[key] for f in fields], [key] * len(fields[0][key]))
        elif stream == "inspire_state":
            fields = [extract_inspire_fields(r) for r in records]
            from .extract import inspire_joint_labels, inspire_force_labels

            stack_field("left_q", [f["left_q"] for f in fields], inspire_joint_labels()[:6])
            stack_field("right_q", [f["right_q"] for f in fields], inspire_joint_labels()[6:])
            stack_field("left_force_raw", [f["left_force_raw"] for f in fields], inspire_force_labels()[:6])
            stack_field("right_force_raw", [f["right_force_raw"] for f in fields], inspire_force_labels()[6:])
            stack_field("left_force_N", [f["left_force_N"] for f in fields], inspire_force_labels()[:6])
            stack_field("right_force_N", [f["right_force_N"] for f in fields], inspire_force_labels()[6:])
        elif stream == "imu":
            fields = [extract_imu_fields(r) for r in records]
            for key in ("angular_velocity", "linear_acceleration", "orientation"):
                stack_field(key, [f[key] for f in fields], [f"{key}_{i}" for i in range(len(fields[0][key]))])
        elif stream == "inspire_action":
            fields = [extract_inspire_action(r) for r in records]
            from .extract import inspire_joint_labels

            stack_field("left_hand_desired", [f["left_hand_desired"] for f in fields], inspire_joint_labels()[:6])
            stack_field("right_hand_desired", [f["right_hand_desired"] for f in fields], inspire_joint_labels()[6:])
        elif stream == "dynamixel_action":
            fields = [extract_dynamixel_action(r) for r in records]
            stack_field("head_yaw_pitch", [f["head_yaw_pitch"] for f in fields], ["yaw", "pitch"])
        elif stream == "dynamixel_state":
            fields = [extract_dynamixel_fields(r) for r in records]
            stack_field("yaw_pitch", [f["yaw_pitch"] for f in fields], ["yaw", "pitch"])
        elif stream == "camera_pose":
            fields = [extract_pose_fields(r) for r in records]
            from .extract import pose_labels

            stack_field("pose7", [f["pose7"] for f in fields], pose_labels())
        elif stream in ("left", "depth"):
            # Pixel decode is expensive; use metadata only unless decode_image_stats=True.
            nbytes = [float(r.get("data_nbytes", 0)) for r in records]
            out[f"raw/{stream}/data_nbytes"] = TimeSeries(
                name=f"raw/{stream}/data_nbytes",
                stamp_ns=stamps,
                t_sec=t_sec,
                values=nbytes,
                source=stream,
            )
        return out

    def raw_image_mean_series(
        self,
        episode: EpisodeData,
        stream: str,
        *,
        max_samples: int = 32,
    ) -> TimeSeries | None:
        """Decode up to max_samples frames and return per-frame mean (for spot-check plots)."""
        records = episode.raw_streams.get(stream) or []
        if not records:
            return None
        stride = max(1, len(records) // max(1, max_samples))
        sample_indices = list(range(0, len(records), stride))[:max_samples]
        sampled = [records[i] for i in sample_indices]
        stamps: list[int] = []
        means: list[float] = []
        from .decode import decode_depth_image

        video_means: dict[int, float] = {}
        if stream == "left" and episode.video_path is not None:
            video_means = _sample_video_frame_means(episode.video_path, sample_indices)
        depth_zarr_means: dict[int, float] = {}
        if stream == "depth":
            depth_zarr_means = _sample_depth_zarr_means(episode.raw_dir, sample_indices)
        depth_png_means: dict[int, float] = {}
        if stream == "depth" and not depth_zarr_means:
            depth_png_means = _sample_depth_png_means(episode.raw_dir, records, sample_indices)

        for idx, r in zip(sample_indices, sampled, strict=False):
            stamp = stamp_ns_of(r)
            if stamp is None:
                continue
            if stream == "left":
                val = video_means.get(idx, float("nan"))
            else:
                if idx in depth_zarr_means:
                    val = depth_zarr_means[idx]
                elif idx in depth_png_means:
                    val = depth_png_means[idx]
                else:
                    dep = decode_depth_image(r)
                    valid = dep[np.isfinite(dep)] if dep is not None else np.array([])
                    val = float(valid.mean()) if valid.size else float("nan")
            stamps.append(int(stamp))
            means.append(val)
        if not stamps:
            return None
        return TimeSeries(
            name=f"raw/{stream}/mean_sampled",
            stamp_ns=stamps,
            t_sec=times_sec(stamps, episode.start_ns),
            values=means,
            source=(
                f"{episode.video_path.name} (sampled)"
                if stream == "left" and episode.video_path
                else "depth.zarr (sampled)"
                if stream == "depth" and depth_zarr_means
                else "depth_png/*.png (sampled)"
                if stream == "depth" and depth_png_means
                else f"{stream}.jsonl (sampled)"
            ),
        )

    def parquet_timeseries(self, episode: EpisodeData) -> dict[str, TimeSeries]:
        p = episode.parquet
        if not p or "timestamp" not in p:
            return {}
        stamps = [episode.start_ns + int(t * 1e9) for t in p["timestamp"].tolist()]
        t_sec = p["timestamp"].tolist()
        out: dict[str, TimeSeries] = {}
        for key in (
            "observation.state",
            "action.latent_state",
            "action.measured_state",
            "action.hand_action",
            "action.camera_action",
        ):
            if key not in p:
                continue
            arr = p[key]
            out[f"parquet/{key}"] = TimeSeries(
                name=f"parquet/{key}",
                stamp_ns=stamps,
                t_sec=t_sec,
                values=arr.tolist(),
                source=str(episode.parquet_path),
            )
        return out
