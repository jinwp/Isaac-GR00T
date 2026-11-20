from __future__ import annotations

import base64
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .extract import stamp_ns_of

# Match test_all.py CLI defaults (used when inferring BEV grid from recorded maps).
_DEFAULT_OCC_BOUNDS = (-6.0, 6.0, -6.0, 6.0)
_DEFAULT_OCC_RES = 0.05
_DEFAULT_LOCAL_HALF_M = 4.0
_DEFAULT_LOCAL_RES_M = 0.6
_DEFAULT_OCC_PIXEL_SCALE = 4
_BOXER_BEV_EVOLUTION_MP4_NAME = "occupancy_uncertainty_global_local_evolution_boxer_object.mp4"
_BOXER_BEV_PIXEL_SCALE = 3  # grid render scale (review_bev default); labels use PIL separately
_DEFAULT_PROC_SIZE_WH = (672, 376)  # ZED Mini native (matches boxer/ZED_Mini.yaml Camera.width/height)
# Hardcoded from boxer/ZED_Mini.yaml Camera1.* at native resolution (672x376).
_ZED_MINI_NATIVE_WH = (672, 376)
_ZED_MINI_FX = 359.222
_ZED_MINI_FY = 359.222
_ZED_MINI_CX = 324.324
_ZED_MINI_CY = 187.275
_DEFAULT_INTRINSIC = np.array(
    [[_ZED_MINI_FX, 0.0, _ZED_MINI_CX], [0.0, _ZED_MINI_FY, _ZED_MINI_CY], [0.0, 0.0, 1.0]],
    dtype=np.float64,
)
_BOX_EDGES = (
    (0, 1),
    (1, 3),
    (3, 2),
    (2, 0),
    (4, 5),
    (5, 7),
    (7, 6),
    (6, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
)


def _slug(text: str, max_len: int = 48) -> str:
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(text).strip())
    return (s[:max_len] if s else "unknown").strip("_") or "unknown"


def _pick_sample_indices(n: int, max_samples: int) -> list[int]:
    if n <= 0:
        return []
    if n <= max_samples:
        return list(range(n))
    step = max(1, n // max_samples)
    idx = list(range(0, n, step))
    if idx[-1] != n - 1:
        idx.append(n - 1)
    return idx[:max_samples]


def _uncertainty_to_rgb(field: np.ndarray) -> np.ndarray:
    """Simple jet-like heatmap (H,W,3) uint8 for QA."""
    arr = np.asarray(field, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"expected 2D uncertainty, got shape {arr.shape}")
    finite = np.isfinite(arr)
    if not np.any(finite):
        return np.zeros((arr.shape[0], arr.shape[1], 3), dtype=np.uint8)
    vmin = float(np.min(arr[finite]))
    vmax = float(np.max(arr[finite]))
    if vmax <= vmin:
        vmax = vmin + 1.0
    norm = np.clip((arr - vmin) / (vmax - vmin), 0.0, 1.0)
    norm[~finite] = 0.0
    r = np.clip(1.5 - np.abs(4.0 * norm - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * norm - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * norm - 1.0), 0.0, 1.0)
    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255.0).astype(np.uint8)


def _save_rgb_png(path: Path, rgb: np.ndarray) -> None:
    try:
        import cv2
    except ImportError:
        import matplotlib.pyplot as plt

        plt.imsave(str(path), rgb)
        return
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)


def _record_stamp(record: dict[str, Any]) -> int:
    stamp = stamp_ns_of(record)
    return int(stamp if stamp is not None else record.get("stamp_ns", 0))


def _causal_record(records: list[dict[str, Any]], stamp_ns: int, *, max_slop_ns: int) -> dict[str, Any] | None:
    best = _causal_record_before_stamp(records, stamp_ns)
    if best is None:
        return None
    if stamp_ns - _record_stamp(best) > max_slop_ns:
        return None
    return best


def _causal_record_before_stamp(records: list[dict[str, Any]], stamp_ns: int) -> dict[str, Any] | None:
    """Latest record with stamp <= query (hold-forward causal join, no slop cutoff)."""
    best: dict[str, Any] | None = None
    best_stamp = -1
    for rec in records:
        rec_stamp = _record_stamp(rec)
        if rec_stamp <= stamp_ns and rec_stamp > best_stamp:
            best = rec
            best_stamp = rec_stamp
        if rec_stamp > stamp_ns:
            break
    return best


def _causal_record_with_fallback(
    records: list[dict[str, Any]], stamp_ns: int, *, max_slop_ns: int
) -> dict[str, Any] | None:
    """Match FrameAssembler recording: causal within slop, else latest stamp <= master."""
    rec = _causal_record(records, stamp_ns, max_slop_ns=max_slop_ns)
    if rec is not None:
        return rec
    return _causal_record_before_stamp(records, stamp_ns)


def _pose_tcw(record: dict[str, Any] | None) -> np.ndarray | None:
    if not record:
        return None
    raw = record.get("camera_pose_world")
    if raw is None:
        raw = record.get("camera_poses_world")
    if raw is not None:
        arr = np.asarray(raw, dtype=np.float64)
        if arr.size == 16:
            return np.linalg.inv(arr.reshape(4, 4))
    raw = record.get("Tcw")
    if raw is not None:
        arr = np.asarray(raw, dtype=np.float64)
        if arr.size == 16:
            return arr.reshape(4, 4)
    raw = record.get("Twc")
    if raw is not None:
        arr = np.asarray(raw, dtype=np.float64)
        if arr.size == 16:
            return np.linalg.inv(arr.reshape(4, 4))
    return None


def _intrinsic_for_image(width: int, height: int) -> np.ndarray:
    native_w, native_h = _ZED_MINI_NATIVE_WH
    scale_x = float(width) / float(native_w)
    scale_y = float(height) / float(native_h)
    return np.array(
        [
            [_ZED_MINI_FX * scale_x, 0.0, _ZED_MINI_CX * scale_x],
            [0.0, _ZED_MINI_FY * scale_y, _ZED_MINI_CY * scale_y],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _object_corners_world(obj: dict[str, Any]) -> np.ndarray | None:
    raw_t = obj.get("T_world_object")
    raw_size = obj.get("bbox_size_xyz")
    if raw_t is None or raw_size is None:
        return None
    T = np.asarray(raw_t, dtype=np.float64)
    size = np.asarray(raw_size, dtype=np.float64).reshape(-1)
    if T.size != 16 or size.size < 3:
        return None
    T = T.reshape(4, 4)
    half = size[:3] * 0.5
    local = np.array(
        [
            [-half[0], -half[1], -half[2], 1.0],
            [half[0], -half[1], -half[2], 1.0],
            [-half[0], half[1], -half[2], 1.0],
            [half[0], half[1], -half[2], 1.0],
            [-half[0], -half[1], half[2], 1.0],
            [half[0], -half[1], half[2], 1.0],
            [-half[0], half[1], half[2], 1.0],
            [half[0], half[1], half[2], 1.0],
        ],
        dtype=np.float64,
    )
    return (T @ local.T).T[:, :3]


def _project_points(points_world: np.ndarray, Tcw: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pts_h = np.concatenate([points_world, np.ones((points_world.shape[0], 1), dtype=np.float64)], axis=1)
    pts_cam = (Tcw @ pts_h.T).T[:, :3]
    z = pts_cam[:, 2]
    valid = z > 1e-4
    uv = np.full((points_world.shape[0], 2), np.nan, dtype=np.float64)
    uv[valid, 0] = K[0, 0] * (pts_cam[valid, 0] / z[valid]) + K[0, 2]
    uv[valid, 1] = K[1, 1] * (pts_cam[valid, 1] / z[valid]) + K[1, 2]
    return uv, valid


def _object_projected_in_image(
    obj: dict[str, Any],
    Tcw: np.ndarray,
    K: np.ndarray,
    width: int,
    height: int,
    *,
    margin_px: float = 0.0,
) -> bool:
    corners = _object_corners_world(obj)
    if corners is None:
        return False
    uv, valid = _project_points(corners, Tcw, K)
    if not any(valid[a] and valid[b] for a, b in _BOX_EDGES):
        return False
    finite_uv = uv[valid]
    if finite_uv.size == 0:
        return False
    x_min = float(np.nanmin(finite_uv[:, 0]))
    x_max = float(np.nanmax(finite_uv[:, 0]))
    y_min = float(np.nanmin(finite_uv[:, 1]))
    y_max = float(np.nanmax(finite_uv[:, 1]))
    return (
        x_max >= -margin_px
        and x_min <= float(width - 1) + margin_px
        and y_max >= -margin_px
        and y_min <= float(height - 1) + margin_px
    )


def _object_center_size(obj: dict[str, Any]) -> tuple[np.ndarray, np.ndarray] | None:
    center_raw = obj.get("bbox_center_world")
    if center_raw is not None:
        center = np.asarray(center_raw, dtype=np.float64).reshape(-1)
    else:
        raw_t = obj.get("T_world_object")
        if raw_t is None:
            return None
        T = np.asarray(raw_t, dtype=np.float64)
        if T.size != 16:
            return None
        center = T.reshape(4, 4)[:3, 3]
    size_raw = obj.get("bbox_size_xyz")
    if size_raw is None:
        return None
    size = np.asarray(size_raw, dtype=np.float64).reshape(-1)
    if center.size < 3 or size.size < 3:
        return None
    return center[:3], np.maximum(size[:3], 1e-6)


def _dedupe_boxer_objects(
    objects: list[dict[str, Any]],
    *,
    center_m: float = 0.35,
    size_ratio: float = 0.5,
) -> list[dict[str, Any]]:
    if len(objects) < 2:
        return objects

    def priority(obj: dict[str, Any]) -> tuple[int, int, float, int, int]:
        state_rank = 2 if str(obj.get("track_state", "")).upper() == "ACTIVE" else 1
        score = obj.get("track_panel_score", obj.get("crop_panel_score", 0.0))
        try:
            score_f = float(score)
        except (TypeError, ValueError):
            score_f = 0.0
        return (
            state_rank,
            int(obj.get("bbox_stamp_ns") or obj.get("stamp_ns") or 0),
            score_f,
            int(obj.get("language_label_stamp_ns") or 0),
            -int(obj.get("track_id", 0)),
        )

    kept: list[dict[str, Any]] = []
    kept_geom: list[tuple[np.ndarray, np.ndarray]] = []
    for obj in sorted(objects, key=priority, reverse=True):
        geom = _object_center_size(obj)
        if geom is None:
            kept.append(obj)
            continue
        center, size = geom
        duplicate = False
        for kept_center, kept_size in kept_geom:
            center_dist = float(np.linalg.norm(center - kept_center))
            per_axis_ratio = np.minimum(size, kept_size) / np.maximum(size, kept_size)
            if center_dist <= center_m and float(np.min(per_axis_ratio)) >= size_ratio:
                duplicate = True
                break
        if duplicate:
            continue
        kept.append(obj)
        kept_geom.append((center, size))

    return sorted(kept, key=lambda obj: int(obj.get("track_id", -1)))


def filter_boxer_record(
    boxer_record: dict[str, Any],
    pose_record: dict[str, Any] | None,
    *,
    object_policy: str,
    image_width: int,
    image_height: int,
) -> dict[str, Any]:
    """
    Apply the same Boxer object filter as ``export_boxer_viz`` / ``test_data_loader.py``.

    ``pose_record`` should be the aligned BEV record (floor_aligned ``camera_pose_world``)
    when ``object_policy`` is ``active_inactive_dedup`` (matches ``--boxer-viz-pose-stream bev_output``).
    """
    if object_policy not in ("recorded", "active", "active_inactive_dedup", "active_memory"):
        raise ValueError(f"unsupported boxer object_policy={object_policy!r}")
    filtered, _ = _filtered_boxer_record_for_viz(
        boxer_record,
        pose_record or {},
        int(image_width),
        int(image_height),
        object_policy=object_policy,
    )
    return filtered


def _filtered_boxer_record_for_viz(
    boxer_record: dict[str, Any],
    pose_record: dict[str, Any],
    width: int,
    height: int,
    *,
    object_policy: str,
) -> tuple[dict[str, Any], int]:
    objects = boxer_record.get("objects") or []
    if not isinstance(objects, list):
        return boxer_record, 0
    original_count = len(objects)
    if object_policy == "recorded":
        return boxer_record, original_count

    Tcw = _pose_tcw(pose_record)
    K = _intrinsic_for_image(width, height)
    filtered: list[dict[str, Any]] = []
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        state = str(obj.get("track_state", "")).upper()
        if object_policy == "active":
            if state == "ACTIVE":
                filtered.append(obj)
        elif object_policy == "active_inactive_dedup":
            if state in ("ACTIVE", "INACTIVE"):
                filtered.append(obj)
        elif object_policy == "active_memory":
            if state == "ACTIVE":
                filtered.append(obj)
            elif state == "INACTIVE" and Tcw is not None:
                if not _object_projected_in_image(obj, Tcw, K, width, height):
                    filtered.append(obj)
        else:
            raise ValueError(f"unsupported boxer object_policy={object_policy!r}")

    out = dict(boxer_record)
    out["objects"] = (
        _dedupe_boxer_objects(filtered)
        if object_policy in ("active_memory", "active_inactive_dedup")
        else filtered
    )
    return out, original_count


def _draw_boxer_overlay(rgb: np.ndarray, boxer_record: dict[str, Any], pose_record: dict[str, Any]) -> np.ndarray:
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("OpenCV is required for --export-boxer-viz") from exc

    out = np.ascontiguousarray(rgb.copy())
    h, w = out.shape[:2]
    K = _intrinsic_for_image(w, h)
    Tcw = _pose_tcw(pose_record)
    if Tcw is None:
        return out
    color = (31, 119, 180)  # RGB, matches test_orb_ros track panel blue.
    for obj in boxer_record.get("objects") or []:
        if not isinstance(obj, dict):
            continue
        corners = _object_corners_world(obj)
        if corners is None:
            continue
        uv, valid = _project_points(corners, Tcw, K)
        if not _object_projected_in_image(obj, Tcw, K, w, h):
            continue
        any_edge = False
        for a, b in _BOX_EDGES:
            if not (valid[a] and valid[b]):
                continue
            p0 = tuple(np.round(uv[a]).astype(int).tolist())
            p1 = tuple(np.round(uv[b]).astype(int).tolist())
            cv2.line(out, p0, p1, color, 2, cv2.LINE_AA)
            any_edge = True
        if not any_edge:
            continue
        finite_uv = uv[valid]
        if finite_uv.size == 0:
            continue
        label = str(obj.get("language_label") or "unknown")[:10]
        score = obj.get("track_panel_score")
        if score is None:
            score = obj.get("crop_panel_score")
        if score is not None:
            text = f"{label} {float(score):.2f}"
        else:
            text = f"{label} #{int(obj.get('track_id', -1))}"
        x = int(np.clip(np.nanmin(finite_uv[:, 0]), 0, max(0, w - 1)))
        y = int(np.clip(np.nanmin(finite_uv[:, 1]) - 4, 12, max(12, h - 1)))
        cv2.putText(out, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return out


def _drawable_box_count(boxer_record: dict[str, Any], pose_record: dict[str, Any], width: int, height: int) -> int:
    Tcw = _pose_tcw(pose_record)
    if Tcw is None:
        return 0
    K = _intrinsic_for_image(width, height)
    count = 0
    for obj in boxer_record.get("objects") or []:
        if not isinstance(obj, dict):
            continue
        corners = _object_corners_world(obj)
        if corners is None:
            continue
        if _object_projected_in_image(obj, Tcw, K, width, height):
            count += 1
    return count


def export_bev_review(
    records: list[dict[str, Any]],
    out_dir: Path,
    *,
    max_samples: int = 12,
) -> list[Path]:
    """Save BEV uncertainty PNGs from recorded bev_output.jsonl (occupancy not in ROS payload)."""
    saved: list[Path] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    indices = _pick_sample_indices(len(records), max_samples)
    for rank, idx in enumerate(indices):
        rec = records[idx]
        stamp = stamp_ns_of(rec)
        frame_idx = rec.get("frame_index", idx)
        for key, stem in (
            ("global_uncertainty", "global_unc"),
            ("local_uncertainty", "local_unc"),
        ):
            raw = rec.get(key) or rec.get(key + "s")
            if raw is None:
                continue
            arr = np.asarray(raw, dtype=np.float32)
            if arr.ndim != 2:
                continue
            rgb = _uncertainty_to_rgb(arr)
            name = f"bev_{rank:03d}_frame_{int(frame_idx):06d}_stamp_{stamp or 0}_t_{stamp or 0}_{stem}.png"
            path = out_dir / name
            _save_rgb_png(path, rgb)
            saved.append(path)

    if records:
        meta = {
            "n_records": len(records),
            "note": (
                "Recorded /test_all/bev_outputs carries global/local uncertainty only. "
                "occupancy_final.png needs test_all --save or extend finalize payload."
            ),
            "sample_indices": indices,
        }
        meta_path = out_dir / "README_bev_review.json"
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        saved.append(meta_path)
    return saved


def export_boxer_crops(
    records: list[dict[str, Any]],
    out_dir: Path,
    *,
    max_frames: int = 40,
) -> list[Path]:
    """Decode crop image b64 from boxer_output.jsonl with frame stamp and language label."""
    saved: list[Path] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    indices = _pick_sample_indices(len(records), max_frames)
    for rank, idx in enumerate(indices):
        rec = records[idx]
        stamp = int(stamp_ns_of(rec) or rec.get("stamp_ns") or 0)
        objects = rec.get("objects") or []
        if not isinstance(objects, list):
            continue
        for obj in objects:
            if not isinstance(obj, dict):
                continue
            encoding = str(obj.get("crop_encoding") or "").lower()
            b64 = obj.get("crop_png_b64")
            ext = "png"
            if not b64:
                b64 = obj.get("crop_jpeg_b64")
                ext = "jpg"
            if encoding in {"png", "jpeg", "jpg"}:
                ext = "jpg" if encoding in {"jpeg", "jpg"} else encoding
            crop_path = obj.get("crop_png_abspath") or obj.get("crop_png_path")
            if not b64 and not crop_path:
                continue
            label = _slug(obj.get("language_label", "unknown"))
            track_id = int(obj.get("track_id", -1))
            if b64:
                try:
                    raw = base64.b64decode(b64, validate=False)
                except Exception:
                    continue
            else:
                try:
                    raw = Path(str(crop_path)).read_bytes()
                except OSError:
                    continue
            name = f"frame_{stamp:019d}_rank_{rank:03d}_track_{track_id:04d}_lab_{label}.{ext}"
            path = out_dir / name
            path.write_bytes(raw)
            saved.append(path)
    return saved


def export_boxer_viz(
    episode_raw_streams: dict[str, list[dict[str, Any]]],
    out_dir: Path,
    *,
    video_path: Path | None = None,
    max_frames: int = 0,
    fps: float = 30.0,
    state_slop_ms: float = 50.0,
    boxer_slop_ms: float = 200.0,
    pose_stream: str = "bev_output",
    object_policy: str = "recorded",
) -> list[Path]:
    """Replay Boxer tracked-object 3D bbox overlay on recorded left MP4 frames."""
    saved: list[Path] = []
    left_records = sorted(episode_raw_streams.get("left") or [], key=_record_stamp)
    boxer_records = sorted(episode_raw_streams.get("boxer_output") or [], key=_record_stamp)
    camera_pose_records = sorted(episode_raw_streams.get("camera_pose") or [], key=_record_stamp)
    bev_pose_records = sorted(episode_raw_streams.get("bev_output") or [], key=_record_stamp)
    if not left_records or not boxer_records:
        return saved
    if pose_stream not in ("camera_pose", "bev_output"):
        raise ValueError(f"unsupported boxer viz pose_stream={pose_stream!r}")
    if object_policy not in ("recorded", "active", "active_inactive_dedup", "active_memory"):
        raise ValueError(f"unsupported boxer viz object_policy={object_policy!r}")

    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("boxer_viz_*.png"):
        old.unlink()
    for old_name in ("boxer_track_overlay.mp4", "README_boxer_viz.json", "boxer_viz_error.txt"):
        old_path = out_dir / old_name
        if old_path.exists():
            old_path.unlink()
    indices = list(range(len(left_records))) if int(max_frames) <= 0 else _pick_sample_indices(len(left_records), max_frames)
    pose_slop_ns = int(float(state_slop_ms) * 1e6)
    boxer_slop_ns = int(float(boxer_slop_ms) * 1e6)
    video_frames = _read_video_frames(video_path, indices)

    def pose_records_for(name: str) -> list[dict[str, Any]]:
        return bev_pose_records if name == "bev_output" else camera_pose_records

    selected_pose_stream = pose_stream
    pose_records = pose_records_for(selected_pose_stream)
    if not pose_records:
        return saved

    frames: list[np.ndarray] = []
    frame_meta: list[dict[str, Any]] = []
    for rank, idx in enumerate(indices):
        left = left_records[idx]
        stamp = _record_stamp(left)
        boxer = _causal_record_with_fallback(boxer_records, stamp, max_slop_ns=boxer_slop_ns)
        pose = _causal_record(pose_records, stamp, max_slop_ns=pose_slop_ns)
        if boxer is None or pose is None:
            continue
        rgb = video_frames.get(idx)
        if rgb is None:
            continue
        boxer_for_viz, original_object_count = _filtered_boxer_record_for_viz(
            boxer,
            pose,
            rgb.shape[1],
            rgb.shape[0],
            object_policy=object_policy,
        )
        overlay = _draw_boxer_overlay(rgb, boxer_for_viz, pose)
        mp4_frame_index = len(frames)
        frames.append(overlay)
        objects = boxer_for_viz.get("objects") or []
        frame_meta.append(
            {
                "mp4_frame_index": mp4_frame_index,
                "rank": rank,
                "left_index": idx,
                "left_stamp_ns": stamp,
                "boxer_stamp_ns": _record_stamp(boxer),
                "pose_stamp_ns": _record_stamp(pose),
                "camera_pose_stamp_ns": _record_stamp(pose),
                "object_count": original_object_count,
                "filtered_object_count": len(objects) if isinstance(objects, list) else 0,
                "drawable_box_count": _drawable_box_count(boxer_for_viz, pose, rgb.shape[1], rgb.shape[0]),
                "pose_stream": selected_pose_stream,
            }
        )

    if not frames:
        err = out_dir / "boxer_viz_error.txt"
        err.write_text(
            "No replay frames were generated. Need left image data plus causal camera_pose "
            "and boxer_output within the requested slop.\n",
            encoding="utf-8",
        )
        return [err]

    try:
        import cv2
    except ImportError as exc:  # pragma: no cover
        err = out_dir / "boxer_viz_error.txt"
        err.write_text(f"OpenCV is required to write MP4: {exc}\n", encoding="utf-8")
        saved.append(err)
    else:
        h, w = frames[0].shape[:2]
        mp4_path = out_dir / "boxer_track_overlay.mp4"
        writer = cv2.VideoWriter(
            str(mp4_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(fps),
            (w, h),
        )
        if writer.isOpened():
            for frame in frames:
                if frame.shape[:2] != (h, w):
                    frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            writer.release()
            saved.append(mp4_path)
        else:
            err = out_dir / "boxer_viz_error.txt"
            err.write_text("Failed to open OpenCV VideoWriter for boxer_track_overlay.mp4\n", encoding="utf-8")
            saved.append(err)

    meta = {
        "n_left_records": len(left_records),
        "n_boxer_records": len(boxer_records),
        "n_camera_pose_records": len(camera_pose_records),
        "n_bev_output_records": len(bev_pose_records),
        "n_rendered_frames": len(frames),
        "left_video_path": str(video_path) if video_path is not None else None,
        "left_image_source": "mp4",
        "requested_pose_stream": pose_stream,
        "selected_pose_stream": selected_pose_stream,
        "object_policy": object_policy,
        "max_frames": int(max_frames),
        "fps": float(fps),
        "state_slop_ms": float(state_slop_ms),
        "boxer_slop_ms": float(boxer_slop_ms),
        "frame_samples": frame_meta,
        "caveats": [
            "Replays saved boxer_output objects, optionally filtered by object_policy.",
            "Older episodes may not contain track_panel_score, so labels fall back to track id.",
            "Projection uses recorded Tcw/Twc and hardcoded ZED Mini intrinsics (359.222/324.324/187.275 @ 672x376), scaled to image size.",
        ],
    }
    meta_path = out_dir / "README_boxer_viz.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    saved.append(meta_path)
    return saved


def _read_video_frames(video_path: Path | None, indices: list[int]) -> dict[int, np.ndarray]:
    if video_path is None or not video_path.exists() or not indices:
        return {}
    try:
        import cv2
    except ImportError:
        return {}
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {}
    out: dict[int, np.ndarray] = {}
    try:
        for idx in sorted(set(int(i) for i in indices if i >= 0)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                continue
            out[idx] = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()
    return out


def _resolve_bev_grid_helpers() -> tuple[Any, Any, Any]:
    repo_root = Path(__file__).resolve().parents[1]
    scripts = repo_root / "ros2_data_hub" / "scripts"
    scripts_str = str(scripts)
    if scripts_str not in sys.path:
        sys.path.insert(0, scripts_str)
    from robot_frame.bev_grid import (  # type: ignore[import-not-found]
        DEFAULT_BEV_GRID_CONFIG,
        camera_pose_world_from_record,
        object_bev_grid_indices,
    )

    return DEFAULT_BEV_GRID_CONFIG, camera_pose_world_from_record, object_bev_grid_indices


def _object_highlight_color(track_id: int) -> tuple[int, int, int]:
    palette = (
        (255, 0, 255),
        (0, 255, 255),
        (255, 255, 0),
        (0, 165, 255),
        (0, 255, 0),
        (255, 128, 0),
    )
    if track_id < 0:
        return palette[0]
    return palette[int(track_id) % len(palette)]


def _pad_bgr_to_height(bgr: np.ndarray, target_h: int) -> np.ndarray:
    """Pad or crop to target height without interpolation (avoids hstack upscale blur)."""
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("OpenCV is required for BEV object evolution MP4") from exc

    h, w = bgr.shape[:2]
    th = int(target_h)
    if h == th:
        return bgr
    if h > th:
        return bgr[:th, :, :]
    pad = th - h
    return cv2.copyMakeBorder(bgr, 0, pad, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))


_LABEL_FONT_CANDIDATES = (
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    Path("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"),
    Path("/usr/share/fonts/truetype/freefont/FreeSansBold.ttf"),
)


def _label_font_px(pixel_scale: int) -> int:
    ps = max(1, int(pixel_scale))
    # ~45% of one grid cell; scales with local panel cell size.
    return max(12, min(40, int(round(ps * 0.45))))


def _load_label_font(font_px: int):
    from PIL import ImageFont

    for path in _LABEL_FONT_CANDIDATES:
        if path.is_file():
            return ImageFont.truetype(str(path), size=int(font_px))
    return ImageFont.load_default()


def _draw_labels_pil_rgb(
    rgb: np.ndarray,
    labels: list[tuple[str, int, int, tuple[int, int, int]]],
    *,
    font_px: int,
) -> np.ndarray:
    """Draw anti-aliased labels on RGB image; grid underneath is untouched."""
    if not labels:
        return rgb
    from PIL import Image, ImageDraw

    pil = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil)
    font = _load_label_font(font_px)
    stroke = max(1, int(font_px) // 10)
    for text, x, y, color_bgr in labels:
        fill = (int(color_bgr[2]), int(color_bgr[1]), int(color_bgr[0]))
        draw.text(
            (int(x), int(y)),
            text,
            font=font,
            fill=fill,
            stroke_width=stroke,
            stroke_fill=(0, 0, 0),
        )
    return np.asarray(pil)


def _draw_grid_highlights_bgr(
    bgr: np.ndarray,
    *,
    grid_h: int,
    grid_w: int,
    pixel_scale: int,
    highlights: list[tuple[int, int, tuple[int, int, int], str | None]],
    thickness: int = 2,
    use_pil_labels: bool = True,
) -> np.ndarray:
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("OpenCV is required for BEV object evolution MP4") from exc

    out = bgr.copy()
    ps = int(pixel_scale)
    font_px = _label_font_px(ps)
    labels_to_draw: list[tuple[str, int, int, tuple[int, int, int]]] = []
    occupied: dict[tuple[int, int], int] = {}
    for ix, iy, color, label in highlights:
        if ix < 0 or iy < 0 or ix >= grid_w or iy >= grid_h:
            continue
        r0 = (grid_h - 1 - iy) * ps
        r1 = (grid_h - iy) * ps
        c0 = ix * ps
        c1 = (ix + 1) * ps
        cv2.rectangle(
            out,
            (c0, r0),
            (max(c0, c1 - 1), max(r0, r1 - 1)),
            color,
            thickness=max(1, min(int(thickness), 2)),
        )
        cx = int(c0 + ps // 2)
        cy = int(r0 + ps // 2)
        cv2.circle(out, (cx, cy), max(1, ps // 4), color, thickness=-1, lineType=cv2.LINE_AA)
        if label:
            slot = occupied.get((ix, iy), 0)
            occupied[(ix, iy)] = slot + 1
            x = int(np.clip(c0 + 2 + slot * (font_px // 2), 0, max(0, out.shape[1] - 1)))
            y = int(np.clip(r0 + max(2, ps // 8), 2, max(2, out.shape[0] - font_px - 2)))
            labels_to_draw.append((label[:10], x, y, color))

    if not labels_to_draw:
        return out

    if use_pil_labels:
        try:
            rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
            rgb = _draw_labels_pil_rgb(rgb, labels_to_draw, font_px=font_px)
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        except Exception:
            pass

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.22, min(0.45, 0.045 * float(ps)))
    line_th = 1
    for text, x, y, color in labels_to_draw:
        baseline_y = int(y + font_px * 0.85)
        cv2.putText(out, text, (x, baseline_y), font, font_scale, (0, 0, 0), line_th + 1, cv2.LINE_AA)
        cv2.putText(out, text, (x, baseline_y), font, font_scale, color, line_th, cv2.LINE_AA)
    return out


def _object_grid_highlights(
    obj: dict[str, Any],
    *,
    bev_record: dict[str, Any] | None,
    grid_config: dict[str, Any] | None,
    object_bev_grid_indices: Any,
    camera_pose_world_from_record: Any,
    default_grid_config: dict[str, Any],
) -> tuple[tuple[int, int] | None, tuple[int, int] | None]:
    gix = obj.get("global_ix")
    giy = obj.get("global_iy")
    lix = obj.get("local_ix")
    liy = obj.get("local_iy")
    if gix is not None and giy is not None and lix is not None and liy is not None:
        global_ij = (int(gix), int(giy))
        local_ij = (int(lix), int(liy))
        if global_ij[0] < 0 or global_ij[1] < 0:
            global_ij = None
        if local_ij[0] < 0 or local_ij[1] < 0:
            local_ij = None
        return global_ij, local_ij

    if bev_record is None:
        return None, None
    c2w = camera_pose_world_from_record(bev_record)
    if c2w is None:
        return None, None
    cfg = grid_config if isinstance(grid_config, dict) else default_grid_config
    indices = object_bev_grid_indices(obj, c2w=c2w, grid_config=cfg)
    global_ij = (int(indices["global_ix"]), int(indices["global_iy"]))
    local_ij = (int(indices["local_ix"]), int(indices["local_iy"]))
    if global_ij[0] < 0 or global_ij[1] < 0:
        global_ij = None
    if local_ij[0] < 0 or local_ij[1] < 0:
        local_ij = None
    return global_ij, local_ij


def _boxer_highlights_for_bev_frame(
    boxer_record: dict[str, Any] | None,
    bev_record: dict[str, Any],
    *,
    grid_config: dict[str, Any] | None,
    object_bev_grid_indices: Any,
    camera_pose_world_from_record: Any,
    default_grid_config: dict[str, Any],
) -> tuple[list[tuple[int, int, tuple[int, int, int], str | None]], list[tuple[int, int, tuple[int, int, int], str | None]]]:
    global_highlights: list[tuple[int, int, tuple[int, int, int], str | None]] = []
    local_highlights: list[tuple[int, int, tuple[int, int, int], str | None]] = []
    if boxer_record is None:
        return global_highlights, local_highlights
    objects = boxer_record.get("objects") or []
    if not isinstance(objects, list):
        return global_highlights, local_highlights
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        track_id = int(obj.get("track_id", -1))
        label = str(obj.get("language_label") or f"#{track_id}")
        color = _object_highlight_color(track_id)
        global_ij, local_ij = _object_grid_highlights(
            obj,
            bev_record=bev_record,
            grid_config=grid_config,
            object_bev_grid_indices=object_bev_grid_indices,
            camera_pose_world_from_record=camera_pose_world_from_record,
            default_grid_config=default_grid_config,
        )
        if global_ij is not None:
            global_highlights.append((global_ij[0], global_ij[1], color, label))
        if local_ij is not None:
            local_highlights.append((local_ij[0], local_ij[1], color, label))
    return global_highlights, local_highlights


def _bev_uncertainty_fields(rec: dict[str, Any]) -> tuple[Any, Any] | None:
    g = rec.get("global_uncertainty")
    if g is None:
        g = rec.get("global_uncertainties")
    l = rec.get("local_uncertainty")
    if l is None:
        l = rec.get("local_uncertainties")
    if g is None or l is None:
        return None
    return g, l


def _bev_pose_raw(rec: dict[str, Any]) -> Any | None:
    pose_raw = rec.get("camera_pose_world")
    if pose_raw is None:
        pose_raw = rec.get("camera_poses_world")
    return pose_raw


def _sort_bev_output_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def sort_key(rec: dict[str, Any]) -> tuple[int, int]:
        if rec.get("frame_index") is not None:
            return (0, int(rec["frame_index"]))
        return (1, int(stamp_ns_of(rec) or rec.get("stamp_ns") or 0))

    return sorted(records, key=sort_key)


def _resolve_vda_repo() -> Path | None:
    env = os.environ.get("VIDEO_DEPTH_ANYTHING_ROOT", "").strip()
    if env:
        p = Path(env).expanduser()
        if (p / "data_collection_utils" / "bev.py").exists():
            return p
    candidates = [
        Path(__file__).resolve().parents[2].parent / "Video-Depth-Anything",
        Path.home() / "halab_workspace" / "Video-Depth-Anything",
    ]
    for p in candidates:
        if (p / "data_collection_utils" / "bev.py").exists():
            return p
    return None


def _load_bev_finalize_phase(bev_phase_records: list[dict[str, Any]], t_frames: int) -> dict[str, Any] | None:
    if not bev_phase_records:
        return None
    phase = bev_phase_records[-1].get("phase")
    if not isinstance(phase, dict):
        return None
    out: dict[str, Any] = {}
    for key, value in phase.items():
        if isinstance(value, list):
            arr = np.asarray(value)
            if arr.ndim >= 1 and arr.shape[0] > t_frames:
                arr = arr[:t_frames]
            elif arr.ndim >= 1 and arr.shape[0] < t_frames:
                pad_shape = (t_frames - int(arr.shape[0]), *arr.shape[1:])
                if arr.dtype.kind == "b":
                    pad = np.zeros(pad_shape, dtype=arr.dtype)
                elif key.startswith("target"):
                    pad = np.full(pad_shape, -1, dtype=arr.dtype)
                else:
                    pad = np.full(pad_shape, np.nan, dtype=np.float32)
                arr = np.concatenate([arr, pad.astype(arr.dtype, copy=False)], axis=0)
            out[key] = arr
        else:
            out[key] = value
    if "target_global_ij" not in out and "target_ix" in out and "target_iy" in out:
        tx = np.asarray(out["target_ix"], dtype=np.int32).reshape(-1)
        ty = np.asarray(out["target_iy"], dtype=np.int32).reshape(-1)
        n = min(tx.shape[0], ty.shape[0], t_frames)
        out["target_global_ij"] = np.stack([tx[:n], ty[:n]], axis=1).astype(np.int32)
    if "target_local_ij" not in out and "target_local_ix" in out and "target_local_iy" in out:
        tx = np.asarray(out["target_local_ix"], dtype=np.int32).reshape(-1)
        ty = np.asarray(out["target_local_iy"], dtype=np.int32).reshape(-1)
        n = min(tx.shape[0], ty.shape[0], t_frames)
        out["target_local_ij"] = np.stack([tx[:n], ty[:n]], axis=1).astype(np.int32)
    return out


def _phase_from_bev_output_records(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not records:
        return None
    if not any("is_bos" in rec or "target_ix" in rec for rec in records):
        return None

    is_bos = np.asarray([bool(rec.get("is_bos", False)) for rec in records], dtype=bool)
    target_ix = np.asarray([int(rec.get("target_ix", -1)) for rec in records], dtype=np.int32)
    target_iy = np.asarray([int(rec.get("target_iy", -1)) for rec in records], dtype=np.int32)
    target_local_ix = np.asarray([int(rec.get("target_local_ix", -1)) for rec in records], dtype=np.int32)
    target_local_iy = np.asarray([int(rec.get("target_local_iy", -1)) for rec in records], dtype=np.int32)
    target_local_clipped = np.asarray(
        [bool(rec.get("target_local_clipped", False)) for rec in records],
        dtype=bool,
    )
    return {
        "is_bos": is_bos,
        "target_ix": target_ix,
        "target_iy": target_iy,
        "target_global_ij": np.stack([target_ix, target_iy], axis=1).astype(np.int32),
        "target_local_ix": target_local_ix,
        "target_local_iy": target_local_iy,
        "target_local_ij": np.stack([target_local_ix, target_local_iy], axis=1).astype(np.int32),
        "target_local_clipped": target_local_clipped,
    }


def _optional_phase_scalar(value: Any) -> float | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return None
    out = float(arr[0])
    return out if np.isfinite(out) else None


def _save_phase_timeline_plot_with_optional_kinematic(
    out_path: Path,
    phase: dict[str, Any],
) -> None:
    vda_root = _resolve_vda_repo()
    if vda_root is None:
        raise RuntimeError(
            "Video-Depth-Anything not found. Set VIDEO_DEPTH_ANYTHING_ROOT or install "
            "sibling repo at ../Video-Depth-Anything"
        )
    os.environ.setdefault("NUMBA_CACHE_DIR", str((out_path.parent / ".numba_cache").resolve()))
    vda_str = str(vda_root)
    if vda_str not in sys.path:
        sys.path.insert(0, vda_str)

    required = ("is_bos", "R_t", "G_t", "horizon")
    missing = [key for key in required if key not in phase]
    if missing:
        raise ValueError(
            "bev_phase.jsonl is missing required phase field(s) for phase_bos_boa_timeline.png: "
            + ", ".join(missing)
        )

    is_bos = np.asarray(phase["is_bos"], dtype=bool)
    R_t = np.asarray(phase["R_t"], dtype=np.float64)
    G_t = np.asarray(phase["G_t"], dtype=np.float64)
    horizon = int(np.asarray(phase["horizon"]).reshape(-1)[0])
    if is_bos.ndim != 1 or R_t.ndim != 1 or G_t.ndim != 1:
        raise ValueError("phase fields is_bos/R_t/G_t must be 1D arrays")
    n = min(len(is_bos), len(R_t), len(G_t))
    if n == 0:
        raise ValueError("phase fields are empty")
    is_bos, R_t, G_t = is_bos[:n], R_t[:n], G_t[:n]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    K_raw = phase.get("K_t")
    if K_raw is None:
        from data_collection_utils.phase_bos_boa import save_phase_timeline_plot  # type: ignore[import-not-found]

        save_phase_timeline_plot(
            str(out_path),
            is_bos,
            R_t,
            G_t,
            horizon,
            tau_r_line=_optional_phase_scalar(phase.get("tau_r_effective")),
            tau_g_line=_optional_phase_scalar(phase.get("tau_g_effective")),
        )
        return

    from data_collection_utils import phase_bos_boa as phase_bos_boa_module  # type: ignore[import-not-found]

    plt = phase_bos_boa_module.plt
    K_t = np.asarray(K_raw, dtype=np.float64).reshape(-1)[:n]
    tt = np.arange(n)
    tr = _optional_phase_scalar(phase.get("tau_r_effective"))
    tg = _optional_phase_scalar(phase.get("tau_g_effective"))
    tk = _optional_phase_scalar(phase.get("tau_k_effective"))

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(12, 7.5),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 1, 1, 1.2]},
    )
    axes[0].plot(tt, np.where(np.isnan(R_t), np.nan, R_t), color="tab:blue", linewidth=1.0, label=r"$R_t$")
    if tr is not None:
        axes[0].axhline(tr, color="tab:blue", linestyle="--", linewidth=1.0, alpha=0.7, label=r"$\tau_R$")
    axes[0].set_ylabel(r"$R_t$")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(tt, np.where(np.isnan(G_t), np.nan, G_t), color="tab:orange", linewidth=1.0, label=r"$G_t$")
    if tg is not None:
        axes[1].axhline(tg, color="tab:orange", linestyle="--", linewidth=1.0, alpha=0.7, label=r"$\tau_G$")
    axes[1].set_ylabel(r"$G_t$")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(tt, np.where(np.isnan(K_t), np.nan, K_t), color="tab:purple", linewidth=1.0, label=r"$K_t$")
    if tk is not None:
        axes[2].axhline(tk, color="tab:purple", linestyle="--", linewidth=1.0, alpha=0.7, label=r"$\tau_K$")
    axes[2].set_ylabel("Kinematic")
    axes[2].legend(loc="upper right")
    axes[2].grid(True, alpha=0.3)

    phase_mask = np.where(is_bos, 1.0, 0.0)
    if horizon > 0:
        phase_mask[max(0, n - horizon) :] = 0.0
    axes[3].fill_between(tt, 0, phase_mask, step="mid", alpha=0.35, color="green", label="BOS")
    axes[3].fill_between(tt, 0, 1 - phase_mask, step="mid", alpha=0.2, color="gray", label="BOA / n/a")
    axes[3].set_ylim(-0.05, 1.05)
    axes[3].set_ylabel("Phase")
    axes[3].set_xlabel("timestep t")
    axes[3].legend(loc="upper right")
    axes[3].set_yticks([0, 1])
    axes[3].set_yticklabels(["BOA", "BOS"])
    fig.suptitle(f"BOS/BOA + camera kinematic gate (lookahead = {horizon} frames)")
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)


def export_bev_phase_timeline(
    bev_phase_records: list[dict[str, Any]],
    out_dir: Path,
    *,
    t_frames: int | None = None,
) -> Path:
    phase = _load_bev_finalize_phase(bev_phase_records, t_frames or 10**12)
    if phase is None:
        raise ValueError("raw_streams/bev_phase.jsonl has no finalize phase payload")
    out_path = out_dir / "phase_bos_boa_timeline.png"
    _save_phase_timeline_plot_with_optional_kinematic(out_path, phase)
    return out_path


def export_bev_evolution_mp4(
    bev_output_records: list[dict[str, Any]],
    out_dir: Path,
    *,
    bev_phase_records: list[dict[str, Any]] | None = None,
    fps: float = 30.0,
) -> Path:
    """
    Rebuild occupancy_uncertainty_global_local_evolution.mp4 from episode raw_streams.

    Uses the same renderer as test_all ``save_realtime_bev_outputs`` (global|local
    uncertainty side-by-side). Requires Video-Depth-Anything on PYTHONPATH.
    """
    vda_root = _resolve_vda_repo()
    if vda_root is None:
        raise RuntimeError(
            "Video-Depth-Anything not found. Set VIDEO_DEPTH_ANYTHING_ROOT or install "
            "sibling repo at ../Video-Depth-Anything"
        )
    os.environ.setdefault("NUMBA_CACHE_DIR", str((out_dir / ".numba_cache").resolve()))
    vda_str = str(vda_root)
    if vda_str not in sys.path:
        sys.path.insert(0, vda_str)

    from data_collection_utils.bev import (  # type: ignore[import-not-found]
        RealtimeBEVFinalOutput,
        _save_global_local_uncertainty_mp4,
    )
    from data_collection_utils.occupancy_map import (  # type: ignore[import-not-found]
        OccupancyGridConfig,
        symmetric_ego_grid_config,
    )

    records = _sort_bev_output_records(bev_output_records)
    global_list: list[np.ndarray] = []
    local_list: list[np.ndarray] = []
    poses: list[np.ndarray] = []
    stamps: list[float] = []

    for rec in records:
        unc = _bev_uncertainty_fields(rec)
        if unc is None:
            continue
        g, l = unc
        g_arr = np.asarray(g, dtype=np.float32)
        l_arr = np.asarray(l, dtype=np.float32)
        if g_arr.ndim != 2 or l_arr.ndim != 2:
            continue
        global_list.append(g_arr)
        local_list.append(l_arr)
        pose_raw = _bev_pose_raw(rec)
        if pose_raw is not None:
            poses.append(np.asarray(pose_raw, dtype=np.float64).reshape(4, 4))
        else:
            poses.append(np.eye(4, dtype=np.float64))
        stamp = stamp_ns_of(rec)
        stamps.append(float(stamp if stamp is not None else len(stamps)))

    if not global_list:
        raise ValueError("bev_output.jsonl has no frames with global/local uncertainty")

    global_unc = np.stack(global_list, axis=0)
    local_unc = np.stack(local_list, axis=0)
    t_frames = int(global_unc.shape[0])
    if poses:
        c2w_world = np.stack(poses[:t_frames], axis=0).astype(np.float32)
    else:
        c2w_world = np.repeat(np.eye(4, dtype=np.float32)[None, ...], t_frames, axis=0)

    x_min, x_max, y_min, y_max = _DEFAULT_OCC_BOUNDS
    gh, gw = int(global_unc.shape[1]), int(global_unc.shape[2])
    occ_res = (x_max - x_min) / max(gw, 1)
    occ_grid_cfg = OccupancyGridConfig(
        x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, res=float(occ_res)
    )
    local_grid_cfg = symmetric_ego_grid_config(_DEFAULT_LOCAL_HALF_M, _DEFAULT_LOCAL_RES_M)

    proc_w, proc_h = _DEFAULT_PROC_SIZE_WH
    intrinsics = np.repeat(_intrinsic_for_image(proc_w, proc_h)[None, ...], t_frames, axis=0).astype(np.float32)
    # Prefer per-BEV-record phase fields because they are already aligned to the
    # exact uncertainty frames used for this MP4. The finalize payload can include
    # frames just outside the episode raw-stream window.
    phase = _phase_from_bev_output_records(records)
    if phase is None:
        phase = _load_bev_finalize_phase(bev_phase_records or [], t_frames)

    output = RealtimeBEVFinalOutput(
        frame_count=t_frames,
        floor_ready=bool(bev_phase_records[-1].get("floor_ready", True)) if bev_phase_records else True,
        floor_payload={},
        camera_poses_source=c2w_world.copy(),
        camera_poses_world=c2w_world,
        intrinsics=intrinsics,
        proc_size=(proc_w, proc_h),
        points_world=[],
        final_occupancy_state=np.zeros((gh, gw), dtype=np.uint8),
        final_global_uncertainty=global_unc[-1].copy(),
        final_local_uncertainty=local_unc[-1].copy(),
        occ_states=np.zeros((t_frames, gh, gw), dtype=np.uint8),
        global_uncertainties=global_unc,
        local_uncertainties=local_unc,
        occupancy_grid_config=occ_grid_cfg,
        local_grid_config=local_grid_cfg,
        phase=phase,
        global_local_mp4_inputs={
            "occ.global_local_mp4_render_config": np.array(
                [float(fps), float(_DEFAULT_OCC_PIXEL_SCALE), 0.0], dtype=np.float32
            ),
        },
        timestamps=np.asarray(stamps, dtype=np.float64),
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    mp4_path = out_dir / "occupancy_uncertainty_global_local_evolution.mp4"
    _save_global_local_uncertainty_mp4(output, mp4_path)

    meta = {
        "n_frames": t_frames,
        "mp4": str(mp4_path),
        "source": "raw_streams/bev_output.jsonl + bev_phase.jsonl (episode finalize phase)",
        "search_target_overlay": phase is not None
        and "target_global_ij" in phase
        and "is_bos" in phase,
        "caveats": [
            "Occupancy grid (occ_states) is not in ROS bev_output; MP4 shows uncertainty only.",
            "Intrinsics/proc_size use test_all defaults unless extended in recorded JSON.",
            "Grid bounds inferred from map size and occ_x/y ±6 m defaults.",
        ],
    }
    meta_path = out_dir / "README_bev_evolution_mp4.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return mp4_path


def export_boxer_objects_bev_evolution_mp4(
    bev_output_records: list[dict[str, Any]],
    boxer_output_records: list[dict[str, Any]],
    out_dir: Path,
    *,
    bev_phase_records: list[dict[str, Any]] | None = None,
    fps: float = 30.0,
    boxer_slop_ms: float = 200.0,
    object_policy: str = "recorded",
) -> Path:
    """
    Same side-by-side global|local uncertainty evolution MP4 as review_bev, but highlights
    Boxer object centers on the BEV grids instead of BOS search targets.

    Boxer join matches FrameAssembler recording: causal within boxer_slop, else hold-forward.
    """
    vda_root = _resolve_vda_repo()
    if vda_root is None:
        raise RuntimeError(
            "Video-Depth-Anything not found. Set VIDEO_DEPTH_ANYTHING_ROOT or install "
            "sibling repo at ../Video-Depth-Anything"
        )
    os.environ.setdefault("NUMBA_CACHE_DIR", str((out_dir / ".numba_cache").resolve()))
    vda_str = str(vda_root)
    if vda_str not in sys.path:
        sys.path.insert(0, vda_str)

    from data_collection_utils.occupancy_map import (  # type: ignore[import-not-found]
        OccupancyGridConfig,
        annotate_evolution_frame_index_bgr,
        ego_forward_right_unit_xy_from_c2w,
        fov_left_right_ray_dirs_unit_xy,
        render_uncertainty_field_bgr,
        symmetric_ego_grid_config,
    )

    default_grid_config, camera_pose_world_from_record, object_bev_grid_indices = _resolve_bev_grid_helpers()
    grid_config: dict[str, Any] | None = None
    if bev_phase_records:
        maybe_cfg = bev_phase_records[-1].get("bev_grid_config")
        if isinstance(maybe_cfg, dict):
            grid_config = maybe_cfg

    records = _sort_bev_output_records(bev_output_records)
    boxer_sorted = sorted(boxer_output_records, key=_record_stamp)

    global_list: list[np.ndarray] = []
    local_list: list[np.ndarray] = []
    poses: list[np.ndarray] = []

    for rec in records:
        unc = _bev_uncertainty_fields(rec)
        if unc is None:
            continue
        g, l = unc
        g_arr = np.asarray(g, dtype=np.float32)
        l_arr = np.asarray(l, dtype=np.float32)
        if g_arr.ndim != 2 or l_arr.ndim != 2:
            continue
        global_list.append(g_arr)
        local_list.append(l_arr)
        pose_raw = _bev_pose_raw(rec)
        if pose_raw is not None:
            poses.append(np.asarray(pose_raw, dtype=np.float64).reshape(4, 4))
        else:
            poses.append(np.eye(4, dtype=np.float64))

    if not global_list:
        raise ValueError("bev_output.jsonl has no frames with global/local uncertainty")

    global_unc = np.stack(global_list, axis=0)
    local_unc = np.stack(local_list, axis=0)
    t_frames = int(global_unc.shape[0])
    gh, gw = int(global_unc.shape[1]), int(global_unc.shape[2])
    lh, lw = int(local_unc.shape[1]), int(local_unc.shape[2])

    x_min, x_max, y_min, y_max = _DEFAULT_OCC_BOUNDS
    occ_res = (x_max - x_min) / max(gw, 1)
    occ_grid_cfg = OccupancyGridConfig(
        x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, res=float(occ_res)
    )
    if isinstance(grid_config, dict) and isinstance(grid_config.get("local"), dict):
        local_cfg_raw = grid_config["local"]
        local_grid_cfg = OccupancyGridConfig(
            x_min=float(local_cfg_raw.get("x_min", -_DEFAULT_LOCAL_HALF_M)),
            x_max=float(local_cfg_raw.get("x_max", _DEFAULT_LOCAL_HALF_M)),
            y_min=float(local_cfg_raw.get("y_min", -_DEFAULT_LOCAL_HALF_M)),
            y_max=float(local_cfg_raw.get("y_max", _DEFAULT_LOCAL_HALF_M)),
            res=float(local_cfg_raw.get("res", _DEFAULT_LOCAL_RES_M)),
        )
    else:
        local_grid_cfg = symmetric_ego_grid_config(_DEFAULT_LOCAL_HALF_M, _DEFAULT_LOCAL_RES_M)

    pixel_scale = max(1, min(int(_BOXER_BEV_PIXEL_SCALE), 8))
    # Local grid has far fewer cells; ensure local panel is at least as tall as global, then
    # pad (never upscale) before side-by-side concat so labels stay sharp.
    global_panel_h = gh * pixel_scale
    local_pixel_scale = max(pixel_scale, (global_panel_h + lh - 1) // max(lh, 1))
    panel_h = lh * local_pixel_scale
    boxer_slop_ns = int(float(boxer_slop_ms) * 1e6)
    if object_policy not in ("recorded", "active", "active_inactive_dedup", "active_memory"):
        raise ValueError(f"unsupported boxer BEV object_policy={object_policy!r}")
    major_grid_m = 0.0
    proc_w, proc_h = _DEFAULT_PROC_SIZE_WH

    try:
        import cv2
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("OpenCV is required for boxer BEV evolution MP4") from exc

    out_dir.mkdir(parents=True, exist_ok=True)
    mp4_path = out_dir / _BOXER_BEV_EVOLUTION_MP4_NAME
    writer: cv2.VideoWriter | None = None
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    rendered_frames = 0
    try:
        for t in range(t_frames):
            bev_rec = records[t]
            stamp = _record_stamp(bev_rec)
            boxer = _causal_record_with_fallback(boxer_sorted, stamp, max_slop_ns=boxer_slop_ns)
            if boxer is not None:
                boxer, _ = _filtered_boxer_record_for_viz(
                    boxer,
                    bev_rec,
                    proc_w,
                    proc_h,
                    object_policy=object_policy,
                )
            global_hl, local_hl = _boxer_highlights_for_bev_frame(
                boxer,
                bev_rec,
                grid_config=grid_config,
                object_bev_grid_indices=object_bev_grid_indices,
                camera_pose_world_from_record=camera_pose_world_from_record,
                default_grid_config=default_grid_config,
            )

            c2w_t = poses[min(t, len(poses) - 1)]
            K_t = _intrinsic_for_image(proc_w, proc_h)
            fwd, _ = ego_forward_right_unit_xy_from_c2w(c2w_t)
            fov = fov_left_right_ray_dirs_unit_xy(
                c2w_t,
                K_t,
                proc_w,
                proc_h,
                ground_z=0.0,
            )
            global_bgr = render_uncertainty_field_bgr(
                global_unc[t],
                occ_grid_cfg,
                pixel_scale=pixel_scale,
                major_grid_m=major_grid_m,
                highlight_ij=None,
                cam_xy=(float(c2w_t[0, 3]), float(c2w_t[1, 3])),
                fov_ray_dirs_xy=fov,
                camera_forward_xy=(float(fwd[0]), float(fwd[1])),
            )
            local_bgr = render_uncertainty_field_bgr(
                local_unc[t],
                local_grid_cfg,
                pixel_scale=local_pixel_scale,
                major_grid_m=major_grid_m,
                highlight_ij=None,
                cam_xy=(0.0, 0.0),
                camera_forward_xy=(0.0, 1.0),
            )
            global_bgr = _draw_grid_highlights_bgr(
                global_bgr,
                grid_h=gh,
                grid_w=gw,
                pixel_scale=pixel_scale,
                highlights=global_hl,
                use_pil_labels=True,
            )
            local_bgr = _draw_grid_highlights_bgr(
                local_bgr,
                grid_h=lh,
                grid_w=lw,
                pixel_scale=local_pixel_scale,
                highlights=local_hl,
                use_pil_labels=True,
            )
            global_bgr = _pad_bgr_to_height(global_bgr, panel_h)
            local_bgr = _pad_bgr_to_height(local_bgr, panel_h)
            frame = annotate_evolution_frame_index_bgr(
                np.concatenate([global_bgr, local_bgr], axis=1),
                t,
            )
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(str(mp4_path), fourcc, float(fps), (w, h))
                if not writer.isOpened():
                    raise RuntimeError(f"Failed to open MP4 writer: {mp4_path}")
            writer.write(frame)
            rendered_frames += 1
    finally:
        if writer is not None:
            writer.release()

    if rendered_frames <= 0:
        raise RuntimeError("No frames rendered for boxer BEV evolution MP4")

    meta = {
        "n_frames": rendered_frames,
        "mp4": str(mp4_path),
        "mp4_name": _BOXER_BEV_EVOLUTION_MP4_NAME,
        "source": "raw_streams/bev_output.jsonl + boxer_output.jsonl",
        "overlay": "boxer_object_global_local_ij",
        "object_policy": object_policy,
        "fps": float(fps),
        "boxer_join": "causal_with_fallback (FrameAssembler policy)",
        "boxer_slop_ms": float(boxer_slop_ms),
        "local_pixel_scale": int(local_pixel_scale),
        "global_pixel_scale": int(pixel_scale),
        "label_renderer": "pil_truetype",
        "caveats": [
            "Same side-by-side layout as review_bev/occupancy_uncertainty_global_local_evolution.mp4.",
            "Highlights use boxer object global_ix/local_ix when present, else recomputed from bbox_center_world.",
            "Boxer join: latest stamp <= BEV stamp within boxer_slop, else hold-forward (same as episode recording).",
        ],
    }
    meta_path = out_dir / "README_boxer_bev_evolution_mp4.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return mp4_path


def export_episode_review_assets(
    episode_raw_streams: dict[str, list[dict[str, Any]]],
    output_dir: Path,
    *,
    video_path: Path | None = None,
    bev_max_samples: int = 12,
    boxer_max_frames: int = 40,
    boxer_viz_max_frames: int = -1,
    boxer_viz_fps: float = 30.0,
    boxer_viz_pose_stream: str = "bev_output",
    boxer_viz_object_policy: str = "recorded",
    bev_evolution_mp4: bool = False,
    bev_mp4_fps: float = 30.0,
) -> list[Path]:
    saved: list[Path] = []
    bev = episode_raw_streams.get("bev_output") or []
    if bev and bev_max_samples > 0:
        saved.extend(
            export_bev_review(bev, output_dir / "review_bev", max_samples=bev_max_samples)
        )
    if bev_evolution_mp4 and bev:
        t_frames: int | None = len(_sort_bev_output_records(bev))
        try:
            mp4 = export_bev_evolution_mp4(
                bev,
                output_dir / "review_bev",
                bev_phase_records=episode_raw_streams.get("bev_phase"),
                fps=bev_mp4_fps,
            )
            saved.append(mp4)
            err_path = output_dir / "review_bev" / "bev_evolution_mp4_error.txt"
            if err_path.exists():
                err_path.unlink()
        except Exception as exc:
            err_path = output_dir / "review_bev" / "bev_evolution_mp4_error.txt"
            err_path.parent.mkdir(parents=True, exist_ok=True)
            err_path.write_text(str(exc), encoding="utf-8")
            saved.append(err_path)
        try:
            phase_png = export_bev_phase_timeline(
                episode_raw_streams.get("bev_phase") or [],
                output_dir / "review_bev",
                t_frames=t_frames,
            )
            saved.append(phase_png)
            err_path = output_dir / "review_bev" / "phase_timeline_error.txt"
            if err_path.exists():
                err_path.unlink()
        except Exception as exc:
            err_path = output_dir / "review_bev" / "phase_timeline_error.txt"
            err_path.parent.mkdir(parents=True, exist_ok=True)
            err_path.write_text(str(exc), encoding="utf-8")
            saved.append(err_path)
    boxer = episode_raw_streams.get("boxer_output") or []
    if boxer and boxer_max_frames > 0:
        saved.extend(
            export_boxer_crops(boxer, output_dir / "review_boxer_crops", max_frames=boxer_max_frames)
        )
    if boxer and boxer_viz_max_frames >= 0:
        saved.extend(
            export_boxer_viz(
                episode_raw_streams,
                output_dir / "review_boxer_viz",
                video_path=video_path,
                max_frames=boxer_viz_max_frames,
                fps=boxer_viz_fps,
                pose_stream=boxer_viz_pose_stream,
                object_policy=boxer_viz_object_policy,
            )
        )
        if bev:
            try:
                boxer_bev_mp4 = export_boxer_objects_bev_evolution_mp4(
                    bev,
                    boxer,
                    output_dir / "review_boxer_viz",
                    bev_phase_records=episode_raw_streams.get("bev_phase"),
                    fps=bev_mp4_fps,
                    object_policy=boxer_viz_object_policy,
                )
                saved.append(boxer_bev_mp4)
            except Exception as exc:
                err_path = output_dir / "review_boxer_viz" / "boxer_bev_evolution_mp4_error.txt"
                err_path.parent.mkdir(parents=True, exist_ok=True)
                err_path.write_text(str(exc), encoding="utf-8")
                saved.append(err_path)
    return saved
