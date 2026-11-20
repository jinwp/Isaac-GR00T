"""Compact Boxer object payload helpers for VLA aligned_extras parquet."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

try:
    import pyarrow as pa
except ImportError:  # pragma: no cover
    pa = None  # type: ignore[assignment]


BOXER_OBJECTS_COLUMN = "boxer.objects_active_inactive_dedup"


def boxer_objects_arrow_type() -> Any:
    if pa is None:
        raise ImportError("pyarrow is required for Boxer object parquet schema")
    return pa.list_(
        pa.struct(
            [
                pa.field("track_id", pa.int32()),
                pa.field("track_state", pa.string()),
                pa.field("language_label", pa.string()),
                pa.field("T_world_object", pa.list_(pa.float32())),
                pa.field("bbox_size_xyz", pa.list_(pa.float32())),
                pa.field("bbox_center_world", pa.list_(pa.float32())),
                pa.field("bbox_orientation_xyzw", pa.list_(pa.float32())),
                pa.field("global_ix", pa.int32()),
                pa.field("global_iy", pa.int32()),
                pa.field("local_ix", pa.int32()),
                pa.field("local_iy", pa.int32()),
                pa.field("local_clipped", pa.bool_()),
                pa.field("track_panel_score", pa.float32()),
                pa.field("crop_panel_score", pa.float32()),
                pa.field("crop_png_path", pa.string()),
                pa.field("crop_png_nbytes", pa.int32()),
            ]
        )
    )


def _float_list(value: Any, *, size: int) -> list[float]:
    if value is None:
        return [float("nan")] * size
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    out = np.full(size, np.nan, dtype=np.float32)
    n = min(size, int(arr.shape[0]))
    if n:
        out[:n] = arr[:n]
    return out.tolist()


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def serialize_boxer_objects(objects: Any) -> list[dict[str, Any]]:
    if not isinstance(objects, list):
        return []
    out: list[dict[str, Any]] = []
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        out.append(
            {
                "track_id": int(obj.get("track_id", -1)),
                "track_state": str(obj.get("track_state") or ""),
                "language_label": str(obj.get("language_label") or ""),
                "T_world_object": _float_list(obj.get("T_world_object"), size=16),
                "bbox_size_xyz": _float_list(obj.get("bbox_size_xyz"), size=3),
                "bbox_center_world": _float_list(obj.get("bbox_center_world"), size=3),
                "bbox_orientation_xyzw": _float_list(obj.get("bbox_orientation_xyzw"), size=4),
                "global_ix": int(obj.get("global_ix", -1)),
                "global_iy": int(obj.get("global_iy", -1)),
                "local_ix": int(obj.get("local_ix", -1)),
                "local_iy": int(obj.get("local_iy", -1)),
                "local_clipped": bool(obj.get("local_clipped", False)),
                "track_panel_score": _optional_float(obj.get("track_panel_score")),
                "crop_panel_score": _optional_float(obj.get("crop_panel_score")),
                "crop_png_path": str(obj.get("crop_png_path") or ""),
                "crop_png_nbytes": int(obj.get("crop_png_nbytes", 0) or 0),
            }
        )
    return out


def deserialize_boxer_objects(raw_dir: Path, objects: Any) -> list[dict[str, Any]]:
    if objects is None:
        return []
    if hasattr(objects, "as_py"):
        objects = objects.as_py()
    if not isinstance(objects, list):
        return []
    out: list[dict[str, Any]] = []
    for obj in objects:
        if obj is None:
            continue
        item = dict(obj)
        rel = item.get("crop_png_path") or ""
        if rel:
            item["crop_png_abspath"] = str(raw_dir / str(rel))
        out.append(item)
    return out
