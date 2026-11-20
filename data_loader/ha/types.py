from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EpisodeInfo:
    episode_index: int
    episode_id: str
    length: int
    duration_s: float
    tasks: list[str]
    collection_mode: str
    teleoperator: str = ""


@dataclass
class TimeSeries:
    """One scalar or vector field sampled on a timeline."""

    name: str
    stamp_ns: list[int]
    t_sec: list[float]
    values: list[Any]  # scalar per step or list[float] per step
    unit: str = ""
    source: str = ""

    @property
    def n(self) -> int:
        return len(self.stamp_ns)

    def is_empty(self) -> bool:
        return self.n == 0


@dataclass
class EpisodeData:
    root: Path
    info: EpisodeInfo
    start_ns: int
    fps: float
    parquet: dict[str, Any] = field(default_factory=dict)
    frames: list[dict[str, Any]] = field(default_factory=list)
    raw_streams: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    aligned_extras: dict[str, Any] = field(default_factory=dict)
    video_path: Path | None = None
    frames_path: Path | None = None
    raw_dir: Path | None = None
    parquet_path: Path | None = None
    aligned_extras_path: Path | None = None
    motor_names: list[str] = field(default_factory=list)

    def raw_stream_names(self) -> list[str]:
        return sorted(self.raw_streams.keys())

    def nonempty_raw_streams(self) -> list[str]:
        return [name for name, records in self.raw_streams.items() if records]
