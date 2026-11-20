#!/usr/bin/env python3
"""Summarize HA-Lab task/episode/frame counts from LeRobot metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-parent", type=Path, required=True)
    args = parser.parse_args()

    rows: list[tuple[str, int, int, float]] = []
    for task_root in sorted(args.data_parent.expanduser().resolve().iterdir()):
        info_path = task_root / "meta" / "info.json"
        episodes_path = task_root / "meta" / "episodes.jsonl"
        if not info_path.is_file() or not episodes_path.is_file():
            continue
        info = json.loads(info_path.read_text(encoding="utf-8"))
        episodes = [
            json.loads(line)
            for line in episodes_path.read_text(encoding="utf-8").splitlines()
            if line
        ]
        frames = sum(int(ep["length"]) for ep in episodes)
        fps = float(info.get("fps", 30.0))
        rows.append((task_root.name, len(episodes), frames, frames / fps / 3600.0))

    if not rows:
        raise SystemExit(f"No LeRobot task roots found under {args.data_parent}")

    for name, episodes, frames, hours in rows:
        print(f"{name}: episodes={episodes}, frames={frames}, hours={hours:.2f}")

    total_episodes = sum(row[1] for row in rows)
    total_frames = sum(row[2] for row in rows)
    total_hours = sum(row[3] for row in rows)
    print("---")
    print(f"tasks={len(rows)}")
    print(f"episodes={total_episodes}")
    print(f"frames={total_frames}")
    print(f"hours={total_hours:.2f}")


if __name__ == "__main__":
    main()
