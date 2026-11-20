from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}") from exc
    return records


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    for record in read_jsonl(path):
        yield record
