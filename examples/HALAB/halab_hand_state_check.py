#!/usr/bin/env python3
"""Per-episode check of the recorded hand state vs the hand command in HA-Lab parquet data.

The model's hand input is ``observation.state[62:74]`` (Inspire openness, 1 = open, [L6 | R6]) and
its hand target is ``action.hand_action`` (the commanded openness).  For every episode this writes:

* ``zero_L`` / ``zero_R``: fraction of frames whose 6 state values are all 0 (missing reading,
  which the model sees as a fully closed hand);
* ``lag_R``: shift (frames, 30 Hz) at which the right-hand state best matches the command
  (``state[t + lag] ~ command[t]``; a real hand lags the command, a copy of it has lag 0);
* ``err0_R`` / ``errlag_R``: mean |state - command| at lag 0 / at ``lag_R``;
* ``same_R``: fraction of frames where state == command (< 1e-3);
* ``cmd_range_R``: max - min of the mean right command (whether the episode grasps);
* ``state0_R`` / ``cmd0_R``: mean right openness over the first second;
* ``video_frames``: frames in the episode's left-camera mp4 (should equal the parquet rows ``n``).

    python examples/HALAB/halab_hand_state_check.py --data-parent <halab root> --out <outputs>/hand_state_new.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

HAND = slice(62, 74)


def best_lag(state: np.ndarray, cmd: np.ndarray, lags: range) -> tuple[int, float]:
    best = (0, float("inf"))
    for lag in lags:
        if lag >= 0:
            s, c = state[lag:], cmd[: len(cmd) - lag]
        else:
            s, c = state[:lag], cmd[-lag:]
        if len(s) < 60:
            continue
        err = float(np.abs(s - c).mean())
        if err < best[1]:
            best = (lag, err)
    return best


def video_frames(path: Path) -> int | None:
    """Frame count of the episode's left-camera mp4 (header only)."""
    try:
        import cv2
    except ImportError:
        return None
    video = path.parents[2] / "videos" / path.parent.name / "observation.images.left" / f"{path.stem}.mp4"
    if not video.exists():
        return -1
    cap = cv2.VideoCapture(str(video))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def episode_stats(path: Path) -> dict:
    has_cmd = "action.hand_action" in pq.read_schema(path).names
    table = pq.read_table(path, columns=["observation.state"] + (["action.hand_action"] if has_cmd else []))
    state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)[:, HAND]
    sl, sr = state[:, :6], state[:, 6:]
    out = {
        "n": int(len(state)),
        "video_frames": video_frames(path),
        "zero_L": float((np.abs(sl).max(1) == 0).mean()),
        "zero_R": float((np.abs(sr).max(1) == 0).mean()),
        "state_mean_L": float(sl.mean()),
        "state_mean_R": float(sr.mean()),
        "state0_R": float(sr[:30].mean()),
    }
    if not has_cmd:  # old exports keep the command only in raw jsonl
        return out
    cr = np.asarray(table["action.hand_action"].to_pylist(), dtype=np.float32)[:, 6:12]
    lag, errlag = best_lag(sr, cr, range(-30, 46))
    mean_cmd = cr.mean(1)
    return out | {
        "lag_R": int(lag),
        "err0_R": float(np.abs(sr - cr).mean()),
        "errlag_R": errlag,
        "same_R": float((np.abs(sr - cr).max(1) < 1e-3).mean()),
        "cmd_range_R": float(mean_cmd.max() - mean_cmd.min()),
        "cmd0_R": float(cr[:30].mean()),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-parent", required=True, help="folder holding one folder per task")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    out = {}
    for task in sorted(d for d in Path(args.data_parent).iterdir() if (d / "data").is_dir()):
        files = sorted((task / "data").glob("chunk-*/episode_*.parquet"))
        for f in files:
            key = f"{task.name}/{int(f.stem.split('_')[-1])}"
            try:
                out[key] = episode_stats(f)
            except Exception as e:  # report and continue
                out[key] = {"error": repr(e)}
        eps = [out[f"{task.name}/{int(f.stem.split('_')[-1])}"] for f in files]
        ok = [e for e in eps if "error" not in e]
        if ok:
            vid_bad = sum(1 for e in ok if e["video_frames"] is not None and abs(e["video_frames"] - e["n"]) > 2)
            line = (f"{task.name[:50]:50} eps {len(ok):3}  video!=rows {vid_bad:3}  zeroL {np.mean([e['zero_L'] for e in ok]):.2f}  "
                    f"zeroR {np.mean([e['zero_R'] for e in ok]):.2f}")
            cmd = [e for e in ok if "lag_R" in e]
            if cmd:
                line += (f"  lagR {np.median([e['lag_R'] for e in cmd]):5.1f}  err0R {np.mean([e['err0_R'] for e in cmd]):.3f}"
                         f"  sameR {np.mean([e['same_R'] for e in cmd]):.2f}")
            print(line, flush=True)
        else:
            print(f"{task.name[:50]:50} no readable episodes ({len(eps)} files)", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"done: {args.out} ({len(out)} episodes)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
