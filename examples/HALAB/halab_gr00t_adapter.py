#!/usr/bin/env python3
"""Adapt HA-Lab ``LeRobotVLADataset`` samples to GR00T ``VLAStepData``.

This adapter performs no offline dataset conversion.  It uses the HA-Lab loader
with body velocity enabled, keeps all 76 observation dimensions, selects the
four requested action groups (47 dimensions total), and optionally calls the
GR00T processor.

The HA-Lab loader package is bundled at ``Isaac-GR00T/data_loader/``; a sibling checkout
(``gr00t_baseline/data_loader/`` next to ``Isaac-GR00T/``) is used only when the bundled copy is absent.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info


# Make the HA-Lab package importable when this file lives at ``Isaac-GR00T/examples/HALAB/``:
# the copy bundled in this repo first, else a sibling checkout (also the accidental zip layout
# ``gr00t_baseline/data_loader/data_loader``).
REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = REPO_ROOT.parent
if (REPO_ROOT / "data_loader" / "__init__.py").is_file():
    DATA_LOADER_PACKAGE_PARENT = REPO_ROOT
elif (WORKSPACE_ROOT / "data_loader" / "__init__.py").is_file():
    DATA_LOADER_PACKAGE_PARENT = WORKSPACE_ROOT
elif (WORKSPACE_ROOT / "data_loader" / "data_loader" / "__init__.py").is_file():
    DATA_LOADER_PACKAGE_PARENT = WORKSPACE_ROOT / "data_loader"
else:
    raise FileNotFoundError(
        "Cannot find the HA-Lab data_loader package. Expected either "
        f"{WORKSPACE_ROOT / 'data_loader' / '__init__.py'} or "
        f"{WORKSPACE_ROOT / 'data_loader' / 'data_loader' / '__init__.py'}"
    )
if str(DATA_LOADER_PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(DATA_LOADER_PACKAGE_PARENT))

from data_loader import LeRobotVLADataset  # noqa: E402
from data_loader.ha.episode_split import halab_episode_key, split_episode_ids  # noqa: E402
from data_loader.ha.loader import LeRobotEpisodeLoader  # noqa: E402
from data_loader.ha.vla_task_registry import resolve_dataset_roots  # noqa: E402
from gr00t.data.types import EmbodimentTag, MessageType, VLAStepData  # noqa: E402


STATE_DIMS = {
    "body_q": 29,
    "body_dq": 29,
    "base_quat": 4,
    "inspire_hand": 12,
    "camera": 2,
}

ACTION_DIMS = {
    "body_q": 29,
    "base_quat": 4,
    "inspire_hand": 12,
    "camera": 2,
}

# Rows per action chunk; same switch as halab_47d_config.ACTION_HORIZON.
ACTION_HORIZON = int(os.environ.get("HALAB_ACTION_HORIZON", "32"))
# 30 Hz frames per chunk row (OpenHLM ``--action-fps 30/N``): row j is frame t + N (j + 1), so N = 1 is the
# original t+1 ... t+H chunk and N = 2 is 15 Hz (50 rows = 3.33 s). Same switch in halab_metrics.
ACTION_STRIDE = int(os.environ.get("HALAB_ACTION_STRIDE", "1"))
if ACTION_STRIDE < 1:
    raise ValueError(f"HALAB_ACTION_STRIDE must be >= 1, got {ACTION_STRIDE}")

# Image history: OpenHLM ``--temporal mem`` frame spacing and truncation (openpi.ha.temporal.sample_offsets,
# pi0_ha._truncate_history), fed through GR00T's own multi-frame input (each past frame is one more image in the
# VLM prompt, oldest first, current last) instead of MEM's SigLIP temporal attention. IMAGE_HISTORY past frames,
# IMAGE_HISTORY_STRIDE frames apart; in training each moves by U{-jitter..jitter}. A frame before the episode
# start is left out. Training only, per sample: every past frame is dropped with DROP_ALL, else the oldest
# k ~ U{1..K} with DROP_OLDEST. Evaluation uses the nominal spacing and the full history. No proprio history.
IMAGE_HISTORY = int(os.environ.get("HALAB_IMAGE_HISTORY", "0"))
IMAGE_HISTORY_STRIDE = int(os.environ.get("HALAB_IMAGE_HISTORY_STRIDE", "30"))
IMAGE_HISTORY_JITTER = int(os.environ.get("HALAB_IMAGE_HISTORY_JITTER", "5"))
IMAGE_HISTORY_DROP_ALL = float(os.environ.get("HALAB_IMAGE_HISTORY_DROP_ALL", "0"))
IMAGE_HISTORY_DROP_OLDEST = float(os.environ.get("HALAB_IMAGE_HISTORY_DROP_OLDEST", "0"))
if IMAGE_HISTORY < 0 or IMAGE_HISTORY_STRIDE < 1 or IMAGE_HISTORY_JITTER < 0:
    raise ValueError("HALAB_IMAGE_HISTORY must be >= 0, _STRIDE >= 1, _JITTER >= 0")
if IMAGE_HISTORY and 2 * IMAGE_HISTORY_JITTER >= IMAGE_HISTORY_STRIDE:
    raise ValueError("HALAB_IMAGE_HISTORY_JITTER must stay below stride / 2 so the frames keep their order")
if min(IMAGE_HISTORY_DROP_ALL, IMAGE_HISTORY_DROP_OLDEST) < 0 or IMAGE_HISTORY_DROP_ALL + IMAGE_HISTORY_DROP_OLDEST > 1:
    raise ValueError("HALAB_IMAGE_HISTORY_DROP_ALL / _DROP_OLDEST must be >= 0 with a sum <= 1")


def _numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy().astype(np.float32, copy=False)


_HISTORY_RNG: tuple[int, np.random.Generator] | None = None


def _history_rng() -> np.random.Generator:
    """Per-process generator from OS entropy: loader workers of different ranks must not share draws."""
    global _HISTORY_RNG
    if _HISTORY_RNG is None or _HISTORY_RNG[0] != os.getpid():
        _HISTORY_RNG = (os.getpid(), np.random.default_rng())
    return _HISTORY_RNG[1]


def history_frames(anchor: int, *, train: bool, rng: np.random.Generator | None = None) -> list[int]:
    """Episode frame indices of the past images kept for ``anchor``, oldest first."""
    if IMAGE_HISTORY == 0:
        return []
    rng = rng if rng is not None else _history_rng()
    ages = np.arange(IMAGE_HISTORY, 0, -1, dtype=np.int64) * IMAGE_HISTORY_STRIDE
    if train and IMAGE_HISTORY_JITTER > 0:
        ages = ages + rng.integers(-IMAGE_HISTORY_JITTER, IMAGE_HISTORY_JITTER + 1, size=IMAGE_HISTORY)
    frames = int(anchor) - ages
    keep = frames >= 0
    if train and (IMAGE_HISTORY_DROP_ALL > 0 or IMAGE_HISTORY_DROP_OLDEST > 0):
        coin = rng.random()
        if coin < IMAGE_HISTORY_DROP_ALL:
            keep[:] = False
        elif coin < IMAGE_HISTORY_DROP_ALL + IMAGE_HISTORY_DROP_OLDEST:
            keep[: int(rng.integers(1, IMAGE_HISTORY + 1))] = False
    return [int(frame) for frame in frames[keep]]


def read_history_images(
    halab_dataset: LeRobotVLADataset, index: int, *, train: bool, rng: np.random.Generator | None = None
) -> list[np.ndarray]:
    """Past RGB uint8 (H, W, 3) frames for sample ``index`` of the HA loader, oldest first."""
    task_idx, episode_index, anchor = halab_dataset._locate(index)
    frames = history_frames(anchor, train=train, rng=rng)
    if not frames:
        return []
    handle = halab_dataset._get_io_handle(task_idx, episode_index)
    rows = sorted(set(frames))
    aligned = handle.read_aligned_rows(rows)
    mp4_index = {frame: int(aligned["frame_index"][pos]) for pos, frame in enumerate(rows)}
    cap_key = (task_idx, episode_index)
    return [halab_dataset._read_image(handle, mp4_index[frame], cap_key) for frame in frames]


class HalabGr00tAdapter(Dataset):
    """Thin wrapper from HA-Lab samples to GR00T processor inputs.

    When ``processor`` is ``None``, ``__getitem__`` returns an unprocessed
    ``VLAStepData`` instance.  This mode is used by the CLI shape test below.
    During training, pass the N1.7 processor and the returned value is directly
    consumable by the official GR00T data collator.
    """

    def __init__(self, halab_dataset: LeRobotVLADataset, processor: Any | None = None) -> None:
        self.halab_dataset = halab_dataset
        self.processor = processor

    def __len__(self) -> int:
        return len(self.halab_dataset)

    @staticmethod
    def to_vla_step(sample: dict[str, Any]) -> VLAStepData:
        proprio = sample["observation"]["proprioceptive"]
        action = sample["action"]

        # HA images are (T, C, H, W) float32 in [0, 1].  N1.7 uses
        # Albumentations by default, which requires uint8 channel-last images.
        images = sample["observation"]["image"]
        images_uint8 = (
            (images * 255.0)
            .round()
            .clamp(0, 255)
            .to(torch.uint8)
            .permute(0, 2, 3, 1)
            .contiguous()
            .cpu()
            .numpy()
        )
        if images_uint8.ndim != 4 or images_uint8.shape[-1] != 3:
            raise ValueError(
                f"Expected HA images to convert to (T,H,W,3), got {images_uint8.shape}"
            )

        # Keep the latest observation only.  Slices retain the time dimension:
        # each value has shape (1, D).
        states = {
            "body_q": _numpy(proprio["measured_joint_q"][-1:]),
            "body_dq": _numpy(proprio["measured_joint_dq"][-1:]),
            "base_quat": _numpy(proprio["body_quat"][-1:]),
            "inspire_hand": _numpy(proprio["hand_action"][-1:]),
            "camera": _numpy(proprio["camera_action"][-1:]),
        }

        # HA actions have shape (1, horizon * stride, D); the leading dimension denotes
        # the single anchor.  Drop it, keep every ACTION_STRIDE-th 30 Hz row (frames
        # t + N, t + 2N, ...) and deliberately omit measured_joint_dq.
        rows = slice(ACTION_STRIDE - 1, None, ACTION_STRIDE)
        actions = {
            "body_q": _numpy(action["measured_joint_q"][0][rows]),
            "base_quat": _numpy(action["body_quat"][0][rows]),
            "inspire_hand": _numpy(action["hand_action"][0][rows]),
            "camera": _numpy(action["camera_action"][0][rows]),
        }

        state_dim = sum(value.shape[-1] for value in states.values())
        action_dim = sum(value.shape[-1] for value in actions.values())
        horizons = {value.shape[0] for value in actions.values()}
        if state_dim != 76:
            raise ValueError(f"Expected 76D state, got {state_dim}: {STATE_DIMS}")
        if action_dim != 47:
            raise ValueError(f"Expected 47D action, got {action_dim}: {ACTION_DIMS}")
        if horizons != {ACTION_HORIZON}:
            raise ValueError(f"Expected action horizon {ACTION_HORIZON}, got {sorted(horizons)}")

        return VLAStepData(
            images={"ego_view": [frame for frame in images_uint8]},
            states=states,
            actions=actions,
            text=str(sample["language"]),
            embodiment=EmbodimentTag.NEW_EMBODIMENT,
            metadata={
                "task_key": sample["meta"]["task_key"],
                "episode_index": sample["meta"]["episode_index"],
                "frame_id": sample["meta"]["frame_id"],
                "action_valid_mask": _numpy(
                    sample["meta"]["action_chunk_valid_mask"][0][rows]
                ),
            },
        )

    def __getitem__(self, index: int) -> Any:
        sample = self.halab_dataset[index]
        step = self.to_vla_step(sample)
        if IMAGE_HISTORY > 0 and self.halab_dataset.load_images:
            # Training jitter / truncation follow the processor's mode (set per sample by the datasets below).
            train = bool(getattr(self.processor, "training", False))
            past = read_history_images(self.halab_dataset, index, train=train)
            step.images = {"ego_view": past + step.images["ego_view"]}
        if self.processor is None:
            return step
        processed = self.processor(
            [{"type": MessageType.EPISODE_STEP.value, "content": step}]
        )
        # The HA loader edge-pads short chunks and provides the true validity
        # mask.  GR00T's processor only knows the configured horizon, so apply
        # the per-sample mask here to prevent loss on repeated padding rows.
        valid = torch.from_numpy(step.metadata["action_valid_mask"]).to(
            dtype=processed["action_mask"].dtype
        )
        if processed["action_mask"].ndim != 2:
            raise ValueError(
                "Expected per-sample action_mask (horizon, action_dim), got "
                f"{tuple(processed['action_mask'].shape)}"
            )
        processed["action_mask"][: valid.shape[0]] *= valid[:, None]
        return processed


def _statistics(values: np.ndarray) -> dict[str, list[float]]:
    values = np.asarray(values, dtype=np.float32)
    return {
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "q01": np.quantile(values, 0.01, axis=0).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).tolist(),
    }


def compute_halab_statistics(
    halab_dataset: LeRobotVLADataset,
    *,
    max_samples: int,
) -> dict[str, dict[str, dict[str, dict[str, list[float]]]]]:
    """Estimate processor normalization statistics from deterministic samples."""

    if len(halab_dataset) < 1:
        raise ValueError("Cannot compute statistics from an empty HA-Lab dataset")
    count = len(halab_dataset) if max_samples <= 0 else min(max_samples, len(halab_dataset))
    indices = np.linspace(0, len(halab_dataset) - 1, count, dtype=np.int64)
    collected: dict[str, dict[str, list[np.ndarray]]] = {
        "state": {key: [] for key in STATE_DIMS},
        "action": {key: [] for key in ACTION_DIMS},
    }
    raw_adapter = HalabGr00tAdapter(halab_dataset, processor=None)
    for position, index in enumerate(indices, start=1):
        step = raw_adapter[int(index)]
        for key, value in step.states.items():
            collected["state"][key].append(value)
        valid = np.asarray(step.metadata["action_valid_mask"], dtype=bool)
        for key, value in step.actions.items():
            collected["action"][key].append(value[valid])
        if position == 1 or position % 100 == 0 or position == count:
            print(f"statistics: {position}/{count} samples", flush=True)

    stats = {
        modality: {
            key: _statistics(np.concatenate(chunks, axis=0))
            for key, chunks in groups.items()
        }
        for modality, groups in collected.items()
    }
    return {EmbodimentTag.NEW_EMBODIMENT.value: stats}


def load_or_compute_statistics(
    halab_dataset: LeRobotVLADataset,
    statistics_path: str | Path,
    *,
    max_samples: int,
) -> dict[str, dict[str, dict[str, dict[str, list[float]]]]]:
    path = Path(statistics_path).expanduser()
    if path.is_file():
        print(f"Loading HA-Lab statistics from {path}", flush=True)
        return json.loads(path.read_text(encoding="utf-8"))
    stats = compute_halab_statistics(halab_dataset, max_samples=max_samples)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(f"Saved HA-Lab statistics to {path}", flush=True)
    return stats


def _rank_and_world_size() -> tuple[int, int]:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(), torch.distributed.get_world_size()
    return int(os.environ.get("RANK", "0")), int(os.environ.get("WORLD_SIZE", "1"))


class HalabGr00tTrainingDataset(IterableDataset):
    """Endless, rank- and worker-sharded training stream for the GR00T trainer.

    ``Gr00tTrainer.get_train_dataloader`` builds a plain ``DataLoader`` without
    a distributed sampler (GR00T's own ``ShardedMixtureDataset`` shards by rank
    itself).  A map-style dataset would therefore feed every GPU the same
    samples.  Each epoch is one seeded permutation of all anchors, split
    disjointly over ``world_size * num_workers`` streams.
    """

    def __init__(
        self,
        halab_dataset: LeRobotVLADataset,
        processor: Any,
        statistics: dict[str, Any],
        *,
        seed: int = 42,
    ) -> None:
        self.seed = int(seed)
        self.statistics = statistics
        self.processor = processor
        processor.set_statistics(statistics, override=True)
        self.adapter = HalabGr00tAdapter(halab_dataset, processor=processor)
        self.num_anchors = len(self.adapter)
        # Read at construction, like ShardedMixtureDataset: the process group is
        # initialized before GR00T builds datasets, and workers inherit these.
        self.rank, self.world_size = _rank_and_world_size()

    # No __len__ on purpose: HF then treats the stream as endless and max_steps
    # alone bounds training.  With a length, HF would restart this iterator
    # (from epoch 0, the same permutation) at every assumed epoch boundary.

    def __iter__(self):
        worker = get_worker_info()
        worker_id, num_workers = (worker.id, worker.num_workers) if worker else (0, 1)
        num_streams = self.world_size * num_workers
        stream = self.rank * num_workers + worker_id
        epoch = 0
        while True:
            order = np.random.default_rng([self.seed, epoch]).permutation(self.num_anchors)
            # Drop the remainder so every stream yields the same count per epoch.
            usable = len(order) - len(order) % num_streams
            for index in order[stream:usable:num_streams]:
                # The processor object is shared with the eval dataset; set the
                # mode per sample in case both run in one process.
                self.processor.train()
                yield self.adapter[int(index)]
            epoch += 1

    def get_dataset_statistics(self) -> dict[str, Any]:
        return self.statistics

    def reset_seed(self, seed: int) -> None:
        # Gr00tTrainer calls this on resume (seed + global_step) instead of
        # skipping already-seen batches.
        self.seed = int(seed)


class HalabGr00tEvalDataset(Dataset):
    """Deterministic, map-style evaluation set (HF shards it across ranks)."""

    def __init__(
        self,
        halab_dataset: LeRobotVLADataset,
        processor: Any,
        statistics: dict[str, Any],
        *,
        max_samples: int | None = None,
    ) -> None:
        self.statistics = statistics
        self.processor = processor
        processor.set_statistics(statistics, override=True)
        self.adapter = HalabGr00tAdapter(halab_dataset, processor=processor)
        if max_samples is not None and 0 < int(max_samples) < len(self.adapter):
            self.indices = np.linspace(0, len(self.adapter) - 1, int(max_samples), dtype=np.int64)
        else:
            self.indices = np.arange(len(self.adapter), dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> Any:
        self.processor.eval()
        return self.adapter[int(self.indices[index])]

    def get_dataset_statistics(self) -> dict[str, Any]:
        return self.statistics


def build_halab_dataset(
    dataset_root: str | Path | None = None,
    *,
    data_parent: str | Path | None = None,
    task_names: list[str] | None = None,
    episodes: list[int] | None = None,
    episode_keys: list[str] | None = None,
    max_anchors_per_episode: int | None = None,
    load_images: bool = True,
) -> LeRobotVLADataset:
    """Construct the HA loader with the contract needed by this baseline."""

    if (dataset_root is None) == (data_parent is None):
        raise ValueError("Provide exactly one of dataset_root or data_parent")

    return LeRobotVLADataset(
        dataset_root=Path(dataset_root) if dataset_root is not None else None,
        data_parent=Path(data_parent) if data_parent is not None else None,
        task_names=task_names,
        dataset_name="halab",
        episodes=episodes,
        episode_keys=episode_keys,
        num_past_frames=0,
        action_chunk_size=ACTION_HORIZON * ACTION_STRIDE,  # 30 Hz rows; to_vla_step keeps every N-th
        action_start_offset=1,
        load_images=load_images,
        include_body_vel=True,  # Preserve dq in the 76D observation.
        use_delta_actions=False,
        bev_uncertainty_source="aligned_extras",
        max_anchors_per_episode=max_anchors_per_episode,
    )


def build_openhlm_halab_episode_split(
    dataset_root: str | Path | None = None,
    *,
    data_parent: str | Path | None = None,
    task_names: list[str] | None = None,
    fraction: float = 0.05,
    seed: int = 7,
) -> tuple[list[str], list[str]]:
    """Return the exact OpenHLM global HALAB ``(train_keys, val_keys)`` split."""

    roots = resolve_dataset_roots(
        dataset_root=Path(dataset_root) if dataset_root is not None else None,
        data_parent=Path(data_parent) if data_parent is not None else None,
        task_names=task_names,
    )
    all_keys = [
        halab_episode_key(root.name, episode.episode_index)
        for root in roots
        for episode in LeRobotEpisodeLoader(root).list_episodes()
    ]
    return split_episode_ids(all_keys, fraction=fraction, seed=seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test the HA-Lab -> GR00T mapping")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset-root", type=Path)
    source.add_argument("--data-parent", type=Path)
    parser.add_argument("--task-name", action="append", default=[])
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--no-images", action="store_true")
    args = parser.parse_args()

    halab = build_halab_dataset(
        args.dataset_root,
        data_parent=args.data_parent,
        task_names=args.task_name or None,
        episodes=[args.episode_index],
        max_anchors_per_episode=max(1, args.sample_index + 1),
        load_images=not args.no_images,
    )
    adapter = HalabGr00tAdapter(halab, processor=None)
    step = adapter[args.sample_index]

    print(f"dataset samples: {len(adapter)}")
    print(f"text: {step.text!r}")
    print("images:", {key: [tuple(frame.shape) for frame in value] for key, value in step.images.items()})
    print("states:", {key: tuple(value.shape) for key, value in step.states.items()})
    print("actions:", {key: tuple(value.shape) for key, value in step.actions.items()})
    print("state_dim:", sum(value.shape[-1] for value in step.states.values()))
    print("action_dim:", sum(value.shape[-1] for value in step.actions.values()))
    print("action_horizon:", next(iter(step.actions.values())).shape[0])
    print("metadata:", step.metadata)


if __name__ == "__main__":
    main()
