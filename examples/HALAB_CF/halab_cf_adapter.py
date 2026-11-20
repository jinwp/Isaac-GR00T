#!/usr/bin/env python3
"""GR00T training/eval datasets with exact camera-rotation counterfactuals.

Wraps ``examples/HALAB/halab_gr00t_adapter.py`` without modifying it (the v3 chains
import that file).  Per training sample on a locomotion chunk, with probability
``frac``: draw ``delta ~ N(0, std)`` clipped to ``max``, warp the ego frame by the
scene-rotation homography and rotate the 32 base-quaternion targets by
``label_sign * delta``.  Every frame, train and eval, then gets the same center
crop so warp borders never distinguish augmented from plain samples.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import get_worker_info


HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "HALAB"
for path in (str(HERE), str(BASE)):
    if path not in sys.path:
        sys.path.insert(0, path)

from gr00t.data.types import MessageType  # noqa: E402
from halab_gr00t_adapter import (  # noqa: E402
    HalabGr00tAdapter,
    HalabGr00tEvalDataset,
    HalabGr00tTrainingDataset,
)
import halab_rotation_cf as rcf  # noqa: E402


@dataclass
class CFConfig:
    frac: float = 0.0
    std_deg: float = 4.0
    max_deg: float = 7.0
    crop_frac: float = 0.08
    walk_threshold: float = 0.15
    label_sign: int = 1
    seed: int = 1234

    def to_dict(self) -> dict:
        return asdict(self)


class HalabCFAdapter(HalabGr00tAdapter):
    def __init__(self, halab_dataset, processor, cfg: CFConfig, *, augment: bool) -> None:
        super().__init__(halab_dataset, processor)
        self.cfg = cfg
        self.augment = augment
        self.rng = np.random.default_rng(cfg.seed)

    def transform_step(self, step, sample):
        frames = list(step.images["ego_view"])
        if self.augment and self.cfg.frac > 0.0:
            if rcf.is_walking(step.actions["body_q"], self.cfg.walk_threshold) and self.rng.random() < self.cfg.frac:
                std = math.radians(self.cfg.std_deg)
                lim = math.radians(self.cfg.max_deg)
                delta = float(np.clip(self.rng.normal(0.0, std), -lim, lim))
                c2w = sample["observation"]["camera_pose"][-1].numpy()
                up, _ = rcf.up_axis_in_camera(c2w)
                frames = [rcf.warp_scene_rotation(frame, up, delta) for frame in frames]
                step.actions["base_quat"] = rcf.rotate_quat_yaw(
                    step.actions["base_quat"], self.cfg.label_sign * delta
                ).astype(np.float32)
        if self.cfg.crop_frac > 0.0:
            frames = [rcf.center_crop_resize(frame, self.cfg.crop_frac) for frame in frames]
        step.images["ego_view"] = frames
        return step

    def __getitem__(self, index: int):
        sample = self.halab_dataset[index]
        step = self.transform_step(self.to_vla_step(sample), sample)
        if self.processor is None:
            return step
        processed = self.processor([{"type": MessageType.EPISODE_STEP.value, "content": step}])
        # Same per-sample validity masking as HalabGr00tAdapter.__getitem__.
        valid = torch.from_numpy(step.metadata["action_valid_mask"]).to(dtype=processed["action_mask"].dtype)
        if processed["action_mask"].ndim != 2:
            raise ValueError(
                "Expected per-sample action_mask (horizon, action_dim), got "
                f"{tuple(processed['action_mask'].shape)}"
            )
        processed["action_mask"][: valid.shape[0]] *= valid[:, None]
        return processed


class HalabCFTrainingDataset(HalabGr00tTrainingDataset):
    """Rank/worker-sharded stream (unchanged order) with counterfactual augmentation."""

    def __init__(self, halab_dataset, processor, statistics, cfg: CFConfig, *, seed: int = 42) -> None:
        super().__init__(halab_dataset, processor, statistics, seed=seed)
        self.adapter = HalabCFAdapter(halab_dataset, processor, cfg, augment=True)

    def __iter__(self):
        worker = get_worker_info()
        worker_id = worker.id if worker else 0
        # Reproducible per stream; reset_seed(seed + step) on resume gives fresh draws.
        self.adapter.rng = np.random.default_rng([self.adapter.cfg.seed, self.seed, self.rank, worker_id])
        yield from super().__iter__()


class HalabCFEvalDataset(HalabGr00tEvalDataset):
    """Deterministic eval set with the same crop as training, no rotation."""

    def __init__(self, halab_dataset, processor, statistics, cfg: CFConfig, *, max_samples=None) -> None:
        super().__init__(halab_dataset, processor, statistics, max_samples=max_samples)
        self.adapter = HalabCFAdapter(halab_dataset, processor, cfg, augment=False)
