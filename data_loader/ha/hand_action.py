"""Convert raw HA Inspire hand commands to the shared motion-action schema."""

from __future__ import annotations

import numpy as np


# Constants and conversion functions below are copied from:
# third_party/MotionTokenizers/src/motion_tokenizers/data/hand_mapping.py
# third_party/MotionTokenizers/src/motion_tokenizers/data/adapters/halab_parquet.py
# Keep them local so data_loader remains usable without the submodule.

DEFAULT_REST_FINGER_ANGLE = 0.26
DEFAULT_GRIP_FINGER_ANGLE = 1.30
DEFAULT_REST_THUMB_PITCH = 0.20
DEFAULT_GRIP_THUMB_PITCH = 0.45
DEFAULT_REST_THUMB_YAW = 1.0
DEFAULT_GRIP_THUMB_YAW = 1.3

_HE_TO_OURS_HAND_INSPIRE_PER_SIDE = np.array([5, 4, 3, 2, 1, 0], dtype=np.int64)

# halab hand per-side order is IDENTICAL to HE's Inspire order
# [pinky, ring, middle, index, thumb_bend, thumb_rotate] → reuse HE's per-side
# permutation into our native order [thumb_yaw, thumb_pitch, index, middle,
# ring, pinky], lifted to the full 12-dim [left6 | right6] vector.
_HALAB_HAND_PERM_12 = np.concatenate([
    _HE_TO_OURS_HAND_INSPIRE_PER_SIDE,
    _HE_TO_OURS_HAND_INSPIRE_PER_SIDE + 6,
])

# Per-channel rest (openness=1) / grip (openness=0) angles in OUR native hand
# order [thumb_yaw, thumb_pitch, index, middle, ring, pinky] × {L, R}. These are
# omniretarget's `expand_inspire_hand_to_target_dofs` ramps:
# angle = rest + (1 − openness) · (grip − rest).
_HAND_REST_PER_SIDE = np.array([
    DEFAULT_REST_THUMB_YAW,
    DEFAULT_REST_THUMB_PITCH,
    DEFAULT_REST_FINGER_ANGLE,
    DEFAULT_REST_FINGER_ANGLE,
    DEFAULT_REST_FINGER_ANGLE,
    DEFAULT_REST_FINGER_ANGLE,
], dtype=np.float64)
_HAND_GRIP_PER_SIDE = np.array([
    DEFAULT_GRIP_THUMB_YAW,
    DEFAULT_GRIP_THUMB_PITCH,
    DEFAULT_GRIP_FINGER_ANGLE,
    DEFAULT_GRIP_FINGER_ANGLE,
    DEFAULT_GRIP_FINGER_ANGLE,
    DEFAULT_GRIP_FINGER_ANGLE,
], dtype=np.float64)
HAND_REST_12 = np.tile(_HAND_REST_PER_SIDE, 2)
HAND_GRIP_12 = np.tile(_HAND_GRIP_PER_SIDE, 2)


def remap_hand_halab_to_ours(hand12_raw: np.ndarray) -> np.ndarray:
    """Permute halab/HE hand order to our native order."""
    if hand12_raw.shape[-1] != 12:
        raise ValueError(f"hand shape {hand12_raw.shape} last dim != 12")
    return np.take(hand12_raw, _HALAB_HAND_PERM_12, axis=-1)


def inspire_openness_to_radians(hand12_ours: np.ndarray) -> np.ndarray:
    """Convert 0..1 Inspire openness (1.0 = open) to proximal joint radians."""
    if hand12_ours.shape[-1] != 12:
        raise ValueError(f"hand shape {hand12_ours.shape} last dim != 12")
    grip_fraction = 1.0 - np.clip(hand12_ours, 0.0, 1.0)
    return HAND_REST_12 + grip_fraction * (HAND_GRIP_12 - HAND_REST_12)


def halab_hand_action_to_radians(hand12_raw: np.ndarray) -> np.ndarray:
    """Convert raw HA hand action to native-order float32 joint radians."""
    hand12_openness = remap_hand_halab_to_ours(np.asarray(hand12_raw))
    hand12_rad = inspire_openness_to_radians(hand12_openness)
    return hand12_rad.astype(np.float32, copy=False)
