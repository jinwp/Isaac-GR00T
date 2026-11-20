"""GR00T N1.7 modality configuration for the HA-Lab 76D-state/47D-action baseline."""

import os

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


STATE_KEYS = [
    "body_q",
    "body_dq",
    "base_quat",
    "inspire_hand",
    "camera",
]

ACTION_KEYS = [
    "body_q",
    "base_quat",
    "inspire_hand",
    "camera",
]

# Rows per action chunk.  The env var is the one switch for a whole run
# (loader, modality config, model, metrics); 32 = the v1-v4 runs.  The rows are
# HALAB_ACTION_STRIDE 30 Hz frames apart (1 = 30 Hz, 2 = 15 Hz; see halab_gr00t_adapter).
ACTION_HORIZON = int(os.environ.get("HALAB_ACTION_HORIZON", "32"))
# Past images per sample and their spacing in 30 Hz frames (halab_gr00t_adapter reads them,
# with jitter / truncation in training, so a sample can carry fewer).
IMAGE_HISTORY = int(os.environ.get("HALAB_IMAGE_HISTORY", "0"))
IMAGE_HISTORY_STRIDE = int(os.environ.get("HALAB_IMAGE_HISTORY_STRIDE", "30"))


HALAB_47D_CONFIG = {
    "video": ModalityConfig(
        delta_indices=[-k * IMAGE_HISTORY_STRIDE for k in range(IMAGE_HISTORY, 0, -1)] + [0],
        modality_keys=["ego_view"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=STATE_KEYS,
    ),
    "action": ModalityConfig(
        # The HA-Lab loader already returns the future t+1 ... t+H chunk.
        # These indices describe the H rows received by the GR00T processor.
        delta_indices=list(range(ACTION_HORIZON)),
        modality_keys=ACTION_KEYS,
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            )
            for _ in ACTION_KEYS
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}


register_modality_config(
    HALAB_47D_CONFIG,
    embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
)
