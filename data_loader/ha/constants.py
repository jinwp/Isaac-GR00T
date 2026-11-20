"""Layout constants matching ros2_data_hub robot_frame/feature_extract.py."""

from __future__ import annotations

G1_MOTOR_DIM = 29
G1_BASE_QUAT_DIM = 4
INSPIRE_Q_DIM = 12
DYNAMIXEL_DIM = 2
# observation.state: [29 q | 29 dq | 4 base_quat | 12 hand | 2 cam]
PROPRIO_DIM = G1_MOTOR_DIM + G1_MOTOR_DIM + G1_BASE_QUAT_DIM + INSPIRE_Q_DIM + DYNAMIXEL_DIM
LEGACY_PROPRIO_DIM = G1_MOTOR_DIM + INSPIRE_Q_DIM + DYNAMIXEL_DIM
POSE_DIM = 7
MEASURED_ACTION_DIM = G1_MOTOR_DIM + G1_MOTOR_DIM + 4 + 3

INSPIRE_LEFT_JOINTS = [f"left_{i}" for i in range(6)]
INSPIRE_RIGHT_JOINTS = [f"right_{i}" for i in range(6)]
INSPIRE_FORCE_LEFT = [f"L_force_{i}" for i in range(6)]
INSPIRE_FORCE_RIGHT = [f"R_force_{i}" for i in range(6)]

_Q_END = G1_MOTOR_DIM
_DQ_END = _Q_END + G1_MOTOR_DIM
_QUAT_END = _DQ_END + G1_BASE_QUAT_DIM
_HAND_END = _QUAT_END + INSPIRE_Q_DIM

PROPRIO_SLICES = {
    "g1_motor_q": (0, _Q_END),
    "g1_motor_dq": (_Q_END, _DQ_END),
    "g1_base_quat": (_DQ_END, _QUAT_END),
    "inspire_q": (_QUAT_END, _HAND_END),
    "dynamixel": (_HAND_END, PROPRIO_DIM),
}

RAW_STREAM_NAMES = (
    "left",
    "depth",
    "imu",
    "g1",
    "inspire_state",
    "dynamixel_state",
    "camera_pose",
    "bev_output",
    "bev_phase",
    "boxer_output",
    "g1_debug",
    "inspire_action",
    "dynamixel_action",
    "vla_action",
)

HUB_RAW_STREAMS = ("left", "depth", "imu", "g1", "inspire_state", "dynamixel_state")
LOCAL_RAW_STREAMS = (
    "camera_pose",
    "bev_output",
    "bev_phase",
    "boxer_output",
    "g1_debug",
    "inspire_action",
    "dynamixel_action",
    "vla_action",
)
