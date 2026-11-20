#!/usr/bin/env python3
"""Exact camera-rotation counterfactuals for HA-Lab egocentric frames.

Rotating the scene by ``angle`` (right-handed about world up, through the camera
center) maps camera rays by ``R(u_c, angle)``, where ``u_c`` is world-up in OpenCV
camera coordinates (x right, y down, z forward).  Pixels move by the homography
``H = K R K^-1``, which does not depend on depth.  The matching label rotates the
future base-quaternion targets by the same angle about the IMU world z axis
(``q' = q_z(angle) * q``) and leaves joint, hand and camera targets unchanged:
if the world turns by ``angle`` around the robot, the demonstrated heading plan
turns with it.

``label_sign`` (from ``small_policy/build_cache.py``'s warp calibration) absorbs a
possible sign difference between the floor-aligned camera world and the IMU world.

The robot actually turns about the pelvis, not the camera center, so a real turn
also translates the camera by a few cm; for a 5 deg turn that is about 2 px of
parallax at 1 m on the 672x376 image.
"""

from __future__ import annotations

import math

import cv2
import numpy as np


ZED_MINI_NATIVE_WH = (672, 376)
ZED_MINI_FX = 359.222
ZED_MINI_FY = 359.222
ZED_MINI_CX = 324.324
ZED_MINI_CY = 187.275
FALLBACK_UP = np.array([0.0, -1.0, 0.0], dtype=np.float64)

# G1 29-DoF body_q order: 0-5 left leg, 6-11 right leg, 12 waist yaw.
LEG_JOINTS = slice(0, 12)


def intrinsics(width: int, height: int) -> np.ndarray:
    """ZED Mini left intrinsics scaled to a ``width x height`` image."""

    sx = float(width) / ZED_MINI_NATIVE_WH[0]
    sy = float(height) / ZED_MINI_NATIVE_WH[1]
    return np.array(
        [[ZED_MINI_FX * sx, 0.0, ZED_MINI_CX * sx], [0.0, ZED_MINI_FY * sy, ZED_MINI_CY * sy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def up_axis_in_camera(c2w: np.ndarray, min_abs_y: float = 0.5) -> tuple[np.ndarray, bool]:
    """World-up in camera coordinates from a floor-aligned camera-to-world pose.

    Returns ``(u, valid)``.  Invalid or implausible poses (non-finite, not a
    rotation, camera tilted more than 60 deg from level) fall back to the camera's
    own vertical axis.  The camera is never upside down, so ``u`` is flipped to
    point to image-up (-y) if the world z axis points down.
    """

    pose = np.asarray(c2w, dtype=np.float64).reshape(4, 4)
    rot = pose[:3, :3]
    if not np.all(np.isfinite(rot)) or abs(np.linalg.det(rot) - 1.0) > 0.05:
        return FALLBACK_UP.copy(), False
    u = rot.T @ np.array([0.0, 0.0, 1.0])
    norm = float(np.linalg.norm(u))
    if norm < 1e-6:
        return FALLBACK_UP.copy(), False
    u = u / norm
    if u[1] > 0:
        u = -u
    if abs(u[1]) < min_abs_y:
        return FALLBACK_UP.copy(), False
    return u, True


def rotation_about_axis(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues rotation matrix (right-handed) about a unit axis."""

    x, y, z = (float(v) for v in axis)
    k = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + math.sin(angle) * k + (1.0 - math.cos(angle)) * (k @ k)


def scene_rotation_homography(width: int, height: int, up: np.ndarray, angle: float) -> np.ndarray:
    """Pixel map ``x' ~ H x`` for a scene rotation of ``angle`` rad about ``up``."""

    k = intrinsics(width, height)
    return k @ rotation_about_axis(up, angle) @ np.linalg.inv(k)


def warp_scene_rotation(frame: np.ndarray, up: np.ndarray, angle: float) -> np.ndarray:
    """Counterfactual frame for a scene rotation; borders replicate edge pixels."""

    if angle == 0.0:
        return frame
    h, w = frame.shape[:2]
    hom = scene_rotation_homography(w, h, up, angle)
    return cv2.warpPerspective(
        np.ascontiguousarray(frame), hom, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )


def center_crop_resize(frame: np.ndarray, crop_frac: float) -> np.ndarray:
    """Remove ``crop_frac`` of the width and height on every side, resize back."""

    if crop_frac <= 0.0:
        return frame
    h, w = frame.shape[:2]
    dx, dy = int(round(w * crop_frac)), int(round(h * crop_frac))
    cropped = np.ascontiguousarray(frame[dy : h - dy, dx : w - dx])
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of wxyz quaternions, broadcasting over leading dims."""

    aw, ax, ay, az = np.moveaxis(np.asarray(a, dtype=np.float64), -1, 0)
    bw, bx, by, bz = np.moveaxis(np.asarray(b, dtype=np.float64), -1, 0)
    return np.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=-1,
    )


def yaw_quat(angle: float) -> np.ndarray:
    return np.array([math.cos(angle / 2.0), 0.0, 0.0, math.sin(angle / 2.0)])


def rotate_quat_yaw(q: np.ndarray, angle: float) -> np.ndarray:
    """World-frame yaw rotation of wxyz quaternions (shape ``(..., 4)``)."""

    out = quat_mul(yaw_quat(angle), q)
    return out.astype(np.asarray(q).dtype, copy=False)


def quat_yaw(q: np.ndarray) -> np.ndarray:
    """Yaw (rad, z-up) of wxyz quaternions; unnormalized input is normalized."""

    q = np.asarray(q, dtype=np.float64)
    q = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-9)
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def wrap_angle(a: np.ndarray) -> np.ndarray:
    return (np.asarray(a) + np.pi) % (2.0 * np.pi) - np.pi


def is_walking(body_q_chunk: np.ndarray, threshold: float) -> bool:
    """A chunk is locomotion if any leg joint moves more than ``threshold`` rad."""

    legs = np.asarray(body_q_chunk)[:, LEG_JOINTS]
    return bool(np.max(np.ptp(legs, axis=0)) > threshold)
