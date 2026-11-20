#!/usr/bin/env python3
"""G1 forward-kinematics task-space metrics, a torch port of OpenHLM's ``openpi/ha/g1_fk.py``.

Ported 1:1 for the ``mt47`` layout (``G1TaskSpace.eval_sums`` / ``sums_to_cm``): the same
MotionTokenizers G1 + Inspire MJCF, kinematic chains, keypoints, thresholds and metric
definitions, so ``fk/<name>_cm`` is comparable with OpenHLM's ODE eval:

* ``wrist_torso``         wrist position error, arm chain in the torso frame;
* ``wrist_torso_moving``  the same, only while the recorded wrist moves >= 0.10 m/s;
* ``hand``                mean over wrist + index/pinky/thumb knuckles, pelvis frame;
* ``wrist_pelvis``        wrist, pelvis frame (waist + arm);
* ``ankle_pelvis``        ankle, pelvis frame (leg);
* ``foot``                mean over ankle + four sole corners, pelvis frame;
* ``wrist_foot_planted``  wrist seen from each foot (leg + waist + arm), while the
                          recorded feet stay planted.

Only the input differs: GR00T's 47D order ``[body_q29 | quat4 | hand12 | camera2]`` and its
min/max normalization over q01/q99 (``x = (n + 1) / 2 * (q99 - q01) + q01``).
"""

from __future__ import annotations

import math
import os
from typing import Any, Sequence

import numpy as np
import torch


# OpenHLM openpi/ha/joint_layout.py BODY_JOINT_NAMES_29: the HA loader's body_q order.
BODY_JOINT_NAMES_29: tuple[str, ...] = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)
JOINT_INDEX = {name: i for i, name in enumerate(BODY_JOINT_NAMES_29)}  # body_q is dims 0..28

# Constants from OpenHLM g1_fk.py.
MOVING_SPEED_M_S = 0.10
PLANTED_SPEED_M_S = 0.02
PLANTED_WINDOW_FRAMES = 15  # 30 Hz frames
SIDES: tuple[str, ...] = ("left", "right")
ARM_JOINTS: tuple[str, ...] = (
    "shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw",
)
HAND_POINT_BODIES: dict[str, tuple[str, ...]] = {
    "left": ("left_wrist_yaw_link", "L_index_proximal", "L_pinky_proximal", "L_thumb_proximal_base"),
    "right": ("right_wrist_yaw_link", "R_index_proximal", "R_pinky_proximal", "R_thumb_proximal_base"),
}
FOOT_POINT_BODIES: dict[str, tuple[str, ...]] = {
    side: (f"{side}_ankle_roll_link", *(f"{side}_ankle_roll_sphere_{i}_link" for i in (1, 2, 3, 4)))
    for side in SIDES
}
FK_NAMES: tuple[str, ...] = (
    "wrist_torso", "wrist_torso_moving", "hand", "wrist_pelvis", "ankle_pelvis", "foot", "wrist_foot_planted",
)


def resolve_mjcf_path() -> str:
    """``HALAB_FK_MJCF``, else MotionTokenizers' own resolver (what OpenHLM uses)."""

    path = os.environ.get("HALAB_FK_MJCF", "").strip()
    if path:
        return path
    from motion_tokenizers.replay.mujoco_replay import resolve_mjcf_path as mt_resolve

    return str(mt_resolve())


class Chain:
    """Serial hinge chain below ``base`` (MuJoCo conventions) with points fixed to its last body."""

    def __init__(self, joint_names, body_pos, body_rot, axis, points) -> None:
        self.joint_names = tuple(joint_names)
        skew = np.zeros((len(self.joint_names), 3, 3))
        skew[:, 0, 1], skew[:, 0, 2] = -axis[:, 2], axis[:, 1]
        skew[:, 1, 0], skew[:, 1, 2] = axis[:, 2], -axis[:, 0]
        skew[:, 2, 0], skew[:, 2, 1] = -axis[:, 1], axis[:, 0]
        # R_body(q) = body_rot @ (I + sin q K + (1 - cos q) K^2)  (Rodrigues).
        self.body_rot = torch.tensor(body_rot, dtype=torch.float32)
        self.rot_sin = torch.tensor(body_rot @ skew, dtype=torch.float32)
        self.rot_cos = torch.tensor(body_rot @ skew @ skew, dtype=torch.float32)
        self.body_pos = torch.tensor(body_pos, dtype=torch.float32)
        self.points = torch.tensor(points, dtype=torch.float32)
        self.index = torch.tensor([JOINT_INDEX[n] for n in self.joint_names], dtype=torch.long)

    def to(self, device: torch.device) -> "Chain":
        for key in ("body_rot", "rot_sin", "rot_cos", "body_pos", "points", "index"):
            setattr(self, key, getattr(self, key).to(device))
        return self

    def forward(self, q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(..., n)`` rad -> points ``(..., K, 3)``, last-body rotation ``(..., 3, 3)``, origin ``(..., 3)``."""

        sin, cos = torch.sin(q)[..., None, None], torch.cos(q)[..., None, None]
        rot = pos = None
        for i in range(len(self.joint_names)):
            local = self.body_rot[i] + sin[..., i, :, :] * self.rot_sin[i] + (1.0 - cos[..., i, :, :]) * self.rot_cos[i]
            offset = self.body_pos[i]
            if rot is None:
                rot, pos = local, offset.expand(*q.shape[:-1], 3)
            else:
                pos = pos + torch.einsum("...ij,j->...i", rot, offset)
                rot = torch.einsum("...ij,...jk->...ik", rot, local)
        points = pos[..., None, :] + torch.einsum("...ij,kj->...ki", rot, self.points)
        return points, rot, pos


def build_chain(model: Any, base: str, tip: str, point_bodies: Sequence[str]) -> Chain:
    """Same checks and construction as OpenHLM ``g1_fk.build_chain``."""

    import mujoco

    def body_id(name: str) -> int:
        idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if idx < 0:
            raise KeyError(f"MJCF has no body {name!r}")
        return int(idx)

    base_id, tip_id = body_id(base), body_id(tip)
    bodies: list[int] = []
    cur = tip_id
    while cur != base_id:
        if cur == 0:
            raise ValueError(f"{tip!r} is not below {base!r} in the kinematic tree")
        bodies.append(cur)
        cur = int(model.body_parentid[cur])
    bodies.reverse()
    names, axes = [], []
    for b in bodies:
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b)
        if int(model.body_jntnum[b]) != 1:
            raise ValueError(f"chain body {name!r} must carry exactly one joint")
        j = int(model.body_jntadr[b])
        if int(model.jnt_type[j]) != int(mujoco.mjtJoint.mjJNT_HINGE) or np.abs(model.jnt_pos[j]).max() > 0:
            raise ValueError(f"joint of {name!r} must be a hinge through the body origin")
        names.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j))
        axes.append(np.asarray(model.jnt_axis[j], np.float64) / np.linalg.norm(model.jnt_axis[j]))
    rots = np.zeros((len(bodies), 3, 3))
    for i, b in enumerate(bodies):
        mat = np.zeros(9)
        mujoco.mju_quat2Mat(mat, np.asarray(model.body_quat[b], np.float64))
        rots[i] = mat.reshape(3, 3)

    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    tip_rot, tip_pos = data.xmat[tip_id].reshape(3, 3), data.xpos[tip_id]
    points = []
    for name in point_bodies:
        pid = body_id(name)
        cur = pid
        while cur != tip_id:
            if cur == 0:
                raise ValueError(f"point body {name!r} is not attached to {tip!r}")
            n_jnt, adr = int(model.body_jntnum[cur]), int(model.body_jntadr[cur])
            if cur != pid and n_jnt:
                raise ValueError(f"point body {name!r} is not rigidly attached to {tip!r}")
            if n_jnt and np.abs(model.jnt_pos[adr : adr + n_jnt]).max() > 0:
                raise ValueError(f"joint of point body {name!r} does not pass through its origin")
            cur = int(model.body_parentid[cur])
        points.append(tip_rot.T @ (data.xpos[pid] - tip_pos))
    missing = [n for n in names if n not in JOINT_INDEX]
    if missing:
        raise KeyError(f"47D body_q has no dimension for {missing}")
    return Chain(
        names,
        np.asarray(model.body_pos[bodies], np.float64),
        rots,
        np.stack(axes),
        np.asarray(points, np.float64).reshape(len(point_bodies), 3),
    )


class G1FkMetrics:
    """``eval_sums`` of OpenHLM ``G1TaskSpace`` (mt47: lower body present) on GR00T-normalized 47D chunks."""

    def __init__(self, q01: np.ndarray, q99: np.ndarray, *, action_stride: int = 1) -> None:
        import mujoco

        model = mujoco.MjModel.from_xml_path(resolve_mjcf_path())
        self.arm_torso = {s: build_chain(model, "torso_link", f"{s}_wrist_yaw_link", HAND_POINT_BODIES[s]) for s in SIDES}
        self.arm_pelvis = {s: build_chain(model, "pelvis", f"{s}_wrist_yaw_link", HAND_POINT_BODIES[s]) for s in SIDES}
        self.leg = {s: build_chain(model, "pelvis", f"{s}_ankle_roll_link", FOOT_POINT_BODIES[s]) for s in SIDES}
        for side in SIDES:
            expected = tuple(f"{side}_{j}_joint" for j in ARM_JOINTS)
            if self.arm_torso[side].joint_names != expected:
                raise ValueError(f"unexpected arm chain {self.arm_torso[side].joint_names}")
        # GR00T StateActionProcessor min/max over q01/q99 (range floored at 1e-8).
        lo = np.asarray(q01, np.float64)[:29]
        span = np.maximum(np.asarray(q99, np.float64)[:29] - lo, 1e-8)
        self.lo = torch.tensor(lo, dtype=torch.float32)
        self.span = torch.tensor(span, dtype=torch.float32)
        self.action_stride = int(action_stride)
        self.device: torch.device | None = None

    def _to(self, device: torch.device) -> None:
        if self.device == device:
            return
        for chains in (self.arm_torso, self.arm_pelvis, self.leg):
            for chain in chains.values():
                chain.to(device)
        self.lo, self.span = self.lo.to(device), self.span.to(device)
        self.device = device

    def _forward(self, chains, q29: torch.Tensor):
        out = [chains[s].forward(q29[..., chains[s].index]) for s in SIDES]
        points = torch.stack([o[0] for o in out], dim=-3)  # (..., 2, K, 3)
        rot = torch.stack([o[1] for o in out], dim=-3)  # (..., 2, 3, 3)
        pos = torch.stack([o[2] for o in out], dim=-2)  # (..., 2, 3)
        return points, rot, pos

    def _planted(self, foot_rot: torch.Tensor, foot_pos: torch.Tensor) -> torch.Tensor:
        rel = torch.einsum("...ji,...j->...i", foot_rot[..., 1, :, :], foot_pos[..., 0, :] - foot_pos[..., 1, :])
        rows_per_s = 30.0 / self.action_stride
        moving = (torch.linalg.norm(rel[:, 1:] - rel[:, :-1], dim=-1) * rows_per_s >= PLANTED_SPEED_M_S).float()
        half = max(1, math.ceil((PLANTED_WINDOW_FRAMES // 2) / self.action_stride))
        n_rows = rel.shape[1]
        padded = torch.nn.functional.pad(moving, (half, half))
        return sum(padded[:, k : k + n_rows] for k in range(2 * half)) == 0

    @torch.no_grad()
    def eval_sums(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        """``pred`` / ``target`` (B, T, >=29) GR00T-normalized, ``mask`` (B, T) -> ``fk_sum`` / ``fk_cnt`` (m)."""

        self._to(pred.device)
        q_p = (pred[..., :29].float() + 1.0) / 2.0 * self.span + self.lo
        q_g = (target[..., :29].float() + 1.0) / 2.0 * self.span + self.lo
        m = mask.float()[..., None]  # (B, T, 1) over the two sides
        out: dict[str, torch.Tensor] = {}

        def add(name: str, err: torch.Tensor, where: torch.Tensor | None = None) -> None:
            w = m if where is None else m * where.float()
            out[f"fk_sum/{name}"] = (err * w).sum()
            out[f"fk_cnt/{name}"] = torch.broadcast_to(w, err.shape).sum()

        def dist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return torch.linalg.norm(a - b, dim=-1)

        torso_p = self._forward(self.arm_torso, q_p)[0]
        torso_g = self._forward(self.arm_torso, q_g)[0]
        wrist_err = dist(torso_p[..., 0, :], torso_g[..., 0, :])  # (B, T, 2)
        add("wrist_torso", wrist_err)
        step = dist(torso_g[:, 1:, :, 0, :], torso_g[:, :-1, :, 0, :]) * (30.0 / self.action_stride)
        add("wrist_torso_moving", wrist_err, torch.cat([step[:, :1], step], dim=1) >= MOVING_SPEED_M_S)

        hand_p = self._forward(self.arm_pelvis, q_p)[0]
        hand_g = self._forward(self.arm_pelvis, q_g)[0]
        add("hand", dist(hand_p, hand_g).mean(-1))
        add("wrist_pelvis", dist(hand_p[..., 0, :], hand_g[..., 0, :]))
        foot_p, frot_p, fpos_p = self._forward(self.leg, q_p)
        foot_g, frot_g, fpos_g = self._forward(self.leg, q_g)
        add("ankle_pelvis", dist(foot_p[..., 0, :], foot_g[..., 0, :]))
        add("foot", dist(foot_p, foot_g).mean(-1))

        def wrist_in_feet(wrist, frot, fpos):
            rel = wrist[..., :, None, :] - fpos[..., None, :, :]  # (B, T, hand, foot, 3)
            return torch.einsum("...fji,...hfj->...hfi", frot, rel)

        chain_err = dist(
            wrist_in_feet(hand_p[..., 0, :], frot_p, fpos_p), wrist_in_feet(hand_g[..., 0, :], frot_g, fpos_g)
        ).mean(-1)
        add("wrist_foot_planted", chain_err, self._planted(frot_g, fpos_g)[..., None])
        return out


def sums_to_cm(sums: dict[str, float]) -> dict[str, float]:
    """OpenHLM ``sums_to_cm``: ``{"fk/<name>_cm": 100 * sum / count}``, frames-less names left out."""

    out: dict[str, float] = {}
    for name in FK_NAMES:
        count = float(sums.get(f"fk_cnt/{name}", 0.0))
        if count > 0.0:
            out[f"fk/{name}_cm"] = 100.0 * float(sums[f"fk_sum/{name}"]) / count
    return out
