#!/usr/bin/env python3
"""Are the SONIC sim's feet really off the ground, or only in the rendered video?

For every SONIC run under ``--sonic-root`` with a ``sim_qpos_log.npz`` (ground-truth MuJoCo
``qpos`` of the physics sim, logged by OpenHLM ``_sim_loop_with_qpos_log.py``), after the
elastic band was released:

* **physics**: the sim's own MJCF (samsung_deploy ``mujoco_sim/assets/g1_inspire.xml``) at
  each logged ``qpos``: share of ticks with no robot-floor contact, the lowest foot point
  above the floor, pelvis height;
* **render**: the MJCF the overlay / rollout videos are drawn with (MotionTokenizers
  ``g1_29dof_rev_1_0_with_inspire_hand_DFQ.xml``) at the same root and body joints: the lowest
  foot point.  A gap between the two lowest-foot heights is a drawing artifact, not physics.

Run with a python that has mujoco (e.g. the Isaac-GR00T .venv)::

    python halab_sonic_ground_check.py --sonic-root /coc/flash2/jpark984/gr00t_baseline/outputs/checkpoint20k_sonic/val
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np


def _hinge_names(model: mujoco.MjModel) -> list[str]:
    return [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        for j in range(model.njnt)
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
    ]


def _foot_geoms(model: mujoco.MjModel) -> np.ndarray:
    bodies = {
        b for b in range(model.nbody) if "ankle_roll" in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or "")
    }
    return np.array([g for g in range(model.ngeom) if model.geom_bodyid[g] in bodies], dtype=int)


def _lowest_point(model: mujoco.MjModel, data: mujoco.MjData, geoms: np.ndarray) -> float:
    """Lowest z of the foot geoms' oriented bounding boxes (visual meshes included)."""
    low = np.inf
    for g in geoms:
        center_local, half = model.geom_aabb[g, :3], model.geom_aabb[g, 3:]
        rot = data.geom_xmat[g].reshape(3, 3)
        center_z = data.geom_xpos[g, 2] + rot[2] @ center_local
        low = min(low, center_z - np.abs(rot[2]) @ half)
    return float(low)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sonic-root", required=True)
    parser.add_argument("--sim-mjcf", default="/coc/flash2/jpark984/samsung_deploy/mujoco_sim/assets/g1_inspire.xml")
    parser.add_argument(
        "--render-mjcf",
        default="/coc/flash2/dscho1234/Workspace/OpenHLM/thirdparty/MotionTokenizers/assets/g1_with_inspire_hand/"
        "g1_29dof_rev_1_0_with_inspire_hand_DFQ.xml",
    )
    parser.add_argument("--stride", type=int, default=20, help="every N-th 200 Hz sim tick (20 = 10 Hz)")
    parser.add_argument("--max-runs", type=int, default=0)
    args = parser.parse_args()

    sim = mujoco.MjModel.from_xml_path(args.sim_mjcf)
    ren = mujoco.MjModel.from_xml_path(args.render_mjcf)
    sim_d, ren_d = mujoco.MjData(sim), mujoco.MjData(ren)
    floor = mujoco.mj_name2id(sim, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    sim_feet, ren_feet = _foot_geoms(sim), _foot_geoms(ren)
    sim_names = _hinge_names(sim)
    # Render joints fed from the sim's qpos by joint name (body joints; hands left at 0).
    pairs = []
    for name in _hinge_names(ren):
        if name in sim_names:
            pairs.append((ren.jnt_qposadr[mujoco.mj_name2id(ren, mujoco.mjtObj.mjOBJ_JOINT, name)],
                          sim.jnt_qposadr[mujoco.mj_name2id(sim, mujoco.mjtObj.mjOBJ_JOINT, name)]))
    print(f"sim nq={sim.nq}, render nq={ren.nq}, joints matched by name: {len(pairs)}; floor geom id {floor}; "
          f"foot geoms sim {len(sim_feet)} render {len(ren_feet)}")

    logs = sorted(Path(args.sonic_root).rglob("sim_qpos_log.npz"))
    if args.max_runs:
        logs = logs[: args.max_runs]
    if not logs:
        raise SystemExit(f"no sim_qpos_log.npz under {args.sonic_root}")
    print(f"{'run':60} | airborne% | sim foot z med/p95 (cm) | render foot z med/p95 (cm) | pelvis z med/min (m) | fell")
    for path in logs:
        run = path.parent
        npz = np.load(path)
        qpos = np.asarray(npz["qpos"], dtype=np.float64)
        release = int(npz["release_tick_idx"][0]) if "release_tick_idx" in npz.files else 0
        qpos = qpos[release::args.stride]
        if qpos.shape[1] != sim.nq:
            print(f"{str(run)[-60:]:60} | qpos width {qpos.shape[1]} != sim nq {sim.nq}: wrong --sim-mjcf")
            continue
        airborne, sim_low, ren_low = 0, [], []
        for q in qpos:
            sim_d.qpos[:] = q
            mujoco.mj_forward(sim, sim_d)
            touching = any(
                floor in (sim_d.contact[i].geom1, sim_d.contact[i].geom2) for i in range(sim_d.ncon)
            )
            airborne += not touching
            sim_low.append(_lowest_point(sim, sim_d, sim_feet))
            ren_d.qpos[:] = 0
            ren_d.qpos[:7] = q[:7]
            for ra, sa in pairs:
                ren_d.qpos[ra] = q[sa]
            mujoco.mj_forward(ren, ren_d)
            ren_low.append(_lowest_point(ren, ren_d, ren_feet))
        result = json.loads((run / "result.json").read_text()) if (run / "result.json").exists() else {}
        sl, rl = np.array(sim_low) * 100, np.array(ren_low) * 100
        label = "/".join(run.parts[-3:])[-60:]
        print(
            f"{label:60} | {100 * airborne / len(qpos):8.1f} | {np.median(sl):6.1f} / {np.percentile(sl, 95):6.1f} "
            f"         | {np.median(rl):6.1f} / {np.percentile(rl, 95):6.1f}           | "
            f"{np.median(qpos[:, 2]):.3f} / {qpos[:, 2].min():.3f}      | {result.get('fell')}"
        )
    print("\nairborne% ~0 and sim foot z ~0 cm: the physics robot stands on the floor. If render foot z is several cm\n"
          "above sim foot z, the floating feet in the video come from drawing with the MotionTokenizers model.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
