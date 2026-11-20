"""Render shim for OpenHLM's SONIC overlay video (imported by ``halab_sonic_replay.py``).

OpenHLM ``physics_sim.render_physics_sim_video_from_g1_debug`` passes
``camera_yaw_pitch_orig`` to MotionTokenizers' ``render_overlay_video``; the
MotionTokenizers checkout on the cluster predates that kwarg, so the overlay render
fails with a TypeError.  This wrapper drops kwargs the installed ``render_overlay_video``
does not accept (only the commanded camera arrow is lost) and then calls OpenHLM's
function unchanged.

It must live in its own importable module: ``openpi.ha.viz.isolated.run_isolated``
re-imports the target as ``module:qualname`` in a child process, and this directory is
on the child's PYTHONPATH (the parent's ``sys.path`` is forwarded).
"""

from __future__ import annotations

import functools
import inspect


def _patch_render_overlay_video() -> None:
    from motion_tokenizers.replay import mujoco_replay

    original = mujoco_replay.render_overlay_video
    if getattr(original, "_halab_filtered", False):
        return
    params = inspect.signature(original).parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return

    @functools.wraps(original)
    def filtered(*args, **kwargs):
        return original(*args, **{k: v for k, v in kwargs.items() if k in params})

    filtered._halab_filtered = True
    mujoco_replay.render_overlay_video = filtered


def render_physics_sim_video_from_g1_debug(*args, **kwargs):
    from openpi.ha.viz import physics_sim

    _patch_render_overlay_video()  # physics_sim imports render_overlay_video at call time
    return physics_sim.render_physics_sim_video_from_g1_debug(*args, **kwargs)
