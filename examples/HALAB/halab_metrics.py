#!/usr/bin/env python3
"""OpenHLM-parity action metrics for the HA-Lab GR00T N1.7 run.

Mirrors what OpenHLM's ``scripts/train.py`` logs for HA-pi05, adapted to
GR00T's flow convention and to ``Gr00tTrainer`` (HF ``Trainer``):

* train (every logging step, mean over the steps since the last log):
  ``fm_loss``, the one-step x0 estimate ``action_{mae,mse}_{norm,physical}``,
  per joint group ``mae_physical/<group>`` and per horizon window
  ``action_mae_physical_<window>``.
* eval, per split (``val`` = held-out episodes, ``train`` = a fixed train
  subset with eval transforms): the same one-step metrics, plus the ODE
  sampler (``get_action``, GR00T's inference path, fresh noise) as
  ``ode/action_{mae,mse}_{norm,denorm}``, ``ode/mae_denorm/<group>`` and
  ``ode/action_mae_denorm_<window>``, the G1 forward-kinematics position
  errors ``ode/fk/<name>_cm`` (``halab_fk.py``, OpenHLM ``g1_fk.py``) and
  ``ode/sample_s``.
* also as in OpenHLM: ``param_norm``, ``time/data_s``, host memory
  ``mem/*``.  Not ported: ``camera_views`` (decoding video in the main
  process before the fork-started DataLoader workers can hang them).

Physical units: GR00T normalizes absolute actions to [-1, 1] with q01/q99
(``use_percentiles``) or min/max, so ``x_phys = x_norm * (hi - lo) / 2 + c``;
MAE scales linearly and MSE quadratically, exactly as OpenHLM's
``_action_scale``.  Values clipped by GR00T's ``clip_outliers`` stay clipped.

GR00T's flow is ``x_t = (1 - t) * noise + t * a`` with target velocity
``v = a - noise``, so the one-step estimate ``a_hat = x_t + (1 - t) * v_pred``
has error ``(1 - t) * (v_pred - v)``.  The action head returns the masked
squared velocity error, so only ``t`` must be recorded (``sample_time``).
"""

from __future__ import annotations

import logging
import os
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

from gr00t.experiment.trainer import Gr00tTrainer
from gr00t.experiment.utils import BestMetricCheckpointCallback
from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7ActionHead

import halab_fk


logger = logging.getLogger(__name__)


ACTION_KEYS = ("body_q", "base_quat", "inspire_hand", "camera")
ACTION_DIM = 47
ACTION_HORIZON = int(os.environ.get("HALAB_ACTION_HORIZON", "32"))  # as halab_47d_config
ACTION_STRIDE = int(os.environ.get("HALAB_ACTION_STRIDE", "1"))  # 30 Hz frames per row, as halab_gr00t_adapter

# OpenHLM ``joint_layout.PI_GROUPS`` (names as logged: ``name.split("(")[0]``) as
# indices into our 47D order [body_q29 (G1 order) | base_quat_wxyz4 | hand12 | camera2].
# HA hand order per side (data_loader/ha/hand_action.py), left 6 then right 6:
# thumb_yaw, thumb_pitch, index, middle, ring, pinky.  OpenHLM puts the index
# finger in the pretrained gripper slot and the other five in "hand_rest".
ACTION_GROUPS: tuple[tuple[str, list[int]], ...] = (
    ("left_arm", list(range(15, 22))),
    ("left_gripper_slot", [35]),
    ("right_arm", list(range(22, 29))),
    ("right_gripper_slot", [41]),
    ("left_hand_rest", [33, 34, 36, 37, 38]),
    ("right_hand_rest", [39, 40, 42, 43, 44]),
    ("left_leg", list(range(0, 6))),
    ("right_leg", list(range(6, 12))),
    ("waist", list(range(12, 15))),
    ("camera", [45, 46]),
    ("root_quat", list(range(29, 33))),
)
assert sorted(i for _, idx in ACTION_GROUPS for i in idx) == list(range(ACTION_DIM))

# OpenHLM's HORIZON_WINDOWS in 30 Hz chunk rows (start inclusive, end exclusive).
# A 32-row chunk reaches rows 0..31, so the second window is partial (a 50-row one
# fills it) and the 1.67-3.33 s window is left out, as OpenHLM does for windows a
# chunk misses.
HORIZON_WINDOWS = ((0, 10, "0-0.33s"), (10, 50, "0.33-1.67s"), (50, 100, "1.67-3.33s"))


def action_scale(statistics: dict[str, Any], embodiment: str, use_percentiles: bool) -> np.ndarray:
    """Half-range of GR00T's action normalization, (D,) in 47D model order."""

    lo_key, hi_key = ("q01", "q99") if use_percentiles else ("min", "max")
    action_stats = statistics[embodiment]["action"]
    halves = [
        np.maximum(np.asarray(action_stats[key][hi_key]) - np.asarray(action_stats[key][lo_key]), 1e-8) / 2.0
        for key in ACTION_KEYS
    ]
    scale = np.concatenate(halves).astype(np.float64)
    if scale.shape != (ACTION_DIM,):
        raise ValueError(f"Expected {ACTION_DIM} action scales, got {scale.shape}")
    return scale


def host_memory_gb() -> dict[str, float]:
    """OpenHLM ``_host_memory_gb``: this process's RSS and its (slurm) cgroup."""

    out: dict[str, float] = {}
    try:
        with open("/proc/self/statm") as f:
            out["mem/main_rss_gb"] = int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 1e9
    except Exception:  # noqa: BLE001
        pass
    try:
        with open("/proc/self/cgroup") as f:
            cg = f.read().strip().splitlines()[0].split(":", 2)[2]
        base = f"/sys/fs/cgroup{cg}"
        for key, name in (("mem/cgroup_gb", "memory.current"), ("mem/cgroup_peak_gb", "memory.peak")):
            with open(f"{base}/{name}") as f:
                out[key] = int(f.read()) / 1e9
        d = base
        while d.startswith("/sys/fs/cgroup/"):
            try:
                with open(f"{d}/memory.max") as f:
                    v = f.read().strip()
                if v != "max":
                    out["mem/cgroup_limit_gb"] = int(v) / 1e9
                    break
            except OSError:
                pass
            d = os.path.dirname(d)
    except Exception:  # noqa: BLE001
        pass
    return out


# OpenHLM ``param_norm``: global L2 norm of the kernels (ndim > 1), excluding
# biases / norm scales (1-D here) and the token / position embeddings.
_NOT_KERNEL = re.compile(r"(embed_tokens|pos_embed|position_embedding)\.weight$")


@torch.no_grad()
def param_norm(model: Any) -> float | None:
    total = None
    for name, p in model.named_parameters():
        if hasattr(p, "ds_numel"):
            return None  # ZeRO-3 partitions the weights; a local norm would be wrong
        if p.ndim <= 1 or _NOT_KERNEL.search(name):
            continue
        sq = torch.linalg.vector_norm(p.detach(), 2, dtype=torch.float32).square()
        total = sq if total is None else total + sq
    return None if total is None else float(total.sqrt())


class _ErrorSums:
    """Masked error sums, (D,) per dimension and (T, D) per chunk row."""

    def __init__(self) -> None:
        self.tensors: dict[str, torch.Tensor] | None = None

    def add(self, abs_err: torch.Tensor, mask: torch.Tensor) -> None:
        """``abs_err`` (B, T, D) already zero where invalid; ``mask`` (B, T) valid frames."""

        abs_err = abs_err.detach().float()
        mask = mask.detach().float()
        update = {
            "abs_dim": abs_err.sum(dim=(0, 1)),
            "sq_dim": abs_err.square().sum(dim=(0, 1)),
            "n": mask.sum().reshape(1),
            "abs_row_dim": abs_err.sum(dim=0),
            "n_row": mask.sum(dim=0),
        }
        if self.tensors is None:
            self.tensors = update
        else:
            for key, value in update.items():
                self.tensors[key] += value

    def reduce(self, device: torch.device) -> dict[str, np.ndarray] | None:
        """All-reduce over ranks (collective: every rank must call it) and reset."""

        shapes = {
            "abs_dim": (ACTION_DIM,),
            "sq_dim": (ACTION_DIM,),
            "n": (1,),
            "abs_row_dim": (ACTION_HORIZON, ACTION_DIM),
            "n_row": (ACTION_HORIZON,),
        }
        tensors = self.tensors or {k: torch.zeros(s, device=device) for k, s in shapes.items()}
        flat = torch.cat([tensors[k].to(device).reshape(-1) for k in shapes])
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        self.tensors = None
        out, offset = {}, 0
        values = flat.cpu().double().numpy()
        for key, shape in shapes.items():
            size = int(np.prod(shape))
            out[key] = values[offset : offset + size].reshape(shape)
            offset += size
        return out if out["n"][0] > 0 else None


def _error_metrics(sums: dict[str, np.ndarray], scale: np.ndarray, physical: str) -> dict[str, float]:
    """OpenHLM ``_ode_metrics`` + ``_horizon_window_metrics`` from summed errors."""

    n = sums["n"][0]
    mae_dim, mse_dim = sums["abs_dim"] / n, sums["sq_dim"] / n
    mae_phys, mse_phys = mae_dim * scale, mse_dim * scale**2
    out = {
        "action_mae_norm": float(mae_dim.mean()),
        "action_mse_norm": float(mse_dim.mean()),
        f"action_mae_{physical}": float(mae_phys.mean()),
        f"action_mse_{physical}": float(mse_phys.mean()),
    }
    for name, idx in ACTION_GROUPS:
        out[f"mae_{physical}/{name}"] = float(mae_phys[idx].mean())
    weighted = sums["abs_row_dim"] * scale[None, :]
    # Row j is frame t + N (j + 1), i.e. 30 Hz row (j + 1) N - 1 (= j at 30 Hz): the windows stay in seconds.
    rows = (np.arange(weighted.shape[0]) + 1) * ACTION_STRIDE - 1
    for lo, hi, label in HORIZON_WINDOWS:
        in_window = (rows >= lo) & (rows < hi)
        count = sums["n_row"][in_window].sum()
        if in_window.any() and count > 0:
            out[f"action_mae_{physical}_{label}"] = float(weighted[in_window].sum() / (count * ACTION_DIM))
    return out


class HalabMetrics:
    """Accumulates per-step metrics and patches them into ``Gr00tTrainer``."""

    def __init__(
        self,
        scale: np.ndarray | None = None,
        *,
        samples_per_step: int,
        train_anchors: int,
        ode_eval: bool = True,
    ) -> None:
        # Set from the processor's statistics once GR00T builds the datasets.
        self.scale = scale
        self.samples_per_step = samples_per_step
        self.train_anchors = train_anchors
        self.ode_eval = ode_eval
        self.one_step = _ErrorSums()
        self.ode = _ErrorSums()
        # Training-time RTC (halab_rtc.py): OpenHLM ``ode_rtc/*``, the ODE sampler with the
        # ground-truth prefix pinned for ``rtc_eval_delay`` rows, error on the suffix only.
        self.rtc_eval_delay = 0
        self.ode_rtc = _ErrorSums()
        self.fm_loss_sum = 0.0
        self.fm_loss_count = 0
        self._last_log_time: float | None = None
        self._last_log_step: int | None = None
        self.data_time = 0.0
        self.data_steps = 0
        self.ode_sample_s = 0.0
        # FK task-space metrics; ``fk_built`` decides the collective layout and
        # is the same on every rank, ``fk`` drops to None after a runtime error.
        self.fk: halab_fk.G1FkMetrics | None = None
        self.fk_built = False
        self.fk_sums: dict[str, torch.Tensor] = {}

    def build_fk(self, q01: np.ndarray, q99: np.ndarray) -> None:
        """OpenHLM ``build_task_space``: FK metrics are off when MuJoCo / the MJCF is missing."""

        try:
            self.fk = halab_fk.G1FkMetrics(q01, q99)
            self.fk_built = True
            logger.info("FK task-space metrics on (MJCF %s)", halab_fk.resolve_mjcf_path())
        except Exception as exc:  # noqa: BLE001
            self.fk, self.fk_built = None, False
            logger.warning("FK metrics are off for this run: %s: %s", type(exc).__name__, exc)

    # ------------------------------------------------------------------
    # Accumulation
    # ------------------------------------------------------------------

    @staticmethod
    def _valid_frames(action_mask: torch.Tensor) -> torch.Tensor:
        return action_mask[:, :ACTION_HORIZON, :ACTION_DIM].amax(dim=-1)

    def add_one_step(self, head: Any, outputs: dict[str, Any]) -> None:
        t = getattr(head, "_halab_last_t", None)
        if t is None or "action_loss" not in outputs or "action_mask" not in outputs:
            return
        sq_vel_err = outputs["action_loss"][:, :ACTION_HORIZON, :ACTION_DIM].detach().float()
        mask = outputs["action_mask"][:, :ACTION_HORIZON, :ACTION_DIM]
        one_minus_t = (1.0 - t.detach().float().reshape(-1))[:, None, None]
        self.one_step.add(one_minus_t * sq_vel_err.clamp_min(0).sqrt(), self._valid_frames(mask))
        self.fm_loss_sum += float(outputs["loss"].detach().float())
        self.fm_loss_count += 1

    @torch.no_grad()
    def add_ode(self, model: Any, inputs: dict[str, Any]) -> None:
        batch = inputs["inputs"]
        # Without "action" in the inputs, get_action runs plain flow sampling
        # (with it, GR00T's action head switches to RTC inpainting).
        sample_inputs = {k: v for k, v in batch.items() if k not in ("action", "action_mask")}
        start = time.monotonic()
        pred = model.get_action(sample_inputs)["action_pred"]
        if pred.is_cuda:
            torch.cuda.synchronize(pred.device)
        self.ode_sample_s += time.monotonic() - start
        target = batch["action"].to(pred.device)
        mask = batch["action_mask"].to(pred.device)
        err = (pred.float() - target.float()).abs() * mask.float()
        err = err[:, :ACTION_HORIZON, :ACTION_DIM]
        valid = self._valid_frames(mask)
        self.ode.add(err, valid)
        if self.rtc_eval_delay > 0:
            import halab_rtc

            d = self.rtc_eval_delay
            pred_rtc = model.get_action(sample_inputs, options=halab_rtc.rtc_options(target, d))["action_pred"]
            suffix = (torch.arange(ACTION_HORIZON, device=pred.device) >= d)[None, :]
            err_rtc = (pred_rtc.float() - target.float()).abs() * mask.float()
            self.ode_rtc.add(err_rtc[:, :ACTION_HORIZON, :ACTION_DIM] * suffix[..., None], valid * suffix)
        if self.fk is not None:
            try:
                sums = self.fk.eval_sums(
                    pred[:, :ACTION_HORIZON, :ACTION_DIM], target[:, :ACTION_HORIZON, :ACTION_DIM], valid
                )
                for key, value in sums.items():
                    self.fk_sums[key] = self.fk_sums.get(key, 0.0) + value.detach().double()
            except Exception as exc:  # noqa: BLE001
                # Keep training; the reduction still runs (zeros) so ranks stay in step.
                logger.warning("FK metrics disabled after error: %s: %s", type(exc).__name__, exc)
                self.fk = None

    # ------------------------------------------------------------------
    # Reduction (collective)
    # ------------------------------------------------------------------

    def _reduce_fm_loss(self, device: torch.device) -> float | None:
        totals = torch.tensor([self.fm_loss_sum, float(self.fm_loss_count)], device=device, dtype=torch.float64)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        self.fm_loss_sum, self.fm_loss_count = 0.0, 0
        return float(totals[0] / totals[1]) if totals[1] > 0 else None

    def _reduce_fk(self, device: torch.device) -> dict[str, float]:
        if not self.fk_built:
            return {}
        keys = [f"fk_{kind}/{name}" for name in halab_fk.FK_NAMES for kind in ("sum", "cnt")]
        flat = torch.tensor(
            [float(self.fk_sums.get(k, 0.0)) for k in keys], device=device, dtype=torch.float64
        )
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        self.fk_sums = {}
        return halab_fk.sums_to_cm(dict(zip(keys, flat.cpu().tolist())))

    def add_data_time(self, seconds: float) -> None:
        self.data_time += seconds
        self.data_steps += 1

    def train_logs(self, step: int, device: torch.device, model: Any = None) -> dict[str, float]:
        logs: dict[str, float] = {}
        fm_loss = self._reduce_fm_loss(device)
        if fm_loss is not None:
            logs["fm_loss"] = fm_loss
        sums = self.one_step.reduce(device)
        if sums is not None:
            logs.update(_error_metrics(sums, self.scale, "physical"))
        now = time.monotonic()
        if self._last_log_time is not None and self._last_log_step is not None and step > self._last_log_step:
            step_s = (now - self._last_log_time) / (step - self._last_log_step)
            logs["time/step_s"] = step_s
            logs["time/samples_per_s"] = self.samples_per_step / max(step_s, 1e-9)
        self._last_log_time, self._last_log_step = now, step
        if self.data_steps:
            logs["time/data_s"] = self.data_time / self.data_steps
        self.data_time, self.data_steps = 0.0, 0
        if model is not None:
            norm = param_norm(model)
            if norm is not None:
                logs["param_norm"] = norm
        logs["samples_seen"] = float(step * self.samples_per_step)
        logs["epoch_passes"] = step * self.samples_per_step / max(self.train_anchors, 1)
        logs.update(host_memory_gb())
        return logs

    def eval_logs(self, prefix: str, device: torch.device) -> dict[str, float]:
        logs: dict[str, float] = {}
        fm_loss = self._reduce_fm_loss(device)
        if fm_loss is not None:
            logs[f"{prefix}/fm_loss"] = fm_loss
        sums = self.one_step.reduce(device)
        if sums is not None:
            logs.update({f"{prefix}/{k}": v for k, v in _error_metrics(sums, self.scale, "physical").items()})
        if self.ode_eval:
            sums = self.ode.reduce(device)
            if sums is not None:
                logs.update({f"{prefix}/ode/{k}": v for k, v in _error_metrics(sums, self.scale, "denorm").items()})
            logs.update({f"{prefix}/ode/{k}": v for k, v in self._reduce_fk(device).items()})
            logs[f"{prefix}/ode/sample_s"] = self.ode_sample_s
            if self.rtc_eval_delay > 0:
                sums = self.ode_rtc.reduce(device)
                if sums is not None:
                    logs.update({f"{prefix}/ode_rtc/{k}": v for k, v in _error_metrics(sums, self.scale, "denorm").items()})
                logs[f"{prefix}/ode_rtc/delay"] = float(self.rtc_eval_delay)
        self.ode_sample_s = 0.0
        logs.update({f"{prefix}/{k}": v for k, v in host_memory_gb().items()})
        return logs

    def reset(self) -> None:
        self.one_step.tensors = None
        self.ode.tensors = None
        self.ode_rtc.tensors = None
        self.fm_loss_sum, self.fm_loss_count = 0.0, 0
        self.fk_sums = {}
        self.ode_sample_s = 0.0



def _protect_best_checkpoint(trainer: Any) -> None:
    """Keep GR00T's ``checkpoint-<step>-best-<metric>`` out of HF rotation.

    HF's ``_rotate_checkpoints`` globs every ``checkpoint-*`` directory, orders
    them by mtime and deletes all but ``save_total_limit``; the best copy is
    always older than the next periodic saves, so it would be deleted.  HF only
    spares ``state.best_model_checkpoint``, which GR00T's callback never sets.
    """

    best = sorted(
        (p for p in Path(trainer.args.output_dir).glob("checkpoint-*-best-*") if p.is_dir()),
        key=lambda p: p.stat().st_mtime,
    )
    if best:
        trainer.state.best_model_checkpoint = str(best[-1])


def _restore_best_metric(self, args, state, control, **kwargs):
    """Carry the best value across Slurm legs.

    GR00T's callback keeps it in memory only, so every leg restarted at +/-inf and
    "best" meant "best of the last leg".  The value is in the best directory's name:
    ``checkpoint-<step>-best-<metric>_<value>``.
    """

    values = []
    for path in Path(args.output_dir).glob(f"checkpoint-*-best-{self.metric_name}_*"):
        try:
            values.append(float(path.name.rsplit("_", 1)[1]))
        except ValueError:
            pass
    if values:
        self.best_metric = max(values) if self.greater_is_better else min(values)
    return control


def install(metrics: HalabMetrics) -> None:
    """Patch ``Gr00tTrainer`` / ``Gr00tN1d7ActionHead`` in this process."""

    BestMetricCheckpointCallback.on_train_begin = _restore_best_metric

    original_init = Gr00tTrainer.__init__

    def __init__(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        # Gr00tN1d7.forward(inputs) has no labels and no ``return_loss`` arg, so
        # HF's prediction_step would skip compute_loss during evaluation: no
        # eval_loss, no best checkpoint, no eval metrics.  The model always
        # returns its flow-matching loss, so let HF ask for it.
        self.can_return_loss = True

    Gr00tTrainer.__init__ = __init__

    original_sample_time = Gr00tN1d7ActionHead.sample_time

    def sample_time(self, batch_size, device, dtype):
        t = original_sample_time(self, batch_size, device, dtype)
        self._halab_last_t = t
        return t

    Gr00tN1d7ActionHead.sample_time = sample_time

    original_compute_loss = Gr00tTrainer.compute_loss

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss, outputs = original_compute_loss(
            self, model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
        )
        unwrapped = self.accelerator.unwrap_model(model)
        metrics.add_one_step(unwrapped.action_head, outputs)
        if not model.training and metrics.ode_eval:
            metrics.add_ode(unwrapped, inputs)
        return (loss, outputs) if return_outputs else loss

    Gr00tTrainer.compute_loss = compute_loss

    original_log_save_evaluate = Gr00tTrainer._maybe_log_save_evaluate

    def maybe_log_save_evaluate(self, *args, **kwargs):
        # Same condition as HF's own train log; every rank reaches it together.
        if self.control.should_log and self.state.global_step > self._globalstep_last_logged:
            self.log(metrics.train_logs(self.state.global_step, self.args.device, self.model))
        return original_log_save_evaluate(self, *args, **kwargs)

    Gr00tTrainer._maybe_log_save_evaluate = maybe_log_save_evaluate

    original_get_batch_samples = Gr00tTrainer.get_batch_samples

    def get_batch_samples(self, *args, **kwargs):
        # OpenHLM time/data_s: wait for the next batch, per optimizer step.
        start = time.monotonic()
        result = original_get_batch_samples(self, *args, **kwargs)
        metrics.add_data_time(time.monotonic() - start)
        return result

    Gr00tTrainer.get_batch_samples = get_batch_samples


    original_evaluate = Gr00tTrainer.evaluate

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        target = self.eval_dataset if eval_dataset is None else eval_dataset
        if isinstance(target, dict):
            # HF recurses into evaluate() once per named split.
            return original_evaluate(self, eval_dataset, ignore_keys, metric_key_prefix)
        # Drop train-step leftovers from a partial logging window.
        metrics.reset()
        output = original_evaluate(self, eval_dataset, ignore_keys, metric_key_prefix)
        extra = metrics.eval_logs(metric_key_prefix, self.args.device)
        if extra:
            self.log(extra)
            output.update(extra)
        _protect_best_checkpoint(self)
        return output

    Gr00tTrainer.evaluate = evaluate
