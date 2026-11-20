#!/usr/bin/env python3
"""Fine-tune GR00T N1.7 directly from the HA-Lab LeRobotVLADataset."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import time

import torch
import torch.distributed as dist
from transformers import TrainerCallback

from gr00t.configs.base_config import get_default_config
from gr00t.data.dataset.factory import DatasetFactory
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.experiment.experiment import run
from gr00t.experiment.trainer import Gr00tTrainer

from halab_gr00t_adapter import (
    HalabGr00tEvalDataset,
    HalabGr00tTrainingDataset,
    build_halab_dataset,
    build_openhlm_halab_episode_split,
    load_or_compute_statistics,
)
import halab_metrics


def load_config_module(path: Path) -> None:
    spec = importlib.util.spec_from_file_location("halab_47d_config", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import modality config {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model-path", default="nvidia/GR00T-N1.7-3B")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset-root", type=Path)
    source.add_argument("--data-parent", type=Path)
    parser.add_argument(
        "--task-name",
        action="append",
        default=[],
        help="Optional task filter for --data-parent; repeat for multiple tasks.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--statistics-path",
        type=Path,
        required=True,
        help="Cached HA normalization statistics JSON.",
    )
    parser.add_argument(
        "--modality-config-path",
        type=Path,
        default=Path("examples/HALAB/halab_47d_config.py"),
    )
    parser.add_argument("--episode-index", type=int, action="append", default=[])
    parser.add_argument("--max-anchors-per-episode", type=int, default=0)
    parser.add_argument(
        "--openhlm-episode-split",
        action="store_true",
        help="Use OpenHLM's global deterministic HALAB 95/5 episode split.",
    )
    parser.add_argument("--val-episode-fraction", type=float, default=0.05)
    parser.add_argument("--val-split-seed", type=int, default=7)
    parser.add_argument(
        "--episode-split-path",
        type=Path,
        help="Path for the exact train/validation episode-key manifest.",
    )
    parser.add_argument(
        "--max-eval-samples",
        type=int,
        default=0,
        help="Cap evaluation anchors deterministically; 0 evaluates every validation anchor.",
    )
    parser.add_argument(
        "--train-eval-samples",
        type=int,
        default=256,
        help="Fixed train anchors evaluated alongside validation (eval/train/*); 0 disables.",
    )
    parser.add_argument(
        "--no-ode-eval",
        action="store_true",
        help="Skip ODE-sampled (get_action) metrics during evaluation.",
    )
    parser.add_argument("--max-stat-samples", type=int, default=2048)
    parser.add_argument(
        "--statistics-only",
        action="store_true",
        help="Compute/cache HA normalization statistics and exit before model loading.",
    )
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--global-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--lr-scheduler-type",
        default="cosine",
        help="HF scheduler name, e.g. cosine (GR00T default) or constant_with_warmup.",
    )
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=0,
        help="Absolute warmup steps; overrides --warmup-ratio when > 0.",
    )
    parser.add_argument("--max-steps", type=int)
    parser.add_argument(
        "--epochs",
        type=float,
        help="Train-set passes. Derives max_steps from the actual selected train split.",
    )
    parser.add_argument("--save-steps", type=int, default=20)
    parser.add_argument("--eval-steps", type=int, default=0)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--dataloader-num-workers", type=int, default=2)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume this output directory from its latest checkpoint.",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=Path,
        help="Resume from this exact complete checkpoint directory.",
    )
    parser.add_argument("--tune-llm", action="store_true")
    parser.add_argument("--tune-visual", action="store_true")
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", default="halab-gr00t-n1d7")
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument(
        "--leg-seconds",
        type=float,
        default=0,
        help="Save a checkpoint and exit cleanly this many seconds after launch "
        "(Slurm chain legs); 0 disables.",
    )
    parser.add_argument(
        "--keep-every-steps",
        type=int,
        default=0,
        help="Copy every N-th checkpoint (model only) to <output-dir>/keep-<step>, out of "
        "save_total_limit rotation, for testing during training; 0 disables.",
    )
    parser.add_argument(
        "--state-dropout-prob",
        type=float,
        default=0.0,
        help="Zero the whole state input with this probability in training "
        "(GR00T FinetuneConfig default 0.2; 0.0 = the v2 runs).",
    )
    parser.add_argument(
        "--color-jitter",
        action="store_true",
        help="Train-time color jitter with the NVIDIA fine-tune guide values.",
    )
    parser.add_argument(
        "--train-time-rtc",
        action="store_true",
        help="Training-time RTC as OpenHLM ha_pi05_full (halab_rtc.py): a clean committed prefix of "
        "0..max_delay-1 rows per sample, loss on the suffix; also logs ode_rtc/* in evaluation.",
    )
    parser.add_argument("--rtc-max-delay", type=int, default=10, help="OpenHLM rtc_max_delay (prefix 0..N-1 rows).")
    parser.add_argument(
        "--deepspeed-stage",
        type=int,
        choices=(2, 3),
        default=2,
        help="DeepSpeed ZeRO stage (GR00T default 2; 3 also shards the weights).",
    )
    return parser.parse_args()


def configure_wandb_resume(output_dir: Path) -> None:
    """Reuse one wandb run across Slurm chain legs (rank 0 owns wandb)."""

    if int(os.environ.get("RANK", "0")) != 0:
        return
    import wandb

    id_file = output_dir / "wandb_id.txt"
    if id_file.is_file():
        run_id = id_file.read_text(encoding="utf-8").strip()
    else:
        run_id = wandb.util.generate_id()
        output_dir.mkdir(parents=True, exist_ok=True)
        id_file.write_text(run_id + "\n", encoding="utf-8")
    # GR00T calls wandb.init without an id; wandb reads these instead.
    os.environ["WANDB_RUN_ID"] = run_id
    os.environ["WANDB_RESUME"] = "allow"


class LegDeadlineCallback(TrainerCallback):
    """Checkpoint and stop once the leg's time budget is spent.

    The decision is all-reduced every step so all ranks save together; a
    rank-local decision would deadlock the collective checkpoint save.
    """

    def __init__(self, deadline: float) -> None:
        self.deadline = deadline

    def on_step_end(self, args, state, control, **kwargs):
        expired = torch.tensor([float(time.monotonic() >= self.deadline)], device=args.device)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(expired, op=dist.ReduceOp.MAX)
        if expired.item() > 0:
            print(f"Leg time budget spent at step {state.global_step}; saving and exiting.", flush=True)
            control.should_save = True
            control.should_training_stop = True
        return control


def install_leg_deadline(seconds: float) -> None:
    deadline = time.monotonic() + seconds
    original_init = Gr00tTrainer.__init__

    def __init__(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.add_callback(LegDeadlineCallback(deadline))

    Gr00tTrainer.__init__ = __init__


# Optimizer / DeepSpeed state is only needed to resume; inference loads the rest.
KEEP_IGNORE = shutil.ignore_patterns(
    "global_step*", "latest", "optimizer*", "scheduler.pt", "rng_state*", "zero_to_fp32.py"
)


class KeepCheckpointCallback(TrainerCallback):
    """Copy every ``every``-th checkpoint, model only, to ``<run>/keep-<step>``.

    HF rotation keeps only the newest ``save_total_limit`` ``checkpoint-*`` dirs, so a
    10k checkpoint is gone a few periodic saves later.  ``keep-*`` is outside that glob
    and the chain runner's resume scan, and sits next to ``episode_split.json`` so
    kinematic play / the server load it like any checkpoint.
    """

    def __init__(self, every: int) -> None:
        self.every = every

    def _keep(self, args, state) -> None:
        step = state.global_step
        if not state.is_world_process_zero or step <= 0 or step % self.every:
            return
        output_dir = Path(args.output_dir)
        source = output_dir / f"checkpoint-{step}"
        target = output_dir / f"keep-{step}"
        if target.is_dir() or not (source / "trainer_state.json").is_file():
            return
        partial = output_dir / f".keep-{step}.partial"
        shutil.rmtree(partial, ignore_errors=True)
        shutil.copytree(source, partial, ignore=KEEP_IGNORE)
        # This on_save runs before CheckpointFormatCallback's, so add what it copies into a checkpoint.
        shutil.copytree(output_dir / "experiment_cfg", partial / "experiment_cfg", dirs_exist_ok=True)
        shutil.copytree(output_dir / "processor", partial, dirs_exist_ok=True)
        partial.rename(target)
        print(f"Kept {source.name} (model only) as {target}", flush=True)

    def on_train_begin(self, args, state, control, **kwargs):
        # A leg that died between the save and the copy resumes from that checkpoint.
        self._keep(args, state)

    def on_save(self, args, state, control, **kwargs):
        self._keep(args, state)


def install_keep_checkpoints(every: int) -> None:
    original_init = Gr00tTrainer.__init__

    def __init__(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.add_callback(KeepCheckpointCallback(every))

    Gr00tTrainer.__init__ = __init__


def install_exact_resume(checkpoint: Path) -> None:
    """Resume from ``checkpoint`` instead of HF's newest-numbered directory."""

    checkpoint = checkpoint.resolve()
    if not (checkpoint / "trainer_state.json").is_file():
        raise FileNotFoundError(f"Incomplete checkpoint (no trainer_state.json): {checkpoint}")
    original_train = Gr00tTrainer.train

    def train(self, resume_from_checkpoint=None, **kwargs):
        if resume_from_checkpoint is True:
            resume_from_checkpoint = str(checkpoint)
        return original_train(self, resume_from_checkpoint=resume_from_checkpoint, **kwargs)

    Gr00tTrainer.train = train


def install_action_horizon(horizon: int) -> None:
    """Load the base model with ``action_horizon`` raised to ``horizon``.

    The N1.7 base checkpoint is saved with ``action_horizon=40``, and GR00T's setup
    builds it with ``AutoModel.from_pretrained`` without that field, so a longer chunk
    is passed as a config override here.  No weight depends on it: the action
    position embedding has ``max_seq_len`` (1024) rows.
    """

    from transformers import AutoModel
    import gr00t.model.gr00t_n1d7.setup as n1d7_setup

    class _HorizonAutoModel:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            kwargs.setdefault("action_horizon", horizon)
            out = AutoModel.from_pretrained(*args, **kwargs)
            model = out[0] if isinstance(out, tuple) else out
            if model.config.action_horizon != horizon:
                raise RuntimeError(f"model action_horizon {model.config.action_horizon} != {horizon}")
            print(f"Model action_horizon set to {horizon}", flush=True)
            return out

    n1d7_setup.AutoModel = _HorizonAutoModel


def main() -> None:
    args = parse_args()
    if args.leg_seconds > 0:
        # Clock starts at launch, so model loading counts against the budget.
        install_leg_deadline(args.leg_seconds)
    if args.keep_every_steps > 0:
        install_keep_checkpoints(args.keep_every_steps)
    load_config_module(args.modality_config_path.resolve())

    episodes = args.episode_index or None
    max_anchors = args.max_anchors_per_episode or None
    if args.max_steps is not None and args.epochs is not None:
        raise ValueError("Use one of --max-steps or --epochs, not both.")
    if args.resume and args.resume_from_checkpoint is not None:
        raise ValueError("Use one of --resume or --resume-from-checkpoint, not both.")
    source_kwargs = {
        "dataset_root": args.dataset_root,
        "data_parent": args.data_parent,
        "task_names": args.task_name or None,
    }

    if args.openhlm_episode_split and episodes is not None:
        raise ValueError(
            "--openhlm-episode-split uses all discovered episodes; do not combine it "
            "with --episode-index."
        )
    if args.openhlm_episode_split:
        train_episode_keys, val_episode_keys = build_openhlm_halab_episode_split(
            **source_kwargs,
            fraction=args.val_episode_fraction,
            seed=args.val_split_seed,
        )
        split_manifest = {
            "scheme": "openhlm_global_episode_split",
            "fraction": args.val_episode_fraction,
            "seed": args.val_split_seed,
            "train_keys": train_episode_keys,
            "val_keys": val_episode_keys,
        }
        split_path = args.episode_split_path or (args.output_dir / "episode_split.json")
        if int(os.environ.get("RANK", "0")) == 0:
            split_path.parent.mkdir(parents=True, exist_ok=True)
            split_path.write_text(json.dumps(split_manifest, indent=2) + "\n", encoding="utf-8")
        print(
            "OpenHLM episode split: "
            f"train={len(train_episode_keys)} val={len(val_episode_keys)} "
            f"fraction={args.val_episode_fraction} seed={args.val_split_seed}",
            flush=True,
        )
    else:
        train_episode_keys = None
        val_episode_keys: list[str] = []

    # Statistics do not need image decoding. They still use the HA loader, so
    # hand remapping and action semantics exactly match training.
    stats_source = build_halab_dataset(
        **source_kwargs,
        episodes=episodes,
        episode_keys=train_episode_keys,
        max_anchors_per_episode=max_anchors,
        load_images=False,
    )
    effective_batch_size = args.global_batch_size * args.gradient_accumulation_steps
    steps_per_epoch = math.ceil(len(stats_source) / effective_batch_size)
    if args.epochs is not None:
        if args.epochs <= 0:
            raise ValueError("--epochs must be positive")
        max_steps = math.ceil(args.epochs * steps_per_epoch)
    elif args.max_steps is not None:
        if args.max_steps <= 0:
            raise ValueError("--max-steps must be positive")
        max_steps = args.max_steps
    elif args.statistics_only:
        max_steps = 0
    else:
        raise ValueError("Training requires --epochs or --max-steps")
    print(
        "Training schedule: "
        f"train_samples={len(stats_source)} "
        f"effective_batch={effective_batch_size} "
        f"steps_per_epoch={steps_per_epoch} "
        f"epochs={args.epochs} max_steps={max_steps}",
        flush=True,
    )
    statistics = load_or_compute_statistics(
        stats_source,
        args.statistics_path,
        max_samples=args.max_stat_samples,
    )
    if args.statistics_only:
        return

    embodiment = EmbodimentTag.NEW_EMBODIMENT.value
    metrics = halab_metrics.HalabMetrics(
        samples_per_step=effective_batch_size,
        train_anchors=len(stats_source),
        ode_eval=not args.no_ode_eval,
    )
    halab_metrics.install(metrics)
    if args.train_time_rtc:
        import halab_rtc

        halab_rtc.install(max_delay=args.rtc_max_delay)
        # OpenHLM evaluates the RTC sampler at the longest delay training draws.
        metrics.rtc_eval_delay = args.rtc_max_delay - 1
    if args.use_wandb:
        configure_wandb_resume(args.output_dir)

    original_build = DatasetFactory.build

    def build_halab(self: DatasetFactory, processor):
        del self
        raw_dataset = build_halab_dataset(
            **source_kwargs,
            episodes=episodes,
            episode_keys=train_episode_keys,
            max_anchors_per_episode=max_anchors,
            load_images=True,
        )
        train_dataset = HalabGr00tTrainingDataset(raw_dataset, processor, statistics, seed=42)
        # Physical-unit scale from the exact normalization the processor applies.
        use_percentiles = processor.state_action_processor.use_percentiles
        metrics.scale = halab_metrics.action_scale(statistics, embodiment, use_percentiles)
        if not args.no_ode_eval:
            # FK un-normalizes body_q exactly like the processor (q01/q99 or min/max).
            lo_key, hi_key = ("q01", "q99") if use_percentiles else ("min", "max")
            body_stats = statistics[embodiment]["action"]["body_q"]
            metrics.build_fk(body_stats[lo_key], body_stats[hi_key])
        eval_dataset = None
        if val_episode_keys:
            raw_eval_dataset = build_halab_dataset(
                **source_kwargs,
                episode_keys=val_episode_keys,
                max_anchors_per_episode=max_anchors,
                load_images=True,
            )
            # Named splits: HF logs them as eval_val_* / eval_train_*.
            eval_dataset = {
                "val": HalabGr00tEvalDataset(
                    raw_eval_dataset,
                    processor,
                    statistics,
                    max_samples=args.max_eval_samples or None,
                )
            }
            if args.train_eval_samples > 0:
                eval_dataset["train"] = HalabGr00tEvalDataset(
                    raw_dataset,
                    processor,
                    statistics,
                    max_samples=args.train_eval_samples,
                )
        return train_dataset, eval_dataset

    DatasetFactory.build = build_halab
    try:
        source_path = args.dataset_root if args.dataset_root is not None else args.data_parent
        config = get_default_config().load_dict(
            {
                "data": {
                    "datasets": [
                        {
                            "dataset_paths": [str(source_path)],
                            "mix_ratio": 1.0,
                            "embodiment_tag": embodiment,
                            "dataset_type": "halab_loader",
                        }
                    ]
                }
            }
        )
        config.load_config_path = None

        config.model.tune_llm = args.tune_llm
        config.model.tune_visual = args.tune_visual
        config.model.tune_projector = True
        config.model.tune_diffusion_model = True
        config.model.state_dropout_prob = args.state_dropout_prob
        # NVIDIA new-embodiment fine-tune guide (getting_started/finetune_new_embodiment.md).
        config.model.color_jitter_params = (
            {"brightness": 0.3, "contrast": 0.4, "saturation": 0.5, "hue": 0.08} if args.color_jitter else None
        )
        config.model.load_bf16 = False
        config.model.reproject_vision = False
        config.model.model_name = "nvidia/Cosmos-Reason2-2B"
        config.model.backbone_trainable_params_fp32 = True
        config.model.use_relative_action = False
        horizon = halab_metrics.ACTION_HORIZON  # HALAB_ACTION_HORIZON, as the loader and modality config
        if horizon > config.model.action_horizon:  # the base model's 40 covers the 32-row runs unchanged
            config.model.action_horizon = horizon  # processor padding / action_mask width
            install_action_horizon(horizon)

        config.training.start_from_checkpoint = args.base_model_path
        config.training.optim = "adamw_torch"
        config.training.num_gpus = args.num_gpus
        config.training.global_batch_size = args.global_batch_size
        config.training.gradient_accumulation_steps = args.gradient_accumulation_steps
        config.training.learning_rate = args.learning_rate
        config.training.lr_scheduler_type = args.lr_scheduler_type
        config.training.warmup_ratio = args.warmup_ratio
        config.training.warmup_steps = args.warmup_steps
        config.training.max_steps = max_steps
        config.training.save_steps = args.save_steps
        config.training.save_total_limit = 2
        config.training.dataloader_num_workers = args.dataloader_num_workers
        config.training.gradient_checkpointing = args.gradient_checkpointing
        config.training.eval_strategy = "steps" if val_episode_keys else "no"
        config.training.eval_steps = args.eval_steps or args.save_steps
        config.training.eval_batch_size = args.eval_batch_size
        config.training.save_best_eval_metric_name = "eval_val_loss" if val_episode_keys else ""
        config.training.save_best_eval_metric_greater_is_better = False
        config.training.output_dir = str(args.output_dir)
        config.training.experiment_name = None
        config.training.use_wandb = args.use_wandb
        config.training.wandb_project = args.wandb_project
        config.training.logging_steps = args.logging_steps
        config.training.deepspeed_stage = args.deepspeed_stage
        # TrainingConfig.resume_from_checkpoint is typed bool: OmegaConf rejects a
        # path when GR00T serializes the config. Keep it a bool here and hand
        # the exact directory to Gr00tTrainer.train, which accepts a path.
        config.training.resume_from_checkpoint = bool(
            args.resume or args.resume_from_checkpoint is not None
        )
        if args.resume_from_checkpoint is not None:
            install_exact_resume(args.resume_from_checkpoint)

        run(config)
    finally:
        DatasetFactory.build = original_build


if __name__ == "__main__":
    main()
