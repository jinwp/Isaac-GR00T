#!/bin/bash -l
# Lab-recipe variant: OpenHLM ha_pi05_full schedule on the same HA-Lab data
# (OpenHLM src/openpi4OpenHLM/src/openpi/training/config.py, "ha_pi05_full"):
# batch 256, 1k warmup -> peak 5e-5 held constant, 30,000 steps.
# Not carried over (GR00T trainer has no hook): EMA 0.999, AdamW betas (0.9, 0.95),
# weight decay 1e-10. GR00T defaults stay: betas (0.9, 0.999), weight decay 1e-5.

#SBATCH --job-name=halab_g1_ohlm
#SBATCH --partition=ha-lab
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=a40:4
#SBATCH --cpus-per-task=28
#SBATCH --mem=240G
#SBATCH --time=03:55:00
#SBATCH --output=/coc/flash2/jpark984/gr00t_baseline/halab-g1-ohlm-%j.out
#SBATCH --error=/coc/flash2/jpark984/gr00t_baseline/halab-g1-ohlm-%j.err

export HALAB_EXPERIMENT=halab_g1_47d_openhlm_recipe_b64_g256_lr5e5_const_30k
export HALAB_SUBMIT_SCRIPT=/coc/flash2/jpark984/gr00t_baseline/Isaac-GR00T/jobs/halab_g1_47d_openhlm_recipe.sbatch
export HALAB_NUM_GPUS=4
export HALAB_GLOBAL_BATCH=256
export HALAB_ACCUMULATION=1
export HALAB_LR=5e-5
export HALAB_GRADIENT_CHECKPOINTING=1
export HALAB_DATALOADER_WORKERS=4
export HALAB_MAX_STEPS=30000
export HALAB_LR_SCHEDULER=constant_with_warmup
export HALAB_WARMUP_STEPS=1000
export HALAB_MASTER_PORT=29703  # unique per concurrent job (may share a node)
export HALAB_MAX_NO_PROGRESS=3
# G1 + Inspire MJCF of MotionTokenizers for the FK (cm) metrics, as in OpenHLM g1_fk.py.
# Empty: try motion_tokenizers.replay.mujoco_replay.resolve_mjcf_path; if that fails,
# FK metrics are skipped with a warning (training is unaffected).
export HALAB_FK_MJCF="${HALAB_FK_MJCF:-}"

cd /coc/flash2/jpark984/gr00t_baseline/Isaac-GR00T
bash jobs/halab_chain_runner.sh
