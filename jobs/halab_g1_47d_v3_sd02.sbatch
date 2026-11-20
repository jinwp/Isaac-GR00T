#!/bin/bash -l
# Main v2 run. Official G1+SONIC workflow reference: 20,000 optimizer updates, cosine LR.
# This baseline keeps that step count, but uses the largest verified microbatch:
# 4 A40 GPUs × 64 samples/GPU = real global batch 256.

#SBATCH --job-name=halab_g1_sd02
#SBATCH --partition=ha-lab
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=a40:4
#SBATCH --cpus-per-task=28
#SBATCH --mem=240G
#SBATCH --time=03:55:00
#SBATCH --output=/coc/flash2/jpark984/gr00t_baseline/halab-g1-sd02-%j.out
#SBATCH --error=/coc/flash2/jpark984/gr00t_baseline/halab-g1-sd02-%j.err

# v3 A: v2 + state dropout 0.2 (GR00T FinetuneConfig default; v2 used 0.0).
export HALAB_EXPERIMENT=halab_g1_47d_v3_sd02_b64_g256_lr1e4
export HALAB_SUBMIT_SCRIPT=/coc/flash2/jpark984/gr00t_baseline/Isaac-GR00T/jobs/halab_g1_47d_v3_sd02.sbatch
export HALAB_NUM_GPUS=4
export HALAB_GLOBAL_BATCH=256
export HALAB_ACCUMULATION=1
export HALAB_LR=1e-4
export HALAB_GRADIENT_CHECKPOINTING=1
export HALAB_DATALOADER_WORKERS=4
export HALAB_MAX_STEPS=20000
export HALAB_LR_SCHEDULER=cosine
export HALAB_WARMUP_RATIO=0.05
export HALAB_MASTER_PORT=29704  # unique per concurrent job (may share a node)
export HALAB_STATE_DROPOUT=0.2
export HALAB_COLOR_JITTER=0
export HALAB_MAX_NO_PROGRESS=3
# G1 + Inspire MJCF of MotionTokenizers for the FK (cm) metrics, as in OpenHLM g1_fk.py.
# Empty: try motion_tokenizers.replay.mujoco_replay.resolve_mjcf_path; if that fails,
# FK metrics are skipped with a warning (training is unaffected).
export HALAB_FK_MJCF="${HALAB_FK_MJCF:-/coc/flash2/dscho1234/Workspace/OpenHLM/thirdparty/MotionTokenizers/assets/g1_with_inspire_hand/g1_29dof_rev_1_0_with_inspire_hand_DFQ.xml}"

cd /coc/flash2/jpark984/gr00t_baseline/Isaac-GR00T
bash jobs/halab_chain_runner.sh
