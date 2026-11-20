#!/bin/bash -l
# v4: v2 (20k run) + training-time RTC, nothing else changed, so the RTC effect is isolated.
# RTC as OpenHLM ha_pi05_full (pi0_ha.py): per sample a clean committed prefix of 0..9 rows,
# per-frame flow time, loss on the suffix rows; eval also logs ode_rtc/* (prefix 9 pinned).

#SBATCH --job-name=halab_g1_rtc
#SBATCH --partition=ha-lab
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=a40:4
#SBATCH --cpus-per-task=28
#SBATCH --mem=240G
#SBATCH --time=03:55:00
#SBATCH --output=/coc/flash2/jpark984/gr00t_baseline/halab-g1-rtc-%j.out
#SBATCH --error=/coc/flash2/jpark984/gr00t_baseline/halab-g1-rtc-%j.err

export HALAB_EXPERIMENT=halab_g1_47d_v4_rtc10_b64_g256_lr1e4
export HALAB_SUBMIT_SCRIPT=/coc/flash2/jpark984/gr00t_baseline/Isaac-GR00T/jobs/halab_g1_47d_v4_rtc.sbatch
export HALAB_NUM_GPUS=4
export HALAB_GLOBAL_BATCH=256
export HALAB_ACCUMULATION=1
export HALAB_LR=1e-4
export HALAB_GRADIENT_CHECKPOINTING=1
export HALAB_DATALOADER_WORKERS=4
export HALAB_MAX_STEPS=20000
export HALAB_LR_SCHEDULER=cosine
export HALAB_WARMUP_RATIO=0.05
export HALAB_MASTER_PORT=29706  # unique per concurrent job (may share a node)
export HALAB_STATE_DROPOUT=0.0
export HALAB_COLOR_JITTER=0
export HALAB_TRAIN_RTC=1
export HALAB_RTC_MAX_DELAY=10
export HALAB_MAX_NO_PROGRESS=3
# G1 + Inspire MJCF of MotionTokenizers for the FK (cm) metrics, as in OpenHLM g1_fk.py.
export HALAB_FK_MJCF="${HALAB_FK_MJCF:-/coc/flash2/dscho1234/Workspace/OpenHLM/thirdparty/MotionTokenizers/assets/g1_with_inspire_hand/g1_29dof_rev_1_0_with_inspire_hand_DFQ.xml}"

cd /coc/flash2/jpark984/gr00t_baseline/Isaac-GR00T
bash jobs/halab_chain_runner.sh
