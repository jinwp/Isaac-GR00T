#!/bin/bash -l
# 38-task HA-Lab data (HF sehoonha/ha-lab-dataset rev 702ef02b, downloaded fresh to datasets/ha-lab-dataset-38tasks):
# v5 recipe (50-row chunk + training-time RTC, lr 1e-4 cosine, global batch 256 on 4x A40, no state dropout)
# for 50k steps. Every 10k-step checkpoint is also copied, model only, to <run>/keep-<step> so it survives
# save_total_limit rotation and can be tested while training continues.
# Joint units: this data is mjcf_v1 (robot-A wrists x1.5625, joint 21 URDF-signed), unlike the legacy 12-task runs.

#SBATCH --job-name=halab38c50
#SBATCH --partition=ha-lab
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=a40:4
#SBATCH --cpus-per-task=28
#SBATCH --mem=240G
#SBATCH --time=03:55:00
#SBATCH --output=/coc/flash2/jpark984/gr00t_baseline/halab38-c50-%j.out
#SBATCH --error=/coc/flash2/jpark984/gr00t_baseline/halab38-c50-%j.err

export HALAB_EXPERIMENT=halab38_g1_47d_c50_rtc10_b64_g256_lr1e4_50k
export HALAB_SUBMIT_SCRIPT=/coc/flash2/jpark984/gr00t_baseline/Isaac-GR00T/jobs/halab38_g1_47d_c50_rtc_50k.sbatch
export HALAB_DATA_PARENT=/coc/flash2/jpark984/gr00t_baseline/datasets/ha-lab-dataset-38tasks/halab
export HALAB_ACTION_HORIZON=50
export HALAB_NUM_GPUS=4
export HALAB_GLOBAL_BATCH=256
export HALAB_ACCUMULATION=1
export HALAB_LR=1e-4
export HALAB_GRADIENT_CHECKPOINTING=1
export HALAB_DATALOADER_WORKERS=4
export HALAB_MAX_STEPS=50000
export HALAB_LR_SCHEDULER=cosine
export HALAB_WARMUP_RATIO=0.05
export HALAB_KEEP_EVERY_STEPS=10000
export HALAB_MASTER_PORT=29708  # unique per concurrent job (may share a node)
export HALAB_STATE_DROPOUT=0.0
export HALAB_COLOR_JITTER=0
export HALAB_TRAIN_RTC=1
export HALAB_RTC_MAX_DELAY=10
export HALAB_MAX_NO_PROGRESS=3
# G1 + Inspire MJCF of MotionTokenizers for the FK (cm) metrics, as in OpenHLM g1_fk.py.
export HALAB_FK_MJCF="${HALAB_FK_MJCF:-/coc/flash2/dscho1234/Workspace/OpenHLM/thirdparty/MotionTokenizers/assets/g1_with_inspire_hand/g1_29dof_rev_1_0_with_inspire_hand_DFQ.xml}"

cd /coc/flash2/jpark984/gr00t_baseline/Isaac-GR00T
bash jobs/halab_chain_runner.sh
