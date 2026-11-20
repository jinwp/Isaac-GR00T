#!/bin/bash -l
# Experiment 1, arm (a) control: same continuation (LR, crop, steps), no rotation counterfactuals.
# Continues the 20k run (v2) from checkpoint-20000 for 2,000 steps on 4 A40 GPUs
# (global batch 256, LR 2e-5 constant after 100 warmup steps, full fine-tuning).
# Arms run one after another: a -> b -> c (each submits the next when done).

#SBATCH --job-name=cf1_a
#SBATCH --partition=ha-lab
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=a40:4
#SBATCH --cpus-per-task=28
#SBATCH --mem=240G
#SBATCH --time=03:55:00
#SBATCH --output=/coc/flash2/jpark984/gr00t_baseline/logs/cf_exp1_a-%j.out
#SBATCH --error=/coc/flash2/jpark984/gr00t_baseline/logs/cf_exp1_a-%j.err

export HALAB_EXPERIMENT=halab_cf_a_control_lr2e5_2k
export HALAB_SUBMIT_SCRIPT=/coc/flash2/jpark984/gr00t_baseline/Isaac-GR00T/jobs/cf_exp1_a_control.sbatch
export HALAB_NEXT_SUBMIT_SCRIPT="/coc/flash2/jpark984/gr00t_baseline/Isaac-GR00T/jobs/cf_exp1_b_rot.sbatch"
export HALAB_INIT_CHECKPOINT=/coc/flash2/jpark984/gr00t_baseline/outputs/halab_g1_47d_openhlm_b64_g256_lr1e4_v2/checkpoint-20000
export HALAB_STATS_PATH=/coc/flash2/jpark984/gr00t_baseline/outputs/halab_g1_47d_openhlm_b64_g256_lr1e4_v2/train_statistics.json
export HALAB_NUM_GPUS=4
export HALAB_GLOBAL_BATCH=256
export HALAB_LR=2e-5
export HALAB_LR_SCHEDULER=constant_with_warmup
export HALAB_WARMUP_STEPS=100
export HALAB_MAX_STEPS=2000
export HALAB_DATALOADER_WORKERS=4
export HALAB_MASTER_PORT=29710  # unique vs the v3 chains (29704/29705)
export HALAB_CF_FRAC=0.0
export HALAB_CF_STD_DEG=4.0
export HALAB_CF_MAX_DEG=7.0
export HALAB_CROP_FRAC=0.08
export HALAB_MAX_NO_PROGRESS=3
export HALAB_FK_MJCF="${HALAB_FK_MJCF:-/coc/flash2/dscho1234/Workspace/OpenHLM/thirdparty/MotionTokenizers/assets/g1_with_inspire_hand/g1_29dof_rev_1_0_with_inspire_hand_DFQ.xml}"

cd /coc/flash2/jpark984/gr00t_baseline/Isaac-GR00T
bash jobs/halab_cf_chain_runner.sh
