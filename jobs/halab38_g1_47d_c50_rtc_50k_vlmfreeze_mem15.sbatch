#!/bin/bash -l
# VLM-frozen 38-task run (as halab38_g1_47d_c50_rtc_50k_vlmfreeze.sbatch) plus the senior's OpenHLM recipe
# (README_dscho.md halab38_v7_*: --action-fps 15 --temporal mem --mem-drop-all 0.1 --mem-drop-oldest 0.3):
# * action 15 Hz: 50 rows, one every 2 frames (frames t+2 ... t+100 = 3.33 s); RTC delay up to 10 rows (0.67 s).
# * image history: 5 past frames 30 frames (1 s) apart, +-5 frames jitter in training, through GR00T's
#   multi-frame input (6 images in the VLM prompt) instead of MEM's SigLIP temporal attention (the frozen
#   vision encoder could not learn it); frames before the episode start are left out.
# * history dropout (training only, per sample): all past frames with 0.1, else the oldest k ~ U{1..5} with 0.3.
# * no proprio history (and so no start marker).

#SBATCH --job-name=halab38m15
#SBATCH --partition=ha-lab
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=a40:4
#SBATCH --cpus-per-task=28
#SBATCH --mem=240G
#SBATCH --time=03:55:00
#SBATCH --output=/coc/flash2/jpark984/gr00t_baseline/halab38-m15-%j.out
#SBATCH --error=/coc/flash2/jpark984/gr00t_baseline/halab38-m15-%j.err

export HALAB_EXPERIMENT=halab38_g1_47d_c50_rtc10_b64_g256_lr1e4_50k_vlmfreeze_img6_a15hz
export HALAB_SUBMIT_SCRIPT=/coc/flash2/jpark984/gr00t_baseline/Isaac-GR00T/jobs/halab38_g1_47d_c50_rtc_50k_vlmfreeze_mem15.sbatch
export HALAB_DATA_PARENT=/coc/flash2/jpark984/gr00t_baseline/datasets/ha-lab-dataset-38tasks/halab
export HALAB_TUNE_VLM=0
export HALAB_ACTION_HORIZON=50
export HALAB_ACTION_STRIDE=2
export HALAB_IMAGE_HISTORY=5
export HALAB_IMAGE_HISTORY_STRIDE=30
export HALAB_IMAGE_HISTORY_JITTER=5
export HALAB_IMAGE_HISTORY_DROP_ALL=0.1
export HALAB_IMAGE_HISTORY_DROP_OLDEST=0.3
export HALAB_NUM_GPUS=4
export HALAB_GLOBAL_BATCH=256
export HALAB_ACCUMULATION=1
export HALAB_LR=1e-4
export HALAB_GRADIENT_CHECKPOINTING=1
export HALAB_DATALOADER_WORKERS=6  # 6 video reads per sample; 4 x (1 + 6) = the 28 CPUs
export HALAB_MAX_STEPS=50000
export HALAB_LR_SCHEDULER=cosine
export HALAB_WARMUP_RATIO=0.05
export HALAB_KEEP_EVERY_STEPS=10000
export HALAB_MASTER_PORT=29712  # unique per concurrent job (29702-29708, 29710, 29711 taken)
export HALAB_STATE_DROPOUT=0.0
export HALAB_COLOR_JITTER=0
export HALAB_TRAIN_RTC=1
export HALAB_RTC_MAX_DELAY=10
export HALAB_MAX_NO_PROGRESS=3
# G1 + Inspire MJCF of MotionTokenizers for the FK (cm) metrics, as in OpenHLM g1_fk.py.
export HALAB_FK_MJCF="${HALAB_FK_MJCF:-/coc/flash2/dscho1234/Workspace/OpenHLM/thirdparty/MotionTokenizers/assets/g1_with_inspire_hand/g1_29dof_rev_1_0_with_inspire_hand_DFQ.xml}"

cd /coc/flash2/jpark984/gr00t_baseline/Isaac-GR00T
bash jobs/halab_chain_runner.sh
