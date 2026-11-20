#!/bin/bash -l
# Shared runner for long-running HA-Lab GR00T experiments.

set -uo pipefail

: "${HALAB_EXPERIMENT:?}"
: "${HALAB_SUBMIT_SCRIPT:?}"
: "${HALAB_NUM_GPUS:?}"
: "${HALAB_GLOBAL_BATCH:?}"
: "${HALAB_ACCUMULATION:?}"
: "${HALAB_LR:?}"
: "${HALAB_GRADIENT_CHECKPOINTING:?}"

ROOT=/coc/flash2/jpark984/gr00t_baseline
REPO=${ROOT}/Isaac-GR00T
DATA_PARENT=${HALAB_DATA_PARENT:-${ROOT}/datasets/ha-lab-dataset/halab}
OUTPUT_DIR=${ROOT}/outputs/${HALAB_EXPERIMENT}
STATS_PATH=${OUTPUT_DIR}/train_statistics.json
MAX_STEPS=${HALAB_MAX_STEPS:-20000}
# Each leg checkpoints and exits cleanly after LEG_SECONDS (counted from launch,
# so model loading is included). TIMEOUT_SECONDS is only a hard fallback; both
# stay under the sbatch --time=03:55:00 (14,100 s) limit.
LEG_SECONDS=${HALAB_LEG_SECONDS:-12000}
TIMEOUT_SECONDS=13200
# Periodic saves are a crash safety net only; the leg-end save carries progress.
SAVE_STEPS=${HALAB_SAVE_STEPS:-500}
EVAL_STEPS=1000
MAX_EVAL_SAMPLES=1024
TRAIN_EVAL_SAMPLES=256
WANDB_PROJECT=${HALAB_WANDB_PROJECT:-halab-gr00t-n1d7}
HALAB_DATALOADER_WORKERS=${HALAB_DATALOADER_WORKERS:-4}
CODE_DIR=${HALAB_CODE_DIR:-examples/HALAB}

cd "${REPO}"
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export HF_HOME=${ROOT}/huggingface

latest_step() {
  local step=0 candidate name value
  for candidate in "${OUTPUT_DIR}"/checkpoint-*; do
    [ -d "${candidate}" ] || continue
    name=${candidate##*/}
    value=${name#checkpoint-}
    value=${value%%-*}
    case "${value}" in
      ''|*[!0-9]*) ;;
      *)
        [ "${name}" = "checkpoint-${value}" ] || continue
        [ -f "${candidate}/trainer_state.json" ] || continue
        [ "${value}" -gt "${step}" ] && step=${value}
        ;;
    esac
  done
  printf '%s\n' "${step}"
}

quarantine_incomplete_checkpoints() {
  local candidate name value destination
  for candidate in "${OUTPUT_DIR}"/checkpoint-*; do
    [ -d "${candidate}" ] || continue
    name=${candidate##*/}
    value=${name#checkpoint-}
    case "${value}" in
      ''|*[!0-9]*) continue ;;
    esac
    [ "${name}" = "checkpoint-${value}" ] || continue
    [ -f "${candidate}/trainer_state.json" ] && continue
    destination="${candidate}.incomplete-${SLURM_JOB_ID}"
    echo "Moving incomplete checkpoint ${candidate} to ${destination}"
    mv "${candidate}" "${destination}"
  done
}

quarantine_incomplete_checkpoints
START_STEP=$(latest_step)

if [ ! -f "${STATS_PATH}" ]; then
  python -u "${CODE_DIR}/launch_halab_finetune.py" \
    --base-model-path nvidia/GR00T-N1.7-3B \
    --data-parent "${DATA_PARENT}" \
    --output-dir "${OUTPUT_DIR}" \
    --statistics-path "${STATS_PATH}" \
    --openhlm-episode-split \
    --val-episode-fraction 0.05 \
    --val-split-seed 7 \
    --max-stat-samples 2048 \
    --num-gpus "${HALAB_NUM_GPUS}" \
    --global-batch-size "${HALAB_GLOBAL_BATCH}" \
    --gradient-accumulation-steps "${HALAB_ACCUMULATION}" \
    --max-steps "${MAX_STEPS}" \
    --modality-config-path "${CODE_DIR}/halab_47d_config.py" \
    --statistics-only
fi

TRAIN_ARGS=(
  --modality-config-path "${CODE_DIR}/halab_47d_config.py"
  --base-model-path nvidia/GR00T-N1.7-3B
  --data-parent "${DATA_PARENT}"
  --output-dir "${OUTPUT_DIR}"
  --statistics-path "${STATS_PATH}"
  --openhlm-episode-split
  --val-episode-fraction 0.05
  --val-split-seed 7
  --max-stat-samples 2048
  --max-eval-samples "${MAX_EVAL_SAMPLES}"
  --num-gpus "${HALAB_NUM_GPUS}"
  --global-batch-size "${HALAB_GLOBAL_BATCH}"
  --gradient-accumulation-steps "${HALAB_ACCUMULATION}"
  --learning-rate "${HALAB_LR}"
  --lr-scheduler-type "${HALAB_LR_SCHEDULER:-cosine}"
  --warmup-ratio "${HALAB_WARMUP_RATIO:-0.05}"
  --warmup-steps "${HALAB_WARMUP_STEPS:-0}"
  --dataloader-num-workers "${HALAB_DATALOADER_WORKERS}"
  --max-steps "${MAX_STEPS}"
  --save-steps "${SAVE_STEPS}"
  --eval-steps "${EVAL_STEPS}"
  --eval-batch-size 2
  --train-eval-samples "${TRAIN_EVAL_SAMPLES}"
  --leg-seconds "${LEG_SECONDS}"
  --keep-every-steps "${HALAB_KEEP_EVERY_STEPS:-0}"
  --use-wandb
  --state-dropout-prob "${HALAB_STATE_DROPOUT:-0.0}"
  --wandb-project "${WANDB_PROJECT}"
)

# HALAB_TUNE_VLM=1 (default): full fine-tune incl. the Cosmos-Reason2 LLM and vision encoder.
# HALAB_TUNE_VLM=0: GR00T default, VLM frozen; only the projector and action head train.
if [ "${HALAB_TUNE_VLM:-1}" = "1" ]; then
  TRAIN_ARGS+=(--tune-llm --tune-visual)
fi

if [ "${HALAB_COLOR_JITTER:-0}" = "1" ]; then
  TRAIN_ARGS+=(--color-jitter)
fi
if [ "${HALAB_TRAIN_RTC:-0}" = "1" ]; then
  TRAIN_ARGS+=(--train-time-rtc --rtc-max-delay "${HALAB_RTC_MAX_DELAY:-10}")
fi
if [ "${HALAB_GRADIENT_CHECKPOINTING}" = "1" ]; then
  TRAIN_ARGS+=(--gradient-checkpointing)
fi
if [ "${START_STEP}" -gt 0 ]; then
  TRAIN_ARGS+=(--resume-from-checkpoint "${OUTPUT_DIR}/checkpoint-${START_STEP}")
fi

set +e
# --cpu-bind=none: on some nodes (hk47) the job's CPUs are scattered and srun's default
# binding fails with "Unable to satisfy cpu bind request"; torchrun needs no pinning.
# The job stays inside its own cgroup CPU allocation either way.
timeout --signal=TERM --kill-after=120 "${TIMEOUT_SECONDS}" \
  srun --cpu-bind=none python -u -m torch.distributed.run \
    --nproc_per_node="${HALAB_NUM_GPUS}" \
    --master_port "${HALAB_MASTER_PORT}" \
    "${CODE_DIR}/launch_halab_finetune.py" "${TRAIN_ARGS[@]}"
TRAIN_STATUS=$?
set -e

END_STEP=$(latest_step)
echo "chain status: start_step=${START_STEP} end_step=${END_STEP} train_status=${TRAIN_STATUS}"

if [ "${END_STEP}" -ge "${MAX_STEPS}" ]; then
  echo "${HALAB_EXPERIMENT}: reached max_steps=${MAX_STEPS}; chain complete."
  exit 0
fi

NO_PROGRESS=${HALAB_NO_PROGRESS:-0}
MAX_NO_PROGRESS=${HALAB_MAX_NO_PROGRESS:-3}
if [ "${END_STEP}" -le "${START_STEP}" ]; then
  NO_PROGRESS=$((NO_PROGRESS + 1))
else
  NO_PROGRESS=0
fi
if [ "${NO_PROGRESS}" -ge "${MAX_NO_PROGRESS}" ]; then
  echo "${HALAB_EXPERIMENT}: ${NO_PROGRESS} consecutive legs made no checkpoint progress; stopping chain." >&2
  exit 1
fi

echo "${HALAB_EXPERIMENT}: queueing continuation from checkpoint-${END_STEP}."
sbatch \
  --dependency="afterany:${SLURM_JOB_ID}" \
  --export="ALL,HALAB_NO_PROGRESS=${NO_PROGRESS}" \
  "${HALAB_SUBMIT_SCRIPT}"
