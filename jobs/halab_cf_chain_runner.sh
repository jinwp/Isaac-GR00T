#!/bin/bash -l
# Chain runner for the counterfactual continuation arms (Experiment 1).
# Same leg/resume/quarantine logic as halab_chain_runner.sh, plus:
#   - initializes from HALAB_INIT_CHECKPOINT (fresh optimizer, new LR schedule)
#   - reuses the 20k run's normalization statistics (HALAB_STATS_PATH, read-only)
#   - rotation counterfactual / crop flags, label sign from the warp calibration
#   - submits HALAB_NEXT_SUBMIT_SCRIPT when this arm reaches max steps

set -uo pipefail

: "${HALAB_EXPERIMENT:?}"
: "${HALAB_SUBMIT_SCRIPT:?}"
: "${HALAB_NUM_GPUS:?}"
: "${HALAB_GLOBAL_BATCH:?}"
: "${HALAB_LR:?}"
: "${HALAB_INIT_CHECKPOINT:?}"
: "${HALAB_STATS_PATH:?}"
: "${HALAB_CF_FRAC:?}"

ROOT=/coc/flash2/jpark984/gr00t_baseline
REPO=${ROOT}/Isaac-GR00T
DATA_PARENT=${ROOT}/datasets/ha-lab-dataset/halab
OUTPUT_DIR=${ROOT}/outputs/${HALAB_EXPERIMENT}
CALIBRATION=${HALAB_CALIBRATION:-${ROOT}/outputs/cf_cache/calibration.json}
MAX_STEPS=${HALAB_MAX_STEPS:-2000}
LEG_SECONDS=${HALAB_LEG_SECONDS:-12000}
TIMEOUT_SECONDS=13200
SAVE_STEPS=${HALAB_SAVE_STEPS:-500}
EVAL_STEPS=1000
MAX_EVAL_SAMPLES=1024
TRAIN_EVAL_SAMPLES=256
WANDB_PROJECT=${HALAB_WANDB_PROJECT:-halab-gr00t-n1d7}
HALAB_DATALOADER_WORKERS=${HALAB_DATALOADER_WORKERS:-4}

cd "${REPO}"
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export HF_HOME=${ROOT}/huggingface

LABEL_SIGN=$(python -c "import json; print(int(json.load(open('${CALIBRATION}'))['label_sign']))")
if [ "${LABEL_SIGN}" != "1" ] && [ "${LABEL_SIGN}" != "-1" ]; then
  if python -c "import sys; sys.exit(0 if float('${HALAB_CF_FRAC}') > 0 else 1)"; then
    echo "Warp calibration failed (label_sign=${LABEL_SIGN} in ${CALIBRATION}); refusing to train with rotation labels." >&2
    exit 1
  fi
  LABEL_SIGN=1  # unused: this arm has no rotation counterfactuals
fi

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

TRAIN_ARGS=(
  --modality-config-path examples/HALAB/halab_47d_config.py
  --base-model-path nvidia/GR00T-N1.7-3B
  --init-checkpoint "${HALAB_INIT_CHECKPOINT}"
  --data-parent "${DATA_PARENT}"
  --output-dir "${OUTPUT_DIR}"
  --statistics-path "${HALAB_STATS_PATH}"
  --openhlm-episode-split
  --val-episode-fraction 0.05
  --val-split-seed 7
  --max-stat-samples 2048
  --max-eval-samples "${MAX_EVAL_SAMPLES}"
  --num-gpus "${HALAB_NUM_GPUS}"
  --global-batch-size "${HALAB_GLOBAL_BATCH}"
  --gradient-accumulation-steps 1
  --learning-rate "${HALAB_LR}"
  --lr-scheduler-type "${HALAB_LR_SCHEDULER:-constant_with_warmup}"
  --warmup-ratio 0.0
  --warmup-steps "${HALAB_WARMUP_STEPS:-100}"
  --dataloader-num-workers "${HALAB_DATALOADER_WORKERS}"
  --max-steps "${MAX_STEPS}"
  --save-steps "${SAVE_STEPS}"
  --eval-steps "${EVAL_STEPS}"
  --eval-batch-size 2
  --train-eval-samples "${TRAIN_EVAL_SAMPLES}"
  --leg-seconds "${LEG_SECONDS}"
  --use-wandb
  --state-dropout-prob 0.0
  --wandb-project "${WANDB_PROJECT}"
  --tune-llm
  --tune-visual
  --gradient-checkpointing
  --rotation-cf-frac "${HALAB_CF_FRAC}"
  --rotation-cf-std-deg "${HALAB_CF_STD_DEG:-4.0}"
  --rotation-cf-max-deg "${HALAB_CF_MAX_DEG:-7.0}"
  --crop-frac "${HALAB_CROP_FRAC:-0.08}"
  --walk-threshold "${HALAB_WALK_THRESHOLD:-0.15}"
  --label-sign "${LABEL_SIGN}"
  --cf-seed "${HALAB_CF_SEED:-1234}"
)

if [ "${START_STEP}" -gt 0 ]; then
  TRAIN_ARGS+=(--resume-from-checkpoint "${OUTPUT_DIR}/checkpoint-${START_STEP}")
fi

set +e
timeout --signal=TERM --kill-after=120 "${TIMEOUT_SECONDS}" \
  srun --cpu-bind=none python -u -m torch.distributed.run \
    --nproc_per_node="${HALAB_NUM_GPUS}" \
    --master_port "${HALAB_MASTER_PORT}" \
    examples/HALAB_CF/launch_halab_cf.py "${TRAIN_ARGS[@]}"
TRAIN_STATUS=$?
set -e

END_STEP=$(latest_step)
echo "chain status: start_step=${START_STEP} end_step=${END_STEP} train_status=${TRAIN_STATUS}"

if [ "${END_STEP}" -ge "${MAX_STEPS}" ]; then
  echo "${HALAB_EXPERIMENT}: reached max_steps=${MAX_STEPS}; arm complete."
  if [ -n "${HALAB_NEXT_SUBMIT_SCRIPT:-}" ]; then
    echo "Submitting next arm: ${HALAB_NEXT_SUBMIT_SCRIPT}"
    sbatch --export="ALL,HALAB_NO_PROGRESS=0" "${HALAB_NEXT_SUBMIT_SCRIPT}"
  fi
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
