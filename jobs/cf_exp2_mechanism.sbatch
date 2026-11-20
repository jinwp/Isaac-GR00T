#!/bin/bash -l
# Experiment 2: mechanism study on 4 A40 GPUs (one task per GPU).
#   - GR00T gain probes of the existing checkpoints (and of the Experiment 1 arms
#     once they exist)
#   - small-policy SNR-law sweep (33 runs) and retention arms A-H2 (27 runs)
# Resumable: the queue stops 5 min before the time limit, training tasks save,
# and this script resubmits itself until every task is done (stops after two legs
# without progress).  Summary: outputs/cf_exp2/summary.md

#SBATCH --job-name=cf2_mech
#SBATCH --partition=ha-lab
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=a40:4
#SBATCH --cpus-per-task=28
#SBATCH --mem=200G
#SBATCH --time=03:55:00
#SBATCH --output=/coc/flash2/jpark984/gr00t_baseline/logs/cf_exp2-%j.out
#SBATCH --error=/coc/flash2/jpark984/gr00t_baseline/logs/cf_exp2-%j.err

set -uo pipefail
SELF=/coc/flash2/jpark984/gr00t_baseline/Isaac-GR00T/jobs/cf_exp2_mechanism.sbatch
OUT=/coc/flash2/jpark984/gr00t_baseline/outputs/cf_exp2
cd /coc/flash2/jpark984/gr00t_baseline/Isaac-GR00T
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export HF_HOME=/coc/flash2/jpark984/gr00t_baseline/huggingface

python -u examples/HALAB_CF/small_policy/run_queue.py --deadline-seconds 13500
STATUS=$?
python -u examples/HALAB_CF/small_policy/analyze.py > /dev/null || true

if [ "${STATUS}" -eq 3 ]; then
  COMPLETED=$(cat "${OUT}/last_leg_completed.txt" 2>/dev/null || echo 0)
  NO_PROGRESS=${EXP2_NO_PROGRESS:-0}
  if [ "${COMPLETED}" -eq 0 ]; then NO_PROGRESS=$((NO_PROGRESS + 1)); else NO_PROGRESS=0; fi
  if [ "${NO_PROGRESS}" -ge 2 ]; then
    echo "Experiment 2: two legs without a completed task; not resubmitting." >&2
    exit 1
  fi
  sbatch --dependency="afterany:${SLURM_JOB_ID}" --export="ALL,EXP2_NO_PROGRESS=${NO_PROGRESS}" "${SELF}"
fi
