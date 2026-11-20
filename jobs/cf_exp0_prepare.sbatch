#!/bin/bash -l
# Step 0 (run once, before Experiments 1 and 2): low-resolution frame cache for the
# small-policy study plus the warp-sign calibration that both experiments read.
# CPU work only; the GPU is requested because the ha-lab partition schedules by GPU.
# Writes /coc/flash2/jpark984/gr00t_baseline/outputs/cf_cache (about 19 GB).

#SBATCH --job-name=cf0_prep
#SBATCH --partition=ha-lab
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=a40:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:30:00
#SBATCH --output=/coc/flash2/jpark984/gr00t_baseline/logs/cf_exp0-%j.out
#SBATCH --error=/coc/flash2/jpark984/gr00t_baseline/logs/cf_exp0-%j.err

set -euo pipefail
cd /coc/flash2/jpark984/gr00t_baseline/Isaac-GR00T
source .venv/bin/activate
export PYTHONUNBUFFERED=1
python -u examples/HALAB_CF/small_policy/build_cache.py --workers 16
