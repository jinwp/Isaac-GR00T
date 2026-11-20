#!/bin/bash -l
# Zero-shot ω-0 (HF keycharon/Omega-zero) on HA-Lab val episodes, then the SONIC physics sim of its tokens.
#
#   1. stats : rebuild ω-0's unpublished action/state normalization stats from ω-HOME (once)
#   2. infer : open-loop token prediction on the GR00T 20k run (v2) val split, 1 episode per task (same seed as
#              the GR00T evals), recorded image + state at anchors every REPLAN rows
#   3. sim   : predicted and recorded token streams through SONIC (decoder only, protocol v4), arm override off,
#              first SIM_SECONDS s; overlay videos + body-joint MAE table
#
#   sbatch jobs/omega0_zeroshot.sbatch
#   EPISODE_KEYS="pick_tennis_racket_place_black_chair/32" SIM_SECONDS=60 sbatch jobs/omega0_zeroshot.sbatch
#   SIM_ONLY=1 sbatch --exclude=<node> jobs/omega0_zeroshot.sbatch   (infer already done; sim again elsewhere)
#   SIM_ONLY=1 SIM_OUT=$GB/outputs/omega0_zeroshot_gmroverride SIM_ARGS="--which gt --arm-override --tasks umbrella" \
#     sbatch --exclude=megabot jobs/omega0_zeroshot.sbatch
#                                    (recorded token + recorded GMR arms -> $SIM_OUT/val/<task>/<episode>/physics_sim/gt_armov/)
#   HOME_TASKS="clean_bed pick_garbage wipe_table" OUT=$GB/outputs/omega0_home_check sbatch --exclude=megabot jobs/omega0_zeroshot.sbatch
#                                    (deploy check on ω-0's own ω-HOME data: needs <task>/<episode>/ego.mp4 downloaded;
#                                     -> $OUT/home/<task>/<episode>/, first episode of each task)

#SBATCH --job-name=omega0_zs
#SBATCH --partition=ha-lab
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:55:00
#SBATCH --output=/coc/flash2/jpark984/gr00t_baseline/omega0-zeroshot-%j.out
#SBATCH --error=/coc/flash2/jpark984/gr00t_baseline/omega0-zeroshot-%j.err

set -eo pipefail
GB=/coc/flash2/jpark984/gr00t_baseline
OM=$GB/omega
CKPT=$OM/Omega-zero
EXT=$OM/external
PY_OMEGA=$GB/OMEGA-0/.venv/bin/python
PY_GR00T=$GB/Isaac-GR00T/.venv/bin/python
HALAB=$GB/Isaac-GR00T/examples/HALAB
SPLIT=$GB/outputs/halab_g1_47d_openhlm_b64_g256_lr1e4_v2/episode_split.json
OUT=${OUT:-$GB/outputs/omega0_zeroshot}
SIM_OUT=${SIM_OUT:-$OUT}  # sim runs + sim_log.txt; inputs always come from $OUT
REPLAN=${REPLAN:-8}
DATA=val; [ -n "${HOME_TASKS:-}" ] && DATA=home  # infer output split dir, sim input
SIM_SECONDS=${SIM_SECONDS:-30}
BIN=/coc/flash2/jpark984/samsung_deploy/target/release/g1_deploy_onnx_ref
# No .pyc into the senior's MotionTokenizers checkout (imported read-only by the video render).
export HF_HOME=$GB/huggingface PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
mkdir -p "$OUT" "$SIM_OUT"
echo "out: $OUT  sim out: $SIM_OUT"

if ! grep -qa token_state "$BIN"; then
  echo "$BIN has no protocol v4 token_state input: rebuild it from wb_teleop before the token sim"
  exit 1
fi
# The binary's unitree SDK aborts ("getpwuid error") on nodes that cannot resolve our uid.
if ! getent passwd "$(id -u)" > /dev/null; then
  echo "$(hostname) cannot resolve uid $(id -u) (SONIC dies with getpwuid error): resubmit with --exclude=$(hostname)"
  exit 1
fi
# The MuJoCo sim's unitree_sdk2py writes a CycloneDDS trace to a fixed file; another user's copy of it kills the sim.
CDDS_LOG=$(grep -o '/tmp/[^<]*\.LOG' /coc/flash2/jpark984/samsung_deploy/thirdparty/unitree_sdk2_python/unitree_sdk2py/core/channel_config.py)
if [ -e "$CDDS_LOG" ] && [ ! -w "$CDDS_LOG" ]; then
  echo "$(hostname): $CDDS_LOG belongs to $(stat -c %U "$CDDS_LOG") (sim dies with 'channel factory init error'): resubmit with --exclude=$(hostname)"
  exit 1
fi

if [ ! -f "$EXT/stats/wam_stat_no_human.json" ] || [ ! -f "$EXT/stats/wam_stat_state_no_human.json" ]; then
  "$PY_OMEGA" "$HALAB/omega0_home_stats.py" --home-root "$OM/omega-HOME" \
    --run-config "$CKPT/run_config.json" --out-dir "$EXT/stats"
fi

if [ -z "${SIM_ONLY:-}" ]; then
  if [ -n "${HOME_TASKS:-}" ]; then SRC=(--home-root "$OM/omega-HOME" --tasks $HOME_TASKS)
  else SRC=(--episode-split "$SPLIT" ${EPISODE_KEYS:+--episode-keys $EPISODE_KEYS}); fi
  "$PY_OMEGA" "$HALAB/omega0_infer.py" --checkpoint "$CKPT" --external-models "$EXT" "${SRC[@]}" \
    --replan "$REPLAN" --output-dir "$OUT" 2>&1 | tee "$OUT/infer_log.txt"
fi

cd "$GB/Isaac-GR00T"
flock /tmp/jpark984_sonic.lock "$PY_GR00T" examples/HALAB/omega0_sim.py --root "$OUT/$DATA" \
  --max-seconds "$SIM_SECONDS" --skip-existing --out-root "$SIM_OUT/$DATA" ${SIM_ARGS:-} 2>&1 | tee "$SIM_OUT/sim_log.txt"
echo "done: $OUT"
