#!/bin/bash -l
# Live SONIC-in-the-loop sim of the GR00T baseline: the real deploy client (run_vla_sonic.py --policy gr00t,
# live replanning every DELAY rows) against the SONIC binary driving a MuJoCo G1 -- the setup of
# samsung_deploy vla_deploy/README.md "SONIC-in-the-loop without a robot", with OpenHLM's
# _sim_loop_with_qpos_log.py as the physics (the loop the offline SONIC evals ran on).
# Vision is open loop: the policy sees the recorded episode's camera, not the sim.
#
#   sbatch jobs/halab_live_sim.sbatch
#   TASK=pick_puppy_doll_place_blue_basket EPISODE=26 INSTRUCTION="pick a puppy doll on the shelf and place it in the blue basket" \
#     sbatch jobs/halab_live_sim.sbatch
#
# Only copies owned by jpark984 are used: the client from gr00t_baseline/samsung_deploy (wb_teleop + gr00t),
# the binary / .venv_teleop / MuJoCo sim from /coc/flash2/jpark984/samsung_deploy, OpenHLM from gr00t_baseline/OpenHLM.

#SBATCH --job-name=halab_live_sim
#SBATCH --partition=ha-lab
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=a40:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/coc/flash2/jpark984/gr00t_baseline/halab-live-sim-%j.out
#SBATCH --error=/coc/flash2/jpark984/gr00t_baseline/halab-live-sim-%j.err

set -u
GB=/coc/flash2/jpark984/gr00t_baseline
SD_RUN=$GB/samsung_deploy                      # deploy client: wb_teleop + vla_deploy/policy/gr00t.py
SD_BIN=/coc/flash2/jpark984/samsung_deploy     # built SONIC binary, .deps, .venv_teleop, pico/ MuJoCo sim
OPENHLM=$GB/OpenHLM
PY_TELEOP=$SD_BIN/.venv_teleop/bin/python
PY_GR00T=$GB/Isaac-GR00T/.venv/bin/python
CKPT=${CKPT:-$GB/outputs/halab_g1_47d_openhlm_b64_g256_lr1e4_v2/checkpoint-20000}
TASK=${TASK:-pick_tennis_racket_place_black_chair}
EPISODE=${EPISODE:-32}
INSTRUCTION=${INSTRUCTION:-"Pick a tennis racket on top of the shelf and place it on top of the black chair"}
RUN_SECONDS=${RUN_SECONDS:-30}
POLICY=${POLICY:-gr00t}                        # hold = no model: is the robot standing at all?
DELAY=${DELAY:-8}                              # rows; inference here is ~210-250 ms (GPU shared with SONIC + MuJoCo)
CHUNK_LEN=${CHUNK_LEN:-48}                     # >= 2 * DELAY + 29 keeps SONIC's 46-frame lookahead fed
GR00T_PORT=${GR00T_PORT:-5590}
OUT=$GB/outputs/live_sim/${POLICY}_${TASK}_ep${EPISODE}_d${DELAY}_c${CHUNK_LEN}_${SLURM_JOB_ID:-local}
mkdir -p "$OUT"
echo "out: $OUT"

for f in "$PY_TELEOP" "$PY_GR00T" "$CKPT" "$SD_BIN/target/release/g1_deploy_onnx_ref" "$SD_RUN/vla_deploy/policy/gr00t.py" \
         "$OPENHLM/src/openpi4OpenHLM/scripts/ha/_sim_loop_with_qpos_log.py" \
         "$GB/datasets/ha-lab-dataset/halab/$TASK/videos/chunk-000/observation.images.left/episode_$(printf %06d "$EPISODE").mp4"; do
  [ -e "$f" ] || { echo "missing: $f"; exit 1; }
done
BIN=$SD_BIN/target/release/g1_deploy_onnx_ref
if ! grep -qa zmq_manager "$BIN"; then
  echo "the binary in $SD_BIN has no --input-type zmq_manager (needed by run_vla_sonic.py); it must be rebuilt from wb_teleop"
  exit 1
fi
ARM_FLAG=""
if ! grep -qa gmr_arm "$BIN"; then
  ARM_FLAG="--no-arm-override"
  echo "binary has no GMR arm override: client runs with --no-arm-override (arms follow SONIC)"
fi
# wb_teleop fixed zmq_manager dropping start=True in STREAMED_MOTION ("CONTROL will start"). An older build
# never leaves WAIT_FOR_CONTROL on the client's command (robot held by plain PD, falls once the band is
# released): start CONTROL with the binary's own ']' key instead, as the zmq-input sim runner does.
START_KEYS=""
if ! grep -qa "CONTROL will start" "$BIN"; then
  START_KEYS="]"
  echo "binary predates the zmq_manager start fix: CONTROL is started with ']' once the binary is ready"
fi

# Runtime env of the deploy binary (OpenHLM sonic_runner._runtime_env).
NV=$GB/Isaac-GR00T/.venv/lib/python3.12/site-packages/nvidia
export LD_LIBRARY_PATH=$SD_BIN/.deps/onnxruntime-linux-x64-gpu-1.19.2/lib:$SD_BIN/.deps/TensorRT-10.16.1.11-cuda12.9/lib:$SD_BIN/.deps/cppdeps/lib:$SD_BIN/thirdparty/unitree_sdk2/thirdparty/lib/x86_64:$NV/cuda_runtime/lib:$NV/cublas/lib:$NV/cudnn/lib:$NV/cufft/lib:$NV/curand/lib:$NV/cusolver/lib:$NV/cusparse/lib:$NV/nvjitlink/lib:${LD_LIBRARY_PATH:-}
export CYCLONEDDS_URI="<CycloneDDS><Domain><SharedMemory><Enable>false</Enable></SharedMemory></Domain></CycloneDDS>"
export PYTHONUNBUFFERED=1

PIDS=()
cleanup() {
  for pid in "${PIDS[@]}"; do kill -TERM "$pid" 2>/dev/null; done
  sleep 8
  for pid in "${PIDS[@]}"; do kill -KILL "$pid" 2>/dev/null; done
}
trap cleanup EXIT

wait_for() {  # file pattern timeout_s
  local t=0
  until grep -q "$2" "$1" 2>/dev/null; do
    sleep 2; t=$((t + 2))
    [ "$t" -ge "$3" ] && { echo "timeout waiting for '$2' in $1"; tail -20 "$1"; exit 1; }
  done
}

# 1. GR00T server first: CONTROL may start as soon as the binary is ready (']'), so the client must follow quickly.
if [ "$POLICY" = gr00t ]; then
  (cd "$GB/Isaac-GR00T" && exec "$PY_GR00T" examples/HALAB/halab_gr00t_server.py --checkpoint "$CKPT" --port "$GR00T_PORT") \
    > "$OUT/gr00t_server.log" 2>&1 &
  PIDS+=($!)
  wait_for "$OUT/gr00t_server.log" "listening on" 900
  grep "inference latency" "$OUT/gr00t_server.log"
  POLICY_ARGS=(--policy gr00t --policy-arg server=tcp://127.0.0.1:$GR00T_PORT --policy-arg chunk_len=$CHUNK_LEN
               --rtc-inference-delay "$DELAY" --rtc-replan-interval "$DELAY")
else
  POLICY_ARGS=(--policy "$POLICY")
fi

# 2. MuJoCo physics (DDS on lo), elastic band lowered / released by sentinel files.
LOWER=$OUT/lower_band.sentinel
RELEASE=$OUT/release_band.sentinel
(cd "$SD_BIN" && SAMSUNG_DEPLOY_ROOT=$SD_BIN QPOS_LOG_OUT=$OUT/sim_qpos_log.npz LOWER_BAND_SENTINEL=$LOWER \
  RELEASE_BAND_SENTINEL=$RELEASE BAND_TARGET_HEIGHT=0.793 BAND_LOWER_SECONDS=1.0 \
  exec "$PY_TELEOP" "$OPENHLM/src/openpi4OpenHLM/scripts/ha/_sim_loop_with_qpos_log.py" \
  --no-enable-onscreen --with-hands --interface lo) > "$OUT/sim.log" 2>&1 &
PIDS+=($!)
sleep 4

# 3. Robot-side state: the sim's rt/lowstate -> g1_debug-shaped ZMQ on :5559.
(cd "$SD_RUN" && exec "$PY_TELEOP" vla_deploy/tests/sim_lowstate_bridge.py --interface lo --bind "tcp://*:5559" --rate 100) \
  > "$OUT/bridge.log" 2>&1 &
PIDS+=($!)

# 4. SONIC binary, waiting in WAIT_FOR_CONTROL for the client's start command (zmq_manager).
DEPLOY_CMD="target/release/g1_deploy_onnx_ref lo policy/release/model_decoder.onnx reference/example/ \
--obs-config policy/release/observation_config.yaml --encoder-file policy/release/model_encoder.onnx \
--planner-file planner/target_vel/V2/planner_sonic.onnx --input-type zmq_manager --output-type zmq \
--zmq-host localhost --zmq-port 5556 --zmq-out-port 5557 --zmq-out-topic g1_debug --disable-crc-check \
--enable-motor-target-transition --motor-target-transition-duration 1.0"
[ -f "$SD_BIN/config/debug_arm_pd_gains.yaml" ] && DEPLOY_CMD="$DEPLOY_CMD --debug-arm-pd-gains config/debug_arm_pd_gains.yaml"
(cd "$SD_RUN" && exec "$PY_TELEOP" vla_deploy/tests/sim_deploy_pty.py --workdir "$SD_BIN" --cmd "$DEPLOY_CMD" \
  --log "$OUT/deploy.log" --marker "$OUT/deploy.ready" --keys "$START_KEYS") > "$OUT/deploy_pty.log" 2>&1 &
PIDS+=($!)

# 5. Frame server: bridge state + the episode's recorded camera + echoed hand / head commands.
(cd "$SD_RUN" && exec "$PY_TELEOP" vla_deploy/tests/sim_frame_server.py --port 5572 --g1-debug tcp://127.0.0.1:5559 \
  --halab-task "$GB/datasets/ha-lab-dataset/halab/$TASK" --episode "$EPISODE" \
  --hand-endpoint tcp://127.0.0.1:5564 --head-endpoint tcp://127.0.0.1:5558 --vla-action-endpoint tcp://127.0.0.1:5566 \
  --out "$OUT" --capture "$OUT/g1_capture.npz") > "$OUT/frame_server.log" 2>&1 &
PIDS+=($!)

echo "waiting for the binary (a TensorRT engine build can take minutes)..."
t=0
until [ -e "$OUT/deploy.ready" ]; do
  sleep 5; t=$((t + 5))
  [ "$t" -ge 2400 ] && { echo "binary never ready"; tail -30 "$OUT/deploy.log"; exit 1; }
done

# 6. The binary's own g1_debug (SONIC targets + measured, CONTROL ticks): shows whether CONTROL ever ran.
(cd "$SD_RUN" && exec "$PY_TELEOP" "$OPENHLM/src/openpi4OpenHLM/scripts/ha/_g1_debug_subscriber.py" \
  --host localhost --port 5557 --topic g1_debug --out "$OUT/g1_debug_ctrl.npz") > "$OUT/g1_debug_sub.log" 2>&1 &
PIDS+=($!)

# 7. Band: lowered onto the feet once the client has sent the start command (GATE 1), released 3 s later,
#    still inside the 5 s settle (before the policy acts).
(
  wait_for "$OUT/client.log" "settle remeasured at gate 1" 1200
  touch "$LOWER"; echo "$(date +%T) gate 1 passed: band lowering"
  sleep 3
  touch "$RELEASE"; echo "$(date +%T) band released"
) > "$OUT/band.log" 2>&1 &
PIDS+=($!)

# 8. The deploy client, exactly as on the robot (minus the operator gates).
cd "$SD_RUN"
"$PY_TELEOP" vla_deploy/run_vla_sonic.py "${POLICY_ARGS[@]}" \
  --task "$INSTRUCTION" --max-seconds "$RUN_SECONDS" --settle-seconds 5 --yes $ARM_FLAG \
  --openhlm-root "$OPENHLM" --venv-python "$PY_TELEOP" --out-dir "$OUT/client" 2>&1 | tee "$OUT/client.log"
CLIENT_RC=${PIPESTATUS[0]}
echo "client exit $CLIENT_RC"

cleanup
trap - EXIT
"$PY_TELEOP" vla_deploy/tests/analyze_sonic_sim_run.py --run-dir "$OUT/client" --g1-debug "$OUT/g1_capture.npz" \
  --frame-server-log "$OUT/frame_server.log" --g1-debug-ctrl "$OUT/g1_debug_ctrl.npz" 2>&1 | tee "$OUT/analysis.txt"
echo "sim falls / resets: $(grep -c 'Robot has fallen' "$OUT/sim.log")" | tee -a "$OUT/analysis.txt"
"$PY_TELEOP" - "$OUT/sim_qpos_log.npz" <<'EOF' | tee -a "$OUT/analysis.txt"
import sys, numpy as np
z = np.load(sys.argv[1]); q = z["qpos"]; r = int(z["release_tick_idx"][0]) if "release_tick_idx" in z.files else 0
pz = q[r:, 2]
print(f"sim pelvis z after release: median {np.median(pz):.3f} m, min {pz.min():.3f} m, last {pz[-1]:.3f} m, ticks {len(pz)}")
EOF
MUJOCO_GL=egl "$PY_GR00T" "$GB/Isaac-GR00T/examples/HALAB/halab_live_sim_render.py" --qpos "$OUT/sim_qpos_log.npz" \
  --mjcf "$SD_BIN/mujoco_sim/assets/g1_inspire.xml" --out "$OUT/sim.mp4"
ln -sf "$GB/datasets/ha-lab-dataset/halab/$TASK/videos/chunk-000/observation.images.left/episode_$(printf %06d "$EPISODE").mp4" \
  "$OUT/policy_camera.mp4"  # what the policy saw (open-loop vision), from the first vla_action on
echo "done: $OUT"
