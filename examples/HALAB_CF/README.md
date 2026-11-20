# HALAB_CF — camera-rotation counterfactual experiments

Two experiments that test the corrective-gain question from both research reports
(`research_report_2026-10-19.md` and the "Research Exploration" doc), using only the
existing HA-Lab data and checkpoints.  Nothing in `examples/HALAB/` or any senior
repository is modified; datasets and existing checkpoints are only read.

## The shared instrument

A scene rotation by `delta` about world up (through the camera center) changes the
ego image by an exact homography `H = K R K^-1`, independent of depth.  The matching
label rotates the 32 future base-quaternion targets by `delta` about the IMU z axis.
The visual heading gain

    G_v(k) = sum label_sign * delta * dyaw_k / sum delta^2

is 1 for a policy that turns with the scene (visual servo) and 0 for one that ignores
the image.  World up in the camera frame comes from the floor-aligned
`bev_camera_pose_world`; the sign convention is checked against real robot turns in
`outputs/cf_cache/calibration.json` (`label_sign` must be +1 or -1, otherwise no
rotation labels are used).

## Step 0 — `jobs/cf_exp0_prepare.sbatch` (1 GPU slot, CPU work, ~20–60 min)

`small_policy/build_cache.py`: 192x108 frames of every 6th frame of all 544
episodes, 30 Hz state/action rows, locomotion flags, world-up axes, and the warp
calibration.  Output `outputs/cf_cache/` (~19 GB).

## Experiment 1 — GR00T scale (4 A40, arms run in sequence, ~7 h each)

Continue the 20k run (v2) from `checkpoint-20000` for 2,000 steps, global batch
256, LR 2e-5 constant after 100 warmup steps, full fine-tuning, 8% center crop on
every frame in every arm, v2's normalization statistics.

| arm | output dir | rotation counterfactuals |
| --- | --- | --- |
| a | `outputs/halab_cf_a_control_lr2e5_2k` | none (continuation control) |
| b | `outputs/halab_cf_b_rot_f050_s4_lr2e5_2k` | 50% of locomotion samples, delta ~ N(0, 4 deg), clip 7 |
| c | `outputs/halab_cf_c_rot_f0125_s4_lr2e5_2k` | 12.5% of locomotion samples (fixed steps) |

Questions: does exact counterfactual supervision raise G_v at 3B scale (b vs a)?
Does dilution lower it (c vs b)?  Is anchored val MAE unchanged (within +3% relative
of arm a)?  Val metrics use the cropped frames, so compare arms with each other,
not with the uncropped 20k run.

## Experiment 2 — mechanism study (4 A40, one task per GPU, ~2–5 h)

`small_policy/run_queue.py` runs, resumably:

1. **GR00T probes** (`probe_gr00t_gain.py`, inference only): G_v, G_p, the turning
   slope and the yaw error on 512 locomotion and 1,024 nominal val anchors, for the
   20k run (v2) at 4 and 16 denoising steps, the 30k run (recipe), the latest
   v3_sd02 and v3_sd02_cj checkpoints, and the Experiment 1 arms once they exist.
2. **SNR-law sweep** (Idea 1): a small CNN + flow-matching policy, 12k steps,
   3 seeds per arm: rotation spread sigma = 0/2/4/6 deg; dilution f = 0.5, 0.25 at
   matched exposure (24k/48k steps) and at fixed steps; image blur 2x/4x; gauge-only.
   The law `G = G_max * S / (S + N)` is fit on the sigma arms and predicts the others.
3. **Retention arms** (ChatGPT report, Candidate 1): an old model trained on
   32 rotated (R) + 96 plain old (O) samples per batch for 10k steps, then 5k-step
   continuations: A (32R+96O), B (16R+48O+64E), C (32R+48O+48E), D/E/F (C plus
   nominal / pair-endpoint / pair-response distillation from the old model),
   H (A plus 48 new samples), H2 (A plus 48 duplicated old samples).  For one seed,
   R samples, angles, noise and flow times are identical in every arm.

What distinguishes the two theories:

| comparison | SNR law (data prior) predicts | fluent-data interference (ChatGPT H1) predicts |
| --- | --- | --- |
| C vs A (same R share, half O swapped for new E) | no difference | C below A |
| B vs A (R share halved) | B below A | B below A |
| H vs H2 (extra new vs extra old plain data) | equal, both below A | H below H2 |
| matched-exposure dilution arms | follow the fitted curve | — |

Kill criteria (predeclared): G_v of the 20k run already >= 0.8 (no headroom on
heading); sigma 0 -> 8 deg raises small-policy G_v by < 0.2 (counterfactuals do not
teach the response); matched-exposure dilution changes G_v by < 0.05 while sigma or
blur clearly change it (SNR law falsified for heading); GR00T arm b worsens anchored
val MAE by > 3% relative to arm a.

## Outputs

* `outputs/cf_cache/calibration.json` — warp sign check (read this first).
* `outputs/cf_exp2/summary.md` — all tables (refreshed after every Experiment 2 leg).
* `outputs/cf_exp2/arms.csv`, `runs/<arm>_s<seed>/results.json`, `probes/<name>/probe.json`.
* Experiment 1: wandb project `halab-gr00t-n1d7`, the output dirs above.
