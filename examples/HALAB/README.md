# HA-Lab GR00T N1.7 baseline

GR00T N1.7-3B 를 HA-Lab G1 teleop 데이터로 full fine-tune 한 whole-body baseline (47D 절대 관절 출력).
NVIDIA Isaac-GR00T `51d4c89` 위에 파일만 추가했고, GR00T 원본 파일은 수정하지 않았습니다.

## 구성

| 위치 | 내용 |
|---|---|
| `data_loader/` | HA-Lab 데이터 로더 (`ha/`). adapter 가 이 폴더를 먼저 찾고, 없으면 `Isaac-GR00T` 옆의 `data_loader/` 를 씁니다 |
| `examples/HALAB/halab_47d_config.py` | NEW_EMBODIMENT 47D modality config (state 76D: q29, dq29, quat4, hand12, cam2 / action 47D: q29, quat4, hand12, cam2) |
| `examples/HALAB/halab_gr00t_adapter.py` | HA-Lab 샘플 → GR00T `VLAStepData`, GPU x worker 별로 나눠 읽는 학습 dataset |
| `examples/HALAB/launch_halab_finetune.py` | 학습 launcher (OpenHLM split, eval, keep 체크포인트, RTC, chunk 길이) |
| `examples/HALAB/halab_metrics.py`, `halab_fk.py` | OpenHLM 과 같은 정의의 eval 지표 (ODE MAE, FK cm) |
| `examples/HALAB/halab_rtc.py` | training-time RTC (arXiv 2512.05964) + prefix 고정 샘플러 |
| `examples/HALAB/halab_kinematic_play.py` | 오프라인 rollout (`replan` = 매 chunk 기록 state, `state-feedback` = 모델 출력을 다음 state 로) |
| `examples/HALAB/halab_sonic_*.py` | SONIC 물리 sim 재생 / 비교 (OpenHLM + samsung_deploy 필요) |
| `examples/HALAB/halab_gr00t_server.py` | 배포용 추론 서버 (ZMQ, 기본 포트 5590) |
| `examples/HALAB/halab_deploy_replay_compare.py` | 로봇 data-replay 로그와 오프라인 추론 비교 |
| `examples/HALAB/halab_input_sensitivity.py` | 이미지만 / state 만 바꿨을 때 출력 변화 (copycat 측정) |
| `examples/HALAB/halab_hand_state_check.py` | 데이터의 손 state vs 손 명령, 영상 길이 점검 |
| `examples/HALAB/omega0_*.py` | OMEGA-0 zero-shot 비교용 |
| `examples/HALAB_CF/` | counterfactual 실험 (별도) |
| `jobs/` | Skynet sbatch + chain runner (3:55 leg 마다 저장 후 재제출) |
| `deploy/gr00t.py`, `deploy/register_gr00t.py` | samsung_deploy `vla_deploy` 용 `--policy gr00t` 플러그인 |

`jobs/*.sbatch` 와 일부 스크립트 기본값에는 Skynet 경로 (`/coc/flash2/jpark984/gr00t_baseline/...`) 가 들어 있습니다.

## 환경

Isaac-GR00T 기본 설치 (`.venv`) 에 `opencv-python`, `pyarrow` 가 추가로 필요합니다 (HA-Lab 로더).
데이터: HF `sehoonha/ha-lab-dataset` 의 `halab/` (task 폴더들의 부모 폴더를 `--data-parent` 로 지정).

- 12-task 데이터: legacy 관절 단위.
- 38-task 데이터 (rev `702ef02b`): mjcf_v1 단위 (robot A 손목 x1.5625, 21번 관절 부호). 배포 시 `joint_units=mjcf_v1`.

## 학습

```bash
sbatch jobs/halab38_g1_47d_c50_rtc_50k.sbatch   # 38 task, chunk 50, RTC, 50k step, 4x A40
```

sbatch 는 `HALAB_*` 환경변수만 정하고 `jobs/halab_chain_runner.sh` → `launch_halab_finetune.py` 를 실행합니다.
주요 값: `HALAB_ACTION_HORIZON=50`, `HALAB_GLOBAL_BATCH=256`, `HALAB_LR=1e-4` (cosine, warmup 5%),
`HALAB_TRAIN_RTC=1`, `HALAB_RTC_MAX_DELAY=10`, `HALAB_STATE_DROPOUT=0.0`, `HALAB_KEEP_EVERY_STEPS=10000`.
VLM 포함 전체 학습 (`tune_llm`, `tune_visual` 켬; GR00T 기본값은 둘 다 꺼짐). `HALAB_TUNE_VLM=0` 이면 VLM freeze (GR00T 기본값).

OpenHLM `--action-fps` / `--temporal mem` 대응 (기본값은 꺼짐, 기존 런과 같음):

| 변수 | 의미 |
|---|---|
| `HALAB_ACTION_STRIDE=N` | chunk 행 간격 (30 Hz 프레임 수). 2 = 15 Hz, 행 j = 프레임 t + N(j+1) |
| `HALAB_IMAGE_HISTORY=K` | 과거 이미지 K장 (GR00T 여러 장 이미지 입력, 오래된 것부터, 현재가 마지막). 에피소드 시작 전 프레임은 뺌 |
| `HALAB_IMAGE_HISTORY_STRIDE`, `_JITTER` | 과거 이미지 간격 (기본 30) 과 학습 때 ±jitter (기본 5) |
| `HALAB_IMAGE_HISTORY_DROP_ALL`, `_DROP_OLDEST` | 학습 때 샘플마다 과거 이미지 전부를 빼는 확률, 아니면 가장 오래된 k ~ U{1..K} 장을 빼는 확률 |

학습 시 `HALAB_*` 값은 `<run>/halab_env.json` 에 저장됩니다.

| run | sbatch |
|---|---|
| v2 (12 task, chunk 32, 20k) | `jobs/halab_g1_47d_v2.sbatch` |
| v5 (12 task, chunk 50 + RTC, 20k) | `jobs/halab_g1_47d_v5_c50_rtc.sbatch` |
| 38 task (chunk 50 + RTC, 50k) | `jobs/halab38_g1_47d_c50_rtc_50k.sbatch` |
| 38 task, VLM freeze (나머지 위와 같음) | `jobs/halab38_g1_47d_c50_rtc_50k_vlmfreeze.sbatch` |
| 38 task, VLM freeze + 과거 이미지 5장 (1초 간격, dropout 0.1/0.3) + action 15 Hz | `jobs/halab38_g1_47d_c50_rtc_50k_vlmfreeze_mem15.sbatch` |

## 평가

```bash
export HALAB_ACTION_HORIZON=50   # 체크포인트 chunk 길이
.venv/bin/python examples/HALAB/halab_kinematic_play.py --checkpoint <run>/checkpoint-N \
    --data-parent <halab> --output-dir <out> --rollout state-feedback
.venv/bin/python examples/HALAB/halab_input_sensitivity.py --checkpoints <ckpt1> <ckpt2> \
    --data-parent <halab> --episode-keys <task>/<episode> ... --out <out>.json
```

## 배포 (로봇 PC)

```bash
# 1) samsung_deploy 에 플러그인 등록 (gr00t.py 복사 + policy/__init__.py 에 2줄)
python deploy/register_gr00t.py <samsung_deploy 경로>
# 2) GR00T 서버
.venv/bin/python examples/HALAB/halab_gr00t_server.py --checkpoint <ckpt> --rtc-max-delay 10
# 3) samsung_deploy 쪽: --policy gr00t --policy-arg joint_units=mjcf_v1 (38-task 체크포인트)
```
