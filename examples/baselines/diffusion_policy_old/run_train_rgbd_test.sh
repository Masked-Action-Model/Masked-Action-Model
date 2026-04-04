#!/bin/bash
set -euo pipefail

# Train script for examples/baselines/diffusion_policy_old/train_rgbd_test.py
# Override any value by exporting the variable before running this script.

EXP_NAME="${EXP_NAME:-test_0}"
SEED="${SEED:-1}"
TORCH_DETERMINISTIC="${TORCH_DETERMINISTIC:-true}"
CUDA="${CUDA:-true}"
TRACK="${TRACK:-false}"
WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME:-ManiSkill}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
CAPTURE_VIDEO="${CAPTURE_VIDEO:-true}"

ENV_ID="${ENV_ID:-PickCube-v1}"
DEMO_PATH="${DEMO_PATH:-demos/data_1/data_1_concat_3D_points_0.03_train.h5}"
NUM_DEMOS="${NUM_DEMOS:-10}"
NUM_EVAL_DEMOS="${NUM_EVAL_DEMOS:-5}"
EVAL_DEMO_SELECTION_SEED="${EVAL_DEMO_SELECTION_SEED:-$SEED}"
TOTAL_ITERS="${TOTAL_ITERS:-30000}"
BATCH_SIZE="${BATCH_SIZE:-32}"

LR="${LR:-1e-4}"
OBS_HORIZON="${OBS_HORIZON:-2}"
ACT_HORIZON="${ACT_HORIZON:-8}"
PRED_HORIZON="${PRED_HORIZON:-16}"
DIFFUSION_STEP_EMBED_DIM="${DIFFUSION_STEP_EMBED_DIM:-64}"

OBS_MODE="${OBS_MODE:-rgb+depth}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-100}"
LOG_FREQ="${LOG_FREQ:-1000}"
EVAL_FREQ="${EVAL_FREQ:-5000}"
SAVE_FREQ="${SAVE_FREQ:-}"
NUM_EVAL_EPISODES="${NUM_EVAL_EPISODES:-5}"
NUM_EVAL_ENVS="${NUM_EVAL_ENVS:-1}"
EVAL_RESET_SEED="${EVAL_RESET_SEED:-}"
ACTION_NORM_PATH="${ACTION_NORM_PATH:-demos/data_1/data_1_norm.json}"
SIM_BACKEND="${SIM_BACKEND:-physx_cpu}"
NUM_DATALOAD_WORKERS="${NUM_DATALOAD_WORKERS:-0}"
CONTROL_MODE="${CONTROL_MODE:-pd_ee_pose}"
DEMO_TYPE="${DEMO_TYPE:-}"

ARGS=(
  --seed "$SEED"
  --wandb-project-name "$WANDB_PROJECT_NAME"
  --env-id "$ENV_ID"
  --demo-path "$DEMO_PATH"
  --total-iters "$TOTAL_ITERS"
  --batch-size "$BATCH_SIZE"
  --lr "$LR"
  --obs-horizon "$OBS_HORIZON"
  --act-horizon "$ACT_HORIZON"
  --pred-horizon "$PRED_HORIZON"
  --diffusion-step-embed-dim "$DIFFUSION_STEP_EMBED_DIM"
  --obs-mode "$OBS_MODE"
  --log-freq "$LOG_FREQ"
  --eval-freq "$EVAL_FREQ"
  --num-eval-episodes "$NUM_EVAL_EPISODES"
  --num-eval-envs "$NUM_EVAL_ENVS"
  --sim-backend "$SIM_BACKEND"
  --num-dataload-workers "$NUM_DATALOAD_WORKERS"
  --control-mode "$CONTROL_MODE"
)

if [[ -n "$EXP_NAME" ]]; then
  ARGS+=(--exp-name "$EXP_NAME")
fi
if [[ "$TORCH_DETERMINISTIC" == "true" ]]; then
  ARGS+=(--torch-deterministic)
else
  ARGS+=(--no-torch-deterministic)
fi
if [[ "$CUDA" == "true" ]]; then
  ARGS+=(--cuda)
else
  ARGS+=(--no-cuda)
fi
if [[ "$TRACK" == "true" ]]; then
  ARGS+=(--track)
else
  ARGS+=(--no-track)
fi
if [[ -n "$WANDB_ENTITY" ]]; then
  ARGS+=(--wandb-entity "$WANDB_ENTITY")
fi
if [[ "$CAPTURE_VIDEO" == "true" ]]; then
  ARGS+=(--capture-video)
else
  ARGS+=(--no-capture-video)
fi
if [[ -n "$NUM_DEMOS" ]]; then
  ARGS+=(--num-demos "$NUM_DEMOS")
fi
if [[ -n "$NUM_EVAL_DEMOS" ]]; then
  ARGS+=(--num-eval-demos "$NUM_EVAL_DEMOS")
fi
if [[ -n "$EVAL_DEMO_SELECTION_SEED" ]]; then
  ARGS+=(--eval-demo-selection-seed "$EVAL_DEMO_SELECTION_SEED")
fi
if [[ -n "$EVAL_RESET_SEED" ]]; then
  ARGS+=(--eval-reset-seed "$EVAL_RESET_SEED")
fi
if [[ -z "$ACTION_NORM_PATH" ]]; then
  echo "缺少 ACTION_NORM_PATH（动作归一化参数 json）。" >&2
  echo "示例：" >&2
  echo "  ACTION_NORM_PATH=demos/data_1/data_1_norm.json \\" >&2
  echo "  bash examples/baselines/diffusion_policy_old/run_train_rgbd_test.sh" >&2
  exit 2
fi
ARGS+=(--action-norm-path "$ACTION_NORM_PATH")
if [[ -n "$MAX_EPISODE_STEPS" ]]; then
  ARGS+=(--max-episode-steps "$MAX_EPISODE_STEPS")
fi
if [[ -n "$SAVE_FREQ" ]]; then
  ARGS+=(--save-freq "$SAVE_FREQ")
fi
if [[ -n "$DEMO_TYPE" ]]; then
  ARGS+=(--demo-type "$DEMO_TYPE")
fi

python examples/baselines/diffusion_policy_old/train_rgbd_test.py "${ARGS[@]}"
