#!/bin/bash
set -euo pipefail

# Unified RGB/RGBD diffusion-policy baseline.
# Override any value by exporting the variable before running this script.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-maniskill}"

# -----------------------------------------------------------------------------
# 1. Experiment
# -----------------------------------------------------------------------------

EXP_NAME="${EXP_NAME:-benchmark-baseline-transformer-rgb-split}"
SEED="${SEED:-1}"
TORCH_DETERMINISTIC="${TORCH_DETERMINISTIC:-true}"
CUDA="${CUDA:-true}"
TRACK="${TRACK:-false}"
WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME:-ManiSkill}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
CAPTURE_VIDEO="${CAPTURE_VIDEO:-false}"

# -----------------------------------------------------------------------------
# 2. Raw dataset and split
# -----------------------------------------------------------------------------

RAW_DEMO_H5="${RAW_DEMO_H5:-demos/exp_4/PickCube-v1/motionplanning/experiment_4.rgb.pd_ee_pose.physx_cpu.h5}"
RAW_DEMO_JSON="${RAW_DEMO_JSON:-${RAW_DEMO_H5%.h5}.json}"
SPLIT_OUTPUT_ROOT="${SPLIT_OUTPUT_ROOT:-demos/baseline_splits}"
SPLIT_OUTPUT_PREFIX="${SPLIT_OUTPUT_PREFIX:-$(basename "${RAW_DEMO_H5%.h5}")}"
SPLIT_SEED="${SPLIT_SEED:-0}"
SPLIT_OUTPUT_DIR="${SPLIT_OUTPUT_DIR:-${SPLIT_OUTPUT_ROOT}/${SPLIT_OUTPUT_PREFIX}_s${SPLIT_SEED}}"
SPLIT_NUM_TRAJ="${SPLIT_NUM_TRAJ:-}"
OVERWRITE_SPLIT="${OVERWRITE_SPLIT:-false}"

# -----------------------------------------------------------------------------
# 3. Environment
# -----------------------------------------------------------------------------

ENV_ID="${ENV_ID:-PickCube-v1}"
CONTROL_MODE="${CONTROL_MODE:-pd_ee_pose}"
OBS_MODE="${OBS_MODE:-rgb}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-100}"
SIM_BACKEND="${SIM_BACKEND:-physx_cpu}"

# -----------------------------------------------------------------------------
# 4. Model
# -----------------------------------------------------------------------------

NOISE_MODEL="${NOISE_MODEL:-Transformer}" # Transformer or Unet
DIFFUSION_STEP_EMBED_DIM="${DIFFUSION_STEP_EMBED_DIM:-64}"
UNET_DIMS="${UNET_DIMS:-64 128 256}"
N_GROUPS="${N_GROUPS:-8}"

# -----------------------------------------------------------------------------
# 5. Training
# -----------------------------------------------------------------------------

NUM_DEMOS="${NUM_DEMOS:-100}"
TOTAL_ITERS="${TOTAL_ITERS:-100000}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LR="${LR:-1e-4}"
OBS_HORIZON="${OBS_HORIZON:-2}"
ACT_HORIZON="${ACT_HORIZON:-8}"
PRED_HORIZON="${PRED_HORIZON:-16}"
NUM_DATALOAD_WORKERS="${NUM_DATALOAD_WORKERS:-0}"
ACTION_NORM_PATH="${ACTION_NORM_PATH:-}"
DEMO_TYPE="${DEMO_TYPE:-baseline_${NOISE_MODEL}}"

# -----------------------------------------------------------------------------
# 6. Eval, logging, and checkpointing
# -----------------------------------------------------------------------------

LOG_FREQ="${LOG_FREQ:-1000}"
EVAL_FREQ="${EVAL_FREQ:-5000}"
SAVE_FREQ="${SAVE_FREQ:-50000}"
NUM_EVAL_DEMOS="${NUM_EVAL_DEMOS:-100}"
NUM_EVAL_EPISODES="${NUM_EVAL_EPISODES:-${NUM_EVAL_DEMOS}}"
NUM_EVAL_ENVS="${NUM_EVAL_ENVS:-10}"

# -----------------------------------------------------------------------------
# 7. Validate required inputs
# -----------------------------------------------------------------------------

if [[ ! -f "$RAW_DEMO_H5" ]]; then
  echo "ERROR: raw demo h5 not found: $RAW_DEMO_H5" >&2
  exit 1
fi
if [[ ! -f "$RAW_DEMO_JSON" ]]; then
  echo "ERROR: raw demo json not found: $RAW_DEMO_JSON" >&2
  exit 1
fi
case "$NOISE_MODEL" in
  Transformer|Unet) ;;
  *)
    echo "ERROR: NOISE_MODEL must be Transformer or Unet, got: $NOISE_MODEL" >&2
    exit 1
    ;;
esac

# -----------------------------------------------------------------------------
# 8. Build args
# -----------------------------------------------------------------------------

ARGS=(
  --seed "$SEED"
  --wandb-project-name "$WANDB_PROJECT_NAME"
  --env-id "$ENV_ID"
  --raw-demo-h5 "$RAW_DEMO_H5"
  --raw-demo-json "$RAW_DEMO_JSON"
  --split-output-dir "$SPLIT_OUTPUT_DIR"
  --split-output-prefix "$SPLIT_OUTPUT_PREFIX"
  --split-seed "$SPLIT_SEED"
  --noise-model "$NOISE_MODEL"
  --total-iters "$TOTAL_ITERS"
  --batch-size "$BATCH_SIZE"
  --lr "$LR"
  --obs-horizon "$OBS_HORIZON"
  --act-horizon "$ACT_HORIZON"
  --pred-horizon "$PRED_HORIZON"
  --diffusion-step-embed-dim "$DIFFUSION_STEP_EMBED_DIM"
  --unet-dims $UNET_DIMS
  --n-groups "$N_GROUPS"
  --obs-mode "$OBS_MODE"
  --log-freq "$LOG_FREQ"
  --eval-freq "$EVAL_FREQ"
  --num-eval-episodes "$NUM_EVAL_EPISODES"
  --num-eval-envs "$NUM_EVAL_ENVS"
  --num-dataload-workers "$NUM_DATALOAD_WORKERS"
  --control-mode "$CONTROL_MODE"
  --sim-backend "$SIM_BACKEND"
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
if [[ "$OVERWRITE_SPLIT" == "true" ]]; then
  ARGS+=(--overwrite-split)
else
  ARGS+=(--no-overwrite-split)
fi
if [[ -n "$SPLIT_NUM_TRAJ" ]]; then
  ARGS+=(--split-num-traj "$SPLIT_NUM_TRAJ")
fi
if [[ -n "$NUM_DEMOS" ]]; then
  ARGS+=(--num-demos "$NUM_DEMOS")
fi
if [[ -n "$NUM_EVAL_DEMOS" ]]; then
  ARGS+=(--num-eval-demos "$NUM_EVAL_DEMOS")
fi
if [[ -n "$ACTION_NORM_PATH" ]]; then
  ARGS+=(--action-norm-path "$ACTION_NORM_PATH")
fi
if [[ -n "$MAX_EPISODE_STEPS" ]]; then
  ARGS+=(--max-episode-steps "$MAX_EPISODE_STEPS")
fi
if [[ -n "$SAVE_FREQ" ]]; then
  ARGS+=(--save-freq "$SAVE_FREQ")
fi
if [[ -n "$DEMO_TYPE" ]]; then
  ARGS+=(--demo-type "$DEMO_TYPE")
fi

python examples/baselines/diffusion_policy/train_baseline.py "${ARGS[@]}"
