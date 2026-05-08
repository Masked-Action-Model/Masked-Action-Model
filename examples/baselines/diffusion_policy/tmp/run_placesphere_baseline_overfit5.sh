#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/examples/baselines/diffusion_policy:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-maniskill}"

EXP_NAME="${EXP_NAME:-PlaceSphere_baseline_overfit5}"
SEED="${SEED:-1}"
CUDA="${CUDA:-true}"
TRACK="${TRACK:-false}"
CAPTURE_VIDEO="${CAPTURE_VIDEO:-false}"
WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME:-ManiSkill}"
WANDB_ENTITY="${WANDB_ENTITY:-}"

DEMO_H5="${DEMO_H5:-demos/PlaceSphere-v1/PlaceSphere-v1.rgb.pd_ee_pose.physx_cpu.h5}"
DEMO_JSON="${DEMO_JSON:-${DEMO_H5%.h5}.json}"
ENV_ID="${ENV_ID:-PlaceSphere-v1}"
ACTION_DIM="${ACTION_DIM:-7}"
CONTROL_MODE="${CONTROL_MODE:-pd_ee_pose}"
OBS_MODE="${OBS_MODE:-rgb}"
SIM_BACKEND="${SIM_BACKEND:-physx_cpu}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-150}"

NOISE_MODEL="${NOISE_MODEL:-Transformer}"
TOTAL_ITERS="${TOTAL_ITERS:-10000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-1e-4}"
OBS_HORIZON="${OBS_HORIZON:-2}"
ACT_HORIZON="${ACT_HORIZON:-8}"
PRED_HORIZON="${PRED_HORIZON:-16}"
NUM_DATALOAD_WORKERS="${NUM_DATALOAD_WORKERS:-0}"
DIFFUSION_STEP_EMBED_DIM="${DIFFUSION_STEP_EMBED_DIM:-64}"
DIT_HIDDEN_DIM="${DIT_HIDDEN_DIM:-512}"
DIT_NUM_BLOCKS="${DIT_NUM_BLOCKS:-6}"
DIT_DIM_FEEDFORWARD="${DIT_DIM_FEEDFORWARD:-2048}"
UNET_DIMS="${UNET_DIMS:-64 128 256}"
N_GROUPS="${N_GROUPS:-8}"

LOG_FREQ="${LOG_FREQ:-1000}"
EVAL_FREQ="${EVAL_FREQ:-5000}"
SAVE_FREQ="${SAVE_FREQ:-5000}"
NUM_DEMOS="${NUM_DEMOS:-5}"
NUM_EVAL_DEMOS="${NUM_EVAL_DEMOS:-5}"
NUM_EVAL_EPISODES="${NUM_EVAL_EPISODES:-5}"
NUM_EVAL_ENVS="${NUM_EVAL_ENVS:-1}"
DEMO_TYPE="${DEMO_TYPE:-placesphere_baseline_overfit5}"

if [[ ! -f "$DEMO_H5" ]]; then
  echo "ERROR: replayed demo h5 not found: $DEMO_H5" >&2
  echo "Run: python -m mani_skill.trajectory.replay_trajectory --traj-path demos/PlaceSphere-v1/PlaceSphere-v1.h5 --save-traj --obs-mode rgb --target-control-mode pd_ee_pose --count 5 --num-envs 1 --sim-backend physx_cpu --allow-failure" >&2
  exit 1
fi
if [[ ! -f "$DEMO_JSON" ]]; then
  echo "ERROR: replayed demo json not found: $DEMO_JSON" >&2
  exit 1
fi

ARGS=(
  --exp-name "$EXP_NAME"
  --seed "$SEED"
  --wandb-project-name "$WANDB_PROJECT_NAME"
  --env-id "$ENV_ID"
  --action-dim "$ACTION_DIM"
  --demo-path "$DEMO_H5"
  --eval-demo-path "$DEMO_H5"
  --eval-demo-metadata-path "$DEMO_JSON"
  --num-demos "$NUM_DEMOS"
  --num-eval-demos "$NUM_EVAL_DEMOS"
  --total-iters "$TOTAL_ITERS"
  --batch-size "$BATCH_SIZE"
  --lr "$LR"
  --obs-horizon "$OBS_HORIZON"
  --act-horizon "$ACT_HORIZON"
  --pred-horizon "$PRED_HORIZON"
  --diffusion-step-embed-dim "$DIFFUSION_STEP_EMBED_DIM"
  --noise-model "$NOISE_MODEL"
  --dit-hidden-dim "$DIT_HIDDEN_DIM"
  --dit-num-blocks "$DIT_NUM_BLOCKS"
  --dit-dim-feedforward "$DIT_DIM_FEEDFORWARD"
  --unet-dims $UNET_DIMS
  --n-groups "$N_GROUPS"
  --obs-mode "$OBS_MODE"
  --max-episode-steps "$MAX_EPISODE_STEPS"
  --log-freq "$LOG_FREQ"
  --eval-freq "$EVAL_FREQ"
  --save-freq "$SAVE_FREQ"
  --num-eval-episodes "$NUM_EVAL_EPISODES"
  --num-eval-envs "$NUM_EVAL_ENVS"
  --num-dataload-workers "$NUM_DATALOAD_WORKERS"
  --control-mode "$CONTROL_MODE"
  --sim-backend "$SIM_BACKEND"
  --demo-type "$DEMO_TYPE"
)

if [[ "$CUDA" == "true" ]]; then ARGS+=(--cuda); else ARGS+=(--no-cuda); fi
if [[ "$TRACK" == "true" ]]; then ARGS+=(--track); else ARGS+=(--no-track); fi
if [[ "$CAPTURE_VIDEO" == "true" ]]; then ARGS+=(--capture-video); else ARGS+=(--no-capture-video); fi
if [[ -n "$WANDB_ENTITY" ]]; then ARGS+=(--wandb-entity "$WANDB_ENTITY"); fi

python examples/baselines/diffusion_policy/tmp/train_baseline_placesphere_overfit5.py "${ARGS[@]}"
