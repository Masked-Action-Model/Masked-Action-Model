#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

CONDA_BIN="${CONDA_BIN:-/home/hebu/miniconda3/envs/maniskill_py311/bin}"
if [[ -d "${CONDA_BIN}" ]]; then
  export PATH="${CONDA_BIN}:${PATH}"
fi

DATASET_PATH="${DATASET_PATH:-franka_train/data/franka_real.h5}"
if [[ ! -f "${DATASET_PATH}" ]]; then
  echo "ERROR: Franka dataset h5 not found: ${DATASET_PATH}" >&2
  exit 1
fi

DATASET_PATH="${DATASET_PATH}" \
TASK_NAME="${TASK_NAME:-franka_real}" \
TASK_DESCRIPTION="${TASK_DESCRIPTION:-pick up a cup and place it next to the plate}" \
OUTPUT_DIR="${OUTPUT_DIR:-STPM_franka}" \
STATE_PATHS="${STATE_PATHS:-['obs/agent/qpos','obs/agent/qvel','obs/extra/tcp_pose']}" \
CAMERA_NAMES="${CAMERA_NAMES:-['base_camera']}" \
CAMERA_POSES_JSON="${CAMERA_POSES_JSON:-}" \
VISION_CKPT="${VISION_CKPT:-pretrained/clip-vit-base-patch32}" \
SEED="${SEED:-42}" \
DEVICE="${DEVICE:-cuda}" \
BATCH_SIZE="${BATCH_SIZE:-32}" \
NUM_WORKERS="${NUM_WORKERS:-0}" \
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-${BATCH_SIZE:-32}}" \
VAL_NUM_WORKERS="${VAL_NUM_WORKERS:-${NUM_WORKERS:-0}}" \
D_MODEL="${D_MODEL:-768}" \
N_LAYERS="${N_LAYERS:-8}" \
N_HEADS="${N_HEADS:-12}" \
DROPOUT="${DROPOUT:-0.1}" \
N_OBS_STEPS="${N_OBS_STEPS:-6}" \
FRAME_GAP="${FRAME_GAP:-2}" \
NO_STATE="${NO_STATE:-false}" \
RESUME_TRAINING="${RESUME_TRAINING:-false}" \
MODEL_PATH="${MODEL_PATH:-}" \
LR="${LR:-5e-5}" \
WEIGHT_DECAY="${WEIGHT_DECAY:-5e-3}" \
BETA1="${BETA1:-0.9}" \
BETA2="${BETA2:-0.95}" \
EPS="${EPS:-1e-8}" \
WARMUP_STEPS="${WARMUP_STEPS:-1000}" \
TOTAL_STEPS="${TOTAL_STEPS:-100000}" \
NUM_EPOCHS="${NUM_EPOCHS:-2}" \
GRAD_CLIP="${GRAD_CLIP:-1.0}" \
LOG_EVERY="${LOG_EVERY:-50}" \
EVAL_EVERY="${EVAL_EVERY:-1}" \
SAVE_EVERY="${SAVE_EVERY:-5000}" \
VAL_PORTION="${VAL_PORTION:-0.1}" \
PREPARE_ONLY="${PREPARE_ONLY:-false}" \
bash STPM/run_train_stpm.sh
