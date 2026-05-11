#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/STPM:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-maniskill}"

PYTHON="${PYTHON:-python}"

STPM_CKPT_PATH="${STPM_CKPT_PATH:-STPM_PickCube/checkpoints/reward_best.pt}"
STPM_CONFIG_PATH="${STPM_CONFIG_PATH:-STPM_PickCube/config.yaml}"
EVAL_DATASET_PATH="${EVAL_DATASET_PATH:-demos/stpm_offline_eval_new20/PickCube-v1/motionplanning/pickcube_new20_seed10000_raw.rgbd.pd_ee_pose.physx_cpu.h5}"
NUM_EVAL_DEMO="${NUM_EVAL_DEMO:-20}"

BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-0}"
DEVICE="${DEVICE:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
PROGRESS_BAR="${PROGRESS_BAR:-true}"

ARGS=(
  --checkpoint "${STPM_CKPT_PATH}"
  --config "${STPM_CONFIG_PATH}"
  --dataset "${EVAL_DATASET_PATH}"
  --num-eval-demo "${NUM_EVAL_DEMO}"
  --batch-size "${BATCH_SIZE}"
  --num-workers "${NUM_WORKERS}"
)

if [[ -n "${DEVICE}" ]]; then
  ARGS+=(--device "${DEVICE}")
fi

if [[ -n "${OUTPUT_DIR}" ]]; then
  ARGS+=(--output-dir "${OUTPUT_DIR}")
fi

if [[ "${PROGRESS_BAR}" == "true" ]]; then
  ARGS+=(--progress-bar)
else
  ARGS+=(--no-progress-bar)
fi

"${PYTHON}" STPM/eval_offline.py "${ARGS[@]}"
