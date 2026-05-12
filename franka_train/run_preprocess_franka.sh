#!/usr/bin/env bash
set -euo pipefail

INPUT_H5="${INPUT_H5:-franka_train/data/franka_real.h5}"
OUTPUT_H5="${OUTPUT_H5:-}"
OUTPUT_DIR="${OUTPUT_DIR:-franka_train/data}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-franka_real}"
ACTION_DIM="${ACTION_DIM:-7}"
MASK_TYPE="${MASK_TYPE:-random_mask}"
RETAIN_RATIO="${RETAIN_RATIO:-0.2}"
MASK_SEQ_LEN="${MASK_SEQ_LEN:-20}"
MASK_SEED="${MASK_SEED:-0}"
ENV_ID="${ENV_ID:-FrankaReal-v1}"
CONTROL_MODE="${CONTROL_MODE:-pd_ee_pose}"
PYTHON_BIN="${PYTHON_BIN:-python}"
if [[ "$PYTHON_BIN" == "python" && -x /home/hebu/miniconda3/envs/maniskill_py311/bin/python ]]; then
  PYTHON_BIN=/home/hebu/miniconda3/envs/maniskill_py311/bin/python
fi

cmd=(
  "$PYTHON_BIN" franka_train/preprocess_franka.py
  --input-h5 "$INPUT_H5"
  --output-dir "$OUTPUT_DIR"
  --output-prefix "$OUTPUT_PREFIX"
  --env-id "$ENV_ID"
  --control-mode "$CONTROL_MODE"
  --action-dim "$ACTION_DIM"
  --mask-type "$MASK_TYPE"
  --retain-ratio "$RETAIN_RATIO"
  --mask-seq-len "$MASK_SEQ_LEN"
  --mask-seed "$MASK_SEED"
)

if [[ -n "$OUTPUT_H5" ]]; then
  cmd+=(--output-h5 "$OUTPUT_H5")
fi
if [[ "${OVERWRITE:-false}" == "true" ]]; then
  cmd+=(--overwrite)
fi
if [[ -n "${NUM_TRAJ:-}" ]]; then
  cmd+=(--num-traj "$NUM_TRAJ")
fi

"${cmd[@]}"
