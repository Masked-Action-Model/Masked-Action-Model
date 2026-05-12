#!/usr/bin/env bash
set -euo pipefail

DEMO_PATH="${DEMO_PATH:-franka_train/data/franka_real_random_mask_0.2_train.h5}"
EXP_NAME="${EXP_NAME:-FrankaReal_subgoal}"
TOTAL_ITERS="${TOTAL_ITERS:-100000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
SAVE_START_ITER="${SAVE_START_ITER:-0}"
SAVE_FREQ="${SAVE_FREQ:-5000}"
LOG_FREQ="${LOG_FREQ:-1000}"
OBS_MODE="${OBS_MODE:-rgb}"
ACTION_DIM="${ACTION_DIM:-7}"
PYTHON_BIN="${PYTHON_BIN:-python}"
if [[ "$PYTHON_BIN" == "python" && -x /home/hebu/miniconda3/envs/maniskill_py311/bin/python ]]; then
  PYTHON_BIN=/home/hebu/miniconda3/envs/maniskill_py311/bin/python
fi

"$PYTHON_BIN" franka_train/train_subgoal_franka.py \
  --exp-name "$EXP_NAME" \
  --demo-path "$DEMO_PATH" \
  --total-iters "$TOTAL_ITERS" \
  --batch-size "$BATCH_SIZE" \
  --save-start-iter "$SAVE_START_ITER" \
  --save-freq "$SAVE_FREQ" \
  --log-freq "$LOG_FREQ" \
  --obs-mode "$OBS_MODE" \
  --action-dim "$ACTION_DIM"
