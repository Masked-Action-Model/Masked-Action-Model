#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

# -----------------------------------------------------------------------------
# 1. Basic info
# -----------------------------------------------------------------------------

ENV_ID="${ENV_ID:-PickCube-v1}"
EXP_NAME="${EXP_NAME:-PickCube_relative_action_overfit5}"
SEED="${SEED:-1}"
TORCH_DETERMINISTIC="${TORCH_DETERMINISTIC:-true}"
CUDA="${CUDA:-true}"
WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME:-ManiSkill}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
TRACK="${TRACK:-false}"
CAPTURE_VIDEO="${CAPTURE_VIDEO:-false}"

# -----------------------------------------------------------------------------
# 2. Raw paths
# -----------------------------------------------------------------------------

RAW_DEMO_H5="${RAW_DEMO_H5:-demos/data_1/data_1.h5}"
RAW_DEMO_JSON="${RAW_DEMO_JSON:-demos/data_1/data_1.json}"

# -----------------------------------------------------------------------------
# 3. Preprocess params
# -----------------------------------------------------------------------------

PREPROCESSED_ROOT_DIR="${PREPROCESSED_ROOT_DIR:-demos/data_1_preprocessed}"
PREPROCESSED_DATA_DIR="${PREPROCESSED_DATA_DIR:-${PREPROCESSED_ROOT_DIR}/mixed}"
PREPROCESSED_DATA_PREFIX="${PREPROCESSED_DATA_PREFIX:-data_1}"
PREPROCESS_MASK_VALUE="${PREPROCESS_MASK_VALUE:-0}"
PREPROCESS_NUM_TRAJ="${PREPROCESS_NUM_TRAJ:-}"
PREPROCESS_MASK_ASSIGN_MODE="${PREPROCESS_MASK_ASSIGN_MODE:-composition}"

TRAIN_NUM_MASK_TYPE="${TRAIN_NUM_MASK_TYPE:-${NUM_MASK_TYPE:-1}}"
TRAIN_MASK_TYPE_LIST=${TRAIN_MASK_TYPE_LIST:-${MASK_TYPE_LIST:-'["random_mask"]'}}
TRAIN_MASK_COMPOSITION_LIST=${TRAIN_MASK_COMPOSITION_LIST:-${MASK_COMPOSITION_LIST:-'[1]'}}
TRAIN_MASK_RATIO_LIST=${TRAIN_MASK_RATIO_LIST:-${MASK_RATIO_LIST:-'[0.2]'}}

EVAL_NUM_MASK_TYPE="${EVAL_NUM_MASK_TYPE:-${TRAIN_NUM_MASK_TYPE}}"
EVAL_MASK_TYPE_LIST=${EVAL_MASK_TYPE_LIST:-${TRAIN_MASK_TYPE_LIST}}
EVAL_MASK_COMPOSITION_LIST=${EVAL_MASK_COMPOSITION_LIST:-${TRAIN_MASK_COMPOSITION_LIST}}
EVAL_MASK_RATIO_LIST=${EVAL_MASK_RATIO_LIST:-${TRAIN_MASK_RATIO_LIST}}

export \
  TRAIN_NUM_MASK_TYPE TRAIN_MASK_TYPE_LIST TRAIN_MASK_COMPOSITION_LIST TRAIN_MASK_RATIO_LIST \
  EVAL_NUM_MASK_TYPE EVAL_MASK_TYPE_LIST EVAL_MASK_COMPOSITION_LIST EVAL_MASK_RATIO_LIST \
  PREPROCESSED_DATA_PREFIX PREPROCESS_MASK_ASSIGN_MODE

# -----------------------------------------------------------------------------
# 4. STPM params
# -----------------------------------------------------------------------------

STPM_CONFIG_PATH="${STPM_CONFIG_PATH:-STPM_PickCube/pick up the cube and place it at the goal/config.yaml}"
STPM_CKPT_PATH="${STPM_CKPT_PATH:-STPM_PickCube/pick up the cube and place it at the goal/checkpoints/reward_best.pt}"

# -----------------------------------------------------------------------------
# 5. Overfit train/eval params
# -----------------------------------------------------------------------------

NUM_DEMOS="${NUM_DEMOS:-5}"
NUM_EVAL_DEMOS="${NUM_EVAL_DEMOS:-5}"
NUM_EVAL_EPISODES="${NUM_EVAL_EPISODES:-5}"
NUM_EVAL_ENVS="${NUM_EVAL_ENVS:-5}"
TOTAL_ITERS="${TOTAL_ITERS:-20000}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LR="${LR:-1e-4}"
NUM_DATALOAD_WORKERS="${NUM_DATALOAD_WORKERS:-0}"
CONTROL_MODE="${CONTROL_MODE:-pd_ee_delta_pose}"
DEMO_TYPE="${DEMO_TYPE:-overfit5}"

# -----------------------------------------------------------------------------
# 6. MAS and diffusion-policy params
# -----------------------------------------------------------------------------

OBS_HORIZON="${OBS_HORIZON:-2}"
ACT_HORIZON="${ACT_HORIZON:-8}"
PRED_HORIZON="${PRED_HORIZON:-16}"
LONG_WINDOW_HORIZON="${LONG_WINDOW_HORIZON:-32}"
DIFFUSION_STEP_EMBED_DIM="${DIFFUSION_STEP_EMBED_DIM:-64}"
SHORT_WINDOW_HORIZON="${SHORT_WINDOW_HORIZON:-0}"
MAS_LONG_ENCODE_MODE="${MAS_LONG_ENCODE_MODE:-2DConv}"
MAS_LONG_CONV_OUTPUT_DIM="${MAS_LONG_CONV_OUTPUT_DIM:-64}"
LOSS_MODE="${LOSS_MODE:-average}"
LOSS_MASK_AREA_WEIGHT="${LOSS_MASK_AREA_WEIGHT:-0.2}"
RELATIVE_POS_SCALE="${RELATIVE_POS_SCALE:-0.2}"
RELATIVE_ROT_SCALE="${RELATIVE_ROT_SCALE:-0.7}"
DELTA_POS_SCALE="${DELTA_POS_SCALE:-0.1}"
DELTA_ROT_SCALE="${DELTA_ROT_SCALE:-0.1}"

# -----------------------------------------------------------------------------
# 7. Eval, logging, and checkpoint params
# -----------------------------------------------------------------------------

MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-100}"
LOG_FREQ="${LOG_FREQ:-100}"
EVAL_FREQ="${EVAL_FREQ:-500}"
SAVE_FREQ="${SAVE_FREQ:-100}"
CAPTURE_VIDEO_FREQ="${CAPTURE_VIDEO_FREQ:-1}"
SIM_BACKEND="${SIM_BACKEND:-gpu}"

# -----------------------------------------------------------------------------
# 8. Derived preprocessed paths
# -----------------------------------------------------------------------------

compute_mixed_file_stem() {
  python - <<'PY'
import os
from types import SimpleNamespace

from examples.baselines.diffusion_policy.data_preprocess_mixed import (
    build_output_stem,
    normalize_split_mask_config,
)

args = SimpleNamespace(
    train_num_mask_type=int(os.environ["TRAIN_NUM_MASK_TYPE"]),
    train_mask_type_list=os.environ["TRAIN_MASK_TYPE_LIST"],
    train_mask_composition_list=os.environ["TRAIN_MASK_COMPOSITION_LIST"],
    train_mask_ratio_list=os.environ["TRAIN_MASK_RATIO_LIST"],
    eval_num_mask_type=int(os.environ["EVAL_NUM_MASK_TYPE"]),
    eval_mask_type_list=os.environ["EVAL_MASK_TYPE_LIST"],
    eval_mask_composition_list=os.environ["EVAL_MASK_COMPOSITION_LIST"],
    eval_mask_ratio_list=os.environ["EVAL_MASK_RATIO_LIST"],
    mask_assign_mode=os.environ["PREPROCESS_MASK_ASSIGN_MODE"],
)

train_mask_specs = normalize_split_mask_config(
    args,
    "train",
    mask_assign_mode=args.mask_assign_mode,
)
eval_mask_specs = normalize_split_mask_config(
    args,
    "eval",
    mask_assign_mode=args.mask_assign_mode,
)
print(
    build_output_stem(
        os.environ["PREPROCESSED_DATA_PREFIX"],
        train_mask_specs,
        eval_mask_specs,
        mask_assign_mode=args.mask_assign_mode,
    )
)
PY
}

PREPROCESSED_FILE_STEM="${PREPROCESSED_FILE_STEM:-$(compute_mixed_file_stem)}"
DEMO_PATH="${DEMO_PATH:-${PREPROCESSED_DATA_DIR}/${PREPROCESSED_FILE_STEM}_train.h5}"
TRAIN_DEMO_METADATA_PATH="${TRAIN_DEMO_METADATA_PATH:-${PREPROCESSED_DATA_DIR}/${PREPROCESSED_FILE_STEM}_train.json}"

# Overfit key: evaluation reuses exactly the selected train split and train metadata,
# so reset seeds and MAS demos are aligned with the same source episodes.
TEST_DEMO_PATH="${TEST_DEMO_PATH:-${DEMO_PATH}}"
EVAL_DEMO_METADATA_PATH="${EVAL_DEMO_METADATA_PATH:-${TRAIN_DEMO_METADATA_PATH}}"
ACTION_NORM_PATH="${ACTION_NORM_PATH:-${DEMO_PATH}}"

# -----------------------------------------------------------------------------
# 9. Preprocess helper
# -----------------------------------------------------------------------------

ensure_preprocessed_dataset() {
  local eval_h5="${PREPROCESSED_DATA_DIR}/${PREPROCESSED_FILE_STEM}_eval.h5"
  local eval_json="${PREPROCESSED_DATA_DIR}/${PREPROCESSED_FILE_STEM}_eval.json"
  local missing=0
  for path in "$DEMO_PATH" "$TRAIN_DEMO_METADATA_PATH" "$eval_h5" "$eval_json"; do
    if [[ ! -f "$path" ]]; then
      missing=1
      break
    fi
  done

  if [[ "$missing" -eq 0 ]]; then
    echo "[overfit-preprocess] reuse existing dataset: ${PREPROCESSED_DATA_DIR}"
    return
  fi

  if [[ ! -f "$RAW_DEMO_H5" ]]; then
    echo "ERROR: raw demo h5 not found: $RAW_DEMO_H5" >&2
    exit 1
  fi
  if [[ ! -f "$RAW_DEMO_JSON" ]]; then
    echo "ERROR: raw demo json not found: $RAW_DEMO_JSON" >&2
    exit 1
  fi

  mkdir -p "$PREPROCESSED_DATA_DIR"
  echo "[overfit-preprocess] generating dataset into ${PREPROCESSED_DATA_DIR}"
  PREPROCESS_ARGS=(
    --input-h5 "$RAW_DEMO_H5"
    --input-json "$RAW_DEMO_JSON"
    --output-dir "$PREPROCESSED_DATA_DIR"
    --output-prefix "$PREPROCESSED_DATA_PREFIX"
    --env-id "$ENV_ID"
    --mask-assign-mode "$PREPROCESS_MASK_ASSIGN_MODE"
    --train-num-mask-type "$TRAIN_NUM_MASK_TYPE"
    --train-mask-type-list "$TRAIN_MASK_TYPE_LIST"
    --train-mask-composition-list "$TRAIN_MASK_COMPOSITION_LIST"
    --train-mask-ratio-list "$TRAIN_MASK_RATIO_LIST"
    --eval-num-mask-type "$EVAL_NUM_MASK_TYPE"
    --eval-mask-type-list "$EVAL_MASK_TYPE_LIST"
    --eval-mask-composition-list "$EVAL_MASK_COMPOSITION_LIST"
    --eval-mask-ratio-list "$EVAL_MASK_RATIO_LIST"
    --mask-value "$PREPROCESS_MASK_VALUE"
  )
  if [[ -n "$PREPROCESS_NUM_TRAJ" ]]; then
    PREPROCESS_ARGS+=(--num-traj "$PREPROCESS_NUM_TRAJ")
  fi
  python examples/baselines/diffusion_policy/data_preprocess_mixed.py "${PREPROCESS_ARGS[@]}"
}

ensure_preprocessed_dataset

# -----------------------------------------------------------------------------
# 10. Validate required files
# -----------------------------------------------------------------------------

if [[ "$CONTROL_MODE" != "pd_ee_delta_pose" ]]; then
  echo "ERROR: overfit5 expects CONTROL_MODE=pd_ee_delta_pose, got: $CONTROL_MODE" >&2
  exit 1
fi
for path in "$DEMO_PATH" "$TEST_DEMO_PATH" "$TRAIN_DEMO_METADATA_PATH" "$EVAL_DEMO_METADATA_PATH" "$ACTION_NORM_PATH" "$STPM_CONFIG_PATH" "$STPM_CKPT_PATH"; do
  if [[ ! -f "$path" ]]; then
    echo "ERROR: required file not found: $path" >&2
    exit 1
  fi
done

if [[ "$TEST_DEMO_PATH" != "$DEMO_PATH" ]]; then
  echo "WARNING: TEST_DEMO_PATH differs from DEMO_PATH; this is no longer a strict same-demo overfit run." >&2
fi
if [[ "$EVAL_DEMO_METADATA_PATH" != "$TRAIN_DEMO_METADATA_PATH" ]]; then
  echo "WARNING: EVAL_DEMO_METADATA_PATH differs from TRAIN_DEMO_METADATA_PATH; reset seeds may not match train demos." >&2
fi

echo "[overfit5] train/eval demo path: $DEMO_PATH"
echo "[overfit5] train/eval metadata: $TRAIN_DEMO_METADATA_PATH"
echo "[overfit5] selected source demos: train=$NUM_DEMOS eval=$NUM_EVAL_DEMOS episodes=$NUM_EVAL_EPISODES envs=$NUM_EVAL_ENVS"

# -----------------------------------------------------------------------------
# 11. Build train args
# -----------------------------------------------------------------------------

ARGS=(
  --seed "$SEED"
  --wandb-project-name "$WANDB_PROJECT_NAME"
  --env-id "$ENV_ID"
  --demo-path "$DEMO_PATH"
  --test-demo-path "$TEST_DEMO_PATH"
  --eval-demo-metadata-path "$EVAL_DEMO_METADATA_PATH"
  --stpm-ckpt-path "$STPM_CKPT_PATH"
  --stpm-config-path "$STPM_CONFIG_PATH"
  --total-iters "$TOTAL_ITERS"
  --batch-size "$BATCH_SIZE"
  --lr "$LR"
  --obs-horizon "$OBS_HORIZON"
  --act-horizon "$ACT_HORIZON"
  --pred-horizon "$PRED_HORIZON"
  --long-window-horizon "$LONG_WINDOW_HORIZON"
  --diffusion-step-embed-dim "$DIFFUSION_STEP_EMBED_DIM"
  --short-window-horizon "$SHORT_WINDOW_HORIZON"
  --mas-long-encode-mode "$MAS_LONG_ENCODE_MODE"
  --mas-long-conv-output-dim "$MAS_LONG_CONV_OUTPUT_DIM"
  --loss-mode "$LOSS_MODE"
  --loss-mask-area-weight "$LOSS_MASK_AREA_WEIGHT"
  --relative-pos-scale "$RELATIVE_POS_SCALE"
  --relative-rot-scale "$RELATIVE_ROT_SCALE"
  --delta-pos-scale "$DELTA_POS_SCALE"
  --delta-rot-scale "$DELTA_ROT_SCALE"
  --log-freq "$LOG_FREQ"
  --eval-freq "$EVAL_FREQ"
  --num-eval-episodes "$NUM_EVAL_EPISODES"
  --num-eval-demos "$NUM_EVAL_DEMOS"
  --num-eval-envs "$NUM_EVAL_ENVS"
  --capture-video-freq "$CAPTURE_VIDEO_FREQ"
  --action-norm-path "$ACTION_NORM_PATH"
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
if [[ -n "$MAX_EPISODE_STEPS" ]]; then
  ARGS+=(--max-episode-steps "$MAX_EPISODE_STEPS")
fi
if [[ -n "$SAVE_FREQ" ]]; then
  ARGS+=(--save-freq "$SAVE_FREQ")
fi
if [[ -n "$DEMO_TYPE" ]]; then
  ARGS+=(--demo-type "$DEMO_TYPE")
fi

python examples/baselines/diffusion_policy/train_relative_action_overfit5.py "${ARGS[@]}"
