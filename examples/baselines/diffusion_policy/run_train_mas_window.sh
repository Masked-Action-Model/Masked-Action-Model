#!/bin/bash
set -euo pipefail

# Train script for examples/baselines/diffusion_policy/train_mas_window.py
# Override any value by exporting the variable before running this script.
# Online control error is computed automatically during each eval round and
# reuses the existing eval demo / action norm / STPM arguments below.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT_DIR}"

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

# -----------------------------------------------------------------------------
# 1. Basic info
# -----------------------------------------------------------------------------

ENV_ID="${ENV_ID:-PickCube-v1}"
EXP_NAME="${EXP_NAME:-PickCube}"
SEED="${SEED:-1}"
TORCH_DETERMINISTIC="${TORCH_DETERMINISTIC:-true}"
CUDA="${CUDA:-true}"
WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME:-ManiSkill}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
TRACK="${TRACK:-false}"
CAPTURE_VIDEO="${CAPTURE_VIDEO:-true}"

# -----------------------------------------------------------------------------
# 2. Raw paths
# -----------------------------------------------------------------------------

RAW_DEMO_H5="${RAW_DEMO_H5:-demos/data_1/data_1.h5}"
RAW_DEMO_JSON="${RAW_DEMO_JSON:-demos/data_1/data_1.json}"

# -----------------------------------------------------------------------------
# 3. Preprocess params (single mask type)
# -----------------------------------------------------------------------------

PREPROCESSED_ROOT_DIR="${PREPROCESSED_ROOT_DIR:-demos/data_1_preprocessed}"
PREPROCESSED_DATA_PREFIX="${PREPROCESSED_DATA_PREFIX:-data_1}"
PREPROCESS_MASK_TYPE="${PREPROCESS_MASK_TYPE:-3D_points}"
PREPROCESS_RETAIN_RATIO="${PREPROCESS_RETAIN_RATIO:-0.5}"
PREPROCESS_MASK_SEQ_LEN="${PREPROCESS_MASK_SEQ_LEN:-20}"
PREPROCESS_MASK_VALUE="${PREPROCESS_MASK_VALUE:-0}"
PREPROCESS_NUM_TRAJ="${PREPROCESS_NUM_TRAJ:-}"

# -----------------------------------------------------------------------------
# 4. Derived preprocessed paths
# -----------------------------------------------------------------------------

case "$PREPROCESS_MASK_TYPE" in
  pose_AnyGrasp|pose_motion_planning|points|3D_points|random_mask)
    PREPROCESS_DIR_SUFFIX="${PREPROCESS_MASK_TYPE}_${PREPROCESS_RETAIN_RATIO}"
    ;;
  2D_partial_trajectory|local_planner)
    PREPROCESS_DIR_SUFFIX="${PREPROCESS_MASK_TYPE}_seq${PREPROCESS_MASK_SEQ_LEN}"
    ;;
  2D_video_trajectory|2D_image_trajectory)
    PREPROCESS_DIR_SUFFIX="${PREPROCESS_MASK_TYPE}"
    ;;
  *)
    PREPROCESS_DIR_SUFFIX="${PREPROCESS_MASK_TYPE}"
    ;;
esac
PREPROCESSED_DATA_DIR="${PREPROCESSED_DATA_DIR:-${PREPROCESSED_ROOT_DIR}/${PREPROCESS_DIR_SUFFIX}}"
PREPROCESSED_FILE_STEM="${PREPROCESSED_FILE_STEM:-${PREPROCESSED_DATA_PREFIX}_${PREPROCESS_DIR_SUFFIX}}"
DEMO_PATH="${DEMO_PATH:-${PREPROCESSED_DATA_DIR}/${PREPROCESSED_FILE_STEM}_train.h5}"
TEST_DEMO_PATH="${TEST_DEMO_PATH:-${PREPROCESSED_DATA_DIR}/${PREPROCESSED_FILE_STEM}_eval.h5}"
EVAL_DEMO_METADATA_PATH="${EVAL_DEMO_METADATA_PATH:-${PREPROCESSED_DATA_DIR}/${PREPROCESSED_FILE_STEM}_eval.json}"
TRAIN_DEMO_METADATA_PATH="${TRAIN_DEMO_METADATA_PATH:-${PREPROCESSED_DATA_DIR}/${PREPROCESSED_FILE_STEM}_train.json}"

# -----------------------------------------------------------------------------
# 5. STPM params
# -----------------------------------------------------------------------------

STPM_CONFIG_PATH="${STPM_CONFIG_PATH:-STPM_PickCube/pick up the cube and place it at the goal/config.yaml}"
STPM_CKPT_PATH="${STPM_CKPT_PATH:-STPM_PickCube/pick up the cube and place it at the goal/checkpoints/reward_best.pt}"

# -----------------------------------------------------------------------------
# 6. Train basic params
# -----------------------------------------------------------------------------

NUM_DEMOS="${NUM_DEMOS:-100}"
TOTAL_ITERS="${TOTAL_ITERS:-1000000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-1e-4}"
NUM_DATALOAD_WORKERS="${NUM_DATALOAD_WORKERS:-0}"
CONTROL_MODE="${CONTROL_MODE:-pd_ee_pose}"
OBS_MODE="${OBS_MODE:-rgb+depth}"  # rgb or rgb+depth; rgb ignores dataset depth for policy input
DEMO_TYPE="${DEMO_TYPE:-}"

# -----------------------------------------------------------------------------
# 7. MAS-window and diffusion-policy params
# -----------------------------------------------------------------------------

OBS_HORIZON="${OBS_HORIZON:-2}"
ACT_HORIZON="${ACT_HORIZON:-8}"
PRED_HORIZON="${PRED_HORIZON:-16}"
LONG_WINDOW_BACKWARD_LENGTH="${LONG_WINDOW_BACKWARD_LENGTH:-0}" #  look hisory
LONG_WINDOW_FORWARD_LENGTH="${LONG_WINDOW_FORWARD_LENGTH:-${PRED_HORIZON}}"  #  look future
DIFFUSION_STEP_EMBED_DIM="${DIFFUSION_STEP_EMBED_DIM:-64}"
SHORT_WINDOW_HORIZON="${SHORT_WINDOW_HORIZON:-8}"
MAS_LONG_ENCODE_MODE="${MAS_LONG_ENCODE_MODE:-2DConv}"
MAS_LONG_CONV_OUTPUT_DIM="${MAS_LONG_CONV_OUTPUT_DIM:-64}"
LOSS_MODE="${LOSS_MODE:-average}" #average or weighted
LOSS_MASK_AREA_WEIGHT="${LOSS_MASK_AREA_WEIGHT:-0.2}"

# -----------------------------------------------------------------------------
# 8. Eval, logging, and checkpoint params
# -----------------------------------------------------------------------------

MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-100}"
LOG_FREQ="${LOG_FREQ:-1000}"
EVAL_FREQ="${EVAL_FREQ:-1000}"
SAVE_FREQ="${SAVE_FREQ:-}"
NUM_EVAL_EPISODES="${NUM_EVAL_EPISODES:-100}"
NUM_EVAL_DEMOS="${NUM_EVAL_DEMOS:-100}"
NUM_EVAL_ENVS="${NUM_EVAL_ENVS:-10}"
INPAINTING="${INPAINTING:-false}"
EVAL_PROGRESS_BAR="${EVAL_PROGRESS_BAR:-false}"
CAPTURE_VIDEO_FREQ="${CAPTURE_VIDEO_FREQ:-10}"
ACTION_NORM_PATH="${ACTION_NORM_PATH:-${DEMO_PATH}}"
SIM_BACKEND="${SIM_BACKEND:-physx_cpu}"

# -----------------------------------------------------------------------------
# 9. Preprocess helper
# -----------------------------------------------------------------------------

ensure_preprocessed_dataset() {
  local missing=0
  for path in "$DEMO_PATH" "$TEST_DEMO_PATH" "$TRAIN_DEMO_METADATA_PATH" "$EVAL_DEMO_METADATA_PATH"; do
    if [[ ! -f "$path" ]]; then
      missing=1
      break
    fi
  done

  if [[ "$missing" -eq 0 ]]; then
    echo "[preprocess] reuse existing dataset: ${PREPROCESSED_DATA_DIR}"
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
  echo "[preprocess] generating dataset into ${PREPROCESSED_DATA_DIR}"
  PREPROCESS_ARGS=(
    --input-h5 "$RAW_DEMO_H5"
    --input-json "$RAW_DEMO_JSON"
    --output-dir "$PREPROCESSED_DATA_DIR"
    --output-prefix "$PREPROCESSED_DATA_PREFIX"
    --env-id "$ENV_ID"
    --mask-type "$PREPROCESS_MASK_TYPE"
  )
  if [[ -n "$PREPROCESS_RETAIN_RATIO" ]]; then
    PREPROCESS_ARGS+=(--retain-ratio "$PREPROCESS_RETAIN_RATIO")
  fi
  if [[ -n "$PREPROCESS_MASK_SEQ_LEN" ]]; then
    PREPROCESS_ARGS+=(--mask-seq-len "$PREPROCESS_MASK_SEQ_LEN")
  fi
  if [[ -n "$PREPROCESS_MASK_VALUE" ]]; then
    PREPROCESS_ARGS+=(--mask-value "$PREPROCESS_MASK_VALUE")
  fi
  if [[ -n "$PREPROCESS_NUM_TRAJ" ]]; then
    PREPROCESS_ARGS+=(--num-traj "$PREPROCESS_NUM_TRAJ")
  fi
  python examples/baselines/diffusion_policy/data_preprocess.py "${PREPROCESS_ARGS[@]}"
}

ensure_preprocessed_dataset

# -----------------------------------------------------------------------------
# 10. Build train args
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
  --long-window-backward-length "$LONG_WINDOW_BACKWARD_LENGTH"
  --long-window-forward-length "$LONG_WINDOW_FORWARD_LENGTH"
  --diffusion-step-embed-dim "$DIFFUSION_STEP_EMBED_DIM"
  --short-window-horizon "$SHORT_WINDOW_HORIZON"
  --mas-long-encode-mode "$MAS_LONG_ENCODE_MODE"
  --mas-long-conv-output-dim "$MAS_LONG_CONV_OUTPUT_DIM"
  --loss-mode "$LOSS_MODE"
  --loss-mask-area-weight "$LOSS_MASK_AREA_WEIGHT"
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
  --obs-mode "$OBS_MODE"
)

# bool/optional 参数单独追加，确保 tyro 能正确解析 true/false 开关。
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
if [[ "$INPAINTING" == "true" ]]; then
  ARGS+=(--inpainting)
else
  ARGS+=(--no-inpainting)
fi
if [[ "$EVAL_PROGRESS_BAR" == "true" ]]; then
  ARGS+=(--eval-progress-bar)
else
  ARGS+=(--no-eval-progress-bar)
fi
if [[ -n "$NUM_DEMOS" ]]; then
  ARGS+=(--num-demos "$NUM_DEMOS")
fi

# -----------------------------------------------------------------------------
# 11. Validate required files
# -----------------------------------------------------------------------------

if [[ ! -f "$DEMO_PATH" ]]; then
  echo "ERROR: demo file not found: $DEMO_PATH" >&2
  exit 1
fi
if [[ -z "$TEST_DEMO_PATH" ]]; then
  echo "ERROR: TEST_DEMO_PATH is required"
  exit 1
fi
if [[ ! -f "$TEST_DEMO_PATH" ]]; then
  echo "ERROR: test demo file not found: $TEST_DEMO_PATH" >&2
  exit 1
fi
if [[ -n "$EVAL_DEMO_METADATA_PATH" && ! -f "$EVAL_DEMO_METADATA_PATH" ]]; then
  echo "ERROR: eval demo metadata file not found: $EVAL_DEMO_METADATA_PATH" >&2
  exit 1
fi
if [[ ! -f "$STPM_CONFIG_PATH" ]]; then
  echo "ERROR: STPM config file not found: $STPM_CONFIG_PATH" >&2
  exit 1
fi
if [[ ! -f "$STPM_CKPT_PATH" ]]; then
  echo "ERROR: STPM checkpoint file not found: $STPM_CKPT_PATH" >&2
  exit 1
fi
if [[ -z "$ACTION_NORM_PATH" ]]; then
  echo "缺少 ACTION_NORM_PATH。" >&2
  echo "示例：" >&2
  echo "  ACTION_NORM_PATH=$DEMO_PATH \\" >&2
  echo "  bash examples/baselines/diffusion_policy/run_train_mas_window.sh" >&2
  exit 2
fi
if [[ ! -f "$ACTION_NORM_PATH" ]]; then
  echo "ERROR: action norm path not found: $ACTION_NORM_PATH" >&2
  exit 1
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

python examples/baselines/diffusion_policy/train_mas_window.py "${ARGS[@]}"
