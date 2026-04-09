#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT_DIR}"

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

EXP_NAME="${EXP_NAME:-PickCube_window_mixed}"
SEED="${SEED:-1}"
TORCH_DETERMINISTIC="${TORCH_DETERMINISTIC:-true}"
CUDA="${CUDA:-true}"
TRACK="${TRACK:-false}"
WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME:-ManiSkill}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
CAPTURE_VIDEO="${CAPTURE_VIDEO:-false}"

ENV_ID="${ENV_ID:-PickCube-v1}"
RAW_DEMO_H5="${RAW_DEMO_H5:-demos/data_1/data_1.h5}"
RAW_DEMO_JSON="${RAW_DEMO_JSON:-demos/data_1/data_1.json}"
PREPROCESSED_ROOT_DIR="${PREPROCESSED_ROOT_DIR:-demos/data_1_preprocessed}"
PREPROCESSED_DATA_DIR="${PREPROCESSED_DATA_DIR:-${PREPROCESSED_ROOT_DIR}/mixed}"
PREPROCESSED_DATA_PREFIX="${PREPROCESSED_DATA_PREFIX:-data_1}"

TRAIN_NUM_MASK_TYPE="${TRAIN_NUM_MASK_TYPE:-${NUM_MASK_TYPE:-2}}"
TRAIN_MASK_TYPE_LIST=${TRAIN_MASK_TYPE_LIST:-${MASK_TYPE_LIST:-'["random_mask","points"]'}}
TRAIN_MASK_TYPE_RATIO_LIST=${TRAIN_MASK_TYPE_RATIO_LIST:-${MASK_TYPE_RATIO_LIST:-'[0.5,0.5]'}}
TRAIN_MASK_PARAM_LIST=${TRAIN_MASK_PARAM_LIST:-${MASK_PARAM_LIST:-'[0.2,0.2]'}}
EVAL_NUM_MASK_TYPE="${EVAL_NUM_MASK_TYPE:-${TRAIN_NUM_MASK_TYPE}}"
EVAL_MASK_TYPE_LIST=${EVAL_MASK_TYPE_LIST:-${TRAIN_MASK_TYPE_LIST}}
EVAL_MASK_TYPE_RATIO_LIST=${EVAL_MASK_TYPE_RATIO_LIST:-${TRAIN_MASK_TYPE_RATIO_LIST}}
EVAL_MASK_PARAM_LIST=${EVAL_MASK_PARAM_LIST:-${TRAIN_MASK_PARAM_LIST}}
PREPROCESS_MASK_VALUE="${PREPROCESS_MASK_VALUE:-0}"
PREPROCESS_NUM_TRAJ="${PREPROCESS_NUM_TRAJ:-}"

STPM_CONFIG_PATH="${STPM_CONFIG_PATH:-STPM_PickCube/pick up the cube and place it at the goal/config.yaml}"
STPM_CKPT_PATH="${STPM_CKPT_PATH:-STPM_PickCube/pick up the cube and place it at the goal/checkpoints/reward_best.pt}"

NUM_DEMOS="${NUM_DEMOS:-100}"
TOTAL_ITERS="${TOTAL_ITERS:-200}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-1e-4}"
OBS_HORIZON="${OBS_HORIZON:-2}"
ACT_HORIZON="${ACT_HORIZON:-8}"
PRED_HORIZON="${PRED_HORIZON:-16}"
LONG_WINDOW_HORIZON="${LONG_WINDOW_HORIZON:-${PRED_HORIZON}}"
DIFFUSION_STEP_EMBED_DIM="${DIFFUSION_STEP_EMBED_DIM:-64}"
SHORT_WINDOW_HORIZON="${SHORT_WINDOW_HORIZON:-2}"
MAS_LONG_ENCODE_MODE="${MAS_LONG_ENCODE_MODE:-2DConv}"
MAS_LONG_CONV_OUTPUT_DIM="${MAS_LONG_CONV_OUTPUT_DIM:-64}"

MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-100}"
LOG_FREQ="${LOG_FREQ:-100}"
EVAL_FREQ="${EVAL_FREQ:-100}"
SAVE_FREQ="${SAVE_FREQ:-100}"
NUM_EVAL_EPISODES="${NUM_EVAL_EPISODES:-100}"
NUM_EVAL_DEMOS="${NUM_EVAL_DEMOS:-100}"
NUM_EVAL_ENVS="${NUM_EVAL_ENVS:-10}"
SIM_BACKEND="${SIM_BACKEND:-gpu}"
NUM_DATALOAD_WORKERS="${NUM_DATALOAD_WORKERS:-0}"
CONTROL_MODE="${CONTROL_MODE:-pd_ee_pose}"
DEMO_TYPE="${DEMO_TYPE:-}"

export \
  TRAIN_NUM_MASK_TYPE TRAIN_MASK_TYPE_LIST TRAIN_MASK_TYPE_RATIO_LIST TRAIN_MASK_PARAM_LIST \
  EVAL_NUM_MASK_TYPE EVAL_MASK_TYPE_LIST EVAL_MASK_TYPE_RATIO_LIST EVAL_MASK_PARAM_LIST \
  PREPROCESSED_DATA_PREFIX

compute_mixed_file_stem() {
  python -c 'import os; from types import SimpleNamespace; from examples.baselines.diffusion_policy.data_preprocess_mixed import build_output_stem, normalize_split_mask_config; args = SimpleNamespace(train_num_mask_type=int(os.environ["TRAIN_NUM_MASK_TYPE"]), train_mask_type_list=os.environ["TRAIN_MASK_TYPE_LIST"], train_mask_type_ratio_list=os.environ["TRAIN_MASK_TYPE_RATIO_LIST"], train_mask_param_list=os.environ["TRAIN_MASK_PARAM_LIST"], eval_num_mask_type=int(os.environ["EVAL_NUM_MASK_TYPE"]), eval_mask_type_list=os.environ["EVAL_MASK_TYPE_LIST"], eval_mask_type_ratio_list=os.environ["EVAL_MASK_TYPE_RATIO_LIST"], eval_mask_param_list=os.environ["EVAL_MASK_PARAM_LIST"]); print(build_output_stem(os.environ["PREPROCESSED_DATA_PREFIX"], normalize_split_mask_config(args, "train"), normalize_split_mask_config(args, "eval")))'
}

PREPROCESSED_FILE_STEM="${PREPROCESSED_FILE_STEM:-$(compute_mixed_file_stem)}"
DEMO_PATH="${DEMO_PATH:-${PREPROCESSED_DATA_DIR}/${PREPROCESSED_FILE_STEM}_train.h5}"
TEST_DEMO_PATH="${TEST_DEMO_PATH:-${PREPROCESSED_DATA_DIR}/${PREPROCESSED_FILE_STEM}_eval.h5}"
EVAL_DEMO_METADATA_PATH="${EVAL_DEMO_METADATA_PATH:-${PREPROCESSED_DATA_DIR}/${PREPROCESSED_FILE_STEM}_eval.json}"
TRAIN_DEMO_METADATA_PATH="${TRAIN_DEMO_METADATA_PATH:-${PREPROCESSED_DATA_DIR}/${PREPROCESSED_FILE_STEM}_train.json}"
ACTION_NORM_PATH="${ACTION_NORM_PATH:-${DEMO_PATH}}"

ensure_preprocessed_dataset() {
  local missing=0
  for path in "$DEMO_PATH" "$TEST_DEMO_PATH" "$TRAIN_DEMO_METADATA_PATH" "$EVAL_DEMO_METADATA_PATH"; do
    if [[ ! -f "$path" ]]; then
      missing=1
      break
    fi
  done

  if [[ "$missing" -eq 0 ]]; then
    echo "[mixed-preprocess] reuse existing dataset: ${PREPROCESSED_DATA_DIR}"
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
  echo "[mixed-preprocess] generating dataset into ${PREPROCESSED_DATA_DIR}"
  PREPROCESS_ARGS=(
    --input-h5 "$RAW_DEMO_H5"
    --input-json "$RAW_DEMO_JSON"
    --output-dir "$PREPROCESSED_DATA_DIR"
    --output-prefix "$PREPROCESSED_DATA_PREFIX"
    --env-id "$ENV_ID"
    --train-num-mask-type "$TRAIN_NUM_MASK_TYPE"
    --train-mask-type-list "$TRAIN_MASK_TYPE_LIST"
    --train-mask-type-ratio-list "$TRAIN_MASK_TYPE_RATIO_LIST"
    --train-mask-param-list "$TRAIN_MASK_PARAM_LIST"
    --eval-num-mask-type "$EVAL_NUM_MASK_TYPE"
    --eval-mask-type-list "$EVAL_MASK_TYPE_LIST"
    --eval-mask-type-ratio-list "$EVAL_MASK_TYPE_RATIO_LIST"
    --eval-mask-param-list "$EVAL_MASK_PARAM_LIST"
    --mask-value "$PREPROCESS_MASK_VALUE"
  )
  if [[ -n "$PREPROCESS_NUM_TRAJ" ]]; then
    PREPROCESS_ARGS+=(--num-traj "$PREPROCESS_NUM_TRAJ")
  fi
  python examples/baselines/diffusion_policy/data_preprocess_mixed.py "${PREPROCESS_ARGS[@]}"
}

ensure_preprocessed_dataset

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
  --log-freq "$LOG_FREQ"
  --eval-freq "$EVAL_FREQ"
  --num-eval-episodes "$NUM_EVAL_EPISODES"
  --num-eval-demos "$NUM_EVAL_DEMOS"
  --num-eval-envs "$NUM_EVAL_ENVS"
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
if [[ ! -f "$DEMO_PATH" ]]; then
  echo "ERROR: demo file not found: $DEMO_PATH" >&2
  exit 1
fi
if [[ -z "$TEST_DEMO_PATH" ]]; then
  echo "ERROR: TEST_DEMO_PATH is required" >&2
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
if [[ -z "$ACTION_NORM_PATH" || ! -f "$ACTION_NORM_PATH" ]]; then
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

python examples/baselines/diffusion_policy/train_mas_window_mixed.py "${ARGS[@]}"
