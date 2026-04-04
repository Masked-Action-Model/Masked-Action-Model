#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT_DIR}"

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
# Online control error is computed automatically during each eval round and
# reuses the existing eval demo / action norm / STPM arguments below.

CONDA_BIN="${CONDA_BIN:-/home/hebu/miniconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-maniskill_py311}"
EXP_NAME="${EXP_NAME:-test_ce}"

ENTRYPOINT="${ENTRYPOINT:-examples/baselines/diffusion_policy/train_mas_window_test.py}"
ENV_ID="${ENV_ID:-PickCube-v1}"
RAW_DEMO_H5="${RAW_DEMO_H5:-demos/data_1/data_1.h5}"
RAW_DEMO_JSON="${RAW_DEMO_JSON:-demos/data_1/data_1.json}"
PREPROCESSED_ROOT_DIR="${PREPROCESSED_ROOT_DIR:-demos/data_1_preprocessed}"
PREPROCESS_MASK_TYPE="${PREPROCESS_MASK_TYPE:-3D_points}"
PREPROCESS_RETAIN_RATIO="${PREPROCESS_RETAIN_RATIO:-0.1}"
PREPROCESS_MASK_SEQ_LEN="${PREPROCESS_MASK_SEQ_LEN:-20}"
PREPROCESS_MASK_VALUE="${PREPROCESS_MASK_VALUE:-0}"
PREPROCESS_NUM_TRAJ="${PREPROCESS_NUM_TRAJ:-}"
PREPROCESSED_DATA_PREFIX="${PREPROCESSED_DATA_PREFIX:-data_1}"
case "$PREPROCESS_MASK_TYPE" in
  pose_AnyGrasp|pose_motion_planning|points|3D_points|random_mask)
    PREPROCESS_DIR_SUFFIX="${PREPROCESS_MASK_TYPE}_${PREPROCESS_RETAIN_RATIO}"
    ;;
  2D_partial_trajectory|local_planner)
    PREPROCESS_DIR_SUFFIX="${PREPROCESS_MASK_TYPE}_seq${PREPROCESS_MASK_SEQ_LEN}"
    ;;
  *)
    PREPROCESS_DIR_SUFFIX="${PREPROCESS_MASK_TYPE}"
    ;;
esac
PREPROCESSED_DATA_DIR="${PREPROCESSED_DATA_DIR:-${PREPROCESSED_ROOT_DIR}/${PREPROCESS_DIR_SUFFIX}}"
PREPROCESSED_FILE_STEM="${PREPROCESSED_FILE_STEM:-${PREPROCESSED_DATA_PREFIX}_${PREPROCESS_DIR_SUFFIX}}"
DEMO_PATH="${DEMO_PATH:-${PREPROCESSED_DATA_DIR}/${PREPROCESSED_FILE_STEM}_train.h5}"
TRAIN_DEMO_METADATA_PATH="${TRAIN_DEMO_METADATA_PATH:-${PREPROCESSED_DATA_DIR}/${PREPROCESSED_FILE_STEM}_train.json}"
# Overfit-test default: evaluate on the same demos used for training.
TEST_DEMO_PATH="${TEST_DEMO_PATH:-${DEMO_PATH}}"
EVAL_DEMO_METADATA_PATH="${EVAL_DEMO_METADATA_PATH:-${TRAIN_DEMO_METADATA_PATH}}"
ACTION_NORM_PATH="${ACTION_NORM_PATH:-${DEMO_PATH}}"
STPM_CONFIG_PATH="${STPM_CONFIG_PATH:-STPM_PickCube/pick up the cube and place it at the goal/config.yaml}"
STPM_CKPT_PATH="${STPM_CKPT_PATH:-STPM_PickCube/pick up the cube and place it at the goal/checkpoints/reward_best.pt}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-150}"
SIM_BACKEND="${SIM_BACKEND:-cuda}"
NUM_DATALOAD_WORKERS="${NUM_DATALOAD_WORKERS:-4}"

TOTAL_ITERS="${TOTAL_ITERS:-30000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_DEMOS="${NUM_DEMOS:-10}"
NUM_EVAL_DEMOS="${NUM_EVAL_DEMOS:-5}"
NUM_EVAL_EPISODES="${NUM_EVAL_EPISODES:-5}"
NUM_EVAL_ENVS="${NUM_EVAL_ENVS:-1}"
PRED_HORIZON="${PRED_HORIZON:-16}"
LONG_WINDOW_HORIZON="${LONG_WINDOW_HORIZON:-${PRED_HORIZON}}"
EVAL_FREQ="${EVAL_FREQ:-5000}"
LOG_FREQ="${LOG_FREQ:-1000}"
SHORT_WINDOW_HORIZON="${SHORT_WINDOW_HORIZON:-8}"
MAS_LONG_ENCODE_MODE="${MAS_LONG_ENCODE_MODE:-2DConv}"
MAS_LONG_CONV_OUTPUT_DIM="${MAS_LONG_CONV_OUTPUT_DIM:-64}"
SEED="${SEED:-1}"
CUDA_FLAG="${CUDA_FLAG:---cuda}"
CAPTURE_VIDEO_FLAG="${CAPTURE_VIDEO_FLAG:---no-capture-video}"

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
  "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
    python examples/baselines/diffusion_policy/data_preprocess.py "${PREPROCESS_ARGS[@]}"
}

ensure_preprocessed_dataset

if [[ ! -f "${DEMO_PATH}" ]]; then
  echo "demo file not found: ${DEMO_PATH}" >&2
  exit 1
fi

if [[ ! -f "${TEST_DEMO_PATH}" ]]; then
  echo "test demo file not found: ${TEST_DEMO_PATH}" >&2
  exit 1
fi

if [[ -n "${EVAL_DEMO_METADATA_PATH}" && ! -f "${EVAL_DEMO_METADATA_PATH}" ]]; then
  echo "eval demo metadata file not found: ${EVAL_DEMO_METADATA_PATH}" >&2
  exit 1
fi

if [[ ! -f "${ACTION_NORM_PATH}" ]]; then
  echo "action norm file not found: ${ACTION_NORM_PATH}" >&2
  exit 1
fi

if [[ ! -f "${STPM_CONFIG_PATH}" ]]; then
  echo "STPM config file not found: ${STPM_CONFIG_PATH}" >&2
  exit 1
fi

if [[ ! -f "${STPM_CKPT_PATH}" ]]; then
  echo "STPM checkpoint file not found: ${STPM_CKPT_PATH}" >&2
  exit 1
fi

ARGS=(
  --env-id "${ENV_ID}"
  --demo-path "${DEMO_PATH}"
  --test-demo-path "${TEST_DEMO_PATH}"
  --eval-demo-metadata-path "${EVAL_DEMO_METADATA_PATH}"
  --action-norm-path "${ACTION_NORM_PATH}"
  --stpm-ckpt-path "${STPM_CKPT_PATH}"
  --stpm-config-path "${STPM_CONFIG_PATH}"
  --max-episode-steps "${MAX_EPISODE_STEPS}"
  --sim-backend "${SIM_BACKEND}"
  --num-dataload-workers "${NUM_DATALOAD_WORKERS}"
  --total-iters "${TOTAL_ITERS}"
  --batch-size "${BATCH_SIZE}"
  --num-demos "${NUM_DEMOS}"
  --num-eval-demos "${NUM_EVAL_DEMOS}"
  --num-eval-episodes "${NUM_EVAL_EPISODES}"
  --num-eval-envs "${NUM_EVAL_ENVS}"
  --pred-horizon "${PRED_HORIZON}"
  --long-window-horizon "${LONG_WINDOW_HORIZON}"
  --eval-freq "${EVAL_FREQ}"
  --log-freq "${LOG_FREQ}"
  --short-window-horizon "${SHORT_WINDOW_HORIZON}"
  --mas-long-encode-mode "${MAS_LONG_ENCODE_MODE}"
  --mas-long-conv-output-dim "${MAS_LONG_CONV_OUTPUT_DIM}"
  --seed "${SEED}"
)

if [[ -n "${EXP_NAME}" ]]; then
  ARGS+=(--exp-name "${EXP_NAME}")
fi

if [[ "${CUDA_FLAG}" == "--cuda" ]]; then
  "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
    python - <<'PY'
import sys
import torch

if not torch.cuda.is_available():
    print("ERROR: CUDA requested but torch.cuda.is_available() is False.", file=sys.stderr)
    sys.exit(2)

print(f"[cuda-check] using GPU: {torch.cuda.get_device_name(0)}")
PY
fi

exec "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
  python "${ENTRYPOINT}" \
  "${ARGS[@]}" \
  "${CUDA_FLAG}" \
  "${CAPTURE_VIDEO_FLAG}"
