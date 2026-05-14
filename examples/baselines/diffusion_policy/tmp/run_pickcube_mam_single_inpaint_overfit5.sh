#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/examples/baselines/diffusion_policy:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-maniskill}"

BASE_DIR="${BASE_DIR:-examples/baselines/diffusion_policy/tmp/pickcube_overfit5}"
DATA_DIR="${DATA_DIR:-${BASE_DIR}/data/mam_single_random_mask_0.2}"
LOG_DIR="${LOG_DIR:-${BASE_DIR}/logs}"
mkdir -p "${DATA_DIR}" "${LOG_DIR}"

RAW_DEMO_H5="${RAW_DEMO_H5:-demos/data_1/data_1.h5}"
RAW_DEMO_JSON="${RAW_DEMO_JSON:-demos/data_1/data_1.json}"
DEMO_PATH="${DEMO_PATH:-${DATA_DIR}/pickcube_mam_single_inpaint_overfit5_train.h5}"
TRAIN_DEMO_METADATA_PATH="${TRAIN_DEMO_METADATA_PATH:-${DATA_DIR}/pickcube_mam_single_inpaint_overfit5_train.json}"
TEST_DEMO_PATH="${TEST_DEMO_PATH:-${DATA_DIR}/pickcube_mam_single_inpaint_overfit5_eval.h5}"
EVAL_DEMO_METADATA_PATH="${EVAL_DEMO_METADATA_PATH:-${DATA_DIR}/pickcube_mam_single_inpaint_overfit5_eval.json}"
ACTION_NORM_PATH="${ACTION_NORM_PATH:-${DEMO_PATH}}"

MASK_TYPE_LIST="${MASK_TYPE_LIST:-[\"random_mask\"]}"
MASK_RATIO_LIST="${MASK_RATIO_LIST:-[0.2]}"
MASK_COMPOSITION_LIST="${MASK_COMPOSITION_LIST:-[1.0]}"
PREPROCESS_MASK_ASSIGN_MODE="${PREPROCESS_MASK_ASSIGN_MODE:-composition}"
PREPROCESS_MASK_VALUE="${PREPROCESS_MASK_VALUE:-0}"
MASK_SEED="${MASK_SEED:-0}"

if [[ ! -f "${DEMO_PATH}" || ! -f "${TEST_DEMO_PATH}" || ! -f "${TRAIN_DEMO_METADATA_PATH}" || ! -f "${EVAL_DEMO_METADATA_PATH}" ]]; then
  python examples/baselines/diffusion_policy/tmp/prepare_placesphere_overfit5_data.py \
    --input-h5 "${RAW_DEMO_H5}" \
    --input-json "${RAW_DEMO_JSON}" \
    --train-h5 "${DEMO_PATH}" \
    --train-json "${TRAIN_DEMO_METADATA_PATH}" \
    --eval-h5 "${TEST_DEMO_PATH}" \
    --eval-json "${EVAL_DEMO_METADATA_PATH}" \
    --env-id PickCube-v1 \
    --num-traj 5 \
    --action-dim 7 \
    --mask-assign-mode "${PREPROCESS_MASK_ASSIGN_MODE}" \
    --train-num-mask-type 1 \
    --train-mask-type-list "${MASK_TYPE_LIST}" \
    --train-mask-composition-list "${MASK_COMPOSITION_LIST}" \
    --train-mask-ratio-list "${MASK_RATIO_LIST}" \
    --eval-num-mask-type 1 \
    --eval-mask-type-list "${MASK_TYPE_LIST}" \
    --eval-mask-composition-list "${MASK_COMPOSITION_LIST}" \
    --eval-mask-ratio-list "${MASK_RATIO_LIST}" \
    --mask-value "${PREPROCESS_MASK_VALUE}" \
    --mask-seed "${MASK_SEED}" \
    --overwrite
fi

export \
  EXP_NAME="${EXP_NAME:-PickCube_mam_single_inpaint_overfit5_20k}" \
  ENV_ID="${ENV_ID:-PickCube-v1}" \
  ACTION_DIM="${ACTION_DIM:-7}" \
  RAW_DEMO_H5 RAW_DEMO_JSON \
  PREPROCESSED_ROOT_DIR="${PREPROCESSED_ROOT_DIR:-${BASE_DIR}/data}" \
  PREPROCESSED_DATA_DIR="${DATA_DIR}" \
  PREPROCESSED_DATA_PREFIX="${PREPROCESSED_DATA_PREFIX:-pickcube_mam_single_inpaint_overfit5}" \
  DEMO_PATH TRAIN_DEMO_METADATA_PATH TEST_DEMO_PATH EVAL_DEMO_METADATA_PATH ACTION_NORM_PATH \
  MASK_TYPE_LIST MASK_RATIO_LIST MASK_COMPOSITION_LIST PREPROCESS_MASK_ASSIGN_MODE PREPROCESS_MASK_VALUE \
  STPM_CONFIG_PATH="${STPM_CONFIG_PATH:-STPM_PickCube/config.yaml}" \
  STPM_CKPT_PATH="${STPM_CKPT_PATH:-STPM_PickCube/checkpoints/reward_best.pt}" \
  SEED="${SEED:-1}" \
  TORCH_DETERMINISTIC="${TORCH_DETERMINISTIC:-true}" \
  CUDA="${CUDA:-true}" \
  TRACK="${TRACK:-false}" \
  CAPTURE_VIDEO="${CAPTURE_VIDEO:-false}" \
  CONTROL_MODE="${CONTROL_MODE:-pd_ee_pose}" \
  OBS_MODE="${OBS_MODE:-rgb}" \
  SIM_BACKEND="${SIM_BACKEND:-physx_cpu}" \
  MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-50}" \
  NOISE_MODEL="${NOISE_MODEL:-Transformer}" \
  DIFFUSION_STEP_EMBED_DIM="${DIFFUSION_STEP_EMBED_DIM:-64}" \
  DIT_HIDDEN_DIM="${DIT_HIDDEN_DIM:-512}" \
  DIT_NUM_BLOCKS="${DIT_NUM_BLOCKS:-6}" \
  DIT_DIM_FEEDFORWARD="${DIT_DIM_FEEDFORWARD:-2048}" \
  UNET_DIMS="${UNET_DIMS:-64 128 256}" \
  N_GROUPS="${N_GROUPS:-8}" \
  TOTAL_ITERS="${TOTAL_ITERS:-20000}" \
  BATCH_SIZE="${BATCH_SIZE:-64}" \
  LR="${LR:-1e-4}" \
  OBS_HORIZON="${OBS_HORIZON:-2}" \
  ACT_HORIZON="${ACT_HORIZON:-8}" \
  PRED_HORIZON="${PRED_HORIZON:-16}" \
  NUM_DATALOAD_WORKERS="${NUM_DATALOAD_WORKERS:-2}" \
  LONG_WINDOW_BACKWARD_LENGTH="${LONG_WINDOW_BACKWARD_LENGTH:-0}" \
  LONG_WINDOW_FORWARD_LENGTH="${LONG_WINDOW_FORWARD_LENGTH:-16}" \
  SHORT_WINDOW_HORIZON="${SHORT_WINDOW_HORIZON:-2}" \
  MAS_LONG_ENCODE_MODE="${MAS_LONG_ENCODE_MODE:-2DConv}" \
  MAS_LONG_CONV_OUTPUT_DIM="${MAS_LONG_CONV_OUTPUT_DIM:-64}" \
  LOSS_MODE="${LOSS_MODE:-average}" \
  LOSS_MASK_AREA_WEIGHT="${LOSS_MASK_AREA_WEIGHT:-0.2}" \
  LOG_FREQ="${LOG_FREQ:-1000}" \
  EVAL_FREQ="${EVAL_FREQ:-5000}" \
  SAVE_FREQ="${SAVE_FREQ:-5000}" \
  NUM_DEMOS="${NUM_DEMOS:-5}" \
  NUM_EVAL_DEMOS="${NUM_EVAL_DEMOS:-5}" \
  NUM_EVAL_EPISODES="${NUM_EVAL_EPISODES:-5}" \
  NUM_EVAL_ENVS="${NUM_EVAL_ENVS:-1}" \
  INPAINTING="${INPAINTING:-true}" \
  EVAL_PROGRESS_BAR="${EVAL_PROGRESS_BAR:-false}" \
  CAPTURE_VIDEO_FREQ="${CAPTURE_VIDEO_FREQ:-5}" \
  DEMO_TYPE="${DEMO_TYPE:-pickcube_mam_single_inpaint_overfit5}"

bash examples/baselines/diffusion_policy/run_mam.sh
