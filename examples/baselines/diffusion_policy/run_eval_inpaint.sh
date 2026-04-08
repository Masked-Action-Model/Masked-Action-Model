#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT_DIR}"

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

CONDA_BIN="${CONDA_BIN:-/home/hebu/miniconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-maniskill_py311}"
ENTRYPOINT="${ENTRYPOINT:-examples/baselines/diffusion_policy/eval_inpaint.py}"

CHECKPOINT_PT_PATH="${CHECKPOINT_PT_PATH:-runs/window_random_0.2/best_eval_success_once.pt}"
EVAL_DEMO_PATH="${EVAL_DEMO_PATH:-demos/data_1_preprocessed/random_mask_0.2/data_1_random_mask_0.2_eval.h5}"
EVAL_DEMO_METADATA_PATH="${EVAL_DEMO_METADATA_PATH:-demos/data_1_preprocessed/random_mask_0.2/data_1_random_mask_0.2_eval.json}"
STPM_CONFIG_PATH="${STPM_CONFIG_PATH:-STPM_PickCube/pick up the cube and place it at the goal/config.yaml}"
STPM_CKPT_PATH="${STPM_CKPT_PATH:-STPM_PickCube/pick up the cube and place it at the goal/checkpoints/reward_best.pt}"

ENV_ID="${ENV_ID:-PickCube-v1}"
CONTROL_MODE="${CONTROL_MODE:-pd_ee_pose}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-160}"
NUM_EVAL_DEMOS="${NUM_EVAL_DEMOS:-100}"
NUM_EVAL_ENVS="${NUM_EVAL_ENVS:-1}"
SIM_BACKEND="${SIM_BACKEND:-physx_cpu}"
RENDER_BACKEND="${RENDER_BACKEND:-gpu}"

OBS_HORIZON="${OBS_HORIZON:-2}"
ACT_HORIZON="${ACT_HORIZON:-8}"
PRED_HORIZON="${PRED_HORIZON:-16}"
LONG_WINDOW_HORIZON="${LONG_WINDOW_HORIZON:-${PRED_HORIZON}}"
SHORT_WINDOW_HORIZON="${SHORT_WINDOW_HORIZON:-20}"
MAS_LONG_ENCODE_MODE="${MAS_LONG_ENCODE_MODE:-2DConv}"
MAS_LONG_CONV_OUTPUT_DIM="${MAS_LONG_CONV_OUTPUT_DIM:-0}"
DIFFUSION_STEP_EMBED_DIM="${DIFFUSION_STEP_EMBED_DIM:-64}"
LEGACY_SHORT_MASK_BRANCH="${LEGACY_SHORT_MASK_BRANCH:-true}"

JUMP_LENGTH="${JUMP_LENGTH:-5}"
NUM_RESAMPLE="${NUM_RESAMPLE:-5}"

CHECKPOINT_KEY="${CHECKPOINT_KEY:-auto}"
SEED="${SEED:-1}"
CUDA_FLAG="${CUDA_FLAG:---cuda}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
OUTPUT_JSON_PATH="${OUTPUT_JSON_PATH:-}"
SAVE_PER_TRAJ_FLAG="${SAVE_PER_TRAJ_FLAG:-}"

if [[ -z "${CHECKPOINT_PT_PATH}" ]]; then
  echo "CHECKPOINT_PT_PATH is required" >&2
  exit 1
fi
if [[ -z "${EVAL_DEMO_PATH}" ]]; then
  echo "EVAL_DEMO_PATH is required" >&2
  exit 1
fi
if [[ ! -f "${CHECKPOINT_PT_PATH}" ]]; then
  echo "checkpoint not found: ${CHECKPOINT_PT_PATH}" >&2
  exit 1
fi
if [[ ! -f "${EVAL_DEMO_PATH}" ]]; then
  echo "eval demo not found: ${EVAL_DEMO_PATH}" >&2
  exit 1
fi
if [[ -n "${EVAL_DEMO_METADATA_PATH}" && ! -f "${EVAL_DEMO_METADATA_PATH}" ]]; then
  echo "eval demo metadata not found: ${EVAL_DEMO_METADATA_PATH}" >&2
  exit 1
fi
if [[ ! -f "${STPM_CONFIG_PATH}" ]]; then
  echo "STPM config not found: ${STPM_CONFIG_PATH}" >&2
  exit 1
fi
if [[ ! -f "${STPM_CKPT_PATH}" ]]; then
  echo "STPM checkpoint not found: ${STPM_CKPT_PATH}" >&2
  exit 1
fi

ARGS=(
  --checkpoint-pt-path "${CHECKPOINT_PT_PATH}"
  --eval-demo-path "${EVAL_DEMO_PATH}"
  --stpm-ckpt-path "${STPM_CKPT_PATH}"
  --stpm-config-path "${STPM_CONFIG_PATH}"
  --env-id "${ENV_ID}"
  --control-mode "${CONTROL_MODE}"
  --max-episode-steps "${MAX_EPISODE_STEPS}"
  --num-eval-demos "${NUM_EVAL_DEMOS}"
  --num-eval-envs "${NUM_EVAL_ENVS}"
  --sim-backend "${SIM_BACKEND}"
  --render-backend "${RENDER_BACKEND}"
  --obs-horizon "${OBS_HORIZON}"
  --act-horizon "${ACT_HORIZON}"
  --pred-horizon "${PRED_HORIZON}"
  --long-window-horizon "${LONG_WINDOW_HORIZON}"
  --short-window-horizon "${SHORT_WINDOW_HORIZON}"
  --mas-long-encode-mode "${MAS_LONG_ENCODE_MODE}"
  --mas-long-conv-output-dim "${MAS_LONG_CONV_OUTPUT_DIM}"
  --diffusion-step-embed-dim "${DIFFUSION_STEP_EMBED_DIM}"
  --jump-length "${JUMP_LENGTH}"
  --num-resample "${NUM_RESAMPLE}"
  --checkpoint-key "${CHECKPOINT_KEY}"
  --seed "${SEED}"
)

if [[ -n "${EVAL_DEMO_METADATA_PATH}" ]]; then
  ARGS+=(--eval-demo-metadata-path "${EVAL_DEMO_METADATA_PATH}")
fi
if [[ -n "${OUTPUT_DIR}" ]]; then
  ARGS+=(--output-dir "${OUTPUT_DIR}")
fi
if [[ -n "${OUTPUT_JSON_PATH}" ]]; then
  ARGS+=(--output-json-path "${OUTPUT_JSON_PATH}")
fi

echo "[resolved-config] entrypoint=${ENTRYPOINT}"
echo "[resolved-config] checkpoint_pt_path=${CHECKPOINT_PT_PATH}"
echo "[resolved-config] eval_demo_path=${EVAL_DEMO_PATH}"
echo "[resolved-config] eval_demo_metadata_path=${EVAL_DEMO_METADATA_PATH}"
echo "[resolved-config] stpm_ckpt_path=${STPM_CKPT_PATH}"
echo "[resolved-config] stpm_config_path=${STPM_CONFIG_PATH}"
echo "[resolved-config] env_id=${ENV_ID}"
echo "[resolved-config] sim_backend=${SIM_BACKEND}"
echo "[resolved-config] obs_horizon=${OBS_HORIZON}"
echo "[resolved-config] act_horizon=${ACT_HORIZON}"
echo "[resolved-config] pred_horizon=${PRED_HORIZON}"
echo "[resolved-config] long_window_horizon=${LONG_WINDOW_HORIZON}"
echo "[resolved-config] short_window_horizon=${SHORT_WINDOW_HORIZON}"
echo "[resolved-config] legacy_short_mask_branch=${LEGACY_SHORT_MASK_BRANCH}"
echo "[resolved-config] jump_length=${JUMP_LENGTH}"
echo "[resolved-config] num_resample=${NUM_RESAMPLE}"
echo "[resolved-config] num_eval_demos=${NUM_EVAL_DEMOS}"
echo "[resolved-config] num_eval_envs=${NUM_EVAL_ENVS}"

if [[ "${LEGACY_SHORT_MASK_BRANCH}" == "true" ]]; then
  ARGS+=(--legacy-short-mask-branch)
fi

CMD=(
  "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}"
  python "${ENTRYPOINT}"
  "${ARGS[@]}"
  "${CUDA_FLAG}"
)

if [[ -n "${SAVE_PER_TRAJ_FLAG}" ]]; then
  CMD+=("${SAVE_PER_TRAJ_FLAG}")
fi

exec "${CMD[@]}"
