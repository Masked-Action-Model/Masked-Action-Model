#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

CONDA_BIN="${CONDA_BIN:-/home/hebu/miniconda3/envs/maniskill_py311/bin}"
if [[ -d "${CONDA_BIN}" ]]; then
  export PATH="${CONDA_BIN}:${PATH}"
fi

REPLAY_DATASET_PATH="${REPLAY_DATASET_PATH:-demos/PullCubeTool-v1/PullCubeTool-v1.rgb.pd_ee_pose.physx_cpu.h5}"
AUTO_REPLAY="${AUTO_REPLAY:-false}"

if [[ ! -f "${REPLAY_DATASET_PATH}" ]]; then
  if [[ "${AUTO_REPLAY}" == "true" ]]; then
    echo "[pullcubetool] replay dataset not found, start replay: ${REPLAY_DATASET_PATH}"
    bash examples/baselines/diffusion_policy/tmp/run_pullcube_replay_full.sh
  else
    echo "ERROR: replay dataset not found: ${REPLAY_DATASET_PATH}" >&2
    echo "Run replay first:" >&2
    echo "  NUM_ENVS=10 bash examples/baselines/diffusion_policy/tmp/run_pullcube_replay_full.sh" >&2
    exit 1
  fi
fi

DATASET_PATH="${DATASET_PATH:-${REPLAY_DATASET_PATH}}" \
TASK_NAME="${TASK_NAME:-pullcubetool}" \
TASK_DESCRIPTION="${TASK_DESCRIPTION:-use the tool to pull a cube that is out of reach}" \
OUTPUT_DIR="${OUTPUT_DIR:-STPM_pullcubetool}" \
STATE_PATHS="${STATE_PATHS:-auto}" \
CAMERA_NAMES="${CAMERA_NAMES:-auto}" \
BATCH_SIZE="${BATCH_SIZE:-64}" \
NUM_WORKERS="${NUM_WORKERS:-4}" \
VAL_NUM_WORKERS="${VAL_NUM_WORKERS:-4}" \
NUM_EPOCHS="${NUM_EPOCHS:-2}" \
bash STPM/run_train_stpm.sh
