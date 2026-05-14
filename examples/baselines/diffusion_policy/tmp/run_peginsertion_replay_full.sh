#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "${ROOT_DIR}"

CONDA_BIN="${CONDA_BIN:-/home/hebu/miniconda3/envs/maniskill_py311/bin}"
if [[ -d "${CONDA_BIN}" ]]; then
  export PATH="${CONDA_BIN}:${PATH}"
fi
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-maniskill}"

SRC_H5="${SRC_H5:-demos/PegInsertionSide-v1/PegInsertionSide-v1.h5}"
NUM_ENVS="${NUM_ENVS:-10}"
SIM_BACKEND="${SIM_BACKEND:-physx_cpu}"
OBS_MODE="${OBS_MODE:-rgb}"
CONTROL_MODE="${CONTROL_MODE:-pd_ee_pose}"

echo "[peginsertion replay] parallel num-envs=${NUM_ENVS}, backend=${SIM_BACKEND}"

python -m mani_skill.trajectory.replay_trajectory \
  --traj-path "${SRC_H5}" \
  --save-traj \
  --obs-mode "${OBS_MODE}" \
  --target-control-mode "${CONTROL_MODE}" \
  --num-envs "${NUM_ENVS}" \
  --sim-backend "${SIM_BACKEND}" \
  --allow-failure
