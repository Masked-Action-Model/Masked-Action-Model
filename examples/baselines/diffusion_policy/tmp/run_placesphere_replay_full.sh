#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-maniskill}"

SRC_H5="${SRC_H5:-demos/PlaceSphere-v1/PlaceSphere-v1.h5}"
SRC_JSON="${SRC_JSON:-demos/PlaceSphere-v1/PlaceSphere-v1.json}"
REPLAY_SRC_DIR="${REPLAY_SRC_DIR:-demos/PlaceSphere-v1_full}"
REPLAY_STEM="${REPLAY_STEM:-PlaceSphere-v1_full}"
NUM_ENVS="${NUM_ENVS:-8}"
SIM_BACKEND="${SIM_BACKEND:-physx_cpu}"
OBS_MODE="${OBS_MODE:-rgb}"
CONTROL_MODE="${CONTROL_MODE:-pd_ee_pose}"

mkdir -p "$REPLAY_SRC_DIR"
SRC_H5_ABS="$(realpath "$SRC_H5")"
SRC_JSON_ABS="$(realpath "$SRC_JSON")"
ln -sf "$SRC_H5_ABS" "$REPLAY_SRC_DIR/${REPLAY_STEM}.h5"
ln -sf "$SRC_JSON_ABS" "$REPLAY_SRC_DIR/${REPLAY_STEM}.json"

python -m mani_skill.trajectory.replay_trajectory \
  --traj-path "$REPLAY_SRC_DIR/${REPLAY_STEM}.h5" \
  --save-traj \
  --obs-mode "$OBS_MODE" \
  --target-control-mode "$CONTROL_MODE" \
  --num-envs "$NUM_ENVS" \
  --sim-backend "$SIM_BACKEND" \
  --allow-failure
