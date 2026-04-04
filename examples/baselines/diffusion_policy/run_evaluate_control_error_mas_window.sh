#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT_DIR}"

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

PYTHON_BIN="${PYTHON_BIN:-python}"

# --------------------------------------------------------------------------- #
# Parameters forwarded to evaluate_control_error_mas_window.py
# --------------------------------------------------------------------------- #
CHECKPOINT_PT_PATH="${CHECKPOINT_PT_PATH:-runs/window_2D_64_2/checkpoints/best_eval_success_once.pt}"
EVAL_DEMO_PATH="${EVAL_DEMO_PATH:-demos/data_1_preprocessed/3D_points_0.03/data_1_concat_3D_points_0.03_train.h5}"
EVAL_DEMO_METADATA_PATH="${EVAL_DEMO_METADATA_PATH:-demos/data_1_preprocessed/3D_points_0.03/data_1_concat_3D_points_0.03_train.json}"
STPM_CKPT_PATH="${STPM_CKPT_PATH:-STPM_PickCube/pick up the cube and place it at the goal/checkpoints/reward_best.pt}"
STPM_CONFIG_PATH="${STPM_CONFIG_PATH:-STPM_PickCube/pick up the cube and place it at the goal/config.yaml}"

ENV_ID="${ENV_ID:-PickCube-v1}"
CONTROL_MODE="${CONTROL_MODE:-pd_ee_pose}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-200}"
NUM_EVAL_DEMOS="${NUM_EVAL_DEMOS:-5}"
NUM_EVAL_ENVS="${NUM_EVAL_ENVS:-1}"
SIM_BACKEND="${SIM_BACKEND:-physx_cpu}"
RENDER_BACKEND="${RENDER_BACKEND:-gpu}"

OBS_HORIZON="${OBS_HORIZON:-2}"
ACT_HORIZON="${ACT_HORIZON:-8}"
PRED_HORIZON="${PRED_HORIZON:-16}"
LONG_WINDOW_HORIZON="${LONG_WINDOW_HORIZON:-16}"
SHORT_WINDOW_HORIZON="${SHORT_WINDOW_HORIZON:-2}"
MAS_LONG_ENCODE_MODE="${MAS_LONG_ENCODE_MODE:-2DConv}"
MAS_LONG_CONV_OUTPUT_DIM="${MAS_LONG_CONV_OUTPUT_DIM:-64}"
DIFFUSION_STEP_EMBED_DIM="${DIFFUSION_STEP_EMBED_DIM:-64}"

CHECKPOINT_KEY="${CHECKPOINT_KEY:-auto}"
SEED="${SEED:-1}"
TORCH_DETERMINISTIC="${TORCH_DETERMINISTIC:-true}"
CUDA="${CUDA:-true}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
OUTPUT_JSON_PATH="${OUTPUT_JSON_PATH:-}"
SAVE_PER_TRAJ="${SAVE_PER_TRAJ:-true}"


ensure_default_eval_dataset() {
  if [[ -f "$EVAL_DEMO_PATH" && -f "$EVAL_DEMO_METADATA_PATH" ]]; then
    echo "[preprocess] reuse existing dataset: ${EVAL_DEMO_PATH}"
    return
  fi

  local default_eval_h5="demos/data_1_preprocessed/3D_points_0.03/data_1_concat_3D_points_0.03_train.h5"
  local default_eval_json="demos/data_1_preprocessed/3D_points_0.03/data_1_concat_3D_points_0.03_train.json"

  if [[ "$EVAL_DEMO_PATH" != "$default_eval_h5" || "$EVAL_DEMO_METADATA_PATH" != "$default_eval_json" ]]; then
    echo "ERROR: eval demo files not found:" >&2
    echo "  $EVAL_DEMO_PATH" >&2
    echo "  $EVAL_DEMO_METADATA_PATH" >&2
    echo "请手动提供现成数据，或使用默认路径让脚本自动生成匹配 window_2D_64_2 的数据集。" >&2
    exit 1
  fi

  local raw_demo_h5="demos/data_1/data_1.h5"
  local raw_demo_json="demos/data_1/data_1.json"
  local output_dir="demos/data_1_preprocessed/3D_points_0.03"

  if [[ ! -f "$raw_demo_h5" || ! -f "$raw_demo_json" ]]; then
    echo "ERROR: raw demo files not found:" >&2
    echo "  $raw_demo_h5" >&2
    echo "  $raw_demo_json" >&2
    exit 1
  fi

  mkdir -p "$output_dir"
  echo "[preprocess] generating default dataset for window_2D_64_2 into ${output_dir}"
  "$PYTHON_BIN" examples/baselines/diffusion_policy/data_preprocess.py \
    --input-h5 "$raw_demo_h5" \
    --input-json "$raw_demo_json" \
    --output-dir "$output_dir" \
    --output-prefix "data_1_concat" \
    --env-id "$ENV_ID" \
    --mask-type "3D_points" \
    --retain-ratio "0.03" \
    --overwrite
}


ensure_default_eval_dataset

for path in "$CHECKPOINT_PT_PATH" "$EVAL_DEMO_PATH" "$EVAL_DEMO_METADATA_PATH" "$STPM_CKPT_PATH" "$STPM_CONFIG_PATH"; do
  if [[ ! -f "$path" ]]; then
    echo "ERROR: file not found: $path" >&2
    exit 1
  fi
done

ARGS=(
  --checkpoint-pt-path "$CHECKPOINT_PT_PATH"
  --eval-demo-path "$EVAL_DEMO_PATH"
  --eval-demo-metadata-path "$EVAL_DEMO_METADATA_PATH"
  --stpm-ckpt-path "$STPM_CKPT_PATH"
  --stpm-config-path "$STPM_CONFIG_PATH"
  --env-id "$ENV_ID"
  --control-mode "$CONTROL_MODE"
  --max-episode-steps "$MAX_EPISODE_STEPS"
  --num-eval-demos "$NUM_EVAL_DEMOS"
  --num-eval-envs "$NUM_EVAL_ENVS"
  --sim-backend "$SIM_BACKEND"
  --render-backend "$RENDER_BACKEND"
  --obs-horizon "$OBS_HORIZON"
  --act-horizon "$ACT_HORIZON"
  --pred-horizon "$PRED_HORIZON"
  --long-window-horizon "$LONG_WINDOW_HORIZON"
  --short-window-horizon "$SHORT_WINDOW_HORIZON"
  --mas-long-encode-mode "$MAS_LONG_ENCODE_MODE"
  --mas-long-conv-output-dim "$MAS_LONG_CONV_OUTPUT_DIM"
  --diffusion-step-embed-dim "$DIFFUSION_STEP_EMBED_DIM"
  --checkpoint-key "$CHECKPOINT_KEY"
  --seed "$SEED"
)

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
if [[ -n "$OUTPUT_DIR" ]]; then
  ARGS+=(--output-dir "$OUTPUT_DIR")
fi
if [[ -n "$OUTPUT_JSON_PATH" ]]; then
  ARGS+=(--output-json-path "$OUTPUT_JSON_PATH")
fi
if [[ "$SAVE_PER_TRAJ" == "true" ]]; then
  ARGS+=(--save-per-traj)
else
  ARGS+=(--no-save-per-traj)
fi

"$PYTHON_BIN" examples/baselines/diffusion_policy/evaluate_control_error_mas_window.py "${ARGS[@]}"
