#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-maniskill}"

# -----------------------------------------------------------------------------
# 1. Basic info
# -----------------------------------------------------------------------------

ENV_ID="${ENV_ID:-PickCube-v1}"
ACTION_DIM="${ACTION_DIM:-7}" # 7=有夹爪，6=无夹爪/panda_stick
EXP_NAME="${EXP_NAME:-PickCube_subgoal_condition}"
SEED="${SEED:-1}"
TORCH_DETERMINISTIC="${TORCH_DETERMINISTIC:-true}"
CUDA="${CUDA:-true}"
WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME:-ManiSkill}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
TRACK="${TRACK:-false}"
CAPTURE_VIDEO="${CAPTURE_VIDEO:-false}"

# -----------------------------------------------------------------------------
# 2. Raw paths and mask preprocess params
# -----------------------------------------------------------------------------

RAW_DEMO_H5="${RAW_DEMO_H5:-demos/data_1/data_1.h5}"
RAW_DEMO_JSON="${RAW_DEMO_JSON:-demos/data_1/data_1.json}"
PREPROCESSED_ROOT_DIR="${PREPROCESSED_ROOT_DIR:-demos/data_1_preprocessed}"
PREPROCESSED_DATA_PREFIX="${PREPROCESSED_DATA_PREFIX:-data_1}"
PREPROCESSED_DATA_DIR="${PREPROCESSED_DATA_DIR:-}"
PREPROCESS_MASK_VALUE="${PREPROCESS_MASK_VALUE:-0}"
PREPROCESS_NUM_TRAJ="${PREPROCESS_NUM_TRAJ:-}"

# Subgoal condition only supports one mask. Keep the same list interface as MAM:
#   MASK_TYPE_LIST='["3D_points"]'
#   MASK_RATIO_LIST='[0.5]'      # retain_ratio, or seq_len for seq masks
MASK_TYPE_LIST="${MASK_TYPE_LIST:-[\"3D_points\"]}"
MASK_RATIO_LIST="${MASK_RATIO_LIST:-[0.5]}"

MASK_ASSIGNMENTS="$(
MASK_TYPE_LIST="$MASK_TYPE_LIST" \
MASK_RATIO_LIST="$MASK_RATIO_LIST" \
python - <<'PY'
import ast
import os
import shlex

def parse_list(value):
    try:
        parsed = ast.literal_eval(value)
    except Exception:
        parsed = [value]
    if isinstance(parsed, (str, bytes)):
        parsed = [parsed]
    return list(parsed)

def parse_mask_param(value):
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null"}:
        return None
    return float(value)

mask_types = [str(v) for v in parse_list(os.environ["MASK_TYPE_LIST"])]
mask_params = [parse_mask_param(v) for v in parse_list(os.environ["MASK_RATIO_LIST"])]
if len(mask_types) != 1:
    raise ValueError(
        "run_subgoal_condition.sh only supports one mask. "
        f"Got MASK_TYPE_LIST={mask_types!r}"
    )
if len(mask_params) == 0:
    mask_params = [0.2]
if len(mask_params) > 1:
    raise ValueError(
        "run_subgoal_condition.sh expects exactly one MASK_RATIO_LIST value. "
        f"Got MASK_RATIO_LIST={mask_params!r}"
    )

mask_param = mask_params[0]
assignments = {
    "MASK_TYPE_LIST": f"[{mask_types[0]!r}]",
    "MASK_RATIO_LIST": f"[{0 if mask_param is None else mask_param}]",
    "SINGLE_MASK_TYPE": mask_types[0],
    "SINGLE_MASK_PARAM": "0" if mask_param is None else format(float(mask_param), "g"),
}
for key, value in assignments.items():
    print(f"{key}={shlex.quote(value)}")
PY
)"
eval "$MASK_ASSIGNMENTS"

export \
  MASK_TYPE_LIST MASK_RATIO_LIST PREPROCESSED_DATA_PREFIX

SINGLE_MASK_RETAIN_RATIO="1.0"
SINGLE_MASK_SEQ_LEN="1"
case "$SINGLE_MASK_TYPE" in
  2D_partial_trajectory|local_planner)
    SINGLE_MASK_SEQ_LEN="${SINGLE_MASK_PARAM}"
    ;;
  *)
    SINGLE_MASK_RETAIN_RATIO="${SINGLE_MASK_PARAM}"
    ;;
esac
echo "[subgoal-mask] type=${SINGLE_MASK_TYPE}"
echo "[subgoal-mask] ratio=${SINGLE_MASK_PARAM}"

case "$SINGLE_MASK_TYPE" in
  pose_AnyGrasp|pose_motion_planning|points|3D_points|random_mask)
    PREPROCESS_DIR_SUFFIX="${SINGLE_MASK_TYPE}_${SINGLE_MASK_RETAIN_RATIO}"
    ;;
  2D_partial_trajectory|local_planner)
    PREPROCESS_DIR_SUFFIX="${SINGLE_MASK_TYPE}_seq${SINGLE_MASK_SEQ_LEN}"
    ;;
  2D_video_trajectory|2D_image_trajectory)
    PREPROCESS_DIR_SUFFIX="${SINGLE_MASK_TYPE}"
    ;;
  *)
    PREPROCESS_DIR_SUFFIX="${SINGLE_MASK_TYPE}"
    ;;
esac
PREPROCESSED_DATA_DIR="${PREPROCESSED_DATA_DIR:-${PREPROCESSED_ROOT_DIR}/${PREPROCESS_DIR_SUFFIX}}"
PREPROCESSED_FILE_STEM="${PREPROCESSED_FILE_STEM:-${PREPROCESSED_DATA_PREFIX}_${PREPROCESS_DIR_SUFFIX}}"
DEMO_PATH="${DEMO_PATH:-${PREPROCESSED_DATA_DIR}/${PREPROCESSED_FILE_STEM}_train.h5}"
EVAL_DEMO_PATH="${EVAL_DEMO_PATH:-${PREPROCESSED_DATA_DIR}/${PREPROCESSED_FILE_STEM}_eval.h5}"
EVAL_DEMO_METADATA_PATH="${EVAL_DEMO_METADATA_PATH:-${PREPROCESSED_DATA_DIR}/${PREPROCESSED_FILE_STEM}_eval.json}"
TRAIN_DEMO_METADATA_PATH="${TRAIN_DEMO_METADATA_PATH:-${PREPROCESSED_DATA_DIR}/${PREPROCESSED_FILE_STEM}_train.json}"
ACTION_NORM_PATH="${ACTION_NORM_PATH:-${DEMO_PATH}}"

case "$ACTION_DIM" in
  6|7) ;;
  *)
    echo "ERROR: ACTION_DIM must be 6 or 7, got: $ACTION_DIM" >&2
    exit 1
    ;;
esac

# -----------------------------------------------------------------------------
# 3. Environment
# -----------------------------------------------------------------------------

CONTROL_MODE="${CONTROL_MODE:-pd_ee_pose}"
OBS_MODE="${OBS_MODE:-rgb}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-100}"
SIM_BACKEND="${SIM_BACKEND:-physx_cpu}"

# -----------------------------------------------------------------------------
# 4. Model
# -----------------------------------------------------------------------------

NOISE_MODEL="${NOISE_MODEL:-Transformer}" # Transformer or Unet
DIFFUSION_STEP_EMBED_DIM="${DIFFUSION_STEP_EMBED_DIM:-64}"
DIT_HIDDEN_DIM="${DIT_HIDDEN_DIM:-512}"
DIT_NUM_BLOCKS="${DIT_NUM_BLOCKS:-6}"
DIT_DIM_FEEDFORWARD="${DIT_DIM_FEEDFORWARD:-2048}"
UNET_DIMS="${UNET_DIMS:-64 128 256}"
N_GROUPS="${N_GROUPS:-8}"
case "$NOISE_MODEL" in
  Transformer|Unet) ;;
  *)
    echo "ERROR: NOISE_MODEL must be Transformer or Unet, got: $NOISE_MODEL" >&2
    exit 1
    ;;
esac

# -----------------------------------------------------------------------------
# 5. Training
# -----------------------------------------------------------------------------

NUM_DEMOS="${NUM_DEMOS:-100}"
TOTAL_ITERS="${TOTAL_ITERS:-100000}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LR="${LR:-1e-4}"
OBS_HORIZON="${OBS_HORIZON:-2}"
ACT_HORIZON="${ACT_HORIZON:-8}"
PRED_HORIZON="${PRED_HORIZON:-16}"
NUM_DATALOAD_WORKERS="${NUM_DATALOAD_WORKERS:-0}"
DEMO_TYPE="${DEMO_TYPE:-subgoal_condition_${NOISE_MODEL}}"

# -----------------------------------------------------------------------------
# 6. Eval, logging, and checkpointing
# -----------------------------------------------------------------------------

LOG_FREQ="${LOG_FREQ:-1000}"
EVAL_FREQ="${EVAL_FREQ:-5000}"
SAVE_FREQ="${SAVE_FREQ:-50000}"
NUM_EVAL_DEMOS="${NUM_EVAL_DEMOS:-100}"
NUM_EVAL_EPISODES="${NUM_EVAL_EPISODES:-${NUM_EVAL_DEMOS}}"
NUM_EVAL_ENVS="${NUM_EVAL_ENVS:-10}"

# -----------------------------------------------------------------------------
# 7. Preprocess helper
# -----------------------------------------------------------------------------

ensure_preprocessed_dataset() {
  local missing=0
  for path in "$DEMO_PATH" "$EVAL_DEMO_PATH" "$TRAIN_DEMO_METADATA_PATH" "$EVAL_DEMO_METADATA_PATH"; do
    if [[ ! -f "$path" ]]; then
      missing=1
      break
    fi
  done

  if [[ "$missing" -eq 0 ]]; then
    echo "[subgoal-preprocess] reuse existing dataset: ${PREPROCESSED_DATA_DIR}"
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
  echo "[subgoal-preprocess] generating dataset into ${PREPROCESSED_DATA_DIR}"
  PREPROCESS_ARGS=(
    --input-h5 "$RAW_DEMO_H5"
    --input-json "$RAW_DEMO_JSON"
    --output-dir "$PREPROCESSED_DATA_DIR"
    --output-prefix "$PREPROCESSED_DATA_PREFIX"
    --env-id "$ENV_ID"
    --action-dim "$ACTION_DIM"
    --mask-type "$SINGLE_MASK_TYPE"
    --retain-ratio "$SINGLE_MASK_RETAIN_RATIO"
    --mask-seq-len "$SINGLE_MASK_SEQ_LEN"
    --mask-value "$PREPROCESS_MASK_VALUE"
  )
  if [[ -n "$PREPROCESS_NUM_TRAJ" ]]; then
    PREPROCESS_ARGS+=(--num-traj "$PREPROCESS_NUM_TRAJ")
  fi
  python examples/baselines/diffusion_policy/data_preprocess/data_preprocess.py "${PREPROCESS_ARGS[@]}"
}

ensure_preprocessed_dataset

# -----------------------------------------------------------------------------
# 8. Validate required files
# -----------------------------------------------------------------------------

for path in "$DEMO_PATH" "$EVAL_DEMO_PATH" "$EVAL_DEMO_METADATA_PATH" "$ACTION_NORM_PATH"; do
  if [[ ! -f "$path" ]]; then
    echo "ERROR: required file not found: $path" >&2
    exit 1
  fi
done

# -----------------------------------------------------------------------------
# 9. Build train args
# -----------------------------------------------------------------------------

ARGS=(
  --seed "$SEED"
  --wandb-project-name "$WANDB_PROJECT_NAME"
  --env-id "$ENV_ID"
  --action-dim "$ACTION_DIM"
  --demo-path "$DEMO_PATH"
  --eval-demo-path "$EVAL_DEMO_PATH"
  --eval-demo-metadata-path "$EVAL_DEMO_METADATA_PATH"
  --action-norm-path "$ACTION_NORM_PATH"
  --noise-model "$NOISE_MODEL"
  --total-iters "$TOTAL_ITERS"
  --batch-size "$BATCH_SIZE"
  --lr "$LR"
  --obs-horizon "$OBS_HORIZON"
  --act-horizon "$ACT_HORIZON"
  --pred-horizon "$PRED_HORIZON"
  --diffusion-step-embed-dim "$DIFFUSION_STEP_EMBED_DIM"
  --dit-hidden-dim "$DIT_HIDDEN_DIM"
  --dit-num-blocks "$DIT_NUM_BLOCKS"
  --dit-dim-feedforward "$DIT_DIM_FEEDFORWARD"
  --unet-dims $UNET_DIMS
  --n-groups "$N_GROUPS"
  --obs-mode "$OBS_MODE"
  --log-freq "$LOG_FREQ"
  --eval-freq "$EVAL_FREQ"
  --num-eval-episodes "$NUM_EVAL_EPISODES"
  --num-eval-envs "$NUM_EVAL_ENVS"
  --num-dataload-workers "$NUM_DATALOAD_WORKERS"
  --control-mode "$CONTROL_MODE"
  --sim-backend "$SIM_BACKEND"
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
if [[ -n "$NUM_EVAL_DEMOS" ]]; then
  ARGS+=(--num-eval-demos "$NUM_EVAL_DEMOS")
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

python examples/baselines/diffusion_policy/train_subgoal_condition.py "${ARGS[@]}"
