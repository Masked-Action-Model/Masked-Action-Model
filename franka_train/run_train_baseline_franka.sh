#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/examples/baselines/diffusion_policy:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-maniskill}"

PYTHON_BIN="${PYTHON_BIN:-python}"
if [[ "$PYTHON_BIN" == "python" && -x /home/hebu/miniconda3/envs/maniskill_py311/bin/python ]]; then
  PYTHON_BIN=/home/hebu/miniconda3/envs/maniskill_py311/bin/python
fi

# 1. Experiment
EXP_NAME="${EXP_NAME:-FrankaReal_baseline}"
SEED="${SEED:-1}"
TORCH_DETERMINISTIC="${TORCH_DETERMINISTIC:-true}"
CUDA="${CUDA:-true}"
TRACK="${TRACK:-false}"
WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME:-ManiSkill}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
CAPTURE_VIDEO="${CAPTURE_VIDEO:-false}"

# 2. Raw real dataset and train-time preprocess
RAW_DEMO_H5="${RAW_DEMO_H5:-franka_train/data/franka_real.h5}"
PREPROCESSED_DATA_DIR="${PREPROCESSED_DATA_DIR:-franka_train/data}"
PREPROCESSED_DATA_PREFIX="${PREPROCESSED_DATA_PREFIX:-franka_real_baseline}"
PREPROCESSED_H5="${PREPROCESSED_H5:-${PREPROCESSED_DATA_DIR}/${PREPROCESSED_DATA_PREFIX}_train.h5}"
DEMO_PATH="${DEMO_PATH:-${PREPROCESSED_H5}}"
RUN_PREPROCESS="${RUN_PREPROCESS:-true}" # true / false / auto
OVERWRITE_PREPROCESS="${OVERWRITE_PREPROCESS:-true}"
PREPROCESS_MASK_VALUE="${PREPROCESS_MASK_VALUE:-0}"
PREPROCESS_NUM_TRAJ="${PREPROCESS_NUM_TRAJ:-}"
MASK_SEED="${MASK_SEED:-0}"
ACTION_ROBUST_MARGIN="${ACTION_ROBUST_MARGIN:-0.01}"
STATE_ROBUST_MARGIN="${STATE_ROBUST_MARGIN:-0.01}"
IMAGE_SIZE="${IMAGE_SIZE:-256}"
NUM_DEMOS="${NUM_DEMOS:-}"

# 3. Environment/action schema
ENV_ID="${ENV_ID:-FrankaReal-v1}"
ACTION_DIM="${ACTION_DIM:-${action_dim:-auto}}"
ACTION_SPACE="${ACTION_SPACE:-${action_space:-absolute}}"
CONTROL_MODE="${CONTROL_MODE:-pd_ee_pose}"
OBS_MODE="${OBS_MODE:-rgb}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-}"
SIM_BACKEND="${SIM_BACKEND:-physx_cpu}"

if [[ "$ACTION_DIM" == "auto" ]]; then
  INFER_H5="$RAW_DEMO_H5"
  if [[ ! -f "$INFER_H5" && -f "$DEMO_PATH" ]]; then
    INFER_H5="$DEMO_PATH"
  fi
  if [[ ! -f "$INFER_H5" ]]; then
    echo "ERROR: cannot infer ACTION_DIM because h5 not found: ${RAW_DEMO_H5} or ${DEMO_PATH}" >&2
    exit 1
  fi
  ACTION_DIM="$("$PYTHON_BIN" - "$INFER_H5" <<'PY'
import h5py
import sys
with h5py.File(sys.argv[1], "r") as f:
    key = sorted(k for k in f.keys() if k.startswith("traj_"))[0]
    print(int(f[key]["actions"].shape[1]))
PY
)"
  echo "[action-dim] inferred ACTION_DIM=${ACTION_DIM} from ${INFER_H5}"
fi
case "$ACTION_DIM" in
  6|7) ;;
  *)
    echo "ERROR: ACTION_DIM must be 6 or 7, got: ${ACTION_DIM}" >&2
    exit 1
    ;;
esac
case "$ACTION_SPACE" in
  absolute|relative) ;;
  *)
    echo "ERROR: ACTION_SPACE must be absolute or relative, got: ${ACTION_SPACE}" >&2
    exit 1
    ;;
esac

need_preprocess=false
if [[ "$RUN_PREPROCESS" == "true" ]]; then
  need_preprocess=true
elif [[ "$RUN_PREPROCESS" == "auto" && ! -f "$DEMO_PATH" ]]; then
  need_preprocess=true
fi

if [[ "$need_preprocess" == "true" ]]; then
  if [[ ! -f "$RAW_DEMO_H5" ]]; then
    echo "ERROR: RAW_DEMO_H5 not found: ${RAW_DEMO_H5}" >&2
    exit 1
  fi
  PREPROCESS_ARGS=(
    --input-h5 "$RAW_DEMO_H5"
    --output-h5 "$DEMO_PATH"
    --env-id "$ENV_ID"
    --control-mode "$CONTROL_MODE"
    --action-dim "$ACTION_DIM"
    --mask-type full
    --mask-value "$PREPROCESS_MASK_VALUE"
    --mask-seed "$MASK_SEED"
    --action-robust-margin "$ACTION_ROBUST_MARGIN"
    --state-robust-margin "$STATE_ROBUST_MARGIN"
    --image-size "$IMAGE_SIZE"
  )
  if [[ -n "$PREPROCESS_NUM_TRAJ" ]]; then
    PREPROCESS_ARGS+=(--num-traj "$PREPROCESS_NUM_TRAJ")
  fi
  if [[ "$OVERWRITE_PREPROCESS" == "true" ]]; then
    PREPROCESS_ARGS+=(--overwrite)
  fi
  echo "[baseline-preprocess] write ${DEMO_PATH}"
  "$PYTHON_BIN" franka_train/preprocess_franka.py "${PREPROCESS_ARGS[@]}"
fi

if [[ ! -f "$DEMO_PATH" ]]; then
  echo "ERROR: demo h5 not found: ${DEMO_PATH}" >&2
  exit 1
fi

ACTION_NORM_PATH="${ACTION_NORM_PATH:-${DEMO_PATH}}"

# 4. Model
NOISE_MODEL="${NOISE_MODEL:-Transformer}" # Transformer or Unet
VISION_ENCODER="${VISION_ENCODER:-dino2}" # resnet, dino2, or dino3
DINO_MODEL_PATH="${DINO_MODEL_PATH:-}"
DINO_DATA_AUG="${DINO_DATA_AUG:-false}"
DIFFUSION_STEP_EMBED_DIM="${DIFFUSION_STEP_EMBED_DIM:-64}"
DIT_HIDDEN_DIM="${DIT_HIDDEN_DIM:-512}"
DIT_NUM_BLOCKS="${DIT_NUM_BLOCKS:-6}"
DIT_DIM_FEEDFORWARD="${DIT_DIM_FEEDFORWARD:-2048}"
UNET_DIMS="${UNET_DIMS:-64 128 256}"
N_GROUPS="${N_GROUPS:-8}"
case "$NOISE_MODEL" in
  Transformer|Unet) ;;
  *)
    echo "ERROR: NOISE_MODEL must be Transformer or Unet, got: ${NOISE_MODEL}" >&2
    exit 1
    ;;
esac
case "$VISION_ENCODER" in
  resnet|dino2|dino3) ;;
  *)
    echo "ERROR: VISION_ENCODER must be resnet, dino2, or dino3, got: ${VISION_ENCODER}" >&2
    exit 1
    ;;
esac
if [[ -z "$DINO_MODEL_PATH" ]]; then
  case "$VISION_ENCODER" in
    dino2) DINO_MODEL_PATH="${ROOT_DIR}/Dino/dinov2-small" ;;
    dino3) DINO_MODEL_PATH="${ROOT_DIR}/Dino/dinov3-vits16plus-pretrain-lvd1689m" ;;
  esac
fi
if [[ "$VISION_ENCODER" == dino* && ! -f "${DINO_MODEL_PATH}/config.json" ]]; then
  echo "ERROR: DINO_MODEL_PATH is incomplete; missing ${DINO_MODEL_PATH}/config.json" >&2
  exit 1
fi
case "$DINO_DATA_AUG" in
  true|false) ;;
  *)
    echo "ERROR: DINO_DATA_AUG must be true or false, got: ${DINO_DATA_AUG}" >&2
    exit 1
    ;;
esac

# 5. Training
TOTAL_ITERS="${TOTAL_ITERS:-100000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
LR="${LR:-1e-4}"
OBS_HORIZON="${OBS_HORIZON:-2}"
ACT_HORIZON="${ACT_HORIZON:-8}"
PRED_HORIZON="${PRED_HORIZON:-16}"
NUM_DATALOAD_WORKERS="${NUM_DATALOAD_WORKERS:-0}"
DEMO_TYPE="${DEMO_TYPE:-franka_real_baseline}"

# 6. Logging/checkpointing. Eval args are accepted for script parity but no online eval runs.
LOG_FREQ="${LOG_FREQ:-1000}"
EVAL_FREQ="${EVAL_FREQ:-0}"
VALID_FREQ="${VALID_FREQ:-5000}"
NUM_VALIDATION_SET="${NUM_VALIDATION_SET:-10}"
NUM_EVAL_EPISODES="${NUM_EVAL_EPISODES:-0}"
NUM_EVAL_ENVS="${NUM_EVAL_ENVS:-0}"
NUM_EVAL_DEMOS="${NUM_EVAL_DEMOS:-}"
SAVE_START_ITER="${SAVE_START_ITER:-0}"
SAVE_FREQ="${SAVE_FREQ:-5000}"

ARGS=(
  --exp-name "$EXP_NAME"
  --seed "$SEED"
  --wandb-project-name "$WANDB_PROJECT_NAME"
  --env-id "$ENV_ID"
  --action-dim "$ACTION_DIM"
  --action-space "$ACTION_SPACE"
  --demo-path "$DEMO_PATH"
  --noise-model "$NOISE_MODEL"
  --vision-encoder "$VISION_ENCODER"
  --dino-model-path "$DINO_MODEL_PATH"
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
  --valid-freq "$VALID_FREQ"
  --num-validation-set "$NUM_VALIDATION_SET"
  --num-eval-episodes "$NUM_EVAL_EPISODES"
  --num-eval-envs "$NUM_EVAL_ENVS"
  --save-start-iter "$SAVE_START_ITER"
  --save-freq "$SAVE_FREQ"
  --num-dataload-workers "$NUM_DATALOAD_WORKERS"
  --control-mode "$CONTROL_MODE"
  --sim-backend "$SIM_BACKEND"
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
if [[ "$TRACK" == "true" ]]; then
  ARGS+=(--track)
else
  ARGS+=(--no-track)
fi
if [[ "$CAPTURE_VIDEO" == "true" ]]; then
  ARGS+=(--capture-video)
else
  ARGS+=(--no-capture-video)
fi
if [[ "$DINO_DATA_AUG" == "true" ]]; then
  ARGS+=(--dino-data-aug)
else
  ARGS+=(--no-dino-data-aug)
fi
if [[ -n "$WANDB_ENTITY" ]]; then
  ARGS+=(--wandb-entity "$WANDB_ENTITY")
fi
if [[ -n "$NUM_DEMOS" ]]; then
  ARGS+=(--num-demos "$NUM_DEMOS")
fi
if [[ -n "$NUM_EVAL_DEMOS" ]]; then
  ARGS+=(--num-eval-demos "$NUM_EVAL_DEMOS")
fi
if [[ -n "$ACTION_NORM_PATH" ]]; then
  ARGS+=(--action-norm-path "$ACTION_NORM_PATH")
fi
if [[ -n "$MAX_EPISODE_STEPS" ]]; then
  ARGS+=(--max-episode-steps "$MAX_EPISODE_STEPS")
fi
if [[ -n "$DEMO_TYPE" ]]; then
  ARGS+=(--demo-type "$DEMO_TYPE")
fi

"$PYTHON_BIN" franka_train/train_baseline_franka.py "${ARGS[@]}"
