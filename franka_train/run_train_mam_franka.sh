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

# -----------------------------------------------------------------------------
# 1. Basic info
# -----------------------------------------------------------------------------

ENV_ID="${ENV_ID:-FrankaReal-v1}"
ACTION_DIM="${ACTION_DIM:-${action_dim:-auto}}"
CONTROL_MODE="${CONTROL_MODE:-pd_ee_pose}"
EXP_NAME="${EXP_NAME:-FrankaReal_mam}"
SEED="${SEED:-1}"
TORCH_DETERMINISTIC="${TORCH_DETERMINISTIC:-true}"
CUDA="${CUDA:-true}"
WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME:-ManiSkill}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
TRACK="${TRACK:-false}"
CAPTURE_VIDEO="${CAPTURE_VIDEO:-false}"

# -----------------------------------------------------------------------------
# 2. Raw paths
# -----------------------------------------------------------------------------

RAW_DEMO_H5="${RAW_DEMO_H5:-franka_train/data/franka_real.h5}"

# -----------------------------------------------------------------------------
# 3. Preprocess params: MAM mask pipeline
# -----------------------------------------------------------------------------

PREPROCESSED_ROOT_DIR="${PREPROCESSED_ROOT_DIR:-franka_train/data}"
PREPROCESSED_DATA_DIR="${PREPROCESSED_DATA_DIR:-}"
PREPROCESSED_DATA_PREFIX="${PREPROCESSED_DATA_PREFIX:-franka_real_mam}"
PREPROCESSED_FILE_STEM="${PREPROCESSED_FILE_STEM:-}"
RUN_PREPROCESS="${RUN_PREPROCESS:-true}" # true / false / auto
OVERWRITE_PREPROCESS="${OVERWRITE_PREPROCESS:-true}"

PREPROCESS_MASK_VALUE="${PREPROCESS_MASK_VALUE:-0}"
PREPROCESS_NUM_TRAJ="${PREPROCESS_NUM_TRAJ:-}"
PREPROCESS_MASK_ASSIGN_MODE="${PREPROCESS_MASK_ASSIGN_MODE:-composition}" # composition / one_demo_multi_mask
MASK_SEED="${MASK_SEED:-0}"
ACTION_ROBUST_MARGIN="${ACTION_ROBUST_MARGIN:-0.01}"
STATE_ROBUST_MARGIN="${STATE_ROBUST_MARGIN:-0.01}"
IMAGE_SIZE="${IMAGE_SIZE:-256}"

case "$PREPROCESS_MASK_ASSIGN_MODE" in
  composition|one_demo_multi_mask) ;;
  *)
    echo "ERROR: PREPROCESS_MASK_ASSIGN_MODE must be composition or one_demo_multi_mask, got: ${PREPROCESS_MASK_ASSIGN_MODE}" >&2
    exit 1
    ;;
esac

# Preferred interface:
#   MASK_TYPE_LIST='["random_mask","3D_points"]'
#   MASK_RATIO_LIST='[0.2,0.5]'             # retain_ratio, or seq_len for seq masks
#   MASK_COMPOSITION_LIST='[0.5,0.5]'       # optional; composition mode defaults to uniform
#   PREPROCESS_MASK_ASSIGN_MODE=composition # or one_demo_multi_mask
MASK_TYPE="${MASK_TYPE:-random_mask}"
RETAIN_RATIO="${RETAIN_RATIO:-0.2}"
MASK_SEQ_LEN="${MASK_SEQ_LEN:-20}"
MASK_TYPE_LIST="${MASK_TYPE_LIST:-[\"${MASK_TYPE}\"]}"
MASK_RATIO_LIST="${MASK_RATIO_LIST:-[${RETAIN_RATIO}]}"
MASK_COMPOSITION_LIST="${MASK_COMPOSITION_LIST:-}"

eval "$(
MASK_TYPE_LIST="$MASK_TYPE_LIST" \
MASK_RATIO_LIST="$MASK_RATIO_LIST" \
MASK_COMPOSITION_LIST="$MASK_COMPOSITION_LIST" \
"$PYTHON_BIN" - <<'PY'
import ast
import json
import os
import shlex


def parse_list(value):
    text = "" if value is None else str(value).strip()
    if len(text) == 0:
        return []
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        inner = text[1:-1].strip()
        if inner:
            text = inner
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
        except Exception:
            continue
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, (str, bytes)):
            inner = parsed.decode("utf-8") if isinstance(parsed, bytes) else parsed
            inner = inner.strip()
            if inner and inner != text:
                return parse_list(inner)
            return [parsed]
        return [parsed]
    if "," in text:
        return [item.strip() for item in text.split(",")]
    return [text]


def normalize(values, count, default):
    values = list(values)
    if count <= 0:
        return []
    if not values:
        return [default for _ in range(count)]
    if len(values) >= count:
        return values[:count]
    if len(values) == 1:
        return values * count
    return values + [values[-1] for _ in range(count - len(values))]


def normalize_composition(raw_value, count):
    values = parse_list(raw_value) if str(raw_value or "").strip() else []
    if count <= 0:
        return []
    if not values:
        return [1.0 / count for _ in range(count)]
    values = [float(v) for v in normalize(values, count, 1.0 / count)]
    total = sum(values)
    if total <= 0:
        raise ValueError(f"mask composition must have positive sum, got {values}")
    return [v / total for v in values]


mask_types_raw = parse_list(os.environ["MASK_TYPE_LIST"])
num_mask_type = len(mask_types_raw)
mask_types = [str(v) for v in normalize(mask_types_raw, num_mask_type, "random_mask")]
mask_params = [
    float(v)
    for v in normalize(parse_list(os.environ["MASK_RATIO_LIST"]), num_mask_type, 0.2)
]
mask_composition = normalize_composition(
    os.environ.get("MASK_COMPOSITION_LIST"),
    num_mask_type,
)

single = num_mask_type == 1
assignments = {
    "MASK_TYPE_LIST": json.dumps(mask_types),
    "MASK_RATIO_LIST": json.dumps(mask_params),
    "MASK_COMPOSITION_LIST": json.dumps(mask_composition),
    "TRAIN_MASK_TYPE_LIST": json.dumps(mask_types),
    "TRAIN_MASK_COMPOSITION_LIST": json.dumps(mask_composition),
    "TRAIN_MASK_RATIO_LIST": json.dumps(mask_params),
    "TRAIN_NUM_MASK_TYPE": str(num_mask_type),
}
if single:
    mask_param = mask_params[0] if mask_params else 0.2
    assignments["SINGLE_MASK_COMPAT"] = "true"
    assignments["SINGLE_MASK_TYPE"] = mask_types[0]
    assignments["SINGLE_MASK_PARAM"] = format(float(mask_param), "g")
else:
    assignments["SINGLE_MASK_COMPAT"] = "false"

for key, value in assignments.items():
    print(f"{key}={shlex.quote(value)}")
PY
)"

export \
  MASK_TYPE_LIST MASK_COMPOSITION_LIST MASK_RATIO_LIST \
  TRAIN_NUM_MASK_TYPE TRAIN_MASK_TYPE_LIST TRAIN_MASK_COMPOSITION_LIST TRAIN_MASK_RATIO_LIST \
  PREPROCESSED_DATA_PREFIX PREPROCESS_MASK_ASSIGN_MODE

echo "[mam-mask] assign_mode=${PREPROCESS_MASK_ASSIGN_MODE}"
echo "[mam-mask] types=${MASK_TYPE_LIST}"
echo "[mam-mask] ratios=${MASK_RATIO_LIST}"
echo "[mam-mask] composition=${MASK_COMPOSITION_LIST}"

PREPROCESS_MODE="mixed"
if [[ "$SINGLE_MASK_COMPAT" == "true" ]]; then
  PREPROCESS_MODE="single"
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
  echo "[mam] single mask type detected: ${SINGLE_MASK_TYPE}; using single-mask preprocess"
fi

compute_mixed_file_stem() {
  "$PYTHON_BIN" - <<'PY'
import os
from types import SimpleNamespace

from examples.baselines.diffusion_policy.data_preprocess.data_preprocess_mixed import (
    build_output_stem,
    normalize_split_mask_config,
)

args = SimpleNamespace(
    train_num_mask_type=int(os.environ["TRAIN_NUM_MASK_TYPE"]),
    train_mask_type_list=os.environ["TRAIN_MASK_TYPE_LIST"],
    train_mask_composition_list=os.environ["TRAIN_MASK_COMPOSITION_LIST"],
    train_mask_ratio_list=os.environ["TRAIN_MASK_RATIO_LIST"],
    eval_num_mask_type=int(os.environ["TRAIN_NUM_MASK_TYPE"]),
    eval_mask_type_list=os.environ["TRAIN_MASK_TYPE_LIST"],
    eval_mask_composition_list=os.environ["TRAIN_MASK_COMPOSITION_LIST"],
    eval_mask_ratio_list=os.environ["TRAIN_MASK_RATIO_LIST"],
    mask_assign_mode=os.environ["PREPROCESS_MASK_ASSIGN_MODE"],
)
train_specs = normalize_split_mask_config(args, "train", mask_assign_mode=args.mask_assign_mode)
eval_specs = normalize_split_mask_config(args, "eval", mask_assign_mode=args.mask_assign_mode)
print(
    build_output_stem(
        os.environ["PREPROCESSED_DATA_PREFIX"],
        train_specs,
        eval_specs,
        mask_assign_mode=args.mask_assign_mode,
    )
)
PY
}

if [[ "$PREPROCESS_MODE" == "single" ]]; then
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
else
  PREPROCESSED_DATA_DIR="${PREPROCESSED_DATA_DIR:-${PREPROCESSED_ROOT_DIR}/mixed}"
  PREPROCESSED_FILE_STEM="${PREPROCESSED_FILE_STEM:-$(compute_mixed_file_stem)}"
fi

DEMO_PATH="${DEMO_PATH:-${PREPROCESSED_DATA_DIR}/${PREPROCESSED_FILE_STEM}_train.h5}"
ACTION_NORM_PATH="${ACTION_NORM_PATH:-${DEMO_PATH}}"
TEST_DEMO_PATH="${TEST_DEMO_PATH:-}"
EVAL_DEMO_METADATA_PATH="${EVAL_DEMO_METADATA_PATH:-}"

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

ensure_preprocessed_dataset() {
  if [[ "$RUN_PREPROCESS" == "false" && -f "$DEMO_PATH" ]]; then
    return
  fi
  if [[ "$RUN_PREPROCESS" == "auto" && -f "$DEMO_PATH" ]]; then
    echo "[mam-preprocess] reuse existing ${PREPROCESS_MODE} dataset: ${DEMO_PATH}"
    return
  fi
  if [[ "$RUN_PREPROCESS" == "false" && ! -f "$DEMO_PATH" ]]; then
    echo "ERROR: demo file not found and RUN_PREPROCESS=false: ${DEMO_PATH}" >&2
    exit 1
  fi
  if [[ ! -f "$RAW_DEMO_H5" ]]; then
    echo "ERROR: raw demo h5 not found: ${RAW_DEMO_H5}" >&2
    exit 1
  fi

  mkdir -p "$(dirname "$DEMO_PATH")"
  echo "[mam-preprocess] generating ${PREPROCESS_MODE} dataset -> ${DEMO_PATH}"
  if [[ "$PREPROCESS_MODE" == "single" ]]; then
    PREPROCESS_ARGS=(
      --input-h5 "$RAW_DEMO_H5"
      --output-h5 "$DEMO_PATH"
      --env-id "$ENV_ID"
      --control-mode "$CONTROL_MODE"
      --action-dim "$ACTION_DIM"
      --mask-type "$SINGLE_MASK_TYPE"
      --retain-ratio "$SINGLE_MASK_RETAIN_RATIO"
      --mask-seq-len "$SINGLE_MASK_SEQ_LEN"
      --mask-value "$PREPROCESS_MASK_VALUE"
      --mask-seed "$MASK_SEED"
      --action-robust-margin "$ACTION_ROBUST_MARGIN"
      --state-robust-margin "$STATE_ROBUST_MARGIN"
      --image-size "$IMAGE_SIZE"
    )
  else
    PREPROCESS_ARGS=(
      --input-h5 "$RAW_DEMO_H5"
      --output-h5 "$DEMO_PATH"
      --env-id "$ENV_ID"
      --control-mode "$CONTROL_MODE"
      --action-dim "$ACTION_DIM"
      --mask-assign-mode "$PREPROCESS_MASK_ASSIGN_MODE"
      --train-num-mask-type "$TRAIN_NUM_MASK_TYPE"
      --train-mask-type-list "$TRAIN_MASK_TYPE_LIST"
      --train-mask-composition-list "$TRAIN_MASK_COMPOSITION_LIST"
      --train-mask-ratio-list "$TRAIN_MASK_RATIO_LIST"
      --mask-value "$PREPROCESS_MASK_VALUE"
      --mask-seed "$MASK_SEED"
      --action-robust-margin "$ACTION_ROBUST_MARGIN"
      --state-robust-margin "$STATE_ROBUST_MARGIN"
      --image-size "$IMAGE_SIZE"
    )
  fi
  if [[ -n "$PREPROCESS_NUM_TRAJ" ]]; then
    PREPROCESS_ARGS+=(--num-traj "$PREPROCESS_NUM_TRAJ")
  fi
  if [[ "$OVERWRITE_PREPROCESS" == "true" ]]; then
    PREPROCESS_ARGS+=(--overwrite)
  fi
  "$PYTHON_BIN" franka_train/preprocess_franka.py "${PREPROCESS_ARGS[@]}"
}

ensure_preprocessed_dataset

if [[ ! -f "$DEMO_PATH" ]]; then
  echo "ERROR: demo file not found: ${DEMO_PATH}" >&2
  exit 1
fi

# -----------------------------------------------------------------------------
# 4. STPM params. Training is in run_train_stpm_franka.sh; these are accepted for parity/future eval.
# -----------------------------------------------------------------------------

STPM_CONFIG_PATH="${STPM_CONFIG_PATH:-STPM_franka/config.yaml}"
STPM_CKPT_PATH="${STPM_CKPT_PATH:-}"

# -----------------------------------------------------------------------------
# 5. Model
# -----------------------------------------------------------------------------

NOISE_MODEL="${NOISE_MODEL:-Unet}" # Transformer or Unet
VISION_ENCODER="${VISION_ENCODER:-dino2}" # resnet, dino2, or dino3
DINO_MODEL_PATH="${DINO_MODEL_PATH:-}"
DINO_DATA_AUG="${DINO_DATA_AUG:-false}"
DIFFUSION_STEP_EMBED_DIM="${DIFFUSION_STEP_EMBED_DIM:-64}"
DIT_HIDDEN_DIM="${DIT_HIDDEN_DIM:-512}"
DIT_NUM_BLOCKS="${DIT_NUM_BLOCKS:-6}"
DIT_DIM_FEEDFORWARD="${DIT_DIM_FEEDFORWARD:-2048}"
UNET_DIMS="${UNET_DIMS:-256 512 1024}"
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

# -----------------------------------------------------------------------------
# 6. Train basic params
# -----------------------------------------------------------------------------

NUM_DEMOS="${NUM_DEMOS:-50}"
TOTAL_ITERS="${TOTAL_ITERS:-100000}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LR="${LR:-1e-4}"
NUM_DATALOAD_WORKERS="${NUM_DATALOAD_WORKERS:-0}"
OBS_MODE="${OBS_MODE:-rgb}"
DEMO_TYPE="${DEMO_TYPE:-franka_real_mam}"
OBS_HORIZON="${OBS_HORIZON:-2}"
ACT_HORIZON="${ACT_HORIZON:-8}"
PRED_HORIZON="${PRED_HORIZON:-16}"

# -----------------------------------------------------------------------------
# 7. MAS window and loss params
# -----------------------------------------------------------------------------

LONG_WINDOW_BACKWARD_LENGTH="${LONG_WINDOW_BACKWARD_LENGTH:-0}"
LONG_WINDOW_FORWARD_LENGTH="${LONG_WINDOW_FORWARD_LENGTH:-32}"
SHORT_WINDOW_HORIZON="${SHORT_WINDOW_HORIZON:-0}"
MAS_LONG_ENCODE_MODE="${MAS_LONG_ENCODE_MODE:-2DConv}"
MAS_LONG_CONV_OUTPUT_DIM="${MAS_LONG_CONV_OUTPUT_DIM:-128}"
LOSS_MODE="${LOSS_MODE:-average}" # average or weighted
LOSS_MASK_AREA_WEIGHT="${LOSS_MASK_AREA_WEIGHT:-0}"

# -----------------------------------------------------------------------------
# 8. Logging/checkpoint params. Eval args are accepted for parity; no online eval runs.
# -----------------------------------------------------------------------------

MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-}"
LOG_FREQ="${LOG_FREQ:-1000}"
EVAL_FREQ="${EVAL_FREQ:-0}"
VALID_FREQ="${VALID_FREQ:-0}"
NUM_VALIDATION_SET="${NUM_VALIDATION_SET:-0}"
SAVE_START_ITER="${SAVE_START_ITER:-0}"
SAVE_FREQ="${SAVE_FREQ:-5000}"
NUM_EVAL_EPISODES="${NUM_EVAL_EPISODES:-0}"
NUM_EVAL_DEMOS="${NUM_EVAL_DEMOS:-0}"
NUM_EVAL_ENVS="${NUM_EVAL_ENVS:-0}"
INPAINTING="${INPAINTING:-false}"
EVAL_PROGRESS_BAR="${EVAL_PROGRESS_BAR:-false}"
CAPTURE_VIDEO_FREQ="${CAPTURE_VIDEO_FREQ:-20}"
SIM_BACKEND="${SIM_BACKEND:-physx_cpu}"

# -----------------------------------------------------------------------------
# 9. Build train args
# -----------------------------------------------------------------------------

ARGS=(
  --exp-name "$EXP_NAME"
  --seed "$SEED"
  --wandb-project-name "$WANDB_PROJECT_NAME"
  --env-id "$ENV_ID"
  --action-dim "$ACTION_DIM"
  --demo-path "$DEMO_PATH"
  --action-norm-path "$ACTION_NORM_PATH"
  --stpm-config-path "$STPM_CONFIG_PATH"
  --total-iters "$TOTAL_ITERS"
  --batch-size "$BATCH_SIZE"
  --lr "$LR"
  --obs-horizon "$OBS_HORIZON"
  --act-horizon "$ACT_HORIZON"
  --pred-horizon "$PRED_HORIZON"
  --long-window-backward-length "$LONG_WINDOW_BACKWARD_LENGTH"
  --long-window-forward-length "$LONG_WINDOW_FORWARD_LENGTH"
  --noise-model "$NOISE_MODEL"
  --vision-encoder "$VISION_ENCODER"
  --dino-model-path "$DINO_MODEL_PATH"
  --diffusion-step-embed-dim "$DIFFUSION_STEP_EMBED_DIM"
  --dit-hidden-dim "$DIT_HIDDEN_DIM"
  --dit-num-blocks "$DIT_NUM_BLOCKS"
  --dit-dim-feedforward "$DIT_DIM_FEEDFORWARD"
  --unet-dims $UNET_DIMS
  --n-groups "$N_GROUPS"
  --short-window-horizon "$SHORT_WINDOW_HORIZON"
  --mas-long-encode-mode "$MAS_LONG_ENCODE_MODE"
  --mas-long-conv-output-dim "$MAS_LONG_CONV_OUTPUT_DIM"
  --loss-mode "$LOSS_MODE"
  --loss-mask-area-weight "$LOSS_MASK_AREA_WEIGHT"
  --obs-mode "$OBS_MODE"
  --log-freq "$LOG_FREQ"
  --eval-freq "$EVAL_FREQ"
  --valid-freq "$VALID_FREQ"
  --num-validation-set "$NUM_VALIDATION_SET"
  --num-eval-episodes "$NUM_EVAL_EPISODES"
  --num-eval-envs "$NUM_EVAL_ENVS"
  --capture-video-freq "$CAPTURE_VIDEO_FREQ"
  --save-start-iter "$SAVE_START_ITER"
  --save-freq "$SAVE_FREQ"
  --sim-backend "$SIM_BACKEND"
  --num-dataload-workers "$NUM_DATALOAD_WORKERS"
  --control-mode "$CONTROL_MODE"
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
if [[ "$INPAINTING" == "true" ]]; then
  ARGS+=(--inpainting)
else
  ARGS+=(--no-inpainting)
fi
if [[ "$EVAL_PROGRESS_BAR" == "true" ]]; then
  ARGS+=(--eval-progress-bar)
else
  ARGS+=(--no-eval-progress-bar)
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
if [[ -n "$TEST_DEMO_PATH" ]]; then
  ARGS+=(--test-demo-path "$TEST_DEMO_PATH")
fi
if [[ -n "$EVAL_DEMO_METADATA_PATH" ]]; then
  ARGS+=(--eval-demo-metadata-path "$EVAL_DEMO_METADATA_PATH")
fi
if [[ -n "$STPM_CKPT_PATH" ]]; then
  ARGS+=(--stpm-ckpt-path "$STPM_CKPT_PATH")
fi
if [[ -n "$MAX_EPISODE_STEPS" ]]; then
  ARGS+=(--max-episode-steps "$MAX_EPISODE_STEPS")
fi
if [[ -n "$DEMO_TYPE" ]]; then
  ARGS+=(--demo-type "$DEMO_TYPE")
fi

"$PYTHON_BIN" franka_train/train_mam_franka.py "${ARGS[@]}"
