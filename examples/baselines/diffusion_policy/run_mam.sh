#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

# -----------------------------------------------------------------------------
# 1. Basic info 
# -----------------------------------------------------------------------------

ENV_ID="${ENV_ID:-PickCube-v1}"
EXP_NAME="${EXP_NAME:-PickCube_mam}"
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

RAW_DEMO_H5="${RAW_DEMO_H5:-demos/data_1/data_1.h5}"
RAW_DEMO_JSON="${RAW_DEMO_JSON:-demos/data_1/data_1.json}"

# -----------------------------------------------------------------------------
# 3. Preprocess params (mask type) 
# -----------------------------------------------------------------------------

PREPROCESSED_ROOT_DIR="${PREPROCESSED_ROOT_DIR:-demos/data_1_preprocessed}"
PREPROCESSED_DATA_DIR="${PREPROCESSED_DATA_DIR:-}"
PREPROCESSED_DATA_PREFIX="${PREPROCESSED_DATA_PREFIX:-${ENV_ID}}" #预处理输出文件名基础前缀
PREPROCESS_MASK_TYPE="${PREPROCESS_MASK_TYPE:-3D_points}"
PREPROCESS_RETAIN_RATIO="${PREPROCESS_RETAIN_RATIO:-0.5}"
PREPROCESS_MASK_SEQ_LEN="${PREPROCESS_MASK_SEQ_LEN:-20}"
PREPROCESS_MASK_VALUE="${PREPROCESS_MASK_VALUE:-0}"
PREPROCESS_NUM_TRAJ="${PREPROCESS_NUM_TRAJ:-}"
PREPROCESS_MASK_ASSIGN_MODE="${PREPROCESS_MASK_ASSIGN_MODE:-composition}" # composition 或 one_demo_multi_mask

TRAIN_NUM_MASK_TYPE="${TRAIN_NUM_MASK_TYPE:-${NUM_MASK_TYPE:-2}}"
if [[ "$TRAIN_NUM_MASK_TYPE" == "1" ]]; then
  TRAIN_MASK_TYPE_LIST=${TRAIN_MASK_TYPE_LIST:-${MASK_TYPE_LIST:-"[\"${PREPROCESS_MASK_TYPE}\"]"}}
  TRAIN_MASK_COMPOSITION_LIST=${TRAIN_MASK_COMPOSITION_LIST:-${MASK_COMPOSITION_LIST:-'[1.0]'}}
  TRAIN_MASK_RATIO_LIST=${TRAIN_MASK_RATIO_LIST:-${MASK_RATIO_LIST:-"[${PREPROCESS_RETAIN_RATIO}]"}}
else
  TRAIN_MASK_TYPE_LIST=${TRAIN_MASK_TYPE_LIST:-${MASK_TYPE_LIST:-'["random_mask","points"]'}}
  TRAIN_MASK_COMPOSITION_LIST=${TRAIN_MASK_COMPOSITION_LIST:-${MASK_COMPOSITION_LIST:-'[0.5,0.5]'}}
  TRAIN_MASK_RATIO_LIST=${TRAIN_MASK_RATIO_LIST:-${MASK_RATIO_LIST:-'[0.2,0.2]'}}
fi

EVAL_NUM_MASK_TYPE="${EVAL_NUM_MASK_TYPE:-${TRAIN_NUM_MASK_TYPE}}"
EVAL_MASK_TYPE_LIST=${EVAL_MASK_TYPE_LIST:-${TRAIN_MASK_TYPE_LIST}}
EVAL_MASK_COMPOSITION_LIST=${EVAL_MASK_COMPOSITION_LIST:-${TRAIN_MASK_COMPOSITION_LIST}}
EVAL_MASK_RATIO_LIST=${EVAL_MASK_RATIO_LIST:-${TRAIN_MASK_RATIO_LIST}}

eval "$(
TRAIN_NUM_MASK_TYPE="$TRAIN_NUM_MASK_TYPE" \
TRAIN_MASK_TYPE_LIST="$TRAIN_MASK_TYPE_LIST" \
TRAIN_MASK_COMPOSITION_LIST="$TRAIN_MASK_COMPOSITION_LIST" \
EVAL_MASK_TYPE_LIST="$EVAL_MASK_TYPE_LIST" \
EVAL_NUM_MASK_TYPE="$EVAL_NUM_MASK_TYPE" \
EVAL_MASK_COMPOSITION_LIST="$EVAL_MASK_COMPOSITION_LIST" \
TRAIN_MASK_RATIO_LIST="$TRAIN_MASK_RATIO_LIST" \
EVAL_MASK_RATIO_LIST="$EVAL_MASK_RATIO_LIST" \
python - <<'PY'
import ast
import json
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


train_num = int(os.environ.get("TRAIN_NUM_MASK_TYPE", "0"))
eval_num = int(os.environ.get("EVAL_NUM_MASK_TYPE", str(train_num)))
train_types = [str(v) for v in normalize(parse_list(os.environ.get("TRAIN_MASK_TYPE_LIST", "[]")), train_num, "random_mask")]
eval_types = [str(v) for v in normalize(parse_list(os.environ.get("EVAL_MASK_TYPE_LIST", "[]")), eval_num, train_types[0] if train_types else "random_mask")]
train_ratios = [float(v) for v in normalize(parse_list(os.environ.get("TRAIN_MASK_RATIO_LIST", "[]")), train_num, 0.2)]
eval_ratios = [float(v) for v in normalize(parse_list(os.environ.get("EVAL_MASK_RATIO_LIST", "[]")), eval_num, train_ratios[0] if train_ratios else 0.2)]
train_comp = [float(v) for v in normalize(parse_list(os.environ.get("TRAIN_MASK_COMPOSITION_LIST", "[]")), train_num, 1.0 / train_num if train_num else 1.0)]
eval_comp = [float(v) for v in normalize(parse_list(os.environ.get("EVAL_MASK_COMPOSITION_LIST", "[]")), eval_num, 1.0 / eval_num if eval_num else 1.0)]

single = train_num == 1 and eval_num == 1 and len(train_types) == 1 and len(eval_types) == 1 and train_types[0] == eval_types[0]
assignments = {
    "TRAIN_MASK_TYPE_LIST": json.dumps(train_types),
    "EVAL_MASK_TYPE_LIST": json.dumps(eval_types),
    "TRAIN_MASK_COMPOSITION_LIST": json.dumps(train_comp),
    "EVAL_MASK_COMPOSITION_LIST": json.dumps(eval_comp),
    "TRAIN_MASK_RATIO_LIST": json.dumps(train_ratios),
    "EVAL_MASK_RATIO_LIST": json.dumps(eval_ratios),
}
if single:
    ratio = train_ratios[0] if train_ratios else (eval_ratios[0] if eval_ratios else 0.2)
    assignments["SINGLE_MASK_COMPAT"] = "true"
    assignments["SINGLE_MASK_TYPE"] = train_types[0]
    assignments["SINGLE_MASK_PARAM"] = format(float(ratio), "g")
else:
    assignments["SINGLE_MASK_COMPAT"] = "false"

for key, value in assignments.items():
    print(f"{key}={shlex.quote(value)}")
PY
)"

export \
  TRAIN_NUM_MASK_TYPE TRAIN_MASK_TYPE_LIST TRAIN_MASK_COMPOSITION_LIST TRAIN_MASK_RATIO_LIST \
  EVAL_NUM_MASK_TYPE EVAL_MASK_TYPE_LIST EVAL_MASK_COMPOSITION_LIST EVAL_MASK_RATIO_LIST \
  PREPROCESSED_DATA_PREFIX PREPROCESS_MASK_ASSIGN_MODE

# -----------------------------------------------------------------------------
# 4. STPM params 
# -----------------------------------------------------------------------------

STPM_CONFIG_PATH="${STPM_CONFIG_PATH:-STPM_PickCube/pick up the cube and place it at the goal/config.yaml}"
STPM_CKPT_PATH="${STPM_CKPT_PATH:-STPM_PickCube/pick up the cube and place it at the goal/checkpoints/reward_best.pt}"

# -----------------------------------------------------------------------------
# 5. Train basic params 
# -----------------------------------------------------------------------------

NUM_DEMOS="${NUM_DEMOS:-100}"
TOTAL_ITERS="${TOTAL_ITERS:-200}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-1e-4}"
NUM_DATALOAD_WORKERS="${NUM_DATALOAD_WORKERS:-0}"
CONTROL_MODE="${CONTROL_MODE:-pd_ee_pose}"
OBS_MODE="${OBS_MODE:-rgb+depth}"  # rgb or rgb+depth; rgb ignores dataset depth for policy input
DEMO_TYPE="${DEMO_TYPE:-}"

# -----------------------------------------------------------------------------
# 6. MAS and diffusion-policy params 
# -----------------------------------------------------------------------------

OBS_HORIZON="${OBS_HORIZON:-2}"
ACT_HORIZON="${ACT_HORIZON:-8}"
PRED_HORIZON="${PRED_HORIZON:-16}"
LONG_WINDOW_BACKWARD_LENGTH="${LONG_WINDOW_BACKWARD_LENGTH:-0}"
LONG_WINDOW_FORWARD_LENGTH="${LONG_WINDOW_FORWARD_LENGTH:-${PRED_HORIZON}}"
DIFFUSION_STEP_EMBED_DIM="${DIFFUSION_STEP_EMBED_DIM:-64}"
SHORT_WINDOW_HORIZON="${SHORT_WINDOW_HORIZON:-2}"
MAS_LONG_ENCODE_MODE="${MAS_LONG_ENCODE_MODE:-2DConv}"
MAS_LONG_CONV_OUTPUT_DIM="${MAS_LONG_CONV_OUTPUT_DIM:-64}"
LOSS_MODE="${LOSS_MODE:-average}" #average or weighted
LOSS_MASK_AREA_WEIGHT="${LOSS_MASK_AREA_WEIGHT:-0.2}"

# -----------------------------------------------------------------------------
# 7. Eval, logging, and checkpoint params
# -----------------------------------------------------------------------------

MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-100}"
LOG_FREQ="${LOG_FREQ:-100}"
EVAL_FREQ="${EVAL_FREQ:-100}"
SAVE_FREQ="${SAVE_FREQ:-100}"
NUM_EVAL_EPISODES="${NUM_EVAL_EPISODES:-100}"
NUM_EVAL_DEMOS="${NUM_EVAL_DEMOS:-100}"
NUM_EVAL_ENVS="${NUM_EVAL_ENVS:-10}"
INPAINTING="${INPAINTING:-false}"
EVAL_PROGRESS_BAR="${EVAL_PROGRESS_BAR:-false}"
CAPTURE_VIDEO_FREQ="${CAPTURE_VIDEO_FREQ:-5}"
SIM_BACKEND="${SIM_BACKEND:-gpu}"

PREPROCESS_MODE="mixed"
if [[ "$SINGLE_MASK_COMPAT" == "true" ]]; then
  PREPROCESS_MODE="single"
  PREPROCESS_MASK_TYPE="$SINGLE_MASK_TYPE"
  case "$PREPROCESS_MASK_TYPE" in
    2D_partial_trajectory|local_planner)
      PREPROCESS_MASK_SEQ_LEN="${SINGLE_MASK_PARAM}"
      ;;
    *)
      PREPROCESS_RETAIN_RATIO="${SINGLE_MASK_PARAM}"
      ;;
  esac
  echo "[mam] single mask type detected: ${PREPROCESS_MASK_TYPE}; using single-mask preprocess and train_mam.py"
fi

# -----------------------------------------------------------------------------
# 8. Derived preprocessed paths 
# -----------------------------------------------------------------------------

compute_mixed_file_stem() {
  python - <<'PY'
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
    eval_num_mask_type=int(os.environ["EVAL_NUM_MASK_TYPE"]),
    eval_mask_type_list=os.environ["EVAL_MASK_TYPE_LIST"],
    eval_mask_composition_list=os.environ["EVAL_MASK_COMPOSITION_LIST"],
    eval_mask_ratio_list=os.environ["EVAL_MASK_RATIO_LIST"],
    mask_assign_mode=os.environ["PREPROCESS_MASK_ASSIGN_MODE"],
)

train_mask_specs = normalize_split_mask_config(
    args,
    "train",
    mask_assign_mode=args.mask_assign_mode,
)
eval_mask_specs = normalize_split_mask_config(
    args,
    "eval",
    mask_assign_mode=args.mask_assign_mode,
)
output_stem = build_output_stem(
    os.environ["PREPROCESSED_DATA_PREFIX"],
    train_mask_specs,
    eval_mask_specs,
    mask_assign_mode=args.mask_assign_mode,
)
print(output_stem)
PY
}

if [[ "$PREPROCESS_MODE" == "single" ]]; then
  case "$PREPROCESS_MASK_TYPE" in
    pose_AnyGrasp|pose_motion_planning|points|3D_points|random_mask)
      PREPROCESS_DIR_SUFFIX="${PREPROCESS_MASK_TYPE}_${PREPROCESS_RETAIN_RATIO}"
      ;;
    2D_partial_trajectory|local_planner)
      PREPROCESS_DIR_SUFFIX="${PREPROCESS_MASK_TYPE}_seq${PREPROCESS_MASK_SEQ_LEN}"
      ;;
    2D_video_trajectory|2D_image_trajectory)
      PREPROCESS_DIR_SUFFIX="${PREPROCESS_MASK_TYPE}"
      ;;
    *)
      PREPROCESS_DIR_SUFFIX="${PREPROCESS_MASK_TYPE}"
      ;;
  esac
  PREPROCESSED_DATA_DIR="${PREPROCESSED_DATA_DIR:-${PREPROCESSED_ROOT_DIR}/${PREPROCESS_DIR_SUFFIX}}"
  PREPROCESSED_FILE_STEM="${PREPROCESSED_FILE_STEM:-${PREPROCESSED_DATA_PREFIX}_${PREPROCESS_DIR_SUFFIX}}"
else
  PREPROCESSED_DATA_DIR="${PREPROCESSED_DATA_DIR:-${PREPROCESSED_ROOT_DIR}/mixed}"
  PREPROCESSED_FILE_STEM="${PREPROCESSED_FILE_STEM:-$(compute_mixed_file_stem)}"
fi
DEMO_PATH="${DEMO_PATH:-${PREPROCESSED_DATA_DIR}/${PREPROCESSED_FILE_STEM}_train.h5}"
TEST_DEMO_PATH="${TEST_DEMO_PATH:-${PREPROCESSED_DATA_DIR}/${PREPROCESSED_FILE_STEM}_eval.h5}"
EVAL_DEMO_METADATA_PATH="${EVAL_DEMO_METADATA_PATH:-${PREPROCESSED_DATA_DIR}/${PREPROCESSED_FILE_STEM}_eval.json}"
TRAIN_DEMO_METADATA_PATH="${TRAIN_DEMO_METADATA_PATH:-${PREPROCESSED_DATA_DIR}/${PREPROCESSED_FILE_STEM}_train.json}"
ACTION_NORM_PATH="${ACTION_NORM_PATH:-${DEMO_PATH}}"

# -----------------------------------------------------------------------------
# 9. Preprocess helper 
# -----------------------------------------------------------------------------

ensure_preprocessed_dataset() {
  local missing=0
  for path in "$DEMO_PATH" "$TEST_DEMO_PATH" "$TRAIN_DEMO_METADATA_PATH" "$EVAL_DEMO_METADATA_PATH"; do
    if [[ ! -f "$path" ]]; then
      missing=1
      break
    fi
  done

  if [[ "$missing" -eq 0 ]]; then
    echo "[mam-preprocess] reuse existing ${PREPROCESS_MODE} dataset: ${PREPROCESSED_DATA_DIR}"
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
  echo "[mam-preprocess] generating ${PREPROCESS_MODE} dataset into ${PREPROCESSED_DATA_DIR}"
  if [[ "$PREPROCESS_MODE" == "single" ]]; then
    PREPROCESS_ARGS=(
      --input-h5 "$RAW_DEMO_H5"
      --input-json "$RAW_DEMO_JSON"
      --output-dir "$PREPROCESSED_DATA_DIR"
      --output-prefix "$PREPROCESSED_DATA_PREFIX"
      --env-id "$ENV_ID"
      --mask-type "$PREPROCESS_MASK_TYPE"
      --retain-ratio "$PREPROCESS_RETAIN_RATIO"
      --mask-seq-len "$PREPROCESS_MASK_SEQ_LEN"
      --mask-value "$PREPROCESS_MASK_VALUE"
    )
    if [[ -n "$PREPROCESS_NUM_TRAJ" ]]; then
      PREPROCESS_ARGS+=(--num-traj "$PREPROCESS_NUM_TRAJ")
    fi
    python examples/baselines/diffusion_policy/data_preprocess/data_preprocess.py "${PREPROCESS_ARGS[@]}"
  else
    PREPROCESS_ARGS=(
      --input-h5 "$RAW_DEMO_H5"
      --input-json "$RAW_DEMO_JSON"
      --output-dir "$PREPROCESSED_DATA_DIR"
      --output-prefix "$PREPROCESSED_DATA_PREFIX"
      --env-id "$ENV_ID"
      --mask-assign-mode "$PREPROCESS_MASK_ASSIGN_MODE"
      --train-num-mask-type "$TRAIN_NUM_MASK_TYPE"
      --train-mask-type-list "$TRAIN_MASK_TYPE_LIST"
      --train-mask-composition-list "$TRAIN_MASK_COMPOSITION_LIST"
      --train-mask-ratio-list "$TRAIN_MASK_RATIO_LIST"
      --eval-num-mask-type "$EVAL_NUM_MASK_TYPE"
      --eval-mask-type-list "$EVAL_MASK_TYPE_LIST"
      --eval-mask-composition-list "$EVAL_MASK_COMPOSITION_LIST"
      --eval-mask-ratio-list "$EVAL_MASK_RATIO_LIST"
      --mask-value "$PREPROCESS_MASK_VALUE"
    )
    if [[ -n "$PREPROCESS_NUM_TRAJ" ]]; then
      PREPROCESS_ARGS+=(--num-traj "$PREPROCESS_NUM_TRAJ")
    fi
    python examples/baselines/diffusion_policy/data_preprocess/data_preprocess_mixed.py "${PREPROCESS_ARGS[@]}"
  fi
}

ensure_preprocessed_dataset

# -----------------------------------------------------------------------------
# 10. Build train args 
# -----------------------------------------------------------------------------

ARGS=(
  --seed "$SEED"
  --wandb-project-name "$WANDB_PROJECT_NAME"
  --env-id "$ENV_ID"
  --demo-path "$DEMO_PATH"
  --test-demo-path "$TEST_DEMO_PATH"
  --eval-demo-metadata-path "$EVAL_DEMO_METADATA_PATH"
  --stpm-ckpt-path "$STPM_CKPT_PATH"
  --stpm-config-path "$STPM_CONFIG_PATH"
  --total-iters "$TOTAL_ITERS"
  --batch-size "$BATCH_SIZE"
  --lr "$LR"
  --obs-horizon "$OBS_HORIZON"
  --act-horizon "$ACT_HORIZON"
  --pred-horizon "$PRED_HORIZON"
  --long-window-backward-length "$LONG_WINDOW_BACKWARD_LENGTH"
  --long-window-forward-length "$LONG_WINDOW_FORWARD_LENGTH"
  --diffusion-step-embed-dim "$DIFFUSION_STEP_EMBED_DIM"
  --short-window-horizon "$SHORT_WINDOW_HORIZON"
  --mas-long-encode-mode "$MAS_LONG_ENCODE_MODE"
  --mas-long-conv-output-dim "$MAS_LONG_CONV_OUTPUT_DIM"
  --loss-mode "$LOSS_MODE"
  --loss-mask-area-weight "$LOSS_MASK_AREA_WEIGHT"
  --log-freq "$LOG_FREQ"
  --eval-freq "$EVAL_FREQ"
  --num-eval-episodes "$NUM_EVAL_EPISODES"
  --num-eval-demos "$NUM_EVAL_DEMOS"
  --num-eval-envs "$NUM_EVAL_ENVS"
  --capture-video-freq "$CAPTURE_VIDEO_FREQ"
  --action-norm-path "$ACTION_NORM_PATH"
  --sim-backend "$SIM_BACKEND"
  --num-dataload-workers "$NUM_DATALOAD_WORKERS"
  --control-mode "$CONTROL_MODE"
  --obs-mode "$OBS_MODE"
)

# bool/optional 参数单独追加，确保 tyro 能正确解析 true/false 开关。
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
if [[ -n "$NUM_DEMOS" ]]; then
  ARGS+=(--num-demos "$NUM_DEMOS")
fi

# -----------------------------------------------------------------------------
# 11. Validate required files
# -----------------------------------------------------------------------------

if [[ ! -f "$DEMO_PATH" ]]; then
  echo "ERROR: demo file not found: $DEMO_PATH" >&2
  exit 1
fi
if [[ -z "$TEST_DEMO_PATH" ]]; then
  echo "ERROR: TEST_DEMO_PATH is required" >&2
  exit 1
fi
if [[ ! -f "$TEST_DEMO_PATH" ]]; then
  echo "ERROR: test demo file not found: $TEST_DEMO_PATH" >&2
  exit 1
fi
if [[ -n "$EVAL_DEMO_METADATA_PATH" && ! -f "$EVAL_DEMO_METADATA_PATH" ]]; then
  echo "ERROR: eval demo metadata file not found: $EVAL_DEMO_METADATA_PATH" >&2
  exit 1
fi
if [[ ! -f "$STPM_CONFIG_PATH" ]]; then
  echo "ERROR: STPM config file not found: $STPM_CONFIG_PATH" >&2
  exit 1
fi
if [[ ! -f "$STPM_CKPT_PATH" ]]; then
  echo "ERROR: STPM checkpoint file not found: $STPM_CKPT_PATH" >&2
  exit 1
fi
if [[ -z "$ACTION_NORM_PATH" || ! -f "$ACTION_NORM_PATH" ]]; then
  echo "ERROR: action norm path not found: $ACTION_NORM_PATH" >&2
  exit 1
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


python examples/baselines/diffusion_policy/train_mam.py "${ARGS[@]}"
