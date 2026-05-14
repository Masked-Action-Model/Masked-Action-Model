#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/examples/baselines/diffusion_policy:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-maniskill}"

EXP_NAME="${EXP_NAME:-PlaceSphere_mam_overfit5}"
SEED="${SEED:-1}"
CUDA="${CUDA:-true}"
TRACK="${TRACK:-false}"
CAPTURE_VIDEO="${CAPTURE_VIDEO:-false}"
WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME:-ManiSkill}"
WANDB_ENTITY="${WANDB_ENTITY:-}"

RAW_RGB_DEMO_H5="${RAW_RGB_DEMO_H5:-demos/PlaceSphere-v1/PlaceSphere-v1.rgb.pd_ee_pose.physx_cpu.h5}"
RAW_RGB_DEMO_JSON="${RAW_RGB_DEMO_JSON:-${RAW_RGB_DEMO_H5%.h5}.json}"
OVERFIT_NUM_DEMOS="${OVERFIT_NUM_DEMOS:-5}"
OVERFIT_DATA_ROOT="${OVERFIT_DATA_ROOT:-examples/baselines/diffusion_policy/tmp/placesphere_overfit_data}"
PREPROCESSED_DATA_DIR="${PREPROCESSED_DATA_DIR:-${OVERFIT_DATA_ROOT}/overfit${OVERFIT_NUM_DEMOS}/mam}"
DEMO_PATH="${DEMO_PATH:-${PREPROCESSED_DATA_DIR}/placesphere_mam_overfit${OVERFIT_NUM_DEMOS}_train.h5}"
TRAIN_DEMO_METADATA_PATH="${TRAIN_DEMO_METADATA_PATH:-${PREPROCESSED_DATA_DIR}/placesphere_mam_overfit${OVERFIT_NUM_DEMOS}_train.json}"
TEST_DEMO_PATH="${TEST_DEMO_PATH:-${PREPROCESSED_DATA_DIR}/placesphere_mam_overfit${OVERFIT_NUM_DEMOS}_eval.h5}"
EVAL_DEMO_METADATA_PATH="${EVAL_DEMO_METADATA_PATH:-${PREPROCESSED_DATA_DIR}/placesphere_mam_overfit${OVERFIT_NUM_DEMOS}_eval.json}"

ENV_ID="${ENV_ID:-PlaceSphere-v1}"
ACTION_DIM="${ACTION_DIM:-7}"
CONTROL_MODE="${CONTROL_MODE:-pd_ee_pose}"
OBS_MODE="${OBS_MODE:-rgb}"
SIM_BACKEND="${SIM_BACKEND:-physx_cpu}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-150}"

STPM_CONFIG_PATH="${STPM_CONFIG_PATH:-STPM_placesphere/config.yaml}"
STPM_CKPT_PATH="${STPM_CKPT_PATH:-STPM_placesphere/checkpoints/reward_best.pt}"

MASK_TYPE_LIST="${MASK_TYPE_LIST:-[\"random_mask\"]}"
MASK_RATIO_LIST="${MASK_RATIO_LIST:-[0.2]}"
MASK_COMPOSITION_LIST="${MASK_COMPOSITION_LIST:-[1]}"
NUM_MASK_TYPE="${NUM_MASK_TYPE:-1}"
PREPROCESS_MASK_ASSIGN_MODE="${PREPROCESS_MASK_ASSIGN_MODE:-composition}"
PREPROCESS_MASK_VALUE="${PREPROCESS_MASK_VALUE:-0}"
PREPROCESS_OVERWRITE="${PREPROCESS_OVERWRITE:-true}"
MASK_SEED="${MASK_SEED:-0}"

eval "$(
MASK_TYPE_LIST="$MASK_TYPE_LIST" \
MASK_RATIO_LIST="$MASK_RATIO_LIST" \
MASK_COMPOSITION_LIST="$MASK_COMPOSITION_LIST" \
python - <<'PY'
import ast
import json
import os
import shlex


ALIASES = {"random": "random_mask"}


def parse_list(value):
    text = str(value or "").strip()
    if not text:
        return []
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
            if isinstance(parsed, list):
                return parsed
            if isinstance(parsed, str):
                return [parsed]
        except Exception:
            pass
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1].strip()
    if "," in text:
        return [item.strip().strip("'\"") for item in text.split(",") if item.strip()]
    return [text.strip("'\"")]


def normalize(values, count, default):
    values = list(values)
    if count <= 0:
        return []
    if not values:
        return [default for _ in range(count)]
    if len(values) == count:
        return values
    if len(values) == 1:
        return values * count
    if len(values) > count:
        return values[:count]
    return values + [values[-1] for _ in range(count - len(values))]


mask_types = [ALIASES.get(str(v), str(v)) for v in parse_list(os.environ["MASK_TYPE_LIST"])]
count = len(mask_types)
mask_ratios = [float(v) for v in normalize(parse_list(os.environ["MASK_RATIO_LIST"]), count, 0.2)]
mask_comp = [float(v) for v in normalize(parse_list(os.environ["MASK_COMPOSITION_LIST"]), count, 1.0)]
total = sum(mask_comp)
if total <= 0:
    raise ValueError(f"MASK_COMPOSITION_LIST must have positive sum, got {mask_comp}")
mask_comp = [v / total for v in mask_comp]

assignments = {
    "MASK_TYPE_LIST": json.dumps(mask_types),
    "MASK_RATIO_LIST": json.dumps(mask_ratios),
    "MASK_COMPOSITION_LIST": json.dumps(mask_comp),
    "NUM_MASK_TYPE": str(count),
}
for key, value in assignments.items():
    print(f"{key}={shlex.quote(value)}")
PY
)"

TOTAL_ITERS="${TOTAL_ITERS:-30000}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LR="${LR:-1e-4}"
NUM_DATALOAD_WORKERS="${NUM_DATALOAD_WORKERS:-0}"
OBS_HORIZON="${OBS_HORIZON:-2}"
ACT_HORIZON="${ACT_HORIZON:-8}"
PRED_HORIZON="${PRED_HORIZON:-16}"
LONG_WINDOW_BACKWARD_LENGTH="${LONG_WINDOW_BACKWARD_LENGTH:-0}"
LONG_WINDOW_FORWARD_LENGTH="${LONG_WINDOW_FORWARD_LENGTH:-${PRED_HORIZON}}"
NOISE_MODEL="${NOISE_MODEL:-Transformer}"
DIFFUSION_STEP_EMBED_DIM="${DIFFUSION_STEP_EMBED_DIM:-64}"
DIT_HIDDEN_DIM="${DIT_HIDDEN_DIM:-512}"
DIT_NUM_BLOCKS="${DIT_NUM_BLOCKS:-6}"
DIT_DIM_FEEDFORWARD="${DIT_DIM_FEEDFORWARD:-2048}"
UNET_DIMS="${UNET_DIMS:-64 128 256}"
N_GROUPS="${N_GROUPS:-8}"
SHORT_WINDOW_HORIZON="${SHORT_WINDOW_HORIZON:-0}"
MAS_LONG_ENCODE_MODE="${MAS_LONG_ENCODE_MODE:-2DConv}"
MAS_LONG_CONV_OUTPUT_DIM="${MAS_LONG_CONV_OUTPUT_DIM:-128}"
LOSS_MODE="${LOSS_MODE:-average}"
LOSS_MASK_AREA_WEIGHT="${LOSS_MASK_AREA_WEIGHT:-0.2}"

LOG_FREQ="${LOG_FREQ:-1000}"
EVAL_FREQ="${EVAL_FREQ:-2000}"
SAVE_FREQ="${SAVE_FREQ:-2000}"
NUM_DEMOS="${NUM_DEMOS:-${OVERFIT_NUM_DEMOS}}"
NUM_EVAL_DEMOS="${NUM_EVAL_DEMOS:-${OVERFIT_NUM_DEMOS}}"
NUM_EVAL_EPISODES="${NUM_EVAL_EPISODES:-${OVERFIT_NUM_DEMOS}}"
NUM_EVAL_ENVS="${NUM_EVAL_ENVS:-1}"
INPAINTING="${INPAINTING:-true}"
EVAL_PROGRESS_BAR="${EVAL_PROGRESS_BAR:-false}"
CAPTURE_VIDEO_FREQ="${CAPTURE_VIDEO_FREQ:-5}"
DEMO_TYPE="${DEMO_TYPE:-placesphere_mam_overfit5}"
ACTION_NORM_PATH="${ACTION_NORM_PATH:-$DEMO_PATH}"

if [[ ! -f "$RAW_RGB_DEMO_H5" ]]; then
  echo "ERROR: replayed rgb/pd_ee_pose demo h5 not found: $RAW_RGB_DEMO_H5" >&2
  echo "Replay PlaceSphere first, or set RAW_RGB_DEMO_H5 to the replayed h5 path." >&2
  exit 1
fi
if [[ ! -f "$RAW_RGB_DEMO_JSON" ]]; then
  echo "ERROR: replayed rgb/pd_ee_pose demo json not found: $RAW_RGB_DEMO_JSON" >&2
  exit 1
fi

PREPROCESS_ARGS=(
  --input-h5 "$RAW_RGB_DEMO_H5"
  --input-json "$RAW_RGB_DEMO_JSON"
  --output-root "$OVERFIT_DATA_ROOT"
  --num-demos-list "$OVERFIT_NUM_DEMOS"
  --env-id "$ENV_ID"
  --action-dim "$ACTION_DIM"
  --mask-assign-mode "$PREPROCESS_MASK_ASSIGN_MODE"
  --num-mask-type "$NUM_MASK_TYPE"
  --mask-type-list "$MASK_TYPE_LIST"
  --mask-composition-list "$MASK_COMPOSITION_LIST"
  --mask-ratio-list "$MASK_RATIO_LIST"
  --mask-value "$PREPROCESS_MASK_VALUE"
  --mask-seed "$MASK_SEED"
)
if [[ "$PREPROCESS_OVERWRITE" == "true" ]]; then
  PREPROCESS_ARGS+=(--overwrite)
fi
python examples/baselines/diffusion_policy/tmp/prepare_placesphere_overfit_data.py "${PREPROCESS_ARGS[@]}"

for path in "$DEMO_PATH" "$TRAIN_DEMO_METADATA_PATH" "$TEST_DEMO_PATH" "$EVAL_DEMO_METADATA_PATH"; do
  if [[ ! -f "$path" ]]; then
    echo "ERROR: expected overfit dataset file not found: $path" >&2
    exit 1
  fi
done

export \
  EXP_NAME SEED CUDA TRACK CAPTURE_VIDEO WANDB_PROJECT_NAME WANDB_ENTITY \
  RAW_DEMO_H5="$RAW_RGB_DEMO_H5" RAW_DEMO_JSON="$RAW_RGB_DEMO_JSON" \
  PREPROCESSED_DATA_DIR DEMO_PATH TRAIN_DEMO_METADATA_PATH TEST_DEMO_PATH EVAL_DEMO_METADATA_PATH \
  ENV_ID ACTION_DIM CONTROL_MODE OBS_MODE SIM_BACKEND MAX_EPISODE_STEPS \
  STPM_CONFIG_PATH STPM_CKPT_PATH MASK_TYPE_LIST MASK_RATIO_LIST MASK_COMPOSITION_LIST \
  PREPROCESS_MASK_ASSIGN_MODE PREPROCESS_MASK_VALUE TOTAL_ITERS BATCH_SIZE LR \
  NUM_DATALOAD_WORKERS OBS_HORIZON ACT_HORIZON PRED_HORIZON \
  LONG_WINDOW_BACKWARD_LENGTH LONG_WINDOW_FORWARD_LENGTH NOISE_MODEL DIFFUSION_STEP_EMBED_DIM \
  DIT_HIDDEN_DIM DIT_NUM_BLOCKS DIT_DIM_FEEDFORWARD UNET_DIMS N_GROUPS SHORT_WINDOW_HORIZON \
  MAS_LONG_ENCODE_MODE MAS_LONG_CONV_OUTPUT_DIM LOSS_MODE LOSS_MASK_AREA_WEIGHT \
  LOG_FREQ EVAL_FREQ SAVE_FREQ NUM_DEMOS NUM_EVAL_DEMOS NUM_EVAL_EPISODES NUM_EVAL_ENVS \
  INPAINTING EVAL_PROGRESS_BAR CAPTURE_VIDEO_FREQ DEMO_TYPE ACTION_NORM_PATH

bash examples/baselines/diffusion_policy/run_mam.sh
