#!/bin/bash
set -euo pipefail

# Launcher for examples/baselines/diffusion_policy/concat_mas_h5.py
# Override any value by exporting the variable before running this script.

DEFAULT_BASE="/home/hebu/code/ManiSkill/demos/data_1"
ORIG="${ORIG:-${DEFAULT_BASE}/data_1.h5}"
MASKED="${MASKED:-${DEFAULT_BASE}/data_1_masked.h5}"
OUT="${OUT:-${DEFAULT_BASE}/data_1_concat.h5}"
MASK_NAME="${MASK_NAME:-mask}"
MAS_NAME="${MAS_NAME:-mas}"
ACTIONS_NAME="${ACTIONS_NAME:-actions}"
TRAIN_RATIO="${TRAIN_RATIO:-5}"
EVAL_RATIO="${EVAL_RATIO:-1}"

if [[ -z "$MASKED" ]]; then
  echo "缺少 MASKED（带掩码的数据集路径）。" >&2
  echo "示例：" >&2
  echo "  ORIG=${ORIG} \\" >&2
  echo "  MASKED=${DEFAULT_BASE}/data_1_masked_*.h5 \\" >&2
  echo "  OUT=${OUT} \\" >&2
  echo "  bash $0" >&2
  exit 1
fi

python examples/baselines/diffusion_policy/concat_mas_h5.py \
  --orig "$ORIG" \
  --masked "$MASKED" \
  --out "$OUT" \
  --mask-name "$MASK_NAME" \
  --mas-name "$MAS_NAME" \
  --actions-name "$ACTIONS_NAME" \
  --train-ratio "$TRAIN_RATIO" \
  --eval-ratio "$EVAL_RATIO"
