#!/bin/bash

INPUT="/data1/tangjielong_2/mask_action_model/Masked-Action-Model/demo_0828/PlugCharger-v1/motionplanning/action.h5"
OUTPUT="./output/0830_test_PlugCharger_local_planner_padding.h5"
MASK_TYPE="local_planner"
RETAIN_RATIO=0.3
SIZE=5
MASK_SEQ_LEN=20 #local_planner使用掩码的连续序列长度

python mask_for_sft.py \
  --input "$INPUT" \
  --output "$OUTPUT" \
  --mask_type "$MASK_TYPE" \
  --retain_ratio "$RETAIN_RATIO" \
  --size "$SIZE" \
  --mask_seq_len "$MASK_SEQ_LEN" \
  --normalize \
  --enable_padding #是否padding
