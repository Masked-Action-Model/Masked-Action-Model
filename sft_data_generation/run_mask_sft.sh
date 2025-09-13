#!/bin/bash

INPUT="/data1/tangjielong_2/mask_action_model/Masked-Action-Model/demo_0828/PushCube-v1/motionplanning/action.h5"
OUTPUT="./output/0914_PushCube_points_mask_w_padding.h5"
MASK_TYPE="points"
RETAIN_RATIO=0.1
SIZE=1
MASK_SEQ_LEN=30 #local_planner使用掩码的连续序列长度 或者2D_partial_trajectory保留的连续序列长度

python mask_for_sft.py \
  --input "$INPUT" \
  --output "$OUTPUT" \
  --mask_type "$MASK_TYPE" \
  --retain_ratio "$RETAIN_RATIO" \
  --size "$SIZE" \
  --mask_seq_len "$MASK_SEQ_LEN" \
  --normalize \
  --enable_padding #是否padding
