#!/bin/bash

#INPUT="/data1/tangjielong/VLA_project/se3_raw/StackPyramid-v1/motionplanning/20250716_205039_mask.h5"
INPUT="../Data_maniskill3/arlen_data/demos/PullCube-v1/motionplanning/20250716_195936_mask.h5"
OUTPUT="./test_2D_image_trajectory1_mask.h5"
MASK_TYPE="2D_image_trajectory"
RETAIN_RATIO=0.9

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
