#!/bin/bash

INPUT="/data1/tangjielong/VLA_project/se3_raw/StackPyramid-v1/motionplanning/20250716_205039_mask.h5"
OUTPUT="./test_random_mask.h5"
MASK_TYPE="random_mask"
RETAIN_RATIO=0.1
SIZE=5

python mask_for_sft.py \
  --input "$INPUT" \
  --output "$OUTPUT" \
  --mask_type "$MASK_TYPE" \
  --retain_ratio "$RETAIN_RATIO" \
  --size "$SIZE" \
  --normalize
