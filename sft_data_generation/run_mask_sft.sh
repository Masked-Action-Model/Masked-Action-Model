#!/bin/bash
# 生成mask 
INPUT="/cephfs/shared/Yanbang/MAM/Masked-Action-Model/sft_data_generation/PickCube-v1/motionplanning/action.h5"
MASK_TYPE="points"
RETAIN_RATIO=0.1
SIZE=1
MASK_SEQ_LEN=30 #local_planner使用掩码的连续序列长度 或者2D_partial_trajectory保留的连续序列长度

# 生成带时间戳的输出文件名
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT="/cephfs/shared/Yanbang/MAM/Masked-Action-Model/sft_data_generation/output_mask_PickCube_${MASK_TYPE}_${TIMESTAMP}.h5"

python mask_for_sft.py \
  --input "$INPUT" \
  --output "$OUTPUT" \
  --mask_type "$MASK_TYPE" \
  --retain_ratio "$RETAIN_RATIO" \
  --size "$SIZE" \
  --mask_seq_len "$MASK_SEQ_LEN" \
  --normalize \
  --enable_padding #是否padding
