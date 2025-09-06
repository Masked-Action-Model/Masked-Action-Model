#!/bin/bash

# 示例：转换h5文件为zarr格式
INPUT_H5="/data1/tangjielong_2/mask_action_model/Masked-Action-Model/demos/StackCube-v1/motionplanning/20250716_190451_mask.h5"
OUTPUT_ZARR="./output_data.zarr"

python /data1/tangjielong_2/mask_action_model/Masked-Action-Model/sft_data_generation/convert_zarr/h5_to_zarr.py \
  --input "$INPUT_H5" \
  --output "$OUTPUT_ZARR" \
  --compression gzip

echo "转换完成！"