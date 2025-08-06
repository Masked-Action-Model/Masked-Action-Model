#!/bin/bash

# 示例：转换h5文件为zarr格式
INPUT_H5="./test_local_planner_mask_0.h5"
OUTPUT_ZARR="./output_data.zarr"

python h5_to_zarr.py \
  --input "$INPUT_H5" \
  --output "$OUTPUT_ZARR" \
  --compression gzip

echo "转换完成！"