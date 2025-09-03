#!/bin/bash

# Diffusion Policy训练数据生成脚本
# 用于从.h5文件和对应.mp4视频生成.zarr格式的训练数据

# 设置默认参数
INPUT_H5="../demo_0828/PlugCharger-v1/motionplanning/action_normed.h5"
OUTPUT_ZARR="./output_zarr/test_dp.zarr"
INPUT_CONDITION="./output/0830_test_PlugCharger_local_planner_padding_1.h5"
USE_GPU=False

# 创建输出目录
OUTPUT_DIR=$(dirname "$OUTPUT_ZARR")
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "Diffusion Policy训练数据生成"
echo "=========================================="
echo "输入文件: $INPUT_H5"
echo "输出文件: $OUTPUT_ZARR"
echo "Condition文件: $INPUT_CONDITION"
echo "GPU加速: $USE_GPU"
echo "=========================================="

# 运行Python脚本
python gen_dp_training_arr.py \
    --input "$INPUT_H5" \
    --output "$OUTPUT_ZARR" \
    --input_condition "$INPUT_CONDITION"
