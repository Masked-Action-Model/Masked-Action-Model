# Zarr数据生成

用于生成机械臂的训练集。

## 📁 文件结构

```
sft_data_generation/
├── README_zarr_gen.md                   # 本文档
├── mask_for_sft.py                     # 数据掩码处理工具
├── gen_dp_training_arr.py              # Diffusion Policy训练数据生成 (CPU)
├── gen_dp_training_arr_gpu.py          # Diffusion Policy训练数据生成 (GPU)
├── gen_mam_training_arr.py             # Masked Action Model训练数据生成 (CPU)
├── gen_mam_training_arr_gpu.py         # Masked Action Model训练数据生成 (GPU)
├── run_gen_dp_training.sh              # DP训练数据生成启动脚本
├── run_gen_dp_training_gpu.sh          # DP训练数据生成启动脚本 (GPU)
├── run_gen_mam_training.sh             # MAM训练数据生成启动脚本
└── run_gen_mam_training_gpu.sh         # MAM训练数据生成启动脚本 (GPU)
```

## 🚀 快速开始

### 1. 环境准备

```bash
# 安装依赖
pip install h5py numpy opencv-python zarr termcolor

# 安装FFmpeg (GPU加速需要)
sudo apt update
sudo apt install ffmpeg

# 检查GPU支持
nvidia-smi
```

### 2. 数据准备

确保您有以下文件结构：
```
your_data_directory/
├── action_normed.h5                    # 包含trajectory数据的H5文件
├── 0/                                  # 视角0的视频文件
│   ├── 0.mp4
│   ├── 1.mp4
│   └── ...
└── 1/                                  # 视角1的视频文件
    ├── 0.mp4
    ├── 1.mp4
    └── ...
```

### 3. 运行示例

#### Diffusion Policy训练数据生成

```bash
# CPU版本
./run_gen_dp_training.sh

# GPU加速版本
./run_gen_dp_training_gpu.sh
```

#### Masked Action Model训练数据生成

```bash
# CPU版本
./run_gen_mam_training.sh

# GPU加速版本
./run_gen_mam_training_gpu.sh
```