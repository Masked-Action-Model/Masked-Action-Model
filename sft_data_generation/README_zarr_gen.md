# Zarr数据生成

用于生成机械臂的训练集。

## 📁 文件结构

```
sft_data_generation/
├── README_zarr_gen.md                   # 本文档
├── mask_for_sft.py                     # 数据掩码处理工具
├── gen_dp_training_arr.py              # Diffusion Policy训练数据生成 (CPU)
├── gen_dp_training_arr_gpu.py          # Diffusion Policy训练数据生成 (硬件加速)
├── gen_mam_training_arr.py             # Masked Action Model训练数据生成 (CPU)
├── gen_mam_training_arr_gpu.py         # Masked Action Model训练数据生成 (硬件加速)
├── run_gen_dp_training.sh              # DP训练数据生成启动脚本
├── run_gen_dp_training_gpu.sh          # DP训练数据生成启动脚本 (硬件加速)
├── run_gen_mam_training.sh             # MAM训练数据生成启动脚本
└── run_gen_mam_training_gpu.sh         # MAM训练数据生成启动脚本 (硬件加速)
```

## 🚀 快速开始

### 1. 环境准备

```bash
# 安装依赖
pip install h5py numpy opencv-python zarr termcolor

# 安装FFmpeg
sudo apt update
sudo apt install ffmpeg

# 检查硬件加速支持
ffmpeg -hwaccels
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

# 硬件加速版本
./run_gen_dp_training_gpu.sh
```

#### Masked Action Model训练数据生成

```bash
# CPU版本
./run_gen_mam_training.sh

# 硬件加速版本
./run_gen_mam_training_gpu.sh
```

## ⚡ 硬件加速支持

### **支持的硬件加速类型**

根据您的系统FFmpeg版本，支持以下硬件加速：

1. **cuvid** (NVIDIA GPU) - 推荐
   - 支持NVIDIA显卡
   - 使用NPP库进行GPU缩放
   - 性能提升：3-8倍

2. **vaapi** (Intel GPU)
   - 支持Intel集成显卡
   - 使用VAAPI进行硬件加速
   - 性能提升：2-5倍

3. **vdpau** (NVIDIA GPU)
   - 较老的NVIDIA显卡支持
   - 性能提升：2-4倍

### **自动检测和回退**

脚本会自动：
1. 检测系统支持的硬件加速类型
2. 选择最佳的可用加速方案
3. 如果硬件加速失败，自动回退到CPU模式

### **性能提升预期**

- **硬件加速**: 2-8倍性能提升（取决于硬件）
- **批量处理**: 2-3倍性能提升
- **总体提升**: 4-24倍性能提升

## 🔧 故障排除

### **常见问题**

1. **FFmpeg未找到**
   ```bash
   sudo apt install ffmpeg
   ```

2. **硬件加速不可用**
   ```bash
   # 检查支持的硬件加速
   ffmpeg -hwaccels
   
   # 检查NVIDIA驱动
   nvidia-smi
   ```

3. **cuvid加速失败**
   - 确保安装了NVIDIA驱动
   - 检查FFmpeg是否支持cuvid
   - 尝试使用vaapi或vdpau

4. **内存不足**
   - 减少批处理大小
   - 使用CPU模式
   - 分批处理数据

### **硬件加速兼容性**

| 系统 | FFmpeg版本 | 支持的加速 | 推荐方案 |
|------|------------|------------|----------|
| Ubuntu 18.04 | 3.4.11 | cuvid, vaapi, vdpau | cuvid |
| Ubuntu 20.04+ | 4.x+ | cuda, cuvid, vaapi | cuda |
| CentOS/RHEL | 4.x+ | cuda, cuvid, vaapi | cuda |

### **解决方案**

如果遇到硬件加速问题：

1. **使用现有加速**: 脚本会自动选择可用的最佳方案
2. **回退到CPU**: 硬件加速失败时自动使用CPU模式
3. **升级FFmpeg**: 考虑升级到支持CUDA的版本

## 📊 性能优化建议

1. **优先使用硬件加速**: 选择GPU版本脚本
2. **调整批处理大小**: 根据内存情况调整batch_size
3. **并行处理**: 对于大量数据，考虑并行处理
4. **存储优化**: 使用SSD存储提升I/O性能

## 🤝 贡献

欢迎提交Issue和Pull Request来改进这个工具集！

## �� 许可证

本项目采用MIT许可证。