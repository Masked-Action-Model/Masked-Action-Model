# Diffusion Policy - MAM 快速上手

本仓库在 Diffusion Policy 框架下，提供了基于图像与低维特征的 MAM 与 IP_MAM 策略训练流程与配置。

- 环境配置：如何创建与激活 Conda 环境
- 数据准备：如何生成示例 zarr 数据并与配置对齐
- 训练启动：如何使用 Hydra 配置训练 MAM
- 文件总览：MAM 相关核心文件列表

---
# 如把condition去掉，训练dp：
仅需在 MAM.yaml 里把 condition 从 shape_meta 中删除/注释，模型就不会再把 condition 当作输入使用。

<!-- condition: 
      shape: [800]
      type: low_dim -->

你的数据集样本里若仍然包含 condition 键，也不会报错（编码器只会根据 shape_meta 中的键构建与取用特征，dataset/normalizer 允许存在“多余”的键），只是该特征不再参与模型输入。

## 1. 环境配置

1) 创建并激活 Conda 环境（优先使用项目自带环境文件）

```bash
conda env create -f diffusion_policy/conda_environment.yaml -n robodiff
```

```bash
conda activate robodiff
```




---

## 2. 数据准备

本项目使用 zarr 数据作为训练输入，数据键位与配置需对齐。

- IP_MAM 数据默认路径：`data/custom_IP_MAM_replay.zarr`
- MAM 数据默认路径：`data/custom_replay.zarr`



A) 生成 MAM 示例数据
```bash
python generate_simple_zarr.py
```
- 确保输出路径与 `diffusion_policy/config/task/MAM.yaml` 中的 `dataset.zarr_path: data/custom_replay.zarr` 保持一致。

如需修改输出路径，请同步更新对应 task 配置文件的 `dataset.zarr_path` 字段。

---

## 3. 开始训练

进入项目子目录（包含 train.py 的目录）
```bash
cd diffusion_policy
```

 训练 MAM （dp）策略（若你使用 MAM 任务与数据）
```bash
python train.py --config-dir=. --config-name=train_diffusion_unet_MAM_policy.yaml training.seed=42 training.device=cuda:2 hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'logging.mode=disabled dataloader.batch_size=32
```

提示：
- 若你修改了图像大小、低维键或动作维度，请同时确保相应 task 的 `shape_meta`、数据集与策略配置一致。
- 训练时会根据配置启用 EMA、日志与检查点管理功能。

---

## 4. MAM 相关文件总览

策略（Policy）
- diffusion_policy/policy/MAMdiffusion_unet_image_policy.py  
  基于 ConditionalUnet1D 的 MAM 图像策略，实现图像（多视角）与低维特征融合、全局条件控制与扩散采样。
- diffusion_policy/policy/IP_MAMdiffusion_unet_image_policy.py  
  IP_MAM 变体的图像策略，数据/键位与 `IP_MAM.yaml` 对齐。

工作区（Workspace）
- diffusion_policy/workspace/MAM_train_diffusion_unet_image_workspace.py  
  MAM 训练工作流：模型/优化器初始化、数据加载、训练循环、EMA、评估、日志与检查点管理。


数据集（Dataset）
- diffusion_policy/dataset/MAM_pusht_image_dataset.py  
  MAM 的 PushT 图像数据集定义（ReplayBuffer + SequenceSampler）。


任务配置（Task Config）
- diffusion_policy/config/task/MAM.yaml  
  定义 MAM 的 `shape_meta`（img1、img2、agent_pos、condition、action 等）与 env_runner、dataset 等。


训练配置（Training Config）
- diffusion_policy/train_diffusion_unet_IP_MAM_policy.yaml  
  使用 IP_MAM 工作区与策略的训练配置，包含 noise_scheduler、obs_encoder 与关键超参。
- image_pusht_MAM_cnn.yaml  
  面向 MAM 的图像训练配置示例（含 dataloader、EMA、logging、多轮运行目录等）。

辅助（编码器）
- diffusion_policy/model/vision/multi_image_obs_encoder.py  
  多图像观测编码器，支持 img1、img2 等 RGB 键与低维键（如 agent_pos、condition/MAS）拼接为特征。
  - 示例：在实现中，低维键会直接作为特征拼接到编码后的图像特征中。

---


## 5. 目录结构（节选）

- diffusion_policy/train.py：Hydra 入口
- diffusion_policy/requirements.txt：pip 依赖
- diffusion_policy/conda_environment.yaml：Conda 环境
- generate_IP_MAM_simple_zarr.py：生成 IP_MAM 示例数据
- generate_simple_zarr.py：生成 MAM 示例数据
- diffusion_policy/config/task/*.yaml：任务级别配置（shape_meta / 数据集路径）
- diffusion_policy/policy/*.py：策略实现
- diffusion_policy/workspace/*.py：训练工作流
- diffusion_policy/dataset/*.py：数据集定义

如需进一步定制（例如修改图像尺寸或新增观测键位），请同步更新：数据生成脚本 → zarr 数据键位 → task 的 `shape_meta` → obs_encoder → policy 及 workspace 超参，保持一致性。