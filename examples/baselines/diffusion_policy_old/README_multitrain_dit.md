# 多卡训练说明（`multitrain_dit.py`）

本文档说明如何使用新建的 DDP 训练脚本进行多卡训练（不修改原始 `train_rgbd.py` / `run_train_rgbd.sh`）。

## 1. 相关文件

- 训练代码：`examples/baselines/diffusion_policy/multitrain_dit.py`
- 启动脚本：`examples/baselines/diffusion_policy/run_multitrain_dit.sh`

## 2. 启动前准备

在 `ManiSkill/ManiSkill` 目录下确认：

1. Demo 数据存在（`DEMO_PATH`）
2. 动作反归一化参数 JSON 存在（`ACTION_NORM_PATH`，必须包含 `min` / `max`）
3. 环境与依赖可正常运行（PyTorch、ManiSkill、diffusers 等）

## 3. 一键启动（示例）

```bash
cd ManiSkill/ManiSkill

export CUDA_VISIBLE_DEVICES=0,1,2,3
export NPROC_PER_NODE=4
export ENV_ID=PickCube-v1
export DEMO_PATH=demos/data_1/data_1_concat_train.h5
export ACTION_NORM_PATH=demos/data_1/data_1_norm.json
export MAX_EPISODE_STEPS=160
export BATCH_SIZE=128
export NUM_DATALOAD_WORKERS=4

bash examples/baselines/diffusion_policy/run_multitrain_dit.sh
```

## 4. 关键参数说明

### `BATCH_SIZE`

在当前脚本中表示**目标全局 batch**（requested global batch）。

脚本会自动计算：

- `per_gpu_batch = BATCH_SIZE // world_size`
- `global_batch(effective) = per_gpu_batch * world_size`

若 `BATCH_SIZE` 不能被 `world_size` 整除，会有 warning，且 `effective` 会小于 `requested`。

### `NPROC_PER_NODE`

单机训练时通常等于你要用的 GPU 数量，即 `world_size`。

### `NUM_DATALOAD_WORKERS`

每个 GPU 进程各自 DataLoader 的 worker 数量。

- 总 worker 数约为：`NUM_DATALOAD_WORKERS * world_size`
- Linux 常用起步值：`4`
- 若数据加载瓶颈明显可试 `8`
- 若 CPU 打满或系统抖动，降到 `2` 或 `0`

## 5. 训练日志会打印什么

脚本在 rank0（主进程）启动时会打印：

- `global_batch(requested)`
- `global_batch(effective)`
- `per_gpu_batch`
- `world_size`
- `num_workers(per_gpu)`
- `num_workers(total)`

用于快速判断 batch 与数据加载配置是否合理。

## 6. 训练过程要点

1. 使用 `torchrun` 启动多进程 DDP
2. 每个 rank 构建本地 DataLoader，并通过 `DistributedSampler` 切分数据
3. 模型通过 `DDP(model)` 包装
4. 主进程负责日志、评估、保存 best checkpoint

## 7. 输出目录

默认输出在：`runs/<run_name>/`

- `checkpoints/`：模型保存（含 best）
- `videos/`：评估视频（开启 `capture_video` 时）
- TensorBoard 日志

查看曲线：

```bash
tensorboard --logdir runs
```

## 8. 常见问题

### Q1: 为什么 `global_batch(requested)` 和 `effective` 不一致？

因为 `BATCH_SIZE` 未被 `world_size` 整除，脚本按整除计算每卡 batch，导致 `effective` 变小。

### Q2: loss 正常但速度慢？

优先检查：

1. `NUM_DATALOAD_WORKERS` 是否过低（数据加载成为瓶颈）
2. `per_gpu_batch` 是否过小（GPU 吃不满）
3. CPU 核数是否不足以支撑总 worker 数

### Q3: 必须传 `ACTION_NORM_PATH` 吗？

是。该文件用于将模型输出动作从归一化区间反归一化后再执行。缺失会直接报错。
