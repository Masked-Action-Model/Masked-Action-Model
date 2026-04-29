# Benchmark 

**我们在这个文档里就一些问题进行交互，我会每次在这个文档里写上进一步指示，然后你按我的要求执行**
**这个文档里我写的内容你不要直接删除，只按照我的要求增加或者改动**

## 背景描述

之前我用dit已经跑通了baseline（去除depth），详见Optimize.md 3、Model部分。

接下来我需要基于此，改一个Unet版的baseline。
（注意直接从Dit那版里照搬所有设计，除了去噪模型不用transformer而是换乘原本的Unet设计）

但是可以看到diffusion policy old文件夹里本来就有很多unet的旧代码，请你对比分析，把之前代码和dit不同也就是有问题的地方找出来。如果有符合要求“（注意直接从Dit那版里照搬所有设计，除了去噪模型不用transformer而是换乘原本的Unet设计）”的版本就直接使用，如果没有就自己创建副本改一个版本

## 执行记录

### 结论

现有旧 UNet 文件不完全符合要求，不能直接使用：

- `train_rgbd_unet.py` 旧版缺少当前 DiT baseline 中的 action min/max 归一化训练与 rollout 反归一化逻辑。
- `run_train_rgbd_unet.sh` 旧版默认数据、默认参数和调用路径仍是旧口径，并且调用到 `examples/baselines/diffusion_policy/...`，不是当前 `diffusion_policy_old/...`。
- `train_unet.py` 虽然已有 action denorm 相关逻辑，但不是从当前已跑通的 `train_rgbd.py` 直接派生，和“只换去噪模型”这个控制变量要求不够干净。

因此已用当前 `train_rgbd.py` 作为基准重建 UNet 版，只替换去噪网络。

### 已修改

- `examples/baselines/diffusion_policy_old/train_rgbd_unet.py`
  - 从当前 `train_rgbd.py` 派生。
  - `DiTNoiseNet` 替换为 `ConditionalUnet1D`。
  - 保留 DiT baseline 的：
    - action min/max 归一化训练
    - eval rollout 反归一化
    - `pd_ee_pose`
    - `rgb`/`rgb+depth` 可切换
    - 同一套 dataset padding / eval / checkpoint 逻辑
  - 新增可调：
    - `--unet-dims`
    - `--n-groups`

- `examples/baselines/diffusion_policy_old/run_train_rgbd_unet.sh`
  - 从当前 `run_train_rgbd.sh` 派生。
  - 默认数据与 DiT rgb baseline 对齐：
    - `experiment_4.rgb.pd_ee_pose.physx_cpu.h5`
    - `OBS_MODE=rgb`
    - `CONTROL_MODE=pd_ee_pose`
  - 最后一行调用 `examples/baselines/diffusion_policy_old/train_rgbd_unet.py`。

### 启动命令

```bash
PATH=/home/hebu/miniconda3/envs/maniskill_py311/bin:$PATH \
MPLCONFIGDIR=/tmp/matplotlib-maniskill \
EXP_NAME=benchmark-pose-unet-rgb \
SEED=1 \
TORCH_DETERMINISTIC=true \
CUDA=true \
TRACK=false \
WANDB_PROJECT_NAME=ManiSkill \
WANDB_ENTITY= \
CAPTURE_VIDEO=false \
ENV_ID=PickCube-v1 \
DEMO_PATH=demos/exp_4/PickCube-v1/motionplanning/experiment_4.rgb.pd_ee_pose.physx_cpu.h5 \
NUM_DEMOS=100 \
TOTAL_ITERS=100000 \
BATCH_SIZE=64 \
LR=1e-4 \
OBS_HORIZON=2 \
ACT_HORIZON=8 \
PRED_HORIZON=16 \
DIFFUSION_STEP_EMBED_DIM=64 \
UNET_DIMS="64 128 256" \
N_GROUPS=8 \
OBS_MODE=rgb \
MAX_EPISODE_STEPS=100 \
LOG_FREQ=1000 \
EVAL_FREQ=5000 \
SAVE_FREQ=50000 \
NUM_EVAL_EPISODES=100 \
NUM_EVAL_ENVS=10 \
ACTION_NORM_PATH= \
SIM_BACKEND=physx_cpu \
NUM_DATALOAD_WORKERS=0 \
CONTROL_MODE=pd_ee_pose \
DEMO_TYPE=unet_rgb \
bash examples/baselines/diffusion_policy_old/run_train_rgbd_unet.sh
```

### 验证

```bash
python -m py_compile examples/baselines/diffusion_policy_old/train_rgbd_unet.py
bash -n examples/baselines/diffusion_policy_old/run_train_rgbd_unet.sh
```

均通过。

## Debug

这个baseline肯定改的有问题，我按照你的这个bash跑了，训了10000success还是0,一模一样的数据用dit训练的时候已经有20% success里。
请你仔细对比两者训练数据等，找出问题。

结论简要写在这里（找到确凿问题为止，不要给我似是而非的结论）：

### Debug 记录

已核对实际 run 的 TensorBoard hyperparameters：
因此这次 0 success **不是跑错数据、不是 control mode 错、不是 num demos 没覆盖、也不是 action norm 路径不一致**。

代码 diff 也确认 `train_rgbd_unet.py` 相比 `train_rgbd.py` 只有预期差异：
- `DiTNoiseNet` -> `ConditionalUnet1D`
- 新增 `unet_dims / n_groups`
- UNet 需要把 obs condition flatten 成 `(B, obs_horizon * obs_dim)`

当前更细结论：

- **已基本排除**
  - 数据路径错误：UNet/DiT 都是 `experiment_4.rgb.pd_ee_pose.physx_cpu.h5`。
  - demo 数不一致：实际 run 中 `num_demos=100`。
  - control mode 不一致：都是 `pd_ee_pose`。
  - action norm 路径不一致：实际 run 都从同一数据集 stats 产生/使用。
  - gripper 主因：UNet 2500 iter 离线生成 gripper 已接近专家，MAE 约 `0.027~0.030`。

### 继续 Debug 记录

#### 已回退

- 已按要求回退欧拉角 unwrap/wrap 相关改动；当前不再把欧拉角作为主线。
- `train_rgbd_unet.py` 里未保留 `unwrap_euler_actions / wrap_to_pi / EULER_ACTION_DIMS`。
- action norm 文件路径恢复为原始 `.action_norm.json`。

#### 新发现：UNet 闭环动作分布异常贴边

新加诊断脚本：

```bash
python examples/baselines/diffusion_policy_old/debug_action_saturation.py \
  --model unet \
  --checkpoint runs/debug-unet-rgb-5k/checkpoints/2500.pt \
  --device cuda
```

该脚本只做诊断：真实 eval 环境 rollout，同时统计反归一化前的 normalized action 是否靠近 `[-1, 1]` 边界。

结果摘要：

- 专家数据 normalized action 贴边比例 `abs(a)>0.98`：
  - x/y/z: `[0.0076, 0.0049, 0.0449]`
  - roll/gripper: `[1.0, 1.0]`
  - 说明 roll/gripper 天然贴边，x/y/z 不应大面积贴边。
- DiT `50000.pt` rollout 贴边比例 `abs(a)>0.98`：
  - x/y/z: `[0.0, 0.0, 0.0289]`
  - 符合专家分布。
- UNet `2500.pt` rollout 贴边比例 `abs(a)>0.98`：
  - x/y/z: `[0.1635, 0.3654, 0.4327]`
  - 明显远高于专家和 DiT。
- UNet `1000.pt` 也有同类问题：
  - x/y/z: `[0.3654, 0.3462, 0.2885]`
- 关闭 DDPM `clip_sample` 后：
  - UNet raw action 直接发散到几十/上百量级，`abs(a)>0.98` 约等于全维度 `99%~100%`。
  - DiT no-clip 仍保持在合理范围，x/y/z 贴边仍为 `[0.0, 0.0, 0.0289]`。
  - 因此不是 `clip_sample=True` 把好动作夹坏，而是 UNet denoising 在闭环状态下本身已经发散；clip 只是把发散压成边界动作。

闭环 trace 也吻合这个现象：

- UNet 100 steps 没有 grasp，TCP 从 `[0.012, 0.038, 0.182]` 漂到 `[0.113, -0.103, 0.534]`，明显偏离目标 `[0.027, -0.002, 0.289]`。
- DiT 同 seed 能把 TCP 保持在目标附近，后段成功 grasp/放置。

#### 当前判断

- 目前最像 bug 的点不是 gripper，也不是欧拉角，而是 **UNet denoising 在闭环 rollout 中发散，对 x/y/z 生成大量归一化边界动作**，导致执行后迅速进入训练分布外状态，然后继续漂移。
- 这解释了为什么离线 loss/MAE 看起来不离谱，但 eval success 仍为 0：离线单步误差没有覆盖闭环分布崩塌。
- `train_rgbd_unet.py` 与 DiT 版的代码 diff 很干净，目前没看到数据管线、horizon、action norm 或 control mode 的隐藏差异。

#### 还需确认

- 需要一个保存了 `5000/10000/final` 的 UNet checkpoint 来确认这不是 `2500` 欠训练；原始 `benchmark-pose-unet-rgb` 只保存了 `0.pt`，所以不能直接 trace 10000。
- 我尝试补训 `debug-unet-rgb-10k-final`，但当前 GPU 被 `benchmark-pose-dit-rgbd-depth-norm` 及其 eval 子进程占满，`BATCH_SIZE=64` 和 `BATCH_SIZE=16` 都 OOM。等该训练结束或释放显存后，应继续跑 10k checkpoint 的同一饱和诊断。

### 新数据结论

已查看新增训练记录：

- `benchmark-pose-unet-rgb` 实际跑到 `18000` step；`5000/10000/15000` 的 `eval/success_once` 和 `eval/success_at_end` 全部为 `0`。同时 loss 已从 `1.177` 降到 `10000: 0.0245`、`18000: 0.0074`。
- `debug-unet-rgb-2k` 保存了 `1000.pt`；`500` loss 为 `0.247`，`1000` loss 为 `0.0977`。
- `debug-unet-rgb-10k-final` / `debug-unet-rgb-10k-bs16` 只有 hparams event，没有标量和 checkpoint，属于启动后未真正训练完成。

对 `1000.pt` 继续跑动作分布诊断：

- 专家数据 normalized action 贴边比例 `abs(a)>0.98` 的 x/y/z 为 `[0.0076, 0.0049, 0.0449]`。
- UNet `1000.pt` 闭环 rollout 的 x/y/z 贴边比例为 `[0.3558, 0.5865, 0.2308]`。
- UNet `1000.pt` 关闭 `clip_sample` 后 raw action 直接爆到几十/几百量级，几乎所有维度 `abs(a)>0.98`。
- UNet `2500.pt` 仍然异常：x/y/z 贴边比例 `[0.1923, 0.6827, 0.1635]`。
- DiT `50000.pt` 对照正常：x/y/z 贴边比例 `[0.0, 0.0, 0.0289]`。

因此现在可以确认：问题不是数据路径、control mode、action norm、eval 配置，也不是简单欠训练。`ConditionalUnet1D` 在当前 `pd_ee_pose + rgb` 链路中反向扩散采样不稳定，连续位姿维度生成大量边界动作；`clip_sample=True` 只是把发散动作夹到 `[-1, 1]`，不是根因。eval success 为 0 的直接原因就是 UNet 采样动作分布已经坏掉，后续应针对 UNet denoiser/conditioning 本身修，而不是继续查数据管线。
