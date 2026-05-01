# Optimaize

**我们在这个文档里就一些问题进行交互，我会每次在这个文档里写上进一步指示，然后你按我的要求执行**
**这个文档里我写的内容你不要直接删除，只按照我的要求增加或者改动**

到此为止，整个模型已经完整了（主体内容全部在dffusion_policy文件夹中），接下来我们需要对模型performance做优化调整。

先做一个小改动，把所有sh文件里的MASK_TYPE_RATIO_LIST改成MASK_COMPOSITION_LIST，把所有MASK_PARAM_LIST改成MASK_RATIO_LIST，并且对代码库相关地方做替换。

## 1、Dataset 采集

### 夹爪第七维

采集Panda MP链路：
- 二值化是在 [interactive_panda.py]触发`planner.open_gripper()/close_gripper()`，而默认值`OPEN = 1`、`CLOSED = -1`。
- Panda 底层控制器本身支持连续值 `PDJointPosMimicControllerConfig(lower=-0.01, upper=0.04)`，把归一化动作从 `[-1,1]` 映射到实际夹爪开合位置。

- 这个信号是 **目标位置**。控制器在 `set_action()` 里把目标 qpos 写给控制侧 finger joint，再按 mimic 关系同步到另一侧 finger joint。
- 真正执行时，SAPIEN 关节驱动按 PD 目标去追这个 target，并受 `stiffness / damping / force_limit` 约束；所以如果夹爪在闭合过程中碰到物体，真实 qpos 不会继续达到目标值，而是停在“接触约束 + PD 驱动力 + force_limit”平衡出来的位置。也就是说**控制信号和真实状态不必相等**。

### 新数据采集

#### 可调参数列表

走MP自动采集入口 `mani_skill/examples/motionplanning/panda/run.py`，当前可直接调的参数有：

- `env_id`： `PickCube-v1`、
- `obs_mode`：`none`
- `num_traj`：要采多少条轨迹
- `only_count_success`： 是
- `reward_mode`：`none`
- `sim_backend`：`cpu`
- `render_mode`：`rgb_array`成
- `vis`： 是
- `save_video`： 否
- `traj_name`：输出 h5/json 文件名
- `shader`：默认
- `record_dir`：demos/...
- `num_procs`：`1`

#### 采集任务1

PickCube 2400

```bash
python mani_skill/examples/motionplanning/panda/run.py \
  -e PickCube-v1 \
  -o none \
  -n 2400 \
  --only-count-success \
  -b cpu \
  --render-mode rgb_array \
  --vis \
  --traj-name experiment_3 \
  --record-dir demos/exp_3
```

replay成pd_ee_pose和pd_ee_delta_pose两个controlmode，用rgbd的obsmode给我命令行：

```bash
python mani_skill/trajectory/replay_trajectory.py \
  --traj-path demos/exp_3/PickCube-v1/motionplanning/experiment_3.h5 \
  -b cpu \
  -o rgbd \
  -c pd_ee_pose \
  --save-traj
```

```bash
python mani_skill/trajectory/replay_trajectory.py \
  --traj-path demos/exp_3/PickCube-v1/motionplanning/experiment_3.h5 \
  -b cpu \
  -o rgbd \
  -c pd_ee_delta_pose \
  --save-traj
```
## 2、Dataset 预处理

这一部分的改动只涉及multi-mask training这条线路，不用管only_mas和mas_window这两个线路，只用考虑train_window_mixed这条线路（一个trainpy文件和一个sh文件）

### 增加mask_type 'none'

组合multimask组合的时候加入type 'none'

`none` 语义：`mas[:, :7]` 全部填 `mask_value`，`mask[:, :7]` 全部为 `False`，只保留 progress 列用于时间对齐。

```bash
TRAIN_NUM_MASK_TYPE=3 \
TRAIN_MASK_TYPE_LIST='["none","random_mask","points"]' \
TRAIN_MASK_COMPOSITION_LIST='[0.2,0.4,0.4]' \
TRAIN_MASK_RATIO_LIST='[null,0.2,0.2]' \
```
### 1demo-multi-mask 设计

#### 修改计划

目标：在 `train_window_mixed` 这条链路里增加两种 preprocess 模式：
- `composition`：保持当前逻辑。
- `one_demo_multi_mask`：新增逻辑。每条 source demo 复制成 `num_mask_type` 条轨迹，每条分别应用一种 mask。
(若 `NUM_DEMOS=10` 且 `num_mask_type=5`，训练实际使用 `10*5=50` 条轨迹；eval 同理。)

同时需要支持重复 mask type 搭配不同 param，和 `composition` 模式保持一致。例如：

```bash
TRAIN_NUM_MASK_TYPE=3
TRAIN_MASK_TYPE_LIST='["points","points","random_mask"]'
TRAIN_MASK_RATIO_LIST='[0.1,0.3,0.2]'
```

这里不能把两个 `points` 合并；它们应分别成为 `points#0` 和 `points#1` 两个 mask slot。

#### 执行记录

- `run_train_window_mixed.sh`
  - 新增 `PREPROCESS_MASK_ASSIGN_MODE`，默认 `composition`。
  - 可设为 `one_demo_multi_mask`。

- `data_preprocess_mixed.py`
  - 新增 `--mask-assign-mode composition|one_demo_multi_mask`。
  - `composition` 保持原逻辑。
  - `one_demo_multi_mask` 下，每个 source demo 会和每个 mask slot 做笛卡尔积。
  - 支持重复 mask type + 不同 param，例如 `points#0` / `points#1`。
  - h5 仍使用 `traj_0`, `traj_1`, ... 作为 group key，避免破坏 loader 排序。
  - 每个 traj 新增：
    - `source_episode_id`
    - `source_mask_copy_key`
    - `source_mask_copy_index`
    - `mask_type`
    - `mask_type_slot`
    - `mask_slot_index`
  - h5 meta / json preprocess_info 新增：
    - `mask_assign_mode`
    - `source_num_episodes`
    - `expanded_num_episodes`
    - expanded source id 信息

- `train_mas_window_mixed.py`
  - 读取 h5 meta 中的 `mask_assign_mode`。
  - `composition` 下仍按 expanded traj + composition 采样。
  - `one_demo_multi_mask` 下，`NUM_DEMOS` / `NUM_EVAL_DEMOS` 表示 source demo 数，随后自动包含这些 source demo 的全部 mask copy。

使用示例：

```bash
PREPROCESS_MASK_ASSIGN_MODE=one_demo_multi_mask \
TRAIN_NUM_MASK_TYPE=4 \
TRAIN_MASK_TYPE_LIST='["none","points","points","full"]' \
TRAIN_MASK_RATIO_LIST='[null,0.1,0.3,null]' \
EVAL_NUM_MASK_TYPE=4 \
EVAL_MASK_TYPE_LIST='["none","points","points","full"]' \
EVAL_MASK_RATIO_LIST='[null,0.1,0.3,null]' \
NUM_DEMOS=10 \
NUM_EVAL_DEMOS=10 \
bash examples/baselines/diffusion_policy/run_train_window_mixed.sh
```

此时训练实际使用 `10 * 4 = 40` 条 expanded traj，eval 同理。

已验证：
- `python -m py_compile` 通过。
- `bash -n run_train_window_mixed.sh` 通过。
- 小数据 preprocess：`6` 条 source demo，`4` 个 mask slot，生成 train `5 source / 20 expanded`、eval `1 source / 4 expanded`。
- h5 检查通过：
  - `none#0` 的 `mas[:, :7]` 全 0，`mask[:, :7]` 全 False。
  - `full#0` 的 `mas[:, :7]` 等于 normalized action，`mask[:, :7]` 全 True。
  - 每个 source demo 都包含 `none#0`, `points#0`, `points#1`, `full#0`。
- 训练选择逻辑验证通过：`NUM_DEMOS=2` 时选中 `2 * 4 = 8` 条 expanded traj。

## 3、Model

这部分的改动要针对整个diffusionpolicy文件夹进行，就是onlymas、maswindow、windowmixed三条线路同时改动。

### Benchmark test Debug

在之前的degub过程中，benchmark test存在一定问题（详见Debug4.md）
现在需要重新跑benchmark test，但这次为了保证准确，我将直接从maniskill官网上pull原始dp文件夹，重新采集并且使用pd delta mode的模式的数据集。
请你：
1、给我从maniskill上把dp的文件夹copy下来，放在现在的diffusionpolicy文件夹旁边，然后命名为diffusion_policy_origin
2、（环境用maniskill_py311）用命令行，用panda的mp功能采集600条新的训练数据（命名experiment_4），然后replay成（pdeedeltapose、rgbd）
3、最后给我启动训练脚本的命令行，里面要有所有可改参数，写到这个md文件里。

#### experiment_4 采集与 rgbd replay 命令

- `examples/baselines/diffusion_policy_origin` 已放在当前 `diffusion_policy` 旁边。
- `demos/exp_4/PickCube-v1/motionplanning/experiment_4.h5` 当前已确认是 `600` 条轨迹。

```bash
PATH=/home/hebu/miniconda3/envs/maniskill_py311/bin:$PATH \
python mani_skill/examples/motionplanning/panda/run.py \
  -e PickCube-v1 \
  -o none \
  -n 600 \
  --only-count-success \
  -b cpu \
  --render-mode rgb_array \
  --vis \
  --traj-name experiment_4 \
  --record-dir demos/exp_4
```

```bash
PATH=/home/hebu/miniconda3/envs/maniskill_py311/bin:$PATH \
python mani_skill/trajectory/replay_trajectory.py \
  --traj-path demos/exp_4/PickCube-v1/motionplanning/experiment_4.h5 \
  -b cpu \
  -o rgbd \
  -c pd_ee_delta_pose \
  --save-traj
```

#### origin RGBD benchmark 训练命令

```bash
PATH=/home/hebu/miniconda3/envs/maniskill_py311/bin:$PATH \
MPLCONFIGDIR=/tmp/matplotlib-maniskill \
python examples/baselines/diffusion_policy_origin/train_rgbd.py \
  --exp-name benchmark-delta-unet \
  --seed 1 \
  --torch-deterministic \
  --cuda \
  --no-track \
  --wandb-project-name ManiSkill \
  --wandb-entity None \
  --no-capture-video \
  --env-id PickCube-v1 \
  --demo-path demos/exp_4/PickCube-v1/motionplanning/experiment_4.rgbd.pd_ee_delta_pose.physx_cpu.h5 \
  --num-demos 50 \
  --total-iters 100000 \
  --batch-size 64 \
  --lr 1e-4 \
  --obs-horizon 2 \
  --act-horizon 8 \
  --pred-horizon 16 \
  --diffusion-step-embed-dim 64 \
  --unet-dims 64 128 256 \
  --n-groups 8 \
  --obs-mode rgb+depth \
  --max-episode-steps 100 \
  --log-freq 1000 \
  --eval-freq 5000 \
  --save-freq 50000 \
  --num-eval-episodes 100 \
  --num-eval-envs 10 \
  --sim-backend physx_cpu \
  --num-dataload-workers 0 \
  --control-mode pd_ee_delta_pose \
  --demo-type None
```

### 一个新发现

在对Unet测试的过程中，我发现不加入depth会很大提升performance（详见ManiSkill_origin/analyse.md），这很可能就是debug4.md中问题的最终解释。

为了确认，我打算跑一个对比的dit测试，用diffusionpolicy_old中的脚本，和之前跑失败的dit做对比。

请你控制变量，给我replay采数据的命令行（rgb mode，其他量都和原来一致），然后帮我修改sh文件，以便我开始对比实验。

#### rgb 对照 replay 命令（默认 DiT 口径）

这里按 `run_train_rgbd.sh` / `train_rgbd.py` / `diffusion_policy/evaluate.py` 这套默认链路执行，控制模式用 `pd_ee_pose`，只把观测改成 `rgb`：

```bash
PATH=/home/hebu/miniconda3/envs/maniskill_py311/bin:$PATH \
python mani_skill/trajectory/replay_trajectory.py \
  --traj-path demos/exp_4/PickCube-v1/motionplanning/experiment_4.h5 \
  -b cpu \
  -o rgb \
  -c pd_ee_pose \
  --save-traj
```

#### 默认 DiT 对照训练脚本

已修改：
- `examples/baselines/diffusion_policy_old/run_train_rgbd.sh`
  - 已由 origin 版本替代
  - 最后一行调用 `train_rgbd.py`
  - 默认数据切到 `experiment_4.rgb.pd_ee_pose.physx_cpu.h5`
  - 默认 `obs_mode=rgb`
  - 默认 `control_mode=pd_ee_pose`
- `examples/baselines/diffusion_policy_old/train_rgbd.py`
  - 已由 origin 版本替代
  - 改为调用 `diffusion_policy.evaluate`
  - `ACTION_NORM_PATH` 允许留空，留空时直接把模型输出送进 `env.step()`
- `examples/baselines/diffusion_policy_old/diffusion_policy/evaluate.py`
  - 已由原 origin evaluate 实现替代

启动命令（参数都可改）：

```bash
PATH=/home/hebu/miniconda3/envs/maniskill_py311/bin:$PATH \
MPLCONFIGDIR=/tmp/matplotlib-maniskill \
EXP_NAME=benchmark-pose-dit-origin-rgb \
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
DEMO_TYPE= \
bash examples/baselines/diffusion_policy_old/run_train_rgbd.sh
```

### 进一步实验

单纯把depth去掉效果挺不错的，现在需要再对比一下，把depth归一化后效果如何（之前是直接除了1024,并没有归一化到-1-1），先在改一个副本，把depth归一化到[-1,1]后参与训练。

请你1、修改副本 2、像上面那样给我运行训练的命令行

记录：

#### depth 归一化副本

已新增：
- `examples/baselines/diffusion_policy_old/train_rgbd_depth_norm.py`
- `examples/baselines/diffusion_policy_old/run_train_rgbd_depth_norm.sh`

核心改动：
```python
# 原始 origin:
depth = obs_seq["depth"].float() / 1024.0

# depth-norm 副本:
depth = obs_seq["depth"].float() / 512.0 - 1.0
```

即把 depth 从 `[0, 1024]` 近似映射到 `[-1, 1]`，再与 rgb 拼接输入 DiT。

#### rgbd + pd_ee_pose replay 命令

若还没有 `experiment_4.rgbd.pd_ee_pose.physx_cpu.h5`，先执行：

```bash
PATH=/home/hebu/miniconda3/envs/maniskill_py311/bin:$PATH \
python mani_skill/trajectory/replay_trajectory.py \
  --traj-path demos/exp_4/PickCube-v1/motionplanning/experiment_4.h5 \
  -b cpu \
  -o rgbd \
  -c pd_ee_pose \
  --save-traj
```

#### depth-norm DiT 对照训练命令

```bash
PATH=/home/hebu/miniconda3/envs/maniskill_py311/bin:$PATH \
MPLCONFIGDIR=/tmp/matplotlib-maniskill \
EXP_NAME=benchmark-pose-dit-rgbd-depth-norm \
SEED=1 \
TORCH_DETERMINISTIC=true \
CUDA=true \
TRACK=false \
WANDB_PROJECT_NAME=ManiSkill \
WANDB_ENTITY= \
CAPTURE_VIDEO=false \
ENV_ID=PickCube-v1 \
DEMO_PATH=demos/exp_4/PickCube-v1/motionplanning/experiment_4.rgbd.pd_ee_pose.physx_cpu.h5 \
NUM_DEMOS=100 \
TOTAL_ITERS=100000 \
BATCH_SIZE=64 \
LR=1e-4 \
OBS_HORIZON=2 \
ACT_HORIZON=8 \
PRED_HORIZON=16 \
DIFFUSION_STEP_EMBED_DIM=64 \
OBS_MODE=rgb+depth \
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
DEMO_TYPE=depth_norm \
bash examples/baselines/diffusion_policy_old/run_train_rgbd_depth_norm.sh
```

验证：
```bash
python -m py_compile examples/baselines/diffusion_policy_old/train_rgbd_depth_norm.py
bash -n examples/baselines/diffusion_policy_old/run_train_rgbd_depth_norm.sh
```

##### 训练结果

实验数据：
- depth-norm DiT：`runs/benchmark-pose-dit-rgbd-depth-norm`
  - 本次日志写到 `24k iters`，最后一次 eval 在 `20k`。
  - `20k`：`success_once = 0.03`，`success_at_end = 0.02`，`return = 0.98`。
  - 最优也出现在 `20k`：`best success_once = 0.03`，`best success_at_end = 0.02`。
  - `losses/total_loss` 从 `1.0253` 降到 `24k` 的 `0.0108`，说明训练 loss 能正常下降。
- rgb-only DiT 对照：`runs/benchmark-pose-dit-origin-rgb`
  - 同样 `20k`：`success_once = 0.38`，`success_at_end = 0.11`，`return = 5.02`。
  - 全程最优：`85k` 时 `success_once = 0.60`，`success_at_end = 0.29`，`return = 10.08`。

结论：
- 把 depth 从 `/1024 -> [0,1]` 改成 `/512 - 1 -> [-1,1]` 后，**没有恢复 DiT 的性能**。
- 这个实验虽然没有跑满 `100k`，但在同样 `20k` 位置，depth-norm 只有 `3%`，而 rgb-only 已经到 `38%`；差距很大，基本可以排除“depth 只是因为数值范围没归一化到 `[-1,1]` 才拖垮训练”的解释。
- depth-norm 的 loss 正常下降，但 eval 成功率几乎不涨，说明问题更像是 **depth 作为输入模态本身给当前 DiT / 数据规模 / 训练口径带来了负收益**，而不是简单的优化没收敛。
- 当前最可靠的方向仍然是 **去掉 depth，使用 rgb-only 作为默认 DiT 视觉输入**；若后续还要用 depth，应优先尝试更强的 depth encoder / modality dropout / 单独 depth ablation，而不是只改线性归一化。


## 4、Loss 设计

这部分的改动要针对整个diffusionpolicy文件夹进行，就是onlymas、maswindow、windowmixed三条线路同时改动。

### known-area与unknown-area分层设计

我需要在计算loss时，能够手动设置已知区域和未知区域的占比。
原本是对所有点做平均MSE，现在改成加权平均。

1、在三个sh文件的MAS WINDOW设置这块，加入loss_mode参数，如果设置为‘average’就维持原样，如果设置为‘weighted’就再接收一个 `known_area_weight`，例如 `0.2` 表示 known area 占 `0.2`，unknown area 占 `0.8`。
2、在train文件中做相应修改（计算loss的函数里传入mask或者mas，然后做相应计算）

具体实现方案：

loss 计算设计：

```python
per_elem_loss = (noise_pred - noise).square()
known_mask = action_mask.to(torch.bool)
unknown_mask = ~known_mask

known_loss = masked_mean(per_elem_loss, known_mask)
unknown_loss = masked_mean(per_elem_loss, unknown_mask)
loss = known_area_weight * known_loss + (1 - known_area_weight) * unknown_loss
```

## 5、control mode调整

### research on control mode

末端可用的三种 control mode 分别是：

- `pd_ee_pose`
- `pd_ee_delta_pose`
- `pd_ee_target_delta_pose`

这里真正要区分的是：**DP 预测出来的 action 数值，到底被 controller 当成绝对 pose、相对当前真实 TCP 的 delta，还是相对上一 target 的 delta。**

#### 通用控制链路

ManiSkill controller
对 EE pose controller 来说，action 先被转成目标末端位姿：
再由 IK 转成目标关节位置：
最后 SAPIEN 关节驱动用 PD 追踪目标关节位置：

#### `pd_ee_pose`

`pd_ee_pose` 把 action 前 6 维解释为 **绝对 TCP 目标位姿**
```text
a_t = [x_t, y_t, z_t, r^x_t, r^y_t, r^z_t, g_t]
```

#### `pd_ee_delta_pose`

`pd_ee_delta_pose` 把 action 前 6 维解释为 **相对当前真实 TCP 位姿的增量**

controller 先把动作映射成小位姿增量。Panda 默认平移范围是 `[-0.1, 0.1]` 米，旋转增量量级约 `0.1` rad：

```text
\delta p_t = scale_pos(a_t[0:3])
\delta r_t = scale_rot(a_t[3:6])
\Delta R_t = R_XYZ(\delta r_t)
```

然后读取当前真实 TCP 位姿 `T_t=(p_t,R_t)`，生成本步目标：

```text
\bar p_t = p_t + \delta p_t
\bar R_t = \Delta R_t R_t
\bar q_t = IK(\bar p_t, \bar R_t; q_t)
```

#### `pd_ee_target_delta_pose`

`pd_ee_target_delta_pose` 也把 action 前 6 维解释为小位姿增量，但它的参考点不是当前真实 TCP，而是 controller 内部保存的上一目标位姿

reset 时：

```text
\bar T_0 = T_0
```

每一步：

```text
\delta p_t = scale_pos(a_t[0:3])
\delta r_t = scale_rot(a_t[3:6])
\Delta R_t = R_XYZ(\delta r_t)

\bar p_t = \bar p_{t-1} + \delta p_t
\bar R_t = \Delta R_t \bar R_{t-1}
\bar q_t = IK(\bar p_t, \bar R_t; q_t)
```

关键点：这里的参考点是 **上一 target**。即使真实 TCP 没追上，target 仍然会继续按预测 delta 积分往前走。

#### 对 diffusion policy 的影响

1. `pd_ee_pose` 学的是绝对目标序列：动作语义更全局，数值范围更大；预测错一个绝对 pose 会直接给 controller 一个错误目标。

2. `pd_ee_delta_pose` 学的是当前状态附近的小步修正：动作分布更集中，通常更适合 DP；但 rollout 偏离 demo 后，后续 delta 会作用在偏离后的真实 TCP 上。

3. `pd_ee_target_delta_pose` 学的是目标轨迹的积分增量：执行会更像追一条连续 target path；但模型误差会在 target 上累积，target 可能逐渐跑到真实 TCP 前面。

### A new Design on control mode

**这部分的改动针对diffusionpolicy文件夹中的mixedmask脚本链路进行，创建副本链路，命名train_delta_action.py以及相应sh及evaluate文件**

我打算对模型做一个重要修改，大体思路如下图所示。
![alt text](image.png)
除了我所说的改动之外，其他都与原pipeline保持一致。

思路：
trainingtime
1、数据集采pd_ee_pose的数据，已知点等信息都统一到绝对action space中。
2、将数据集从pd_ee_pose转化为pd_delta_pose or pd_target_delta_pose
3、在每一个时间步，取出当前的7维pose，通过与当前时间步的mas_window做差得到当前时间步的mdas_window(masked delta action space)
4、输入mdas_window等相关condition信息，直接让模型预测噪声并reconstuct relative action space，注意这里预测出的动作序列为相对当前帧的relative action space而不是真正的delta action值
5、用预测出的relative action值之间做差，获得真正的delta action space后与步骤2中得到的真值算loss并更新参数。
inferencetime：
1、从测试集中获得只有稀疏点的绝对mas
2、在每一个时间步与当前位姿做差得到mdas_window并用于重建relative action space，做差得到delta action space
3、用pd_delta_pose or pd_target_delta_pose，执行当前步骤与预测出的动作

请你帮我1、分析这个方案的合理性 2、分析使用哪个delta control mode

## 6、mixed版本代码兼容性修改

原本diffusionpolicy文件夹中有train mas window 和 train window mixed两个线路，现在可以给mixed版做一个兼容，如果mixed的sh文件中，mask type list中只输入了一种mask type，那就默认回退到单一mask type，也就是mas window这个训练pipeline。

请你给train window mixed和reconstuct relative action space的版本这个两个线路都做兼容

在下面适当记录关键信息，别堆没用的：

### 修改记录

1. `run_train_window_mixed.sh`
   - 增加单 mask 检测：当 train/eval 的 `MASK_TYPE_LIST` 都只有一种且相同时，进入兼容模式。
   - 兼容模式直接 fallback 到 `run_train_mas_window.sh`。
   - 自动映射：
     - `PREPROCESS_MASK_TYPE = 单一 mask type`
     - `PREPROCESS_RETAIN_RATIO = 单一 mask ratio`
   - 这样单 mask 不再走 mixed preprocess / mixed train，而是回到原 `train_mas_window.py` pipeline。

2. `run_train_relative_action.sh`
   - 增加单 mask 检测。
   - relative action 目前没有独立 single-mask 训练脚本，因此保持 relative 链路，但自动修正 mixed mask 配置：
     - `TRAIN_NUM_MASK_TYPE=1`
     - `EVAL_NUM_MASK_TYPE=1`
     - `TRAIN_MASK_COMPOSITION_LIST=[1.0]`
     - `EVAL_MASK_COMPOSITION_LIST=[1.0]`
     - ratio list 保留单一 mask ratio
   - 解决 `MASK_TYPE_LIST=["random_mask"]` 但 `NUM_MASK_TYPE` 默认仍为 2 的不一致问题。

3. 验证
   ```bash
   bash -n examples/baselines/diffusion_policy/run_train_window_mixed.sh
   bash -n examples/baselines/diffusion_policy/run_train_relative_action.sh
   ```
   均通过。

4. 单 mask mixed fallback 检查
   ```bash
   TRAIN_MASK_TYPE_LIST='["random_mask"]' \
   EVAL_MASK_TYPE_LIST='["random_mask"]' \
   TRAIN_MASK_RATIO_LIST='[0.2]' \
   EVAL_MASK_RATIO_LIST='[0.2]' \
   bash examples/baselines/diffusion_policy/run_train_window_mixed.sh
   ```
   日志会出现：
   ```text
   [mixed-compat] single mask type detected: random_mask; fallback to run_train_mas_window.sh
   ```

## 7、MAS backward设计

这部分的改动针对diffusionpolicy文件夹中的maswindow、windowmixed（以及附带的relative线路）三条线路同时改动。

在之前的mas long window 设计中，mas long window只有一个长度参数，并且默认为从当前帧向前（forward）看一定长度，并用于卷积。
先在需要给mas向后看的能力，设计两个参数，一个long_window_backward_length(表示t-1方向)，一个long_window_forward_length（表示t+1方向）（包含当前帧t），如果前向超出则做重复padding。
根据两个长度参数获取mas后，卷积等操作都维持不变。

做这个修改并留适当记录：

### 修改记录

1. 新增 long MAS 双向窗口参数：
   - `long_window_backward_length`：取 `t-1` 方向的历史 MAS 点数。
   - `long_window_forward_length`：取 `t+1` 方向的未来 MAS 点数，包含当前帧 `t`。
   - 实际 long window 长度为二者之和。
   - 旧外部参数 `long_window_horizon` 已删除；不再作为 CLI / sh 参数暴露。
   - 未显式传 `long_window_forward_length` 时，默认使用 `pred_horizon`，因此默认仍等价于原 forward-only 逻辑。

2. 修改范围：
   - `utils/progress_utils.py`
     - 新增 `build_mas_window_around_step(...)`。
     - `build_mas_long_window_from_future(...)` 支持 backward/forward 参数。
     - `build_dual_mas_window_obs_horizon(...)` 支持双向 long window。
   - `utils/stpm_utils.py`
     - online eval 的 `build_dual_mas_window_condition_batch(...)` 同步支持双向 long window。
   - `train_mas_window.py`
   - `train_mas_window_mixed.py`
   - `train_relative_action.py`
   - `evaluate/evaluate_mas_window.py`
   - `evaluate/evaluate_mas_window_mixed.py`
   - `evaluate/evaluate_relative_action.py`
   - `evaluate/evaluate_relative_action_mixed.py`

3. 三个 sh 入口新增环境变量：
   ```bash
   LONG_WINDOW_BACKWARD_LENGTH=4 \
   LONG_WINDOW_FORWARD_LENGTH=12 \
   bash examples/baselines/diffusion_policy/run_train_mas_window.sh
   ```
   `run_train_window_mixed.sh` 和 `run_train_relative_action.sh` 同样支持这两个变量。

4. padding 语义：
   - backward 越过开头时重复第一帧。
   - forward 越过结尾时重复最后一帧。

### 一个修复

为什么还保留着LONG_WINDOW_HORIZON="${LONG_WINDOW_HORIZON:-${PRED_HORIZON}}"这个参数，这个应该删去，只保留，forward和backward两个参数，相加来决定最终mas长度

修复记录：
- 已从三条入口脚本删除 `LONG_WINDOW_HORIZON` 和 `--long-window-horizon`：
  - `run_train_mas_window.sh`
  - `run_train_window_mixed.sh`
  - `run_train_relative_action.sh`
- `LONG_WINDOW_FORWARD_LENGTH` 默认改为 `${PRED_HORIZON}`，`LONG_WINDOW_BACKWARD_LENGTH` 默认仍为 `0`。
- 已从三条训练脚本的 CLI 参数中删除 `long_window_horizon`：
  - `train_mas_window.py`
  - `train_mas_window_mixed.py`
  - `train_relative_action.py`
- `args.long_window_horizon` 现在只作为内部派生值：
  ```python
  args.long_window_horizon = (
      args.long_window_backward_length + args.long_window_forward_length
  )
  ```
