# Optimaize

**我们在这个文档里就一些问题进行交互，我会每次在这个文档里写上进一步指示，然后你按我的要求执行**
**这个文档里我写的内容你不要直接删除，只按照我的要求增加或者改动**

到此为止，整个模型已经完整了（主体内容全部在dffusion_policy文件夹中），接下来我们需要对模型performance做优化调整。

先做一个小改动，把所有sh文件里的MASK_TYPE_RATIO_LIST改成MASK_COMPOSITION_LIST，把所有MASK_PARAM_LIST改成MASK_RATIO_LIST，并且对代码库相关地方做替换。

## Dataset 采集

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
- `render_mode`：`rgb_array`
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
## Dataset 预处理

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

## Model

这部分的改动要针对整个diffusionpolicy文件夹进行，就是onlymas、maswindow、windowmixed三条线路同时改动。

### Benchmark test Debug

在之前的degub过程中，benchmark test存在一定问题（详见Debug4.md）
现在需要重新跑benchmark test，但这次为了保证准确，我将直接从maniskill官网上pull原始dp文件夹，重新采集并且使用pd delta mode的模式的数据集。
请你：
1、给我从maniskill上把dp的文件夹copy下来，放在现在的diffusionpolicy文件夹旁边，然后命名为diffusion_policy_origin
2、（环境用maniskill_py311）用命令行，用panda的mp功能采集600条新的训练数据（命名experiment_4），然后replay成（pdeedeltapose、rgbd）
3、最后给我启动训练脚本的命令行，里面要有所有可改参数，写到这个md文件里。

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

## Loss 设计

这部分的改动要针对整个diffusionpolicy文件夹进行，就是onlymas、maswindow、windowmixed三条线路同时改动。

### mask-area与none-mask-area分层设计

我需要在计算loss时，能够手动设置已知和位置区域的占比。
原本是对所有点做平均MSE，现在改成加权平均。

1、在三个sh文件的MAS WINDOW设置这块，加入loss_mode参数，如果设置为‘average’就维持原样，如果设置为‘weighted’就再接收一个0.2，位置区域*0.8。
2、在train文件中做相应修改（计算loss的函数里传入mask或者mas，然后做相应计算）

具体实现方案：

loss 计算设计：

```python
per_elem_loss = (noise_pred - noise).square()
known_mask = action_mask.to(torch.bool)
unknown_mask = ~known_mask

known_loss = masked_mean(per_elem_loss, known_mask)
unknown_loss = masked_mean(per_elem_loss, unknown_mask)
loss = known_w * known_loss + unknown_w * unknown_loss
```

### Eval video frequency 拆分

#### 执行记录

- 新增 `utils/eval_video_sampling_utils.py`：
  - `validate_eval_video_config(...)`：统一校验 `num_eval_episodes` 必须能被 `num_eval_envs` 和 `capture_video_freq` 整除。
  - `build_capture_indices(...)`：按全局 eval episode 下标生成 `0, freq, 2*freq, ...`。
  - `build_eval_batches(...)`：把需要采样的视频 episode 尽量放到 env0，并用 `valid_count` 区分真实 episode 和 padding/重复 rollout。
- 修改三条 train 链路：
  - 新增 `capture_video_freq: int = 10`。
  - `only_mas` 移除硬编码 `eval_freq * 10` 的视频保存周期。
  - `mas_window` 只在 batch 的 env0 对应采样 index 时归档视频，否则删除新视频。
  - `window_mixed` 在 mixed evaluator 内按全局 eval episode index 采样，不按 mask slot 内部 index 采样。
- 修改三个 sh：
  - 新增 `CAPTURE_VIDEO_FREQ="${CAPTURE_VIDEO_FREQ:-10}"`。
  - 透传 `--capture-video-freq "$CAPTURE_VIDEO_FREQ"`。
- 已验证：
  - `python -m py_compile` 通过。
  - `bash -n run_train_only_mas.sh` 通过。
  - `bash -n run_train_mas_window.sh` 通过。
  - `bash -n run_train_window_mixed.sh` 通过。
  - 小单元检查通过：
    - `100 / env=10 / freq=10` 生成 10 个采样 index。
    - `100 / env=5 / freq=10` 生成 10 个采样 index。
    - `100 / env=10 / freq=20` 生成 5 个采样 index。
    - `100 % 6 != 0` 与 `100 % 12 != 0` 都会报错。
