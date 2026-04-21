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

### 1. sh 参数设计

只改 `examples/baselines/diffusion_policy/run_train_window_mixed.sh` 的 mixed preprocess 分区。

新增：

```bash
PREPROCESS_MASK_ASSIGN_MODE="${PREPROCESS_MASK_ASSIGN_MODE:-composition}"
```

可选值：
- `composition`
- `one_demo_multi_mask`

`composition` 模式继续使用：
- `TRAIN_NUM_MASK_TYPE`
- `TRAIN_MASK_TYPE_LIST`
- `TRAIN_MASK_COMPOSITION_LIST`
- `TRAIN_MASK_RATIO_LIST`
- eval 对应一套参数

`one_demo_multi_mask` 模式使用：
- `TRAIN_NUM_MASK_TYPE`
- `TRAIN_MASK_TYPE_LIST`
- `TRAIN_MASK_RATIO_LIST`
- eval 对应一套参数

在 `one_demo_multi_mask` 模式下，`*_MASK_COMPOSITION_LIST` 不参与轨迹分配；可以保留为兼容字段，但不要求用户手动设置。metadata 中可写成 uniform ratio，仅用于记录。

重复 mask type 规则：
- `mask_type_list` 允许重复。
- 同一 `mask_type` 的不同 param 视为不同 slot。
- slot 命名继续使用已有规则：`points#0`、`points#1`、`random_mask#0`。
- `one_demo_multi_mask` 展开时按 slot 展开，而不是按 base mask type 去重。

### 2. data_preprocess_mixed.py 参数设计

新增 CLI 参数：

```bash
--mask-assign-mode composition|one_demo_multi_mask
```

内部新增或改造函数：

- `normalize_split_mask_config(args, split, mode)`
  - `composition`：保持现有校验，要求 composition list 长度等于 mask type 数量且总和为 1。
  - `one_demo_multi_mask`：只要求 type list 和 param list 长度等于 mask type 数量；composition 自动设为 `1 / num_mask_type`，仅用于 metadata。

- `build_output_stem(...)`
  - 文件名中加入 mode，避免 `composition` 和 `one_demo_multi_mask` 产物冲突。
  - 例如：`data_1_mixed_...` 与 `data_1_onedemo_...` 区分。

### 3. 轨迹展开逻辑

新增 helper：

```python
build_mask_jobs(source_episode_ids, mask_specs, mode, seed)
```

返回 job list，每个 job 至少包含：
- `source_episode_id`
- `mask_spec`
- `mask_slot_index`
- `mask_type_slot`
- `source_mask_copy_key`，例如 `traj_12_0`

两种模式：

- `composition`
  - 先按当前最大余数法给 source episode 分配 mask。
  - 每个 source episode 只生成 1 个 job。

- `one_demo_multi_mask`
  - 对每个 source episode 和每个 mask spec 做笛卡尔积。
  - 每个 source episode 生成 `num_mask_type` 个 job。
  - 若 mask type 重复，仍按 `mask_type_slot` 展开；例如 `points#0` 和 `points#1` 是同一 source demo 的两份不同 copy。
  - 再用固定 seed shuffle job list，使 h5 中 mask type 顺序打乱，降低 batch 内同类型聚集风险。

### 4. h5/json 写入规则

为了兼容现有 loader，h5 group 仍建议使用：

```text
traj_0
traj_1
traj_2
...
```

不直接用 `traj_0_0` 作为 h5 key，因为当前 loader 依赖 `int(key.split("_")[-1])` 排序，多级编号可能引入排序歧义。

但在每个 traj 里新增字段记录可读编号：

- `source_episode_id`
- `source_mask_copy_key`，如 `traj_0_0`
- `mask_type`
- `mask_type_slot`
- `mask_slot_index`

json 的 `episodes[*]` 同步写入：

- `episode_id`：展开后的本地 id
- `source_episode_id`：原始 demo id
- `source_mask_copy_key`：如 `traj_0_0`
- 当前 mask type 和参数

meta / preprocess_info 新增：

- `mask_assign_mode`
- `source_num_episodes`
- `expanded_num_episodes`
- `num_mask_type`

### 5. 训练侧 NUM_DEMOS / NUM_EVAL_DEMOS 语义

当前 `train_mas_window_mixed.py` 的 `num_demos` 是按 h5 中的 expanded traj 数选择的；这不符合新需求。

需要改为：

- `composition` 模式：保持当前语义，`NUM_DEMOS` 表示选择多少条 h5 轨迹。
- `one_demo_multi_mask` 模式：`NUM_DEMOS` 表示选择多少个 source demo，然后自动包含这些 source demo 的所有 mask copy。

建议新增 helper：

```python
select_train_demo_indices_by_mode(
    source_episode_ids,
    mask_type_slots,
    num_source_demos,
    mode,
    seed,
)
```

输出仍然是 h5 local traj indices，传给 Dataset。

eval 同理：

- `NUM_EVAL_DEMOS=10`
- `num_mask_type=5`
- 实际 eval expanded traj 数为 `50`
- reset seeds 允许重复，因为同一个 source demo 的不同 mask copy 应在相同初始状态下评估。

### 6. load_eval / metadata 对齐

`load_traj_mask_types()`、`load_traj_mask_type_slots()` 可继续按 h5 traj 读取。

需要确认或补充：
- `source_episode_ids` 对 expanded traj 一一对应。
- `infer_eval_reset_seeds_from_demo()` 从 json 读取 expanded episodes 时，每个 mask copy 都保留原 source demo 的 reset seed。
- `eval_traj_ids` 与 expanded traj 数一致。

### 7. 验证计划

最小验证：

1. 单元级验证：
   - `mask_assign_mode=composition` 时，输出 job 数等于 source episode 数。
   - `mask_assign_mode=one_demo_multi_mask` 时，输出 job 数等于 `source episode 数 * num_mask_type`。
   - 每个 source episode 都包含完整 mask type 集合。
   - 重复 mask type 不应被合并；`points#0` 和 `points#1` 应分别生成 copy，并保留各自参数。

2. 小数据 preprocess：
   - `PREPROCESS_NUM_TRAJ=6`
   - `TRAIN_NUM_MASK_TYPE=3`
   - 预期 train/eval split 后，每个 split 的 expanded 数等于 source split 数乘 3。

3. h5 检查：
   - `none` copy 的 `mas[:, :7]` 全 0，`mask[:, :7]` 全 False。
   - `full` copy 的 `mas[:, :7]` 等于 normalized action，`mask[:, :7]` 全 True。
   - 同一个 `source_episode_id` 下存在多个 `mask_type_slot`。

4. train smoke：
   - `TOTAL_ITERS=1`
   - `NUM_DEMOS=2`
   - `TRAIN_NUM_MASK_TYPE=3`
   - 确认实际 Dataset 选择到 `2*3=6` 条 source-mask copies。

#### 执行记录

已实现 `one_demo_multi_mask`：

- `run_train_window_mixed.sh`
  - 新增 `PREPROCESS_MASK_ASSIGN_MODE`，默认 `composition`。
  - 可设为 `one_demo_multi_mask`。
  - 文件名计算和 `data_preprocess_mixed.py` 调用都会传入该 mode。

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
