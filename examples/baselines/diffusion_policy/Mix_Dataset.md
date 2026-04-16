# Training with Mixed Dataset

**我们在这个文档里就当前问题进行交互，我会每次在这个文档里写上进一步指示，然后你按我的要求执行**

**这个文档里我写的内容你不要直接删除，只按照我的要求增加或者改动**

## 背景

在之前的训练实验过程中（细节见debug1.md），我们设置了不同的mask type，并且每次训练只使用单一种类的mask。为了获得范化性能更好的policy，我们打算采用混合数据集训练 (Mixed Dataset Strategy)，也就是在train_set和eval_set中使用由不同masktype数据混合而成的数据集。

具体输入方式：（sh文件中）
- num_mask_type:n
- mask_type_list:['type1','type2',...] (读取输入的列表，验证总数是否为n，以及名字合法)
- mask_type_ratio_list:[0.2,0.2,0.5,...] (读取输入的列表，验证是否和为1)
**指每种mask的占比，在train和eval集中保持一致**
- mask_param_list:[0.3,0.3,20,...] (读取输入的列表，验证masktype和param是否对应)
**注意这个mask_param列指的是每种mask所需要的额外参数，详见后面mask类型说明中，前四种masktype需要‘retain_ratio’,后两种需要‘mask_seq_len’,最后一种不需要额外参数。因此需要验证，若所需ratio则在01之间，若所需len，则在1-100之间，若不需要额外参数，则不读取该位置的参数** 
- mask_value (保留这个参数，一般仍设为0)

## 单一mask相关配置

注意：
  - 当前 diffusion preprocess 处理的是 `(T, 7)` 动作；先对 7 维动作做 mask，之后再通过 `progress_utils.py` 补一列 progress，最终 `mas` 变成 `(T, 8)`。

mask 类型说明：
  - `points`：需要 `retain_ratio`；随机保留 `int(T * retain_ratio)` 个 step 的 `x,y`。
  - `3D_points`：需要 `retain_ratio`；随机保留 `int(T * retain_ratio)` 个 step 的 `x,y,z`。
  - `pose_motion_planning`：需要 `retain_ratio`；随机保留 `int(T * retain_ratio)` 个 step 的完整 pose/action（前 7 维）。  
  - `random_mask`：需要 `retain_ratio`；在所有位置中随机保留 `retain_ratio` 比例的元素。

  - `mix0`：不需要额外参数；保留所有 step 的 `x,y`，再额外保留 1 个唯一时间步的前 7 维，以及另外 3 个唯一时间步的前 3 维；要求 `T >= 4`。
  - `2D_partial_trajectory`：需要 `mask_seq_len`；随机保留一个连续长度为 `mask_seq_len` 的 `x,y` 子序列。
  - `local_planner`：需要 `mask_seq_len`；先保留全部，再随机 mask 掉一个连续长度为 `mask_seq_len` 的前 7 维子序列。

  - `2D_video_trajectory`：不需要额外参数；保留每个 step 的 `x,y`。同`2D_image_trajectory` 

  - `auto_regressive`和`pose_AnyGrasp`：这两个模式之后不需要了，不用再保留。
  - `none`或`full`：当选择num_mask_type=0时，即默认不需要mask

当前data preprocess流程：

  - 调用 `examples/baselines/diffusion_policy/data_preprocess.py`，这个脚本再调用 `data_preprocess_tools/*` 里的函数。

- 具体调用链：
  - `run_train_mas_window.sh`
  - `data_preprocess.py`
  - `data_preprocess_tools/mask_utils.py`
  - `data_preprocess_tools/normalize_utils.py`
  - `data_preprocess_tools/obs_utils.py`
  - `data_preprocess_tools/progress_utils.py`
  - 然后再进入 `train_mas_window.py`
  - 训练加载时才会用到 `utils/load_train_data_utils.py`、`utils/build_progress_window_utils.py`

- shell 脚本阶段做的事情：
  - 先根据 `PREPROCESS_MASK_TYPE`、`PREPROCESS_RETAIN_RATIO`、`PREPROCESS_MASK_SEQ_LEN` 拼出输出目录和文件名后缀。
  - 检查以下四个文件是否已经存在：`train.h5`、`eval.h5`、`train.json`、`eval.json`。
  - 如果都存在，就直接复用，不再预处理。
  - 如果缺任意一个，就调用：
    - `python examples/baselines/diffusion_policy/data_preprocess.py ...`
  - 然后再调用：
    - `python examples/baselines/diffusion_policy/train_mas_window.py ...`

## mix dataset pipeline 设计

### 总体原则

- 不改训练侧数据格式：最终产物仍然是一个 `train.h5 + train.json + eval.h5 + eval.json`。
- mixed dataset 采用独立脚本和独立 py 入口，不与单一 mask 复用顶层入口文件。
- 单一 mask 继续使用 `data_preprocess.py`。
- mixed mask 单独新增 `data_preprocess_mixed.py`。
- 底层 mask / normalize / progress 工具仍然共用 `data_preprocess_tools/*`。

### 整体流程

1. shell 读取 mixed 配置。
2. `data_preprocess_mixed.py` 解析并校验 mixed 配置。
3. 先按原逻辑把 raw episode 划成 train/eval。
4. 在 train split 内部分配每条轨迹使用哪一种 mask；在 eval split 内也独立按同一组 ratio 分配。
5. 每条轨迹仍然只生成一次，只是它的 `mask_type/mask_param` 来自当前 split 的分配结果。
6. 生成出的 h5 结构保持不变：`actions / mas / mask / obs / meta`。
7. mixed train / eval 使用独立 py 文件，但底层工具函数可以复用。

### ratio 与 episode 分配规则

- `num_mask_type = 0`：
  - 等价于单一 `none`。
  - 不读取 `mask_type_list / mask_type_ratio_list / mask_param_list`。

- `num_mask_type > 0`：
  - `len(mask_param_list) == len(mask_type_list) == len(mask_type_ratio_list) == num_mask_type`

- `mask_type_ratio_list` 校验：
  - 每个值都在 `[0,1]`
  - 总和为 `1`

- `mask_param_list` 校验：
  - `points / 3D_points / pose_motion_planning / random_mask`：读取为 `retain_ratio`，要求在 `(0,1]`
  - `2D_partial_trajectory / local_planner`：读取为 `mask_seq_len`，要求在 `[1,100]`
  - `2D_video_trajectory / 2D_image_trajectory / mix0`：忽略该位置参数

- split 内分配算法：
  - 输入：该 split 的 `source_episode_ids` 和 `mask_type_ratio_list`
  - 先算每个 mask 的期望数量 `ratio * num_episodes`
  - 再用“最大余数法”取整，保证总数严格等于该 split 的 episode 数
  - 然后用固定随机种子打乱该 split 的 episode 顺序，按数量切块分配给各个 mask
  - 这样 train 和 eval 都能各自近似保持同一组比例

### 文件设计

#### 1. `examples/baselines/diffusion_policy/run_train_window_mixed.sh`

- 作用：
  - mixed dataset 的正式训练入口
  - 接收 mixed 配置
  - 组织 preprocess 参数
  - 调用 `data_preprocess_mixed.py`
  - 然后调用 `train_mas_window_mixed.py`

- 改动：
  - 不再复用单一 mask 的 `run_train_mas_window.sh`
  - 只保留 mixed 所需变量：
    - `NUM_MASK_TYPE`
    - `MASK_TYPE_LIST`
    - `MASK_TYPE_RATIO_LIST`
    - `MASK_PARAM_LIST`
  - 始终调用 `data_preprocess_mixed.py`
  - 输出目录后缀不要再只看单一 `PREPROCESS_MASK_TYPE`，而是根据整组 mixed spec 生成一个 canonical suffix，避免不同 mixed 配置写到同一路径

#### 2. `examples/baselines/diffusion_policy/data_preprocess.py`

- 作用：
  - 保持单一 mask preprocess 入口
  - 继续负责 raw data 读取、归一化、split、单一 mask 应用、写 h5/json

- 结论：
  - 主体逻辑尽量不动
  - 最多只抽一些通用 helper 给 `data_preprocess_mixed.py` 复用

#### 3. `examples/baselines/diffusion_policy/data_preprocess_mixed.py`

- 作用：
  - mixed mask 的独立 preprocess 入口
  - 负责 raw data 读取、归一化、split、mixed mask 分配、写 h5/json

- 新增参数解析：
  - `--num-mask-type`
  - `--mask-type-list`
  - `--mask-type-ratio-list`
  - `--mask-param-list`

- 新增/调整函数设计：
  - `parse_args()`
    - 增加 mixed 参数
    - 保留 `mask_value / num_traj / split_seed / mask_seed / output_dir` 等通用参数

  - `build_output_stem(...)`
    - mixed 时根据完整 spec 生成稳定文件名前缀

  - `normalize_mixed_mask_config(args)`
    - 把 mixed 输入统一整理成 `mask_specs`
    - 输出统一格式，例如：
      - `[{mask_type, ratio, retain_ratio, mask_seq_len}, ...]`

  - `split_source_episode_ids(...)`
    - 可直接复用单一 preprocess 的现有逻辑

  - `assign_mask_specs_to_episodes(source_episode_ids, mask_specs, seed)`
    - 新函数
    - 在某个 split 内，把每个 `source_episode_id` 映射到一个 `mask_spec`

  - `write_split_h5(...)`
    - 从“整个 split 用同一个 mask_type”改成“每条轨迹查自己的 `mask_spec`”
    - 对每条轨迹：
      - 先做 action/state normalize
      - 再按该轨迹分配到的 `mask_type + param` 调 `apply_mask_to_actions()`
      - 再补 progress，写入 `mas`
    - 同时把每条轨迹实际使用的 `mask_type / retain_ratio / mask_seq_len` 写进 traj 级别字段，方便追踪

  - `build_split_metadata_json(...)`
    - `preprocess_info` 中新增 mixed 配置摘要
    - `episodes[*]` 中新增该 episode 的实际 `mask_type` 和参数

  - `main()`
    - 先统一 mixed `mask_specs`
    - 再 split
    - 再分别给 train/eval 分配 mask
    - 最后写出 mixed train/eval 数据集

#### 4. `examples/baselines/diffusion_policy/train_mas_window_mixed.py`

- 作用：
  - mixed dataset 的独立训练入口
  - 负责加载 mixed 预处理数据、训练 policy、触发 mixed eval

- 设计：
  - 不直接复用 `train_mas_window.py` 作为顶层入口
  - 可以复用其中稳定的模型组件、dataset 工具、优化器逻辑
  - 额外新增 mixed 相关 log：
    - 当前 mixed 配置摘要
    - train/eval 中每种 mask 的样本数量
    - 每轮 eval 的 overall success 和 per-mask success

#### 5. `examples/baselines/diffusion_policy/evaluate/evaluate_mas_window_mixed.py`

- 作用：
  - mixed dataset 的独立 eval 逻辑
  - 按不同 `mask_type` 分组统计 success rate

- 新增/调整函数设计：
  - `group_eval_data_by_mask_type(...)`
    - 按 eval demo 对应的 `mask_type` 分组

  - `evaluate_one_mask_type_group(...)`
    - 对某一个 mask group 单独 rollout 并统计 success

  - `evaluate_mas_window_mixed(...)`
    - 汇总 overall success
    - 同时返回：
      - `success_rate_overall`
      - `success_rate_by_mask_type`
      - 每个 mask type 的 episode 数

- 说明：
  - 由于 mixed eval 需要分 mask type 统计 success rate，因此不再复用原来的 eval 入口

#### 6. `examples/baselines/diffusion_policy/data_preprocess_tools/mask_utils.py`

- 作用：
  - 保持“单条 action 序列如何应用某个 mask”的底层逻辑

- 新增/调整函数设计：
  - `SUPPORTED_MASK_TYPES`
    - 去掉不再使用的：
      - `auto_regressive`
      - `pose_AnyGrasp`
    - 保留：
      - `points`
      - `3D_points`
      - `pose_motion_planning`
      - `random_mask`
      - `mix0`
      - `2D_partial_trajectory`
      - `local_planner`
      - `2D_video_trajectory`
      - `none`

  - `validate_mask_config(...)`
    - 继续负责“单个 mask_type 的参数是否合法”

  - `validate_mixed_mask_config(num_mask_type, mask_type_list, mask_type_ratio_list, mask_param_list)`
    - 新函数
    - 负责 mixed 级别校验：数量、名字、ratio 总和、参数类型和范围

  - `build_mask_spec(mask_type, raw_param, ratio)`
    - 新函数
    - 把原始输入整理成标准 spec

  - `apply_mask_to_actions(...)`
    - 主体逻辑基本不变
    - 继续只处理一条 `(T,7)` 动作和一个 `mask_type`

#### 7. `examples/baselines/diffusion_policy/data_preprocess_tools/progress_utils.py`

- 作用：
  - 不改 mixed 逻辑
  - 继续负责给 masked action 补 progress，生成 `(T,8)` 的 `mas`

- 结论：
  - 无需新增函数

#### 8. `examples/baselines/diffusion_policy/utils/load_train_data_utils.py`

- 作用：
  - 从 h5 读 `observations/actions/mas/mask/meta`

- 结论：
  - 可以继续复用
  - 若需要，也可以补一个读取 traj 级别 `mask_type` 的 helper

### h5 / json 写出规则

- `meta` 级别新增：
  - `mixed_mask_enabled`
  - `num_mask_type`
  - `mask_type_list`
  - `mask_type_ratio_list`
  - `mask_param_list`

- `traj` 级别新增：
  - `mask_type`
  - 若适用则写：
    - `retain_ratio`
    - `mask_seq_len`

- 现有字段保持不变：
  - `actions`
  - `mas`
  - `mask`
  - `obs/state`

### 最小改动原则

- mixed 的核心应放在独立的 `data_preprocess_mixed.py`，不放在 dataloader。
- `apply_mask_to_actions()` 继续保持“单轨迹、单 mask”的简单接口。
- 顶层入口文件与 eval 统计逻辑独立新建，但底层公共工具尽量复用。

## 进一步改动指令

需要进行两个改动:

1、之前在设计的过程中没有考虑到同一masktype重复出现的情况。
现在我需要实现统一masktype重复使用的功能。如demos/data_1_mixed_random_mask__random_mask__random_mask__points__points__3D_points__pose_motion_planning_b9f756f1cc_train.json里的设置，如果使用这种设置就会出问题，因为程序只读取了某个masktype第一次出现时的参数。
我需要你进行修改，判断masktype是否有重复，如果有重复，对多次重复的相同masktype进行编号，然后视作不同mask，读取不同的param，然后以其ratio创建新的数据集。

2、之前在设计中训练集和测试集使用的是同样的masktype和比例。
现在将这两个设置分开，测试集和训练集用相同的一套参数模式，人为手动调节其成分和比例，参数设置和规则与训练集一致，请你加入这部分设计（py文件和sh文件）

先不改代码，把方案写出来

## 改进方案

### 总体目标

这次改动只解决两个问题：

1. mixed 配置里允许同一 `mask_type` 重复出现，并且重复项要被视为不同的 mask 配置项，各自读取自己的参数和 ratio。
2. train split 和 eval split 不再强制共用同一套 mixed 配置，而是分别提供两套配置，参数规则完全一致，但成分和比例可以人工独立调节。

本轮先只出方案，不直接改代码。

### 问题 1：重复 `mask_type` 被错误合并

#### 当前问题

当前 mixed preprocess 虽然允许输入：

- `["random_mask", "random_mask", "random_mask", "points", "points", "3D_points", "pose_motion_planning"]`

但训练侧在读取 h5 `meta` 时，会把：

- `mask_type_list_json`
- `mask_type_ratio_list_json`

压成一个 `dict[str, float]`。

这样一来，重复 key 会被覆盖，导致：

- `random_mask` 只保留最后一次 ratio
- `points` 只保留最后一次 ratio
- 不同重复项对应的不同 param 也无法区分
- 后续分层抽样总数出错，或统计被错误合并

#### 改动原则

- 底层真正执行 mask 时，仍然使用原始 `mask_type`。
- 但在 mixed 配置、split 分配、h5/json 元信息、训练采样和日志统计里，引入“槽位级别”的唯一标识，避免重复类型被合并。

#### 新概念：`mask_slot`

对 mixed 输入列表中的每一项，都视作一个独立 `mask_slot`。

每个 `mask_slot` 至少包含：

- `mask_type_base`
  - 原始 mask 类型，如 `random_mask`
- `mask_slot_index`
  - 同一 `mask_type_base` 内部的出现序号
- `mask_slot_name`
  - 唯一名字，如：
    - `random_mask#0`
    - `random_mask#1`
    - `random_mask#2`
    - `points#0`
    - `points#1`
    - `3D_points`
    - `pose_motion_planning`
- `ratio`
- `retain_ratio`
- `mask_seq_len`

编号规则：

- 按输入顺序扫描 `mask_type_list`
- 对每个 `mask_type_base` 单独计数
- 若某个类型总共只出现一次，可直接保留原名，也可以统一写成 `type#0`
- 内部统一使用带编号形式，即所有项都生成 `mask_slot_name`

例如输入：

- `["random_mask", "random_mask", "random_mask", "points", "points", "3D_points", "pose_motion_planning"]`

会标准化为：

- `random_mask#0`
- `random_mask#1`
- `random_mask#2`
- `points#0`
- `points#1`
- `3D_points#0`
- `pose_motion_planning#0`

注意：

- `apply_mask_to_actions()` 仍只接收 `mask_type_base`
- `mask_slot_name` 只用于“区分不同配置项”

#### 需要调整的数据结构

当前 `build_mask_spec()` 输出的结构不够区分重复项，建议扩展为：

```python
{
    "mask_type": "random_mask",          # 原始 base type，保留现有字段名以兼容底层逻辑
    "mask_type_base": "random_mask",     # 显式记录 base type，便于后续代码阅读
    "mask_slot_name": "random_mask#1",   # mixed 级别唯一标识
    "mask_slot_index": 1,                # 同类内部编号
    "ratio": 0.2,
    "retain_ratio": 0.1,
    "mask_seq_len": None,
}
```

这样每个配置项既保留“它是什么 mask”，又保留“它是第几个同类配置”。

#### preprocess 侧改动

`data_preprocess_mixed.py` 中：

- `normalize_mixed_mask_config(...)`
  - 不再只返回“按类型区分”的 `mask_specs`
  - 而是返回带 `mask_slot_name` 的 `mask_specs`

- `assign_mask_specs_to_episodes(...)`
  - 继续按 `ratio` 分配
  - 但分配单位改为 `mask_slot`
  - 即 `random_mask#0`、`random_mask#1`、`random_mask#2` 各自单独拿到自己的 episode 子集

- `build_output_stem(...)`
  - 文件名前缀的 canonical spec 必须包含 `mask_slot_name + param + ratio`
  - 不能只看 `mask_type`
  - 否则不同重复配置仍然可能算出相同 stem

#### h5 / json 写出规则改动

为避免训练侧再次合并，meta 和 traj 都要补足“base type”和“slot name”两层信息。

`meta` 级别建议新增：

- `mask_slot_name_list_json`
- `mask_slot_ratio_list_json`
- `mask_slot_param_list_json`
- `mask_slot_specs_json`

原有字段建议保留，但语义调整为“原始输入顺序”：

- `mask_type_list_json`
- `mask_type_ratio_list_json`
- `mask_param_list_json`

`traj` 级别建议新增：

- `mask_type_slot`
  - 如 `random_mask#1`
- `mask_slot_index`

保留原有：

- `mask_type`
  - 仍写 base type，如 `random_mask`

`json` 里的 `episodes[*]` 也同步新增：

- `mask_type_slot`
- `mask_slot_index`

这样：

- 按基础类型统计时，用 `mask_type`
- 按重复项区分时，用 `mask_type_slot`

#### train / eval 侧读取改动

`train_mas_window_mixed.py` 当前的关键问题是把 ratio 读成了 `dict[str, float]`。

训练时应该不受影响、就是评估时需要分不同masktypeslot统计成功率

这里需要改为：

- 新增 `load_target_mask_ratio_by_slot(...)`
  - 从 `mask_slot_name_list_json` 和 `mask_slot_ratio_list_json` 读取
  - 返回“有序列表”或“list[tuple[slot_name, ratio]]”
  - 不再返回 `dict`

- 新增读取轨迹 slot 的 helper
  - 训练采样时优先读取 `traj/mask_type_slot`
  - 若旧数据没有该字段，再回退到 `traj/mask_type`

- `stratified_select_demo_indices(...)`
  - 输入不再是按 `mask_type` 的字典
  - 改成：
    - `slot_names: list[str]`
    - `target_slots: list[str]`
    - `target_ratios: list[float]`
  - 保持顺序，禁止 key 合并

#### 日志和评估统计改动

为了兼顾“总类别表现”和“重复项细节”，建议 mixed eval 同时输出两套统计：

- `by_mask_type`
  - 按基础类型合并
- `by_mask_slot`
  - 按带编号槽位区分

例如：

- `random_mask` 总体 success
- `random_mask#0 / #1 / #2` 各自 success

这样既能看大类，又能看同类不同参数的差异。

### 问题 2：train / eval 使用独立 mixed 配置

#### 当前问题

当前 `data_preprocess_mixed.py` 和 `run_train_window_mixed.sh` 都只有一套：

- `NUM_MASK_TYPE`
- `MASK_TYPE_LIST`
- `MASK_TYPE_RATIO_LIST`
- `MASK_PARAM_LIST`

然后 train split 和 eval split 只是：

- 用同一组 `mask_specs`
- 分别按不同 seed 分配 episode

这不满足现在的新需求：train 和 eval 应该允许使用两套独立 mixed 配置。

#### 新设计原则

- train 和 eval 使用同一套“参数格式规则”
- 但配置内容相互独立
- 可以类型相同、比例不同
- 也可以类型集合不同、参数不同

#### shell 参数改动

`run_train_window_mixed.sh` 建议拆成两组环境变量：

训练集配置：

- `TRAIN_NUM_MASK_TYPE`
- `TRAIN_MASK_TYPE_LIST`
- `TRAIN_MASK_TYPE_RATIO_LIST`
- `TRAIN_MASK_PARAM_LIST`

测试集配置：

- `EVAL_NUM_MASK_TYPE`
- `EVAL_MASK_TYPE_LIST`
- `EVAL_MASK_TYPE_RATIO_LIST`
- `EVAL_MASK_PARAM_LIST`

其余通用参数继续保留：

- `PREPROCESS_MASK_VALUE`
- `PREPROCESS_NUM_TRAJ`
- `RAW_DEMO_H5`
- `RAW_DEMO_JSON`

为了兼容旧用法，建议保留一个过渡策略：

- 若 `EVAL_*` 未显式提供，则默认复制 `TRAIN_*`

这样：

- 老脚本还能跑
- 新需求也能直接启用

#### preprocess 参数改动

`data_preprocess_mixed.py` 的参数解析也拆成两组：

训练 split：

- `--train-num-mask-type`
- `--train-mask-type-list`
- `--train-mask-type-ratio-list`
- `--train-mask-param-list`

评估 split：

- `--eval-num-mask-type`
- `--eval-mask-type-list`
- `--eval-mask-type-ratio-list`
- `--eval-mask-param-list`

内部处理上：

- `normalize_mixed_mask_config(...)`
  - 改成可接收前缀，或拆成：
    - `normalize_split_mask_config(args, split="train")`
    - `normalize_split_mask_config(args, split="eval")`

最终得到：

- `train_mask_specs`
- `eval_mask_specs`

#### split 分配逻辑改动

当前逻辑：

- `train_ids` 用同一组 `mask_specs`
- `eval_ids` 用同一组 `mask_specs`

改成：

- `train_ids` 只用 `train_mask_specs`
- `eval_ids` 只用 `eval_mask_specs`

即：

```python
train_assignments = assign_mask_specs_to_episodes(train_ids, train_mask_specs, seed=...)
eval_assignments = assign_mask_specs_to_episodes(eval_ids, eval_mask_specs, seed=...)
```

train/eval 的分配算法仍相同，只是输入 spec 不再共享。

#### 输出文件命名改动

因为最终一个 mixed 目录下的四个文件：

- `*_train.h5`
- `*_train.json`
- `*_eval.h5`
- `*_eval.json`

同时依赖 train config 和 eval config，所以 `build_output_stem(...)` 也必须改。

建议 stem 的 canonical digest 基于：

```python
{
    "train_mask_specs": [...],
    "eval_mask_specs": [...],
}
```

否则：

- train 配置一样
- eval 配置不同

时，会错误复用同一套旧文件。

#### meta / json 记录改动

由于 train.h5 和 eval.h5 是两个独立文件，各自只需要写自己 split 实际使用的配置即可。

也就是说：

- `train.h5` / `train.json` 记录 `train_mask_specs`
- `eval.h5` / `eval.json` 记录 `eval_mask_specs`

此外，建议在 `preprocess_info` 里补一个“全局来源摘要”，便于追踪：

- `requested_train_mask_specs`
- `requested_eval_mask_specs`

这样即使单独看某个 split 文件，也知道整次 mixed preprocess 是怎么配的。

### 文件级改动建议

#### 1. `run_train_window_mixed.sh`

改动点：

- 增加 `TRAIN_*` / `EVAL_*` 两套 mixed 配置变量
- `compute_mixed_file_stem()` 同时传入 train/eval 两套 spec
- `ensure_preprocessed_dataset()` 调 `data_preprocess_mixed.py` 时同时传两套参数
- 复用旧文件前，文件名天然已由新 stem 区分 train/eval 配置差异

#### 2. `data_preprocess_mixed.py`

改动点：

- 参数解析改成 train/eval 两套
- `normalize_mixed_mask_config(...)` 拆成 split-aware 版本
- 生成 `mask_slot_name`
- `assign_mask_specs_to_episodes(...)` 按 `mask_slot` 分配
- `write_split_h5(...)` / `build_split_metadata_json(...)`
  - 同时写 `mask_type` 和 `mask_type_slot`
- `build_output_stem(...)`
  - digest 同时包含 train/eval spec

#### 3. `data_preprocess_tools/mask_utils.py`

改动点：

- `validate_mixed_mask_config(...)`
  - 继续校验合法性
  - 但不再假设 `mask_type` 唯一
- `build_mask_spec(...)`
  - 扩展为支持 `mask_slot_name` / `mask_slot_index`
  - 或者保留旧接口，再在 mixed preprocess 里补 slot 信息

建议：

- 尽量不要把“编号逻辑”塞进单一 mask 工具
- 编号属于 mixed 层语义，更适合放在 `data_preprocess_mixed.py`

#### 4. `train_mas_window_mixed.py`

改动点：

- 停止使用 `dict[str, float]` 保存目标比例
- 改读 `mask_type_slot`
- 分层采样按 `mask_slot` 进行
- 日志同时打印：
  - `train mask_slot counts`
  - `train mask_type counts`
  - `eval mask_slot counts`
  - `eval mask_type counts`

#### 5. `evaluate/evaluate_mas_window_mixed.py`

改动点：

- 分组函数同时支持：
  - 按 `mask_type`
  - 按 `mask_type_slot`
- 默认主汇报仍保留 `overall success`
- 额外输出：
  - `success_rate_by_mask_type`
  - `success_rate_by_mask_slot`

### 兼容性策略

建议兼容旧 mixed 数据集，避免一次性全部失效。

训练读取阶段采用以下优先级：

1. 若存在 `mask_type_slot` / `mask_slot_name_list_json`
   - 按新逻辑处理
2. 否则回退到旧字段：
   - `mask_type`
   - `mask_type_list_json`
   - `mask_type_ratio_list_json`

这样：

- 新数据可以支持重复类型
- 旧数据仍可继续训练

但要注意：

- 旧格式 mixed 数据本身不支持区分重复类型
- 所以旧数据若真的包含重复 `mask_type`，仍然应该重新 preprocess

### 建议实施顺序

1. 先改 `run_train_window_mixed.sh` 和 `data_preprocess_mixed.py`
   - 解决 train/eval 双配置
   - 解决重复 `mask_type` 编号和写盘
2. 再改 `train_mas_window_mixed.py`
   - 解决训练侧分层抽样读错 ratio 的问题
3. 最后改 `evaluate_mas_window_mixed.py`
   - 增加 `by_mask_slot` 统计

### 建议验收项

最少做以下几组检查：

1. 不重复类型、train/eval 同配置
   - 结果应与当前 mixed 行为一致
2. 重复类型、train/eval 同配置
   - 如 `random_mask` 重复 3 次、`points` 重复 2 次
   - 训练不应再出现 ratio 被覆盖
3. 不重复类型、train/eval 不同配置
   - train 和 eval 输出文件 stem 应不同于旧逻辑
   - 两个 split 的 meta/json 应分别记录各自配置
4. 重复类型、train/eval 不同配置
   - 这是最终目标场景
5. `num_mask_type=0`
   - 仍应等价于 `none`

### 本轮结论

建议的核心不是“把重复 `mask_type` 强行合并”，而是：

- 在 mixed 层引入唯一 `mask_slot`
- 在底层 mask 执行层继续保留原始 `mask_type`

这样才能同时满足：

- 同类 mask 多参数重复出现
- train/eval 配置独立
- 训练采样和评估统计都不再发生信息丢失


## 进一步指示

根据plan改动代码

## 已实现内容

本轮已完成以下代码改动：

- `run_train_window_mixed.sh`
  - 支持 `TRAIN_*` / `EVAL_*` 两套 mixed 配置
  - `EVAL_*` 未提供时默认回退到 `TRAIN_*`
  - 输出 stem 哈希同时包含 train/eval 两边 spec

- `data_preprocess_mixed.py`
  - 支持 train/eval 两套 mixed 配置解析
  - 对重复 `mask_type` 自动生成 `mask_type_slot`
  - train/eval split 分别使用各自 spec 分配 mask
  - h5 / json 增加 `mask_type_slot` 相关元信息

- `train_mas_window_mixed.py`
  - 不再把 mixed ratio 压成 `dict[str, float]`
  - 分层采样改为保留输入顺序，支持重复标签
  - 兼容读取新的 `mask_type_slot`

- `evaluate/evaluate_mas_window_mixed.py`
  - 保留 overall 指标
  - 同时输出 `by_mask_type` 和 `by_mask_slot` 两套统计

- `utils/load_eval_data_utils.py`
  - 新增 `mask_type_slot` 的读取与回退逻辑

当前已做的最小验证：

- Python 语法编译通过
- `run_train_window_mixed.sh` 的 `bash -n` 检查通过
- 重复 `mask_type` 的 `mask_slot` 生成 smoke test 通过
- `mask_type_slot` 读盘 smoke test 通过
