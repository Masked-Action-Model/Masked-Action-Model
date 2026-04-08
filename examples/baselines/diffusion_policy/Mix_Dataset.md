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
  - `2D_video_trajectory`：忽略该位置参数

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
