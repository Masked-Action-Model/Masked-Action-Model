我要对我们的模型数据预处理部分做一些调整。
将一些对原始数据的预处理操作整合起来放到 `data_preprocess.py` 里，并统一在主函数中调用，简化数据入口。

下面是详细 plan，同时也同步记录当前已经落地的实现状态。

---

## 更新（2026-03-26）

当前这份方案已经落地了首版实现：

- 新增 `examples/baselines/diffusion_policy/data_preprocess/data_preprocess.py`
- 新增 `examples/baselines/diffusion_policy/data_preprocess/utils/`
- 已支持：
  - 原始 `h5/json` 读取
  - action 前 6 维全局 min-max 归一化
  - 基于 `agent + extra` 的 state 提取与全局 min-max 归一化
  - mask 生成
  - progress 增广
  - 5:1 train/eval split
  - 将 `action_min/max`、`state_min/max` 等统计量写入 `h5/meta`
  - 新增 `mix0`：
    - 全轨迹保留 `x,y`
    - 额外保留 1 个唯一时间步的前 7 维
    - 再额外保留 3 个其他唯一时间步的前 3 维
    - 不需要额外参数

另外顺手补了两个兼容点：

- `utils/denormalize_utils.py` 现在支持直接从预处理后的 `h5/meta` 读取 `action_min/max`
- `utils/progress_utils.py` 现在对已经是 `(T, 8)` 的 `mas` 幂等，不会重复加 progress

## 更新（2026-04-17）

针对文档末尾“multimask training 时每个训练 batch 是否混合”的问题，结合当前代码实现，结论如下：

- 当前 **mixed preprocess 是按轨迹分配 mask type**，不是按 step 分配。
  - 见 `data_preprocess_mixed.py` 中的 `assign_mask_specs_to_episodes(...)`
  - 先按 ratio 给每条 `source_episode_id` 分配一个 `mask_spec`
  - 每条轨迹在写入 `train/eval.h5` 时只会带一种 `mask_type`

- 当前 **mixed train 的一个 batch 通常是混合的**，不是“一个 mask type 训完再训下一个”。
  - 见 `train_mas_window_mixed.py`
  - 训练集先按 `mask_type_slot` 做一次分层抽样，保证选入的 demo 总体比例接近目标比例
  - 之后 `Dataset` 会把每条轨迹展开成很多 `slice`
  - 再用 `RandomSampler(dataset, replacement=False)` 对这些 slice 随机打乱
  - `BatchSampler` 是直接从这个随机顺序里按 batch size 切块
  - 所以一个 batch 里的样本通常会来自不同轨迹，也通常会来自不同 mask type

- 但要注意：当前实现只是“随机混合”，**不是每个 batch 都严格按 mask ratio 配平**。
  - 某个 batch 可能刚好全是同一种 mask，也可能是多种混合
  - 从长期统计上看会混合，但单个 batch 没有硬约束
  - 另外由于采样单位是 `slice`，轨迹更长的 demo 会贡献更多训练样本，因此训练时实际看到的 mask 占比更接近“按 slice 数加权后的比例”，不完全等于“按轨迹条数”的比例

两种方式的利弊如下。

### 方式 A：当前实现这种“batch 内随机混合”

优点：

- 每个优化 step 都更容易看到多种 mask 条件，梯度更平滑
- 不容易连续很多 step 都只朝某一种 mask 过拟合
- 更接近“共享主干、统一策略”这类多任务训练的常见做法
- 实现简单，直接复用标准 `RandomSampler`

缺点：

- 单个 batch 的 mask 组成不可控，方差较大
- 如果不同 mask 难度差异很大，容易互相干扰
- 更长轨迹会自然占更多 slice，导致实际训练占比和预设 ratio 不完全一致

### 方式 B：按 mask type 分组，依次训练

优点：

- 每个 batch 语义更纯，梯度方向更集中
- 更容易做分组对比、debug 和 curriculum
- 如果某些 mask 需要更高训练强度，可以单独调度

缺点：

- 连续很多 step 只看一种 mask，容易对当前组短期过拟合
- 组与组之间梯度切换更剧烈，训练更不平稳
- 共享策略会更容易出现“刚学会 A 又被 B 冲掉”的现象
- 需要额外实现 grouped sampler / grouped batch sampler，训练管线更复杂

因此，按当前代码和目标来看：

- **现在的实现是“轨迹级分配 + batch 级随机混合”**
- 如果目标是训练一个能统一处理多种 mask 的单策略，这通常是更合适的默认方案
- 如果后面发现某些 mask 明显被压制，再考虑引入“按 mask 分组采样”或“每 batch 强制配比”的 sampler 会更稳妥

当前 smoke test 命令：

```bash
python examples/baselines/diffusion_policy/data_preprocess/data_preprocess.py \
  --input-h5 demos/data_1/data_1.h5 \
  --mask-type mix0 \
  --num-traj 6 \
  --output-dir /tmp/data_1_preprocessed_smoke \
  --overwrite
```

说明：

- 这次实现完成的是本 plan 里的 **Step 1**，并补了少量低风险兼容逻辑
- 训练主脚本还没有全面切到“直接消费预处理后 state / 非 padding MAS”的最终形态，尤其 `only_mas` 路径仍保留旧的固定长度假设

---

## 1. 目标

把目前分散在：

- 原始 demo 读取
- action 归一化
- state 归一化
- mask 生成
- progress 增广
- train/eval split
- 训练入口参数拼装

这些步骤，收拢为一条清晰的数据预处理流水线，输出一套**可直接训练/评估**的数据文件，使训练脚本不再承担数据修补工作。

最终目标是：

1. 训练脚本只接收“已经预处理好”的 `train/eval h5 + json`
2. action 反归一化参数从 `h5/meta` 直接读取，不再依赖单独的 `norm.json`
3. state 在预处理阶段就完成归一化，训练中不再动态归一化
4. MAS 在预处理阶段直接生成好，不再在训练时临时补 padding / 加 progress

---

## 2. 现状约束

先明确几个当前数据事实，后续实现必须和这些事实对齐：

1. 原始输入是：
   - `demos/data_1/data_1.h5`
   - `demos/data_1/data_1.json`

2. 当前原始 `actions` shape 是 `(T, 7)`

3. 当前已有的 `demos/data_1/data_1_norm.json` 维度长度是 `6`
   - 说明现有 action 归一化逻辑实际上是：
   - **只归一化前 6 维连续控制量**
   - 第 7 维保持原值

4. 你希望新生成的 MAS 最终是 `(T, 8)`
   - 前 7 维：mask 后的动作条件
   - 第 8 维：progress

5. 新方案里不再需要“把 MAS padding 到 max_length 再给训练脚本二次处理”这一套旧逻辑

---

## 3. 新的数据产物设计

把预处理后的输出固定成一套目录规范，例如：

```text
demos/data_1_preprocessed/
  data_1_<mask_type>_<param>_train.h5
  data_1_<mask_type>_<param>_train.json
  data_1_<mask_type>_<param>_eval.h5
  data_1_<mask_type>_<param>_eval.json
```

其中：

- 需要 `retain_ratio` 的 mask，如 `3D_points`，文件名带 `<ratio>`
- 需要 `mask_seq_len` 的 mask，如 `2D_partial_trajectory`，文件名带 `_seq<len>`
- 不需要额外参数的 mask，如 `mix0`，文件名直接是 `data_1_mix0_train.h5`

每个输出 `h5` 里保留每条 `traj_*` 的基础结构，但在 `meta` 中额外补全训练需要的统计量。

建议 `meta` 至少包含：

- `action_min`: shape `(6,)`
- `action_max`: shape `(6,)`
- `state_min`: shape `(S,)`
- `state_max`: shape `(S,)`
- `state_dim`
- `action_dim`
- `mas_dim`，固定为 `8`
- `mask_type`
- `retain_ratio` 或其他 mask 参数；若 mask 不需要额外参数，如 `mix0`，则不写这些字段
- `num_episodes`
- `split`
- `source_h5`
- `source_json`

这样训练脚本就能只依赖 `h5` 本身，不必再额外传 `action_norm.json`。

---

## 4. `data_preprocess.py` 的职责拆分

建议把新文件 `examples/baselines/diffusion_policy/data_preprocess/data_preprocess.py` 设计成下面几个阶段。

### 阶段 A：读取原始数据

输入：

- 原始 `h5`
- 原始 `json`

处理：

1. 遍历所有 `traj_*`
2. 读取：
   - `actions`
   - `obs`
   - `success/terminated/truncated`
   - 其他训练仍需要保留的字段
3. 保持原始 episode 顺序，不在这里打乱

输出：

- 内存中的 trajectory 列表
- 与之对齐的 metadata/episode 信息

这里建议先做一层统一校验：

- 每条轨迹都有 `actions`
- `obs["state"]` 长度等于 `T+1`
- action 最后一维是 `7`
- json episode 数量与 h5 trajectory 可对齐

---

### 阶段 B：计算并写入 action 归一化

这一阶段仿照 `sft_data_generation/mask_for_sft.py` 的思路，但目标不是生成额外 `norm.json`，而是把统计量写回输出 `h5/meta`。

处理规则建议明确为：

1. 全数据集统计 action 前 6 维的全局 `min/max`
2. 仅对前 6 维做 min-max 归一化到 `[-1, 1]`
3. 第 7 维保持原值

这样可以与当前已有训练/反归一化逻辑保持一致，避免无谓的动作定义漂移。

输出：

- 每条 trajectory 的 `actions` 都已归一化
- `meta/action_min`
- `meta/action_max`

这里要特别注意：

1. 归一化统计量必须先在**全量原始数据**上计算，再分别写入 train/eval
2. train/eval 两个 split 必须共享同一套 action min/max，不能各算各的

---

### 阶段 C：计算并写入 state 归一化

目标是把 state 归一化前移到预处理阶段。

处理建议：

1. 先从所有 trajectory 的 `obs/state` 中提取训练需要的 state 表达
2. 在**全量原始数据**上统计全局 `state_min/state_max`
3. 对所有输出轨迹中的 state 统一归一化到 `[-1, 1]`

这里需要先明确一个实现选择：

- 如果训练里实际使用的是 `convert_obs + build_state_obs_extractor` 之后的 state
- 那么预处理阶段就应该基于**同一套 extractor 逻辑**生成归一化后的 state

换句话说，不能对原始 `obs` 里所有字段盲目归一化，而应该对“训练真正拼给模型的 state”归一化。

因此这一步 plan 上建议：

1. 复用当前 `build_state_obs_extractor`
2. 在预处理时直接得到最终训练态 `state`
3. 写回到新的输出数据中

这样后面训练脚本可以删除动态 `compute_state_min_max / set_state_normalizer / normalize_state` 相关逻辑。

---

### 阶段 D：生成 MAS

这一阶段负责从归一化后的动作轨迹生成 mask 条件。

处理规则：

1. 输入是每条轨迹的动作序列 `(T, 7)`
2. 按 `mask_type` 与对应参数执行 mask
3. 输出 `mas` shape 为 `(T, 7)`
4. 不再做全局 `max_length` padding

当前补充的 `mix0` 语义固定为：

- 所有 step 保留 `x,y`
- 再随机选 4 个互不重复的时间步
- 第 1 个时间步保留前 7 维
- 其余 3 个时间步保留前 3 维
- 要求轨迹长度 `T >= 4`

这里建议明确两条边界：

1. `mask_type` 对应参数要在预处理入口强校验
   - 如 `retain_ratio`
   - `mask_seq_len`
   - 其他 mask 专属参数

2. mask 逻辑本身尽量复用 `sft_data_generation/mask_for_sft.py` 的规则，避免出现“同名 mask_type 语义不一致”

但这里不建议直接硬拷脚本，而是：

1. 提取一层纯函数
2. 让 `data_preprocess.py` 直接调用

这样后面 train pipeline 和 sft pipeline 的 mask 语义才能统一。

---

### 阶段 E：为 MAS 增加 progress

在得到 `(T, 7)` 的 MAS 后，直接在预处理阶段扩成 `(T, 8)`。

规则建议与当前 `augment_mas_with_progress` 保持一致：

1. progress 只表示该轨迹内部的执行进度
2. progress 范围统一为 `[0, 1]`
3. 每条轨迹按自身长度 `T` 生成

由于新方案不再 padding 到全局 `max_length`，这里 progress 不需要再处理“有效段 + padding 段”两段逻辑，而是直接对真实轨迹长度生成即可。

最终每条 trajectory 的 `mas`：

- shape: `(T, 8)`
- 前 7 维：masked action
- 第 8 维：progress

---

### 阶段 F：train/eval split

这一阶段负责把处理好的轨迹按 5:1 划分。

建议规范如下：

1. split 基于 trajectory 维度，不基于 frame
2. 默认按固定随机种子打乱后再 split
3. 输出：
   - `train.h5/json`
   - `eval.h5/json`

把 split 信息也写入输出 `json` 与 `h5/meta`，保证可追溯。

保存：

- `split_seed`
- `train_indices`
- `eval_indices`

这样后面任何训练结果都能追溯到具体 episode 集合。

---

## 5. 新训练入口应如何简化

预处理完成后，训练入口应该只接收“预处理后数据”，不再接收零散中间产物。

### `run_train_mas_window.sh` 建议保留的核心参数

- `DEMO_PATH`
- `EVAL_DEMO_PATH`
- `EVAL_DEMO_METADATA_PATH`
- `MASK_TYPE`
- 与该 mask 对应的参数，如 `RETAIN_RATIO` / `MASK_SEQ_LEN`

### 建议删除的旧参数

- `ACTION_NORM_PATH`
  - 因为反归一化参数应从 `h5/meta` 直接读取
- TEST_DEMO_PATH
- EVAL_DEMO_METADATA_PATH
- ACTION_NORM_PATH

以及所有只服务于旧预处理逻辑、但在新数据格式中已经冗余的参数。

### 主函数的新调用顺序

建议未来训练主函数统一成：

1. 读取预处理后 `h5`
2. 从 `meta` 读取 action/state 统计量
3. 构建 dataset
4. 训练
5. 评估

不要再在训练主函数里临时做：

- action norm 文件读取
- state min/max 统计
- progress 增广
- MAS padding 修补

这些都应该在预处理阶段完成。

---

## 6. 代码改造顺序 plan

建议实际实施时按下面顺序推进，避免一次改太多。

### Step 1：先落 `data_preprocess.py`

先只做离线预处理脚本，功能包括：

1. 读取原始 `data_1.h5/json`
2. action 归一化
3. state 归一化
4. MAS 生成
5. progress 增广
6. 5:1 split
7. 写出 train/eval h5/json

这一阶段先不动训练代码。

### Step 2：让训练代码先“兼容新数据”

在训练脚本里先支持：

1. 从 `meta` 读取 action min/max
2. 如果 state 已经归一化，则跳过动态 state normalize
3. 直接读取 `(T, 8)` 的 `mas`

这一阶段可以保留对旧数据格式的兼容分支，方便回滚。

### Step 3：删掉旧数据入口逻辑

当新数据路径验证稳定后，再删除：

- 独立 `action_norm.json` 依赖
- 训练中 state 动态归一化
- 训练中 MAS 补 progress
- 训练中 MAS padding 相关逻辑

### Step 4：最后清理 shell 参数

等训练主脚本彻底切到新格式后，再精简 `run_train_mas_window.sh` 参数，避免中途脚本和数据格式不匹配。

---

## 7. 验证 checklist

这个 plan 落地时，建议每一步都做下面这些校验。

### 数据内容校验

1. 输出 `train/eval.h5` 的 trajectory 数量之和等于输入总数
2. `train/eval.json` 的 episode 数量与 h5 对齐
3. 每条轨迹：
   - `actions.shape == (T, 7)`
   - `mas.shape == (T, 8)`
   - `obs/state.shape[0] == T+1`

### 统计量校验

1. `meta/action_min/max` 维度为 `(6,)`
2. action 前 6 维归一化后大致落在 `[-1,1]`
3. state 归一化后大致落在 `[-1,1]`

### mask 校验

1. 不同 `mask_type` 输出符合预期
2. 第 8 维 progress 单调不减
3. progress 首尾值合理

### 训练兼容性校验

1. 新数据能被 dataset 正常读取
2. 不再依赖 `action_norm.json`
3. 训练首个 batch 的 shape 全部正确

---

## 8. 主要风险点

### 风险 1：state 归一化对象不一致

如果预处理阶段归一化的是原始 state，而训练时真正使用的是 extractor 后的 state，会造成 train/eval 不一致。

所以必须统一：

- **以训练真正喂给模型的 state 定义为准**

### 风险 2：mask 逻辑与旧 SFT 语义漂移

如果新脚本自己重写 mask，但细节与 `mask_for_sft.py` 不同，后面实验就无法横向对比。

所以建议：

- mask 规则复用旧实现
- 或至少逐个 `mask_type` 做行为对齐测试

### 风险 3：train/eval split 统计量泄漏

如果 action/state min/max 分别在 train/eval 各自计算，会导致反归一化和归一化口径不一致。

所以建议固定：

- 统计量在 split 前的全量数据上统一计算

### 风险 4：旧代码里仍残留二次归一化

如果预处理已经做完 state/action normalize，但训练时又做一次，就会直接把输入搞坏。

因此未来真正改代码时，要优先清理：

- `load_action_denorm_stats(action_norm_path)`
- `compute_state_min_max`
- `set_state_normalizer`
- `normalize_state`

这些与新数据格式冲突的路径。

---

## 9. 我建议的最终落地顺序

最稳妥的顺序是：

1. 先实现 `data_preprocess.py`
2. 先只生成一小份样例数据做 shape/统计检查
3. 再让 `multitrain_dit.py` 先接新数据
4. 验证纯 DiT 路径稳定
5. 再接入 `mas_window` / `only_mas`
6. 最后统一清理 shell 参数与旧兼容代码

这样改动面最可控，也最容易 debug。

## 下一步指示

在增加multimasktraining的工作流后，数据集进行了混合，但我需要你调研确认，在每个训练batch里是否混合，还是每种类别依次训练？
并且分析两种方式的利弊。
