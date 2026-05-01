# diffusion_policy 说明

该目录实现三条 Diffusion Policy 训练/评估链路，其中核心模型是 MAM（Masked Action Model）。

## 1. Baseline DP

- 启动脚本：`run_baseline.sh`
- 训练入口：`train_baseline.py`
- 评估入口：`evaluate/evaluate_baseline.py`
- 数据输入：原始 ManiSkill demo，或由 `--raw-demo-h5/--raw-demo-json` 自动切成 train/eval。
- 模型：RGB/RGBD observation encoder + `DiTNoiseNet` 或 `ConditionalUnet1D`，直接预测 action chunk。
- 关键设计：只对动作前 6 维做 min/max 归一化，第 7 维 gripper 保持原值；eval 会从 demo metadata 读取 reset seed，保证评估 episode 对齐。

## 2. Subgoal Condition DP

- 启动脚本：`run_subgoal_condition.sh`
- 预处理入口：`data_preprocess/data_preprocess.py`
- 训练入口：`train_subgoal_condition.py`
- 评估入口：`evaluate/evaluate_subgoal_condition.py`
- 数据输入：单一 mask 类型的预处理数据，包含 `actions`、`mas`、`mask` 和归一化后的 `obs/state`。
- 模型：Baseline DP 加一个 flatten 后的 subgoal/MAS 条件。
- 关键设计：`mas` 是 masked action sequence；训练时把整段 MAS padding 到固定长度后拼入 observation condition；eval 时按 demo 顺序把对应 subgoal 附加到每个 env obs。

## 3. MAM / MAS Window DP

- 启动脚本：`run_mam.sh`
- 预处理入口：`data_preprocess/data_preprocess.py` 或 `data_preprocess/data_preprocess_mixed.py`
- 训练入口：`train_mam.py`
- 在线评估入口：`evaluate/evaluate_mam.py`
- 离线控制误差评估：`eval_ce.py`
- RePaint/inpainting 评估：`eval_inpaint.py`
- 数据输入：带 `mas`/`mask` 的预处理数据；支持单 mask、mixed composition、one-demo-multi-mask。
- 模型：DP 主干 + MAS long/short window encoder。long window 可用 2D/1D conv，short window 和当前 mask/动作信息一起作为条件。

## MAM 核心设计

MAM 的目标是在标准 Diffusion Policy 的 observation condition 之外，引入一段“不完整但有结构的动作提示”：

- `actions`：完整专家动作，是 diffusion 训练目标。
- `mas`：Masked Action Sequence，由完整动作按某种 mask 策略遮挡后得到。
- `mask`：标记 MAS 中哪些动作维度是已知控制点。
- `progress`：MAS 的第 8 维，表示轨迹进度，范围约为 `[0, 1]`。

训练时，MAM 不直接把整条 MAS 全量 flatten 后塞给模型，而是围绕当前 step 构造局部条件：

- long window：较长时间范围的 MAS 片段，用 `MasConv` 或 `MasConv1D` 编码成紧凑特征。
- short window：更短的 MAS 片段，保留较细粒度的近期/未来控制提示。
- mask window：与 MAS window 同步输入，让模型区分真实控制点和被 mask 的填充值。
- obs condition：最终 condition 为 `visual_feature + state + mas_long_feature + mas_short_feature`，送入 `DiTNoiseNet` 预测动作噪声。

评估时没有专家当前 step，因此 MAM 用 STPM 从 rollout history 估计当前 progress，再根据 progress 在 eval demo 的 MAS 中找对应位置，在线构造 long/short window condition。这使模型可以在执行过程中持续对齐“当前执行进度”和“参考 masked action sequence”。

mask 设计支持三类实验：

- 单一 mask：一组 demo 使用同一种 mask 类型和比例。
- mixed composition：不同 demo 按比例分配不同 mask 类型。
- one-demo-multi-mask：同一条源 demo 复制成多个 mask slot，便于比较不同 mask 条件。

损失默认是普通 diffusion noise MSE；当 `loss_mode=weighted` 时，`loss_mask_area_weight` 会调整 masked/unmasked 区域权重，用来控制模型更关注已知控制区域还是补全区域。

整体上，MAM 可以理解为：**用 MAS 提供稀疏动作先验，用 progress 对齐在线执行位置，用 Diffusion Policy 补全可执行的动作 chunk。**

## 目录结构

- `data_preprocess/`：demo split、action/state normalization、mask/MAS 生成。
- `models/`：DP 主干和 MAS encoder。
- `evaluate/`：三条链路的在线 rollout 评估。
- `utils/`：共享工具。
  - `split_eval_utils.py`：raw demo split、action norm 读写、eval reset kwargs。
  - `progress_utils.py`：MAS progress 增广、padding、long/short window 构造。
  - `load_train_data_utils.py` / `load_eval_data_utils.py`：HDF5 数据读取与 eval 子集构造。
  - `stpm_utils.py`：STPM progress 预测和在线 MAS condition。
  - `control_error_utils.py` / `inpainting_utils.py` / `video_utils.py`：专项评估与可视化辅助。

## 数据约定

- HDF5 轨迹 key 使用 `traj_0 ... traj_N`。
- 预处理 HDF5 的 `meta` 里保存 `action_min/max`、`state_min/max`、`source_episode_ids`、mask 配置和归一化标记。
- `actions` 是训练目标，`mas` 是 masked action sequence + progress，`mask` 表示哪些动作维度为已知控制点。
- eval JSON metadata 保存 reset 信息；训练脚本会用它还原评估 seed。

## 维护约定

- 新增通用 split/eval reset 逻辑放到 `utils/split_eval_utils.py`。
- 新增 MAS progress/window 逻辑放到 `utils/progress_utils.py`。
- 只保留被入口脚本或训练/评估代码引用的模型与工具；实验草稿应放到目录外的计划文档或临时目录。
