# Control Error: a new Metric
（我写在这个文档里的东西你不要删除，只按照我的指示增补）
（如果我写的部分有不够合理的部分也可以做修改，但要向我说明）
（每次下达修改命令之后，先不要改代码，先把实现方案整理清楚，标明参数、函数、变量 shape 和关键定义，尽量简练。）

## 背景
我们用 mask action model 将高层控制信号引入低层动作规划，现在需要一个 metric 衡量 rollout 对控制信号的贴合程度。

定义 rollout 轨迹 `p:[T,7]`，`mas:[T_mas,8]`。对每个有效控制点 `k`，若该点有 `d_k` 个已知维度，则
$$
e_k^{raw}=\min_{t\in[1,T]}\|S_kp_t-c_k\|_2,\qquad e_k=\frac{e_k^{raw}}{d_k}
$$
最终
$$
\mathrm{CE}=\frac{1}{K}\sum_{k=1}^{K}e_k
$$
即：先对每个控制点求最小 L2 误差，再除以该点已知维度数，最后对所有有效控制点平均。

## 离线版本算法

离线版本概述：给定训练好的 policy 权重和含 `mas` 的数据集，逐条 rollout 并计算 CE。
1. 读取 `eval_demo_path` 中的 `mas/mask/actions`
2. 加载 `checkpoint_pt_path` 的 policy 权重和 STPM encoder
3. 按 `train_mas_window.py` / `evaluate_mas_window.py` 的方式 rollout
4. 记录每条轨迹的 `success_once`、`success_at_end`、`rollout_traj:[T,7]`
5. 用 `rollout_traj` 和 `mas/mask` 计算单条 `ce_traj`
6. 分别统计 `ce_all`、`ce_success`、`ce_failed`

详细算法设计方案：
### 0. 固定实现定义
为避免语义漂移，固定如下：
1. rollout 轨迹 `p` 定义为 rollout 时实际送入 `env.step()` 的 7 维动作序列，不是 `tcp_pose`
2. “有效控制点”按 `mask` 的行定义：若 `mask[k,:7]` 至少有一个 `1`，则第 `k` 行为一个控制点

另外统一规定：
- `mas[:,7]` 是 `progress`，不参与 CE
- CE 比较空间统一为归一化后的控制空间
- 成功/失败划分使用 `success_once`
- `success_at_end` 只记录，不参与 `ce_success / ce_failed`
- 点级误差先除以该点已知维度数 `d_k`

### 1. 输入、输出、文件
建议新增：
- `examples/baselines/diffusion_policy/evaluate_control_error_mas_window.py`
- `examples/baselines/diffusion_policy/run_evaluate_control_error_mas_window.sh`

输入参数建议：
- 数据：`eval_demo_path`、`eval_demo_metadata_path`、`num_eval_demos`
- 权重：`checkpoint_pt_path`、`stpm_ckpt_path`、`stpm_config_path`
- 环境：`env_id`、`control_mode`、`max_episode_steps`、`num_eval_envs`、`sim_backend`
- 模型：`obs_horizon`、`act_horizon`、`pred_horizon`、`long_window_horizon`、`short_window_horizon`、`mas_long_encode_mode`、`mas_long_conv_output_dim`
- 输出：`output_json_path`、可选 `save_per_traj`

说明：
- 当前 checkpoint 大概率只有 `state_dict`，结构超参先在 `.sh` 中显式传入
- 动作归一化参数优先从数据集 meta 读取 `action_min/action_max`

输出 json 至少包含：`ce_all`、`ce_success`、`ce_failed`、`num_total_valid_ce`、`num_success_valid_ce`、`num_failed_valid_ce`、`num_success`、`num_failed`、`per_traj_results`。

### 2. 关键变量与 shape
对第 `i` 条 demo：
- `mas_i:(T_mas_i,8)`
- `mask_i_raw:(T_mas_i,7)`
- `mask_i:(T_mas_i,8)`，由 mask augment 得到，但 CE 只看前 7 维
- `executed_actions_denorm_i:(T_exec_i,7)`
- `executed_actions_norm_i:(T_exec_i,7)`

对单个控制点 `k`：
- `valid_dims_k = np.where(mask_i[k,:7] > 0)[0]`，shape `(d_k,)`
- `d_k = len(valid_dims_k)`
- `target_k = mas_i[k, valid_dims_k]`，shape `(d_k,)`
- `pred_tk = executed_actions_norm_i[t, valid_dims_k]`，shape `(d_k,)`

辅助量：
- `known_idx_i = [k | sum(mask_i[k,:7]) > 0]`
- `K_i = len(known_idx_i)`

### 3. rollout 轨迹 `p` 的定义
固定
$$
p_t := executed\_actions\_norm_i[t]\in\mathbb{R}^7
$$
即 `p_t` 是 rollout 实际执行动作重新归一化后的 7 维控制向量。

归一化规则：
- 前 6 维：`2 * (a - action_min) / (action_max - action_min) - 1`
- 第 7 维 gripper：若数据本身已在 `[-1,1]`，则直接保留

### 4. CE 实现公式
对每个有效控制点 `k`：
- `D_k = valid_dims_k`
- `c_k = mas_i[k, D_k]`
- 实现时不显式构造选择矩阵 `S_k`，直接用索引 `D_k`

点级原始误差：
$$
ce_k^{raw}=\min_{t\in[1,T_{exec}]}\|p_t[D_k]-c_k\|_2
$$
点级归一化误差：
$$
ce_k=\frac{ce_k^{raw}}{d_k}
$$
单轨迹误差：
$$
ce_{traj}=\frac{1}{K_i}\sum_{k\in known\_idx_i} ce_k
$$

补充记录建议：`best_match_t_k = argmin_t ||p_t[D_k]-c_k||_2`，可选记录 `ce_raw_k`、`d_k`。

边界情况：
- 若 `K_i == 0`，则 `ce_traj = NaN`
- 该轨迹不参与聚合平均

### 5. 整体流程
#### Step 1. 加载 eval 数据
从 `eval_demo_path` 读取 `mas`、`mask`、`actions`（用于 `traj_len`）。建议参考：`load_eval_mas_window_data`、`infer_eval_reset_seeds_from_demo`、`infer_eval_traj_ids_from_demo`、`subset_eval_data`。原始数据通常是 `mas:(T,8)`、`mask:(T,7)`；计算 CE 前先 augment mask，但实际只用 `mask[:,:7]`。

#### Step 2. 构建 agent 和 env
复用 `train_mas_window.py` 中的 `build_eval_stpm_encoder`、`make_eval_envs`、`Agent`、`set_action_denormalizer`。加载 checkpoint 时优先 `ema_agent`，没有再回退 `agent`。

#### Step 3. rollout 并记录执行动作
参照 `evaluate/evaluate_mas_window.py`，但额外记录：
- `demo_local_idx`
- `success_once`
- `success_at_end`
- `executed_actions_denorm`
- `episode_len`

注意：若一个 action chunk 中途 episode 提前结束，只记录真正执行到 `env.step()` 的前 `executed_steps` 个动作。

#### Step 4. 动作重新归一化
从数据集 meta 读取 `action_min/action_max`，将 `executed_actions_denorm_i` 转为 `executed_actions_norm_i`，保证与 `mas_i[:,:7]` 在同一空间比较。

#### Step 5. 计算单轨迹 CE
流程：
1. 取 `known_idx_i`
2. 对每个 `k` 取 `valid_dims_k`
3. 在所有 rollout 时刻 `t` 上计算 L2 距离
4. 取最小值得到 `ce_raw_k`
5. 用 `d_k` 归一化得到 `ce_k`
6. 对所有 `ce_k` 取平均得到 `ce_traj`

单轨迹输出建议包含：`demo_local_idx`、`source_episode_id`、`success_once`、`success_at_end`、`episode_len`、`num_known_points`、`ce_traj`，可选 `best_match_t_list`。

#### Step 6. 聚合
将所有有效 `ce_traj` 分三组：
- 全部：`ce_all`
- `success_once=True`：`ce_success`
- `success_once=False`：`ce_failed`

并输出：`num_total_valid_ce`、`num_success_valid_ce`、`num_failed_valid_ce`。若某组为空，则均值记为 `NaN`，不要记成 `0`。

### 6. 建议函数拆分
- `load_ce_eval_data(...)`：加载 `mas/mask/traj_len/reset_seed/traj_id`
- `build_ce_agent_and_env(...)`：建 env、建 STPM、建 agent、加载 checkpoint
- `rollout_collect_executed_actions(...)`：rollout 并记录动作与 success
- `renormalize_executed_actions(actions_denorm, action_min, action_max)`：`(T,7)->(T,7)`
- `compute_control_error_for_one_traj(actions_norm, mas_t, mask_t)`：计算单轨迹 CE
- `aggregate_control_error(per_traj_results)`：生成 `ce_all / ce_success / ce_failed`
- `save_control_error_json(...)`：保存结果

### 7. 必过 sanity check
- 若直接用 demo 自己的 `normalized_actions` 作为 `p_t`，再和同轨迹 `mas/mask` 计算 CE，则 `ce_traj` 应非常接近 `0`

若不接近 0，优先检查：
- 是否错误把 `progress` 算进 CE
- 是否错误使用 `mask[:,7]`
- 是否把控制点按列而不是按行处理
- 是否对 rollout 动作重复做了归一化/反归一化
- 是否漏掉 `ce_k = ce_raw_k / d_k`

## 在线版本算法（plan）

### 0. 目标与约束
目标：把 CE 融入训练时的每轮 eval pipeline，在 `train_mas_window.py` 和 `train_mas_window_test.py` 中，每次 eval 都输出：
- `ce_all`
- `ce_success`
- `ce_failed`

并额外保存一张 `iter_ce` 折线图到对应 `runs/<run_name>/...` 目录下。

约束：
- online CE 的定义必须与上面的离线版本完全一致
- 尽量复用已有函数，避免在 train 脚本里重复写 CE 细节
- best checkpoint 逻辑先保持不变，仍按 `success_once / success_at_end` 保存
- 你后面写的“根据 mas 的 K 个点列逐一计算 ce”，这里仍按前文已固定定义实现：`K` 指有效时间步行，不按列算

### 1. 建议改动文件
- `examples/baselines/diffusion_policy/train_mas_window.py`
- `examples/baselines/diffusion_policy/train_mas_window_test.py`
- `examples/baselines/diffusion_policy/evaluate/evaluate_mas_window.py`
- `examples/baselines/diffusion_policy/evaluate/evaluate_mas_window_test.py`
- `examples/baselines/diffusion_policy/utils/control_error_utils.py`
- `examples/baselines/diffusion_policy/utils/draw_p_t_curve_utils.py`
- 对应 `run_train_mas_window*.sh`

### 2. 训练侧输入、输出与关键变量
训练脚本在每轮 eval 时已有：
- `eval_mam_data`
- `eval_reset_seeds`
- `denorm_mins / denorm_maxs`
- `ema_agent`

建议把训练侧 eval 数据加载从 `load_eval_mas_window_data(...)` 改为 `load_ce_eval_data(...)`，这样 `eval_mam_data` 除已有字段外，再多：
- `demo_local_indices: list[int]`
- `source_episode_ids: list[int]`

这样 `subset_eval_data(...)` 后，batch 内也能保留稳定的 traj 标识，便于 CE 聚合与 debug。

在线 CE 计算时的关键变量：
- `mas_i:(T_mas_i,8)`
- `mask_i:(T_mas_i,8)`
- `executed_actions_denorm_i:(T_exec_i,7)`
- `executed_actions_norm_i:(T_exec_i,7)`
- `rollout_records: list[dict]`，长度为当前 eval 完成的轨迹数
- `per_traj_ce_results: list[dict]`

### 3. evaluate_mas_window 的接口改动
给 `evaluate/evaluate_mas_window.py` 和 `_test.py` 都加一个新参数：
- `return_rollout_records: bool = False`

新增后保持兼容：
- 默认 `False` 时，返回值行为与现在完全一致
- `return_rollout_records=True, return_progress_curves=False` 时，返回 `eval_metrics, rollout_records`
- 两个都为 `True` 时，返回 `eval_metrics, progress_curve_records, rollout_records`

`rollout_records` 中每条轨迹建议记录：
- `demo_local_idx:int`
- `source_episode_id:int`
- `success_once:bool`
- `success_at_end:bool`
- `episode_len:int`
- `executed_actions_denorm:(T_exec,7)`

### 4. evaluate 阶段如何收集 rollout action
这部分直接复用离线 CE evaluator 的做法。

在 `evaluate_mas_window(...)` 内：
- 每个 env 维护 `executed_action_chunks[env_idx]`
- 每次 `agent.get_action(obs)` 得到 `action_seq:(B, act_horizon, 7)`
- 若本 chunk 中只真正执行了前 `executed_steps` 步，则只追加 `action_seq[env_idx, :executed_steps]`
- episode 结束时，对该 env 的 chunk 做 `concatenate`，得到 `executed_actions_denorm_i:(T_exec_i,7)`

注意：
- 这里记录的必须是实际执行到 `env.step()` 的动作
- 不能把 chunk 里未执行的尾部动作算进 CE

### 5. CE 计算建议抽成共享 helper
当前 `utils/control_error_utils.py` 已经有：
- `load_ce_eval_data`
- `normalize_rollout_actions`
- `compute_control_error_for_traj`
- `aggregate_control_error`

为了避免 train 脚本里重复写“逐轨迹拼接 CE 结果”的逻辑，把离线脚本 `evaluate_control_error_mas_window.py` 里的 `_compute_control_error_results(...)` 提升到 `utils/control_error_utils.py`，变成共享函数，例如：
- `compute_control_error_results_from_rollouts(eval_data, rollout_records, action_min, action_max, save_per_traj=False)`

输入：
- `eval_data`
- `rollout_records`
- `action_min/action_max`

输出：
- `per_traj_ce_results: list[dict]`

每条结果至少包含：
- `demo_local_idx`
- `source_episode_id`
- `success_once`
- `success_at_end`
- `episode_len`
- `executed_steps`
- `ce_traj`
- `num_known_points`

### 6. train_mas_window.py 中的 online CE 流程
在 `evaluate_and_save_best(iteration)` 中：

1. 先像现在一样跑 `evaluate_mas_window(...)`
2. 额外要求 `return_rollout_records=True`
3. 拿到 `rollout_records` 后，调用共享 helper 生成 `per_traj_ce_results`
4. 用 `aggregate_control_error(...)` 得到：
   - `ce_all`
   - `ce_success`
   - `ce_failed`
5. 将这三个值：
   - `print(...)`
   - `writer.add_scalar("eval/ce_all", ..., iteration)`
   - `writer.add_scalar("eval/ce_success", ..., iteration)`
   - `writer.add_scalar("eval/ce_failed", ..., iteration)`

这里直接复用训练脚本前面已经加载好的 `denorm_mins / denorm_maxs`，不要再额外读一次文件。

若当前 eval 是分 batch 跑的，则做法是：
- 每个 batch 先得到 `batch_rollout_records`
- 当场算出 `batch_per_traj_ce_results`
- 最后把所有 batch 的 `per_traj_ce_results` 合并，再统一 `aggregate`

这样不会打乱现有 `eval_metrics` 的聚合逻辑。

### 7. train_mas_window_test.py 中的 online CE 流程
`train_mas_window_test.py` 与主训练脚本同一套逻辑，只是它当前 eval 基本是单条 demo 顺序跑。

因此直接做同样改动即可：
- `evaluate_mas_window(..., return_rollout_records=True)`
- 每条 demo eval 后取回 `rollout_records`
- 累积 `per_traj_ce_results`
- 最后统一算 `ce_all / ce_success / ce_failed`
- 同样写入 `writer` 并打印

### 8. iter_ce 折线图
建议不要把画图逻辑塞进 train 主循环里，改为在 `utils/draw_p_t_curve_utils.py` 增加新函数，例如：
- `save_control_error_curve(run_dir, iterations, ce_all, ce_success, ce_failed)`

输入 shape：
- `iterations:(N,)`
- `ce_all:(N,)`
- `ce_success:(N,)`
- `ce_failed:(N,)`

输出：
- 保存到 `runs/<run_name>/graphs/iter_ce.png`

训练脚本中维护 4 个 history：
- `ce_curve_iters:list[int]`
- `ce_curve_all:list[float]`
- `ce_curve_success:list[float]`
- `ce_curve_failed:list[float]`

每次 eval 完后 append 一次，并覆盖保存一次 png 即可。若某次某组为空，则该点允许是 `NaN`，画图时保留断点即可，不要强行补 `0`。

### 9. 对应 sh 文件
优先方案：不新增 CE 相关超参，online CE 默认在每轮 eval 自动计算。

这样对应 `.sh` 文件只需要确认已有参数继续正确传入：
- `--test-demo-path`
- `--eval-demo-metadata-path`
- `--action-norm-path`
- `--stpm-ckpt-path`
- `--stpm-config-path`

若后续你希望 CE 可手动开关，再额外加：
- `--enable-control-error`
- `--save-control-error-curve`

但当前这版 plan 里不建议先加，先把默认流程跑通。

### 10. online 版本必须过的 sanity check
1. 每轮 eval 后，`len(rollout_records)` 应等于本轮真正完成的 eval traj 数
2. 每条 `executed_actions_denorm` 都应满足 shape `(T_exec,7)`
3. `ce_success` 只由 `success_once=True` 的轨迹聚合
4. `ce_failed` 只由 `success_once=False` 的轨迹聚合
5. 若某条轨迹 `num_known_points == 0`，则该轨迹 `ce_traj=NaN`，但不参与均值
6. `writer` 中能看到 `eval/ce_all`、`eval/ce_success`、`eval/ce_failed`
7. `runs/<run_name>/graphs/iter_ce.png` 会随着 eval 迭代不断更新

### 11. 实现顺序建议
1. 先扩展 `evaluate_mas_window.py` / `_test.py`，把 `rollout_records` 返回打通
2. 再把共享 CE 汇总 helper 抽到 `utils/control_error_utils.py`
3. 然后接入 `train_mas_window.py`
4. 再同步接入 `train_mas_window_test.py`
5. 最后补 `iter_ce` 画图与 `.sh` 检查


## 进一步指示

接下来我需要将CE的计算融合到training的pipeline中。

（同时改动trainmaswindow 和 trainmaswindowtest，以及他们对应的sh文件）

具体实现如下：

- 每轮eval时，收集对应的mas tensor和rollout action tensor（然后进行归一化），并且收集sucess（once）和failed数据

- 按照之前的流程根据mas的K个点列逐一计算ce

- 最后将计算的ce取平均后print出来（3个值），同时做一个iter_ce的折线图，保存在runs文件夹里

- 尽量调用之前utils里已经有的函数，避免重复性工作

把plan写到## 在线版本算法（plan），要求和之前一样
