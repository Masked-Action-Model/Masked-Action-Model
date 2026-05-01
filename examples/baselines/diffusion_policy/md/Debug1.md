# only-mas pipeline

documents：
- `examples/baselines/diffusion_policy/train_only_mas.py`
- `examples/baselines/diffusion_policy/train_only_mas_test.py`
- `examples/baselines/diffusion_policy/evaluate/evaluate_only_mas.py`
- `examples/baselines/diffusion_policy/evaluate/evaluate_only_mas_test.py`
- `examples/baselines/diffusion_policy/run_train_only_mas.sh`
- `examples/baselines/diffusion_policy/run_train_only_mas_test.sh`
- `examples/baselines/diffusion_policy/models/*`
- `examples/baselines/diffusion_policy/utils/*`
- `STPM/models/stpm_encoder.py`

## 1. overview

`train_only_mas.py` trains a Diffusion Policy with `DiTNoiseNet`。  
在常规视觉和状态条件之外，额外接收两条条件分支：
1. `mas`
   demo masked action sequence，作为“全局演示”
2. `progress`有 3 种注入方式：
   - `MLP8`
   - `MAS8`
   - `AP8`

## 2. 入口

### 2.1 训练入口
`run_train_only_mas.sh` ：
```bash
python examples/baselines/diffusion_policy/train_only_mas.py ...
```
- 训练 demo：`demo_path`
- 评估 demo：`test_demo_path`
- STPM checkpoint/config：用于评估时在线预测 progress
- action norm json：用于 rollout 时把动作从 `[-1, 1]` 反归一化
- `progress_mode`：控制 progress 如何进入 DiT
- `mas_encode_mode`：控制整条 `mas` 如何编码

### 2.2 test 入口
逻辑与正式版相同，只是默认：
- `total_iters` 很小
- `num_demos / num_eval_demos / num_eval_episodes` 更小
- 更适合快速检查数据流、shape 和 overfit 行为

## 3. 训练前的准备

主入口 `train_only_mas.py` 在 `__main__` 中依次完成：
1. 解析参数 `Args`
2. 设置随机种子
3. 从评估 demo metadata 推断 reset seed
4. 读取动作反归一化参数
5. 构建 `STPMEncoder`
6. 构建评估环境 `envs`
7. 构建训练数据集 `SmallDemoDataset_DiffusionPolicy`
8. 读取评估侧 only-mas 条件数据 `eval_mam_data`
9. 创建 `Agent`
10. 创建优化器、lr scheduler、EMA
11. 进入训练循环，并定期调用 `evaluate_only_mas`

当前和数据加载/归一化相关的 helper 已按用途拆开in utils


## 4. 训练数据是如何构造的

核心类：`SmallDemoDataset_DiffusionPolicy`
### 4.1 原始数据读取
通过 `load_demo_dataset(..., concat=False)` 逐轨迹读取。
读取内容：
- `observations`
- `actions`
- `mas`
- `env_states`
- `success`
- `terminated`
- `truncated`
其中每条轨迹通常满足：
- `observations["state"]` 长度是 `L+1`
- `actions` 形状是 `(L, A)`
- 原始 `mas` 形状是 `(T, 7)`
记号：

- `B`：batch size
- `H`：`obs_horizon`
- `Ah`：`act_horizon`
- `P`：`pred_horizon`
- `A`：动作维度
- `T`：MAS 最大时间长度，也等于 dataset 里的 `max_length`
- `S`：state 维度

### 4.2 观测预处理
训练集里的 observation 会先经 `convert_obs(...)` 变成：
- `rgb`: `(L+1, 3*k, H_img, W_img)`，后续转成 tensor
- `depth`: `(L+1, k, H_img, W_img)`，后续转成 tensor
- `state`: `(L+1, S)`

### 4.3 给 MAS 增加 progress 列
关键函数：
- `build_progress_column(traj_len, mas_len, device) -> (mas_len, 1)`
- `augment_mas_with_progress(mas_t, traj_len) -> (T, 8)`
这两个函数现在单独位于：
- `examples/baselines/diffusion_policy/utils/progress_utils.py`
逻辑：
1. 用 `traj_len=L` 在 `[0, 1]` 之间线性生成 progress
2. 如果 `traj_len < T`，剩余 padding 部分全部补成 `1.0`
3. 把这列 progress 拼到原始 `mas` 最后一维
(增强后 `mas_t`: `(T, 8)`)

### 4.4 样本切片方式
`self.slices` 里每个元素是：
```python
(traj_idx, start, start + pred_horizon)
```
每个训练样本都会返回：
- `observations`: 一个长度为 `H` 的观测窗口
- `actions`: 一个长度为 `P` 的动作窗口
padding 规则：
- trajectory 开头不够时，观测和动作会向前 pad
- state/action/rgbd 都重复首帧或首动作
- trajectory 结尾不够时，动作用最后一个动作重复

### 4.5 `__getitem__` 返回的主要字段与 shape
对单个样本：
- `obs_seq["rgb"]`: `(H, 3*k, H_img, W_img)`
- `obs_seq["depth"]`: `(H, k, H_img, W_img)`
- `obs_seq["state"]`: `(H, S)`
- `obs_seq["mas"]`: `(H, T*8)`
- `obs_seq["progress"]`: 取决于 `progress_mode`
- `act_seq`: `(P, A)`

其中 `obs_seq["mas"]` 的生成方式固定不变：
1. 取增强后的整条 `mas_traj`，shape `(T, 8)` and flatten 成 `(T*8,)`
2. 沿时间维重复 `H` 次 and 得到 `(H, T*8)`

## 5. progress 的 3 种注入方式 (while training)

### Dataset:

#### `MLP8`
`build_progress_mlp_horizon_window`
从 `mas_t (T, 8)` 最后一列取 progress，围绕当前时间步 `t` 截出一个观测窗口：
- 输入：`mas_t`, `current_step=t`, `obs_horizon=H`
- 输出：`(H, 1)` (padding repeat)

#### `MAS8`
`build_mas_horizon_window`
直接从增强后的 `mas_t (T, 8)` 中截取局部 MAS window：
- 输出 shape：`(H, 8)` (padding repeat)

这里放进 `obs["progress"]` 的虽然名字叫 progress，但内容其实已经不是标量进度，而是“局部 MAS 八维向量窗口”。

#### `AP8`
`build_action_progress_window`
for `obs_horizon` 里的每个 anchor 分别做一次 future rollout：
1. 从 `mas_t[:, -1]` 读取 progress 列
2. 先确定观测窗口 anchor(开头越界的 anchor 会钳到第 `0` 帧)：
3. 对每个 anchor，各自向前取 `act_horizon` 个 progress(越界补 `1.0`)
4. 最终直接得到 `(H, 8)`
```text
[
  [p_{t-H+1}, ..., p_{t-H+8}],
  ...,
  [p_t, ..., p_{t+7}]
]
```
- `obs["progress"]`: `(H, 8)`

### `Agent.encode_progress` 中：

#### `MLP8`
1. 输入 `raw_progress`: `(B, H, 1)`
2. reshape 成 `(B*H, 1)`
3. 送入 `ProgressMLP(in_dim=1, out_dim=8)`
4. 输出 reshape 回 `(B, H, 8)`
#### not MLP8
- `progress_mode != "MLP8"`，则不经过 `ProgressMLP`
- 直接原样返回


## 6. Agent 把条件编码后送进 DiT

核心类：`Agent`

### 6.1 视觉分支

`obs_conditioning` 中:
normalize:
- `rgb`: `(B, H, 3*k, H_img, W_img)`，归一化到 `[0,1]`
- `depth`: `(B, H, k, H_img, W_img)`，除以 `1024`
concat:
- `img_seq`: `(B, H, 4*k, H_img, W_img)`
flatten:
- `(B*H, 4*k, H_img, W_img)`
`PlainConv(out_dim=256)`：
- `visual_feature`: `(B*H, 256)`
reshape：
- `(B, H, 256)`

### 6.2 state 分支

`state` normalize 到 `[-1, 1]`：
- 输入：`(B, H, S)`
- 输出：`(B, H, S)`

### 6.3 mas 分支

- `obs["mas"]`: `(B, H, T*8)`
分三种编码方式：

#### `no_encode`
直接把 `-1` 替换成 `0`：
- 输出：`(B, H, T*8)`

#### `2D`
先把 `(B, H, T*8)` reshape 成：
- `mas_value`: `(B, H, T, 8)`
- `mas_mask`: `(B, H, T, 8)`
再调维并堆叠 value/mask，形成：
- `(B*H, 2, 8, T)`
送入 `MasConv(out_dim=mas_encode_output_dim)`，输出：
- `(B*H, D_mas)`再 reshape：`(B, H, D_mas)`

#### `1D`
前面 reshape 一样，但 `MasConv1D` 会把每个 MAS 维度当成独立通道组做时序卷积，输出：
- `(B, H, D_mas)`

### 6.4 progress 分支

由上一节的 `MLP8 / MAS8 / AP8` 决定，最终统一输出：
- `progress_feature`: `(B, H, 8)`

### 6.5 最终 observation condition

`obs_conditioning` 最后拼接：
```text
[visual_feature, state, mas_feature, progress_feature]
```
- `obs_cond`: `(B, H, obs_dim)`

```text
obs_dim = 256 + S + mas_feature_dim + 8
```

## 7. DiTNoiseNet 内部的数据流

核心类：`DiTNoiseNet`
输入：
- `noise_actions`: `(B, P, A)`
- `timesteps`: `(B,)`
- `obs_enc`: `(B, H, obs_dim)`

### 7.1 encoder
`forward_enc(obs_enc)`：
1. `obs_proj` 把 `obs_dim -> hidden_dim`
2. 得到 `(B, H, hidden_dim)`
3. transpose 成 `(H, B, hidden_dim)`
4. 加位置编码
5. 送入 Transformer encoder

输出是多层 cache：
- `enc_cache`: `list[L]`
- 每层元素 shape：`(H, B, hidden_dim)`

### 7.2 decoder
`forward_dec(noise_actions, time, enc_cache)`：
1. `noise_actions` 经过 `ac_proj`
2. 从 `(B, P, A)` 变成 `(B, P, hidden_dim)`
3. transpose 成 `(P, B, hidden_dim)`
4. 加 decoder 位置参数 `dec_pos`
5. diffusion timestep 经 `time_net` 编成 `(B, hidden_dim)`
6. decoder 每层使用对应 encoder cache 作为条件
7. `eps_out` 输出预测噪声

最终输出：
- `noise_pred`: `(B, P, A)`

## 8. 训练流程

### 8.1 dataloader 输出
每个 batch：
- `data_batch["observations"]`
  - `rgb`: `(B, H, 3*k, H_img, W_img)`
  - `depth`: `(B, H, k, H_img, W_img)`
  - `state`: `(B, H, S)`
  - `mas`: `(B, H, T*8)`
  - `progress`: `(B, H, 8)`
- `data_batch["actions"]`: `(B, P, A)`

### 8.2 `Agent.compute_loss`

(标准 diffusion 的噪声回归损失)
步骤如下：
1. `obs_conditioning(obs_seq)` 得到 `obs_cond: (B, H, obs_dim)`
2. 采样高斯噪声 `noise: (B, P, A)`
3. 随机采样扩散步 `timesteps: (B,)`
4. 用 `noise_scheduler.add_noise(action_seq, noise, timesteps)` 得到
   `noisy_action_seq: (B, P, A)`
5. `noise_pred = noise_pred_net(noisy_action_seq, timesteps, obs_cond)`
6. 损失为：
```python
F.mse_loss(noise_pred, noise)
```

### 8.3 参数更新

每轮训练：
1. `optimizer.zero_grad()`
2. `loss.backward()`
3. `optimizer.step()`
4. `lr_scheduler.step()`
5. `ema.step(agent.parameters())`


## 9. Inference: 动作生成

Agent.get_action(...):
- 输入当前 rollout observation 
- 输出执行 action chunk

### 9.1 加载动作反归一化参数

在进入评估循环前，主函数先做：
```python
denorm_mins, denorm_maxs = load_action_denorm_stats(args.action_norm_path)
agent.set_action_denormalizer(denorm_mins, denorm_maxs, device)
ema_agent.set_action_denormalizer(denorm_mins, denorm_maxs, device)
```
这一步现在是强制要求：

- only-mas 评估必须先成功加载并设置 denormalizer
- 如果 `get_action(...)` 发现 `action_denorm_min/max` 或 `action_denorm_dims` 没有设置好，会直接报错终止

映射公式：
```text
real_action = min + 0.5 * (norm_action + 1) * (max - min)
```

### 9.2 `get_action(...)` input

评估循环在每一轮调用 `agent.get_action(obs)` 时，`obs` 里已经包含：
- `state`: `(B, H, S)`
- `rgb`: `(B, H, H_img, W_img, 3)`
- `depth`: `(B, H, H_img, W_img, 1)`
- `mas`: `(B, H, T*8)`
- `progress`: 取决于 `progress_mode`
其中：
- `B` = 当前并行 env 数
- `H` = `obs_horizon`
- `S` = state dim
- `T` = demo MAS 最大长度
- `P` = `pred_horizon`
- `Ah` = `act_horizon`
- `A` = action dim

### 9.3 `get_action(...)` ouput

1. 改image shape
   - `rgb -> (B, H, 3, H_img, W_img)`
   - `depth -> (B, H, 1, H_img, W_img)`
2. 调用：
   `obs_cond = obs_conditioning(obs_seq, eval_mode=True)`:
   拼成：
   - `obs_cond: (B, H, obs_dim)`
3. 初始化高斯噪声动作：
   - `noisy_action_seq ~ N(0, I)`
   - shape `(B, P, A)`
4. 对 diffusion scheduler 的所有 timestep 依次去噪：
   - `noise_pred = DiTNoiseNet(noisy_action_seq, t, obs_cond)`
   - `noisy_action_seq = scheduler.step(...).prev_sample`
5. 得到完整动作序列后，只取:
- `action_seq = noisy_action_seq[:, H-1 : H-1+Ah]`
- shape `(B, Ah, A)`
6. 最后必须把前 `d` 维动作映射回真实动作范围(如果 denormalizer 未设置直接报错)

## 10. whole evaluation process

### 10.0 key function

- `evaluate_only_mas(...)`闭环：

```text
reset env
-> 初始化 rollout history
-> 从 demo 取 mas condition
-> 从 rollout history 用 STPM 推 progress
-> 构造 obs["progress"]
-> agent.get_action(obs)
-> env.step(action)
-> 把新观测追加到 history
-> 重复直到 episode 结束
```
with:
- `mas` 来自 demo
- `progress` 来自 rollout 历史上的 STPM 在线推理

### 10.1 prepare eval_demo 和对齐信息:

- 把评估时会用到的 demo、seed、traj_id、MAS condition 全部提前对齐好

#### 先load eval_demo from path

#### `infer_eval_reset_seeds_from_demo(...)`
输入：
- `demo_path: str`
- `num_traj: Optional[int]`
- `metadata_path: Optional[str]`
输出：
- `eval_reset_seeds: List[int]`
- `(N_eval,)`
#### `infer_eval_traj_ids_from_demo(...)`
输入：
- `demo_path: str`
- `num_traj: Optional[int]`
输出：
- `eval_traj_ids: List[int]`
- `(N_eval,)`
#### `select_eval_demo_indices(...)`
- 从全部 demo 里选出“这次实际要评估的子集下标”
输入：
- `total_demos: int`
- `num_eval_demos: int`
输出：
- `eval_demo_indices: List[int]`
- `(N_used,)`
#### `eval_reset_seeds / eval_traj_ids` 子集裁剪
输入：
- `eval_demo_indices`
- `eval_reset_seeds`
- `eval_traj_ids`
输出：
- 裁剪后的 `eval_reset_seeds`
- 裁剪后的 `eval_traj_ids`
if _test
- 按 demo seed 精确评估，代码会强制 `args.num_eval_envs == 1`
- 逐条 demo 对齐，不是无约束并行 rollout
#### `load_eval_only_mas_data(...)`
输入：
- `data_path: str`
- `device: torch.device`
- `expected_mas_flat_dim: int`
- `num_traj: Optional[int]`
输出：
- `eval_mam_data["mas_flat"]`
  - list of `(T*8,)`
- `eval_mam_data["mas"]`
  - list of `(T, 8)`
- `eval_mam_data["traj_lengths"]`
  - 每条 traj 未做 padding 的真实 `action` 长度
  - 给 `AP8` 按当前 traj 计算 `delta = 1 / traj_len`
#### `subset_eval_data(...)`
- 对 `eval_mam_data` 里所有 list 类型字段按同一组 index 做裁剪
输出：
- 子集版 `eval_mam_data`

#### 这一段执行完后，主函数有哪些 eval 数据:
- `eval_demo_path`
  - demo set
- `eval_reset_seeds`
  - 每个评估 episode 的 reset seed
- `eval_traj_ids`
  - 每个评估 episode 对应 demo 轨迹
- `eval_demo_indices`
  - 当前实际评估的是 demo 总集里的哪几条
- `eval_mam_data["mas_flat"]`
  - 评估时 `obs["mas"]` 的来源
- `eval_mam_data["mas"]`
  - `MAS8 / AP8` 进度条件的来源
- `eval_mam_data["traj_lengths"]`
  - `AP8` 每条轨迹各自的时间步长基准

### 10.2 进入 `evaluate_only_mas(...)`

preparation:

1. `validate_stpm_eval_setup(...)`
   - 检查 STPM checkpoint 是否与当前评估兼容
2. `agent.eval()`
3. 初始化：
   - `eval_metrics`
   - `traj_cursor`
   - `num_traj`
4.第一次 `reset`：`obs, info = eval_envs.reset(...)`
and keep history of ["rgb"] ["depth"] ["state"]

Additionally, 初始化:
- `traj_ids: (B,)`当前每个 env 对应哪条 demo
- `step_ptr: (B,)`当前 rollout 已经执行了多少步

### 10.3 rollout 前，构造 `obs["mas"]` and `obs["progress"]`
- `build_mas_condition_batch(...)`
- `obs["mas"]`: `(B, H, T*8)`

#### `MLP8` 
- `predict_progress_from_histories(...)`：

1. 对每个 env，当前 history 长度应为：
   - `obs_horizon + step_ptr[env_idx]`
2. 对该 env 的 observation window，一共构造 `H` 个 anchor
3. 对每个 anchor：
   - `_sample_stpm_history_indices(...)` 取一段长度为 `stpm_n_obs_steps + 1` 的index
   - `_build_stpm_rgbd_frame(...)` 构造单帧 `rgbd`
   - `_reorder_pickcube_rollout_state_for_stpm(...)` 重排单帧 `state`
4. 所有 env、所有 anchor 合并后，得到：
   - `rgbd_batch`: `(B*H, T_stpm, 4, H_img, W_img)`
   - `state_batch`: `(B*H, T_stpm, S_stpm)`
5. 调用：
   - `stpm_encoder.predict_progress(...)`
6. STPM 输出：
   - `progress: (B*H,) `reshape成`(B, H, 1)` (标量窗口)

#### `MAS8` 
- `predict_current_progress_from_histories(...)`
1. 对每个 env，只取当前时刻一个 anchor：
   - `anchor_idx = step_ptr + obs_horizon - 1`
2. 构造一段 STPM 输入序列
3. 得到：
   - `current_progress: (B, 1)

- `build_mas_progress_condition_batch(...)`
1. 对每个 batch 样本，取对应 demo 的增强后：
   - `mas_t: (T, 8)`
2. 取最后一列：
   - `progress_col: (T,)`
3. 用当前 progress 标量 `p` 和 `progress_col` 做最近邻匹配
4. 得到 `nearest_idx`
5. 从该位置向前截一个长度为 `H` 的 MAS window (重复首帧)
- `obs["progress"]`: `(B, H, 8)`

#### `AP8`
- `predict_current_progress_from_histories(...)`
same as above and get `current_progress: (B, 1)`

- `build_ap_progress_condition_batch(...)`
1. 对每个 batch 取出 demo 真实动作长度：
- `traj_len = eval_mam_data["traj_lengths"][traj_id]`
2. 再计算 delta = 1 / traj_len
3. 对每个样本，已知当前 progress 是 `p_t`
4. 先按 observation window 回推出 `H` 个 anchor：
5. 对每个 anchor，各自向前构造长度为 8 的 progress 序列：
```text
[p_anchor, p_anchor+delta, ..., p_anchor+7*delta]
```
6. 最后把所有值裁剪到 `[0, 1]`:`obs["progress"]`: `(B, H, 8)`

### 10.4 rollout once

with `obs["mas"]` and `obs["progress"]` prepared
conduct one step:
- `action_seq = agent.get_action(obs)`
输出：
- `(B, Ah, A)`
and conduct
1. `eval_envs.step(action_seq[:, action_idx])`
2. `prepare_batched_rollout_obs(raw_next_obs, device, obs_horizon)`
3. `append_latest_rollout_frame(histories, next_obs)`
4. `executed_steps += 1`
5. 如果 `truncated.any()`，提前结束本轮 chunk

after one  timestep:
- `obs = next_obs`
- `step_ptr = step_ptr + executed_steps`

### 10.5 episode end handling

1. `append_episode_metrics(eval_metrics, info)` 收集 episode 指标
2. `eps_count += num_envs`
3. 更新 `traj_ids`
4. `step_ptr.zero_()`

## 11. 训练与评估的核心差别

### 11.1 `mas` 分支
训练和评估一致
### 11.2 `progress` 分支
训练：
- 直接从 demo 里的增强后 `mas` 获得 progress 或其衍生形式
评估：
- 先用 rollout 历史喂给 STPM
- 得到当前进度估计
- 再按 `MLP8 / MAS8 / AP8` 转成对应 DiT condition

# mas-window pipeline

最新要求：
把原来的单一 `mas_window` 改成两条条件分支：
1. `mas_long_window`
   - shape: `(B, H, 8 * pred_horizon)`
   - 表示更长的 future MAS window
   - 不直接 flatten 到 condition
   - 先做卷积编码，再输入 condition
   - 编码模式只保留：
     - `1DConv`
     - `2DConv`
   - 并允许配置卷积输出维度
   - 当 `mas_long_conv_output_dim = 0` 时，该分支关闭
2. `mas_short_window`
   - shape: `(B, H, 8 * short_window_horizon)`
   - 表示更短、更局部的 future MAS window
   - 直接 flatten 后输入 condition
   - 当 `short_window_horizon = 0` 时，该分支关闭

其余整体设计保持不变：
- 训练时仍用 GT 时间步直接构造窗口
- 评估时仍由 STPM 在线推断当前 progress 后做时间对齐
- 视觉、状态、DiT 主干、动作去噪流程都保持原逻辑

documents：
- `examples/baselines/diffusion_policy/train_mas_window.py`
- `examples/baselines/diffusion_policy/train_mas_window_test.py`
- `examples/baselines/diffusion_policy/evaluate/evaluate_mas_window.py`
- `examples/baselines/diffusion_policy/evaluate/evaluate_mas_window_test.py`
- `examples/baselines/diffusion_policy/run_train_mas_window.sh`
- `examples/baselines/diffusion_policy/run_train_mas_window_test.sh`
- `examples/baselines/diffusion_policy/models/*`
- `examples/baselines/diffusion_policy/utils/*`
- `STPM/models/stpm_encoder.py`

## 1. overview

`train_mas_window.py` trains a Diffusion Policy with `DiTNoiseNet`。  
与 `only-mas` 不同的是，在常规视觉和状态条件之外，额外接收两条 MAS future condition：
1. `mas_long_window`
   - 从增强后 `mas (T, 8)` 中截取长度为 `pred_horizon` 的 future window
   - 经过卷积编码后再送入 DiT condition
   - 若 `mas_long_conv_output_dim = 0`，则该分支关闭
2. `mas_short_window`
   - 从增强后 `mas (T, 8)` 中截取长度为 `short_window_horizon` 的 future window
   - 直接 flatten 后拼入 condition
   - 若 `short_window_horizon = 0`，则该分支关闭


新增超参数：
- `short_window_horizon`
- `mas_long_encode_mode`：
  - `2DConv`
  - `1DConv`
- `mas_long_conv_output_dim`

两个窗口分支都不是必须的：
- `short_window_horizon = 0`
  - 不使用 `mas_short_window`
- `mas_long_conv_output_dim = 0`
  - 不使用 `mas_long_window`

## 2. 入口

### 2.1 训练入口
同上，改为：
`run_train_mas_window.sh`
```bash
python examples/baselines/diffusion_policy/train_mas_window.py ...
```

### 2.2 test 入口
 `train_mas_window_test.py` / `run_train_mas_window_test.sh`

## 3. 训练前的准备

主入口 `train_mas_window.py` 在 `__main__` 中整体流程同上：
1. 解析参数 `Args`
2. 设置随机种子
3. 从评估 demo metadata 推断 reset seed
4. 读取动作反归一化参数
5. 构建 `STPMEncoder`
6. 构建评估环境 `envs`
7. 构建训练数据集 `SmallDemoDataset_MasWindowDiffusionPolicy`
8. 读取评估侧 mas-window 条件数据 `eval_mas_window_data`
9. 创建 `Agent`
10. 创建优化器、lr scheduler、EMA
11. 进入训练循环，并定期调用 `evaluate_mas_window`

和 only-mas 的主要差别：
- dataset 不再产出 `obs["mas"]`
- dataset 不再产出 `obs["progress"]`
- 改为产出：
  - `obs["mas_long_window"]`
  - `obs["mas_short_window"]`
  - 未启用的分支可视为最后一维为 `0`

## 4. 训练数据是如何构造的

核心类：`SmallDemoDataset_MasWindowDiffusionPolicy`

### 4.1 原始数据读取
同上：
- `observations`
- `actions`
- `mas`
- `env_states`
- `success`
- `terminated`
- `truncated`

记号同上，额外记：
- `Pl = pred_horizon`
- `Hs = short_window_horizon`
- `Dlong = 8 * Pl`
- `Dshort = 8 * Hs`

### 4.2 观测预处理
同上

### 4.3 给 MAS 增加 progress 列
同上，仍复用：
- `build_progress_column(traj_len, mas_len, device)`
- `augment_mas_with_progress(mas_t, traj_len)`

增强后：
- `mas_t: (T, 8)`

### 4.4 样本切片方式
同上：
```python
(traj_idx, start, start + pred_horizon)
```

### 4.5 `mas_long_window / mas_short_window` 的构造逻辑

关键函数可新增为：
- `build_mas_long_window_from_future(mas_t, current_step, pred_horizon) -> (pred_horizon, 8)`
- `build_mas_short_window_from_future(mas_t, current_step, short_window_horizon) -> (short_window_horizon, 8)`
- `build_dual_mas_window_obs_horizon(...)`
  - `-> long_window: (obs_horizon, pred_horizon, 8)`
  - `-> short_window: (obs_horizon, short_window_horizon, 8)`

逻辑：
1. 对训练样本当前时刻 `t`，先确定观测窗口的 `H` 个 anchor（向前取，若越界则重复）
2. 对每个 anchor，在增强后 `mas_t (T, 8)` 中分别构造两种 future window：
   - 长窗口：长度 `pred_horizon`
   - 短窗口：长度 `short_window_horizon`
3. 若未来越界，则重复最后一个 MAS token padding
4. 得到：
   - `mas_long_window_raw`: `(H, pred_horizon, 8)`
   - `mas_short_window_raw`: `(H, short_window_horizon, 8)`
5. flatten 后得到：
   - `obs["mas_long_window"]`: `(H, 8 * pred_horizon)`
   - `obs["mas_short_window"]`: `(H, 8 * short_window_horizon)`

若分支关闭：
- `obs["mas_long_window"]` 可视为 `(H, 0)`
- `obs["mas_short_window"]` 可视为 `(H, 0)`

注意：
- 这里的“当前位置”在训练时直接由样本时间步确定，不需要 STPM
- 训练集里的双窗口都由 GT 轨迹时间对齐直接构造

### 4.6 `__getitem__` 返回的主要字段与 shape
对单个样本：
- `obs_seq["rgb"]`: `(H, 3*k, H_img, W_img)`
- `obs_seq["depth"]`: `(H, k, H_img, W_img)`
- `obs_seq["state"]`: `(H, S)`
- `obs_seq["mas_long_window"]`: `(H, 8*pred_horizon)`
- `obs_seq["mas_short_window"]`: `(H, 8*short_window_horizon)`
- `act_seq`: `(P, A)`

若分支关闭：
- long 关闭时，`obs_seq["mas_long_window"]`: `(H, 0)`
- short 关闭时，`obs_seq["mas_short_window"]`: `(H, 0)`

## 5. 双窗口条件的注入方式 (while training)

### Dataset:

先统一构造两种局部 future window：
- `mas_long_window_raw`: `(H, pred_horizon, 8)`
- `mas_short_window_raw`: `(H, short_window_horizon, 8)`

#### `mas_short_window`
1. 将每个 anchor 对应的 `(short_window_horizon, 8)` 直接 flatten
2. 得到：
   - `obs["mas_short_window"]`: `(H, 8*short_window_horizon)`
3. 直接输入 condition，不额外编码
4. 若 `short_window_horizon = 0`
   - 该分支关闭
   - 输出视为 `(B, H, 0)`

#### `mas_long_window + 2DConv`
1. dataset 侧保存 flatten 结果，或保留 `(H, pred_horizon, 8)` 的中间表示
2. 在 `Agent.encode_mas_long_window` 中 reshape 成卷积所需形式
3. 目标是让二维卷积同时看：
   - 时间维 `pred_horizon`
   - MAS feature 维 `8`

#### `mas_long_window + 1DConv`
1. dataset 侧同样来自 `mas_long_window_raw`
2. 在 `Agent.encode_mas_long_window` 中把 `8` 维看成并行通道组
3. 只沿 `pred_horizon` 时间维卷积，不混合单个 token 内部的相对位置

若 `mas_long_conv_output_dim = 0`
- long 分支关闭
- 不构造 long conv feature
- 输出视为 `(B, H, 0)`

## 6. Agent 把条件编码后送进 DiT

核心类：`Agent`

需和 only-mas 的 padding 语义对齐，可把 `-1` 替换成 `0`

### 6.1 视觉分支
同上

### 6.2 state 分支
同上

### 6.3 mas_long_window 分支

输入统一视为：
- `obs["mas_long_window"]`: `(B, H, 8*pred_horizon)`
  - 若关闭则可视为 `(B, H, 0)`

内部可先 reshape 为：
- `mas_long_value`: `(B, H, pred_horizon, 8)`
- 若需要 mask，也可额外构造：
  - `mas_long_mask`: `(B, H, pred_horizon, 8)`

分两种编码方式：

#### `2DConv`
1. reshape 成：
   - `(B, H, pred_horizon, 8)`
2. 构造 value/mask 双通道后调维，可形成：
   - `(B*H, 2, 8, pred_horizon)`
   或等价二维卷积输入格式
3. 送入 `MasWindowConv(out_dim=mas_long_conv_output_dim)`
4. 输出：
   - `(B*H, mas_long_conv_output_dim)`
5. reshape 回：
   - `(B, H, mas_long_conv_output_dim)`

#### `1DConv`
1. reshape 成：
   - `(B, H, pred_horizon, 8)`
2. 调整成只沿时间维卷积的输入形式，例如：
   - `(B*H, 8, pred_horizon)`
3. 送入 `MasWindowConv1D(out_dim=mas_long_conv_output_dim)`
4. 输出：
   - `(B*H, mas_long_conv_output_dim)`
5. reshape 回：
   - `(B, H, mas_long_conv_output_dim)`

### 6.4 mas_short_window 分支

- `obs["mas_short_window"]`: `(B, H, 8*short_window_horizon)`
  - 若关闭则可视为 `(B, H, 0)`
- 不经过额外编码器
- 仅做 padding 语义清理：
  - 将 `-1` 替换为 `0`
- 输出：
  - `mas_short_feature: (B, H, 8*short_window_horizon)`

### 6.5 最终 observation condition

`obs_conditioning` 最后拼接：
```text
[visual_feature, state, mas_long_feature, mas_short_feature]
```

- `mas_long_feature_dim = mas_long_conv_output_dim`
- `mas_short_feature_dim = 8 * short_window_horizon`

若关闭：
- `mas_long_feature_dim = 0`，当 `mas_long_conv_output_dim = 0`
- `mas_short_feature_dim = 0`，当 `short_window_horizon = 0`

最终：
- `obs_cond`: `(B, H, obs_dim)`

```text
obs_dim = 256 + S + mas_long_feature_dim + mas_short_feature_dim
```

## 7. DiTNoiseNet 内部的数据流

同上

## 8. 训练流程

### 8.1 dataloader 输出
每个 batch：
- `data_batch["observations"]`
  - `rgb`: 同上
  - `depth`: 同上
  - `state`: `(B, H, S)`
  - `mas_long_window`: `(B, H, 8*pred_horizon)` 或 `(B, H, 0)`
  - `mas_short_window`: `(B, H, 8*short_window_horizon)` 或 `(B, H, 0)`
- `data_batch["actions"]`: `(B, P, A)`

### 8.2 `Agent.compute_loss`
同上

区别只在：
1. `obs_conditioning(obs_seq)` 内不再编码 `mas` / `progress`
2. 改为编码：
   - `mas_long_window`
   - `mas_short_window`

### 8.3 参数更新
同上

## 9. Inference: 动作生成

### 9.1 加载动作反归一化参数
同上

### 9.2 `get_action(...)` input

评估循环在每一轮调用 `agent.get_action(obs)` 时，`obs` 里包含：
- `state`: `(B, H, S)`
- `rgb`: `(B, H, H_img, W_img, 3)`
- `depth`: `(B, H, H_img, W_img, 1)`
- `mas_long_window`: `(B, H, 8*pred_horizon)` 或 `(B, H, 0)`
- `mas_short_window`: `(B, H, 8*short_window_horizon)` 或 `(B, H, 0)`

### 9.3 `get_action(...)` output
同上

区别只在：
- `obs_conditioning(obs_seq, eval_mode=True)` 只拼接：
  - `visual`
  - `state`
  - `mas_long_window_feature`
  - `mas_short_window_feature`

## 10. whole evaluation process

### 10.0 key function

- `evaluate_mas_window(...)` 闭环：

```text
reset env
-> 初始化 rollout history
-> 从 demo 取增强后 mas
-> 从 rollout history 用 STPM 推 current progress
-> 基于 current progress 构造 obs["mas_long_window"] 和 obs["mas_short_window"]
-> agent.get_action(obs)
-> env.step(action)
-> 把新观测追加到 history
-> 重复直到 episode 结束
```

with:
- `mas_long_window / mas_short_window` 来自 demo 中增强后的 `mas`
- 当前对齐位置来自 rollout 历史上的 STPM 在线推理

### 10.1 prepare eval_demo 和对齐信息
同上

需要保留的 eval 数据：
- `eval_demo_path`
- `eval_reset_seeds`
- `eval_traj_ids`
- `eval_demo_indices`
- `eval_mas_window_data["mas"]`
  - list of `(T, 8)`，用于在线构造 long/short window
- `eval_mas_window_data["traj_lengths"]`
  - 若需要把 STPM 输出 progress 映射到离散时间索引时使用

可不再需要：
- `eval_mam_data["mas_flat"]`

### 10.2 进入 `evaluate_mas_window(...)`
同上

初始化：
- `traj_ids: (B,)`
- `step_ptr: (B,)`
- rollout histories of `rgb / depth / state`

### 10.3 rollout 前，构造 `obs["mas_long_window"]` 和 `obs["mas_short_window"]`

关键函数可新增为：
- `predict_current_progress_from_histories(...)`
- `build_mas_long_window_condition_batch(...)`
- `build_mas_short_window_condition_batch(...)`
- 或一个统一 helper：
  - `build_dual_mas_window_condition_batch(...)`

#### 第一步：在线预测当前 progress

********注意这里只取当前帧于此progress，也就是只预测了一个progress值，其余obs_horizon个维度直接通过这个预测出来的列向前推得到。

`predict_current_progress_from_histories(...)`
1. 对每个 env，只取当前时刻一个 anchor：
   - `anchor_idx = step_ptr + obs_horizon - 1`
2. 构造一段 STPM 输入序列
3. 调用：
   - `stpm_encoder.predict_progress(...)`
4. 得到：
   - `current_progress: (B, 1)`

#### 第二步：由 current progress 构造 future long/short window

`build_dual_mas_window_condition_batch(...)`
对每个 batch 样本：
1. 取对应 demo 的增强后：
   - `mas_t: (T, 8)`
2. 取最后一列：
   - `progress_col: (T,)`
3. 用当前 progress 标量 `p` 和 `progress_col` 做最近邻匹配
4. 得到 `nearest_idx`
5. 从该位置开始，分别向后截取：
   - 长窗口：长度 `pred_horizon`
   - 短窗口：长度 `short_window_horizon`
6. 若越界，则重复最后一个 token
7. 对观测窗口中的其他 anchor 不再分别预测progress，而是根据上一个anchor直接向前推一帧得到其余窗口：
   - 长窗口原始形状：`(B, H, pred_horizon, 8)`
   - 短窗口原始形状：`(B, H, short_window_horizon, 8)`
8. 最终按模型输入需要变成：
   - `obs["mas_long_window"]`: `(B, H, 8*pred_horizon)` 或 `(B, H, 0)`
   - `obs["mas_short_window"]`: `(B, H, 8*short_window_horizon)` 或 `(B, H, 0)`

！！评估时不为每个 anchor 都单独跑一遍 STPM，采用：
- 只预测当前时刻 `p_t`
- 再按固定 `delta` 回推出 `H` 个 anchor 对应的 progress

### 10.4 rollout once
同上

with `obs["mas_long_window"]` and `obs["mas_short_window"]` prepared:
- `action_seq = agent.get_action(obs)`

### 10.5 episode end handling
同上

## 11. implementation checklist

为了后续真正写代码，建议按下面顺序落地：
1. 更新 `Args`
   - 删除旧的 `mas_horizon`
   - 删除旧的单分支 `mas_window_mode`
   - 新增：
     - `short_window_horizon`
     - `mas_long_encode_mode`
     - `mas_long_conv_output_dim`
   - 可选语义：
     - `short_window_horizon = 0` 关闭 short 分支
     - `mas_long_conv_output_dim = 0` 关闭 long 分支
2. 更新 dataset / helper
   - 新增双窗口构造 helper
   - `__getitem__` 改为同时返回：
     - `mas_long_window`
     - `mas_short_window`
3. 更新 `Agent`
   - 把原 `encode_mas_window` 拆成：
     - `encode_mas_long_window`
     - `encode_mas_short_window` 或直接 inline
   - 保留 long 分支的 value/mask 语义
4. 更新 `obs_conditioning`
   - condition 改为 `visual + state + long_feature + short_feature`
   - 更新 `expected_obs_cond_dim`
5. 更新 loss / rollout 输入约定
   - 训练和推理都统一读新字段名
6. 更新评估 helper
   - 在线从 STPM progress 构造 long/short 双窗口
   - 保持 reset/history/video 逻辑不变
7. 更新 `train_mas_window.py` / `_test.py`
   - 数据流、日志打印、shape check 全部切到双窗口命名
8. 更新 `evaluate_mas_window.py` / `_test.py`
   - rollout 前构造双窗口 condition
9. 更新 `run_train_mas_window.sh` / `_test.sh`
   - 参数名与默认值切到新设计
10. 验证
   - 先跑 test 版检查 shape
   - 再跑小规模训练确认 overfit 和 eval 不报错
