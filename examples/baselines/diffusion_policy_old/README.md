# Diffusion Policy（Only MAS）Conditioning 与数据流

本文档说明：观测与其他输入如何作为条件注入模型、做了哪些预处理/归一化，以及张量形状在流程中的变化。基于 `train_only_mas.py` 与 `diffusion_policy/evaluate_only_mas.py`。

## 总览

模型在每个时间步使用三类条件：

1. **视觉特征**：来自 RGB + 深度图。
2. **状态向量**：环境的 state（本代码中不做归一化）。
3. **MAS（masked action space）**：轨迹级别的扁平化向量。

三者沿特征维拼接，作为 diffusion action 模型的 FiLM 风格条件：

```
feature = concat(visual_feature, state, mas)
```

## 数据来源

### 训练数据（离线）
通过 `SmallDemoDataset_DiffusionPolicy` 中的 `load_demo_dataset` 从 demo `.h5` 读取：

- `observations`（包含 `rgb`, `depth`, `state`）
- `actions`
- `mas`（轨迹级矩阵，已 padding 到最长长度）
- `env_states`, `success`, `terminated`, `truncated`（辅助信息）

### 评估数据（在线）
来自 `evaluate_only_mas.py`：

- 环境每步返回 `obs`。
- MAS 从 `eval_mam_data["mas_flat"]` 注入，并在观测窗口上重复。

## 预处理与归一化

### 视觉观测
在 `Agent.encode_obs` 中处理：

- `rgb`：`uint8` → `float`，**除以 255.0**
- `depth`：`float` → `float`，**除以 1024.0**

未做其他视觉归一化。

### 状态观测
- **本代码中不做归一化。**
- `state` 直接转 tensor 后拼接。

如果需要归一化（如 z-score / min-max），需自行添加并确保训练/推理一致。

### MAS
- 将 `-1` 替换为 `0`（mask）。
- 每条轨迹的 MAS 展平成 `(T * act_dim,)`。

### 动作
- **训练**：actions 直接用于 diffusion 训练。
- **评估**：若提供 `action_norm_path`，会在 `env.step()` 前**反归一化**：
  - `action_seq[..., :d] = min + 0.5 * (a + 1) * (max - min)`

## 条件注入细节

### 1）视觉特征

输入形状（RGB 与 Depth）：`(B, obs_horizon, H, W, C)`

处理流程：

1. 在 `convert_obs`（训练）或 `get_action`（在线）中转为通道优先：
   - RGB: `(B, L, H, W, 3*k)` → `(B, L, 3*k, H, W)`
   - Depth: `(B, L, H, W, 1*k)` → `(B, L, 1*k, H, W)`
2. 归一化：RGB/255，Depth/1024
3. 通道维拼接：
   - `(B, L, C_rgb, H, W)` + `(B, L, C_d, H, W)` → `(B, L, C, H, W)`
4. 把时间维展平到 batch 维：
   - `(B, L, C, H, W)` → `(B*L, C, H, W)`
5. `PlainConv` 编码 → `(B*L, D)`
6. 还原：
   - `(B*L, D)` → `(B, L, D)`

### 2）State

输入形状：`(B, obs_horizon, state_dim)`

- 无归一化。
- 直接与视觉特征、MAS 拼接。

### 3）MAS

#### 训练
- 每条轨迹原始形状：`(T, act_dim)`
- 替换 `-1` 为 `0`
- 展平：`(T * act_dim,)`
- 在 obs_horizon 维度上重复：
  - `(obs_horizon, T * act_dim)`

#### 评估
- `mas_flat_list[i]` 形状为 `(T * act_dim,)`
- 对 batch 维堆叠为 `(B, T * act_dim)`
- `unsqueeze + repeat`：
  - `(B, 1, T * act_dim)` → `(B, obs_horizon, T * act_dim)`

### 最终条件张量

拼接：

```
obs_cond = concat(
    visual_feature,        # (B, obs_horizon, D)，D-视觉编码器输出的特征维度（256）
    state,                 # (B, obs_horizon, state_dim)
    mas                    # (B, obs_horizon, T*act_dim)
)
```

最终形状：

```
obs_cond: (B, obs_horizon, D + state_dim + T*act_dim)
```

该 `obs_cond` 作为条件输入给 `DiTNoiseNet`。

## 动作序列形状

- diffusion 生成 `pred_horizon` 动作：
  - `(B, pred_horizon, act_dim)`
- 只执行 `act_horizon` 窗口：
  - `start = obs_horizon - 1`
  - `action_seq = action_seq[:, start:start+act_horizon]`

## 如果需要 state 归一化，建议位置

要在训练与推理中保持一致：

- **训练**：在 dataset 预处理时，对 `obs_traj_dict["state"]` 归一化
- **推理**：在 `Agent.encode_obs` 或在线 rollout 的 `to_tensor(obs, device)` 之后归一化

并确保使用同一套统计量（mean/std 或 min/max）。
