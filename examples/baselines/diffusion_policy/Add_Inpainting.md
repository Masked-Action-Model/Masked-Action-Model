# Add_Inpainting

**我们在这个文档里就inpainting的问题进行交互，我会每次在这个文档里写上进一步指示，然后你按我的要求执行**

**这个文档里我写的内容你不要删，只按照我的要求增改**

## 方法概述：

在完成了对diffusionpolicy的condition部分改在之后（见Debug1，我们最终选择了mas_window的方案），我需要再对其inference阶段进行改造，来加强mas中控制信号的作用效力。

我们将要学习借鉴Repaint这一来自图像生成领域的training-free的inference-time inpainting 方法（https://arxiv.org/pdf/2201.09865），将其迁移到动作生成领域。

也就是在维持模型训练步骤（核心是一个噪声预测模型）不变的情况下，改造其inference流程来获得更好的rollout效果。

### Repaint伪代码：

 Inputs:
   model        : 训练好的 diffusion denoiser
   x_known_0    : 已知区域的真值（原始图像中未被mask掉的部分）#   mask         : 1表示known，0表示unknown
   T            : 总diffusion步数
   r            : 每个anchor位置的resampling轮数
   j            : jump length，每次向回跳多少步
   scheduler    : 给出 alpha_bar[t], beta[t] 等参数

 Output:
   x_0          : 最终inpaint结果

- 把已知区域 x_known_0 加噪到第 t 步
def forward_noise_known(x_known_0, t):
    eps = sample_standard_gaussian_like(x_known_0)
    return sqrt(alpha_bar[t]) * x_known_0 + sqrt(1 - alpha_bar[t]) * eps
- 在第 t 步，把 known / unknown 合并
**注意：known部分必须处在“当前噪声层级t**
def merge_known_unknown(x_unknown_t, x_known_0, mask, t):
    x_known_t = forward_noise_known(x_known_0, t)
    return mask * x_known_t + (1 - mask) * x_unknown_t
- 单步 reverse: x_t -> x_{t-1}
def reverse_step(model, x_t, t):
    **这里代表标准DDPM的一步去噪**
    return sample_from_p_theta(model, x_t, t)
- 单步 forward: x_t -> x_{t+1}
def forward_step(x_t, t):
    **从 x_t 重新加噪到 x_{t+1}**
    eps = sample_standard_gaussian_like(x_t)
    return sqrt(1 - beta[t + 1]) * x_t + sqrt(beta[t + 1]) * eps
- 连续 forward 跳 j 步: x_t -> x_{t+j}
def jump_forward_j_steps(x_t, t, j, T):
    x = x_t
    cur_t = t
    for _ in range(j):
        if cur_t >= T:
            break
        x = forward_step(x, cur_t)
        cur_t += 1
    return x, cur_t

- Main inference

def repaint_inference(model, x_known_0, mask, T, r, j, scheduler):
    **1. 初始化：从纯噪声 x_T 开始**
    x_t = sample_standard_gaussian(shape_of(x_known_0))
    t = T

**2. 主循环：不断从高噪声往低噪声走**
    while t > 0:

        (A) 先连续 reverse j 步，走到一个anchor位置，从 x_t 走到 x_{t-j}
        steps_down = min(j, t)
        for _ in range(steps_down):

            x_prev = reverse_step(model, x_t, t)
            x_prev = merge_known_unknown(x_prev, x_known_0, mask, t - 1)
            x_t = x_prev
            t = t - 1

**现在位于某个 anchor：x_t**

        (B) 在当前anchor附近做 r-1 轮 resampling，每一轮：jump forward j步，再 reverse j步回来
        if t > 0:
            for _ in range(r - 1):

                x_jump, t_jump = jump_forward_j_steps(x_t, t, j, T)
                x_tmp = x_jump
                cur_t = t_jump
                while cur_t > t:
                    x_prev = reverse_step(model, x_tmp, cur_t)
                    x_prev = merge_known_unknown(x_prev, x_known_0, mask, cur_t - 1)

                    x_tmp = x_prev
                    cur_t -= 1

                x_t = x_tmp
    return x_t

### MAM+Inpainting伪代码

#### 内层repaint inference循环

Inputs:
  obs_seq         : 当前观测窗口（rgb/depth/state/mas_window/...），batch 形式
  mas_inpaint     : 当前推理步单独取出的 inpainting 条件，shape `(B, P, 7)`
  mas_inpaint_mask: 对应 mask，shape `(B, P, 7)`，1=known, 0=unknown
  a_t             : 当前扩散层级下的动作序列，shape `(B, P, A)`；初始化为高斯噪声
  action_known_0  : 由 `mas_inpaint` 对齐到 `pred_horizon` 后得到的已知动作，shape `(B, P, A)`
  action_mask     : 由 `mas_inpaint_mask` 对齐到 `pred_horizon` 后得到的 mask，shape `(B, P, A)`
  policy_model    : diffusion policy 的 denoiser
  obs_encoder     : 条件编码器
  scheduler       : 当前使用的 DDPM scheduler
  T               : `scheduler.config.num_train_timesteps`
  r               : 每个 anchor 的 resampling 次数
  j               : jump length
  H               : obs_horizon
  P               : pred_horizon
  A               : action_dim

Output:
  action_seq_0    : 生成出的 normalized action chunk, `(B, P, A)`

def build_inpaint_mas_mask(
    mas_inpaint,
    mas_inpaint_mask,
    H,
    P,
    A,
):
    B = mas_inpaint.shape[0]
    start = H - 1
    A_known = min(A, 7)

    action_known_0 = zeros(B, P, A)
    action_mask = zeros(B, P, A)

    copy_len = min(mas_inpaint.shape[1], P - start)
    action_known_0[:, start:start+copy_len, :A_known] = mas_inpaint[:, :copy_len, :A_known]
    action_mask[:, start:start+copy_len, :A_known] = (
        mas_inpaint_mask[:, :copy_len, :A_known] > 0.5
    ).float()
    return action_known_0, action_mask

def overwrite_known_action(a_unknown_t, action_known_0, action_mask, t, scheduler):
    # t > 0 时使用：先把 clean known action 加噪到当前层级，再覆盖
    eps = randn_like(action_known_0)
    t_batch = full((action_known_0.shape[0],), t, dtype=long, device=action_known_0.device)
    a_known_t = scheduler.add_noise(action_known_0, eps, t_batch)
    return action_mask * a_known_t + (1 - action_mask) * a_unknown_t

def overwrite_known_action_clean(a_unknown_0, action_known_0, action_mask):
    # t == 0 时使用：已经到 clean level，直接覆盖
    return action_mask * action_known_0 + (1 - action_mask) * a_unknown_0

def dp_repaint_inference(
    obs_seq,
    mas_inpaint,
    mas_inpaint_mask,
    policy_model,
    obs_encoder,
    scheduler,
    T,
    r,
    j,
    H,
    P,
    A,
):
    B = obs_seq["state"].shape[0]
    max_t = T - 1

    action_known_0, action_mask = build_inpaint_mas_mask(
        mas_inpaint=mas_inpaint,
        mas_inpaint_mask=mas_inpaint_mask,
        H=H,
        P=P,
        A=A,
    )

    # 编码观测条件
    cond = obs_encoder(obs_seq)

    # 初始化动作序列为高斯噪声
    a_t = randn(shape=(B, P, A), device=obs_seq["state"].device)
    t = max_t

    while t >= 0:
        steps_down = min(j, t + 1)

        for _ in range(steps_down):
            t_batch = full((B,), t, dtype=long, device=a_t.device)
            noise_pred = policy_model(a_t, t_batch, cond)
            a_prev = scheduler.step(
                model_output=noise_pred,
                timestep=t,
                sample=a_t,
            ).prev_sample

            if t > 0:
                a_prev = overwrite_known_action(a_prev, action_known_0, action_mask, t - 1, scheduler)
            else:
                a_prev = overwrite_known_action_clean(a_prev, action_known_0, action_mask)

            a_t = a_prev
            t = t - 1
            if t < 0:
                break
        if t < 0:
            break

        for _ in range(r - 1):
            a_jump, t_jump = forward_jump_j_steps(
                a_cur=a_t,
                t_cur=t,
                j=j,
                T=max_t,
                scheduler=scheduler,
            )

            a_tmp = a_jump
            cur_t = t_jump

            while cur_t > t:
                t_batch = full((B,), cur_t, dtype=long, device=a_tmp.device)
                noise_pred = policy_model(a_tmp, t_batch, cond)
                a_prev = scheduler.step(
                    model_output=noise_pred,
                    timestep=cur_t,
                    sample=a_tmp,
                ).prev_sample

                if cur_t > 0:
                    a_prev = overwrite_known_action(a_prev, action_known_0, action_mask, cur_t - 1, scheduler)
                else:
                    a_prev = overwrite_known_action_clean(a_prev, action_known_0, action_mask)

                a_tmp = a_prev
                cur_t = cur_t - 1

            a_t = a_tmp

    return a_t

#### 外层滑动action chunk循环：

while not done:
    obs_seq = build_current_obs_window_from_history(...)
    current_progress = predict_current_progress(...)

    # 1. 构造 obs condition 用的 dual MAS window
    dual_window = build_dual_mas_window_condition_batch(...)
    dual_window_mask = build_dual_mas_window_condition_batch(...)
    obs_seq["mas_long_window"] = dual_window["mas_long_window"]
    obs_seq["mas_short_window"] = dual_window["mas_short_window"]
    obs_seq["mas_long_window_mask"] = dual_window_mask["mas_long_window"]
    obs_seq["mas_short_window_mask"] = dual_window_mask["mas_short_window"]

    # 2. 单独构造动作 inpainting 用的 mas_inpaint
    mas_inpaint, mas_inpaint_mask = build_current_inpaint_mas_mask(
        mas_list=eval_mas_window_data["mas"],
        mas_mask_list=eval_mas_window_data["mas_mask"],
        traj_ids=traj_ids,
        current_progress=current_progress,
        pred_horizon=pred_horizon,
        device=obs_seq["state"].device,
        dtype=obs_seq["state"].dtype,
    )

    full_action_seq_norm = dp_repaint_inference(
        obs_seq         = obs_seq,
        mas_inpaint     = mas_inpaint,
        mas_inpaint_mask= mas_inpaint_mask,
        policy_model    = policy_model,
        obs_encoder     = obs_encoder,
        scheduler       = scheduler,
        T               = scheduler.config.num_train_timesteps,
        r               = r,
        j               = j,
        H               = obs_horizon,
        P               = pred_horizon,
        A               = action_dim,
    )

    start = obs_horizon - 1
    end = start + act_horizon
    exec_action_seq_norm = full_action_seq_norm[:, start:end]
    exec_action_seq = denormalize_action(exec_action_seq_norm)

    for k in range(act_horizon):
        obs, reward, done, info = env.step(exec_action_seq[:, k])
        obs_history.append(obs)
        if done:
            break

## 离线版代码设计

我们先实现一个离线版的inpainting inference。

用一个sh启动脚本，输入checkpoints文件，测试集路径，模型（噪声预测模型）相关信息
以及inpainting参数：j(jump_length)与r(num_resample) 
（注意j>=1，i可以为0，i取0时即回退到hardoverwrite版本）。

用一个utils文件放相关工具函数，尽量把主要的工具函数都方里面，方便后面迁移为在线版。

用一个py脚本实现加载checkpoints文件、数据集，然后在maniskill环境中进行rollout，每一步inference执行一次inpainting，注意之前progress以及window的相关设计（详见debug1.md），然后记录once、end成功率，以及计算三个ce值（很关键）。


### 总体策略

- 先做离线版，不改训练流程，不改 `compute_loss()`。
- 先不直接改现有 `Agent.get_action()`，避免影响现有 `mas_window` 评估；离线版单独走一条 inpainting rollout 路径。
- 最大化复用现有代码：
  - checkpoint / env / STPM / CE 统计复用 `evaluate_control_error_mas_window.py`
  - dual-window condition 复用 `build_dual_mas_window_condition_batch(...)`
  - 成功率与 CE 计算复用 `utils/control_error_utils.py`

### 计划新增文件

- `examples/baselines/diffusion_policy/utils/inpainting_utils.py`
  - 放 RePaint 动作版的核心工具函数。
- `examples/baselines/diffusion_policy/evaluate_control_error_mas_window_inpaint.py`
  - 离线入口脚本。
- `examples/baselines/diffusion_policy/run_eval_mas_window_inpaint.sh`
  - 启动脚本，负责整理参数并调用上面的 py 文件。

### `utils/inpainting_utils.py` 计划包含的函数

- `build_inpaint_mas_mask(...)`
  - 把 `mas_inpaint / mas_inpaint_mask` 对齐到 `(B, P, A)`。
  - 只写前 7 维 action。
  - 起始位置按 `start = obs_horizon - 1` 对齐。
- `overwrite_known_action(...)`
  - `t > 0` 时用，把 clean known action 加噪到当前层级后再覆盖。
- `overwrite_known_action_clean(...)`
  - `t == 0` 时直接在 clean level 覆盖。
- `forward_step_ddpm(...)`
  - 单步前向加噪，供 jump 使用。
- `forward_jump_j_steps(...)`
  - 从当前层级向高噪声跳 `j` 步。
- `build_current_inpaint_mas_mask(...)`
  - 根据 `current_progress + traj_ids + eval_data["mas"]/["mas_mask"]` 构造当前 rollout step 的 `mas_inpaint / mas_inpaint_mask`。
  - 输出 shape 固定为 `(B, pred_horizon, 7)`。
- `dp_repaint_inference(...)`
  - 输入 `agent / obs_seq / mas_inpaint / mas_inpaint_mask / j / r`
  - 内部调用 `agent.obs_conditioning(...)`、`agent.noise_pred_net(...)`、`agent.noise_scheduler.step(...)`
  - 输出 normalized 的 `(B, pred_horizon, action_dim)`。
- `sample_inpaint_action_chunk(...)`
  - 对 `dp_repaint_inference(...)` 再包一层。
  - 负责截取 `[:, obs_horizon-1 : obs_horizon-1+act_horizon]`
  - 再做动作反归一化，直接返回可 `env.step()` 的 action chunk。

### `evaluate_control_error_mas_window_inpaint.py` 实现思路

- 以 `evaluate_control_error_mas_window.py` 为模板复制一份，不从零写。
- 保留已有部分：
  - `Args`
  - checkpoint 加载
  - env 构造
  - STPM 构造
  - eval demo / reset seed 读取
  - CE 汇总与 json 保存
- 主要改 `_rollout_one_eval_batch(...)`：
  - 仍然先用 `predict_current_progress_from_histories(...)` 得到 `current_progress`
  - 仍然用 `build_dual_mas_window_condition_batch(...)` 构造 `obs["mas_long_window"]` 和 `obs["mas_short_window"]`
  - 再调用 `build_current_inpaint_mas_mask(...)` 构造当前 step 的 `mas_inpaint / mas_inpaint_mask`
  - 再调用 `sample_inpaint_action_chunk(...)` 代替当前的 `agent.get_action(obs)`
- rollout 结束后仍然沿用现有逻辑：
  - 记录 `success_once`
  - 记录 `success_at_end`
  - 记录执行动作
  - 计算 `ce_all / ce_success / ce_failed`

### `run_eval_mas_window_inpaint.sh` 参数设计

- 基础参数直接对齐现有离线 CE 入口：
  - `CHECKPOINT_PT_PATH`
  - `EVAL_DEMO_PATH`
  - `EVAL_DEMO_METADATA_PATH`
  - `ACTION_NORM_PATH`
  - `STPM_CKPT_PATH`
  - `STPM_CONFIG_PATH`
  - `ENV_ID`
  - `NUM_EVAL_DEMOS`
  - `NUM_EVAL_ENVS`
  - `OBS_HORIZON`
  - `ACT_HORIZON`
  - `PRED_HORIZON`
  - `LONG_WINDOW_HORIZON`
  - `SHORT_WINDOW_HORIZON`
  - `MAS_LONG_ENCODE_MODE`
  - `MAS_LONG_CONV_OUTPUT_DIM`
- 新增 inpainting 参数：
  - `INPAINT_J`
  - `INPAINT_R`
- 约束：
  - `INPAINT_J >= 1`
  - `INPAINT_R >= 1`
  - `INPAINT_R = 1` 时退化为不做 resample、只做 hard overwrite 的版本


### 第一版落地顺序

1. 先写 `utils/inpainting_utils.py`。
2. 再复制并修改 `evaluate_control_error_mas_window.py` 为 inpaint 版入口。
3. 再补 `run_eval_mas_window_inpaint.sh`。
4. 先在 `num_eval_demos=1, num_eval_envs=1` 下做 smoke test。
5. 再检查输出的 `success_once / success_at_end / ce_all / ce_success / ce_failed` 是否完整。

### 验证点

- `mas_inpaint / mas_inpaint_mask` shape 是否稳定为 `(B, pred_horizon, 7)`。
- known action 是否从 `obs_horizon - 1` 位置开始对齐，而不是从 `0` 开始。
- inpainting 内部始终使用 normalized 动作；只在 `env.step()` 前反归一化。
- `INPAINT_R=1` 时流程应能稳定退化为无 resample 版本。
- rollout 结束后 json 中必须有：
  - `success_once`
  - `success_at_end`
  - `ce_all`
  - `ce_success`
  - `ce_failed`
