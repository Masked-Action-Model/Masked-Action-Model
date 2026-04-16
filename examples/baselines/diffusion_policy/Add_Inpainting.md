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
   r            : 每个anchor位置额外执行的resampling轮数
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

        (B) 在当前anchor附近做 r 轮 resampling，每一轮：jump forward j步，再 reverse j步回来
        if t > 0:
            for _ in range(r):

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
  r               : 每个 anchor 位置额外执行的 resampling 次数
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

        for _ in range(r):
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
（注意 `j>=1`，`r>=0`，`r=0` 时即回退到 hard overwrite 版本）。

用一个utils文件放相关工具函数，尽量把主要的工具函数都方里面，方便后面迁移为在线版。

用一个py脚本实现加载checkpoints文件、数据集，然后在maniskill环境中进行rollout，每一步inference执行一次inpainting，注意之前progress以及window的相关设计（详见debug1.md），然后记录once、end成功率，以及计算三个ce值（很关键）。


### 总体策略

- 先做离线版，不改训练流程，不改 `compute_loss()`。
- 先不直接改现有 `Agent.get_action()`，避免影响现有 `mas_window` 评估；离线版单独走一条 inpainting rollout 路径。
- 最大化复用现有代码：
  - checkpoint / env / STPM / CE 统计复用 `evaluate_control_error_mas_window.py`
  - dual-window condition 复用 `build_dual_mas_window_condition_batch(...)`
  - 成功率与 CE 计算复用 `utils/control_error_utils.py`

### 当前实现文件

- `examples/baselines/diffusion_policy/utils/inpainting_utils.py`
  - 放 RePaint 动作版的核心工具函数。
- `examples/baselines/diffusion_policy/eval_inpaint.py`
  - 离线入口脚本。
- `examples/baselines/diffusion_policy/run_eval_inpaint.sh`
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

### `eval_inpaint.py` 实现思路

- 以 `eval_ce.py` 为模板复制一份，不从零写。
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

### `run_eval_inpaint.sh` 参数设计

- 基础参数直接对齐现有离线 CE 入口：
  - `CHECKPOINT_PT_PATH`
  - `EVAL_DEMO_PATH`
  - `EVAL_DEMO_METADATA_PATH`
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
  - `JUMP_LENGTH`
  - `NUM_RESAMPLE`
- 约束：
  - `JUMP_LENGTH >= 1`
  - `NUM_RESAMPLE >= 0`
  - `NUM_RESAMPLE = 0` 时退化为不做 resample、只做 hard overwrite 的版本
  - 当前实现不单独传 `ACTION_NORM_PATH`，动作反归一化统计直接从 `EVAL_DEMO_PATH` 读取


### 第一版落地顺序

1. 先写 `utils/inpainting_utils.py`。
2. 再复制并修改 `eval_ce.py` 为 inpaint 版入口。
3. 再补 `run_eval_inpaint.sh`。
4. 先在 `num_eval_demos=1, num_eval_envs=1` 下做 smoke test。
5. 再检查输出的 `success_once / success_at_end / ce_all / ce_success / ce_failed` 是否完整。

### 验证点

- `mas_inpaint / mas_inpaint_mask` shape 是否稳定为 `(B, pred_horizon, 7)`。
- known action 是否从 `obs_horizon - 1` 位置开始对齐，而不是从 `0` 开始。
- inpainting 内部始终使用 normalized 动作；只在 `env.step()` 前反归一化。
- `NUM_RESAMPLE=0` 时流程应能稳定退化为无 resample 版本。
- rollout 结束后 json 中必须有：
  - `success_once`
  - `success_at_end`
  - `ce_all`
  - `ce_success`
  - `ce_failed`

## 离线版实验结果分析

指令：
我先后对3个预训练好的checkpoints.py文件进行了离线inpainting的实验。
mask参数分别是random0.1、random0.2、3dpoints0.2
他们的原始checkpoints文件、events文件以及eval-inpainting后的json记录文件分别在runs文件夹下的window_random_0.1,window_random_0.2,window_3dpoints_0.2
(重点放在前两个，3dpoints那个很多实验还没跑完)
针对每个checkpoints文件，我小范围改动了j和r的值，做了一些测试，得到相关ce值、successonce、successend等的数据

现在需要你来分析inpainting方法在动作inference中的作用（正作用还是负作用），不同masktype对inpainting的影响，jr值的影响（重点分析jr值增大后的看似发生的负作用，以及其原理），结合ce探究动作生成和图像生成的不同之处，尝试分析现象背后的原理，得出一些初步的结论，并且列出下一步建议的实验计划。

### 结果整理与初步分析

### 先和原始 no-inpainting 结果做对比

这里优先做“同一 checkpoint”的公平比较。

原因：

- 当前离线 inpainting 实验用的都是 `best_eval_success_once.pt`。
- 因此最公平的 no-inpainting 基线，不是训练全程任意一步的最好值，而是 `tfevents` 中对应 `best_eval_success_once` 那个 step 的原始评估结果。

| run | no-inpaint 基线来源 | success_once | success_end |
|---|---|---:|---:|
| `random0.1` | `best_eval_success_once` 对应 step=`340000` | 0.43 | 0.19 |
| `random0.2` | `best_eval_success_once` 对应 step=`70000` | 0.86 | 0.37 |
| `3dpoints0.2` | `best_eval_success_once` 对应 step=`265000` | 0.66 | 0.34 |
| `3dpoints0.5` | `best_eval_success_once` 对应 step=`55000` | 0.62 | 0.32 |

补充说明：

- `random0.1` 训练过程中 `success_end` 的全程最好值是 `0.24`。
- `random0.2` 训练过程中 `success_end` 的全程最好值是 `0.58`。
- `3dpoints0.2` 训练过程中 `success_end` 的全程最好值是 `0.40`。
- `3dpoints0.5` 训练过程中 `success_once` 的最后平滑值大约是 `0.52`，和你提供的观察一致；它的全程最好 `success_end` 是 `0.36`，出现在 step=`80000`。

基于这个公平基线，当前结论先给出来：

- `random0.1`：使用 inpainting 后整体是正作用。即使最朴素的 `j=1,r=0`，也把 `success_once` 从 `0.43` 提到 `0.50`，把 `success_end` 从 `0.19` 提到 `0.24`。
- `random0.2`：使用 inpainting 后也是明显正作用。最佳已测配置 `j=5,r=5` 把 `success_once` 从 `0.86` 提到 `0.96`，把 `success_end` 从 `0.37` 提到 `0.56`。
- `3dpoints0.2`：使用 inpainting 后整体是负作用。最好的已测结果仍只有 `0.58/0.24`，低于 no-inpaint 的 `0.66/0.34`。
- `3dpoints0.5`：使用 inpainting 后不是明显正作用。对同一个 `best_eval_success_once` checkpoint 而言，`j=1,r=0` 的 `success_once` 与 no-inpaint 基线持平，`success_end` 则从 `0.32` 小幅降到 `0.30`；但它仍显著好于 `3dpoints0.2`，说明提高 3D 点密度确实在缓解问题。

### 结果汇总

#### 1. `random0.1`

| j | r | success_once | success_end | ce_all | ce_success | ce_failed |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 0 | 0.50 | 0.24 | 0.00422 | 0.000736 | 0.00770 |
| 1 | 5 | 0.42 | 0.21 | 0.00701 | 0.000667 | 0.01160 |
| 5 | 3 | 0.39 | 0.21 | 0.00653 | 0.001524 | 0.00972 |
| 5 | 5 | 0.48 | 0.26 | 0.00341 | 0.001317 | 0.00534 |
| 10 | 5 | 0.43 | 0.25 | 0.00566 | 0.000705 | 0.00939 |

结论：

- 大多数大 `j/r` 组合都会让 `success_once` 下降。
- 但 `j=5,r=5` 是一个例外：`success_end` 从 `0.24` 升到 `0.26`，`ce_all` 从 `0.00422` 降到 `0.00341`，说明适中的 jump 配合较强 resample 可能存在一个局部 sweet spot。
- 单纯把 `r` 拉大但 `j` 很小（`j=1,r=5`）是负作用最明显的组合之一。
- 如果和原始 no-inpaint 基线比，`random0.1` 上至少 `j=1,r=0` 和 `j=5,r=5` 是明确正收益；但也有一些大 `j/r` 组合会把 `success_once` 拉回到基线附近甚至略低，说明对这类随机 mask，inpainting 是“有帮助但不稳”。

#### 2. `random0.2`

| j | r | success_once | success_end | ce_all | ce_success | ce_failed |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 0 | 0.95 | 0.48 | 0.001615 | 0.001339 | 0.00687 |
| 1 | 5 | 0.95 | 0.55 | 0.001603 | 0.001477 | 0.00400 |
| 5 | 5 | 0.96 | 0.56 | 0.001439 | 0.001387 | 0.00268 |

结论：

- `random0.2` 上 inpainting 明显更稳定，且总体是正作用。
- `success_once` 基本不掉，`success_end` 明显提升。
- `ce_failed` 明显下降，说明 inpainting 主要在“挽救失败轨迹”这件事上有效。
- `j=5,r=5` 是当前这组里最好的已测组合。
- 如果和原始 no-inpaint 基线比，提升非常明显，说明在已知动作较密时，inpainting 不只是“微调”，而是在系统性改善 rollout。

#### 3. `3dpoints0.2`

| j | r | success_once | success_end | ce_all | ce_success | ce_failed |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 0 | 0.58 | 0.24 | 0.00981 | 0.000833 | 0.02220 |
| 1 | 3 | 0.44 | 0.18 | 0.00998 | 0.001385 | 0.01673 |
| 1 | 5 | 0.50 | 0.20 | 0.01105 | 0.001121 | 0.02098 |
| 1 | 8 | 0.50 | 0.16 | 0.01122 | 0.000977 | 0.02146 |

结论：

- 当前已测结果里，`3dpoints0.2` 上 resample 仍然整体是负作用。
- 相比 `r=0`，`r=3/5/8` 的 `success_once` 和 `success_end` 都没有超过基线。
- `r=3` 时 `ce_failed` 降得最多，但成功率掉得也最明显，说明它更像是在“贴已知点”，而不是在提升任务可执行性。
- 继续把 `r` 提到 `5/8` 后，`success_once` 相比 `r=3` 有一定回升，但 `ce_all` 和 `success_end` 仍劣于 `r=0`，而且 `ce_failed` 也开始反弹，说明更强 resample 并没有真正修复问题，只是在不同失败模式之间移动。
- 更关键的是，如果和原始 no-inpaint 基线 `0.66/0.34` 比，连 `r=0` 的 hard overwrite 版本都已经是负作用，这说明问题不只是“resample 太强”，而是当前这类 sparse 3D 条件本身就不适合直接拿来做动作 inpainting。

#### 4. `3dpoints0.5`

| j | r | success_once | success_end | ce_all | ce_success | ce_failed |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 0 | 0.62 | 0.30 | 0.01698 | 0.00307 | 0.03967 |
| 5 | 1 | 0.44 | 0.24 | 0.02303 | 0.00318 | 0.03864 |

结论：

- `3dpoints0.5` 的表现明显好于 `3dpoints0.2`，说明仅仅把 3D 点的保留比例从 `0.2` 提高到 `0.5`，就足以显著改善动作 inpainting 的可用性。
- 在当前已测结果里，`j=1,r=0` 明显优于 `j=5,r=1`；也就是说，对 `3dpoints0.5` 这类较密但仍不完整的条件，hard overwrite 版本已经够用，而较大的 jump 反而在破坏前缀稳定性。
- 如果和同 checkpoint 的 no-inpaint 基线 `0.62/0.32` 比，`j=1,r=0` 更接近“近似持平但略负”，而不是明确正收益；不过它仍然远好于 `3dpoints0.2`，说明提高点密度是有效方向。

补充观察：

- 对当前参与评估的 50 条 demo，`3dpoints0.5` 的 `num_known_points` 均值约为 `38.8`，而 `3dpoints0.2` 只有 `15.22`。
- 这里的 `num_known_points` 统计的是“有至少一个已知维度的时间步数”，不是已知坐标总数。
- 这说明 `3dpoints0.5` 的改善很可能首先来自“条件更密”，而不一定意味着只靠 `xyz` 三维点本身就足够了。


### `j/r` 增大后为何常出现负作用

我认为主要有 5 个原因：

1. `forward jump` 会主动把当前已经部分去噪好的动作 chunk 再次推回更高噪声层。
   对图像来说，这种做法有机会提升多样性并重新协调局部内容；但对动作来说，这会直接破坏近未来动作前缀的稳定性。

2. 动作生成更看重 prefix 的精确性，而图像生成更看重整体一致性。
   当前 rollout 实际只执行 `pred_horizon` 里的前 `act_horizon` 段，因此最重要的是前几步动作是否稳定、平滑、动态可执行。`j/r` 增大时，对整段 chunk 反复加噪和回退，很容易先伤到最关键的前缀。

3. RePaint 在图像里覆盖的是“真实已知像素”，但这里覆盖的是“部分动作维度/部分时间步”。
   这两者信息完备性不同。图像 known region 往往就是 ground truth；动作 known region 往往只是稀疏控制点。模型要补的不是“剩余像素”，而是完整可执行控制序列，所以问题更病态。

4. 动作成功依赖环境动力学闭环，而不只是轨迹相似度。
   图像中像素更像 ground truth，通常就是更好；但动作中 `ce` 更低，不一定就更容易抓取/放置成功。尤其在 `3dpoints0.2` 里，`ce_failed` 下降但 success 下降，就是这个现象的直接证据。

5. 当前模型训练时并没有显式见过大量“反复 jump-forward / reverse-back”的推理轨迹。
   当 `j/r` 较大时，inference path 和训练分布偏差会更大，模型可能会在未充分训练覆盖的噪声路径上输出不稳定动作。

### 结合 CE 对“动作生成 vs 图像生成”的差异理解

这批结果说明：动作任务里，CE 是重要指标，但不是充分指标。

- 在 `random0.2` 上，inpainting 既降低了 `ce_failed`，也提升了 `success_end`，说明这里 CE 改善和任务成功大体一致。
- 在 `random0.1` 上，部分组合会让 `ce_all` 变好，但 `success_once` 仍下降，说明“更像演示”不等于“更容易完成任务”。
- 在 `3dpoints0.2` 上，这种背离更明显：失败轨迹的 CE 下降，但整体成功率下降，说明模型在 sparse known points 周围生成了更像演示的局部运动，却没有补出正确的姿态、抓取时机和后续动作衔接。
- 在 `3dpoints0.5` 上，又出现了另一种背离：`j=1,r=0` 的 success 明显高于 `3dpoints0.2`，但 `ce_all` 反而更高。这说明已知点变多后，模型可能为了满足更多局部 3D 点约束，牺牲了一部分全局轨迹相似度，但闭环任务反而更容易成功。

因此，对动作 inpainting 来说：

- CE 更像是“条件跟随程度”指标；
- success 更像是“动力学可执行性 + 任务达成性”指标；
- 两者必须联合看，不能直接把图像 inpainting 的直觉照搬过来。

### 初步结论

1. 和原始 no-inpaint 基线相比，inpainting 不是普遍正作用，它对 mask type 很敏感。
2. 当 known action 较密、且覆盖维度较完整时，inpainting 更可能带来明确正收益。
   `random0.2` 是当前最明显的正例。
3. `random0.1` 也整体偏正作用，但对 `j/r` 更敏感，说明它处于“有帮助但不稳定”的中间区域。
4. `3dpoints` 系列表明，“条件密度”和“条件完整性”都重要。
   `3dpoints0.2` 明显过稀，因此连 hard overwrite 都可能是负作用；
   而 `3dpoints0.5` 在密度提高后，至少已经从“明显负作用”走到了“接近 no-inpaint 基线的中间状态”。
5. 当 known action 稀疏、且只覆盖局部几何维度时，不只是 resample，连 hard overwrite 本身都可能是负作用。
   `3dpoints0.2` 是当前最明显的反例。
6. `j/r` 不是越大越好；大多数情况下，过大的重采样会破坏已经成形的动作前缀。`3dpoints0.2` 上从 `r=3` 加到 `r=5/8` 也没有翻正，而 `3dpoints0.5` 上 `j=5,r=1` 也明显差于 `j=1,r=0`，说明“继续加大 jump/resample”并不是主要出路。
7. 当前结果更像在说明：动作版 RePaint 需要更保守的使用方式，尤其要优先保护近未来 prefix 的稳定性，并且前提是 known condition 本身要足够强。

### 下一步建议实验计划

#### A. 先补全最关键的 ablation

- 在三个 mask setting 上都补一个统一的小网格：
  - `j in {1, 3, 5}`
  - `r in {0, 1, 3, 5}`
- 所有实验固定同一批 demo index、同一随机种子，避免采样方差混进结论。

#### B. 明确区分 “jump 的作用” 和 “resample 的作用”

- 固定 `r=0`，只扫 `j`。
  这样可以确认：仅仅改 anchor 间隔是否就会影响结果。
- 固定 `j=1`，只扫 `r`。
  这样可以单独测“重复 resample 本身”到底是帮忙还是捣乱。

#### C. 优先在 `random0.2` 上做更细搜索

- 因为它是当前唯一稳定显示正收益的 setting。
- 可以继续测：
  - `j=3,r=5`
  - `j=5,r=3`
  - `j=7,r=3`
- 目标是看最佳点是否在 `j≈3~5`、`r≈3~5` 这一带，而不是继续盲目增大。

#### D. 对 `3dpoints` 系列做“条件密度/完整性”验证

- 继续把 `3dpoints0.2 / 3dpoints0.5` 放在一起分析。
- 增加一个包含姿态/夹爪信息的变体 mask，与纯 `3dpoints` 对照。
- 如果补全姿态信息后 inpainting 进一步变好，就能直接支持“当前瓶颈主要来自 known condition 不完整”这一判断。
- 如果只提高点密度、不补姿态也能持续变好，就说明“条件过稀”同样是主要瓶颈。

#### E. 增加 prefix-stability 指标

- 除了 `ce_all / ce_success / ce_failed`，建议额外记录：
  - 只在执行前 `act_horizon` 上算的 CE
  - 连续 action 差分范数，例如 `||a_t - a_{t-1}||`
  - orientation / gripper 维度单独 CE
- 这样可以直接检验“大 `j/r` 是否主要破坏了动作前缀平滑性”。

#### F. 若后续继续改算法，优先尝试更保守的动作版 RePaint

- 只对 `obs_horizon-1` 之后的更远 future 段做强 resample，尽量少动最前面的执行前缀。
- 或者让 `r` 随 timestep 衰减：高噪声层稍多，低噪声层少做甚至不做。
- 或者限制 overwrite/resample 只作用在 known 较密的局部时间段，而不是整段 chunk。

### 当前最值得保留的判断

- `random0.2`: 相比原始 no-inpaint 基线，inpainting 是明确正作用，值得继续挖。
- `random0.1`: 相比原始 no-inpaint 基线，inpainting 也是正作用，但效果高度敏感，说明方法并不稳，需要更细参数和更强约束设计。
- `3dpoints0.2`: 相比原始 no-inpaint 基线，inpainting 是负作用。新增 `r=5/8` 后结论依旧没变，更应该先补条件信息或改更保守的推理策略。
- `3dpoints0.5`: 比 `3dpoints0.2` 乐观很多，但按同 checkpoint 公平比较，`j=1,r=0` 目前仍只是“接近 no-inpaint 基线”，还不能算明确正作用；它更适合被定义为“值得继续补更多 `j/r` 组合验证的中间状态”。

把结论写在下面：
