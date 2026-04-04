# Debug4 

**我们在这个文档里就当前这个debg问题进行交互，我会每次在这个文档里写上进一步指示，然后按我的要求进行，在指定位置添加我需要的答复**

**这个文档里我写的内容你不要动，只按照我的要求增改**

## 问题概述

我在本机上重新跑了run_train_mas_window_dit.sh，跑了100k多的iter，但是sucessrate几乎是0,（训练数据见runs/PickCube_window_dit_new）说明pipeline绝对有问题（因为之前的dp_old里，dit100k已经可以达到四五十的成功率）。现在需要找到bug。

---

## 2026-03-31 Debug 结论：当前问题的本质

这次后面又补了 3 个直接实验，结论要更新，已经被排除的点：

下面这些点现在都**不是主因**：

- `window=0/0` 分支本身没有残留条件  
  checkpoint / `obs_conditioning` / `mas_long_window` / `mas_short_window` 都核过了，pure DiT 分支确实只吃 `rgbd + state`

- eval seed / H5 / JSON 没有错位  
  我重新按真实评估构造 `gym.make -> FlattenRGBDObservationWrapper -> FrameStack`，对齐了全部 `100` 个 eval seed：
  - `eval reset state` 归一化后和 `eval.h5 traj_i/obs/state[0]` **逐维完全一致**
  - `rgb/depth` 也基本逐像素一致，depth 完全相同，rgb 只有渲染量化级别的微小差异

- `is_grasped` 首帧污染不是当前主因  
  这个 bug 真实存在，也已经修过 raw / preprocessed 数据，但修完后纯 DiT 现象没有根本变化

- `num_demos=100` 的“前100前缀偏置”不是决定性主因  
  这个问题也真实存在，但我做了不改原代码的随机子集副本，重跑同口径 pure DiT：
  - baseline `PickCube_window_dit_new @ 10k`：`success_once = 0.01`
  - random-subset 副本 `@ 10k`：`success_once = 0.00`
  说明“前100前缀偏置”是个问题，但它**不是这次 pure DiT 跑坏的决定性根因**

- state normalization 不是主因  
  我又做了一个 `no_state_norm` 副本，只改成：
  - 训练读 raw state
  - agent 完全跳过 state normalization
  
  结果：
  - `PickCube_window_dit_no_state_norm_10k @ 10k`：`success_once = 0.01`
  
  这和 baseline 基本一样，所以 **state normalization 不是根因**

- 当前 zero-window 评估链不是在压低 pure DiT 成绩  
  我把同一个 pure DiT checkpoint `runs/PickCube_window_dit_new/checkpoints/best_eval_success_once.pt`
  用两种评估器在同一批 `100` 个 eval seed 上直接对打：
  - 当前 `evaluate_mas_window(window=0/0)`：`success_once = 0.03`, `success_at_end = 0.01`
  - 旧式简单 rollout 评估：`success_once = 0.01`, `success_at_end = 0.01`
  
  结论：
  - **现在这条 zero-window mas eval 链没有把 pure DiT 评低**
  - 如果真要说差异，反而是当前 eval 略高一点

这次一路 debug 到现在，可以明确写成：

- **当前没有证据支持“zero-window / pure DiT 实现本身有 bug”**
- **seed、H5、JSON、state、rgb/depth、eval 链都已经逐项核过**
- **前100前缀偏置、state normalization、is_grasped 首帧污染都是真问题，但都不是这次失败的决定性主因**


## blank_window 实验

现在主要的问题就是，当我使用train_mas_window.py训练的时候，取long和short window大于零时，训练效果非常不错，但取两者都为0时，dit完全训不起来，所以非常值得怀疑。
（因为dit本身肯定是没有问题的，毕竟在diffusion_policy_old里的代码能跑到40-50%，用的是old文件夹里run_train_rgbd.sh的脚本）

现在我需要你：

- 做一个实验（就叫dit_blank_window），给dit加上一些mas_window(只用long取16的horizon，conv后变为64维，shortwindow horizon取8)，然后在输入模型前把window里的值改为白噪音，然后其他参数保持和examples/baselines/diffusion_policy/run_train_mas_window_dit.sh里一致。具体来说，你先改一版副本文件（sh、py），用副本文件跑测试。（大概运行100k，环境用maniskill_py311），然后实验结束后结论写在## blank_window 实验部分。

- 已按要求创建副本文件：
  - `examples/baselines/diffusion_policy/train_mas_window_blank_window.py`
  - `examples/baselines/diffusion_policy/run_train_mas_window_dit_blank_window.sh`

- 当前副本配置：
  - `EXP_NAME=dit_blank_window`
  - `long_window_horizon=16`
  - `mas_long_conv_output_dim=64`
  - `short_window_horizon=8`
  - 其余训练主参数对齐 `run_train_mas_window_dit.sh`
  - 我额外把 `CAPTURE_VIDEO=false` 作为运行假设，只是为了减少 IO，不改模型/数据/评估逻辑

- 简要结果：
  - `dit_blank_window` 训练到 `70k` 时，`loss` 已明显下降，但 `success_once` 最高只有 `0.03`，`success_at_end` 最高只有 `0.02`，整体仍接近 pure DiT 的低水平。
  - checkpoint 维度检查说明 long-window 和 short-window 分支都真实接进了模型，不存在“分支没生效”的问题。

- 简要结论：
  - 这个实验基本排除了“只是多了 window 分支/额外参数量，DiT 就会变好”这一解释。
  - 更合理的结论是：`window>0` 版本真正受益的是 **真实 future window 提供的有效条件信息**；一旦把 window 值替换成白噪音，虽然 `loss` 还能继续下降，但 rollout 成功率起不来。
  - 因此，当前 `window>0` 的高成绩更像是在吃真实 future window 的信息红利，而不只是网络结构更强。

## old dit 兼容性分析

- 结论先写：**能做当前 rollout 测试，但前提是按 zero-window 方式去构建当前 evaluator 的 Agent；不能直接拿当前默认的 window 参数去载入。**

- 兼容性证据：

  - old agent 有 `247` 个 key，current zero-window agent 有 `251` 个 key。
  - current 版本只多出 4 个 buffer：
    - `state_norm_min`
    - `state_norm_max`
    - `state_norm_scale`
    - `has_state_normalizer`
  - 其余公共参数的 shape **全部一致**。
  - 我又直接做了 `new_zero_window.load_state_dict(old.state_dict(), strict=False)`：
    - `missing = ['state_norm_min', 'state_norm_max', 'state_norm_scale', 'has_state_normalizer']`
    - `unexpected = []`
  - 这说明如果 checkpoint 真的是 old `train_rgbd.py` 产物，那么它在**window 关闭**时可以被当前 rollout agent 正常吃下，主体网络并不存在结构不兼容问题。

- 额外注意：
  - 当前仓库里的 `examples/baselines/diffusion_policy_old/run_train_rgbd.sh` 脚本本身是陈旧的，它现在指向了不存在的 `examples/baselines/diffusion_policy/train_rgbd.py`。
  - 所以“用这个脚本在当前 repo 里重新复现 old 训练”这件事现在本身就不成立；但**已经训练好的 old ckpt**，只要按 zero-window 参数去建当前 evaluator，仍然可以做 rollout 测试。

## old dit 实验

- 我用old dit的sh文件（examples/baselines/diffusion_policy_old/run_train_rgbd.sh）重新做了实验，结果发现到6万iter基本sucess为0（runs/dit_old.），但是之前用同样的脚本训练却能达到很好的效果（见runs/dit_old_original里面有之前训练的曲线图以及checkpoints文件），请你仔细分析为什么，尤其是我怀疑数据集之类的影响了训练或者是其他任何原因，总之给我找出来，不要动现有代码，需要跑测试可以建副本，最终结论用简洁的话写在## old dit 实验里。


- 这次我重点核了 `examples/baselines/diffusion_policy_old/train_rgbd.py` 的**数据契约**，结论已经比较明确：
  - **当前 `runs/dit_old` 跑坏的决定性原因，不是 DiT 结构本身，而是当前 `demos/data_1/data_1.h5` 里的 `actions` 已经不符合 old rgbd pipeline 的假设。**

- 直接证据：
  - `old train_rgbd.py` 训练时**直接读取 H5 里的 `actions`** 做 diffusion，没有训练内归一化；
  - 但 rollout 时又会对前 6 维执行  
    `a = min + 0.5 * (a + 1) * (max - min)`  
    也就是说它默认 **H5 里的训练动作已经是归一化到 `[-1, 1]` 的动作**。
  - 而我检查了当前 `demos/data_1/data_1.h5`，其中 `actions` 明显是 **raw action**，不是 normalized action：
    - 第 4 维角度范围就是 `[-pi, pi]`
    - 用 `data_1_norm.json` 去归一化后，前 6 维才会落回接近 `[-1, 1]`
  - 这说明现在是：
    - 训练：吃 raw action
    - rollout：却按 normalized action 再做一次 denorm
    - 训练/评估动作空间已经**彻底错位**

- 我又做了一个数值核对：如果把当前 H5 里的 raw action 直接代入 old rollout 的 denorm 公式，前 6 维会被二次映射；其中旋转那一维的平均绝对偏差约是 **6.72 rad**。这已经不是“小漂移”，而是足以把 rollout 直接打坏的量级。

- 这也和 old 目录里的其他证据一致：
  - `concat_mas_h5.py` 明确写了 old 拼接数据时，最终写进 `actions` 的应该来自 **normalized dataset**
  - 当前 `data_1.h5` 修改时间是 **2026-03-31 17:29**，`data_1_norm.json` 修改时间是 **2026-04-02 20:56**，说明这套数据/统计量最近确实被重写过

- 所以这次 `runs/dit_old` 和以前 `runs/dit_old_original` 对不上的最合理解释是：
  - **要么以前的 `data_1.h5` 还是 normalized-actions 版本**
  - **要么当时跑 `dit_old_original` 时并没有使用现在这套 denorm 口径**
  - 但无论是哪一种，至少可以确定：**“现在这份 `data_1.h5` + 现在这版 old rgbd train/eval” 这一组合本身就是不一致的，不能指望复现以前的 old dit 好结果。**

- 一句话结论：
  - **old dit 现在训不起来，主因就是当前数据文件里的 action 已经是 raw space，但 old pipeline 仍按 normalized-action pipeline 在 rollout；根因是数据契约错位，不是 DiT 本体坏了。**

- 我又继续检查了你接下来要用的 `demos/data_1/data_1_normed.h5`，结论是：
  - **`data_1_normed.h5` 本身没有发现数据契约问题，它和 old `train_rgbd.py` / `data_1_norm.json` 是匹配的，可以用来训练 old dit。**

- 具体证据：
  - `data_1_normed.h5` 和 `data_1.h5` 的轨迹数、轨迹长度、obs / success / terminated / truncated 都对齐；
  - `data_1_normed.h5` 的前 6 维 action 范围已经落在 `[-1, 1]` 附近，且和  
    `data_1_norm.json` 的 min-max 公式逐值对上：
    - 逐元素绝对误差均值大约在 `1e-8`
    - 最大误差大约在 `1e-7`
  - gripper 维仍保持 `{-1, 1}`，没有额外漂移
  - 我还直接做了一个训练层 smoke test：
    - `dataset -> dataloader -> agent.compute_loss -> backward -> optimizer.step`
    - 全链路通过，`loss = 1.0671`，没有出现 shape / dtype / denorm 相关报错

- 所以对这一步可以直接下结论：
  - **如果你要复现 old dit，训练集应该用 `data_1_normed.h5`，而不是当前 raw-action 的 `data_1.h5`。**
  - **从我现在的检查结果看，用 `data_1_normed.h5` 训练 old dit 这件事本身没有发现新问题。**

- 额外备注（非阻塞）：
  - `examples/baselines/diffusion_policy_old/train_rgbd.py` 里 `compute_loss` 依赖模块级全局 `device`；
  - 这在“被 import 后单独调用”时会报 `NameError`，但**正常按脚本方式启动训练不会受影响**，因为主程序里会先定义这个全局变量。


## 对比结论

**这步至关重要，是找到bug的最关键步骤，一定要仔细认真的找**

- 来到了我们debug最关键阶段，现在新旧dit都跑出结果了，有很明显的差异，runs/PickCube_window_dit_new 里sucess几乎为0,而runs/dit_old中同样是100kiter能到百分之二十左右。

- 现在需要你细致调研这个区别，把真正造成结果差异的因素找出来，注意之前通过实验已经排除了很多不是关键的原因，我需要你一直找，直到把bug找出来。

- 如果你有对bug的猜想，可以新建window dit的副本控制变量后做对比实验，来验证你的猜想。

- 得到结论后简要写在## 对比结论里。

- 目前更接近本质的差异，不是 DiT 主体代码，而是：  
  **new 现在评的是“held-out demo-conditioned generalization”，old 之前评的不是这个东西。**

- 先说 `held-out eval` 到底改变了什么：
  - new `train_mas_window.py` 强制要求 `test_demo_path`，并从 `eval.json` 里读取整套 `eval_reset_seeds`；评估时每个 rollout 都用 **held-out demo 对应的 reset seed**，同时喂这条 held-out demo 对应的 MAS window
  - 这意味着 new 的 `success_once` 测的是：  
    **模型能不能在“没参与训练的 demo / seed”上，靠当前观测去正确落地一条 unseen condition**
  - old `train_rgbd.py` 的 `evaluate.py` 只是直接 `eval_envs.reset()`，**没有 held-out split、没有 demo metadata、没有固定 seed 列表**
  - 所以 old 日志里的 `~20%` 本质上只是一个**非 held-out、非 demo-aligned** 的环境成功率；它不能回答“模型在 unseen demo 上是否泛化”这个问题

- 更关键的是 `overlap`：
  - new 当前配置下，训练用的 train split 和评估用的 eval split 是严格分开的，所以 `new_train_first100` 与 `new_eval100` 的交集是 **0**
  - 但 old 默认训练的前 `100` 条 demo，对应的真实训练 reset seeds 与当前 `new eval100` 的 reset seeds 交集有 **21** 条：
    - `[5, 12, 18, 26, 36, 41, 45, 46, 55, 57, 59, 64, 69, 70, 78, 82, 86, 90, 94, 95, 99]`
  - 这件事非常关键，因为它给出一个很强的解释：
    - **如果 old policy 主要是在记住见过的 demo，而对真正 unseen 的 held-out seed 几乎不会做，那么它在这套 eval 上天然就可能落在 `21/100 ≈ 21%` 这个量级**
    - 这个数字和你现在看到的 `old ≈ 20%` 几乎重合

- 所以目前最稳的结论应该改成：
  - **`old 20% vs new 0%` 不能直接当成“new 实现有 bug、old 真泛化更强”的证据**
  - 它更像是在说明：  
    **以前 old 策略的泛化能力可能比我们以为的差得多，`20%` 这个量级本身就可能主要由 overlap / memorization 解释，而不是真正的 held-out 泛化**

- 这不等于说 new zero-window 一定没 bug；它只说明：
  - **当前最强的差异解释，首先是 held-out generalization 口径变化，以及 old/new 对 eval overlap 的本质差别**
  - 如果接下来还要继续追 zero-window 的真实 bug，最有判别力的实验应该是：  
    **直接把 old ckpt 放到当前这 100 个 held-out seeds 上逐条评估，并按 overlap / non-overlap 分组看 success。**


## 进一步核实与验证

- 这一步我按“**reset seed 口径**”把 old / new 当前实际对比所对应的训练集、检验集都重新列了一遍。这里的口径是：
  - **old** 指 `examples/baselines/diffusion_policy_old/run_train_rgbd.sh` 当前默认配置：`NUM_DEMOS=100`
  - **new** 指 `examples/baselines/diffusion_policy/run_train_mas_window_dit.sh` 当前默认配置：`NUM_DEMOS=100, NUM_EVAL_DEMOS=100`

- old 训练集对应的 reset seeds：
  - `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100]`

- new 训练集对应的 reset seeds：
  - `[0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23, 24, 25, 27, 28, 29, 30, 31, 32, 33, 34, 35, 37, 38, 39, 40, 42, 43, 44, 47, 48, 49, 50, 52, 53, 54, 56, 58, 60, 61, 62, 63, 65, 66, 67, 68, 71, 72, 73, 74, 75, 76, 77, 79, 80, 81, 83, 84, 85, 87, 88, 89, 91, 92, 93, 96, 97, 98, 100, 101, 103, 105, 106, 107, 108, 109, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 125]`

- new 检验集对应的 reset seeds：
  - `[5, 12, 18, 26, 36, 41, 45, 46, 55, 57, 59, 64, 69, 70, 78, 82, 86, 90, 94, 95, 99, 102, 104, 110, 124, 128, 141, 143, 146, 154, 161, 162, 163, 165, 166, 199, 206, 208, 213, 228, 230, 233, 245, 247, 252, 263, 275, 279, 293, 295, 297, 299, 305, 313, 322, 328, 333, 335, 344, 354, 359, 364, 371, 387, 390, 396, 401, 405, 423, 433, 439, 443, 449, 454, 460, 461, 464, 474, 478, 503, 506, 507, 510, 513, 514, 520, 525, 526, 527, 529, 530, 532, 542, 563, 566, 577, 581, 584, 585, 595]`

- old 检验集（`train_rgbd.py` 当前 `evaluate.py`）：
  - **没有固定 demo seed 列表**；代码是直接 `eval_envs.reset()`，不读任何 `eval.json` / metadata。
  - 当前默认配置下（`physx_cpu, num_eval_envs=10, num_eval_episodes=100`），old eval 的 reset seed 不是 100 个独立样本，而是：
    - **10 个 unique seeds**
    - **每个 seed 重复 10 次**
    - seed 范围约在 `[40194941, 2143436973]`
  - 这 10 个 unique seeds 是：
    - `[40194941, 1642857692, 2143436973, 56726128, 486979249, 1096548337, 214636784, 768078775, 796319576, 228722706]`
  - 原因是 ManiSkill 单环境 CPU env 在 unseeded `reset()` 时，会从固定 `RandomState(2022)` 里顺序采样 `randint(2**31)`；而 old 的 10 个 eval worker 都是独立单环境 CPU env，所以每轮 10 个 worker 会拿到同一个 seed。
  - 影响：
    - **old eval 不是 held-out demo eval**
    - **old eval 也不是均匀随机覆盖 seed 空间**
    - **它只是对 10 个内部随机 seed 反复测了 10 次**
  - 这些高位整数 seed **不意味着 old dit 没有问题**，也**不能证明没有泄漏**；它只说明 old eval 的口径和当前 new 的 held-out demo-seed eval 完全不是一回事，因此 old 的 `~20%` 不能直接当成“old 泛化正常”的证据。

- 这一步核实以后，可以把 seed 层面的核心差异说得更具体：
  - **old 训练 seeds 是一段前缀式 seed 集合**（大致在 `0~100` 这一段）
  - **new 训练 seeds 是 train split 的前 100 条**
  - **new 检验 seeds 是固定的 100 条 held-out demo seeds**
  - **old 检验则根本不是固定 seed 列表**

- 因此现在“old 20% vs new 0%”这个差异里，至少已经可以明确看到两类 seed 口径差别：
  - **new 在固定 held-out seeds 上测**
  - **old 不在固定 held-out seeds 上测**
  - 以及 old 训练 seeds 与 new 检验 seeds 之间确实存在前面算过的 **21 条重合**


## overlap eval 对比

- 我按当前 new 的 held-out `eval100` seeds，对两个 old ckpt 都做了逐 seed cross-eval。为了不动现有代码，我新建了副本脚本：
  - `examples/baselines/diffusion_policy_old/eval_rgbd_overlap_ckpt.py`

- 评估口径：
  - checkpoint 1：`runs/dit_old/checkpoints/best_eval_success_once.pt`
  - checkpoint 2：`runs/dit_old_original/best_eval_success_once.pt`
  - eval seeds：当前 new 的 held-out `100` 个 reset seeds
  - overlap 定义：是否落在 old 前 `100` 条训练 demo 的 reset seed 集合里（共 `21` 条）

- `dit_old`（训练到约 100k 的那个）结果：
  - 总体：`success_once = 0.24`，`success_at_end = 0.20`
  - overlap 21 条：`success_once = 1.00`，`success_at_end = 0.9048`
  - non-overlap 79 条：`success_once = 0.0380`，`success_at_end = 0.0127`
  - non-overlap 里只有 `3/79` 条 seed 曾经成功一次：`[124, 206, 252]`
  - non-overlap 里只有 `1/79` 条 seed 末尾成功：`[252]`

- `dit_old_original`（训练很久的那个）结果：
  - 总体：`success_once = 0.66`，`success_at_end = 0.35`
  - overlap 21 条：`success_once = 0.9524`，`success_at_end = 0.8571`
  - non-overlap 79 条：`success_once = 0.5823`，`success_at_end = 0.2152`

- 最关键的结论：
  - **old 100k ckpt 基本就是在吃 overlap。** 它在 overlap seeds 上几乎全会，但一到 non-overlap 就几乎全灭。
  - 这说明之前看到的 `old ~20%`，非常可能主要就是由这 `21/100` 的 overlap 撑起来，而不是它真的具备了稳定的 held-out 泛化能力。
  - **old_original 虽然确实比 100k ckpt 强，而且在 non-overlap 上也有一定能力，但它同样存在非常明显的 overlap 优势。**
  - 所以现在可以更有把握地说：  
    **old/new 的核心差异首先是 eval protocol 和 overlap，而不是“new zero-window 已经被证明有实现 bug”。**


## 进一步指示

- 我现在要做最后一步的对比尝试，我需要你修改olddp的数据加载逻辑，把olddp里的train和eval与newdp完全对齐，训练集还是与之前相同（data_1_norm.h5），但是选取100条训练数据时，完全按照newdp的resetseed选取；同样的，eval时，完全选用newdo测试集中的100个seed编码，然后我将再次进行对比实验，来获得确凿的证据。
修改:
train_rgbd.py/evaluate.py/....sh

- 已完成对齐改动：
  - `examples/baselines/diffusion_policy_old/diffusion_policy/utils.py` 现已支持按指定 `traj_ids` 读取 H5，不再只能“取前 N 条”。
  - `examples/baselines/diffusion_policy_old/train_rgbd.py` 新增了 `source_demo_metadata_path / train_seed_reference_metadata_path / eval_demo_metadata_path` 三个入口：
    - 训练时会先读取 new train metadata 的前 `100` 个 `reset_seed`，再回到 `demos/data_1/data_1.json` 里按 seed 映射出 old `data_1_normed.h5` 对应的 `traj_id`，然后只加载这些轨迹训练。
    - 评估时会直接读取 new eval metadata 的前 `100` 个 `reset_kwargs`，逐条 `env.reset(**reset_kwargs)` 做 rollout，而不再走 old 默认的随机 `eval_envs.reset()`。
  - `examples/baselines/diffusion_policy_old/diffusion_policy/evaluate.py` 已支持 `reset_kwargs_list`，因此 old eval 现在可以和 new 一样按固定 seed 列表逐条评。
  - `examples/baselines/diffusion_policy_old/run_train_rgbd.sh` 默认已经切到这套对齐口径：
    - 训练参考 metadata：`demos/data_1_preprocessed/3D_points_0.1/data_1_3D_points_0.1_train.json`
    - 评估 metadata：`demos/data_1_preprocessed/3D_points_0.1/data_1_3D_points_0.1_eval.json`
    - source metadata：`demos/data_1/data_1.json`
    - `NUM_EVAL_ENVS` 默认改成了 `1`，避免 fixed reset list 被并行 env 打乱

- 轻量验证已通过：
  - 训练对齐不是“简单取前 100 条”，而是真按 seed 回查 old H5。比如：
    - `seed 52 -> traj 51`
    - `seed 53 -> traj 52`
    - `seed 54 -> traj 53`
    - 这说明代码已经正确跨过了 `seed=51` 缺失造成的错位。
  - 我直接在 `maniskill_py311` 环境里验证过：
    - 对齐后的训练前 10 个 `traj_id` 为 `[0, 1, 2, 3, 4, 6, 7, 8, 9, 10]`
    - 它对应的训练前 10 个 `reset_seed` 也正是 `[0, 1, 2, 3, 4, 6, 7, 8, 9, 10]`
    - 对齐后的评估前 10 个 `reset_seed` 为 `[5, 12, 18, 26, 36, 41, 45, 46, 55, 57]`
  - 同时我也验证了 `data_1_normed.h5` 确实能按这些映射后的 `traj_id` 正常读取。

- 现在可以直接用：
  - `examples/baselines/diffusion_policy_old/run_train_rgbd.sh`
  - 跑出来的 old 训练/评估口径就已经与当前 new 的 `3D_points_0.1` train/eval split 对齐。
