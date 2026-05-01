# Reconstuct Relative Action Space

**我们在这个文档里就一些问题进行交互，我会每次在这个文档里写上进一步指示，然后你按我的要求执行**
**这个文档里我写的内容你不要直接删除，只按照我的要求增加或者改动**
**这部分的改动针对diffusionpolicy文件夹中的train_window_mixed脚本链路进行，创建副本链路，命名train_relative_action.py以及相应一个sh及一个evaluate文件**

除了下述改动，其他细节与原train_window_mixed链路保持一致

## pipeline：

trainingtime：
1、训练数据集仍收集pd_ee_pose形式，将所有已知信息统一到absolute action space
2、对数据集里的acition，根据action_pred_horizon，对每一帧动作做向后切片（H，7），然后将这个切片转化为以首帧为基准的relative_action_chunk
3、对数据集里的mas，根据mas_window设置，对每一帧切片并截取mas_window,然后根据当前帧的绝对pose（注意不从数据集的action中读取，而是从对应时间步的obs/tcp_pose中读取，并且将四元数转为统一的欧拉角），计算得到相对当前帧的masked relative action space window（简称mras_window）
4、训练时，对于每一个时间步，输入3中计算的mras_window以及其他condition，模型预测噪声，将预测出的relative_action_chunk与2中计算的真值chunk计算loss并更新参数

inferencetime：
1、对于测试数据集中的每条测试，读取初始值
2、对于每帧，读取当前仿真环境中的真实末端绝对pose，根据这个pose与绝对mas_window做差，转化为mras_window
3、对于每一帧输入mras_window等condition，模型去噪得到当前帧的relative_action_chunk
4、将relative_action_chunk做差得到delta_action_chunk，并用pd_ee_deltapse的control_mode输入控制器

## code plan

### 0. 核心语义约定

1. 原始数据、预处理数据、MAS 仍保持 `pd_ee_pose` 的 absolute action space：前 6 维是 `[x, y, z, roll, pitch, yaw]`，第 7 维是 gripper。
2. 新链路只在训练采样和评估 rollout 时动态转换空间，不修改现有 `data_preprocess_mixed.py` 的默认产物。
3. 为保证 training / inference 一致，`relative_action_chunk` 和 `mras_window` 使用同一个 reference：当前帧真实 TCP pose，即 `obs/extra/tcp_pose`，不是 `actions[t]`。
4. 文档中“以首帧为基准”在代码中解释为：以当前 sample 的执行起点帧为基准。执行起点为 `exec_start = start + obs_horizon - 1`，reference 从该帧 observation 的 `tcp_pose` 读取。
5. `tcp_pose` 是 `[x, y, z, qw, qx, qy, qz]`；action / MAS 是 `[x, y, z, euler_xyz]`。所有 relative / delta 旋转必须通过 quaternion / rotation matrix 计算，不能直接 Euler 相减。
6. 第 7 维 gripper 不参与 pose relative / delta 差分，始终沿用对应时间步的 gripper command。
7. 默认 eval control mode 先按本文需求使用 `pd_ee_delta_pose`；若后续发现多步 chunk 因实际 TCP 未精确到达目标而漂移，再保留 `pd_ee_target_delta_pose` 作为 ablation。

### 1. 新增文件

1. `examples/baselines/diffusion_policy/train_relative_action.py`
   - 从 `train_mas_window_mixed.py` 复制。
   - 保留 mixed mask、one-demo-multi-mask、双 MAS window、STPM progress、online eval、CE 统计、视频归档等逻辑。
   - 只替换 dataset action target、MAS condition、agent inference action 输出的空间转换。

2. `examples/baselines/diffusion_policy/run_train_relative_action.sh`
   - 从 `run_train_window_mixed.sh` 复制。
   - 最后一行调用 `train_relative_action.py`。
   - 默认 `EXP_NAME=PickCube_relative_action_mixed`。
   - 默认 `CONTROL_MODE=pd_ee_delta_pose`。
   - 仍复用 `data_preprocess_mixed.py` 生成的 absolute-normalized mixed 数据。

3. `examples/baselines/diffusion_policy/evaluate/evaluate_relative_action.py`
   - 从 `evaluate_mas_window.py` 复制。
   - 每步 rollout 从当前 env observation 中读取真实 TCP pose，构造当前 reference。
   - 将 absolute MAS window 动态转换为 MRAS window。
   - 调用 agent 得到 relative chunk，再转成 `pd_ee_delta_pose` action chunk 后执行。

4. `examples/baselines/diffusion_policy/evaluate/evaluate_relative_action_mixed.py`
   - 从 `evaluate_mas_window_mixed.py` 复制。
   - 保持按 `mask_type_slot` 分组、视频采样、progress curve、rollout records 的返回格式。
   - 内部调用 `evaluate_relative_action(...)`。

5. `examples/baselines/diffusion_policy/utils/relative_action_utils.py`
   - 集中放 absolute / relative / delta 转换函数，训练和评估共用。

### 2. 转换工具设计

`relative_action_utils.py` 实现：

1. `denormalize_abs_action(action_norm, action_min, action_max)`
   - 只反归一化前 6 维。
   - 第 7 维 gripper 原样保留。

2. `normalize_abs_action(action_abs, action_min, action_max)`
   - 只用于调试或必要回写。
   - 只归一化前 6 维。

3. `tcp_pose_to_abs_euler(tcp_pose)`
   - 输入 `[x, y, z, qw, qx, qy, qz]`。
   - 输出 `[x, y, z, roll, pitch, yaw]`，旋转统一为 controller 使用的 XYZ Euler。

4. `absolute_pose_to_relative(abs_pose, ref_pose)`
   - position：`target_pos - ref_pos`。
   - rotation：`q_rel = q_target * inv(q_ref)`，再转 XYZ Euler。
   - 支持 shape `(..., 6)`。

5. `absolute_action_sequence_to_relative(abs_seq, ref_pose)`
   - 对前 6 维调用 `absolute_pose_to_relative`。
   - 第 7 维 gripper 保持 `abs_seq[..., 6]`。

6. `relative_sequence_to_delta_actions(relative_seq)`
   - `delta_0 = relative_0`。
   - `delta_k = relative_k relative_{k-1}^{-1}` 的 pose 差分语义：
     - position：`rel_pos_k - rel_pos_{k-1}`。
     - rotation：`q_delta_k = q_rel_k * inv(q_rel_{k-1})`。
   - 第 7 维 gripper 直接取当前步 gripper，不差分。

7. `normalize_delta_for_pd_ee_delta_pose(delta_seq, pos_scale=0.1, rot_scale=0.1)`
   - Panda `pd_ee_delta_pose` 默认 position / rotation 范围都是 `[-0.1, 0.1]`。
   - 前 3 维除以 `pos_scale` 并 clip 到 `[-1, 1]`。
   - 旋转 3 维除以 `rot_scale` 并按 norm / clip 约束。
   - gripper 保持 controller 需要的原始 gripper command。

8. `integrate_delta_actions(ref_pose, delta_seq)`
   - 调试用，从 reference 和 delta action 重新积分出 absolute target path。
   - 用于验证 `absolute -> relative -> delta -> integrate` 能还原原 absolute path。

### 3. Dataset 修改点

在 `SmallDemoDataset_MasWindowDiffusionPolicy` 副本中：

1. 初始化时从 h5 `meta/action_min`、`meta/action_max` 读取 absolute action normalization stats。
2. 预处理 observation 时，在 `convert_obs(...)` 前从原始 `obs_traj_dict["extra"]["tcp_pose"]` 抽取每帧 TCP pose，保存为 `trajectories["tcp_pose"][traj_idx]`。
3. 若数据集中没有 `extra/tcp_pose`，直接报错；不要从 normalized state 里猜字段位置。
4. `__getitem__` 中保持原逻辑切：
   - `obs_seq`
   - `mas_long_window`
   - `mas_short_window`
   - `action_mask_seq`
5. 计算当前 reference：
   - `current_step = start + obs_horizon - 1`
   - `ref_idx = clamp(current_step, 0, L)`
   - `ref_pose = tcp_pose_to_abs_euler(tcp_pose_traj[ref_idx])`
6. 将 `act_seq` 从 normalized absolute action 反归一化到 physical absolute action。
7. 用 `ref_pose` 构造：
   - `relative_action_seq = absolute_action_sequence_to_relative(abs_action_seq, ref_pose)`
8. 对 `mas_long_window` / `mas_short_window`：
   - 前 7 维从 normalized absolute MAS 反归一化到 physical absolute action。
   - 对 mask 为 true 的点转成相对 `ref_pose` 的 MRAS。
   - mask 为 false 的点保持 mask value。
   - 第 8 维 progress 保持原值和原 mask 语义。
9. `action_mask_seq` 继续沿用原 mask 切片，但其语义变为 relative-space mask；第 7 维 gripper mask 不变。
10. 返回：
    - `"actions"`：`relative_action_seq`，作为 diffusion clean target。
    - `"action_mask"`：relative-space mask。
    - `"observations"`：其中 MAS window 已替换成 MRAS window。

### 4. Agent 修改点

1. `ALGO_NAME = "BC_Diffusion_rgbd_DiT_RelativeAction_Mixed"`。
2. `Args` 新增：
   - `relative_pos_scale: float = 0.1`
   - `relative_rot_scale: float = 0.1`
   - `delta_pos_scale: float = 0.1`
   - `delta_rot_scale: float = 0.1`
   - `control_mode: str = "pd_ee_delta_pose"`
3. `compute_loss`：
   - 默认 clean sample 是 `relative_action_seq`。
   - 先保持现有 noise-prediction MSE，保证改动最小。
4. `get_action`：
   - diffusion 输出 relative action sequence。
   - 截取执行 chunk：`relative_seq[:, obs_horizon - 1 : obs_horizon - 1 + act_horizon]`。
   - 转成 delta chunk。
   - 按 `pd_ee_delta_pose` action space 归一化 / clip。
   - 返回 env 可直接执行的 normalized delta action chunk。
5. 移除或改名原 absolute action denormalizer 的推理用途；它只在 dataset 和 MRAS 构造中用于 absolute 数据反归一化，不再用于 env action 输出。

### 5. Evaluation 修改点

`evaluate_relative_action.py` 中：

1. reset 后初始化 histories，保持 STPM progress 推理逻辑不变。
2. 每个 rollout step：
   - 从当前 `obs["state"]` 或未归一化原始 observation 中取当前真实 TCP pose。
   - 更稳妥的第一版：在 `FlattenRGBDObservationWrapper` 输出前保留 / 附加 `tcp_pose`，或在 wrapper 后按 PickCube state layout 提取 tcp_pose，并写清检查。
3. 用 STPM progress 从 eval demo 中取 absolute MAS / mask window。
4. 用当前真实 TCP reference 将 absolute MAS window 转成 MRAS window。
5. `obs["mas_long_window"]` / `obs["mas_short_window"]` 写入 MRAS，mask window 保持 mask + progress 语义。
6. `agent.get_action(obs)` 返回 normalized `pd_ee_delta_pose` action chunk。
7. 逐步执行 chunk；由于 `pd_ee_delta_pose` 相对真实当前 EE pose，每一步不需要维护外部 target reference。
8. rollout record 中保存实际执行的 delta action；若要继续计算 CE，可额外保存由 delta 积分得到的 reconstructed absolute target path。

`evaluate_relative_action_mixed.py` 中：

1. 保持 mixed grouping 和 summary 逻辑不变。
2. 将内部调用从 `evaluate_mas_window(...)` 替换为 `evaluate_relative_action(...)`。
3. 返回格式与原 mixed eval 对齐，训练主循环尽量少改。

### 6. Shell 脚本修改点

`run_train_relative_action.sh`：

1. 默认配置：
   ```bash
   EXP_NAME="${EXP_NAME:-PickCube_relative_action_mixed}"
   CONTROL_MODE="${CONTROL_MODE:-pd_ee_delta_pose}"
   ```
2. 新增参数：
   ```bash
   RELATIVE_POS_SCALE="${RELATIVE_POS_SCALE:-0.1}"
   RELATIVE_ROT_SCALE="${RELATIVE_ROT_SCALE:-0.1}"
   DELTA_POS_SCALE="${DELTA_POS_SCALE:-0.1}"
   DELTA_ROT_SCALE="${DELTA_ROT_SCALE:-0.1}"
   ```
3. 追加 CLI：
   ```bash
   --relative-pos-scale "$RELATIVE_POS_SCALE"
   --relative-rot-scale "$RELATIVE_ROT_SCALE"
   --delta-pos-scale "$DELTA_POS_SCALE"
   --delta-rot-scale "$DELTA_ROT_SCALE"
   ```
4. 启动前检查：
   - raw demo / preprocessed meta 的 source control mode 应是 `pd_ee_pose`。
   - eval env `CONTROL_MODE` 应是 `pd_ee_delta_pose`。
   - 若不满足，打印明确 warning 或直接报错。

### 7. 验证计划

1. 静态检查：
   ```bash
   python -m py_compile \
     examples/baselines/diffusion_policy/train_relative_action.py \
     examples/baselines/diffusion_policy/evaluate/evaluate_relative_action.py \
     examples/baselines/diffusion_policy/evaluate/evaluate_relative_action_mixed.py \
     examples/baselines/diffusion_policy/utils/relative_action_utils.py
   ```

2. shell 检查：
   ```bash
   bash -n examples/baselines/diffusion_policy/run_train_relative_action.sh
   ```

3. 转换单测 / smoke check：
   - `tcp_pose -> abs_euler -> relative` 维度和旋转顺序正确。
   - `absolute -> relative -> delta -> integrate` 能还原 absolute target path。
   - gripper 维不差分。
   - mask=false 的 MAS 点保持 mask value。
   - progress 维不参与 pose 转换。

4. dataset sample check：
   - 打印 `relative_action_seq`、`mras_long_window` 的 shape / min / max。
   - 检查第一个执行动作对应的 relative pose 是否相对当前 `obs/tcp_pose`。

5. 最小训练 smoke：
   ```bash
   TOTAL_ITERS=1 \
   NUM_DEMOS=2 \
   NUM_EVAL_DEMOS=2 \
   NUM_EVAL_EPISODES=2 \
   NUM_EVAL_ENVS=1 \
   CAPTURE_VIDEO=false \
   TRACK=false \
   bash examples/baselines/diffusion_policy/run_train_relative_action.sh
   ```

6. 日志必须能确认：
   - train target space 是 relative action。
   - MAS condition space 是 MRAS。
   - env.step 输入是 `pd_ee_delta_pose` normalized delta action。
   - reference 来自当前真实 TCP pose，而不是 action target。

## 归一化设置

### relative pipeline 的 action 归一化链路

当前 relative 链路同时存在两套归一化：

1. **absolute 数据集 min/max 归一化**：来自 `data_preprocess_mixed.py`。
2. **relative / delta 固定尺度归一化**：来自 `relative_pos_scale`、`relative_rot_scale`、`delta_pos_scale`、`delta_rot_scale`，默认都为 `0.1`。

具体链路如下：

1. 原始数据： `pd_ee_pose` `[x, y, z, roll, pitch, yaw, gripper]`
2. 预处理：action 前 6 维归一化到 `[-1, 1]`，gripper 保持原值
3. 预处理：`mas` 基于 normalized absolute action mask + progress 
4. Dataset读取: `action` 反归一化得 physical absolute action
5. Dataset读取: `tcp_pose` 转成 `[x,y,z,roll,pitch,yaw]` 作 reference
6. physical absolute action chunk 通过 quaternion / rotation matrix 转成 physical relative action chunk；gripper 沿用
7. physical relative action归一化：除 `relative_pos_scale`和 `relative_rot_scale`，并clip 到 `[-1,1]`，gripper 不变
8. 训练：normalized relative action space 上做 epsilon MSE
9. 训练 MAS condition：normalized absolute MAS -> 反归一化前 6 维 -> 相对同一个 TCP reference 转 MRAS -> 再用 relative scale 归一化/clip
10. 推理：eval MAS 仍来自 preprocessed normalized absolute MAS，用当前env中的真实 TCP pose 转 MRAS，归一化同上
11. 推理：输出 normalized relative chunk；先用 relative scale 反归一化为 physical relative chunk，再差分成 physical delta chunk

12. physical delta chunk 用 `normalize_delta_for_pd_ee_delta_pose(...)` 转成 env action：位置除以 `delta_pos_scale` 并 clip 到 `[-1,1]`；旋转除以 `-delta_rot_scale` 并按 rotation norm clip。这里使用负号是为了匹配 ManiSkill `PDEEPoseController` 对 normalized rotation 乘 `rot_lower=-0.1` 的实现。

13. env control mode：`pd_ee_delta_pose`；Panda 默认 controller 会把 normalized position action 映射到物理 `[-0.1, 0.1]`，rotation 也按 `rot_lower=-0.1` 转回物理 delta rotation。

14. CE：eval record 保存 reconstructed absolute target path，再用原始 `action_min/action_max` 归一化回 absolute normalized space，与 normalized MAS 控制点比较。


## 过拟合测试

我需要在5demo上做一个过拟合测试（即使用相同的环境种子和demo进行训练和测试），看能不能在较少的iter达到100%正确率，请你1、写一份用于过拟合测试的副本（不改动原训练文件）2、给我运行测试的命令行值令（要求包含一些必要的可改动参数）

过程记录在这里：

### 实现记录

1. 新增训练副本：
   - `examples/baselines/diffusion_policy/train_relative_action_overfit5.py`
   - 从 `train_relative_action.py` 复制，不改动原训练文件。
   - `ALGO_NAME` 改为 `BC_Diffusion_rgbd_DiT_RelativeAction_Mixed_Overfit5`，便于区分日志。

2. 新增启动脚本：
   - `examples/baselines/diffusion_policy/run_train_relative_action_overfit5.sh`
   - 默认 `NUM_DEMOS=5`
   - 默认 `NUM_EVAL_DEMOS=5`
   - 默认 `NUM_EVAL_EPISODES=5`
   - 默认 `NUM_EVAL_ENVS=5`
   - 默认 `TEST_DEMO_PATH="$DEMO_PATH"`
   - 默认 `EVAL_DEMO_METADATA_PATH="$TRAIN_DEMO_METADATA_PATH"`

3. overfit 关键语义：
   - train/eval 使用同一个 preprocessed train h5。
   - eval reset seed 从 train metadata 读取。
   - 因此测试环境种子和 MAS demo 与训练 demo 对齐。

### 推荐运行命令

```bash
TOTAL_ITERS=20000 \
EVAL_FREQ=500 \
SAVE_FREQ=500 \
BATCH_SIZE=64 \
LR=1e-4 \
NUM_DEMOS=5 \
NUM_EVAL_DEMOS=5 \
NUM_EVAL_EPISODES=5 \
NUM_EVAL_ENVS=5 \
CAPTURE_VIDEO=false \
TRACK=false \
bash examples/baselines/diffusion_policy/run_train_relative_action_overfit5.sh
```

### Debug

刚才进行了overfit测试（用的sh脚本里的参数，跑了10000），按理说应该已经过拟合了，说明代码或pipleine绝对有问题。
请你查看分析训练数据，(如loss下降但eval不行是否说明是inference有问题)，细致检查从数据集预处理到加载Dataset到Train到eval的全流程，重点检查新增relativespace相关设置以及归一化等易出错点。

把结论写在这里（注意只记录确凿的有证据的问题，没用的废话别写，以及如果存在疑惑，请刨根问底的查询分析）：

#### 结论 1：relative reference 坐标系用错

确凿问题：
- `pd_ee_pose` action / MAS 是 Panda EE controller frame，也就是 robot root/base frame。
- `obs/extra/tcp_pose` 是 world frame。
- 原 relative 链路直接用 world-frame `tcp_pose` 做 reference，导致 absolute action 与 reference 不在同一坐标系。

证据：
- 在 `demos/data_1_preprocessed/mixed/data_1_mixed_random_mask-s0-r0.2-mix1_b9ed7cd6aa_train.h5` 的 `traj_0` 中：
  - 第 0 帧 action position: `[0.6272538, 0.03801131, 0.18215266]`
  - 第 0 帧 world tcp position: `[0.01225353, 0.03801134, 0.18215224]`
  - x 方向差约 `0.615m`
- PickCube 里 Panda root pose 是 `[-0.615, 0, 0]`，见 `mani_skill/envs/tasks/tabletop/pick_cube.py` 的 `_load_agent(... sapien.Pose(p=[-0.615, 0, 0]))`。
- 把 tcp 从 world frame 转到 Panda base frame 后，前 10 帧 action/tcp position 误差从：
  - 修复前 maxabs: `[0.61500025, 0.00370219, 0.02030376]`
  - 修复后 maxabs: `[0.00304234, 0.00370219, 0.02030376]`

影响：
- 修复前 `relative_pos_scale=0.1` 下，执行 chunk 的 x 方向 relative target **100% 超过 [-1,1] 并被 clip**。
- 这会导致 training target 本身被破坏；loss 可以下降，但模型学到的是被截断后的错误 relative action，eval 不可能稳定过拟合。

修复：
- 新增 `tcp_pose_to_panda_base_abs_euler(...)`：
  ```python
  abs_pose[..., :3] = abs_pose[..., :3] - [-0.615, 0, 0]
  ```
- 已替换：
  - `train_relative_action.py`
  - `train_relative_action_overfit5.py`
  - `evaluate/evaluate_relative_action.py`

修复后位置 clip 检查：
- 执行 chunk relative position normalized 后：
  - max: `[0.5341166, 0.65932375, 1.3712426]`
  - frac > 1: `[0.0, 0.0, 0.02992958]`
- 执行 chunk delta position normalized 后：
  - max: `[0.1642263, 0.20047085, 0.39840326]`
  - frac > 1: `[0.0, 0.0, 0.0]`

验证：
```bash
python -m py_compile \
  examples/baselines/diffusion_policy/utils/relative_action_utils.py \
  examples/baselines/diffusion_policy/evaluate/evaluate_relative_action.py \
  examples/baselines/diffusion_policy/train_relative_action.py \
  examples/baselines/diffusion_policy/train_relative_action_overfit5.py
```

### debug2

我发现另一个bug，如果mas中值是与state中的tcp做relative，那实际上relative action chunk也应该如此，不直接做差，而是与tcppose做差得到relative。
（也就是把Line13中的action首帧值获取方式改掉）

请你修改这个问题（并且仔细检查是否会因此带来其他附加问题）
然后重新给我过拟合测试的命令行（注意我新跑的过拟合测试还是过不了）

#### 结论 2：eval 多步 open-loop chunk 逻辑已恢复

保留的事实：
- Dataset 中 `relative_action_chunk` 已经使用 `ref_pose = obs/extra/tcp_pose[current_step]`，不是 `actions[start]`。
- `mras_window` 也使用同一个 `ref_pose`。
- eval 中第 1 步严格是“当前真实 TCP -> 目标 action”；后续步按相邻预测 target 差分执行，这是恢复后的预期 open-loop chunk 语义。

### debug3

又重新跑了一次过拟合测试，2000iter还是没有反应。请你仔细查看分析训练数据，(如loss下降但eval不行是否说明是inference有问题)，细致检查从数据集预处理到加载Dataset到Train到eval的全流程，重点检查新增relativespace相关设置以及归一化等易出错点，一直追查直到找到问题为止。

把结论写在这里（注意只记录确凿的有证据的问题，没用的废话别写，以及如果存在疑惑，请刨根问底的查询分析）并且先不要改代码，只写分析：

#### debug3 分析结论

1. 现象确认：不是训练完全没学到 noise target。
   - TensorBoard: `runs/PickCube_relative_action_overfit5/events.out.tfevents.1777308456...`
   - `losses/total_loss`: iter 0 = `1.042487`，iter 2400 = `0.010436`。
   - `eval/success_once`: iter 0/500/1000/1500/2000 全部 `0.0`。
   - `eval/ce_all`: iter 0 = `0.075287`，iter 2000 = `0.354077`，训练 loss 下降但 eval 控制误差变坏。

2. overfit5 实际选中的 5 条 train/eval demo 是同一组：
   - dataset: `demos/data_1_preprocessed/mixed/data_1_mixed_random_mask-s0-r0.2-mix1_b9ed7cd6aa_train.h5`
   - `NUM_DEMOS=5, NUM_EVAL_DEMOS=5, SEED=1, composition random_mask#0`
   - 选中 traj index: `[39, 170, 249, 275, 281]`。

Clip问题：


- traj: `[39, 170, 249, 275, 281]`
- scale: `RELATIVE_POS_SCALE=0.1`，`RELATIVE_ROT_SCALE=0.1`，`DELTA_POS_SCALE=0.1`，`DELTA_ROT_SCALE=0.1`

1. `RELATIVE_POS_SCALE=0.1`

   - 影响：action target clip 不算极端，但 MRAS position condition 有约 `7.7%` 已知 scalar 被截断，z 维约 `11.5%`。

2. `RELATIVE_ROT_SCALE=0.1`

   - 影响：这是最严重的问题。yaw 维大量被截断；MRAS 已知 yaw 点约 `33.8%` 被压成边界值，模型输入 condition 和监督 target 都丢失真实几何幅度。

3. `DELTA_POS_SCALE=0.1`
   - 影响：没有截断证据。

4. `DELTA_ROT_SCALE=0.1`
     - norm clip frac: `0.0013`

   - 影响：只有极少数 delta rotation 超过 controller `0.1rad` bound，不是当前主要问题。

更好的 scale 选择：
```bash
RELATIVE_POS_SCALE=0.2
RELATIVE_ROT_SCALE=0.7
DELTA_POS_SCALE=0.1
DELTA_ROT_SCALE=0.1
```

### debug4

还是跑不出来，继续只做分析，先不改代码。

#### debug4 分析结论

1. 新 run 确认已使用 debug3 推荐 scale，但 eval 仍失败。
   - TensorBoard: `runs/PickCube_relative_action_overfit5/events.out.tfevents.1777364338...`
   - hparams:
     - `relative_pos_scale=0.2`
     - `relative_rot_scale=0.7`
     - `delta_pos_scale=0.1`
     - `delta_rot_scale=0.1`
     - `act_horizon=8`
     - `control_mode=pd_ee_delta_pose`
   - `losses/total_loss`: iter 0 = `1.042354`，iter 4400 = `0.009590`
   - `eval/success_once`: iter 0/500/1000/1500/2000/2500/3000/3500/4000 全部 `0.0`
   - `eval/ce_all`: iter 0 = `0.096929`，iter 4000 = `0.241521`

2. overfit5 的 seed/demo 对齐没有问题。
   - 选中 traj: `[39, 170, 249, 275, 281]`
   - 对应 reset seeds: `[47, 209, 302, 332, 340]`
   - 对应 action lengths: `[83, 79, 74, 74, 70]`

3. 确凿问题：`pd_ee_delta_pose + act_horizon=8 open-loop target-to-target delta` 本身不能复现 expert。
   - 测试方式：不用模型，直接读取同 5 条 demo 的 expert absolute `pd_ee_pose` action。
   - 每个 chunk 开始用当前 env TCP 作 reference，构造 relative action chunk。
   - 再按当前 eval 逻辑做 `relative_sequence_to_delta_actions(...)`，用 `pd_ee_delta_pose` 执行完整 chunk。
   - 结果：
     - `act_horizon=1`: success_once `5/5`，success_end `5/5`
     - `act_horizon=2`: success_once `5/5`，success_end `4/5`
     - `act_horizon=4`: success_once `1/5`，success_end `1/5`
     - `act_horizon=8`: success_once `0/5`，success_end `0/5`

4. 反证：基础 delta 转换和 seeds 没坏。
   - 同样 expert action，如果每一步都用当前真实 TCP 重新计算 delta，再用 `pd_ee_delta_pose` 执行：
     - success_once `5/5`
     - success_end `5/5`
   - 说明 `tcp_pose -> panda_base`、absolute action denorm、delta action normalization、reset seed 对齐都不是主要问题。

5. 进一步反证：H=8 target-to-target chunk 与 `pd_ee_target_delta_pose` 匹配。
   - 同样 expert H=8 open-loop chunk，控制模式改为 `pd_ee_target_delta_pose`：
     - success_once `5/5`
     - success_end `5/5`
   - 结论：当前 eval 失败的硬问题是控制语义错配：
     - `relative_sequence_to_delta_actions(...)` 生成的是 target-to-target delta。
     - `pd_ee_delta_pose` 每步相对的是实际当前 EE pose。
     - 多步 open-loop 时实际 EE 没有精确到达上一 target，误差会在 chunk 内积累；`act_horizon=8` 已经足够让 expert 都失败。
     - `pd_ee_target_delta_pose` 才符合 target-to-target delta 的 open-loop 语义。

6. 当前最有证据的修复方向：
   - 如果必须保持 `act_horizon=8` 多步 DP 执行，eval control mode 应改为 `pd_ee_target_delta_pose`。
   - 如果必须保持 `pd_ee_delta_pose`，则 `act_horizon` 需要降到 `1` 或至多 `2`，但这会显著增加 DP 推理频率。
   - 过拟合排错优先建议：保持 `ACT_HORIZON=8`，改 `CONTROL_MODE=pd_ee_target_delta_pose` 做对照，因为 oracle expert 已证明该组合能 5/5 成功。

## 最终结论

最终问题不是模型过拟合能力、seed 对齐、基础坐标变换或归一化的主问题，而是 **relative action chunk 与 `pd_ee_delta_pose` 控制语义不等价**。

`relative_action_chunk` 表达的是一段相对同一个当前 TCP reference 的目标位姿序列；将相邻 relative target 相减得到的 delta，本质上是 **target-to-target delta**。但 `pd_ee_delta_pose` 在环境中每一步都相对真实当前 EE pose 执行，而真实 EE 不会在 open-loop chunk 内精确到达上一 target，因此 target-to-target delta 不能被稳定转换成 `pd_ee_delta_pose` action。`act_horizon=8` 时，即使用 expert action 做 oracle 转换也会失败，说明这是控制模式语义错配，而不是模型预测误差导致的。

最强证据是 **demo 直接转化测试**：完全绕过模型，直接把同一批 expert demo 的 absolute `pd_ee_pose` action 转成 relative chunk，再转成 delta chunk 执行。如果使用 `pd_ee_delta_pose`，`act_horizon=8` 的 expert oracle 结果是 `0/5` 成功；但同样的转换在 `pd_ee_target_delta_pose` 下是 `5/5` 成功。也就是说，即使预测完美，当前 `relative -> delta -> pd_ee_delta_pose` 链路也无法复现原 demo。

因此，若继续使用多步 open-loop chunk，应使用与 target-to-target delta 语义匹配的 `pd_ee_target_delta_pose`；若必须使用 `pd_ee_delta_pose`，则需要每步基于真实当前 EE 重新计算 delta，即把 `act_horizon` 降到 1（或很小）。
