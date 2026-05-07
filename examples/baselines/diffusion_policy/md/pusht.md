# 兼容DrawTriangle-v1和PushT-v1

*这是一个和codex的交互文档，请你完成我在文档中布置的任务，但不要动我写的东西，只在指定位置做增删改，并且注意保持语言简练明白*

最终目标：用当前diffusion_policy中的三条pipeline跑Push-T和DrawTriangle的两个任务。由于机械臂设置不同以及其他task-specific
的内容，需要对代码做一些修改才能跑通这两个任务。

## 当前不兼容点调研

请你阅读当前pipeline，然后在代码库中调研这两个任务的具体信息，记录在这里，然后对比代码库中现有信息，记录下不兼容点。

### 任务具体信息

- 当前 diffusion_policy 三条 pipeline：
  - Baseline DP：`run_baseline.sh` -> `train_baseline.py` -> `evaluate/evaluate_baseline.py`，直接用 raw demo 的 `obs/actions` 训练。
  - Subgoal Condition DP：`run_subgoal_condition.sh` -> `data_preprocess.py` -> `train_subgoal_condition.py`，要求预处理数据含 `actions/mas/mask`。
  - MAM / MAS Window DP：`run_mam.sh` -> `data_preprocess.py` 或 `data_preprocess_mixed.py` -> `train_mam.py`，要求预处理数据含 `actions/mas/mask`，评估还依赖 STPM。
- PushT-v1：
  - 源码：`mani_skill/envs/tasks/tabletop/push_t.py`；默认机器人 `panda_stick`，默认 horizon 100。
  - 支持 `pd_ee_pose`、`pd_ee_delta_pose` 等控制；实测 `pd_ee_pose/pd_ee_delta_pose` action dim = 6，`pd_joint_pos` action dim = 7。
  - 默认 `base_camera` 为 128x128；`rgb+depth` + `FlattenRGBDObservationWrapper` 后 state dim = 21。
  - `state` obs 会额外含 `goal_pos`、`obj_pose`，state dim = 31；视觉 obs 只额外含 `tcp_pose`。
  - 成功条件：T block 覆盖目标区域面积 >= 90%。
- DrawTriangle-v1：
  - 源码：`mani_skill/envs/tasks/drawing/draw_triangle.py`；默认机器人 `panda_stick`，默认 horizon 300。
  - 支持 `pd_ee_pose`、`pd_ee_delta_pose` 等控制；实测 `pd_ee_pose/pd_ee_delta_pose` action dim = 6，`pd_joint_pos` action dim = 7。
  - 默认 `base_camera` 为 320x240；`rgb+depth` + `FlattenRGBDObservationWrapper` 后 state dim = 21。
  - `state` obs 会额外含 `goal_pose`、`tcp_to_verts_pos`、`goal_pos`、`vertices`，state dim = 49；视觉 obs 只额外含 `tcp_pose`。
  - 成功条件：画出的点覆盖三角形参考点，距离阈值源码为 `THRESHOLD = 0.025`。
- 数据现状：
  - `mani_skill/utils/download_demo.py` 列出 PushT-v1 和 DrawTriangle-v1 官方 demo 源。
  - 本地 `demos/` 当前未发现 PushT-v1 / DrawTriangle-v1 h5/json。
  - motionplanning 里有 DrawTriangle solver，默认产出 `pd_joint_pos` demo；未发现 PushT motionplanning solver。

### 不兼容点

- 动作维度硬编码不兼容：
  - `data_preprocess/utils/progress_utils.py` 固定 `MAS_ACTION_DIM = 7`、`MAS_STEP_DIM = 8`。
  - `data_preprocess/utils/mask_utils.py` 要求 action dim = 7。
  - `data_preprocess.py` / `data_preprocess_mixed.py` 要求原始 `actions` 为 `(T, 7)`，且只归一化前 6 维。
  - `train_mam.py` 固定 `MAS_STEP_DIM = 8`；`train_subgoal_condition.py` 固定 `mas[:, :7]`；CE / inpainting 工具也有多处 `:7`。
  - PushT / DrawTriangle 若用当前默认 `pd_ee_pose`，action dim 是 6，会直接卡在预处理和 MAS 相关代码。
- 控制模式和 demo 不一定一致：
  - 当前 run 脚本默认 `CONTROL_MODE=pd_ee_pose`，但 DrawTriangle motionplanning solver 生成的是 `pd_joint_pos`。
  - 如果使用官方 raw demo 或 motionplanning demo，需要先确认 json 里的 `control_mode`，必要时 replay/转换到目标控制模式。
- 数据输入不兼容：
  - 三条 pipeline 都需要带 `obs` 的 h5；官方下载源通常是 raw dataset，需要 replay 成 `rgb` 或 `rgb+depth` 观测后再训练。
  - Subgoal/MAM 还需要预处理生成 `mas/mask/meta(action_min/max,state_min/max)`。
- 任务默认参数需要显式切换：
  - `ENV_ID`、`RAW_DEMO_H5/JSON`、`PREPROCESSED_ROOT_DIR`、`PREPROCESSED_DATA_PREFIX` 目前默认指向 PickCube / data_1。
  - `MAX_EPISODE_STEPS` 对 PushT 应为 100，对 DrawTriangle 应为 300；当前脚本默认 100，会截断 DrawTriangle。
- 视觉尺寸不同但模型基本可接：
  - PushT 是 128x128，DrawTriangle 是 320x240；`PlainConv(pool_feature_map=True)` 有 adaptive pool，训练/评估同任务同相机时可跑。
  - 若混用不同任务或不同 replay 相机，需要固定 camera config，避免同一训练集内图像尺寸不一致。
- MAM 额外依赖 STPM：
  - `run_mam.sh` 默认 STPM 路径是 PickCube；PushT / DrawTriangle 需要对应任务的 STPM config 和 ckpt。
  - STPM 的 camera/state 维度必须和 diffusion eval env 的 `base_camera`、state schema 对齐。


## 兼容性修改

如果数据集replay正确（仍然使用pd_ee_pose），参数填写正确，按理说，只有action_dim会有不一致情况。因此请你做一下修改：

1、三个sh文件的ENVID参数后面加一行action_dim参数。可选7（有夹爪）或6（无夹爪）。

2、据此修改一系列原本写死的参数设置（如mas维度从8变为7，action从7变为6）

3、STPM应该是不受影响的，如果STPM训练受action-dim的影响的话，先向我汇报，不改代码

在这里写修改记录：

- 三个入口脚本已新增 `ACTION_DIM`：
  - `run_baseline.sh`
  - `run_subgoal_condition.sh`
  - `run_mam.sh`
  - 默认 `7`；PushT / DrawTriangle 使用 `pd_ee_pose` 时应设 `ACTION_DIM=6`。
- 三个训练入口已新增 `--action-dim` 并校验 env action space：
  - `train_baseline.py`
  - `train_subgoal_condition.py`
  - `train_mam.py`
  - 训练集 actions 维度也会在 dataset 加载阶段提前校验。
- 预处理链路已支持 6/7 维：
  - `data_preprocess.py`
  - `data_preprocess_mixed.py`
  - `mask_utils.py`
  - `progress_utils.py`
  - `mas_dim = action_dim + 1`；7 维时仍只归一化前 6 维，保留旧 gripper 语义。
- MAM / subgoal / CE / inpainting 中原本写死的 `7/8` 维度已改为按 `action_dim` 推导。
- STPM 检查结果：当前 STPM 代码只搜索到 state/rgb/progress 相关逻辑，未发现训练或推理直接依赖 action dim；因此未改 STPM。
- 验证：
  - `bash -n` 通过三个入口脚本。
  - `py_compile` 通过修改过的 Python 文件。
  - 已用 synthetic `(T, 6)` action 验证 mask、MAS progress、MAS window shape：`mas=(T,7)`。
