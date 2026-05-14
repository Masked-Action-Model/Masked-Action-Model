# Diffusion Policy 与 STPM 使用说明

本文档整理两个模块的常用入口：

- Diffusion Policy：`examples/baselines/diffusion_policy`
- STPM：`STPM`

建议先激活环境：

```bash
conda activate maniskill_py311
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
```

环境配置可参考：

```bash
environment_maniskill_py311.yml
```

## 1. 数据要求

Diffusion Policy 和 STPM 默认读取 ManiSkill demo h5/json：

```text
xxx.h5
xxx.json
```

常见 h5 内容：

```text
traj_*/actions
traj_*/obs/agent/*
traj_*/obs/extra/*
traj_*/obs/sensor_data/*/rgb
traj_*/obs/sensor_data/*/depth
```

`ACTION_DIM` 支持：

```text
7: Panda 带 gripper
6: Panda stick / 无 gripper
auto: 从 h5 自动推断，部分脚本支持
```

## 2. Baseline Diffusion Policy

入口：

```bash
examples/baselines/diffusion_policy/run_baseline.sh
```

示例：

```bash
RAW_DEMO_H5=demos/PlaceSphere-v1/PlaceSphere-v1.rgb.pd_ee_pose.physx_cpu.h5 \
RAW_DEMO_JSON=demos/PlaceSphere-v1/PlaceSphere-v1.rgb.pd_ee_pose.physx_cpu.json \
ENV_ID=PlaceSphere-v1 ACTION_DIM=7 OBS_MODE=rgb \
NUM_DEMOS=5 TOTAL_ITERS=20000 \
bash examples/baselines/diffusion_policy/run_baseline.sh
```

关键参数：

```text
NOISE_MODEL=Transformer|Unet
ACTION_ROBUST_MARGIN=0.01   # 可选，action 用 1%/99% robust min-max + clip
USE_PREPARED_SPLIT=true|false
DEMO_PATH / EVAL_DEMO_PATH  # 使用已有 train/eval split
RAW_DEMO_H5 / RAW_DEMO_JSON # 自动切分 raw demo
```

## 3. Subgoal Condition DP

入口：

```bash
examples/baselines/diffusion_policy/run_subgoal_condition.sh
```

示例：

```bash
RAW_DEMO_H5=demos/data_1/data_1.h5 \
RAW_DEMO_JSON=demos/data_1/data_1.json \
ENV_ID=PickCube-v1 ACTION_DIM=auto \
MASK_TYPE_LIST='["3D_points"]' \
MASK_RATIO_LIST='[0.5]' \
TOTAL_ITERS=100000 \
bash examples/baselines/diffusion_policy/run_subgoal_condition.sh
```

train/eval 可使用不同 mask：

```bash
TRAIN_MASK_TYPE_LIST='["random_mask"]' \
TRAIN_MASK_RATIO_LIST='[0.2]' \
EVAL_MASK_TYPE_LIST='["3D_points"]' \
EVAL_MASK_RATIO_LIST='[0.5]' \
bash examples/baselines/diffusion_policy/run_subgoal_condition.sh
```

## 4. MAM / MAS Window DP

入口：

```bash
examples/baselines/diffusion_policy/run_mam.sh
```

示例：

```bash
RAW_DEMO_H5=demos/data_1/data_1.h5 \
RAW_DEMO_JSON=demos/data_1/data_1.json \
ENV_ID=PickCube-v1 ACTION_DIM=auto \
MASK_TYPE_LIST='["random_mask","3D_points"]' \
MASK_RATIO_LIST='[0.2,0.5]' \
MASK_COMPOSITION_LIST='[0.5,0.5]' \
PREPROCESS_MASK_ASSIGN_MODE=composition \
STPM_CONFIG_PATH=STPM_PickCube/config.yaml \
STPM_CKPT_PATH=STPM_PickCube/checkpoints/reward_best.pt \
bash examples/baselines/diffusion_policy/run_mam.sh
```

train/eval mask 分开配置：

```bash
TRAIN_MASK_TYPE_LIST='["random_mask"]' \
TRAIN_MASK_RATIO_LIST='[0.2]' \
EVAL_MASK_TYPE_LIST='["3D_points"]' \
EVAL_MASK_RATIO_LIST='[0.5]' \
bash examples/baselines/diffusion_policy/run_mam.sh
```

MAM 评估依赖 STPM：

```text
STPM_CONFIG_PATH
STPM_CKPT_PATH
```

## 5. Mask 配置

常用 mask type：

```text
random_mask
points
3D_points
pose_motion_planning
pose_AnyGrasp
2D_partial_trajectory
local_planner
full
none
```

mixed mask：

```bash
MASK_TYPE_LIST='["random_mask","3D_points"]'
MASK_RATIO_LIST='[0.2,0.5]'
MASK_COMPOSITION_LIST='[0.5,0.5]'
```

含义：

```text
MASK_RATIO_LIST:
  random_mask / points / 3D_points / pose_motion_planning -> retain_ratio
  2D_partial_trajectory / local_planner -> mask_seq_len

MASK_COMPOSITION_LIST:
  mixed composition 中不同 mask type 分配给 demo 的比例
```

渐变 mask：

```text
multi_random    -> random_mask
multi_points    -> points
multi_3D_points -> 3D_points
multi_pose      -> pose_motion_planning
```

示例：

```bash
MASK_TYPE_LIST='["multi_random","multi_points"]' \
MASK_RATIO_LIST='[[0.2,0.4],[0,0.5]]' \
MASK_COMPOSITION_LIST='[0.5,0.5]' \
bash examples/baselines/diffusion_policy/run_mam.sh
```

如果某个 `multi_*` mask 分到 `n` 条 demo，会按 `np.linspace(start, end, n)` 给每条 demo 分配递增 retain ratio。

## 6. 预处理输出

Subgoal / MAM 会自动调用：

```text
data_preprocess/data_preprocess.py
data_preprocess/data_preprocess_mixed.py
```

输出位置由以下参数控制：

```text
PREPROCESSED_ROOT_DIR
PREPROCESSED_DATA_DIR
PREPROCESSED_DATA_PREFIX
PREPROCESSED_FILE_STEM
DEMO_PATH
EVAL_DEMO_PATH / TEST_DEMO_PATH
```

预处理 h5 会包含：

```text
actions  # normalized expert action
mas      # masked action sequence + progress
mask     # known action mask
obs/state
meta/action_min
meta/action_max
meta/state_min
meta/state_max
```

## 7. STPM 训练

通用入口：

```bash
STPM/run_train_stpm.sh
```

示例：

```bash
DATASET_PATH=demos/data_1/data_1.h5 \
TASK_NAME=pickcube \
TASK_DESCRIPTION='pick up the cube and place it at the target' \
OUTPUT_DIR=STPM_PickCube \
STATE_PATHS=auto \
CAMERA_NAMES=auto \
BATCH_SIZE=32 NUM_EPOCHS=2 \
bash STPM/run_train_stpm.sh
```

只生成 config 和 state normalizer，不训练：

```bash
PREPARE_ONLY=true \
DATASET_PATH=demos/data_1/data_1.h5 \
OUTPUT_DIR=STPM_PickCube \
bash STPM/run_train_stpm.sh
```

输出：

```text
OUTPUT_DIR/config.yaml
OUTPUT_DIR/state_norm.json
OUTPUT_DIR/checkpoints/reward_best.pt
```

任务快捷脚本：

```bash
bash STPM/run_train_peginsertion_stpm.sh
bash STPM/run_train_pullcubetool_stpm.sh
```

## 8. STPM 离线评估

入口：

```bash
STPM/run_eval_offline.sh
```

示例：

```bash
STPM_CKPT_PATH=STPM_PickCube/checkpoints/reward_best.pt \
STPM_CONFIG_PATH=STPM_PickCube/config.yaml \
EVAL_DATASET_PATH=demos/stpm_eval/PickCube-v1/eval.h5 \
NUM_EVAL_DEMO=20 \
bash STPM/run_eval_offline.sh
```

常用参数：

```text
BATCH_SIZE
NUM_WORKERS
DEVICE
OUTPUT_DIR
PROGRESS_BAR=true|false
```

## 9. 常用调参

Diffusion Policy：

```text
EXP_NAME
SEED
CUDA=true|false
OBS_MODE=rgb|rgb+depth
NOISE_MODEL=Transformer|Unet
TOTAL_ITERS
BATCH_SIZE
LR
OBS_HORIZON
ACT_HORIZON
PRED_HORIZON
NUM_DATALOAD_WORKERS
ACTION_ROBUST_MARGIN
```

MAM 特有：

```text
LONG_WINDOW_BACKWARD_LENGTH
LONG_WINDOW_FORWARD_LENGTH
SHORT_WINDOW_HORIZON
MAS_LONG_ENCODE_MODE=2DConv|1DConv
MAS_LONG_CONV_OUTPUT_DIM
LOSS_MODE=average|weighted
LOSS_MASK_AREA_WEIGHT
INPAINTING=true|false
```

STPM：

```text
DATASET_PATH
TASK_NAME
TASK_DESCRIPTION
OUTPUT_DIR
STATE_PATHS
CAMERA_NAMES
VISION_CKPT
N_OBS_STEPS
FRAME_GAP
D_MODEL
N_LAYERS
N_HEADS
TOTAL_STEPS
NUM_EPOCHS
```

## 10. 最小推荐流程

1. 训练 STPM：

```bash
DATASET_PATH=demos/data_1/data_1.h5 OUTPUT_DIR=STPM_PickCube bash STPM/run_train_stpm.sh
```

2. 训练 MAM：

```bash
RAW_DEMO_H5=demos/data_1/data_1.h5 \
RAW_DEMO_JSON=demos/data_1/data_1.json \
STPM_CONFIG_PATH=STPM_PickCube/config.yaml \
STPM_CKPT_PATH=STPM_PickCube/checkpoints/reward_best.pt \
bash examples/baselines/diffusion_policy/run_mam.sh
```

3. 训练 Subgoal：

```bash
RAW_DEMO_H5=demos/data_1/data_1.h5 \
RAW_DEMO_JSON=demos/data_1/data_1.json \
bash examples/baselines/diffusion_policy/run_subgoal_condition.sh
```
