# Franka Real Training

真机 Franka 入口：复用 diffusion_policy / STPM 的训练代码，但跳过 ManiSkill 在线 eval，并适配 25 维 state。

## 数据格式

raw h5 至少包含：

```text
traj_N/actions                         (T, 7)
traj_N/success                         (T,)
traj_N/obs/agent/qpos                  (T+1, 9)
traj_N/obs/agent/qvel                  (T+1, 9)
traj_N/obs/extra/tcp_pose              (T+1, 7)
traj_N/obs/sensor_data/base_camera/rgb (T+1, H, W, 3)
```

建议 `meta` 写入：`env_id`、`control_mode`、`action_dim`、`state_dim=25`、`state_paths`、`camera_names`、`max_episode_steps`、`actions_normalized=false`、`states_normalized=false`。

## 训练命令

Baseline 默认先刷新预处理 raw h5，再训练：

```bash
RAW_DEMO_H5=franka_train/data/franka_real.h5 \
TOTAL_ITERS=100000 SAVE_START_ITER=10000 SAVE_FREQ=5000 \
bash franka_train/run_train_baseline_franka.sh
```

Subgoal 默认先刷新预处理 raw h5，再训练：

```bash
RAW_DEMO_H5=franka_train/data/franka_real.h5 \
MASK_TYPE=random_mask RETAIN_RATIO=0.2 \
bash franka_train/run_train_subgoal_franka.sh
```

Baseline / Subgoal / MAM 的训练脚本都会先运行 `franka_train/preprocess_franka.py`，默认覆盖刷新 `DEMO_PATH`，再开始训练。预处理默认会将 `rgb/depth` resize 到 `256x256`，并对 action/state 使用 `1%/99%` robust min-max 后 clip 到范围内再归一化。可调：

```bash
ACTION_ROBUST_MARGIN=0.01 STATE_ROBUST_MARGIN=0.01 IMAGE_SIZE=256 \
bash franka_train/run_train_mam_franka.sh
```

如需复用已有预处理 h5，可设 `RUN_PREPROCESS=auto`；如需完全跳过预处理，可设 `RUN_PREPROCESS=false`。

MAM 默认先刷新预处理 raw h5，并支持原 pipeline 的 mixed mask 设置：

```bash
RAW_DEMO_H5=franka_train/data/franka_real.h5 \
MASK_TYPE_LIST='["random_mask","3D_points"]' \
MASK_RATIO_LIST='[0.2,0.5]' \
MASK_COMPOSITION_LIST='[0.5,0.5]' \
PREPROCESS_MASK_ASSIGN_MODE=composition \
LONG_WINDOW_BACKWARD_LENGTH=0 LONG_WINDOW_FORWARD_LENGTH=32 \
SHORT_WINDOW_HORIZON=0 LOSS_MASK_AREA_WEIGHT=0 \
bash franka_train/run_train_mam_franka.sh
```

如需同一条 demo 复制出多种 mask：

```bash
PREPROCESS_MASK_ASSIGN_MODE=one_demo_multi_mask \
MASK_TYPE_LIST='["random_mask","3D_points"]' \
MASK_RATIO_LIST='[0.2,0.5]' \
bash franka_train/run_train_mam_franka.sh
```

STPM 真机单独训练，默认 state paths 为 `qpos/qvel/tcp_pose`：

```bash
DATASET_PATH=franka_train/data/franka_real.h5 \
OUTPUT_DIR=STPM_franka \
TASK_DESCRIPTION='pick up a cup and place it next to the plate' \
bash franka_train/run_train_stpm_franka.sh
```

只生成 STPM config 和 `state_norm.json`：

```bash
DATASET_PATH=franka_train/data/franka_real.h5 \
OUTPUT_DIR=STPM_franka \
PREPARE_ONLY=true \
bash franka_train/run_train_stpm_franka.sh
```

## 常用可调参数

- 公共：`EXP_NAME`、`SEED`、`CUDA`、`NUM_DEMOS`、`TOTAL_ITERS`、`BATCH_SIZE`、`LR`、`OBS_HORIZON`、`ACT_HORIZON`、`PRED_HORIZON`、`NUM_DATALOAD_WORKERS`。
- 模型：`NOISE_MODEL=Transformer|Unet`、`DIT_HIDDEN_DIM`、`DIT_NUM_BLOCKS`、`DIT_DIM_FEEDFORWARD`、`UNET_DIMS`、`N_GROUPS`。
- MAM：`MASK_TYPE_LIST`、`MASK_RATIO_LIST`、`MASK_COMPOSITION_LIST`、`PREPROCESS_MASK_ASSIGN_MODE`、`LONG_WINDOW_BACKWARD_LENGTH`、`LONG_WINDOW_FORWARD_LENGTH`、`SHORT_WINDOW_HORIZON`、`MAS_LONG_ENCODE_MODE`、`MAS_LONG_CONV_OUTPUT_DIM`、`LOSS_MODE`、`LOSS_MASK_AREA_WEIGHT`。
- checkpoint：`SAVE_START_ITER`、`SAVE_FREQ`；规则为 `step > SAVE_START_ITER && step % SAVE_FREQ == 0`，同时刷新 `latest.pt`。

真机入口默认 `EVAL_FREQ=0`，不会创建 ManiSkill eval env，也不会保存 `best_eval_*`。
