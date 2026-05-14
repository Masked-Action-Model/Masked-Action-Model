# Debug PegInsertion

*这是一个和codex的交互文档，请你完成我在文档中布置的任务，但不要动我写的东西，只在指定位置做增删改，并且注意保持语言简练明白*

## 问题描述

我需要跑通PegInsertion的过拟合实验，最好再后期也试一试正式训练。

## 执行记录

1、replay数据
bash examples/baselines/diffusion_policy/tmp/run_peginsertion_replay_full.sh

2、STPM训练
bash run_train_peginsertion_stpm.sh

3、baseline 过拟合

PATH=/home/hebu/miniconda3/envs/maniskill_py311/bin:$PATH \
PYTHONPATH=/home/hebu/code/ManiSkill:/home/hebu/code/ManiSkill/examples/baselines/diffusion_policy:$PYTHONPATH \
TOTAL_ITERS=20000 \
python examples/baselines/diffusion_policy/train_baseline.py \
    --exp-name PegInsertion_baseline_5demotrain_5demoeval \
    --env-id PegInsertionSide-v1 \
    --demo-path examples/baselines/diffusion_policy/tmp/peginsertion_overfit5/data/baseline_success5demo/peginsertion_baseline_success5demotrain.h5 \
    --eval-demo-path examples/baselines/diffusion_policy/tmp/peginsertion_overfit5/data/baseline_success5demo/peginsertion_baseline_success5demoeval.h5 \
    --eval-demo-metadata-path examples/baselines/diffusion_policy/tmp/peginsertion_overfit5/data/baseline_success5demo/peginsertion_baseline_success5demoeval.json \
    --num-demos 5 \
    --num-eval-demos 5 \
    --num-eval-episodes 5 \
    --num-eval-envs 1 \
    --action-dim 7 \
    --control-mode pd_ee_pose \
    --obs-mode rgb \
    --sim-backend physx_cpu \
    --max-episode-steps 200 \
    --noise-model Transformer \
    --batch-size 64 \
    --lr 1e-4 \
    --obs-horizon 2 \
    --act-horizon 8 \
    --pred-horizon 16 \
    --eval-freq 2000 \
    --save-freq 2000 \
    --log-freq 1000 \
    --num-dataload-workers 0 \
    --demo-type peginsertion_baseline_5demotrain_5demoeval \
    --cuda \
    --no-track \
    --no-capture-video

4、mam 过拟合

先生成 MAM 的 5demotrain / 5demoeval，train 和 eval 使用同一批 source demo：

```bash
PATH=/home/hebu/miniconda3/envs/maniskill_py311/bin:$PATH \
PYTHONPATH=/home/hebu/code/ManiSkill:/home/hebu/code/ManiSkill/examples/baselines/diffusion_policy:$PYTHONPATH \
python examples/baselines/diffusion_policy/tmp/prepare_placesphere_overfit5_data.py \
    --input-h5 examples/baselines/diffusion_policy/tmp/peginsertion_overfit5/data/baseline_success5demo/peginsertion_baseline_success5demotrain.h5 \
    --input-json examples/baselines/diffusion_policy/tmp/peginsertion_overfit5/data/baseline_success5demo/peginsertion_baseline_success5demotrain.json \
    --train-h5 examples/baselines/diffusion_policy/tmp/peginsertion_overfit5/data/mam_5demo/peginsertion_mam_5demotrain.h5 \
    --train-json examples/baselines/diffusion_policy/tmp/peginsertion_overfit5/data/mam_5demo/peginsertion_mam_5demotrain.json \
    --eval-h5 examples/baselines/diffusion_policy/tmp/peginsertion_overfit5/data/mam_5demo/peginsertion_mam_5demoeval.h5 \
    --eval-json examples/baselines/diffusion_policy/tmp/peginsertion_overfit5/data/mam_5demo/peginsertion_mam_5demoeval.json \
    --env-id PegInsertionSide-v1 \
    --num-traj 5 \
    --action-dim 7 \
    --mask-assign-mode composition \
    --train-num-mask-type 1 \
    --train-mask-type-list '["random_mask"]' \
    --train-mask-composition-list '[1]' \
    --train-mask-ratio-list '[0.2]' \
    --eval-num-mask-type 2 \
    --eval-mask-type-list '["random_mask"]' \
    --eval-mask-composition-list '[]' \
    --eval-mask-ratio-list '[0.2]' \
    --mask-value 0 \
    --mask-seed 0 \
    --overwrite
```

然后启动 MAM 过拟合训练：

```bash
PATH=/home/hebu/miniconda3/envs/maniskill_py311/bin:$PATH \
PYTHONPATH=/home/hebu/code/ManiSkill:/home/hebu/code/ManiSkill/examples/baselines/diffusion_policy:$PYTHONPATH \
EXP_NAME=PegInsertion_mam_5demotrain_5demoeval \
RAW_DEMO_H5=examples/baselines/diffusion_policy/tmp/peginsertion_overfit5/data/baseline_success5demo/peginsertion_baseline_success5demotrain.h5 \
RAW_DEMO_JSON=examples/baselines/diffusion_policy/tmp/peginsertion_overfit5/data/baseline_success5demo/peginsertion_baseline_success5demotrain.json \
PREPROCESSED_DATA_DIR=examples/baselines/diffusion_policy/tmp/peginsertion_overfit5/data/mam_5demo \
DEMO_PATH=examples/baselines/diffusion_policy/tmp/peginsertion_overfit5/data/mam_5demo/peginsertion_mam_5demotrain.h5 \
TRAIN_DEMO_METADATA_PATH=examples/baselines/diffusion_policy/tmp/peginsertion_overfit5/data/mam_5demo/peginsertion_mam_5demotrain.json \
TEST_DEMO_PATH=examples/baselines/diffusion_policy/tmp/peginsertion_overfit5/data/mam_5demo/peginsertion_mam_5demoeval.h5 \
EVAL_DEMO_METADATA_PATH=examples/baselines/diffusion_policy/tmp/peginsertion_overfit5/data/mam_5demo/peginsertion_mam_5demoeval.json \
STPM_CONFIG_PATH=STPM_peginsertion/config.yaml \
STPM_CKPT_PATH=STPM_peginsertion/checkpoints/reward_best.pt \
ENV_ID=PegInsertionSide-v1 \
ACTION_DIM=7 \
CONTROL_MODE=pd_ee_pose \
OBS_MODE=rgb \
SIM_BACKEND=physx_cpu \
MAX_EPISODE_STEPS=200 \
NOISE_MODEL=Transformer \
BATCH_SIZE=64 \
LR=1e-4 \
TOTAL_ITERS=30000 \
OBS_HORIZON=2 \
ACT_HORIZON=8 \
PRED_HORIZON=16 \
LONG_WINDOW_BACKWARD_LENGTH=0 \
LONG_WINDOW_FORWARD_LENGTH=16 \
SHORT_WINDOW_HORIZON=2 \
MAS_LONG_ENCODE_MODE=2DConv \
MAS_LONG_CONV_OUTPUT_DIM=64 \
LOSS_MODE=average \
LOSS_MASK_AREA_WEIGHT=0.2 \
NUM_DEMOS=5 \
NUM_EVAL_DEMOS=5 \
NUM_EVAL_EPISODES=5 \
NUM_EVAL_ENVS=1 \
LOG_FREQ=1000 \
EVAL_FREQ=2000 \
SAVE_FREQ=2000 \
NUM_DATALOAD_WORKERS=0 \
DEMO_TYPE=peginsertion_mam_5demotrain_5demoeval \
TRACK=false \
CAPTURE_VIDEO=false \
INPAINTING=false \
EVAL_PROGRESS_BAR=false \
capture_video_freq=5 \
bash examples/baselines/diffusion_policy/run_mam.sh
```

## Debug1

对比刚才跑的baseline和mam的过拟合测试，baseline能到达100%，但mam跑了14000还是0,一定是有bug。

请你细致对比分析，检查从datapreproces、policy训练、STPM表现、rollout和测试的每一个环节，找出为什么加入mas后无法过拟合

（别跟我扯什么方法不行，方法没有问题，一定是有bug）

找出确凿的问题和证据：

### 排查结论

1. baseline 数据问题已排除：当前使用的 `baseline_success5demo` 全是成功 demo；TensorBoard 里 baseline 到 `24000` 时 `eval/success_once=1.0`，`success_at_end=0.8`。

2. STPM 不是主因：用 `STPM_peginsertion/checkpoints/reward_best.pt` 在这 5 条成功 demo 上离线评估，`stpm_eval_mean_mse=0.001813`，`last_step_mse=0.000342`，预测进度从 0 到 1 基本贴合真实进度。

3. train/eval 的 MAS window 构造对齐：用 oracle progress 对比 `build_dual_mas_window_obs_horizon` 和 eval 的 `build_dual_mas_window_condition_batch`，`max mas diff=0.0`，`max mask diff=0.0`。所以当前没有发现 rollout 条件窗口错位。

4. 真正不等价的是 MAM 过拟合数据设置：
   - 当前 `mask_assign_mode=composition`，5 条 demo 被拆成 `points=2` 条、`random_mask=3` 条，不是“每条 demo 同时拥有两种 mask”。
   - `points` 在代码里只保留动作维度 `x,y`，PegInsertion 的 `z/rotation/gripper` 全为 0 mask；实际保留率只有 `0.0562`，不是 0.2。
   - `random_mask` 约保留 0.2，但对 7 维动作是稀疏随机点。
   - 当前 `LOSS_MODE=average` 会完全忽略 `action_mask`，`loss_mask_area_weight` 在该设置下不起作用。

5. 失败不是因为 STPM 进度差：对 `runs/PegInsertion_mam_5demotrain_5demoeval/checkpoints/12000.pt` 跑 CE rollout，`success_once=0`，但 `ce_all≈0.0127`。这说明动作和 demo 已经比较接近，但 PegInsertion 对末端姿态/插入时序很敏感；当前稀疏 MAS 设置不足以稳定闭环成功。

### 建议下一步

先重做一个严格过拟合版 MAM：`mask_assign_mode=one_demo_multi_mask`，让每条 source demo 同时生成每种 mask；同时去掉 `points` 或改成更完整的 mask（例如只用 `random_mask`，或提高 retain ratio），并把 `LOSS_MODE` 改成 `weighted`。这样才能和 baseline 的 5demo 过拟合公平比较。

## 严格过拟合

```bash
PATH=/home/hebu/miniconda3/envs/maniskill_py311/bin:$PATH \
PYTHONPATH=/home/hebu/code/ManiSkill:/home/hebu/code/ManiSkill/examples/baselines/diffusion_policy:$PYTHONPATH \
EXP_NAME=PegInsertion_mam_5demotrain_5demoeval \
RAW_DEMO_H5=examples/baselines/diffusion_policy/tmp/peginsertion_overfit5/data/baseline_success5demo/peginsertion_baseline_success5demotrain.h5 \
RAW_DEMO_JSON=examples/baselines/diffusion_policy/tmp/peginsertion_overfit5/data/baseline_success5demo/peginsertion_baseline_success5demotrain.json \
PREPROCESSED_DATA_DIR=examples/baselines/diffusion_policy/tmp/peginsertion_overfit5/data/mam_5demo \
DEMO_PATH=examples/baselines/diffusion_policy/tmp/peginsertion_overfit5/data/mam_5demo/peginsertion_mam_5demotrain.h5 \
TRAIN_DEMO_METADATA_PATH=examples/baselines/diffusion_policy/tmp/peginsertion_overfit5/data/mam_5demo/peginsertion_mam_5demotrain.json \
TEST_DEMO_PATH=examples/baselines/diffusion_policy/tmp/peginsertion_overfit5/data/mam_5demo/peginsertion_mam_5demoeval.h5 \
EVAL_DEMO_METADATA_PATH=examples/baselines/diffusion_policy/tmp/peginsertion_overfit5/data/mam_5demo/peginsertion_mam_5demoeval.json \
STPM_CONFIG_PATH=STPM_peginsertion/config.yaml \
STPM_CKPT_PATH=STPM_peginsertion/checkpoints/reward_best.pt \
ENV_ID=PegInsertionSide-v1 \
ACTION_DIM=7 \
CONTROL_MODE=pd_ee_pose \
OBS_MODE=rgb \
SIM_BACKEND=physx_cpu \
MAX_EPISODE_STEPS=200 \
NOISE_MODEL=Transformer \
BATCH_SIZE=64 \
LR=1e-4 \
TOTAL_ITERS=30000 \
OBS_HORIZON=2 \
ACT_HORIZON=8 \
PRED_HORIZON=16 \
LONG_WINDOW_BACKWARD_LENGTH=0 \
LONG_WINDOW_FORWARD_LENGTH=16 \
SHORT_WINDOW_HORIZON=2 \
MAS_LONG_ENCODE_MODE=2DConv \
MAS_LONG_CONV_OUTPUT_DIM=64 \
LOSS_MODE=average \
LOSS_MASK_AREA_WEIGHT=0.2 \
NUM_DEMOS=1 \
NUM_EVAL_DEMOS=1 \
NUM_EVAL_EPISODES=1 \
NUM_EVAL_ENVS=1 \
LOG_FREQ=1000 \
EVAL_FREQ=2000 \
SAVE_FREQ=2000 \
NUM_DATALOAD_WORKERS=0 \
DEMO_TYPE=peginsertion_mam_5demotrain_5demoeval \
TRACK=false \
CAPTURE_VIDEO=false \
INPAINTING=false \
EVAL_PROGRESS_BAR=false \
bash examples/baselines/diffusion_policy/run_mam.sh
```
## debug2

### 已确认问题

1. 这轮“严格 1demo 过拟合”没有真正使用新 mask 数据。
   - 文档里新设置想用 `random_mask` 单 mask，但实际 H5 仍是旧数据：
     `mask_type_list_json=["random_mask","points"]`，`mask_composition_list_json=[0.5,0.5]`。
   - 文件时间也证明没有重生成：`peginsertion_mam_5demotrain.h5` / `eval.h5` 仍是 `2026-05-12 19:05`，早于这轮训练。
   - 原因是 `run_mam.sh` 只要 `DEMO_PATH/TEST_DEMO_PATH/json` 已存在就直接 `[mam-preprocess] reuse existing ...`，不会校验当前请求的 mask 配置、source、action_dim 是否匹配。这是脚本级 bug，容易静默复用脏数据。

2. 当前 1demo 实际选到的不是你以为的“新生成 random_mask 1demo 数据”。
   - 现有 H5 里 5 条为：`traj_0/1=points`，`traj_2/3/4=random_mask`。
   - 按 `NUM_DEMOS=1, seed=1, composition` 的选择逻辑，训练内部实际选中 `traj_2`，不是新生成的数据集。

3. env reset 和 action denorm 本身不是失败原因。
   - 我用同一 H5 的归一化动作反归一化后，在 `PegInsertionSide-v1` 中直接 replay：
     `traj_0 seed=2 success_once=True success_end=True`；
     `traj_2 seed=9 success_once=True success_end=True`。
   - 说明 demo 动作、seed、action_min/max、eval 环境基本对齐；policy rollout 失败应继续查 policy 输出 / rollout 条件，而不是先怀疑 demo replay。

4. `eval_ce.py` 另有一个已确认 bug：即使用 `--no-save-per-traj`，保存 JSON 时仍会把 `dataset_meta` 里的 bytes 直接写入，报 `TypeError: Object of type bytes is not JSON serializable`。文件会留下半截 JSON，但脚本退出码为 1。

### 需要先修/验证

1. 给 `run_mam.sh` 加配置一致性校验，至少校验已存在 H5 的 `mask_type_list_json / mask_ratio_list_json / mask_assign_mode / source_h5 / action_dim`，不一致就报错或强制重生成。
2. 重新生成一个全新目录的 1demo 数据，不要复用 `mam_5demo`，再跑真正的 1demo 过拟合。
3. 修掉 `eval_ce.py` 的 bytes JSON 序列化问题，方便后续稳定记录每条 rollout 证据。

## 新 1demo random_mask 数据

已生成并校验：

- train: `examples/baselines/diffusion_policy/tmp/peginsertion_overfit5/data/mam_1demo_random_mask/peginsertion_mam_1demo_random_mask_train.h5`
- eval: `examples/baselines/diffusion_policy/tmp/peginsertion_overfit5/data/mam_1demo_random_mask/peginsertion_mam_1demo_random_mask_eval.h5`
- 内容：1 条 demo，`seed=2`，`success=True`，`mask_type=random_mask`，`maskmean≈0.1998`。

生成命令：

```bash
PATH=/home/hebu/miniconda3/envs/maniskill_py311/bin:$PATH \
PYTHONPATH=/home/hebu/code/ManiSkill:/home/hebu/code/ManiSkill/examples/baselines/diffusion_policy:$PYTHONPATH \
LD_LIBRARY_PATH=/home/hebu/miniconda3/envs/maniskill_py311/lib:$LD_LIBRARY_PATH \
MPLCONFIGDIR=/tmp/matplotlib-maniskill \
python examples/baselines/diffusion_policy/tmp/prepare_placesphere_overfit5_data.py \
    --input-h5 examples/baselines/diffusion_policy/tmp/peginsertion_overfit5/data/baseline_success5demo/peginsertion_baseline_success5demotrain.h5 \
    --input-json examples/baselines/diffusion_policy/tmp/peginsertion_overfit5/data/baseline_success5demo/peginsertion_baseline_success5demotrain.json \
    --train-h5 examples/baselines/diffusion_policy/tmp/peginsertion_overfit5/data/mam_1demo_random_mask/peginsertion_mam_1demo_random_mask_train.h5 \
    --train-json examples/baselines/diffusion_policy/tmp/peginsertion_overfit5/data/mam_1demo_random_mask/peginsertion_mam_1demo_random_mask_train.json \
    --eval-h5 examples/baselines/diffusion_policy/tmp/peginsertion_overfit5/data/mam_1demo_random_mask/peginsertion_mam_1demo_random_mask_eval.h5 \
    --eval-json examples/baselines/diffusion_policy/tmp/peginsertion_overfit5/data/mam_1demo_random_mask/peginsertion_mam_1demo_random_mask_eval.json \
    --env-id PegInsertionSide-v1 \
    --num-traj 1 \
    --action-dim 7 \
    --mask-assign-mode composition \
    --train-num-mask-type 1 \
    --train-mask-type-list '["random_mask"]' \
    --train-mask-composition-list '[1.0]' \
    --train-mask-ratio-list '[0.2]' \
    --eval-num-mask-type 1 \
    --eval-mask-type-list '["random_mask"]' \
    --eval-mask-composition-list '[1.0]' \
    --eval-mask-ratio-list '[0.2]' \
    --mask-value 0 \
    --mask-seed 0 \
    --overwrite
```

过拟合训练命令：

```bash
PATH=/home/hebu/miniconda3/envs/maniskill_py311/bin:$PATH \
PYTHONPATH=/home/hebu/code/ManiSkill:/home/hebu/code/ManiSkill/examples/baselines/diffusion_policy:$PYTHONPATH \
LD_LIBRARY_PATH=/home/hebu/miniconda3/envs/maniskill_py311/lib:$LD_LIBRARY_PATH \
MPLCONFIGDIR=/tmp/matplotlib-maniskill \
EXP_NAME=PegInsertion_mam_1demo_random_mask_overfit \
RAW_DEMO_H5=examples/baselines/diffusion_policy/tmp/peginsertion_overfit5/data/baseline_success5demo/peginsertion_baseline_success5demotrain.h5 \
RAW_DEMO_JSON=examples/baselines/diffusion_policy/tmp/peginsertion_overfit5/data/baseline_success5demo/peginsertion_baseline_success5demotrain.json \
PREPROCESSED_DATA_DIR=examples/baselines/diffusion_policy/tmp/peginsertion_overfit5/data/mam_1demo_random_mask \
DEMO_PATH=examples/baselines/diffusion_policy/tmp/peginsertion_overfit5/data/mam_1demo_random_mask/peginsertion_mam_1demo_random_mask_train.h5 \
TRAIN_DEMO_METADATA_PATH=examples/baselines/diffusion_policy/tmp/peginsertion_overfit5/data/mam_1demo_random_mask/peginsertion_mam_1demo_random_mask_train.json \
TEST_DEMO_PATH=examples/baselines/diffusion_policy/tmp/peginsertion_overfit5/data/mam_1demo_random_mask/peginsertion_mam_1demo_random_mask_eval.h5 \
EVAL_DEMO_METADATA_PATH=examples/baselines/diffusion_policy/tmp/peginsertion_overfit5/data/mam_1demo_random_mask/peginsertion_mam_1demo_random_mask_eval.json \
ACTION_NORM_PATH=examples/baselines/diffusion_policy/tmp/peginsertion_overfit5/data/mam_1demo_random_mask/peginsertion_mam_1demo_random_mask_train.h5 \
STPM_CONFIG_PATH=STPM_peginsertion/config.yaml \
STPM_CKPT_PATH=STPM_peginsertion/checkpoints/reward_best.pt \
ENV_ID=PegInsertionSide-v1 \
ACTION_DIM=7 \
CONTROL_MODE=pd_ee_pose \
OBS_MODE=rgb \
SIM_BACKEND=physx_cpu \
MAX_EPISODE_STEPS=200 \
NOISE_MODEL=Transformer \
BATCH_SIZE=64 \
LR=1e-4 \
TOTAL_ITERS=30000 \
OBS_HORIZON=2 \
ACT_HORIZON=8 \
PRED_HORIZON=16 \
LONG_WINDOW_BACKWARD_LENGTH=0 \
LONG_WINDOW_FORWARD_LENGTH=16 \
SHORT_WINDOW_HORIZON=2 \
MAS_LONG_ENCODE_MODE=2DConv \
MAS_LONG_CONV_OUTPUT_DIM=64 \
LOSS_MODE=average \
LOSS_MASK_AREA_WEIGHT=0.2 \
NUM_DEMOS=1 \
NUM_EVAL_DEMOS=1 \
NUM_EVAL_EPISODES=1 \
NUM_EVAL_ENVS=1 \
LOG_FREQ=1000 \
EVAL_FREQ=2000 \
SAVE_FREQ=2000 \
NUM_DATALOAD_WORKERS=0 \
DEMO_TYPE=peginsertion_mam_1demo_random_mask_overfit \
TRACK=false \
CAPTURE_VIDEO=false \
INPAINTING=false \
EVAL_PROGRESS_BAR=false \
bash examples/baselines/diffusion_policy/run_mam.sh
```

## debug2

1demo test的过拟合在6000时过了，请你调研两个问题：

1、1demo test的时间是不是偏长了，检查一下bug，尤其检查STPM的对应问题，以及state归一化相关问题有没有bug，以及camera相关设置有没有bug

2、根据这个1demo test的过拟合时间，推测之前5demo的测试有没有bug，并且排查为什么1-->5会有bug

### 调研结果

1. 1demo 确实在 `6000` 第一次成功：
   - `eval/success_once`: `0,0,0,0,1.0`，对应 step `0,0,2000,4000,6000`。
   - `eval/success_at_end` 同样在 `6000` 到 `1.0`。
   - `eval/ce_all` 在 `6000` 为 `0.003979`。
   - 因为 `EVAL_FREQ=2000`，只能说明成功首次被观测在 `6000`，真实转折可能在 `4000~6000` 之间。

2. 暂未发现 STPM bug：
   - 对新 1demo 数据跑 STPM offline eval：`mean_loss=0.001291`，`sample_weighted_mse=0.001342`，`last_step_mse=0.000226`。
   - 该误差低于之前 5demo STPM offline 的 `mean_loss=0.001813`，所以 1demo 过拟合慢不能归因于 STPM 进度明显失准。

3. 暂未发现 state 归一化 bug：
   - 新 1demo H5: `state_dim=25`，`state_paths=[qpos,qvel,tcp_pose]`，与 `STPM_peginsertion/config.yaml` 完全一致。
   - 用 raw `qpos/qvel/tcp_pose` 和 H5 `meta/state_min|max` 重算 `obs/state`，`max_abs_diff=0.0`。
   - 训练时读取的是已归一化 `obs/state`，eval 时 raw state 会用同一份 `state_min|max` 归一化，路径上没有发现不一致。

4. 暂未发现 camera 设置 bug：
   - 新 1demo H5 中 `base_camera` 和 `hand_camera` 都是 `(179,128,128,3)`。
   - STPM config 的 camera_names 是 `['base_camera','hand_camera']`。
   - 默认 eval env 经 `FlattenRGBDObservationWrapper` 后 `rgb shape=(1,128,128,6)`，正好是两个 128x128 RGB camera 拼接；与训练数据和 STPM camera 数一致。
   - 注意：`build_eval_sensor_configs` 当前只从 `camera_poses`/CLI override 生成 sensor config，不读取 `camera_info` 的 width/height；本任务默认 env 已是 128，所以没有造成错配。但这对其他任务可能是隐患。

5. 1demo 的 `6000` 不算明显异常：
   - 这是 MAM + STPM online progress + Transformer denoiser 的闭环 eval，不是纯 supervised open-loop。
   - 1demo loss 从 `1.02` 降到 `0.0085` 后成功，CE 也降到 `0.004`；曲线符合“先拟合动作，再闭环成功”的过程。
   - 当前没有证据说明 `6000` 是由 STPM/state/camera bug 导致。

6. 之前 5demo 结果不能直接作为“1-->5 有 bug”的证据：
   - 新 1demo 数据是干净的单 mask：`random_mask#0`，source=`0`，maskmean≈`0.1998`。
   - 旧 5demo 数据不是 5 条同构 random_mask：`traj_0/1=points`，`traj_2/3/4=random_mask`。
   - 旧 5demo 的 `points` 实际只保留动作 `x,y`，整体 maskmean≈`0.0562`；它和新 1demo 的 random_mask 设置不等价。
   - 旧 5demo run 还混有两次 event，且前一轮曾复用脏数据；所以不能用它直接推断 1demo 到 5demo 的 scaling bug。

### 下一步判断标准

要判断真正的 `1-->5` 是否有 bug，需要重新生成一个干净的 5demo 数据集：5 条全部 `random_mask#0`，保留率 `0.2`，独立目录，独立 `EXP_NAME`。如果这个干净 5demo 仍然在足够 iter 后失败，再继续查多 demo selection、state/action normalization 范围、per-demo CE 和 rollout 轨迹。

## 新 5demo random_mask 数据

已生成并校验：

- train: `examples/baselines/diffusion_policy/tmp/peginsertion_overfit5/data/mam_5demo_random_mask/peginsertion_mam_5demo_random_mask_train.h5`
- eval: `examples/baselines/diffusion_policy/tmp/peginsertion_overfit5/data/mam_5demo_random_mask/peginsertion_mam_5demo_random_mask_eval.h5`
- 内容：5 条 demo，seeds=`[2,6,9,10,11]`，全部 `success=True`，全部 `mask_type=random_mask` / `mask_type_slot=random_mask#0`，maskmean 约 `0.199~0.200`。

数据生成命令：

```bash
PATH=/home/hebu/miniconda3/envs/maniskill_py311/bin:$PATH \
PYTHONPATH=/home/hebu/code/ManiSkill:/home/hebu/code/ManiSkill/examples/baselines/diffusion_policy:$PYTHONPATH \
LD_LIBRARY_PATH=/home/hebu/miniconda3/envs/maniskill_py311/lib:$LD_LIBRARY_PATH \
MPLCONFIGDIR=/tmp/matplotlib-maniskill \
python examples/baselines/diffusion_policy/tmp/prepare_placesphere_overfit5_data.py \
    --input-h5 examples/baselines/diffusion_policy/tmp/peginsertion_overfit5/data/baseline_success5demo/peginsertion_baseline_success5demotrain.h5 \
    --input-json examples/baselines/diffusion_policy/tmp/peginsertion_overfit5/data/baseline_success5demo/peginsertion_baseline_success5demotrain.json \
    --train-h5 examples/baselines/diffusion_policy/tmp/peginsertion_overfit5/data/mam_5demo_random_mask/peginsertion_mam_5demo_random_mask_train.h5 \
    --train-json examples/baselines/diffusion_policy/tmp/peginsertion_overfit5/data/mam_5demo_random_mask/peginsertion_mam_5demo_random_mask_train.json \
    --eval-h5 examples/baselines/diffusion_policy/tmp/peginsertion_overfit5/data/mam_5demo_random_mask/peginsertion_mam_5demo_random_mask_eval.h5 \
    --eval-json examples/baselines/diffusion_policy/tmp/peginsertion_overfit5/data/mam_5demo_random_mask/peginsertion_mam_5demo_random_mask_eval.json \
    --env-id PegInsertionSide-v1 \
    --num-traj 5 \
    --action-dim 7 \
    --mask-assign-mode composition \
    --train-num-mask-type 1 \
    --train-mask-type-list '["random_mask"]' \
    --train-mask-composition-list '[1.0]' \
    --train-mask-ratio-list '[0.2]' \
    --eval-num-mask-type 1 \
    --eval-mask-type-list '["random_mask"]' \
    --eval-mask-composition-list '[1.0]' \
    --eval-mask-ratio-list '[0.2]' \
    --mask-value 0 \
    --mask-seed 0 \
    --overwrite
```

过拟合训练命令：

```bash
PATH=/home/hebu/miniconda3/envs/maniskill_py311/bin:$PATH \
PYTHONPATH=/home/hebu/code/ManiSkill:/home/hebu/code/ManiSkill/examples/baselines/diffusion_policy:$PYTHONPATH \
LD_LIBRARY_PATH=/home/hebu/miniconda3/envs/maniskill_py311/lib:$LD_LIBRARY_PATH \
MPLCONFIGDIR=/tmp/matplotlib-maniskill \
EXP_NAME=PegInsertion_mam_5demo_random_mask_overfit \
RAW_DEMO_H5=examples/baselines/diffusion_policy/tmp/peginsertion_overfit5/data/baseline_success5demo/peginsertion_baseline_success5demotrain.h5 \
RAW_DEMO_JSON=examples/baselines/diffusion_policy/tmp/peginsertion_overfit5/data/baseline_success5demo/peginsertion_baseline_success5demotrain.json \
PREPROCESSED_DATA_DIR=examples/baselines/diffusion_policy/tmp/peginsertion_overfit5/data/mam_5demo_random_mask \
DEMO_PATH=examples/baselines/diffusion_policy/tmp/peginsertion_overfit5/data/mam_5demo_random_mask/peginsertion_mam_5demo_random_mask_train.h5 \
TRAIN_DEMO_METADATA_PATH=examples/baselines/diffusion_policy/tmp/peginsertion_overfit5/data/mam_5demo_random_mask/peginsertion_mam_5demo_random_mask_train.json \
TEST_DEMO_PATH=examples/baselines/diffusion_policy/tmp/peginsertion_overfit5/data/mam_5demo_random_mask/peginsertion_mam_5demo_random_mask_eval.h5 \
EVAL_DEMO_METADATA_PATH=examples/baselines/diffusion_policy/tmp/peginsertion_overfit5/data/mam_5demo_random_mask/peginsertion_mam_5demo_random_mask_eval.json \
ACTION_NORM_PATH=examples/baselines/diffusion_policy/tmp/peginsertion_overfit5/data/mam_5demo_random_mask/peginsertion_mam_5demo_random_mask_train.h5 \
STPM_CONFIG_PATH=STPM_peginsertion/config.yaml \
STPM_CKPT_PATH=STPM_peginsertion/checkpoints/reward_best.pt \
ENV_ID=PegInsertionSide-v1 \
ACTION_DIM=7 \
CONTROL_MODE=pd_ee_pose \
OBS_MODE=rgb \
SIM_BACKEND=physx_cpu \
MAX_EPISODE_STEPS=200 \
NOISE_MODEL=Transformer \
BATCH_SIZE=64 \
LR=1e-4 \
TOTAL_ITERS=30000 \
OBS_HORIZON=2 \
ACT_HORIZON=8 \
PRED_HORIZON=16 \
LONG_WINDOW_BACKWARD_LENGTH=0 \
LONG_WINDOW_FORWARD_LENGTH=16 \
SHORT_WINDOW_HORIZON=2 \
MAS_LONG_ENCODE_MODE=2DConv \
MAS_LONG_CONV_OUTPUT_DIM=64 \
LOSS_MODE=average \
LOSS_MASK_AREA_WEIGHT=0.2 \
NUM_DEMOS=5 \
NUM_EVAL_DEMOS=5 \
NUM_EVAL_EPISODES=5 \
NUM_EVAL_ENVS=1 \
LOG_FREQ=1000 \
EVAL_FREQ=2000 \
SAVE_FREQ=2000 \
NUM_DATALOAD_WORKERS=0 \
DEMO_TYPE=peginsertion_mam_5demo_random_mask_overfit \
TRACK=false \
CAPTURE_VIDEO=false \
INPAINTING=false \
EVAL_PROGRESS_BAR=false \
CAPTURE_VIDEO_FREQ=5 \
bash examples/baselines/diffusion_policy/run_mam.sh
```
