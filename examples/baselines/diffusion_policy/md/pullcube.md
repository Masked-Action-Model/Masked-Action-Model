# Debug pullcube

*这是一个和codex的交互文档，请你完成我在文档中布置的任务，但不要动我写的东西，只在指定位置做增删改，并且注意保持语言简练明白*

## 问题描述

我需要跑通PullCube的过拟合实验，最好再后期也试一试正式训练。

## 执行记录

1、replay数据
bash examples/baselines/diffusion_policy/tmp/run_pullcube_replay_full.sh

2、STPM训练
bash STPM/run_train_pullcubetool_stpm.sh

3、5demo 数据

已准备：

- baseline train: `examples/baselines/diffusion_policy/tmp/pullcubetool_overfit5/data/baseline_5demo/pullcubetool_baseline_5demo_train.h5`
- baseline eval: `examples/baselines/diffusion_policy/tmp/pullcubetool_overfit5/data/baseline_5demo/pullcubetool_baseline_5demo_eval.h5`
- mam train: `examples/baselines/diffusion_policy/tmp/pullcubetool_overfit5/data/mam_5demo_random_mask/pullcubetool_mam_5demo_random_mask_train.h5`
- mam eval: `examples/baselines/diffusion_policy/tmp/pullcubetool_overfit5/data/mam_5demo_random_mask/pullcubetool_mam_5demo_random_mask_eval.h5`

校验：5 条 demo，`action_dim=7`，`mask_type=random_mask`，`retain_ratio=0.2`，实际 `mask_mean≈0.1996~0.2`。

4、baseline 5demo test

训练步数可调：默认 `TRAIN_ITERS=20000`，运行前可改。

```bash
TRAIN_ITERS=${TRAIN_ITERS:-20000}

PATH=/home/hebu/miniconda3/envs/maniskill_py311/bin:$PATH \
PYTHONPATH=/home/hebu/code/ManiSkill:/home/hebu/code/ManiSkill/examples/baselines/diffusion_policy:$PYTHONPATH \
LD_LIBRARY_PATH=/home/hebu/miniconda3/envs/maniskill_py311/lib:$LD_LIBRARY_PATH \
MPLCONFIGDIR=/tmp/matplotlib-maniskill \
python examples/baselines/diffusion_policy/train_baseline.py \
    --exp-name PullCubeTool_baseline_5demo_test \
    --env-id PullCubeTool-v1 \
    --demo-path examples/baselines/diffusion_policy/tmp/pullcubetool_overfit5/data/baseline_5demo/pullcubetool_baseline_5demo_train.h5 \
    --eval-demo-path examples/baselines/diffusion_policy/tmp/pullcubetool_overfit5/data/baseline_5demo/pullcubetool_baseline_5demo_eval.h5 \
    --eval-demo-metadata-path examples/baselines/diffusion_policy/tmp/pullcubetool_overfit5/data/baseline_5demo/pullcubetool_baseline_5demo_eval.json \
    --num-demos 5 \
    --num-eval-demos 5 \
    --num-eval-episodes 5 \
    --num-eval-envs 1 \
    --action-dim 7 \
    --control-mode pd_ee_pose \
    --obs-mode rgb \
    --sim-backend physx_cpu \
    --max-episode-steps 300 \
    --noise-model Transformer \
    --batch-size 64 \
    --lr 1e-4 \
    --total-iters "$TRAIN_ITERS" \
    --obs-horizon 2 \
    --act-horizon 8 \
    --pred-horizon 16 \
    --eval-freq 2000 \
    --save-freq 2000 \
    --log-freq 1000 \
    --num-dataload-workers 0 \
    --demo-type pullcubetool_baseline_5demo_test \
    --cuda \
    --no-track \
    --no-capture-video
```

5、mam 5demo random_mask=0.2 test

训练步数可调：默认 `TRAIN_ITERS=30000`，运行前可改。

```bash
TRAIN_ITERS=${TRAIN_ITERS:-30000}

PATH=/home/hebu/miniconda3/envs/maniskill_py311/bin:$PATH \
PYTHONPATH=/home/hebu/code/ManiSkill:/home/hebu/code/ManiSkill/examples/baselines/diffusion_policy:$PYTHONPATH \
LD_LIBRARY_PATH=/home/hebu/miniconda3/envs/maniskill_py311/lib:$LD_LIBRARY_PATH \
MPLCONFIGDIR=/tmp/matplotlib-maniskill \
EXP_NAME=PullCubeTool_mam_5demo_random_mask_test \
RAW_DEMO_H5=examples/baselines/diffusion_policy/tmp/pullcubetool_overfit5/data/baseline_5demo/pullcubetool_baseline_5demo_train.h5 \
RAW_DEMO_JSON=examples/baselines/diffusion_policy/tmp/pullcubetool_overfit5/data/baseline_5demo/pullcubetool_baseline_5demo_train.json \
PREPROCESSED_DATA_DIR=examples/baselines/diffusion_policy/tmp/pullcubetool_overfit5/data/mam_5demo_random_mask \
PREPROCESS_NUM_TRAJ=5 \
PREPROCESS_MASK_ASSIGN_MODE=composition \
MASK_TYPE_LIST='["random_mask"]' \
MASK_RATIO_LIST='[0.2]' \
MASK_COMPOSITION_LIST='[1.0]' \
DEMO_PATH=examples/baselines/diffusion_policy/tmp/pullcubetool_overfit5/data/mam_5demo_random_mask/pullcubetool_mam_5demo_random_mask_train.h5 \
TRAIN_DEMO_METADATA_PATH=examples/baselines/diffusion_policy/tmp/pullcubetool_overfit5/data/mam_5demo_random_mask/pullcubetool_mam_5demo_random_mask_train.json \
TEST_DEMO_PATH=examples/baselines/diffusion_policy/tmp/pullcubetool_overfit5/data/mam_5demo_random_mask/pullcubetool_mam_5demo_random_mask_eval.h5 \
EVAL_DEMO_METADATA_PATH=examples/baselines/diffusion_policy/tmp/pullcubetool_overfit5/data/mam_5demo_random_mask/pullcubetool_mam_5demo_random_mask_eval.json \
ACTION_NORM_PATH=examples/baselines/diffusion_policy/tmp/pullcubetool_overfit5/data/mam_5demo_random_mask/pullcubetool_mam_5demo_random_mask_train.h5 \
STPM_CONFIG_PATH=STPM_pullcubetool/config.yaml \
STPM_CKPT_PATH=STPM_pullcubetool/checkpoints/reward_best.pt \
ENV_ID=PullCubeTool-v1 \
ACTION_DIM=7 \
CONTROL_MODE=pd_ee_pose \
OBS_MODE=rgb \
SIM_BACKEND=physx_cpu \
MAX_EPISODE_STEPS=300 \
NOISE_MODEL=Transformer \
BATCH_SIZE=64 \
LR=1e-4 \
TOTAL_ITERS="$TRAIN_ITERS" \
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
DEMO_TYPE=pullcubetool_mam_5demo_random_mask_test \
TRACK=false \
CAPTURE_VIDEO=false \
INPAINTING=false \
EVAL_PROGRESS_BAR=false \
CAPTURE_VIDEO_FREQ=5 \
bash examples/baselines/diffusion_policy/run_mam.sh
```

## baseline debug

现象：`PullCubeTool_baseline_5demo_test` 到 `12000` 时 eval 又回到 0，怀疑 baseline 有 bug。

排查结论：

1. eval 数据和环境对齐没问题：
   - 5 条 eval demo 直接在当前 `PullCubeTool-v1 / pd_ee_pose / max_episode_steps=300` 环境 replay，全部成功。
   - `success_any=True`，`success_last=True`。

2. 训练不是完全没学到：
   - TensorBoard 里 `6000` 曾到 `success_once=0.2`，不是一直 0。
   - `12000.pt` 在专家观测帧上预测动作已经很准：
     `denorm_mse≈0.000032`，各维平均绝对误差约 `[0.0026, 0.0029, 0.0015, 0.0091, 0.0007, 0.0003, 0.0031]`。

3. 真正失败点在闭环 rollout：
   - seed 0 的 12000 checkpoint 前 80 步能正常接近并抓住工具，和专家轨迹基本一致。
   - 后续移动工具到 cube 后方时发生闭环漂移：专家在 `192` 步时 tool 约在 `[-0.037, -0.275, 0.025]`，策略在同阶段约为 `[0.030, -0.081, 0.029]`，y 方向明显偏离，导致工具没有勾到 cube，cube 一直不动。

4. 当前更像观测/闭环鲁棒性问题，而不是 eval reset、action denorm 或数据成功标签 bug：
   - 当前 `rgb` replay 只有 `base_camera` 一个相机。
   - `obs/extra` 只有 `tcp_pose`，没有 `cube_pose/tool_pose/is_grasping_tool` 这类低维任务信息。
   - 对 PullCubeTool 这种长时序接触任务，单 base camera + 5demo 的小误差会在工具放置阶段放大。

下一步建议：

1. 先跑一个 privileged debug 版：让 `rgb` obs 也带 `cube_pose/tool_pose/is_grasping_tool`，重新 replay/STPM/baseline，验证闭环能否过拟合。
2. 或加第二视角/hand camera 后重放数据，优先确认视觉观测是否足够。
3. 同时可以固定 diffusion eval seed，减少 eval 指标随机波动，但这不是主因。
