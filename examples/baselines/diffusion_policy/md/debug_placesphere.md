# Debug PlaceSphere

*这是一个和codex的交互文档，请你完成我在文档中布置的任务，但不要动我写的东西，只在指定位置做增删改，并且注意保持语言简练明白*

## 基本信息

STPM相关模块在STPM_placesphere。
原始数据集在demos/PlaceSphere-v1

## 过拟合测试

请你
1、在diffusion_policy的文件夹里创建一个tmp文件夹，创建train_baseline 和 mam的5demo过拟合测试副本：保持pipeline完全一致，但使用五条训练数据，eval使用相同的五条demo

2、把原始数据replay成训练可用的rgb、pdeepose模式

3、同时撰写两个跑过拟合测试的sh文件，并填好参数，我会自己跑

4、做一下简单的参数、数据、pipeline的核对，看看有没有很明显的问题

## 一个bug修复

核实发现STPM的训练有问题，应该改为：

1、针对不同的task。
- 需要先从其数据h5文件里读取其state信息。不仅要读取维度，还要读取state内容，（类似于state path参数）并且都写去task的config文件里。

- 也需要从h5文件里读取camera的相关信息，有几个camera，以及其信息，都要写进config文件

2、在训练STPM前，需要对所有维度的state做归一化，并且放入json文件

3、同时做好mam训练时的alignment。如，输入STPM模型前state和camera的对齐等等。做一下相关检查。

过程简要记录：

- 已将 `STPM/utils/generate_state_norm_json.py` 改成按 `state_paths` 读取 h5 state 并生成全维 norm；不再硬编码 PickCube 字段，也不再 `state[:, :state_dim]` 截断。
- 生成的 norm json 现在写入 `meta.state_paths/state_schema/camera_names/camera_info`，便于检查 state 与 camera 对齐。
- `STPM/train_STPM.py` 训练初始化时会从 h5 推断/保存 `state_schema` 和 `camera_info`；若未配置 `state_paths`，会从 h5 的 `obs` state leaf 自动推断。
- 若 `state_norm.json` 缺失，训练前会按当前 `state_paths` 自动生成；若已有 json 维度不足或 meta 中的 `state_paths` 与 config 不一致，会直接报错。
- MAM eval 侧增加 camera 顺序检查：rollout raw obs 的 camera 顺序必须与 STPM checkpoint/config 的 `camera_names` 一致；state 映射继续按 STPM `state_paths` 从 rollout state 中切片。
- MAM 数据预处理侧已明确对 `obs/state` 全部维度归一化，并在 h5 meta 写入 `state_paths/state_schema_json/normalized_state_dims`；`train_mam.py` 加载时会检查预处理 state schema 与 env rollout state schema 一致，不一致直接报错。
- 临时验证：
  - PlaceSphere auto state: `qpos(9)+qvel(9)+is_grasped(1)+tcp_pose(7)+bin_pos(3)=29`，camera=`['base_camera']`。
  - StackPyramid: 自动推断 `qpos(9)+qvel(9)+tcp_pose(7)=25`，camera=`['base_camera','hand_camera']`。
  - 当传入错误 `--state_dim 29` 但 `state_paths` 实际为 28 维时，脚本会报错拒绝截断。

## 测试

现在我需要以placesphere为例，replaydata数据（全部），用replay完的数据在本机上训练STPM模块，并用这个新生成的模块重新跑mam的5demo测试（baseline已经通过）。

给我
1、replay的命令行（并行，加速replay）
2、训练STPM的命令行
3、修改5demo过拟合测试的sh文件（如果需要的话）
4、跑训练的命令行

命令：

```bash
cd /home/hebu/code/ManiSkill

PATH=/home/hebu/miniconda3/envs/maniskill_py311/bin:$PATH python -m mani_skill.trajectory.replay_trajectory \
  --traj-path demos/PlaceSphere-v1/PlaceSphere-v1.h5 \
  --save-traj \
  --obs-mode rgb \
  --target-control-mode pd_ee_pose \
  --num-envs 8 \
  --sim-backend physx_cpu \
  --allow-failure
```

```bash
cd /home/hebu/code/ManiSkill

PATH=/home/hebu/miniconda3/envs/maniskill_py311/bin:$PATH \
DATASET_PATH=demos/PlaceSphere-v1/PlaceSphere-v1.rgb.pd_ee_pose.physx_cpu.h5 \
TASK_NAME=placesphere \
TASK_DESCRIPTION="place the sphere into the shallow bin" \
OUTPUT_DIR=STPM_placesphere \
STATE_PATHS=auto \
CAMERA_NAMES=auto \
BATCH_SIZE=64 \
NUM_WORKERS=4 \
VAL_NUM_WORKERS=4 \
NUM_EPOCHS=2 \
bash STPM/run_train_stpm.sh
```
