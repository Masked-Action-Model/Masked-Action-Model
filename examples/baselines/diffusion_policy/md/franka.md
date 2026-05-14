# 真机Franka实验

*这是一个和codex的交互文档，请你完成我在文档中布置的任务，但不要动我写的东西，只在指定位置做增删改，并且注意保持语言简练明白*

核心目标：我们需要在真机上测试diffusion_policy文件夹里的policy
（包括baseline、subgoal_conditioning、mam）
（目前task是夹起一个杯子后放在盘子旁边）

## 核心差异

1、真机数据集内容构成会与仿真不同
（去除了truncated和terminated数值，所有轨迹执行到maxlength）
（去除了obs/extra/is_grasped、obs/extra/goal_pos、obs/sensor_param/base_camera/*，用不到）

2、训练过程中不需要也无法进行eval环节

3、checkpoint文件记录遵循一下规则，在n iteration后，每m iteration保存一次checkpoint文件

## 真机数据集

真机数据集格式

h5：

traj_N/
  actions                         (T, 7) float32 （拼接replay/EE_pose_FK和replay/gripper_events）
  success                         (T,) bool      （手动标注最后一次夹爪action，此后为success）

  obs/agent/qpos                  (T+1, 9) float32   （直接记录在FK_state/joint_pos）
  obs/agent/qvel                  (T+1, 9) float32   （直接记录在FK_state/joint_velocity） 
  obs/extra/tcp_pose              (T+1, 7) float32   （直接记录在FK_state/tcp_pose（应7维））

  obs/sensor_data/base_camera/rgb (T+1, 128, 128, 3) uint8 gzip （视频数据转化）

*同时补齐 `meta`：至少记录 `env_id`、`control_mode`、`action_dim`、`state_dim=25`、`state_paths=[qpos,qvel,tcp_pose]`、`camera_names`、`max_episode_steps`、`actions_normalized/states_normalized`


## 改动目标

新建文件夹franka_train:
复用STPM与diffusion_policy文件里使用大部分的py\sh脚本，只修改部分以适配真机的特殊性质：


让代码适配新的state配置与camera配置（rgb）

- `terminated/truncated` 可以不存，但现有 `train_mam.py` 会强制读取二者。短期建议训练脚本兼容缺省字段：若缺失则自动生成全 `False`，长度为 `actions.shape[0]`；不要要求真机数据伪造仿真终止信息。

- `obs/extra/is_grasped` 和 `goal_pos` 缺失会让当前默认 state 从 PickCube 的 29 维变为真机 25 维。baseline/subgoal 只要训练和推理 state schema 一致即可；MAM/STPM 若仍用旧 PickCube STPM checkpoint 会不匹配，建议真机单独训练/配置 STPM，或先关闭依赖旧 29 维 state 的路径。

让代码适配无eval的

- 真机无法在线 eval 时，训练脚本应增加 `--no-eval` 或允许 `eval_freq <= 0` 跳过 `evaluate_and_save_best()`。否则现有脚本会创建 ManiSkill eval env，并按 `success_once/success_at_end` 保存 best checkpoint，不适合真机。

checkpoint 规则建议改成：`iteration > n_start_save` 且 `iteration % save_freq == 0` 时保存普通 checkpoint；同时保留 `latest.pt`。无 eval 时不要保存 `best_eval_*`，选择模型应看 train loss、离线 action MSE/CE、以及真机试跑记录。

让代码适配无truncate等的数据，直接执行到maxlength

- `success` 字段可用于离线统计和切分阶段，但不应作为训练终止信号。由于所有轨迹执行到 maxlength，采样窗口仍按 `actions` 长度切；成功后的尾段如果动作基本静止，建议单独检查是否会让模型过度学习“停住”。

- 关键风险是动作语义：`actions=(EE_pose_FK + gripper_events)` 必须确认是绝对 `pd_ee_pose` 目标，且位姿坐标系、四元数顺序、夹爪开闭量纲和当前训练/eval 代码一致。若真机执行端实际吃 delta action，应新建真机专用 action adapter，不要混在原 baseline 里。
