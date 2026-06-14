# Codex 
**這是我和codex的交互文檔，我在裏面寫的東西不要動，只按照我的要求在指定位置做增改**

概述：這個文件夾裏的腳本是用於做franka_data裏採集的franka真機數據的預處理的。

## 處理一：修正關於數據記錄起始時間的錯誤

問題說明：在採集腳本中（見home/robot/chenyu/data_colletcion），按下r鍵開始數據replay時，franka會先移動至數據採集的初始位置，然後開始replay。按理說應該在移動到初始位置之後再開始進行數據的記錄和採集，但這一版本的採集腳本有問題，設置成了‘按r後直接開始記錄數據’，造成了數據的冗餘，現在需要進行裁剪

我需要你：
- 先調研pick_cup_place_next_to_bowl/formal/traj_x/replay下記錄的幾個文件中，長度是否都一致，是否都是從按下r就開始採集
- 然後我需要你調研是否能通過‘夾爪末端停止在某一位置+夾爪打開’的方式判別真正的動作起始點
- 如果不能，則需要想新的辦法處理數據；如果能，先給我結論，不要執行裁剪

按照joint和gripper的時間戳對齊裁剪視頻。

## 處理二：真機replay獲得eepose信息

問題說明：目前採集的數據裏沒有記錄末端的信息，需要通過replay的方式補上。

我需要你：
- 我需要先用FK算法根據joint的位姿解出末端absolute aciton的理論值，記錄在action_FK.npz裏
- 調研home/robot/franka_ros2_ws，看一下裏面是不是已經有一個末端replay腳本，可以根據末端的數值replay franka，並記錄末端真實數值
- 我需要你對這個replay腳本進行修改，實現以下功能
- 讀取固定數據集的末端action（action_FK.npz）信息，用末端控制進行replay，並讀取記錄ee_pose_traj.npz，其他功能保持不變
- 調研完後再在這個md文件中記錄下replay腳本的具體用法，給我運行的bash order：

已完成的修改：
- `data_preprocess/generate_action_fk.py`：從每條 `traj_N/replay/joint_trajectory.npz` 讀取 7 維 joint，按 FR3 URDF 做 FK，生成 `traj_N/replay/action_FK.npz`。這個文件是理論 action，key 為 `action_FK`，格式是 `[x, y, z, qx, qy, qz, qw]`。
- `/home/robot/franka_ros2_ws/src/franka_example_controllers/scripts/ee_control_replay.py` 和 install 入口已修改：優先讀取 `traj_N/replay/action_FK.npz`，用 `action_FK` 做末端控制 replay，真機實際末端狀態寫入 `traj_N/replay/ee_pose_traj.npz`，key 為 `ee_pose`，格式同樣是 `[x, y, z, qx, qy, qz, qw]`。原有 `traj_N/FK_state/*.npz` 診斷輸出保留。

bash order：


1. 終端 1：啟動真機 EE replay controller：

```bash
cd /home/robot/franka_data
source /opt/ros/humble/setup.bash
source /home/robot/franka_ros2_ws/install/setup.bash
ros2 launch franka_bringup ee_control_replay.launch.py
```

2. 終端 2：批量用 `action_FK.npz` 做末端 replay，並記錄真實末端 state：

```bash
cd /home/robot/franka_data
source /opt/ros/humble/setup.bash
source /home/robot/franka_ros2_ws/install/setup.bash
ros2 run franka_example_controllers ee_control_replay.py \
  --data_dir /home/robot/franka_data/pick_cup_place_next_to_bowl/formal \
  --start_traj 0 \
  --overwrite \
  --pose_key action_FK
```

如果想一條一條測試，在第 3 步最後加 `--test`。每條軌跡完成後，action 和 state 的區分是：
- action：`traj_N/replay/action_FK.npz` 裏的 `action_FK`
- state：`traj_N/replay/ee_pose_traj.npz` 裏的 `ee_pose`

## 处理三：预处理数据

我需要将franka_data/pick_cup_place_next_to_bowl/formal里的数据文件整理成一个训练用的h5文件。
**用于训练diffusion policy，代码详见（hebu/Masked-Action-Model/franka_train/train_baseline_franka.py）**
**生成h5文件还可以参照之前训练使用的（hebu/robot_recordings/pickcup_20hz_new.h5）**

我需要你对照训练代码和现有数据集，撰写预处理脚本，将数据处理成训练所需的h5格式，具体内容如下：

traj_N/
  actions                         (T, 7) float32 （拼接action_FK和gripper_events）
  success                         (T,) bool      （手动标注最后一次夹爪action，此后为success）
  obs/agent/qpos                  (T+1, 9) float32   （从replay/joint_trajectory中提取）
  obs/agent/qvel                  (T+1, 9) float32   （从replay/joint_velocities中提取） 
  obs/extra/tcp_pose              (T+1, 7) float32   （从replay/ee_pose_traj中提取）
  obs/sensor_data/base_camera/rgb (T+1, 128, 128, 3) uint8 gzip （从视频中根据时间戳提取）
  obs/sensor_data/wrist_camera/rgb (T+1, 128, 128, 3) uint8 gzip （从视频中根据时间戳提取）


- 将所有信息进行时间戳对齐：目前机械臂state的采集频率与相机频率不同，请你统一按照时间戳，做20hz的帧对齐与近似匹配
- 注意目前训练代码中只使用单相机（256*256），之后会改成双相机（256*256），请你做两版数据：一版直接将两个相机画面直接压缩成256*256；另一版将basecamera裁剪后再压成256*256
（裁剪方式见robot_recordings中的crop文件）
- 注意action值从理论计算的末端位姿中获得，state里的末端从实际replay得到的末端中获得
- 尤其注意角度的格式转换问题，注意rpy和四元数之间的转换不要错位，action用的是rpy值，state里的末端位姿用的是四元数（顺序不要搞错，和训练代码对应！！）
- 最后检查归一化逻辑，如果训练代码中已对state和action做了归一化则不需归一化，若训练代码中未做归一化则需预处理时归一化
- 整个数据集的meta信息写成json文件

如果有以任何额外问题，参考训练代码和之前的h5文件进行对照。