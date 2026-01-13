# 生成对应七种SFT需要的数据集。


## 执行mask_for_sft脚本

编辑 `run_mask_sft.sh`，设置好对应参数：

```bash
INPUT="/path/to/input.h5" # 原始h5文件路径
OUTPUT="./test_local_planner_mask.h5" # 输出h5文件路径（如size>1会自动加编号后缀）
MASK_TYPE="local_planner" # 掩码类型（如local_planner、pose_AnyGrasp、points等）
RETAIN_RATIO=0.2 # 1-mask_ratio
SIZE=5 # 生成数据集数量
```
执行脚本
```bash
bash run_mask_sft.sh
```

---

## 掩码类型说明（mask_type）

- `2D_video_trajectory`：只保留x, y, time，其余mask
- `2D_image_trajectory`：只保留x, y，其余mask
- `pose_AnyGrasp`：随机保留一个pose: x, y, z, dx, dy, dz, gripper，其余mask
- `points`：随机保留retain_ratio比例的x, y，其余mask
- `pose_motion_planning`：随机保留retain_ratio比例的pose: x, y, z, dx, dy, dz, gripper，其余mask
- `local_planner`：保留所有time数据，mask掉连续长度为mask_seq_len（默认20）的任意一个子序列（0-6列），其余部分都保留
- `auto_regressive`：随机选一个(i, j)，保留(i, j)前所有原始数据，其后全部mask
- `random_mask`：随机保留retain_ratio比例的任意点数据，其余mask

---

## 归一化说明

- 默认对action的x, y, z, dx, dy, dz（0-5列）归一化到(-1, 1]区间
- 全局min/max从所有trajectory下所有group的action统计
- 归一化参数会自动保存为`输入文件名_norm.json`
- 若不需要归一化，可在shell脚本或命令行去掉`--normalize`参数

action.h5 (原始数据)
    ↓
[mask_for_sft.py --normalize] （min_max_norm）
    ↓
action_norm.json (归一化参数) + action_norm.json (归一化数据)
    ↓
output_mask (生成的mask)
---

## padding说明

- 将所有traj添加padding（-1）至最大traj长度
---

## 输出说明

- 输出h5文件结构与原始一致，action等数据已mask、归一化、并按time step升序排列（mask部分放在序列最后面）
- 若size>1，输出文件名自动加编号后缀（如output_0.h5, output_1.h5...）
- 归一化参数保存在同名json文件

---

## 数据分析工具

### analyze_data.py

用于分析h5文件中action数据的形状和统计信息。


**使用方法**：
```bash
python analyze_data.py
```

**输出信息**：
- 最大action形状：所有轨迹中action数据的最大维度
- 最大轨迹长度：所有轨迹中的最大步数
- 总轨迹数量：文件中包含的轨迹总数
- 各轨迹详情：每个轨迹的action形状和数据类型

**注意**：需要在脚本中修改`h5_file_path`变量指向要分析的h5文件路径。

---

