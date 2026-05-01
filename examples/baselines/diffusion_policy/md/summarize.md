# summarize

**我们在这个文档里就diffusion_policy文件夹的清理和整理问题进行交互，我会每次在这个文档里写上进一步指示，然后按我的要求进行，在指定位置添加我需要的答复**
**这个文档里我写的内容你不要动，只按照我的要求增改**

## 当前总结

仔细阅读diffusion_policy文件夹里的所有文件，给出所有文件的用途。
便于我做后续清理

## 本次阅读结果：文件用途总表

说明：下面覆盖当前目录的非缓存文件；`__pycache__/` 下 147 个 `.pyc` 都是 Python 运行缓存，可整体删除；`runs/` 当前无文件，是训练输出目录。

### 总体判断

- 当前核心链路：`data_preprocess.py` / `data_preprocess_mixed.py` -> `train_mas_window.py` / `train_mas_window_mixed.py` / `train_relative_action.py` -> `evaluate/*` + `utils/*`。
- 可优先清理：所有 `__pycache__`、空 `runs/`、明显过时的 test/overfit 脚本。
- 需要修复或确认后再留：`run_eval_inpaint_test.sh` 指向不存在的 `run_eval_mas_window_inpaint.sh`；`run_train_mas_window_dit.sh` 使用 `--long-window-horizon`，但当前 `train_mas_window.py` 入口使用 `--long-window-backward-length/--long-window-forward-length`。

### 顶层文档、配置、资源

| 文件 | 用途 | 清理建议 |
|---|---|---|
| `.gitignore` | 忽略 `__pycache__/`、`runs/`、`wandb/`、`*.egg-info/`。 | 保留。 |
| `summarize.md` | 本清理总结工作文档。 | 保留。 |
| `Add_CE.md` | Control Error 指标的设计、离线/在线接入方案、验证 checklist。 | 可归档为设计文档。 |
| `Add_Inpainting.md` | 动作版 RePaint/Inpainting 的算法设计、实现记录、实验结论。 | 若继续做 inpainting，保留。 |
| `DataPreprocess.md` | 新预处理流程设计：action/state 归一化、MAS、progress、split。 | 保留作预处理规范。 |
| `Debug1.md` | only-MAS 和 MAS-window pipeline 的详细数据流/debug 说明。 | 可归档，当前代码说明价值高。 |
| `Debug4.md` | blank-window、old DiT 兼容性、bug 对比实验记录。 | 历史 debug，可归档。 |
| `Mix_Dataset.md` | mixed mask dataset 设计、重复 mask slot、train/eval 独立配置。 | 保留作 mixed 规范。 |
| `Optimize.md` | 数据采集、模型、loss、control mode、relative action 等优化记录。 | 历史实验笔记，可归档。 |
| `add_multitrain.md` | DDP/multitrain 添加记录。 | 可归档。 |
| `reconstruct_relative.md` | relative/delta action space 重构设计和 debug 结论。 | 若保留 relative 链路则保留。 |
| `image.png` | “Reconstruction in Delta Action Space” 示意图，用于 relative/delta action 方案说明。 | 若文档引用则保留，否则可移动到 docs/assets。 |

### 数据预处理

| 文件 | 用途 | 清理建议 |
|---|---|---|
| `data_preprocess.py` | 单一 mask 预处理入口：读取 raw h5/json，计算 action/state min-max，生成 normalized `actions`、`mas`、`mask`、progress、train/eval h5/json。 | 核心，保留。 |
| `data_preprocess_mixed.py` | mixed mask 预处理入口：支持 `composition` / `one_demo_multi_mask`，支持 train/eval 独立 mask 配置、重复 mask slot、expanded demo、mixed meta。 | 核心，保留。 |
| `data_preprocess/utils/__init__.py` | 导出预处理工具 API。 | 保留。 |
| `data_preprocess/utils/io_utils.py` | JSON/HDF5 基础 IO、traj key 排序、字符串 dataset 写入。 | 保留。 |
| `data_preprocess/utils/mask_utils.py` | mask 类型校验和 `apply_mask_to_actions` 实现。 | 核心，保留。 |
| `data_preprocess/utils/normalize_utils.py` | min-max 统计、归一化、从 h5/json 加载 action stats。 | 保留。 |
| `data_preprocess/utils/obs_utils.py` | 从 ManiSkill obs 中 flatten state。 | 保留。 |
| `data_preprocess/utils/progress_utils.py` | MAS 7 维 + progress 第 8 维的 numpy/torch 增广和 padding。 | 保留。 |

### 训练入口

| 文件 | 用途 | 清理建议 |
|---|---|---|
| `train_mas_window.py` | 当前单 mask MAS-window 主训练：RGB/RGBD + state + long/short MAS window，支持 weighted loss、online CE、视频归档、可选 inpainting、backward/forward long window。 | 核心，保留。 |
| `train_mas_window_mixed.py` | 单卡 mixed mask MAS-window 训练：按 mask slot/type 抽样与评估，输出 per-mask 指标和 CE。 | 核心，保留或与 DDP 版合并。 |
| `train_mas_window_test.py` | 旧 test/overfit 版本：使用 `evaluate_mas_window_test.py`，参数和功能少于主线。 | 候选归档/删除。 |
| `train_only_mas.py` | 旧 only-MAS 链路：把整条 MAS flatten 后作为条件，支持 `MLP8/MAS8/AP8` progress。 | 若不再做 only-MAS ablation，可归档。 |
| `train_relative_action.py` | mixed relative-action 主训练：数据仍是 absolute MAS/action，训练时转 relative/MRAS，推理输出 `pd_ee_delta_pose`。 | 若 relative 仍在实验，保留。 |
| `train_relative_action_overfit5.py` | relative-action 的 5 demo overfit/debug 副本，使用旧 `long_window_horizon` 参数。 | 候选归档/删除。 |
| `multitrain_dit.py` | DDP 纯 DiT/RGBD diffusion policy baseline，不使用 MAS。 | 若需要 baseline，保留。 |
| `multitrain_mas_window_mixed.py` | DDP mixed MAS-window 训练：DistributedSampler/DDP，main rank 评估、per-mask 指标、CE。 | 多卡核心，保留。 |

### 评估入口与评估 helper

| 文件 | 用途 | 清理建议 |
|---|---|---|
| `eval_ce.py` | 独立离线 CE 评估入口：加载 checkpoint、eval demo、STPM，rollout 后计算 CE JSON。 | 保留。 |
| `eval_inpaint.py` | 独立 inpainting/RePaint 评估入口，继承 CE 参数并加入 `jump_length/num_resample`。 | 若继续做 inpainting，保留。 |
| `evaluate/evaluate_mas_window.py` | 当前单 mask MAS-window rollout helper：STPM 估 progress，构造 long/short window，可返回 progress 曲线和 rollout records。 | 核心，保留。 |
| `evaluate/evaluate_mas_window_mixed.py` | mixed eval wrapper：按 mask type/slot 分组调用 MAS-window eval，聚合 per-mask 指标和视频。 | 核心，保留。 |
| `evaluate/evaluate_mas_window_test.py` | 旧 test 版 MAS-window evaluator。 | 候选归档/删除。 |
| `evaluate/evaluate_only_mas.py` | only-MAS evaluator，按 STPM progress 构造 `mas/progress` 条件。 | 仅 only-MAS ablation 需要。 |
| `evaluate/evaluate_relative_action.py` | relative-action evaluator：在线把 absolute MAS 转 MRAS，执行 delta action，并保存重构 absolute path 供 CE。 | relative 链路保留。 |
| `evaluate/evaluate_relative_action_mixed.py` | relative mixed eval wrapper，按 mask type/slot 分组聚合。 | relative 链路保留。 |

### 模型

| 文件 | 用途 | 清理建议 |
|---|---|---|
| `models/modeling_ditdp.py` | DiT diffusion noise predictor `DiTNoiseNet`；后半部分多为注释掉的 LeRobot/Policy 参考代码。 | `DiTNoiseNet` 核心保留，可清理大段注释。 |
| `models/plain_conv.py` | RGB/RGBD 图像 encoder 和通用 `make_mlp`。 | 保留。 |
| `models/mas_conv2d.py` | 2D Conv MAS window encoder，输入 value/mask 双通道。 | 保留。 |
| `models/mas_conv1d.py` | 1D temporal Conv MAS encoder，只沿时间维卷积。 | 保留。 |
| `models/progress_mlp.py` | progress 标量到特征的 MLP。 | only-MAS 用，按链路决定。 |

### 共享工具

| 文件 | 用途 | 清理建议 |
|---|---|---|
| `utils/add_progress_to_mas_utils.py` | torch 版 MAS/mask 加 progress 和 padding。 | 保留。 |
| `utils/build_progress_window_utils.py` | 构造 MLP/MAS/AP progress 窗口、future/bidirectional MAS long/short window。 | 核心，保留。 |
| `utils/control_error_utils.py` | CE 数据加载、rollout action 归一化、单轨迹 CE、聚合和 JSON 保存。 | 保留。 |
| `utils/denormalize_utils.py` | 加载 action denorm stats、计算 state min/max。 | 保留。 |
| `utils/draw_p_t_curve_utils.py` | 保存 progress 曲线和 iter CE 曲线。 | 保留。 |
| `utils/eval_video_sampling_utils.py` | eval batch 划分和视频 capture index 选择。 | 保留。 |
| `utils/inpainting_utils.py` | 动作版 RePaint/inpainting 的 known-action overwrite、jump/resample 推理。 | inpainting 保留。 |
| `utils/load_eval_data_utils.py` | eval seed/traj 推断、eval MAS 数据加载、mask type/slot 读取。 | 核心，保留。 |
| `utils/load_train_data_utils.py` | HDF5 训练数据读取、key 映射、meta 加载。 | 保留。 |
| `utils/loss_utils.py` | action mask slicing 和 known/unknown weighted noise MSE。 | 保留。 |
| `utils/make_env.py` | 创建 CPU/GPU eval vector env，挂 FrameStack/RecordEpisode/ManiSkillVectorEnv。 | 保留。 |
| `utils/relative_action_utils.py` | absolute/relative/delta action 转换、PickCube TCP pose 解析、MRAS 转换。 | relative 链路保留。 |
| `utils/stpm_utils.py` | STPM rollout obs/history 维护、progress 预测、MAS 条件 batch 构造、episode metric 收集。 | 核心，保留。 |
| `utils/utils.py` | IterationBasedBatchSampler、worker init、obs convert、obs space/state extractor。 | 保留。 |
| `utils/video_utils.py` | eval 视频快照、成功/失败分类移动、旧 artifact 清理。 | 保留。 |

### Shell 启动脚本

| 文件 | 用途 | 清理建议 |
|---|---|---|
| `run_train_mas_window.sh` | 当前单 mask MAS-window 主启动脚本：自动预处理、训练、CE/inpainting 参数透传。 | 核心，保留。 |
| `run_train_window_mixed.sh` | 单卡 mixed MAS-window 启动脚本：自动 mixed 预处理；单 mask 时 fallback 到 `run_train_mas_window.sh`。 | 核心，保留。 |
| `run_train_relative_action.sh` | relative-action mixed 启动脚本，强制 `pd_ee_delta_pose`。 | relative 链路保留。 |
| `run_multitrain_window_mixed.sh` | 多卡 DDP mixed MAS-window 启动脚本。 | 多卡保留。 |
| `run_multitrain_dit.sh` | 多卡 DDP 纯 DiT baseline 启动脚本。 | baseline 需要则保留。 |
| `run_train_only_mas.sh` | only-MAS 启动脚本。 | only-MAS 不做则归档。 |
| `run_train_mas_window_test.sh` | test/overfit CE 启动脚本，默认 conda env。 | 候选归档。 |
| `run_train_relative_action_overfit5.sh` | relative-action overfit5 启动脚本。 | 候选归档。 |
| `run_train_mas_window_dit.sh` | 旧/特殊 DiT 配置脚本，但当前参数疑似不兼容。 | 先修复或删除。 |
| `run_eval_ce.sh` | 独立 CE 评估脚本，缺默认数据时可触发单 mask 预处理。 | 保留。 |
| `run_eval_inpaint.sh` | 独立 inpainting 评估脚本。 | inpainting 保留。 |
| `run_eval_inpaint_test.sh` | inpainting test wrapper，但调用不存在的 `run_eval_mas_window_inpaint.sh`。 | 修复或删除。 |
| `watch_dit_blank_window_progress.sh` | 轮询 TensorBoard event，显示 `runs/dit_blank_window` 的 iter 和 best success。 | 临时监控脚本，可归档。 |

### 生成物/缓存

| 路径 | 用途 | 清理建议 |
|---|---|---|
| `__pycache__/`、`*/__pycache__/` | Python 字节码缓存，其中还包含不少已删除/旧文件名的缓存，如 `train_rgbd*`、`evaluate_mam*`、`conditional_unet1d*`。 | 可整体删除。 |
| `runs/` | 训练日志、checkpoint、视频输出目录；当前为空。 | 保留目录规则即可，不需要纳入源码。 |

### 建议清理顺序

1. 先删除所有 `__pycache__`，不影响源码。
2. 修复或删除明显坏脚本：`run_eval_inpaint_test.sh`、`run_train_mas_window_dit.sh`。
3. 把 test/overfit 副本移到 archive：`train_mas_window_test.py`、`evaluate/evaluate_mas_window_test.py`、`run_train_mas_window_test.sh`、`train_relative_action_overfit5.py`、`run_train_relative_action_overfit5.sh`。
4. 若确认不再做 only-MAS ablation，可归档 `train_only_mas.py`、`evaluate/evaluate_only_mas.py`、`run_train_only_mas.sh`、`models/progress_mlp.py` 中 only-MAS 相关依赖。
5. 文档可按主题归档到 `docs/`：CE、Inpainting、DataPreprocess、Mixed、Relative、Debug/Optimize。
