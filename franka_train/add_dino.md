# Add Dino-v3 as Alternative vision encoder

*这是一个和codex的交互文档，请你完成我在文档中布置的任务，但不要动我写的东西，只在指定位置做增删改，并且注意保持语言简练明白*

## 大致计划

我打算在franka_train文件夹里的三条管线（baseline、sc、mam）中，同时引入供替换的另一视觉编码器 DINO-v3 ViT-S+/16（29M）。

在sh文件中新增参数‘vision-encoder’，可选resnet（维持原版）与dino（使用新版encoder）

## 调研与部署前问题

### DINO-v3 ViT-S+/16 基本信息相关

- 目标模型：`facebook/dinov3-vits16plus-pretrain-lvd1689m`，即 DINOv3 ViT-S+/16，约 29M 参数
- 架构要点：patch size 16、hidden size 384、register tokens 4、6 heads、SwiGLU FFN、RoPE

- 加载方式用：
  - HuggingFace Transformers：`AutoImageProcessor.from_pretrained(...)` + `AutoModel.from_pretrained(...)`
  - 预训练权重提前加载到本地，并在sh文件中新增参数，需要手动输入与训练权重的路径，只用DINO_MODEL_PATH这一个路径参数

- 使用官方 LVD-1689M 预处理：RGB 图像 resize 到指定尺寸，转 float，并用 ImageNet mean/std：`mean=(0.485,0.456,0.406)`，`std=(0.229,0.224,0.225)`。
- 输出使用：CLS token，直接输出全图特征

### 部署前需要决定的问题

1. 用 CLS / `pooler_output`，维度 384，再通过 `Linear(384, visual_feature_dim)` 投到 256，尽量保持三条管线下游维度不变。

3. 输入预处理
   - 预处理默认 `IMAGE_SIZE=256`
   - 不使用HF processor，直接在 PyTorch 里实现同样的 resize + normalize，然后直接喂给 AutoModel
   - 不需要担心depth，全局没有depth

4. 先使用冻结 DINO，只训练线性投影和 diffusion policy，后续如果需要finetune可以再加

5. 数据增强
   - 在sh文件里做可选参数‘augmentation’，true/false 
   - 若true，则使用轻量ColorJitter和RandomResizedCrop(scale=0.9~1.0)

6. 当前真机只使用但相机，若有多机位输入，直接报错

7. checkpoint 兼容
   - `vision_encoder=resnet/plainconv` 与 `vision_encoder=dino` 的 checkpoint 不兼容。
   - 保存参数里需要记录 `vision_encoder`、DINO 模型名/路径、freeze 状态、feature 类型、输入尺寸、是否使用 depth。

8. 依赖与环境
   - HuggingFace 路线需要确认当前环境是否已有 `transformers>=4.56`。
   - 目前环境里已满足，但真机环境可能需要安装 

9. batch size 可能会受限

10. 实现接口
   - sh 新增：
   `VISION_ENCODER=resnet|dino`、
   `DINO_MODEL_PATH`、
   `DINO_DATA_AUG=true|false`、
   - py 新增同名参数，三条 Franka wrapper 都透传给对应 diffusion_policy `Agent`。
   - 最小改动方式：新增一个 `VisionEncoder` factory，在 `Agent.__init__` 中替代硬编码 `PlainConv`，并保持 `visual_feature_dim=256` 不变。

## 执行记录

- 新增 `examples/baselines/diffusion_policy/models/vision_encoder.py`：
  - `resnet` 分支保持原 `PlainConv`
  - `dino` 分支使用本地 `DINO_MODEL_PATH` 通过 HuggingFace `AutoModel.from_pretrained(..., local_files_only=True)` 加载
  - DINO 冻结，只训练 `Linear(384, 256)` 投影层
  - 输入使用 RGB，resize 到 `256x256`，再做 ImageNet mean/std normalize
  - DINO 只支持单相机 3 通道 RGB，多通道或 depth 会报错
- 修改三条 diffusion policy Agent：
  - `train_baseline.py`
  - `train_subgoal_condition.py`
  - `train_mam.py`
  - 将硬编码 `PlainConv` 替换为 `make_vision_encoder(...)`
  - 新增 `vision_encoder`、`dino_model_path`、`dino_data_aug` 参数
  - `DINO_DATA_AUG=true` 时启用轻量 `RandomResizedCrop(scale=0.9~1.0)` 和 `ColorJitter`
- 修改三条 Franka wrapper：
  - `train_baseline_franka.py`
  - `train_subgoal_franka.py`
  - `train_mam_franka.py`
  - 同步新增并透传 `vision_encoder`、`dino_model_path`、`dino_data_aug`
- 修改三条 sh：
  - `run_train_baseline_franka.sh`
  - `run_train_subgoal_franka.sh`
  - `run_train_mam_franka.sh`
  - 新增 `VISION_ENCODER=resnet|dino`
  - 新增 `DINO_MODEL_PATH`
  - 新增 `DINO_DATA_AUG=true|false`
  - 当 `VISION_ENCODER=dino` 且 `DINO_MODEL_PATH` 为空时直接报错
- 验证：
  - `py_compile` 通过相关 Python 文件
  - `bash -n` 通过三条 Franka sh 文件


## 新增DINOv2 ViT-S/14

由于dino-v3需要审批，现在打算再加入dino-v2这一可选项：
   `VISION_ENCODER=resnet|dino3|dino2`、
   `DINO_MODEL_PATH`、
   `DINO_DATA_AUG=true|false`、

请你像刚才一样，给我列出模型基本信息以及部署前需要决定的问题

### DINO-v2 ViT-S/14 基本信息相关

- 目标模型：`facebook/dinov2-small`，对应 DINOv2 ViT-S/14，约 21M 参数。
- 架构要点：patch size 14、hidden size 384、无 register tokens，先用普通 ViT-S/14。
- 加载方式仍用 HuggingFace：
  - `AutoModel.from_pretrained(DINO_MODEL_PATH, local_files_only=True)`
  - 权重提前下载到本地，训练时只读 `DINO_MODEL_PATH`，避免联网。
- 预处理：
  - RGB 图像转 float
  - resize 到固定尺寸
  - ImageNet mean/std：`mean=(0.485,0.456,0.406)`，`std=(0.229,0.224,0.225)`
- 输出使用：
  - 第一版继续用 CLS / `pooler_output`
  - 输出维度同样是 384，再接 `Linear(384, 256)`，保持下游条件维度不变。
- 和 DINO-v3 的主要差别：
  - DINOv2 ViT-S/14 更容易下载，不需要 DINOv3 gated 审批。
  - patch size 从 16 变 14，所以输入尺寸最好能被 14 整除。
  - 若继续使用当前 `IMAGE_SIZE=256`，patch grid 不是整数； DINOv2 分支内部 resize 到 `252x252`。


### DINO-v2 部署前需要决定的问题

1. 模型路径
   - 建议默认目录：`Dino/dinov2-small`
   - `VISION_ENCODER=dino2` 时 `DINO_MODEL_PATH` 指向该目录。

2. 输入尺寸
   - 当前 H5 预处理仍保持 `IMAGE_SIZE=256`，不建议为了 DINOv2 改全局数据。
   - DINOv2 encoder 内部单独 resize 到 `252x252` 

3. 与 DINO-v3 共用参数
   - 继续只用一个 `DINO_MODEL_PATH`，由 `VISION_ENCODER=dino2|dino3` 决定加载哪类权重。
   - `DINO_DATA_AUG` 继续复用，不额外拆成 DINO2/DINO3 两套。

4. 特征选择
   - 第一版仍用 CLS / `pooler_output`。
  
5. 冻结策略
   - 第一版继续冻结 DINOv2，只训练投影层和 policy。
   - 若后续 finetune，需要单独加 LR 参数和参数组。

6. 单相机限制
   - 和 DINO-v3 一样，DINOv2 只接受单相机 RGB 3 通道。
   - 多相机或 depth 输入先直接报错。

7. checkpoint 兼容
   - `resnet`、`dino2`、`dino3` 三类 checkpoint 互不兼容。
   - wandb / tensorboard hyperparameters 中需要记录 `vision_encoder` 和 `dino_model_path`。

8. 实现接口建议
   - sh 改为：`VISION_ENCODER=resnet|dino3|dino2`
   - py 类型同步改为：`Literal["resnet", "dino3", "dino2"]`
   - `vision_encoder.py` 中新增 `DinoVisionEncoder(model_path, out_dim, image_size)`，根据 `dino2/dino3` 设置内部 resize size：
     - `dino3`: 256
     - `dino2`: 252

### DINO-v2 执行记录

- 修改 `examples/baselines/diffusion_policy/models/vision_encoder.py`：
  - `make_vision_encoder(...)` 支持 `resnet|dino2|dino3`
  - `dino2` 内部 resize 到 `252x252`
  - `dino3` 内部 resize 到 `256x256`
  - `make_dino_data_aug(...)` 根据 `dino2/dino3` 使用对应 crop size
- 修改三条 diffusion policy Agent：
  - `train_baseline.py`
  - `train_subgoal_condition.py`
  - `train_mam.py`
  - `vision_encoder` 类型改为 `Literal["resnet", "dino2", "dino3"]`
  - depth 限制和数据增强逻辑同步适配 `dino2/dino3`
- 修改三条 Franka wrapper：
  - `train_baseline_franka.py`
  - `train_subgoal_franka.py`
  - `train_mam_franka.py`
  - `vision_encoder` 类型同步改为 `resnet|dino2|dino3`
- 修改三条 sh：
  - `run_train_baseline_franka.sh`
  - `run_train_subgoal_franka.sh`
  - `run_train_mam_franka.sh`
  - `VISION_ENCODER` 支持 `resnet|dino2|dino3`
  - `dino2` 默认路径：`${ROOT_DIR}/Dino/dinov2-small`
  - `dino3` 默认路径：`${ROOT_DIR}/Dino/dinov3-vits16plus-pretrain-lvd1689m`
  - DINO 分支继续检查 `config.json`
- 验证：
  - `py_compile` 通过相关 Python 文件
  - `bash -n` 通过三条 Franka sh 文件
