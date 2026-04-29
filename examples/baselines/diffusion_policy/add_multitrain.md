# add multi train

**act according to my command in this file and do not change what i wrote but only add content at assigned placed**

i need you to learn from multitrain_dit.py and multitrain_dit.sh summarize how to modify codes to enable training with multi gpu in part summarize.

after that, i need you to learn from the pattern and made a copy of train_mas_window_mixed.py and run_train_window_mixed.sh, alter the into multitraining mode , named as multitrain_mas_window_mixed.py and run_multitrain_window_mixed.py

## Summarize

- `multitrain_dit.py` 的核心改法可以概括为 6 点：
  1. 引入分布式依赖：增加 `torch.distributed`、`DistributedDataParallel(DDP)`、`DistributedSampler`。
  2. 增加分布式启动/清理函数：读取 `WORLD_SIZE` / `RANK` / `LOCAL_RANK`，初始化 `process_group`，并把每个进程绑到各自 GPU。
  3. 数据加载改为按 rank 切分：用 `DistributedSampler` 代替单卡随机采样；`batch_size` 解释为全局 batch，再换算成 `per_gpu_batch`。
  4. 模型训练改为 DDP：模型先 `.to(device)`，再包一层 `DDP(...)`；训练时通过 DDP 包装后的模型做 forward/backward。
  5. 只让主进程做副作用：`wandb`、`SummaryWriter`、评估、保存 best checkpoint、保存视频都只放在 `rank0`。
  6. 启动脚本改为 `torchrun`：shell 脚本里增加 `NPROC_PER_NODE`、`NNODES`、`NODE_RANK`、`MASTER_ADDR`、`MASTER_PORT`，最后用 `torchrun` 或 `python -m torch.distributed.run` 启动。

- 从 `multitrain_dit.sh` 学到的 shell 层模式：
  1. 单独保留原训练参数拼装逻辑，不改 Tyro 参数接口。
  2. 仅在最后一跳把 `python xxx.py` 替换为 `torchrun ... xxx.py`。
  3. `BATCH_SIZE` 继续作为“目标全局 batch”，Python 侧再按 `world_size` 切成本地 batch。
  4. 脚本里保留 `torchrun` 不存在时回退到 `python -m torch.distributed.run` 的兼容分支。

- 套到 `train_mas_window_mixed.py` 时，需要额外注意 3 点：
  1. 这个脚本不是直接 `forward()`，原本主要走 `compute_loss()`，所以多卡版需要补一个 `forward()` 包装，才能让 DDP 正常接管前向传播。
  2. 评估链路比较重，且带 STPM / 视频导出，所以主进程之外不应再做日志、评估、保存；我在多卡版里把这些都收敛到 `rank0`。
  3. 原脚本运行入口是 shell 脚本，因此按仓库现有命名习惯，新增运行脚本实际落成 `run_multitrain_window_mixed.sh`。

## Code changing records

- 新增 `examples/baselines/diffusion_policy/multitrain_mas_window_mixed.py`
  - 由 `train_mas_window_mixed.py` 复制而来，保留原始单卡文件不动。
  - 加入 `setup_distributed()` / `cleanup_distributed()`。
  - 加入 `DDP`、`DistributedSampler`，并把 dataloader 改成按 rank 切分。
  - 新增 `Agent.forward()`，内部仍复用原 `compute_loss()` 逻辑。
  - 将训练入口改成 `ddp_agent(...)`，并把优化器参数源切到 `ddp_agent.parameters()`。
  - 将 `run_name` 在 rank0 生成后广播给其他进程，避免多进程产出不同目录名。
  - 将随机种子改成 `seed + rank`，worker seed 改成 `seed + rank * 1000`，减少多进程完全同随机流的风险。
  - 仅 `rank0` 执行 TensorBoard / wandb / evaluate / checkpoint / video。
  - checkpoint 中额外保存了 `iter` 和 `world_size`。

- 新增 `examples/baselines/diffusion_policy/run_multitrain_window_mixed.sh`
  - 由 `run_train_window_mixed.sh` 复制而来，保留原单卡脚本不动。
  - 新增 `NPROC_PER_NODE`、`NNODES`、`NODE_RANK`、`MASTER_ADDR`、`MASTER_PORT`。
  - 默认实验名改为 `PickCube_window_mixed_ddp`。
  - 最后一跳改为启动 `multitrain_mas_window_mixed.py`，优先用 `torchrun`，否则回退到 `python -m torch.distributed.run`。
