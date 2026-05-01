# summarize

**我们在这个文档里就diffusion_policy文件夹的清理和整理问题进行交互，我会每次在这个文档里写上进一步指示，然后按我的要求进行，在指定位置添加我需要的答复**
**这个文档里我写的内容你不要动，只按照我的要求增改**

## 合并两条mas线路

我需要你合并train_mas_window和train_mas_window_mixed这两条线路（以及对应的sh文件），evaluate文件要不要合并你来决定。

1、合并后只保留一个py文件train_mam.py和run_mam.sh，evaluate文件也改名evaluate_mam.py

2、也就是在mixed版本代码的基础上实现single mask的功能。确保当num_mask_type填1的时候，代码就可以完全回退到single mask的训练功能，要和原来的train_mas_window实现完全一致的功能。

合并改动记录在这里：

- 已合并为 `train_mam.py`、`run_mam.sh`、`evaluate/evaluate_mam.py`。
- `train_mam.py` 以 mixed 训练逻辑为主体，新增 single-mask 检测：当 mask label/slot 只有一种时，训练和 eval demo 都按原 single 线路取前 N 条，不再做 mixed stratified shuffle。
- `run_mam.sh` 支持 `NUM_MASK_TYPE=1` 自动走 single-mask `data_preprocess.py`，多 mask 仍走 `data_preprocess_mixed.py`；最终统一调用 `train_mam.py`。
- 已删除旧入口：`train_mas_window.py`、`train_mas_window_mixed.py`、`run_train_mas_window.sh`、`run_train_window_mixed.sh`、`evaluate_mas_window.py`、`evaluate_mas_window_mixed.py`。
- `eval_ce.py`、`eval_inpaint.py` 的训练入口导入已改为 `train_mam.py`。

## Subgoal Condition线路

目前dp文件里有两条线路，一条baseline一条mam，现在为了对比实验，需要新增一条线路，其核心原理就是在baseline的基础上，在obs中拼接整个mas（flatten后直接接入obs），要求obsmode可选rgb，diffusion网络只需要transformer，具体实现如下：

1、读取sh文件中的原始数据path、masktype相关信息同mam线路（但是这条线路不需要设计multimask，只需要single mask type的设计），据此做数据预处理，action，state的归一化等等

2、构建dataset时读取mas，padding到maxlength（保证每张mas都是maxlength*7）

3、mas加载进obs，flatten后拼接在obs中，（B，H，7*maxlength）在obs_horizon这一维度上做复制，也就是拼进去H个同样的mas

训练即eval等等和之前其他pipeline一致

实现记录：

- 新增 `train_subgoal_condition.py`、`run_subgoal_condition.sh`、`evaluate/evaluate_subgoal_condition.py`。
- `run_subgoal_condition.sh` 只支持 `NUM_MASK_TYPE=1`，复用 single-mask `data_preprocess.py` 生成带 `mas/mask`、归一化 action/state 的 train/eval 数据。
- `train_subgoal_condition.py` 基于 baseline Transformer：读取每条 demo 的 `mas[:, :7]`，pad 到 train/eval 最大长度后 flatten 成 `subgoal`，并在 dataset 中复制到 `(obs_horizon, 7*maxlength)` 后拼进模型条件。
- `evaluate_subgoal_condition.py` 按 eval demo reset kwargs 对齐 rollout，并为每个并行 env 注入对应 demo 的 subgoal 条件。
- 已检查：`python -m py_compile` 覆盖 mam/subgoal 相关 py 文件；`bash -n` 覆盖 `run_mam.sh` 和 `run_subgoal_condition.sh`。

### 过拟合测试

我需要你创建一个tmp文件夹，复制5demo的过拟合测试副本，让我分别测试三条链路进行debug，分别是subgoalconditioning，singlemask训练，multimask训练，创建完之后给我bashorder让我跑过拟合。

实现记录：

- 已创建 `examples/baselines/diffusion_policy/tmp/overfit_5demo/`。
- 已复制 5demo raw 副本到 `tmp/overfit_5demo/raw/`。
- 已新增 `make_overfit_preprocessed.py`，生成三套 train/eval 都覆盖同一批 5demo 的过拟合数据：
  - `subgoal_condition`：5 train / 5 eval
  - `mam_singlemask`：5 train / 5 eval
  - `mam_multimask`：5 source demo，`one_demo_multi_mask` 展开为 10 train / 10 eval
- bash order 已写入 `examples/baselines/diffusion_policy/tmp/overfit_5demo/bash_order.md`。
