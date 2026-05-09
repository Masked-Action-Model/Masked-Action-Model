# Add Unet to SC and MAM

*这是一个和codex的交互文档，请你完成我在文档中布置的任务，但不要动我写的东西，只在指定位置做增删改，并且注意保持语言简练明白*

## Goal

在diffusionpolicy中现在有三条pipeline
请你仿照baseline.py（sh文件、py文件）中的unet/transformer切换的设置，迁移到subgoal_conditioning和MAM两条pipeline，同步修改sh和py文件。(目前后两种pipeline中只能实现dit)
参数设计仿照baseline如下：
‘
 4. Model
NOISE_MODEL="${NOISE_MODEL:-Transformer}" # Transformer or Unet
DIFFUSION_STEP_EMBED_DIM="${DIFFUSION_STEP_EMBED_DIM:-64}"
DIT_HIDDEN_DIM="${DIT_HIDDEN_DIM:-512}"
DIT_NUM_BLOCKS="${DIT_NUM_BLOCKS:-6}"
DIT_DIM_FEEDFORWARD="${DIT_DIM_FEEDFORWARD:-2048}"
UNET_DIMS="${UNET_DIMS:-64 128 256}"
N_GROUPS="${N_GROUPS:-8}"
‘

## Design

- 保持三条 pipeline 的模型参数接口一致：`NOISE_MODEL=Transformer|Unet`，DiT 参数和 UNet 参数都从 sh 透传到训练入口。
- `Transformer` 继续使用 `DiTNoiseNet`，条件输入保持 `(B, obs_horizon, cond_dim)`。
- `Unet` 使用已有 `ConditionalUnet1D`，动作序列输入仍为 `(B, pred_horizon, action_dim)`，条件输入改为 `obs_horizon * cond_dim` 的 flatten 向量。
- Subgoal conditioning 的 `cond_dim = visual_feature + state + subgoal_flat`，其中 `subgoal_flat` 仍由 padded MAS action 部分得到。
- MAM 的 `cond_dim = visual_feature + state + mas_long_feature + mas_short_feature`，MAS window 编码逻辑不变，只替换 denoiser backbone。
- action dim、MAS step dim、STPM 对齐逻辑不随本次修改改变。

## Implementation

- `run_subgoal_condition.sh`
  - 新增 `NOISE_MODEL`、`UNET_DIMS`、`N_GROUPS`。
  - 已将 denoiser 相关参数拆到独立 `Model` 分区，和 baseline 保持一致。
  - 新增 `NOISE_MODEL` 合法性检查。
  - 训练参数改为透传 `--noise-model "$NOISE_MODEL"`、`--unet-dims $UNET_DIMS`、`--n-groups "$N_GROUPS"`。
  - 默认 `DEMO_TYPE` 改为 `subgoal_condition_${NOISE_MODEL}`。
- `train_subgoal_condition.py`
  - `noise_model` 从只支持 `Transformer` 改为支持 `Transformer|Unet`。
  - `Transformer` 分支保持 `DiTNoiseNet`。
  - `Unet` 分支新增 `ConditionalUnet1D`，并在 `prepare_noise_condition()` 中 flatten 条件。
- `run_mam.sh`
  - 新增 `NOISE_MODEL`、`UNET_DIMS`、`N_GROUPS`。
  - 已将 denoiser 相关参数拆到独立 `Model` 分区；MAS window/loss 参数单独保留。
  - 新增 `NOISE_MODEL` 合法性检查。
  - 训练参数透传 `--noise-model`、`--unet-dims`、`--n-groups`。
- `train_mam.py`
  - 新增 `ConditionalUnet1D` 分支。
  - `Transformer` 保持 `(B, obs_horizon, cond_dim)` 条件；`Unet` 使用 flatten 后的全局条件。
  - MAM 的 long/short MAS window 编码、loss mask、action denorm、STPM eval 流程保持不变。
- `utils/inpainting_utils.py`
  - inpainting 采样时复用 agent 的 `prepare_noise_condition()`，使 MAM Unet 也能正常评估。
  - depth 只在存在时 permute，避免 `OBS_MODE=rgb` 时 inpainting 直接缺 key。
- `eval_ce.py`
  - 补齐 MAM Agent 构建所需的 DiT/UNet 参数。
  - CE / inpaint 独立评估支持按 `noise_model` 校验 DiT 或 UNet checkpoint 条件维度。
- 验证：
  - `bash -n` 通过 `run_subgoal_condition.sh`、`run_mam.sh`。
  - `py_compile` 通过 `train_subgoal_condition.py`、`train_mam.py`、`utils/inpainting_utils.py`、`eval_ce.py`、`eval_inpaint.py`。
