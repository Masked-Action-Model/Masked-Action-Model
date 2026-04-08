from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Literal, Optional

import numpy as np
import torch
import tyro
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper

from train_mas_window import (
    MAS_STEP_DIM,
    Agent,
    _load_state_norm_stats_from_meta,
    build_eval_batch_indices,
    build_eval_stpm_encoder,
    validate_only_mas_eval_layout,
)
from utils.control_error_utils import (
    aggregate_control_error,
    compute_control_error_for_traj,
    load_ce_eval_data,
    normalize_rollout_actions,
    save_control_error_json,
)
from utils.denormalize_utils import load_action_denorm_stats
from utils.load_eval_data_utils import (
    infer_eval_reset_seeds_from_demo,
    select_eval_demo_indices,
    subset_eval_data,
)
from utils.load_train_data_utils import load_dataset_meta
from utils.make_env import make_eval_envs
from utils.stpm_utils import (
    append_episode_metrics,
    append_latest_rollout_frame,
    build_dual_mas_window_condition_batch,
    init_env_histories_from_reset_obs,
    predict_current_progress_from_histories,
    prepare_batched_rollout_obs,
    validate_stpm_eval_setup,
)


@dataclass
class Args:
    checkpoint_pt_path: str
    eval_demo_path: str
    eval_demo_metadata_path: Optional[str] = None
    stpm_ckpt_path: str = ""
    stpm_config_path: str = "STPM/config/rewind_maniskill.yaml"

    env_id: str = "PickCube-v1"
    control_mode: str = "pd_ee_pose"
    max_episode_steps: int = 200
    num_eval_demos: int = 5
    num_eval_envs: int = 1
    sim_backend: str = "physx_cpu"
    render_backend: str = "gpu"

    obs_horizon: int = 2
    act_horizon: int = 8
    pred_horizon: int = 16
    long_window_horizon: Optional[int] = None
    short_window_horizon: int = 2
    mas_long_encode_mode: Literal["1DConv", "2DConv"] = "2DConv"
    mas_long_conv_output_dim: int = 64
    diffusion_step_embed_dim: int = 64
    legacy_short_mask_branch: bool = False

    checkpoint_key: Literal["auto", "ema_agent", "agent"] = "auto"
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True

    output_dir: Optional[str] = None
    output_json_path: Optional[str] = None
    save_per_traj: bool = True


def _to_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _scalar_bool(value) -> bool:
    return bool(_to_numpy(value).reshape(-1)[0])


def _scalar_int(value) -> int:
    return int(_to_numpy(value).reshape(-1)[0])


def _resolve_output_json_path(args: Args) -> str:
    if args.output_json_path is not None and len(args.output_json_path.strip()) > 0:
        return args.output_json_path

    ckpt_path = Path(args.checkpoint_pt_path).resolve()
    default_output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir is not None and len(args.output_dir.strip()) > 0
        else ckpt_path.parent.parent / "control_error"
    )
    return str(default_output_dir / f"{ckpt_path.stem}_control_error.json")


def _select_checkpoint_state_dict(
    checkpoint: dict,
    checkpoint_key: Literal["auto", "ema_agent", "agent"],
) -> tuple[dict, str]:
    if checkpoint_key == "ema_agent":
        if "ema_agent" not in checkpoint:
            raise KeyError("checkpoint does not contain key 'ema_agent'")
        return checkpoint["ema_agent"], "ema_agent"
    if checkpoint_key == "agent":
        if "agent" not in checkpoint:
            raise KeyError("checkpoint does not contain key 'agent'")
        return checkpoint["agent"], "agent"

    if "ema_agent" in checkpoint:
        return checkpoint["ema_agent"], "ema_agent"
    if "agent" in checkpoint:
        return checkpoint["agent"], "agent"
    if all(torch.is_tensor(v) for v in checkpoint.values()):
        return checkpoint, "state_dict"
    raise KeyError(
        "unable to locate policy weights in checkpoint; expected 'ema_agent', 'agent', "
        "or a plain state_dict"
    )


def _compute_expected_obs_cond_dim(
    envs,
    mas_long_conv_output_dim: int,
    short_window_horizon: int,
    legacy_short_mask_branch: bool = False,
) -> int:
    visual_feature_dim = 256
    obs_state_dim = int(envs.single_observation_space["state"].shape[1])
    if legacy_short_mask_branch:
        mas_long_feature_dim = 0
    else:
        mas_long_feature_dim = max(0, int(mas_long_conv_output_dim))
    if int(short_window_horizon) <= 0:
        mas_short_feature_dim = 0
    elif legacy_short_mask_branch:
        mas_short_feature_dim = 2 * int(short_window_horizon) * MAS_STEP_DIM
    else:
        mas_short_feature_dim = int(short_window_horizon) * MAS_STEP_DIM
    return visual_feature_dim + obs_state_dim + mas_long_feature_dim + mas_short_feature_dim


def _build_legacy_short_mask_feature(
    mas_short_window: torch.Tensor,
    mas_short_window_mask: torch.Tensor,
    obs_horizon: int,
    short_window_horizon: int,
    state_dtype: torch.dtype,
) -> torch.Tensor:
    expected_dim = int(short_window_horizon) * MAS_STEP_DIM
    if mas_short_window.shape[-1] != expected_dim:
        raise ValueError(
            "legacy short-window 兼容模式下 short window 维度不匹配: "
            f"got {tuple(mas_short_window.shape)}, expected_last_dim={expected_dim}"
        )
    if mas_short_window_mask.shape != mas_short_window.shape:
        raise ValueError(
            "legacy short-window 兼容模式下 short window mask 形状不匹配: "
            f"mas={tuple(mas_short_window.shape)}, mask={tuple(mas_short_window_mask.shape)}"
        )

    batch_size = mas_short_window.shape[0]
    mas_value = mas_short_window.to(dtype=state_dtype).reshape(
        batch_size,
        obs_horizon,
        short_window_horizon,
        MAS_STEP_DIM,
    )
    mas_mask = mas_short_window_mask.to(dtype=state_dtype).reshape(
        batch_size,
        obs_horizon,
        short_window_horizon,
        MAS_STEP_DIM,
    )
    raw_mas = mas_short_window.reshape(
        batch_size,
        obs_horizon,
        short_window_horizon,
        MAS_STEP_DIM,
    )
    mas_value[..., :-1] = mas_value[..., :-1] * mas_mask[..., :-1]
    mas_mask[..., -1] = raw_mas[..., -1].to(dtype=state_dtype)
    mas_value = mas_value.reshape(batch_size, obs_horizon, -1)
    mas_mask = mas_mask.reshape(batch_size, obs_horizon, -1)
    return torch.cat((mas_value, mas_mask), dim=-1)


def _validate_checkpoint_obs_cond_dim(args: Args, envs, state_dict: dict) -> None:
    obs_proj_weight = state_dict.get("noise_pred_net.obs_proj.0.weight", None)
    if obs_proj_weight is None:
        raise KeyError("checkpoint is missing noise_pred_net.obs_proj.0.weight")

    checkpoint_obs_dim = int(obs_proj_weight.shape[1])
    expected_obs_dim = _compute_expected_obs_cond_dim(
        envs=envs,
        mas_long_conv_output_dim=args.mas_long_conv_output_dim,
        short_window_horizon=args.short_window_horizon,
        legacy_short_mask_branch=args.legacy_short_mask_branch,
    )
    if checkpoint_obs_dim != expected_obs_dim:
        if args.legacy_short_mask_branch:
            raise ValueError(
                "checkpoint 的 obs_conditioning 维度与 legacy short-window+mask 布局不匹配。"
                f" checkpoint_obs_dim={checkpoint_obs_dim}, expected_obs_dim={expected_obs_dim}. "
                "请检查 short_window_horizon 是否和旧版训练配置一致。"
            )
        raise ValueError(
            "checkpoint 的 obs_conditioning 维度与当前固定 short-window 布局不匹配。"
            f" checkpoint_obs_dim={checkpoint_obs_dim}, expected_obs_dim={expected_obs_dim}. "
            "如果这是旧版 short-window 显式 mask 分支 checkpoint，请加 "
            "`--legacy-short-mask-branch`。"
        )


def _build_agent_runtime_args(args: Args):
    long_window_horizon = args.long_window_horizon
    if long_window_horizon is None:
        long_window_horizon = args.pred_horizon
    short_window_horizon = int(args.short_window_horizon)
    mas_long_conv_output_dim = int(args.mas_long_conv_output_dim)
    if args.legacy_short_mask_branch:
        short_window_horizon = short_window_horizon * 2
        mas_long_conv_output_dim = 0
    return SimpleNamespace(
        obs_horizon=int(args.obs_horizon),
        act_horizon=int(args.act_horizon),
        pred_horizon=int(args.pred_horizon),
        long_window_horizon=int(long_window_horizon),
        short_window_horizon=short_window_horizon,
        mas_long_encode_mode=args.mas_long_encode_mode,
        mas_long_conv_output_dim=mas_long_conv_output_dim,
        diffusion_step_embed_dim=int(args.diffusion_step_embed_dim),
    )


def _build_agent(
    args: Args,
    envs,
    device: torch.device,
    checkpoint_state_dict: dict,
) -> tuple[Agent, np.ndarray, np.ndarray, dict]:
    dataset_meta = load_dataset_meta(args.eval_demo_path)
    action_min, action_max = load_action_denorm_stats(args.eval_demo_path)
    state_min, state_max = _load_state_norm_stats_from_meta(dataset_meta)

    _validate_checkpoint_obs_cond_dim(
        args=args,
        envs=envs,
        state_dict=checkpoint_state_dict,
    )
    runtime_args = _build_agent_runtime_args(args=args)
    agent = Agent(envs, runtime_args).to(device)
    missing_keys, unexpected_keys = agent.load_state_dict(
        checkpoint_state_dict,
        strict=False,
    )
    allowed_missing = {
        "state_norm_min",
        "state_norm_max",
        "state_norm_scale",
        "has_state_normalizer",
    }
    missing_keys = [key for key in missing_keys if key not in allowed_missing]
    if len(missing_keys) > 0 or len(unexpected_keys) > 0:
        raise RuntimeError(
            "checkpoint 加载失败: "
            f"missing_keys={missing_keys}, unexpected_keys={unexpected_keys}"
        )

    agent.set_action_denormalizer(action_min, action_max, device)
    if state_min is not None and state_max is not None:
        agent.set_state_normalizer(state_min, state_max, device)

    return agent, action_min, action_max, dataset_meta


def _build_eval_envs(args: Args):
    env_kwargs = dict(
        control_mode=args.control_mode,
        reward_mode="sparse",
        obs_mode="rgb+depth",
        render_mode="rgb_array",
        render_backend=args.render_backend,
        human_render_camera_configs=dict(shader_pack="default"),
        max_episode_steps=int(args.max_episode_steps),
    )
    return make_eval_envs(
        args.env_id,
        args.num_eval_envs,
        args.sim_backend,
        env_kwargs,
        dict(obs_horizon=args.obs_horizon),
        video_dir=None,
        wrappers=[FlattenRGBDObservationWrapper],
    )


def _load_selected_eval_data(
    args: Args,
    device: torch.device,
) -> tuple[dict, list[int], list[int]]:
    eval_reset_seeds = infer_eval_reset_seeds_from_demo(
        args.eval_demo_path,
        num_traj=None,
        metadata_path=args.eval_demo_metadata_path,
    )
    if len(eval_reset_seeds) == 0:
        raise ValueError(
            "无法从 eval demo metadata 推断 reset seeds，CE 评估需要和 demo 对齐的 seeds。"
        )

    eval_data = load_ce_eval_data(
        data_path=args.eval_demo_path,
        device=device,
        num_traj=None,
    )
    if len(eval_reset_seeds) != len(eval_data["mas"]):
        raise ValueError(
            "eval seeds 数量和 eval traj 数量不一致: "
            f"{len(eval_reset_seeds)} vs {len(eval_data['mas'])}"
        )

    selected_indices = select_eval_demo_indices(
        total_demos=len(eval_data["mas"]),
        num_eval_demos=args.num_eval_demos,
    )
    eval_data = subset_eval_data(eval_data, selected_indices)
    eval_data["demo_local_indices"] = list(range(len(eval_data["mas"])))
    eval_reset_seeds = [eval_reset_seeds[i] for i in selected_indices]

    if len(eval_reset_seeds) == 0:
        raise ValueError("选中的 eval demos 为空")
    if len(eval_reset_seeds) % int(args.num_eval_envs) != 0:
        raise ValueError(
            "选中的 eval demos 必须能被 num_eval_envs 整除，"
            f"当前为 {len(eval_reset_seeds)} 和 {args.num_eval_envs}"
        )

    return eval_data, eval_reset_seeds, selected_indices


def _infer_batched_env_count(episode_info: dict) -> int:
    for value in episode_info.values():
        array = _to_numpy(value)
        if array.ndim >= 1:
            return int(array.shape[0])
    return 1


def _slice_episode_value(value, env_idx: int, total_envs: int):
    if torch.is_tensor(value):
        if value.ndim >= 1 and value.shape[0] == total_envs:
            return value[env_idx]
        return value
    array = np.asarray(value)
    if array.ndim >= 1 and array.shape[0] == total_envs:
        return array[env_idx]
    return value


def _extract_final_episode_records(info: dict, num_envs: int | None = None) -> list[dict]:
    final_info = info["final_info"]
    if not isinstance(final_info, dict):
        return [one_final_info["episode"] for one_final_info in final_info]

    episode_info = final_info["episode"]
    total_envs = int(num_envs or _infer_batched_env_count(episode_info))
    done_mask = info.get("_final_info", None)
    if done_mask is None:
        done_indices = list(range(total_envs))
    else:
        done_mask = _to_numpy(done_mask).reshape(-1).astype(bool)
        done_indices = np.flatnonzero(done_mask).tolist()
    return [
        {
            key: _slice_episode_value(value, env_idx, total_envs)
            for key, value in episode_info.items()
        }
        for env_idx in done_indices
    ]


def _rollout_one_eval_batch(
    agent,
    eval_envs,
    device: torch.device,
    sim_backend: str,
    batch_eval_data: dict,
    obs_horizon: int,
    long_window_horizon: int,
    short_window_horizon: int,
    legacy_short_mask_branch: bool,
    stpm_encoder,
    stpm_n_obs_steps: int,
    stpm_frame_gap: int,
    reset_seeds: list[int],
) -> tuple[dict, list[dict]]:
    validate_stpm_eval_setup(stpm_encoder, stpm_n_obs_steps, stpm_frame_gap)

    num_envs = eval_envs.num_envs
    if len(reset_seeds) != num_envs:
        raise ValueError(
            f"reset_seeds length must match num_envs={num_envs}, got {len(reset_seeds)}"
        )
    if len(batch_eval_data["mas"]) != num_envs:
        raise ValueError(
            "batch_eval_data length must match num_envs, "
            f"got {len(batch_eval_data['mas'])} vs {num_envs}"
        )

    agent_was_training = agent.training
    agent.eval()

    with torch.no_grad():
        eval_metrics = defaultdict(list)
        obs, info = eval_envs.reset(seed=[int(seed) for seed in reset_seeds])
        obs = prepare_batched_rollout_obs(obs, device, obs_horizon)
        histories = init_env_histories_from_reset_obs(obs, obs_horizon)
        traj_ids = torch.arange(num_envs, device=device, dtype=torch.long)
        step_ptr = torch.zeros((num_envs,), device=device, dtype=torch.long)
        executed_action_chunks = [[] for _ in range(num_envs)]

        while True:
            current_progress = predict_current_progress_from_histories(
                histories=histories,
                step_ptr=step_ptr,
                obs_horizon=obs_horizon,
                stpm_encoder=stpm_encoder,
                stpm_n_obs_steps=stpm_n_obs_steps,
                stpm_frame_gap=stpm_frame_gap,
                target_device=obs["state"].device,
                target_dtype=obs["state"].dtype,
            )
            dual_window_cond = build_dual_mas_window_condition_batch(
                mas_list=batch_eval_data["mas"],
                traj_lengths=batch_eval_data["traj_lengths"],
                traj_ids=traj_ids,
                current_progress=current_progress,
                obs_horizon=obs_horizon,
                long_window_horizon=long_window_horizon,
                short_window_horizon=short_window_horizon,
                device=obs["state"].device,
                dtype=obs["state"].dtype,
            )
            dual_window_mask_cond = build_dual_mas_window_condition_batch(
                mas_list=batch_eval_data["mas_mask"],
                traj_lengths=batch_eval_data["traj_lengths"],
                traj_ids=traj_ids,
                current_progress=current_progress,
                obs_horizon=obs_horizon,
                long_window_horizon=long_window_horizon,
                short_window_horizon=short_window_horizon,
                device=obs["state"].device,
                dtype=obs["state"].dtype,
            )
            obs["mas_long_window"] = dual_window_cond["mas_long_window"]
            obs["mas_short_window"] = dual_window_cond["mas_short_window"]
            obs["mas_long_window_mask"] = dual_window_mask_cond["mas_long_window"]
            obs["mas_short_window_mask"] = dual_window_mask_cond["mas_short_window"]
            if legacy_short_mask_branch:
                batch_size = obs["state"].shape[0]
                obs["mas_long_window_feature"] = obs["state"].new_empty(
                    (batch_size, obs_horizon, 0)
                )
                obs["mas_short_window_feature"] = _build_legacy_short_mask_feature(
                    mas_short_window=obs["mas_short_window"],
                    mas_short_window_mask=obs["mas_short_window_mask"],
                    obs_horizon=obs_horizon,
                    short_window_horizon=short_window_horizon,
                    state_dtype=obs["state"].dtype,
                )

            action_seq = agent.get_action(obs)
            action_seq_np = _to_numpy(action_seq).astype(np.float32)
            action_seq_env = action_seq_np if sim_backend == "physx_cpu" else action_seq

            executed_steps = 0
            next_obs = obs
            for action_idx in range(action_seq_np.shape[1]):
                raw_next_obs, rew, terminated, truncated, info = eval_envs.step(
                    action_seq_env[:, action_idx]
                )
                next_obs = prepare_batched_rollout_obs(raw_next_obs, device, obs_horizon)
                append_latest_rollout_frame(histories, next_obs)
                executed_steps = action_idx + 1
                if truncated.any():
                    break

            if executed_steps > 0:
                for env_idx in range(num_envs):
                    executed_action_chunks[env_idx].append(
                        action_seq_np[env_idx, :executed_steps].copy()
                    )

            obs = next_obs
            step_ptr = step_ptr + executed_steps

            if truncated.any():
                assert truncated.all() == truncated.any(), (
                    "all episodes should truncate at the same time for fair evaluation"
                )
                append_episode_metrics(eval_metrics, info)
                final_episodes = _extract_final_episode_records(info)
                rollout_records = []
                for env_idx, episode in enumerate(final_episodes):
                    action_chunks = executed_action_chunks[env_idx]
                    if len(action_chunks) == 0:
                        executed_actions_denorm = np.zeros((0, 7), dtype=np.float32)
                    else:
                        executed_actions_denorm = np.concatenate(action_chunks, axis=0)
                    rollout_records.append(
                        dict(
                            demo_local_idx=int(batch_eval_data["demo_local_indices"][env_idx]),
                            source_episode_id=int(
                                batch_eval_data["source_episode_ids"][env_idx]
                            ),
                            success_once=_scalar_bool(episode["success_once"]),
                            success_at_end=_scalar_bool(episode["success_at_end"]),
                            episode_len=_scalar_int(episode["episode_len"]),
                            executed_actions_denorm=executed_actions_denorm,
                        )
                    )
                break

    if agent_was_training:
        agent.train()
    return eval_metrics, rollout_records


def _summarize_eval_metrics(eval_metrics: dict) -> dict:
    summary = {}
    for key, values in eval_metrics.items():
        arr = np.asarray(values).reshape(-1)
        if arr.size == 0:
            continue
        summary[key] = float(np.mean(arr))
    return summary


def _compute_control_error_results(
    eval_data: dict,
    rollout_records: list[dict],
    action_min: np.ndarray,
    action_max: np.ndarray,
    save_per_traj: bool,
) -> list[dict]:
    per_traj_results = []
    for record in sorted(rollout_records, key=lambda item: int(item["demo_local_idx"])):
        demo_local_idx = int(record["demo_local_idx"])
        executed_actions_denorm = np.asarray(
            record["executed_actions_denorm"],
            dtype=np.float32,
        )
        executed_actions_norm = normalize_rollout_actions(
            executed_actions_denorm,
            action_min=action_min,
            action_max=action_max,
        )

        mas_t = _to_numpy(eval_data["mas"][demo_local_idx]).astype(np.float32)
        mask_t = _to_numpy(eval_data["mas_mask"][demo_local_idx]).astype(np.float32)
        ce_result = compute_control_error_for_traj(
            rollout_actions_norm=executed_actions_norm,
            mas_t=mas_t,
            mask_t=mask_t,
        )

        traj_result = dict(
            demo_local_idx=demo_local_idx,
            source_episode_id=int(record["source_episode_id"]),
            success_once=bool(record["success_once"]),
            success_at_end=bool(record["success_at_end"]),
            episode_len=int(record["episode_len"]),
            executed_steps=int(executed_actions_denorm.shape[0]),
            ce_traj=ce_result["ce_traj"],
            num_known_points=int(ce_result["num_known_points"]),
            point_error_raw_mean=ce_result["point_error_raw_mean"],
            point_error_normalized_mean=ce_result["point_error_normalized_mean"],
        )
        if save_per_traj:
            traj_result.update(
                executed_actions_denorm=executed_actions_denorm,
                executed_actions_norm=executed_actions_norm,
                known_step_indices=ce_result["known_step_indices"],
                best_match_t_list=ce_result["best_match_t_list"],
                point_errors=ce_result["point_errors"],
                point_errors_normalized=ce_result["point_errors_normalized"],
                point_errors_raw=ce_result["point_errors_raw"],
                known_dims_list=ce_result["known_dims_list"],
            )
        per_traj_results.append(traj_result)
    return per_traj_results


def main():
    # 1. 解析命令行并补齐和训练脚本一致的默认窗口长度。
    args = tyro.cli(Args)

    if args.long_window_horizon is None:
        args.long_window_horizon = args.pred_horizon
    args.long_window_horizon = int(args.long_window_horizon)

    # 2. 做最基本的参数合法性检查，避免 rollout 跑到一半才报错。
    if args.obs_horizon + args.act_horizon - 1 > args.pred_horizon:
        raise ValueError("obs_horizon + act_horizon - 1 must be <= pred_horizon")
    if args.long_window_horizon < 0:
        raise ValueError("long_window_horizon must be non-negative")

    # 3. 固定随机种子，并确定本次评估使用的 torch device。
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    output_json_path = _resolve_output_json_path(args)

    # 4. 加载 policy checkpoint，并选出真正用于 rollout 的 state_dict。
    checkpoint = torch.load(args.checkpoint_pt_path, map_location=device)
    checkpoint_state_dict, checkpoint_source = _select_checkpoint_state_dict(
        checkpoint=checkpoint,
        checkpoint_key=args.checkpoint_key,
    )

    # 5. 加载 eval demo、对齐 reset seeds，并构建 STPM 和环境。
    eval_data, eval_reset_seeds, selected_indices = _load_selected_eval_data(
        args=args,
        device=device,
    )
    stpm_encoder, stpm_n_obs_steps, stpm_frame_gap = build_eval_stpm_encoder(
        args.stpm_ckpt_path,
        args.stpm_config_path,
        device,
    )
    envs = _build_eval_envs(args)

    try:
        # 6. 构建 agent，并校验 checkpoint 与当前固定 short-window 布局一致。
        validate_only_mas_eval_layout(envs, stpm_encoder)
        agent, action_min, action_max, dataset_meta = _build_agent(
            args=args,
            envs=envs,
            device=device,
            checkpoint_state_dict=checkpoint_state_dict,
        )

        print(
            f"[ce-eval] running {len(eval_reset_seeds)} demos with num_eval_envs={args.num_eval_envs}, "
            f"checkpoint_source={checkpoint_source}"
        )

        # 7. 按 batch 做真实 rollout，记录 episode metrics 和实际执行动作轨迹。
        all_eval_metrics = defaultdict(list)
        all_rollout_records = []
        batch_indices_list = build_eval_batch_indices(
            total_items=len(eval_reset_seeds),
            batch_size=args.num_eval_envs,
        )
        for batch_indices in batch_indices_list:
            batch_seeds = [eval_reset_seeds[idx] for idx in batch_indices]
            batch_eval_data = subset_eval_data(eval_data, batch_indices)
            batch_metrics, batch_rollout_records = _rollout_one_eval_batch(
                agent=agent,
                eval_envs=envs,
                device=device,
                sim_backend=args.sim_backend,
                batch_eval_data=batch_eval_data,
                obs_horizon=args.obs_horizon,
                long_window_horizon=(
                    args.long_window_horizon
                    if (args.mas_long_conv_output_dim > 0 and not args.legacy_short_mask_branch)
                    else 0
                ),
                short_window_horizon=args.short_window_horizon,
                legacy_short_mask_branch=args.legacy_short_mask_branch,
                stpm_encoder=stpm_encoder,
                stpm_n_obs_steps=stpm_n_obs_steps,
                stpm_frame_gap=stpm_frame_gap,
                reset_seeds=batch_seeds,
            )
            for key, values in batch_metrics.items():
                all_eval_metrics[key].extend(np.asarray(values).reshape(-1).tolist())
            all_rollout_records.extend(batch_rollout_records)

        # 8. 将执行动作重新映回归一化控制空间，再逐条轨迹计算 control error。
        per_traj_results = _compute_control_error_results(
            eval_data=eval_data,
            rollout_records=all_rollout_records,
            action_min=action_min,
            action_max=action_max,
            save_per_traj=args.save_per_traj,
        )
        ce_summary = aggregate_control_error(per_traj_results)
        eval_metric_summary = _summarize_eval_metrics(all_eval_metrics)

        # 9. 汇总所有统计并落盘，便于后续做离线分析或画图。
        results = dict(
            **ce_summary,
            checkpoint_pt_path=str(Path(args.checkpoint_pt_path).resolve()),
            checkpoint_source=checkpoint_source,
            eval_demo_path=str(Path(args.eval_demo_path).resolve()),
            eval_demo_metadata_path=(
                str(Path(args.eval_demo_metadata_path).resolve())
                if args.eval_demo_metadata_path is not None
                else None
            ),
            output_json_path=str(Path(output_json_path).resolve()),
            num_eval_demos=len(selected_indices),
            selected_demo_indices=selected_indices,
            num_eval_envs=int(args.num_eval_envs),
            eval_metrics_mean=eval_metric_summary,
            dataset_meta=dataset_meta,
            run_config=asdict(args),
            per_traj_results=per_traj_results,
        )
        save_control_error_json(results, output_json_path)

        print(
            f"[ce-eval] saved results to {output_json_path}\n"
            f"[ce-eval] ce_all={ce_summary['ce_all']}, "
            f"ce_success={ce_summary['ce_success']}, "
            f"ce_failed={ce_summary['ce_failed']}"
        )
    finally:
        # 10. 无论成功或失败都关闭 env，避免残留渲染/物理资源。
        envs.close()


if __name__ == "__main__":
    main()
