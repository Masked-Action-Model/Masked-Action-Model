from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
import sys

import numpy as np
import torch
import tyro
from tqdm import tqdm

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

try:
    from eval_ce import (
        Args as BaseArgs,
        _build_agent,
        _build_eval_envs,
        _build_legacy_short_mask_feature,
        _compute_control_error_results,
        _extract_final_episode_records,
        _load_selected_eval_data,
        _scalar_bool,
        _scalar_int,
        _select_checkpoint_state_dict,
        _summarize_eval_metrics,
    )
except ModuleNotFoundError:
    from examples.baselines.diffusion_policy.eval_ce import (
        Args as BaseArgs,
        _build_agent,
        _build_eval_envs,
        _build_legacy_short_mask_feature,
        _compute_control_error_results,
        _extract_final_episode_records,
        _load_selected_eval_data,
        _scalar_bool,
        _scalar_int,
        _select_checkpoint_state_dict,
        _summarize_eval_metrics,
    )

from train_mas_window import (
    build_eval_batch_indices,
    build_eval_stpm_encoder,
    validate_only_mas_eval_layout,
)
from utils.control_error_utils import aggregate_control_error, save_control_error_json
from utils.inpainting_utils import (
    build_current_inpaint_mas_mask,
    sample_inpaint_action_chunk,
)
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
class Args(BaseArgs):
    jump_length: int = 1
    num_resample: int = 0
    save_per_traj: bool = False


def _resolve_output_json_path(args: Args) -> str:
    if args.output_json_path is not None and len(args.output_json_path.strip()) > 0:
        return args.output_json_path

    ckpt_path = Path(args.checkpoint_pt_path).resolve()
    if args.output_dir is not None and len(args.output_dir.strip()) > 0:
        output_dir = Path(args.output_dir).resolve()
    else:
        output_dir = ckpt_path.parent.parent / "control_error_inpaint"
    filename = (
        f"{ckpt_path.stem}_inpaint_j{int(args.jump_length)}_r{int(args.num_resample)}.json"
    )
    return str(output_dir / filename)


def _summarize_success_results(per_traj_results: list[dict]) -> dict:
    num_rollouts = len(per_traj_results)
    num_success_once = sum(bool(r.get("success_once", False)) for r in per_traj_results)
    num_success_at_end = sum(
        bool(r.get("success_at_end", False)) for r in per_traj_results
    )
    if num_rollouts <= 0:
        success_once_rate = None
        success_at_end_rate = None
    else:
        success_once_rate = float(num_success_once / num_rollouts)
        success_at_end_rate = float(num_success_at_end / num_rollouts)
    return dict(
        num_rollouts=num_rollouts,
        num_success_once=num_success_once,
        num_failed_once=num_rollouts - num_success_once,
        success_once_rate=success_once_rate,
        num_success_at_end=num_success_at_end,
        num_failed_at_end=num_rollouts - num_success_at_end,
        success_at_end_rate=success_at_end_rate,
    )


def _rollout_one_eval_batch_inpaint(
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
    jump_length: int,
    num_resample: int,
    progress_desc: str | None = None,
    rollout_step_total: int | None = None,
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
    step_pbar = None
    if progress_desc is not None and rollout_step_total is not None and rollout_step_total > 0:
        step_pbar = tqdm(
            total=int(rollout_step_total),
            desc=progress_desc,
            unit="step",
            leave=False,
            dynamic_ncols=True,
        )

    try:
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

                mas_inpaint, mas_inpaint_mask = build_current_inpaint_mas_mask(
                    mas_list=batch_eval_data["mas"],
                    mas_mask_list=batch_eval_data["mas_mask"],
                    traj_ids=traj_ids,
                    current_progress=current_progress,
                    pred_horizon=agent.pred_horizon,
                    device=obs["state"].device,
                    dtype=obs["state"].dtype,
                )

                action_seq = sample_inpaint_action_chunk(
                    agent=agent,
                    obs_seq=obs,
                    mas_inpaint=mas_inpaint,
                    mas_inpaint_mask=mas_inpaint_mask,
                    jump_length=jump_length,
                    num_resample=num_resample,
                )
                action_seq_np = action_seq.detach().cpu().numpy().astype(np.float32)
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
                    if step_pbar is not None:
                        remaining = max(int(step_pbar.total) - int(step_pbar.n), 0)
                        step_pbar.update(min(int(executed_steps), remaining))

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
    finally:
        if step_pbar is not None:
            step_pbar.close()
    if agent_was_training:
        agent.train()
    return eval_metrics, rollout_records


def main():
    args = tyro.cli(Args)

    if args.long_window_horizon is None:
        args.long_window_horizon = args.pred_horizon
    args.long_window_horizon = int(args.long_window_horizon)

    if args.obs_horizon + args.act_horizon - 1 > args.pred_horizon:
        raise ValueError("obs_horizon + act_horizon - 1 must be <= pred_horizon")
    if args.long_window_horizon < 0:
        raise ValueError("long_window_horizon must be non-negative")
    if args.jump_length < 1:
        raise ValueError("jump_length must be >= 1")
    if args.num_resample < 0:
        raise ValueError("num_resample must be >= 0")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    output_json_path = _resolve_output_json_path(args)

    checkpoint = torch.load(args.checkpoint_pt_path, map_location=device)
    checkpoint_state_dict, checkpoint_source = _select_checkpoint_state_dict(
        checkpoint=checkpoint,
        checkpoint_key=args.checkpoint_key,
    )

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
        validate_only_mas_eval_layout(envs, stpm_encoder)
        agent, action_min, action_max, dataset_meta = _build_agent(
            args=args,
            envs=envs,
            device=device,
            checkpoint_state_dict=checkpoint_state_dict,
        )

        print(
            f"[ce-inpaint-eval] running {len(eval_reset_seeds)} demos with "
            f"num_eval_envs={args.num_eval_envs}, checkpoint_source={checkpoint_source}, "
            f"jump_length={args.jump_length}, num_resample={args.num_resample}"
        )

        all_eval_metrics = defaultdict(list)
        all_rollout_records = []
        batch_indices_list = build_eval_batch_indices(
            total_items=len(eval_reset_seeds),
            batch_size=args.num_eval_envs,
        )
        overall_pbar = tqdm(
            total=len(eval_reset_seeds),
            desc="[ce-inpaint-eval] demos",
            unit="demo",
            dynamic_ncols=True,
        )
        try:
            for batch_idx, batch_indices in enumerate(batch_indices_list):
                batch_seeds = [eval_reset_seeds[idx] for idx in batch_indices]
                batch_eval_data = _subset_eval_data(eval_data, batch_indices)
                batch_metrics, batch_rollout_records = _rollout_one_eval_batch_inpaint(
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
                    jump_length=args.jump_length,
                    num_resample=args.num_resample,
                    progress_desc=(
                        f"[ce-inpaint-eval] batch {batch_idx + 1}/{len(batch_indices_list)}"
                    ),
                    rollout_step_total=args.max_episode_steps,
                )
                for key, values in batch_metrics.items():
                    all_eval_metrics[key].extend(np.asarray(values).reshape(-1).tolist())
                all_rollout_records.extend(batch_rollout_records)
                overall_pbar.update(len(batch_indices))
                completed = len(all_rollout_records)
                num_success_once = sum(
                    bool(record.get("success_once", False)) for record in all_rollout_records
                )
                num_success_at_end = sum(
                    bool(record.get("success_at_end", False))
                    for record in all_rollout_records
                )
                if completed > 0:
                    overall_pbar.set_postfix(
                        success_once=f"{num_success_once}/{completed}",
                        success_end=f"{num_success_at_end}/{completed}",
                    )
        finally:
            overall_pbar.close()

        per_traj_results = _compute_control_error_results(
            eval_data=eval_data,
            rollout_records=all_rollout_records,
            action_min=action_min,
            action_max=action_max,
            save_per_traj=args.save_per_traj,
        )
        ce_summary = aggregate_control_error(per_traj_results)
        success_summary = _summarize_success_results(per_traj_results)
        eval_metric_summary = _summarize_eval_metrics(all_eval_metrics)

        results = dict(
            **ce_summary,
            **success_summary,
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
            f"[ce-inpaint-eval] saved results to {output_json_path}\n"
            f"[ce-inpaint-eval] success_once="
            f"{success_summary['num_success_once']}/{success_summary['num_rollouts']} "
            f"({success_summary['success_once_rate']}), "
            f"success_at_end="
            f"{success_summary['num_success_at_end']}/{success_summary['num_rollouts']} "
            f"({success_summary['success_at_end_rate']})\n"
            f"[ce-inpaint-eval] ce_all={ce_summary['ce_all']}, "
            f"ce_success={ce_summary['ce_success']}, "
            f"ce_failed={ce_summary['ce_failed']}"
        )
    finally:
        envs.close()


def _subset_eval_data(eval_data: dict, indices: list[int]):
    subset = {}
    for key, value in eval_data.items():
        if isinstance(value, list):
            subset[key] = [value[i] for i in indices]
        else:
            subset[key] = value
    return subset


if __name__ == "__main__":
    main()
