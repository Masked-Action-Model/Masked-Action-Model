from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm

from utils.stpm_utils import (
    append_episode_metrics,
    append_latest_rollout_frame,
    build_dual_mas_window_condition_batch,
    init_env_histories_from_reset_obs,
    predict_current_progress_from_histories,
    prepare_batched_rollout_obs,
    validate_stpm_eval_setup,
)


def _to_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _scalar_bool(value) -> bool:
    return bool(_to_numpy(value).reshape(-1)[0])


def _scalar_int(value) -> int:
    return int(_to_numpy(value).reshape(-1)[0])


def _extract_final_episode_records(info: dict) -> list[dict]:
    if isinstance(info["final_info"], dict):
        return [info["final_info"]["episode"]]
    return [final_info["episode"] for final_info in info["final_info"]]


# --------------------------------------------------------------------------- #
# Main evaluation loop
# --------------------------------------------------------------------------- #
def evaluate_mas_window(
    n: int,
    agent,
    eval_envs,
    device,
    sim_backend: str,
    eval_mas_window_data: dict,
    obs_horizon: int,
    long_window_horizon: int,
    short_window_horizon: int,
    stpm_encoder,
    stpm_n_obs_steps: int,
    stpm_frame_gap: int,
    progress_bar: bool = True,
    reset_seed: int = None,
    reset_seeds: list[int] | None = None,
    return_progress_curves: bool = False,
    return_rollout_records: bool = False,
):
    """Evaluate dual-window MAS conditioning with online STPM progress inference."""

    validate_stpm_eval_setup(stpm_encoder, stpm_n_obs_steps, stpm_frame_gap)

    mas_list = eval_mas_window_data["mas"]
    if len(mas_list) == 0:
        raise ValueError("eval_mas_window_data['mas'] must be non-empty")
    num_traj = len(mas_list)
    traj_cursor = 0
    demo_local_indices = eval_mas_window_data.get("demo_local_indices")
    if demo_local_indices is None:
        demo_local_indices = list(range(num_traj))
    source_episode_ids = eval_mas_window_data.get("source_episode_ids")
    if source_episode_ids is None:
        source_episode_ids = demo_local_indices
    if len(demo_local_indices) != num_traj or len(source_episode_ids) != num_traj:
        raise ValueError("demo_local_indices/source_episode_ids length must match num_traj")

    agent.eval()
    if progress_bar:
        pbar = tqdm(total=n)
    with torch.no_grad():
        eval_metrics = defaultdict(list)
        progress_curve_records = [] if return_progress_curves else None
        rollout_records = [] if return_rollout_records else None
        num_envs = eval_envs.num_envs
        if reset_seeds is not None:
            if len(reset_seeds) != num_envs:
                raise ValueError(
                    f"reset_seeds length must match num_envs={num_envs}, got {len(reset_seeds)}"
                )
            if reset_seed is not None:
                raise ValueError("reset_seed and reset_seeds cannot be set at the same time")
            reset_seed_arg = [int(seed) for seed in reset_seeds]
            obs, info = eval_envs.reset(seed=reset_seed_arg)
        else:
            if reset_seed is None:
                obs, info = eval_envs.reset()
            else:
                if num_envs > 1:
                    print(
                        f"[eval-mas-window] reset_seed={reset_seed} with num_envs={num_envs}: "
                        "only env 0 uses that exact seed; other envs will use derived seeds."
                    )
                obs, info = eval_envs.reset(seed=reset_seed)
        obs = prepare_batched_rollout_obs(obs, device, obs_horizon)
        histories = init_env_histories_from_reset_obs(obs, obs_horizon)
        traj_ids = torch.tensor(
            [(traj_cursor + i) % num_traj for i in range(num_envs)],
            device=device,
            dtype=torch.long,
        )
        traj_cursor = (traj_cursor + num_envs) % num_traj
        step_ptr = torch.zeros((num_envs,), device=device, dtype=torch.long)
        episode_curve_steps = [[] for _ in range(num_envs)] if return_progress_curves else None
        episode_curve_progress = (
            [[] for _ in range(num_envs)] if return_progress_curves else None
        )
        executed_action_chunks = [[] for _ in range(num_envs)] if return_rollout_records else None
        eps_count = 0

    while eps_count < n:
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
            current_progress_for_curve = current_progress[:, 0]
            dual_window_cond = build_dual_mas_window_condition_batch(
                mas_list=eval_mas_window_data["mas"],
                traj_lengths=eval_mas_window_data["traj_lengths"],
                traj_ids=traj_ids,
                current_progress=current_progress,
                obs_horizon=obs_horizon,
                long_window_horizon=long_window_horizon,
                short_window_horizon=short_window_horizon,
                device=obs["state"].device,
                dtype=obs["state"].dtype,
            )
            dual_window_mask_cond = build_dual_mas_window_condition_batch(
                mas_list=eval_mas_window_data["mas_mask"],
                traj_lengths=eval_mas_window_data["traj_lengths"],
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

            if return_progress_curves:
                for env_idx in range(num_envs):
                    episode_curve_steps[env_idx].append(int(step_ptr[env_idx].item()))
                    episode_curve_progress[env_idx].append(
                        float(current_progress_for_curve[env_idx].item())
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

            if return_rollout_records and executed_steps > 0:
                for env_idx in range(num_envs):
                    executed_action_chunks[env_idx].append(
                        action_seq_np[env_idx, :executed_steps].copy()
                    )

            obs = next_obs
            step_ptr = step_ptr + executed_steps

            # episode end handling
            if truncated.any():
                assert truncated.all() == truncated.any(), (
                    "all episodes should truncate at the same time for fair evaluation with other algorithms"
                )
                append_episode_metrics(eval_metrics, info)
                completed_traj_ids = traj_ids.clone()
                final_episodes = (
                    _extract_final_episode_records(info) if return_rollout_records else None
                )

                if return_progress_curves:
                    final_progress = predict_current_progress_from_histories(
                        histories=histories,
                        step_ptr=step_ptr,
                        obs_horizon=obs_horizon,
                        stpm_encoder=stpm_encoder,
                        stpm_n_obs_steps=stpm_n_obs_steps,
                        stpm_frame_gap=stpm_frame_gap,
                        target_device=obs["state"].device,
                        target_dtype=obs["state"].dtype,
                    )[:, 0]
                    if isinstance(info["final_info"], dict):
                        success_flags = [
                            bool(
                                np.asarray(
                                    info["final_info"]["episode"]["success_at_end"]
                                ).reshape(-1)[0]
                            )
                        ]
                    else:
                        success_flags = [
                            bool(
                                np.asarray(final_info["episode"]["success_at_end"]).reshape(
                                    -1
                                )[0]
                            )
                            for final_info in info["final_info"]
                        ]

                    for env_idx in range(num_envs):
                        final_step = int(step_ptr[env_idx].item())
                        final_value = float(final_progress[env_idx].item())
                        if (
                            len(episode_curve_steps[env_idx]) == 0
                            or episode_curve_steps[env_idx][-1] != final_step
                        ):
                            episode_curve_steps[env_idx].append(final_step)
                            episode_curve_progress[env_idx].append(final_value)
                        else:
                            episode_curve_progress[env_idx][-1] = final_value

                        progress_curve_records.append(
                            dict(
                                demo_local_idx=int(
                                    demo_local_indices[int(completed_traj_ids[env_idx].item())]
                                ),
                                steps=np.asarray(
                                    episode_curve_steps[env_idx], dtype=np.int64
                                ),
                                progress=np.asarray(
                                    episode_curve_progress[env_idx], dtype=np.float32
                                ),
                                success_at_end=success_flags[env_idx],
                            )
                        )

                if return_rollout_records:
                    for env_idx in range(num_envs):
                        action_chunks = executed_action_chunks[env_idx]
                        if len(action_chunks) == 0:
                            executed_actions_denorm = np.zeros((0, 7), dtype=np.float32)
                        else:
                            executed_actions_denorm = np.concatenate(action_chunks, axis=0)
                        traj_idx = int(completed_traj_ids[env_idx].item())
                        episode = final_episodes[env_idx]
                        rollout_records.append(
                            dict(
                                demo_local_idx=int(demo_local_indices[traj_idx]),
                                source_episode_id=int(source_episode_ids[traj_idx]),
                                success_once=_scalar_bool(episode["success_once"]),
                                success_at_end=_scalar_bool(episode["success_at_end"]),
                                episode_len=_scalar_int(episode["episode_len"]),
                                executed_actions_denorm=executed_actions_denorm,
                            )
                        )

                eps_count += num_envs
                traj_ids = torch.tensor(
                    [(traj_cursor + i) % num_traj for i in range(num_envs)],
                    device=device,
                    dtype=torch.long,
                )
                traj_cursor = (traj_cursor + num_envs) % num_traj
                step_ptr.zero_()

                if eps_count < n:
                    if reset_seeds is not None:
                        obs, info = eval_envs.reset(seed=reset_seed_arg)
                    elif reset_seed is None:
                        obs, info = eval_envs.reset()
                    else:
                        obs, info = eval_envs.reset(seed=reset_seed)
                    obs = prepare_batched_rollout_obs(obs, device, obs_horizon)
                    histories = init_env_histories_from_reset_obs(obs, obs_horizon)
                    if return_progress_curves:
                        episode_curve_steps = [[] for _ in range(num_envs)]
                        episode_curve_progress = [[] for _ in range(num_envs)]
                    if return_rollout_records:
                        executed_action_chunks = [[] for _ in range(num_envs)]

                if progress_bar:
                    pbar.update(num_envs)

    agent.train()
    for k in eval_metrics.keys():
        eval_metrics[k] = np.stack(eval_metrics[k])
    if return_progress_curves and return_rollout_records:
        return eval_metrics, progress_curve_records, rollout_records
    if return_progress_curves:
        return eval_metrics, progress_curve_records
    if return_rollout_records:
        return eval_metrics, rollout_records
    return eval_metrics
