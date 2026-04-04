from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm

from utils.stpm_utils import (
    append_episode_metrics,
    append_latest_rollout_frame,
    build_ap_progress_condition_batch,
    build_mas_condition_batch,
    build_mas_progress_condition_batch,
    init_env_histories_from_reset_obs,
    predict_current_progress_from_histories,
    predict_progress_from_histories,
    prepare_batched_rollout_obs,
    validate_stpm_eval_setup,
)


# --------------------------------------------------------------------------- #
# Main evaluation loop
# --------------------------------------------------------------------------- #
def evaluate_only_mas(
    n: int,
    agent,
    eval_envs,
    device,
    sim_backend: str,
    eval_mam_data: dict,
    obs_horizon: int,
    pred_horizon: int,
    progress_mode: str,
    stpm_encoder,
    stpm_n_obs_steps: int,
    stpm_frame_gap: int,
    progress_bar: bool = True,
    reset_seed: int = None,
    reset_seeds: list[int] | None = None,
    return_progress_curves: bool = False,
):
    """Evaluate only-mas conditioning with online STPM progress inference.

    `pred_horizon` is kept for interface compatibility with the training script.
    """
    del pred_horizon

    # preparation: validate eval_mam_data and STPM setup before starting evaluation rollouts
    validate_stpm_eval_setup(stpm_encoder, stpm_n_obs_steps, stpm_frame_gap)

    mas_flat_list = eval_mam_data["mas_flat"]
    mas_mask_flat_list = eval_mam_data["mas_mask_flat"]
    if len(mas_flat_list) == 0:
        raise ValueError("eval_mam_data['mas_flat'] must be non-empty")
    num_traj = len(mas_flat_list)
    traj_cursor = 0

    # set agent to eval mode
    agent.eval()
    if progress_bar:
        pbar = tqdm(total=n)
    with torch.no_grad():
        eval_metrics = defaultdict(list)
        progress_curve_records = [] if return_progress_curves else None
        num_envs = eval_envs.num_envs
        if reset_seeds is not None:
            if len(reset_seeds) != num_envs:
                raise ValueError(
                    f"reset_seeds length must match num_envs={num_envs}, got {len(reset_seeds)}"
                )
            if reset_seed is not None:
                raise ValueError("reset_seed and reset_seeds cannot be set at the same time")
            reset_seed_arg = [int(seed) for seed in reset_seeds]
        else:
            reset_seed_arg = reset_seed
            if reset_seed is not None and num_envs > 1:
                print(
                    f"[eval-only-mas] reset_seed={reset_seed} with num_envs={num_envs}: "
                    "only env 0 uses that exact seed; other envs will use derived seeds."
                )
        obs, info = eval_envs.reset(seed=reset_seed_arg)
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
        eps_count = 0

    # preparing observations for the first step
    while eps_count < n:
            if progress_mode == "MLP8":
                obs["progress"] = predict_progress_from_histories(
                    histories=histories,
                    step_ptr=step_ptr,
                    obs_horizon=obs_horizon,
                    stpm_encoder=stpm_encoder,
                    stpm_n_obs_steps=stpm_n_obs_steps,
                    stpm_frame_gap=stpm_frame_gap,
                    target_device=obs["state"].device,
                    target_dtype=obs["state"].dtype,
                )
                current_progress_for_curve = obs["progress"][:, -1, 0]
            elif progress_mode == "MAS8":
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
                obs["progress"] = build_mas_progress_condition_batch(
                    mas_list=eval_mam_data["mas"],
                    traj_ids=traj_ids,
                    current_progress=current_progress,
                    obs_horizon=obs_horizon,
                    device=obs["state"].device,
                    dtype=obs["state"].dtype,
                )
                obs["progress_mask"] = build_mas_progress_condition_batch(
                    mas_list=eval_mam_data["mas_mask"],
                    traj_ids=traj_ids,
                    current_progress=current_progress,
                    obs_horizon=obs_horizon,
                    device=obs["state"].device,
                    dtype=obs["state"].dtype,
                )
                current_progress_for_curve = current_progress[:, 0]
            elif progress_mode == "AP8":
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
                obs["progress"] = build_ap_progress_condition_batch(
                    current_progress=current_progress,
                    traj_lengths=eval_mam_data["traj_lengths"],
                    traj_ids=traj_ids,
                    obs_horizon=obs_horizon,
                    device=obs["state"].device,
                    dtype=obs["state"].dtype,
                )
                current_progress_for_curve = current_progress[:, 0]
            else:
                raise ValueError(f"Unsupported progress_mode: {progress_mode}")

            if return_progress_curves:
                for env_idx in range(num_envs):
                    episode_curve_steps[env_idx].append(int(step_ptr[env_idx].item()))
                    episode_curve_progress[env_idx].append(
                        float(current_progress_for_curve[env_idx].item())
                    )

            # Keep MAS conditioning on the original flattened demo signal.
            obs["mas"] = build_mas_condition_batch(
                mas_flat_list=mas_flat_list,
                traj_ids=traj_ids,
                obs_horizon=obs_horizon,
                device=obs["state"].device,
                dtype=obs["state"].dtype,
            )
            obs["mas_mask"] = build_mas_condition_batch(
                mas_flat_list=mas_mask_flat_list,
                traj_ids=traj_ids,
                obs_horizon=obs_horizon,
                device=obs["state"].device,
                dtype=obs["state"].dtype,
            )

            # rollout with the agent
            action_seq = agent.get_action(obs)
            if sim_backend == "physx_cpu":
                action_seq = action_seq.cpu().numpy()

            executed_steps = 0
            next_obs = obs
            for action_idx in range(action_seq.shape[1]):
                raw_next_obs, rew, terminated, truncated, info = eval_envs.step(
                    action_seq[:, action_idx]
                )
                next_obs = prepare_batched_rollout_obs(raw_next_obs, device, obs_horizon)
                append_latest_rollout_frame(histories, next_obs)
                executed_steps = action_idx + 1
                if truncated.any():
                    break

            obs = next_obs
            step_ptr = step_ptr + executed_steps

            # episode end handling
            if truncated.any():
                assert truncated.all() == truncated.any(), (
                    "all episodes should truncate at the same time for fair evaluation with other algorithms"
                )
                append_episode_metrics(eval_metrics, info)
                completed_traj_ids = traj_ids.clone()

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
                                demo_local_idx=int(completed_traj_ids[env_idx].item()),
                                steps=np.asarray(
                                    episode_curve_steps[env_idx], dtype=np.int64
                                ),
                                progress=np.asarray(
                                    episode_curve_progress[env_idx], dtype=np.float32
                                ),
                                success_at_end=success_flags[env_idx],
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

                if progress_bar:
                    pbar.update(num_envs)

    agent.train()
    for k in eval_metrics.keys():
        eval_metrics[k] = np.stack(eval_metrics[k])
    if return_progress_curves:
        return eval_metrics, progress_curve_records
    return eval_metrics
