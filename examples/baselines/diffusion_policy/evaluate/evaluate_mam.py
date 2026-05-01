from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm

from utils.draw_p_t_curve_utils import save_progress_curve_for_video
from utils.eval_video_sampling_utils import build_eval_batches
from utils.inpainting_utils import (
    build_current_inpaint_mas_mask,
    sample_inpaint_action_chunk,
)
from utils.load_eval_data_utils import subset_eval_data
from utils.stpm_utils import (
    append_episode_metrics,
    append_latest_rollout_frame,
    build_dual_mas_window_condition_batch,
    init_env_histories_from_reset_obs,
    predict_current_progress_from_histories,
    prepare_batched_rollout_obs,
    validate_stpm_eval_setup,
)
from utils.video_utils import (
    collect_failed_video,
    collect_success_video,
    delete_new_video_files,
    snapshot_video_files,
)


def _to_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _scalar_bool(value) -> bool:
    return bool(_to_numpy(value).reshape(-1)[0])


def _scalar_int(value) -> int:
    return int(_to_numpy(value).reshape(-1)[0])


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
    long_window_backward_length: int = 0,
    long_window_forward_length: int | None = None,
    progress_bar: bool = True,
    reset_seed: int = None,
    reset_seeds: list[int] | None = None,
    return_progress_curves: bool = False,
    return_rollout_records: bool = False,
    inpainting: bool = False,
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
                long_window_backward_length=long_window_backward_length,
                long_window_forward_length=long_window_forward_length,
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
                long_window_backward_length=long_window_backward_length,
                long_window_forward_length=long_window_forward_length,
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

            if inpainting:
                mas_inpaint, mas_inpaint_mask = build_current_inpaint_mas_mask(
                    mas_list=eval_mas_window_data["mas"],
                    mas_mask_list=eval_mas_window_data["mas_mask"],
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
                    jump_length=0,
                    num_resample=0,
                )
            else:
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
                    _extract_final_episode_records(info, num_envs=num_envs)
                    if (return_progress_curves or return_rollout_records)
                    else None
                )
                if final_episodes is not None and len(final_episodes) != num_envs:
                    raise ValueError(
                        "final episode record count does not match num_envs: "
                        f"{len(final_episodes)} vs {num_envs}"
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
                    success_flags = [
                        _scalar_bool(episode["success_at_end"]) for episode in final_episodes
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


def _slice_metric_dict(metrics: dict, valid_batch_size: int) -> dict:
    sliced = {}
    for key, value in metrics.items():
        array = np.asarray(value).reshape(-1)[:valid_batch_size]
        sliced[key] = array
    return sliced


def _summarize_metric_dict(metrics: dict) -> dict:
    summary = {}
    num_episodes = 0
    for key, value in metrics.items():
        array = np.asarray(value).reshape(-1)
        num_episodes = max(num_episodes, int(array.shape[0]))
        summary[key] = float(np.mean(array)) if array.size > 0 else float("nan")
    summary["num_episodes"] = num_episodes
    return summary


def _group_indices_by_labels(
    labels: list[str] | None,
    total_items: int,
    label_name: str,
) -> dict[str, list[int]]:
    if labels is None:
        raise ValueError(f"MAM eval requires eval_mas_window_data[{label_name!r}]")
    if len(labels) != total_items:
        raise ValueError(
            f"{label_name} length must match eval trajectory count: "
            f"{len(labels)} vs {total_items}"
        )
    grouped = defaultdict(list)
    for idx, label in enumerate(labels):
        grouped[str(label)].append(idx)
    return dict(sorted(grouped.items(), key=lambda item: item[0]))


def group_eval_data_by_mask_type(eval_mas_window_data: dict) -> dict[str, list[int]]:
    total_items = len(eval_mas_window_data["mas"])
    return _group_indices_by_labels(
        eval_mas_window_data.get("mask_types"),
        total_items=total_items,
        label_name="mask_types",
    )


def group_eval_data_by_mask_type_slot(eval_mas_window_data: dict) -> dict[str, list[int]]:
    total_items = len(eval_mas_window_data["mas"])
    slot_labels = eval_mas_window_data.get("mask_type_slots")
    if slot_labels is None:
        slot_labels = eval_mas_window_data.get("mask_types")
    return _group_indices_by_labels(
        slot_labels,
        total_items=total_items,
        label_name="mask_type_slots",
    )


def _append_metric_dict(dst: dict[str, list[float]], src: dict) -> None:
    for key, value in src.items():
        dst[key].extend(np.asarray(value).reshape(-1).tolist())


def _metric_dict_from_lists(metric_lists: dict[str, list[float]]) -> dict:
    return {key: np.asarray(values) for key, values in metric_lists.items()}


def evaluate_one_mask_group(
    group_label: str,
    group_indices: list[int],
    mask_type_label: str,
    mask_type_slot_label: str,
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
    long_window_backward_length: int = 0,
    long_window_forward_length: int | None = None,
    progress_bar: bool = False,
    reset_seeds: list[int] | None = None,
    return_progress_curves: bool = False,
    return_rollout_records: bool = False,
    capture_indices: set[int] | None = None,
    video_dir: str | None = None,
    iteration: int | None = None,
    eval_traj_ids: list[int] | None = None,
    inpainting: bool = False,
):
    if len(group_indices) == 0:
        raise ValueError(f"group {group_label!r} is empty")
    if video_dir is not None and iteration is None:
        raise ValueError("iteration is required when video_dir is provided")

    aggregated_metrics = defaultdict(list)
    aggregated_progress_records = [] if return_progress_curves else None
    aggregated_rollout_records = [] if return_rollout_records else None
    batch_size = int(eval_envs.num_envs)

    eval_batches = build_eval_batches(
        group_indices,
        batch_size,
        capture_indices=capture_indices,
    )
    for eval_batch in eval_batches:
        batch_indices = eval_batch.indices
        valid_batch_size = eval_batch.valid_count
        batch_eval_data = subset_eval_data(eval_mas_window_data, batch_indices)
        batch_reset_seeds = None
        if reset_seeds is not None:
            batch_reset_seeds = [reset_seeds[i] for i in batch_indices]
        batch_return_progress_curves = return_progress_curves and (
            video_dir is None or eval_batch.capture_index is not None
        )
        video_snapshot = None
        if video_dir is not None:
            video_snapshot = snapshot_video_files(video_dir)

        one_result = evaluate_mas_window(
            valid_batch_size,
            agent,
            eval_envs,
            device,
            sim_backend,
            eval_mas_window_data=batch_eval_data,
            obs_horizon=obs_horizon,
            long_window_horizon=long_window_horizon,
            short_window_horizon=short_window_horizon,
            long_window_backward_length=long_window_backward_length,
            long_window_forward_length=long_window_forward_length,
            stpm_encoder=stpm_encoder,
            stpm_n_obs_steps=stpm_n_obs_steps,
            stpm_frame_gap=stpm_frame_gap,
            progress_bar=progress_bar,
            reset_seeds=batch_reset_seeds,
            return_progress_curves=batch_return_progress_curves,
            return_rollout_records=return_rollout_records,
            inpainting=inpainting,
        )

        if batch_return_progress_curves and return_rollout_records:
            batch_metrics, batch_progress_records, batch_rollout_records = one_result
        elif batch_return_progress_curves:
            batch_metrics, batch_progress_records = one_result
            batch_rollout_records = None
        elif return_rollout_records:
            batch_metrics, batch_rollout_records = one_result
            batch_progress_records = None
        else:
            batch_metrics = one_result
            batch_progress_records = None
            batch_rollout_records = None

        if video_dir is not None:
            if eval_batch.capture_index is not None:
                recorded_env_idx = 0
                success_at_end = bool(
                    np.asarray(batch_metrics["success_at_end"]).reshape(-1)[recorded_env_idx]
                )
                demo_idx = (
                    eval_traj_ids[int(eval_batch.capture_index)]
                    if eval_traj_ids is not None
                    else int(eval_batch.capture_index)
                )
                if success_at_end:
                    archived_video_path = collect_success_video(
                        video_dir=video_dir,
                        previous_snapshot=video_snapshot,
                        iteration=int(iteration),
                        demo_idx=demo_idx,
                    )
                else:
                    archived_video_path = collect_failed_video(
                        video_dir=video_dir,
                        previous_snapshot=video_snapshot,
                        iteration=int(iteration),
                        demo_idx=demo_idx,
                    )
                if (
                    archived_video_path is not None
                    and batch_progress_records is not None
                    and recorded_env_idx < len(batch_progress_records)
                ):
                    save_progress_curve_for_video(
                        video_path=archived_video_path,
                        video_dir=video_dir,
                        timesteps=batch_progress_records[recorded_env_idx]["steps"],
                        progress=batch_progress_records[recorded_env_idx]["progress"],
                    )
            else:
                delete_new_video_files(video_dir, video_snapshot)

        batch_metrics = _slice_metric_dict(batch_metrics, valid_batch_size)
        _append_metric_dict(aggregated_metrics, batch_metrics)

        if return_progress_curves and batch_progress_records is not None:
            for record in batch_progress_records[:valid_batch_size]:
                record = dict(record)
                record["mask_type"] = mask_type_label
                record["mask_type_slot"] = mask_type_slot_label
                aggregated_progress_records.append(record)

        if return_rollout_records:
            for record in batch_rollout_records[:valid_batch_size]:
                record = dict(record)
                record["mask_type"] = mask_type_label
                record["mask_type_slot"] = mask_type_slot_label
                aggregated_rollout_records.append(record)

    grouped_metrics = _metric_dict_from_lists(aggregated_metrics)
    if return_progress_curves and return_rollout_records:
        return grouped_metrics, aggregated_progress_records, aggregated_rollout_records
    if return_progress_curves:
        return grouped_metrics, aggregated_progress_records
    if return_rollout_records:
        return grouped_metrics, aggregated_rollout_records
    return grouped_metrics


def evaluate_mam(
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
    long_window_backward_length: int = 0,
    long_window_forward_length: int | None = None,
    progress_bar: bool = True,
    reset_seed: int = None,
    reset_seeds: list[int] | None = None,
    return_progress_curves: bool = False,
    return_rollout_records: bool = False,
    capture_indices: set[int] | None = None,
    video_dir: str | None = None,
    iteration: int | None = None,
    eval_traj_ids: list[int] | None = None,
    inpainting: bool = False,
):
    if reset_seed is not None:
        raise ValueError("MAM eval expects reset_seeds instead of a single reset_seed")
    total_items = len(eval_mas_window_data["mas"])
    if n != total_items:
        raise ValueError(
            f"MAM eval expects n to equal len(eval_mas_window_data['mas'])={total_items}, got {n}"
        )
    if reset_seeds is not None and len(reset_seeds) != total_items:
        raise ValueError(
            "reset_seeds length must match eval trajectory count: "
            f"{len(reset_seeds)} vs {total_items}"
        )

    mask_types = eval_mas_window_data.get("mask_types")
    mask_type_slots = eval_mas_window_data.get("mask_type_slots")
    if mask_type_slots is None:
        mask_type_slots = mask_types

    _group_indices_by_labels(mask_types, total_items=total_items, label_name="mask_types")
    slot_groups = _group_indices_by_labels(
        mask_type_slots,
        total_items=total_items,
        label_name="mask_type_slots",
    )

    aggregated_metrics = defaultdict(list)
    per_mask_type_metric_lists = defaultdict(lambda: defaultdict(list))
    per_mask_slot_summary = {}
    progress_curve_records = [] if return_progress_curves else None
    rollout_records = [] if return_rollout_records else None

    for mask_type_slot, group_indices in slot_groups.items():
        mask_type = str(mask_types[group_indices[0]])
        one_result = evaluate_one_mask_group(
            group_label=mask_type_slot,
            group_indices=group_indices,
            mask_type_label=mask_type,
            mask_type_slot_label=str(mask_type_slot),
            agent=agent,
            eval_envs=eval_envs,
            device=device,
            sim_backend=sim_backend,
            eval_mas_window_data=eval_mas_window_data,
            obs_horizon=obs_horizon,
            long_window_horizon=long_window_horizon,
            short_window_horizon=short_window_horizon,
            long_window_backward_length=long_window_backward_length,
            long_window_forward_length=long_window_forward_length,
            stpm_encoder=stpm_encoder,
            stpm_n_obs_steps=stpm_n_obs_steps,
            stpm_frame_gap=stpm_frame_gap,
            progress_bar=progress_bar,
            reset_seeds=reset_seeds,
            return_progress_curves=return_progress_curves,
            return_rollout_records=return_rollout_records,
            capture_indices=capture_indices,
            video_dir=video_dir,
            iteration=iteration,
            eval_traj_ids=eval_traj_ids,
            inpainting=inpainting,
        )

        if return_progress_curves and return_rollout_records:
            group_metrics, group_progress_records, group_rollout_records = one_result
        elif return_progress_curves:
            group_metrics, group_progress_records = one_result
            group_rollout_records = None
        elif return_rollout_records:
            group_metrics, group_rollout_records = one_result
            group_progress_records = None
        else:
            group_metrics = one_result
            group_progress_records = None
            group_rollout_records = None

        _append_metric_dict(aggregated_metrics, group_metrics)
        _append_metric_dict(per_mask_type_metric_lists[mask_type], group_metrics)
        per_mask_slot_summary[str(mask_type_slot)] = _summarize_metric_dict(group_metrics)

        if return_progress_curves:
            progress_curve_records.extend(group_progress_records)
        if return_rollout_records:
            rollout_records.extend(group_rollout_records)

    per_mask_type_summary = {
        str(mask_type): _summarize_metric_dict(_metric_dict_from_lists(metric_lists))
        for mask_type, metric_lists in sorted(
            per_mask_type_metric_lists.items(), key=lambda item: item[0]
        )
    }
    overall_metrics = _metric_dict_from_lists(aggregated_metrics)

    if return_progress_curves and return_rollout_records:
        return (
            overall_metrics,
            per_mask_type_summary,
            per_mask_slot_summary,
            progress_curve_records,
            rollout_records,
        )
    if return_progress_curves:
        return overall_metrics, per_mask_type_summary, per_mask_slot_summary, progress_curve_records
    if return_rollout_records:
        return overall_metrics, per_mask_type_summary, per_mask_slot_summary, rollout_records
    return overall_metrics, per_mask_type_summary, per_mask_slot_summary
