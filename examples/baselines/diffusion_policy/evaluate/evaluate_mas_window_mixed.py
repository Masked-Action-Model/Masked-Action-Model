from __future__ import annotations

from collections import defaultdict

import numpy as np

try:
    from evaluate.evaluate_mas_window import evaluate_mas_window
    from utils.load_eval_data_utils import subset_eval_data
except ModuleNotFoundError:
    from examples.baselines.diffusion_policy.evaluate.evaluate_mas_window import (
        evaluate_mas_window,
    )
    from examples.baselines.diffusion_policy.utils.load_eval_data_utils import (
        subset_eval_data,
    )


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


def _build_padded_eval_batches(total_items: int, batch_size: int):
    if total_items <= 0:
        return []
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    batches = []
    for start in range(0, total_items, batch_size):
        valid_indices = list(range(start, min(start + batch_size, total_items)))
        padded_indices = valid_indices[:]
        while len(padded_indices) < batch_size:
            padded_indices.append(valid_indices[-1])
        batches.append((padded_indices, len(valid_indices)))
    return batches


def group_eval_data_by_mask_type(eval_mas_window_data: dict) -> dict[str, list[int]]:
    mask_types = eval_mas_window_data.get("mask_types")
    if mask_types is None:
        raise ValueError("mixed eval requires eval_mas_window_data['mask_types']")
    if len(mask_types) != len(eval_mas_window_data["mas"]):
        raise ValueError(
            "mask_types length must match eval trajectory count: "
            f"{len(mask_types)} vs {len(eval_mas_window_data['mas'])}"
        )
    grouped = defaultdict(list)
    for idx, mask_type in enumerate(mask_types):
        grouped[str(mask_type)].append(idx)
    return dict(sorted(grouped.items(), key=lambda item: item[0]))


def evaluate_one_mask_type_group(
    mask_type: str,
    group_indices: list[int],
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
    progress_bar: bool = False,
    reset_seeds: list[int] | None = None,
    return_progress_curves: bool = False,
    return_rollout_records: bool = False,
):
    if len(group_indices) == 0:
        raise ValueError(f"group {mask_type!r} is empty")

    grouped_eval_data = subset_eval_data(eval_mas_window_data, group_indices)
    grouped_reset_seeds = None
    if reset_seeds is not None:
        grouped_reset_seeds = [reset_seeds[i] for i in group_indices]

    aggregated_metrics = defaultdict(list)
    aggregated_progress_records = [] if return_progress_curves else None
    aggregated_rollout_records = [] if return_rollout_records else None
    batch_size = int(eval_envs.num_envs)

    for padded_indices, valid_batch_size in _build_padded_eval_batches(
        len(group_indices), batch_size
    ):
        batch_eval_data = subset_eval_data(grouped_eval_data, padded_indices)
        batch_reset_seeds = None
        if grouped_reset_seeds is not None:
            batch_reset_seeds = [grouped_reset_seeds[i] for i in padded_indices]

        one_result = evaluate_mas_window(
            batch_size,
            agent,
            eval_envs,
            device,
            sim_backend,
            eval_mas_window_data=batch_eval_data,
            obs_horizon=obs_horizon,
            long_window_horizon=long_window_horizon,
            short_window_horizon=short_window_horizon,
            stpm_encoder=stpm_encoder,
            stpm_n_obs_steps=stpm_n_obs_steps,
            stpm_frame_gap=stpm_frame_gap,
            progress_bar=progress_bar,
            reset_seeds=batch_reset_seeds,
            return_progress_curves=return_progress_curves,
            return_rollout_records=return_rollout_records,
        )

        if return_progress_curves and return_rollout_records:
            batch_metrics, batch_progress_records, batch_rollout_records = one_result
        elif return_progress_curves:
            batch_metrics, batch_progress_records = one_result
            batch_rollout_records = None
        elif return_rollout_records:
            batch_metrics, batch_rollout_records = one_result
            batch_progress_records = None
        else:
            batch_metrics = one_result
            batch_progress_records = None
            batch_rollout_records = None

        batch_metrics = _slice_metric_dict(batch_metrics, valid_batch_size)
        for key, value in batch_metrics.items():
            aggregated_metrics[key].extend(np.asarray(value).reshape(-1).tolist())

        if return_progress_curves:
            for record in batch_progress_records[:valid_batch_size]:
                record = dict(record)
                record["mask_type"] = mask_type
                aggregated_progress_records.append(record)

        if return_rollout_records:
            for record in batch_rollout_records[:valid_batch_size]:
                record = dict(record)
                record["mask_type"] = mask_type
                aggregated_rollout_records.append(record)

    grouped_metrics = {
        key: np.asarray(values) for key, values in aggregated_metrics.items()
    }
    if return_progress_curves and return_rollout_records:
        return grouped_metrics, aggregated_progress_records, aggregated_rollout_records
    if return_progress_curves:
        return grouped_metrics, aggregated_progress_records
    if return_rollout_records:
        return grouped_metrics, aggregated_rollout_records
    return grouped_metrics


def evaluate_mas_window_mixed(
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
    if reset_seed is not None:
        raise ValueError("mixed eval expects reset_seeds instead of a single reset_seed")
    mask_groups = group_eval_data_by_mask_type(eval_mas_window_data)
    total_items = len(eval_mas_window_data["mas"])
    if n != total_items:
        raise ValueError(
            f"mixed eval expects n to equal len(eval_mas_window_data['mas'])={total_items}, got {n}"
        )
    if reset_seeds is not None and len(reset_seeds) != total_items:
        raise ValueError(
            "reset_seeds length must match eval trajectory count: "
            f"{len(reset_seeds)} vs {total_items}"
        )

    aggregated_metrics = defaultdict(list)
    per_mask_summary = {}
    progress_curve_records = [] if return_progress_curves else None
    rollout_records = [] if return_rollout_records else None

    for mask_type, group_indices in mask_groups.items():
        group_reset_seeds = None if reset_seeds is None else reset_seeds
        one_result = evaluate_one_mask_type_group(
            mask_type=mask_type,
            group_indices=group_indices,
            agent=agent,
            eval_envs=eval_envs,
            device=device,
            sim_backend=sim_backend,
            eval_mas_window_data=eval_mas_window_data,
            obs_horizon=obs_horizon,
            long_window_horizon=long_window_horizon,
            short_window_horizon=short_window_horizon,
            stpm_encoder=stpm_encoder,
            stpm_n_obs_steps=stpm_n_obs_steps,
            stpm_frame_gap=stpm_frame_gap,
            progress_bar=progress_bar,
            reset_seeds=group_reset_seeds,
            return_progress_curves=return_progress_curves,
            return_rollout_records=return_rollout_records,
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

        for key, value in group_metrics.items():
            aggregated_metrics[key].extend(np.asarray(value).reshape(-1).tolist())
        per_mask_summary[mask_type] = _summarize_metric_dict(group_metrics)

        if return_progress_curves:
            progress_curve_records.extend(group_progress_records)
        if return_rollout_records:
            rollout_records.extend(group_rollout_records)

    overall_metrics = {
        key: np.asarray(values) for key, values in aggregated_metrics.items()
    }
    if return_progress_curves and return_rollout_records:
        return overall_metrics, per_mask_summary, progress_curve_records, rollout_records
    if return_progress_curves:
        return overall_metrics, per_mask_summary, progress_curve_records
    if return_rollout_records:
        return overall_metrics, per_mask_summary, rollout_records
    return overall_metrics, per_mask_summary
