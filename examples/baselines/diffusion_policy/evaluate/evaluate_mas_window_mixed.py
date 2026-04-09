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


def _group_indices_by_labels(
    labels: list[str] | None,
    total_items: int,
    label_name: str,
) -> dict[str, list[int]]:
    if labels is None:
        raise ValueError(f"mixed eval requires eval_mas_window_data[{label_name!r}]")
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
    progress_bar: bool = False,
    reset_seeds: list[int] | None = None,
    return_progress_curves: bool = False,
    return_rollout_records: bool = False,
):
    if len(group_indices) == 0:
        raise ValueError(f"group {group_label!r} is empty")

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
        _append_metric_dict(aggregated_metrics, batch_metrics)

        if return_progress_curves:
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
            stpm_encoder=stpm_encoder,
            stpm_n_obs_steps=stpm_n_obs_steps,
            stpm_frame_gap=stpm_frame_gap,
            progress_bar=progress_bar,
            reset_seeds=reset_seeds,
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

        _append_metric_dict(aggregated_metrics, group_metrics)
        _append_metric_dict(per_mask_type_metric_lists[mask_type], group_metrics)
        per_mask_slot_summary[str(mask_type_slot)] = _summarize_metric_dict(group_metrics)

        if return_progress_curves:
            progress_curve_records.extend(group_progress_records)
        if return_rollout_records:
            rollout_records.extend(group_rollout_records)

    per_mask_type_summary = {
        str(mask_type): _summarize_metric_dict(_metric_dict_from_lists(metric_lists))
        for mask_type, metric_lists in sorted(per_mask_type_metric_lists.items(), key=lambda item: item[0])
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
