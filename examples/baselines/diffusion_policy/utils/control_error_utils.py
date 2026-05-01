from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from h5py import File
import torch

try:
    from data_preprocess.utils.normalize_utils import normalize_selected_dims
except ModuleNotFoundError:
    from examples.baselines.diffusion_policy.data_preprocess.utils.normalize_utils import (
        normalize_selected_dims,
    )

try:
    from utils.load_eval_data_utils import load_eval_mas_window_data
except ModuleNotFoundError:
    from examples.baselines.diffusion_policy.utils.load_eval_data_utils import (
        load_eval_mas_window_data,
    )


def load_source_episode_ids(data_path: str, num_traj: int | None = None) -> list[int]:
    """Read source episode ids aligned with traj_0, traj_1, ... ordering."""
    with File(data_path, "r") as f:
        traj_keys = sorted(
            [k for k in f.keys() if k.startswith("traj_")],
            key=lambda x: int(x.split("_")[-1]),
        )
        if num_traj is not None:
            traj_keys = traj_keys[:num_traj]

        source_episode_ids = []
        for traj_key in traj_keys:
            traj_group = f[traj_key]
            if "source_episode_id" in traj_group:
                source_episode_ids.append(
                    int(np.asarray(traj_group["source_episode_id"][()]).reshape(-1)[0])
                )
            else:
                source_episode_ids.append(int(traj_key.split("_")[-1]))
    return source_episode_ids


def load_ce_eval_data(
    data_path: str,
    device,
    num_traj: int | None = None,
) -> dict:
    """Load eval MAS data plus stable per-traj identifiers for CE aggregation."""
    eval_data = load_eval_mas_window_data(
        data_path=data_path,
        device=device,
        num_traj=num_traj,
    )
    source_episode_ids = load_source_episode_ids(data_path=data_path, num_traj=num_traj)
    if len(source_episode_ids) != len(eval_data["mas"]):
        raise ValueError(
            "source_episode_ids length mismatch: "
            f"{len(source_episode_ids)} vs {len(eval_data['mas'])}"
        )
    eval_data["source_episode_ids"] = source_episode_ids
    eval_data["demo_local_indices"] = list(range(len(source_episode_ids)))
    return eval_data


def _to_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def normalize_rollout_actions(
    actions_denorm: np.ndarray,
    action_min: np.ndarray,
    action_max: np.ndarray,
) -> np.ndarray:
    """Map executed rollout actions back into the normalized control space."""
    actions_denorm = np.asarray(actions_denorm, dtype=np.float32)
    if actions_denorm.ndim != 2 or actions_denorm.shape[1] != 7:
        raise ValueError(
            f"expected executed actions shape (T, 7), got {actions_denorm.shape}"
        )

    action_min = np.asarray(action_min, dtype=np.float32)
    action_max = np.asarray(action_max, dtype=np.float32)
    if action_min.shape != action_max.shape or action_min.ndim != 1:
        raise ValueError(
            f"invalid action min/max shape: {action_min.shape} vs {action_max.shape}"
        )

    norm_dims = min(int(action_min.shape[0]), actions_denorm.shape[1] - 1)
    actions_norm = normalize_selected_dims(
        actions_denorm,
        mins=action_min,
        maxs=action_max,
        dims=range(norm_dims),
    )
    # Gripper dim is already in [-1, 1] for current datasets.
    actions_norm[:, -1] = actions_denorm[:, -1]
    return actions_norm.astype(np.float32)


def extract_known_control_points(
    mas_t: np.ndarray,
    mask_t: np.ndarray,
) -> list[dict]:
    """Return valid masked control points, one item per time step with known dims."""
    mas_t = np.asarray(mas_t, dtype=np.float32)
    mask_t = np.asarray(mask_t, dtype=np.float32)
    if mas_t.ndim != 2 or mas_t.shape[1] < 7:
        raise ValueError(f"expected mas shape (T, >=7), got {mas_t.shape}")
    if mask_t.ndim != 2 or mask_t.shape[0] != mas_t.shape[0] or mask_t.shape[1] < 7:
        raise ValueError(
            f"expected mask shape ({mas_t.shape[0]}, >=7), got {mask_t.shape}"
        )

    points = []
    for step_idx in range(mas_t.shape[0]):
        dims_idx = np.where(mask_t[step_idx, :7] > 0.5)[0].astype(np.int64)
        if dims_idx.size == 0:
            continue
        points.append(
            dict(
                step_idx=int(step_idx),
                dims_idx=dims_idx,
                target_vec=mas_t[step_idx, dims_idx].astype(np.float32),
            )
        )
    return points


def compute_control_error_for_traj(
    rollout_actions_norm: np.ndarray,
    mas_t: np.ndarray,
    mask_t: np.ndarray,
) -> dict:
    """Compute CE for a single rollout trajectory."""
    rollout_actions_norm = np.asarray(rollout_actions_norm, dtype=np.float32)
    if rollout_actions_norm.ndim != 2 or rollout_actions_norm.shape[1] != 7:
        raise ValueError(
            f"expected rollout_actions_norm shape (T, 7), got {rollout_actions_norm.shape}"
        )

    known_points = extract_known_control_points(mas_t=mas_t, mask_t=mask_t)
    if len(known_points) == 0:
        return dict(
            ce_traj=np.nan,
            num_known_points=0,
            known_step_indices=[],
            best_match_t_list=[],
            point_errors=[],
            point_errors_normalized=[],
            point_errors_raw=[],
            point_error_raw_mean=np.nan,
            point_error_normalized_mean=np.nan,
            known_dims_list=[],
        )
    if rollout_actions_norm.shape[0] == 0:
        raise ValueError("rollout trajectory is empty but known control points exist")

    point_errors = []
    point_errors_raw = []
    best_match_t_list = []
    known_step_indices = []
    known_dims_list = []
    for point in known_points:
        dims_idx = point["dims_idx"]
        target_vec = point["target_vec"]
        num_known_dims = int(dims_idx.size)
        if num_known_dims <= 0:
            raise ValueError("each known control point must contain at least one known dim")
        distances = np.linalg.norm(
            rollout_actions_norm[:, dims_idx] - target_vec[None, :],
            axis=1,
        )
        best_match_t = int(np.argmin(distances))
        point_error_raw = float(distances[best_match_t])
        point_error_normalized = point_error_raw / float(num_known_dims)
        point_errors_raw.append(point_error_raw)
        point_errors.append(point_error_normalized)
        best_match_t_list.append(best_match_t)
        known_step_indices.append(int(point["step_idx"]))
        known_dims_list.append(dims_idx.tolist())

    return dict(
        ce_traj=float(np.mean(point_errors)),
        num_known_points=len(point_errors),
        known_step_indices=known_step_indices,
        best_match_t_list=best_match_t_list,
        point_errors=point_errors,
        point_errors_normalized=point_errors,
        point_errors_raw=point_errors_raw,
        point_error_raw_mean=float(np.mean(point_errors_raw)),
        point_error_normalized_mean=float(np.mean(point_errors)),
        known_dims_list=known_dims_list,
    )


def aggregate_control_error(per_traj_results: list[dict]) -> dict:
    """Aggregate CE over all trajectories, success_once, and failed groups."""

    def _is_valid_ce(value) -> bool:
        return value is not None and np.isfinite(value)

    valid_results = [r for r in per_traj_results if _is_valid_ce(r.get("ce_traj"))]
    success_results = [r for r in valid_results if bool(r.get("success_once", False))]
    failed_results = [r for r in valid_results if not bool(r.get("success_once", False))]

    def _mean_or_nan(results: list[dict]) -> float:
        if len(results) == 0:
            return float("nan")
        return float(np.mean([float(r["ce_traj"]) for r in results]))

    return dict(
        ce_all=_mean_or_nan(valid_results),
        ce_success=_mean_or_nan(success_results),
        ce_failed=_mean_or_nan(failed_results),
        num_total_valid_ce=len(valid_results),
        num_success_valid_ce=len(success_results),
        num_failed_valid_ce=len(failed_results),
        num_success=sum(bool(r.get("success_once", False)) for r in per_traj_results),
        num_failed=sum(not bool(r.get("success_once", False)) for r in per_traj_results),
    )


def compute_control_error_results_from_rollouts(
    eval_data: dict,
    rollout_records: list[dict],
    action_min: np.ndarray,
    action_max: np.ndarray,
    save_per_traj: bool = False,
) -> list[dict]:
    """Convert rollout records into per-traj CE results in a stable traj-id order."""
    demo_local_indices = eval_data.get("demo_local_indices")
    if demo_local_indices is None:
        demo_local_indices = list(range(len(eval_data["mas"])))
    local_pos_by_demo_idx = {
        int(demo_idx): local_pos for local_pos, demo_idx in enumerate(demo_local_indices)
    }

    per_traj_results = []
    for record in sorted(rollout_records, key=lambda item: int(item["demo_local_idx"])):
        demo_local_idx = int(record["demo_local_idx"])
        if demo_local_idx not in local_pos_by_demo_idx:
            raise KeyError(
                f"demo_local_idx {demo_local_idx} not found in current eval_data mapping"
            )
        local_pos = local_pos_by_demo_idx[demo_local_idx]
        executed_actions_denorm = np.asarray(
            record["executed_actions_denorm"],
            dtype=np.float32,
        )
        executed_actions_norm = normalize_rollout_actions(
            executed_actions_denorm,
            action_min=action_min,
            action_max=action_max,
        )

        mas_t = _to_numpy(eval_data["mas"][local_pos]).astype(np.float32)
        mask_t = _to_numpy(eval_data["mas_mask"][local_pos]).astype(np.float32)
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


def _to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, tuple):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, bytes):
        return obj.decode("utf-8")
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bytes_):
        return obj.decode("utf-8")
    if isinstance(obj, np.floating):
        value = float(obj)
        return None if not np.isfinite(value) else value
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, float):
        return None if not np.isfinite(obj) else obj
    return obj


def save_control_error_json(results: dict, output_json_path: str) -> None:
    output_path = Path(output_json_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(_to_jsonable(results), f, ensure_ascii=False, indent=2)
