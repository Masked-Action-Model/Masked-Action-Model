from typing import List

import numpy as np
import torch

try:
    from data_preprocess.utils.normalize_utils import load_action_stats_from_path
except ModuleNotFoundError:
    from examples.baselines.diffusion_policy.data_preprocess.utils.normalize_utils import (
        load_action_stats_from_path,
    )


def load_action_denorm_stats(action_norm_path: str):
    if action_norm_path is None or len(action_norm_path.strip()) == 0:
        raise ValueError(
            "action_norm_path is required. Please pass --action-norm-path to provide min/max for denormalization."
        )
    mins, maxs = load_action_stats_from_path(action_norm_path)
    print(
        f"[denorm] loaded action norm stats from {action_norm_path}, dims={mins.shape[0]}"
    )
    return mins, maxs


def compute_state_min_max(state_traj_list: List[torch.Tensor]):
    assert len(state_traj_list) > 0, "Empty state trajectory list"
    state_min = state_traj_list[0].amin(dim=0).to(dtype=torch.float32)
    state_max = state_traj_list[0].amax(dim=0).to(dtype=torch.float32)
    for state_traj in state_traj_list[1:]:
        state_traj = state_traj.to(dtype=torch.float32)
        state_min = torch.minimum(state_min, state_traj.amin(dim=0))
        state_max = torch.maximum(state_max, state_traj.amax(dim=0))
    return state_min, state_max
