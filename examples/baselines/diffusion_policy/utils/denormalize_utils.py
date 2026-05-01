from typing import List

import torch

try:
    from utils.split_eval_utils import load_action_denorm_stats
except ModuleNotFoundError:
    from examples.baselines.diffusion_policy.utils.split_eval_utils import (
        load_action_denorm_stats,
    )


def compute_state_min_max(state_traj_list: List[torch.Tensor]):
    assert len(state_traj_list) > 0, "Empty state trajectory list"
    state_min = state_traj_list[0].amin(dim=0).to(dtype=torch.float32)
    state_max = state_traj_list[0].amax(dim=0).to(dtype=torch.float32)
    for state_traj in state_traj_list[1:]:
        state_traj = state_traj.to(dtype=torch.float32)
        state_min = torch.minimum(state_min, state_traj.amin(dim=0))
        state_max = torch.maximum(state_max, state_traj.amax(dim=0))
    return state_min, state_max
