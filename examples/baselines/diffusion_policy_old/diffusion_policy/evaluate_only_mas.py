from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm

from mani_skill.utils import common


def evaluate_only_mas(
    n: int,
    agent,
    eval_envs,
    device,
    sim_backend: str,
    eval_mam_data: dict,
    obs_horizon: int,
    pred_horizon: int,
    progress_bar: bool = True,
):
    """Evaluate with explicit mas conditioning during online rollout.

    eval_mam_data format:
        {
            "mas_flat": list[Tensor],   # each: (obs_mas_dim,)
        }
    """
    mas_flat_list = eval_mam_data["mas_flat"]
    assert len(mas_flat_list) > 0

    num_traj = len(mas_flat_list)
    traj_cursor = 0

    agent.eval()
    if progress_bar:
        pbar = tqdm(total=n)

    with torch.no_grad():
        eval_metrics = defaultdict(list)
        obs, info = eval_envs.reset()
        B = eval_envs.num_envs

        traj_ids = torch.tensor(
            [(traj_cursor + i) % num_traj for i in range(B)],
            device=device,
            dtype=torch.long,
        )
        traj_cursor = (traj_cursor + B) % num_traj
        step_ptr = torch.zeros((B,), device=device, dtype=torch.long)

        eps_count = 0
        while eps_count < n:
            obs = common.to_tensor(obs, device)

            obs_mas = torch.stack([mas_flat_list[int(i.item())] for i in traj_ids], dim=0)
            obs_mas = obs_mas.unsqueeze(1).repeat(1, obs_horizon, 1)
            obs["mas"] = obs_mas

            action_seq = agent.get_action(obs)
            # train_mam.Agent.get_action() already denormalizes rollout actions when configured.
            if sim_backend == "physx_cpu":
                action_seq = action_seq.cpu().numpy()

            executed_steps = 0
            for i in range(action_seq.shape[1]):
                obs, rew, terminated, truncated, info = eval_envs.step(action_seq[:, i])
                executed_steps = i + 1
                if truncated.any():
                    break

            step_ptr = step_ptr + executed_steps

            if truncated.any():
                assert truncated.all() == truncated.any(), (
                    "all episodes should truncate at the same time for fair evaluation with other algorithms"
                )
                if isinstance(info["final_info"], dict):
                    for k, v in info["final_info"]["episode"].items():
                        eval_metrics[k].append(v.float().cpu().numpy())
                else:
                    for final_info in info["final_info"]:
                        for k, v in final_info["episode"].items():
                            eval_metrics[k].append(v)

                eps_count += B
                traj_ids = torch.tensor(
                    [(traj_cursor + i) % num_traj for i in range(B)],
                    device=device,
                    dtype=torch.long,
                )
                traj_cursor = (traj_cursor + B) % num_traj
                step_ptr.zero_()

                if progress_bar:
                    pbar.update(B)

    agent.train()
    for k in eval_metrics.keys():
        eval_metrics[k] = np.stack(eval_metrics[k])
    return eval_metrics
