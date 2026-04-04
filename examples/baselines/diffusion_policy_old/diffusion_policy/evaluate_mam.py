from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm

from mani_skill.utils import common


def _slice_mask_like_train(mask_traj: torch.Tensor, step_ptr: int, obs_horizon: int, pred_horizon: int) -> torch.Tensor:
    """Match the mask slicing/padding logic used by the training dataset."""
    L = mask_traj.shape[0]
    if L == 0:
        raise ValueError("mask trajectory must be non-empty")
    start = step_ptr - (obs_horizon - 1)
    end = start + pred_horizon

    mask_seq = mask_traj[max(0, start) : end]
    if start < 0 and mask_seq.shape[0] > 0:
        mask_seq = torch.cat([mask_seq[0].repeat(-start, 1), mask_seq], dim=0)
    if mask_seq.shape[0] < pred_horizon:
        pad_len = pred_horizon - mask_seq.shape[0]
        pad_mask = torch.zeros((pad_len, mask_seq.shape[1]), device=mask_seq.device, dtype=mask_seq.dtype)
        mask_seq = torch.cat([mask_seq, pad_mask], dim=0)
    elif mask_seq.shape[0] > pred_horizon:
        mask_seq = mask_seq[:pred_horizon]
    return mask_seq


def _slice_mas_like_train(mas_traj: torch.Tensor, step_ptr: int, obs_horizon: int, pred_horizon: int) -> torch.Tensor:
    """Match the mas slicing/padding logic used by the training dataset."""
    L = mas_traj.shape[0]
    if L == 0:
        raise ValueError("mas trajectory must be non-empty")
    start = step_ptr - (obs_horizon - 1)
    end = start + pred_horizon

    mas_seq = mas_traj[max(0, start) : end]
    if start < 0 and mas_seq.shape[0] > 0:
        mas_seq = torch.cat([mas_seq[0].repeat(-start, 1), mas_seq], dim=0)
    if mas_seq.shape[0] < pred_horizon:
        pad_len = pred_horizon - mas_seq.shape[0]
        pad_mas = torch.zeros((pad_len, mas_seq.shape[1]), device=mas_seq.device, dtype=mas_seq.dtype)
        mas_seq = torch.cat([mas_seq, pad_mas], dim=0)
    elif mas_seq.shape[0] > pred_horizon:
        mas_seq = mas_seq[:pred_horizon]
    return mas_seq


def evaluate_mam(
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
    """Evaluate with explicit mas/mask conditioning during online rollout.

    eval_mam_data format:
        {
            "mas_flat": list[Tensor],   # each: (obs_mas_dim,)
            "mas": list[Tensor],        # each: (L, act_dim)
            "mask": list[Tensor],       # each: (L, act_dim)
        }
    """
    mas_flat_list = eval_mam_data["mas_flat"]
    mas_list = eval_mam_data["mas"]
    mask_list = eval_mam_data["mask"]
    assert len(mas_flat_list) > 0 and len(mas_list) > 0 and len(mask_list) > 0
    assert len(mas_flat_list) == len(mas_list) == len(mask_list)

    num_traj = len(mask_list)
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

            mask_seq = torch.stack(
                [
                    _slice_mask_like_train(
                        mask_list[int(tid.item())],
                        int(sp.item()),
                        obs_horizon,
                        pred_horizon,
                    )
                    for tid, sp in zip(traj_ids, step_ptr)
                ],
                dim=0,
            )

            mas_seq = torch.stack(
                [
                    _slice_mas_like_train(
                        mas_list[int(tid.item())],
                        int(sp.item()),
                        obs_horizon,
                        pred_horizon,
                    )
                    for tid, sp in zip(traj_ids, step_ptr)
                ],
                dim=0,
            )

            action_seq = agent.get_action(obs, mask_seq=mask_seq, a0_seq=mas_seq)
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
