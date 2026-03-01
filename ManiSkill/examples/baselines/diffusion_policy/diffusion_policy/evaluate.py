from __future__ import annotations

from collections import defaultdict
from functools import partial
from typing import Dict, Optional

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from mani_skill.utils import common
from diffusion_policy.utils import build_state_obs_extractor, convert_obs, load_demo_dataset


def reorder_keys(d, ref_dict):
    out = dict()
    for k, v in ref_dict.items():
        if isinstance(v, dict) or isinstance(v, spaces.Dict):
            out[k] = reorder_keys(d[k], ref_dict[k])
        else:
            out[k] = d[k]
    return out


class EvalMAMDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        obs_process_fn,
        obs_space,
        device,
        obs_horizon: int,
        pred_horizon: int,
        control_mode: str,
        num_traj: Optional[int] = None,
    ) -> None:
        load_keys = ["observations", "actions", "mas", "mask"]
        trajectories = load_demo_dataset(data_path, keys=load_keys, num_traj=num_traj, concat=False)
        missing_keys = [k for k in load_keys if k not in trajectories]
        assert not missing_keys, f"Missing keys in loaded trajectories: {missing_keys}"

        obs_traj_dict_list = []
        for obs_traj_dict in trajectories["observations"]:
            _obs_traj_dict = reorder_keys(obs_traj_dict, obs_space)
            _obs_traj_dict = obs_process_fn(_obs_traj_dict)
            _obs_traj_dict["depth"] = torch.Tensor(_obs_traj_dict["depth"].astype(np.float32)).to(
                device=device, dtype=torch.float16
            )
            _obs_traj_dict["rgb"] = torch.from_numpy(_obs_traj_dict["rgb"]).to(device)
            _obs_traj_dict["state"] = torch.from_numpy(_obs_traj_dict["state"]).to(device)
            obs_traj_dict_list.append(_obs_traj_dict)
        trajectories["observations"] = obs_traj_dict_list
        self.obs_keys = list(_obs_traj_dict.keys())

        for i in range(len(trajectories["actions"])):
            trajectories["actions"][i] = torch.Tensor(trajectories["actions"][i]).to(device=device)
        for i in range(len(trajectories["mas"])):
            trajectories["mas"][i] = torch.Tensor(trajectories["mas"][i]).to(device=device)
        for i in range(len(trajectories["mask"])):
            trajectories["mask"][i] = torch.Tensor(trajectories["mask"][i]).to(device=device)
        # Ensure mask is aligned to action length (some datasets store mask with L+1).
        for i in range(len(trajectories["actions"])):
            L = trajectories["actions"][i].shape[0]
            t = trajectories["mask"][i]
            if t.shape[0] > L:
                t = t[:L]
            elif t.shape[0] < L:
                pad = torch.zeros((L - t.shape[0], t.shape[1]), device=t.device, dtype=t.dtype)
                t = torch.cat([t, pad], dim=0)
            trajectories["mask"][i] = t

        # Only pd_ee_pose is supported here; use fixed bounds for action normalization.
        # Box([-2,-2,-2,-2π,-2π,-2π,-1], [2,2,2,2π,2π,2π,1])
        self.action_low = torch.tensor(
            [-2.0, -2.0, -2.0, -2.0 * np.pi, -2.0 * np.pi, -2.0 * np.pi, -1.0],
            device=device,
            dtype=torch.float32,
        )
        self.action_high = torch.tensor(
            [2.0, 2.0, 2.0, 2.0 * np.pi, 2.0 * np.pi, 2.0 * np.pi, 1.0],
            device=device,
            dtype=torch.float32,
        )

        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.control_mode = control_mode
        self.slices = []
        num_traj = len(trajectories["actions"])
        for traj_idx in range(num_traj):
            L = trajectories["actions"][traj_idx].shape[0]
            pad_before = obs_horizon - 1
            pad_after = pred_horizon - obs_horizon
            self.slices += [
                (traj_idx, start, start + pred_horizon)
                for start in range(-pad_before, L - pred_horizon + pad_after)
            ]

        self.trajectories = trajectories

    def __len__(self) -> int:
        return len(self.slices)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        traj_idx, start, end = self.slices[index]
        L, _ = self.trajectories["actions"][traj_idx].shape

        obs_traj = self.trajectories["observations"][traj_idx]
        obs_seq = {}
        for k, v in obs_traj.items():
            obs_seq[k] = v[max(0, start) : start + self.obs_horizon]
            if start < 0:
                pad_len = -start
                if k == "state":
                    # keep delta ~ 0 by linear extrapolation
                    if obs_seq[k].shape[0] >= 2:
                        d = obs_seq[k][1] - obs_seq[k][0]
                    else:
                        d = torch.zeros_like(obs_seq[k][0])
                    pad_obs = [obs_seq[k][0] - d * n for n in range(pad_len, 0, -1)]
                    obs_seq[k] = torch.cat((torch.stack(pad_obs, dim=0), obs_seq[k]), dim=0)
                else:
                    # repeat the first frame for visual obs
                    pad_obs_seq = torch.stack([obs_seq[k][0]] * pad_len, dim=0)
                    obs_seq[k] = torch.cat((pad_obs_seq, obs_seq[k]), dim=0)

        mas_traj = self.trajectories["mas"][traj_idx]  # (T, 7)
        mas_traj = torch.where(mas_traj == -1, torch.zeros_like(mas_traj), mas_traj)
        mas_flat = mas_traj.reshape(-1)
        obs_seq["mas"] = mas_flat.unsqueeze(0).repeat(self.obs_horizon, 1)

        action_seq = self.trajectories["actions"][traj_idx][max(0, start) : end]
        if start < 0:
            # keep delta ~ 0 by linear extrapolation: a[-1] = a[0] - (a[1] - a[0])
            if action_seq.shape[0] >= 2:
                d = action_seq[1] - action_seq[0]
            else:
                d = torch.zeros_like(action_seq[0])
            pad_len = -start
            pad_actions = [action_seq[0] - d * k for k in range(pad_len, 0, -1)]
            action_seq = torch.cat([torch.stack(pad_actions, dim=0), action_seq], dim=0)
        if end > L:
            pad_action = action_seq[-1]
            action_seq = torch.cat([action_seq, pad_action.repeat(end - L, 1)], dim=0)
        # Normalize actions to [-1, 1] using fixed pd_ee_pose bounds
        action_seq = (action_seq - 0.5 * (self.action_high + self.action_low)) / (
            0.5 * (self.action_high - self.action_low)
        )

        a0_seq = mas_traj[max(0, start) : end]
        if start < 0:
            a0_seq = torch.cat([a0_seq[0].repeat(-start, 1), a0_seq], dim=0)
        if end > L:
            pad_len = end - L
            pad_a0 = torch.zeros(
                (pad_len, a0_seq.shape[1]), device=a0_seq.device, dtype=a0_seq.dtype
            )
            a0_seq = torch.cat([a0_seq, pad_a0], dim=0)

        mask_traj = self.trajectories["mask"][traj_idx]
        mask_seq = mask_traj[max(0, start) : end]
        if start < 0:
            mask_seq = torch.cat([mask_seq[0].repeat(-start, 1), mask_seq], dim=0)
        if end > L:
            pad_len = end - L
            pad_mask = torch.zeros(
                (pad_len, mask_seq.shape[1]), device=mask_seq.device, dtype=mask_seq.dtype
            )
            mask_seq = torch.cat([mask_seq, pad_mask], dim=0)
        if mask_seq.shape[0] > self.pred_horizon:
            mask_seq = mask_seq[: self.pred_horizon]
        elif mask_seq.shape[0] < self.pred_horizon:
            pad_len = self.pred_horizon - mask_seq.shape[0]
            pad_mask = torch.zeros(
                (pad_len, mask_seq.shape[1]), device=mask_seq.device, dtype=mask_seq.dtype
            )
            mask_seq = torch.cat([mask_seq, pad_mask], dim=0)
        if a0_seq.shape[0] > self.pred_horizon:
            a0_seq = a0_seq[: self.pred_horizon]
        elif a0_seq.shape[0] < self.pred_horizon:
            pad_len = self.pred_horizon - a0_seq.shape[0]
            pad_a0 = torch.zeros(
                (pad_len, a0_seq.shape[1]), device=a0_seq.device, dtype=a0_seq.dtype
            )
            a0_seq = torch.cat([a0_seq, pad_a0], dim=0)
        assert action_seq.shape[0] == self.pred_horizon
        assert mask_seq.shape[0] == self.pred_horizon
        assert a0_seq.shape[0] == self.pred_horizon

        return {
            "observations": obs_seq,
            "actions": action_seq,
            "mask": mask_seq,
            "a0_actions": a0_seq,
        }


def _evaluate_from_h5(
    eval_h5_path: str,
    env_id: str,
    env_kwargs: dict,
    agent,
    device,
    obs_horizon: int,
    pred_horizon: int,
    batch_size: int,
    num_workers: int,
    progress_bar: bool,
    num_eval_data: Optional[int] = None,
) -> Dict[str, float]:
    env = gym.make(env_id, **env_kwargs)
    obs_space = env.observation_space
    env.close()

    obs_process_fn = partial(
        convert_obs,
        concat_fn=partial(np.concatenate, axis=-1),
        transpose_fn=partial(np.transpose, axes=(0, 3, 1, 2)),
        state_obs_extractor=build_state_obs_extractor(env_id),
        depth=True,
    )

    dataset = EvalMAMDataset(
        data_path=eval_h5_path,
        obs_process_fn=obs_process_fn,
        obs_space=obs_space,
        device=device,
        obs_horizon=obs_horizon,
        pred_horizon=pred_horizon,
        control_mode="pd_ee_pose",
        num_traj=num_eval_data,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
    )

    agent.eval()
    total = 0
    mse_all_sum = 0.0
    mse_unknown_sum = 0.0

    if progress_bar:
        pbar = tqdm(total=len(dataloader))

    with torch.no_grad():
        for batch in dataloader:
            obs_seq = batch["observations"]
            action_seq = batch["actions"]
            mask_seq = batch["mask"]
            a0_seq = batch["a0_actions"]

            B = action_seq.shape[0]
            obs_cond = agent.encode_obs(obs_seq, eval_mode=True)

            noisy_action_seq = torch.randn(
                (B, pred_horizon, action_seq.shape[-1]), device=device
            )
            mask_seq = mask_seq.to(device=device, dtype=noisy_action_seq.dtype)
            a0_seq = a0_seq.to(device=device, dtype=noisy_action_seq.dtype)
            noisy_action_seq = mask_seq * a0_seq + (1.0 - mask_seq) * noisy_action_seq

            for k in agent.noise_scheduler.timesteps:
                a_in = mask_seq * a0_seq + (1.0 - mask_seq) * noisy_action_seq
                timesteps = torch.full((B,), k, dtype=torch.long, device=device)
                eps_hat = agent.noise_pred_net(a_in, timesteps, obs_cond, mask=mask_seq)
                noisy_action_seq = agent.noise_scheduler.step(
                    model_output=eps_hat,
                    timestep=k,
                    sample=noisy_action_seq,
                ).prev_sample
                noisy_action_seq = mask_seq * a0_seq + (1.0 - mask_seq) * noisy_action_seq

            pred_actions = noisy_action_seq
            diff = pred_actions - action_seq.to(device=device, dtype=pred_actions.dtype)
            mse_all = (diff ** 2).mean()
            unknown = 1.0 - mask_seq
            denom = unknown.sum().clamp_min(1.0)
            mse_unknown = ((diff ** 2) * unknown).sum() / denom

            mse_all_sum += mse_all.item() * B
            mse_unknown_sum += mse_unknown.item() * B
            total += B

            if progress_bar:
                pbar.update(1)

    agent.train()
    if total == 0:
        return {"mse_all": 0.0, "mse_unknown": 0.0}
    return {
        "mse_all": mse_all_sum / total,
        "mse_unknown": mse_unknown_sum / total,
    }


def evaluate(
    n: int,
    agent,
    eval_envs,
    device,
    sim_backend: str,
    progress_bar: bool = True,
    eval_h5_path: Optional[str] = None,
    env_id: Optional[str] = None,
    env_kwargs: Optional[dict] = None,
    batch_size: int = 256,
    num_workers: int = 0,
    obs_horizon: Optional[int] = None,
    pred_horizon: Optional[int] = None,
    num_eval_data: Optional[int] = None,
):
    if eval_h5_path is not None:
        assert env_id is not None and env_kwargs is not None, "env_id/env_kwargs required for H5 evaluation"
        assert obs_horizon is not None and pred_horizon is not None, "obs_horizon/pred_horizon required"
        return _evaluate_from_h5(
            eval_h5_path=eval_h5_path,
            env_id=env_id,
            env_kwargs=env_kwargs,
            agent=agent,
            device=device,
            obs_horizon=obs_horizon,
            pred_horizon=pred_horizon,
            batch_size=batch_size,
            num_workers=num_workers,
            progress_bar=progress_bar,
            num_eval_data=num_eval_data,
        )

    agent.eval()
    if progress_bar:
        pbar = tqdm(total=n)
    # Action rescale helper: map [-1,1] -> env action space
    action_low = None
    action_high = None
    if hasattr(eval_envs, "single_action_space"):
        action_low = eval_envs.single_action_space.low
        action_high = eval_envs.single_action_space.high
    with torch.no_grad():
        eval_metrics = defaultdict(list)
        obs, info = eval_envs.reset()
        eps_count = 0
        while eps_count < n:
            obs = common.to_tensor(obs, device)
            action_seq = agent.get_action(obs)
            if action_low is not None and action_high is not None:
                if sim_backend == "physx_cpu":
                    # numpy path
                    a = action_seq.cpu().numpy()
                    action_seq = 0.5 * (action_high + action_low) + 0.5 * (
                        action_high - action_low
                    ) * a
                else:
                    # torch path
                    low_t = common.to_tensor(action_low, device)
                    high_t = common.to_tensor(action_high, device)
                    action_seq = 0.5 * (high_t + low_t) + 0.5 * (high_t - low_t) * action_seq
            if sim_backend == "physx_cpu":
                action_seq = action_seq.cpu().numpy()
            for i in range(action_seq.shape[1]):
                obs, rew, terminated, truncated, info = eval_envs.step(action_seq[:, i])
                if truncated.any():
                    break

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
                eps_count += eval_envs.num_envs
                if progress_bar:
                    pbar.update(eval_envs.num_envs)
    agent.train()
    for k in eval_metrics.keys():
        eval_metrics[k] = np.stack(eval_metrics[k])
    return eval_metrics
