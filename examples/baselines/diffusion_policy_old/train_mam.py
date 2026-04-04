ALGO_NAME = "BC_Diffusion_rgbd_DiT"

import os
import random
import time
import json
from collections import defaultdict
from dataclasses import dataclass, field
from functools import partial
from typing import List, Optional

import gymnasium as gym
from gymnasium.vector.vector_env import VectorEnv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import tyro
from diffusers.optimization import get_scheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from h5py import File
from gymnasium import spaces
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import BatchSampler, RandomSampler
from torch.utils.tensorboard import SummaryWriter

from diffusion_policy.modeling_ditdp import DiTNoiseNet
from diffusion_policy.evaluate_mam import evaluate_mam
from diffusion_policy.make_env import make_eval_envs
from diffusion_policy.plain_conv import PlainConv
from diffusion_policy.utils import (IterationBasedBatchSampler,
                                    build_state_obs_extractor, convert_obs,
                                    worker_init_fn)
from diffusion_policy.utils import load_demo_dataset


@dataclass
class Args:
    exp_name: Optional[str] = None
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "ManiSkill"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    env_id: str = "PegInsertionSide-v1"
    """the id of the environment"""
    demo_path: str = (
        "demos/PegInsertionSide-v1/trajectory.state.pd_ee_delta_pose.physx_cpu.h5"
    )
    """the path of demo dataset, it is expected to be a ManiSkill dataset h5py format file"""
    test_demo_path: Optional[str] = None
    """the path of test demo dataset used for online eval mas/mask conditioning (required)"""
    num_demos: Optional[int] = None
    """number of trajectories to load from the demo dataset"""
    total_iters: int = 1_000_000
    """total timesteps of the experiment"""
    batch_size: int = 256
    """the batch size of sample from the replay memory"""

    # Diffusion Policy specific arguments
    lr: float = 1e-4
    """the learning rate of the diffusion policy"""
    obs_horizon: int = 2  # Seems not very important in ManiSkill, 1, 2, 4 work well
    act_horizon: int = 8  # Seems not very important in ManiSkill, 4, 8, 15 work well
    pred_horizon: int = (
        16  # 16->8 leads to worse performance, maybe it is like generate a half image; 16->32, improvement is very marginal
    )
    diffusion_step_embed_dim: int = 64  # not very important

    # Environment/experiment specific arguments
    max_episode_steps: Optional[int] = None
    """Change the environments' max_episode_steps to this value. Sometimes necessary if the demonstrations being imitated are too short. Typically the default
    max episode steps of environments in ManiSkill are tuned lower so reinforcement learning agents can learn faster."""
    log_freq: int = 1000
    """the frequency of logging the training metrics"""
    eval_freq: int = 5000
    """the frequency of evaluating the agent on the evaluation environments"""
    save_freq: Optional[int] = None
    """the frequency of saving the model checkpoints. By default this is None and will only save checkpoints based on the best evaluation metrics."""
    num_eval_episodes: int = 100
    """the number of episodes to evaluate the agent on"""
    num_eval_envs: int = 10
    """the number of parallel environments to evaluate the agent on"""
    action_norm_path: Optional[str] = None
    """Path to action normalization json (must contain keys min/max). Required for rollout denormalization."""
    sim_backend: str = "physx_cpu"
    """the simulation backend to use for evaluation environments. can be "cpu" or "gpu"""
    num_dataload_workers: int = 0
    """the number of workers to use for loading the training data in the torch dataloader"""
    control_mode: str = "pd_ee_pose"
    """the control mode to use for the evaluation environments. Must match the control mode of the demonstration dataset."""

    # additional tags/configs for logging purposes to wandb and shared comparisons with other algorithms
    demo_type: Optional[str] = None


def reorder_keys(d, ref_dict):
    out = dict()
    for k, v in ref_dict.items():
        if isinstance(v, dict) or isinstance(v, spaces.Dict):
            out[k] = reorder_keys(d[k], ref_dict[k])
        else:
            out[k] = d[k]
    return out


def load_action_denorm_stats(action_norm_path: str):
    if action_norm_path is None or len(action_norm_path.strip()) == 0:
        raise ValueError(
            "action_norm_path is required. Please pass --action-norm-path to provide min/max for denormalization."
        )
    if not os.path.exists(action_norm_path):
        raise FileNotFoundError(f"action norm json not found: {action_norm_path}")
    with open(action_norm_path, "r") as f:
        data = json.load(f)
    if "min" not in data or "max" not in data:
        raise ValueError(
            f"action norm json must contain keys 'min' and 'max': {action_norm_path}"
        )
    mins = np.asarray(data["min"], dtype=np.float32)
    maxs = np.asarray(data["max"], dtype=np.float32)
    if mins.shape != maxs.shape or mins.ndim != 1 or mins.shape[0] == 0:
        raise ValueError(
            f"invalid min/max shape in action norm json: min={mins.shape}, max={maxs.shape}"
        )
    print(f"[denorm] loaded action norm stats from {action_norm_path}, dims={mins.shape[0]}")
    return mins, maxs


def load_eval_mam_data(
    data_path: str,
    device: torch.device,
    expected_act_dim: int,
    expected_mas_flat_dim: int,
):

    keys = ["mas", "mask"]
    trajectories = load_demo_dataset(data_path, keys=keys, num_traj=None, concat=False)
    missing_keys = [k for k in keys if k not in trajectories]
    assert not missing_keys, f"Missing keys in eval trajectories: {missing_keys}"
    assert len(trajectories["mask"]) > 0, "Empty eval trajectories"

    mas_flat_list = []
    mas_list = []
    mask_list = []
    for i in range(len(trajectories["mask"])):
        mas_t = torch.as_tensor(trajectories["mas"][i], device=device, dtype=torch.float32)
        mask_t = torch.as_tensor(trajectories["mask"][i], device=device, dtype=torch.float32)

        assert mas_t.ndim == 2, f"mas[{i}] must be 2D, got shape {tuple(mas_t.shape)}"
        assert mask_t.ndim == 2, f"mask[{i}] must be 2D, got shape {tuple(mask_t.shape)}"
        assert mask_t.shape[1] == expected_act_dim, (
            f"mask[{i}] act_dim mismatch: {mask_t.shape[1]} vs expected {expected_act_dim}"
        )
        if mask_t.shape[0] > mas_t.shape[0]:
            mask_t = mask_t[: mas_t.shape[0]]
        elif mask_t.shape[0] < mas_t.shape[0]:
            pad = torch.zeros(
                (mas_t.shape[0] - mask_t.shape[0], mask_t.shape[1]),
                device=device,
                dtype=mask_t.dtype,
            )
            mask_t = torch.cat([mask_t, pad], dim=0)

        mas_t = torch.where(mas_t == -1, torch.zeros_like(mas_t), mas_t)
        mas_flat = mas_t.reshape(-1)
        assert mas_flat.shape[0] == expected_mas_flat_dim, (
            f"mas_flat_dim mismatch at traj {i}: {mas_flat.shape[0]} vs expected {expected_mas_flat_dim}"
        )

        mas_list.append(mas_t)
        mas_flat_list.append(mas_flat)
        mask_list.append(mask_t)

    return {"mas_flat": mas_flat_list, "mas": mas_list, "mask": mask_list}


def read_obs_mas_dim_from_meta(data_path: str, act_dim: int) -> int:
    with File(data_path, "r") as f:
        assert "meta" in f, f"Missing 'meta' group in dataset: {data_path}"
        meta = f["meta"]
        assert "max_length" in meta, f"Missing 'meta/max_length' in dataset: {data_path}"
        max_length = int(meta["max_length"][()])
    obs_mas_dim = max_length * act_dim
    assert obs_mas_dim > 0, f"Invalid obs_mas_dim from meta: max_length={max_length}, act_dim={act_dim}"
    return obs_mas_dim


class SmallDemoDataset_DiffusionPolicy(Dataset):  # Load everything into memory
    def __init__(self, data_path, obs_process_fn, obs_space, device, num_traj):

        load_keys = ["observations", "actions", "mas", "mask", "env_states", "success", "terminated", "truncated"]
        trajectories = load_demo_dataset(data_path, keys=load_keys, num_traj=num_traj, concat=False)
        print(f"Loaded trajectory keys: {sorted(trajectories.keys())}")
        missing_keys = [k for k in load_keys if k not in trajectories]
        assert not missing_keys, f"Missing keys in loaded trajectories: {missing_keys}"

        # trajectories['observations'] is a list of dict, each dict is a traj, with keys in obs_space, values with length L+1
        # trajectories['actions'] is a list of np.ndarray (L, act_dim)
        # trajectories['mas'] is a list of np.ndarray (T, 7), per-traj masked action space (padded to max length)
        # trajectories['mask'] is a list of np.ndarray (T, 7), mask aligned with mas (0/1)
        # trajectories['env_states'] is a list of dict, per-traj environment states
        # trajectories['success'] is a list of np.ndarray (L,), per-step success flags
        # trajectories['terminated'] is a list of np.ndarray (L,), per-step termination flags
        # trajectories['truncated'] is a list of np.ndarray (L,), per-step truncation flags
        
        print("Raw trajectory loaded, beginning observation pre-processing...")

        # Pre-process the observations, make them align with the obs returned by the obs_wrapper
        obs_traj_dict_list = []
        for obs_traj_dict in trajectories["observations"]:
            _obs_traj_dict = reorder_keys(obs_traj_dict, obs_space)  # key order in demo is different from key order in env obs
            _obs_traj_dict = obs_process_fn(_obs_traj_dict)
            _obs_traj_dict["depth"] = torch.Tensor(_obs_traj_dict["depth"].astype(np.float32)).to(device=device, dtype=torch.float16)
            _obs_traj_dict["rgb"] = torch.from_numpy(_obs_traj_dict["rgb"]).to(device)  # still uint8
            _obs_traj_dict["state"] = torch.from_numpy(_obs_traj_dict["state"]).to(device)
            obs_traj_dict_list.append(_obs_traj_dict)
        trajectories["observations"] = obs_traj_dict_list
        self.obs_keys = list(_obs_traj_dict.keys())
        
        # Pre-process the actions\mas\mask
        for i in range(len(trajectories["actions"])):
            trajectories["actions"][i] = torch.Tensor(trajectories["actions"][i]).to(device=device)
        if "mas" in trajectories:
            for i in range(len(trajectories["mas"])):
                trajectories["mas"][i] = torch.Tensor(trajectories["mas"][i]).to(device=device)
        if "mask" in trajectories:
            for i in range(len(trajectories["mask"])):
                trajectories["mask"][i] = torch.Tensor(trajectories["mask"][i]).to(device=device)
        # Ensure mask is aligned to action length (some datasets store mask with L+1).
        for i in range(len(trajectories["actions"])):
            L = trajectories["actions"][i].shape[0]
            if "mask" not in trajectories:
                continue
            t = trajectories["mask"][i]
            if t.shape[0] > L:
                t = t[:L]
            elif t.shape[0] < L:
                pad = torch.zeros((L - t.shape[0], t.shape[1]), device=t.device, dtype=t.dtype)
                t = torch.cat([t, pad], dim=0)
            trajectories["mask"][i] = t
        if "mas" in trajectories and len(trajectories["mas"]) > 0:
            self.mas_flat_dim = int(trajectories["mas"][0].numel())
        else:
            self.mas_flat_dim = 0
        print("Obs/action pre-processing is done, start to pre-compute the slice indices...")

        self.obs_horizon, self.pred_horizon = obs_horizon, pred_horizon = (args.obs_horizon,args.pred_horizon,)
        self.slices = []
        num_traj = len(trajectories["actions"])
        total_transitions = 0
        for traj_idx in range(num_traj):
            L = trajectories["actions"][traj_idx].shape[0]
            obs_len = trajectories["observations"][traj_idx]["state"].shape[0]
            assert obs_len >= L + 1
            total_transitions += L

            # |o|o|                             observations: 2
            # | |a|a|a|a|a|a|a|a|               actions executed: 8
            # |p|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p| actions predicted: 16
            pad_before = obs_horizon - 1
            # Pad before the trajectory, so the first action of an episode is in "actions executed"
            # obs_horizon - 1 is the number of "not used actions"
            pad_after = pred_horizon - obs_horizon
            # Pad after the trajectory, so all the observations are utilized in training
            # Note that in the original code, pad_after = act_horizon - 1, but I think this is not the best choice
            self.slices += [
                (traj_idx, start, start + pred_horizon)
                for start in range(-pad_before, L - pred_horizon + pad_after)
            ]  # slice indices follow convention [start, end)

        print(f"Total transitions: {total_transitions}, Total obs sequences: {len(self.slices)}")

        self.trajectories = trajectories

    def __getitem__(self, index):
        traj_idx, start, end = self.slices[index]
        L, act_dim = self.trajectories["actions"][traj_idx].shape

        obs_traj = self.trajectories["observations"][traj_idx]
        obs_seq = {}
        for k, v in obs_traj.items():
            obs_seq[k] = v[max(0, start) : start + self.obs_horizon]  # start+self.obs_horizon is at least 1
            if start < 0:  # pad before the trajectory
                pad_len = -start
                if k == "state":
                    # keep delta ~ 0 by linear extrapolation: o[-1] = o[0] - (o[1] - o[0])
                    if obs_seq[k].shape[0] >= 2:
                        d = obs_seq[k][1] - obs_seq[k][0]
                    else:
                        d = torch.zeros_like(obs_seq[k][0])
                    pad_obs = [obs_seq[k][0] - d * n for n in range(pad_len, 0, -1)]
                    obs_seq[k] = torch.cat((torch.stack(pad_obs, dim=0), obs_seq[k]), dim=0)
                
                else:
                    # repeat the first frame for visual obs to avoid uint8 underflow/overflow
                    pad_obs_seq = torch.stack([obs_seq[k][0]] * pad_len, dim=0)
                    obs_seq[k] = torch.cat((pad_obs_seq, obs_seq[k]), dim=0)
        
        
        mas_traj = self.trajectories["mas"][traj_idx]  # (T, 7)
        mas_traj = torch.where(mas_traj == -1, torch.zeros_like(mas_traj), mas_traj)
        mas_flat = mas_traj.reshape(-1)  # (T*7,)
        mas_seq = mas_flat.unsqueeze(0).repeat(self.obs_horizon, 1)  # (obs_horizon, T*7)
        obs_seq["mas"] = mas_seq

        # Slice action window directly from full trajectory length.
        act_seq = self.trajectories["actions"][traj_idx][max(0, start) : end]
        if start < 0:  # pad before the trajectory
            # keep delta ~ 0 by linear extrapolation: a[-1] = a[0] - (a[1] - a[0])
            if act_seq.shape[0] >= 2:
                d = act_seq[1] - act_seq[0]
            else:
                d = torch.zeros_like(act_seq[0])
            pad_len = -start
            pad_actions = [act_seq[0] - d * k for k in range(pad_len, 0, -1)]
            act_seq = torch.cat([torch.stack(pad_actions, dim=0), act_seq], dim=0)
        if end > L:  # pad after the trajectory
            pad_action = act_seq[-1]
            act_seq = torch.cat([act_seq, pad_action.repeat(end - L, 1)], dim=0)
            # making the robot (arm and gripper) stay still
        
        mask_seq = None
        if "mask" in self.trajectories:
            mask_traj = self.trajectories["mask"][traj_idx]
            mask_seq = mask_traj[max(0, start) : end]
            if start < 0:
                mask_seq = torch.cat([mask_seq[0].repeat(-start, 1), mask_seq], dim=0)
            if end > L:
                pad_len = end - L
                pad_mask = torch.zeros(
                    (pad_len, mask_seq.shape[1]),
                    device=mask_seq.device,
                    dtype=mask_seq.dtype,
                )
                mask_seq = torch.cat([mask_seq, pad_mask], dim=0)
        
        assert (obs_seq["state"].shape[0] == self.obs_horizon and act_seq.shape[0] == self.pred_horizon)
        
        return {
            "observations": obs_seq,
            "actions": act_seq,
            "mask": mask_seq,
        }

    def __len__(self):
        return len(self.slices)

    def debug_print_sample(self, sample_idx: int = 0):
        sample = self.__getitem__(sample_idx)
        obs = sample.get("observations", {})
        print("=== Dataset Sample Debug ===")
        print(f"sample_idx: {sample_idx}")
        print(f"observations keys: {list(obs.keys())}")
        for k, v in obs.items():
            if hasattr(v, "shape"):
                print(f"  obs[{k}]: shape={tuple(v.shape)}, dtype={v.dtype}")
            else:
                print(f"  obs[{k}]: type={type(v)}")
        actions = sample.get("actions")
        if actions is not None and hasattr(actions, "shape"):
            print(f"actions: shape={tuple(actions.shape)}, dtype={actions.dtype}")
        else:
            print(f"actions: type={type(actions)}")
        mask = sample.get("mask")
        if mask is not None and hasattr(mask, "shape"):
            print(f"mask: shape={tuple(mask.shape)}, dtype={mask.dtype}")
        else:
            print(f"mask: {mask}")
        print("============================")


class Agent(nn.Module):
    def __init__(self, env: VectorEnv, args: Args, obs_mas_dim: int):
        super().__init__()
        self.obs_horizon = args.obs_horizon
        self.act_horizon = args.act_horizon
        self.pred_horizon = args.pred_horizon
        self.include_mas = True
        assert (len(env.single_observation_space["state"].shape) == 2)  # (obs_horizon, obs_dim)
        assert len(env.single_action_space.shape) == 1  # (act_dim, )
        self.act_dim = env.single_action_space.shape[0]
        obs_state_dim = env.single_observation_space["state"].shape[1]
        assert obs_mas_dim > 0, "obs_mas_dim must be > 0 for MAM conditioning"
        self.obs_mas_dim = obs_mas_dim
        total_visual_channels = (env.single_observation_space["rgb"].shape[-1]+ env.single_observation_space["depth"].shape[-1])

        visual_feature_dim = 256
        self.visual_encoder = PlainConv(in_channels=total_visual_channels, out_dim=visual_feature_dim, pool_feature_map=True)
        self.noise_pred_net = DiTNoiseNet(
            ac_dim=self.act_dim,
            ac_chunk=self.pred_horizon,
            obs_dim=visual_feature_dim + obs_state_dim + obs_mas_dim,
            time_dim=args.diffusion_step_embed_dim,
            use_mask=True,
        )
        self.num_diffusion_iters = 100
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.num_diffusion_iters,
            beta_schedule="squaredcos_cap_v2",  # has big impact on performance, try not to change
            clip_sample=True,  # clip output to [-1,1] to improve stability
            prediction_type="epsilon",  # predict noise (instead of denoised action)
        )
        self.action_denorm_min = None
        self.action_denorm_max = None
        self.action_denorm_dims = 0

    def set_action_denormalizer(self, mins: np.ndarray, maxs: np.ndarray, device):
        self.action_denorm_dims = int(mins.shape[0])
        self.action_denorm_min = torch.as_tensor(mins, device=device, dtype=torch.float32)
        self.action_denorm_max = torch.as_tensor(maxs, device=device, dtype=torch.float32)

    def encode_obs(self, obs_seq, eval_mode):
        rgb = obs_seq["rgb"].float() / 255.0  # (B, obs_horizon, 3*k, H, W)
        depth = obs_seq["depth"].float() / 1024.0  # (B, obs_horizon, 1*k, H, W)
        img_seq = torch.cat([rgb, depth], dim=2)  # (B, obs_horizon, C, H, W)
        batch_size = img_seq.shape[0]
        img_seq = img_seq.flatten(end_dim=1)  # (B*obs_horizon, C, H, W)
        if hasattr(self, "aug") and not eval_mode:
            img_seq = self.aug(img_seq)  # (B*obs_horizon, C, H, W)
        visual_feature = self.visual_encoder(img_seq)  # (B*obs_horizon, D)
        visual_feature = visual_feature.reshape(batch_size, self.obs_horizon, visual_feature.shape[1])  # (B, obs_horizon, D)y
        mas = obs_seq.get("mas", None)
        if mas is None:
            # Online rollout observations may not include mas; use zeros to keep feature size consistent.
            mas = torch.zeros(
                (batch_size, self.obs_horizon, self.obs_mas_dim),
                device=obs_seq["state"].device,
                dtype=obs_seq["state"].dtype,
            )
        else:
            mas = mas.to(device=obs_seq["state"].device, dtype=obs_seq["state"].dtype)
        feature = torch.cat((visual_feature, obs_seq["state"], mas), dim=-1)
        return feature  # (B, obs_horizon, D+obs_state_dim+obs_mas_dim)

    def compute_loss(self, obs_seq, action_seq, mask_seq):
        B = obs_seq["state"].shape[0]

        # observation as FiLM conditioning
        obs_cond = self.encode_obs(obs_seq, eval_mode=False)  # (B, obs_horizon, obs_dim)

        # sample noise to add to actions
        noise = torch.randn((B, self.pred_horizon, self.act_dim), device=device)

        # sample a diffusion iteration for each data point
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (B,), device=device).long()

        # add noise to the clean images(actions) according to the noise magnitude at each diffusion iteration
        # (this is the forward diffusion process)
        noisy_action_seq = self.noise_scheduler.add_noise(action_seq, noise, timesteps)

        assert mask_seq is not None, "mask_seq is required for masked diffusion policy training"
        mask_seq = mask_seq.to(device=noisy_action_seq.device, dtype=noisy_action_seq.dtype)
        # hard overwrite known actions into the noisy sequence
        a_in = mask_seq * action_seq + (1.0 - mask_seq) * noisy_action_seq

        # predict the noise residual
        noise_pred = self.noise_pred_net(a_in, timesteps, obs_cond, mask=mask_seq)

        diff = (noise - noise_pred) * (1.0 - mask_seq)
        denom = (1.0 - mask_seq).sum().clamp_min(1.0)
        return (diff ** 2).sum() / denom

    def get_action(self, obs_seq, mask_seq=None, a0_seq=None):
        # init scheduler
        # self.noise_scheduler.set_timesteps(self.num_diffusion_iters)
        # set_timesteps will change noise_scheduler.timesteps is only used in noise_scheduler.step()
        # noise_scheduler.step() is only called during inference
        # if we use DDPM, and inference_diffusion_steps == train_diffusion_steps, then we can skip this

        # obs_seq['state']: (B, obs_horizon, obs_state_dim)
        B = obs_seq["state"].shape[0]
        with torch.no_grad():
            obs_seq["rgb"] = obs_seq["rgb"].permute(0, 1, 4, 2, 3)
            obs_seq["depth"] = obs_seq["depth"].permute(0, 1, 4, 2, 3)

            obs_cond = self.encode_obs(obs_seq, eval_mode=True)  # (B, obs_horizon, obs_dim)

            # initialize action from Guassian noise
            noisy_action_seq = torch.randn(
                (B, self.pred_horizon, self.act_dim), device=obs_seq["state"].device
            )

            if mask_seq is None or a0_seq is None:
                # Online rollout path without explicit masked-action constraints.
                a0_seq = torch.zeros_like(noisy_action_seq)
                mask_seq = torch.zeros_like(noisy_action_seq)
            else:
                a0_seq = a0_seq[:, : self.pred_horizon].to(
                    device=noisy_action_seq.device, dtype=noisy_action_seq.dtype
                )
                mask_seq = mask_seq[:, : self.pred_horizon].to(
                    device=noisy_action_seq.device, dtype=noisy_action_seq.dtype
                )

            noisy_action_seq = (
                mask_seq * a0_seq + (1.0 - mask_seq) * noisy_action_seq
            )

            for k in self.noise_scheduler.timesteps:
                # predict noise
                a_in = mask_seq * a0_seq + (1.0 - mask_seq) * noisy_action_seq
                timesteps = torch.full(
                    (B,), k, dtype=torch.long, device=noisy_action_seq.device
                )
                noise_pred = self.noise_pred_net(a_in, timesteps, obs_cond, mask=mask_seq)

                # inverse diffusion step (remove noise)
                noisy_action_seq = self.noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=k,
                    sample=noisy_action_seq,
                ).prev_sample
                noisy_action_seq = (
                    mask_seq * a0_seq + (1.0 - mask_seq) * noisy_action_seq
                )

        # only take act_horizon number of actions
        start = self.obs_horizon - 1
        end = start + self.act_horizon
        action_seq = noisy_action_seq[:, start:end]  # (B, act_horizon, act_dim)
        if self.action_denorm_dims > 0:
            d = min(self.action_denorm_dims, action_seq.shape[-1])
            mins = self.action_denorm_min[:d]
            maxs = self.action_denorm_max[:d]
            action_seq = action_seq.clone()
            action_seq[..., :d] = mins + 0.5 * (action_seq[..., :d] + 1.0) * (maxs - mins)
        return action_seq

def save_ckpt(run_name, tag):
    os.makedirs(f"runs/{run_name}/checkpoints", exist_ok=True)
    ema.copy_to(ema_agent.parameters())
    torch.save(
        {
            "agent": agent.state_dict(),
            "ema_agent": ema_agent.state_dict(),
        },
        f"runs/{run_name}/checkpoints/{tag}.pt",
    )


if __name__ == "__main__":
    args = tyro.cli(Args)

    if args.exp_name is None:
        args.exp_name = os.path.basename(__file__)[: -len(".py")]
        run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    else:
        run_name = args.exp_name

    assert args.obs_horizon + args.act_horizon - 1 <= args.pred_horizon
    assert args.obs_horizon >= 1 and args.act_horizon >= 1 and args.pred_horizon >= 1

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    denorm_mins, denorm_maxs = load_action_denorm_stats(args.action_norm_path)
    print("[denorm] eval actions will be denormalized before env.step().")

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # create evaluation environment
    env_kwargs = dict(
        control_mode=args.control_mode,
        reward_mode="sparse",
        obs_mode="rgb+depth",
        render_mode="rgb_array",
        human_render_camera_configs=dict(shader_pack="default")
    )
    assert args.max_episode_steps != None, "max_episode_steps must be specified as imitation learning algorithms task solve speed is dependent on the data you train on"
    env_kwargs["max_episode_steps"] = args.max_episode_steps
    other_kwargs = dict(obs_horizon=args.obs_horizon)
    
    envs = make_eval_envs(
        args.env_id,
        args.num_eval_envs,
        args.sim_backend,
        env_kwargs,
        other_kwargs,
        video_dir=f"runs/{run_name}/videos" if args.capture_video else None,
        wrappers=[FlattenRGBDObservationWrapper],
    )

    # whether to setup wandb logging
    if args.track:
        import wandb
        config = vars(args)
        config["eval_env_cfg"] = dict(**env_kwargs, num_envs=args.num_eval_envs, env_id=args.env_id, env_horizon=args.max_episode_steps)
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=config,
            name=run_name,
            save_code=True,
            group="DiffusionPolicy",
            tags=["diffusion_policy"],
        )
    
    # setup tensorboard writer
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # create temporary env to get original observation space as AsyncVectorEnv (CPU parallelization) doesn't permit that
    tmp_env = gym.make(args.env_id, **env_kwargs)
    orignal_obs_space = tmp_env.observation_space
    tmp_env.close()

    # preprocess observations for dataset
    obs_process_fn = partial(
        convert_obs,
        concat_fn=partial(np.concatenate, axis=-1),
        transpose_fn=partial(np.transpose, axes=(0, 3, 1, 2)),  # (B, H, W, C) -> (B, C, H, W)
        state_obs_extractor=build_state_obs_extractor(args.env_id),
        depth=True,
    )

    dataset = SmallDemoDataset_DiffusionPolicy(
        data_path=args.demo_path,
        obs_process_fn=obs_process_fn,
        obs_space=orignal_obs_space,
        device=device,
        num_traj=args.num_demos
    )
    obs_mas_dim = read_obs_mas_dim_from_meta(args.demo_path, envs.single_action_space.shape[0])
    assert obs_mas_dim == dataset.mas_flat_dim, (
        f"obs_mas_dim from meta ({obs_mas_dim}) != flattened mas dim from data ({dataset.mas_flat_dim})"
    )
    dataset.debug_print_sample(0)
    eval_mam_data = load_eval_mam_data(
        data_path=args.test_demo_path,
        device=device,
        expected_act_dim=envs.single_action_space.shape[0],
        expected_mas_flat_dim=obs_mas_dim,
    )
    
    sampler = RandomSampler(dataset, replacement=False)
    batch_sampler = BatchSampler(sampler, batch_size=args.batch_size, drop_last=True)
    batch_sampler = IterationBasedBatchSampler(batch_sampler, args.total_iters)
    train_dataloader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=args.num_dataload_workers,
        worker_init_fn=lambda worker_id: worker_init_fn(worker_id, base_seed=args.seed),
        persistent_workers=(args.num_dataload_workers > 0),
    )

    agent = Agent(envs, args, obs_mas_dim=obs_mas_dim).to(device)
    agent.set_action_denormalizer(denorm_mins, denorm_maxs, device)

    optimizer = optim.AdamW(
        params=agent.parameters(), lr=args.lr, betas=(0.95, 0.999), weight_decay=1e-6
    )

    # Cosine LR schedule with linear warmup
    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=args.total_iters,
    )

    # Exponential Moving Average
    # accelerates training and improves stability
    # holds a copy of the model weights
    ema = EMAModel(parameters=agent.parameters(), power=0.75)
    ema_agent = Agent(envs, args, obs_mas_dim=obs_mas_dim).to(device)
    ema_agent.set_action_denormalizer(denorm_mins, denorm_maxs, device)

    best_eval_metrics = defaultdict(float)
    timings = defaultdict(float)

    # define evaluation and logging functions
    def evaluate_and_save_best(iteration):
        if iteration % args.eval_freq == 0:
            last_tick = time.time()
            ema.copy_to(ema_agent.parameters())
            eval_metrics = evaluate_mam(
                args.num_eval_episodes,
                ema_agent,
                envs,
                device,
                args.sim_backend,
                eval_mam_data=eval_mam_data,
                obs_horizon=args.obs_horizon,
                pred_horizon=args.pred_horizon,
            )
            timings["eval"] += time.time() - last_tick

            for k, v in eval_metrics.items():
                metric_value = float(np.mean(v)) if isinstance(v, np.ndarray) else float(v)
                writer.add_scalar(f"eval/{k}", metric_value, iteration)
                print(f"{k}: {metric_value:.6f}")
            for k in ["success_once", "success_at_end"]:
                if k in eval_metrics:
                    metric_value = float(np.mean(eval_metrics[k]))
                    if k not in best_eval_metrics or metric_value > best_eval_metrics[k]:
                        best_eval_metrics[k] = metric_value
                        save_ckpt(run_name, f"best_eval_{k}")
                        print(f"New best {k}: {metric_value:.6f}. Saving checkpoint.")
    def log_metrics(iteration):
        if iteration % args.log_freq == 0:
            writer.add_scalar(
                "charts/learning_rate", optimizer.param_groups[0]["lr"], iteration
            )
            writer.add_scalar("losses/total_loss", total_loss.item(), iteration)
            for k, v in timings.items():
                writer.add_scalar(f"time/{k}", v, iteration)

    # ---------------------------------------------------------------------------- #
    # Training begins.
    # ---------------------------------------------------------------------------- #
    agent.train()
    pbar = tqdm(total=args.total_iters)
    last_tick = time.time()
    for iteration, data_batch in enumerate(train_dataloader):
        timings["data_loading"] += time.time() - last_tick

        # forward and compute loss
        last_tick = time.time()
        total_loss = agent.compute_loss(
            obs_seq=data_batch["observations"],  # obs_batch_dict['state'] is (B, L, obs_dim)
            action_seq=data_batch["actions"],  # (B, L, act_dim)
            mask_seq=data_batch["mask"],  # (B, L, act_dim)
        )
        timings["forward"] += time.time() - last_tick

        # backward
        last_tick = time.time()
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        lr_scheduler.step()  # step lr scheduler every batch, this is different from standard pytorch behavior
        timings["backward"] += time.time() - last_tick

        # ema step
        last_tick = time.time()
        ema.step(agent.parameters())
        timings["ema"] += time.time() - last_tick

        # Evaluation
        evaluate_and_save_best(iteration)
        log_metrics(iteration)

        # Checkpoint
        if args.save_freq is not None and iteration % args.save_freq == 0:
            save_ckpt(run_name, str(iteration))
        pbar.update(1)
        pbar.set_postfix({"loss": total_loss.item()})
        last_tick = time.time()

    evaluate_and_save_best(args.total_iters)
    log_metrics(args.total_iters)

    envs.close()
    writer.close()
