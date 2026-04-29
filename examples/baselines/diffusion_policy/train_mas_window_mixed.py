ALGO_NAME = "BC_Diffusion_rgbd_DiT_MASWindow_Mixed"
MAS_STEP_DIM = 8

import os
import random
import time
import sys
import json
from collections import defaultdict
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import List, Literal, Optional

import gymnasium as gym
from gymnasium.vector.vector_env import VectorEnv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import tyro
from diffusers.optimization import get_scheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from h5py import File
from gymnasium import spaces
from mani_skill.utils import common
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import BatchSampler, RandomSampler
from torch.utils.tensorboard import SummaryWriter

from evaluate.evaluate_mas_window_mixed import evaluate_mas_window_mixed
from models.mas_conv1d import MasConv1D
from models.mas_conv2d import MasConv
from models.modeling_ditdp import DiTNoiseNet
from models.plain_conv import PlainConv
from utils.add_progress_to_mas_utils import (
    augment_mask_with_progress,
    augment_mas_with_progress,
)
from utils.build_progress_window_utils import (
    build_dual_mas_window_obs_horizon,
)
from utils.denormalize_utils import (compute_state_min_max,
                                     load_action_denorm_stats)
from utils.control_error_utils import (
    aggregate_control_error,
    compute_control_error_results_from_rollouts,
    load_ce_eval_data,
    load_source_episode_ids,
)
from utils.draw_p_t_curve_utils import (
    save_control_error_curve,
)
from utils.eval_video_sampling_utils import (
    build_capture_indices,
    validate_eval_video_config,
)
from utils.load_eval_data_utils import (infer_eval_reset_seeds_from_demo,
                                  infer_eval_traj_ids_from_demo,
                                  load_traj_mask_type_slots,
                                  load_traj_mask_types,
                                  subset_eval_data)
from utils.load_train_data_utils import load_dataset_meta, load_demo_dataset
from utils.loss_utils import (
    compute_mask_weighted_noise_mse,
    slice_action_mask_sequence,
)
from utils.make_env import make_eval_envs
from utils.utils import (IterationBasedBatchSampler, build_state_obs_extractor,
                         convert_obs, worker_init_fn)
from utils.video_utils import (
    clear_iteration_artifacts,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from STPM.models.stpm_encoder import STPMEncoder


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
    capture_video_freq: int = 10
    """archive one eval video every capture_video_freq eval episodes"""

    env_id: str = "PegInsertionSide-v1"
    """the id of the environment"""
    demo_path: str = (
        "demos/PegInsertionSide-v1/trajectory.state.pd_ee_delta_pose.physx_cpu.h5"
    )
    """the path of demo dataset, it is expected to be a ManiSkill dataset h5py format file"""
    test_demo_path: Optional[str] = None
    """optional eval demo dataset used for online mas conditioning; falls back to demo_path when unset"""
    eval_demo_metadata_path: Optional[str] = None
    """optional eval demo metadata json used to infer reset seeds; falls back to filename-based guessing when unset"""
    stpm_ckpt_path: str = ""
    """Path to the STPM checkpoint used to infer rollout progress during evaluation."""
    stpm_config_path: str = "STPM/config/rewind_maniskill.yaml"
    """Path to the STPM config paired with stpm_ckpt_path."""
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
    long_window_backward_length: Optional[int] = None
    """MAS long window length before t; defaults to 0"""
    long_window_forward_length: Optional[int] = None
    """MAS long window length from t onward, including t; defaults to pred_horizon"""
    diffusion_step_embed_dim: int = 64  # not very important
    short_window_horizon: int = 8
    """future MAS short window length; set to 0 to disable the short-window branch"""
    mas_long_encode_mode: Literal["1DConv", "2DConv"] = "2DConv"
    """how to encode the long MAS window before concatenating it into the observation conditioning"""
    mas_long_conv_output_dim: int = 64
    """Output feature dim used by long-window conv encoders; set to 0 to disable the long-window branch."""
    loss_mode: Literal["average", "weighted"] = "average"
    """average keeps plain MSE; weighted balances known area and unknown area"""
    loss_mask_area_weight: float = 0.2
    """known area weight when loss_mode=weighted; unknown area uses 1 - this value"""
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
    num_eval_demos: int = 100
    """number of demo seeds to evaluate from the loaded demos"""
    num_eval_envs: int = 10
    """the number of parallel environments to evaluate the agent on"""
    inpainting: bool = False
    """whether to use online j=0,r=0 inpainting overwrite during each evaluation rollout"""
    eval_progress_bar: bool = False
    """whether to show per-eval episode progress bars inside evaluation"""
    action_norm_path: Optional[str] = None
    """Path to action normalization stats. Supports legacy json or preprocessed h5/meta; required for rollout denormalization."""
    sim_backend: str = "physx_cpu"
    """the simulation backend to use for evaluation environments. can be "cpu" or "gpu"""
    num_dataload_workers: int = 0
    """the number of workers to use for loading the training data in the torch dataloader"""
    control_mode: str = "pd_ee_pose"
    """the control mode to use for the evaluation environments. Must match the control mode of the demonstration dataset."""
    obs_mode: Literal["rgb", "rgb+depth"] = "rgb+depth"
    """visual modality consumed by the policy. rgb ignores depth even when the dataset/eval env provide it."""

    # additional tags/configs for logging purposes to wandb and shared comparisons with other algorithms
    demo_type: Optional[str] = None


def policy_uses_depth(obs_mode: str) -> bool:
    if obs_mode == "rgb":
        return False
    if obs_mode == "rgb+depth":
        return True
    raise ValueError(f"Unsupported obs_mode={obs_mode!r}; expected 'rgb' or 'rgb+depth'.")


def stpm_eval_env_obs_mode(policy_obs_mode: str) -> str:
    if policy_obs_mode not in ("rgb", "rgb+depth"):
        raise ValueError(
            f"Unsupported obs_mode={policy_obs_mode!r}; expected 'rgb' or 'rgb+depth'."
        )
    return "rgb+depth"


def reorder_keys(d, ref_dict):
    out = dict()
    for k, v in ref_dict.items():
        if isinstance(v, dict) or isinstance(v, spaces.Dict):
            out[k] = reorder_keys(d[k], ref_dict[k])
        else:
            out[k] = d[k]
    return out


def build_eval_stpm_encoder(
    stpm_ckpt_path: str, stpm_config_path: str, device: torch.device
):
    if len(stpm_ckpt_path.strip()) == 0:
        raise ValueError(
            "stpm_ckpt_path is required for mas-window evaluation. "
            "Please pass --stpm-ckpt-path."
        )

    resolved_ckpt = STPMEncoder._resolve_path(stpm_ckpt_path)
    resolved_config = STPMEncoder._resolve_path(stpm_config_path)
    if not os.path.exists(resolved_ckpt):
        raise FileNotFoundError(f"STPM checkpoint not found: {resolved_ckpt}")
    if not os.path.exists(resolved_config):
        raise FileNotFoundError(f"STPM config not found: {resolved_config}")

    stpm_encoder = STPMEncoder(
        ckpt_path=resolved_ckpt,
        config_path=resolved_config,
        device=device,
    )
    stpm_n_obs_steps = int(stpm_encoder.cfg.model.n_obs_steps)
    stpm_frame_gap = int(stpm_encoder.cfg.model.frame_gap)
    stpm_sequence_length = stpm_n_obs_steps + 1
    print(
        f"[stpm] loaded checkpoint={resolved_ckpt}, config={resolved_config}, "
        f"camera={stpm_encoder.camera_name}, state_dim={stpm_encoder.state_dim}, "
        f"n_obs_steps={stpm_n_obs_steps}, frame_gap={stpm_frame_gap}, "
        f"sequence_length={stpm_sequence_length}"
    )
    return stpm_encoder, stpm_n_obs_steps, stpm_frame_gap


def validate_only_mas_eval_layout(envs: VectorEnv, stpm_encoder):
    rgb_shape = envs.single_observation_space["rgb"].shape
    depth_shape = envs.single_observation_space["depth"].shape
    state_shape = envs.single_observation_space["state"].shape

    if getattr(stpm_encoder, "camera_name", None) != "base_camera":
        raise ValueError(
            "Only STPM configs with a single 'base_camera' are supported, "
            f"got camera_name={getattr(stpm_encoder, 'camera_name', None)!r}."
        )
    if rgb_shape[-1] != 3 or depth_shape[-1] != 1:
        raise ValueError(
            "Only single-camera rollout observations are supported for STPM-driven "
            f"mas-window evaluation, got rgb shape {rgb_shape} and depth shape {depth_shape}."
        )
    if state_shape[-1] != int(stpm_encoder.state_dim):
        raise ValueError(
            "Current rollout state cannot be mapped to the STPM checkpoint state layout. "
            f"Expected last dim {stpm_encoder.state_dim}, got {state_shape[-1]}."
        )


def move_batch_to_device(batch, device):
    out = {}
    for k, v in batch.items():
        if isinstance(v, dict):
            out[k] = move_batch_to_device(v, device)
        elif torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def _meta_flag(meta: dict, key: str, default: bool = False) -> bool:
    if key not in meta:
        return default
    value = np.asarray(meta[key])
    if value.shape == ():
        return bool(value.item())
    return bool(np.all(value))


def _load_state_norm_stats_from_meta(meta: dict):
    if "state_min" not in meta or "state_max" not in meta:
        return None, None
    return (
        np.asarray(meta["state_min"], dtype=np.float32),
        np.asarray(meta["state_max"], dtype=np.float32),
    )

class SmallDemoDataset_MasWindowDiffusionPolicy(Dataset):  # Load everything into memory
    def __init__(self, data_path, obs_process_fn, obs_space, device, num_traj, traj_indices=None):
        self.include_depth = policy_uses_depth(args.obs_mode)

        load_keys = ["observations", "actions", "mas", "mask", "env_states", "success", "terminated", "truncated"]
        trajectories = load_demo_dataset(
            data_path,
            keys=load_keys,
            num_traj=num_traj if traj_indices is None else None,
            concat=False,
            traj_indices=traj_indices,
        )
        dataset_meta = load_dataset_meta(data_path)
        self.state_is_normalized = _meta_flag(dataset_meta, "states_normalized", False)
        print(
            f"[obs-mode] policy obs_mode={args.obs_mode}, "
            f"dataset depth used by policy={self.include_depth}"
        )
        print(f"Loaded trajectory keys: {sorted(trajectories.keys())}")
        missing_keys = [k for k in load_keys if k not in trajectories]
        assert not missing_keys, f"Missing keys in loaded trajectories: {missing_keys}"

        # trajectories['observations'] is a list of dict, each dict is a traj, with keys in obs_space, values with length L+1
        # trajectories['actions'] is a list of np.ndarray (L, act_dim)
        # trajectories['mas'] is a list of np.ndarray (T, 7), per-traj masked action space (padded to max length)
        # trajectories['mask'] is a list of np.ndarray (T, 7), per-traj explicit keep mask
        # trajectories['env_states'] is a list of dict, per-traj environment states
        # trajectories['success'] is a list of np.ndarray (L,), per-step success flags
        # trajectories['terminated'] is a list of np.ndarray (L,), per-step termination flags
        # trajectories['truncated'] is a list of np.ndarray (L,), per-step truncation flags
        
        print("Raw trajectory loaded, beginning observation pre-processing...")

        # Pre-process the observations, make them align with the obs returned by the obs_wrapper
        obs_traj_dict_list = []
        for obs_traj_dict in trajectories["observations"]:
            precomputed_state = obs_traj_dict.get("state", None)
            _obs_traj_dict = reorder_keys(obs_traj_dict, obs_space)  # key order in demo is different from key order in env obs
            _obs_traj_dict = obs_process_fn(_obs_traj_dict)
            if self.include_depth:
                _obs_traj_dict["depth"] = torch.Tensor(
                    _obs_traj_dict["depth"].astype(np.float32)
                ).to(dtype=torch.float16)
            _obs_traj_dict["rgb"] = torch.from_numpy(_obs_traj_dict["rgb"])  # still uint8
            if precomputed_state is not None and self.state_is_normalized:
                _obs_traj_dict["state"] = torch.from_numpy(
                    np.asarray(precomputed_state, dtype=np.float32)
                ).to(dtype=torch.float32)
            else:
                _obs_traj_dict["state"] = torch.from_numpy(_obs_traj_dict["state"]).to(dtype=torch.float32)
            obs_traj_dict_list.append(_obs_traj_dict)
        trajectories["observations"] = obs_traj_dict_list
        self.obs_keys = list(_obs_traj_dict.keys())
        state_meta_min, state_meta_max = _load_state_norm_stats_from_meta(dataset_meta)
        if self.state_is_normalized:
            if state_meta_min is None or state_meta_max is None:
                raise ValueError(
                    f"{data_path} marks states_normalized=true but missing meta/state_min|max"
                )
            self.state_min = torch.as_tensor(state_meta_min, dtype=torch.float32)
            self.state_max = torch.as_tensor(state_meta_max, dtype=torch.float32)
            print(
                f"[state_norm] using preprocessed normalized state from dataset meta, dim={self.state_min.shape[0]}"
            )
        else:
            self.state_min, self.state_max = compute_state_min_max(
                [obs_traj_dict["state"] for obs_traj_dict in obs_traj_dict_list]
            )
            print(
                f"[state_norm] computed state min/max from dataset, dim={self.state_min.shape[0]}"
            )
        
        # Pre-process the actions and mas
        for i in range(len(trajectories["actions"])):
            trajectories["actions"][i] = torch.as_tensor(trajectories["actions"][i], dtype=torch.float32)
        if "mas" in trajectories:
            for i in range(len(trajectories["mas"])):
                raw_mas_t = torch.as_tensor(trajectories["mas"][i], dtype=torch.float32)
                raw_mask_t = torch.as_tensor(trajectories["mask"][i], dtype=torch.float32)
                traj_len = int(trajectories["actions"][i].shape[0])
                if raw_mas_t.shape[-1] == MAS_STEP_DIM:
                    trajectories["mas"][i] = raw_mas_t
                else:
                    trajectories["mas"][i] = augment_mas_with_progress(raw_mas_t, traj_len)
                trajectories["mask"][i] = augment_mask_with_progress(
                    raw_mask_t,
                    traj_len=traj_len,
                    mas_t=raw_mas_t,
                )
                if trajectories["mas"][i].shape[-1] != MAS_STEP_DIM:
                    raise ValueError(
                        f"[check1] augmented mas traj {i} expected last dim {MAS_STEP_DIM}, "
                        f"got shape {tuple(trajectories['mas'][i].shape)}"
                    )
                if trajectories["mask"][i].shape != trajectories["mas"][i].shape:
                    raise ValueError(
                        f"[check1] mask/mas shape mismatch at traj {i}: "
                        f"mask={tuple(trajectories['mask'][i].shape)} vs mas={tuple(trajectories['mas'][i].shape)}"
                    )
                if i == 0:
                    print(
                        f"[check1] augment_mas_with_progress output shape="
                        f"{tuple(trajectories['mas'][i].shape)}"
                    )
        print("Obs/action pre-processing is done, start to pre-compute the slice indices...")
        self._printed_dataset_checks = False

        self.obs_horizon, self.pred_horizon = obs_horizon, pred_horizon = (args.obs_horizon,args.pred_horizon,)
        self.long_window_horizon = int(args.long_window_horizon)
        self.long_window_backward_length = int(args.long_window_backward_length)
        self.long_window_forward_length = int(args.long_window_forward_length)
        self.act_horizon = args.act_horizon
        self.short_window_horizon = args.short_window_horizon
        self.enable_long_window = args.mas_long_conv_output_dim > 0
        self.enable_short_window = args.short_window_horizon > 0
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
        # add padding
        for k, v in obs_traj.items():
            obs_seq[k] = v[max(0, start) : start + self.obs_horizon]  # start+self.obs_horizon is at least 1
            if start < 0:  # pad before the trajectory
                pad_len = -start
                if k == "state":
                    pad_obs_seq = torch.stack([obs_seq[k][0]] * pad_len, dim=0)
                    obs_seq[k] = torch.cat((pad_obs_seq, obs_seq[k]), dim=0)
                else:
                    # repeat the first frame for visual obs to avoid uint8 underflow/overflow
                    pad_obs_seq = torch.stack([obs_seq[k][0]] * pad_len, dim=0)
                    obs_seq[k] = torch.cat((pad_obs_seq, obs_seq[k]), dim=0)
        current_step = start + self.obs_horizon - 1
        mas_traj = self.trajectories["mas"][traj_idx]
        mask_traj = self.trajectories["mask"][traj_idx]
        mas_long_window, mas_short_window = build_dual_mas_window_obs_horizon(
            mas_traj,
            current_step=current_step,
            obs_horizon=self.obs_horizon,
            long_window_horizon=self.long_window_horizon if self.enable_long_window else 0,
            short_window_horizon=self.short_window_horizon if self.enable_short_window else 0,
            long_window_backward_length=self.long_window_backward_length if self.enable_long_window else 0,
            long_window_forward_length=self.long_window_forward_length if self.enable_long_window else 0,
        )
        mas_long_window_mask, mas_short_window_mask = build_dual_mas_window_obs_horizon(
            mask_traj,
            current_step=current_step,
            obs_horizon=self.obs_horizon,
            long_window_horizon=self.long_window_horizon if self.enable_long_window else 0,
            short_window_horizon=self.short_window_horizon if self.enable_short_window else 0,
            long_window_backward_length=self.long_window_backward_length if self.enable_long_window else 0,
            long_window_forward_length=self.long_window_forward_length if self.enable_long_window else 0,
        )
        obs_seq["mas_long_window"] = mas_long_window.reshape(self.obs_horizon, -1)
        obs_seq["mas_short_window"] = mas_short_window.reshape(self.obs_horizon, -1)
        obs_seq["mas_long_window_mask"] = mas_long_window_mask.reshape(self.obs_horizon, -1)
        obs_seq["mas_short_window_mask"] = mas_short_window_mask.reshape(self.obs_horizon, -1)
        if not self._printed_dataset_checks:
            print(
                f"[check2] dataset obs['mas_long_window'] shape={tuple(obs_seq['mas_long_window'].shape)} "
                f"(enabled={self.enable_long_window}, expected ({self.obs_horizon}, {MAS_STEP_DIM * (self.long_window_horizon if self.enable_long_window else 0)}))"
            )
            print(
                f"[check2] dataset obs['mas_short_window'] shape={tuple(obs_seq['mas_short_window'].shape)} "
                f"(enabled={self.enable_short_window}, expected ({self.obs_horizon}, {MAS_STEP_DIM * (self.short_window_horizon if self.enable_short_window else 0)}))"
            )
            self._printed_dataset_checks = True
        obs_seq["state_is_normalized"] = torch.tensor(self.state_is_normalized, dtype=torch.bool)

        # Slice action window directly from full trajectory length.
        act_seq = self.trajectories["actions"][traj_idx][max(0, start) : end]
        action_mask_seq = slice_action_mask_sequence(
            mask_traj=mask_traj,
            start=start,
            end=end,
            action_len=L,
            act_dim=act_dim,
            pred_horizon=self.pred_horizon,
        )
        if start < 0:  # pad before the trajectory
            pad_len = -start
            pad_actions = act_seq[0].unsqueeze(0).repeat(pad_len, 1)
            act_seq = torch.cat([pad_actions, act_seq], dim=0)
        if end > L:  # pad after the trajectory
            pad_action = act_seq[-1]
            act_seq = torch.cat([act_seq, pad_action.repeat(end - L, 1)], dim=0)
            # making the robot (arm and gripper) stay still
        
        assert (
            obs_seq["state"].shape[0] == self.obs_horizon
            and act_seq.shape[0] == self.pred_horizon
            and action_mask_seq.shape == act_seq.shape
        )
        
        return {
            "observations": obs_seq,
            "actions": act_seq,
            "action_mask": action_mask_seq,
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
        print("============================")


class Agent(nn.Module):
    def __init__(self, env: VectorEnv, args: Args):
        super().__init__()
        self.obs_horizon = args.obs_horizon
        self.act_horizon = args.act_horizon
        self.pred_horizon = args.pred_horizon
        self.long_window_horizon = int(args.long_window_horizon)
        self.long_window_backward_length = int(args.long_window_backward_length)
        self.long_window_forward_length = int(args.long_window_forward_length)
        self.short_window_horizon = int(args.short_window_horizon)
        self.enable_long_window = int(args.mas_long_conv_output_dim) > 0
        self.enable_short_window = self.short_window_horizon > 0
        self.loss_mode = args.loss_mode
        self.loss_mask_area_weight = float(args.loss_mask_area_weight)
        self.mas_long_window_dim = self.long_window_horizon * MAS_STEP_DIM
        self.mas_short_window_dim = self.short_window_horizon * MAS_STEP_DIM
        self.mas_long_encode_mode = args.mas_long_encode_mode
        assert (len(env.single_observation_space["state"].shape) == 2)  # (obs_horizon, obs_dim)
        assert len(env.single_action_space.shape) == 1  # (act_dim, )
        self.act_dim = env.single_action_space.shape[0]
        obs_state_dim = env.single_observation_space["state"].shape[1]
        self.mas_step_dim = MAS_STEP_DIM
        if self.mas_long_window_dim % self.mas_step_dim != 0:
            raise ValueError(
                "mas_long_window_dim "
                f"({self.mas_long_window_dim}) must be divisible by mas_step_dim ({self.mas_step_dim})"
            )
        self.include_depth = policy_uses_depth(args.obs_mode)
        obs_space_keys = set(env.single_observation_space.spaces.keys())
        if "rgb" not in obs_space_keys:
            raise ValueError("MAS-window policy requires rgb observations.")
        total_visual_channels = env.single_observation_space["rgb"].shape[-1]
        if self.include_depth:
            if "depth" not in obs_space_keys:
                raise ValueError(
                    "obs_mode='rgb+depth' requires depth observations from the eval env."
                )
            total_visual_channels += env.single_observation_space["depth"].shape[-1]

        visual_feature_dim = 256
        self.visual_encoder = PlainConv(in_channels=total_visual_channels, out_dim=visual_feature_dim, pool_feature_map=True)

        mas_long_feature_dim = max(0, int(args.mas_long_conv_output_dim))
        if self.enable_long_window:
            if self.mas_long_encode_mode == "2DConv":
                self.mas_long_window_encoder = MasConv(
                    in_channels=2, out_dim=mas_long_feature_dim
                )
            elif self.mas_long_encode_mode == "1DConv":
                self.mas_long_window_encoder = MasConv1D(
                    in_channels=2, mas_dim=self.mas_step_dim, out_dim=mas_long_feature_dim
                )
            else:
                raise ValueError(
                    f"Unsupported mas_long_encode_mode: {self.mas_long_encode_mode}"
                )
        else:
            self.mas_long_window_encoder = None
        mas_short_feature_dim = self.mas_short_window_dim if self.enable_short_window else 0

        self.noise_pred_net = DiTNoiseNet(
            ac_dim=self.act_dim,
            ac_chunk=self.pred_horizon,
            obs_dim=visual_feature_dim + obs_state_dim + mas_long_feature_dim + mas_short_feature_dim,
            time_dim=args.diffusion_step_embed_dim,
            use_mask=False,
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
        self.register_buffer(
            "state_norm_min",
            torch.zeros(obs_state_dim, dtype=torch.float32),
        )
        self.register_buffer(
            "state_norm_max",
            torch.zeros(obs_state_dim, dtype=torch.float32),
        )
        self.register_buffer(
            "state_norm_scale",
            torch.ones(obs_state_dim, dtype=torch.float32),
        )
        self.register_buffer(
            "has_state_normalizer",
            torch.tensor(False, dtype=torch.bool),
        )
        self.expected_obs_cond_dim = (
            visual_feature_dim + obs_state_dim + mas_long_feature_dim + mas_short_feature_dim
        )
        self._printed_obs_cond_check = False
        self._printed_action_denorm_check = False

    def _reshape_mas_window_value_and_mask(
        self,
        mas_window: torch.Tensor,
        mas_window_mask: torch.Tensor,
        window_horizon: int,
        expected_dim: int,
        state_dtype: torch.dtype,
    ):
        if mas_window.shape[-1] != expected_dim:
            raise ValueError(
                f"Expected mas window last dim {expected_dim}, got shape {tuple(mas_window.shape)}"
            )
        if mas_window_mask.shape != mas_window.shape:
            raise ValueError(
                f"Expected mas_window_mask shape {tuple(mas_window.shape)}, got {tuple(mas_window_mask.shape)}"
            )
        mas_value = mas_window.to(dtype=state_dtype)
        mas_mask = mas_window_mask.to(dtype=state_dtype)
        mas_value = mas_value.reshape(
            mas_window.shape[0], self.obs_horizon, window_horizon, self.mas_step_dim
        )
        mas_mask = mas_mask.reshape(
            mas_window.shape[0], self.obs_horizon, window_horizon, self.mas_step_dim
        )
        raw_mas_window = mas_window.reshape(
            mas_window.shape[0], self.obs_horizon, window_horizon, self.mas_step_dim
        )
        mas_value[..., :-1] = mas_value[..., :-1] * mas_mask[..., :-1]
        mas_mask[..., -1] = raw_mas_window[..., -1].to(dtype=state_dtype)
        return mas_value, mas_mask

    def set_action_denormalizer(self, mins: np.ndarray, maxs: np.ndarray, device):
        self.action_denorm_dims = int(mins.shape[0])
        self.action_denorm_min = torch.as_tensor(mins, device=device, dtype=torch.float32)
        self.action_denorm_max = torch.as_tensor(maxs, device=device, dtype=torch.float32)

    def set_state_normalizer(self, mins, maxs, device):
        mins = torch.as_tensor(mins, device=device, dtype=torch.float32)
        maxs = torch.as_tensor(maxs, device=device, dtype=torch.float32)
        if mins.shape != maxs.shape or mins.ndim != 1 or mins.shape[0] == 0:
            raise ValueError(
                f"invalid state min/max shape: min={tuple(mins.shape)}, max={tuple(maxs.shape)}"
            )
        if mins.shape != self.state_norm_min.shape:
            raise ValueError(
                f"state normalizer dim mismatch: expected {tuple(self.state_norm_min.shape)}, "
                f"got min={tuple(mins.shape)}, max={tuple(maxs.shape)}"
            )
        scale = maxs - mins
        scale = torch.where(scale > 1e-6, scale, torch.ones_like(scale))
        self.state_norm_min.copy_(mins)
        self.state_norm_max.copy_(maxs)
        self.state_norm_scale.copy_(scale)
        self.has_state_normalizer.fill_(True)

    def normalize_state(self, state: torch.Tensor):
        if not bool(self.has_state_normalizer.item()):
            return state
        mins = self.state_norm_min.to(device=state.device, dtype=state.dtype)
        scale = self.state_norm_scale.to(device=state.device, dtype=state.dtype)
        return ((state - mins) / scale) * 2.0 - 1.0

    def maybe_normalize_state(self, obs_seq):
        state = obs_seq["state"]
        state_is_normalized = obs_seq.get("state_is_normalized", None)
        if state_is_normalized is None:
            return self.normalize_state(state)
        if torch.is_tensor(state_is_normalized):
            already_normalized = bool(state_is_normalized.all().item())
        else:
            already_normalized = bool(state_is_normalized)
        if already_normalized:
            return state
        return self.normalize_state(state)

    def encode_mas_long_window(self, raw_mas_long_window, raw_mas_long_window_mask, state: torch.Tensor):
        batch_size = state.shape[0]
        if not self.enable_long_window:
            return state.new_empty((batch_size, self.obs_horizon, 0))
        if raw_mas_long_window is None or raw_mas_long_window_mask is None:
            mas_long_window = torch.zeros(
                (batch_size, self.obs_horizon, self.mas_long_window_dim),
                device=state.device,
                dtype=state.dtype,
            )
            mas_long_window_mask = torch.zeros_like(mas_long_window)
            mas_long_window_is_missing = True
        else:
            mas_long_window = raw_mas_long_window.to(
                device=state.device, dtype=state.dtype
            )
            mas_long_window_mask = raw_mas_long_window_mask.to(
                device=state.device, dtype=state.dtype
            )
            mas_long_window_is_missing = False
        if mas_long_window_is_missing:
            mas_value = mas_long_window.reshape(
                batch_size, self.obs_horizon, self.long_window_horizon, self.mas_step_dim
            )
            mas_mask = mas_long_window_mask.reshape(
                batch_size, self.obs_horizon, self.long_window_horizon, self.mas_step_dim
            )
        else:
            mas_value, mas_mask = self._reshape_mas_window_value_and_mask(
                mas_long_window,
                mas_long_window_mask,
                window_horizon=self.long_window_horizon,
                expected_dim=self.mas_long_window_dim,
                state_dtype=state.dtype,
            )
        mas_value = mas_value.permute(0, 1, 3, 2)
        mas_mask = mas_mask.permute(0, 1, 3, 2)
        mas_and_mask = torch.stack((mas_value, mas_mask), dim=2)
        mas_and_mask = mas_and_mask.reshape(
            batch_size * self.obs_horizon, 2, self.mas_step_dim, self.long_window_horizon
        )
        return self.mas_long_window_encoder(mas_and_mask).reshape(
            batch_size, self.obs_horizon, -1
        )

    def encode_mas_short_window(self, raw_mas_short_window, raw_mas_short_window_mask, state: torch.Tensor):
        batch_size = state.shape[0]
        if not self.enable_short_window:
            return state.new_empty((batch_size, self.obs_horizon, 0))
        if raw_mas_short_window is None or raw_mas_short_window_mask is None:
            mas_short_window = torch.zeros(
                (batch_size, self.obs_horizon, self.mas_short_window_dim),
                device=state.device,
                dtype=state.dtype,
            )
            mas_short_window_mask = torch.zeros_like(mas_short_window)
        else:
            mas_short_window = raw_mas_short_window.to(
                device=state.device, dtype=state.dtype
            )
            mas_short_window_mask = raw_mas_short_window_mask.to(
                device=state.device, dtype=state.dtype
            )
        mas_value, mas_mask = self._reshape_mas_window_value_and_mask(
            mas_short_window,
            mas_short_window_mask,
            window_horizon=self.short_window_horizon,
            expected_dim=self.mas_short_window_dim,
            state_dtype=state.dtype,
        )
        del mas_mask
        return mas_value.reshape(batch_size, self.obs_horizon, -1)

    def obs_conditioning(self, obs_seq, eval_mode):
        rgb = obs_seq["rgb"].float() / 255.0
        img_seq = rgb
        if self.include_depth:
            depth = obs_seq["depth"].float() / 1024.0
            img_seq = torch.cat([rgb, depth], dim=2)
        batch_size = img_seq.shape[0]
        img_seq = img_seq.flatten(end_dim=1)
        if hasattr(self, "aug") and not eval_mode:
            img_seq = self.aug(img_seq)
        visual_feature = self.visual_encoder(img_seq)
        visual_feature = visual_feature.reshape(batch_size, self.obs_horizon, visual_feature.shape[1])

        mas_long_window_feature = obs_seq.get("mas_long_window_feature", None)
        if mas_long_window_feature is None:
            mas_long_window_feature = self.encode_mas_long_window(
                obs_seq.get("mas_long_window", None),
                obs_seq.get("mas_long_window_mask", None),
                obs_seq["state"],
            )
        else:
            mas_long_window_feature = mas_long_window_feature.to(
                device=obs_seq["state"].device, dtype=obs_seq["state"].dtype
            )
        mas_short_window_feature = obs_seq.get("mas_short_window_feature", None)
        if mas_short_window_feature is None:
            mas_short_window_feature = self.encode_mas_short_window(
                obs_seq.get("mas_short_window", None),
                obs_seq.get("mas_short_window_mask", None),
                obs_seq["state"],
            )
        else:
            mas_short_window_feature = mas_short_window_feature.to(
                device=obs_seq["state"].device, dtype=obs_seq["state"].dtype
            )
        state = self.maybe_normalize_state(obs_seq)

        feature = torch.cat(
            (visual_feature, state, mas_long_window_feature, mas_short_window_feature),
            dim=-1,
        )
        if feature.shape[-1] != self.expected_obs_cond_dim:
            raise ValueError(
                f"[check4] obs_conditioning dim mismatch: got {feature.shape[-1]}, "
                f"expected {self.expected_obs_cond_dim}"
            )
        if not self._printed_obs_cond_check:
            print(
                f"[check2] obs['mas_long_window'] batch shape before encoding="
                f"{tuple(obs_seq['mas_long_window'].shape) if obs_seq.get('mas_long_window', None) is not None else None}"
            )
            print(
                f"[check2] obs['mas_short_window'] batch shape before encoding="
                f"{tuple(obs_seq['mas_short_window'].shape) if obs_seq.get('mas_short_window', None) is not None else None}"
            )
            print(
                f"[check2] long_window_enabled={self.enable_long_window}, "
                f"short_window_enabled={self.enable_short_window}"
            )
            print(
                f"[check4] obs_conditioning output shape={tuple(feature.shape)}, "
                f"expected_last_dim={self.expected_obs_cond_dim}"
            )
            self._printed_obs_cond_check = True
        return feature

    def compute_loss(self, obs_seq, action_seq, action_mask=None):
        B = obs_seq["state"].shape[0]

        # observation as FiLM conditioning
        obs_cond = self.obs_conditioning(obs_seq, eval_mode=False)  # (B, obs_horizon, obs_dim)

        # sample noise to add to actions
        noise = torch.randn((B, self.pred_horizon, self.act_dim), device=device)

        # sample a diffusion iteration for each data point
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (B,), device=device).long()

        # add noise to the clean images(actions) according to the noise magnitude at each diffusion iteration
        # (this is the forward diffusion process)
        noisy_action_seq = self.noise_scheduler.add_noise(action_seq, noise, timesteps)

        # predict the noise residual
        noise_pred = self.noise_pred_net(noisy_action_seq, timesteps, obs_cond)

        return compute_mask_weighted_noise_mse(
            noise_pred=noise_pred,
            noise=noise,
            action_mask=action_mask,
            loss_mode=self.loss_mode,
            mask_area_weight=self.loss_mask_area_weight,
        )

    def get_action(self, obs_seq):
        # init scheduler
        # self.noise_scheduler.set_timesteps(self.num_diffusion_iters)
        # set_timesteps will change noise_scheduler.timesteps is only used in noise_scheduler.step()
        # noise_scheduler.step() is only called during inference
        # if we use DDPM, and inference_diffusion_steps == train_diffusion_steps, then we can skip this

        # process observation and compute conditioning
        B = obs_seq["state"].shape[0]
        with torch.no_grad():
            obs_seq = common.to_tensor(obs_seq, obs_seq["state"].device)
            obs_seq["rgb"] = obs_seq["rgb"].permute(0, 1, 4, 2, 3)
            if self.include_depth:
                obs_seq["depth"] = obs_seq["depth"].permute(0, 1, 4, 2, 3)

            obs_cond = self.obs_conditioning(obs_seq, eval_mode=True)  # (B, obs_horizon, obs_dim)

            # initialize action from Guassian noise
            noisy_action_seq = torch.randn(
                (B, self.pred_horizon, self.act_dim), device=obs_seq["state"].device
            )

            for k in self.noise_scheduler.timesteps:
                timesteps = torch.full(
                    (B,), k, dtype=torch.long, device=noisy_action_seq.device
                )
                noise_pred = self.noise_pred_net(noisy_action_seq, timesteps, obs_cond)

                # inverse diffusion step (remove noise)
                noisy_action_seq = self.noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=k,
                    sample=noisy_action_seq,
                ).prev_sample

        # only take act_horizon number of actions
        start = self.obs_horizon - 1
        end = start + self.act_horizon
        action_seq = noisy_action_seq[:, start:end]  # (B, act_horizon, act_dim)
        if (
            self.action_denorm_dims <= 0
            or self.action_denorm_min is None
            or self.action_denorm_max is None
        ):
            raise RuntimeError(
                "Action denormalizer must be configured before get_action() is used "
                "for mas-window evaluation."
            )
        d = min(self.action_denorm_dims, action_seq.shape[-1])
        mins = self.action_denorm_min[:d]
        maxs = self.action_denorm_max[:d]
        action_seq = action_seq.clone()
        action_seq[..., :d] = mins + 0.5 * (action_seq[..., :d] + 1.0) * (maxs - mins)
        if not self._printed_action_denorm_check:
            print(
                f"[check7] action denormalization applied: denorm_dims={d}, "
                f"output_shape={tuple(action_seq.shape)}, "
                f"min={float(action_seq[..., :d].min().item()):.6f}, "
                f"max={float(action_seq[..., :d].max().item()):.6f}"
            )
            self._printed_action_denorm_check = True
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


def summarize_label_counts(labels):
    counts = defaultdict(int)
    for label in labels:
        counts[str(label)] += 1
    return dict(sorted(counts.items(), key=lambda item: item[0]))


def _decode_meta_string(value):
    value = np.asarray(value)
    if value.shape == ():
        value = value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def load_target_mask_composition_entries(data_path: str) -> list[tuple[str, float]] | None:
    meta = load_dataset_meta(data_path)
    mixed_enabled = _meta_flag(meta, "mixed_mask_enabled", False)
    if not mixed_enabled:
        return None
    if "mask_slot_name_list_json" in meta and "mask_slot_ratio_list_json" in meta:
        labels = json.loads(_decode_meta_string(meta["mask_slot_name_list_json"]))
        ratios = json.loads(_decode_meta_string(meta["mask_slot_ratio_list_json"]))
        if len(labels) != len(ratios):
            raise ValueError(
                f"mask_slot_name_list_json/mask_slot_ratio_list_json length mismatch in {data_path}"
            )
        return [(str(label), float(ratio)) for label, ratio in zip(labels, ratios)]
    if "mask_type_list_json" in meta and "mask_composition_list_json" in meta:
        labels = json.loads(_decode_meta_string(meta["mask_type_list_json"]))
        ratios = json.loads(_decode_meta_string(meta["mask_composition_list_json"]))
        if len(labels) != len(ratios):
            raise ValueError(
                f"mask_type_list_json/mask_composition_list_json length mismatch in {data_path}"
            )
        return [(str(label), float(ratio)) for label, ratio in zip(labels, ratios)]
    return None


def load_mask_assign_mode(data_path: str) -> str:
    meta = load_dataset_meta(data_path)
    if "mask_assign_mode" not in meta:
        return "composition"
    mode = _decode_meta_string(meta["mask_assign_mode"])
    if mode not in {"composition", "one_demo_multi_mask"}:
        raise ValueError(f"unsupported mask_assign_mode={mode!r} in {data_path}")
    return mode


def largest_remainder_counts(total_items: int, ratios: list[float]) -> list[int]:
    if total_items < 0:
        raise ValueError(f"total_items must be non-negative, got {total_items}")
    if len(ratios) == 0:
        raise ValueError("ratios must be non-empty")
    if total_items == 0:
        return [0 for _ in ratios]
    raw = np.asarray(ratios, dtype=np.float64) * float(total_items)
    counts = np.floor(raw).astype(np.int64)
    remaining = int(total_items - int(counts.sum()))
    if remaining > 0:
        remainders = raw - counts.astype(np.float64)
        order = sorted(range(len(ratios)), key=lambda idx: (-remainders[idx], idx))
        for idx in order[:remaining]:
            counts[idx] += 1
    return [int(v) for v in counts.tolist()]


def select_source_demo_indices(
    source_episode_ids: list[int],
    num_source_demos: int | None,
    seed: int,
) -> list[int]:
    if len(source_episode_ids) == 0:
        return []

    source_to_indices = defaultdict(list)
    ordered_source_ids = []
    for local_idx, source_episode_id in enumerate(source_episode_ids):
        source_episode_id = int(source_episode_id)
        if source_episode_id not in source_to_indices:
            ordered_source_ids.append(source_episode_id)
        source_to_indices[source_episode_id].append(int(local_idx))

    if num_source_demos is None or num_source_demos >= len(ordered_source_ids):
        selected_source_ids = set(ordered_source_ids)
    elif num_source_demos <= 0:
        selected_source_ids = set()
    else:
        rng = np.random.default_rng(seed)
        shuffled_source_ids = list(ordered_source_ids)
        rng.shuffle(shuffled_source_ids)
        selected_source_ids = set(shuffled_source_ids[:num_source_demos])

    selected_indices = []
    for source_episode_id in ordered_source_ids:
        if source_episode_id in selected_source_ids:
            selected_indices.extend(source_to_indices[source_episode_id])
    return sorted(selected_indices)


def select_demo_indices_by_mask_assign_mode(
    labels: list[str],
    source_episode_ids: list[int],
    num_select: int | None,
    seed: int,
    target_composition_entries: list[tuple[str, float]] | None,
    mask_assign_mode: str,
) -> list[int]:
    if mask_assign_mode == "composition":
        return stratified_select_demo_indices(
            labels=labels,
            num_select=num_select,
            seed=seed,
            target_composition_entries=target_composition_entries,
        )
    if mask_assign_mode == "one_demo_multi_mask":
        return select_source_demo_indices(
            source_episode_ids=source_episode_ids,
            num_source_demos=num_select,
            seed=seed,
        )
    raise ValueError(f"unsupported mask_assign_mode={mask_assign_mode!r}")


def stratified_select_demo_indices(
    labels: list[str],
    num_select: int | None,
    seed: int,
    target_composition_entries: list[tuple[str, float]] | None = None,
) -> list[int]:
    total_items = len(labels)
    if num_select is None or num_select >= total_items:
        return list(range(total_items))
    if num_select <= 0:
        return []

    grouped_indices = defaultdict(list)
    for idx, label in enumerate(labels):
        grouped_indices[str(label)].append(int(idx))

    if target_composition_entries is None:
        ordered_labels = sorted(grouped_indices.keys())
        target_items = [
            (label, len(grouped_indices[label]) / float(total_items))
            for label in ordered_labels
        ]
    else:
        target_items = []
        for label, composition in target_composition_entries:
            label = str(label)
            if label not in grouped_indices:
                raise ValueError(
                    f"label={label!r} from meta composition not found in dataset selection"
                )
            target_items.append((label, float(composition)))

    target_counts = largest_remainder_counts(
        num_select, [ratio for _, ratio in target_items]
    )
    rng = np.random.default_rng(seed)
    shuffled_grouped_indices = {}
    for label, candidates in grouped_indices.items():
        shuffled_candidates = list(candidates)
        rng.shuffle(shuffled_candidates)
        shuffled_grouped_indices[label] = shuffled_candidates

    selected = []
    for (label, _), target_count in zip(target_items, target_counts):
        candidates = shuffled_grouped_indices[label]
        if target_count > len(candidates):
            raise ValueError(
                f"Cannot sample {target_count} demos for label={label!r}; "
                f"only {len(candidates)} available"
            )
        selected.extend(candidates[:target_count])
        del candidates[:target_count]

    if len(selected) != num_select:
        raise AssertionError(
            f"stratified selection size mismatch: expected {num_select}, got {len(selected)}"
        )
    return sorted(selected)


if __name__ == "__main__":
    args = tyro.cli(Args)

    # ------------------------------------------------------------------------ #
    # 1. Run configuration and global random seeds
    # ------------------------------------------------------------------------ #
    if args.exp_name is None:
        args.exp_name = os.path.basename(__file__)[: -len(".py")]
        run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    else:
        run_name = args.exp_name
    video_dir = f"runs/{run_name}/videos"

    assert args.obs_horizon + args.act_horizon - 1 <= args.pred_horizon
    assert args.obs_horizon >= 1 and args.act_horizon >= 1 and args.pred_horizon >= 1
    if not 0.0 <= float(args.loss_mask_area_weight) <= 1.0:
        raise ValueError(
            f"loss_mask_area_weight must be in [0, 1], got {args.loss_mask_area_weight}"
        )
    if args.long_window_backward_length is None:
        args.long_window_backward_length = 0
    args.long_window_backward_length = int(args.long_window_backward_length)
    if args.long_window_forward_length is None:
        args.long_window_forward_length = args.pred_horizon
    args.long_window_forward_length = int(args.long_window_forward_length)
    if args.long_window_backward_length < 0:
        raise ValueError(
            "long_window_backward_length must be non-negative, "
            f"got {args.long_window_backward_length}"
        )
    if args.long_window_forward_length < 0:
        raise ValueError(
            "long_window_forward_length must be non-negative, "
            f"got {args.long_window_forward_length}"
        )
    args.long_window_horizon = (
        args.long_window_backward_length + args.long_window_forward_length
    )
    if int(args.mas_long_conv_output_dim) > 0 and args.long_window_horizon <= 0:
        raise ValueError(
            "long_window_backward_length + long_window_forward_length must be positive "
            "when mas_long_conv_output_dim > 0"
        )
    if int(args.mas_long_conv_output_dim) > 0 and args.long_window_forward_length <= 0:
        raise ValueError(
            "long_window_forward_length must be positive when long MAS window is enabled"
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(
        "[runtime] "
        f"torch_device={device}, "
        f"cuda_enabled={args.cuda}, "
        f"cuda_available={torch.cuda.is_available()}, "
        f"eval_sim_backend={args.sim_backend}, "
        f"num_eval_envs={args.num_eval_envs}"
    )

    # ------------------------------------------------------------------------ #
    # 2. Evaluation demo alignment: demo path, reset seeds, traj ids, subsets
    # ------------------------------------------------------------------------ #
    if args.test_demo_path is None:
        raise ValueError(
            "train_mas_window_mixed.py requires --test-demo-path. "
            "Evaluation must use an explicit eval demo dataset and cannot reuse demo_path."
        )
    eval_demo_path = args.test_demo_path
    eval_reset_seeds = infer_eval_reset_seeds_from_demo(
        eval_demo_path,
        num_traj=None,
        metadata_path=args.eval_demo_metadata_path,
    )
    eval_traj_ids = infer_eval_traj_ids_from_demo(
        eval_demo_path, num_traj=None
    )
    eval_mask_types_all = load_traj_mask_types(
        eval_demo_path,
        num_traj=None,
    )
    eval_mask_type_slots_all = load_traj_mask_type_slots(
        eval_demo_path,
        num_traj=None,
    )
    eval_source_episode_ids_all = load_source_episode_ids(
        eval_demo_path,
        num_traj=None,
    )
    eval_mask_assign_mode = load_mask_assign_mode(eval_demo_path)
    eval_target_composition_entries = load_target_mask_composition_entries(eval_demo_path)
    if len(eval_reset_seeds) == 0:
        raise ValueError(
            "Failed to infer evaluation reset seeds from eval demo metadata. "
            "mas-window evaluation requires demo-aligned reset seeds to keep MAS conditioning aligned."
        )
    print(f"[seed-infer] auto inferred {len(eval_reset_seeds)} eval seeds")
    total_eval_demos = len(eval_traj_ids) if len(eval_traj_ids) > 0 else len(eval_reset_seeds)
    eval_selection_labels = (
        eval_mask_type_slots_all
        if len(eval_mask_type_slots_all) == len(eval_mask_types_all)
        else eval_mask_types_all
    )
    eval_num_select = (
        min(total_eval_demos, args.num_eval_demos)
        if eval_mask_assign_mode == "composition"
        else args.num_eval_demos
    )
    eval_demo_indices = select_demo_indices_by_mask_assign_mode(
        labels=eval_selection_labels[:total_eval_demos],
        source_episode_ids=eval_source_episode_ids_all[:total_eval_demos],
        num_select=eval_num_select,
        seed=args.seed,
        target_composition_entries=(
            eval_target_composition_entries
            if eval_mask_assign_mode == "composition"
            else None
        ),
        mask_assign_mode=eval_mask_assign_mode,
    )
    if len(eval_reset_seeds) > 0:
        eval_reset_seeds = [
            eval_reset_seeds[i] for i in eval_demo_indices if i < len(eval_reset_seeds)
        ]
        args.num_eval_episodes = len(eval_reset_seeds)
    if len(eval_traj_ids) > 0:
        eval_traj_ids = [
            eval_traj_ids[i] for i in eval_demo_indices if i < len(eval_traj_ids)
        ]
    if len(eval_reset_seeds) > 0:
        assert len(eval_traj_ids) == len(eval_reset_seeds), (
            "failed to align eval traj ids with eval seeds for video naming"
        )
    validate_eval_video_config(
        num_eval_episodes=args.num_eval_episodes,
        num_eval_envs=args.num_eval_envs,
        capture_video_freq=args.capture_video_freq,
    )
    print(
        f"[mas-window] num_demos={args.num_demos}, demo_path={args.demo_path}, "
        f"test_demo_path={args.test_demo_path}, eval_demo_path={eval_demo_path}, "
        f"eval_demo_metadata_path={args.eval_demo_metadata_path}, "
        f"eval_mask_assign_mode={eval_mask_assign_mode}, "
        f"eval_demo_indices={eval_demo_indices}, "
        f"eval_reset_seeds={eval_reset_seeds if len(eval_reset_seeds) > 0 else None}, "
        f"eval_traj_ids={eval_traj_ids if len(eval_traj_ids) > 0 else None}, "
        f"inpainting={args.inpainting}"
    )

    # ------------------------------------------------------------------------ #
    # 3. Evaluation-only runtime requirements: action denormalization and STPM
    # ------------------------------------------------------------------------ #
    denorm_mins, denorm_maxs = load_action_denorm_stats(args.action_norm_path)
    print("[denorm] eval actions will be denormalized before env.step().")

    stpm_encoder, stpm_n_obs_steps, stpm_frame_gap = build_eval_stpm_encoder(
        args.stpm_ckpt_path, args.stpm_config_path, device
    )

    # ------------------------------------------------------------------------ #
    # 4. Build evaluation environment and experiment logging
    # ------------------------------------------------------------------------ #
    env_kwargs = dict(
        control_mode=args.control_mode,
        reward_mode="sparse",
        obs_mode=stpm_eval_env_obs_mode(args.obs_mode),
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
        video_dir=video_dir if args.capture_video else None,
        wrappers=[FlattenRGBDObservationWrapper],
    )
    validate_only_mas_eval_layout(envs, stpm_encoder)

    if args.track:
        import wandb

        config = vars(args)
        config["eval_env_cfg"] = dict(
            **env_kwargs,
            num_envs=args.num_eval_envs,
            env_id=args.env_id,
            env_horizon=args.max_episode_steps,
        )
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

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # ------------------------------------------------------------------------ #
    # 5. Resolve observation preprocessing from a temporary single env
    # ------------------------------------------------------------------------ #
    tmp_env_kwargs = dict(env_kwargs, obs_mode=args.obs_mode)
    tmp_env = gym.make(args.env_id, **tmp_env_kwargs)
    orignal_obs_space = tmp_env.observation_space
    tmp_env.close()

    obs_process_fn = partial(
        convert_obs,
        concat_fn=partial(np.concatenate, axis=-1),  # merge camera outputs
        transpose_fn=partial(np.transpose, axes=(0, 3, 1, 2)),  # (B, H, W, C) -> (B, C, H, W)
        state_obs_extractor=build_state_obs_extractor(args.env_id),
        depth=policy_uses_depth(args.obs_mode),
    )

    # ------------------------------------------------------------------------ #
    # 6. Load training dataset and dataloader
    # ------------------------------------------------------------------------ #
    train_mask_types_all = load_traj_mask_types(
        data_path=args.demo_path,
        num_traj=None,
    )
    train_mask_type_slots_all = load_traj_mask_type_slots(
        data_path=args.demo_path,
        num_traj=None,
    )
    train_source_episode_ids_all = load_source_episode_ids(
        data_path=args.demo_path,
        num_traj=None,
    )
    train_mask_assign_mode = load_mask_assign_mode(args.demo_path)
    train_target_composition_entries = load_target_mask_composition_entries(args.demo_path)
    train_selection_labels = (
        train_mask_type_slots_all
        if len(train_mask_type_slots_all) == len(train_mask_types_all)
        else train_mask_types_all
    )
    train_demo_indices = select_demo_indices_by_mask_assign_mode(
        labels=train_selection_labels,
        source_episode_ids=train_source_episode_ids_all,
        num_select=args.num_demos,
        seed=args.seed,
        target_composition_entries=(
            train_target_composition_entries
            if train_mask_assign_mode == "composition"
            else None
        ),
        mask_assign_mode=train_mask_assign_mode,
    )
    selected_train_mask_types = [train_mask_types_all[i] for i in train_demo_indices]
    selected_train_mask_type_slots = [train_mask_type_slots_all[i] for i in train_demo_indices]
    dataset = SmallDemoDataset_MasWindowDiffusionPolicy(
        data_path=args.demo_path,
        obs_process_fn=obs_process_fn,
        obs_space=orignal_obs_space,
        device=device,
        num_traj=None,
        traj_indices=train_demo_indices,
    )

    dataset.debug_print_sample(0)

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

    # ------------------------------------------------------------------------ #
    # 7. Load evaluation-only MAS conditioning data
    # ------------------------------------------------------------------------ #
    eval_mam_data = load_ce_eval_data(
        data_path=eval_demo_path,
        device=device,
        num_traj=None,
    )
    if len(eval_demo_indices) > 0:
        eval_mam_data = subset_eval_data(eval_mam_data, eval_demo_indices)
    eval_mask_type_counts = summarize_label_counts(eval_mam_data.get("mask_types", []))
    eval_mask_slot_counts = summarize_label_counts(eval_mam_data.get("mask_type_slots", []))
    train_mask_type_counts = summarize_label_counts(selected_train_mask_types)
    train_mask_slot_counts = summarize_label_counts(selected_train_mask_type_slots)
    train_total_mask_type_counts = summarize_label_counts(train_mask_types_all)
    train_total_mask_slot_counts = summarize_label_counts(train_mask_type_slots_all)
    print(f"[mixed-train] train mask type counts: {train_mask_type_counts}")
    print(f"[mixed-train] train mask slot counts: {train_mask_slot_counts}")
    print(f"[mixed-train] train total mask type counts: {train_total_mask_type_counts}")
    print(f"[mixed-train] train total mask slot counts: {train_total_mask_slot_counts}")
    print(f"[mixed-train] eval mask type counts: {eval_mask_type_counts}")
    print(f"[mixed-train] eval mask slot counts: {eval_mask_slot_counts}")
    print(
        f"[mixed-train] mask_assign_mode: train={train_mask_assign_mode}, "
        f"eval={eval_mask_assign_mode}, selected_train={len(train_demo_indices)}, "
        f"selected_eval={len(eval_demo_indices)}"
    )

    # ------------------------------------------------------------------------ #
    # 8. Build agent, EMA agent, optimizer, and LR scheduler
    # ------------------------------------------------------------------------ #
    agent = Agent(envs, args).to(device)
    agent.set_action_denormalizer(denorm_mins, denorm_maxs, device)
    if dataset.state_min is not None and dataset.state_max is not None:
        agent.set_state_normalizer(dataset.state_min, dataset.state_max, device)

    optimizer = optim.AdamW(
        params=agent.parameters(), lr=args.lr, betas=(0.95, 0.999), weight_decay=1e-6
    )

    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=args.total_iters,
    )

    ema = EMAModel(parameters=agent.parameters(), power=0.75)
    ema_agent = Agent(envs, args).to(device)
    ema_agent.set_action_denormalizer(denorm_mins, denorm_maxs, device)
    if dataset.state_min is not None and dataset.state_max is not None:
        ema_agent.set_state_normalizer(dataset.state_min, dataset.state_max, device)

    # ------------------------------------------------------------------------ #
    # 9. Tracking buffers and helper functions for eval/logging/checkpointing
    # ------------------------------------------------------------------------ #
    best_eval_metrics = defaultdict(float)
    timings = defaultdict(float)
    ce_curve_iters = []
    ce_curve_all = []
    ce_curve_success = []
    ce_curve_failed = []
    capture_indices = (
        build_capture_indices(args.num_eval_episodes, args.capture_video_freq)
        if args.capture_video
        else set()
    )

    def evaluate_and_save_best(iteration):
        if iteration % args.eval_freq == 0:
            last_tick = time.time()
            ema.copy_to(ema_agent.parameters())
            ce_summary = None
            per_mask_type_summary = {}
            per_mask_slot_summary = {}
            if args.capture_video:
                clear_iteration_artifacts(video_dir=video_dir, iteration=iteration)
            mixed_eval_result = evaluate_mas_window_mixed(
                args.num_eval_episodes,
                ema_agent,
                envs,
                device,
                args.sim_backend,
                eval_mas_window_data=eval_mam_data,
                obs_horizon=args.obs_horizon,
                long_window_horizon=(
                    args.long_window_horizon if args.mas_long_conv_output_dim > 0 else 0
                ),
                long_window_backward_length=(
                    args.long_window_backward_length if args.mas_long_conv_output_dim > 0 else 0
                ),
                long_window_forward_length=(
                    args.long_window_forward_length if args.mas_long_conv_output_dim > 0 else 0
                ),
                short_window_horizon=args.short_window_horizon,
                stpm_encoder=stpm_encoder,
                stpm_n_obs_steps=stpm_n_obs_steps,
                stpm_frame_gap=stpm_frame_gap,
                progress_bar=args.eval_progress_bar,
                reset_seeds=eval_reset_seeds if len(eval_reset_seeds) > 0 else None,
                return_progress_curves=args.capture_video,
                return_rollout_records=True,
                capture_indices=capture_indices if args.capture_video else None,
                video_dir=video_dir if args.capture_video else None,
                iteration=iteration,
                eval_traj_ids=eval_traj_ids if len(eval_traj_ids) > 0 else None,
                inpainting=args.inpainting,
            )
            if args.capture_video:
                (
                    eval_metrics,
                    per_mask_type_summary,
                    per_mask_slot_summary,
                    progress_curve_records,
                    rollout_records,
                ) = mixed_eval_result
            else:
                (
                    eval_metrics,
                    per_mask_type_summary,
                    per_mask_slot_summary,
                    rollout_records,
                ) = mixed_eval_result
                progress_curve_records = None
            ce_summary = aggregate_control_error(
                compute_control_error_results_from_rollouts(
                    eval_data=eval_mam_data,
                    rollout_records=rollout_records,
                    action_min=denorm_mins,
                    action_max=denorm_maxs,
                    save_per_traj=False,
                )
            )
            timings["eval"] += time.time() - last_tick

            for k, v in eval_metrics.items():
                metric_value = float(np.mean(v)) if isinstance(v, np.ndarray) else float(v)
                writer.add_scalar(f"eval/{k}", metric_value, iteration)
                print(f"{k}: {metric_value:.6f}")
            for mask_type, one_summary in per_mask_type_summary.items():
                for key, value in one_summary.items():
                    metric_value = float(value)
                    writer.add_scalar(f"eval_by_mask_type/{mask_type}/{key}", metric_value, iteration)
                    print(f"[{mask_type}] {key}: {metric_value:.6f}")
            for mask_slot, one_summary in per_mask_slot_summary.items():
                for key, value in one_summary.items():
                    metric_value = float(value)
                    writer.add_scalar(f"eval_by_mask_slot/{mask_slot}/{key}", metric_value, iteration)
                    print(f"[{mask_slot}] {key}: {metric_value:.6f}")
            if ce_summary is not None:
                for k in ["ce_all", "ce_success", "ce_failed"]:
                    metric_value = float(ce_summary[k])
                    writer.add_scalar(f"eval/{k}", metric_value, iteration)
                    print(f"{k}: {metric_value:.6f}")
                ce_curve_iters.append(int(iteration))
                ce_curve_all.append(float(ce_summary["ce_all"]))
                ce_curve_success.append(float(ce_summary["ce_success"]))
                ce_curve_failed.append(float(ce_summary["ce_failed"]))
                save_control_error_curve(
                    run_dir=writer.log_dir,
                    iterations=ce_curve_iters,
                    ce_all=ce_curve_all,
                    ce_success=ce_curve_success,
                    ce_failed=ce_curve_failed,
                )
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

        # move batch to target device
        data_batch = move_batch_to_device(data_batch, device)

        # forward and compute loss
        last_tick = time.time()
        total_loss = agent.compute_loss(
            obs_seq=data_batch["observations"],  # obs_batch_dict['state'] is (B, L, obs_dim)
            action_seq=data_batch["actions"],  # (B, L, act_dim)
            action_mask=data_batch["action_mask"],  # (B, L, act_dim)
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
