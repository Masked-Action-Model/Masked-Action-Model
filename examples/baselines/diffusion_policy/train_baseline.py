ALGO_NAME = "BC_Diffusion_rgbd_Baseline"

import os
import random
import time
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, List, Literal, Optional

import gymnasium as gym
import h5py
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
from gymnasium import spaces
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import BatchSampler, RandomSampler
from torch.utils.tensorboard import SummaryWriter

# Allow running this file directly via `python .../train_baseline.py`.
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
BASELINE_ROOT = Path(__file__).resolve().parent
if str(BASELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(BASELINE_ROOT))

from data_preprocess.utils.normalize_utils import (
    compute_global_min_max,
    load_action_stats_from_path,
    normalize_selected_dims,
)
from evaluate.evaluate_baseline import evaluate
from models.conditional_unet1d import ConditionalUnet1D
from models.modeling_ditdp import DiTNoiseNet
from models.plain_conv import PlainConv
from utils.load_train_data_utils import load_demo_dataset
from utils.make_env import make_eval_envs
from utils.utils import (IterationBasedBatchSampler, build_state_obs_extractor,
                         convert_obs, worker_init_fn)


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
    """train split h5 path. If raw_demo_h5 is set, this is overwritten by the generated train split."""
    eval_demo_path: Optional[str] = None
    """eval split h5 path. If raw_demo_h5 is set, this is overwritten by the generated eval split."""
    eval_demo_metadata_path: Optional[str] = None
    """optional path to eval demo metadata json. If omitted, inferred from eval_demo_path"""
    raw_demo_h5: Optional[str] = None
    """optional raw demo h5. When set, train_baseline.py creates a 5:1 train/eval split without normalization or mask."""
    raw_demo_json: Optional[str] = None
    """optional raw demo json. Defaults to raw_demo_h5 with .json suffix."""
    split_output_dir: Optional[str] = None
    """directory for generated train/eval split files."""
    split_output_prefix: Optional[str] = None
    """filename prefix for generated split files."""
    split_seed: int = 0
    """random seed used for 5:1 train/eval split."""
    split_num_traj: Optional[int] = None
    """optional number of raw trajectories to include before splitting."""
    overwrite_split: bool = False
    """whether to overwrite existing generated split files."""
    num_demos: Optional[int] = None
    """number of trajectories to load from the demo dataset"""
    num_eval_demos: Optional[int] = None
    """number of eval trajectories to load from eval_demo_path. If omitted, all eval trajectories are used"""
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
    noise_model: Literal["Transformer", "Unet"] = "Transformer"
    """denoiser backbone. Transformer uses DiTNoiseNet; Unet uses ConditionalUnet1D."""
    unet_dims: List[int] = field(default_factory=lambda: [64, 128, 256])
    """UNet channel dimensions, used only when noise_model=Unet."""
    n_groups: int = 8
    """GroupNorm groups for UNet, used only when noise_model=Unet."""


    # Environment/experiment specific arguments
    obs_mode: str = "rgb+depth"
    """The observation mode to use for the environment, which dictates what visual inputs to pass to the model. Can be "rgb", "depth", or "rgb+depth"."""
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
    """Optional action normalization stats. If omitted, stats are computed from the train split and saved beside it."""
    sim_backend: str = "physx_cpu"
    """the simulation backend to use for evaluation environments. can be "cpu" or "gpu"""
    num_dataload_workers: int = 0
    """the number of workers to use for loading the training data in the torch dataloader"""
    control_mode: str = "pd_ee_pose"
    """fixed control mode"""

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


def _list_traj_keys(h5_file: h5py.File, num_traj: Optional[int] = None) -> list[str]:
    keys = sorted(
        [key for key in h5_file.keys() if key.startswith("traj_")],
        key=lambda x: int(x.split("_")[-1]),
    )
    if num_traj is not None:
        if num_traj > len(keys):
            raise ValueError(f"split_num_traj={num_traj} > available trajectories={len(keys)}")
        keys = keys[:num_traj]
    if len(keys) == 0:
        raise ValueError("raw demo contains no traj_* groups")
    return keys


def _split_source_episode_ids(source_episode_ids: list[int], split_seed: int):
    if len(source_episode_ids) == 1:
        return source_episode_ids[:], []
    rng = np.random.default_rng(split_seed)
    perm = rng.permutation(np.asarray(source_episode_ids, dtype=np.int64))
    eval_count = max(1, len(source_episode_ids) // 6)
    eval_ids = sorted(int(x) for x in perm[:eval_count].tolist())
    train_ids = sorted(int(x) for x in perm[eval_count:].tolist())
    if len(train_ids) == 0:
        train_ids, eval_ids = eval_ids, []
    return train_ids, eval_ids


def _copy_non_traj_groups(src_file: h5py.File, dst_file: h5py.File) -> None:
    for key in src_file.keys():
        if not key.startswith("traj_"):
            src_file.copy(src_file[key], dst_file, key)


def _write_split_h5(raw_h5: str, output_h5: str, source_episode_ids: list[int]) -> None:
    os.makedirs(os.path.dirname(output_h5), exist_ok=True)
    with h5py.File(raw_h5, "r") as src_file, h5py.File(output_h5, "w") as dst_file:
        _copy_non_traj_groups(src_file, dst_file)
        for local_episode_id, source_episode_id in enumerate(source_episode_ids):
            src_key = f"traj_{source_episode_id}"
            if src_key not in src_file:
                raise KeyError(f"missing source trajectory {src_key} in {raw_h5}")
            dst_key = f"traj_{local_episode_id}"
            src_file.copy(src_file[src_key], dst_file, dst_key)
            if "source_episode_id" not in dst_file[dst_key]:
                dst_file[dst_key].create_dataset(
                    "source_episode_id", data=np.int32(source_episode_id)
                )


def _write_split_json(
    raw_json: str,
    output_json: str,
    split_name: str,
    source_episode_ids: list[int],
    raw_h5: str,
    split_seed: int,
) -> None:
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(raw_json, "r", encoding="utf-8") as f:
        source_metadata = json.load(f)
    episodes = source_metadata.get("episodes", [])
    output_metadata = {
        "env_info": source_metadata.get("env_info", {}),
        "commit_info": source_metadata.get("commit_info", {}),
        "episodes": [],
        "split_info": {
            "split": split_name,
            "source_h5": raw_h5,
            "source_json": raw_json,
            "source_episode_ids": [int(x) for x in source_episode_ids],
            "split_seed": int(split_seed),
            "split_rule": "rng permutation, eval_count=max(1, N//6), sorted within each split",
            "normalization": "none",
            "mask": "none",
        },
    }
    for local_episode_id, source_episode_id in enumerate(source_episode_ids):
        if source_episode_id >= len(episodes):
            raise IndexError(
                f"source_episode_id {source_episode_id} exceeds json episodes ({len(episodes)})"
            )
        episode = dict(episodes[source_episode_id])
        episode["episode_id"] = int(local_episode_id)
        episode["source_episode_id"] = int(source_episode_id)
        output_metadata["episodes"].append(episode)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(output_metadata, f, indent=2)


def ensure_raw_train_eval_split(args: Args) -> None:
    if args.raw_demo_h5 is None or len(args.raw_demo_h5.strip()) == 0:
        return
    raw_h5 = os.path.abspath(args.raw_demo_h5)
    if not os.path.exists(raw_h5):
        raise FileNotFoundError(f"raw_demo_h5 not found: {raw_h5}")
    raw_json = (
        os.path.abspath(args.raw_demo_json)
        if args.raw_demo_json is not None and len(args.raw_demo_json.strip()) > 0
        else os.path.splitext(raw_h5)[0] + ".json"
    )
    if not os.path.exists(raw_json):
        raise FileNotFoundError(f"raw_demo_json not found: {raw_json}")

    raw_stem = os.path.splitext(os.path.basename(raw_h5))[0]
    output_dir = (
        os.path.abspath(args.split_output_dir)
        if args.split_output_dir is not None and len(args.split_output_dir.strip()) > 0
        else os.path.join(os.path.dirname(raw_h5), f"{raw_stem}_baseline_split_s{args.split_seed}")
    )
    output_prefix = args.split_output_prefix or raw_stem
    output_stem = f"{output_prefix}_split-s{args.split_seed}"
    train_h5 = os.path.join(output_dir, f"{output_stem}_train.h5")
    train_json = os.path.join(output_dir, f"{output_stem}_train.json")
    eval_h5 = os.path.join(output_dir, f"{output_stem}_eval.h5")
    eval_json = os.path.join(output_dir, f"{output_stem}_eval.json")

    expected = [train_h5, train_json, eval_h5, eval_json]
    if all(os.path.exists(path) for path in expected) and not args.overwrite_split:
        print(f"[baseline-split] reuse existing split: {output_dir}")
    else:
        if not args.overwrite_split:
            existing = [path for path in expected if os.path.exists(path)]
            if existing:
                raise FileExistsError(
                    "partial split outputs exist; set --overwrite-split to replace them: "
                    + ", ".join(existing)
                )
        with h5py.File(raw_h5, "r") as raw_file:
            traj_keys = _list_traj_keys(raw_file, num_traj=args.split_num_traj)
        source_episode_ids = [int(key.split("_")[-1]) for key in traj_keys]
        train_ids, eval_ids = _split_source_episode_ids(
            source_episode_ids=source_episode_ids,
            split_seed=args.split_seed,
        )
        if len(eval_ids) == 0:
            raise ValueError("baseline split produced empty eval set; need at least 2 raw trajectories")
        print(
            f"[baseline-split] generating split into {output_dir}: "
            f"train={len(train_ids)}, eval={len(eval_ids)}, seed={args.split_seed}"
        )
        _write_split_h5(raw_h5, train_h5, train_ids)
        _write_split_h5(raw_h5, eval_h5, eval_ids)
        _write_split_json(raw_json, train_json, "train", train_ids, raw_h5, args.split_seed)
        _write_split_json(raw_json, eval_json, "eval", eval_ids, raw_h5, args.split_seed)

    args.demo_path = train_h5
    args.eval_demo_path = eval_h5
    args.eval_demo_metadata_path = eval_json
    print(f"[baseline-split] train demo: {args.demo_path}")
    print(f"[baseline-split] eval demo: {args.eval_demo_path}")


def load_action_denorm_stats(action_norm_path: str):
    if action_norm_path is None or len(action_norm_path.strip()) == 0:
        raise ValueError(
            "action_norm_path is required. Please pass --action-norm-path to provide min/max for denormalization."
        )
    mins, maxs = load_action_stats_from_path(action_norm_path)
    print(f"[denorm] loaded action norm stats from {action_norm_path}, dims={mins.shape[0]}")
    return mins, maxs


def save_action_norm_stats(action_norm_path: str, mins: np.ndarray, maxs: np.ndarray):
    os.makedirs(os.path.dirname(action_norm_path), exist_ok=True)
    with open(action_norm_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "min": np.asarray(mins, dtype=np.float32).tolist(),
                "max": np.asarray(maxs, dtype=np.float32).tolist(),
            },
            f,
            indent=2,
        )


def _candidate_metadata_paths(demo_path: str, explicit_meta_path: Optional[str] = None):
    candidates = []
    if explicit_meta_path is not None and len(explicit_meta_path.strip()) > 0:
        candidates.append(explicit_meta_path)
    if demo_path.endswith(".json"):
        candidates.append(demo_path)
    else:
        base, _ = os.path.splitext(demo_path)
        candidates.append(f"{base}.json")
        suffixes = [
            "_concat_train",
            "_concat_eval",
            "_concat",
            "_train",
            "_eval",
            "_masked",
            "_normed",
        ]
        for suffix in suffixes:
            if base.endswith(suffix):
                candidates.append(f"{base[:-len(suffix)]}.json")

    seen = set()
    out = []
    for path in candidates:
        if path not in seen:
            out.append(path)
            seen.add(path)
    return out


def infer_loaded_traj_ids_from_demo(demo_path: str, num_traj: Optional[int] = None):
    if demo_path.endswith(".json"):
        with open(demo_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        traj_ids = [
            int(episode.get("episode_id", i))
            for i, episode in enumerate(meta.get("episodes", []))
        ]
    else:
        if not os.path.exists(demo_path):
            raise FileNotFoundError(f"eval demo path not found: {demo_path}")
        with h5py.File(demo_path, "r") as f:
            traj_keys = [key for key in f.keys() if key.startswith("traj_")]
        if len(traj_keys) == 0:
            raise ValueError(f"no traj_* found in eval demo path: {demo_path}")
        traj_ids = sorted(int(key.split("_")[-1]) for key in traj_keys)

    if num_traj is not None:
        if num_traj > len(traj_ids):
            raise ValueError(
                f"num_eval_demos={num_traj} > available eval trajectories={len(traj_ids)}"
            )
        traj_ids = traj_ids[:num_traj]
    return traj_ids


def _episode_to_reset_kwargs(episode: dict[str, Any]) -> dict[str, Any]:
    reset_kwargs = dict(episode.get("reset_kwargs", {}) or {})
    reset_seed = reset_kwargs.get("seed", episode.get("episode_seed", None))
    if isinstance(reset_seed, list):
        reset_seed = reset_seed[0] if len(reset_seed) > 0 else None
    if reset_seed is not None:
        reset_kwargs["seed"] = int(reset_seed)
    return reset_kwargs


def load_eval_reset_kwargs_list(
    eval_demo_path: str,
    eval_demo_metadata_path: Optional[str] = None,
    num_traj: Optional[int] = None,
) -> list[dict[str, Any]]:
    traj_ids = infer_loaded_traj_ids_from_demo(eval_demo_path, num_traj=num_traj)
    for meta_path in _candidate_metadata_paths(eval_demo_path, eval_demo_metadata_path):
        if not os.path.exists(meta_path):
            continue
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            episodes = meta.get("episodes", [])
            episode_by_id = {
                int(episode.get("episode_id", i)): episode
                for i, episode in enumerate(episodes)
            }
            reset_kwargs_list = []
            for traj_id in traj_ids:
                episode = episode_by_id.get(traj_id, None)
                if episode is None and 0 <= traj_id < len(episodes):
                    episode = episodes[traj_id]
                if episode is None:
                    reset_kwargs_list = []
                    break
                reset_kwargs_list.append(_episode_to_reset_kwargs(episode))
            if len(reset_kwargs_list) == len(traj_ids):
                seeds = [kwargs.get("seed", None) for kwargs in reset_kwargs_list]
                print(
                    f"[eval-split] loaded {len(reset_kwargs_list)} eval reset kwargs "
                    f"from {meta_path}"
                )
                print(f"[eval-split] eval trajectory ids: {traj_ids[:20]}")
                print(f"[eval-split] eval seeds: {seeds[:20]}")
                return reset_kwargs_list
        except Exception as exc:
            print(f"[eval-split] failed reading {meta_path}: {exc}")

    raise ValueError(
        "failed to infer eval reset kwargs from eval demo metadata. "
        "Pass --eval-demo-metadata-path explicitly if the json is not next to eval_demo_path."
    )


class SmallDemoDataset_DiffusionPolicy(Dataset):  # Load everything into memory
    def __init__(
        self,
        data_path,
        obs_process_fn,
        obs_space,
        include_rgb,
        include_depth,
        obs_horizon,
        pred_horizon,
        device,
        num_traj,
        action_norm_path=None,
    ):
        self.include_rgb = include_rgb
        self.include_depth = include_depth
        self.action_min = None
        self.action_max = None
        self.action_norm_path = None

        trajectories = load_demo_dataset(data_path, num_traj=num_traj, concat=False)
        # trajectories['observations'] is a list of dict, each dict is a traj, with keys in obs_space, values with length L+1
        # trajectories['actions'] is a list of np.ndarray (L, act_dim)
        print("Raw trajectory loaded, beginning observation pre-processing...")

        # Pre-process the observations, make them align with the obs returned by the obs_wrapper
        obs_traj_dict_list = []
        for obs_traj_dict in trajectories["observations"]:
            _obs_traj_dict = reorder_keys(obs_traj_dict, obs_space)  # key order in demo is different from key order in env obs
            _obs_traj_dict = obs_process_fn(_obs_traj_dict)
            if self.include_depth:
                _obs_traj_dict["depth"] = torch.Tensor(_obs_traj_dict["depth"].astype(np.float32)).to(device=device, dtype=torch.float16)
            if self.include_rgb:
                _obs_traj_dict["rgb"] = torch.from_numpy(_obs_traj_dict["rgb"]).to(device)  # still uint8
            _obs_traj_dict["state"] = torch.from_numpy(_obs_traj_dict["state"]).to(device)
            obs_traj_dict_list.append(_obs_traj_dict)
        trajectories["observations"] = obs_traj_dict_list
        self.obs_keys = list(_obs_traj_dict.keys())
        raw_actions = [np.asarray(action, dtype=np.float32) for action in trajectories["actions"]]
        if action_norm_path is not None and len(action_norm_path.strip()) > 0:
            action_min, action_max = load_action_denorm_stats(action_norm_path)
            self.action_norm_path = action_norm_path
            print(f"[action_norm] using provided stats: {action_norm_path}")
        else:
            action_min, action_max = compute_global_min_max(raw_actions)
            self.action_norm_path = os.path.splitext(data_path)[0] + ".action_norm.json"
            save_action_norm_stats(self.action_norm_path, action_min, action_max)
            print(
                f"[action_norm] computed stats from dataset and saved to {self.action_norm_path}"
            )
        self.action_min = np.asarray(action_min, dtype=np.float32)
        self.action_max = np.asarray(action_max, dtype=np.float32)
        print(f"[action_norm] min={self.action_min}")
        print(f"[action_norm] max={self.action_max}")

        # Pre-process the actions into normalized space for training.
        for i, action in enumerate(raw_actions):
            normalized_action = normalize_selected_dims(
                action,
                mins=self.action_min,
                maxs=self.action_max,
            )
            trajectories["actions"][i] = torch.from_numpy(normalized_action).to(device=device)
        print("Obs/action pre-processing is done, start to pre-compute the slice indices...")

         # Fixed to pd_ee_pose: pad with final action to keep target unchanged.
        print("Using fixed control mode pd_ee_pose, padding with final action.")

        self.obs_horizon, self.pred_horizon = obs_horizon, pred_horizon
        
        self.slices = []
        num_traj = len(trajectories["actions"])
        total_transitions = 0
        for traj_idx in range(num_traj):
            L = trajectories["actions"][traj_idx].shape[0]
            assert trajectories["observations"][traj_idx]["state"].shape[0] == L + 1
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

        print(
            f"Total transitions: {total_transitions}, Total obs sequences: {len(self.slices)}"
        )

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
        
        assert (
            obs_seq["state"].shape[0] == self.obs_horizon
            and act_seq.shape[0] == self.pred_horizon
        )
        return {
            "observations": obs_seq,
            "actions": act_seq,
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
        self.noise_model = args.noise_model
        assert (len(env.single_observation_space["state"].shape) == 2)  # (obs_horizon, obs_dim)
        assert len(env.single_action_space.shape) == 1  # (act_dim, )

        self.act_dim = env.single_action_space.shape[0]
        obs_state_dim = env.single_observation_space["state"].shape[1]
        total_visual_channels = 0
        self.include_rgb = "rgb" in env.single_observation_space.keys()
        self.include_depth = "depth" in env.single_observation_space.keys()

        if self.include_rgb:
            total_visual_channels += env.single_observation_space["rgb"].shape[-1]
        if self.include_depth:
            total_visual_channels += env.single_observation_space["depth"].shape[-1]

        visual_feature_dim = 256
        self.visual_encoder = PlainConv(
            in_channels=total_visual_channels, out_dim=visual_feature_dim, pool_feature_map=True
        )
        if args.noise_model == "Transformer":
            self.noise_pred_net = DiTNoiseNet(
                ac_dim=self.act_dim,
                ac_chunk=self.pred_horizon,
                obs_dim=visual_feature_dim + obs_state_dim,
                time_dim=args.diffusion_step_embed_dim,
                use_mask=False,
            )
        elif args.noise_model == "Unet":
            self.noise_pred_net = ConditionalUnet1D(
                input_dim=self.act_dim,
                global_cond_dim=self.obs_horizon * (visual_feature_dim + obs_state_dim),
                diffusion_step_embed_dim=args.diffusion_step_embed_dim,
                down_dims=args.unet_dims,
                n_groups=args.n_groups,
            )
        else:
            raise ValueError(f"unsupported noise_model={args.noise_model!r}")
        self.num_diffusion_iters = 100
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.num_diffusion_iters,
            beta_schedule="squaredcos_cap_v2",  # has big impact on performance, try not to change
            clip_sample=True,  # clip output to [-1,1] to improve stability
            prediction_type="epsilon",  # predict noise (instead of denoised action)
        )
        self.register_buffer(
            "action_denorm_min", torch.empty(0, dtype=torch.float32), persistent=True
        )
        self.register_buffer(
            "action_denorm_max", torch.empty(0, dtype=torch.float32), persistent=True
        )

    def set_action_denormalizer(self, mins: np.ndarray, maxs: np.ndarray, device):
        self.action_denorm_min = torch.as_tensor(mins, device=device, dtype=torch.float32)
        self.action_denorm_max = torch.as_tensor(maxs, device=device, dtype=torch.float32)

    def encode_obs(self, obs_seq, eval_mode):
        if self.include_rgb:
            rgb = obs_seq["rgb"].float() / 255.0  # (B, obs_horizon, 3*k, H, W)
            img_seq = rgb
        if self.include_depth:
            depth = obs_seq["depth"].float() / 1024.0  # (B, obs_horizon, 1*k, H, W)
            img_seq = depth
        if self.include_rgb and self.include_depth:
            img_seq = torch.cat([rgb, depth], dim=2)  # (B, obs_horizon, C, H, W), C=4*k
        batch_size = img_seq.shape[0]
        img_seq = img_seq.flatten(end_dim=1)  # (B*obs_horizon, C, H, W)
        if hasattr(self, "aug") and not eval_mode:
            img_seq = self.aug(img_seq)  # (B*obs_horizon, C, H, W)
        visual_feature = self.visual_encoder(img_seq)  # (B*obs_horizon, D)
        visual_feature = visual_feature.reshape(batch_size, self.obs_horizon, visual_feature.shape[1])  # (B, obs_horizon, D)
        feature = torch.cat((visual_feature, obs_seq["state"]), dim=-1)  # (B, obs_horizon, D+obs_state_dim)
        return feature  # (B, obs_horizon, D+obs_state_dim)

    def prepare_noise_condition(self, obs_cond):
        if self.noise_model == "Unet":
            return obs_cond.flatten(start_dim=1)
        return obs_cond

    def compute_loss(self, obs_seq, action_seq):
        B = obs_seq["state"].shape[0]

        # observation as FiLM conditioning
        obs_cond = self.encode_obs(
            obs_seq, eval_mode=False
        )  # (B, obs_horizon * obs_dim)
        obs_cond = self.prepare_noise_condition(obs_cond)

        # sample noise to add to actions
        noise = torch.randn((B, self.pred_horizon, self.act_dim), device=device)

        # sample a diffusion iteration for each data point
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (B,), device=device).long()

        # add noise to the clean images(actions) according to the noise magnitude at each diffusion iteration
        # (this is the forward diffusion process)
        noisy_action_seq = self.noise_scheduler.add_noise(action_seq, noise, timesteps)

        # predict the noise residual
        noise_pred = self.noise_pred_net(noisy_action_seq, timesteps, obs_cond)

        return F.mse_loss(noise_pred, noise)

    def get_action(self, obs_seq):
        # init scheduler
        # self.noise_scheduler.set_timesteps(self.num_diffusion_iters)
        # set_timesteps will change noise_scheduler.timesteps is only used in noise_scheduler.step()
        # noise_scheduler.step() is only called during inference
        # if we use DDPM, and inference_diffusion_steps == train_diffusion_steps, then we can skip this

        # obs_seq['state']: (B, obs_horizon, obs_state_dim)
        B = obs_seq["state"].shape[0]
        with torch.no_grad():
            if self.include_rgb:
                obs_seq["rgb"] = obs_seq["rgb"].permute(0, 1, 4, 2, 3)
            if self.include_depth:
                obs_seq["depth"] = obs_seq["depth"].permute(0, 1, 4, 2, 3)

            obs_cond = self.encode_obs(
                obs_seq, eval_mode=True
            )  # (B, obs_horizon * obs_dim)
            obs_cond = self.prepare_noise_condition(obs_cond)

            # initialize action from Guassian noise
            noisy_action_seq = torch.randn(
                (B, self.pred_horizon, self.act_dim), device=obs_seq["state"].device
            )

            for k in self.noise_scheduler.timesteps:
                # predict noise
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
        if self.action_denorm_min.numel() > 0 and self.action_denorm_max.numel() > 0:
            d = min(int(self.action_denorm_min.shape[0]), action_seq.shape[-1])
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
    ensure_raw_train_eval_split(args)

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
    if args.eval_demo_path is None or len(args.eval_demo_path.strip()) == 0:
        raise ValueError(
            "eval_demo_path is required for split train/eval. "
            "Pass --eval-demo-path to select eval reset seeds from an eval h5/json."
        )
    eval_reset_kwargs_list = load_eval_reset_kwargs_list(
        eval_demo_path=args.eval_demo_path,
        eval_demo_metadata_path=args.eval_demo_metadata_path,
        num_traj=args.num_eval_demos,
    )
    if args.num_eval_episodes != len(eval_reset_kwargs_list):
        print(
            f"[eval-split] overriding num_eval_episodes={args.num_eval_episodes} "
            f"to selected eval demos={len(eval_reset_kwargs_list)}"
        )
        args.num_eval_episodes = len(eval_reset_kwargs_list)
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # create evaluation environment
    env_kwargs = dict(
        control_mode=args.control_mode,
        reward_mode="sparse",
        obs_mode=args.obs_mode,
        render_mode="rgb_array",
        sensor_configs=dict(shader_pack="default"),
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
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # create temporary env to get original observation space as AsyncVectorEnv (CPU parallelization) doesn't permit that
    tmp_env = gym.make(args.env_id, **env_kwargs)
    orignal_obs_space = tmp_env.observation_space
    # determine whether the env will return rgb and/or depth data
    include_rgb = tmp_env.unwrapped.obs_mode_struct.visual.rgb
    include_depth = tmp_env.unwrapped.obs_mode_struct.visual.depth
    tmp_env.close()

    obs_process_fn = partial(
    convert_obs,
    concat_fn=partial(np.concatenate, axis=-1),
    transpose_fn=partial(np.transpose, axes=(0, 3, 1, 2)),  # (B, H, W, C) -> (B, C, H, W)
    state_obs_extractor=build_state_obs_extractor(args.env_id),
    depth = include_depth
    )

    dataset = SmallDemoDataset_DiffusionPolicy(
        data_path=args.demo_path,
        obs_process_fn=obs_process_fn,
        obs_space=orignal_obs_space,
        include_rgb=include_rgb,
        include_depth=include_depth,
        obs_horizon=args.obs_horizon,
        pred_horizon=args.pred_horizon,
        device=device,
        num_traj=args.num_demos,
        action_norm_path=args.action_norm_path,
    )
    denorm_mins = dataset.action_min
    denorm_maxs = dataset.action_max
    print(
        f"[denorm] training uses normalized pd_ee_pose actions; rollout will denormalize with {dataset.action_norm_path}"
    )
    debug_sample_idx = 0
    print("Debug printing a sample from the dataset...")
    dataset.debug_print_sample(debug_sample_idx)
    
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

    agent = Agent(envs, args).to(device)

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
    ema_agent = Agent(envs, args).to(device)
    agent.set_action_denormalizer(denorm_mins, denorm_maxs, device)
    ema_agent.set_action_denormalizer(denorm_mins, denorm_maxs, device)

    best_eval_metrics = defaultdict(float)
    timings = defaultdict(float)

    # define evaluation and logging functions
    def evaluate_and_save_best(iteration):
        if iteration % args.eval_freq == 0:
            last_tick = time.time()
            ema.copy_to(ema_agent.parameters())
            eval_metrics = evaluate(
                args.num_eval_episodes,
                ema_agent,
                envs,
                device,
                args.sim_backend,
                reset_kwargs_list=eval_reset_kwargs_list,
            )
            timings["eval"] += time.time() - last_tick

            print(f"Evaluated {len(eval_metrics['success_at_end'])} episodes")
            for k in eval_metrics.keys():
                eval_metrics[k] = np.mean(eval_metrics[k])
                writer.add_scalar(f"eval/{k}", eval_metrics[k], iteration)
                print(f"{k}: {eval_metrics[k]:.4f}")

            save_on_best_metrics = ["success_once", "success_at_end"]
            for k in save_on_best_metrics:
                if k in eval_metrics and eval_metrics[k] > best_eval_metrics[k]:
                    best_eval_metrics[k] = eval_metrics[k]
                    save_ckpt(run_name, f"best_eval_{k}")
                    print(
                        f"New best {k}_rate: {eval_metrics[k]:.4f}. Saving checkpoint."
                    )
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
