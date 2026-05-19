from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Optional

import numpy as np
import torch
import tyro

from common import (
    build_policy_env_stub,
    build_open_loop_validator,
    infer_action_dim,
    install_maniskill_stubs,
    load_demo_dataset_with_optional_done,
    load_meta,
    make_obs_process_fn,
    make_run_name,
    seed_everything,
    split_train_validation_traj_indices,
    train_no_eval,
)

install_maniskill_stubs()
import train_baseline as baseline_mod


def _meta_bool(meta: dict, key: str, default: bool = False) -> bool:
    if key not in meta:
        return bool(default)
    value = np.asarray(meta[key])
    if value.shape == ():
        return bool(value.item())
    return bool(value.reshape(-1)[0])


class FrankaBaselineDataset(baseline_mod.SmallDemoDataset_DiffusionPolicy):
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
        action_dim,
        action_norm_path=None,
        traj_indices=None,
    ):
        meta = load_meta(data_path)
        actions_normalized = _meta_bool(meta, "actions_normalized", False)
        states_normalized = _meta_bool(meta, "states_normalized", False)
        if not actions_normalized:
            if traj_indices is not None:
                raise ValueError(
                    "validation split for raw baseline data is not supported here; "
                    "use preprocessed normalized Franka data."
                )
            super().__init__(
                data_path=data_path,
                obs_process_fn=obs_process_fn,
                obs_space=obs_space,
                include_rgb=include_rgb,
                include_depth=include_depth,
                obs_horizon=obs_horizon,
                pred_horizon=pred_horizon,
                device=device,
                num_traj=num_traj,
                action_dim=action_dim,
                action_norm_path=action_norm_path,
            )
            return

        self.action_dim = int(action_dim)
        if self.action_dim not in (6, 7):
            raise ValueError(f"action_dim must be 6 or 7, got {self.action_dim}")
        self.include_rgb = include_rgb
        self.include_depth = include_depth
        self.action_norm_path = action_norm_path or data_path
        self.action_min = np.asarray(meta["action_min"], dtype=np.float32)
        self.action_max = np.asarray(meta["action_max"], dtype=np.float32)
        if self.action_min.shape[0] > self.action_dim:
            raise ValueError(
                f"action norm dim ({self.action_min.shape[0]}) exceeds action_dim={self.action_dim}"
            )

        trajectories = load_demo_dataset_with_optional_done(
            data_path,
            num_traj=num_traj if traj_indices is None else None,
            concat=False,
            traj_indices=traj_indices,
        )
        print("Preprocessed Franka trajectory loaded, beginning observation pre-processing...")

        obs_traj_dict_list = []
        for obs_traj_dict in trajectories["observations"]:
            precomputed_state = obs_traj_dict.get("state", None)
            obs_traj_dict = baseline_mod.reorder_keys(obs_traj_dict, obs_space)
            obs_traj_dict = obs_process_fn(obs_traj_dict)
            if include_depth:
                obs_traj_dict["depth"] = torch.Tensor(
                    obs_traj_dict["depth"].astype(np.float32)
                ).to(dtype=torch.float16)
            if include_rgb:
                obs_traj_dict["rgb"] = torch.from_numpy(obs_traj_dict["rgb"])
            if states_normalized and precomputed_state is not None:
                obs_traj_dict["state"] = torch.from_numpy(
                    np.asarray(precomputed_state, dtype=np.float32)
                ).to(dtype=torch.float32)
            else:
                obs_traj_dict["state"] = torch.from_numpy(obs_traj_dict["state"]).to(
                    dtype=torch.float32
                )
            obs_traj_dict_list.append(obs_traj_dict)
        trajectories["observations"] = obs_traj_dict_list
        self.obs_keys = list(obs_traj_dict_list[0].keys())

        for traj_idx, action in enumerate(trajectories["actions"]):
            action_np = np.asarray(action, dtype=np.float32)
            if action_np.ndim != 2 or action_np.shape[1] != self.action_dim:
                raise ValueError(
                    f"traj_{traj_idx} actions must have shape (T, {self.action_dim}), got {action_np.shape}"
                )
            trajectories["actions"][traj_idx] = torch.from_numpy(action_np).to(dtype=torch.float32)
        print(
            f"[action_norm] using preprocessed normalized actions: {self.action_norm_path}"
        )
        print(f"[action_norm] min={self.action_min}")
        print(f"[action_norm] max={self.action_max}")

        print("Obs/action pre-processing is done, start to pre-compute the slice indices...")
        print("Using fixed control mode pd_ee_pose, padding with final action.")
        self.obs_horizon, self.pred_horizon = obs_horizon, pred_horizon
        self.slices = []
        num_traj = len(trajectories["actions"])
        total_transitions = 0
        for traj_idx in range(num_traj):
            length = trajectories["actions"][traj_idx].shape[0]
            assert trajectories["observations"][traj_idx]["state"].shape[0] == length + 1
            total_transitions += length
            pad_before = obs_horizon - 1
            pad_after = pred_horizon - obs_horizon
            self.slices += [
                (traj_idx, start, start + pred_horizon)
                for start in range(-pad_before, length - pred_horizon + pad_after)
            ]
        print(
            f"Total transitions: {total_transitions}, Total obs sequences: {len(self.slices)}"
        )
        self.trajectories = trajectories


@dataclass
class Args:
    exp_name: Optional[str] = None
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "ManiSkill"
    wandb_entity: Optional[str] = None
    capture_video: bool = False

    env_id: str = "FrankaReal-v1"
    demo_path: str = "franka_train/data/franka_real.h5"
    num_demos: Optional[int] = None
    action_dim: Optional[int] = 7
    action_norm_path: Optional[str] = None

    total_iters: int = 100_000
    batch_size: int = 256
    lr: float = 1e-4
    obs_horizon: int = 2
    act_horizon: int = 8
    pred_horizon: int = 16
    diffusion_step_embed_dim: int = 64
    noise_model: Literal["Transformer", "Unet"] = "Transformer"
    vision_encoder: Literal["resnet", "dino2", "dino3"] = "resnet"
    dino_model_path: str = ""
    dino_data_aug: bool = False
    dit_hidden_dim: int = 512
    dit_num_blocks: int = 6
    dit_dim_feedforward: int = 2048
    unet_dims: List[int] = field(default_factory=lambda: [64, 128, 256])
    n_groups: int = 8

    obs_mode: Literal["rgb", "rgb+depth"] = "rgb"
    max_episode_steps: Optional[int] = None
    sim_backend: str = "physx_cpu"
    log_freq: int = 1000
    eval_freq: int = 0
    valid_freq: int = 0
    num_validation_set: int = 0
    num_eval_episodes: int = 0
    num_eval_envs: int = 0
    num_eval_demos: Optional[int] = None
    save_start_iter: int = 0
    save_freq: Optional[int] = 5000
    num_dataload_workers: int = 0
    control_mode: str = "pd_ee_pose"
    demo_type: Optional[str] = "franka_real_baseline"


def main() -> None:
    args = tyro.cli(Args)
    args.action_dim = infer_action_dim(args.demo_path, args.action_dim)
    run_name = make_run_name(
        Path(__file__).stem,
        args.env_id,
        args.exp_name,
        args.seed,
    )
    if args.obs_horizon + args.act_horizon - 1 > args.pred_horizon:
        raise ValueError("obs_horizon + act_horizon - 1 must be <= pred_horizon")

    seed_everything(args.seed, args.torch_deterministic)
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    baseline_mod.device = device

    env, raw_obs_space, include_rgb, include_depth = build_policy_env_stub(
        args.demo_path,
        action_dim=args.action_dim,
        obs_horizon=args.obs_horizon,
        obs_mode=args.obs_mode,
    )
    obs_process_fn = make_obs_process_fn(include_depth)
    train_traj_indices, val_traj_indices = split_train_validation_traj_indices(
        args.demo_path,
        args.num_demos,
        args.num_validation_set,
        args.seed,
    )

    dataset = FrankaBaselineDataset(
        data_path=args.demo_path,
        obs_process_fn=obs_process_fn,
        obs_space=raw_obs_space,
        include_rgb=include_rgb,
        include_depth=include_depth,
        obs_horizon=args.obs_horizon,
        pred_horizon=args.pred_horizon,
        device=device,
        num_traj=args.num_demos if train_traj_indices is None else None,
        action_dim=args.action_dim,
        action_norm_path=args.action_norm_path,
        traj_indices=train_traj_indices,
    )
    dataset.debug_print_sample(0)
    val_dataset = None
    validate_fn = None
    if val_traj_indices:
        val_dataset = FrankaBaselineDataset(
            data_path=args.demo_path,
            obs_process_fn=obs_process_fn,
            obs_space=raw_obs_space,
            include_rgb=include_rgb,
            include_depth=include_depth,
            obs_horizon=args.obs_horizon,
            pred_horizon=args.pred_horizon,
            device=device,
            num_traj=None,
            action_dim=args.action_dim,
            action_norm_path=args.action_norm_path,
            traj_indices=val_traj_indices,
        )

    agent = baseline_mod.Agent(env, args).to(device)
    ema_agent = baseline_mod.Agent(env, args).to(device)
    agent.set_action_denormalizer(dataset.action_min, dataset.action_max, device)
    ema_agent.set_action_denormalizer(dataset.action_min, dataset.action_max, device)
    if val_dataset is not None:
        validate_fn = build_open_loop_validator(
            dataset=val_dataset,
            device=device,
        )

    def compute_loss(model, batch):
        return model.compute_loss(
            obs_seq=batch["observations"],
            action_seq=batch["actions"],
        )

    train_no_eval(
        run_name=run_name,
        args=args,
        dataset=dataset,
        agent=agent,
        ema_agent=ema_agent,
        device=device,
        compute_loss=compute_loss,
        validate_fn=validate_fn,
    )
    env.close()


if __name__ == "__main__":
    main()
