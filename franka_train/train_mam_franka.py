from __future__ import annotations

from dataclasses import dataclass, field
import os
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
    load_action_stats_from_h5,
    load_demo_dataset_with_optional_done,
    make_obs_process_fn,
    make_run_name,
    seed_everything,
    split_train_validation_traj_indices,
    state_schema_from_raw_obs_space,
    train_no_eval,
)

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
install_maniskill_stubs()
import train_mam as mam_mod


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
    demo_path: str = "franka_train/data/franka_real_random_mask_0.2_train.h5"
    test_demo_path: Optional[str] = None
    eval_demo_metadata_path: Optional[str] = None
    action_norm_path: Optional[str] = None
    num_demos: Optional[int] = None
    action_dim: Optional[int] = 7
    stpm_ckpt_path: str = ""
    stpm_config_path: str = "STPM/config/rewind_maniskill.yaml"

    total_iters: int = 100_000
    batch_size: int = 256
    lr: float = 1e-4
    obs_horizon: int = 2
    act_horizon: int = 8
    pred_horizon: int = 16
    long_window_backward_length: Optional[int] = None
    long_window_forward_length: Optional[int] = None
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
    short_window_horizon: int = 8
    mas_long_encode_mode: Literal["1DConv", "2DConv"] = "2DConv"
    mas_long_conv_output_dim: int = 64
    loss_mode: Literal["average", "weighted"] = "average"
    loss_mask_area_weight: float = 0.2

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
    capture_video_freq: int = 20
    inpainting: bool = False
    eval_progress_bar: bool = False
    save_start_iter: int = 0
    save_freq: Optional[int] = 5000
    num_dataload_workers: int = 0
    control_mode: str = "pd_ee_pose"
    demo_type: Optional[str] = "franka_real_mam"


def configure_args(args: Args) -> None:
    args.action_dim = infer_action_dim(args.demo_path, args.action_dim)
    mam_mod.configure_mas_dimensions(args.action_dim)
    if args.long_window_backward_length is None:
        args.long_window_backward_length = 0
    if args.long_window_forward_length is None:
        args.long_window_forward_length = args.pred_horizon
    args.long_window_backward_length = int(args.long_window_backward_length)
    args.long_window_forward_length = int(args.long_window_forward_length)
    args.long_window_horizon = (
        args.long_window_backward_length + args.long_window_forward_length
    )
    if args.obs_horizon + args.act_horizon - 1 > args.pred_horizon:
        raise ValueError("obs_horizon + act_horizon - 1 must be <= pred_horizon")
    if not 0.0 <= float(args.loss_mask_area_weight) <= 1.0:
        raise ValueError("loss_mask_area_weight must be in [0, 1]")


def patch_mam_module(raw_obs_space) -> None:
    original_validate = mam_mod._validate_mam_state_alignment

    def validate_franka_state_alignment(data_path, dataset_meta, obs_space, state_obs_extractor):
        if "state_schema_json" in dataset_meta:
            return original_validate(data_path, dataset_meta, obs_space, state_obs_extractor)
        env_schema = state_schema_from_raw_obs_space(obs_space)
        env_dim = sum(int(entry["dim"]) for entry in env_schema)
        if "state_dim" in dataset_meta:
            state_dim = int(np.asarray(dataset_meta["state_dim"]).item())
            if state_dim != env_dim:
                raise ValueError(
                    f"state_dim mismatch: meta={state_dim}, inferred={env_dim}"
                )
        print(f"[franka-mam] no state_schema_json; inferred state_dim={env_dim}")

    mam_mod.load_demo_dataset = load_demo_dataset_with_optional_done
    mam_mod._validate_mam_state_alignment = validate_franka_state_alignment


def main() -> None:
    args = tyro.cli(Args)
    configure_args(args)
    run_name = make_run_name(
        Path(__file__).stem,
        args.env_id,
        args.exp_name,
        args.seed,
    )
    seed_everything(args.seed, args.torch_deterministic)
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    mam_mod.args = args
    mam_mod.device = device
    env, raw_obs_space, _include_rgb, include_depth = build_policy_env_stub(
        args.demo_path,
        action_dim=args.action_dim,
        obs_horizon=args.obs_horizon,
        obs_mode=args.obs_mode,
    )
    patch_mam_module(raw_obs_space)
    obs_process_fn = make_obs_process_fn(include_depth)
    train_traj_indices, val_traj_indices = split_train_validation_traj_indices(
        args.demo_path,
        args.num_demos,
        args.num_validation_set,
        args.seed,
    )

    dataset = mam_mod.SmallDemoDataset_MasWindowDiffusionPolicy(
        data_path=args.demo_path,
        obs_process_fn=obs_process_fn,
        obs_space=raw_obs_space,
        device=device,
        num_traj=args.num_demos if train_traj_indices is None else None,
        traj_indices=train_traj_indices,
        state_obs_extractor=None,
    )
    dataset.debug_print_sample(0)
    val_dataset = None
    if val_traj_indices:
        val_dataset = mam_mod.SmallDemoDataset_MasWindowDiffusionPolicy(
            data_path=args.demo_path,
            obs_process_fn=obs_process_fn,
            obs_space=raw_obs_space,
            device=device,
            num_traj=None,
            traj_indices=val_traj_indices,
            state_obs_extractor=None,
        )

    denorm_mins, denorm_maxs = load_action_stats_from_h5(
        args.action_norm_path or args.demo_path
    )
    agent = mam_mod.Agent(env, args).to(device)
    ema_agent = mam_mod.Agent(env, args).to(device)
    agent.set_action_denormalizer(denorm_mins, denorm_maxs, device)
    ema_agent.set_action_denormalizer(denorm_mins, denorm_maxs, device)
    if dataset.state_min is not None and dataset.state_max is not None:
        agent.set_state_normalizer(dataset.state_min, dataset.state_max, device)
        ema_agent.set_state_normalizer(dataset.state_min, dataset.state_max, device)
    validate_fn = None
    if val_dataset is not None:
        validate_fn = build_open_loop_validator(
            dataset=val_dataset,
            device=device,
        )

    def compute_loss(model, batch):
        return model.compute_loss(
            obs_seq=batch["observations"],
            action_seq=batch["actions"],
            action_mask=batch["action_mask"],
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
