from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Optional

import torch
import tyro

from common import (
    build_policy_env_stub,
    infer_action_dim,
    install_maniskill_stubs,
    make_obs_process_fn,
    make_run_name,
    seed_everything,
    train_no_eval,
)

install_maniskill_stubs()
import train_baseline as baseline_mod


@dataclass
class Args:
    exp_name: Optional[str] = None
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True

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
    dit_hidden_dim: int = 512
    dit_num_blocks: int = 6
    dit_dim_feedforward: int = 2048
    unet_dims: List[int] = field(default_factory=lambda: [64, 128, 256])
    n_groups: int = 8

    obs_mode: Literal["rgb", "rgb+depth"] = "rgb"
    log_freq: int = 1000
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

    dataset = baseline_mod.SmallDemoDataset_DiffusionPolicy(
        data_path=args.demo_path,
        obs_process_fn=obs_process_fn,
        obs_space=raw_obs_space,
        include_rgb=include_rgb,
        include_depth=include_depth,
        obs_horizon=args.obs_horizon,
        pred_horizon=args.pred_horizon,
        device=device,
        num_traj=args.num_demos,
        action_dim=args.action_dim,
        action_norm_path=args.action_norm_path,
    )
    dataset.debug_print_sample(0)

    agent = baseline_mod.Agent(env, args).to(device)
    ema_agent = baseline_mod.Agent(env, args).to(device)
    agent.set_action_denormalizer(dataset.action_min, dataset.action_max, device)
    ema_agent.set_action_denormalizer(dataset.action_min, dataset.action_max, device)

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
    )
    env.close()


if __name__ == "__main__":
    main()
