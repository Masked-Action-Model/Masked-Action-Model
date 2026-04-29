ALGO_NAME = "BC_Diffusion_rgbd_DiT_DDP"

import os
import random
import time
import json
from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from typing import Optional

import gymnasium as gym
from gymnasium.vector.vector_env import VectorEnv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
import tyro
from diffusers.optimization import get_scheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from gymnasium import spaces
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from evaluate.evaluate import evaluate
from models.modeling_ditdp import DiTNoiseNet
from models.plain_conv import PlainConv
from utils.denormalize_utils import compute_state_min_max
from utils.load_train_data_utils import load_demo_dataset
from utils.make_env import make_eval_envs
from utils.utils import (IterationBasedBatchSampler, build_state_obs_extractor,
                         convert_obs, worker_init_fn)

@dataclass
class Args:
    exp_name: Optional[str] = None
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "ManiSkill"
    wandb_entity: Optional[str] = None
    capture_video: bool = True

    env_id: str = "PegInsertionSide-v1"
    demo_path: str = "demos/PegInsertionSide-v1/trajectory.state.pd_ee_delta_pose.physx_cpu.h5"
    num_demos: Optional[int] = None
    total_iters: int = 1_000_000
    batch_size: int = 256

    lr: float = 1e-4
    obs_horizon: int = 2
    act_horizon: int = 8
    pred_horizon: int = 16
    diffusion_step_embed_dim: int = 64

    obs_mode: str = "rgb+depth"
    max_episode_steps: Optional[int] = None
    log_freq: int = 1000
    eval_freq: int = 5000
    save_freq: Optional[int] = None
    num_eval_episodes: int = 100
    num_eval_envs: int = 10
    action_norm_path: Optional[str] = None
    sim_backend: str = "physx_cpu"
    num_dataload_workers: int = 0
    control_mode: str = "pd_ee_pose"

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
        raise ValueError("action_norm_path is required.")
    if not os.path.exists(action_norm_path):
        raise FileNotFoundError(f"action norm json not found: {action_norm_path}")
    with open(action_norm_path, "r") as f:
        data = json.load(f)
    if "min" not in data or "max" not in data:
        raise ValueError(f"action norm json must contain keys 'min' and 'max': {action_norm_path}")
    mins = np.asarray(data["min"], dtype=np.float32)
    maxs = np.asarray(data["max"], dtype=np.float32)
    if mins.shape != maxs.shape or mins.ndim != 1 or mins.shape[0] == 0:
        raise ValueError(f"invalid min/max shape in action norm json: min={mins.shape}, max={maxs.shape}")
    return mins, maxs


def setup_distributed(args: Args):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1

    if distributed:
        backend = "nccl" if args.cuda and torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")

    if args.cuda and torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    is_main_process = rank == 0
    return distributed, world_size, rank, local_rank, is_main_process, device


def cleanup_distributed(distributed: bool):
    if distributed and dist.is_initialized():
        dist.destroy_process_group()


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


class SmallDemoDataset_DiffusionPolicy(Dataset):
    def __init__(self, data_path, obs_process_fn, obs_space, include_rgb, include_depth, obs_horizon, pred_horizon, num_traj):
        self.include_rgb = include_rgb
        self.include_depth = include_depth

        from utils.load_train_data_utils import load_demo_dataset
        trajectories = load_demo_dataset(data_path, num_traj=num_traj, concat=False)
        print("Raw trajectory loaded, beginning observation pre-processing...")

        obs_traj_dict_list = []
        for obs_traj_dict in trajectories["observations"]:
            _obs_traj_dict = reorder_keys(obs_traj_dict, obs_space)
            _obs_traj_dict = obs_process_fn(_obs_traj_dict)
            if self.include_depth:
                _obs_traj_dict["depth"] = torch.from_numpy(_obs_traj_dict["depth"].astype(np.float32))
            if self.include_rgb:
                _obs_traj_dict["rgb"] = torch.from_numpy(_obs_traj_dict["rgb"])
            _obs_traj_dict["state"] = torch.from_numpy(_obs_traj_dict["state"])
            obs_traj_dict_list.append(_obs_traj_dict)

        trajectories["observations"] = obs_traj_dict_list
        self.obs_keys = list(obs_traj_dict_list[0].keys())

        for i in range(len(trajectories["actions"])):
            trajectories["actions"][i] = torch.from_numpy(trajectories["actions"][i].astype(np.float32))

        self.obs_horizon, self.pred_horizon = obs_horizon, pred_horizon

        self.slices = []
        num_traj = len(trajectories["actions"])
        total_transitions = 0
        for traj_idx in range(num_traj):
            L = trajectories["actions"][traj_idx].shape[0]
            assert trajectories["observations"][traj_idx]["state"].shape[0] == L + 1
            total_transitions += L

            pad_before = obs_horizon - 1
            pad_after = pred_horizon - obs_horizon
            self.slices += [
                (traj_idx, start, start + pred_horizon)
                for start in range(-pad_before, L - pred_horizon + pad_after)
            ]

        print(f"Total transitions: {total_transitions}, Total obs sequences: {len(self.slices)}")
        self.trajectories = trajectories

    def __getitem__(self, index):
        traj_idx, start, end = self.slices[index]
        L, _ = self.trajectories["actions"][traj_idx].shape

        obs_traj = self.trajectories["observations"][traj_idx]
        obs_seq = {}
        for k, v in obs_traj.items():
            obs_seq[k] = v[max(0, start): start + self.obs_horizon]
            if start < 0:
                pad_len = -start
                if k == "state":
                    if obs_seq[k].shape[0] >= 2:
                        d = obs_seq[k][1] - obs_seq[k][0]
                    else:
                        d = torch.zeros_like(obs_seq[k][0])
                    pad_obs = [obs_seq[k][0] - d * n for n in range(pad_len, 0, -1)]
                    obs_seq[k] = torch.cat((torch.stack(pad_obs, dim=0), obs_seq[k]), dim=0)
                else:
                    pad_obs_seq = torch.stack([obs_seq[k][0]] * pad_len, dim=0)
                    obs_seq[k] = torch.cat((pad_obs_seq, obs_seq[k]), dim=0)

        act_seq = self.trajectories["actions"][traj_idx][max(0, start): end]
        if start < 0:
            if act_seq.shape[0] >= 2:
                d = act_seq[1] - act_seq[0]
            else:
                d = torch.zeros_like(act_seq[0])
            pad_len = -start
            pad_actions = [act_seq[0] - d * k for k in range(pad_len, 0, -1)]
            act_seq = torch.cat([torch.stack(pad_actions, dim=0), act_seq], dim=0)
        if end > L:
            pad_action = act_seq[-1]
            act_seq = torch.cat([act_seq, pad_action.repeat(end - L, 1)], dim=0)

        return {"observations": obs_seq, "actions": act_seq}

    def __len__(self):
        return len(self.slices)


class Agent(nn.Module):
    def __init__(self, env: VectorEnv, args: Args):
        super().__init__()
        self.obs_horizon = args.obs_horizon
        self.act_horizon = args.act_horizon
        self.pred_horizon = args.pred_horizon

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
        self.visual_encoder = PlainConv(in_channels=total_visual_channels, out_dim=visual_feature_dim, pool_feature_map=True)
        self.noise_pred_net = DiTNoiseNet(
            ac_dim=self.act_dim,
            ac_chunk=self.pred_horizon,
            obs_dim=visual_feature_dim + obs_state_dim,
            time_dim=args.diffusion_step_embed_dim,
            use_mask=False,
        )
        self.num_diffusion_iters = 100
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.num_diffusion_iters,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type="epsilon",
        )
        self.action_denorm_min = None
        self.action_denorm_max = None
        self.action_denorm_dims = 0

    def set_action_denormalizer(self, mins: np.ndarray, maxs: np.ndarray, device):
        self.action_denorm_dims = int(mins.shape[0])
        self.action_denorm_min = torch.as_tensor(mins, device=device, dtype=torch.float32)
        self.action_denorm_max = torch.as_tensor(maxs, device=device, dtype=torch.float32)

    def encode_obs(self, obs_seq, eval_mode):
        if self.include_rgb:
            rgb = obs_seq["rgb"].float() / 255.0
            img_seq = rgb
        if self.include_depth:
            depth = obs_seq["depth"].float() / 1024.0
            img_seq = depth
        if self.include_rgb and self.include_depth:
            img_seq = torch.cat([rgb, depth], dim=2)

        batch_size = img_seq.shape[0]
        img_seq = img_seq.flatten(end_dim=1)
        visual_feature = self.visual_encoder(img_seq)
        visual_feature = visual_feature.reshape(batch_size, self.obs_horizon, visual_feature.shape[1])
        feature = torch.cat((visual_feature, obs_seq["state"]), dim=-1)
        return feature

    def compute_loss(self, obs_seq, action_seq):
        B = obs_seq["state"].shape[0]
        dev = obs_seq["state"].device
        obs_cond = self.encode_obs(obs_seq, eval_mode=False)

        noise = torch.randn((B, self.pred_horizon, self.act_dim), device=dev)
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (B,), device=dev).long()
        noisy_action_seq = self.noise_scheduler.add_noise(action_seq, noise, timesteps)
        noise_pred = self.noise_pred_net(noisy_action_seq, timesteps, obs_cond)
        return F.mse_loss(noise_pred, noise)

    def forward(self, obs_seq, action_seq):
        return self.compute_loss(obs_seq, action_seq)

    def get_action(self, obs_seq):
        B = obs_seq["state"].shape[0]
        with torch.no_grad():
            if self.include_rgb:
                obs_seq["rgb"] = obs_seq["rgb"].permute(0, 1, 4, 2, 3)
            if self.include_depth:
                obs_seq["depth"] = obs_seq["depth"].permute(0, 1, 4, 2, 3)

            obs_cond = self.encode_obs(obs_seq, eval_mode=True)
            noisy_action_seq = torch.randn((B, self.pred_horizon, self.act_dim), device=obs_seq["state"].device)

            for k in self.noise_scheduler.timesteps:
                timesteps = torch.full((B,), k, dtype=torch.long, device=noisy_action_seq.device)
                noise_pred = self.noise_pred_net(noisy_action_seq, timesteps, obs_cond)
                noisy_action_seq = self.noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=k,
                    sample=noisy_action_seq,
                ).prev_sample

        start = self.obs_horizon - 1
        end = start + self.act_horizon
        action_seq = noisy_action_seq[:, start:end]
        if self.action_denorm_dims > 0:
            d = min(self.action_denorm_dims, action_seq.shape[-1])
            mins = self.action_denorm_min[:d]
            maxs = self.action_denorm_max[:d]
            action_seq = action_seq.clone()
            action_seq[..., :d] = mins + 0.5 * (action_seq[..., :d] + 1.0) * (maxs - mins)
        return action_seq


def main():
    args = tyro.cli(Args)

    assert args.obs_horizon + args.act_horizon - 1 <= args.pred_horizon, (
        "invalid horizons: require obs_horizon + act_horizon - 1 <= pred_horizon"
    )
    assert args.obs_horizon >= 1 and args.act_horizon >= 1 and args.pred_horizon >= 1, (
        "obs_horizon/act_horizon/pred_horizon must be >= 1"
    )

    distributed, world_size, rank, local_rank, is_main_process, device = setup_distributed(args)

    if args.exp_name is None:
        base_name = os.path.basename(__file__)[:-len(".py")]
        run_name = f"{args.env_id}__{base_name}__{args.seed}__{int(time.time())}"
    else:
        run_name = args.exp_name

    if distributed:
        shared = [run_name if is_main_process else None]
        dist.broadcast_object_list(shared, src=0)
        run_name = shared[0]

    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    denorm_mins, denorm_maxs = load_action_denorm_stats(args.action_norm_path)

    env_kwargs = dict(
        control_mode=args.control_mode,
        reward_mode="sparse",
        obs_mode=args.obs_mode,
        render_mode="rgb_array",
        human_render_camera_configs=dict(shader_pack="default")
    )
    assert args.max_episode_steps is not None, "max_episode_steps must be specified"
    env_kwargs["max_episode_steps"] = args.max_episode_steps

    other_kwargs = dict(obs_horizon=args.obs_horizon)
    envs = make_eval_envs(
        args.env_id,
        args.num_eval_envs,
        args.sim_backend,
        env_kwargs,
        other_kwargs,
        video_dir=f"runs/{run_name}/videos" if args.capture_video and is_main_process else None,
        wrappers=[FlattenRGBDObservationWrapper],
    )

    if args.track and is_main_process:
        import wandb
        config = vars(args)
        config["ddp_world_size"] = world_size
        config["eval_env_cfg"] = dict(**env_kwargs, num_envs=args.num_eval_envs, env_id=args.env_id, env_horizon=args.max_episode_steps)
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=config,
            name=run_name,
            save_code=True,
            group="DiffusionPolicy_DDP",
            tags=["diffusion_policy", "ddp"],
        )

    writer = SummaryWriter(f"runs/{run_name}") if is_main_process else None
    if writer is not None:
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )

    tmp_env = gym.make(args.env_id, **env_kwargs)
    original_obs_space = tmp_env.observation_space
    include_rgb = tmp_env.unwrapped.obs_mode_struct.visual.rgb
    include_depth = tmp_env.unwrapped.obs_mode_struct.visual.depth
    tmp_env.close()

    obs_process_fn = partial(
        convert_obs,
        concat_fn=partial(np.concatenate, axis=-1),
        transpose_fn=partial(np.transpose, axes=(0, 3, 1, 2)),
        state_obs_extractor=build_state_obs_extractor(args.env_id),
        depth=include_depth,
    )

    dataset = SmallDemoDataset_DiffusionPolicy(
        data_path=args.demo_path,
        obs_process_fn=obs_process_fn,
        obs_space=original_obs_space,
        include_rgb=include_rgb,
        include_depth=include_depth,
        obs_horizon=args.obs_horizon,
        pred_horizon=args.pred_horizon,
        num_traj=args.num_demos,
    )

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
    local_batch_size = max(1, args.batch_size // world_size)
    if args.batch_size % world_size != 0 and is_main_process:
        print(f"[warning] batch_size={args.batch_size} not divisible by world_size={world_size}. local_batch_size={local_batch_size}")

    global_batch_effective = local_batch_size * world_size
    total_num_workers = args.num_dataload_workers * world_size
    if is_main_process:
        print(
            "[ddp] "
            f"global_batch(requested)={args.batch_size}, "
            f"global_batch(effective)={global_batch_effective}, "
            f"per_gpu_batch={local_batch_size}, "
            f"world_size={world_size}, "
            f"num_workers(per_gpu)={args.num_dataload_workers}, "
            f"num_workers(total)={total_num_workers}"
        )
        if args.num_dataload_workers == 0:
            print("[hint] num_workers=0 最稳但吞吐可能较低；Linux 可尝试 4/8 提升数据加载速度。")
        elif total_num_workers > os.cpu_count():
            print(
                f"[hint] total workers ({total_num_workers}) > CPU cores ({os.cpu_count()})，"
                "可能导致上下文切换开销增加，可考虑降低 num_workers。"
            )

    train_dataloader = DataLoader(
        dataset,
        batch_size=local_batch_size,
        sampler=sampler,
        num_workers=args.num_dataload_workers,
        worker_init_fn=lambda worker_id: worker_init_fn(worker_id, base_seed=args.seed + rank * 1000),
        persistent_workers=(args.num_dataload_workers > 0),
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    agent = Agent(envs, args).to(device)
    agent.set_action_denormalizer(denorm_mins, denorm_maxs, device)

    if distributed:
        ddp_agent = DDP(
            agent,
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None,
            broadcast_buffers=False,
        )
    else:
        ddp_agent = agent

    optimizer = optim.AdamW(params=ddp_agent.parameters(), lr=args.lr, betas=(0.95, 0.999), weight_decay=1e-6)
    lr_scheduler = get_scheduler(name="cosine", optimizer=optimizer, num_warmup_steps=500, num_training_steps=args.total_iters)

    ema = EMAModel(parameters=agent.parameters(), power=0.75)
    ema_agent = Agent(envs, args).to(device)
    ema_agent.set_action_denormalizer(denorm_mins, denorm_maxs, device)

    best_eval_metrics = defaultdict(float)
    timings = defaultdict(float)

    def save_ckpt(tag):
        if not is_main_process:
            return
        os.makedirs(f"runs/{run_name}/checkpoints", exist_ok=True)
        ema.copy_to(ema_agent.parameters())
        torch.save(
            {
                "agent": agent.state_dict(),
                "ema_agent": ema_agent.state_dict(),
                "iter": current_iter,
                "world_size": world_size,
            },
            f"runs/{run_name}/checkpoints/{tag}.pt",
        )

    def evaluate_and_save_best(iteration):
        if (not is_main_process) or (iteration % args.eval_freq != 0):
            return
        last_tick = time.time()
        ema.copy_to(ema_agent.parameters())
        eval_metrics = evaluate(args.num_eval_episodes, ema_agent, envs, device, args.sim_backend)
        timings["eval"] += time.time() - last_tick

        for k in eval_metrics.keys():
            eval_metrics[k] = np.mean(eval_metrics[k])
            writer.add_scalar(f"eval/{k}", eval_metrics[k], iteration)

        save_on_best_metrics = ["success_once", "success_at_end"]
        for k in save_on_best_metrics:
            if k in eval_metrics and eval_metrics[k] > best_eval_metrics[k]:
                best_eval_metrics[k] = eval_metrics[k]
                save_ckpt(f"best_eval_{k}")

    def log_metrics(iteration, total_loss):
        if (not is_main_process) or (iteration % args.log_freq != 0):
            return
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], iteration)
        writer.add_scalar("losses/total_loss", total_loss.item(), iteration)
        for k, v in timings.items():
            writer.add_scalar(f"time/{k}", v, iteration)

    ddp_agent.train()
    data_iter = iter(train_dataloader)
    pbar = tqdm(total=args.total_iters, disable=not is_main_process)
    last_tick = time.time()

    global current_iter
    current_iter = 0

    for iteration in range(args.total_iters):
        current_iter = iteration
        try:
            data_batch = next(data_iter)
        except StopIteration:
            sampler.set_epoch(iteration)
            data_iter = iter(train_dataloader)
            data_batch = next(data_iter)

        timings["data_loading"] += time.time() - last_tick

        last_tick = time.time()
        data_batch = move_batch_to_device(data_batch, device)
        total_loss = ddp_agent(data_batch["observations"], data_batch["actions"])
        timings["forward"] += time.time() - last_tick

        last_tick = time.time()
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()
        lr_scheduler.step()
        timings["backward"] += time.time() - last_tick

        last_tick = time.time()
        ema.step(agent.parameters())
        timings["ema"] += time.time() - last_tick

        evaluate_and_save_best(iteration)
        log_metrics(iteration, total_loss)

        if args.save_freq is not None and iteration % args.save_freq == 0:
            save_ckpt(str(iteration))

        if is_main_process:
            pbar.update(1)
            pbar.set_postfix({"loss": float(total_loss.item())})
        last_tick = time.time()

    evaluate_and_save_best(args.total_iters)
    log_metrics(args.total_iters, total_loss)

    envs.close()
    if writer is not None:
        writer.close()

    cleanup_distributed(distributed)


if __name__ == "__main__":
    main()
