import argparse
import sys
from pathlib import Path

import gymnasium as gym
import mani_skill.envs  # noqa: F401
import numpy as np
import torch
from mani_skill.utils import common
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diffusion_policy.make_env import make_eval_envs  # noqa: E402
from train_rgbd import Agent as DiTAgent  # noqa: E402
from train_rgbd import Args as DiTArgs  # noqa: E402
from train_rgbd import load_action_stats_from_path as load_dit_action_stats  # noqa: E402
from train_rgbd_unet import Agent as UNetAgent  # noqa: E402
from train_rgbd_unet import Args as UNetArgs  # noqa: E402
from train_rgbd_unet import load_action_stats_from_path as load_unet_action_stats  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["unet", "dit"], required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--action-norm-path",
        default="demos/exp_4/PickCube-v1/motionplanning/experiment_4.rgb.pd_ee_pose.physx_cpu.action_norm.json",
    )
    parser.add_argument("--env-id", default="PickCube-v1")
    parser.add_argument("--obs-mode", default="rgb")
    parser.add_argument("--control-mode", default="pd_ee_pose")
    parser.add_argument("--sim-backend", default="physx_cpu")
    parser.add_argument("--max-episode-steps", type=int, default=100)
    parser.add_argument("--obs-horizon", type=int, default=2)
    parser.add_argument("--act-horizon", type=int, default=8)
    parser.add_argument("--pred-horizon", type=int, default=16)
    parser.add_argument("--diffusion-step-embed-dim", type=int, default=64)
    parser.add_argument("--unet-dims", nargs="+", type=int, default=[64, 128, 256])
    parser.add_argument("--n-groups", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-clip-sample", action="store_true")
    return parser.parse_args()


def build_args(cli):
    cls = UNetArgs if cli.model == "unet" else DiTArgs
    args = cls()
    args.env_id = cli.env_id
    args.obs_mode = cli.obs_mode
    args.control_mode = cli.control_mode
    args.sim_backend = cli.sim_backend
    args.max_episode_steps = cli.max_episode_steps
    args.obs_horizon = cli.obs_horizon
    args.act_horizon = cli.act_horizon
    args.pred_horizon = cli.pred_horizon
    args.diffusion_step_embed_dim = cli.diffusion_step_embed_dim
    args.cuda = cli.device == "cuda"
    if cli.model == "unet":
        args.unet_dims = cli.unet_dims
        args.n_groups = cli.n_groups
    return args


def build_env(args):
    env_kwargs = dict(
        control_mode=args.control_mode,
        reward_mode="sparse",
        obs_mode=args.obs_mode,
        render_mode="rgb_array",
        sensor_configs=dict(shader_pack="default"),
        human_render_camera_configs=dict(shader_pack="default"),
        max_episode_steps=args.max_episode_steps,
    )
    return make_eval_envs(
        args.env_id,
        1,
        args.sim_backend,
        env_kwargs,
        dict(obs_horizon=args.obs_horizon),
        video_dir=None,
        wrappers=[FlattenRGBDObservationWrapper],
    )


def sample_action(agent, obs_seq, model_name):
    bsz = obs_seq["state"].shape[0]
    with torch.no_grad():
        local_obs = dict(obs_seq)
        if agent.include_rgb:
            local_obs["rgb"] = local_obs["rgb"].permute(0, 1, 4, 2, 3)
        if agent.include_depth:
            local_obs["depth"] = local_obs["depth"].permute(0, 1, 4, 2, 3)

        obs_cond = agent.encode_obs(local_obs, eval_mode=True)
        if model_name == "unet":
            obs_cond = obs_cond.flatten(start_dim=1)

        noisy_action_seq = torch.randn(
            (bsz, agent.pred_horizon, agent.act_dim), device=local_obs["state"].device
        )
        for step in agent.noise_scheduler.timesteps:
            timesteps = torch.full(
                (bsz,), step, dtype=torch.long, device=noisy_action_seq.device
            )
            noise_pred = agent.noise_pred_net(noisy_action_seq, timesteps, obs_cond)
            noisy_action_seq = agent.noise_scheduler.step(
                model_output=noise_pred,
                timestep=step,
                sample=noisy_action_seq,
            ).prev_sample

    start = agent.obs_horizon - 1
    end = start + agent.act_horizon
    raw_action_seq = noisy_action_seq[:, start:end]
    action_seq = raw_action_seq
    if agent.action_denorm_min.numel() > 0 and agent.action_denorm_max.numel() > 0:
        d = min(int(agent.action_denorm_min.shape[0]), action_seq.shape[-1])
        mins = agent.action_denorm_min[:d]
        maxs = agent.action_denorm_max[:d]
        action_seq = action_seq.clone()
        action_seq[..., :d] = mins + 0.5 * (action_seq[..., :d] + 1.0) * (maxs - mins)
    return action_seq, raw_action_seq


def load_agent(cli, envs, device):
    args = build_args(cli)
    agent_cls = UNetAgent if cli.model == "unet" else DiTAgent
    load_stats = load_unet_action_stats if cli.model == "unet" else load_dit_action_stats
    agent = agent_cls(envs, args).to(device)
    mins, maxs = load_stats(cli.action_norm_path)
    agent.set_action_denormalizer(mins, maxs, device)
    ckpt = torch.load(cli.checkpoint, map_location=device)
    state = ckpt["ema_agent"] if "ema_agent" in ckpt else ckpt["agent"]
    agent.load_state_dict(state)
    if cli.no_clip_sample:
        agent.noise_scheduler.config.clip_sample = False
    agent.eval()
    return agent


def main():
    cli = parse_args()
    np.random.seed(cli.seed)
    torch.manual_seed(cli.seed)
    device = torch.device(cli.device if torch.cuda.is_available() and cli.device == "cuda" else "cpu")
    args = build_args(cli)
    envs = build_env(args)
    agent = load_agent(cli, envs, device)

    obs, info = envs.reset(seed=cli.seed)
    raw_all = []
    action_all = []
    success_trace = []
    grasp_trace = []
    truncated = np.array([False])
    steps = 0
    while not truncated.any():
        obs_t = common.to_tensor(obs, device)
        action_seq, raw_seq = sample_action(agent, obs_t, cli.model)
        raw_all.append(raw_seq.detach().cpu().numpy().reshape(-1, raw_seq.shape[-1]))
        action_np = action_seq.detach().cpu().numpy()
        action_all.append(action_np.reshape(-1, action_np.shape[-1]))
        for i in range(action_np.shape[1]):
            obs, rew, terminated, truncated, info = envs.step(action_np[:, i])
            steps += 1
            final = info.get("final_info", None)
            if final is not None and isinstance(final, dict) and "episode" in final:
                episode = final["episode"]
                if "success_at_end" in episode:
                    success_trace.append(episode["success_at_end"])
            if "is_grasped" in info:
                grasp_trace.append(info["is_grasped"])
            if truncated.any():
                break

    raw = np.concatenate(raw_all, axis=0)
    action = np.concatenate(action_all, axis=0)
    sat_098 = (np.abs(raw) > 0.98).mean(axis=0)
    sat_090 = (np.abs(raw) > 0.90).mean(axis=0)
    print(f"MODEL {cli.model}")
    print(f"CHECKPOINT {cli.checkpoint}")
    print(f"CLIP_SAMPLE {agent.noise_scheduler.config.clip_sample}")
    print(f"STEPS {steps}")
    print("RAW_MEAN", np.array2string(raw.mean(axis=0), precision=5))
    print("RAW_STD", np.array2string(raw.std(axis=0), precision=5))
    print("RAW_MIN", np.array2string(raw.min(axis=0), precision=5))
    print("RAW_MAX", np.array2string(raw.max(axis=0), precision=5))
    print("SAT_ABS_GT_098", np.array2string(sat_098, precision=5))
    print("SAT_ABS_GT_090", np.array2string(sat_090, precision=5))
    print("ACTION_MEAN", np.array2string(action.mean(axis=0), precision=5))
    print("ACTION_STD", np.array2string(action.std(axis=0), precision=5))
    if "final_info" in info and isinstance(info["final_info"], dict):
        print("FINAL_EPISODE", info["final_info"].get("episode", {}))
    envs.close()


if __name__ == "__main__":
    main()
