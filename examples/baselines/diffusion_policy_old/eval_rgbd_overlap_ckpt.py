import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import mani_skill.envs  # noqa: F401
import numpy as np
import torch
import tyro
from mani_skill.utils import common
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper

from diffusion_policy.make_env import make_eval_envs
from train_rgbd import Agent, Args as TrainArgs, load_action_denorm_stats


def unwrap_singleton(value):
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            return unwrap_singleton(value.reshape(-1)[0])
        if value.size == 1:
            return unwrap_singleton(value.reshape(-1)[0].item())
    return value


def load_reset_kwargs_list(meta_path: str) -> list[dict[str, Any]]:
    with open(meta_path, "r") as f:
        meta = json.load(f)
    episodes = meta.get("episodes", [])
    reset_kwargs_list: list[dict[str, Any]] = []
    for episode in episodes:
        reset_kwargs = dict(episode.get("reset_kwargs", {}) or {})
        reset_seed = reset_kwargs.get("seed", episode.get("episode_seed", None))
        if isinstance(reset_seed, list):
            reset_seed = reset_seed[0] if len(reset_seed) > 0 else None
        if reset_seed is not None:
            reset_kwargs["seed"] = int(reset_seed)
        reset_kwargs_list.append(reset_kwargs)
    return reset_kwargs_list


def load_old_train_reset_seeds(meta_path: str, num_demos: int) -> list[int]:
    with open(meta_path, "r") as f:
        meta = json.load(f)
    episodes = meta.get("episodes", [])[:num_demos]
    out = []
    for episode in episodes:
        seed = episode.get("reset_kwargs", {}).get("seed", episode.get("episode_seed"))
        if isinstance(seed, list):
            seed = seed[0] if len(seed) > 0 else None
        out.append(int(seed))
    return out


def build_agent(
    envs,
    env_id: str,
    obs_mode: str,
    max_episode_steps: int,
    action_norm_path: str,
    device: torch.device,
):
    args = TrainArgs()
    args.env_id = env_id
    args.obs_mode = obs_mode
    args.max_episode_steps = max_episode_steps
    args.action_norm_path = action_norm_path
    args.control_mode = "pd_ee_pose"
    args.sim_backend = "physx_cpu"
    args.num_eval_envs = 1
    args.capture_video = False
    agent = Agent(envs, args).to(device)
    mins, maxs = load_action_denorm_stats(action_norm_path)
    agent.set_action_denormalizer(mins, maxs, device)
    return agent


def evaluate_checkpoint(
    ckpt_path: str,
    eval_meta_path: str,
    old_train_meta_path: str,
    num_old_train_demos: int,
    max_eval_episodes: int | None,
    env_id: str,
    obs_mode: str,
    max_episode_steps: int,
    action_norm_path: str,
    sim_backend: str,
    device: torch.device,
):
    env_kwargs = dict(
        control_mode="pd_ee_pose",
        reward_mode="sparse",
        obs_mode=obs_mode,
        render_mode="rgb_array",
        human_render_camera_configs=dict(shader_pack="default"),
        max_episode_steps=max_episode_steps,
    )
    envs = make_eval_envs(
        env_id,
        1,
        sim_backend,
        env_kwargs,
        dict(obs_horizon=2),
        video_dir=None,
        wrappers=[FlattenRGBDObservationWrapper],
    )

    agent = build_agent(
        envs=envs,
        env_id=env_id,
        obs_mode=obs_mode,
        max_episode_steps=max_episode_steps,
        action_norm_path=action_norm_path,
        device=device,
    )
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt["ema_agent"] if "ema_agent" in ckpt else ckpt["agent"]
    missing, unexpected = agent.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"[load_state_dict] missing={missing} unexpected={unexpected}")
    agent.eval()

    reset_kwargs_list = load_reset_kwargs_list(eval_meta_path)
    if max_eval_episodes is not None:
        reset_kwargs_list = reset_kwargs_list[:max_eval_episodes]
    heldout_seeds = [int(kwargs.get("seed")) for kwargs in reset_kwargs_list]
    old_train_seeds = set(load_old_train_reset_seeds(old_train_meta_path, num_old_train_demos))

    results = []
    with torch.no_grad():
        for episode_idx, reset_kwargs in enumerate(reset_kwargs_list):
            obs, _ = envs.reset(**reset_kwargs)
            success_once = False
            success_at_end = False
            episode_len = 0
            while True:
                obs_tensor = common.to_tensor(obs, device)
                action_seq = agent.get_action(obs_tensor)
                if sim_backend == "physx_cpu":
                    action_seq = action_seq.cpu().numpy()
                truncated_flag = False
                for i in range(action_seq.shape[1]):
                    obs, rew, terminated, truncated, info = envs.step(action_seq[:, i])
                    episode_len += 1
                    if bool(np.asarray(truncated).reshape(-1)[0]):
                        final_info = unwrap_singleton(info.get("final_info", {}))
                        episode_metrics = unwrap_singleton(final_info.get("episode", {}))
                        success_once = bool(
                            unwrap_singleton(episode_metrics.get("success_once", False))
                        )
                        success_at_end = bool(
                            unwrap_singleton(episode_metrics.get("success_at_end", False))
                        )
                        truncated_flag = True
                        break
                if truncated_flag:
                    break

            seed = int(reset_kwargs.get("seed"))
            results.append(
                dict(
                    episode_idx=episode_idx,
                    seed=seed,
                    overlap=(seed in old_train_seeds),
                    success_once=success_once,
                    success_at_end=success_at_end,
                    episode_len=episode_len,
                )
            )
            print(
                f"[episode {episode_idx:03d}] seed={seed} overlap={int(seed in old_train_seeds)} "
                f"success_once={int(success_once)} success_at_end={int(success_at_end)} len={episode_len}"
            )

    envs.close()

    def summarize(items: list[dict[str, Any]]):
        if len(items) == 0:
            return dict(count=0, success_once_rate=float("nan"), success_at_end_rate=float("nan"))
        return dict(
            count=len(items),
            success_once_rate=float(np.mean([x["success_once"] for x in items])),
            success_at_end_rate=float(np.mean([x["success_at_end"] for x in items])),
        )

    overlap_items = [x for x in results if x["overlap"]]
    non_overlap_items = [x for x in results if not x["overlap"]]

    return dict(
        ckpt_path=ckpt_path,
        total=summarize(results),
        overlap=summarize(overlap_items),
        non_overlap=summarize(non_overlap_items),
        overlap_seeds=[x["seed"] for x in overlap_items],
        non_overlap_success_once_seeds=[x["seed"] for x in non_overlap_items if x["success_once"]],
        results=results,
        heldout_seeds=heldout_seeds,
        old_train_seeds=sorted(old_train_seeds),
    )


@dataclass
class EvalArgs:
    ckpt_path: str
    eval_meta_path: str = "demos/data_1_preprocessed/3D_points_0.1/data_1_3D_points_0.1_eval.json"
    old_train_meta_path: str = "demos/data_1/data_1.json"
    num_old_train_demos: int = 100
    max_eval_episodes: int | None = None
    env_id: str = "PickCube-v1"
    obs_mode: str = "rgb+depth"
    max_episode_steps: int = 100
    action_norm_path: str = "demos/data_1/data_1_norm.json"
    sim_backend: str = "physx_cpu"
    cuda: bool = True
    output_json: str | None = None


if __name__ == "__main__":
    args = tyro.cli(EvalArgs)
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    result = evaluate_checkpoint(
        ckpt_path=args.ckpt_path,
        eval_meta_path=args.eval_meta_path,
        old_train_meta_path=args.old_train_meta_path,
        num_old_train_demos=args.num_old_train_demos,
        max_eval_episodes=args.max_eval_episodes,
        env_id=args.env_id,
        obs_mode=args.obs_mode,
        max_episode_steps=args.max_episode_steps,
        action_norm_path=args.action_norm_path,
        sim_backend=args.sim_backend,
        device=device,
    )
    print(json.dumps(result["total"], indent=2))
    print(json.dumps(result["overlap"], indent=2))
    print(json.dumps(result["non_overlap"], indent=2))
    if args.output_json is not None:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2))
        print(f"[saved] {out_path}")
