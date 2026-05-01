from collections import defaultdict
from typing import Any, Optional

import numpy as np
import torch
from tqdm import tqdm

from mani_skill.utils import common


def _normalize_reset_kwargs(reset_kwargs: Optional[dict[str, Any]]) -> dict[str, Any]:
    out = dict(reset_kwargs or {})
    seed = out.get("seed", None)
    if isinstance(seed, list):
        seed = seed[0] if len(seed) > 0 else None
    if seed is not None:
        out["seed"] = int(seed)
    if out.get("options", None) is None:
        out.pop("options", None)
    return out


def _reset_with_kwargs_batch(eval_envs, reset_kwargs_batch: list[dict[str, Any]]):
    reset_kwargs_batch = [
        _normalize_reset_kwargs(reset_kwargs) for reset_kwargs in reset_kwargs_batch
    ]
    if len(reset_kwargs_batch) == 1:
        return eval_envs.reset(**reset_kwargs_batch[0])

    reset_keys = set()
    for reset_kwargs in reset_kwargs_batch:
        reset_keys.update(reset_kwargs.keys())
    non_seed_keys = sorted(reset_keys - {"seed"})

    batched_kwargs: dict[str, Any] = {}
    for key in non_seed_keys:
        value = reset_kwargs_batch[0].get(key, None)
        if any(reset_kwargs.get(key, None) != value for reset_kwargs in reset_kwargs_batch):
            raise ValueError(
                "Batched eval reset only supports identical non-seed reset kwargs. "
                "Set --num-eval-envs 1 for per-demo options."
            )
        if value is not None:
            batched_kwargs[key] = value

    seeds = [reset_kwargs.get("seed", None) for reset_kwargs in reset_kwargs_batch]
    if any(seed is not None for seed in seeds):
        if any(seed is None for seed in seeds):
            raise ValueError("Batched eval reset requires every demo to have a seed.")
        batched_kwargs["seed"] = [int(seed) for seed in seeds]
    return eval_envs.reset(**batched_kwargs)


def evaluate(
    n: int,
    agent,
    eval_envs,
    device,
    sim_backend: str,
    progress_bar: bool = True,
    reset_kwargs_list: Optional[list[dict[str, Any]]] = None,
    eval_subgoal_data: Optional[dict[str, Any]] = None,
):
    agent.eval()
    if progress_bar:
        pbar = tqdm(total=n)
    with torch.no_grad():
        eval_metrics = defaultdict(list)
        use_demo_resets = reset_kwargs_list is not None and len(reset_kwargs_list) > 0

        def reset_eval_envs(ep_start_idx: int):
            if not use_demo_resets:
                return eval_envs.reset()
            reset_kwargs_batch = [
                reset_kwargs_list[(ep_start_idx + i) % len(reset_kwargs_list)]
                for i in range(eval_envs.num_envs)
            ]
            return _reset_with_kwargs_batch(eval_envs, reset_kwargs_batch)

        if use_demo_resets:
            print(
                f"[eval-split] evaluating with {len(reset_kwargs_list)} reset kwargs "
                "loaded from eval demo path"
            )
        subgoal_list = None
        if eval_subgoal_data is not None:
            subgoal_list = eval_subgoal_data.get("subgoal_flat")
            if subgoal_list is None or len(subgoal_list) == 0:
                raise ValueError("eval_subgoal_data must contain non-empty subgoal_flat")
            if use_demo_resets and len(subgoal_list) != len(reset_kwargs_list):
                raise ValueError(
                    "subgoal_flat length must match reset_kwargs_list length: "
                    f"{len(subgoal_list)} vs {len(reset_kwargs_list)}"
                )

        def attach_subgoal(obs_dict: dict, ep_start_idx: int):
            if subgoal_list is None:
                return obs_dict
            batch_size = int(obs_dict["state"].shape[0])
            obs_horizon = int(obs_dict["state"].shape[1])
            subgoal_batch = [
                torch.as_tensor(
                    subgoal_list[(ep_start_idx + i) % len(subgoal_list)],
                    device=obs_dict["state"].device,
                    dtype=obs_dict["state"].dtype,
                )
                for i in range(batch_size)
            ]
            obs_dict["subgoal"] = torch.stack(subgoal_batch, dim=0).unsqueeze(1).repeat(
                1, obs_horizon, 1
            )
            return obs_dict

        obs, info = reset_eval_envs(0)
        eps_count = 0
        while eps_count < n:
            obs = common.to_tensor(obs, device)
            obs = attach_subgoal(obs, eps_count)
            action_seq = agent.get_action(obs)
            if sim_backend == "physx_cpu":
                action_seq = action_seq.cpu().numpy()
            for i in range(action_seq.shape[1]):
                obs, rew, terminated, truncated, info = eval_envs.step(action_seq[:, i])
                if truncated.any():
                    break

            if truncated.any():
                assert truncated.all() == truncated.any(), "all episodes should truncate at the same time for fair evaluation with other algorithms"
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
                if use_demo_resets and eps_count < n:
                    obs, info = reset_eval_envs(eps_count)
    agent.train()
    for k in eval_metrics.keys():
        eval_metrics[k] = np.stack(eval_metrics[k])
    return eval_metrics
