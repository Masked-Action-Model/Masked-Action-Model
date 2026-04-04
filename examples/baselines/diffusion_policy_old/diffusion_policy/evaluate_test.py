from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm

from mani_skill.utils import common


def evaluate(
    n: int,
    agent,
    eval_envs,
    device,
    sim_backend: str,
    progress_bar: bool = True,
    reset_seed: int = None,
    reset_kwargs_list: list[dict] = None,
):
    agent.eval()
    if progress_bar:
        pbar = tqdm(total=n)
    with torch.no_grad():
        eval_metrics = defaultdict(list)
        if reset_kwargs_list is not None and len(reset_kwargs_list) > 0:
            if eval_envs.num_envs != 1:
                raise ValueError(
                    "reset_kwargs_list evaluation only supports num_envs=1."
                )
            print(
                f"[eval-test] evaluating with {len(reset_kwargs_list)} demo reset kwargs"
            )
        elif reset_seed is not None and eval_envs.num_envs > 1:
            print(
                f"[eval-test] reset_seed={reset_seed} with num_envs={eval_envs.num_envs}: "
                "only env 0 uses that exact seed; other envs will use derived seeds."
            )
        eps_count = 0

        def reset_env_for_episode(ep_idx: int):
            if reset_kwargs_list is not None and len(reset_kwargs_list) > 0:
                reset_kwargs = dict(reset_kwargs_list[ep_idx % len(reset_kwargs_list)])
                return eval_envs.reset(**reset_kwargs)
            if reset_seed is None:
                return eval_envs.reset()
            return eval_envs.reset(seed=reset_seed)

        obs, info = reset_env_for_episode(eps_count)
        while eps_count < n:
            obs = common.to_tensor(obs, device)
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
                if eps_count < n:
                    obs, info = reset_env_for_episode(eps_count)
    agent.train()
    for k in eval_metrics.keys():
        eval_metrics[k] = np.stack(eval_metrics[k])
    return eval_metrics
