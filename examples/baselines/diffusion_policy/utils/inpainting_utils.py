from __future__ import annotations

import torch

from mani_skill.utils import common


def _pad_last_step(seq: torch.Tensor, target_len: int) -> torch.Tensor:
    if seq.ndim != 2:
        raise ValueError(f"expected 2D sequence, got shape {tuple(seq.shape)}")
    if seq.shape[0] <= 0:
        raise ValueError("cannot pad empty sequence")
    if target_len <= 0:
        raise ValueError(f"target_len must be positive, got {target_len}")
    if seq.shape[0] >= target_len:
        return seq[:target_len]
    pad = seq[-1:].repeat(target_len - seq.shape[0], 1)
    return torch.cat((seq, pad), dim=0)


def build_current_inpaint_mas_mask(
    mas_list,
    mas_mask_list,
    traj_ids: torch.Tensor,
    current_progress: torch.Tensor,
    pred_horizon: int,
    device,
    dtype,
):
    if pred_horizon <= 0:
        raise ValueError(f"pred_horizon must be positive, got {pred_horizon}")
    if current_progress.ndim != 2 or current_progress.shape[1] != 1:
        raise ValueError(
            f"expected current_progress shape (B, 1), got {tuple(current_progress.shape)}"
        )
    if traj_ids.ndim != 1 or traj_ids.shape[0] != current_progress.shape[0]:
        raise ValueError(
            f"traj_ids shape must match batch, got traj_ids={tuple(traj_ids.shape)}, "
            f"current_progress={tuple(current_progress.shape)}"
        )

    current_progress = current_progress.to(device=device, dtype=dtype)
    inpaint_list = []
    inpaint_mask_list = []

    for batch_idx, traj_id in enumerate(traj_ids):
        local_traj_id = int(traj_id.item())
        mas_t = mas_list[local_traj_id].to(device=device, dtype=dtype)
        mask_t = mas_mask_list[local_traj_id].to(device=device, dtype=dtype)
        if mas_t.ndim != 2 or mas_t.shape[1] < 8:
            raise ValueError(f"expected augmented mas shape (T, >=8), got {tuple(mas_t.shape)}")
        if mask_t.shape != mas_t.shape:
            raise ValueError(
                f"expected mask shape {tuple(mas_t.shape)}, got {tuple(mask_t.shape)}"
            )

        progress_col = mas_t[:, -1]
        target_progress = torch.clamp(current_progress[batch_idx, 0], min=0.0, max=1.0)
        anchor_idx = int(torch.argmin(torch.abs(progress_col - target_progress)).item())
        end_idx = min(anchor_idx + pred_horizon, mas_t.shape[0])

        future_mas = mas_t[anchor_idx:end_idx, :7]
        future_mask = mask_t[anchor_idx:end_idx, :7]
        future_mas = _pad_last_step(future_mas, pred_horizon)
        future_mask = _pad_last_step(future_mask, pred_horizon)

        inpaint_list.append(future_mas)
        inpaint_mask_list.append(future_mask)

    return (
        torch.stack(inpaint_list, dim=0),
        torch.stack(inpaint_mask_list, dim=0),
    )


def build_inpaint_mas_mask(
    mas_inpaint: torch.Tensor,
    mas_inpaint_mask: torch.Tensor,
    obs_horizon: int,
    pred_horizon: int,
    action_dim: int,
):
    if mas_inpaint.ndim != 3 or mas_inpaint_mask.ndim != 3:
        raise ValueError(
            f"expected 3D tensors, got mas_inpaint={tuple(mas_inpaint.shape)}, "
            f"mask={tuple(mas_inpaint_mask.shape)}"
        )
    if mas_inpaint.shape != mas_inpaint_mask.shape:
        raise ValueError(
            f"mas_inpaint/mask shape mismatch: {tuple(mas_inpaint.shape)} vs "
            f"{tuple(mas_inpaint_mask.shape)}"
        )
    if obs_horizon <= 0 or pred_horizon <= 0 or action_dim <= 0:
        raise ValueError(
            f"invalid horizons/dim: obs_horizon={obs_horizon}, "
            f"pred_horizon={pred_horizon}, action_dim={action_dim}"
        )

    batch_size = mas_inpaint.shape[0]
    start = obs_horizon - 1
    if start >= pred_horizon:
        raise ValueError(
            f"obs_horizon - 1 must be < pred_horizon, got {obs_horizon} and {pred_horizon}"
        )

    action_known_0 = mas_inpaint.new_zeros((batch_size, pred_horizon, action_dim))
    action_mask = mas_inpaint.new_zeros((batch_size, pred_horizon, action_dim))
    known_dims = min(int(action_dim), int(mas_inpaint.shape[-1]), 7)
    copy_len = min(int(mas_inpaint.shape[1]), pred_horizon - start)

    action_known_0[:, start : start + copy_len, :known_dims] = mas_inpaint[
        :, :copy_len, :known_dims
    ]
    action_mask[:, start : start + copy_len, :known_dims] = (
        mas_inpaint_mask[:, :copy_len, :known_dims] > 0.5
    ).to(dtype=mas_inpaint.dtype)
    return action_known_0, action_mask


def overwrite_known_action(
    a_unknown_t: torch.Tensor,
    action_known_0: torch.Tensor,
    action_mask: torch.Tensor,
    timestep: int,
    scheduler,
):
    eps = torch.randn_like(action_known_0)
    t_batch = torch.full(
        (action_known_0.shape[0],),
        int(timestep),
        dtype=torch.long,
        device=action_known_0.device,
    )
    a_known_t = scheduler.add_noise(action_known_0, eps, t_batch)
    return action_mask * a_known_t + (1.0 - action_mask) * a_unknown_t


def overwrite_known_action_clean(
    a_unknown_0: torch.Tensor,
    action_known_0: torch.Tensor,
    action_mask: torch.Tensor,
):
    return action_mask * action_known_0 + (1.0 - action_mask) * a_unknown_0


def forward_step_ddpm(a_t: torch.Tensor, timestep: int, scheduler):
    next_t = int(timestep) + 1
    betas = scheduler.betas.to(device=a_t.device, dtype=a_t.dtype)
    beta_next = betas[next_t]
    alpha_next = 1.0 - beta_next
    eps = torch.randn_like(a_t)
    a_next = torch.sqrt(alpha_next) * a_t + torch.sqrt(beta_next) * eps
    return a_next, next_t


def forward_jump_j_steps(a_cur: torch.Tensor, t_cur: int, j: int, max_t: int, scheduler):
    if j <= 0:
        raise ValueError(f"jump length j must be positive, got {j}")
    a_jump = a_cur
    cur_t = int(t_cur)
    for _ in range(j):
        if cur_t >= max_t:
            break
        a_jump, cur_t = forward_step_ddpm(a_jump, cur_t, scheduler)
    return a_jump, cur_t


def _prepare_obs_for_agent(obs_seq):
    obs_seq = common.to_tensor(obs_seq, obs_seq["state"].device)
    obs_seq["rgb"] = obs_seq["rgb"].permute(0, 1, 4, 2, 3)
    obs_seq["depth"] = obs_seq["depth"].permute(0, 1, 4, 2, 3)
    return obs_seq


def _denormalize_action_seq(agent, action_seq_norm: torch.Tensor):
    if (
        agent.action_denorm_dims <= 0
        or agent.action_denorm_min is None
        or agent.action_denorm_max is None
    ):
        raise RuntimeError("action denormalizer must be configured before inpaint rollout")

    denorm_dims = min(int(agent.action_denorm_dims), int(action_seq_norm.shape[-1]))
    mins = agent.action_denorm_min[:denorm_dims].to(
        device=action_seq_norm.device,
        dtype=action_seq_norm.dtype,
    )
    maxs = agent.action_denorm_max[:denorm_dims].to(
        device=action_seq_norm.device,
        dtype=action_seq_norm.dtype,
    )
    action_seq = action_seq_norm.clone()
    action_seq[..., :denorm_dims] = mins + 0.5 * (
        action_seq[..., :denorm_dims] + 1.0
    ) * (maxs - mins)
    return action_seq


def dp_repaint_inference(
    agent,
    obs_seq,
    mas_inpaint: torch.Tensor,
    mas_inpaint_mask: torch.Tensor,
    jump_length: int,
    num_resample: int,
):
    if jump_length < 0:
        raise ValueError(f"jump_length must be >= 0, got {jump_length}")
    if num_resample < 0:
        raise ValueError(f"num_resample must be >= 0, got {num_resample}")
    if jump_length == 0 and num_resample != 0:
        raise ValueError("num_resample must be 0 when jump_length is 0")

    obs_seq = _prepare_obs_for_agent(obs_seq)
    batch_size = obs_seq["state"].shape[0]
    pred_horizon = int(agent.pred_horizon)
    action_dim = int(agent.act_dim)
    obs_horizon = int(agent.obs_horizon)
    max_t = int(agent.noise_scheduler.config.num_train_timesteps) - 1

    action_known_0, action_mask = build_inpaint_mas_mask(
        mas_inpaint=mas_inpaint.to(device=obs_seq["state"].device, dtype=obs_seq["state"].dtype),
        mas_inpaint_mask=mas_inpaint_mask.to(
            device=obs_seq["state"].device,
            dtype=obs_seq["state"].dtype,
        ),
        obs_horizon=obs_horizon,
        pred_horizon=pred_horizon,
        action_dim=action_dim,
    )

    with torch.no_grad():
        obs_cond = agent.obs_conditioning(obs_seq, eval_mode=True)
        a_t = torch.randn(
            (batch_size, pred_horizon, action_dim),
            device=obs_seq["state"].device,
            dtype=obs_seq["state"].dtype,
        )
        t = max_t
        no_known_action = bool(torch.count_nonzero(action_mask).item() == 0)

        if jump_length == 0:
            for k in agent.noise_scheduler.timesteps:
                t = int(k)
                t_batch = torch.full(
                    (batch_size,),
                    t,
                    dtype=torch.long,
                    device=a_t.device,
                )
                noise_pred = agent.noise_pred_net(a_t, t_batch, obs_cond)
                a_prev = agent.noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=t,
                    sample=a_t,
                ).prev_sample

                if not no_known_action:
                    if t > 0:
                        a_prev = overwrite_known_action(
                            a_unknown_t=a_prev,
                            action_known_0=action_known_0,
                            action_mask=action_mask,
                            timestep=t - 1,
                            scheduler=agent.noise_scheduler,
                        )
                    else:
                        a_prev = overwrite_known_action_clean(
                            a_unknown_0=a_prev,
                            action_known_0=action_known_0,
                            action_mask=action_mask,
                        )
                a_t = a_prev
            return a_t

        while t >= 0:
            steps_down = min(int(jump_length), t + 1)
            for _ in range(steps_down):
                t_batch = torch.full(
                    (batch_size,),
                    t,
                    dtype=torch.long,
                    device=a_t.device,
                )
                noise_pred = agent.noise_pred_net(a_t, t_batch, obs_cond)
                a_prev = agent.noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=t,
                    sample=a_t,
                ).prev_sample

                if not no_known_action:
                    if t > 0:
                        a_prev = overwrite_known_action(
                            a_unknown_t=a_prev,
                            action_known_0=action_known_0,
                            action_mask=action_mask,
                            timestep=t - 1,
                            scheduler=agent.noise_scheduler,
                        )
                    else:
                        a_prev = overwrite_known_action_clean(
                            a_unknown_0=a_prev,
                            action_known_0=action_known_0,
                            action_mask=action_mask,
                        )

                a_t = a_prev
                t -= 1
                if t < 0:
                    break
            if t < 0:
                break

            for _ in range(int(num_resample)):
                a_jump, t_jump = forward_jump_j_steps(
                    a_cur=a_t,
                    t_cur=t,
                    j=jump_length,
                    max_t=max_t,
                    scheduler=agent.noise_scheduler,
                )
                a_tmp = a_jump
                cur_t = t_jump

                while cur_t > t:
                    t_batch = torch.full(
                        (batch_size,),
                        cur_t,
                        dtype=torch.long,
                        device=a_tmp.device,
                    )
                    noise_pred = agent.noise_pred_net(a_tmp, t_batch, obs_cond)
                    a_prev = agent.noise_scheduler.step(
                        model_output=noise_pred,
                        timestep=cur_t,
                        sample=a_tmp,
                    ).prev_sample

                    if not no_known_action:
                        if cur_t > 0:
                            a_prev = overwrite_known_action(
                                a_unknown_t=a_prev,
                                action_known_0=action_known_0,
                                action_mask=action_mask,
                                timestep=cur_t - 1,
                                scheduler=agent.noise_scheduler,
                            )
                        else:
                            a_prev = overwrite_known_action_clean(
                                a_unknown_0=a_prev,
                                action_known_0=action_known_0,
                                action_mask=action_mask,
                            )

                    a_tmp = a_prev
                    cur_t -= 1

                a_t = a_tmp

    return a_t


def sample_inpaint_action_chunk(
    agent,
    obs_seq,
    mas_inpaint: torch.Tensor,
    mas_inpaint_mask: torch.Tensor,
    jump_length: int,
    num_resample: int,
):
    full_action_seq_norm = dp_repaint_inference(
        agent=agent,
        obs_seq=obs_seq,
        mas_inpaint=mas_inpaint,
        mas_inpaint_mask=mas_inpaint_mask,
        jump_length=jump_length,
        num_resample=num_resample,
    )
    start = int(agent.obs_horizon) - 1
    end = start + int(agent.act_horizon)
    exec_action_seq_norm = full_action_seq_norm[:, start:end]
    return _denormalize_action_seq(agent, exec_action_seq_norm)
