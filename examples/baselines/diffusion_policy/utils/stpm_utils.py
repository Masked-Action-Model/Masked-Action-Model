import torch

from mani_skill.utils import common
from utils.progress_utils import (
    build_mas_long_window_from_future,
    build_mas_short_window_from_future,
)


_DEBUG_PRINT_ONCE_TAGS = set()


def _debug_print_once(tag: str, message: str):
    if tag not in _DEBUG_PRINT_ONCE_TAGS:
        print(message)
        _DEBUG_PRINT_ONCE_TAGS.add(tag)


def _progress_delta_from_traj_len(traj_len: int) -> float:
    if traj_len <= 0:
        raise ValueError(f"traj_len must be positive, got {traj_len}")
    if traj_len == 1:
        return 0.0
    return 1.0 / float(traj_len - 1)


PICKCUBE_QPOS_DIM = 9
PICKCUBE_QVEL_DIM = 9
PICKCUBE_IS_GRASPED_DIM = 1
PICKCUBE_TCP_POSE_DIM = 7
PICKCUBE_GOAL_POS_DIM = 3
PICKCUBE_ROLLOUT_STATE_DIM = (
    PICKCUBE_QPOS_DIM
    + PICKCUBE_QVEL_DIM
    + PICKCUBE_IS_GRASPED_DIM
    + PICKCUBE_TCP_POSE_DIM
    + PICKCUBE_GOAL_POS_DIM
)


def validate_stpm_eval_setup(stpm_encoder, stpm_n_obs_steps: int, stpm_frame_gap: int):
    if stpm_encoder is None:
        raise ValueError(
            "stpm_encoder is required for only-mas evaluation. "
            "Pass a valid STPM checkpoint from the training entrypoint."
        )
    if getattr(stpm_encoder, "camera_name", None) != "base_camera":
        raise ValueError(
            "Only STPM configs with a single 'base_camera' are supported for "
            f"only-mas evaluation, got camera_name={getattr(stpm_encoder, 'camera_name', None)!r}."
        )
    if int(getattr(stpm_encoder, "state_dim", -1)) != PICKCUBE_ROLLOUT_STATE_DIM:
        raise ValueError(
            "Only PickCube-compatible STPM checkpoints are supported right now. "
            f"Expected state_dim={PICKCUBE_ROLLOUT_STATE_DIM}, got {getattr(stpm_encoder, 'state_dim', None)}."
        )
    if int(stpm_n_obs_steps) <= 0:
        raise ValueError(f"stpm_n_obs_steps must be positive, got {stpm_n_obs_steps}")
    if int(stpm_frame_gap) <= 0:
        raise ValueError(f"stpm_frame_gap must be positive, got {stpm_frame_gap}")
    _debug_print_once(
        "check5_validate",
        f"[check5] STPM eval setup validated: camera=base_camera, "
        f"checkpoint_state_dim={int(getattr(stpm_encoder, 'state_dim', -1))}, "
        f"expected_rollout_state_dim={PICKCUBE_ROLLOUT_STATE_DIM}",
    )


def prepare_batched_rollout_obs(obs, device, obs_horizon: int):
    obs = common.to_tensor(obs, device)

    if "rgb" not in obs or "state" not in obs:
        raise ValueError(
            f"Expected rollout obs to contain keys rgb/state, got {sorted(obs.keys())}"
        )

    if obs["rgb"].ndim == 4:
        obs["rgb"] = obs["rgb"].unsqueeze(0)
    if obs["state"].ndim == 2:
        obs["state"] = obs["state"].unsqueeze(0)

    if obs["rgb"].ndim != 5 or obs["state"].ndim != 3:
        raise ValueError(
            "Unexpected rollout obs shapes after batching: "
            f"rgb={tuple(obs['rgb'].shape)}, state={tuple(obs['state'].shape)}"
        )
    if (
        obs["rgb"].shape[1] != obs_horizon
        or obs["state"].shape[1] != obs_horizon
    ):
        raise ValueError(
            f"Expected obs_horizon={obs_horizon} in rollout obs, got "
            f"rgb={tuple(obs['rgb'].shape)}, state={tuple(obs['state'].shape)}"
        )

    return obs


def _extract_single_frame(stacked_obs: dict, env_idx: int, frame_idx: int):
    return {
        "rgb": stacked_obs["rgb"][env_idx, frame_idx].clone(),
        "state": stacked_obs["state"][env_idx, frame_idx].clone(),
    }


def init_env_histories_from_reset_obs(stacked_obs: dict, obs_horizon: int):
    histories = []
    num_envs = stacked_obs["state"].shape[0]
    for env_idx in range(num_envs):
        history = dict(rgb=[], state=[])
        for frame_idx in range(obs_horizon):
            frame = _extract_single_frame(stacked_obs, env_idx, frame_idx)
            for key in history.keys():
                history[key].append(frame[key])
        histories.append(history)
    return histories


def append_latest_rollout_frame(histories, stacked_obs: dict):
    for env_idx, history in enumerate(histories):
        frame = _extract_single_frame(stacked_obs, env_idx, -1)
        for key in history.keys():
            history[key].append(frame[key])


def _reorder_pickcube_rollout_state_for_stpm(
    rollout_state: torch.Tensor, expected_state_dim: int
):
    if rollout_state.ndim != 1:
        raise ValueError(
            f"Expected single rollout state frame to be 1D, got shape {tuple(rollout_state.shape)}"
        )
    if rollout_state.numel() != PICKCUBE_ROLLOUT_STATE_DIM:
        raise ValueError(
            "Current only-mas evaluator only supports PickCube-compatible rollout "
            f"states with dim={PICKCUBE_ROLLOUT_STATE_DIM}, got {rollout_state.numel()}."
        )
    if expected_state_dim != PICKCUBE_ROLLOUT_STATE_DIM:
        raise ValueError(
            "Current only-mas evaluator only supports PickCube-compatible STPM "
            f"state_dim={PICKCUBE_ROLLOUT_STATE_DIM}, got {expected_state_dim}."
        )

    qpos_end = PICKCUBE_QPOS_DIM
    qvel_end = qpos_end + PICKCUBE_QVEL_DIM
    is_grasped_end = qvel_end + PICKCUBE_IS_GRASPED_DIM
    tcp_pose_end = is_grasped_end + PICKCUBE_TCP_POSE_DIM
    goal_pos_end = tcp_pose_end + PICKCUBE_GOAL_POS_DIM

    qpos = rollout_state[:qpos_end]
    qvel = rollout_state[qpos_end:qvel_end]
    is_grasped = rollout_state[qvel_end:is_grasped_end]
    tcp_pose = rollout_state[is_grasped_end:tcp_pose_end]
    goal_pos = rollout_state[tcp_pose_end:goal_pos_end]

    reordered = torch.cat((qpos, qvel, goal_pos, tcp_pose, is_grasped), dim=0)
    _debug_print_once(
        "check5_reorder",
        f"[check5] STPM state reorder validated: input_dim={rollout_state.numel()}, "
        f"reordered_dim={reordered.numel()}, order=(qpos,qvel,goal_pos,tcp_pose,is_grasped)",
    )
    return reordered


def _build_stpm_rgb_frame(rgb_frame: torch.Tensor):
    if rgb_frame.ndim != 3:
        raise ValueError(
            f"Expected rgb frame to be 3D, got rgb={tuple(rgb_frame.shape)}"
        )

    if rgb_frame.shape[-1] == 3:
        rgb_chw = rgb_frame.permute(2, 0, 1)
    elif rgb_frame.shape[0] == 3:
        rgb_chw = rgb_frame
    else:
        raise ValueError(
            "Only single-camera 'base_camera' rollout observations are supported. "
            f"Expected rgb channel dim 3, got shape {tuple(rgb_frame.shape)}."
        )

    if rgb_chw.shape[0] != 3:
        raise ValueError(
            f"Expected base_camera rgb channel count 3, got {rgb_chw.shape[0]}."
        )

    return rgb_chw.to(dtype=torch.float32)


def _sample_stpm_history_indices(
    anchor_idx: int, history_len: int, n_obs_steps: int, frame_gap: int
):
    relative_indices = range(-n_obs_steps * frame_gap, 1, frame_gap)
    sampled = [
        min(max(anchor_idx + relative_idx, 0), history_len - 1)
        for relative_idx in relative_indices
    ]
    expected_seq_len = n_obs_steps + 1
    if len(sampled) != expected_seq_len:
        raise ValueError(
            f"Expected STPM sequence length {expected_seq_len}, got indices {sampled}"
        )
    return sampled


def predict_progress_from_histories(
    histories,
    step_ptr: torch.Tensor,
    obs_horizon: int,
    stpm_encoder,
    stpm_n_obs_steps: int,
    stpm_frame_gap: int,
    target_device,
    target_dtype,
):
    rgb_windows = []
    state_windows = []

    for env_idx, history in enumerate(histories):
        expected_history_len = obs_horizon + int(step_ptr[env_idx].item())
        actual_history_len = len(history["state"])
        if actual_history_len != expected_history_len:
            raise ValueError(
                f"History length mismatch for env {env_idx}: expected {expected_history_len}, got {actual_history_len}"
            )
        if env_idx == 0:
            _debug_print_once(
                "check6_histories_mlp8",
                f"[check6] history/step_ptr sync (MLP8 path): "
                f"step_ptr={int(step_ptr[env_idx].item())}, obs_horizon={obs_horizon}, "
                f"expected_history_len={expected_history_len}, actual_history_len={actual_history_len}",
            )

        for offset in range(obs_horizon):
            anchor_idx = int(step_ptr[env_idx].item()) + offset
            sampled_indices = _sample_stpm_history_indices(
                anchor_idx=anchor_idx,
                history_len=actual_history_len,
                n_obs_steps=stpm_n_obs_steps,
                frame_gap=stpm_frame_gap,
            )

            rgb_seq = []
            state_seq = []
            for hist_idx in sampled_indices:
                rgb_seq.append(_build_stpm_rgb_frame(history["rgb"][hist_idx]))
                state_seq.append(
                    _reorder_pickcube_rollout_state_for_stpm(
                        history["state"][hist_idx],
                        expected_state_dim=stpm_encoder.state_dim,
                    )
                )

            rgb_windows.append(torch.stack(rgb_seq, dim=0))
            state_windows.append(torch.stack(state_seq, dim=0))

    rgb_batch = torch.stack(rgb_windows, dim=0)
    state_batch = torch.stack(state_windows, dim=0)
    progress = stpm_encoder.predict_progress(rgbd=rgb_batch, state=state_batch)
    return progress.to(device=target_device, dtype=target_dtype).reshape(
        len(histories), obs_horizon, 1
    )


def predict_current_progress_from_histories(
    histories,
    step_ptr: torch.Tensor,
    obs_horizon: int,
    stpm_encoder,
    stpm_n_obs_steps: int,
    stpm_frame_gap: int,
    target_device,
    target_dtype,
):
    rgb_windows = []
    state_windows = []

    for env_idx, history in enumerate(histories):
        expected_history_len = obs_horizon + int(step_ptr[env_idx].item())
        actual_history_len = len(history["state"])
        if actual_history_len != expected_history_len:
            raise ValueError(
                f"History length mismatch for env {env_idx}: expected {expected_history_len}, got {actual_history_len}"
            )
        if env_idx == 0:
            _debug_print_once(
                "check6_histories_current",
                f"[check6] history/step_ptr sync (current-progress path): "
                f"step_ptr={int(step_ptr[env_idx].item())}, obs_horizon={obs_horizon}, "
                f"expected_history_len={expected_history_len}, actual_history_len={actual_history_len}",
            )

        anchor_idx = int(step_ptr[env_idx].item()) + obs_horizon - 1
        sampled_indices = _sample_stpm_history_indices(
            anchor_idx=anchor_idx,
            history_len=actual_history_len,
            n_obs_steps=stpm_n_obs_steps,
            frame_gap=stpm_frame_gap,
        )

        rgb_seq = []
        state_seq = []
        for hist_idx in sampled_indices:
            rgb_seq.append(_build_stpm_rgb_frame(history["rgb"][hist_idx]))
            state_seq.append(
                _reorder_pickcube_rollout_state_for_stpm(
                    history["state"][hist_idx],
                    expected_state_dim=stpm_encoder.state_dim,
                )
            )

        rgb_windows.append(torch.stack(rgb_seq, dim=0))
        state_windows.append(torch.stack(state_seq, dim=0))

    rgb_batch = torch.stack(rgb_windows, dim=0)
    state_batch = torch.stack(state_windows, dim=0)
    progress = stpm_encoder.predict_progress(rgbd=rgb_batch, state=state_batch)
    return progress.to(device=target_device, dtype=target_dtype).reshape(
        len(histories), 1
    )


def build_mas_condition_batch(
    mas_flat_list,
    traj_ids: torch.Tensor,
    obs_horizon: int,
    device,
    dtype,
):
    obs_mas = torch.stack([mas_flat_list[int(i.item())] for i in traj_ids], dim=0)
    return obs_mas.to(device=device, dtype=dtype).unsqueeze(1).repeat(1, obs_horizon, 1)


def build_mas_progress_condition_batch(
    mas_list,
    traj_ids: torch.Tensor,
    current_progress: torch.Tensor,
    obs_horizon: int,
    device,
    dtype,
):
    if current_progress.ndim != 2 or current_progress.shape[1] != 1:
        raise ValueError(
            f"Expected current_progress shape (B, 1), got {tuple(current_progress.shape)}"
        )

    conds = []
    for batch_idx, traj_id in enumerate(traj_ids):
        mas_t = mas_list[int(traj_id.item())].to(device=device, dtype=dtype)
        if mas_t.ndim != 2 or mas_t.shape[-1] <= 0:
            raise ValueError(f"Expected MAS trajectory to be 2D, got {tuple(mas_t.shape)}")

        progress_col = mas_t[:, -1]
        target = current_progress[batch_idx, 0].to(device=device, dtype=dtype)
        nearest_idx = int(torch.argmin(torch.abs(progress_col - target)).item())
        start = nearest_idx - obs_horizon + 1
        mas_seq = mas_t[max(0, start) : nearest_idx + 1]
        if start < 0:
            pad_len = -start
            mas_seq = torch.cat((mas_seq[:1].repeat(pad_len, 1), mas_seq), dim=0)
        if mas_seq.shape != (obs_horizon, mas_t.shape[-1]):
            raise ValueError(
                f"Expected MAS progress condition shape {(obs_horizon, mas_t.shape[-1])}, got {tuple(mas_seq.shape)}"
            )
        conds.append(mas_seq)
    return torch.stack(conds, dim=0)


def build_ap_progress_condition_batch(
    current_progress: torch.Tensor,
    traj_lengths,
    traj_ids: torch.Tensor,
    obs_horizon: int,
    device,
    dtype,
):
    if current_progress.ndim != 2 or current_progress.shape[1] != 1:
        raise ValueError(
            f"Expected current_progress shape (B, 1), got {tuple(current_progress.shape)}"
        )
    current_progress = current_progress.to(device=device, dtype=dtype)

    conds = []
    for batch_idx, traj_id in enumerate(traj_ids):
        traj_len = int(traj_lengths[int(traj_id.item())])
        if traj_len <= 0:
            raise ValueError(f"traj_len must be positive, got {traj_len}")

        delta = _progress_delta_from_traj_len(traj_len)
        anchor_offsets = (
            torch.arange(obs_horizon, device=device, dtype=dtype) - (obs_horizon - 1)
        ) * delta
        future_offsets = torch.arange(8, device=device, dtype=dtype) * delta
        anchor_progress = current_progress[batch_idx : batch_idx + 1] + anchor_offsets.unsqueeze(0)
        ap_progress = anchor_progress.unsqueeze(-1) + future_offsets.view(1, 1, 8)
        conds.append(torch.clamp(ap_progress, min=0.0, max=1.0).squeeze(0))
    return torch.stack(conds, dim=0)


def build_mas_window_condition_batch(
    mas_list,
    traj_lengths,
    traj_ids: torch.Tensor,
    current_progress: torch.Tensor,
    obs_horizon: int,
    mas_horizon: int,
    device,
    dtype,
):
    if current_progress.ndim != 2 or current_progress.shape[1] != 1:
        raise ValueError(
            f"Expected current_progress shape (B, 1), got {tuple(current_progress.shape)}"
        )
    if obs_horizon <= 0:
        raise ValueError(f"obs_horizon must be positive, got {obs_horizon}")
    if mas_horizon <= 0:
        raise ValueError(f"mas_horizon must be positive, got {mas_horizon}")

    current_progress = current_progress.to(device=device, dtype=dtype)
    conds = []
    for batch_idx, traj_id in enumerate(traj_ids):
        local_traj_id = int(traj_id.item())
        mas_t = mas_list[local_traj_id].to(device=device, dtype=dtype)
        traj_len = int(traj_lengths[local_traj_id])
        if traj_len <= 0:
            raise ValueError(f"traj_len must be positive, got {traj_len}")
        if mas_t.ndim != 2 or mas_t.shape[-1] <= 0:
            raise ValueError(f"Expected MAS trajectory to be 2D, got {tuple(mas_t.shape)}")

        progress_col = mas_t[:, -1]
        delta = _progress_delta_from_traj_len(traj_len)
        anchor_progress = current_progress[batch_idx, 0] + (
            torch.arange(obs_horizon, device=device, dtype=dtype) - (obs_horizon - 1)
        ) * delta
        anchor_progress = torch.clamp(anchor_progress, min=0.0, max=1.0)

        sample_windows = []
        for anchor_p in anchor_progress:
            nearest_idx = int(torch.argmin(torch.abs(progress_col - anchor_p)).item())
            sample_windows.append(
                build_mas_long_window_from_future(
                    mas_t,
                    current_step=nearest_idx,
                    long_window_horizon=mas_horizon,
                )
            )
        sample_windows = torch.stack(sample_windows, dim=0)
        conds.append(sample_windows.reshape(obs_horizon, -1))
    return torch.stack(conds, dim=0)


def build_dual_mas_window_condition_batch(
    mas_list,
    traj_lengths,
    traj_ids: torch.Tensor,
    current_progress: torch.Tensor,
    obs_horizon: int,
    long_window_horizon: int,
    short_window_horizon: int,
    device,
    dtype,
    long_window_backward_length: int = 0,
    long_window_forward_length: int | None = None,
):
    if current_progress.ndim != 2 or current_progress.shape[1] != 1:
        raise ValueError(
            f"Expected current_progress shape (B, 1), got {tuple(current_progress.shape)}"
        )
    if obs_horizon <= 0:
        raise ValueError(f"obs_horizon must be positive, got {obs_horizon}")
    if long_window_horizon < 0:
        raise ValueError(
            f"long_window_horizon must be non-negative, got {long_window_horizon}"
        )
    if short_window_horizon < 0:
        raise ValueError(
            f"short_window_horizon must be non-negative, got {short_window_horizon}"
        )
    if long_window_backward_length < 0:
        raise ValueError(
            "long_window_backward_length must be non-negative, "
            f"got {long_window_backward_length}"
        )
    if long_window_forward_length is not None and long_window_forward_length < 0:
        raise ValueError(
            "long_window_forward_length must be non-negative, "
            f"got {long_window_forward_length}"
        )
    effective_long_window_horizon = (
        long_window_horizon
        if long_window_forward_length is None
        else long_window_backward_length + long_window_forward_length
    )

    current_progress = current_progress.to(device=device, dtype=dtype)
    long_conds = []
    short_conds = []
    for batch_idx, traj_id in enumerate(traj_ids):
        local_traj_id = int(traj_id.item())
        mas_t = mas_list[local_traj_id].to(device=device, dtype=dtype)
        traj_len = int(traj_lengths[local_traj_id])
        if traj_len <= 0:
            raise ValueError(f"traj_len must be positive, got {traj_len}")
        if mas_t.ndim != 2 or mas_t.shape[-1] <= 0:
            raise ValueError(f"Expected MAS trajectory to be 2D, got {tuple(mas_t.shape)}")

        progress_col = mas_t[:, -1]
        delta = _progress_delta_from_traj_len(traj_len)
        anchor_progress = current_progress[batch_idx, 0] + (
            torch.arange(obs_horizon, device=device, dtype=dtype) - (obs_horizon - 1)
        ) * delta
        anchor_progress = torch.clamp(anchor_progress, min=0.0, max=1.0)

        sample_long_windows = []
        sample_short_windows = []
        for anchor_p in anchor_progress:
            nearest_idx = int(torch.argmin(torch.abs(progress_col - anchor_p)).item())
            if effective_long_window_horizon > 0:
                sample_long_windows.append(
                    build_mas_long_window_from_future(
                        mas_t,
                        current_step=nearest_idx,
                        long_window_horizon=long_window_horizon,
                        long_window_backward_length=long_window_backward_length,
                        long_window_forward_length=long_window_forward_length,
                    )
                )
            if short_window_horizon > 0:
                sample_short_windows.append(
                    build_mas_short_window_from_future(
                        mas_t,
                        current_step=nearest_idx,
                        short_window_horizon=short_window_horizon,
                    )
                )
        if effective_long_window_horizon > 0:
            sample_long_windows = torch.stack(sample_long_windows, dim=0)
        else:
            sample_long_windows = mas_t.new_empty((obs_horizon, 0, mas_t.shape[-1]))
        if short_window_horizon > 0:
            sample_short_windows = torch.stack(sample_short_windows, dim=0)
        else:
            sample_short_windows = mas_t.new_empty(
                (obs_horizon, 0, mas_t.shape[-1])
            )
        long_conds.append(sample_long_windows.reshape(obs_horizon, -1))
        short_conds.append(sample_short_windows.reshape(obs_horizon, -1))

    return {
        "mas_long_window": torch.stack(long_conds, dim=0),
        "mas_short_window": torch.stack(short_conds, dim=0),
    }


def append_episode_metrics(eval_metrics: dict, info: dict):
    if isinstance(info["final_info"], dict):
        for k, v in info["final_info"]["episode"].items():
            eval_metrics[k].append(v.float().cpu().numpy())
    else:
        for final_info in info["final_info"]:
            for k, v in final_info["episode"].items():
                eval_metrics[k].append(v)
