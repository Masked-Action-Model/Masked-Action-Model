import torch

try:
    from data_preprocess.utils.progress_utils import (
        MAS_ACTION_DIM,
        MAS_STEP_DIM,
        augment_mas_with_progress_torch,
        build_progress_column_torch,
        pad_augmented_mas_torch,
    )
except ModuleNotFoundError:
    from examples.baselines.diffusion_policy.data_preprocess.utils.progress_utils import (
        MAS_ACTION_DIM,
        MAS_STEP_DIM,
        augment_mas_with_progress_torch,
        build_progress_column_torch,
        pad_augmented_mas_torch,
    )


def build_progress_column(traj_len: int, mas_len: int, device: torch.device) -> torch.Tensor:
    return build_progress_column_torch(traj_len=traj_len, mas_len=mas_len, device=device)


def augment_mas_with_progress(mas_t: torch.Tensor, traj_len: int) -> torch.Tensor:
    return augment_mas_with_progress_torch(mas_t=mas_t, traj_len=traj_len)


def augment_mask_with_progress(
    mask_t: torch.Tensor,
    traj_len: int,
    mas_t: torch.Tensor | None = None,
) -> torch.Tensor:
    if mask_t.ndim != 2:
        raise ValueError(f"mask must be 2D, got shape {tuple(mask_t.shape)}")
    mask_t = mask_t.to(dtype=torch.float32)
    if mask_t.shape[-1] == MAS_STEP_DIM:
        return mask_t
    if mask_t.shape[-1] != MAS_ACTION_DIM:
        raise ValueError(
            f"expected mask last dim {MAS_ACTION_DIM} or {MAS_STEP_DIM}, got shape {tuple(mask_t.shape)}"
        )
    if mas_t is not None:
        progress_source = augment_mas_with_progress_torch(
            mas_t=mas_t.to(device=mask_t.device, dtype=torch.float32),
            traj_len=traj_len,
        )
        progress_col = progress_source[:, -1:].to(device=mask_t.device, dtype=mask_t.dtype)
    else:
        progress_col = build_progress_column_torch(
            traj_len=traj_len,
            mas_len=mask_t.shape[0],
            device=mask_t.device,
        ).to(dtype=mask_t.dtype)
    return torch.cat((mask_t, progress_col), dim=1)


def pad_mas_to_length(
    mas_t: torch.Tensor,
    target_len: int,
    traj_len: int,
    masked_value: float = 0.0,
) -> torch.Tensor:
    return pad_augmented_mas_torch(
        mas_t=mas_t,
        target_len=target_len,
        traj_len=traj_len,
        masked_value=masked_value,
    )


def pad_mask_to_length(
    mask_t: torch.Tensor,
    target_len: int,
    traj_len: int,
    mas_t: torch.Tensor | None = None,
) -> torch.Tensor:
    mask_t = augment_mask_with_progress(mask_t=mask_t, traj_len=traj_len, mas_t=mas_t)
    if target_len < mask_t.shape[0]:
        raise ValueError(
            f"target_len ({target_len}) cannot be smaller than mask length ({mask_t.shape[0]})"
        )
    if target_len == mask_t.shape[0]:
        return mask_t
    pad = mask_t.new_zeros((target_len - mask_t.shape[0], MAS_STEP_DIM))
    progress_col = build_progress_column_torch(
        traj_len=traj_len,
        mas_len=target_len,
        device=mask_t.device,
    ).to(dtype=mask_t.dtype)
    pad[:, -1] = progress_col[mask_t.shape[0] :, 0]
    return torch.cat((mask_t, pad), dim=0)


def build_progress_mlp_horizon_window(
    mas_t: torch.Tensor, current_step: int, obs_horizon: int
) -> torch.Tensor:
    if mas_t.ndim != 2:
        raise ValueError(f"mas must be 2D, got shape {tuple(mas_t.shape)}")
    if obs_horizon <= 0:
        raise ValueError(f"obs_horizon must be positive, got {obs_horizon}")
    if mas_t.shape[0] <= 0:
        raise ValueError(f"mas length must be positive, got shape {tuple(mas_t.shape)}")

    current_idx = min(max(int(current_step), 0), mas_t.shape[0] - 1)
    start = current_idx - obs_horizon + 1
    progress_seq = mas_t[max(0, start) : current_idx + 1, -1:].clone()
    if start < 0:
        pad_len = -start
        pad_progress = progress_seq[:1].repeat(pad_len, 1)
        progress_seq = torch.cat((pad_progress, progress_seq), dim=0)
    if progress_seq.shape != (obs_horizon, 1):
        raise ValueError(
            f"Expected progress window shape {(obs_horizon, 1)}, got {tuple(progress_seq.shape)}"
        )
    return progress_seq


def build_mas_horizon_window(
    mas_t: torch.Tensor, current_step: int, obs_horizon: int
) -> torch.Tensor:
    if mas_t.ndim != 2:
        raise ValueError(f"mas must be 2D, got shape {tuple(mas_t.shape)}")
    if mas_t.shape[-1] != MAS_STEP_DIM:
        raise ValueError(
            f"Expected augmented mas last dim {MAS_STEP_DIM}, got shape {tuple(mas_t.shape)}"
        )
    if obs_horizon <= 0:
        raise ValueError(f"obs_horizon must be positive, got {obs_horizon}")
    if mas_t.shape[0] <= 0:
        raise ValueError(f"mas length must be positive, got shape {tuple(mas_t.shape)}")

    current_idx = min(max(int(current_step), 0), mas_t.shape[0] - 1)
    start = current_idx - obs_horizon + 1
    mas_seq = mas_t[max(0, start) : current_idx + 1].clone()
    if start < 0:
        pad_len = -start
        pad_mas = mas_seq[:1].repeat(pad_len, 1)
        mas_seq = torch.cat((pad_mas, mas_seq), dim=0)
    if mas_seq.shape != (obs_horizon, MAS_STEP_DIM):
        raise ValueError(
            f"Expected MAS window shape {(obs_horizon, MAS_STEP_DIM)}, got {tuple(mas_seq.shape)}"
        )
    return mas_seq


def build_action_progress_window(
    mas_t: torch.Tensor, current_step: int, act_horizon: int, obs_horizon: int
) -> torch.Tensor:
    if act_horizon != MAS_STEP_DIM:
        raise ValueError(
            f"AP8 expects act_horizon == {MAS_STEP_DIM}, got {act_horizon}"
        )
    if obs_horizon <= 0:
        raise ValueError(f"obs_horizon must be positive, got {obs_horizon}")
    if mas_t.ndim != 2 or mas_t.shape[-1] != MAS_STEP_DIM:
        raise ValueError(
            f"Expected augmented mas shape (T, {MAS_STEP_DIM}), got {tuple(mas_t.shape)}"
        )

    current_idx = min(max(int(current_step), 0), mas_t.shape[0] - 1)
    progress_col = mas_t[:, -1]
    start = current_idx - obs_horizon + 1
    ap_windows = []
    for h_offset in range(obs_horizon):
        anchor_idx = min(max(start + h_offset, 0), progress_col.shape[0] - 1)
        future_progress = []
        for act_offset in range(act_horizon):
            future_idx = anchor_idx + act_offset
            if future_idx >= progress_col.shape[0]:
                future_progress.append(
                    torch.ones((), device=mas_t.device, dtype=mas_t.dtype)
                )
            else:
                future_progress.append(progress_col[future_idx])
        ap_windows.append(torch.stack(future_progress, dim=0))
    ap_window = torch.stack(ap_windows, dim=0)
    if ap_window.shape != (obs_horizon, MAS_STEP_DIM):
        raise ValueError(
            f"Expected AP window shape {(obs_horizon, MAS_STEP_DIM)}, got {tuple(ap_window.shape)}"
        )
    return ap_window


def build_mas_window_from_future(
    mas_t: torch.Tensor, current_step: int, mas_horizon: int
) -> torch.Tensor:
    if mas_t.ndim != 2 or mas_t.shape[-1] != MAS_STEP_DIM:
        raise ValueError(
            f"Expected augmented mas shape (T, {MAS_STEP_DIM}), got {tuple(mas_t.shape)}"
        )
    if mas_horizon <= 0:
        raise ValueError(f"mas_horizon must be positive, got {mas_horizon}")

    current_idx = min(max(int(current_step), 0), mas_t.shape[0] - 1)
    end = current_idx + mas_horizon
    mas_window = mas_t[current_idx:min(end, mas_t.shape[0])].clone()
    if end > mas_t.shape[0]:
        pad_len = end - mas_t.shape[0]
        mas_window = torch.cat((mas_window, mas_window[-1:].repeat(pad_len, 1)), dim=0)
    if mas_window.shape != (mas_horizon, MAS_STEP_DIM):
        raise ValueError(
            f"Expected mas-window shape {(mas_horizon, MAS_STEP_DIM)}, got {tuple(mas_window.shape)}"
        )
    return mas_window


def build_mas_window_around_step(
    mas_t: torch.Tensor,
    current_step: int,
    backward_length: int,
    forward_length: int,
) -> torch.Tensor:
    if mas_t.ndim != 2 or mas_t.shape[-1] != MAS_STEP_DIM:
        raise ValueError(
            f"Expected augmented mas shape (T, {MAS_STEP_DIM}), got {tuple(mas_t.shape)}"
        )
    if backward_length < 0:
        raise ValueError(f"backward_length must be non-negative, got {backward_length}")
    if forward_length <= 0:
        raise ValueError(f"forward_length must be positive, got {forward_length}")
    if mas_t.shape[0] <= 0:
        raise ValueError(f"mas length must be positive, got shape {tuple(mas_t.shape)}")

    current_idx = min(max(int(current_step), 0), mas_t.shape[0] - 1)
    offsets = range(-backward_length, forward_length)
    indices = [
        min(max(current_idx + offset, 0), mas_t.shape[0] - 1)
        for offset in offsets
    ]
    mas_window = mas_t[indices].clone()
    expected_horizon = backward_length + forward_length
    if mas_window.shape != (expected_horizon, MAS_STEP_DIM):
        raise ValueError(
            "Expected bidirectional mas-window shape "
            f"{(expected_horizon, MAS_STEP_DIM)}, got {tuple(mas_window.shape)}"
        )
    return mas_window


def build_mas_long_window_from_future(
    mas_t: torch.Tensor,
    current_step: int,
    long_window_horizon: int,
    long_window_backward_length: int = 0,
    long_window_forward_length: int | None = None,
) -> torch.Tensor:
    if long_window_forward_length is not None or long_window_backward_length > 0:
        forward_length = (
            long_window_horizon
            if long_window_forward_length is None
            else long_window_forward_length
        )
        return build_mas_window_around_step(
            mas_t=mas_t,
            current_step=current_step,
            backward_length=long_window_backward_length,
            forward_length=forward_length,
        )
    return build_mas_window_from_future(
        mas_t=mas_t, current_step=current_step, mas_horizon=long_window_horizon
    )


def build_mas_short_window_from_future(
    mas_t: torch.Tensor, current_step: int, short_window_horizon: int
) -> torch.Tensor:
    if short_window_horizon == 0:
        return mas_t.new_empty((0, MAS_STEP_DIM))
    return build_mas_window_from_future(
        mas_t=mas_t,
        current_step=current_step,
        mas_horizon=short_window_horizon,
    )


def build_mas_window_obs_horizon(
    mas_t: torch.Tensor, current_step: int, obs_horizon: int, mas_horizon: int
) -> torch.Tensor:
    if obs_horizon <= 0:
        raise ValueError(f"obs_horizon must be positive, got {obs_horizon}")

    current_idx = min(max(int(current_step), 0), mas_t.shape[0] - 1)
    start = current_idx - obs_horizon + 1
    mas_windows = []
    for h_offset in range(obs_horizon):
        anchor_idx = min(max(start + h_offset, 0), mas_t.shape[0] - 1)
        mas_windows.append(
            build_mas_window_from_future(
                mas_t, current_step=anchor_idx, mas_horizon=mas_horizon
            )
        )
    mas_window = torch.stack(mas_windows, dim=0)
    if mas_window.shape != (obs_horizon, mas_horizon, MAS_STEP_DIM):
        raise ValueError(
            "Expected mas-window horizon shape "
            f"{(obs_horizon, mas_horizon, MAS_STEP_DIM)}, got {tuple(mas_window.shape)}"
        )
    return mas_window


def build_dual_mas_window_obs_horizon(
    mas_t: torch.Tensor,
    current_step: int,
    obs_horizon: int,
    long_window_horizon: int,
    short_window_horizon: int,
    long_window_backward_length: int = 0,
    long_window_forward_length: int | None = None,
):
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

    current_idx = min(max(int(current_step), 0), mas_t.shape[0] - 1)
    start = current_idx - obs_horizon + 1
    long_windows = []
    short_windows = []
    for h_offset in range(obs_horizon):
        anchor_idx = min(max(start + h_offset, 0), mas_t.shape[0] - 1)
        if effective_long_window_horizon > 0:
            long_windows.append(
                build_mas_long_window_from_future(
                    mas_t,
                    current_step=anchor_idx,
                    long_window_horizon=long_window_horizon,
                    long_window_backward_length=long_window_backward_length,
                    long_window_forward_length=long_window_forward_length,
                )
            )
        if short_window_horizon > 0:
            short_windows.append(
                build_mas_short_window_from_future(
                    mas_t,
                    current_step=anchor_idx,
                    short_window_horizon=short_window_horizon,
                )
            )

    if effective_long_window_horizon > 0:
        long_window = torch.stack(long_windows, dim=0)
    else:
        long_window = mas_t.new_empty((obs_horizon, 0, MAS_STEP_DIM))
    if short_window_horizon > 0:
        short_window = torch.stack(short_windows, dim=0)
    else:
        short_window = mas_t.new_empty((obs_horizon, 0, MAS_STEP_DIM))
    if long_window.shape != (obs_horizon, effective_long_window_horizon, MAS_STEP_DIM):
        raise ValueError(
            "Expected long-window horizon shape "
            f"{(obs_horizon, effective_long_window_horizon, MAS_STEP_DIM)}, got {tuple(long_window.shape)}"
        )
    if short_window.shape != (obs_horizon, short_window_horizon, MAS_STEP_DIM):
        raise ValueError(
            "Expected short-window horizon shape "
            f"{(obs_horizon, short_window_horizon, MAS_STEP_DIM)}, got {tuple(short_window.shape)}"
        )
    return long_window, short_window
