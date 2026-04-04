import torch


MAS_STEP_DIM = 8


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


def build_mas_long_window_from_future(
    mas_t: torch.Tensor, current_step: int, long_window_horizon: int
) -> torch.Tensor:
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

    current_idx = min(max(int(current_step), 0), mas_t.shape[0] - 1)
    start = current_idx - obs_horizon + 1
    long_windows = []
    short_windows = []
    for h_offset in range(obs_horizon):
        anchor_idx = min(max(start + h_offset, 0), mas_t.shape[0] - 1)
        if long_window_horizon > 0:
            long_windows.append(
                build_mas_long_window_from_future(
                    mas_t,
                    current_step=anchor_idx,
                    long_window_horizon=long_window_horizon,
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

    if long_window_horizon > 0:
        long_window = torch.stack(long_windows, dim=0)
    else:
        long_window = mas_t.new_empty((obs_horizon, 0, MAS_STEP_DIM))
    if short_window_horizon > 0:
        short_window = torch.stack(short_windows, dim=0)
    else:
        short_window = mas_t.new_empty((obs_horizon, 0, MAS_STEP_DIM))
    if long_window.shape != (obs_horizon, long_window_horizon, MAS_STEP_DIM):
        raise ValueError(
            "Expected long-window horizon shape "
            f"{(obs_horizon, long_window_horizon, MAS_STEP_DIM)}, got {tuple(long_window.shape)}"
        )
    if short_window.shape != (obs_horizon, short_window_horizon, MAS_STEP_DIM):
        raise ValueError(
            "Expected short-window horizon shape "
            f"{(obs_horizon, short_window_horizon, MAS_STEP_DIM)}, got {tuple(short_window.shape)}"
        )
    return long_window, short_window
