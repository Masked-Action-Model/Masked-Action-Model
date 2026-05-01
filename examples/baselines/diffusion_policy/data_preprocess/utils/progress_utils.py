from __future__ import annotations

import numpy as np
import torch


MAS_ACTION_DIM = 7
MAS_STEP_DIM = MAS_ACTION_DIM + 1


def build_progress_array(
    traj_len: int,
    mas_len: int | None = None,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    if traj_len < 0:
        raise ValueError(f"traj_len must be non-negative, got {traj_len}")
    mas_len = traj_len if mas_len is None else int(mas_len)
    if mas_len <= 0:
        raise ValueError(f"mas_len must be positive, got {mas_len}")
    if traj_len > mas_len:
        raise ValueError(f"traj_len ({traj_len}) cannot exceed mas_len ({mas_len})")

    if traj_len == 0:
        progress = np.ones((mas_len,), dtype=dtype)
    else:
        progress = np.linspace(0.0, 1.0, num=traj_len, dtype=dtype)
        if traj_len < mas_len:
            progress = np.concatenate(
                [progress, np.ones((mas_len - traj_len,), dtype=dtype)],
                axis=0,
            )
    return progress.reshape(-1, 1)


def augment_mas_with_progress_np(mas: np.ndarray, traj_len: int | None = None) -> np.ndarray:
    mas = np.asarray(mas, dtype=np.float32)
    if mas.ndim != 2:
        raise ValueError(f"mas must be 2D, got shape {mas.shape}")
    if mas.shape[1] == MAS_STEP_DIM:
        return mas.astype(np.float32, copy=False)
    if mas.shape[1] != MAS_ACTION_DIM:
        raise ValueError(
            f"expected mas last dim {MAS_ACTION_DIM} or {MAS_STEP_DIM}, got {mas.shape}"
        )
    traj_len = mas.shape[0] if traj_len is None else int(traj_len)
    progress = build_progress_array(traj_len=traj_len, mas_len=mas.shape[0], dtype=np.float32)
    return np.concatenate([mas, progress], axis=1).astype(np.float32)


def build_progress_column_torch(
    traj_len: int,
    mas_len: int,
    device: torch.device,
) -> torch.Tensor:
    if mas_len <= 0:
        raise ValueError(f"mas_len must be positive, got {mas_len}")
    if traj_len < 0:
        raise ValueError(f"traj_len must be non-negative, got {traj_len}")
    if traj_len > mas_len:
        raise ValueError(
            f"traj_len ({traj_len}) cannot exceed mas_len ({mas_len}) when building progress."
        )

    if traj_len == 0:
        progress = torch.ones((mas_len,), device=device, dtype=torch.float32)
    else:
        progress = torch.linspace(0.0, 1.0, steps=traj_len, device=device)
        if traj_len < mas_len:
            progress = torch.cat(
                (
                    progress,
                    torch.ones(
                        (mas_len - traj_len,),
                        device=device,
                        dtype=progress.dtype,
                    ),
                ),
                dim=0,
            )
    return progress.unsqueeze(-1)


def augment_mas_with_progress_torch(mas_t: torch.Tensor, traj_len: int) -> torch.Tensor:
    if mas_t.ndim != 2:
        raise ValueError(f"mas must be 2D, got shape {tuple(mas_t.shape)}")
    if mas_t.shape[-1] == MAS_STEP_DIM:
        return mas_t
    if mas_t.shape[-1] != MAS_ACTION_DIM:
        raise ValueError(
            f"expected mas last dim {MAS_ACTION_DIM} or {MAS_STEP_DIM}, got shape {tuple(mas_t.shape)}"
        )
    progress_col = build_progress_column_torch(traj_len, mas_t.shape[0], mas_t.device).to(
        dtype=mas_t.dtype
    )
    return torch.cat((mas_t, progress_col), dim=1)


def pad_augmented_mas_np(
    mas: np.ndarray,
    target_len: int,
    traj_len: int | None = None,
    masked_value: float = 0.0,
) -> np.ndarray:
    mas = augment_mas_with_progress_np(mas, traj_len=traj_len)
    if target_len < mas.shape[0]:
        raise ValueError(
            f"target_len ({target_len}) cannot be smaller than mas length ({mas.shape[0]})"
        )
    if target_len == mas.shape[0]:
        return mas
    pad = np.full(
        (target_len - mas.shape[0], MAS_STEP_DIM),
        fill_value=np.float32(masked_value),
        dtype=np.float32,
    )
    pad[:, -1] = 1.0
    return np.concatenate((mas, pad), axis=0)


def pad_augmented_mas_torch(
    mas_t: torch.Tensor,
    target_len: int,
    traj_len: int | None = None,
    masked_value: float = 0.0,
) -> torch.Tensor:
    traj_len = int(mas_t.shape[0]) if traj_len is None else int(traj_len)
    mas_t = augment_mas_with_progress_torch(mas_t, traj_len=traj_len)
    if target_len < mas_t.shape[0]:
        raise ValueError(
            f"target_len ({target_len}) cannot be smaller than mas length ({mas_t.shape[0]})"
        )
    if target_len == mas_t.shape[0]:
        return mas_t
    pad = mas_t.new_full((target_len - mas_t.shape[0], MAS_STEP_DIM), masked_value)
    pad[:, -1] = 1.0
    return torch.cat((mas_t, pad), dim=0)
