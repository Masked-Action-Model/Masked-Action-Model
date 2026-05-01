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
