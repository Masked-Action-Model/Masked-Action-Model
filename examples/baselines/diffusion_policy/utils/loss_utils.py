from typing import Optional

import torch
import torch.nn.functional as F


def slice_action_mask_sequence(
    mask_traj: torch.Tensor,
    start: int,
    end: int,
    action_len: int,
    act_dim: int,
    pred_horizon: int,
) -> torch.Tensor:
    action_mask_seq = mask_traj[max(0, start) : min(end, action_len), :act_dim]
    if action_mask_seq.ndim != 2:
        raise ValueError(
            f"expected action mask sequence to be 2D, got shape {tuple(action_mask_seq.shape)}"
        )
    if action_mask_seq.shape[1] != act_dim:
        raise ValueError(
            f"expected action mask dim {act_dim}, got shape {tuple(action_mask_seq.shape)}"
        )
    if action_mask_seq.shape[0] == 0:
        raise ValueError(
            f"empty action mask slice for start={start}, end={end}, action_len={action_len}"
        )
    if start < 0:
        pad_len = -start
        pad_mask = action_mask_seq[0].unsqueeze(0).repeat(pad_len, 1)
        action_mask_seq = torch.cat([pad_mask, action_mask_seq], dim=0)
    if end > action_len:
        pad_len = end - action_len
        pad_mask = action_mask_seq[-1].unsqueeze(0).repeat(pad_len, 1)
        action_mask_seq = torch.cat([action_mask_seq, pad_mask], dim=0)
    if action_mask_seq.shape != (pred_horizon, act_dim):
        raise ValueError(
            f"expected action mask shape {(pred_horizon, act_dim)}, got {tuple(action_mask_seq.shape)}"
        )
    return action_mask_seq


def _masked_mean(
    values: torch.Tensor,
    mask: torch.Tensor,
) -> Optional[torch.Tensor]:
    if mask.shape != values.shape:
        raise ValueError(
            f"mask shape {tuple(mask.shape)} must match values shape {tuple(values.shape)}"
        )
    if not torch.any(mask):
        return None
    return values[mask].mean()


def compute_mask_weighted_noise_mse(
    noise_pred: torch.Tensor,
    noise: torch.Tensor,
    action_mask: Optional[torch.Tensor],
    loss_mode: str,
    mask_area_weight: float,
) -> torch.Tensor:
    if loss_mode == "average":
        return F.mse_loss(noise_pred, noise)
    if loss_mode != "weighted":
        raise ValueError(f"unsupported loss_mode={loss_mode!r}")
    if action_mask is None:
        raise ValueError("weighted loss requires action_mask")
    if action_mask.shape != noise_pred.shape:
        raise ValueError(
            f"action_mask shape {tuple(action_mask.shape)} must match noise_pred shape {tuple(noise_pred.shape)}"
        )
    if not 0.0 <= float(mask_area_weight) <= 1.0:
        raise ValueError(
            f"loss_mask_area_weight must be in [0, 1], got {mask_area_weight}"
        )

    per_elem_loss = (noise_pred - noise).square()
    known_mask = action_mask.to(device=per_elem_loss.device, dtype=torch.bool)
    unknown_mask = ~known_mask
    known_loss = _masked_mean(per_elem_loss, known_mask)
    unknown_loss = _masked_mean(per_elem_loss, unknown_mask)

    known_weight = float(mask_area_weight)
    unknown_weight = 1.0 - known_weight
    weighted_loss = None
    total_weight = 0.0

    if known_loss is not None and known_weight > 0.0:
        weighted_loss = known_loss * known_weight
        total_weight += known_weight
    if unknown_loss is not None and unknown_weight > 0.0:
        term = unknown_loss * unknown_weight
        weighted_loss = term if weighted_loss is None else weighted_loss + term
        total_weight += unknown_weight
    if weighted_loss is not None and total_weight > 0.0:
        return weighted_loss / total_weight

    present_losses = [
        region_loss
        for region_loss in (known_loss, unknown_loss)
        if region_loss is not None
    ]
    if present_losses:
        return torch.stack(present_losses).mean()
    return per_elem_loss.mean()
