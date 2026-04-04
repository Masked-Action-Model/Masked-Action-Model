from __future__ import annotations

from typing import Tuple

import numpy as np


MASKED_VALUE = 0.0
MASK_TYPES_REQUIRING_RATIO = {
    "points",
    "3D_points",
    "pose_motion_planning",
    "pose_AnyGrasp",
    "random_mask",
}
MASK_TYPES_REQUIRING_SEQ_LEN = {
    "2D_partial_trajectory",
    "local_planner",
}
SUPPORTED_MASK_TYPES = {
    "none",
    "full",
    "2D_video_trajectory",
    "2D_image_trajectory",
    "2D_partial_trajectory",
    "pose_AnyGrasp",
    "pose_motion_planning",
    "points",
    "3D_points",
    "auto_regressive",
    "local_planner",
    "random_mask",
}


def validate_mask_config(
    mask_type: str,
    retain_ratio: float | None = None,
    mask_seq_len: int | None = None,
) -> None:
    if mask_type not in SUPPORTED_MASK_TYPES:
        raise ValueError(
            f"Unsupported mask_type={mask_type!r}. Supported: {sorted(SUPPORTED_MASK_TYPES)}"
        )
    if mask_type in MASK_TYPES_REQUIRING_RATIO:
        if retain_ratio is None:
            raise ValueError(f"mask_type={mask_type!r} requires retain_ratio")
        if not (0.0 <= retain_ratio <= 1.0):
            raise ValueError(
                f"retain_ratio must be in [0, 1], got {retain_ratio} for {mask_type!r}"
            )
    if mask_type in MASK_TYPES_REQUIRING_SEQ_LEN:
        if mask_seq_len is None:
            raise ValueError(f"mask_type={mask_type!r} requires mask_seq_len")
        if mask_seq_len <= 0:
            raise ValueError(
                f"mask_seq_len must be positive for {mask_type!r}, got {mask_seq_len}"
            )


def _ensure_2d_action(action: np.ndarray) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32)
    if action.ndim != 2:
        raise ValueError(f"action must be 2D, got shape {action.shape}")
    if action.shape[1] != 7:
        raise ValueError(f"expected action dim 7, got shape {action.shape}")
    return action


def _retain_random_rows(mask: np.ndarray, rng: np.random.Generator, retain_num: int) -> np.ndarray:
    if retain_num <= 0 or mask.shape[0] == 0:
        return np.empty((0,), dtype=np.int64)
    retain_num = min(retain_num, mask.shape[0])
    retain_idx = np.arange(mask.shape[0], dtype=np.int64)
    rng.shuffle(retain_idx)
    return retain_idx[:retain_num]


def apply_mask_to_actions(
    action: np.ndarray,
    mask_type: str,
    rng: np.random.Generator,
    retain_ratio: float | None = None,
    mask_seq_len: int | None = None,
    masked_value: float = MASKED_VALUE,
) -> Tuple[np.ndarray, np.ndarray]:
    validate_mask_config(
        mask_type=mask_type,
        retain_ratio=retain_ratio,
        mask_seq_len=mask_seq_len,
    )
    action = _ensure_2d_action(action)
    n, m = action.shape
    masked = np.full_like(action, fill_value=np.float32(masked_value))
    keep_mask = np.zeros((n, m), dtype=bool)

    if mask_type in {"none", "full"}:
        masked = action.copy()
        keep_mask[:] = True
        return masked, keep_mask

    if n == 0:
        return masked, keep_mask

    if mask_type in {"2D_video_trajectory", "2D_image_trajectory"}:
        keep_mask[:, 0:2] = True
    elif mask_type == "2D_partial_trajectory":
        if mask_seq_len >= n:
            raise ValueError(
                f"mask_seq_len ({mask_seq_len}) must be smaller than trajectory length ({n})"
            )
        start_idx = int(rng.integers(0, n - mask_seq_len + 1))
        keep_mask[start_idx : start_idx + mask_seq_len, 0:2] = True
    elif mask_type == "pose_AnyGrasp":
        retain_idx = _retain_random_rows(keep_mask, rng, retain_num=1)
        keep_mask[retain_idx, :] = True
    elif mask_type == "pose_motion_planning":
        retain_num = int(n * float(retain_ratio))
        retain_idx = _retain_random_rows(keep_mask, rng, retain_num=retain_num)
        keep_mask[retain_idx, :] = True
    elif mask_type == "points":
        retain_num = int(n * float(retain_ratio))
        retain_idx = _retain_random_rows(keep_mask, rng, retain_num=retain_num)
        keep_mask[retain_idx, 0:2] = True
    elif mask_type == "3D_points":
        retain_num = int(n * float(retain_ratio))
        retain_idx = _retain_random_rows(keep_mask, rng, retain_num=retain_num)
        keep_mask[retain_idx, 0:3] = True
    elif mask_type == "auto_regressive":
        row_idx = int(rng.integers(0, n))
        col_idx = int(rng.integers(0, m))
        keep_mask[:row_idx, :] = True
        keep_mask[row_idx, :col_idx] = True
    elif mask_type == "local_planner":
        if mask_seq_len >= n:
            raise ValueError(
                f"mask_seq_len ({mask_seq_len}) must be smaller than trajectory length ({n})"
            )
        keep_mask[:, :] = True
        start_idx = int(rng.integers(0, n - mask_seq_len + 1))
        keep_mask[start_idx : start_idx + mask_seq_len, :] = False
    elif mask_type == "random_mask":
        total_entries = n * m
        retain_num = int(total_entries * float(retain_ratio))
        if retain_num > 0:
            flat_indices = np.arange(total_entries, dtype=np.int64)
            rng.shuffle(flat_indices)
            flat_keep = flat_indices[:retain_num]
            keep_mask.reshape(-1)[flat_keep] = True
    else:
        raise AssertionError(f"unreachable mask_type={mask_type!r}")

    masked[keep_mask] = action[keep_mask]
    return masked.astype(np.float32), keep_mask
