from __future__ import annotations

from typing import Any, Tuple

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
    "mix0",
    "2D_partial_trajectory",
    "pose_AnyGrasp",
    "pose_motion_planning",
    "points",
    "3D_points",
    "auto_regressive",
    "local_planner",
    "random_mask",
}
MIXED_SUPPORTED_MASK_TYPES = {
    "none",
    "full",
    "2D_video_trajectory",
    "2D_image_trajectory",
    "mix0",
    "2D_partial_trajectory",
    "pose_motion_planning",
    "points",
    "3D_points",
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


def build_mask_spec(
    mask_type: str,
    raw_param: Any = None,
    ratio: float = 1.0,
) -> dict[str, Any]:
    if mask_type not in MIXED_SUPPORTED_MASK_TYPES:
        raise ValueError(
            f"Unsupported mixed mask_type={mask_type!r}. Supported: "
            f"{sorted(MIXED_SUPPORTED_MASK_TYPES)}"
        )
    ratio = float(ratio)
    if not (0.0 <= ratio <= 1.0):
        raise ValueError(f"mask ratio must be in [0, 1], got {ratio} for {mask_type!r}")

    retain_ratio = None
    mask_seq_len = None
    if mask_type in MASK_TYPES_REQUIRING_RATIO:
        if raw_param is None:
            raise ValueError(f"mask_type={mask_type!r} requires retain_ratio")
        retain_ratio = float(raw_param)
        if not (0.0 < retain_ratio <= 1.0):
            raise ValueError(
                f"retain_ratio must be in (0, 1], got {retain_ratio} for {mask_type!r}"
            )
    elif mask_type in MASK_TYPES_REQUIRING_SEQ_LEN:
        if raw_param is None:
            raise ValueError(f"mask_type={mask_type!r} requires mask_seq_len")
        try:
            mask_seq_len = int(raw_param)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"mask_seq_len must be an integer-like value for {mask_type!r}, got {raw_param!r}"
            ) from exc
        if not (1 <= mask_seq_len <= 100):
            raise ValueError(
                f"mask_seq_len must be in [1, 100], got {mask_seq_len} for {mask_type!r}"
            )

    validate_mask_config(
        mask_type=mask_type,
        retain_ratio=retain_ratio,
        mask_seq_len=mask_seq_len,
    )
    return {
        "mask_type": mask_type,
        "ratio": ratio,
        "retain_ratio": retain_ratio,
        "mask_seq_len": mask_seq_len,
    }


def validate_mixed_mask_config(
    num_mask_type: int,
    mask_type_list: list[str],
    mask_type_ratio_list: list[float],
    mask_param_list: list[Any],
) -> None:
    if int(num_mask_type) < 0:
        raise ValueError(f"num_mask_type must be non-negative, got {num_mask_type}")
    if int(num_mask_type) == 0:
        if len(mask_type_list) != 0 or len(mask_type_ratio_list) != 0 or len(mask_param_list) != 0:
            raise ValueError(
                "num_mask_type=0 expects empty mask_type_list/mask_type_ratio_list/mask_param_list"
            )
        return

    if len(mask_type_list) != int(num_mask_type):
        raise ValueError(
            f"len(mask_type_list)={len(mask_type_list)} != num_mask_type={num_mask_type}"
        )
    if len(mask_type_ratio_list) != int(num_mask_type):
        raise ValueError(
            f"len(mask_type_ratio_list)={len(mask_type_ratio_list)} != num_mask_type={num_mask_type}"
        )
    if len(mask_param_list) != int(num_mask_type):
        raise ValueError(
            f"len(mask_param_list)={len(mask_param_list)} != num_mask_type={num_mask_type}"
        )

    total_ratio = 0.0
    for mask_type, ratio, raw_param in zip(
        mask_type_list, mask_type_ratio_list, mask_param_list
    ):
        spec = build_mask_spec(mask_type=mask_type, raw_param=raw_param, ratio=ratio)
        total_ratio += float(spec["ratio"])
    if not np.isclose(total_ratio, 1.0, atol=1e-6):
        raise ValueError(
            f"mask_type_ratio_list must sum to 1, got {total_ratio:.8f}"
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
    elif mask_type == "mix0":
        if n < 4:
            raise ValueError(
                f"mix0 requires trajectory length >= 4, got {n}"
            )
        keep_mask[:, 0:2] = True
        retain_idx = _retain_random_rows(keep_mask, rng, retain_num=4)
        keep_mask[retain_idx[0], :] = True
        keep_mask[retain_idx[1:], 0:3] = True
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
