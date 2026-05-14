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
MULTI_MASK_TYPE_TO_BASE = {
    "multi_random": "random_mask",
    "multi_points": "points",
    "multi_3D_points": "3D_points",
    "multi_pose": "pose_motion_planning",
}
MASK_TYPES_REQUIRING_RATIO_RANGE = set(MULTI_MASK_TYPE_TO_BASE.keys())
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
    *MASK_TYPES_REQUIRING_RATIO_RANGE,
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
    *MASK_TYPES_REQUIRING_RATIO_RANGE,
}


def is_multi_ratio_mask_type(mask_type: str) -> bool:
    return str(mask_type) in MULTI_MASK_TYPE_TO_BASE


def resolve_base_mask_type(mask_type: str) -> str:
    return MULTI_MASK_TYPE_TO_BASE.get(str(mask_type), str(mask_type))


def parse_retain_ratio_range(raw_param: Any, mask_type: str) -> tuple[float, float]:
    if raw_param is None:
        raise ValueError(f"mask_type={mask_type!r} requires retain_ratio range [start, end]")
    if isinstance(raw_param, str):
        text = raw_param.strip()
        if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
            text = text[1:-1].strip()
        if text.startswith("[") and text.endswith("]"):
            try:
                import ast

                raw_param = ast.literal_eval(text)
            except Exception as exc:
                raise ValueError(
                    f"failed to parse retain_ratio range for {mask_type!r}: {raw_param!r}"
                ) from exc
        elif "," in text:
            raw_param = [item.strip() for item in text.split(",")]
        else:
            raise ValueError(
                f"retain_ratio range for {mask_type!r} must be [start, end], got {raw_param!r}"
            )
    if not isinstance(raw_param, (list, tuple)) or len(raw_param) != 2:
        raise ValueError(
            f"retain_ratio range for {mask_type!r} must be [start, end], got {raw_param!r}"
        )
    start, end = float(raw_param[0]), float(raw_param[1])
    if not (0.0 <= start <= end <= 1.0):
        raise ValueError(
            f"retain_ratio range for {mask_type!r} must satisfy 0 <= start <= end <= 1, got {raw_param!r}"
        )
    return start, end


def validate_mask_config(
    mask_type: str,
    retain_ratio: float | None = None,
    mask_seq_len: int | None = None,
    retain_ratio_range: Any | None = None,
) -> None:
    if mask_type not in SUPPORTED_MASK_TYPES:
        raise ValueError(
            f"Unsupported mask_type={mask_type!r}. Supported: {sorted(SUPPORTED_MASK_TYPES)}"
        )
    if mask_type in MASK_TYPES_REQUIRING_RATIO_RANGE:
        if retain_ratio_range is not None:
            parse_retain_ratio_range(retain_ratio_range, mask_type)
            return
        mask_type = resolve_base_mask_type(mask_type)
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
    retain_ratio_range = None
    mask_seq_len = None
    if mask_type in MASK_TYPES_REQUIRING_RATIO_RANGE:
        retain_ratio_range = parse_retain_ratio_range(raw_param, mask_type)
    elif mask_type in MASK_TYPES_REQUIRING_RATIO:
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
        retain_ratio_range=retain_ratio_range,
    )
    return {
        "mask_type": mask_type,
        "base_mask_type": resolve_base_mask_type(mask_type),
        "is_multi_ratio": bool(mask_type in MASK_TYPES_REQUIRING_RATIO_RANGE),
        "ratio": ratio,
        "retain_ratio": retain_ratio,
        "retain_ratio_range": (
            [float(retain_ratio_range[0]), float(retain_ratio_range[1])]
            if retain_ratio_range is not None
            else None
        ),
        "mask_seq_len": mask_seq_len,
    }


def validate_mixed_mask_config(
    num_mask_type: int,
    mask_type_list: list[str],
    mask_composition_list: list[float] | None = None,
    mask_ratio_list: list[Any] | None = None,
) -> None:
    if mask_composition_list is None:
        mask_composition_list = []
    if mask_ratio_list is None:
        mask_ratio_list = []

    if int(num_mask_type) < 0:
        raise ValueError(f"num_mask_type must be non-negative, got {num_mask_type}")
    if int(num_mask_type) == 0:
        if (
            len(mask_type_list) != 0
            or len(mask_composition_list) != 0
            or len(mask_ratio_list) != 0
        ):
            raise ValueError(
                "num_mask_type=0 expects empty mask_type_list/mask_composition_list/mask_ratio_list"
            )
        return

    if len(mask_type_list) != int(num_mask_type):
        raise ValueError(
            f"len(mask_type_list)={len(mask_type_list)} != num_mask_type={num_mask_type}"
        )
    if len(mask_composition_list) != int(num_mask_type):
        raise ValueError(
            f"len(mask_composition_list)={len(mask_composition_list)} != num_mask_type={num_mask_type}"
        )
    if len(mask_ratio_list) != int(num_mask_type):
        raise ValueError(
            f"len(mask_ratio_list)={len(mask_ratio_list)} != num_mask_type={num_mask_type}"
        )

    total_ratio = 0.0
    for mask_type, ratio, raw_param in zip(
        mask_type_list, mask_composition_list, mask_ratio_list
    ):
        spec = build_mask_spec(mask_type=mask_type, raw_param=raw_param, ratio=ratio)
        total_ratio += float(spec["ratio"])
    if not np.isclose(total_ratio, 1.0, atol=1e-6):
        raise ValueError(
            f"mask_composition_list must sum to 1, got {total_ratio:.8f}"
        )


def _ensure_2d_action(action: np.ndarray, action_dim: int = 7) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32)
    action_dim = int(action_dim)
    if action_dim not in (6, 7):
        raise ValueError(f"action_dim must be 6 or 7, got {action_dim}")
    if action.ndim != 2:
        raise ValueError(f"action must be 2D, got shape {action.shape}")
    if action.shape[1] != action_dim:
        raise ValueError(f"expected action dim {action_dim}, got shape {action.shape}")
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
    action_dim: int = 7,
) -> Tuple[np.ndarray, np.ndarray]:
    mask_type = resolve_base_mask_type(mask_type)
    validate_mask_config(
        mask_type=mask_type,
        retain_ratio=retain_ratio,
        mask_seq_len=mask_seq_len,
    )
    action = _ensure_2d_action(action, action_dim=action_dim)
    n, m = action.shape
    masked = np.full_like(action, fill_value=np.float32(masked_value))
    keep_mask = np.zeros((n, m), dtype=bool)

    if mask_type == "none":
        return masked, keep_mask

    if mask_type == "full":
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
