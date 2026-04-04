from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, Sequence

import h5py
import numpy as np


EPS = 1e-8


def compute_global_min_max(array_iterable: Iterable[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    global_min = None
    global_max = None
    for array in array_iterable:
        array = np.asarray(array, dtype=np.float32)
        if array.ndim != 2:
            raise ValueError(f"expected 2D array when computing min/max, got shape {array.shape}")
        local_min = array.min(axis=0)
        local_max = array.max(axis=0)
        if global_min is None:
            global_min = local_min
            global_max = local_max
        else:
            global_min = np.minimum(global_min, local_min)
            global_max = np.maximum(global_max, local_max)
    if global_min is None or global_max is None:
        raise ValueError("cannot compute min/max from empty iterable")
    return global_min.astype(np.float32), global_max.astype(np.float32)


def normalize_selected_dims(
    data: np.ndarray,
    mins: np.ndarray,
    maxs: np.ndarray,
    dims: Sequence[int] | None = None,
) -> np.ndarray:
    data = np.asarray(data, dtype=np.float32)
    mins = np.asarray(mins, dtype=np.float32)
    maxs = np.asarray(maxs, dtype=np.float32)
    if mins.shape != maxs.shape or mins.ndim != 1:
        raise ValueError(f"mins/maxs must be 1D and shape-matched, got {mins.shape} and {maxs.shape}")

    normalized = data.copy()
    dims = list(range(mins.shape[0])) if dims is None else [int(dim) for dim in dims]
    for dim in dims:
        denom = float(maxs[dim] - mins[dim])
        if abs(denom) < EPS:
            normalized[..., dim] = 0.0
        else:
            normalized[..., dim] = 2.0 * (data[..., dim] - mins[dim]) / denom - 1.0
    return normalized.astype(np.float32)


def _load_action_stats_from_h5(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as f:
        if "meta" not in f:
            raise ValueError(f"missing meta group in h5 dataset: {path}")
        meta = f["meta"]
        if "action_min" not in meta or "action_max" not in meta:
            raise ValueError(
                f"h5 dataset meta must contain action_min/action_max: {path}"
            )
        mins = np.asarray(meta["action_min"][()], dtype=np.float32)
        maxs = np.asarray(meta["action_max"][()], dtype=np.float32)
    if mins.shape != maxs.shape or mins.ndim != 1 or mins.shape[0] == 0:
        raise ValueError(
            f"invalid action_min/action_max shape in h5 meta: {mins.shape}, {maxs.shape}"
        )
    return mins, maxs


def _load_action_stats_from_json(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "min" not in data or "max" not in data:
        raise ValueError(f"action norm json must contain min/max: {path}")
    mins = np.asarray(data["min"], dtype=np.float32)
    maxs = np.asarray(data["max"], dtype=np.float32)
    if mins.shape != maxs.shape or mins.ndim != 1 or mins.shape[0] == 0:
        raise ValueError(
            f"invalid action min/max shape in json: {mins.shape}, {maxs.shape}"
        )
    return mins, maxs


def load_action_stats_from_path(path: str) -> tuple[np.ndarray, np.ndarray]:
    if path is None or len(path.strip()) == 0:
        raise ValueError("action stats path is required")
    if not os.path.exists(path):
        raise FileNotFoundError(f"action stats path not found: {path}")
    lower_path = path.lower()
    if lower_path.endswith(".h5") or lower_path.endswith(".hdf5"):
        return _load_action_stats_from_h5(path)
    return _load_action_stats_from_json(path)
