from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import h5py
import numpy as np


STRING_DTYPE = h5py.string_dtype(encoding="utf-8")


def ensure_parent_dir(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, payload: Any) -> None:
    path = ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def list_traj_keys(file: h5py.File, num_traj: int | None = None) -> list[str]:
    keys = sorted(
        [key for key in file.keys() if key.startswith("traj_")],
        key=lambda key: int(key.split("_")[-1]),
    )
    if num_traj is not None:
        if num_traj <= 0:
            raise ValueError(f"num_traj must be positive, got {num_traj}")
        if num_traj > len(keys):
            raise ValueError(
                f"num_traj ({num_traj}) exceeds available trajectories ({len(keys)})"
            )
        keys = keys[:num_traj]
    return keys


def write_string_dataset(
    group: h5py.Group,
    key: str,
    value: str | Sequence[str],
) -> None:
    if isinstance(value, str):
        data = np.asarray(value, dtype=STRING_DTYPE)
    else:
        data = np.asarray(list(value), dtype=STRING_DTYPE)
    group.create_dataset(key, data=data)
