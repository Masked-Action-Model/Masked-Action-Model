from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from maniskill_dataset import (  # noqa: E402
    infer_camera_info_from_h5,
    infer_camera_names_from_h5,
    infer_state_paths_from_h5,
    infer_state_schema_from_h5,
)


def parse_list_arg(value: str | None) -> list[str] | None:
    if value is None or len(str(value).strip()) == 0:
        return None
    text = str(value).strip()
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
            if isinstance(parsed, list):
                return [str(item) for item in parsed]
        except Exception:
            pass
    if "," in text:
        return [item.strip() for item in text.split(",") if item.strip()]
    return [text]


def parse_config_list(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        if value.strip().lower() == "auto":
            return None
        return parse_list_arg(value)
    try:
        parsed = list(value)
    except TypeError:
        return [str(value)]
    parsed = [str(item) for item in parsed]
    if len(parsed) == 0 or any(item.strip().lower() == "auto" for item in parsed):
        return None
    return parsed


def _load_state_path(traj_group: h5py.Group, path: str) -> np.ndarray:
    try:
        value = np.asarray(traj_group[path][()], dtype=np.float32)
    except KeyError:
        raise KeyError(
            f"Configured STPM state path {path!r} is missing from trajectory {traj_group.name}."
        ) from None
    if value.ndim == 1:
        value = value.reshape(-1, 1)
    elif value.ndim > 2:
        value = value.reshape(value.shape[0], -1)
    return value


def build_state(traj_group: h5py.Group, state_paths: list[str]) -> np.ndarray:
    return np.concatenate(
        [_load_state_path(traj_group, state_path) for state_path in state_paths],
        axis=-1,
    ).astype(np.float32)


def compute_state_stats(h5_path: Path, state_paths: list[str]) -> tuple[dict[str, Any], int, int]:
    all_states = []
    with h5py.File(h5_path, "r") as dataset:
        traj_keys = sorted(
            [key for key in dataset.keys() if key.startswith("traj_")],
            key=lambda key: int(key.split("_")[1]),
        )
        if not traj_keys:
            raise ValueError(f"No traj_* groups found in {h5_path}")

        for traj_key in traj_keys:
            state = build_state(dataset[traj_key], state_paths=state_paths)
            all_states.append(state)

    stacked = np.concatenate(all_states, axis=0).astype(np.float32)
    std = np.std(stacked, axis=0).clip(min=1e-2)
    stats = {
        "mean": np.mean(stacked, axis=0).tolist(),
        "std": std.tolist(),
        "q01": np.quantile(stacked, 0.01, axis=0).tolist(),
        "q99": np.quantile(stacked, 0.99, axis=0).tolist(),
    }
    return stats, int(stacked.shape[1]), int(stacked.shape[0])


def load_config_defaults(config_path: Path | None):
    if config_path is None:
        return None, None, None, None, None
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(config_path)
    h5_path = Path(str(cfg.general.repo_id))
    output_path = Path(str(cfg.general.state_norm_path))
    state_paths = parse_config_list(getattr(cfg.general, "state_paths", None)) or []
    camera_names = parse_config_list(getattr(cfg.general, "camera_names", None))
    return cfg, h5_path, output_path, state_paths, camera_names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate STPM state normalization stats from the exact state_paths "
            "used by a ManiSkill H5 dataset/config."
        )
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--h5_path", type=Path, default=None)
    parser.add_argument("--output_path", type=Path, default=None)
    parser.add_argument(
        "--state-paths",
        type=str,
        default=None,
        help="JSON/Python/comma list of H5 paths. Defaults to config state_paths, then all obs state leaves.",
    )
    parser.add_argument(
        "--state_dim",
        type=int,
        default=None,
        help="Optional sanity check. Must equal the inferred state dim; no truncation is performed.",
    )
    parser.add_argument(
        "--write-config",
        action="store_true",
        help="When --config is provided, write resolved state/camera metadata back to that config.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing output_path.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg, cfg_h5_path, cfg_output_path, cfg_state_paths, cfg_camera_names = load_config_defaults(args.config)
    h5_path = args.h5_path or cfg_h5_path
    output_path = args.output_path or cfg_output_path
    if h5_path is None:
        raise ValueError("Provide --h5_path or --config with general.repo_id.")
    if output_path is None:
        raise ValueError("Provide --output_path or --config with general.state_norm_path.")
    h5_path = h5_path.resolve()
    output_path = output_path.resolve()

    if not h5_path.exists():
        raise FileNotFoundError(f"H5 dataset not found: {h5_path}")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"output_path already exists: {output_path}. Add --overwrite.")

    state_paths = parse_list_arg(args.state_paths)
    if state_paths is None:
        state_paths = cfg_state_paths if cfg_state_paths else infer_state_paths_from_h5(h5_path)
    state_schema = infer_state_schema_from_h5(h5_path, state_paths=state_paths)
    inferred_camera_names = infer_camera_names_from_h5(h5_path)
    camera_names = cfg_camera_names if cfg_camera_names is not None else inferred_camera_names
    missing_camera_names = sorted(set(camera_names) - set(inferred_camera_names))
    if missing_camera_names:
        raise ValueError(
            f"Configured camera_names missing from H5: {missing_camera_names}; "
            f"available={inferred_camera_names}"
        )
    all_camera_info = infer_camera_info_from_h5(h5_path)
    camera_info = {camera_name: all_camera_info[camera_name] for camera_name in camera_names}

    stats, state_dim, num_state_frames = compute_state_stats(h5_path, state_paths=state_paths)
    if args.state_dim is not None and int(args.state_dim) != state_dim:
        raise ValueError(
            f"--state_dim={args.state_dim} does not match state_paths dim={state_dim}. "
            "Refusing to truncate or pad STPM state normalizer stats."
        )

    output = {
        "norm_stats": {"state": stats},
        "meta": {
            "source_h5": str(h5_path),
            "state_paths": state_paths,
            "state_schema": state_schema,
            "state_dim": state_dim,
            "num_state_frames": num_state_frames,
            "camera_names": camera_names,
            "camera_info": camera_info,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    if args.write_config:
        if cfg is None or args.config is None:
            raise ValueError("--write-config requires --config.")
        from omegaconf import OmegaConf

        cfg.general.repo_id = str(h5_path)
        cfg.general.state_norm_path = str(output_path)
        cfg.general.state_paths = state_paths
        cfg.general.state_schema = state_schema
        cfg.general.camera_names = camera_names
        cfg.general.camera_info = camera_info
        cfg.model.state_dim = state_dim
        OmegaConf.save(cfg, args.config)

    print(
        f"Saved STPM state normalizer stats to {output_path} "
        f"(state_dim={state_dim}, paths={state_paths}, cameras={camera_names})"
    )


if __name__ == "__main__":
    main()
