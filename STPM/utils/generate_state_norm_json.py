import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def build_state(traj_group: h5py.Group) -> np.ndarray:
    agent_group = traj_group["obs"]["agent"]
    extra_group = traj_group["obs"]["extra"]
    return np.concatenate(
        [
            agent_group["qpos"][:].astype(np.float32),
            agent_group["qvel"][:].astype(np.float32),
            extra_group["goal_pos"][:].astype(np.float32),
            extra_group["tcp_pose"][:].astype(np.float32),
            extra_group["is_grasped"][:].astype(np.float32).reshape(-1, 1),
        ],
        axis=-1,
    )


def compute_state_stats(h5_path: Path, state_dim: int) -> dict:
    all_states = []
    with h5py.File(h5_path, "r") as dataset:
        traj_keys = sorted(
            [key for key in dataset.keys() if key.startswith("traj_")],
            key=lambda key: int(key.split("_")[1]),
        )
        if not traj_keys:
            raise ValueError(f"No traj_* groups found in {h5_path}")

        for traj_key in traj_keys:
            state = build_state(dataset[traj_key])
            if state.shape[1] < state_dim:
                raise ValueError(
                    f"Trajectory {traj_key} only has state dim {state.shape[1]}, expected at least {state_dim}"
                )
            all_states.append(state[:, :state_dim])

    stacked = np.concatenate(all_states, axis=0).astype(np.float32)
    std = np.std(stacked, axis=0).clip(min=1e-2)
    return {
        "mean": np.mean(stacked, axis=0).tolist(),
        "std": std.tolist(),
        "q01": np.quantile(stacked, 0.01, axis=0).tolist(),
        "q99": np.quantile(stacked, 0.99, axis=0).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description="Generate ManiSkill state normalizer stats JSON.")
    parser.add_argument("--h5_path", required=True, type=Path)
    parser.add_argument("--output_path", required=True, type=Path)
    parser.add_argument("--state_dim", type=int, default=29)
    args = parser.parse_args()

    if not args.h5_path.exists():
        raise FileNotFoundError(f"H5 dataset not found: {args.h5_path}")

    stats = compute_state_stats(args.h5_path, args.state_dim)
    output = {"norm_stats": {"state": stats}}
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Saved state normalizer stats to {args.output_path}")


if __name__ == "__main__":
    main()
