import os
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple
import random
import h5py


def _normalize_lengths(lengths, eval_video: bool = False):
    """Convert collated lengths from [B, 1] to [B] for the reward model."""
    if hasattr(lengths, "dim") and lengths.dim() > 0 and lengths.shape[-1] == 1:
        lengths = lengths.squeeze(-1)
    return lengths.unsqueeze(0) if eval_video else lengths


def adapt_maniskill_batch_rewind(
    batch: dict,
    camera_names: List[str] = ["top_camera-images-rgb"],
    eval_video: bool = False
) -> dict:
    """Convert a FrameManiskillDataset batch into the training batch format.
    
    Args:
        batch: Batched output from FrameManiskillDataset.
        camera_names: Camera keys to place under ``image_frames``.
        eval_video: If True, wrap tensors with an additional batch dimension.
    """
    def maybe_unsqueeze(x):
        return x.unsqueeze(0) if eval_video else x

    result = {
        "image_frames": {},
        "targets": maybe_unsqueeze(batch["targets"]),
        "lengths": _normalize_lengths(batch["lengths"], eval_video=eval_video),
        "tasks": [batch["task"]] if eval_video else batch["task"],
        "state": maybe_unsqueeze(batch["state"]),
        "frame_relative_indices": maybe_unsqueeze(batch["frame_relative_indices"]),
    }

    for cam_name in camera_names:
        result["image_frames"][cam_name] = maybe_unsqueeze(batch[cam_name])

    return result



def get_valid_episodes(repo_id: str) -> List[int]:
    """
    Collect valid episode indices from a ManiSkill .h5 file or legacy lerobot cache.

    Args:
        repo_id (str): Dataset path or legacy repo/cache identifier.

    Returns:
        List[int]: Sorted list of valid episode indices.
    """
    dataset_path = Path(repo_id)
    if dataset_path.suffix == ".h5":
        if not dataset_path.exists():
            raise FileNotFoundError(f"Data file not found: {dataset_path}")

        episode_pattern = re.compile(r"traj_(\d+)")
        valid_episodes = []
        with h5py.File(dataset_path, "r") as dataset:
            for key in dataset.keys():
                match = episode_pattern.fullmatch(key)
                if match:
                    valid_episodes.append(int(match.group(1)))
        return sorted(valid_episodes)

    base_path = Path.home() / ".cache" / "huggingface" / "lerobot" / repo_id / "data"
    episode_pattern = re.compile(r"episode_(\d+)\.parquet")

    valid_episodes = []

    if not base_path.exists():
        raise FileNotFoundError(f"Data directory not found: {base_path}")

    for chunk_dir in base_path.glob("chunk-*"):
        if not chunk_dir.is_dir():
            continue
        for file in chunk_dir.glob("episode_*.parquet"):
            match = episode_pattern.match(file.name)
            if match:
                ep_idx = int(match.group(1))
                valid_episodes.append(ep_idx)

    return sorted(valid_episodes)

def split_train_eval_episodes(valid_episodes: List[int], train_ratio: float = 0.9, seed: int = 42) -> Tuple[List[int], List[int]]:
    """
    Randomly split valid episode indices into training and evaluation sets.

    Args:
        valid_episodes (List[int]): List of valid episode indices.
        train_ratio (float): Fraction of episodes to use for training (default: 0.9).
        seed (int): Random seed for reproducibility (default: 42).

    Returns:
        Tuple[List[int], List[int]]: (train_episodes, eval_episodes)
    """
    random.seed(seed)
    episodes = valid_episodes.copy()
    random.shuffle(episodes)

    split_index = int(len(episodes) * train_ratio)
    train_episodes = episodes[:split_index]
    eval_episodes = episodes[split_index:]

    return train_episodes, eval_episodes
