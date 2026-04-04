from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch


class FrameManiskillDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        repo_id: str,
        episodes: list[int] | None = None,
        n_obs_steps: int = 1,
        frame_gap: int = 1,
        image_names: list[str] | None = None,
        task_name: str = "",
    ):
        self.repo_id = repo_id
        self.path = Path(repo_id)
        if not self.path.exists():
            raise FileNotFoundError(f"ManiSkill dataset not found: {self.path}")
        if self.path.suffix != ".h5":
            raise ValueError(f"Expected an .h5 dataset path, got: {self.path}")

        self.n_obs_steps = int(n_obs_steps)
        self.frame_gap = int(frame_gap)
        self.image_names = image_names or ["base_camera"]
        self.task_name = task_name
        self.sequence_length = self.n_obs_steps + 1
        self.frame_relative_indices = np.arange(
            -self.n_obs_steps * self.frame_gap,
            1,
            self.frame_gap,
            dtype=np.int64,
        )

        unsupported = [name for name in self.image_names if name != "base_camera"]
        if unsupported:
            raise ValueError(
                f"Only 'base_camera' is supported by FrameManiskillDataset for now. "
                f"Got unsupported image_names={unsupported}"
            )

        self._h5_file: h5py.File | None = None
        self.episode_lengths: dict[int, int] = {}
        self.sample_lookup: list[tuple[int, int]] = []

        traj_pattern = re.compile(r"traj_(\d+)")
        with h5py.File(self.path, "r") as dataset:
            available_episode_ids = []
            for key in dataset.keys():
                match = traj_pattern.fullmatch(key)
                if match:
                    available_episode_ids.append(int(match.group(1)))

            available_episode_ids.sort()
            if not available_episode_ids:
                raise ValueError(f"No traj_* groups found in ManiSkill dataset: {self.path}")

            if episodes is None:
                selected_episode_ids = available_episode_ids
            else:
                selected_episode_ids = [int(ep) for ep in episodes]
                missing = sorted(set(selected_episode_ids) - set(available_episode_ids))
                if missing:
                    raise ValueError(
                        f"Requested episodes not found in {self.path}: {missing}"
                    )

            for episode_index in selected_episode_ids:
                traj_group = dataset[f"traj_{episode_index}"]
                obs_len = int(traj_group["obs"]["agent"]["qpos"].shape[0])
                self.episode_lengths[episode_index] = obs_len
                self.sample_lookup.extend((episode_index, anchor_index) for anchor_index in range(obs_len))

    def __len__(self) -> int:
        return len(self.sample_lookup)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_h5_file"] = None
        return state

    def _get_h5(self) -> h5py.File:
        if self._h5_file is None:
            self._h5_file = h5py.File(self.path, "r")
        return self._h5_file

    def _get_sample_indices(self, episode_obs_len: int, anchor_index: int) -> np.ndarray:
        sampled_indices = anchor_index + self.frame_relative_indices
        return np.clip(sampled_indices, 0, episode_obs_len - 1)

    @staticmethod
    def _take_rows(dataset: h5py.Dataset, sampled_indices: np.ndarray) -> np.ndarray:
        return np.stack([dataset[int(i)] for i in sampled_indices], axis=0)

    def _load_camera_tensor(self, traj_group: h5py.Group, sampled_indices: np.ndarray) -> torch.Tensor:
        camera_group = traj_group["obs"]["sensor_data"]["base_camera"]
        rgb = self._take_rows(camera_group["rgb"], sampled_indices).astype(np.float32)
        depth = self._take_rows(camera_group["depth"], sampled_indices).astype(np.float32)
        rgb = np.transpose(rgb, (0, 3, 1, 2))
        depth = np.transpose(depth, (0, 3, 1, 2))
        camera = np.concatenate([rgb, depth], axis=1)
        return torch.from_numpy(camera)

    def _load_state_tensor(self, traj_group: h5py.Group, sampled_indices: np.ndarray) -> torch.Tensor:
        agent_group = traj_group["obs"]["agent"]
        extra_group = traj_group["obs"]["extra"]
        state = np.concatenate(
            [
                self._take_rows(agent_group["qpos"], sampled_indices).astype(np.float32),
                self._take_rows(agent_group["qvel"], sampled_indices).astype(np.float32),
                self._take_rows(extra_group["goal_pos"], sampled_indices).astype(np.float32),
                self._take_rows(extra_group["tcp_pose"], sampled_indices).astype(np.float32),
                self._take_rows(extra_group["is_grasped"], sampled_indices).astype(np.float32).reshape(-1, 1),
            ],
            axis=-1,
        )
        return torch.from_numpy(state)

    def __getitem__(self, index: int) -> dict[str, Any]:
        episode_index, anchor_index = self.sample_lookup[index]
        episode_obs_len = self.episode_lengths[episode_index]
        sampled_indices = self._get_sample_indices(episode_obs_len, anchor_index)

        dataset = self._get_h5()
        traj_group = dataset[f"traj_{episode_index}"]

        denominator = max(episode_obs_len - 1, 1)
        targets = sampled_indices.astype(np.float32) / float(denominator)

        item = {
            "base_camera": self._load_camera_tensor(traj_group, sampled_indices),
            "state": self._load_state_tensor(traj_group, sampled_indices),
            "targets": torch.from_numpy(targets),
            "lengths": torch.tensor([self.sequence_length], dtype=torch.long),
            "frame_relative_indices": torch.from_numpy(self.frame_relative_indices.copy()),
            "task": self.task_name,
            "episode_index": episode_index,
            "anchor_index": anchor_index,
        }
        return item
