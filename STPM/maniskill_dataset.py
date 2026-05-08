from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch


TRAJ_PATTERN = re.compile(r"traj_(\d+)")
STATELESS_OBS_KEYS = {"sensor_data", "sensor_param"}

DEFAULT_STATE_PATHS = [
    "obs/agent/qpos",
    "obs/agent/qvel",
    "obs/extra/is_grasped",
    "obs/extra/tcp_pose",
    "obs/extra/goal_pos",
]


def _list_traj_keys(dataset: h5py.File) -> list[str]:
    traj_keys = sorted(key for key in dataset.keys() if TRAJ_PATTERN.fullmatch(key))
    if not traj_keys:
        raise ValueError(f"No traj_* groups found in ManiSkill dataset")
    return traj_keys


def _state_leaf_dim(dataset: h5py.Dataset) -> int:
    frame_shape = dataset.shape[1:]
    return int(np.prod(frame_shape)) if len(frame_shape) > 0 else 1


def _walk_state_datasets(group: h5py.Group, prefix: str) -> list[tuple[str, h5py.Dataset]]:
    out = []
    for key in group.keys():
        if prefix == "obs" and key in STATELESS_OBS_KEYS:
            continue
        value = group[key]
        path = f"{prefix}/{key}" if prefix else key
        if isinstance(value, h5py.Group):
            out.extend(_walk_state_datasets(value, path))
        elif isinstance(value, h5py.Dataset):
            out.append((path, value))
    return out


def infer_state_schema_from_h5(
    repo_id: str | Path, state_paths: list[str] | None = None
) -> list[dict[str, Any]]:
    path = Path(repo_id)
    if not path.exists():
        raise FileNotFoundError(f"ManiSkill dataset not found: {path}")
    if path.suffix != ".h5":
        raise ValueError(f"Expected an .h5 dataset path, got: {path}")

    with h5py.File(path, "r") as dataset:
        traj_key = _list_traj_keys(dataset)[0]
        traj_group = dataset[traj_key]
        if state_paths is None:
            state_items = _walk_state_datasets(traj_group["obs"], "obs")
        else:
            state_items = []
            for state_path in state_paths:
                try:
                    state_items.append((state_path, traj_group[state_path]))
                except KeyError:
                    raise KeyError(
                        f"Configured STPM state path {state_path!r} is missing from {path}."
                    ) from None
        schema = []
        for state_path, dataset_value in state_items:
            schema.append(
                {
                    "path": state_path,
                    "shape": [int(v) for v in dataset_value.shape[1:]],
                    "dtype": str(dataset_value.dtype),
                    "dim": int(_state_leaf_dim(dataset_value)),
                }
            )
    return schema


def infer_state_paths_from_h5(repo_id: str | Path) -> list[str]:
    return [entry["path"] for entry in infer_state_schema_from_h5(repo_id)]


def infer_camera_info_from_h5(repo_id: str | Path) -> dict[str, Any]:
    path = Path(repo_id)
    if not path.exists():
        raise FileNotFoundError(f"ManiSkill dataset not found: {path}")
    if path.suffix != ".h5":
        raise ValueError(f"Expected an .h5 dataset path, got: {path}")

    with h5py.File(path, "r") as dataset:
        traj_key = _list_traj_keys(dataset)[0]
        obs_group = dataset[f"{traj_key}/obs"]
        sensor_data_path = "sensor_data"
        if sensor_data_path not in obs_group:
            raise ValueError(
                f"No obs/sensor_data found in {path}. "
                "STPM visual training requires an rgb/rgbd H5 dataset."
            )
        sensor_data = obs_group[sensor_data_path]
        sensor_param = obs_group.get("sensor_param", {})
        camera_info = {}
        for camera_name in sensor_data.keys():
            camera_group = sensor_data[camera_name]
            modalities = {}
            for modality in camera_group.keys():
                value = camera_group[modality]
                modalities[modality] = {
                    "shape": [int(v) for v in value.shape[1:]],
                    "dtype": str(value.dtype),
                }
            params = {}
            if camera_name in sensor_param:
                for param_name in sensor_param[camera_name].keys():
                    value = sensor_param[camera_name][param_name]
                    params[param_name] = {
                        "shape": [int(v) for v in value.shape[1:]],
                        "dtype": str(value.dtype),
                    }
            camera_info[camera_name] = {
                "modalities": modalities,
                "params": params,
            }
    return camera_info


def infer_camera_names_from_h5(repo_id: str | Path) -> list[str]:
    path = Path(repo_id)
    if not path.exists():
        raise FileNotFoundError(f"ManiSkill dataset not found: {path}")
    if path.suffix != ".h5":
        raise ValueError(f"Expected an .h5 dataset path, got: {path}")

    with h5py.File(path, "r") as dataset:
        traj_keys = _list_traj_keys(dataset)
        sensor_data_path = f"{traj_keys[0]}/obs/sensor_data"
        if sensor_data_path not in dataset:
            raise ValueError(
                f"No obs/sensor_data found in {path}. "
                "STPM visual training requires an rgb/rgbd H5 dataset."
            )
        sensor_data = dataset[sensor_data_path]
        camera_names = [
            camera_name
            for camera_name in sensor_data.keys()
            if "rgb" in sensor_data[camera_name]
        ]

    if not camera_names:
        raise ValueError(f"No RGB camera datasets found under {sensor_data_path} in {path}")
    return camera_names


def resolve_camera_names(repo_id: str | Path, image_names: list[str] | str | None) -> list[str]:
    if image_names is None:
        return infer_camera_names_from_h5(repo_id)
    if isinstance(image_names, str):
        if image_names.lower() == "auto":
            return infer_camera_names_from_h5(repo_id)
        return [image_names]
    image_names = list(image_names)
    if len(image_names) == 0 or any(str(name).lower() == "auto" for name in image_names):
        return infer_camera_names_from_h5(repo_id)
    return image_names


def resolve_state_paths(state_paths: list[str] | None) -> list[str]:
    return list(state_paths or DEFAULT_STATE_PATHS)


def infer_state_dim_from_h5(repo_id: str | Path, state_paths: list[str] | None = None) -> int:
    path = Path(repo_id)
    if not path.exists():
        raise FileNotFoundError(f"ManiSkill dataset not found: {path}")
    if path.suffix != ".h5":
        raise ValueError(f"Expected an .h5 dataset path, got: {path}")

    resolved_state_paths = (
        infer_state_paths_from_h5(path) if state_paths is None else list(state_paths)
    )
    with h5py.File(path, "r") as dataset:
        traj_keys = _list_traj_keys(dataset)
        traj_group = dataset[traj_keys[0]]
        state_dim = 0
        for state_path in resolved_state_paths:
            try:
                value = traj_group[state_path]
            except KeyError:
                raise KeyError(
                    f"Configured STPM state path {state_path!r} is missing from {path}. "
                    f"state_paths={resolved_state_paths!r}"
                ) from None
            state_dim += _state_leaf_dim(value)
    return state_dim


class FrameManiskillDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        repo_id: str,
        episodes: list[int] | None = None,
        n_obs_steps: int = 1,
        frame_gap: int = 1,
        image_names: list[str] | str | None = None,
        task_name: str = "",
        task_description: str | None = None,
        state_paths: list[str] | None = None,
    ):
        self.repo_id = repo_id
        self.path = Path(repo_id)
        if not self.path.exists():
            raise FileNotFoundError(f"ManiSkill dataset not found: {self.path}")
        if self.path.suffix != ".h5":
            raise ValueError(f"Expected an .h5 dataset path, got: {self.path}")

        self.n_obs_steps = int(n_obs_steps)
        self.frame_gap = int(frame_gap)
        self.image_names = resolve_camera_names(self.path, image_names)
        self.task_description = task_description if task_description is not None else task_name
        self.sequence_length = self.n_obs_steps + 1
        self.frame_relative_indices = np.arange(
            -self.n_obs_steps * self.frame_gap,
            1,
            self.frame_gap,
            dtype=np.int64,
        )

        self.state_paths = resolve_state_paths(state_paths)

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

    def _load_camera_tensor(
        self,
        traj_group: h5py.Group,
        sampled_indices: np.ndarray,
        camera_name: str,
    ) -> torch.Tensor:
        try:
            camera_group = traj_group["obs"]["sensor_data"][camera_name]
        except KeyError:
            available = []
            if "sensor_data" in traj_group["obs"]:
                available = list(traj_group["obs"]["sensor_data"].keys())
            raise KeyError(
                f"Camera {camera_name!r} is missing from trajectory. "
                f"Available cameras: {available}"
            ) from None
        if "rgb" not in camera_group:
            raise KeyError(f"Camera {camera_name!r} has no rgb dataset.")
        rgb = self._take_rows(camera_group["rgb"], sampled_indices).astype(np.float32)
        rgb = np.transpose(rgb, (0, 3, 1, 2))
        return torch.from_numpy(rgb)

    def _load_state_path(
        self, traj_group: h5py.Group, path: str, sampled_indices: np.ndarray
    ) -> np.ndarray:
        try:
            dataset = traj_group[path]
        except KeyError:
            raise KeyError(
                f"Configured STPM state path {path!r} is missing from trajectory. "
                f"Available state paths must be under this traj group in {self.path}."
            ) from None
        value = self._take_rows(dataset, sampled_indices).astype(np.float32)
        if value.ndim == 1:
            value = value.reshape(-1, 1)
        elif value.ndim > 2:
            value = value.reshape(value.shape[0], -1)
        return value

    def _load_state_tensor(self, traj_group: h5py.Group, sampled_indices: np.ndarray) -> torch.Tensor:
        state = np.concatenate(
            [
                self._load_state_path(traj_group, path, sampled_indices)
                for path in self.state_paths
            ],
            axis=-1,
        ).astype(np.float32)
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
            "state": self._load_state_tensor(traj_group, sampled_indices),
            "targets": torch.from_numpy(targets),
            "lengths": torch.tensor([self.sequence_length], dtype=torch.long),
            "frame_relative_indices": torch.from_numpy(self.frame_relative_indices.copy()),
            "task": self.task_description,
            "episode_index": episode_index,
            "anchor_index": anchor_index,
        }
        for camera_name in self.image_names:
            item[camera_name] = self._load_camera_tensor(
                traj_group, sampled_indices, camera_name
            )
        return item
