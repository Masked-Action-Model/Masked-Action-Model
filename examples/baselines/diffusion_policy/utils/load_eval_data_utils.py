import json
import os
from typing import List, Optional

import numpy as np
import torch
from h5py import File

try:
    from utils.progress_utils import (
        augment_mask_with_progress,
        augment_mas_with_progress,
        pad_mas_to_length,
        pad_mask_to_length,
    )
    from utils.load_train_data_utils import load_demo_dataset
except ModuleNotFoundError:
    from examples.baselines.diffusion_policy.utils.progress_utils import (
        augment_mask_with_progress,
        augment_mas_with_progress,
        pad_mas_to_length,
        pad_mask_to_length,
    )
    from examples.baselines.diffusion_policy.utils.load_train_data_utils import (
        load_demo_dataset,
    )


def _decode_string_value(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return _decode_string_value(value.item())
        if value.size == 1:
            return _decode_string_value(value.reshape(-1)[0])
    return str(value)


def _candidate_metadata_paths(demo_path: str):
    base, _ = os.path.splitext(demo_path)
    candidates = [f"{base}.json"]
    suffixes = [
        "_concat_train",
        "_concat_eval",
        "_concat",
        "_train",
        "_eval",
        "_masked",
        "_normed",
    ]
    for suffix in suffixes:
        if base.endswith(suffix):
            candidates.append(f"{base[:-len(suffix)]}.json")
    seen = set()
    out = []
    for path in candidates:
        if path not in seen:
            out.append(path)
            seen.add(path)
    return out


def infer_eval_reset_seed_from_demo(
    demo_path: str,
    metadata_path: Optional[str] = None,
):
    seeds = infer_eval_reset_seeds_from_demo(
        demo_path=demo_path,
        num_traj=1,
        metadata_path=metadata_path,
    )
    if len(seeds) == 0:
        print("[seed-infer] failed to infer seed from demo metadata")
        return None
    return seeds[0]


def infer_eval_reset_seeds_from_demo(
    demo_path: str,
    num_traj: Optional[int] = None,
    metadata_path: Optional[str] = None,
):
    if not os.path.exists(demo_path):
        print(f"[seed-infer] demo path not found: {demo_path}")
        return []
    with File(demo_path, "r") as f:
        traj_keys = [k for k in f.keys() if k.startswith("traj_")]
    if len(traj_keys) == 0:
        print(f"[seed-infer] no traj_* found in: {demo_path}")
        return []
    traj_keys = sorted(traj_keys, key=lambda x: int(x.split("_")[-1]))
    if num_traj is not None:
        traj_keys = traj_keys[:num_traj]
    episode_ids = [int(k.split("_")[-1]) for k in traj_keys]

    meta_candidates = []
    if metadata_path is not None and len(metadata_path.strip()) > 0:
        meta_candidates.append(metadata_path)
    meta_candidates.extend(_candidate_metadata_paths(demo_path))

    seen = set()
    deduped_meta_candidates = []
    for meta_path in meta_candidates:
        if meta_path not in seen:
            deduped_meta_candidates.append(meta_path)
            seen.add(meta_path)

    for meta_path in deduped_meta_candidates:
        if not os.path.exists(meta_path):
            continue
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
            episodes = meta.get("episodes", [])
            seeds = []
            for episode_id in episode_ids:
                if episode_id >= len(episodes):
                    seeds = []
                    break
                episode = episodes[episode_id]
                reset_seed = episode.get("reset_kwargs", {}).get("seed", None)
                if isinstance(reset_seed, list):
                    reset_seed = reset_seed[0] if len(reset_seed) > 0 else None
                if reset_seed is None:
                    reset_seed = episode.get("episode_seed", None)
                if reset_seed is None:
                    seeds = []
                    break
                seeds.append(int(reset_seed))
            if len(seeds) == len(episode_ids):
                print(f"[seed-infer] loaded {len(seeds)} eval seeds from {meta_path}")
                return seeds
        except Exception as e:
            print(f"[seed-infer] failed reading {meta_path}: {e}")
    print("[seed-infer] failed to infer evaluation seed list from demo metadata")
    return []


def select_eval_demo_indices(total_demos: int, num_eval_demos: int):
    if total_demos <= 0:
        return []
    return list(range(min(total_demos, num_eval_demos)))


def infer_eval_traj_ids_from_demo(demo_path: str, num_traj: Optional[int] = None):
    if not os.path.exists(demo_path):
        print(f"[traj-infer] demo path not found: {demo_path}")
        return []
    with File(demo_path, "r") as f:
        traj_keys = [k for k in f.keys() if k.startswith("traj_")]
    if len(traj_keys) == 0:
        print(f"[traj-infer] no traj_* found in: {demo_path}")
        return []
    traj_ids = sorted(int(k.split("_")[-1]) for k in traj_keys)
    if num_traj is not None:
        traj_ids = traj_ids[:num_traj]
    return traj_ids


def subset_eval_data(eval_data: dict, indices: List[int]):
    subset = {}
    for k, v in eval_data.items():
        if isinstance(v, list):
            subset[k] = [v[i] for i in indices]
        else:
            subset[k] = v
    return subset


def read_augmented_mas_length(data_path: str, mas_step_dim: int = 8) -> int:
    with File(data_path, "r") as f:
        traj_keys = sorted(
            [k for k in f.keys() if k.startswith("traj_")],
            key=lambda x: int(x.split("_")[-1]),
        )
        assert len(traj_keys) > 0, f"No traj_* groups found in dataset: {data_path}"
        inferred_max_length = 0
        for traj_key in traj_keys:
            traj = f[traj_key]
            assert "mas" in traj, f"Missing 'mas' in trajectory: {traj_key}"
            mas_shape = traj["mas"].shape
            assert len(mas_shape) == 2, f"Expected 2D mas in {traj_key}, got shape {mas_shape}"
            if int(mas_shape[1]) == mas_step_dim:
                expected_step_dim = mas_step_dim
            elif int(mas_shape[1]) + 1 == mas_step_dim:
                expected_step_dim = int(mas_shape[1]) + 1
            else:
                raise AssertionError(
                    f"Expected raw/augmented mas step dim to map to {mas_step_dim}, "
                    f"got shape {mas_shape} in {traj_key}"
                )
            assert expected_step_dim == mas_step_dim
            inferred_max_length = max(inferred_max_length, int(mas_shape[0]))

        if "meta" in f and "max_length" in f["meta"]:
            meta_max_length = int(f["meta"]["max_length"][()])
            assert meta_max_length == inferred_max_length, (
                f"meta/max_length ({meta_max_length}) != inferred max length ({inferred_max_length}) "
                f"in dataset {data_path}"
            )
            return meta_max_length
    return inferred_max_length


def _load_traj_string_field(
    data_path: str,
    field_name: str,
    num_traj: Optional[int] = None,
    meta_fallback_field: Optional[str] = None,
    default_value: str = "unknown",
) -> list[str]:
    with File(data_path, "r") as f:
        traj_keys = sorted(
            [k for k in f.keys() if k.startswith("traj_")],
            key=lambda x: int(x.split("_")[-1]),
        )
        if num_traj is not None:
            traj_keys = traj_keys[:num_traj]

        default_field_value = None
        if (
            meta_fallback_field is not None
            and "meta" in f
            and meta_fallback_field in f["meta"]
        ):
            default_field_value = _decode_string_value(f["meta"][meta_fallback_field][()])

        values = []
        for traj_key in traj_keys:
            traj_group = f[traj_key]
            if field_name in traj_group:
                values.append(_decode_string_value(traj_group[field_name][()]))
            elif default_field_value is not None:
                values.append(default_field_value)
            else:
                values.append(str(default_value))
    return values


def load_traj_mask_types(data_path: str, num_traj: Optional[int] = None) -> list[str]:
    return _load_traj_string_field(
        data_path=data_path,
        field_name="mask_type",
        num_traj=num_traj,
        meta_fallback_field="mask_type",
        default_value="unknown",
    )


def load_traj_mask_type_slots(data_path: str, num_traj: Optional[int] = None) -> list[str]:
    slots = _load_traj_string_field(
        data_path=data_path,
        field_name="mask_type_slot",
        num_traj=num_traj,
        meta_fallback_field="mask_type_slot",
        default_value="",
    )
    if any(len(str(slot)) > 0 for slot in slots):
        return [
            str(slot) if len(str(slot)) > 0 else str(mask_type)
            for slot, mask_type in zip(
                slots,
                load_traj_mask_types(data_path=data_path, num_traj=num_traj),
            )
        ]
    return load_traj_mask_types(data_path=data_path, num_traj=num_traj)


def load_eval_only_mas_data(
    data_path: str,
    device: torch.device,
    expected_mas_flat_dim: int,
    num_traj: Optional[int] = None,
):
    if expected_mas_flat_dim <= 0 or expected_mas_flat_dim % 8 != 0:
        raise ValueError(
            f"expected_mas_flat_dim must be a positive multiple of 8, got {expected_mas_flat_dim}"
        )
    target_mas_len = expected_mas_flat_dim // 8
    keys = ["mas", "mask", "actions"]
    trajectories = load_demo_dataset(data_path, keys=keys, num_traj=num_traj, concat=False)
    missing_keys = [k for k in keys if k not in trajectories]
    assert not missing_keys, f"Missing keys in eval trajectories: {missing_keys}"
    assert len(trajectories["mas"]) > 0, "Empty eval trajectories"

    mas_flat_list = []
    mas_mask_flat_list = []
    mas_list = []
    mas_mask_list = []
    traj_lengths = []
    for i in range(len(trajectories["mas"])):
        raw_mas_t = torch.as_tensor(trajectories["mas"][i], device=device, dtype=torch.float32)
        raw_mask_t = torch.as_tensor(trajectories["mask"][i], device=device, dtype=torch.float32)
        traj_len = int(np.asarray(trajectories["actions"][i]).shape[0])

        assert raw_mas_t.ndim == 2, f"mas[{i}] must be 2D, got shape {tuple(raw_mas_t.shape)}"
        if raw_mas_t.shape[0] != traj_len:
            raise ValueError(
                f"eval raw mas[{i}] must keep real action length {traj_len}, got {tuple(raw_mas_t.shape)}"
            )
        if raw_mask_t.ndim != 2 or raw_mask_t.shape[0] != traj_len:
            raise ValueError(
                f"eval raw mask[{i}] must keep real action length {traj_len}, got {tuple(raw_mask_t.shape)}"
            )
        if target_mas_len < traj_len:
            raise ValueError(
                f"target_mas_len {target_mas_len} is smaller than eval traj_len {traj_len}"
            )
        mas_t = pad_mas_to_length(raw_mas_t, target_len=target_mas_len, traj_len=traj_len)
        mask_t = pad_mask_to_length(
            raw_mask_t,
            target_len=target_mas_len,
            traj_len=traj_len,
            mas_t=raw_mas_t,
        )

        mas_flat = mas_t.reshape(-1)
        mas_mask_flat = mask_t.reshape(-1)
        assert mas_flat.shape[0] == expected_mas_flat_dim, (
            f"mas_flat_dim mismatch at traj {i}: {mas_flat.shape[0]} vs expected {expected_mas_flat_dim}"
        )
        if mas_mask_flat.shape[0] != expected_mas_flat_dim:
            raise ValueError(
                f"mas_mask_flat_dim mismatch at traj {i}: {mas_mask_flat.shape[0]} vs expected {expected_mas_flat_dim}"
            )

        mas_list.append(mas_t)
        mas_mask_list.append(mask_t)
        mas_flat_list.append(mas_flat)
        mas_mask_flat_list.append(mas_mask_flat)
        traj_lengths.append(traj_len)
        if i == 0:
            print(
                f"[only_mas_eval] keep raw HDF5 length then pad in memory: "
                f"raw_len={traj_len}, target_len={target_mas_len}, padded_shape={tuple(mas_t.shape)}"
            )

    return {
        "mas_flat": mas_flat_list,
        "mas_mask_flat": mas_mask_flat_list,
        "mas": mas_list,
        "mas_mask": mas_mask_list,
        "traj_lengths": traj_lengths,
    }


def load_eval_mas_window_data(
    data_path: str,
    device: torch.device,
    num_traj: Optional[int] = None,
):
    keys = ["mas", "mask", "actions"]
    trajectories = load_demo_dataset(data_path, keys=keys, num_traj=num_traj, concat=False)
    missing_keys = [k for k in keys if k not in trajectories]
    assert not missing_keys, f"Missing keys in eval trajectories: {missing_keys}"
    assert len(trajectories["mas"]) > 0, "Empty eval trajectories"

    mas_list = []
    mas_mask_list = []
    traj_lengths = []
    mask_types = load_traj_mask_types(data_path=data_path, num_traj=num_traj)
    mask_type_slots = load_traj_mask_type_slots(data_path=data_path, num_traj=num_traj)
    for i in range(len(trajectories["mas"])):
        raw_mas_t = torch.as_tensor(trajectories["mas"][i], device=device, dtype=torch.float32)
        raw_mask_t = torch.as_tensor(trajectories["mask"][i], device=device, dtype=torch.float32)
        traj_len = int(np.asarray(trajectories["actions"][i]).shape[0])
        assert raw_mas_t.ndim == 2, f"mas[{i}] must be 2D, got shape {tuple(raw_mas_t.shape)}"
        mas_list.append(augment_mas_with_progress(raw_mas_t, traj_len))
        mas_mask_list.append(
            augment_mask_with_progress(raw_mask_t, traj_len=traj_len, mas_t=raw_mas_t).to(
                device=device
            )
        )
        traj_lengths.append(traj_len)

    return {
        "mas": mas_list,
        "mas_mask": mas_mask_list,
        "traj_lengths": traj_lengths,
        "mask_types": mask_types,
        "mask_type_slots": mask_type_slots,
    }


def read_obs_mas_dim_from_meta(data_path: str, mas_step_dim: int = 8) -> int:
    max_length = read_augmented_mas_length(data_path, mas_step_dim=mas_step_dim)
    obs_mas_dim = max_length * mas_step_dim
    assert obs_mas_dim > 0, (
        f"Invalid obs_mas_dim from meta: max_length={max_length}, mas_step_dim={mas_step_dim}"
    )
    return obs_mas_dim
