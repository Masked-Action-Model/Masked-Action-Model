from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import h5py
import numpy as np

from common import list_traj_keys, read_camera_names
from data_preprocess.utils.mask_utils import (
    MASK_TYPES_REQUIRING_RATIO,
    MASK_TYPES_REQUIRING_SEQ_LEN,
    apply_mask_to_actions,
    validate_mask_config,
)
from data_preprocess.utils.normalize_utils import (
    compute_global_min_max,
)
from data_preprocess.utils.obs_utils import (
    build_default_state_obs_extractor,
    build_state_schema_from_obs,
    flatten_state_from_obs,
)
from data_preprocess.utils.progress_utils import (
    augment_mas_with_progress_np,
    mas_step_dim_for_action_dim,
    validate_action_dim,
)
from data_preprocess.utils.io_utils import write_string_dataset
from data_preprocess.data_preprocess_mixed import (
    build_mask_jobs,
    build_output_stem as build_mixed_output_stem,
    normalize_split_mask_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess real Franka h5 for diffusion-policy subgoal/MAM training."
    )
    parser.add_argument("--input-h5", type=Path, required=True)
    parser.add_argument("--output-h5", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("franka_train/data"))
    parser.add_argument("--output-prefix", type=str, default=None)
    parser.add_argument("--env-id", type=str, default="FrankaReal-v1")
    parser.add_argument("--control-mode", type=str, default="pd_ee_pose")
    parser.add_argument("--action-dim", type=int, choices=[6, 7], default=7)
    parser.add_argument("--mask-type", type=str, default="random_mask")
    parser.add_argument("--retain-ratio", type=float, default=0.2)
    parser.add_argument("--mask-seq-len", type=int, default=20)
    parser.add_argument("--mask-value", type=float, default=0.0)
    parser.add_argument(
        "--mask-assign-mode",
        choices=["composition", "one_demo_multi_mask"],
        default="composition",
    )
    parser.add_argument("--num-mask-type", type=int, default=None)
    parser.add_argument("--mask-type-list", type=str, default=None)
    parser.add_argument("--mask-ratio-list", type=str, default=None)
    parser.add_argument("--mask-composition-list", type=str, default=None)
    parser.add_argument("--train-num-mask-type", type=int, default=None)
    parser.add_argument("--train-mask-type-list", type=str, default=None)
    parser.add_argument("--train-mask-ratio-list", type=str, default=None)
    parser.add_argument("--train-mask-composition-list", type=str, default=None)
    parser.add_argument(
        "--action-robust-margin",
        type=float,
        default=0.01,
        help="action robust min/max margin; 0.01 means 1%/99% quantiles, then clip before normalization",
    )
    parser.add_argument(
        "--state-robust-margin",
        type=float,
        default=0.01,
        help="state robust min/max margin; 0.01 means 1%/99% quantiles, then clip before normalization",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="resize obs/sensor_data/*/(rgb|depth) to image_size x image_size during preprocessing; <=0 keeps raw size",
    )
    parser.add_argument("--num-traj", type=int, default=None)
    parser.add_argument("--mask-seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def uses_mixed_mask(args: argparse.Namespace) -> bool:
    return any(
        value is not None
        for value in [
            args.num_mask_type,
            args.mask_type_list,
            args.mask_ratio_list,
            args.mask_composition_list,
            args.train_num_mask_type,
            args.train_mask_type_list,
            args.train_mask_ratio_list,
            args.train_mask_composition_list,
        ]
    )


def single_output_path(args: argparse.Namespace) -> Path:
    if args.output_h5 is not None:
        return args.output_h5
    prefix = args.output_prefix or args.input_h5.stem
    suffix = args.mask_type
    if args.mask_type in MASK_TYPES_REQUIRING_RATIO:
        suffix = f"{suffix}_{format(float(args.retain_ratio), 'g')}"
    if args.mask_type in MASK_TYPES_REQUIRING_SEQ_LEN:
        suffix = f"{suffix}_seq{int(args.mask_seq_len)}"
    return args.output_dir / f"{prefix}_{suffix}_train.h5"


def mixed_mask_args(args: argparse.Namespace) -> SimpleNamespace:
    mask_type_list = args.mask_type_list
    mask_ratio_list = args.mask_ratio_list
    mask_composition_list = args.mask_composition_list
    if mask_type_list is None:
        mask_type_list = json.dumps([args.mask_type])
    if mask_ratio_list is None:
        raw_param = (
            args.mask_seq_len
            if args.mask_type in MASK_TYPES_REQUIRING_SEQ_LEN
            else args.retain_ratio
        )
        mask_ratio_list = json.dumps([raw_param])
    if mask_composition_list is None:
        mask_composition_list = json.dumps([1.0])
    num_mask_type = args.num_mask_type
    if num_mask_type is None:
        try:
            num_mask_type = len(json.loads(mask_type_list))
        except Exception:
            num_mask_type = len([x for x in str(mask_type_list).split(",") if x.strip()])
    return SimpleNamespace(
        num_mask_type=int(num_mask_type),
        mask_type_list=mask_type_list,
        mask_ratio_list=mask_ratio_list,
        mask_composition_list=mask_composition_list,
        train_num_mask_type=args.train_num_mask_type,
        train_mask_type_list=args.train_mask_type_list,
        train_mask_ratio_list=args.train_mask_ratio_list,
        train_mask_composition_list=args.train_mask_composition_list,
        eval_num_mask_type=args.train_num_mask_type,
        eval_mask_type_list=args.train_mask_type_list,
        eval_mask_ratio_list=args.train_mask_ratio_list,
        eval_mask_composition_list=args.train_mask_composition_list,
        mask_assign_mode=args.mask_assign_mode,
    )


def mixed_output_path(args: argparse.Namespace, train_mask_specs: list[dict]) -> Path:
    if args.output_h5 is not None:
        return args.output_h5
    prefix = args.output_prefix or args.input_h5.stem
    stem = build_mixed_output_stem(
        prefix,
        train_mask_specs,
        train_mask_specs,
        mask_assign_mode=args.mask_assign_mode,
    )
    return args.output_dir / f"{stem}_train.h5"


def load_state_matrix(obs_group: h5py.Group) -> np.ndarray:
    obs = {
        "agent": {
            key: np.asarray(obs_group["agent"][key][()])
            for key in obs_group["agent"].keys()
        },
        "extra": {
            key: np.asarray(obs_group["extra"][key][()])
            for key in obs_group["extra"].keys()
        },
    }
    return flatten_state_from_obs(
        obs,
        state_obs_extractor=build_default_state_obs_extractor(),
    )


def iter_action_arrays(h5_file: h5py.File, traj_keys: list[str], action_dim: int):
    norm_dims = min(6, int(action_dim))
    for traj_key in traj_keys:
        actions = np.asarray(h5_file[traj_key]["actions"][()], dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != int(action_dim):
            raise ValueError(
                f"{traj_key}/actions must have shape (T, {action_dim}), got {actions.shape}"
            )
        yield actions[:, :norm_dims]


def iter_state_arrays(h5_file: h5py.File, traj_keys: list[str]):
    for traj_key in traj_keys:
        yield load_state_matrix(h5_file[traj_key]["obs"])


def validate_robust_margin(margin: float, name: str) -> float:
    margin = float(margin)
    if not np.isfinite(margin):
        raise ValueError(f"{name} must be finite, got {margin}")
    if margin < 0.0 or margin >= 0.5:
        raise ValueError(f"{name} must be in [0, 0.5), got {margin}")
    return margin


def compute_robust_min_max(array_iterable, margin: float, name: str):
    margin = validate_robust_margin(margin, name)
    if margin <= 0.0:
        return compute_global_min_max(array_iterable)

    arrays = []
    for array in array_iterable:
        array = np.asarray(array, dtype=np.float32)
        if array.ndim != 2:
            raise ValueError(f"expected 2D array for {name}, got shape {array.shape}")
        if array.shape[0] == 0:
            raise ValueError(f"cannot compute {name} from empty array")
        arrays.append(array)
    if len(arrays) == 0:
        raise ValueError(f"cannot compute {name} from empty iterable")
    stacked = np.concatenate(arrays, axis=0)
    lower = np.quantile(stacked, margin, axis=0)
    upper = np.quantile(stacked, 1.0 - margin, axis=0)
    return lower.astype(np.float32), upper.astype(np.float32)


def normalize_clip_selected_dims(
    data: np.ndarray,
    mins: np.ndarray,
    maxs: np.ndarray,
    dims,
) -> np.ndarray:
    data = np.asarray(data, dtype=np.float32)
    mins = np.asarray(mins, dtype=np.float32)
    maxs = np.asarray(maxs, dtype=np.float32)
    normalized = data.copy()
    for dim in [int(dim) for dim in dims]:
        denom = float(maxs[dim] - mins[dim])
        if abs(denom) < 1e-8:
            normalized[..., dim] = 0.0
            continue
        values = np.clip(data[..., dim], mins[dim], maxs[dim])
        normalized[..., dim] = 2.0 * (values - mins[dim]) / denom - 1.0
    return normalized.astype(np.float32)


def copy_attrs(src_obj, dst_obj) -> None:
    for key, value in src_obj.attrs.items():
        dst_obj.attrs[key] = value


def resize_visual_array(array: np.ndarray, image_size: int) -> np.ndarray:
    image_size = int(image_size)
    array = np.asarray(array)
    if image_size <= 0 or array.ndim < 3:
        return array

    has_channel_dim = array.shape[-1] in (1, 3, 4)
    if has_channel_dim:
        leading_shape = array.shape[:-3]
        height, width, channels = array.shape[-3:]
        flat = array.reshape((-1, height, width, channels))
    else:
        leading_shape = array.shape[:-2]
        height, width = array.shape[-2:]
        channels = 1
        flat = array.reshape((-1, height, width, 1))
    if int(height) == image_size and int(width) == image_size:
        return array

    import torch
    import torch.nn.functional as F

    dtype = array.dtype
    tensor = torch.as_tensor(flat).permute(0, 3, 1, 2).to(dtype=torch.float32)
    resized = F.interpolate(
        tensor,
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    )
    resized = resized.permute(0, 2, 3, 1).cpu().numpy()
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        resized = np.clip(np.rint(resized), info.min, info.max)
    resized = resized.astype(dtype, copy=False)
    if has_channel_dim:
        return resized.reshape((*leading_shape, image_size, image_size, channels))
    return resized.reshape((*leading_shape, image_size, image_size))


def copy_group_with_visual_resize(
    src_group: h5py.Group,
    dst_parent: h5py.Group,
    name: str,
    image_size: int,
) -> h5py.Group:
    dst_group = dst_parent.create_group(name)
    copy_attrs(src_group, dst_group)
    for key in src_group.keys():
        src_item = src_group[key]
        if isinstance(src_item, h5py.Group):
            copy_group_with_visual_resize(src_item, dst_group, key, image_size)
        elif isinstance(src_item, h5py.Dataset) and key in {"rgb", "depth"}:
            data = resize_visual_array(src_item[()], image_size)
            dst_item = dst_group.create_dataset(key, data=data, dtype=data.dtype)
            copy_attrs(src_item, dst_item)
        else:
            src_group.file.copy(src_item, dst_group, name=key)
    return dst_group


def copy_obs_with_normalized_state(
    src_obs: h5py.Group,
    dst_traj: h5py.Group,
    normalized_state: np.ndarray,
    image_size: int,
) -> None:
    dst_obs = copy_group_with_visual_resize(src_obs, dst_traj, "obs", image_size)
    if "state" in dst_obs:
        del dst_obs["state"]
    dst_obs.create_dataset("state", data=normalized_state)


def validate_raw_traj(h5_file: h5py.File, traj_key: str, action_dim: int) -> None:
    traj = h5_file[traj_key]
    actions = traj["actions"]
    if actions.ndim != 2 or actions.shape[1] != int(action_dim):
        raise ValueError(f"{traj_key}/actions shape invalid: {actions.shape}")
    state = load_state_matrix(traj["obs"])
    if state.shape[0] != actions.shape[0] + 1:
        raise ValueError(
            f"{traj_key} state length must be T+1, got {state.shape[0]} vs {actions.shape[0]}"
        )
    rgb = traj["obs"]["sensor_data"]["base_camera"]["rgb"]
    if rgb.shape[0] != actions.shape[0] + 1:
        raise ValueError(f"{traj_key} rgb length must be T+1")
    tcp_pose = traj["obs"]["extra"]["tcp_pose"][()]
    quat_norm = np.linalg.norm(np.asarray(tcp_pose[:, 3:7], dtype=np.float32), axis=1)
    if not np.all(np.isfinite(quat_norm)) or np.max(np.abs(quat_norm - 1.0)) > 1e-2:
        raise ValueError(f"{traj_key} tcp_pose quaternion norm is not close to 1")


def build_state_schema(obs_group: h5py.Group) -> list[dict]:
    return build_state_schema_from_obs(obs_group, has_leading_axis=True)


def mask_param_repr(mask_spec: dict[str, Any]) -> Any:
    if mask_spec["mask_type"] in MASK_TYPES_REQUIRING_RATIO:
        return float(mask_spec["retain_ratio"])
    if mask_spec["mask_type"] in MASK_TYPES_REQUIRING_SEQ_LEN:
        return int(mask_spec["mask_seq_len"])
    return None


def write_mixed_mask_meta(
    meta: h5py.Group,
    mask_specs: list[dict[str, Any]],
    args: argparse.Namespace,
    source_num_episodes: int,
    expanded_num_episodes: int,
) -> None:
    meta.create_dataset("mixed_mask_enabled", data=np.bool_(True))
    meta.create_dataset("num_mask_type", data=np.int32(len(mask_specs)))
    meta.create_dataset("source_num_episodes", data=np.int32(source_num_episodes))
    meta.create_dataset("expanded_num_episodes", data=np.int32(expanded_num_episodes))
    write_string_dataset(meta, "mask_assign_mode", str(args.mask_assign_mode))
    write_string_dataset(meta, "mask_specs_json", json.dumps(mask_specs, sort_keys=True))
    write_string_dataset(
        meta,
        "mask_type_list_json",
        json.dumps([spec["mask_type"] for spec in mask_specs]),
    )
    write_string_dataset(
        meta,
        "mask_composition_list_json",
        json.dumps([float(spec["ratio"]) for spec in mask_specs]),
    )
    write_string_dataset(
        meta,
        "mask_ratio_list_json",
        json.dumps([mask_param_repr(spec) for spec in mask_specs]),
    )
    write_string_dataset(
        meta,
        "mask_slot_name_list_json",
        json.dumps([spec["mask_type_slot"] for spec in mask_specs]),
    )
    write_string_dataset(
        meta,
        "mask_slot_ratio_list_json",
        json.dumps([float(spec["ratio"]) for spec in mask_specs]),
    )
    write_string_dataset(
        meta,
        "mask_slot_param_list_json",
        json.dumps([mask_param_repr(spec) for spec in mask_specs]),
    )
    write_string_dataset(meta, "mask_slot_specs_json", json.dumps(mask_specs, sort_keys=True))
    write_string_dataset(meta, "requested_train_mask_specs_json", json.dumps(mask_specs, sort_keys=True))
    write_string_dataset(meta, "requested_eval_mask_specs_json", json.dumps(mask_specs, sort_keys=True))


def write_preprocessed_h5(
    *,
    input_h5: Path,
    output_h5: Path,
    traj_keys: list[str],
    action_min: np.ndarray,
    action_max: np.ndarray,
    state_min: np.ndarray,
    state_max: np.ndarray,
    state_schema: list[dict],
    args: argparse.Namespace,
    mask_jobs: list[dict[str, Any]] | None = None,
    mask_specs: list[dict[str, Any]] | None = None,
) -> None:
    normalized_action_dims = np.arange(min(6, int(args.action_dim)), dtype=np.int32)
    source_episode_ids = [int(key.split("_")[-1]) for key in traj_keys]
    source_key_by_episode_id = {
        int(key.split("_")[-1]): key
        for key in traj_keys
    }
    if mask_jobs is None:
        mask_spec = {
            "mask_type": args.mask_type,
            "mask_type_slot": args.mask_type,
            "mask_slot_index": 0,
            "mask_spec_index": 0,
            "ratio": 1.0,
            "retain_ratio": (
                args.retain_ratio
                if args.mask_type in MASK_TYPES_REQUIRING_RATIO
                else None
            ),
            "mask_seq_len": (
                args.mask_seq_len
                if args.mask_type in MASK_TYPES_REQUIRING_SEQ_LEN
                else None
            ),
        }
        mask_jobs = [
            {
                "source_episode_id": source_episode_id,
                "source_mask_copy_key": f"traj_{source_episode_id}",
                "rng_seed": int(args.mask_seed) + int(source_episode_id),
                "mask_spec": mask_spec,
            }
            for source_episode_id in source_episode_ids
        ]
    camera_names = read_camera_names(str(input_h5))
    output_h5.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(input_h5, "r") as src_file, h5py.File(output_h5, "w") as dst_file:
        max_episode_steps = max(
            int(src_file[key]["actions"].shape[0]) for key in traj_keys
        )
        meta = dst_file.create_group("meta")
        meta.create_dataset("action_min", data=np.asarray(action_min, dtype=np.float32))
        meta.create_dataset("action_max", data=np.asarray(action_max, dtype=np.float32))
        meta.create_dataset("state_min", data=np.asarray(state_min, dtype=np.float32))
        meta.create_dataset("state_max", data=np.asarray(state_max, dtype=np.float32))
        meta.create_dataset("action_dim", data=np.int32(args.action_dim))
        meta.create_dataset("state_dim", data=np.int32(state_min.shape[0]))
        meta.create_dataset("normalized_action_dims", data=normalized_action_dims)
        meta.create_dataset(
            "normalized_state_dims",
            data=np.arange(state_min.shape[0], dtype=np.int32),
        )
        meta.create_dataset("mas_dim", data=np.int32(mas_step_dim_for_action_dim(args.action_dim)))
        meta.create_dataset("num_episodes", data=np.int32(len(mask_jobs)))
        meta.create_dataset("mask_value", data=np.float32(args.mask_value))
        meta.create_dataset("mask_seed", data=np.int32(args.mask_seed))
        meta.create_dataset("action_robust_margin", data=np.float32(args.action_robust_margin))
        meta.create_dataset("state_robust_margin", data=np.float32(args.state_robust_margin))
        meta.create_dataset("actions_clipped_to_norm_range", data=np.bool_(True))
        meta.create_dataset("states_clipped_to_norm_range", data=np.bool_(True))
        meta.create_dataset("image_size", data=np.int32(args.image_size))
        meta.create_dataset("visual_obs_resized", data=np.bool_(int(args.image_size) > 0))
        meta.create_dataset("actions_normalized", data=np.bool_(True))
        meta.create_dataset("states_normalized", data=np.bool_(True))
        meta.create_dataset("mas_has_progress", data=np.bool_(True))
        meta.create_dataset("max_episode_steps", data=np.int32(max_episode_steps))
        expanded_source_episode_ids = [
            int(job["source_episode_id"]) for job in mask_jobs
        ]
        meta.create_dataset(
            "source_episode_ids",
            data=np.asarray(expanded_source_episode_ids, dtype=np.int32),
        )
        meta.create_dataset(
            "unique_source_episode_ids",
            data=np.asarray(source_episode_ids, dtype=np.int32),
        )
        write_string_dataset(meta, "split", "train")
        write_string_dataset(meta, "env_id", args.env_id)
        write_string_dataset(meta, "control_mode", args.control_mode)
        write_string_dataset(
            meta,
            "normalization_method",
            "robust_min_max_clip"
            if args.action_robust_margin > 0.0 or args.state_robust_margin > 0.0
            else "min_max_clip",
        )
        write_string_dataset(meta, "state_path", "obs/state")
        write_string_dataset(meta, "state_paths", [str(entry["path"]) for entry in state_schema])
        write_string_dataset(meta, "state_schema_json", json.dumps(state_schema, sort_keys=True))
        write_string_dataset(meta, "camera_names", json.dumps(camera_names))
        write_string_dataset(meta, "mask_type", args.mask_type)
        write_string_dataset(meta, "source_h5", str(input_h5))
        if mask_specs is not None:
            write_mixed_mask_meta(
                meta,
                mask_specs=mask_specs,
                args=args,
                source_num_episodes=len(source_episode_ids),
                expanded_num_episodes=len(mask_jobs),
            )
        if args.mask_type in MASK_TYPES_REQUIRING_RATIO:
            meta.create_dataset("retain_ratio", data=np.float32(args.retain_ratio))
        if args.mask_type in MASK_TYPES_REQUIRING_SEQ_LEN:
            meta.create_dataset("mask_seq_len", data=np.int32(args.mask_seq_len))

        for local_idx, job in enumerate(mask_jobs):
            source_episode_id = int(job["source_episode_id"])
            traj_key = source_key_by_episode_id[source_episode_id]
            mask_spec = job["mask_spec"]
            src_traj = src_file[traj_key]
            dst_traj = dst_file.create_group(f"traj_{local_idx}")

            actions = np.asarray(src_traj["actions"][()], dtype=np.float32)
            normalized_actions = normalize_clip_selected_dims(
                actions,
                mins=action_min,
                maxs=action_max,
                dims=range(len(normalized_action_dims)),
            )
            state = load_state_matrix(src_traj["obs"])
            normalized_state = normalize_clip_selected_dims(
                state,
                mins=state_min,
                maxs=state_max,
                dims=range(state.shape[1]),
            )
            rng = np.random.default_rng(int(job["rng_seed"]))
            masked_actions, keep_mask = apply_mask_to_actions(
                normalized_actions,
                mask_type=mask_spec["mask_type"],
                rng=rng,
                retain_ratio=mask_spec["retain_ratio"],
                mask_seq_len=mask_spec["mask_seq_len"],
                masked_value=args.mask_value,
                action_dim=args.action_dim,
            )
            mas = augment_mas_with_progress_np(
                masked_actions,
                traj_len=normalized_actions.shape[0],
                action_dim=args.action_dim,
            )

            for key in src_traj.keys():
                if key in {"actions", "mas", "mask"}:
                    continue
                if key == "obs":
                    copy_obs_with_normalized_state(
                        src_traj["obs"],
                        dst_traj,
                        normalized_state=normalized_state,
                        image_size=args.image_size,
                    )
                else:
                    src_file.copy(src_traj[key], dst_traj, name=key)
            dst_traj.create_dataset("actions", data=normalized_actions)
            dst_traj.create_dataset("mas", data=mas)
            dst_traj.create_dataset("mask", data=keep_mask)
            dst_traj.create_dataset("source_episode_id", data=np.int32(source_episode_id))
            write_string_dataset(dst_traj, "mask_type", str(mask_spec["mask_type"]))
            write_string_dataset(dst_traj, "mask_type_slot", str(mask_spec["mask_type_slot"]))
            dst_traj.create_dataset("mask_slot_index", data=np.int32(mask_spec["mask_slot_index"]))
            if mask_spec["retain_ratio"] is not None:
                dst_traj.create_dataset("retain_ratio", data=np.float32(mask_spec["retain_ratio"]))
            if mask_spec["mask_seq_len"] is not None:
                dst_traj.create_dataset("mask_seq_len", data=np.int32(mask_spec["mask_seq_len"]))


def write_extra_franka_meta(output_h5: Path, input_h5: Path, args: argparse.Namespace) -> None:
    camera_names = read_camera_names(str(input_h5))
    with h5py.File(output_h5, "a") as f:
        meta = f["meta"]
        traj_keys = list_traj_keys(f)
        max_episode_steps = max(int(f[key]["actions"].shape[0]) for key in traj_keys)
        for key in [
            "control_mode",
            "camera_names",
            "max_episode_steps",
            "source_" + "json",
        ]:
            if key in meta:
                del meta[key]
        write_string_dataset(meta, "control_mode", args.control_mode)
        write_string_dataset(meta, "camera_names", json.dumps(camera_names))
        meta.create_dataset("max_episode_steps", data=np.int32(max_episode_steps))


def main() -> None:
    args = parse_args()
    args.action_dim = validate_action_dim(args.action_dim)
    mixed_enabled = uses_mixed_mask(args)
    if not mixed_enabled:
        validate_mask_config(
            mask_type=args.mask_type,
            retain_ratio=(
                args.retain_ratio if args.mask_type in MASK_TYPES_REQUIRING_RATIO else None
            ),
            mask_seq_len=(
                args.mask_seq_len if args.mask_type in MASK_TYPES_REQUIRING_SEQ_LEN else None
            ),
        )
    mixed_args = mixed_mask_args(args)
    train_mask_specs = (
        normalize_split_mask_config(
            mixed_args,
            split="train",
            mask_assign_mode=args.mask_assign_mode,
        )
        if mixed_enabled
        else None
    )
    input_h5 = args.input_h5.resolve()
    out_h5 = (
        mixed_output_path(args, train_mask_specs)
        if mixed_enabled
        else single_output_path(args)
    ).resolve()
    if out_h5.exists() and not args.overwrite:
        raise FileExistsError(f"output exists: {out_h5}. Pass --overwrite to replace it.")

    with h5py.File(input_h5, "r") as src_file:
        traj_keys = list_traj_keys(src_file, num_traj=args.num_traj)
        for traj_key in traj_keys:
            validate_raw_traj(src_file, traj_key, args.action_dim)
        action_min, action_max = compute_robust_min_max(
            iter_action_arrays(src_file, traj_keys, args.action_dim),
            margin=args.action_robust_margin,
            name="action_robust_margin",
        )
        state_min, state_max = compute_robust_min_max(
            iter_state_arrays(src_file, traj_keys),
            margin=args.state_robust_margin,
            name="state_robust_margin",
        )
        state_schema = build_state_schema(src_file[traj_keys[0]]["obs"])

    if mixed_enabled:
        source_episode_ids = [int(key.split("_")[-1]) for key in traj_keys]
        mask_jobs = build_mask_jobs(
            source_episode_ids=source_episode_ids,
            mask_specs=train_mask_specs,
            mask_assign_mode=args.mask_assign_mode,
            seed=args.mask_seed,
        )
        write_preprocessed_h5(
            input_h5=input_h5,
            output_h5=out_h5,
            traj_keys=traj_keys,
            action_min=action_min,
            action_max=action_max,
            state_min=state_min,
            state_max=state_max,
            state_schema=state_schema,
            args=args,
            mask_jobs=mask_jobs,
            mask_specs=train_mask_specs,
        )
        written_trajs = len(mask_jobs)
    else:
        write_preprocessed_h5(
            input_h5=input_h5,
            output_h5=out_h5,
            traj_keys=traj_keys,
            action_min=action_min,
            action_max=action_max,
            state_min=state_min,
            state_max=state_max,
            state_schema=state_schema,
            args=args,
        )
        written_trajs = len(traj_keys)
    print(
        f"[franka-preprocess] wrote {out_h5} "
        f"trajs={written_trajs} state_dim={state_min.shape[0]} "
        f"action_dim={args.action_dim} "
        f"action_margin={args.action_robust_margin:g} "
        f"state_margin={args.state_robust_margin:g} "
        f"image_size={args.image_size}"
    )


if __name__ == "__main__":
    main()
