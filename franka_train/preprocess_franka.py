from __future__ import annotations

import argparse
import json
from pathlib import Path

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
    normalize_selected_dims,
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
    parser.add_argument("--num-traj", type=int, default=None)
    parser.add_argument("--mask-seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def output_path(args: argparse.Namespace) -> Path:
    if args.output_h5 is not None:
        return args.output_h5
    prefix = args.output_prefix or args.input_h5.stem
    suffix = args.mask_type
    if args.mask_type in MASK_TYPES_REQUIRING_RATIO:
        suffix = f"{suffix}_{format(float(args.retain_ratio), 'g')}"
    if args.mask_type in MASK_TYPES_REQUIRING_SEQ_LEN:
        suffix = f"{suffix}_seq{int(args.mask_seq_len)}"
    return args.output_dir / f"{prefix}_{suffix}_train.h5"


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
) -> None:
    normalized_action_dims = np.arange(min(6, int(args.action_dim)), dtype=np.int32)
    source_episode_ids = [int(key.split("_")[-1]) for key in traj_keys]
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
        meta.create_dataset("num_episodes", data=np.int32(len(traj_keys)))
        meta.create_dataset("mask_value", data=np.float32(args.mask_value))
        meta.create_dataset("mask_seed", data=np.int32(args.mask_seed))
        meta.create_dataset("actions_normalized", data=np.bool_(True))
        meta.create_dataset("states_normalized", data=np.bool_(True))
        meta.create_dataset("mas_has_progress", data=np.bool_(True))
        meta.create_dataset("max_episode_steps", data=np.int32(max_episode_steps))
        meta.create_dataset("source_episode_ids", data=np.asarray(source_episode_ids, dtype=np.int32))
        write_string_dataset(meta, "split", "train")
        write_string_dataset(meta, "env_id", args.env_id)
        write_string_dataset(meta, "control_mode", args.control_mode)
        write_string_dataset(meta, "state_path", "obs/state")
        write_string_dataset(meta, "state_paths", [str(entry["path"]) for entry in state_schema])
        write_string_dataset(meta, "state_schema_json", json.dumps(state_schema, sort_keys=True))
        write_string_dataset(meta, "camera_names", json.dumps(camera_names))
        write_string_dataset(meta, "mask_type", args.mask_type)
        write_string_dataset(meta, "source_h5", str(input_h5))
        if args.mask_type in MASK_TYPES_REQUIRING_RATIO:
            meta.create_dataset("retain_ratio", data=np.float32(args.retain_ratio))
        if args.mask_type in MASK_TYPES_REQUIRING_SEQ_LEN:
            meta.create_dataset("mask_seq_len", data=np.int32(args.mask_seq_len))

        for local_idx, traj_key in enumerate(traj_keys):
            src_traj = src_file[traj_key]
            dst_traj = dst_file.create_group(f"traj_{local_idx}")
            source_episode_id = int(traj_key.split("_")[-1])

            actions = np.asarray(src_traj["actions"][()], dtype=np.float32)
            normalized_actions = normalize_selected_dims(
                actions,
                mins=action_min,
                maxs=action_max,
                dims=range(len(normalized_action_dims)),
            )
            state = load_state_matrix(src_traj["obs"])
            normalized_state = normalize_selected_dims(
                state,
                mins=state_min,
                maxs=state_max,
                dims=range(state.shape[1]),
            )
            rng = np.random.default_rng(int(args.mask_seed) + source_episode_id)
            masked_actions, keep_mask = apply_mask_to_actions(
                normalized_actions,
                mask_type=args.mask_type,
                rng=rng,
                retain_ratio=(
                    args.retain_ratio
                    if args.mask_type in MASK_TYPES_REQUIRING_RATIO
                    else None
                ),
                mask_seq_len=(
                    args.mask_seq_len
                    if args.mask_type in MASK_TYPES_REQUIRING_SEQ_LEN
                    else None
                ),
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
                    src_file.copy(src_traj["obs"], dst_traj, name="obs")
                    if "state" in dst_traj["obs"]:
                        del dst_traj["obs"]["state"]
                    dst_traj["obs"].create_dataset("state", data=normalized_state)
                else:
                    src_file.copy(src_traj[key], dst_traj, name=key)
            dst_traj.create_dataset("actions", data=normalized_actions)
            dst_traj.create_dataset("mas", data=mas)
            dst_traj.create_dataset("mask", data=keep_mask)
            dst_traj.create_dataset("source_episode_id", data=np.int32(source_episode_id))


def main() -> None:
    args = parse_args()
    args.action_dim = validate_action_dim(args.action_dim)
    validate_mask_config(
        mask_type=args.mask_type,
        retain_ratio=(
            args.retain_ratio if args.mask_type in MASK_TYPES_REQUIRING_RATIO else None
        ),
        mask_seq_len=(
            args.mask_seq_len if args.mask_type in MASK_TYPES_REQUIRING_SEQ_LEN else None
        ),
    )
    input_h5 = args.input_h5.resolve()
    out_h5 = output_path(args).resolve()
    if out_h5.exists() and not args.overwrite:
        raise FileExistsError(f"output exists: {out_h5}. Pass --overwrite to replace it.")

    with h5py.File(input_h5, "r") as src_file:
        traj_keys = list_traj_keys(src_file, num_traj=args.num_traj)
        for traj_key in traj_keys:
            validate_raw_traj(src_file, traj_key, args.action_dim)
        action_min, action_max = compute_global_min_max(
            iter_action_arrays(src_file, traj_keys, args.action_dim)
        )
        state_min, state_max = compute_global_min_max(iter_state_arrays(src_file, traj_keys))
        state_schema = build_state_schema(src_file[traj_keys[0]]["obs"])

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
    print(
        f"[franka-preprocess] wrote {out_h5} "
        f"trajs={len(traj_keys)} state_dim={state_min.shape[0]} "
        f"action_dim={args.action_dim}"
    )


if __name__ == "__main__":
    main()
