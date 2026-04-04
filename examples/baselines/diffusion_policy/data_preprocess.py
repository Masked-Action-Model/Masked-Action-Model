from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np

try:
    from data_preprocess_tools.io_utils import (
        ensure_parent_dir,
        list_traj_keys,
        read_json,
        write_json,
        write_string_dataset,
    )
    from data_preprocess_tools.mask_utils import (
        MASK_TYPES_REQUIRING_RATIO,
        MASK_TYPES_REQUIRING_SEQ_LEN,
        apply_mask_to_actions,
        validate_mask_config,
    )
    from data_preprocess_tools.normalize_utils import (
        compute_global_min_max,
        normalize_selected_dims,
    )
    from data_preprocess_tools.obs_utils import (
        build_default_state_obs_extractor,
        flatten_state_from_obs,
    )
    from data_preprocess_tools.progress_utils import (
        MAS_ACTION_DIM,
        MAS_STEP_DIM,
        augment_mas_with_progress_np,
    )
except ModuleNotFoundError:
    from examples.baselines.diffusion_policy.data_preprocess_tools.io_utils import (
        ensure_parent_dir,
        list_traj_keys,
        read_json,
        write_json,
        write_string_dataset,
    )
    from examples.baselines.diffusion_policy.data_preprocess_tools.mask_utils import (
        MASK_TYPES_REQUIRING_RATIO,
        MASK_TYPES_REQUIRING_SEQ_LEN,
        apply_mask_to_actions,
        validate_mask_config,
    )
    from examples.baselines.diffusion_policy.data_preprocess_tools.normalize_utils import (
        compute_global_min_max,
        normalize_selected_dims,
    )
    from examples.baselines.diffusion_policy.data_preprocess_tools.obs_utils import (
        build_default_state_obs_extractor,
        flatten_state_from_obs,
    )
    from examples.baselines.diffusion_policy.data_preprocess_tools.progress_utils import (
        MAS_ACTION_DIM,
        MAS_STEP_DIM,
        augment_mas_with_progress_np,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess ManiSkill diffusion-policy demos into train/eval HDF5 datasets."
    )
    parser.add_argument(
        "--input-h5",
        type=Path,
        default=Path("demos/data_1/data_1.h5"),
        help="原始 demo h5 路径",
    )
    parser.add_argument(
        "--input-json",
        type=Path,
        default=None,
        help="原始 demo metadata json 路径，默认取 input_h5 同名 json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="输出目录，默认写到 input 目录同级的 *_preprocessed 目录",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default=None,
        help="输出文件名前缀，默认使用 input_h5 stem",
    )
    parser.add_argument(
        "--env-id",
        type=str,
        default="PickCube-v1",
        help="环境 id，仅用于记录 meta 和后续对齐",
    )
    parser.add_argument(
        "--mask-type",
        type=str,
        default="3D_points",
        help=(
            "mask 类型，支持例如 "
            "2D_video_trajectory、2D_image_trajectory、2D_partial_trajectory、"
            "pose_AnyGrasp、pose_motion_planning、points、3D_points、"
            "local_planner、random_mask 等"
        ),
    )
    parser.add_argument(
        "--retain-ratio",
        type=float,
        default=0.03,
        help="points / 3D_points / pose_motion_planning / random_mask 等类型使用的保留比例",
    )
    parser.add_argument(
        "--mask-seq-len",
        type=int,
        default=20,
        help="local_planner / 2D_partial_trajectory 的连续窗口长度",
    )
    parser.add_argument(
        "--mask-value",
        type=float,
        default=0.0,
        help="mask 后填充值",
    )
    parser.add_argument(
        "--num-traj",
        type=int,
        default=None,
        help="只处理前 num_traj 条轨迹，便于 smoke test",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=0,
        help="train/eval split 随机种子",
    )
    parser.add_argument(
        "--mask-seed",
        type=int,
        default=0,
        help="mask 随机种子基数；每条轨迹会基于 source_episode_id 派生自己的种子",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖已有输出文件",
    )
    return parser.parse_args()


def default_output_dir(input_h5: Path) -> Path:
    parent = input_h5.parent
    if parent.parent != parent:
        return parent.parent / f"{parent.name}_preprocessed"
    return parent / f"{parent.name}_preprocessed"


def format_float_suffix(value: float) -> str:
    return format(float(value), "g")


def build_output_stem(
    output_prefix: str,
    mask_type: str,
    retain_ratio: float | None,
    mask_seq_len: int | None,
) -> str:
    if mask_type in MASK_TYPES_REQUIRING_RATIO:
        return f"{output_prefix}_{mask_type}_{format_float_suffix(retain_ratio)}"
    if mask_type in MASK_TYPES_REQUIRING_SEQ_LEN:
        return f"{output_prefix}_{mask_type}_seq{int(mask_seq_len)}"
    return f"{output_prefix}_{mask_type}"


def validate_inputs(
    input_h5: Path,
    input_json: Path,
    output_dir: Path,
    overwrite: bool,
) -> None:
    if not input_h5.exists():
        raise FileNotFoundError(f"input h5 not found: {input_h5}")
    if not input_json.exists():
        raise FileNotFoundError(f"input json not found: {input_json}")
    output_dir.mkdir(parents=True, exist_ok=True)
    if not overwrite:
        existing = list(output_dir.glob("*"))
        if len(existing) > 0:
            print(
                f"[warn] output_dir {output_dir} 已存在内容，若要覆盖请显式添加 --overwrite。"
            )


def load_state_matrix_from_obs_group(obs_group: h5py.Group) -> np.ndarray:
    state_obs = {
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
        state_obs,
        state_obs_extractor=build_default_state_obs_extractor(),
    )


def iter_selected_action_arrays(
    h5_file: h5py.File,
    traj_keys: Iterable[str],
) -> Iterable[np.ndarray]:
    for traj_key in traj_keys:
        actions = np.asarray(h5_file[traj_key]["actions"][()], dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != MAS_ACTION_DIM:
            raise ValueError(
                f"{traj_key}/actions must have shape (T, {MAS_ACTION_DIM}), got {actions.shape}"
            )
        yield actions[:, :6]


def iter_selected_state_arrays(
    h5_file: h5py.File,
    traj_keys: Iterable[str],
) -> Iterable[np.ndarray]:
    for traj_key in traj_keys:
        yield load_state_matrix_from_obs_group(h5_file[traj_key]["obs"])


def validate_dataset_alignment(h5_file: h5py.File, metadata: dict, traj_keys: list[str]) -> None:
    episodes = metadata.get("episodes", [])
    all_traj_keys = list_traj_keys(h5_file)
    if len(all_traj_keys) != len(episodes):
        raise ValueError(
            f"h5/json episode count mismatch: {len(all_traj_keys)} trajs vs {len(episodes)} json episodes"
        )
    for traj_key in traj_keys:
        traj_group = h5_file[traj_key]
        action_len = int(traj_group["actions"].shape[0])
        state_len = int(load_state_matrix_from_obs_group(traj_group["obs"]).shape[0])
        if state_len != action_len + 1:
            raise ValueError(
                f"{traj_key} state length must equal action length + 1, got {state_len} vs {action_len}"
            )


def split_source_episode_ids(
    source_episode_ids: list[int],
    split_seed: int,
) -> tuple[list[int], list[int]]:
    if len(source_episode_ids) == 0:
        raise ValueError("source_episode_ids is empty")
    if len(source_episode_ids) == 1:
        return source_episode_ids[:], []

    rng = np.random.default_rng(split_seed)
    perm = rng.permutation(np.asarray(source_episode_ids, dtype=np.int64))
    eval_count = max(1, len(source_episode_ids) // 6)
    eval_ids = sorted(int(x) for x in perm[:eval_count].tolist())
    train_ids = sorted(int(x) for x in perm[eval_count:].tolist())
    if len(train_ids) == 0:
        train_ids, eval_ids = eval_ids, []
    return train_ids, eval_ids


def build_split_metadata_json(
    source_metadata: dict,
    split_name: str,
    source_episode_ids: list[int],
    input_h5: Path,
    input_json: Path,
    env_id: str,
    mask_type: str,
    retain_ratio: float | None,
    mask_seq_len: int | None,
    split_seed: int,
    mask_seed: int,
) -> dict:
    output_metadata = {
        "env_info": copy.deepcopy(source_metadata.get("env_info", {})),
        "commit_info": copy.deepcopy(source_metadata.get("commit_info", {})),
        "episodes": [],
        "preprocess_info": {
            "env_id": env_id,
            "split": split_name,
            "source_h5": str(input_h5),
            "source_json": str(input_json),
            "source_episode_ids": [int(x) for x in source_episode_ids],
            "mask_type": mask_type,
            "retain_ratio": retain_ratio,
            "mask_seq_len": mask_seq_len,
            "split_seed": int(split_seed),
            "mask_seed": int(mask_seed),
            "mas_action_dim": MAS_ACTION_DIM,
            "mas_dim": MAS_STEP_DIM,
        },
    }
    episodes = source_metadata.get("episodes", [])
    for local_episode_id, source_episode_id in enumerate(source_episode_ids):
        if source_episode_id >= len(episodes):
            raise IndexError(
                f"source_episode_id {source_episode_id} exceeds json episodes ({len(episodes)})"
            )
        episode = copy.deepcopy(episodes[source_episode_id])
        episode["episode_id"] = int(local_episode_id)
        episode["source_episode_id"] = int(source_episode_id)
        output_metadata["episodes"].append(episode)
    return output_metadata


def write_split_h5(
    input_h5: Path,
    output_h5: Path,
    split_name: str,
    source_episode_ids: list[int],
    action_min: np.ndarray,
    action_max: np.ndarray,
    state_min: np.ndarray,
    state_max: np.ndarray,
    env_id: str,
    mask_type: str,
    retain_ratio: float | None,
    mask_seq_len: int | None,
    mask_value: float,
    split_seed: int,
    mask_seed: int,
    input_json: Path,
) -> None:
    ensure_parent_dir(output_h5)
    with h5py.File(input_h5, "r") as src_file, h5py.File(output_h5, "w") as dst_file:
        meta = dst_file.create_group("meta")
        meta.create_dataset("action_min", data=np.asarray(action_min, dtype=np.float32))
        meta.create_dataset("action_max", data=np.asarray(action_max, dtype=np.float32))
        meta.create_dataset("state_min", data=np.asarray(state_min, dtype=np.float32))
        meta.create_dataset("state_max", data=np.asarray(state_max, dtype=np.float32))
        meta.create_dataset("action_dim", data=np.int32(MAS_ACTION_DIM))
        meta.create_dataset("state_dim", data=np.int32(state_min.shape[0]))
        meta.create_dataset("mas_dim", data=np.int32(MAS_STEP_DIM))
        meta.create_dataset("num_episodes", data=np.int32(len(source_episode_ids)))
        meta.create_dataset("mask_value", data=np.float32(mask_value))
        meta.create_dataset("split_seed", data=np.int32(split_seed))
        meta.create_dataset("mask_seed", data=np.int32(mask_seed))
        meta.create_dataset("actions_normalized", data=np.bool_(True))
        meta.create_dataset("states_normalized", data=np.bool_(True))
        meta.create_dataset("mas_has_progress", data=np.bool_(True))
        meta.create_dataset("state_path", data=np.asarray("obs/state", dtype=h5py.string_dtype("utf-8")))
        meta.create_dataset("normalized_action_dims", data=np.asarray([0, 1, 2, 3, 4, 5], dtype=np.int32))
        meta.create_dataset(
            "source_episode_ids",
            data=np.asarray(source_episode_ids, dtype=np.int32),
        )
        write_string_dataset(meta, "split", split_name)
        write_string_dataset(meta, "env_id", env_id)
        write_string_dataset(meta, "mask_type", mask_type)
        write_string_dataset(meta, "source_h5", str(input_h5))
        write_string_dataset(meta, "source_json", str(input_json))
        if retain_ratio is not None:
            meta.create_dataset("retain_ratio", data=np.float32(retain_ratio))
        if mask_seq_len is not None:
            meta.create_dataset("mask_seq_len", data=np.int32(mask_seq_len))

        for local_episode_id, source_episode_id in enumerate(source_episode_ids):
            src_traj = src_file[f"traj_{source_episode_id}"]
            dst_traj = dst_file.create_group(f"traj_{local_episode_id}")

            actions = np.asarray(src_traj["actions"][()], dtype=np.float32)
            normalized_actions = normalize_selected_dims(
                actions,
                mins=action_min,
                maxs=action_max,
                dims=range(6),
            )
            state = load_state_matrix_from_obs_group(src_traj["obs"])
            normalized_state = normalize_selected_dims(
                state,
                mins=state_min,
                maxs=state_max,
                dims=range(state.shape[1]),
            )
            if normalized_state.shape[0] != normalized_actions.shape[0] + 1:
                raise ValueError(
                    "normalized state/action length mismatch for "
                    f"traj_{source_episode_id}: {normalized_state.shape[0]} vs {normalized_actions.shape[0]}"
                )

            traj_rng = np.random.default_rng(mask_seed + int(source_episode_id))
            masked_actions, keep_mask = apply_mask_to_actions(
                normalized_actions,
                mask_type=mask_type,
                rng=traj_rng,
                retain_ratio=retain_ratio,
                mask_seq_len=mask_seq_len,
                masked_value=mask_value,
            )
            mas = augment_mas_with_progress_np(
                masked_actions,
                traj_len=normalized_actions.shape[0],
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
            dst_traj.create_dataset(
                "source_episode_id",
                data=np.int32(source_episode_id),
            )


def main() -> None:
    args = parse_args()
    validate_mask_config(
        mask_type=args.mask_type,
        retain_ratio=args.retain_ratio if args.mask_type in MASK_TYPES_REQUIRING_RATIO else None,
        mask_seq_len=args.mask_seq_len if args.mask_type in MASK_TYPES_REQUIRING_SEQ_LEN else None,
    )

    input_h5 = args.input_h5.resolve()
    input_json = (
        args.input_json.resolve()
        if args.input_json is not None
        else args.input_h5.with_suffix(".json").resolve()
    )
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else default_output_dir(input_h5).resolve()
    )
    validate_inputs(
        input_h5=input_h5,
        input_json=input_json,
        output_dir=output_dir,
        overwrite=args.overwrite,
    )

    output_prefix = args.output_prefix or input_h5.stem
    output_stem = build_output_stem(
        output_prefix=output_prefix,
        mask_type=args.mask_type,
        retain_ratio=args.retain_ratio if args.mask_type in MASK_TYPES_REQUIRING_RATIO else None,
        mask_seq_len=args.mask_seq_len if args.mask_type in MASK_TYPES_REQUIRING_SEQ_LEN else None,
    )

    with h5py.File(input_h5, "r") as src_file:
        traj_keys = list_traj_keys(src_file, num_traj=args.num_traj)
        metadata = read_json(input_json)
        validate_dataset_alignment(src_file, metadata, traj_keys)
        source_episode_ids = [int(traj_key.split("_")[-1]) for traj_key in traj_keys]
        action_min, action_max = compute_global_min_max(
            iter_selected_action_arrays(src_file, traj_keys)
        )
        state_min, state_max = compute_global_min_max(
            iter_selected_state_arrays(src_file, traj_keys)
        )

    train_ids, eval_ids = split_source_episode_ids(
        source_episode_ids=source_episode_ids,
        split_seed=args.split_seed,
    )

    train_h5 = output_dir / f"{output_stem}_train.h5"
    train_json = output_dir / f"{output_stem}_train.json"
    eval_h5 = output_dir / f"{output_stem}_eval.h5"
    eval_json = output_dir / f"{output_stem}_eval.json"

    if not args.overwrite:
        for path in [train_h5, train_json, eval_h5, eval_json]:
            if path.exists():
                raise FileExistsError(
                    f"output already exists: {path}. Add --overwrite to replace it."
                )

    write_split_h5(
        input_h5=input_h5,
        output_h5=train_h5,
        split_name="train",
        source_episode_ids=train_ids,
        action_min=action_min,
        action_max=action_max,
        state_min=state_min,
        state_max=state_max,
        env_id=args.env_id,
        mask_type=args.mask_type,
        retain_ratio=args.retain_ratio if args.mask_type in MASK_TYPES_REQUIRING_RATIO else None,
        mask_seq_len=args.mask_seq_len if args.mask_type in MASK_TYPES_REQUIRING_SEQ_LEN else None,
        mask_value=args.mask_value,
        split_seed=args.split_seed,
        mask_seed=args.mask_seed,
        input_json=input_json,
    )
    write_json(
        train_json,
        build_split_metadata_json(
            source_metadata=metadata,
            split_name="train",
            source_episode_ids=train_ids,
            input_h5=input_h5,
            input_json=input_json,
            env_id=args.env_id,
            mask_type=args.mask_type,
            retain_ratio=args.retain_ratio if args.mask_type in MASK_TYPES_REQUIRING_RATIO else None,
            mask_seq_len=args.mask_seq_len if args.mask_type in MASK_TYPES_REQUIRING_SEQ_LEN else None,
            split_seed=args.split_seed,
            mask_seed=args.mask_seed,
        ),
    )

    if len(eval_ids) > 0:
        write_split_h5(
            input_h5=input_h5,
            output_h5=eval_h5,
            split_name="eval",
            source_episode_ids=eval_ids,
            action_min=action_min,
            action_max=action_max,
            state_min=state_min,
            state_max=state_max,
            env_id=args.env_id,
            mask_type=args.mask_type,
            retain_ratio=args.retain_ratio if args.mask_type in MASK_TYPES_REQUIRING_RATIO else None,
            mask_seq_len=args.mask_seq_len if args.mask_type in MASK_TYPES_REQUIRING_SEQ_LEN else None,
            mask_value=args.mask_value,
            split_seed=args.split_seed,
            mask_seed=args.mask_seed,
            input_json=input_json,
        )
        write_json(
            eval_json,
            build_split_metadata_json(
                source_metadata=metadata,
                split_name="eval",
                source_episode_ids=eval_ids,
                input_h5=input_h5,
                input_json=input_json,
                env_id=args.env_id,
                mask_type=args.mask_type,
                retain_ratio=args.retain_ratio if args.mask_type in MASK_TYPES_REQUIRING_RATIO else None,
                mask_seq_len=args.mask_seq_len if args.mask_type in MASK_TYPES_REQUIRING_SEQ_LEN else None,
                split_seed=args.split_seed,
                mask_seed=args.mask_seed,
            ),
        )

    print(
        "[data_preprocess] done: "
        f"train={len(train_ids)} trajs -> {train_h5}, "
        f"eval={len(eval_ids)} trajs -> {eval_h5 if len(eval_ids) > 0 else 'N/A'}"
    )
    print(
        "[data_preprocess] stats: "
        f"action_dim={MAS_ACTION_DIM}, state_dim={state_min.shape[0]}, mas_dim={MAS_STEP_DIM}"
    )


if __name__ == "__main__":
    main()
