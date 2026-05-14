from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np


THIS_FILE = Path(__file__).resolve()
DP_ROOT = THIS_FILE.parents[1]
REPO_ROOT = THIS_FILE.parents[4]

for path in (REPO_ROOT, DP_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from data_preprocess.data_preprocess import (  # noqa: E402
    build_state_schema_from_obs_group,
    iter_selected_action_arrays,
    iter_selected_state_arrays,
    validate_dataset_alignment,
)
from data_preprocess.data_preprocess_mixed import (  # noqa: E402
    build_mask_jobs,
    build_split_metadata_json,
    normalize_split_mask_config,
    write_split_h5,
)
from data_preprocess.utils.io_utils import list_traj_keys, read_json, write_json  # noqa: E402
from data_preprocess.utils.normalize_utils import compute_global_min_max  # noqa: E402
from data_preprocess.utils.progress_utils import validate_action_dim  # noqa: E402


def parse_num_demos(raw: str) -> list[int]:
    values = [int(item.strip()) for item in str(raw).split(",") if item.strip()]
    if not values:
        raise ValueError("--num-demos-list must contain at least one value")
    if any(value <= 0 for value in values):
        raise ValueError(f"num demos must be positive, got {values}")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare all-success PlaceSphere train=eval datasets for overfit tests."
    )
    parser.add_argument(
        "--input-h5",
        type=Path,
        default=Path("demos/PlaceSphere-v1/PlaceSphere-v1.rgb.pd_ee_pose.physx_cpu.h5"),
    )
    parser.add_argument("--input-json", type=Path, default=None)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("examples/baselines/diffusion_policy/tmp/placesphere_overfit_data"),
    )
    parser.add_argument("--num-demos-list", type=str, default="1,5")
    parser.add_argument("--env-id", type=str, default="PlaceSphere-v1")
    parser.add_argument("--action-dim", type=int, choices=[6, 7], default=7)
    parser.add_argument("--mask-assign-mode", choices=["composition", "one_demo_multi_mask"], default="composition")
    parser.add_argument("--num-mask-type", type=int, default=2)
    parser.add_argument("--mask-type-list", type=str, default='["random_mask","points"]')
    parser.add_argument("--mask-composition-list", type=str, default="[0.5,0.5]")
    parser.add_argument("--mask-ratio-list", type=str, default="[0.2,0.2]")
    parser.add_argument("--mask-value", type=float, default=0.0)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--mask-seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def select_success_episode_ids(metadata: dict, h5_file: h5py.File, count: int) -> list[int]:
    available_keys = set(list_traj_keys(h5_file))
    selected: list[int] = []
    for episode in metadata.get("episodes", []):
        if not bool(episode.get("success", False)):
            continue
        episode_id = int(episode["episode_id"])
        if f"traj_{episode_id}" not in available_keys:
            continue
        selected.append(episode_id)
        if len(selected) == count:
            return selected
    raise ValueError(f"only found {len(selected)} successful trajectories, need {count}")


def copy_non_traj_groups(src_file: h5py.File, dst_file: h5py.File) -> None:
    for key in src_file.keys():
        if not key.startswith("traj_"):
            src_file.copy(src_file[key], dst_file, key)


def write_baseline_subset(
    input_h5: Path,
    input_json: Path,
    output_h5: Path,
    output_json: Path,
    split_name: str,
    source_episode_ids: list[int],
    metadata: dict,
) -> None:
    output_h5.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(input_h5, "r") as src_file, h5py.File(output_h5, "w") as dst_file:
        copy_non_traj_groups(src_file, dst_file)
        for local_episode_id, source_episode_id in enumerate(source_episode_ids):
            src_key = f"traj_{source_episode_id}"
            dst_key = f"traj_{local_episode_id}"
            src_file.copy(src_file[src_key], dst_file, dst_key)
            if "source_episode_id" in dst_file[dst_key]:
                del dst_file[dst_key]["source_episode_id"]
            dst_file[dst_key].create_dataset("source_episode_id", data=np.int32(source_episode_id))

    episodes = metadata.get("episodes", [])
    output_metadata = {
        "episodes": [],
        "env_info": copy.deepcopy(metadata.get("env_info", {})),
        "commit_info": copy.deepcopy(metadata.get("commit_info", {})),
        "overfit_info": {
            "env_id": metadata.get("env_info", {}).get("env_id", "PlaceSphere-v1"),
            "split": split_name,
            "source_h5": str(input_h5),
            "source_json": str(input_json),
            "source_episode_ids": [int(x) for x in source_episode_ids],
            "all_success": True,
            "train_equals_eval": True,
            "normalization": "none",
            "mask": "none",
        },
    }
    for local_episode_id, source_episode_id in enumerate(source_episode_ids):
        episode = copy.deepcopy(episodes[source_episode_id])
        if not bool(episode.get("success", False)):
            raise ValueError(f"source episode {source_episode_id} is not success")
        episode["episode_id"] = int(local_episode_id)
        episode["source_episode_id"] = int(source_episode_id)
        output_metadata["episodes"].append(episode)
    write_json(output_json, output_metadata)


def existing_outputs(paths: list[Path], overwrite: bool) -> bool:
    existing = [path.exists() for path in paths]
    if all(existing) and not overwrite:
        return True
    if any(existing) and not overwrite:
        missing = [str(path) for path, exists in zip(paths, existing) if not exists]
        raise FileExistsError(f"partial outputs exist; missing={missing}; rerun with --overwrite")
    return False


def main() -> None:
    args = parse_args()
    args.action_dim = validate_action_dim(args.action_dim)
    num_demos_list = parse_num_demos(args.num_demos_list)

    input_h5 = args.input_h5.resolve()
    input_json = (
        args.input_json.resolve()
        if args.input_json is not None
        else args.input_h5.with_suffix(".json").resolve()
    )
    output_root = args.output_root.resolve()

    if not input_h5.exists():
        raise FileNotFoundError(f"input h5 not found: {input_h5}")
    if not input_json.exists():
        raise FileNotFoundError(f"input json not found: {input_json}")

    metadata = read_json(input_json)
    train_mask_specs = normalize_split_mask_config(
        SimpleNamespace(
            train_num_mask_type=args.num_mask_type,
            train_mask_type_list=args.mask_type_list,
            train_mask_composition_list=args.mask_composition_list,
            train_mask_ratio_list=args.mask_ratio_list,
            eval_num_mask_type=args.num_mask_type,
            eval_mask_type_list=args.mask_type_list,
            eval_mask_composition_list=args.mask_composition_list,
            eval_mask_ratio_list=args.mask_ratio_list,
        ),
        split="train",
        mask_assign_mode=args.mask_assign_mode,
    )
    eval_mask_specs = copy.deepcopy(train_mask_specs)

    with h5py.File(input_h5, "r") as src_file:
        max_num = max(num_demos_list)
        selected_by_count = {
            count: select_success_episode_ids(metadata, src_file, count)
            for count in num_demos_list
        }

    for count in num_demos_list:
        source_episode_ids = selected_by_count[count]
        source_traj_keys = [f"traj_{episode_id}" for episode_id in source_episode_ids]

        baseline_dir = output_root / f"overfit{count}" / "baseline"
        baseline_train_h5 = baseline_dir / f"placesphere_baseline_overfit{count}_train.h5"
        baseline_train_json = baseline_dir / f"placesphere_baseline_overfit{count}_train.json"
        baseline_eval_h5 = baseline_dir / f"placesphere_baseline_overfit{count}_eval.h5"
        baseline_eval_json = baseline_dir / f"placesphere_baseline_overfit{count}_eval.json"
        if not existing_outputs(
            [baseline_train_h5, baseline_train_json, baseline_eval_h5, baseline_eval_json],
            overwrite=args.overwrite,
        ):
            write_baseline_subset(
                input_h5=input_h5,
                input_json=input_json,
                output_h5=baseline_train_h5,
                output_json=baseline_train_json,
                split_name="train",
                source_episode_ids=source_episode_ids,
                metadata=metadata,
            )
            write_baseline_subset(
                input_h5=input_h5,
                input_json=input_json,
                output_h5=baseline_eval_h5,
                output_json=baseline_eval_json,
                split_name="eval",
                source_episode_ids=source_episode_ids,
                metadata=metadata,
            )

        mam_dir = output_root / f"overfit{count}" / "mam"
        mam_train_h5 = mam_dir / f"placesphere_mam_overfit{count}_train.h5"
        mam_train_json = mam_dir / f"placesphere_mam_overfit{count}_train.json"
        mam_eval_h5 = mam_dir / f"placesphere_mam_overfit{count}_eval.h5"
        mam_eval_json = mam_dir / f"placesphere_mam_overfit{count}_eval.json"
        if not existing_outputs(
            [mam_train_h5, mam_train_json, mam_eval_h5, mam_eval_json],
            overwrite=args.overwrite,
        ):
            with h5py.File(input_h5, "r") as src_file:
                validate_dataset_alignment(src_file, metadata, source_traj_keys)
                action_min, action_max = compute_global_min_max(
                    iter_selected_action_arrays(src_file, source_traj_keys, action_dim=args.action_dim)
                )
                state_min, state_max = compute_global_min_max(
                    iter_selected_state_arrays(src_file, source_traj_keys)
                )
                state_schema = build_state_schema_from_obs_group(src_file[source_traj_keys[0]]["obs"])
                state_schema_dim = sum(int(entry["dim"]) for entry in state_schema)
                if state_schema_dim != int(state_min.shape[0]):
                    raise ValueError(
                        f"state schema dim={state_schema_dim} does not match state stats dim={state_min.shape[0]}"
                    )

            train_jobs = build_mask_jobs(
                source_episode_ids=source_episode_ids,
                mask_specs=train_mask_specs,
                mask_assign_mode=args.mask_assign_mode,
                seed=args.mask_seed,
            )
            eval_jobs = copy.deepcopy(train_jobs)
            for output_path in [mam_train_h5, mam_train_json, mam_eval_h5, mam_eval_json]:
                output_path.parent.mkdir(parents=True, exist_ok=True)

            write_split_h5(
                input_h5=input_h5,
                output_h5=mam_train_h5,
                split_name="train",
                source_episode_ids=source_episode_ids,
                mask_jobs=train_jobs,
                action_min=action_min,
                action_max=action_max,
                state_min=state_min,
                state_max=state_max,
                env_id=args.env_id,
                mask_specs=train_mask_specs,
                requested_train_mask_specs=train_mask_specs,
                requested_eval_mask_specs=eval_mask_specs,
                mask_assign_mode=args.mask_assign_mode,
                mask_value=args.mask_value,
                split_seed=args.split_seed,
                mask_seed=args.mask_seed,
                input_json=input_json,
                action_dim=args.action_dim,
                state_schema=state_schema,
            )
            write_json(
                mam_train_json,
                build_split_metadata_json(
                    source_metadata=metadata,
                    split_name="train",
                    source_episode_ids=source_episode_ids,
                    mask_jobs=train_jobs,
                    input_h5=input_h5,
                    input_json=input_json,
                    env_id=args.env_id,
                    mask_specs=train_mask_specs,
                    requested_train_mask_specs=train_mask_specs,
                    requested_eval_mask_specs=eval_mask_specs,
                    mask_assign_mode=args.mask_assign_mode,
                    split_seed=args.split_seed,
                    mask_seed=args.mask_seed,
                    action_dim=args.action_dim,
                ),
            )
            write_split_h5(
                input_h5=input_h5,
                output_h5=mam_eval_h5,
                split_name="eval",
                source_episode_ids=source_episode_ids,
                mask_jobs=eval_jobs,
                action_min=action_min,
                action_max=action_max,
                state_min=state_min,
                state_max=state_max,
                env_id=args.env_id,
                mask_specs=train_mask_specs,
                requested_train_mask_specs=train_mask_specs,
                requested_eval_mask_specs=eval_mask_specs,
                mask_assign_mode=args.mask_assign_mode,
                mask_value=args.mask_value,
                split_seed=args.split_seed,
                mask_seed=args.mask_seed,
                input_json=input_json,
                action_dim=args.action_dim,
                state_schema=state_schema,
            )
            write_json(
                mam_eval_json,
                build_split_metadata_json(
                    source_metadata=metadata,
                    split_name="eval",
                    source_episode_ids=source_episode_ids,
                    mask_jobs=eval_jobs,
                    input_h5=input_h5,
                    input_json=input_json,
                    env_id=args.env_id,
                    mask_specs=train_mask_specs,
                    requested_train_mask_specs=train_mask_specs,
                    requested_eval_mask_specs=eval_mask_specs,
                    mask_assign_mode=args.mask_assign_mode,
                    split_seed=args.split_seed,
                    mask_seed=args.mask_seed,
                    action_dim=args.action_dim,
                ),
            )

        print(
            f"[placesphere-overfit] overfit{count}: source_success_ids={source_episode_ids}, "
            f"baseline={baseline_dir}, mam={mam_dir}"
        )


if __name__ == "__main__":
    main()
