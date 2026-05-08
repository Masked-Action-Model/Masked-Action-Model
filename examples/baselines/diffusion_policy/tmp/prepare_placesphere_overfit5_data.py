from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import h5py


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare identical train/eval MAM data for PlaceSphere 5-demo overfit."
    )
    parser.add_argument(
        "--input-h5",
        type=Path,
        default=Path("demos/PlaceSphere-v1/PlaceSphere-v1.rgb.pd_ee_pose.physx_cpu.h5"),
    )
    parser.add_argument("--input-json", type=Path, default=None)
    parser.add_argument(
        "--train-h5",
        type=Path,
        default=Path("demos/PlaceSphere-v1_preprocessed/overfit5/placesphere_mam_overfit5_train.h5"),
    )
    parser.add_argument(
        "--train-json",
        type=Path,
        default=Path("demos/PlaceSphere-v1_preprocessed/overfit5/placesphere_mam_overfit5_train.json"),
    )
    parser.add_argument(
        "--eval-h5",
        type=Path,
        default=Path("demos/PlaceSphere-v1_preprocessed/overfit5/placesphere_mam_overfit5_eval.h5"),
    )
    parser.add_argument(
        "--eval-json",
        type=Path,
        default=Path("demos/PlaceSphere-v1_preprocessed/overfit5/placesphere_mam_overfit5_eval.json"),
    )
    parser.add_argument("--env-id", type=str, default="PlaceSphere-v1")
    parser.add_argument("--num-traj", type=int, default=5)
    parser.add_argument("--action-dim", type=int, choices=[6, 7], default=7)
    parser.add_argument("--mask-assign-mode", choices=["composition", "one_demo_multi_mask"], default="composition")
    parser.add_argument("--train-num-mask-type", type=int, default=2)
    parser.add_argument("--train-mask-type-list", type=str, default='["random_mask","points"]')
    parser.add_argument("--train-mask-composition-list", type=str, default="[0.5,0.5]")
    parser.add_argument("--train-mask-ratio-list", type=str, default="[0.2,0.2]")
    parser.add_argument("--eval-num-mask-type", type=int, default=None)
    parser.add_argument("--eval-mask-type-list", type=str, default=None)
    parser.add_argument("--eval-mask-composition-list", type=str, default=None)
    parser.add_argument("--eval-mask-ratio-list", type=str, default=None)
    parser.add_argument("--mask-value", type=float, default=0.0)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--mask-seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def existing_outputs_are_complete(paths: list[Path]) -> bool:
    existing = [path.exists() for path in paths]
    if all(existing):
        return True
    if any(existing):
        missing = [str(path) for path in paths if not path.exists()]
        raise FileExistsError(f"partial overfit outputs exist, missing: {missing}")
    return False


def main() -> None:
    args = parse_args()
    args.action_dim = validate_action_dim(args.action_dim)

    input_h5 = args.input_h5.resolve()
    input_json = (
        args.input_json.resolve()
        if args.input_json is not None
        else args.input_h5.with_suffix(".json").resolve()
    )
    output_paths = [
        args.train_h5.resolve(),
        args.train_json.resolve(),
        args.eval_h5.resolve(),
        args.eval_json.resolve(),
    ]
    if not args.overwrite and existing_outputs_are_complete(output_paths):
        print(f"[placesphere-overfit5] reuse existing MAM data: {output_paths[0].parent}")
        return

    if not input_h5.exists():
        raise FileNotFoundError(f"input h5 not found: {input_h5}")
    if not input_json.exists():
        raise FileNotFoundError(f"input json not found: {input_json}")

    train_mask_specs = normalize_split_mask_config(
        SimpleNamespace(**vars(args)),
        split="train",
        mask_assign_mode=args.mask_assign_mode,
    )
    eval_mask_specs = normalize_split_mask_config(
        SimpleNamespace(**vars(args)),
        split="eval",
        mask_assign_mode=args.mask_assign_mode,
    )

    with h5py.File(input_h5, "r") as src_file:
        traj_keys = list_traj_keys(src_file, num_traj=args.num_traj)
        metadata = read_json(input_json)
        validate_dataset_alignment(src_file, metadata, traj_keys)
        source_episode_ids = [int(traj_key.split("_")[-1]) for traj_key in traj_keys]
        action_min, action_max = compute_global_min_max(
            iter_selected_action_arrays(src_file, traj_keys, action_dim=args.action_dim)
        )
        state_min, state_max = compute_global_min_max(
            iter_selected_state_arrays(src_file, traj_keys)
        )
        state_schema = build_state_schema_from_obs_group(src_file[traj_keys[0]]["obs"])
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

    for output_path in output_paths:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    write_split_h5(
        input_h5=input_h5,
        output_h5=output_paths[0],
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
        output_paths[1],
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
        output_h5=output_paths[2],
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
        output_paths[3],
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
        "[placesphere-overfit5] wrote identical MAM train/eval data: "
        f"source_trajs={len(source_episode_ids)}, expanded_trajs={len(train_jobs)}, "
        f"train={output_paths[0]}, eval={output_paths[2]}"
    )


if __name__ == "__main__":
    main()
