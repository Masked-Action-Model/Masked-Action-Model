#!/usr/bin/env python3
import argparse
import h5py


def traj_key_sort(key: str) -> int:
    if not key.startswith("traj_"):
        return -1
    try:
        return int(key.split("_")[-1])
    except ValueError:
        return -1


def list_traj_keys(h5file):
    keys = [k for k in h5file.keys() if k.startswith("traj_")]
    return sorted(keys, key=traj_key_sort)


def copy_group(src, dst, name):
    dst.copy(src, name)


def copy_non_traj_groups(src, dst):
    for k in src.keys():
        if not k.startswith("traj_"):
            copy_group(src[k], dst, k)


def main():
    parser = argparse.ArgumentParser(
        description="Order traj_* groups and split first N trajectories into train/eval."
    )
    parser.add_argument(
        "--input",
        default="demos/data_1/data_1_concat.h5",
        help="Input concatenated H5 path",
    )
    parser.add_argument(
        "--ordered-out",
        default="demos/data_1/data_1_concat_ordered.h5",
        help="Output H5 with ordered traj_* groups",
    )
    parser.add_argument(
        "--train-out",
        default="demos/data_1/data_1_train.h5",
        help="Train output H5",
    )
    parser.add_argument(
        "--eval-out",
        default="demos/data_1/data_1_eval.h5",
        help="Eval output H5",
    )
    parser.add_argument("--limit", type=int, default=600, help="Total trajectories to use")
    parser.add_argument("--train-count", type=int, default=500, help="Train trajectories count")
    args = parser.parse_args()

    if args.train_count > args.limit:
        raise ValueError("train-count must be <= limit")

    with h5py.File(args.input, "r") as f_in:
        traj_keys = list_traj_keys(f_in)
        if len(traj_keys) < args.limit:
            raise RuntimeError(
                f"Not enough trajectories: found {len(traj_keys)}, need {args.limit}"
            )

        # 1) Write ordered concat file
        with h5py.File(args.ordered_out, "w") as f_ordered:
            for k, v in f_in.attrs.items():
                f_ordered.attrs[k] = v
            copy_non_traj_groups(f_in, f_ordered)
            for k in traj_keys:
                copy_group(f_in[k], f_ordered, k)

        # 2) Split first limit into train/eval
        selected = traj_keys[: args.limit]
        train_keys = selected[: args.train_count]
        eval_keys = selected[args.train_count :]

        with h5py.File(args.train_out, "w") as f_train:
            for k, v in f_in.attrs.items():
                f_train.attrs[k] = v
            copy_non_traj_groups(f_in, f_train)
            for k in train_keys:
                copy_group(f_in[k], f_train, k)

        with h5py.File(args.eval_out, "w") as f_eval:
            for k, v in f_in.attrs.items():
                f_eval.attrs[k] = v
            copy_non_traj_groups(f_in, f_eval)
            for k in eval_keys:
                copy_group(f_in[k], f_eval, k)


if __name__ == "__main__":
    main()
