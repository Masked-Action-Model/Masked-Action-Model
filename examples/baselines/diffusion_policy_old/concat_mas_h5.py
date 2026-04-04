#!/usr/bin/env python3
import argparse
import os
import h5py
import numpy as np


def list_traj_keys(h5file):
    return sorted([k for k in h5file.keys() if k.startswith("traj_")], key=lambda x: int(x.split("_")[-1]))


def copy_group(src_group, dst_group, name):
    # h5py allows copying objects across files by providing the destination group
    dst_group.copy(src_group, name)


def split_h5_train_eval(src_path, train_ratio=5, eval_ratio=1):
    """Split traj_* groups from src_path into <stem>_train.h5 and <stem>_eval.h5 by ratio."""
    if train_ratio <= 0 or eval_ratio <= 0:
        raise ValueError("train_ratio and eval_ratio must be positive integers.")

    base = src_path[:-3] if src_path.endswith(".h5") else src_path
    train_path = f"{base}_train.h5"
    eval_path = f"{base}_eval.h5"

    with h5py.File(src_path, "r") as f_src, h5py.File(train_path, "w") as f_train, h5py.File(eval_path, "w") as f_eval:
        trajs = list_traj_keys(f_src)
        if not trajs:
            raise RuntimeError(f"No traj_* groups found in {src_path}")
        cycle = train_ratio + eval_ratio

        if "meta" in f_src:
            copy_group(f_src["meta"], f_train, "meta")
            copy_group(f_src["meta"], f_eval, "meta")

        for i, t in enumerate(trajs):
            if (i % cycle) < train_ratio:
                copy_group(f_src[t], f_train, t)
            else:
                copy_group(f_src[t], f_eval, t)

    print(f"Split done: {train_path} and {eval_path} (ratio {train_ratio}:{eval_ratio})")


def main():
    parser = argparse.ArgumentParser(description="Concat original actions with masked action space into one H5 dataset.")
    parser.add_argument("--orig", required=True, help="Path to original dataset (with actions)")
    parser.add_argument("--normed", required=True, help="Path to normalized dataset (actions only)")
    parser.add_argument("--masked", required=True, help="Path to masked dataset (with padded actions)")
    parser.add_argument("--out", required=True, help="Output H5 path")
    parser.add_argument("--mask-name", default="mask", help="Dataset name for mask")
    parser.add_argument("--mas-name", default="mas", help="Dataset name for masked action space")
    parser.add_argument("--actions-name", default="actions", help="Dataset name for original actions")
    parser.add_argument("--train-ratio", type=int, default=5, help="Train split ratio")
    parser.add_argument("--eval-ratio", type=int, default=1, help="Eval split ratio")
    args = parser.parse_args()

    with h5py.File(args.orig, "r") as f_orig, h5py.File(args.normed, "r") as f_normed, h5py.File(args.masked, "r") as f_masked, h5py.File(args.out, "w") as f_out:
        # meta
        if "meta" in f_masked:
            copy_group(f_masked["meta"], f_out, "meta")
            padding_value = f_masked["meta"]["padding_value"][()]
        else:
            padding_value = -1.0

        orig_trajs = set(list_traj_keys(f_orig))
        normed_trajs = set(list_traj_keys(f_normed))
        masked_trajs = list_traj_keys(f_masked)
        common_trajs = [t for t in masked_trajs if t in orig_trajs and t in normed_trajs]
        if not common_trajs:
            raise RuntimeError("No common traj_* groups among orig, normed, and masked files.")

        for t in common_trajs:
            g_out = f_out.create_group(t)

            # Copy obs/env_states/flags from original file
            copy_group(f_orig[t]["obs"], g_out, "obs")
            copy_group(f_orig[t]["env_states"], g_out, "env_states")
            for flag in ["success", "terminated", "truncated"]:
                g_out.create_dataset(flag, data=f_orig[t][flag][...], dtype=f_orig[t][flag].dtype)

            # mas comes from masked file (with padding)
            mas = f_masked[t]["actions"][...]
            g_out.create_dataset(args.mas_name, data=mas, dtype=mas.dtype)

            # mask is computed from masked mas, then trimmed to normalized action length (no padding in output mask)
            full_mask = (mas != padding_value).astype(np.float32)
            normed_actions = f_normed[t]["actions"][...]
            if normed_actions.shape[0] > full_mask.shape[0]:
                raise RuntimeError(
                    f"{t}: normed actions length {normed_actions.shape[0]} > masked length {full_mask.shape[0]}"
                )
            mask = full_mask[: normed_actions.shape[0]]
            g_out.create_dataset(args.mask_name, data=mask, dtype=np.float32)

            # Only actions come from normalized file (no padding)
            g_out.create_dataset(args.actions_name, data=normed_actions, dtype=normed_actions.dtype)

            # keep other top-level attributes if any
            for k, v in f_masked[t].attrs.items():
                g_out.attrs[k] = v

    split_h5_train_eval(args.out, train_ratio=args.train_ratio, eval_ratio=args.eval_ratio)
    if os.path.exists(args.out):
        os.remove(args.out)
        print(f"Removed intermediate file: {args.out}")


if __name__ == "__main__":
    main()
