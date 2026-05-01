import json
import os
from typing import Any, Optional

import h5py
import numpy as np

try:
    from data_preprocess.utils.normalize_utils import load_action_stats_from_path
except ModuleNotFoundError:
    from examples.baselines.diffusion_policy.data_preprocess.utils.normalize_utils import (
        load_action_stats_from_path,
    )


def _list_traj_keys(h5_file: h5py.File, num_traj: Optional[int] = None) -> list[str]:
    keys = sorted(
        [key for key in h5_file.keys() if key.startswith("traj_")],
        key=lambda x: int(x.split("_")[-1]),
    )
    if num_traj is not None:
        if num_traj > len(keys):
            raise ValueError(f"split_num_traj={num_traj} > available trajectories={len(keys)}")
        keys = keys[:num_traj]
    if len(keys) == 0:
        raise ValueError("raw demo contains no traj_* groups")
    return keys


def _split_source_episode_ids(source_episode_ids: list[int], split_seed: int):
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


def _copy_non_traj_groups(src_file: h5py.File, dst_file: h5py.File) -> None:
    for key in src_file.keys():
        if not key.startswith("traj_"):
            src_file.copy(src_file[key], dst_file, key)


def _write_split_h5(raw_h5: str, output_h5: str, source_episode_ids: list[int]) -> None:
    os.makedirs(os.path.dirname(output_h5), exist_ok=True)
    with h5py.File(raw_h5, "r") as src_file, h5py.File(output_h5, "w") as dst_file:
        _copy_non_traj_groups(src_file, dst_file)
        for local_episode_id, source_episode_id in enumerate(source_episode_ids):
            src_key = f"traj_{source_episode_id}"
            if src_key not in src_file:
                raise KeyError(f"missing source trajectory {src_key} in {raw_h5}")
            dst_key = f"traj_{local_episode_id}"
            src_file.copy(src_file[src_key], dst_file, dst_key)
            if "source_episode_id" not in dst_file[dst_key]:
                dst_file[dst_key].create_dataset(
                    "source_episode_id", data=np.int32(source_episode_id)
                )


def _write_split_json(
    raw_json: str,
    output_json: str,
    split_name: str,
    source_episode_ids: list[int],
    raw_h5: str,
    split_seed: int,
) -> None:
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(raw_json, "r", encoding="utf-8") as f:
        source_metadata = json.load(f)
    episodes = source_metadata.get("episodes", [])
    output_metadata = {
        "env_info": source_metadata.get("env_info", {}),
        "commit_info": source_metadata.get("commit_info", {}),
        "episodes": [],
        "split_info": {
            "split": split_name,
            "source_h5": raw_h5,
            "source_json": raw_json,
            "source_episode_ids": [int(x) for x in source_episode_ids],
            "split_seed": int(split_seed),
            "split_rule": "rng permutation, eval_count=max(1, N//6), sorted within each split",
            "normalization": "none",
            "mask": "none",
        },
    }
    for local_episode_id, source_episode_id in enumerate(source_episode_ids):
        if source_episode_id >= len(episodes):
            raise IndexError(
                f"source_episode_id {source_episode_id} exceeds json episodes ({len(episodes)})"
            )
        episode = dict(episodes[source_episode_id])
        episode["episode_id"] = int(local_episode_id)
        episode["source_episode_id"] = int(source_episode_id)
        output_metadata["episodes"].append(episode)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(output_metadata, f, indent=2)


def ensure_raw_train_eval_split(args) -> None:
    if args.raw_demo_h5 is None or len(args.raw_demo_h5.strip()) == 0:
        return
    raw_h5 = os.path.abspath(args.raw_demo_h5)
    if not os.path.exists(raw_h5):
        raise FileNotFoundError(f"raw_demo_h5 not found: {raw_h5}")
    raw_json = (
        os.path.abspath(args.raw_demo_json)
        if args.raw_demo_json is not None and len(args.raw_demo_json.strip()) > 0
        else os.path.splitext(raw_h5)[0] + ".json"
    )
    if not os.path.exists(raw_json):
        raise FileNotFoundError(f"raw_demo_json not found: {raw_json}")

    raw_stem = os.path.splitext(os.path.basename(raw_h5))[0]
    output_dir = (
        os.path.abspath(args.split_output_dir)
        if args.split_output_dir is not None and len(args.split_output_dir.strip()) > 0
        else os.path.join(os.path.dirname(raw_h5), f"{raw_stem}_baseline_split_s{args.split_seed}")
    )
    output_prefix = args.split_output_prefix or raw_stem
    output_stem = f"{output_prefix}_split-s{args.split_seed}"
    train_h5 = os.path.join(output_dir, f"{output_stem}_train.h5")
    train_json = os.path.join(output_dir, f"{output_stem}_train.json")
    eval_h5 = os.path.join(output_dir, f"{output_stem}_eval.h5")
    eval_json = os.path.join(output_dir, f"{output_stem}_eval.json")

    expected = [train_h5, train_json, eval_h5, eval_json]
    if all(os.path.exists(path) for path in expected) and not args.overwrite_split:
        print(f"[baseline-split] reuse existing split: {output_dir}")
    else:
        if not args.overwrite_split:
            existing = [path for path in expected if os.path.exists(path)]
            if existing:
                raise FileExistsError(
                    "partial split outputs exist; set --overwrite-split to replace them: "
                    + ", ".join(existing)
                )
        with h5py.File(raw_h5, "r") as raw_file:
            traj_keys = _list_traj_keys(raw_file, num_traj=args.split_num_traj)
        source_episode_ids = [int(key.split("_")[-1]) for key in traj_keys]
        train_ids, eval_ids = _split_source_episode_ids(
            source_episode_ids=source_episode_ids,
            split_seed=args.split_seed,
        )
        if len(eval_ids) == 0:
            raise ValueError("baseline split produced empty eval set; need at least 2 raw trajectories")
        print(
            f"[baseline-split] generating split into {output_dir}: "
            f"train={len(train_ids)}, eval={len(eval_ids)}, seed={args.split_seed}"
        )
        _write_split_h5(raw_h5, train_h5, train_ids)
        _write_split_h5(raw_h5, eval_h5, eval_ids)
        _write_split_json(raw_json, train_json, "train", train_ids, raw_h5, args.split_seed)
        _write_split_json(raw_json, eval_json, "eval", eval_ids, raw_h5, args.split_seed)

    args.demo_path = train_h5
    args.eval_demo_path = eval_h5
    args.eval_demo_metadata_path = eval_json
    print(f"[baseline-split] train demo: {args.demo_path}")
    print(f"[baseline-split] eval demo: {args.eval_demo_path}")


def load_action_denorm_stats(action_norm_path: str):
    if action_norm_path is None or len(action_norm_path.strip()) == 0:
        raise ValueError(
            "action_norm_path is required. Please pass --action-norm-path to provide min/max for denormalization."
        )
    mins, maxs = load_action_stats_from_path(action_norm_path)
    print(f"[denorm] loaded action norm stats from {action_norm_path}, dims={mins.shape[0]}")
    return mins, maxs


def save_action_norm_stats(action_norm_path: str, mins: np.ndarray, maxs: np.ndarray):
    os.makedirs(os.path.dirname(action_norm_path), exist_ok=True)
    with open(action_norm_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "min": np.asarray(mins, dtype=np.float32).tolist(),
                "max": np.asarray(maxs, dtype=np.float32).tolist(),
            },
            f,
            indent=2,
        )


def _candidate_metadata_paths(demo_path: str, explicit_meta_path: Optional[str] = None):
    candidates = []
    if explicit_meta_path is not None and len(explicit_meta_path.strip()) > 0:
        candidates.append(explicit_meta_path)
    if demo_path.endswith(".json"):
        candidates.append(demo_path)
    else:
        base, _ = os.path.splitext(demo_path)
        candidates.append(f"{base}.json")
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


def infer_loaded_traj_ids_from_demo(demo_path: str, num_traj: Optional[int] = None):
    if demo_path.endswith(".json"):
        with open(demo_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        traj_ids = [
            int(episode.get("episode_id", i))
            for i, episode in enumerate(meta.get("episodes", []))
        ]
    else:
        if not os.path.exists(demo_path):
            raise FileNotFoundError(f"eval demo path not found: {demo_path}")
        with h5py.File(demo_path, "r") as f:
            traj_keys = [key for key in f.keys() if key.startswith("traj_")]
        if len(traj_keys) == 0:
            raise ValueError(f"no traj_* found in eval demo path: {demo_path}")
        traj_ids = sorted(int(key.split("_")[-1]) for key in traj_keys)

    if num_traj is not None:
        if num_traj > len(traj_ids):
            raise ValueError(
                f"num_eval_demos={num_traj} > available eval trajectories={len(traj_ids)}"
            )
        traj_ids = traj_ids[:num_traj]
    return traj_ids


def _episode_to_reset_kwargs(episode: dict[str, Any]) -> dict[str, Any]:
    reset_kwargs = dict(episode.get("reset_kwargs", {}) or {})
    reset_seed = reset_kwargs.get("seed", episode.get("episode_seed", None))
    if isinstance(reset_seed, list):
        reset_seed = reset_seed[0] if len(reset_seed) > 0 else None
    if reset_seed is not None:
        reset_kwargs["seed"] = int(reset_seed)
    return reset_kwargs


def load_eval_reset_kwargs_list(
    eval_demo_path: str,
    eval_demo_metadata_path: Optional[str] = None,
    num_traj: Optional[int] = None,
) -> list[dict[str, Any]]:
    traj_ids = infer_loaded_traj_ids_from_demo(eval_demo_path, num_traj=num_traj)
    for meta_path in _candidate_metadata_paths(eval_demo_path, eval_demo_metadata_path):
        if not os.path.exists(meta_path):
            continue
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            episodes = meta.get("episodes", [])
            episode_by_id = {
                int(episode.get("episode_id", i)): episode
                for i, episode in enumerate(episodes)
            }
            reset_kwargs_list = []
            for traj_id in traj_ids:
                episode = episode_by_id.get(traj_id, None)
                if episode is None and 0 <= traj_id < len(episodes):
                    episode = episodes[traj_id]
                if episode is None:
                    reset_kwargs_list = []
                    break
                reset_kwargs_list.append(_episode_to_reset_kwargs(episode))
            if len(reset_kwargs_list) == len(traj_ids):
                seeds = [kwargs.get("seed", None) for kwargs in reset_kwargs_list]
                print(
                    f"[eval-split] loaded {len(reset_kwargs_list)} eval reset kwargs "
                    f"from {meta_path}"
                )
                print(f"[eval-split] eval trajectory ids: {traj_ids[:20]}")
                print(f"[eval-split] eval seeds: {seeds[:20]}")
                return reset_kwargs_list
        except Exception as exc:
            print(f"[eval-split] failed reading {meta_path}: {exc}")

    raise ValueError(
        "failed to infer eval reset kwargs from eval demo metadata. "
        "Pass --eval-demo-metadata-path explicitly if the json is not next to eval_demo_path."
    )
