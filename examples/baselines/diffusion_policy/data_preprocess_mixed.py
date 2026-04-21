from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

try:
    from data_preprocess import (
        default_output_dir,
        iter_selected_action_arrays,
        iter_selected_state_arrays,
        load_state_matrix_from_obs_group,
        split_source_episode_ids,
        validate_dataset_alignment,
        validate_inputs,
    )
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
        build_mask_spec,
        validate_mixed_mask_config,
    )
    from data_preprocess_tools.normalize_utils import (
        compute_global_min_max,
        normalize_selected_dims,
    )
    from data_preprocess_tools.progress_utils import (
        MAS_ACTION_DIM,
        MAS_STEP_DIM,
        augment_mas_with_progress_np,
    )
except ModuleNotFoundError:
    from examples.baselines.diffusion_policy.data_preprocess import (
        default_output_dir,
        iter_selected_action_arrays,
        iter_selected_state_arrays,
        load_state_matrix_from_obs_group,
        split_source_episode_ids,
        validate_dataset_alignment,
        validate_inputs,
    )
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
        build_mask_spec,
        validate_mixed_mask_config,
    )
    from examples.baselines.diffusion_policy.data_preprocess_tools.normalize_utils import (
        compute_global_min_max,
        normalize_selected_dims,
    )
    from examples.baselines.diffusion_policy.data_preprocess_tools.progress_utils import (
        MAS_ACTION_DIM,
        MAS_STEP_DIM,
        augment_mas_with_progress_np,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess ManiSkill diffusion-policy demos into mixed-mask train/eval HDF5 datasets."
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
        "--mask-assign-mode",
        type=str,
        choices=["composition", "one_demo_multi_mask"],
        default="composition",
        help=(
            "mixed mask 分配模式：composition=每条 demo 按比例分配一个 mask；"
            "one_demo_multi_mask=每条 demo 复制为每个 mask slot 各一条。"
        ),
    )
    parser.add_argument(
        "--num-mask-type",
        type=int,
        default=0,
        help="mixed mask 中 mask type 的数量；0 表示 none",
    )
    parser.add_argument(
        "--mask-type-list",
        type=str,
        default="[]",
        help="mask type 列表，支持 JSON / Python list / 逗号分隔字符串",
    )
    parser.add_argument(
        "--mask-composition-list",
        dest="mask_composition_list",
        type=str,
        default="[]",
        help="mask composition 列表，支持 JSON / Python list / 逗号分隔字符串",
    )
    parser.add_argument(
        "--mask-ratio-list",
        dest="mask_ratio_list",
        type=str,
        default="[]",
        help="mask ratio 列表，支持 JSON / Python list / 逗号分隔字符串",
    )
    parser.add_argument(
        "--train-num-mask-type",
        type=int,
        default=None,
        help="训练集 mixed mask 的 mask type 数量；未提供时回退到 --num-mask-type",
    )
    parser.add_argument(
        "--train-mask-type-list",
        type=str,
        default=None,
        help="训练集 mask type 列表；未提供时回退到 --mask-type-list",
    )
    parser.add_argument(
        "--train-mask-composition-list",
        dest="train_mask_composition_list",
        type=str,
        default=None,
        help="训练集 mask composition 列表；未提供时回退到 --mask-composition-list",
    )
    parser.add_argument(
        "--train-mask-ratio-list",
        dest="train_mask_ratio_list",
        type=str,
        default=None,
        help="训练集 mask ratio 列表；未提供时回退到 --mask-ratio-list",
    )
    parser.add_argument(
        "--eval-num-mask-type",
        type=int,
        default=None,
        help="评估集 mixed mask 的 mask type 数量；未提供时默认复制训练集配置",
    )
    parser.add_argument(
        "--eval-mask-type-list",
        type=str,
        default=None,
        help="评估集 mask type 列表；未提供时默认复制训练集配置",
    )
    parser.add_argument(
        "--eval-mask-composition-list",
        dest="eval_mask_composition_list",
        type=str,
        default=None,
        help="评估集 mask composition 列表；未提供时默认复制训练集配置",
    )
    parser.add_argument(
        "--eval-mask-ratio-list",
        dest="eval_mask_ratio_list",
        type=str,
        default=None,
        help="评估集 mask ratio 列表；未提供时默认复制训练集配置",
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
        help="mixed mask 分配和逐轨迹 mask 的随机种子基数",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖已有输出文件",
    )
    return parser.parse_args()


def _parse_list_arg(raw_value: str, arg_name: str) -> list[Any]:
    text = "" if raw_value is None else str(raw_value).strip()
    if len(text) == 0:
        return []

    # Accept shell values like '"[1,2]"' or "'[1,2]'" by peeling one outer quote layer.
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        inner = text[1:-1].strip()
        if len(inner) > 0:
            text = inner

    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
            if isinstance(parsed, list):
                return parsed
            if isinstance(parsed, str) and parsed.strip() != text:
                return _parse_list_arg(parsed, arg_name)
        except Exception:
            pass

    if "," in text:
        return [item.strip() for item in text.split(",")]

    raise ValueError(f"failed to parse {arg_name}: {raw_value!r}")


def format_float_suffix(value: float) -> str:
    return format(float(value), "g")


def _spec_param_token(mask_spec: dict[str, Any]) -> str:
    if mask_spec["mask_type"] in MASK_TYPES_REQUIRING_RATIO:
        return f"r{format_float_suffix(mask_spec['retain_ratio'])}"
    if mask_spec["mask_type"] in MASK_TYPES_REQUIRING_SEQ_LEN:
        return f"seq{int(mask_spec['mask_seq_len'])}"
    return "nop"


def _resolve_split_mask_inputs(
    args: argparse.Namespace,
    split: str,
) -> tuple[int, str, str, str]:
    if split not in {"train", "eval"}:
        raise ValueError(f"unsupported split={split!r}")

    shared_num = int(getattr(args, "num_mask_type", 0))
    shared_type_list = str(getattr(args, "mask_type_list", "[]"))
    shared_composition_list = str(getattr(args, "mask_composition_list", "[]"))
    shared_ratio_list = str(getattr(args, "mask_ratio_list", "[]"))

    train_num = getattr(args, "train_num_mask_type", None)
    train_type_list = getattr(args, "train_mask_type_list", None)
    train_composition_list = getattr(args, "train_mask_composition_list", None)
    train_ratio_list = getattr(args, "train_mask_ratio_list", None)

    if split == "train":
        return (
            shared_num if train_num is None else int(train_num),
            shared_type_list if train_type_list is None else str(train_type_list),
            shared_composition_list
            if train_composition_list is None
            else str(train_composition_list),
            shared_ratio_list if train_ratio_list is None else str(train_ratio_list),
        )

    eval_num = getattr(args, "eval_num_mask_type", None)
    eval_type_list = getattr(args, "eval_mask_type_list", None)
    eval_composition_list = getattr(args, "eval_mask_composition_list", None)
    eval_ratio_list = getattr(args, "eval_mask_ratio_list", None)
    train_resolved = _resolve_split_mask_inputs(args, split="train")
    return (
        train_resolved[0] if eval_num is None else int(eval_num),
        train_resolved[1] if eval_type_list is None else str(eval_type_list),
        train_resolved[2] if eval_composition_list is None else str(eval_composition_list),
        train_resolved[3] if eval_ratio_list is None else str(eval_ratio_list),
    )


def _attach_mask_slot_metadata(mask_specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counters: dict[str, int] = {}
    out = []
    for spec_index, spec in enumerate(mask_specs):
        one_spec = copy.deepcopy(spec)
        base_type = str(one_spec["mask_type"])
        slot_index = counters.get(base_type, 0)
        counters[base_type] = slot_index + 1
        slot_name = f"{base_type}#{slot_index}"
        one_spec["mask_spec_index"] = int(spec_index)
        one_spec["mask_type_base"] = base_type
        one_spec["mask_slot_index"] = int(slot_index)
        one_spec["mask_type_slot"] = slot_name
        one_spec["mask_slot_name"] = slot_name
        out.append(one_spec)
    return out


def _build_readable_spec_token(mask_spec: dict[str, Any]) -> str:
    return (
        f"{mask_spec['mask_type_base']}-s{int(mask_spec['mask_slot_index'])}-"
        f"{_spec_param_token(mask_spec)}-mix{format_float_suffix(mask_spec['ratio'])}"
    )


def build_output_stem(
    output_prefix: str,
    train_mask_specs: list[dict[str, Any]],
    eval_mask_specs: list[dict[str, Any]] | None = None,
    mask_assign_mode: str = "composition",
) -> str:
    if eval_mask_specs is None:
        eval_mask_specs = train_mask_specs
    if mask_assign_mode not in {"composition", "one_demo_multi_mask"}:
        raise ValueError(f"unsupported mask_assign_mode={mask_assign_mode!r}")
    mode_token = "mixed" if mask_assign_mode == "composition" else "onedemo"

    if (
        len(train_mask_specs) == 1
        and len(eval_mask_specs) == 1
        and train_mask_specs[0]["mask_type"] == eval_mask_specs[0]["mask_type"]
        and train_mask_specs[0]["mask_type"] in {"none", "full"}
    ):
        if mask_assign_mode != "composition":
            return f"{output_prefix}_{mode_token}_{train_mask_specs[0]['mask_type']}"
        return f"{output_prefix}_{train_mask_specs[0]['mask_type']}"

    train_tokens = [_build_readable_spec_token(spec) for spec in train_mask_specs]
    eval_tokens = [_build_readable_spec_token(spec) for spec in eval_mask_specs]
    if train_mask_specs == eval_mask_specs:
        readable = "__".join(train_tokens)
    else:
        readable = f"train__{'__'.join(train_tokens)}__eval__{'__'.join(eval_tokens)}"
    canonical_json = json.dumps(
        {
            "mask_assign_mode": mask_assign_mode,
            "train_mask_specs": train_mask_specs,
            "eval_mask_specs": eval_mask_specs,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha1(canonical_json.encode("utf-8")).hexdigest()[:10]
    if len(readable) > 120:
        train_readable = "__".join(spec["mask_type_base"] for spec in train_mask_specs)
        eval_readable = "__".join(spec["mask_type_base"] for spec in eval_mask_specs)
        readable = (
            train_readable
            if train_mask_specs == eval_mask_specs
            else f"train__{train_readable}__eval__{eval_readable}"
        )
    return f"{output_prefix}_{mode_token}_{readable}_{digest}"


def normalize_split_mask_config(
    args: argparse.Namespace,
    split: str,
    mask_assign_mode: str = "composition",
) -> list[dict[str, Any]]:
    if mask_assign_mode not in {"composition", "one_demo_multi_mask"}:
        raise ValueError(f"unsupported mask_assign_mode={mask_assign_mode!r}")
    num_mask_type, mask_type_list_raw, mask_composition_list_raw, mask_ratio_list_raw = (
        _resolve_split_mask_inputs(args, split=split)
    )
    if int(num_mask_type) == 0:
        return _attach_mask_slot_metadata(
            [build_mask_spec(mask_type="none", raw_param=None, ratio=1.0)]
        )

    mask_type_list = [
        str(v) for v in _parse_list_arg(mask_type_list_raw, f"{split}_mask_type_list")
    ]
    mask_ratio_list = _parse_list_arg(mask_ratio_list_raw, f"{split}_mask_ratio_list")
    if mask_assign_mode == "composition":
        mask_composition_list = [
            float(v)
            for v in _parse_list_arg(mask_composition_list_raw, f"{split}_mask_composition_list")
        ]
    else:
        if len(mask_type_list) != int(num_mask_type):
            raise ValueError(
                f"len(mask_type_list)={len(mask_type_list)} != num_mask_type={num_mask_type}"
            )
        if len(mask_ratio_list) != int(num_mask_type):
            raise ValueError(
                f"len(mask_ratio_list)={len(mask_ratio_list)} != num_mask_type={num_mask_type}"
            )
        mask_composition_list = [
            1.0 / float(num_mask_type) for _ in range(int(num_mask_type))
        ]
    validate_mixed_mask_config(
        num_mask_type=int(num_mask_type),
        mask_type_list=mask_type_list,
        mask_composition_list=mask_composition_list,
        mask_ratio_list=mask_ratio_list,
    )
    return _attach_mask_slot_metadata(
        [
            build_mask_spec(mask_type=mask_type, raw_param=raw_param, ratio=ratio)
            for mask_type, ratio, raw_param in zip(
                mask_type_list, mask_composition_list, mask_ratio_list
            )
        ]
    )


def normalize_mixed_mask_config(args: argparse.Namespace) -> list[dict[str, Any]]:
    return normalize_split_mask_config(
        args,
        split="train",
        mask_assign_mode=getattr(args, "mask_assign_mode", "composition"),
    )


def largest_remainder_counts(total_items: int, ratios: list[float]) -> list[int]:
    if total_items < 0:
        raise ValueError(f"total_items must be non-negative, got {total_items}")
    if len(ratios) == 0:
        raise ValueError("ratios must be non-empty")
    if total_items == 0:
        return [0 for _ in ratios]

    raw = np.asarray(ratios, dtype=np.float64) * float(total_items)
    counts = np.floor(raw).astype(np.int64)
    remaining = int(total_items - int(counts.sum()))
    if remaining > 0:
        remainders = raw - counts.astype(np.float64)
        order = sorted(
            range(len(ratios)),
            key=lambda idx: (-remainders[idx], idx),
        )
        for idx in order[:remaining]:
            counts[idx] += 1
    return [int(v) for v in counts.tolist()]


def assign_mask_specs_to_episodes(
    source_episode_ids: list[int],
    mask_specs: list[dict[str, Any]],
    seed: int,
) -> dict[int, dict[str, Any]]:
    if len(source_episode_ids) == 0:
        return {}
    ratios = [float(spec["ratio"]) for spec in mask_specs]
    counts = largest_remainder_counts(len(source_episode_ids), ratios)
    if sum(counts) != len(source_episode_ids):
        raise AssertionError("assigned mask counts must cover every source episode")

    rng = np.random.default_rng(seed)
    shuffled_ids = rng.permutation(np.asarray(source_episode_ids, dtype=np.int64)).tolist()
    assigned = {}
    cursor = 0
    for spec_idx, spec in enumerate(mask_specs):
        count = counts[spec_idx]
        for source_episode_id in shuffled_ids[cursor : cursor + count]:
            assigned[int(source_episode_id)] = copy.deepcopy(spec)
        cursor += count
    if len(assigned) != len(source_episode_ids):
        raise AssertionError("failed to assign a mask spec to every source episode")
    return assigned


def build_mask_jobs(
    source_episode_ids: list[int],
    mask_specs: list[dict[str, Any]],
    mask_assign_mode: str,
    seed: int,
) -> list[dict[str, Any]]:
    if mask_assign_mode not in {"composition", "one_demo_multi_mask"}:
        raise ValueError(f"unsupported mask_assign_mode={mask_assign_mode!r}")

    if mask_assign_mode == "composition":
        assigned_mask_specs = assign_mask_specs_to_episodes(
            source_episode_ids=source_episode_ids,
            mask_specs=mask_specs,
            seed=seed,
        )
        return [
            dict(
                source_episode_id=int(source_episode_id),
                mask_spec=copy.deepcopy(assigned_mask_specs[int(source_episode_id)]),
                source_mask_copy_key=(
                    f"traj_{int(source_episode_id)}_"
                    f"{int(assigned_mask_specs[int(source_episode_id)]['mask_spec_index'])}"
                ),
                rng_seed=int(seed) + int(source_episode_id),
            )
            for source_episode_id in source_episode_ids
        ]

    jobs = []
    for source_episode_id in source_episode_ids:
        for spec in mask_specs:
            spec_index = int(spec["mask_spec_index"])
            jobs.append(
                dict(
                    source_episode_id=int(source_episode_id),
                    mask_spec=copy.deepcopy(spec),
                    source_mask_copy_key=f"traj_{int(source_episode_id)}_{spec_index}",
                    rng_seed=(
                        int(seed)
                        + int(source_episode_id) * 1009
                        + int(spec_index) * 9176
                    ),
                )
            )
    rng = np.random.default_rng(seed)
    if len(jobs) > 1:
        order = rng.permutation(len(jobs)).tolist()
        jobs = [jobs[i] for i in order]
    return jobs


def _mask_param_repr(mask_spec: dict[str, Any]) -> Any:
    if mask_spec["mask_type"] in MASK_TYPES_REQUIRING_RATIO:
        return float(mask_spec["retain_ratio"])
    if mask_spec["mask_type"] in MASK_TYPES_REQUIRING_SEQ_LEN:
        return int(mask_spec["mask_seq_len"])
    return None


def build_split_metadata_json(
    source_metadata: dict,
    split_name: str,
    source_episode_ids: list[int],
    mask_jobs: list[dict[str, Any]],
    input_h5: Path,
    input_json: Path,
    env_id: str,
    mask_specs: list[dict[str, Any]],
    requested_train_mask_specs: list[dict[str, Any]],
    requested_eval_mask_specs: list[dict[str, Any]],
    mask_assign_mode: str,
    split_seed: int,
    mask_seed: int,
) -> dict:
    mask_composition_list = [float(spec["ratio"]) for spec in mask_specs]
    mask_ratio_list = [_mask_param_repr(spec) for spec in mask_specs]
    expanded_source_episode_ids = [
        int(job["source_episode_id"]) for job in mask_jobs
    ]
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
            "expanded_source_episode_ids": expanded_source_episode_ids,
            "mixed_mask_enabled": True,
            "mask_assign_mode": str(mask_assign_mode),
            "source_num_episodes": int(len(source_episode_ids)),
            "expanded_num_episodes": int(len(mask_jobs)),
            "num_mask_type": int(len(mask_specs)),
            "mask_specs": copy.deepcopy(mask_specs),
            "mask_type_list": [spec["mask_type"] for spec in mask_specs],
            "mask_composition_list": mask_composition_list,
            "mask_ratio_list": mask_ratio_list,
            "mask_type_slot_list": [spec["mask_type_slot"] for spec in mask_specs],
            "mask_slot_name_list": [spec["mask_type_slot"] for spec in mask_specs],
            "mask_slot_ratio_list": mask_composition_list,
            "mask_slot_param_list": mask_ratio_list,
            "requested_train_mask_specs": copy.deepcopy(requested_train_mask_specs),
            "requested_eval_mask_specs": copy.deepcopy(requested_eval_mask_specs),
            "split_seed": int(split_seed),
            "mask_seed": int(mask_seed),
            "mas_action_dim": MAS_ACTION_DIM,
            "mas_dim": MAS_STEP_DIM,
        },
    }
    episodes = source_metadata.get("episodes", [])
    for local_episode_id, job in enumerate(mask_jobs):
        source_episode_id = int(job["source_episode_id"])
        if source_episode_id >= len(episodes):
            raise IndexError(
                f"source_episode_id {source_episode_id} exceeds json episodes ({len(episodes)})"
            )
        episode = copy.deepcopy(episodes[source_episode_id])
        mask_spec = job["mask_spec"]
        episode["episode_id"] = int(local_episode_id)
        episode["source_episode_id"] = int(source_episode_id)
        episode["source_mask_copy_key"] = str(job["source_mask_copy_key"])
        episode["source_mask_copy_index"] = int(mask_spec["mask_spec_index"])
        episode["mask_type"] = str(mask_spec["mask_type"])
        episode["mask_type_slot"] = str(mask_spec["mask_type_slot"])
        episode["mask_slot_name"] = str(mask_spec["mask_type_slot"])
        episode["mask_slot_index"] = int(mask_spec["mask_slot_index"])
        if mask_spec["retain_ratio"] is not None:
            episode["retain_ratio"] = float(mask_spec["retain_ratio"])
        if mask_spec["mask_seq_len"] is not None:
            episode["mask_seq_len"] = int(mask_spec["mask_seq_len"])
        output_metadata["episodes"].append(episode)
    return output_metadata


def _write_mask_spec_meta(
    meta_group: h5py.Group,
    mask_specs: list[dict[str, Any]],
    requested_train_mask_specs: list[dict[str, Any]],
    requested_eval_mask_specs: list[dict[str, Any]],
    mask_assign_mode: str,
    source_num_episodes: int,
    expanded_num_episodes: int,
) -> None:
    meta_group.create_dataset("mixed_mask_enabled", data=np.bool_(True))
    meta_group.create_dataset("num_mask_type", data=np.int32(len(mask_specs)))
    meta_group.create_dataset("source_num_episodes", data=np.int32(source_num_episodes))
    meta_group.create_dataset("expanded_num_episodes", data=np.int32(expanded_num_episodes))
    write_string_dataset(meta_group, "mask_assign_mode", str(mask_assign_mode))
    write_string_dataset(
        meta_group,
        "mask_specs_json",
        json.dumps(mask_specs, sort_keys=True),
    )
    write_string_dataset(
        meta_group,
        "mask_type_list_json",
        json.dumps([spec["mask_type"] for spec in mask_specs]),
    )
    write_string_dataset(
        meta_group,
        "mask_composition_list_json",
        json.dumps([float(spec["ratio"]) for spec in mask_specs]),
    )
    write_string_dataset(
        meta_group,
        "mask_slot_name_list_json",
        json.dumps([spec["mask_type_slot"] for spec in mask_specs]),
    )
    write_string_dataset(
        meta_group,
        "mask_slot_ratio_list_json",
        json.dumps([float(spec["ratio"]) for spec in mask_specs]),
    )
    write_string_dataset(
        meta_group,
        "mask_ratio_list_json",
        json.dumps([_mask_param_repr(spec) for spec in mask_specs]),
    )
    write_string_dataset(
        meta_group,
        "mask_slot_param_list_json",
        json.dumps([_mask_param_repr(spec) for spec in mask_specs]),
    )
    write_string_dataset(
        meta_group,
        "mask_slot_specs_json",
        json.dumps(mask_specs, sort_keys=True),
    )
    write_string_dataset(
        meta_group,
        "requested_train_mask_specs_json",
        json.dumps(requested_train_mask_specs, sort_keys=True),
    )
    write_string_dataset(
        meta_group,
        "requested_eval_mask_specs_json",
        json.dumps(requested_eval_mask_specs, sort_keys=True),
    )


def _write_traj_mask_spec(traj_group: h5py.Group, mask_spec: dict[str, Any]) -> None:
    write_string_dataset(traj_group, "mask_type", str(mask_spec["mask_type"]))
    write_string_dataset(traj_group, "mask_type_slot", str(mask_spec["mask_type_slot"]))
    traj_group.create_dataset("mask_slot_index", data=np.int32(mask_spec["mask_slot_index"]))
    if mask_spec["retain_ratio"] is not None:
        traj_group.create_dataset("retain_ratio", data=np.float32(mask_spec["retain_ratio"]))
    if mask_spec["mask_seq_len"] is not None:
        traj_group.create_dataset("mask_seq_len", data=np.int32(mask_spec["mask_seq_len"]))


def write_split_h5(
    input_h5: Path,
    output_h5: Path,
    split_name: str,
    source_episode_ids: list[int],
    mask_jobs: list[dict[str, Any]],
    action_min: np.ndarray,
    action_max: np.ndarray,
    state_min: np.ndarray,
    state_max: np.ndarray,
    env_id: str,
    mask_specs: list[dict[str, Any]],
    requested_train_mask_specs: list[dict[str, Any]],
    requested_eval_mask_specs: list[dict[str, Any]],
    mask_assign_mode: str,
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
        meta.create_dataset("num_episodes", data=np.int32(len(mask_jobs)))
        meta.create_dataset("mask_value", data=np.float32(mask_value))
        meta.create_dataset("split_seed", data=np.int32(split_seed))
        meta.create_dataset("mask_seed", data=np.int32(mask_seed))
        meta.create_dataset("actions_normalized", data=np.bool_(True))
        meta.create_dataset("states_normalized", data=np.bool_(True))
        meta.create_dataset("mas_has_progress", data=np.bool_(True))
        meta.create_dataset(
            "state_path",
            data=np.asarray("obs/state", dtype=h5py.string_dtype("utf-8")),
        )
        meta.create_dataset(
            "normalized_action_dims",
            data=np.asarray([0, 1, 2, 3, 4, 5], dtype=np.int32),
        )
        meta.create_dataset(
            "source_episode_ids",
            data=np.asarray(
                [int(job["source_episode_id"]) for job in mask_jobs],
                dtype=np.int32,
            ),
        )
        meta.create_dataset(
            "unique_source_episode_ids",
            data=np.asarray(source_episode_ids, dtype=np.int32),
        )
        write_string_dataset(meta, "split", split_name)
        write_string_dataset(meta, "env_id", env_id)
        write_string_dataset(meta, "source_h5", str(input_h5))
        write_string_dataset(meta, "source_json", str(input_json))
        _write_mask_spec_meta(
            meta,
            mask_specs=mask_specs,
            requested_train_mask_specs=requested_train_mask_specs,
            requested_eval_mask_specs=requested_eval_mask_specs,
            mask_assign_mode=mask_assign_mode,
            source_num_episodes=len(source_episode_ids),
            expanded_num_episodes=len(mask_jobs),
        )

        for local_episode_id, job in enumerate(mask_jobs):
            source_episode_id = int(job["source_episode_id"])
            src_traj = src_file[f"traj_{source_episode_id}"]
            dst_traj = dst_file.create_group(f"traj_{local_episode_id}")
            mask_spec = job["mask_spec"]

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

            traj_rng = np.random.default_rng(int(job["rng_seed"]))
            masked_actions, keep_mask = apply_mask_to_actions(
                normalized_actions,
                mask_type=mask_spec["mask_type"],
                rng=traj_rng,
                retain_ratio=mask_spec["retain_ratio"],
                mask_seq_len=mask_spec["mask_seq_len"],
                masked_value=mask_value,
            )
            mas = augment_mas_with_progress_np(
                masked_actions,
                traj_len=normalized_actions.shape[0],
            )

            for key in src_traj.keys():
                if key in {
                    "actions",
                    "mas",
                    "mask",
                    "mask_type",
                    "mask_type_slot",
                    "mask_slot_index",
                    "retain_ratio",
                    "mask_seq_len",
                    "source_episode_id",
                    "source_mask_copy_key",
                    "source_mask_copy_index",
                }:
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
            write_string_dataset(
                dst_traj,
                "source_mask_copy_key",
                str(job["source_mask_copy_key"]),
            )
            dst_traj.create_dataset(
                "source_mask_copy_index",
                data=np.int32(mask_spec["mask_spec_index"]),
            )
            _write_traj_mask_spec(dst_traj, mask_spec)


def _summarize_assignments(
    split_name: str,
    mask_jobs: list[dict[str, Any]],
) -> None:
    slot_counts: dict[str, int] = {}
    type_counts: dict[str, int] = {}
    source_episode_ids = []
    for job in mask_jobs:
        source_episode_ids.append(int(job["source_episode_id"]))
        one_spec = job["mask_spec"]
        mask_type = str(one_spec["mask_type"])
        mask_slot = str(one_spec["mask_type_slot"])
        type_counts[mask_type] = type_counts.get(mask_type, 0) + 1
        slot_counts[mask_slot] = slot_counts.get(mask_slot, 0) + 1
    unique_source_count = len(set(source_episode_ids))
    print(
        f"[mixed_preprocess] {split_name} source trajs={unique_source_count}, "
        f"expanded trajs={len(mask_jobs)}"
    )
    print(f"[mixed_preprocess] {split_name} mask slot counts: {dict(sorted(slot_counts.items()))}")
    print(f"[mixed_preprocess] {split_name} mask type counts: {dict(sorted(type_counts.items()))}")


def main() -> None:
    args = parse_args()
    train_mask_specs = normalize_split_mask_config(
        args,
        split="train",
        mask_assign_mode=args.mask_assign_mode,
    )
    eval_mask_specs = normalize_split_mask_config(
        args,
        split="eval",
        mask_assign_mode=args.mask_assign_mode,
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
        train_mask_specs=train_mask_specs,
        eval_mask_specs=eval_mask_specs,
        mask_assign_mode=args.mask_assign_mode,
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
    train_jobs = build_mask_jobs(
        source_episode_ids=train_ids,
        mask_specs=train_mask_specs,
        mask_assign_mode=args.mask_assign_mode,
        seed=args.mask_seed,
    )
    eval_jobs = build_mask_jobs(
        source_episode_ids=eval_ids,
        mask_specs=eval_mask_specs,
        mask_assign_mode=args.mask_assign_mode,
        seed=args.mask_seed + 1_000_003,
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
    )
    write_json(
        train_json,
        build_split_metadata_json(
            source_metadata=metadata,
            split_name="train",
            source_episode_ids=train_ids,
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
        ),
    )

    if len(eval_ids) > 0:
        write_split_h5(
            input_h5=input_h5,
            output_h5=eval_h5,
            split_name="eval",
            source_episode_ids=eval_ids,
            mask_jobs=eval_jobs,
            action_min=action_min,
            action_max=action_max,
            state_min=state_min,
            state_max=state_max,
            env_id=args.env_id,
            mask_specs=eval_mask_specs,
            requested_train_mask_specs=train_mask_specs,
            requested_eval_mask_specs=eval_mask_specs,
            mask_assign_mode=args.mask_assign_mode,
            mask_value=args.mask_value,
            split_seed=args.split_seed,
            mask_seed=args.mask_seed + 1_000_003,
            input_json=input_json,
        )
        write_json(
            eval_json,
            build_split_metadata_json(
                source_metadata=metadata,
                split_name="eval",
                source_episode_ids=eval_ids,
                mask_jobs=eval_jobs,
                input_h5=input_h5,
                input_json=input_json,
                env_id=args.env_id,
                mask_specs=eval_mask_specs,
                requested_train_mask_specs=train_mask_specs,
                requested_eval_mask_specs=eval_mask_specs,
                mask_assign_mode=args.mask_assign_mode,
                split_seed=args.split_seed,
                mask_seed=args.mask_seed + 1_000_003,
            ),
        )

    print(
        "[data_preprocess_mixed] done: "
        f"mode={args.mask_assign_mode}, "
        f"train={len(train_ids)} source/{len(train_jobs)} expanded trajs -> {train_h5}, "
        f"eval={len(eval_ids)} source/{len(eval_jobs)} expanded trajs -> "
        f"{eval_h5 if len(eval_ids) > 0 else 'N/A'}"
    )
    _summarize_assignments("train", train_jobs)
    if len(eval_ids) > 0:
        _summarize_assignments("eval", eval_jobs)
    print(
        "[data_preprocess_mixed] stats: "
        f"action_dim={MAS_ACTION_DIM}, state_dim={state_min.shape[0]}, mas_dim={MAS_STEP_DIM}"
    )


if __name__ == "__main__":
    main()
