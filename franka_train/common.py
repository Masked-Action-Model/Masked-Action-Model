from __future__ import annotations

import json
import random
import sys
import types
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from gymnasium import spaces
from torch.utils.data.dataloader import DataLoader, default_collate
from torch.utils.data.sampler import BatchSampler, RandomSampler
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
DIFFUSION_ROOT = REPO_ROOT / "examples" / "baselines" / "diffusion_policy"
for path in (REPO_ROOT, DIFFUSION_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from data_preprocess.utils.obs_utils import (  # noqa: E402
    build_default_state_obs_extractor,
    build_state_schema_from_obs,
    flatten_state_from_obs,
)
from utils.load_train_data_utils import (  # noqa: E402
    TARGET_KEY_TO_SOURCE_KEY,
    load_content_from_h5_file,
)
from utils.utils import (  # noqa: E402
    IterationBasedBatchSampler,
    convert_obs,
    worker_init_fn,
)


def install_maniskill_stubs() -> None:
    if "mani_skill" in sys.modules:
        return

    mani_skill = types.ModuleType("mani_skill")
    envs = types.ModuleType("mani_skill.envs")
    utils = types.ModuleType("mani_skill.utils")
    common = types.ModuleType("mani_skill.utils.common")
    sapien_utils = types.ModuleType("mani_skill.utils.sapien_utils")
    gym_utils = types.ModuleType("mani_skill.utils.gym_utils")
    wrappers = types.ModuleType("mani_skill.utils.wrappers")
    flatten = types.ModuleType("mani_skill.utils.wrappers.flatten")
    vector = types.ModuleType("mani_skill.vector")
    vector_wrappers = types.ModuleType("mani_skill.vector.wrappers")
    vector_gymnasium = types.ModuleType("mani_skill.vector.wrappers.gymnasium")

    def to_tensor(value, device=None):
        if torch.is_tensor(value):
            return value.to(device=device) if device is not None else value
        if isinstance(value, dict):
            return {key: to_tensor(val, device=device) for key, val in value.items()}
        return torch.as_tensor(value, device=device)

    def find_max_episode_steps_value(_env):
        return 0

    class _UnusedWrapper:
        def __init__(self, env=None, *args, **kwargs):
            self.env = env

        def __getattr__(self, name):
            return getattr(self.env, name)

    class _UnusedVectorEnv(_UnusedWrapper):
        pass

    common.to_tensor = to_tensor
    gym_utils.find_max_episode_steps_value = find_max_episode_steps_value
    wrappers.CPUGymWrapper = _UnusedWrapper
    wrappers.FrameStack = _UnusedWrapper
    wrappers.RecordEpisode = _UnusedWrapper
    flatten.FlattenRGBDObservationWrapper = _UnusedWrapper
    vector_gymnasium.ManiSkillVectorEnv = _UnusedVectorEnv
    utils.common = common
    utils.sapien_utils = sapien_utils
    utils.gym_utils = gym_utils
    utils.wrappers = wrappers
    mani_skill.envs = envs
    mani_skill.utils = utils
    mani_skill.vector = vector
    vector.wrappers = vector_wrappers
    vector_wrappers.gymnasium = vector_gymnasium

    sys.modules["mani_skill"] = mani_skill
    sys.modules["mani_skill.envs"] = envs
    sys.modules["mani_skill.utils"] = utils
    sys.modules["mani_skill.utils.common"] = common
    sys.modules["mani_skill.utils.sapien_utils"] = sapien_utils
    sys.modules["mani_skill.utils.gym_utils"] = gym_utils
    sys.modules["mani_skill.utils.wrappers"] = wrappers
    sys.modules["mani_skill.utils.wrappers.flatten"] = flatten
    sys.modules["mani_skill.vector"] = vector
    sys.modules["mani_skill.vector.wrappers"] = vector_wrappers
    sys.modules["mani_skill.vector.wrappers.gymnasium"] = vector_gymnasium


def seed_everything(seed: int, torch_deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = torch_deterministic


def args_to_dict(args: Any) -> dict[str, Any]:
    if is_dataclass(args):
        return asdict(args)
    return dict(vars(args))


def list_traj_keys(h5_file: h5py.File, num_traj: int | None = None) -> list[str]:
    keys = sorted(
        [key for key in h5_file.keys() if key.startswith("traj_")],
        key=lambda key: int(key.split("_")[-1]),
    )
    if num_traj is not None:
        keys = keys[: int(num_traj)]
    if len(keys) == 0:
        raise ValueError("no traj_* groups found")
    return keys


def split_train_validation_traj_indices(
    data_path: str,
    num_demos: int | None,
    num_validation_set: int,
    seed: int,
) -> tuple[list[int] | None, list[int]]:
    num_validation_set = int(num_validation_set)
    if num_validation_set <= 0:
        return None, []
    with h5py.File(data_path, "r") as f:
        total = len(list_traj_keys(f))
    if num_demos is not None:
        total = min(total, int(num_demos))
    if num_validation_set >= total:
        raise ValueError(
            f"num_validation_set ({num_validation_set}) must be smaller than selected demos ({total})"
        )
    indices = list(range(total))
    rng = np.random.default_rng(int(seed))
    rng.shuffle(indices)
    val_indices = sorted(indices[:num_validation_set])
    train_indices = sorted(indices[num_validation_set:])
    print(
        f"[validation] split from {total} demos: train={train_indices}, val={val_indices}"
    )
    return train_indices, val_indices


def infer_action_dim(data_path: str, action_dim: int | None = None) -> int:
    if action_dim is not None:
        return int(action_dim)
    with h5py.File(data_path, "r") as f:
        if "meta" in f and "action_dim" in f["meta"]:
            return int(np.asarray(f["meta"]["action_dim"][()]).item())
        first_key = list_traj_keys(f)[0]
        return int(f[first_key]["actions"].shape[1])


def _decode_scalar(value: Any) -> Any:
    value = np.asarray(value)
    if value.shape == ():
        value = value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def load_meta(data_path: str) -> dict[str, Any]:
    with h5py.File(data_path, "r") as f:
        if "meta" not in f:
            return {}
        return {key: f["meta"][key][()] for key in f["meta"].keys()}


def read_camera_names(data_path: str) -> list[str]:
    meta = load_meta(data_path)
    if "camera_names" in meta:
        raw = _decode_scalar(meta["camera_names"])
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    return [str(x) for x in parsed]
            except json.JSONDecodeError:
                return [raw]
        if isinstance(raw, np.ndarray):
            return [str(_decode_scalar(x)) for x in raw.tolist()]
    with h5py.File(data_path, "r") as f:
        first_key = list_traj_keys(f)[0]
        return list(f[first_key]["obs"]["sensor_data"].keys())


def _box_for_dataset(dataset: h5py.Dataset, drop_time_axis: bool = True) -> spaces.Box:
    shape = tuple(int(x) for x in dataset.shape[1:] if drop_time_axis)
    if not drop_time_axis:
        shape = tuple(int(x) for x in dataset.shape)
    dtype = dataset.dtype
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        low = info.min
        high = info.max
    else:
        low = -np.inf
        high = np.inf
    return spaces.Box(low=low, high=high, shape=shape, dtype=dtype)


def _space_from_group(group: h5py.Group) -> spaces.Dict:
    items = {}
    for key in group.keys():
        value = group[key]
        if isinstance(value, h5py.Group):
            items[key] = _space_from_group(value)
        elif isinstance(value, h5py.Dataset):
            items[key] = _box_for_dataset(value, drop_time_axis=True)
    return spaces.Dict(items)


def build_raw_obs_space_from_h5(data_path: str) -> spaces.Dict:
    with h5py.File(data_path, "r") as f:
        first_key = list_traj_keys(f)[0]
        return _space_from_group(f[first_key]["obs"])


def _load_obs_arrays_from_group(obs_group: h5py.Group) -> dict[str, Any]:
    out = {}
    for key in obs_group.keys():
        value = obs_group[key]
        if isinstance(value, h5py.Group):
            out[key] = _load_obs_arrays_from_group(value)
        elif isinstance(value, h5py.Dataset):
            out[key] = np.asarray(value[()])
    return out


def infer_state_dim_from_h5(data_path: str) -> int:
    meta = load_meta(data_path)
    if "state_dim" in meta:
        return int(np.asarray(meta["state_dim"]).item())
    with h5py.File(data_path, "r") as f:
        first_key = list_traj_keys(f)[0]
        obs = _load_obs_arrays_from_group(f[first_key]["obs"])
    state = flatten_state_from_obs(
        obs,
        state_obs_extractor=build_default_state_obs_extractor(),
    )
    return int(state.shape[1])


def infer_image_shapes_from_h5(data_path: str) -> tuple[int, int, int, int]:
    with h5py.File(data_path, "r") as f:
        first_key = list_traj_keys(f)[0]
        sensor_data = f[first_key]["obs"]["sensor_data"]
        rgb_channels = 0
        depth_channels = 0
        height = None
        width = None
        for camera_name in sensor_data.keys():
            camera = sensor_data[camera_name]
            if "rgb" in camera:
                rgb_shape = camera["rgb"].shape
                height = int(rgb_shape[1])
                width = int(rgb_shape[2])
                rgb_channels += int(rgb_shape[3])
            if "depth" in camera:
                depth_shape = camera["depth"].shape
                height = int(depth_shape[1]) if height is None else height
                width = int(depth_shape[2]) if width is None else width
                depth_channels += int(depth_shape[3])
    if height is None or width is None or rgb_channels <= 0:
        raise ValueError(f"{data_path} must contain obs/sensor_data/*/rgb")
    return height, width, rgb_channels, depth_channels


class PolicyEnvStub:
    def __init__(self, single_observation_space: spaces.Dict, action_dim: int):
        self.single_observation_space = single_observation_space
        self.single_action_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(int(action_dim),),
            dtype=np.float32,
        )

    def close(self) -> None:
        return None


def build_policy_env_stub(
    data_path: str,
    action_dim: int,
    obs_horizon: int,
    obs_mode: str = "rgb",
) -> tuple[PolicyEnvStub, spaces.Dict, bool, bool]:
    if obs_mode not in {"rgb", "rgb+depth"}:
        raise ValueError(f"unsupported obs_mode={obs_mode!r}; expected 'rgb' or 'rgb+depth'")
    raw_obs_space = build_raw_obs_space_from_h5(data_path)
    state_dim = infer_state_dim_from_h5(data_path)
    height, width, rgb_channels, depth_channels = infer_image_shapes_from_h5(data_path)
    include_rgb = True
    include_depth = obs_mode == "rgb+depth"
    if include_depth and depth_channels <= 0:
        raise ValueError("obs_mode='rgb+depth' was requested, but no depth data exists")

    policy_spaces: dict[str, spaces.Space] = {
        "state": spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(int(obs_horizon), state_dim),
            dtype=np.float32,
        ),
        "rgb": spaces.Box(
            low=0,
            high=255,
            shape=(int(obs_horizon), height, width, rgb_channels),
            dtype=np.uint8,
        ),
    }
    if include_depth:
        policy_spaces["depth"] = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(int(obs_horizon), height, width, depth_channels),
            dtype=np.float32,
        )
    return PolicyEnvStub(spaces.Dict(policy_spaces), action_dim), raw_obs_space, include_rgb, include_depth


def make_obs_process_fn(include_depth: bool):
    return lambda obs: convert_obs(
        obs,
        concat_fn=lambda arrays: np.concatenate(arrays, axis=-1),
        transpose_fn=lambda array: np.transpose(array, axes=(0, 3, 1, 2)),
        state_obs_extractor=build_default_state_obs_extractor(),
        depth=include_depth,
    )


def save_checkpoint(
    run_name: str,
    tag: str,
    agent: torch.nn.Module,
    ema: Any,
    ema_agent: torch.nn.Module,
) -> Path:
    ckpt_dir = Path("runs") / run_name / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ema.copy_to(ema_agent.parameters())
    ckpt_path = ckpt_dir / f"{tag}.pt"
    torch.save(
        {
            "agent": agent.state_dict(),
            "ema_agent": ema_agent.state_dict(),
        },
        ckpt_path,
    )
    return ckpt_path


def save_latest_checkpoint(
    run_name: str,
    agent: torch.nn.Module,
    ema: Any,
    ema_agent: torch.nn.Module,
) -> Path:
    return save_checkpoint(run_name, "latest", agent, ema, ema_agent)


def should_save_periodic(step: int, save_start_iter: int, save_freq: int | None) -> bool:
    return (
        save_freq is not None
        and int(save_freq) > 0
        and int(step) > int(save_start_iter)
        and int(step) % int(save_freq) == 0
    )


def build_open_loop_validator(
    *,
    dataset,
    device: torch.device,
    points_per_demo: int = 10,
):
    if dataset is None:
        return None
    points_per_demo = max(1, int(points_per_demo))
    sample_indices = _select_validation_sample_indices(dataset, points_per_demo)
    if len(sample_indices) == 0:
        raise ValueError("validation dataset produced no samples")
    print(
        f"[validation] open-loop samples={len(sample_indices)}, "
        f"points_per_demo<={points_per_demo}"
    )

    def validator(model: torch.nn.Module) -> dict[str, float]:
        was_training = model.training
        model.eval()
        mse_sum = 0.0
        mae_sum = 0.0
        count = 0
        with torch.no_grad():
            for sample_idx in sample_indices:
                sample = default_collate([dataset[int(sample_idx)]])
                sample = _move_batch_to_device(sample, device)
                pred = _predict_open_loop_action_from_dataset_obs(
                    model,
                    sample["observations"],
                )
                target = sample["actions"][
                    :,
                    dataset.obs_horizon - 1 : dataset.obs_horizon - 1 + pred.shape[1],
                ]
                diff = pred - target
                mse_sum += float(torch.sum(diff * diff).item())
                mae_sum += float(torch.sum(torch.abs(diff)).item())
                count += int(diff.numel())
        if was_training:
            model.train()
        open_loop_loss = mse_sum / max(1, count)
        return {
            "validation/open_loop_loss": open_loop_loss,
            "validation/open_loop_mse": open_loop_loss,
            "validation/open_loop_mae": mae_sum / max(1, count),
            "validation/open_loop_samples": float(len(sample_indices)),
        }

    return validator


def _predict_open_loop_action_from_dataset_obs(
    model: torch.nn.Module,
    obs_seq: dict[str, Any],
) -> torch.Tensor:
    if hasattr(model, "encode_obs"):
        obs_cond = model.encode_obs(obs_seq, eval_mode=True)
    elif hasattr(model, "obs_conditioning"):
        obs_cond = model.obs_conditioning(obs_seq, eval_mode=True)
    else:
        raise AttributeError("model must provide encode_obs() or obs_conditioning()")
    obs_cond = model.prepare_noise_condition(obs_cond)
    batch_size = obs_seq["state"].shape[0]
    device = obs_seq["state"].device
    noisy_action_seq = torch.randn(
        (batch_size, model.pred_horizon, model.act_dim),
        device=device,
    )
    for k in model.noise_scheduler.timesteps:
        timesteps = torch.full(
            (batch_size,),
            k,
            dtype=torch.long,
            device=device,
        )
        noise_pred = model.noise_pred_net(noisy_action_seq, timesteps, obs_cond)
        noisy_action_seq = model.noise_scheduler.step(
            model_output=noise_pred,
            timestep=k,
            sample=noisy_action_seq,
        ).prev_sample
    start = model.obs_horizon - 1
    end = start + model.act_horizon
    return noisy_action_seq[:, start:end]


def _select_validation_sample_indices(dataset, points_per_demo: int) -> list[int]:
    by_traj: dict[int, list[int]] = {}
    for sample_idx, (traj_idx, start, _end) in enumerate(dataset.slices):
        if start < 0:
            continue
        by_traj.setdefault(int(traj_idx), []).append(int(sample_idx))
    selected = []
    for traj_idx in sorted(by_traj):
        candidates = by_traj[traj_idx]
        if len(candidates) <= points_per_demo:
            selected.extend(candidates)
            continue
        positions = np.linspace(0, len(candidates) - 1, num=points_per_demo)
        selected.extend(candidates[int(round(pos))] for pos in positions)
    return selected


def load_demo_dataset_with_optional_done(
    path,
    keys=("observations", "actions"),
    num_traj=None,
    concat=True,
    traj_indices=None,
):
    print("Loading HDF5 file", path)
    file = h5py.File(path, "r")
    ordered_keys = list_traj_keys(file)
    if traj_indices is not None:
        selected_keys = []
        for idx in traj_indices:
            idx = int(idx)
            if idx < 0 or idx >= len(ordered_keys):
                raise IndexError(f"traj_idx out of range: {idx}")
            selected_keys.append(ordered_keys[idx])
        ordered_keys = selected_keys
    elif num_traj is not None:
        ordered_keys = ordered_keys[: int(num_traj)]
    raw_data = {key: load_content_from_h5_file(file[key]) for key in ordered_keys}
    file.close()
    print("Loaded")

    dataset = {}
    for target_key in keys:
        source_key = TARGET_KEY_TO_SOURCE_KEY[target_key]
        values = []
        for traj_key in ordered_keys:
            traj = raw_data[traj_key]
            if source_key in traj:
                values.append(traj[source_key])
                continue
            if source_key in {"terminated", "truncated", "success"}:
                length = int(np.asarray(traj["actions"]).shape[0])
                values.append(np.zeros(length, dtype=np.bool_))
                continue
            if source_key == "env_states":
                values.append({})
                continue
            raise KeyError(f"key: {source_key} not in {traj_key}: {traj.keys()}")
        dataset[target_key] = values
        if isinstance(values[0], np.ndarray) and concat:
            if target_key in ["observations", "states"] and len(values[0]) > len(raw_data[ordered_keys[0]]["actions"]):
                dataset[target_key] = np.concatenate([v[:-1] for v in values], axis=0)
            elif target_key in ["next_observations", "next_states"] and len(values[0]) > len(raw_data[ordered_keys[0]]["actions"]):
                dataset[target_key] = np.concatenate([v[1:] for v in values], axis=0)
            else:
                dataset[target_key] = np.concatenate(values, axis=0)
            print("Load", target_key, dataset[target_key].shape)
        else:
            print("Load", target_key, len(values), type(values[0]))
    return dataset


def build_writer(run_name: str, args: Any):
    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter(str(Path("runs") / run_name))
    args_dict = args_to_dict(args)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in args_dict.items()])),
    )
    return writer


def train_no_eval(
    *,
    run_name: str,
    args: Any,
    dataset,
    agent: torch.nn.Module,
    ema_agent: torch.nn.Module,
    device: torch.device,
    compute_loss,
    after_agent_built=None,
    validate_fn=None,
) -> None:
    from diffusers.optimization import get_scheduler
    from diffusers.training_utils import EMAModel

    writer = build_writer(run_name, args)
    sampler = RandomSampler(dataset, replacement=False)
    batch_sampler = BatchSampler(sampler, batch_size=args.batch_size, drop_last=True)
    batch_sampler = IterationBasedBatchSampler(batch_sampler, args.total_iters)
    train_dataloader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=args.num_dataload_workers,
        worker_init_fn=lambda worker_id: worker_init_fn(worker_id, base_seed=args.seed),
        persistent_workers=(args.num_dataload_workers > 0),
    )

    optimizer = torch.optim.AdamW(
        params=agent.parameters(),
        lr=args.lr,
        betas=(0.95, 0.999),
        weight_decay=1e-6,
    )
    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=args.total_iters,
    )
    ema = EMAModel(parameters=agent.parameters(), power=0.75)
    if after_agent_built is not None:
        after_agent_built(agent, ema_agent)

    timings: dict[str, float] = {}
    agent.train()
    pbar = tqdm(total=args.total_iters)
    last_tick = None
    last_loss = None
    last_saved_step = None

    for zero_based_iter, data_batch in enumerate(train_dataloader):
        step = zero_based_iter + 1
        if last_tick is None:
            last_tick = 0.0
        data_batch = _move_batch_to_device(data_batch, device)

        optimizer.zero_grad()
        total_loss = compute_loss(agent, data_batch)
        total_loss.backward()
        optimizer.step()
        lr_scheduler.step()
        ema.step(agent.parameters())
        last_loss = float(total_loss.item())

        if step % int(args.log_freq) == 0:
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], step)
            writer.add_scalar("losses/total_loss", last_loss, step)
            for key, value in timings.items():
                writer.add_scalar(f"time/{key}", value, step)

        valid_freq = int(getattr(args, "valid_freq", 0) or 0)
        if validate_fn is not None and valid_freq > 0 and step % valid_freq == 0:
            ema.copy_to(ema_agent.parameters())
            metrics = validate_fn(ema_agent)
            for key, value in metrics.items():
                writer.add_scalar(key, value, step)
            print(
                "[validation] "
                + ", ".join(f"{key}={value:.6g}" for key, value in metrics.items())
                + f", step={step}"
            )
            agent.train()

        if should_save_periodic(step, args.save_start_iter, args.save_freq):
            save_checkpoint(run_name, str(step), agent, ema, ema_agent)
            save_latest_checkpoint(run_name, agent, ema, ema_agent)
            last_saved_step = step
            print(f"[checkpoint] saved step={step}")

        pbar.update(1)
        pbar.set_postfix({"loss": last_loss})

    if last_loss is not None:
        writer.add_scalar("losses/total_loss", last_loss, int(args.total_iters))
    if last_saved_step != int(args.total_iters):
        save_latest_checkpoint(run_name, agent, ema, ema_agent)
        print(f"[checkpoint] saved latest at final step={args.total_iters}")
    writer.close()
    pbar.close()


def _move_batch_to_device(batch, device):
    out = {}
    for key, value in batch.items():
        if isinstance(value, dict):
            out[key] = _move_batch_to_device(value, device)
        elif torch.is_tensor(value):
            out[key] = value.to(device, non_blocking=True)
        else:
            out[key] = value
    return out


def make_run_name(script_stem: str, env_id: str, exp_name: str | None, seed: int) -> str:
    if exp_name:
        return exp_name
    import time

    return f"{env_id}__{script_stem}__{seed}__{int(time.time())}"


def load_action_stats_from_h5(data_path: str) -> tuple[np.ndarray, np.ndarray]:
    meta = load_meta(data_path)
    if "action_min" not in meta or "action_max" not in meta:
        raise ValueError(f"{data_path} has no meta/action_min and meta/action_max")
    return (
        np.asarray(meta["action_min"], dtype=np.float32),
        np.asarray(meta["action_max"], dtype=np.float32),
    )


def state_schema_from_raw_obs_space(raw_obs_space: spaces.Dict) -> list[dict[str, Any]]:
    return build_state_schema_from_obs(raw_obs_space, has_leading_axis=False)
