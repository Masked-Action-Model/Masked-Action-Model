from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Optional

import numpy as np
import torch
import tyro

from common import (
    build_policy_env_stub,
    build_open_loop_validator,
    infer_action_dim,
    install_maniskill_stubs,
    load_demo_dataset_with_optional_done,
    load_meta,
    make_obs_process_fn,
    make_run_name,
    seed_everything,
    split_train_validation_traj_indices,
    train_no_eval,
)

install_maniskill_stubs()
import train_baseline as baseline_mod


def _meta_bool(meta: dict, key: str, default: bool = False) -> bool:
    if key not in meta:
        return bool(default)
    value = np.asarray(meta[key])
    if value.shape == ():
        return bool(value.item())
    return bool(value.reshape(-1)[0])


def _decode_meta_value(value):
    value = np.asarray(value)
    if value.shape == ():
        value = value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        decoded = []
        for item in value.tolist():
            if isinstance(item, bytes):
                decoded.append(item.decode("utf-8"))
            else:
                decoded.append(str(item))
        return decoded
    return value


def _load_state_schema(meta: dict) -> list[dict]:
    if "state_schema_json" in meta:
        raw_schema = _decode_meta_value(meta["state_schema_json"])
        schema = json.loads(raw_schema) if isinstance(raw_schema, str) else raw_schema
        if isinstance(schema, list):
            return schema
        raise ValueError("meta/state_schema_json must decode to a list")

    state_paths = _decode_meta_value(meta.get("state_paths", []))
    if isinstance(state_paths, str):
        try:
            parsed = json.loads(state_paths)
            state_paths = parsed if isinstance(parsed, list) else [state_paths]
        except json.JSONDecodeError:
            state_paths = [state_paths]
    if (
        isinstance(state_paths, list)
        and len(state_paths) > 0
        and state_paths[-1] == "obs/extra/tcp_pose"
        and "state_dim" in meta
    ):
        state_dim = int(np.asarray(meta["state_dim"]).item())
        return [
            {
                "path": "obs/extra/tcp_pose",
                "dim": 7,
                "start": state_dim - 7,
                "end": state_dim,
            }
        ]
    raise ValueError(
        "relative action_space requires meta/state_schema_json, or state_paths with "
        "obs/extra/tcp_pose as the final state component"
    )


def _find_tcp_pose_slice(state_schema: list[dict]) -> slice:
    for entry in state_schema:
        if str(entry.get("path", "")) == "obs/extra/tcp_pose":
            start = int(entry["start"])
            end = int(entry["end"])
            if end <= start:
                raise ValueError(f"invalid tcp_pose state schema entry: {entry}")
            return slice(start, end)
    raise ValueError("state schema does not contain obs/extra/tcp_pose")


def _denormalize_selected_dims(
    data: np.ndarray,
    mins: np.ndarray,
    maxs: np.ndarray,
) -> np.ndarray:
    data = np.asarray(data, dtype=np.float32)
    mins = np.asarray(mins, dtype=np.float32)
    maxs = np.asarray(maxs, dtype=np.float32)
    denormalized = data.copy()
    dim = min(data.shape[-1], mins.shape[0], maxs.shape[0])
    if dim <= 0:
        return denormalized
    denormalized[..., :dim] = mins[:dim] + 0.5 * (
        data[..., :dim] + 1.0
    ) * (maxs[:dim] - mins[:dim])
    return denormalized.astype(np.float32)


def _normalize_selected_dims_tensor(
    data: torch.Tensor,
    mins: torch.Tensor,
    maxs: torch.Tensor,
) -> torch.Tensor:
    normalized = data.clone()
    dim = min(data.shape[-1], mins.shape[0], maxs.shape[0])
    if dim <= 0:
        return normalized
    mins = mins[:dim].to(device=data.device, dtype=data.dtype)
    maxs = maxs[:dim].to(device=data.device, dtype=data.dtype)
    denom = maxs - mins
    values = torch.clamp(data[..., :dim], min=mins, max=maxs)
    scaled = 2.0 * (values - mins) / torch.clamp(denom, min=1e-8) - 1.0
    zero_dims = torch.abs(denom) < 1e-8
    if torch.any(zero_dims):
        scaled[..., zero_dims] = 0.0
    normalized[..., :dim] = scaled
    return normalized


class FrankaBaselineDataset(baseline_mod.SmallDemoDataset_DiffusionPolicy):
    def __init__(
        self,
        data_path,
        obs_process_fn,
        obs_space,
        include_rgb,
        include_depth,
        obs_horizon,
        pred_horizon,
        device,
        num_traj,
        action_dim,
        action_norm_path=None,
        traj_indices=None,
        action_space: Literal["absolute", "relative"] = "absolute",
        relative_action_stats: tuple[np.ndarray, np.ndarray] | None = None,
    ):
        self.action_space = str(action_space)
        if self.action_space not in {"absolute", "relative"}:
            raise ValueError(
                f"action_space must be 'absolute' or 'relative', got {self.action_space!r}"
            )
        self.relative_action_stats = None
        meta = load_meta(data_path)
        actions_normalized = _meta_bool(meta, "actions_normalized", False)
        states_normalized = _meta_bool(meta, "states_normalized", False)
        if not actions_normalized:
            if self.action_space != "absolute":
                raise ValueError(
                    "relative action_space requires preprocessed normalized Franka data"
                )
            if traj_indices is not None:
                raise ValueError(
                    "validation split for raw baseline data is not supported here; "
                    "use preprocessed normalized Franka data."
                )
            super().__init__(
                data_path=data_path,
                obs_process_fn=obs_process_fn,
                obs_space=obs_space,
                include_rgb=include_rgb,
                include_depth=include_depth,
                obs_horizon=obs_horizon,
                pred_horizon=pred_horizon,
                device=device,
                num_traj=num_traj,
                action_dim=action_dim,
                action_norm_path=action_norm_path,
            )
            return

        self.action_dim = int(action_dim)
        if self.action_dim not in (6, 7):
            raise ValueError(f"action_dim must be 6 or 7, got {self.action_dim}")
        self.include_rgb = include_rgb
        self.include_depth = include_depth
        self.action_norm_path = action_norm_path or data_path
        self.action_min = np.asarray(meta["action_min"], dtype=np.float32)
        self.action_max = np.asarray(meta["action_max"], dtype=np.float32)
        if self.action_min.shape[0] > self.action_dim:
            raise ValueError(
                f"action norm dim ({self.action_min.shape[0]}) exceeds action_dim={self.action_dim}"
            )
        self.absolute_action_min = self.action_min.copy()
        self.absolute_action_max = self.action_max.copy()
        self.relative_norm_dim = min(
            self.action_dim,
            self.absolute_action_min.shape[0],
            6,
        )
        if self.action_space == "relative" and self.relative_norm_dim <= 0:
            raise ValueError("relative action_space requires at least one normalized pose dim")
        state_min = np.asarray(meta.get("state_min", []), dtype=np.float32)
        state_max = np.asarray(meta.get("state_max", []), dtype=np.float32)
        if self.action_space == "relative":
            self.tcp_pose_slice = _find_tcp_pose_slice(_load_state_schema(meta))
            if states_normalized and (state_min.size == 0 or state_max.size == 0):
                raise ValueError(
                    "relative action_space with normalized state requires meta/state_min "
                    "and meta/state_max"
                )

        trajectories = load_demo_dataset_with_optional_done(
            data_path,
            num_traj=num_traj if traj_indices is None else None,
            concat=False,
            traj_indices=traj_indices,
        )
        print("Preprocessed Franka trajectory loaded, beginning observation pre-processing...")

        obs_traj_dict_list = []
        relative_state_trajs = []
        for obs_traj_dict in trajectories["observations"]:
            precomputed_state = obs_traj_dict.get("state", None)
            obs_traj_dict = baseline_mod.reorder_keys(obs_traj_dict, obs_space)
            obs_traj_dict = obs_process_fn(obs_traj_dict)
            processed_state = np.asarray(obs_traj_dict["state"], dtype=np.float32)
            if self.action_space == "relative":
                if states_normalized and precomputed_state is not None:
                    relative_state = _denormalize_selected_dims(
                        np.asarray(precomputed_state, dtype=np.float32),
                        mins=state_min,
                        maxs=state_max,
                    )
                else:
                    relative_state = processed_state
                relative_state_trajs.append(
                    torch.from_numpy(relative_state).to(dtype=torch.float32)
                )
            if include_depth:
                obs_traj_dict["depth"] = torch.Tensor(
                    obs_traj_dict["depth"].astype(np.float32)
                ).to(dtype=torch.float16)
            if include_rgb:
                obs_traj_dict["rgb"] = torch.from_numpy(obs_traj_dict["rgb"])
            if states_normalized and precomputed_state is not None:
                obs_traj_dict["state"] = torch.from_numpy(
                    np.asarray(precomputed_state, dtype=np.float32)
                ).to(dtype=torch.float32)
            else:
                obs_traj_dict["state"] = torch.from_numpy(obs_traj_dict["state"]).to(
                    dtype=torch.float32
                )
            obs_traj_dict_list.append(obs_traj_dict)
        trajectories["observations"] = obs_traj_dict_list
        self.obs_keys = list(obs_traj_dict_list[0].keys())

        for traj_idx, action in enumerate(trajectories["actions"]):
            action_np = np.asarray(action, dtype=np.float32)
            if action_np.ndim != 2 or action_np.shape[1] != self.action_dim:
                raise ValueError(
                    f"traj_{traj_idx} actions must have shape (T, {self.action_dim}), got {action_np.shape}"
                )
            if self.action_space == "relative":
                action_np = _denormalize_selected_dims(
                    action_np,
                    mins=self.absolute_action_min,
                    maxs=self.absolute_action_max,
                )
            trajectories["actions"][traj_idx] = torch.from_numpy(action_np).to(
                dtype=torch.float32
            )
        if self.action_space == "relative":
            print(
                f"[action_norm] loaded absolute normalized actions from: {self.action_norm_path}"
            )
        else:
            print(
                f"[action_norm] using preprocessed normalized actions: {self.action_norm_path}"
            )
            print(f"[action_norm] min={self.action_min}")
            print(f"[action_norm] max={self.action_max}")

        print("Obs/action pre-processing is done, start to pre-compute the slice indices...")
        print("Using fixed control mode pd_ee_pose, padding with final action.")
        self.obs_horizon, self.pred_horizon = obs_horizon, pred_horizon
        self.slices = []
        num_traj = len(trajectories["actions"])
        total_transitions = 0
        for traj_idx in range(num_traj):
            length = trajectories["actions"][traj_idx].shape[0]
            assert trajectories["observations"][traj_idx]["state"].shape[0] == length + 1
            total_transitions += length
            pad_before = obs_horizon - 1
            pad_after = pred_horizon - obs_horizon
            self.slices += [
                (traj_idx, start, start + pred_horizon)
                for start in range(-pad_before, length - pred_horizon + pad_after)
            ]
        print(
            f"Total transitions: {total_transitions}, Total obs sequences: {len(self.slices)}"
        )
        self.trajectories = trajectories
        if self.action_space == "relative":
            self.relative_state_trajs = relative_state_trajs
            if relative_action_stats is None:
                self.action_min, self.action_max = self._compute_relative_action_stats()
            else:
                self.action_min = np.asarray(relative_action_stats[0], dtype=np.float32)
                self.action_max = np.asarray(relative_action_stats[1], dtype=np.float32)
            if self.action_min.shape[0] != self.relative_norm_dim:
                raise ValueError(
                    f"relative action_min dim ({self.action_min.shape[0]}) must match "
                    f"relative_norm_dim={self.relative_norm_dim}"
                )
            self.relative_action_stats = (self.action_min.copy(), self.action_max.copy())
            self.relative_action_min_tensor = torch.from_numpy(self.action_min).to(
                dtype=torch.float32
            )
            self.relative_action_max_tensor = torch.from_numpy(self.action_max).to(
                dtype=torch.float32
            )
            print(
                "[action_space] relative: action chunk first "
                f"{self.relative_norm_dim} dims subtract state tcp_pose dims "
                f"{self.tcp_pose_slice.start}:{self.tcp_pose_slice.start + self.relative_norm_dim}"
            )
            print(f"[relative_action_norm] min={self.action_min}")
            print(f"[relative_action_norm] max={self.action_max}")

    def _slice_action_chunk(self, traj_idx: int, start: int, end: int) -> torch.Tensor:
        actions = self.trajectories["actions"][traj_idx]
        length = actions.shape[0]
        act_seq = actions[max(0, start) : end]
        if start < 0:
            if act_seq.shape[0] >= 2:
                delta = act_seq[1] - act_seq[0]
            else:
                delta = torch.zeros_like(act_seq[0])
            pad_len = -start
            pad_actions = [act_seq[0] - delta * k for k in range(pad_len, 0, -1)]
            act_seq = torch.cat([torch.stack(pad_actions, dim=0), act_seq], dim=0)
        if end > length:
            pad_action = act_seq[-1]
            act_seq = torch.cat([act_seq, pad_action.repeat(end - length, 1)], dim=0)
        return act_seq

    def _slice_state_sequence(self, traj_idx: int, start: int) -> torch.Tensor:
        state = self.relative_state_trajs[traj_idx]
        state_seq = state[max(0, start) : start + self.obs_horizon]
        if start < 0:
            if state_seq.shape[0] >= 2:
                delta = state_seq[1] - state_seq[0]
            else:
                delta = torch.zeros_like(state_seq[0])
            pad_len = -start
            pad_states = [state_seq[0] - delta * n for n in range(pad_len, 0, -1)]
            state_seq = torch.cat([torch.stack(pad_states, dim=0), state_seq], dim=0)
        if state_seq.shape[0] != self.obs_horizon:
            raise ValueError(
                f"state sequence length mismatch: got {state_seq.shape[0]}, "
                f"expected {self.obs_horizon}"
            )
        return state_seq

    def _to_relative_action_chunk(
        self,
        traj_idx: int,
        start: int,
        absolute_action_chunk: torch.Tensor,
        normalize: bool,
    ) -> torch.Tensor:
        tcp_pose = self._slice_state_sequence(traj_idx, start)[0, self.tcp_pose_slice]
        if tcp_pose.shape[0] < self.relative_norm_dim:
            raise ValueError(
                f"tcp_pose dim ({tcp_pose.shape[0]}) is smaller than "
                f"relative_norm_dim={self.relative_norm_dim}"
            )
        relative = absolute_action_chunk.clone()
        relative[..., : self.relative_norm_dim] = (
            relative[..., : self.relative_norm_dim]
            - tcp_pose[: self.relative_norm_dim].to(dtype=relative.dtype)
        )
        if not normalize:
            return relative
        return _normalize_selected_dims_tensor(
            relative,
            mins=self.relative_action_min_tensor,
            maxs=self.relative_action_max_tensor,
        )

    def _compute_relative_action_stats(self) -> tuple[np.ndarray, np.ndarray]:
        mins = None
        maxs = None
        for traj_idx, start, end in self.slices:
            absolute_chunk = self._slice_action_chunk(traj_idx, start, end)
            relative_chunk = self._to_relative_action_chunk(
                traj_idx,
                start,
                absolute_chunk,
                normalize=False,
            )
            values = relative_chunk[:, : self.relative_norm_dim].numpy()
            chunk_min = values.min(axis=0)
            chunk_max = values.max(axis=0)
            mins = chunk_min if mins is None else np.minimum(mins, chunk_min)
            maxs = chunk_max if maxs is None else np.maximum(maxs, chunk_max)
        if mins is None or maxs is None:
            raise ValueError("cannot compute relative action stats from empty dataset")
        return mins.astype(np.float32), maxs.astype(np.float32)

    def __getitem__(self, index):
        traj_idx, start, end = self.slices[index]

        obs_traj = self.trajectories["observations"][traj_idx]
        obs_seq = {}
        for key, value in obs_traj.items():
            obs_seq[key] = value[max(0, start) : start + self.obs_horizon]
            if start < 0:
                pad_len = -start
                if key == "state":
                    if obs_seq[key].shape[0] >= 2:
                        delta = obs_seq[key][1] - obs_seq[key][0]
                    else:
                        delta = torch.zeros_like(obs_seq[key][0])
                    pad_obs = [
                        obs_seq[key][0] - delta * n for n in range(pad_len, 0, -1)
                    ]
                    obs_seq[key] = torch.cat(
                        (torch.stack(pad_obs, dim=0), obs_seq[key]),
                        dim=0,
                    )
                else:
                    pad_obs_seq = torch.stack([obs_seq[key][0]] * pad_len, dim=0)
                    obs_seq[key] = torch.cat((pad_obs_seq, obs_seq[key]), dim=0)

        act_seq = self._slice_action_chunk(traj_idx, start, end)
        if self.action_space == "relative":
            act_seq = self._to_relative_action_chunk(
                traj_idx,
                start,
                act_seq,
                normalize=True,
            )

        assert (
            obs_seq["state"].shape[0] == self.obs_horizon
            and act_seq.shape[0] == self.pred_horizon
        )
        return {
            "observations": obs_seq,
            "actions": act_seq,
        }


@dataclass
class Args:
    exp_name: Optional[str] = None
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "ManiSkill"
    wandb_entity: Optional[str] = None
    capture_video: bool = False

    env_id: str = "FrankaReal-v1"
    demo_path: str = "franka_train/data/franka_real.h5"
    num_demos: Optional[int] = None
    action_dim: Optional[int] = 7
    action_norm_path: Optional[str] = None
    action_space: Literal["absolute", "relative"] = "absolute"

    total_iters: int = 100_000
    batch_size: int = 256
    lr: float = 1e-4
    obs_horizon: int = 2
    act_horizon: int = 8
    pred_horizon: int = 16
    diffusion_step_embed_dim: int = 64
    noise_model: Literal["Transformer", "Unet"] = "Transformer"
    vision_encoder: Literal["resnet", "dino2", "dino3"] = "resnet"
    dino_model_path: str = ""
    dino_data_aug: bool = False
    dit_hidden_dim: int = 512
    dit_num_blocks: int = 6
    dit_dim_feedforward: int = 2048
    unet_dims: List[int] = field(default_factory=lambda: [64, 128, 256])
    n_groups: int = 8

    obs_mode: Literal["rgb", "rgb+depth"] = "rgb"
    max_episode_steps: Optional[int] = None
    sim_backend: str = "physx_cpu"
    log_freq: int = 1000
    eval_freq: int = 0
    valid_freq: int = 0
    num_validation_set: int = 0
    num_eval_episodes: int = 0
    num_eval_envs: int = 0
    num_eval_demos: Optional[int] = None
    save_start_iter: int = 0
    save_freq: Optional[int] = 5000
    num_dataload_workers: int = 0
    control_mode: str = "pd_ee_pose"
    demo_type: Optional[str] = "franka_real_baseline"


def main() -> None:
    args = tyro.cli(Args)
    args.action_dim = infer_action_dim(args.demo_path, args.action_dim)
    run_name = make_run_name(
        Path(__file__).stem,
        args.env_id,
        args.exp_name,
        args.seed,
    )
    if args.obs_horizon + args.act_horizon - 1 > args.pred_horizon:
        raise ValueError("obs_horizon + act_horizon - 1 must be <= pred_horizon")

    seed_everything(args.seed, args.torch_deterministic)
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    baseline_mod.device = device

    env, raw_obs_space, include_rgb, include_depth = build_policy_env_stub(
        args.demo_path,
        action_dim=args.action_dim,
        obs_horizon=args.obs_horizon,
        obs_mode=args.obs_mode,
    )
    obs_process_fn = make_obs_process_fn(include_depth)
    train_traj_indices, val_traj_indices = split_train_validation_traj_indices(
        args.demo_path,
        args.num_demos,
        args.num_validation_set,
        args.seed,
    )

    dataset = FrankaBaselineDataset(
        data_path=args.demo_path,
        obs_process_fn=obs_process_fn,
        obs_space=raw_obs_space,
        include_rgb=include_rgb,
        include_depth=include_depth,
        obs_horizon=args.obs_horizon,
        pred_horizon=args.pred_horizon,
        device=device,
        num_traj=args.num_demos if train_traj_indices is None else None,
        action_dim=args.action_dim,
        action_norm_path=args.action_norm_path,
        traj_indices=train_traj_indices,
        action_space=args.action_space,
    )
    dataset.debug_print_sample(0)
    val_dataset = None
    validate_fn = None
    if val_traj_indices:
        val_dataset = FrankaBaselineDataset(
            data_path=args.demo_path,
            obs_process_fn=obs_process_fn,
            obs_space=raw_obs_space,
            include_rgb=include_rgb,
            include_depth=include_depth,
            obs_horizon=args.obs_horizon,
            pred_horizon=args.pred_horizon,
            device=device,
            num_traj=None,
            action_dim=args.action_dim,
            action_norm_path=args.action_norm_path,
            traj_indices=val_traj_indices,
            action_space=args.action_space,
            relative_action_stats=dataset.relative_action_stats,
        )

    agent = baseline_mod.Agent(env, args).to(device)
    ema_agent = baseline_mod.Agent(env, args).to(device)
    agent.set_action_denormalizer(dataset.action_min, dataset.action_max, device)
    ema_agent.set_action_denormalizer(dataset.action_min, dataset.action_max, device)
    if val_dataset is not None:
        validate_fn = build_open_loop_validator(
            dataset=val_dataset,
            device=device,
        )

    def compute_loss(model, batch):
        return model.compute_loss(
            obs_seq=batch["observations"],
            action_seq=batch["actions"],
        )

    train_no_eval(
        run_name=run_name,
        args=args,
        dataset=dataset,
        agent=agent,
        ema_agent=ema_agent,
        device=device,
        compute_loss=compute_loss,
        validate_fn=validate_fn,
    )
    env.close()


if __name__ == "__main__":
    main()
