from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-maniskill")

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from tqdm import tqdm

from maniskill_dataset import (
    FrameManiskillDataset,
    infer_camera_names_from_h5,
    infer_state_dim_from_h5,
    infer_state_paths_from_h5,
    resolve_camera_names,
    resolve_state_paths,
)
from models.clip_encoder import FrozenCLIPEncoder
from models.rewind_reward_model import RewardTransformer
from utils.data_utils import adapt_maniskill_batch_rewind, get_valid_episodes
from utils.train_utils import get_normalizer_from_calculated


def _select_clip_rgb(images: torch.Tensor) -> torch.Tensor:
    if images.shape[2] < 3:
        raise ValueError(
            f"Expected at least 3 image channels for CLIP input, got {tuple(images.shape)}"
        )
    return images[:, :, :3, :, :]


def _resolve_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    if path.exists() or path.is_absolute():
        return path
    repo_root = Path(__file__).resolve().parents[1]
    candidate = repo_root / path
    return candidate if candidate.exists() else path


def _resolve_config_resource(path: str | Path, config_path: Path) -> str:
    path = Path(path).expanduser()
    if path.exists() or path.is_absolute():
        return str(path)

    repo_root = Path(__file__).resolve().parents[1]
    for base in (config_path.parent, repo_root):
        candidate = base / path
        if candidate.exists():
            return str(candidate)
    return str(path)


def _to_int_list(value: Any) -> list[int]:
    if value is None:
        return []
    if torch.is_tensor(value):
        return [int(v) for v in value.detach().cpu().reshape(-1).tolist()]
    if isinstance(value, np.ndarray):
        return [int(v) for v in value.reshape(-1).tolist()]
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    return [int(value)]


def _load_episode_obs_lengths(dataset_path: Path, episode_ids: list[int]) -> dict[int, int]:
    lengths = {}
    with h5py.File(dataset_path, "r") as f:
        for episode_id in episode_ids:
            traj_key = f"traj_{episode_id}"
            if traj_key not in f:
                raise KeyError(f"Missing {traj_key} in {dataset_path}")
            traj = f[traj_key]
            if "obs" not in traj or "agent" not in traj["obs"] or "qpos" not in traj["obs"]["agent"]:
                raise KeyError(
                    f"{traj_key} has no obs/agent/qpos, cannot infer observation length"
                )
            lengths[episode_id] = int(traj["obs"]["agent"]["qpos"].shape[0])
    return lengths


def _build_dataloader(dataset, cfg, batch_size: int | None, num_workers: int | None):
    val_cfg = getattr(cfg, "val_dataloader", {})
    resolved_batch_size = int(batch_size or getattr(val_cfg, "batch_size", 32))
    resolved_num_workers = int(
        getattr(val_cfg, "num_workers", 0) if num_workers is None else num_workers
    )
    kwargs = {
        "batch_size": resolved_batch_size,
        "num_workers": resolved_num_workers,
        "shuffle": False,
        "pin_memory": torch.cuda.is_available()
        and bool(getattr(val_cfg, "pin_memory", False)),
    }
    if resolved_num_workers > 0:
        kwargs["persistent_workers"] = bool(getattr(val_cfg, "persistent_workers", False))
    return torch.utils.data.DataLoader(dataset, **kwargs)


def _build_model(cfg, checkpoint_path: Path, device: torch.device):
    clip_encoder = FrozenCLIPEncoder(cfg.encoders.vision_ckpt, device)
    reward_model = RewardTransformer(
        d_model=cfg.model.d_model,
        vis_emb_dim=512,
        text_emb_dim=512,
        state_dim=int(cfg.model.state_dim),
        n_layers=cfg.model.n_layers,
        n_heads=cfg.model.n_heads,
        dropout=cfg.model.dropout,
        num_cameras=len(cfg.general.camera_names),
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    reward_model.load_state_dict(state_dict)
    reward_model.eval()

    state_normalizer = get_normalizer_from_calculated(
        cfg.general.state_norm_path,
        device,
        state_dim=int(cfg.model.state_dim),
    )
    return clip_encoder, reward_model, state_normalizer


@torch.no_grad()
def _forward_stpm_batch(
    batch: dict,
    cfg,
    camera_names: list[str],
    clip_encoder,
    reward_model,
    state_normalizer,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch = adapt_maniskill_batch_rewind(batch, camera_names=camera_names)
    batch_size, horizon = batch["image_frames"][camera_names[0]].shape[:2]

    img_list = []
    for camera_name in camera_names:
        imgs = _select_clip_rgb(batch["image_frames"][camera_name]).flatten(0, 1).to(device)
        img_list.append(imgs)

    targets = batch["targets"].to(device)
    lengths = batch["lengths"].to(device)
    state = batch["state"].to(device)
    state = state_normalizer.normalize(state)
    if bool(getattr(cfg.model, "no_state", False)):
        state = torch.zeros_like(state)

    imgs_all = torch.cat(img_list, dim=0)
    img_emb = clip_encoder.encode_image(imgs_all)
    img_emb = img_emb.view(len(img_list), batch_size, horizon, -1).permute(1, 0, 2, 3)
    lang_emb = clip_encoder.encode_text(batch["tasks"])
    pred = reward_model(img_emb, lang_emb, state, lengths)
    return pred, targets


def _save_episode_curve(output_dir: Path, episode_id: int, rows: list[dict]) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = sorted(rows, key=lambda x: int(x["anchor_index"]))
    steps = np.asarray([r["anchor_index"] for r in rows], dtype=np.float32)
    pred = np.asarray([r["pred_progress"] for r in rows], dtype=np.float32)
    target = np.asarray([r["target_progress"] for r in rows], dtype=np.float32)
    mse = float(np.mean((pred - target) ** 2))
    mae = float(np.mean(np.abs(pred - target)))

    out_path = output_dir / f"traj_{episode_id:04d}_progress.png"
    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    ax.plot(steps, pred, label="pred", linewidth=1.8)
    ax.plot(steps, target, label="target", linewidth=1.4, linestyle="--")
    ax.set_xlabel("timestep")
    ax.set_ylabel("progress")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlim(left=0.0)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
    ax.set_title(f"traj_{episode_id}  MSE={mse:.6f}  MAE={mae:.6f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def _save_predictions_csv(output_dir: Path, predictions_by_episode: dict[int, list[dict]]) -> Path:
    csv_path = output_dir / "predictions.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "episode_id",
                "anchor_index",
                "pred_progress",
                "target_progress",
                "last_step_mse",
            ],
        )
        writer.writeheader()
        for episode_id in sorted(predictions_by_episode):
            for row in sorted(
                predictions_by_episode[episode_id], key=lambda x: int(x["anchor_index"])
            ):
                writer.writerow({"episode_id": episode_id, **row})
    return csv_path


def _prepare_config(args) -> tuple[Any, Path, list[int], list[str]]:
    config_path = _resolve_path(args.config).resolve()
    dataset_path = _resolve_path(args.dataset).resolve()
    cfg = OmegaConf.load(config_path)
    cfg.general.repo_id = str(dataset_path)
    cfg.general.state_norm_path = _resolve_config_resource(
        cfg.general.state_norm_path,
        config_path,
    )
    cfg.encoders.vision_ckpt = _resolve_config_resource(
        cfg.encoders.vision_ckpt,
        config_path,
    )

    available_episodes = get_valid_episodes(str(dataset_path))
    selected_episodes = available_episodes[: int(args.num_eval_demo)]
    if len(selected_episodes) == 0:
        raise ValueError(f"No eval episodes selected from {dataset_path}")

    configured_cameras = getattr(cfg.general, "camera_names", "auto")
    camera_names = resolve_camera_names(dataset_path, configured_cameras)
    if len(camera_names) == 0:
        camera_names = infer_camera_names_from_h5(dataset_path)
    cfg.general.camera_names = camera_names

    configured_state_paths = list(getattr(cfg.general, "state_paths", []) or [])
    state_paths = (
        resolve_state_paths(configured_state_paths)
        if configured_state_paths
        else infer_state_paths_from_h5(dataset_path)
    )
    cfg.general.state_paths = state_paths

    inferred_state_dim = infer_state_dim_from_h5(dataset_path, state_paths=state_paths)
    if int(cfg.model.state_dim) != int(inferred_state_dim):
        raise ValueError(
            "Config/checkpoint state_dim does not match eval dataset state paths: "
            f"config={int(cfg.model.state_dim)}, dataset={inferred_state_dim}, "
            f"state_paths={state_paths}"
        )

    return cfg, dataset_path, selected_episodes, camera_names


def evaluate(args) -> dict:
    checkpoint_path = _resolve_path(args.checkpoint).resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    cfg, dataset_path, selected_episodes, camera_names = _prepare_config(args)
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else checkpoint_path.parent / f"{checkpoint_path.stem}_offline_eval_curves"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    device_name = args.device or getattr(cfg.general, "device", "cuda")
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")

    dataset = FrameManiskillDataset(
        repo_id=str(dataset_path),
        episodes=selected_episodes,
        n_obs_steps=int(cfg.model.n_obs_steps),
        frame_gap=int(cfg.model.frame_gap),
        image_names=camera_names,
        task_description=str(
            getattr(cfg.general, "task_description", getattr(cfg.general, "task_name", ""))
        ),
        state_paths=list(cfg.general.state_paths),
    )
    dataloader = _build_dataloader(dataset, cfg, args.batch_size, args.num_workers)
    episode_lengths = _load_episode_obs_lengths(dataset_path, selected_episodes)

    clip_encoder, reward_model, state_normalizer = _build_model(cfg, checkpoint_path, device)

    batch_loss_sum = 0.0
    batch_count = 0
    squared_error_sum = 0.0
    squared_error_count = 0
    predictions_by_episode: dict[int, list[dict]] = defaultdict(list)

    pbar = tqdm(dataloader, desc="offline STPM eval", disable=not args.progress_bar)
    for batch in pbar:
        pred, targets = _forward_stpm_batch(
            batch=batch,
            cfg=cfg,
            camera_names=camera_names,
            clip_encoder=clip_encoder,
            reward_model=reward_model,
            state_normalizer=state_normalizer,
            device=device,
        )
        loss = F.mse_loss(pred, targets, reduction="mean")
        batch_loss_sum += float(loss.item())
        batch_count += 1
        squared_error_sum += float(F.mse_loss(pred, targets, reduction="sum").item())
        squared_error_count += int(targets.numel())

        episode_ids = _to_int_list(batch["episode_index"])
        anchor_indices = _to_int_list(batch["anchor_index"])
        pred_last = pred[:, -1].detach().cpu().numpy()
        target_last = targets[:, -1].detach().cpu().numpy()

        for episode_id, anchor_index, pred_progress, target_progress in zip(
            episode_ids,
            anchor_indices,
            pred_last,
            target_last,
        ):
            predictions_by_episode[int(episode_id)].append(
                {
                    "anchor_index": int(anchor_index),
                    "pred_progress": float(pred_progress),
                    "target_progress": float(target_progress),
                    "last_step_mse": float((pred_progress - target_progress) ** 2),
                }
            )

    if batch_count == 0:
        raise ValueError("Evaluation dataloader is empty")

    curve_paths = []
    for episode_id in selected_episodes:
        rows = predictions_by_episode.get(int(episode_id), [])
        expected_len = episode_lengths[int(episode_id)]
        if len(rows) != expected_len:
            raise ValueError(
                f"traj_{episode_id} prediction count mismatch: "
                f"{len(rows)} vs obs_len={expected_len}"
            )
        curve_paths.append(str(_save_episode_curve(output_dir, int(episode_id), rows)))

    predictions_csv = _save_predictions_csv(output_dir, predictions_by_episode)
    avg_loss = batch_loss_sum / float(batch_count)
    sample_weighted_loss = squared_error_sum / float(squared_error_count)
    last_step_mse = float(
        np.mean(
            [
                row["last_step_mse"]
                for rows in predictions_by_episode.values()
                for row in rows
            ]
        )
    )

    summary = {
        "checkpoint": str(checkpoint_path),
        "config": str(_resolve_path(args.config).resolve()),
        "dataset": str(dataset_path),
        "output_dir": str(output_dir),
        "selected_episodes": [int(v) for v in selected_episodes],
        "num_eval_demo": len(selected_episodes),
        "num_eval_points": int(sum(len(v) for v in predictions_by_episode.values())),
        "camera_names": list(camera_names),
        "state_paths": list(cfg.general.state_paths),
        "loss": {
            "stpm_eval_mean_mse": float(avg_loss),
            "sample_weighted_mse": float(sample_weighted_loss),
            "last_step_mse": float(last_step_mse),
        },
        "curve_paths": curve_paths,
        "predictions_csv": str(predictions_csv),
    }
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    summary["summary_path"] = str(summary_path)
    return summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Offline STPM eval on replayed ManiSkill H5 datasets with observations."
    )
    parser.add_argument("--checkpoint", "--ckpt-path", required=True, help="STPM checkpoint .pt path")
    parser.add_argument("--config", required=True, help="STPM config yaml path")
    parser.add_argument("--dataset", "--dataset-path", required=True, help="Replay H5 dataset with obs")
    parser.add_argument("--num-eval-demo", type=int, required=True, help="Number of demos to evaluate")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", default=None, help="cuda/cpu; defaults to config.general.device")
    parser.add_argument("--output-dir", default=None, help="Default: next to checkpoint")
    parser.add_argument("--progress-bar", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


if __name__ == "__main__":
    result = evaluate(parse_args())
    print(f"[offline-stpm] mean_loss={result['loss']['stpm_eval_mean_mse']:.8f}")
    print(f"[offline-stpm] sample_weighted_mse={result['loss']['sample_weighted_mse']:.8f}")
    print(f"[offline-stpm] last_step_mse={result['loss']['last_step_mse']:.8f}")
    print(f"[offline-stpm] output_dir={result['output_dir']}")
    print(f"[offline-stpm] summary={result['summary_path']}")
