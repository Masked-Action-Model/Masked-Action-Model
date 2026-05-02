#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/STPM:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-maniskill}"
export WANDB_MODE="${WANDB_MODE:-disabled}"

DATASET_PATH="${DATASET_PATH:-demos/exp_5/StackPyramid-v1/motionplanning/StackPyramin-v1.rgbd.pd_joint_pos.physx_cpu.h5}"
if [[ -z "$DATASET_PATH" ]]; then
  echo "ERROR: DATASET_PATH is required, e.g. DATASET_PATH=demos/.../trajectory.rgbd....h5" >&2
  exit 1
fi
if [[ ! -f "$DATASET_PATH" ]]; then
  echo "ERROR: dataset h5 not found: $DATASET_PATH" >&2
  exit 1
fi

TASK_NAME="${TASK_NAME:-stackpyramid}"
TASK_DESCRIPTION="${TASK_DESCRIPTION:-pick up the red cube, place it next to the green cube, and stack the blue cube on top of the red and green cube}"
OUTPUT_DIR="${OUTPUT_DIR:-STPM_stackpyramid}"
STATE_PATHS="${STATE_PATHS:-auto}"

SEED="${SEED:-42}"
DEVICE="${DEVICE:-cuda}"
CAMERA_NAMES="${CAMERA_NAMES:-auto}"
VISION_CKPT="${VISION_CKPT:-pretrained/clip-vit-base-patch32}"

BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-0}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-${BATCH_SIZE}}"
VAL_NUM_WORKERS="${VAL_NUM_WORKERS:-${NUM_WORKERS}}"

D_MODEL="${D_MODEL:-768}"
N_LAYERS="${N_LAYERS:-8}"
N_HEADS="${N_HEADS:-12}"
DROPOUT="${DROPOUT:-0.1}"
N_OBS_STEPS="${N_OBS_STEPS:-6}"
FRAME_GAP="${FRAME_GAP:-2}"
NO_STATE="${NO_STATE:-false}"
RESUME_TRAINING="${RESUME_TRAINING:-false}"
MODEL_PATH="${MODEL_PATH:-}"

LR="${LR:-5e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-5e-3}"
BETA1="${BETA1:-0.9}"
BETA2="${BETA2:-0.95}"
EPS="${EPS:-1e-8}"
WARMUP_STEPS="${WARMUP_STEPS:-1000}"
TOTAL_STEPS="${TOTAL_STEPS:-100000}"

NUM_EPOCHS="${NUM_EPOCHS:-2}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
LOG_EVERY="${LOG_EVERY:-50}"
EVAL_EVERY="${EVAL_EVERY:-1}"
SAVE_EVERY="${SAVE_EVERY:-5000}"
VAL_PORTION="${VAL_PORTION:-0.1}"

export \
  DATASET_PATH TASK_NAME TASK_DESCRIPTION OUTPUT_DIR STATE_PATHS SEED DEVICE CAMERA_NAMES VISION_CKPT \
  BATCH_SIZE NUM_WORKERS VAL_BATCH_SIZE VAL_NUM_WORKERS D_MODEL N_LAYERS N_HEADS \
  DROPOUT N_OBS_STEPS FRAME_GAP NO_STATE RESUME_TRAINING MODEL_PATH LR WEIGHT_DECAY \
  BETA1 BETA2 EPS WARMUP_STEPS TOTAL_STEPS NUM_EPOCHS GRAD_CLIP LOG_EVERY EVAL_EVERY \
  SAVE_EVERY VAL_PORTION

CONFIG_PATH="$(
python - <<'PY'
import ast
import json
import os
import re
from pathlib import Path

import h5py
import numpy as np
from omegaconf import OmegaConf


def env_bool(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def slugify(value):
    value = value.strip() or "stpm"
    value = re.sub(r"[^A-Za-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_").lower()
    return value or "stpm"


def h5_get(group, path):
    cur = group
    for part in path.split("/"):
        if not part:
            continue
        cur = cur[part]
    return cur


def is_numeric_leaf(obj):
    return isinstance(obj, h5py.Dataset) and np.issubdtype(obj.dtype, np.number)


def parse_state_paths(value):
    value = value.strip()
    if not value or value.lower() == "auto":
        return None
    try:
        parsed = ast.literal_eval(value)
    except Exception:
        parsed = [x.strip() for x in value.split(",") if x.strip()]
    if isinstance(parsed, (str, bytes)):
        parsed = [parsed]
    return [str(x) for x in parsed]


def parse_camera_names(value):
    value = value.strip()
    if not value or value.lower() == "auto":
        return None
    try:
        parsed = ast.literal_eval(value)
    except Exception:
        parsed = [x.strip() for x in value.split(",") if x.strip()]
    if isinstance(parsed, (str, bytes)):
        parsed = [parsed]
    return [str(x).strip() for x in parsed if str(x).strip()]


def infer_camera_names(traj_group):
    try:
        sensor_data = h5_get(traj_group, "obs/sensor_data")
    except KeyError:
        raise ValueError(
            "Failed to infer STPM cameras: no obs/sensor_data group found in dataset."
        ) from None
    camera_names = [
        name
        for name, value in sensor_data.items()
        if isinstance(value, h5py.Group) and "rgb" in value
    ]
    if not camera_names:
        raise ValueError("Failed to infer STPM cameras: no RGB camera datasets found.")
    return camera_names


def infer_state_paths(traj_group):
    paths = []
    for path in ("obs/agent/qpos", "obs/agent/qvel"):
        try:
            if is_numeric_leaf(h5_get(traj_group, path)):
                paths.append(path)
        except KeyError:
            pass

    extra_paths = []
    try:
        extra = h5_get(traj_group, "obs/extra")
    except KeyError:
        extra = None
    if extra is not None:
        def visit(group, prefix):
            for key, value in group.items():
                path = f"{prefix}/{key}" if prefix else key
                if isinstance(value, h5py.Group):
                    visit(value, path)
                elif is_numeric_leaf(value):
                    extra_paths.append(f"obs/extra/{path}")
        visit(extra, "")

    priority = {"goal_pos": 0, "tcp_pose": 1, "is_grasped": 2}
    extra_paths.sort(key=lambda p: (priority.get(p.split("/")[-1], 99), p))
    paths.extend(extra_paths)
    if not paths:
        raise ValueError("Failed to infer numeric STPM state paths from dataset.")
    return paths


def read_state(traj_group, state_paths):
    pieces = []
    expected_len = None
    for path in state_paths:
        data = h5_get(traj_group, path)[:].astype(np.float32)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        elif data.ndim > 2:
            data = data.reshape(data.shape[0], -1)
        if expected_len is None:
            expected_len = data.shape[0]
        elif data.shape[0] != expected_len:
            raise ValueError(f"State path {path} has length {data.shape[0]}, expected {expected_len}")
        pieces.append(data)
    return np.concatenate(pieces, axis=-1).astype(np.float32)


dataset_path = Path(os.environ["DATASET_PATH"]).expanduser()
task_name = os.environ["TASK_NAME"].strip() or dataset_path.stem
task_description = os.environ["TASK_DESCRIPTION"].strip() or task_name
output_dir = Path(os.environ["OUTPUT_DIR"].strip() or f"STPM_task_{slugify(task_name)}")
output_dir.mkdir(parents=True, exist_ok=True)

with h5py.File(dataset_path, "r") as f:
    traj_keys = sorted(
        [k for k in f.keys() if k.startswith("traj_")],
        key=lambda k: int(k.split("_")[1]),
    )
    if not traj_keys:
        raise ValueError(f"No traj_* groups found in {dataset_path}")
    first_traj = f[traj_keys[0]]
    state_paths = parse_state_paths(os.environ["STATE_PATHS"]) or infer_state_paths(first_traj)
    camera_names = parse_camera_names(os.environ["CAMERA_NAMES"]) or infer_camera_names(first_traj)
    all_states = [read_state(f[k], state_paths) for k in traj_keys]

states = np.concatenate(all_states, axis=0).astype(np.float32)
state_dim = int(states.shape[1])
state_norm_path = output_dir / "state_norm.json"
state_norm_path.write_text(json.dumps({
    "norm_stats": {
        "state": {
            "mean": states.mean(axis=0).tolist(),
            "std": states.std(axis=0).clip(min=1e-2).tolist(),
            "q01": np.quantile(states, 0.01, axis=0).tolist(),
            "q99": np.quantile(states, 0.99, axis=0).tolist(),
        }
    }
}, indent=2))

base_cfg_path = Path("STPM/config/rewind_maniskill.yaml")
cfg = OmegaConf.load(base_cfg_path)
cfg.general.project_name = output_dir.name
cfg.general.output_dir = str(output_dir)
cfg.general.task_name = task_name
cfg.general.task_description = task_description
cfg.general.repo_id = str(dataset_path)
cfg.general.state_norm_path = str(state_norm_path)
cfg.general.state_paths = state_paths
cfg.general.camera_names = camera_names
cfg.general.seed = int(os.environ["SEED"])
cfg.general.device = os.environ["DEVICE"]

cfg.dataloader.batch_size = int(os.environ["BATCH_SIZE"])
cfg.dataloader.num_workers = int(os.environ["NUM_WORKERS"])
cfg.dataloader.persistent_workers = cfg.dataloader.num_workers > 0
cfg.val_dataloader.batch_size = int(os.environ["VAL_BATCH_SIZE"])
cfg.val_dataloader.num_workers = int(os.environ["VAL_NUM_WORKERS"])
cfg.val_dataloader.persistent_workers = cfg.val_dataloader.num_workers > 0

cfg.encoders.vision_ckpt = os.environ["VISION_CKPT"]
cfg.model.d_model = int(os.environ["D_MODEL"])
cfg.model.n_layers = int(os.environ["N_LAYERS"])
cfg.model.n_heads = int(os.environ["N_HEADS"])
cfg.model.dropout = float(os.environ["DROPOUT"])
cfg.model.n_obs_steps = int(os.environ["N_OBS_STEPS"])
cfg.model.frame_gap = int(os.environ["FRAME_GAP"])
cfg.model.state_dim = state_dim
cfg.model.no_state = env_bool("NO_STATE")
cfg.model.resume_training = env_bool("RESUME_TRAINING")
cfg.model.model_path = os.environ["MODEL_PATH"]

cfg.optim.lr = float(os.environ["LR"])
cfg.optim.weight_decay = float(os.environ["WEIGHT_DECAY"])
cfg.optim.betas = [float(os.environ["BETA1"]), float(os.environ["BETA2"])]
cfg.optim.eps = float(os.environ["EPS"])
cfg.optim.warmup_steps = int(os.environ["WARMUP_STEPS"])
cfg.optim.total_steps = int(os.environ["TOTAL_STEPS"])

cfg.train.num_epochs = int(os.environ["NUM_EPOCHS"])
cfg.train.grad_clip = float(os.environ["GRAD_CLIP"])
cfg.train.log_every = int(os.environ["LOG_EVERY"])
cfg.train.eval_every = int(os.environ["EVAL_EVERY"])
cfg.train.save_every = int(os.environ["SAVE_EVERY"])
cfg.train.val_portion = float(os.environ["VAL_PORTION"])

config_path = output_dir / "config.yaml"
OmegaConf.save(cfg, config_path)
print(config_path)
PY
)"

echo "[stpm] config: ${CONFIG_PATH}"
echo "[stpm] output: $(dirname "$CONFIG_PATH")"

if [[ "${PREPARE_ONLY:-false}" == "true" ]]; then
  echo "[stpm] PREPARE_ONLY=true, skip training."
  exit 0
fi

python STPM/train_STPM.py --config "$CONFIG_PATH"
