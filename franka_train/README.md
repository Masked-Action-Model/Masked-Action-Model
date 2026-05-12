# Franka Real Training

This folder contains real-data training entry points that avoid ManiSkill online eval.

## Data

Raw baseline h5:

```text
traj_N/actions
traj_N/success
traj_N/obs/agent/qpos
traj_N/obs/agent/qvel
traj_N/obs/extra/tcp_pose
traj_N/obs/sensor_data/base_camera/rgb
```

Subgoal and MAM need a preprocessed h5 with normalized `actions`, `obs/state`, `mas`, and `mask`:

```bash
INPUT_H5=franka_train/data/franka_real.h5 \
OVERWRITE=true \
bash franka_train/run_preprocess_franka.sh
```

## Train

Baseline uses the raw h5:

```bash
DEMO_PATH=franka_train/data/franka_real.h5 \
bash franka_train/run_train_baseline_franka.sh
```

Subgoal and MAM use the preprocessed h5:

```bash
DEMO_PATH=franka_train/data/franka_real_random_mask_0.2_train.h5 \
bash franka_train/run_train_subgoal_franka.sh

DEMO_PATH=franka_train/data/franka_real_random_mask_0.2_train.h5 \
bash franka_train/run_train_mam_franka.sh
```

Checkpoints are saved when `step > SAVE_START_ITER && step % SAVE_FREQ == 0`; `latest.pt` is also refreshed.
