#!/usr/bin/env python3
"""Trim replay camera videos to the true replay time window.

For old recordings, replay cameras were started when the user pressed "r",
before the controller completed its move-to-start pre-roll.  Joint and gripper
timestamps, however, are relative to the controller's replay_started signal.

This script keeps only the camera frames that correspond to the true replay
window and overwrites the AVI plus its *_frame_ts.npz file when --execute is set.
It does not modify joint or gripper files.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import numpy as np


CAMERAS = ("third_rs", "wrist_zed")


def natural_traj_key(path: Path) -> int:
    try:
        return int(path.name.split("_", 1)[1])
    except (IndexError, ValueError):
        return -1


def replay_duration(replay_dir: Path) -> float:
    joint_path = replay_dir / "joint_trajectory.npz"
    gripper_path = replay_dir / "gripper_events.npz"

    joint = np.load(joint_path)
    joint_ts = joint["timestamps"].astype(np.float64)
    if joint_ts.size == 0:
        raise ValueError(f"{joint_path} has no timestamps")

    duration = float(np.nanmax(joint_ts))

    if gripper_path.exists():
        gripper = np.load(gripper_path, allow_pickle=True)
        if "relative_times" in gripper.files and gripper["relative_times"].size:
            duration = max(duration, float(np.nanmax(gripper["relative_times"])))

    if not np.isfinite(duration) or duration <= 0:
        raise ValueError(f"invalid replay duration for {replay_dir}: {duration}")
    return duration


def video_info(path: Path) -> tuple[int, float, int, int]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        return 0, 0.0, 0, 0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return frame_count, fps, width, height


def write_trimmed_video(src: Path, dst: Path, keep_mask: np.ndarray, fps: float) -> int:
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"failed to open video: {src}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(str(dst), fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        writer.release()
        raise RuntimeError(f"failed to open video writer: {dst}")

    written = 0
    index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if index < keep_mask.size and keep_mask[index]:
                writer.write(frame)
                written += 1
            index += 1
    finally:
        cap.release()
        writer.release()

    return written


def trim_camera(replay_dir: Path, camera: str, duration: float, execute: bool) -> dict:
    avi_path = replay_dir / f"{camera}.avi"
    ts_path = replay_dir / f"{camera}_frame_ts.npz"
    result = {
        "camera": camera,
        "status": "skip",
        "old_frames": 0,
        "new_frames": 0,
        "trim_seconds": 0.0,
    }

    if not avi_path.exists():
        result["reason"] = "missing avi"
        return result
    if not ts_path.exists():
        frame_count, _, _, _ = video_info(avi_path)
        result["old_frames"] = frame_count
        result["reason"] = "missing timestamps"
        return result

    data = np.load(ts_path, allow_pickle=True)
    if "timestamps" not in data.files:
        result["reason"] = "timestamp key missing"
        return result

    timestamps = data["timestamps"].astype(np.float64)
    frame_count, fps, width, height = video_info(avi_path)
    result["old_frames"] = int(frame_count)

    if frame_count <= 0 or width <= 0 or height <= 0:
        result["reason"] = "empty video"
        return result
    if timestamps.size == 0:
        result["reason"] = "empty timestamps"
        return result

    usable = min(frame_count, timestamps.size)
    timestamps = timestamps[:usable]

    replay_t0_path = replay_dir / "replay_t0.npy"
    if replay_t0_path.exists():
        replay_t0 = float(np.load(replay_t0_path))
        window_start = replay_t0
        window_end = replay_t0 + duration
    else:
        window_end = float(timestamps[-1])
        window_start = window_end - duration

    keep_mask = (timestamps >= window_start) & (timestamps <= window_end)
    if keep_mask.size < frame_count:
        padded = np.zeros(frame_count, dtype=bool)
        padded[: keep_mask.size] = keep_mask
        keep_mask = padded

    new_frames = int(keep_mask.sum())
    result["new_frames"] = new_frames
    result["trim_seconds"] = max(0.0, float(timestamps[0] - window_start) * -1.0)

    if new_frames <= 0:
        result["reason"] = "trim would remove all frames"
        return result

    kept_ts = timestamps[(keep_mask[: timestamps.size])]
    if kept_ts.size >= 2:
        result["trim_seconds"] = float(kept_ts[0] - timestamps[0])
    else:
        result["trim_seconds"] = float(window_start - timestamps[0])

    if not execute:
        result["status"] = "dry-run"
        return result

    tmp_avi = avi_path.with_name(f"{avi_path.stem}.trimtmp{avi_path.suffix}")
    tmp_ts = ts_path.with_name(f"{ts_path.stem}.trimtmp{ts_path.suffix}")

    written = write_trimmed_video(avi_path, tmp_avi, keep_mask, fps)
    if written != new_frames:
        if tmp_avi.exists():
            tmp_avi.unlink()
        raise RuntimeError(
            f"{avi_path}: wrote {written} frames, expected {new_frames}"
        )

    out = {}
    for key in data.files:
        value = data[key]
        if key == "timestamps":
            out[key] = kept_ts
        elif hasattr(value, "shape") and value.shape[:1] == (timestamps.size,):
            out[key] = value[: timestamps.size][keep_mask[: timestamps.size]]
        else:
            out[key] = value
    np.savez_compressed(tmp_ts, **out)

    os.replace(tmp_avi, avi_path)
    os.replace(tmp_ts, ts_path)
    result["status"] = "trimmed"
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Trim replay camera pre-roll from traj_*/replay videos."
    )
    parser.add_argument(
        "dataset",
        nargs="?",
        default="pick_cup_place_next_to_bowl/formal",
        type=Path,
        help="Dataset folder containing traj_* directories.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Overwrite AVI and *_frame_ts.npz files. Without this, only report.",
    )
    args = parser.parse_args()

    dataset = args.dataset.expanduser()
    traj_dirs = sorted(
        [p for p in dataset.glob("traj_*") if p.is_dir()],
        key=natural_traj_key,
    )
    if not traj_dirs:
        raise SystemExit(f"No traj_* directories found under {dataset}")

    totals = {
        "traj": 0,
        "trimmed": 0,
        "dry-run": 0,
        "skip": 0,
        "old_frames": 0,
        "new_frames": 0,
    }

    for traj_dir in traj_dirs:
        replay_dir = traj_dir / "replay"
        if not replay_dir.exists():
            continue
        duration = replay_duration(replay_dir)
        totals["traj"] += 1
        parts = []
        for camera in CAMERAS:
            info = trim_camera(replay_dir, camera, duration, args.execute)
            totals[info["status"]] += 1
            totals["old_frames"] += info["old_frames"]
            totals["new_frames"] += info["new_frames"]
            if info["status"] == "skip":
                parts.append(
                    f"{camera}:skip({info.get('reason', 'unknown')},"
                    f" frames={info['old_frames']})"
                )
            else:
                parts.append(
                    f"{camera}:{info['old_frames']}->{info['new_frames']}"
                    f" trim={info['trim_seconds']:.3f}s"
                )
        print(f"{traj_dir.name} duration={duration:.3f}s  " + "  ".join(parts))

    mode = "EXECUTE" if args.execute else "DRY-RUN"
    print(
        f"[{mode}] traj={totals['traj']} trimmed={totals['trimmed']} "
        f"dry_run={totals['dry-run']} skip={totals['skip']} "
        f"frames={totals['old_frames']}->{totals['new_frames']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
