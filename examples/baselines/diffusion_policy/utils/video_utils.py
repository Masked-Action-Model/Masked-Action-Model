from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterable


def snapshot_video_files(video_dir: str | Path) -> set[Path]:
    """Return the current set of mp4 files under a video directory."""
    root = Path(video_dir)
    if not root.exists():
        return set()
    return {path.resolve() for path in root.glob("*.mp4")}


def find_new_video_files(
    video_dir: str | Path, previous_snapshot: Iterable[Path]
) -> list[Path]:
    """Find mp4 files created since the previous snapshot."""
    previous = {Path(path).resolve() for path in previous_snapshot}
    current = snapshot_video_files(video_dir)
    return sorted(current - previous)


def delete_new_video_files(video_dir: str | Path, previous_snapshot: Iterable[Path]):
    for video_path in find_new_video_files(video_dir, previous_snapshot):
        Path(video_path).unlink(missing_ok=True)


def clear_iteration_artifacts(video_dir: str | Path, iteration: int):
    """Remove archived videos/curves for one eval iteration so reruns don't accumulate stale files."""
    video_root = Path(video_dir)
    iter_prefix = f"iter_{iteration:07d}_"
    artifact_roots = [
        video_root,
        video_root / "success",
        video_root / "failed",
        video_root.parent / "graphs" / "success",
        video_root.parent / "graphs" / "failed",
    ]
    for root in artifact_roots:
        if not root.exists():
            continue
        for path in root.glob(f"{iter_prefix}*"):
            if path.is_file():
                path.unlink(missing_ok=True)


def build_success_video_path(
    video_dir: str | Path,
    iteration: int,
    demo_idx: int,
    suffix: str = "",
) -> Path:
    """Build the target success video path for one evaluation rollout."""
    success_dir = Path(video_dir) / "success"
    video_name = f"iter_{iteration:07d}_demo_{demo_idx:04d}"
    if suffix:
        video_name += f"_{suffix}"
    return success_dir / f"{video_name}.mp4"


def build_failed_video_path(
    video_dir: str | Path,
    iteration: int,
    demo_idx: int,
    suffix: str = "",
) -> Path:
    """Build the target failed video path for one evaluation rollout."""
    failed_dir = Path(video_dir) / "failed"
    video_name = f"iter_{iteration:07d}_demo_{demo_idx:04d}"
    if suffix:
        video_name += f"_{suffix}"
    return failed_dir / f"{video_name}.mp4"


def _move_video(
    video_path: str | Path,
    dst: Path,
    video_dir: str | Path,
    iteration: int,
    demo_idx: int,
    suffix: str = "",
    overwrite: bool = False,
    path_builder=None,
) -> Path | None:
    src = Path(video_path)
    if not src.exists():
        return None

    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists() and not overwrite:
        duplicate_id = 1
        while True:
            candidate = path_builder(
                video_dir=video_dir,
                iteration=iteration,
                demo_idx=demo_idx,
                suffix=f"{suffix}_dup{duplicate_id}" if suffix else f"dup{duplicate_id}",
            )
            if not candidate.exists():
                dst = candidate
                break
            duplicate_id += 1

    try:
        return Path(shutil.move(str(src), str(dst)))
    except FileNotFoundError:
        return None


def move_video_to_success(
    video_path: str | Path,
    video_dir: str | Path,
    iteration: int,
    demo_idx: int,
    suffix: str = "",
    overwrite: bool = False,
) -> Path:
    """Move one recorded video into the success folder with a deterministic name."""
    dst = build_success_video_path(
        video_dir=video_dir,
        iteration=iteration,
        demo_idx=demo_idx,
        suffix=suffix,
    )
    return _move_video(
        video_path=video_path,
        dst=dst,
        video_dir=video_dir,
        iteration=iteration,
        demo_idx=demo_idx,
        suffix=suffix,
        overwrite=overwrite,
        path_builder=build_success_video_path,
    )


def move_video_to_failed(
    video_path: str | Path,
    video_dir: str | Path,
    iteration: int,
    demo_idx: int,
    suffix: str = "",
    overwrite: bool = False,
) -> Path:
    """Move one recorded video into the failed folder with a deterministic name."""
    dst = build_failed_video_path(
        video_dir=video_dir,
        iteration=iteration,
        demo_idx=demo_idx,
        suffix=suffix,
    )
    return _move_video(
        video_path=video_path,
        dst=dst,
        video_dir=video_dir,
        iteration=iteration,
        demo_idx=demo_idx,
        suffix=suffix,
        overwrite=overwrite,
        path_builder=build_failed_video_path,
    )


def collect_success_video(
    video_dir: str | Path,
    previous_snapshot: Iterable[Path],
    iteration: int,
    demo_idx: int,
    suffix: str = "",
    overwrite: bool = False,
    expect_single: bool = True,
) -> Path | None:
    """Move the newly created evaluation video into videos/success/ with the target name."""
    new_videos = find_new_video_files(video_dir, previous_snapshot)
    if len(new_videos) == 0:
        return None
    if expect_single and len(new_videos) != 1:
        # In vectorized eval, multiple envs may flush videos between two snapshots.
        # Use the newest file and keep training robust instead of raising.
        new_videos = sorted(new_videos, key=lambda p: p.stat().st_mtime)
    return move_video_to_success(
        video_path=new_videos[-1],
        video_dir=video_dir,
        iteration=iteration,
        demo_idx=demo_idx,
        suffix=suffix,
        overwrite=overwrite,
    )


def collect_failed_video(
    video_dir: str | Path,
    previous_snapshot: Iterable[Path],
    iteration: int,
    demo_idx: int,
    suffix: str = "",
    overwrite: bool = False,
    expect_single: bool = True,
) -> Path | None:
    """Move the newly created evaluation video into videos/failed/ with the target name."""
    new_videos = find_new_video_files(video_dir, previous_snapshot)
    if len(new_videos) == 0:
        return None
    if expect_single and len(new_videos) != 1:
        # In vectorized eval, multiple envs may flush videos between two snapshots.
        # Use the newest file and keep training robust instead of raising.
        new_videos = sorted(new_videos, key=lambda p: p.stat().st_mtime)
    return move_video_to_failed(
        video_path=new_videos[-1],
        video_dir=video_dir,
        iteration=iteration,
        demo_idx=demo_idx,
        suffix=suffix,
        overwrite=overwrite,
    )
