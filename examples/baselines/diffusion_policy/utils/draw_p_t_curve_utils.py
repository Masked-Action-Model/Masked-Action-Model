from __future__ import annotations

from pathlib import Path

import numpy as np

_WARNED_NO_MATPLOTLIB_FOR_CE = False


def build_curve_path_from_video(video_path: str | Path, video_dir: str | Path) -> Path:
    video_path = Path(video_path).resolve()
    video_dir = Path(video_dir).resolve()
    relative_video_path = video_path.relative_to(video_dir)
    return (video_dir.parent / "graphs" / relative_video_path).with_suffix(".png")


def save_progress_curve_for_video(
    video_path: str | Path,
    video_dir: str | Path,
    timesteps,
    progress,
):
    timesteps = np.asarray(timesteps, dtype=np.float32).reshape(-1)
    progress = np.asarray(progress, dtype=np.float32).reshape(-1)
    if timesteps.size == 0 or progress.size == 0:
        raise ValueError("timesteps/progress must be non-empty for curve drawing")
    if timesteps.shape != progress.shape:
        raise ValueError(
            f"timesteps/progress shape mismatch: {timesteps.shape} vs {progress.shape}"
        )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curve_path = build_curve_path_from_video(video_path=video_path, video_dir=video_dir)
    curve_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    ax.plot(timesteps, progress, marker="o", linewidth=1.8, markersize=3.5)
    ax.set_xlabel("timestep")
    ax.set_ylabel("progress")
    ax.set_ylim(0.0, 1.0)
    ax.set_xlim(left=0.0)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
    ax.set_title(curve_path.stem)
    fig.tight_layout()
    fig.savefig(curve_path)
    plt.close(fig)
    return curve_path


def save_control_error_curve(
    run_dir: str | Path,
    iterations,
    ce_all,
    ce_success,
    ce_failed,
):
    global _WARNED_NO_MATPLOTLIB_FOR_CE
    iterations = np.asarray(iterations, dtype=np.float32).reshape(-1)
    ce_all = np.asarray(ce_all, dtype=np.float32).reshape(-1)
    ce_success = np.asarray(ce_success, dtype=np.float32).reshape(-1)
    ce_failed = np.asarray(ce_failed, dtype=np.float32).reshape(-1)

    if iterations.size == 0:
        return None
    if not (
        iterations.shape == ce_all.shape == ce_success.shape == ce_failed.shape
    ):
        raise ValueError(
            "iterations/ce_all/ce_success/ce_failed shape mismatch: "
            f"{iterations.shape}, {ce_all.shape}, {ce_success.shape}, {ce_failed.shape}"
        )

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        if not _WARNED_NO_MATPLOTLIB_FOR_CE:
            print(
                "[control-error-curve] matplotlib not available, skip saving iter_ce.png"
            )
            _WARNED_NO_MATPLOTLIB_FOR_CE = True
        return None

    run_dir = Path(run_dir).resolve()
    curve_path = run_dir / "graphs" / "iter_ce.png"
    curve_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    ax.plot(iterations, ce_all, marker="o", linewidth=1.8, markersize=3.5, label="ce_all")
    ax.plot(
        iterations,
        ce_success,
        marker="o",
        linewidth=1.5,
        markersize=3.0,
        label="ce_success",
    )
    ax.plot(
        iterations,
        ce_failed,
        marker="o",
        linewidth=1.5,
        markersize=3.0,
        label="ce_failed",
    )
    ax.set_xlabel("iteration")
    ax.set_ylabel("control error")
    ax.set_xlim(left=0.0)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
    ax.set_title("iter_ce")
    ax.legend()
    fig.tight_layout()
    fig.savefig(curve_path)
    plt.close(fig)
    return curve_path
