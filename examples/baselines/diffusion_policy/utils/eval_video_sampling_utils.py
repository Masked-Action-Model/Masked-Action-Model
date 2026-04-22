from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class EvalBatch:
    indices: list[int]
    valid_count: int
    capture_index: int | None = None


def validate_eval_video_config(
    num_eval_episodes: int,
    num_eval_envs: int,
    capture_video_freq: int,
) -> None:
    if int(num_eval_envs) <= 0:
        raise ValueError(f"num_eval_envs must be positive, got {num_eval_envs}")
    if int(capture_video_freq) <= 0:
        raise ValueError(
            f"capture_video_freq must be positive, got {capture_video_freq}"
        )
    if int(num_eval_episodes) <= 0:
        raise ValueError(
            f"num_eval_episodes must be positive, got {num_eval_episodes}"
        )
    if int(num_eval_episodes) % int(num_eval_envs) != 0:
        raise ValueError(
            "num_eval_episodes must be divisible by num_eval_envs, "
            f"got num_eval_episodes={num_eval_episodes}, num_eval_envs={num_eval_envs}"
        )
    if int(num_eval_episodes) % int(capture_video_freq) != 0:
        raise ValueError(
            "num_eval_episodes must be divisible by capture_video_freq, "
            f"got num_eval_episodes={num_eval_episodes}, "
            f"capture_video_freq={capture_video_freq}"
        )


def build_capture_indices(total_items: int, capture_video_freq: int) -> set[int]:
    if total_items <= 0:
        return set()
    if capture_video_freq <= 0:
        raise ValueError(
            f"capture_video_freq must be positive, got {capture_video_freq}"
        )
    return set(range(0, int(total_items), int(capture_video_freq)))


def build_eval_batches(
    item_indices: Sequence[int] | int,
    batch_size: int,
    capture_indices: Iterable[int] | None = None,
) -> list[EvalBatch]:
    if isinstance(item_indices, int):
        items = list(range(item_indices))
    else:
        items = [int(idx) for idx in item_indices]
    if len(items) == 0:
        return []
    if int(batch_size) <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    capture_set = set(int(idx) for idx in capture_indices or [])
    capture_items = [idx for idx in items if idx in capture_set]
    other_items = [idx for idx in items if idx not in capture_set]
    batches: list[EvalBatch] = []

    if capture_items:
        other_cursor = 0
        for capture_idx in capture_items:
            batch = [capture_idx]
            valid_count = 1
            while len(batch) < batch_size and other_cursor < len(other_items):
                batch.append(other_items[other_cursor])
                other_cursor += 1
                valid_count += 1
            while len(batch) < batch_size:
                batch.append(batch[-1])
            batches.append(
                EvalBatch(
                    indices=batch,
                    valid_count=valid_count,
                    capture_index=capture_idx,
                )
            )

        remaining = other_items[other_cursor:]
    else:
        remaining = items

    for start in range(0, len(remaining), batch_size):
        batch = remaining[start : start + batch_size]
        valid_count = len(batch)
        if valid_count < batch_size:
            batch = batch + [batch[-1]] * (batch_size - valid_count)
        batches.append(EvalBatch(indices=batch, valid_count=valid_count))

    return batches
