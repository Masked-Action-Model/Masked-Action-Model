from __future__ import annotations

from typing import Any, Callable

import numpy as np


STATE_TOP_LEVEL_KEYS = ("agent", "extra")


def _mapping_items(value: Any):
    if hasattr(value, "spaces"):
        return list(value.spaces.items())
    if hasattr(value, "items"):
        return list(value.items())
    return None


def iter_default_state_obs_items(obs: dict[str, Any]) -> list[tuple[str, Any]]:
    """Return every policy state component in the same order used for flattening."""
    items: list[tuple[str, Any]] = []
    top_items = dict(_mapping_items(obs) or [])

    def visit(value: Any, path: str) -> None:
        children = _mapping_items(value)
        if children is None:
            items.append((path, value))
            return
        for key, child in children:
            visit(child, f"{path}/{key}")

    for top_key in STATE_TOP_LEVEL_KEYS:
        if top_key in top_items:
            visit(top_items[top_key], f"obs/{top_key}")
    return items


def build_default_state_obs_extractor() -> Callable[[dict[str, Any]], list[Any]]:
    return lambda obs: [value for _, value in iter_default_state_obs_items(obs)]


def _component_frame_shape(component: Any, has_leading_axis: bool) -> list[int]:
    shape = getattr(component, "shape", None)
    if shape is None:
        shape = np.asarray(component).shape
    shape = [int(v) for v in shape]
    if has_leading_axis and len(shape) > 0:
        shape = shape[1:]
    return shape


def build_state_schema_from_obs(
    obs: dict[str, Any],
    has_leading_axis: bool,
) -> list[dict[str, Any]]:
    schema: list[dict[str, Any]] = []
    offset = 0
    for path, component in iter_default_state_obs_items(obs):
        shape = _component_frame_shape(component, has_leading_axis=has_leading_axis)
        dim = int(np.prod(shape)) if len(shape) > 0 else 1
        dtype = getattr(component, "dtype", None)
        entry = {
            "path": path,
            "shape": shape,
            "dtype": str(dtype) if dtype is not None else str(np.asarray(component).dtype),
            "dim": dim,
            "start": offset,
            "end": offset + dim,
        }
        schema.append(entry)
        offset += dim
    if len(schema) == 0:
        raise ValueError("No state components found under obs/agent or obs/extra")
    return schema


def _prepare_state_component(component: Any) -> np.ndarray:
    array = np.asarray(component)
    if array.dtype == np.bool_:
        array = array.astype(np.float32)
    elif array.dtype == np.float64:
        array = array.astype(np.float32)
    if array.ndim == 0:
        return array.reshape(1, 1)
    if array.ndim == 1:
        return array.reshape(-1, 1)
    return array.astype(np.float32, copy=False)


def flatten_state_from_obs(
    obs: dict[str, Any],
    state_obs_extractor: Callable[[dict[str, Any]], list[Any]] | None = None,
) -> np.ndarray:
    extractor = state_obs_extractor or build_default_state_obs_extractor()
    state_components = [_prepare_state_component(component) for component in extractor(obs)]
    if len(state_components) == 0:
        raise ValueError("state_obs_extractor returned no components")
    if state_components[0].ndim != 2:
        raise ValueError(
            f"state components must be 2D after preparation, got {state_components[0].shape}"
        )
    first_len = state_components[0].shape[0]
    for component in state_components[1:]:
        if component.ndim != 2:
            raise ValueError(
                f"state components must be 2D after preparation, got {component.shape}"
            )
        if component.shape[0] != first_len:
            raise ValueError(
                "state components have mismatched leading dimensions: "
                f"{first_len} vs {component.shape[0]}"
            )
    return np.concatenate(state_components, axis=1).astype(np.float32)
