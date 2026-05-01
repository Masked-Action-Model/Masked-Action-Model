from __future__ import annotations

from typing import Any, Callable

import numpy as np


def build_default_state_obs_extractor() -> Callable[[dict[str, Any]], list[Any]]:
    return lambda obs: list(obs["agent"].values()) + list(obs["extra"].values())


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
