"""Align feature streams to one shared time grid and build modality masks."""
from __future__ import annotations

import numpy as np

from .feature_io import Sequence


def align_to_grid(
    groups: dict[str, Sequence],
    step_ms: float = 40.0,
    tolerance_ms: float = 25.0,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Nearest-neighbour resampling of every group onto a common grid.

    A grid point is valid for a group only if the group's frame is valid and
    lies within ``tolerance_ms`` of the grid point.
    """
    groups = {k: v for k, v in groups.items() if len(v.timestamps_ms) > 0}
    if not groups:
        raise ValueError("No non-empty feature groups to align")

    t0 = min(seq.timestamps_ms[0] for seq in groups.values())
    t1 = max(seq.timestamps_ms[-1] for seq in groups.values())
    grid = np.arange(t0, t1 + step_ms * 0.5, step_ms)

    feats: dict[str, np.ndarray] = {}
    masks: dict[str, np.ndarray] = {}
    for name, seq in groups.items():
        ts = seq.timestamps_ms
        idx = np.searchsorted(ts, grid, side="left")
        right = np.clip(idx, 0, len(ts) - 1)
        left = np.clip(idx - 1, 0, len(ts) - 1)
        d_right = np.abs(ts[right] - grid)
        d_left = np.abs(ts[left] - grid)
        use_left = d_left < d_right
        best = np.where(use_left, left, right)
        dist = np.where(use_left, d_left, d_right)
        feats[name] = seq.features[best]
        masks[name] = seq.valid[best] & (dist <= tolerance_ms)
    return grid, feats, masks


def combine_masks(masks: dict[str, np.ndarray], core: list[str], policy: str, length: int) -> np.ndarray:
    """Frame validity for one modality.

    ``or``: any group valid. ``and_core``: all present core groups valid.
    ``any_core``: any present core group valid. Without present core groups the
    core policies fall back to ``or``.
    """
    if not masks:
        return np.zeros(length, dtype=bool)
    all_masks = np.stack(list(masks.values()))
    if policy == "or":
        return all_masks.any(axis=0)
    core_masks = [m for name, m in masks.items() if name in core]
    if not core_masks:
        return all_masks.any(axis=0)
    stacked = np.stack(core_masks)
    if policy == "and_core":
        return stacked.all(axis=0)
    if policy == "any_core":
        return stacked.any(axis=0)
    raise ValueError(f"Unknown mask_policy {policy!r}; expected or / and_core / any_core")
