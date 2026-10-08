"""Per-group input adapters with train-set feature standardization."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Standardize(nn.Module):
    """(x - mean) / std with train-set statistics stored as buffers, then clipped."""

    def __init__(self, dim: int, stats: tuple[np.ndarray, np.ndarray] | None = None,
                 clip: float = 10.0, noise_std: float = 0.0) -> None:
        super().__init__()
        mean = torch.zeros(dim) if stats is None else torch.as_tensor(stats[0], dtype=torch.float32)
        std = torch.ones(dim) if stats is None else torch.as_tensor(stats[1], dtype=torch.float32)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        self.clip = clip
        self.noise_std = noise_std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = ((x - self.mean) / self.std).clamp(-self.clip, self.clip)
        if self.training and self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std
        return x


class GroupAdapter(nn.Module):
    def __init__(self, d_in: int, d_out: int, dropout: float, stats=None, noise_std: float = 0.0) -> None:
        super().__init__()
        self.standardize = Standardize(d_in, stats, noise_std=noise_std)
        self.proj = nn.Linear(d_in, d_out)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(F.gelu(self.proj(self.standardize(x))))


class ModalityProjector(nn.Module):
    """Adapts every feature group of one modality and fuses them to ``d_model``."""

    def __init__(self, dims: dict[str, int], modality: str, stats: dict | None,
                 d_adapter: int, d_model: int, dropout: float, noise_std: float) -> None:
        super().__init__()
        self.names = sorted(dims)
        self.adapters = nn.ModuleDict({
            n: GroupAdapter(dims[n], d_adapter, dropout,
                            None if stats is None else stats.get(f"{modality}/{n}"), noise_std)
            for n in self.names
        })
        self.proj = nn.Linear(len(self.names) * d_adapter, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, groups: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.norm(self.proj(torch.cat([self.adapters[n](groups[n]) for n in self.names], dim=-1)))
