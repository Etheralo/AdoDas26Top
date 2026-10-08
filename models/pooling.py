"""Mask-aware attentive statistics pooling."""
from __future__ import annotations

import torch
import torch.nn as nn


def masked_softmax(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Softmax over valid positions; rows without valid positions get all-zero weights."""
    any_valid = mask.any(dim=-1, keepdim=True)
    scores = scores.masked_fill(~mask, float("-inf"))
    scores = torch.where(any_valid, scores, torch.zeros_like(scores))
    w = torch.softmax(scores, dim=-1) * mask.to(scores.dtype)
    return w / w.sum(dim=-1, keepdim=True).clamp(min=1e-6)


class AttentiveStatsPool(nn.Module):
    """Weighted mean and std of frames; attention logits are biased by VAD and
    face-quality signals so speech frames and good faces weigh more."""

    def __init__(self, d_model: int, hidden: int = 128) -> None:
        super().__init__()
        self.score = nn.Sequential(nn.Linear(d_model, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor, mask: torch.Tensor,
                vad: torch.Tensor | None = None, qc: torch.Tensor | None = None) -> torch.Tensor:
        e = self.score(x).squeeze(-1).float()
        if vad is not None:
            e = e + self.alpha * vad.float()
        if qc is not None:
            e = e + self.beta * qc.float()
        w = masked_softmax(e, mask).unsqueeze(-1)
        xf = x.float()
        mean = (w * xf).sum(dim=1)
        var = (w * (xf - mean.unsqueeze(1)) ** 2).sum(dim=1)
        std = torch.sqrt(var.clamp(min=1e-6))
        present = mask.any(dim=-1, keepdim=True).float()
        return torch.cat([mean, std], dim=-1) * present
