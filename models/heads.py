"""Task heads.

- ``A1Head``: three sigmoid logits (D/A/S screening).
- ``QueryCoralHead`` (HEALTH FIRST): 21 learnable item queries cross-attend to
  the participant and session tokens; each item gets a CORAL-style score minus
  three monotonically increasing thresholds.
- ``A1GuidedCorrection`` (HITSZ_HLT): shifts every item's logits by a learned
  multiple of its subscale's A1 logit.
Logit k of an item models P(score > k), k = 0, 1, 2.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def _inv_softplus(y: np.ndarray) -> np.ndarray:
    y = np.maximum(y, 1e-3)
    return y + np.log(-np.expm1(-y))


class A1Head(nn.Module):
    def __init__(self, d: int, dropout: float, prevalence: np.ndarray | None = None) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Dropout(dropout), nn.Linear(d, 3))
        if prevalence is not None:
            with torch.no_grad():
                self.net[-1].bias.copy_(torch.as_tensor(_logit(prevalence), dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CoralThresholds(nn.Module):
    """Per-item increasing thresholds: t0 free, t_k = t_{k-1} + softplus(gap_k)."""

    def __init__(self, n_items: int, n_thresholds: int = 3) -> None:
        super().__init__()
        self.first = nn.Parameter(torch.zeros(n_items, 1))
        self.raw_gaps = nn.Parameter(torch.full((n_items, n_thresholds - 1), float(_inv_softplus(np.array(1.0)))))

    def forward(self) -> torch.Tensor:
        return torch.cat([self.first, self.first + torch.cumsum(F.softplus(self.raw_gaps), dim=-1)], dim=-1)

    @torch.no_grad()
    def init_from_prior(self, cum_probs: np.ndarray) -> None:
        """cum_probs (I, 3) = P(y > k) on train; with a zero score the head then
        predicts the prior."""
        thr = -_logit(np.asarray(cum_probs, dtype=np.float64))
        thr = np.maximum.accumulate(thr, axis=1)
        gaps = np.maximum(np.diff(thr, axis=1), 1e-2)
        self.first.copy_(torch.as_tensor(thr[:, :1], dtype=torch.float32))
        self.raw_gaps.copy_(torch.as_tensor(_inv_softplus(gaps), dtype=torch.float32))


class LinearCoralHead(nn.Module):
    """One linear score per item (ablation of the query head; also used for the
    session-level auxiliary loss)."""

    def __init__(self, d: int, n_items: int = 21) -> None:
        super().__init__()
        self.score = nn.Linear(d, n_items)
        self.thresholds = CoralThresholds(n_items)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.score(x).float().unsqueeze(-1) - self.thresholds().unsqueeze(0)


class QueryCoralHead(nn.Module):
    def __init__(self, d: int, n_layers: int, n_heads: int, dropout: float, n_items: int = 21) -> None:
        super().__init__()
        self.queries = nn.Parameter(torch.randn(n_items, d) * 0.02)
        layer = nn.TransformerDecoderLayer(
            d, n_heads, 4 * d, dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.decoder = nn.TransformerDecoder(layer, n_layers)
        self.norm = nn.LayerNorm(d)
        self.item_w = nn.Parameter(torch.randn(n_items, d) * 0.02)
        self.item_b = nn.Parameter(torch.zeros(n_items))
        self.thresholds = CoralThresholds(n_items)

    def forward(self, memory: torch.Tensor, memory_pad: torch.Tensor) -> torch.Tensor:
        B = memory.shape[0]
        q = self.queries.unsqueeze(0).expand(B, -1, -1).to(memory.dtype)
        h = self.norm(self.decoder(q, memory, memory_key_padding_mask=memory_pad)).float()
        score = (h * self.item_w).sum(-1) + self.item_b
        return score.unsqueeze(-1) - self.thresholds().unsqueeze(0)


class A1GuidedCorrection(nn.Module):
    def __init__(self, item_to_subscale: np.ndarray, detach: bool = True) -> None:
        super().__init__()
        self.register_buffer("item_to_sub", torch.as_tensor(item_to_subscale, dtype=torch.long))
        self.weight = nn.Parameter(torch.zeros(len(item_to_subscale)))
        self.detach = detach

    def forward(self, a2_logits: torch.Tensor, a1_logits: torch.Tensor) -> torch.Tensor:
        src = a1_logits.detach() if self.detach else a1_logits
        shift = src.float()[:, self.item_to_sub] * self.weight
        return a2_logits + shift.unsqueeze(-1)
