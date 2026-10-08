"""Participant-level fusion of the four session tokens."""
from __future__ import annotations

import torch
import torch.nn as nn


class CrossSessionEncoder(nn.Module):
    """[CLS] + four session tokens through a Transformer; missing sessions are
    padded out. With ``n_layers=0`` falls back to masked mean + MLP (baseline-style)."""

    def __init__(self, d: int, n_layers: int, n_heads: int, dropout: float, n_sessions: int = 4) -> None:
        super().__init__()
        self.n_layers = n_layers
        self.session_pos = nn.Parameter(torch.randn(1, n_sessions, d) * 0.02)
        if n_layers > 0:
            self.cls = nn.Parameter(torch.randn(1, 1, d) * 0.02)
            layer = nn.TransformerEncoderLayer(
                d, n_heads, 4 * d, dropout, activation="gelu", batch_first=True, norm_first=True
            )
            self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        else:
            self.mlp = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Dropout(dropout), nn.Linear(d, d))
        self.norm = nn.LayerNorm(d)

    def forward(self, tokens: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B = tokens.shape[0]
        x = tokens + self.session_pos.to(tokens.dtype)
        if self.n_layers == 0:
            m = valid.unsqueeze(-1).to(x.dtype)
            pooled = (x * m).sum(1) / m.sum(1).clamp(min=1.0)
            return self.norm(self.mlp(pooled)), x
        x = torch.cat([self.cls.expand(B, -1, -1).to(x.dtype), x], dim=1)
        pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=valid.device), ~valid], dim=1)
        x = self.norm(self.encoder(x, src_key_padding_mask=pad))
        return x[:, 0], x[:, 1:]
