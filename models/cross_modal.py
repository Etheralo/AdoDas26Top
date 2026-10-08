"""Frame-level bidirectional audio-visual cross-attention (HITSZ_HLT)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conformer import FeedForward, safe_key_mask


class CrossAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.h = n_heads
        self.dh = d_model // n_heads
        self.q = nn.Linear(d_model, d_model)
        self.kv = nn.Linear(d_model, 2 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, ctx: torch.Tensor, ctx_mask: torch.Tensor) -> torch.Tensor:
        N, T, D = x.shape
        S = ctx.shape[1]
        q = self.q(x).view(N, T, self.h, self.dh).transpose(1, 2)
        k, v = self.kv(ctx).view(N, S, 2, self.h, self.dh).permute(2, 0, 3, 1, 4)
        attn_mask = safe_key_mask(ctx_mask)[:, None, None, :]
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0.0
        )
        return self.out(y.transpose(1, 2).reshape(N, T, D))


class CrossModalBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ff_mult: int, dropout: float) -> None:
        super().__init__()
        self.norm_a = nn.LayerNorm(d_model)
        self.norm_v = nn.LayerNorm(d_model)
        self.a_from_v = CrossAttention(d_model, n_heads, dropout)
        self.v_from_a = CrossAttention(d_model, n_heads, dropout)
        self.ff_a = FeedForward(d_model, ff_mult, dropout)
        self.ff_v = FeedForward(d_model, ff_mult, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, a: torch.Tensor, ma: torch.Tensor, v: torch.Tensor, mv: torch.Tensor):
        an, vn = self.norm_a(a), self.norm_v(v)
        # A session whose other modality is entirely missing gets no cross-modal update.
        has_v = mv.any(dim=-1).to(a.dtype)[:, None, None]
        has_a = ma.any(dim=-1).to(v.dtype)[:, None, None]
        a = a + self.drop(self.a_from_v(an, vn, mv)) * has_v
        v = v + self.drop(self.v_from_a(vn, an, ma)) * has_a
        a = a + self.ff_a(a)
        v = v + self.ff_v(v)
        return a * ma.unsqueeze(-1).to(a.dtype), v * mv.unsqueeze(-1).to(v.dtype)
