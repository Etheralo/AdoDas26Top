"""Conformer encoder with rotary self-attention and masked temporal downsampling."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def safe_key_mask(mask: torch.Tensor) -> torch.Tensor:
    """Rows without any valid key attend everywhere (their output is masked later)
    instead of producing NaN from an all -inf softmax."""
    return mask | ~mask.any(dim=-1, keepdim=True)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, base: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, length: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(length, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq.to(device))
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos(), emb.sin()


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    rotated = torch.cat([-x[..., half:], x[..., :half]], dim=-1)
    return x * cos.to(x.dtype) + rotated * sin.to(x.dtype)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float, rotary: bool = True) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.h = n_heads
        self.dh = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = dropout
        self.rope = RotaryEmbedding(self.dh) if rotary else None

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        N, T, D = x.shape
        q, k, v = self.qkv(x).view(N, T, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        if self.rope is not None:
            cos, sin = self.rope(T, x.device)
            q, k = apply_rotary(q, cos, sin), apply_rotary(k, cos, sin)
        attn_mask = safe_key_mask(mask)[:, None, None, :]
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0.0
        )
        return self.out(y.transpose(1, 2).reshape(N, T, D))


class FeedForward(nn.Module):
    def __init__(self, d_model: int, mult: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, mult * d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(mult * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConvModule(nn.Module):
    def __init__(self, d_model: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("conv_kernel must be odd")
        self.norm_in = nn.LayerNorm(d_model)
        self.pw1 = nn.Linear(d_model, 2 * d_model)
        self.dw = nn.Conv1d(d_model, d_model, kernel_size, padding=kernel_size // 2, groups=d_model)
        self.norm = nn.LayerNorm(d_model)
        self.pw2 = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = F.glu(self.pw1(self.norm_in(x)), dim=-1)
        h = h * mask.unsqueeze(-1).to(h.dtype)  # keep padding out of the depthwise conv
        h = self.dw(h.transpose(1, 2)).transpose(1, 2)
        h = F.silu(self.norm(h))
        return self.drop(self.pw2(h))


class ConformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ff_mult: int, conv_kernel: int,
                 dropout: float, rotary: bool = True) -> None:
        super().__init__()
        self.ff1 = FeedForward(d_model, ff_mult, dropout)
        self.norm_att = nn.LayerNorm(d_model)
        self.att = MultiHeadSelfAttention(d_model, n_heads, dropout, rotary)
        self.drop = nn.Dropout(dropout)
        self.conv = ConvModule(d_model, conv_kernel, dropout)
        self.ff2 = FeedForward(d_model, ff_mult, dropout)
        self.norm_out = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = x + 0.5 * self.ff1(x)
        x = x + self.drop(self.att(self.norm_att(x), mask))
        x = x + self.conv(x, mask)
        x = x + 0.5 * self.ff2(x)
        return self.norm_out(x) * mask.unsqueeze(-1).to(x.dtype)


class TemporalDownsample(nn.Module):
    """Strided conv over time; masks become "any valid frame in the window" and
    side signals (VAD, QC) are averaged over valid frames."""

    def __init__(self, d_model: int, factor: int) -> None:
        super().__init__()
        self.factor = int(factor)
        self.conv = nn.Conv1d(d_model, d_model, self.factor, stride=self.factor) if self.factor > 1 else None

    def forward(self, x: torch.Tensor, mask: torch.Tensor, signals: list[torch.Tensor]):
        if self.conv is None:
            return x, mask, signals
        N, T, _ = x.shape
        f = self.factor
        pad = (-T) % f
        maskf = mask.float()
        if pad:
            x = F.pad(x, (0, 0, 0, pad))
            maskf = F.pad(maskf, (0, pad))
            signals = [F.pad(s, (0, pad)) for s in signals]
        T2 = (T + pad) // f
        x = self.conv(x.transpose(1, 2)).transpose(1, 2)
        m = maskf.view(N, T2, f)
        count = m.sum(-1).clamp(min=1.0)
        new_mask = m.sum(-1) > 0
        signals = [(s.view(N, T2, f) * m).sum(-1) / count for s in signals]
        return x * new_mask.unsqueeze(-1).to(x.dtype), new_mask, signals


class ModalityEncoder(nn.Module):
    def __init__(self, projector: nn.Module, d_model: int, n_layers: int, n_heads: int,
                 ff_mult: int, conv_kernel: int, dropout: float, downsample: int, rotary: bool) -> None:
        super().__init__()
        self.projector = projector
        self.down = TemporalDownsample(d_model, downsample)
        self.blocks = nn.ModuleList([
            ConformerBlock(d_model, n_heads, ff_mult, conv_kernel, dropout, rotary) for _ in range(n_layers)
        ])

    def forward(self, groups: dict[str, torch.Tensor], mask: torch.Tensor, vad: torch.Tensor, qc: torch.Tensor):
        x = self.projector(groups) * mask.unsqueeze(-1).float()
        x, mask, (vad, qc) = self.down(x, mask, [vad, qc])
        for block in self.blocks:
            x = block(x, mask)
        return x, mask, vad, qc
