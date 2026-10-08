"""Session-aware ASR text branch (HITSZ_HLT).

Token embeddings come from a frozen Chinese encoder (extracted offline). A
learnable prompt per session type (B01 yesterday / B02 happy / B03 sad) is
prepended so the small Transformer adapts the text to the elicitation prompt.
Sessions without text (A01 reading, missing transcripts) use a learned token.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import math

from .pooling import masked_softmax


class TextBranch(nn.Module):
    def __init__(self, d_in: int, d_text: int, n_layers: int, n_heads: int,
                 dropout: float, n_sessions: int = 4) -> None:
        super().__init__()
        self.in_norm = nn.LayerNorm(d_in)
        self.proj = nn.Linear(d_in, d_text)
        self.session_prompt = nn.Embedding(n_sessions, d_text)
        layer = nn.TransformerEncoderLayer(
            d_text, n_heads, 4 * d_text, dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.attn = nn.Linear(d_text, 1)
        self.out = nn.Sequential(nn.LayerNorm(2 * d_text), nn.Linear(2 * d_text, d_text))
        self.no_text = nn.Parameter(torch.randn(d_text) * 0.02)

    def forward(self, text: torch.Tensor, text_mask: torch.Tensor,
                text_present: torch.Tensor, session_type: torch.Tensor) -> torch.Tensor:
        N = text.shape[0]
        h = self.proj(self.in_norm(text))
        prompt = self.session_prompt(session_type).unsqueeze(1).to(h.dtype)
        h = torch.cat([prompt, h], dim=1)
        # Preserve linguistic word order. Without positions, self-attention and
        # pooling would be invariant to permutations of transcript tokens.
        pos = torch.arange(h.shape[1], device=h.device, dtype=torch.float32)[:, None]
        freq = torch.exp(torch.arange(0, h.shape[-1], 2, device=h.device).float()
                         * (-math.log(10000.0) / h.shape[-1]))
        pe = torch.zeros(h.shape[1], h.shape[-1], device=h.device)
        pe[:, 0::2] = torch.sin(pos * freq)
        pe[:, 1::2] = torch.cos(pos * freq[:pe[:, 1::2].shape[-1]])
        h = h + pe.to(h.dtype).unsqueeze(0)
        valid = torch.cat([torch.ones(N, 1, dtype=torch.bool, device=text.device), text_mask], dim=1)
        h = self.encoder(h, src_key_padding_mask=~valid)
        prompt_out, tokens = h[:, 0], h[:, 1:]
        w = masked_softmax(self.attn(tokens).squeeze(-1).float(), text_mask).unsqueeze(-1)
        pooled = (w * tokens.float()).sum(dim=1)
        out = self.out(torch.cat([prompt_out.float(), pooled], dim=-1))
        return torch.where(text_present.unsqueeze(-1), out, self.no_text.to(out.dtype).expand_as(out))
