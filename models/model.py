"""Full model: session encoder -> cross-session Transformer -> A1 / A2 heads.

Every component named in the AdoDAS paper (Table 3) can be switched off from
the config for ablations:
  dual Conformers (HEALTH FIRST)            model.conformer_layers
  frame-level A-V attention (HITSZ_HLT)     model.use_cross_modal
  session-aware ASR text (HITSZ_HLT)        model.use_text
  cross-session Transformer (both)          model.cross_session_layers (0 = masked mean)
  query-aligned 21-item CORAL (HEALTH FIRST) model.a2_head = query_coral | linear_coral
  A1-guided item correction (HITSZ_HLT)     model.use_a1_guidance
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from data.dass import item_to_subscale

from .adapters import GroupAdapter, ModalityProjector
from .conformer import ModalityEncoder
from .cross_modal import CrossModalBlock
from .cross_session import CrossSessionEncoder
from .heads import A1GuidedCorrection, A1Head, LinearCoralHead, QueryCoralHead
from .pooling import AttentiveStatsPool
from .text_branch import TextBranch

HEAD_MODULES = ("a1_head", "a2_head", "a1_guidance")
N_SESSIONS = 4
D_SESSION_TYPE = 16


class AdoDASModel(nn.Module):
    def __init__(self, m: dict, dims: dict, stats: dict | None = None, priors: dict | None = None) -> None:
        super().__init__()
        self.m = m
        d = int(m["d_model"])
        ds = int(m["d_session"])
        drop = float(m["dropout"])
        noise = float(m["feature_noise_std"])
        self.has_audio = bool(dims["audio"])
        self.has_video = bool(dims["video"])
        if not (self.has_audio or self.has_video):
            raise ValueError("At least one audio or video sequence feature is required")

        def encoder(modality: str) -> ModalityEncoder:
            proj = ModalityProjector(dims[modality], modality, stats, int(m["d_adapter"]), d, drop, noise)
            return ModalityEncoder(proj, d, int(m["conformer_layers"]), int(m["n_heads"]), int(m["ff_mult"]),
                                   int(m["conv_kernel"]), drop, int(m["downsample"]), bool(m["use_rotary"]))

        self.audio_enc = encoder("audio") if self.has_audio else None
        self.video_enc = encoder("video") if self.has_video else None
        self.audio_pool = AttentiveStatsPool(d) if self.has_audio else None
        self.video_pool = AttentiveStatsPool(d) if self.has_video else None
        use_xm = bool(m["use_cross_modal"]) and self.has_audio and self.has_video
        self.cross_modal = nn.ModuleList(
            [CrossModalBlock(d, int(m["n_heads"]), int(m["ff_mult"]), drop) for _ in range(int(m["cross_modal_layers"]))]
        ) if use_xm else nn.ModuleList()

        self.pooled_names = sorted(dims["pooled"])
        self.pooled_adapters = nn.ModuleDict({
            n: GroupAdapter(dims["pooled"][n], int(m["d_adapter"]), drop,
                            None if stats is None else stats.get(f"pooled/{n}"), noise)
            for n in self.pooled_names
        })

        self.use_text = bool(m["use_text"]) and bool(dims.get("text"))
        self.text = TextBranch(int(dims["text"]), int(m["d_text"]), int(m["text_layers"]), int(m["n_heads"]), drop) \
            if self.use_text else None

        self.session_type_emb = nn.Embedding(N_SESSIONS, D_SESSION_TYPE)
        n_flags = int(self.has_audio) + int(self.has_video) + len(self.pooled_names) + int(self.use_text)
        fusion_in = (2 * d) * (int(self.has_audio) + int(self.has_video))
        fusion_in += int(m["d_adapter"]) * len(self.pooled_names)
        fusion_in += int(m["d_text"]) * int(self.use_text) + D_SESSION_TYPE + n_flags
        self.session_fusion = nn.Sequential(
            nn.Linear(fusion_in, ds), nn.GELU(), nn.Dropout(drop), nn.Linear(ds, ds), nn.LayerNorm(ds)
        )

        self.cross_session = CrossSessionEncoder(ds, int(m["cross_session_layers"]), int(m["cross_session_heads"]), drop)

        priors = priors or {}
        self.a1_head = A1Head(ds, drop, priors.get("a1_prevalence"))
        self.query_head = m["a2_head"] == "query_coral"
        if self.query_head:
            self.a2_head = QueryCoralHead(ds, int(m["query_layers"]), int(m["cross_session_heads"]), drop)
        elif m["a2_head"] == "linear_coral":
            self.a2_head = LinearCoralHead(ds)
        else:
            raise ValueError(f"Unknown model.a2_head {m['a2_head']!r}")
        self.a1_guidance = A1GuidedCorrection(item_to_subscale(), bool(m["a1_guidance_detach"])) \
            if m["use_a1_guidance"] else None

        # Session-level auxiliary heads (each session predicts the participant labels).
        self.aux_a1 = nn.Linear(ds, 3)
        self.aux_a2 = LinearCoralHead(ds)

        if "a2_cum_probs" in priors:
            self.a2_head.thresholds.init_from_prior(priors["a2_cum_probs"])
            self.aux_a2.thresholds.init_from_prior(priors["a2_cum_probs"])

        self._encoder_frozen = False

    # ------------------------------------------------------------------ stages
    def freeze_encoder(self, task: str = "both") -> None:
        """Stage 2: only the task heads keep training; the rest runs in eval mode."""
        if task not in ("a1", "a2", "both"):
            raise ValueError(f"Unknown task {task!r}")
        self._stage2_heads = HEAD_MODULES if task == "both" else ("a1_head",) if task == "a1" else ("a2_head", "a1_guidance")
        self._encoder_frozen = True
        for name, module in self.named_children():
            if name not in self._stage2_heads:
                for p in module.parameters():
                    p.requires_grad_(False)
        self.train(self.training)

    def train(self, mode: bool = True):
        super().train(mode)
        if self._encoder_frozen:
            for name, module in self.named_children():
                if name not in self._stage2_heads:
                    module.eval()
        return self

    # ------------------------------------------------------------------ forward
    def encode_sessions(self, batch: dict) -> torch.Tensor:
        st = batch["session_type"]
        ma, mv = batch["mask_audio"], batch["mask_video"]
        p_drop = float(self.m["modality_drop_prob"])
        if self.training and p_drop > 0 and self.has_audio and self.has_video and not self._encoder_frozen:
            r = torch.rand(ma.shape[0], device=ma.device)
            ma = ma & ~(r < p_drop / 2).unsqueeze(-1)
            mv = mv & ~((r >= p_drop / 2) & (r < p_drop)).unsqueeze(-1)

        parts, flags = [], []
        a = v = None
        if self.has_audio:
            a, ma, vad_a, qc_a = self.audio_enc(batch["audio"], ma, batch["vad"], batch["qc"])
        if self.has_video:
            v, mv, vad_v, qc_v = self.video_enc(batch["video"], mv, batch["vad"], batch["qc"])
        for block in self.cross_modal:
            a, v = block(a, ma, v, mv)
        if self.has_audio:
            parts.append(self.audio_pool(a, ma, vad_a, None))
            flags.append(ma.any(-1))
        if self.has_video:
            parts.append(self.video_pool(v, mv, vad_v, qc_v))
            flags.append(mv.any(-1))
        for n in self.pooled_names:
            present = batch["pooled_present"][n]
            parts.append(self.pooled_adapters[n](batch["pooled"][n]).float() * present.unsqueeze(-1).float())
            flags.append(present)
        if self.use_text:
            parts.append(self.text(batch["text"], batch["text_mask"], batch["text_present"], st).float())
            flags.append(batch["text_present"])
        parts.append(self.session_type_emb(st).float())
        parts.append(torch.stack(flags, dim=-1).float())
        return self.session_fusion(torch.cat([p.float() for p in parts], dim=-1))

    def forward(self, batch: dict) -> dict[str, torch.Tensor]:
        valid = batch["session_valid"]
        B = valid.shape[0]
        tokens = self.encode_sessions(batch)
        tokens = tokens * valid.reshape(-1, 1).to(tokens.dtype)
        cls, sess = self.cross_session(tokens.view(B, N_SESSIONS, -1), valid)

        a1 = self.a1_head(cls).float()
        if self.query_head:
            memory = torch.cat([cls.unsqueeze(1), sess], dim=1)
            pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=valid.device), ~valid], dim=1)
            a2 = self.a2_head(memory, pad)
        else:
            a2 = self.a2_head(cls)
        if self.a1_guidance is not None:
            a2 = self.a1_guidance(a2, a1)
        return {
            "a1": a1,
            "a2": a2.float(),
            "sess_a1": self.aux_a1(tokens).float(),
            "sess_a2": self.aux_a2(tokens),
        }


def build_model(model_cfg: dict, dims: dict, stats: dict | None = None, priors: dict | None = None) -> AdoDASModel:
    return AdoDASModel(model_cfg, dims, stats, priors)


def compute_priors(y_a1: np.ndarray, y_a2: np.ndarray) -> dict:
    a1 = y_a1[(y_a1 >= 0).all(1)]
    a2 = y_a2[(y_a2 >= 0).all(1)]
    return {
        "a1_prevalence": a1.mean(0) if len(a1) else np.full(3, 0.2),
        "a2_cum_probs": np.stack([(a2 > k).mean(0) for k in range(3)], axis=1) if len(a2) else np.full((21, 3), 0.2),
    }
