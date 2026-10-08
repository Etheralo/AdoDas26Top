"""Training losses.

total = a1_weight * BCE(A1) + a2_weight * CORAL-BCE(A2)
      + session_aux_weight * (session-level A1 + A2 losses)
      + subscale_weight * (subscale-score regression + A1-from-items BCE)
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from data.dass import decision_midpoints, subscale_membership


def a1_pos_weight(y: np.ndarray) -> np.ndarray:
    y = y[(y >= 0).all(1)]
    pos = y.sum(0)
    neg = len(y) - pos
    return np.clip(np.sqrt(neg / np.maximum(pos, 1)), 1.0, 4.0).astype(np.float32)


def a2_pos_weight(y: np.ndarray, cap: float) -> np.ndarray:
    y = y[(y >= 0).all(1)]
    w = np.ones((y.shape[1], 3), dtype=np.float32)
    for k in range(3):
        p = np.maximum((y > k).mean(0), 1e-4)
        w[:, k] = np.clip(np.sqrt((1 - p) / p), 1.0, cap)
    return w


def monotone_cumprobs(logits: torch.Tensor) -> torch.Tensor:
    """P(y > k) made non-increasing in k: (..., 3) -> (..., 3)."""
    s = torch.sigmoid(logits)
    p1 = s[..., 0]
    p2 = torch.minimum(s[..., 1], p1)
    p3 = torch.minimum(s[..., 2], p2)
    return torch.stack([p1, p2, p3], dim=-1)


class LossComputer:
    def __init__(self, loss_cfg: dict, y_a1_train: np.ndarray, y_a2_train: np.ndarray, device: torch.device) -> None:
        self.c = loss_cfg
        self.pw_a1 = torch.as_tensor(a1_pos_weight(y_a1_train), device=device) if loss_cfg["a1_pos_weight"] else None
        self.pw_a2 = (torch.as_tensor(a2_pos_weight(y_a2_train, float(loss_cfg["a2_pos_weight_cap"])), device=device)
                      if loss_cfg["a2_pos_weight"] else None)
        self.membership = torch.as_tensor(subscale_membership(), device=device)  # (3, 21)
        self.midpoints = torch.as_tensor(decision_midpoints(), device=device)  # (3,)

    def _smooth(self, t: torch.Tensor) -> torch.Tensor:
        ls = float(self.c["label_smoothing"])
        return t * (1 - ls) + 0.5 * ls if ls > 0 else t

    def a1(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        keep = (y >= 0).all(1)
        if not keep.any():
            return logits.sum() * 0.0
        return F.binary_cross_entropy_with_logits(logits[keep], self._smooth(y[keep].float()), pos_weight=self.pw_a1)

    def a2(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        keep = (y >= 0).all(1)
        if not keep.any():
            return logits.sum() * 0.0
        yk = y[keep].long()
        targets = (yk.unsqueeze(-1) > torch.arange(3, device=y.device)).float()
        return F.binary_cross_entropy_with_logits(logits[keep], self._smooth(targets), pos_weight=self.pw_a2)

    def subscale(self, a2_logits: torch.Tensor, y_a1: torch.Tensor, y_a2: torch.Tensor) -> torch.Tensor:
        keep = (y_a2 >= 0).all(1) & (y_a1 >= 0).all(1)
        if not keep.any():
            return a2_logits.sum() * 0.0
        expected = monotone_cumprobs(a2_logits[keep]).sum(-1)       # (B, 21) expected item score
        pred = 2.0 * expected @ self.membership.T                    # (B, 3) doubled 0-42 scale
        true = 2.0 * y_a2[keep].float() @ self.membership.T
        reg = F.smooth_l1_loss(pred / 42.0, true / 42.0)
        cls_logit = (pred - self.midpoints) / float(self.c["subscale_tau"])
        cls = F.binary_cross_entropy_with_logits(cls_logit, y_a1[keep].float())
        return reg + cls

    def __call__(self, out: dict, batch: dict) -> tuple[torch.Tensor, dict[str, float]]:
        c = self.c
        y1, y2 = batch["y_a1"], batch["y_a2"]
        parts = {
            "a1": self.a1(out["a1"], y1),
            "a2": self.a2(out["a2"], y2),
        }
        active = [float(c[f"{tag}_weight"]) * parts[tag] for tag in ("a1", "a2")
                  if float(c[f"{tag}_weight"]) > 0]
        if not active:
            raise ValueError("At least one task loss weight must be positive")
        total = sum(active)

        if float(c["session_aux_weight"]) > 0:
            valid = batch["session_valid"].reshape(-1)
            n_sess = batch["session_valid"].shape[1]
            if valid.any():
                sy1 = y1.repeat_interleave(n_sess, dim=0)[valid]
                sy2 = y2.repeat_interleave(n_sess, dim=0)[valid]
                parts["sess"] = self.a1(out["sess_a1"][valid], sy1) + self.a2(out["sess_a2"][valid], sy2)
                total = total + float(c["session_aux_weight"]) * parts["sess"]

        if float(c["subscale_weight"]) > 0:
            parts["subscale"] = self.subscale(out["a2"], y1, y2)
            total = total + float(c["subscale_weight"]) * parts["subscale"]

        return total, {k: float(v.detach()) for k, v in parts.items()}
