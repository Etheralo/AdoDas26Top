"""Post-hoc calibration, fitted on inner-dev only.

A1: the scorer thresholds probabilities at 0.5, so one logit bias per target is
grid-searched for F1. A2: one offset per item (added to all three threshold
logits) is grid-searched for QWK, separately for each decoding rule; the rule
with the best mean QWK (then lower MAE) is selected.
"""
from __future__ import annotations

import numpy as np

from .metrics import binary_f1, qwk

A2_DECODERS = ("count", "monotonic", "expectation")


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def decode_a2(logits: np.ndarray, method: str) -> np.ndarray:
    """(..., 3) cumulative logits -> integer scores 0..3."""
    s = sigmoid(logits)
    if method == "count":
        return (s > 0.5).sum(-1).astype(np.int64)
    p1 = s[..., 0]
    p2 = np.minimum(s[..., 1], p1)
    p3 = np.minimum(s[..., 2], p2)
    if method == "monotonic":
        probs = np.stack([1 - p1, p1 - p2, p2 - p3, p3], axis=-1)
        return probs.argmax(-1).astype(np.int64)
    if method == "expectation":
        return np.clip(np.round(p1 + p2 + p3), 0, 3).astype(np.int64)
    raise ValueError(f"Unknown decode method {method!r}")


def _grid(lo: float, hi: float, step: float) -> np.ndarray:
    g = np.round(np.arange(lo, hi + step / 2, step), 6)
    return g[np.argsort(np.abs(g), kind="stable")]  # ties resolve to the smallest |offset|


def calibrate_a1(logits: np.ndarray, labels: np.ndarray, lo=-4.0, hi=4.0, step=0.05) -> dict:
    keep = (labels >= 0).all(1)
    logits, labels = logits[keep], labels[keep]
    grid = _grid(lo, hi, step)
    biases, f1s = [], []
    for c in range(logits.shape[1]):
        best_b, best_f1 = 0.0, -1.0
        for b in grid:
            f1 = binary_f1(labels[:, c], sigmoid(logits[:, c] + b) >= 0.5)
            if f1 > best_f1:
                best_b, best_f1 = float(b), f1
        biases.append(best_b)
        f1s.append(best_f1)
    return {"biases": biases, "f1_per": f1s, "f1": float(np.mean(f1s))}


def apply_a1(logits: np.ndarray, calib: dict | None) -> np.ndarray:
    b = np.zeros(logits.shape[1]) if calib is None else np.asarray(calib["biases"])
    return sigmoid(logits + b[None, :])


def calibrate_a2_items(logits: np.ndarray, labels: np.ndarray, method: str,
                       lo=-2.0, hi=2.0, step=0.05) -> tuple[np.ndarray, np.ndarray]:
    """Per-item best offset and its QWK for one decoding rule."""
    grid = _grid(lo, hi, step)
    n_items = logits.shape[1]
    offsets = np.zeros(n_items)
    item_qwk = np.zeros(n_items)
    for j in range(n_items):
        best_b, best_q, best_mae = 0.0, -np.inf, np.inf
        yj = labels[:, j]
        for b in grid:
            pred = decode_a2(logits[:, j] + b, method)
            q = qwk(yj, pred)
            mae = float(np.abs(pred - yj).mean())
            if (q, -mae) > (best_q, -best_mae):
                best_b, best_q, best_mae = float(b), q, mae
        offsets[j], item_qwk[j] = best_b, best_q
    return offsets, item_qwk


def calibrate_a2(logits: np.ndarray, labels: np.ndarray, methods=A2_DECODERS) -> dict:
    keep = (labels >= 0).all(1)
    logits, labels = logits[keep], labels[keep].astype(np.int64)
    per_method = {}
    for m in methods:
        offsets, item_qwk = calibrate_a2_items(logits, labels, m)
        preds = decode_a2(logits + offsets[None, :, None], m)
        per_method[m] = {
            "offsets": offsets.tolist(),
            "item_qwk": item_qwk.tolist(),
            "qwk": float(item_qwk.mean()),
            "mae": float(np.abs(preds - labels).mean()),
        }
    best = max(per_method, key=lambda m: (per_method[m]["qwk"], -per_method[m]["mae"]))
    return {"method": best, "offsets": per_method[best]["offsets"], "qwk": per_method[best]["qwk"],
            "per_method": per_method}


def apply_a2(logits: np.ndarray, calib: dict | None, default_method: str = "expectation") -> np.ndarray:
    if calib is None:
        return decode_a2(logits, default_method)
    offsets = np.asarray(calib["offsets"])
    return decode_a2(logits + offsets[None, :, None], calib["method"])
