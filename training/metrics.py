"""Evaluation metrics, following the AdoDAS scorer (paper Sec. 2.3).

A1: binary F1 per target at a fixed 0.5 threshold, averaged; macro AUROC is the
tie-breaker. A2: mean item-level quadratic weighted kappa; MAE is the tie-breaker.
"""
from __future__ import annotations

import numpy as np


def _rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[order] = np.arange(1, len(x) + 1)
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks)
    return (sums / counts)[inv]


def binary_f1(y: np.ndarray, pred: np.ndarray) -> float:
    y = np.asarray(y).astype(bool)
    pred = np.asarray(pred).astype(bool)
    tp = np.sum(y & pred)
    denom = 2 * tp + np.sum(~y & pred) + np.sum(y & ~pred)
    return float(2 * tp / denom) if denom > 0 else 0.0


def auroc(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y).astype(bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _rankdata(np.asarray(score, dtype=np.float64))
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def qwk(y: np.ndarray, pred: np.ndarray, n_classes: int = 4) -> float:
    y = np.asarray(y).astype(np.int64)
    pred = np.asarray(pred).astype(np.int64)
    K = n_classes
    O = np.bincount(y * K + pred, minlength=K * K).reshape(K, K).astype(np.float64)
    idx = np.arange(K)
    W = (idx[:, None] - idx[None, :]) ** 2 / (K - 1) ** 2
    E = np.outer(O.sum(1), O.sum(0)) / max(len(y), 1)
    den = float((W * E).sum())
    if den == 0:
        return 0.0
    return float(1.0 - (W * O).sum() / den)


def a1_metrics(probs: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> dict:
    keep = (labels >= 0).all(1)
    probs, labels = probs[keep], labels[keep]
    preds = probs >= threshold
    f1s = [binary_f1(labels[:, c], preds[:, c]) for c in range(3)]
    aucs = [auroc(labels[:, c], probs[:, c]) for c in range(3)]
    return {
        "f1": float(np.mean(f1s)),
        "f1_per": f1s,
        "auroc": float(np.nanmean(aucs)) if not np.all(np.isnan(aucs)) else float("nan"),
        "n": int(keep.sum()),
    }


def a2_metrics(preds: np.ndarray, labels: np.ndarray) -> dict:
    keep = (labels >= 0).all(1)
    preds, labels = preds[keep].astype(np.int64), labels[keep].astype(np.int64)
    item_qwk = [qwk(labels[:, j], preds[:, j]) for j in range(labels.shape[1])]
    return {
        "qwk": float(np.mean(item_qwk)),
        "qwk_per": item_qwk,
        "mae": float(np.abs(preds - labels).mean()) if len(labels) else float("nan"),
        "n": int(keep.sum()),
    }
