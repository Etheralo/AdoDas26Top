"""DASS-21 structure shared by Track A1 and Track A2.

Item membership and severity cutoffs follow the standard DASS-21 scoring
(subscale sum x 2 on the 0-42 scale). Track A1 treats "Normal" as negative and
"mild or above" as positive (AdoDAS paper, Sec. 2.1).
"""
from __future__ import annotations

import numpy as np

SESSIONS = ("A01", "B01", "B02", "B03")
ITEM_COLS = tuple(f"d{i:02d}" for i in range(1, 22))
A1_COLS = ("y_D", "y_A", "y_S")
SUBSCALES = ("D", "A", "S")
SUBSCALE_ITEMS = {
    "D": (3, 5, 10, 13, 16, 17, 21),
    "A": (2, 4, 7, 9, 15, 19, 20),
    "S": (1, 6, 8, 11, 12, 14, 18),
}
# Doubled subscale scores at or above these values are "mild or above".
POSITIVE_CUTOFFS = {"D": 10, "A": 8, "S": 15}


def item_to_subscale() -> np.ndarray:
    """(21,) index of the subscale (0=D, 1=A, 2=S) each item belongs to."""
    out = np.zeros(len(ITEM_COLS), dtype=np.int64)
    for s_idx, name in enumerate(SUBSCALES):
        for item in SUBSCALE_ITEMS[name]:
            out[item - 1] = s_idx
    return out


def subscale_membership() -> np.ndarray:
    """(3, 21) 0/1 matrix mapping items to subscales."""
    m = np.zeros((len(SUBSCALES), len(ITEM_COLS)), dtype=np.float32)
    for s_idx, name in enumerate(SUBSCALES):
        for item in SUBSCALE_ITEMS[name]:
            m[s_idx, item - 1] = 1.0
    return m


def cutoffs() -> np.ndarray:
    return np.array([POSITIVE_CUTOFFS[s] for s in SUBSCALES], dtype=np.float32)


def decision_midpoints() -> np.ndarray:
    """Midpoint between the highest negative and lowest positive doubled score.

    Doubled scores are even, so e.g. Stress (cutoff 15) separates 14 from 16.
    """
    mids = []
    for c in cutoffs():
        pos = c if c % 2 == 0 else c + 1
        mids.append(pos - 1.0)
    return np.array(mids, dtype=np.float32)


def subscale_scores(items: np.ndarray) -> np.ndarray:
    """items (N, 21) in 0..3 -> doubled subscale scores (N, 3)."""
    return 2.0 * np.asarray(items, dtype=np.float32) @ subscale_membership().T


def derive_a1(items: np.ndarray) -> np.ndarray:
    return (subscale_scores(items) >= cutoffs()).astype(np.int64)


def check_label_consistency(y_a1: np.ndarray, y_a2: np.ndarray) -> dict:
    """How often the released A1 labels equal the labels derived from A2 items."""
    y_a1 = np.asarray(y_a1)
    y_a2 = np.asarray(y_a2)
    valid = (y_a1 >= 0).all(axis=1) & (y_a2 >= 0).all(axis=1)
    n = int(valid.sum())
    if n == 0:
        return {"n": 0, "agreement": {}, "all_agree": float("nan")}
    derived = derive_a1(y_a2[valid])
    actual = y_a1[valid].astype(np.int64)
    agree = {s: float((derived[:, i] == actual[:, i]).mean()) for i, s in enumerate(SUBSCALES)}
    return {"n": n, "agreement": agree, "all_agree": float((derived == actual).all(axis=1).mean())}
