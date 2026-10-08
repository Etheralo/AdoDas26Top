"""Reader for pre-extracted ASR text token embeddings.

``tools/extract_text_embeddings.py`` writes them in the same layout as the
feature tree: <text_root>/<split>/<school>/<class>/<pid>/text/<tag>/<session>/sequence.npz
with ``features`` (L, D) and optional ``valid_mask`` (L,).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def text_path(root: str | Path, split: str, school: str, cls: str, pid: str, tag: str, session: str) -> Path:
    return Path(root) / split / school / cls / pid / "text" / tag / session / "sequence.npz"


def load_text(
    root: str | Path,
    split: str,
    school: str,
    cls: str,
    pid: str,
    tag: str,
    session: str,
    max_tokens: int,
) -> np.ndarray | None:
    path = text_path(root, split, school, cls, pid, tag, session)
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as data:
        feats = np.asarray(data["features"], dtype=np.float32)
        if "valid_mask" in data.files:
            feats = feats[np.asarray(data["valid_mask"]).astype(bool)]
    if feats.ndim != 2 or len(feats) == 0:
        return None
    feats = np.nan_to_num(feats)
    return feats[:max_tokens] if max_tokens else feats
