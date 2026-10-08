"""Readers for the released feature tree.

Layout: <root>/<split>/<school>/<class>/<pid>/<modality>/<feature>/[<model_tag>/]<session>/
Sequence groups store ``sequence.npz`` with ``features`` (or ``mel_features`` +
``mfcc_features`` for mel_mfcc), ``timestamps_ms`` and ``valid_mask``. Pooled
groups store ``pooled.parquet`` and/or ``pooled.json``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


class FeatureDimError(ValueError):
    """A loaded feature does not have the dimension the model was built for."""


@dataclass
class Sequence:
    features: np.ndarray       # (T, D) float32
    timestamps_ms: np.ndarray  # (T,) float64, sorted
    valid: np.ndarray          # (T,) bool


def feature_dir(
    root: str | Path,
    split: str,
    school: str,
    cls: str,
    pid: str,
    modality: str,
    feature: str,
    session: str,
    tag: str | None = None,
) -> Path:
    path = Path(root) / split / school / cls / pid / modality / feature
    if tag:
        path = path / tag
    return path / session


def load_sequence(directory: Path, feature: str) -> Sequence:
    path = Path(directory) / "sequence.npz"
    with np.load(path, allow_pickle=False) as data:
        if feature == "mel_mfcc":
            feats = np.concatenate(
                [np.asarray(data["mel_features"], np.float32), np.asarray(data["mfcc_features"], np.float32)],
                axis=-1,
            )
        else:
            feats = np.asarray(data["features"], dtype=np.float32)
        ts = np.asarray(data["timestamps_ms"], dtype=np.float64)
        if "valid_mask" in data.files:
            valid = np.asarray(data["valid_mask"]).astype(bool)
        else:
            valid = np.ones(len(ts), dtype=bool)

    if feats.ndim == 1:
        feats = feats[:, None]
    if not (feats.shape[0] == ts.shape[0] == valid.shape[0]):
        raise ValueError(
            f"Length mismatch in {path}: features {feats.shape[0]}, "
            f"timestamps {ts.shape[0]}, valid_mask {valid.shape[0]}"
        )

    finite = np.isfinite(feats).all(axis=1)
    if not finite.all():
        feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
        valid = valid & finite

    if ts.size > 1 and np.any(np.diff(ts) < 0):
        order = np.argsort(ts, kind="stable")
        feats, ts, valid = feats[order], ts[order], valid[order]
    return Sequence(features=feats, timestamps_ms=ts, valid=valid)


def load_pooled(directory: Path) -> np.ndarray | None:
    """First row of pooled.parquet (numeric columns), falling back to pooled.json."""
    directory = Path(directory)
    parquet = directory / "pooled.parquet"
    if parquet.exists():
        import pandas as pd

        df = pd.read_parquet(parquet)
        numeric = df.select_dtypes(include=[np.number])
        if len(numeric) > 0 and numeric.shape[1] > 0:
            return np.nan_to_num(numeric.iloc[0].to_numpy(dtype=np.float32))

    js = directory / "pooled.json"
    if js.exists():
        with open(js, encoding="utf-8") as f:
            meta = json.load(f)
        feats = meta.get("features")
        if isinstance(feats, dict):
            values = list(feats.values())
        elif isinstance(feats, list):
            values = feats
        else:
            return None
        return np.nan_to_num(np.asarray(values, dtype=np.float32))
    return None


def pooled_exists(directory: Path) -> bool:
    directory = Path(directory)
    return (directory / "pooled.parquet").exists() or (directory / "pooled.json").exists()
