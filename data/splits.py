"""Train / inner-dev / val / test participant sets.

Two protocols: inner_dev preserves a held-out official validation report;
official_val trains on all 4,200 training participants and uses the official
600-person validation split for selection, calibration and item experts. Test
reference labels are accepted only by a separate offline scoring tool.
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .dataset import Participant, build_participants, read_manifest

log = logging.getLogger(__name__)

KEY_COLS = ["anon_school", "anon_class", "anon_pid"]


def development_split(cfg: dict) -> str:
    source = cfg["split"].get("development_source", "inner_dev")
    if source not in ("inner_dev", "official_val"):
        raise ValueError("split.development_source must be inner_dev or official_val")
    return "val" if source == "official_val" else "inner_dev"


def default_split_file(cfg: dict) -> Path:
    s = cfg["split"]
    return Path(cfg["paths"]["output_dir"]) / "splits" / f"inner_dev_seed{s['seed']}_r{s['inner_dev_ratio']}.csv"


def make_inner_dev(participants: list[Participant], ratio: float, seed: int) -> pd.DataFrame:
    """Stratify by school and the A1 label pattern so prevalence is preserved."""
    if not 0 < ratio < 1:
        raise ValueError("split.inner_dev_ratio must lie strictly between 0 and 1")
    rng = np.random.default_rng(seed)
    strata: dict[str, list[Participant]] = {}
    for p in participants:
        key = p.school + "|" + "".join(str(int(v)) for v in p.y_a1)
        strata.setdefault(key, []).append(p)
    # Largest-remainder allocation keeps the requested overall size even when
    # many school/label strata are too small for independent rounding.
    target = max(1, min(len(participants) - 1, int(round(ratio * len(participants)))))
    keys = sorted(strata)
    allocation = {key: int(np.floor(ratio * len(strata[key]))) for key in keys}
    tie = {key: rng.random() for key in keys}
    order = sorted(keys, key=lambda k: (-(ratio * len(strata[k]) - allocation[k]), tie[k]))
    for key in order[:target - sum(allocation.values())]:
        allocation[key] += 1
    chosen = []
    for key in keys:
        members = strata[key]
        order = rng.permutation(len(members))
        chosen.extend(members[i] for i in order[:allocation[key]])
    rows = [{"anon_school": p.school, "anon_class": p.cls, "anon_pid": p.pid} for p in chosen]
    return pd.DataFrame(rows, columns=KEY_COLS).sort_values(KEY_COLS).reset_index(drop=True)


def load_or_create_inner_dev(participants: list[Participant], path: Path, ratio: float, seed: int) -> pd.DataFrame:
    path = Path(path)
    if path.exists():
        df = pd.read_csv(path, dtype=str)
        log.info(f"Using inner-dev split {path} ({len(df)} participants)")
        return df
    df = make_inner_dev(participants, ratio, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    log.info(f"Created inner-dev split {path} ({len(df)} participants)")
    return df


def file_sha1(path: Path) -> str:
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()


def load_participant_sets(
    cfg: dict,
    split_file: Path | None = None,
    include_test: bool = False,
) -> tuple[dict[str, list[Participant]], Path]:
    """Returns {"train", "inner_dev", "val"[, "test_hidden"]} and the split file used."""
    manifest_dir = Path(cfg["paths"]["manifest_dir"])
    train_all = build_participants(read_manifest(manifest_dir / "train.csv"), "train")
    val_path = manifest_dir / "val.csv"
    if val_path.exists():
        val = build_participants(read_manifest(val_path), "val")
    elif cfg["split"]["require_official_val"] or cfg["split"]["final"]:
        raise FileNotFoundError(f"Official validation manifest missing: {val_path}")
    else:
        val = []
        log.warning("Official val unavailable: train/inner-dev only; no official validation score will be reported")

    if set(p.file_id for p in train_all) & set(p.file_id for p in val):
        raise ValueError("Official train and val contain overlapping participants")
    if development_split(cfg) == "val":
        if cfg["split"]["final"]:
            raise ValueError("Official validation selection cannot include val in training")
        if not val:
            raise ValueError("Official validation selection requires a nonempty val.csv")
        # The returned manifest records selection IDs; no train participants are withheld.
        split_file = Path(cfg["paths"]["output_dir"]) / "splits" / "official_val_participants.csv"
        split_file.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([dict(zip(KEY_COLS, (p.school, p.cls, p.pid))) for p in val]).to_csv(split_file, index=False)
        sets = {"train": train_all, "inner_dev": [], "val": val}
    else:
        sets, split_file = _inner_sets(cfg, train_all, val, split_file)

    test_path = manifest_dir / "test_hidden.csv"
    if include_test and test_path.exists():
        test = build_participants(read_manifest(test_path), "test_hidden")
        if any((p.y_a1 >= 0).any() or (p.y_a2 >= 0).any() for p in test):
            raise ValueError("test_hidden.csv must contain identifiers only; reference labels belong in offline evaluation")
        if {p.file_id for p in test} & {p.file_id for p in train_all + val}:
            raise ValueError("Test participants overlap train/val")
        sets["test_hidden"] = test
    limit = int(cfg["debug"]["limit_participants"])
    if limit > 0:
        sets = {k: v[: (limit if k == "train" else max(1, limit // 2))] for k, v in sets.items()}
    return sets, split_file


def _inner_sets(cfg, train_all, val, split_file):
    split_file = Path(split_file) if split_file else default_split_file(cfg)
    dev_df = load_or_create_inner_dev(
        train_all, split_file, float(cfg["split"]["inner_dev_ratio"]), int(cfg["split"]["seed"])
    )
    dev_keys = set(map(tuple, dev_df[KEY_COLS].astype(str).values.tolist()))
    inner_dev = [p for p in train_all if (p.school, p.cls, p.pid) in dev_keys]
    train = [p for p in train_all if (p.school, p.cls, p.pid) not in dev_keys]
    if len(inner_dev) != len(dev_keys):
        raise ValueError(f"{split_file} lists participants that are not in train.csv")

    if not train or not inner_dev:
        raise ValueError("Training and inner-dev must both contain participants; adjust the split ratio")
    if set(p.file_id for p in train_all) & set(p.file_id for p in val):
        raise ValueError("Official train and val contain overlapping participants")
    if cfg["split"]["final"]:
        train = train + val

    sets = {"train": train, "inner_dev": inner_dev, "val": val}
    return sets, split_file
