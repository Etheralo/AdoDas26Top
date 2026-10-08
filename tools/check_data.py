#!/usr/bin/env python3
"""Check the downloaded data before training.

    python tools/check_data.py --config configs/stage1.yaml

Reports manifest columns and sizes, sessions per participant, per-feature file
coverage for every split and session, feature dimensions (must be identical
across files), text coverage, and whether the A1 labels follow the DASS-21
cutoffs applied to the A2 items. Exits with code 1 on blocking problems.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.dass import A1_COLS, ITEM_COLS, SESSIONS, check_label_consistency  # noqa: E402
from data.dataset import build_participants, participant_labels, read_manifest  # noqa: E402
from data.feature_io import feature_dir, load_pooled, load_sequence, pooled_exists  # noqa: E402
from data.text_io import text_path  # noqa: E402
from training.config import check_required, load_config  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test_hidden"])
    ap.add_argument("--dim_samples", type=int, default=30, help="Files per feature whose dims are compared")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    check_required(cfg)
    f = cfg["features"]
    root = Path(cfg["paths"]["feature_root"])
    tags = {"ssl_embed": f["audio_ssl_tag"], "vision_ssl_embed": f["video_ssl_tag"]}
    groups = [("audio", n, False) for n in f["audio"]] + [("video", n, False) for n in f["video"]] \
        + [("audio", n, True) for n in f["audio_pooled"]]
    problems = []

    for split in args.splits:
        path = Path(cfg["paths"]["manifest_dir"]) / f"{split}.csv"
        if not path.exists():
            print(f"[{split}] manifest {path} not found")
            if split == "train" or (split == "val" and cfg["split"]["require_official_val"]):
                problems.append(f"missing manifest {path}")
            continue
        df = read_manifest(path)
        label_cols = [c for c in (*A1_COLS, *ITEM_COLS) if c in df.columns]
        print(f"\n[{split}] {len(df)} rows, columns: {list(df.columns)}")
        if split != "test_hidden" and df[label_cols].isna().any().any():
            problems.append(f"{split}: missing labels")
        if split != "test_hidden" and len(label_cols) < len(A1_COLS) + len(ITEM_COLS):
            problems.append(f"{split}: label columns missing ({len(label_cols)} of 24 present)")
        parts = build_participants(df, split)
        print(f"  participants {len(parts)}; sessions per participant: "
              f"{dict(sorted(Counter(len(p.sessions) for p in parts).items()))}")

        coverage = defaultdict(Counter)
        dims = defaultdict(Counter)
        for p in parts:
            for s in p.sessions:
                for modality, name, pooled in groups:
                    d = feature_dir(root, split, p.school, p.cls, p.pid, modality, name, s,
                                    None if pooled else tags.get(name))
                    ok = pooled_exists(d) if pooled else (d / "sequence.npz").exists()
                    coverage[f"{modality}/{name}"][s] += int(ok)
                    key = f"{modality}/{name}"
                    if ok and sum(dims[key].values()) < args.dim_samples:
                        try:
                            dim = len(load_pooled(d)) if pooled else load_sequence(d, name).features.shape[1]
                            dims[key][dim] += 1
                        except Exception as exc:
                            dims[key][f"error: {exc!r}"[:80]] += 1
                if cfg["paths"]["text_root"] and s in f["text_sessions"]:
                    ok = text_path(cfg["paths"]["text_root"], split, p.school, p.cls, p.pid, f["text_tag"], s).exists()
                    coverage["text"][s] += int(ok)

        n_sess = Counter(s for p in parts for s in p.sessions)
        print("  coverage (files found / sessions in manifest):")
        for key in [f"{m}/{n}" for m, n, _ in groups] + (["text"] if cfg["paths"]["text_root"] else []):
            cells = " ".join(f"{s}:{coverage[key][s]}/{n_sess[s]}" for s in SESSIONS if n_sess[s])
            total = sum(coverage[key].values())
            print(f"    {key:<28} {cells}   dims {dict(dims[key]) if key in dims else '-'}")
            if total == 0:
                problems.append(f"{split}: no files at all for {key}")
            if key in dims and any(not isinstance(k, int) for k in dims[key]):
                problems.append(f"{split}: unreadable feature samples for {key}")
            if key in dims and len([k for k in dims[key] if isinstance(k, int)]) > 1:
                problems.append(f"{split}: inconsistent dims for {key}: {dict(dims[key])}")

        if split != "test_hidden" and parts:
            y1, y2 = participant_labels(parts)
            cons = check_label_consistency(y1, y2)
            print(f"  A1 labels derived from items (DASS-21 cutoffs D>=10, A>=8, S>=15): {cons}")
            prev = np.where(y1 >= 0, y1, np.nan)
            print(f"  A1 prevalence D/A/S: {np.round(np.nanmean(prev, axis=0), 4).tolist()}")

    print()
    if problems:
        for pr in problems:
            print(f"PROBLEM: {pr}")
        return 1
    print("OK: no blocking problems found")
    return 0


if __name__ == "__main__":
    sys.exit(main())
