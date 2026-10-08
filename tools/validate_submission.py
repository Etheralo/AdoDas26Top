#!/usr/bin/env python3
"""Check a submission file against the AdoDAS rules (paper Sec. 2.3).

The identifier set must equal the reference set exactly (no duplicates, no
missing or extra ids); A1 needs finite probabilities p_D/p_A/p_S in [0, 1];
A2 needs integers 0..3 in d01..d21.

    python tools/validate_submission.py --task a1 --submission sub.csv --manifest test_hidden.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

A1_OUT_COLS = ["p_D", "p_A", "p_S"]
A2_OUT_COLS = [f"d{i:02d}" for i in range(1, 22)]


def expected_ids_from_manifest(path: str | Path) -> set[str]:
    df = pd.read_csv(path, dtype=str)
    return set((df["anon_school"] + "_" + df["anon_class"] + "_" + df["anon_pid"]).tolist())


def validate_submission(df: pd.DataFrame, task: str, expected_ids: set[str] | None) -> list[str]:
    errors: list[str] = []
    cols = A1_OUT_COLS if task == "a1" else A2_OUT_COLS
    missing_cols = [c for c in ["file_id", *cols] if c not in df.columns]
    if missing_cols:
        return [f"missing columns: {missing_cols}"]

    ids = df["file_id"].astype(str)
    dup = ids[ids.duplicated()].unique().tolist()
    if dup:
        errors.append(f"{len(dup)} duplicated ids, e.g. {dup[:3]}")
    if expected_ids is not None:
        got = set(ids)
        if got - expected_ids:
            errors.append(f"{len(got - expected_ids)} unexpected ids, e.g. {sorted(got - expected_ids)[:3]}")
        if expected_ids - got:
            errors.append(f"{len(expected_ids - got)} missing ids, e.g. {sorted(expected_ids - got)[:3]}")

    values = df[cols].to_numpy()
    if task == "a1":
        v = pd.to_numeric(df[cols].stack(), errors="coerce").to_numpy(dtype=np.float64)
        if not np.isfinite(v).all():
            errors.append("non-finite or non-numeric probabilities")
        elif (v < 0).any() or (v > 1).any():
            errors.append("probabilities outside [0, 1]")
    else:
        v = pd.to_numeric(df[cols].stack(), errors="coerce").to_numpy(dtype=np.float64)
        if not np.isfinite(v).all() or not np.all(v == np.round(v)):
            errors.append("A2 responses must be integers")
        elif (v < 0).any() or (v > 3).any():
            errors.append("A2 responses outside 0..3")
    if values.shape[0] == 0:
        errors.append("empty submission")
    return errors


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["a1", "a2"])
    ap.add_argument("--submission", required=True)
    ap.add_argument("--manifest", default=None, help="Manifest CSV of the split (gives the reference id set)")
    args = ap.parse_args()
    df = pd.read_csv(args.submission, dtype={"file_id": str})
    expected = expected_ids_from_manifest(args.manifest) if args.manifest else None
    errors = validate_submission(df, args.task, expected)
    if errors:
        for e in errors:
            print(f"ERROR: {e}")
        return 1
    print(f"OK: {len(df)} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
