#!/usr/bin/env python3
"""Audit official split IDs/labels and write an ID-only test manifest.

Reference label values are never copied into training manifests. No transcripts
or individual labels are printed. Existing manifests are never overwritten.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.dataset import read_manifest, build_participants, participant_labels
from data.dass import A1_COLS, ITEM_COLS, SESSIONS, check_label_consistency
from data.splits import KEY_COLS


def keys(df):
    return set(map(tuple, df[KEY_COLS].astype(str).itertuples(index=False, name=None)))


def prepare(root: Path):
    report = {"splits": {}, "problems": [], "test_manifest_labels": False}
    sets = {}
    for split, count in (("train", 4200), ("val", 600)):
        df = read_manifest(root / f"{split}.csv")
        if any(c not in df or df[c].isna().any() for c in (*A1_COLS, *ITEM_COLS)):
            raise ValueError(f"{split}: all 24 label columns must be complete")
        sets[split] = keys(df)
        parts = build_participants(df, split)
        y1, y2 = participant_labels(parts)
        if len(parts) != count or any(len(p.sessions) != 4 for p in parts):
            raise ValueError(f"{split}: expected {count} participants, each with 4 sessions")
        report["splits"][split] = {"participants": len(parts), "sessions": len(df),
            "a1_prevalence": y1.mean(0).tolist(), "label_consistency": check_label_consistency(y1, y2)}
    refs = []
    for task, cols in (("a1", A1_COLS), ("a2", ITEM_COLS)):
        df = pd.read_csv(root / "references" / f"{task}_reference_data.csv", dtype={k: str for k in KEY_COLS})
        if df[KEY_COLS].isna().any().any() or df.duplicated(KEY_COLS).any() or len(df) != 1200:
            raise ValueError(f"{task}: invalid reference identifiers/count")
        for col in cols:
            if col not in df or not df[col].isin((0, 1) if task == "a1" else (0, 1, 2, 3)).all():
                raise ValueError(f"{task}: invalid reference label column")
        refs.append(df)
    if keys(refs[0]) != keys(refs[1]):
        raise ValueError("A1/A2 reference participant sets differ")
    sets["test_hidden"] = keys(refs[0])
    for a, b in (("train", "val"), ("train", "test_hidden"), ("val", "test_hidden")):
        if sets[a] & sets[b]:
            raise ValueError(f"Participants overlap between {a} and {b}")
    rows = [{**dict(zip(KEY_COLS, k)), "session": s, "split": "test_hidden"}
            for k in sorted(sets["test_hidden"]) for s in SESSIONS]
    candidate = pd.DataFrame(rows, columns=["split", *KEY_COLS, "session"])
    # Check feature participant set matches the reference ID set exactly.
    for split, ids in sets.items():
        tree = root / "features" / split
        actual = {tuple(p.relative_to(tree).parts) for p in tree.glob("SCH_*/CLS_*/P*") if p.is_dir()}
        if actual != ids:
            raise ValueError(f"{split}: feature/manifest participant set mismatch: missing={len(ids-actual)}, extra={len(actual-ids)}")
        missing = empty = total = 0
        for school, cls, pid in ids:
            for session in ("B01", "B02", "B03"):
                d = root / "transcripts" / split / school / cls / pid / session
                path = d / "normalized_transcript.txt"
                if not path.exists(): path = d / "clean_transcript.txt"
                total += 1
                if not path.exists(): missing += 1
                elif not path.read_text(encoding="utf-8").strip(): empty += 1
        report["splits"].setdefault(split, {"participants": len(ids), "sessions": len(ids)*4})
        report["splits"][split]["transcripts"] = {"expected": total, "missing": missing, "empty": empty}
        if missing: report["problems"].append(f"{split}: {missing} transcripts missing")
    path = root / "test_hidden.csv"
    if path.exists():
        existing = read_manifest(path)
        if any(c in existing for c in (*A1_COLS, *ITEM_COLS)):
            raise ValueError("Existing test manifest contains labels; remove them manually before training")
        if keys(existing) != sets["test_hidden"] or len(existing) != 4800:
            raise ValueError("Existing test manifest differs; refusing overwrite")
    elif not report["problems"]:
        candidate.to_csv(path, index=False)
    report["test_manifest"] = str(path)
    report["participant_disjoint"] = True
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/root/autodl-tmp/dataset-ai")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    report = prepare(Path(args.root))
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return bool(report["problems"])

if __name__ == "__main__": sys.exit(main())
