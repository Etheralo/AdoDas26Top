#!/usr/bin/env python3
"""Offline test scoring only after validation choices are frozen.

Never import this tool from train/infer/ensemble. Do not tune on these scores.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.dass import A1_COLS, ITEM_COLS
from tools.validate_submission import validate_submission, A1_OUT_COLS, A2_OUT_COLS
from training.metrics import a1_metrics, a2_metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["a1", "a2"], required=True)
    ap.add_argument("--submission", required=True)
    ap.add_argument("--reference_root", default="/root/autodl-tmp/dataset-ai/references")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    ref = pd.read_csv(Path(args.reference_root) / f"{args.task}_reference_data.csv")
    ref["file_id"] = ref.anon_school + "_" + ref.anon_class + "_" + ref.anon_pid
    pred = pd.read_csv(args.submission)
    errors = validate_submission(pred, args.task, set(ref.file_id))
    if errors: raise ValueError(errors)
    pred = pred.set_index("file_id").loc[ref.file_id]
    output_cols, label_cols = (A1_OUT_COLS, A1_COLS) if args.task == "a1" else (A2_OUT_COLS, ITEM_COLS)
    values, labels = pred[list(output_cols)].to_numpy(), ref[list(label_cols)].to_numpy()
    metrics = a1_metrics(values, labels) if args.task == "a1" else a2_metrics(values, labels)
    result = {"split": "test_hidden_offline", "task": args.task, "participants": len(ref),
              "metrics": metrics, "submission": args.submission,
              "note": "Offline reference evaluation; not an organizer leaderboard submission. Do not select or calibrate using test scores."}
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))

if __name__ == "__main__": main()
