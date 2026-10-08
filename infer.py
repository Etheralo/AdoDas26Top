#!/usr/bin/env python3
"""Export predictions of a trained run, calibrate on inner-dev, report val and
write submission files.

    python infer.py --run_dir outputs/runs/<run> [--set paths.feature_root=/new/path]

Outputs (inside the run directory):
  predictions/<split>_<a1|a2>.npz   raw logits from best_a1.pt / best_a2.pt
  calibration.json                  A1 biases and A2 offsets + decode rule (fitted on inner-dev)
  report.json                       inner-dev (in-sample) and official val (held-out) metrics
  submissions/submission_<a1|a2>_<split>.csv
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from data.dataset import ParticipantDataset
from data.splits import development_split, load_participant_sets
from models.model import build_model
from tools.validate_submission import A1_OUT_COLS, A2_OUT_COLS, expected_ids_from_manifest, validate_submission
from training.calibration import apply_a1, apply_a2, calibrate_a1, calibrate_a2
from training.config import deep_merge, parse_overrides
from training.metrics import a1_metrics, a2_metrics
from training.trainer import load_checkpoint, make_loader, predict
from training.utils import amp_enabled, resolve_device, save_json, setup_logging

log = logging.getLogger("adodas_sota")

SPLITS = ("inner_dev", "val", "test_hidden")
TAGS = ("a1", "a2")


def export_predictions(run_dir: Path, cfg: dict, sets: dict, device: torch.device, splits=SPLITS) -> None:
    for tag in TAGS:
        ckpt = load_checkpoint(run_dir / "checkpoints" / f"best_{tag}.pt")
        ccfg = ckpt["config"]
        data_cfg = {**ccfg, "paths": cfg["paths"]}  # allow moving the data between machines
        model = build_model(ccfg["model"], ckpt["dims"])
        model.load_state_dict(ckpt["model"])
        model.to(device)
        amp = amp_enabled(bool(ccfg["train"]["amp"]), device)
        for split in splits:
            if not sets.get(split):
                continue
            ds = ParticipantDataset(sets[split], data_cfg, train=False, dims=ckpt["dims"])
            loader = make_loader(ds, int(cfg["train"].get("eval_batch_size", 2)), False,
                                 int(cfg["train"]["num_workers"]), int(ccfg["train"]["seed"]),
                                 prefetch_factor=int(cfg["train"].get("prefetch_factor", 1)),
                                 worker_batch_size=int(cfg["train"].get("worker_batch_size", 0)))
            pred = predict(model, loader, device, amp, f"{tag}:{split}")
            np.savez(run_dir / "predictions" / f"{split}_{tag}.npz", **pred)
            log.info(f"Saved predictions/{split}_{tag}.npz ({len(pred['file_ids'])} participants, epoch {ckpt['epoch']})")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


def load_predictions(run_dir: Path, split: str, tag: str) -> dict | None:
    path = Path(run_dir) / "predictions" / f"{split}_{tag}.npz"
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as d:
        return {k: d[k] for k in d.files}


def write_submissions(out_dir: Path, split: str, file_ids: np.ndarray, a1_probs: np.ndarray | None,
                      a2_scores: np.ndarray | None, manifest_path: Path | None, strict: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    expected = expected_ids_from_manifest(manifest_path) if manifest_path and manifest_path.exists() else None
    for task, values, cols in (("a1", a1_probs, A1_OUT_COLS), ("a2", a2_scores, A2_OUT_COLS)):
        if values is None:
            continue
        df = pd.DataFrame(values, columns=cols)
        if task == "a2":
            df = df.astype(int)
        df.insert(0, "file_id", [str(x) for x in file_ids])
        path = out_dir / f"submission_{task}_{split}.csv"
        df.to_csv(path, index=False)
        errors = validate_submission(df, task, expected)
        if errors:
            msg = f"{path.name} failed validation: {errors}"
            if strict:
                raise ValueError(msg)
            log.warning(msg)
        else:
            log.info(f"Wrote {path} ({len(df)} rows, validated)")


def finalize_run(run_dir: Path, cfg: dict) -> dict:
    dev_key = development_split(cfg)
    dev1 = load_predictions(run_dir, dev_key, "a1")
    dev2 = load_predictions(run_dir, dev_key, "a2")
    cal1 = calibrate_a1(dev1["a1_logits"], dev1["y_a1"])
    cal2 = calibrate_a2(dev2["a2_logits"], dev2["y_a2"])
    save_json({"a1": cal1, "a2": cal2}, run_dir / "calibration.json")

    report = {
        f"{dev_key}_selection_in_sample": {
            "a1": a1_metrics(apply_a1(dev1["a1_logits"], cal1), dev1["y_a1"]),
            "a2": a2_metrics(apply_a2(dev2["a2_logits"], cal2), dev2["y_a2"]),
        },
        "calibration_split": dev_key,
        "note": "Selection/calibration metrics are in-sample. Official val is held out only with inner_dev selection. Test reference labels are never loaded here.",
    }
    strict = int(cfg["debug"]["limit_participants"]) == 0
    manifest_dir = Path(cfg["paths"]["manifest_dir"])
    for split in ("val", "test_hidden"):
        p1, p2 = load_predictions(run_dir, split, "a1"), load_predictions(run_dir, split, "a2")
        if p1 is None or p2 is None:
            continue
        if not np.array_equal(p1["file_ids"], p2["file_ids"]):
            raise ValueError(f"{split}: a1/a2 prediction files list different participants")
        probs = apply_a1(p1["a1_logits"], cal1)
        scores = apply_a2(p2["a2_logits"], cal2)
        if split == "val" and (p1["y_a1"] >= 0).all():
            key = "val_used_for_selection" if dev_key == "val" else "val_in_training" if cfg["split"]["final"] else "val_held_out"
            report[key] = {
                "a1": a1_metrics(probs, p1["y_a1"]),
                "a1_uncalibrated": a1_metrics(apply_a1(p1["a1_logits"], None), p1["y_a1"]),
                "a2": a2_metrics(scores, p2["y_a2"]),
                "a2_uncalibrated": a2_metrics(apply_a2(p2["a2_logits"], None), p2["y_a2"]),
            }
        write_submissions(run_dir / "submissions", split, p1["file_ids"], probs, scores,
                          manifest_dir / f"{split}.csv", strict)
    save_json(report, run_dir / "report.json")
    for key in ("val_held_out", "val_in_training", "val_used_for_selection"):
        if key in report:
            r = report[key]
            log.info(f"{key}: A1 mean F1 {r['a1']['f1']:.4f} (AUROC {r['a1']['auroc']:.4f}) | "
                     f"A2 mean QWK {r['a2']['qwk']:.4f} (MAE {r['a2']['mae']:.4f})")
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--set", nargs="*", default=[], help="Overrides such as paths.feature_root=/data")
    ap.add_argument("--splits", nargs="+", default=list(SPLITS))
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    with open(run_dir / "config_used.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg = deep_merge(cfg, parse_overrides(args.set))
    setup_logging(run_dir / "logs" / "infer.log")
    sets, _ = load_participant_sets(cfg, split_file=run_dir / "inner_dev.csv", include_test=True)
    if development_split(cfg) not in args.splits:
        raise ValueError("Development split predictions are required for calibration")
    export_predictions(run_dir, cfg, sets, resolve_device(), args.splits)
    finalize_run(run_dir, cfg)


if __name__ == "__main__":
    main()
