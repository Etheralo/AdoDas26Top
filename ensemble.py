#!/usr/bin/env python3
"""Ensemble trained runs (HEALTH FIRST "validation-selected item experts").

    python ensemble.py --config configs/ensemble.yaml

A1: mean of member logits (each member's best_a1.pt), then one bias per target
fitted on inner-dev. A2: candidates are every member (best_a2.pt) plus the mean
of all members; for each item the (candidate, decode rule, offset) with the best
inner-dev QWK is chosen, unless it beats the mean-ensemble default by less than
``a2.min_gain``. All members must share the same inner-dev split.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import yaml

from infer import load_predictions, write_submissions
from data.splits import development_split
from training.calibration import A2_DECODERS, apply_a1, calibrate_a1, calibrate_a2_items, decode_a2
from training.metrics import a1_metrics, a2_metrics
from training.utils import make_run_dir, save_json, setup_logging

log = logging.getLogger("adodas_sota")

SPLITS = ("inner_dev", "val", "test_hidden")


def load_member(run_dir: Path, ref_ids: dict[str, np.ndarray] | None, dev_key: str = "inner_dev") -> dict:
    out = {}
    for split in SPLITS:
        for tag in ("a1", "a2"):
            p = load_predictions(run_dir, split, tag)
            if p is None:
                if split == dev_key:
                    raise FileNotFoundError(f"{run_dir}: missing predictions/{split}_{tag}.npz (run infer.py)")
                continue
            if ref_ids is not None and split in ref_ids:
                pos = {fid: i for i, fid in enumerate(p["file_ids"])}
                if set(pos) != set(ref_ids[split]):
                    raise ValueError(f"{run_dir}: {split} participants differ from the first member "
                                     f"(members must share the same inner-dev split)")
                order = np.array([pos[fid] for fid in ref_ids[split]])
                p = {k: v[order] for k, v in p.items()}
            out[(split, tag)] = p
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    primary_task = cfg.get("primary_task", "both")
    if primary_task not in ("a1", "a2", "both"):
        raise ValueError("primary_task must be a1, a2, or both")
    members = [Path(m) for m in cfg["members"]]
    if not members:
        raise ValueError("ensemble config lists no members")
    min_gain = float(cfg.get("a2", {}).get("min_gain", 0.005))
    expert_selection = bool(cfg.get("a2", {}).get("expert_selection", True))
    run_dir = make_run_dir(cfg.get("output_dir", "outputs"), cfg.get("name", "ensemble"))
    setup_logging(run_dir / "logs" / "ensemble.log")
    log.info(f"Members: {[str(m) for m in members]}")

    with open(members[0] / "config_used.yaml", encoding="utf-8") as f:
        member_cfg = yaml.safe_load(f)
    dev_key = development_split(member_cfg)
    for member in members:
        with open(member / "config_used.yaml", encoding="utf-8") as f:
            ccfg = yaml.safe_load(f)
        if development_split(ccfg) != dev_key or ccfg["split"]["final"]:
            raise ValueError("Members must share a selection protocol and cannot train on validation")
    first = load_member(members[0], None, dev_key)
    ref_ids = {split: first[(split, "a1")]["file_ids"] for split in SPLITS if (split, "a1") in first}
    data = [load_member(m, ref_ids, dev_key) for m in members]  # every member (incl. the first) in one row order
    splits = [s for s in SPLITS if all((s, "a1") in d and (s, "a2") in d for d in data)]

    # ---------------- A1: mean logits + bias calibration
    a1_logits = {s: np.mean([d[(s, "a1")]["a1_logits"] for d in data], axis=0) for s in splits}
    y_a1 = {s: data[0][(s, "a1")]["y_a1"] for s in splits}
    cal1 = calibrate_a1(a1_logits[dev_key], y_a1[dev_key])
    a1_probs = {s: apply_a1(a1_logits[s], cal1) for s in splits}

    # ---------------- A2: per-item expert selection
    cands = {f"m{i}": {s: d[(s, "a2")]["a2_logits"] for s in splits} for i, d in enumerate(data)}
    cands["mean"] = {s: np.mean([d[(s, "a2")]["a2_logits"] for d in data], axis=0) for s in splits}
    y_a2 = {s: data[0][(s, "a2")]["y_a2"] for s in splits}
    dev_labels = y_a2[dev_key].astype(np.int64)

    table = {}  # (cand, method) -> (offsets, item_qwk)
    for c, logits in cands.items():
        if not expert_selection and c != "mean":
            continue
        for m in A2_DECODERS:
            table[(c, m)] = calibrate_a2_items(logits[dev_key], dev_labels, m)
    def item_mae(candidate, method, offsets):
        pred = decode_a2(cands[candidate][dev_key] + offsets[None, :, None], method)
        return np.abs(pred - dev_labels).mean(0)
    maes = {key: item_mae(*key, value[0]) for key, value in table.items()}
    default_method = max(A2_DECODERS, key=lambda m: (table[("mean", m)][1].mean(), -maes[("mean", m)].mean()))
    def_off, def_q = table[("mean", default_method)]

    n_items = dev_labels.shape[1]
    choice = []
    for j in range(n_items):
        (c, m), (off, q) = max(table.items(), key=lambda kv: (kv[1][1][j], -maes[kv[0]][j]))
        if q[j] - def_q[j] < min_gain:  # not clearly better than the mean ensemble: keep the default
            c, m, off, q = "mean", default_method, def_off, def_q
        choice.append({"item": f"d{j + 1:02d}", "candidate": c, "method": m,
                       "offset": float(off[j]), "selection_qwk": float(q[j])})

    a2_scores = {}
    for s in splits:
        cols = [decode_a2(cands[ch["candidate"]][s][:, j] + ch["offset"], ch["method"])
                for j, ch in enumerate(choice)]
        a2_scores[s] = np.stack(cols, axis=1)

    report = {"primary_task": primary_task,
              "note": "Only primary_task is an optimized track; development expert selection/calibration scores are in-sample.",
              "calibration_split": dev_key, "members": [str(m) for m in members], "a1_calibration": cal1, "a2_items": choice,
              "a2_default": {"candidate": "mean", "method": default_method},
              f"{dev_key}_selection_in_sample": {"a1": a1_metrics(a1_probs[dev_key], y_a1[dev_key]),
                                      "a2": a2_metrics(a2_scores[dev_key], y_a2[dev_key])}}
    if "val" in splits and (y_a1["val"] >= 0).all():
        report["val_used_for_selection" if dev_key == "val" else "val_held_out"] = {"a1": a1_metrics(a1_probs["val"], y_a1["val"]),
                         "a2": a2_metrics(a2_scores["val"], y_a2["val"])}
        log.info(f"val: A1 mean F1 {report['val_used_for_selection' if dev_key == 'val' else 'val_held_out']['a1']['f1']:.4f} | A2 mean QWK {report['val_used_for_selection' if dev_key == 'val' else 'val_held_out']['a2']['qwk']:.4f} "
                 f"(meaningful only if no member was trained with split.final=true)")

    with open(members[0] / "config_used.yaml", encoding="utf-8") as f:
        member_cfg = yaml.safe_load(f)
    manifest_dir = Path(member_cfg["paths"]["manifest_dir"])
    for s in ("val", "test_hidden"):
        if s in splits:
            write_submissions(run_dir / "submissions", s, ref_ids[s], a1_probs[s], a2_scores[s],
                              manifest_dir / f"{s}.csv", strict=int(member_cfg["debug"]["limit_participants"]) == 0)
    save_json(report, run_dir / "report.json")
    log.info(f"Done: {run_dir}")


if __name__ == "__main__":
    main()
