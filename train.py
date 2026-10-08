#!/usr/bin/env python3
"""Train one model.

Stage 1 (end-to-end):
    python train.py --config configs/stage1.yaml
Stage 2 (freeze the encoder, re-fit task heads; HITSZ_HLT "staged task-head adjustment"):
    python train.py --config configs/stage2.yaml --set train.init_checkpoint=outputs/runs/<run>/checkpoints/best_a1.pt
Smoke test (32 participants, 1 epoch):
    python train.py --config configs/stage1.yaml --smoke

After training, predictions for inner-dev / val / test_hidden are exported,
calibration is fitted on inner-dev and submission files are written (see infer.py).
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch

import infer
from data.dass import check_label_consistency
from data.dataset import ParticipantDataset, compute_feature_stats, participant_labels
from data.splits import development_split, file_sha1, load_participant_sets
from models.model import build_model, compute_priors
from training.config import check_required, load_config, save_config
from training.losses import LossComputer
from training.trainer import fit, load_checkpoint, make_loader
from training.utils import (amp_enabled, environment_info, make_run_dir, resolve_device, save_json,
                            seed_everything, setup_logging)

SMOKE_OVERRIDES = [
    "debug.limit_participants=32",
    "train.epochs=1",
    "train.min_epochs=1",
    "train.num_workers=0",
    "train.preload=false",
    "train.stats_participants=8",
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], help="Overrides, e.g. train.lr=3e-4 model.use_text=true")
    ap.add_argument("--name", default=None, help="Run name prefix")
    ap.add_argument("--smoke", action="store_true", help="Tiny end-to-end check")
    return ap.parse_args()


def run_name(cfg: dict, stage: int) -> str:
    f, m = cfg["features"], cfg["model"]
    parts = [f"stage{stage}", f"a-{f['audio_ssl_tag']}", f"v-{f['video_ssl_tag']}"]
    if m["use_text"]:
        parts.append("text")
    if cfg["split"]["final"]:
        parts.append("final")
    parts.append(f"seed{cfg['train']['seed']}")
    return "_".join(parts)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, list(args.set) + (SMOKE_OVERRIDES if args.smoke else []))
    stage = int(cfg["train"]["stage"])

    init_ckpt, split_file = None, None
    if stage == 2 and not cfg["train"]["init_checkpoint"]:
        raise ValueError("Stage 2 needs train.init_checkpoint (a stage-1 best_a1.pt / best_a2.pt)")
    if cfg["train"]["init_checkpoint"]:
        ckpt_path = Path(cfg["train"]["init_checkpoint"])
        init_ckpt = load_checkpoint(ckpt_path)
        # Architecture, features and data split must match the stage-1 model.
        for section in ("model", "features", "split"):
            cfg[section] = init_ckpt["config"][section]
        split_file = ckpt_path.parent.parent / "inner_dev.csv"
    check_required(cfg)
    if cfg["train"]["require_cuda"] and not torch.cuda.is_available():
        raise RuntimeError("CUDA required by this config; allocate a GPU before training")

    seed_everything(int(cfg["train"]["seed"]), bool(cfg["train"]["deterministic"]))
    name = (args.name + "_" if args.name else "") + run_name(cfg, stage) + ("_smoke" if args.smoke else "")
    run_dir = make_run_dir(cfg["paths"]["output_dir"], name)
    log = setup_logging(run_dir / "logs" / "train.log")
    save_config(cfg, run_dir / "config_used.yaml")
    save_json(environment_info(), run_dir / "environment.json")
    device = resolve_device()
    amp = amp_enabled(bool(cfg["train"]["amp"]), device, log)
    log.info(f"Run dir: {run_dir} | device {device} | bf16 {amp}")
    if init_ckpt is not None and stage == 1:
        warm_start = {"checkpoint": cfg["train"]["init_checkpoint"], "source_epoch": init_ckpt["epoch"],
                      "optimizer_reinitialized": True, "scheduler_reinitialized": True}
        save_json(warm_start, run_dir / "warm_start.json")
        log.info(f"Stage1 warm start from epoch {init_ckpt['epoch']}; optimizer/scheduler reinitialized")

    sets, split_path = load_participant_sets(cfg, split_file=split_file)
    shutil.copy(split_path, run_dir / "inner_dev.csv")
    shutil.copy(split_path, run_dir / "development.csv")
    dev_key = development_split(cfg)
    save_json({"selection_split": dev_key, "calibration_split": dev_key,
               "validation_used_for_selection": dev_key == "val",
               "test_labels_used_for_training": False}, run_dir / "evaluation_protocol.json")
    log.info(f"Participants: train {len(sets['train'])}, inner_dev {len(sets['inner_dev'])}, val {len(sets['val'])} "
             f"(split file sha1 {file_sha1(split_path)[:12]}, final={cfg['split']['final']})")

    y1, y2 = participant_labels(sets["train"])
    consistency = check_label_consistency(y1, y2)
    log.info(f"A1 labels vs. labels derived from A2 items: {consistency}")
    if consistency["n"] and consistency["all_agree"] < 0.99 and float(cfg["loss"]["subscale_weight"]) > 0:
        log.warning("A1 labels do not follow the DASS-21 cutoffs for >1% of participants; "
                    "consider --set loss.subscale_weight=0")

    t = cfg["train"]
    train_ds = ParticipantDataset(sets["train"], cfg, train=True,
                                  dims=init_ckpt["dims"] if init_ckpt else None, preload=bool(t["preload"]))
    dims = train_ds.dims
    dev_ds = ParticipantDataset(sets[dev_key], cfg, train=False, dims=dims, preload=bool(t["preload"]))
    datasets = {"train": train_ds, dev_key: dev_ds}
    if dev_key != "val" and bool(t["report_val"]) and not cfg["split"]["final"] and sets["val"]:
        datasets["val"] = ParticipantDataset(sets["val"], cfg, train=False, dims=dims, preload=bool(t["preload"]))
    if cfg["model"]["use_text"]:
        found, expected = train_ds.text_coverage()
        log.info(f"Text coverage (train): {found}/{expected} open-response sessions")
        if found != expected or not dims.get("text"):
            raise RuntimeError("Text enabled but embeddings are incomplete; extract all official ASR sessions first")

    stats = None
    if init_ckpt is None:
        stats = compute_feature_stats(train_ds, int(t["stats_participants"]), int(t["seed"]))
    model = build_model(cfg["model"], dims, stats, compute_priors(y1, y2))
    if init_ckpt is not None:
        model.load_state_dict(init_ckpt["model"])
        if stage == 2:
            model.freeze_encoder(cfg["train"]["selection_task"])
    model.to(device)
    n_all = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Parameters: {n_all:,} total, {n_train:,} trainable")

    loaders = {
        k: make_loader(ds, int(t["batch_size"] if k == "train" else t["eval_batch_size"]),
                       shuffle=(k == "train"), num_workers=int(t["num_workers"]),
                       seed=int(t["seed"]), prefetch_factor=int(t["prefetch_factor"]), drop_last=False,
                       worker_batch_size=int(t["worker_batch_size"]))
        for k, ds in datasets.items()
    }
    losses = LossComputer(cfg["loss"], y1, y2, device)
    result = fit(model, loaders, losses, cfg, device, amp, run_dir, {"config": cfg, "dims": dims})
    save_json(result, run_dir / "history.json")

    all_sets, _ = load_participant_sets(cfg, split_file=run_dir / "inner_dev.csv", include_test=True)
    infer.export_predictions(run_dir, cfg, all_sets, device)
    infer.finalize_run(run_dir, cfg)
    log.info(f"Done: {run_dir}")


if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy("file_system")
    main()
