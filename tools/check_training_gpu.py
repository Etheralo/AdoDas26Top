#!/usr/bin/env python3
"""Probe the configured architecture on real data and maximum-length inputs.

This checks execution and memory, not model quality or full dataset coverage.
Participant identifiers and feature values are never printed.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from data.collate import make_collate, to_device
from data.dataset import ParticipantDataset, build_participants, participant_labels, read_manifest
from models.model import build_model, compute_priors
from training.config import check_required, load_config
from training.losses import LossComputer
from training.trainer import build_optimizer


def synthetic_batch(dims, participants, frames, text_tokens):
    n = participants * 4
    out = {
        "audio": {k: torch.randn(n, frames, d, device="cuda") for k, d in dims["audio"].items()},
        "video": {k: torch.randn(n, frames, d, device="cuda") for k, d in dims["video"].items()},
        "mask_audio": torch.ones(n, frames, dtype=torch.bool, device="cuda"),
        "mask_video": torch.ones(n, frames, dtype=torch.bool, device="cuda"),
        "vad": torch.ones(n, frames, device="cuda"),
        "qc": torch.ones(n, frames, device="cuda"),
        "pooled": {k: torch.randn(n, d, device="cuda") for k, d in dims["pooled"].items()},
        "pooled_present": {k: torch.ones(n, dtype=torch.bool, device="cuda") for k in dims["pooled"]},
        "session_type": torch.arange(4, device="cuda").repeat(participants),
        "session_valid": torch.ones(participants, 4, dtype=torch.bool, device="cuda"),
        "y_a1": torch.zeros(participants, 3, device="cuda"),
        "y_a2": torch.zeros(participants, 21, device="cuda"),
    }
    if dims.get("text"):
        out.update(text=torch.randn(n, text_tokens, dims["text"], device="cuda"),
                   text_mask=torch.ones(n, text_tokens, dtype=torch.bool, device="cuda"),
                   text_present=torch.ones(n, dtype=torch.bool, device="cuda"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/server_stage1.yaml")
    ap.add_argument("--output", default=None)
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    check_required(cfg)
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")
    torch.manual_seed(42)
    parts = build_participants(read_manifest(Path(cfg["paths"]["manifest_dir"]) / "train.csv"), "train")
    b = int(cfg["train"]["batch_size"])
    ds = ParticipantDataset(parts[:max(b, 2)], cfg, train=True)
    dims = ds.dims
    y1, y2 = participant_labels(parts)
    device = torch.device("cuda")
    model = build_model(cfg["model"], dims, priors=compute_priors(y1, y2)).to(device)
    losses = LossComputer(cfg["loss"], y1, y2, device)
    opt = build_optimizer(model, float(cfg["train"]["lr"]), float(cfg["train"]["weight_decay"]))
    result = {"status": "RUNNING", "torch": str(torch.__version__), "gpu": torch.cuda.get_device_name(),
              "config": args.config, "text_enabled": bool(cfg["model"]["use_text"]),
              "parameters": sum(p.numel() for p in model.parameters()), "checks": []}

    def check(label, batch_factory, train):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()
        batch = batch_factory()
        model.train(train)
        with torch.set_grad_enabled(train):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bool(cfg["train"]["amp"])):
                out = model(batch)
            if not all(torch.isfinite(out[k]).all() for k in ("a1", "a2")):
                raise RuntimeError(f"{label}: nonfinite outputs")
            if train:
                loss, _ = losses(out, batch)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"{label}: nonfinite loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["train"]["grad_clip"]),
                                              error_if_nonfinite=True)
                opt.step()
                opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        record = {"check": label, "status": "PASS", "participants": int(batch["session_valid"].shape[0]),
                  "frames": int(batch["mask_audio"].shape[1]), "seconds": round(time.perf_counter()-start, 3),
                  "peak_allocated_GiB": round(torch.cuda.max_memory_allocated()/2**30, 3),
                  "peak_reserved_GiB": round(torch.cuda.max_memory_reserved()/2**30, 3)}
        result["checks"].append(record)
        print(json.dumps(record), flush=True)
        limit = float(cfg["train"]["gpu_memory_limit_gib"])
        if limit and record["peak_reserved_GiB"] > limit:
            raise RuntimeError(f"{label}: reserved memory exceeds {limit} GiB; reduce batch size")

    try:
        check("real_data_train_step", lambda: to_device(make_collate(dims)([ds[i] for i in range(b)]), device), True)
        check("maximum_train_length_step", lambda: synthetic_batch(dims, b, int(cfg["features"]["max_frames"]),
                                                                   int(cfg["features"]["max_text_tokens"])), True)
        check("maximum_eval_length_forward", lambda: synthetic_batch(dims, int(cfg["train"]["eval_batch_size"]),
                                                                      int(cfg["features"]["eval_max_frames"]),
                                                                      int(cfg["features"]["max_text_tokens"])), False)
        result["status"] = "PASS"
    except Exception as exc:
        result.update(status="FAIL", error=f"{type(exc).__name__}: {exc}")
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    return result["status"] != "PASS"


if __name__ == "__main__":
    sys.exit(main())
