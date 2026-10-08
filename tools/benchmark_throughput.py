#!/usr/bin/env python3
"""Measure real-data training throughput; never print identifiers or features.

Cases are batch_size:num_workers:grad_accum_steps. Each uses the same random
participant pool, full configured architecture and frame limits. Warmup and
timed samples are separate; model weights are disposable and never exported.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch

from data.collate import to_device
from data.dataset import ParticipantDataset, participant_labels
from data.splits import load_participant_sets
from models.model import build_model, compute_priors
from training.config import load_config
from training.losses import LossComputer
from training.trainer import build_optimizer, make_loader
from training.utils import seed_everything


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--split-file", required=True)
    ap.add_argument("--cases", default="8:8:2,16:8:1,16:12:1,16:16:1")
    ap.add_argument("--participants", type=int, default=256)
    ap.add_argument("--warmup-participants", type=int, default=64)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    torch.set_num_threads(1)
    seed_everything(42)
    sets, _ = load_participant_sets(cfg, split_file=Path(args.split_file))
    count = args.participants + args.warmup_participants
    indices = np.random.default_rng(42).permutation(len(sets["train"]))[:count]
    pool = [sets["train"][int(i)] for i in indices]
    y1, y2 = participant_labels(sets["train"])
    ds = ParticipantDataset(pool, cfg, train=True)
    results = []
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    for case in args.cases.split(","):
        batch_size, workers, accum = map(int, case.split(":"))
        if min(batch_size, accum) < 1 or workers < 0:
            raise ValueError("Batch/accumulation must be positive; workers cannot be negative")
        seed_everything(42)
        model = build_model(cfg["model"], ds.dims, priors=compute_priors(y1, y2)).cuda().train()
        losses = LossComputer(cfg["loss"], y1, y2, torch.device("cuda"))
        opt = build_optimizer(model, float(cfg["train"]["lr"]), float(cfg["train"]["weight_decay"]))
        prefetch = int(cfg["train"]["prefetch_factor"])
        loader = make_loader(ds, batch_size, False, workers, 42, prefetch_factor=prefetch,
                             worker_batch_size=int(cfg["train"]["worker_batch_size"]))
        iterator = iter(loader)
        warmup = math.ceil(args.warmup_participants / batch_size)
        timed = args.participants // batch_size
        timings = []
        samples = []
        host_samples = []
        stop = threading.Event()

        def sample_gpu():
            while not stop.is_set():
                try:
                    raw = subprocess.check_output(
                        ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                        text=True, timeout=3,
                    )
                    samples.append(float(raw.strip().splitlines()[0]))
                except (OSError, subprocess.SubprocessError, ValueError):
                    pass
                try:
                    stat = dict(line.split() for line in Path("/sys/fs/cgroup/memory.stat").read_text().splitlines())
                    host_samples.append({
                        "current": int(Path("/sys/fs/cgroup/memory.current").read_text()),
                        "anon_and_shmem": int(stat["anon"]) + int(stat["shmem"]),
                        "shmem": int(stat["shmem"]),
                    })
                except (OSError, KeyError, ValueError):
                    pass
                stop.wait(0.5)

        monitor = None
        started = time.perf_counter()
        try:
            opt.zero_grad(set_to_none=True)
            for i in range(warmup + timed):
                if i == warmup:
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    started = time.perf_counter()
                    monitor = threading.Thread(target=sample_gpu, daemon=True)
                    monitor.start()
                ready = time.perf_counter()
                batch = next(iterator)
                loaded = time.perf_counter()
                batch = to_device(batch, torch.device("cuda"))
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = model(batch)
                loss, _ = losses(out, batch)
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite loss")
                (loss / accum).backward()
                if (i + 1) % accum == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["train"]["grad_clip"]),
                                                  error_if_nonfinite=True)
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                if i >= warmup:
                    timings.append((loaded-ready, time.perf_counter()-loaded))
            elapsed = time.perf_counter() - started
            record = dict(status="PASS", batch_size=batch_size, num_workers=workers,
                          grad_accum_steps=accum, prefetch_factor=prefetch,
                          timed_participants=timed*batch_size, timed_seconds=round(elapsed,3),
                          participants_per_second=round(timed*batch_size/elapsed,3),
                          mean_data_wait_seconds=round(np.mean([x[0] for x in timings]),4),
                          mean_compute_seconds=round(np.mean([x[1] for x in timings]),4),
                          peak_allocated_GiB=round(torch.cuda.max_memory_allocated()/2**30,3),
                          peak_reserved_GiB=round(torch.cuda.max_memory_reserved()/2**30,3))
        finally:
            stop.set()
            if monitor is not None:
                monitor.join(timeout=4)
            # Shut down persistent workers before the next case allocates a pool.
            if hasattr(iterator, "close"):
                iterator.close()
            base = loader.loader if hasattr(loader, "loader") else loader
            if base._iterator is not None:
                base._iterator._shutdown_workers()
        record["gpu_utilization_samples_percent"] = samples
        record["effective_participant_batch"] = batch_size * accum
        record["worker_batch_size"] = int(cfg["train"]["worker_batch_size"])
        if host_samples:
            for key in ("current", "anon_and_shmem", "shmem"):
                record[f"peak_sampled_host_{key}_GiB"] = round(max(x[key] for x in host_samples)/2**30,3)
        record["mean_sampled_gpu_utilization_percent"] = round(float(np.mean(samples)),1) if samples else None
        results.append(record)
        output.write_text(json.dumps({"config":args.config,"results":results}, indent=2)+"\n")
        print(json.dumps(record), flush=True)
        del iterator, loader, model, opt, out, loss, batch
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
