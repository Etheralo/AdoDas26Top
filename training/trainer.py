"""Training loop, prediction and per-epoch evaluation."""
from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.collate import make_collate, merge_session_batches, to_device
from data.dataset import ParticipantDataset
from data.splits import development_split

from .calibration import apply_a1, apply_a2, calibrate_a1, calibrate_a2, sigmoid
from .losses import LossComputer
from .metrics import a1_metrics, a2_metrics
from .utils import save_json, worker_init_fn

log = logging.getLogger("adodas_sota")

NO_DECAY_KEYS = ("queries", "cls", "session_pos", "no_text", "session_prompt", "emb", "item_b", "thresholds")


class MergedLoader:
    """Read small batches in workers, merge to the requested optimizer batch."""
    def __init__(self, loader, batch_size, worker_batch_size, drop_last, pin_memory):
        self.loader = loader
        self.batch_size = batch_size
        self.chunks = batch_size // worker_batch_size
        self.drop_last = drop_last
        self.pin_memory = pin_memory

    def __len__(self):
        n = len(self.loader.dataset)
        return n // self.batch_size if self.drop_last else math.ceil(n / self.batch_size)

    def __iter__(self):
        chunks = []
        for batch in self.loader:
            chunks.append(batch)
            if len(chunks) == self.chunks:
                if not self.drop_last or sum(len(b["file_ids"]) for b in chunks) == self.batch_size:
                    yield merge_session_batches(chunks, pin_memory=self.pin_memory)
                chunks.clear()
        if chunks and not self.drop_last:
            yield merge_session_batches(chunks, pin_memory=self.pin_memory)


def make_loader(ds: ParticipantDataset, batch_size: int, shuffle: bool, num_workers: int,
                seed: int, drop_last: bool = False, prefetch_factor: int = 1,
                worker_batch_size: int = 0):
    g = torch.Generator()
    g.manual_seed(seed)
    chunk_size = min(batch_size, worker_batch_size) if num_workers > 0 and worker_batch_size > 0 else batch_size
    if batch_size % chunk_size:
        raise ValueError("batch_size must be divisible by worker_batch_size")
    merged = chunk_size < batch_size
    pin_memory = torch.cuda.is_available()
    loader = DataLoader(
        ds,
        batch_size=chunk_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=make_collate(ds.dims),
        pin_memory=pin_memory and not merged,
        drop_last=False if merged else drop_last,
        persistent_workers=num_workers > 0,
        worker_init_fn=worker_init_fn,
        generator=g,
        **({"prefetch_factor": prefetch_factor} if num_workers > 0 else {}),
    )
    return MergedLoader(loader, batch_size, chunk_size, drop_last, pin_memory) if merged else loader


def build_optimizer(model: torch.nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or any(k in name for k in NO_DECAY_KEYS):
            no_decay.append(p)
        else:
            decay.append(p)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}], lr=lr
    )


def build_scheduler(opt: torch.optim.Optimizer, total_steps: int, warmup_steps: int, min_ratio: float = 0.01):
    def fn(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(opt, fn)


def train_one_epoch(model, loader, opt, sched, losses: LossComputer, device, amp: bool,
                    grad_clip: float, desc: str, grad_accum_steps: int = 1, gpu_memory_limit_gib: float = 0.0) -> dict:
    model.train()
    if grad_accum_steps < 1:
        raise ValueError("grad_accum_steps must be positive")
    sums: dict[str, float] = {}
    n = 0
    gradient_updates = 0
    batches = len(loader)
    opt.zero_grad(set_to_none=True)
    pbar = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)
    for i, batch in enumerate(pbar):
        batch = to_device(batch, device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
            out = model(batch)
        loss, parts = losses(out, batch)
        if not torch.isfinite(loss):
            opt.zero_grad(set_to_none=True)
            raise RuntimeError("Non-finite training loss; inspect features and learning rate")
        # The last accumulation window may contain fewer microbatches.
        window_start = (i // grad_accum_steps) * grad_accum_steps
        window_size = min(grad_accum_steps, batches - window_start)
        (loss / window_size).backward()
        if device.type == "cuda" and gpu_memory_limit_gib > 0:
            reserved = torch.cuda.memory_reserved(device) / 2**30
            if reserved > gpu_memory_limit_gib:
                raise RuntimeError(f"GPU reserved memory {reserved:.2f} GiB exceeds configured "
                                   f"budget {gpu_memory_limit_gib:.2f} GiB; reduce batch size")
        if (i + 1) % grad_accum_steps == 0 or i + 1 == batches:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], grad_clip,
                error_if_nonfinite=True,
            )
            gradient_updates += 1
            sums["grad_norm"] = sums.get("grad_norm", 0.0) + float(grad_norm.detach())
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
        n += 1
        sums["loss"] = sums.get("loss", 0.0) + float(loss.detach())
        for k, v in parts.items():
            sums[k] = sums.get(k, 0.0) + v
        pbar.set_postfix_str(f"{float(loss.detach()):.4f}")
    pbar.close()
    if n == 0:
        raise RuntimeError("Empty training loader")
    return {k: v / (gradient_updates if k == "grad_norm" else n) for k, v in sums.items()}


@torch.no_grad()
def predict(model, loader, device, amp: bool, desc: str = "Predict") -> dict:
    model.eval()
    ids, a1, a2, y1, y2 = [], [], [], [], []
    for batch in tqdm(loader, desc=desc, leave=False, dynamic_ncols=True):
        ids.extend(batch["file_ids"])
        batch = to_device(batch, device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
            out = model(batch)
        a1.append(out["a1"].float().cpu().numpy())
        a2.append(out["a2"].float().cpu().numpy())
        y1.append(batch["y_a1"].cpu().numpy())
        y2.append(batch["y_a2"].cpu().numpy())
    return {
        "file_ids": np.asarray(ids),
        "a1_logits": np.concatenate(a1),
        "a2_logits": np.concatenate(a2),
        "y_a1": np.concatenate(y1),
        "y_a2": np.concatenate(y2),
    }


def score_predictions(pred: dict, calib_a1: dict | None = None, calib_a2: dict | None = None) -> dict:
    """Raw metrics plus metrics after applying (possibly separately fitted) calibration."""
    res = {
        "a1_raw": a1_metrics(sigmoid(pred["a1_logits"]), pred["y_a1"]),
        "a2_raw": a2_metrics(apply_a2(pred["a2_logits"], None), pred["y_a2"]),
    }
    if calib_a1 is not None:
        res["a1_cal"] = a1_metrics(apply_a1(pred["a1_logits"], calib_a1), pred["y_a1"])
    if calib_a2 is not None:
        res["a2_cal"] = a2_metrics(apply_a2(pred["a2_logits"], calib_a2), pred["y_a2"])
    return res


def fit(model, loaders: dict, losses: LossComputer, cfg: dict, device, amp: bool, run_dir: Path,
        ckpt_extra: dict) -> dict:
    """Trains, keeps ``best_a1.pt`` / ``best_a2.pt`` by inner-dev score and stops
    early on selection_task using selection_metric. Returns the history."""
    t = cfg["train"]
    dev_key = development_split(cfg)
    task = t["selection_task"]
    metric = t["selection_metric"]
    if task not in ("a1", "a2", "both") or metric not in ("raw", "calibrated"):
        raise ValueError("Invalid selection_task or selection_metric")
    log.info(f"Checkpoint/early-stop objective: {task}, {metric}; calibration scores are in-sample")
    epochs = int(t["epochs"])
    accum = int(t["grad_accum_steps"])
    if accum < 1:
        raise ValueError("train.grad_accum_steps must be positive")
    steps_per_epoch = max(1, math.ceil(len(loaders["train"]) / accum))
    log.info(f"Microbatch {t['batch_size']}, accumulation {accum}, "
             f"nominal effective participant batch {int(t['batch_size']) * accum}; "
             f"optimizer steps per epoch {steps_per_epoch}")
    opt = build_optimizer(model, float(t["lr"]), float(t["weight_decay"]))
    sched = build_scheduler(opt, epochs * steps_per_epoch, int(t["warmup_epochs"]) * steps_per_epoch)

    best = {"a1": -np.inf, "a2": -np.inf, "sum": -np.inf}
    best_order = {k: (-np.inf, -np.inf) for k in best}
    bad_epochs = 0
    history = []
    report_val = bool(t["report_val"]) and not cfg["split"]["final"] and "val" in loaders and dev_key != "val"

    for epoch in range(1, epochs + 1):
        tr = train_one_epoch(model, loaders["train"], opt, sched, losses, device, amp,
                             float(t["grad_clip"]), f"Train {epoch}/{epochs}", grad_accum_steps=accum,
                             gpu_memory_limit_gib=float(t["gpu_memory_limit_gib"]))
        dev_pred = predict(model, loaders[dev_key], device, amp, f"Selection:{dev_key}")
        cal1 = calibrate_a1(dev_pred["a1_logits"], dev_pred["y_a1"])
        cal2 = calibrate_a2(dev_pred["a2_logits"], dev_pred["y_a2"])
        dev = score_predictions(dev_pred, cal1, cal2)
        metric_key = "raw" if metric == "raw" else "cal"
        sel_a1 = dev[f"a1_{metric_key}"]["f1"]
        sel_a2 = dev[f"a2_{metric_key}"]["qwk"]
        order_a1 = (sel_a1, dev['a1_raw'].get('auroc', 0.0))
        order_a2 = (sel_a2, -dev[f'a2_{metric_key}'].get('mae', 0.0))
        row = {"epoch": epoch, "lr": opt.param_groups[0]["lr"], "train": tr, dev_key: dev}
        row["selection"] = {"task": task, "metric": metric, "split": dev_key, "a1": sel_a1, "a2": sel_a2}
        row["dev_predicted_positive_rate"] = (sigmoid(dev_pred["a1_logits"]) >= 0.5).mean(0).tolist()
        row["dev_label_positive_rate"] = (dev_pred["y_a1"] == 1).mean(0).tolist()
        if device.type == "cuda":
            row["peak_gpu_memory_GiB"] = torch.cuda.max_memory_allocated(device) / 2**30

        msg = (f"ep {epoch:3d} | loss {tr.get('loss', float('nan')):.4f} | dev A1 F1 raw {dev['a1_raw']['f1']:.4f} "
               f"cal {dev['a1_cal']['f1']:.4f} AUC {dev['a1_raw']['auroc']:.4f} | dev A2 QWK raw {dev['a2_raw']['qwk']:.4f} "
               f"cal {dev['a2_cal']['qwk']:.4f} MAE {dev['a2_cal']['mae']:.4f}")
        if report_val:
            val = score_predictions(predict(model, loaders["val"], device, amp, "Val"), cal1, cal2)
            row["val"] = val
            msg += f" || val A1 F1 {val['a1_cal']['f1']:.4f} A2 QWK {val['a2_cal']['qwk']:.4f}"

        improved = []
        for tag, score, order in (("a1", sel_a1, order_a1), ("a2", sel_a2, order_a2)):
            if order > best_order[tag]:
                best[tag] = score
                best_order[tag] = order
                save_checkpoint(run_dir / "checkpoints" / f"best_{tag}.pt", model, epoch, row, ckpt_extra)
                improved.append(tag)
        if improved:
            msg += f"  * best {'/'.join(improved)}"
        log.info(msg)
        history.append(row)

        total = sel_a1 if task == "a1" else sel_a2 if task == "a2" else sel_a1 + sel_a2
        order = order_a1 if task == 'a1' else order_a2 if task == 'a2' else (total, order_a1[1] + order_a2[1])
        if order > best_order['sum']:
            best["sum"] = total
            best_order['sum'] = order
            bad_epochs = 0
        else:
            bad_epochs += 1
        # Persist progress before the next epoch can be interrupted.
        save_checkpoint(run_dir / "checkpoints" / "last.pt", model, epoch, row, ckpt_extra)
        save_json({"history": history, "best": best}, run_dir / "history.json")
        if epoch >= int(t["min_epochs"]) and bad_epochs >= int(t["patience"]):
            log.info(f"Early stopping at epoch {epoch} (patience {t['patience']})")
            break

    save_checkpoint(run_dir / "checkpoints" / "last.pt", model, epoch, history[-1], ckpt_extra)
    return {"history": history, "best": best}


def save_checkpoint(path: Path, model, epoch: int, metrics: dict, extra: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({"model": model.state_dict(), "epoch": epoch, "metrics": metrics, **extra}, temporary)
    temporary.replace(path)


def load_checkpoint(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)
