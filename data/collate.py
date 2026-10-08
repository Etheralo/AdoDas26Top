"""Batch participants into flat session tensors (4 slots per participant)."""
from __future__ import annotations

import math
from typing import Callable

import numpy as np
import torch

from .dass import SESSIONS


def make_collate(dims: dict) -> Callable[[list[dict]], dict]:
    audio_dims = dims["audio"]
    video_dims = dims["video"]
    pooled_dims = dims["pooled"]
    text_dim = dims.get("text")

    def zeros(*shape: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        if torch.utils.data.get_worker_info() is None:
            return torch.zeros(*shape, dtype=dtype)
        # Match PyTorch's default_collate allocation: write directly to shared
        # storage instead of copying a complete padded batch into shared memory.
        prototype = torch.empty(0, dtype=dtype)
        storage = prototype._typed_storage()._new_shared(math.prod(shape), device=prototype.device)
        return prototype.new(storage).resize_(*shape).zero_()

    def collate(batch: list[dict]) -> dict:
        B = len(batch)
        flat = [s for item in batch for s in item["sessions"]]
        N = len(flat)
        lengths = [len(s["mask_audio"]) if s is not None else 1 for s in flat]
        T = max(lengths)

        audio = {n: zeros(N, T, d) for n, d in audio_dims.items()}
        video = {n: zeros(N, T, d) for n, d in video_dims.items()}
        mask_audio = zeros(N, T, dtype=torch.bool)
        mask_video = zeros(N, T, dtype=torch.bool)
        vad = zeros(N, T)
        qc = zeros(N, T)
        pooled = {n: zeros(N, d) for n, d in pooled_dims.items()}
        pooled_present = {n: zeros(N, dtype=torch.bool) for n in pooled_dims}

        if text_dim:
            text_lens = [len(s["text"]) if s is not None and s.get("text") is not None else 0 for s in flat]
            L = max(1, max(text_lens))
            text = zeros(N, L, text_dim)
            text_mask = zeros(N, L, dtype=torch.bool)

        for i, s in enumerate(flat):
            if s is None:
                continue
            t = lengths[i]
            for n in audio_dims:
                audio[n][i, :t] = torch.from_numpy(np.ascontiguousarray(s["audio"][n]))
            for n in video_dims:
                video[n][i, :t] = torch.from_numpy(np.ascontiguousarray(s["video"][n]))
            mask_audio[i, :t] = torch.from_numpy(np.ascontiguousarray(s["mask_audio"]))
            mask_video[i, :t] = torch.from_numpy(np.ascontiguousarray(s["mask_video"]))
            vad[i, :t] = torch.from_numpy(np.ascontiguousarray(s["vad"]))
            qc[i, :t] = torch.from_numpy(np.ascontiguousarray(s["qc"]))
            for n in pooled_dims:
                pooled[n][i] = torch.from_numpy(s["pooled"][n])
                pooled_present[n][i] = bool(s["pooled_present"][n])
            if text_dim and text_lens[i] > 0:
                text[i, : text_lens[i]] = torch.from_numpy(s["text"])
                text_mask[i, : text_lens[i]] = True

        out = {
            "audio": audio,
            "video": video,
            "mask_audio": mask_audio,
            "mask_video": mask_video,
            "vad": vad,
            "qc": qc,
            "pooled": pooled,
            "pooled_present": pooled_present,
            "session_type": torch.arange(len(SESSIONS)).repeat(B),
            "session_valid": torch.tensor([[s is not None for s in item["sessions"]] for item in batch]),
            "y_a1": torch.from_numpy(np.stack([item["y_a1"] for item in batch])),
            "y_a2": torch.from_numpy(np.stack([item["y_a2"] for item in batch])),
            "file_ids": [item["file_id"] for item in batch],
        }
        if text_dim:
            out["text"] = text
            out["text_mask"] = text_mask
            out["text_present"] = text_mask.any(dim=1)
        return out

    return collate


def merge_session_batches(batches: list[dict], *, pin_memory: bool = False) -> dict:
    """Merge small worker batches into one padded participant batch on CPU."""
    if not batches:
        raise ValueError("Cannot merge an empty batch list")

    def merge(values, pad_time=False):
        first = values[0]
        if isinstance(first, dict):
            return {k: merge([v[k] for v in values], pad_time) for k in first}
        if isinstance(first, list):
            return [item for v in values for item in v]
        if not isinstance(first, torch.Tensor) or first.device.type != "cpu":
            raise TypeError("Batch merger expects CPU tensors and nested dict/list values")
        count = sum(v.shape[0] for v in values)
        if pad_time:
            length = max(v.shape[1] for v in values)
            result = torch.zeros((count, length, *first.shape[2:]), dtype=first.dtype, pin_memory=pin_memory)
            offset = 0
            for value in values:
                result[offset:offset + len(value), :value.shape[1]].copy_(value)
                offset += len(value)
            return result
        result = torch.empty((count, *first.shape[1:]), dtype=first.dtype, pin_memory=pin_memory)
        return torch.cat(values, dim=0, out=result)

    temporal = {"audio", "video", "mask_audio", "mask_video", "vad", "qc", "text", "text_mask"}
    return {k: merge([b[k] for b in batches], k in temporal) for k in batches[0]}


def to_device(obj, device: torch.device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    return obj
