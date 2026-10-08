"""Config defaults, YAML loading and ``--set section.key=value`` overrides.

Unknown keys are rejected so a typo cannot silently fall back to a default.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

DEFAULTS: dict[str, dict[str, Any]] = {
    "paths": {
        "feature_root": None,
        "manifest_dir": None,
        "text_root": None,
        "output_dir": "outputs",
    },
    "features": {
        "audio": ["mel_mfcc", "vad", "ssl_embed"],
        "audio_pooled": ["egemaps"],
        "video": ["headpose_geom", "face_behavior", "qc_stats", "vad_agg",
                  "body_pose", "global_motion", "vision_ssl_embed"],
        "audio_ssl_tag": "chinese-hubert-large",
        "video_ssl_tag": "dinov2-large",
        "text_tag": "chinese-roberta-wwm-ext-large",
        "text_sessions": ["B01", "B02", "B03"],
        "max_text_tokens": 256,
        "grid_step_ms": 40.0,
        "tolerance_ms": 25.0,
        "mask_policy": "and_core",
        "core_audio": ["mel_mfcc", "ssl_embed"],
        "core_video": ["vision_ssl_embed", "qc_stats"],
        "vad_sources": ["vad", "vad_agg"],
        "qc_source": "qc_stats",
        "max_frames": 1500,
        "eval_max_frames": 4500,
        "probe_participants": 50,
    },
    "split": {
        "development_source": "inner_dev",
        "inner_dev_ratio": 0.1,
        "seed": 2026,
        "final": False,
        "require_official_val": True,
    },
    "model": {
        "d_adapter": 64,
        "d_model": 192,
        "n_heads": 4,
        "conformer_layers": 4,
        "conv_kernel": 15,
        "ff_mult": 4,
        "downsample": 2,
        "use_rotary": True,
        "use_cross_modal": True,
        "cross_modal_layers": 2,
        "use_text": False,
        "d_text": 192,
        "text_layers": 2,
        "d_session": 256,
        "cross_session_layers": 2,
        "cross_session_heads": 4,
        "a2_head": "query_coral",
        "query_layers": 2,
        "use_a1_guidance": True,
        "a1_guidance_detach": True,
        "dropout": 0.1,
        "feature_noise_std": 0.0,
        "modality_drop_prob": 0.1,
    },
    "loss": {
        "a1_weight": 1.0,
        "a2_weight": 1.0,
        "session_aux_weight": 0.2,
        "subscale_weight": 0.1,
        "subscale_tau": 2.0,
        "a1_pos_weight": True,
        "a2_pos_weight": True,
        "a2_pos_weight_cap": 4.0,
        "label_smoothing": 0.0,
    },
    "train": {
        "stage": 1,
        "selection_task": "both",
        "selection_metric": "calibrated",
        "gpu_memory_limit_gib": 0.0,
        "init_checkpoint": None,
        "batch_size": 16,
        "eval_batch_size": 2,
        "grad_accum_steps": 1,
        "prefetch_factor": 1,
        "strict_data": True,
        "require_cuda": False,
        "lr": 5.0e-4,
        "weight_decay": 0.05,
        "epochs": 40,
        "warmup_epochs": 2,
        "min_epochs": 5,
        "patience": 8,
        "grad_clip": 1.0,
        "amp": True,
        "num_workers": 4,
        "worker_batch_size": 0,
        "preload": False,
        "seed": 42,
        "deterministic": False,
        "session_drop_prob": 0.1,
        "report_val": True,
        "stats_participants": 256,
    },
    "debug": {
        "limit_participants": 0,
    },
}


def deep_merge(base: dict, override: dict, path: str = "") -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        key = f"{path}{k}"
        if k not in out:
            raise KeyError(f"Unknown config key: {key}")
        if isinstance(out[k], dict):
            if v is None:  # an empty section in the YAML, e.g. "loss:" with nothing below
                continue
            if not isinstance(v, dict):
                raise TypeError(f"Config key {key} must be a mapping")
            out[k] = deep_merge(out[k], v, key + ".")
        else:
            out[k] = v
    return out


def parse_overrides(items: list[str] | None) -> dict:
    out: dict = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"--set expects section.key=value, got {item!r}")
        key, raw = item.split("=", 1)
        value = yaml.safe_load(raw)
        node = out
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = value
    return out


def load_config(path: str | Path, overrides: list[str] | None = None, base: dict | None = None) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, encoding="utf-8") as f:
        user = yaml.safe_load(f) or {}
    cfg = deep_merge(base if base is not None else DEFAULTS, user)
    return deep_merge(cfg, parse_overrides(overrides))


def check_required(cfg: dict) -> None:
    for key in ("feature_root", "manifest_dir"):
        if not cfg["paths"][key]:
            raise ValueError(f"paths.{key} must be set (edit the YAML or pass --set paths.{key}=...)")


def save_config(cfg: dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
