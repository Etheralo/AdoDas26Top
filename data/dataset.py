"""Participant-level dataset: one sample = one participant with four session slots."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm

from .alignment import align_to_grid, combine_masks
from .dass import A1_COLS, ITEM_COLS, SESSIONS
from .feature_io import FeatureDimError, feature_dir, load_pooled, load_sequence, pooled_exists
from .text_io import load_text, text_path

log = logging.getLogger(__name__)

ID_COLS = ("anon_school", "anon_class", "anon_pid", "session")
_MAX_LOGGED_FAILURES = 20


@dataclass
class Participant:
    file_id: str
    school: str
    cls: str
    pid: str
    feature_split: str
    sessions: tuple[str, ...]
    y_a1: np.ndarray  # (3,) float32, -1 when unlabeled
    y_a2: np.ndarray  # (21,) float32, -1 when unlabeled


def read_manifest(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    df = pd.read_csv(path, dtype={c: str for c in ID_COLS})
    missing = [c for c in ID_COLS if c not in df.columns]
    if missing:
        raise KeyError(f"{path} is missing columns {missing}")
    if df[list(ID_COLS)].isna().any().any():
        raise ValueError(f"{path}: missing participant/session identifiers")
    if df.duplicated(list(ID_COLS)).any():
        raise ValueError(f"{path}: duplicate participant/session rows")
    if not df["session"].isin(SESSIONS).all():
        raise ValueError(f"{path}: unknown session names")
    label_cols = [c for c in (*A1_COLS, *ITEM_COLS) if c in df.columns]
    for c in label_cols:
        values = pd.to_numeric(df[c], errors="raise")
        allowed = (0, 1) if c in A1_COLS else (0, 1, 2, 3)
        if not (values.isna() | values.isin(allowed)).all():
            raise ValueError(f"{path}: invalid values in label column {c}")
    for c in label_cols:
        if (df.groupby(list(ID_COLS[:3]))[c].nunique(dropna=False) > 1).any():
            raise ValueError(f"{path}: session rows disagree on participant label {c}")
    return df


def build_participants(df: pd.DataFrame, feature_split: str) -> list[Participant]:
    out: list[Participant] = []
    for (school, cls, pid), group in df.groupby(["anon_school", "anon_class", "anon_pid"], sort=True):
        first = group.iloc[0]

        def _labels(cols: tuple[str, ...]) -> np.ndarray:
            vals = []
            for c in cols:
                v = first[c] if c in group.columns else np.nan
                vals.append(-1.0 if pd.isna(v) else float(v))
            return np.asarray(vals, dtype=np.float32)

        sessions = tuple(s for s in SESSIONS if s in set(group["session"].astype(str)))
        out.append(
            Participant(
                file_id=f"{school}_{cls}_{pid}",
                school=str(school),
                cls=str(cls),
                pid=str(pid),
                feature_split=feature_split,
                sessions=sessions,
                y_a1=_labels(A1_COLS),
                y_a2=_labels(ITEM_COLS),
            )
        )
    return out


class ParticipantDataset(Dataset):
    """Loads, aligns and crops the four sessions of each participant.

    ``dims`` fixes the model input dimensions. When omitted it is probed from
    the data and every configured feature group must be found, otherwise a
    ``RuntimeError`` is raised (no silent feature dropping).
    """

    def __init__(
        self,
        participants: list[Participant],
        cfg: dict,
        *,
        train: bool,
        dims: dict | None = None,
        preload: bool = False,
    ) -> None:
        f = cfg["features"]
        self.cfg_features = f
        self.participants = participants
        self.train = train
        self.feature_root = Path(cfg["paths"]["feature_root"])
        use_text = bool(cfg["model"]["use_text"])
        self.text_root = cfg["paths"].get("text_root") if use_text else None
        if use_text and not self.text_root:
            raise ValueError("model.use_text is true but paths.text_root is not set")
        self.audio_seq = list(f["audio"])
        self.audio_pooled = list(f["audio_pooled"])
        self.video_seq = list(f["video"])
        self.tags = {"ssl_embed": f["audio_ssl_tag"], "vision_ssl_embed": f["video_ssl_tag"]}
        self.max_frames = int(f["max_frames"] if train else f["eval_max_frames"]) or 0
        self.session_drop_prob = float(cfg["train"]["session_drop_prob"]) if train else 0.0
        self.strict_data = bool(cfg["train"]["strict_data"])
        self._failures = 0
        self._cache: list[list[dict | None]] | None = None
        self.dims = dims if dims is not None else self.probe_dims(int(f["probe_participants"]))
        if preload:
            self.preload()

    # ------------------------------------------------------------------ probing
    def _dir(self, p: Participant, modality: str, name: str, session: str, pooled: bool = False) -> Path:
        tag = None if pooled else self.tags.get(name)
        return feature_dir(self.feature_root, p.feature_split, p.school, p.cls, p.pid, modality, name, session, tag)

    def probe_dims(self, max_participants: int) -> dict:
        dims: dict[str, Any] = {"audio": {}, "video": {}, "pooled": {}, "text": None}
        need_text = self.text_root is not None
        for p in self.participants[: max(1, max_participants)]:
            for s in p.sessions:
                for modality, names in (("audio", self.audio_seq), ("video", self.video_seq)):
                    for name in names:
                        if name in dims[modality]:
                            continue
                        d = self._dir(p, modality, name, s)
                        if (d / "sequence.npz").exists():
                            dims[modality][name] = int(load_sequence(d, name).features.shape[1])
                for name in self.audio_pooled:
                    if name in dims["pooled"]:
                        continue
                    vec = load_pooled(self._dir(p, "audio", name, s, pooled=True))
                    if vec is not None:
                        dims["pooled"][name] = int(vec.shape[0])
                if need_text and dims["text"] is None and s in self.cfg_features["text_sessions"]:
                    t = load_text(self.text_root, p.feature_split, p.school, p.cls, p.pid,
                                  self.cfg_features["text_tag"], s, 0)
                    if t is not None:
                        dims["text"] = int(t.shape[1])
            if self._probe_complete(dims, need_text):
                break

        missing = [f"audio/{n}" for n in self.audio_seq if n not in dims["audio"]]
        missing += [f"video/{n}" for n in self.video_seq if n not in dims["video"]]
        missing += [f"pooled/{n}" for n in self.audio_pooled if n not in dims["pooled"]]
        if need_text and dims["text"] is None:
            missing.append("text")
        if missing:
            p = self.participants[0]
            example = self._dir(p, "audio", "ssl_embed", p.sessions[0] if p.sessions else "A01")
            raise RuntimeError(
                f"Configured features not found in the first {max_participants} participants: {missing}. "
                f"Check paths.feature_root, the SSL tags (audio_ssl_tag={self.tags['ssl_embed']!r}, "
                f"video_ssl_tag={self.tags['vision_ssl_embed']!r}) and the feature list. "
                f"Example expected path: {example}"
            )
        log.info(f"Feature dims: {dims}")
        return dims

    def _probe_complete(self, dims: dict, need_text: bool) -> bool:
        return (
            all(n in dims["audio"] for n in self.audio_seq)
            and all(n in dims["video"] for n in self.video_seq)
            and all(n in dims["pooled"] for n in self.audio_pooled)
            and (not need_text or dims["text"] is not None)
        )

    # ------------------------------------------------------------------ loading
    def load_session(self, p: Participant, session: str) -> dict | None:
        f = self.cfg_features
        groups = {}
        for modality, names in (("audio", self.audio_seq), ("video", self.video_seq)):
            for name in names:
                d = self._dir(p, modality, name, session)
                if not (d / "sequence.npz").exists():
                    continue
                seq = load_sequence(d, name)
                if seq.features.shape[1] != self.dims[modality][name]:
                    raise FeatureDimError(
                        f"{d}: dim {seq.features.shape[1]} != expected {self.dims[modality][name]}"
                    )
                if len(seq.timestamps_ms):
                    groups[f"{modality}/{name}"] = seq
        if not groups:
            return None

        grid, feats, masks = align_to_grid(groups, float(f["grid_step_ms"]), float(f["tolerance_ms"]))
        T = len(grid)
        out: dict[str, Any] = {"audio": {}, "video": {}, "group_masks": {}}
        present = {"audio": {}, "video": {}}
        for modality, names in (("audio", self.audio_seq), ("video", self.video_seq)):
            for name in names:
                key = f"{modality}/{name}"
                if key in feats:
                    out[modality][name] = feats[key]
                    out["group_masks"][key] = masks[key]
                    present[modality][name] = masks[key]
                else:
                    out[modality][name] = np.zeros((T, self.dims[modality][name]), dtype=np.float32)
                    out["group_masks"][key] = np.zeros(T, dtype=bool)

        out["mask_audio"] = combine_masks(present["audio"], f["core_audio"], f["mask_policy"], T)
        out["mask_video"] = combine_masks(present["video"], f["core_video"], f["mask_policy"], T)

        vad = np.zeros(T, dtype=np.float32)
        for src in f["vad_sources"]:
            key = next((k for k in (f"audio/{src}", f"video/{src}") if k in feats), None)
            if key is not None:
                vad = feats[key][:, 0].astype(np.float32) * masks[key]
                break
        qc = np.zeros(T, dtype=np.float32)
        qc_key = next((k for k in (f"video/{f['qc_source']}", f"audio/{f['qc_source']}") if k in feats), None)
        if qc_key is not None:
            qc = feats[qc_key][:, 0].astype(np.float32) * masks[qc_key]
        out["vad"], out["qc"] = vad, qc

        out["pooled"], out["pooled_present"] = {}, {}
        for name in self.audio_pooled:
            vec = load_pooled(self._dir(p, "audio", name, session, pooled=True))
            if vec is not None and vec.shape[0] != self.dims["pooled"][name]:
                raise FeatureDimError(f"pooled {name} for {p.file_id}/{session}: dim {vec.shape[0]}")
            out["pooled_present"][name] = vec is not None
            out["pooled"][name] = vec if vec is not None else np.zeros(self.dims["pooled"][name], np.float32)

        out["text"] = None
        if self.text_root is not None and session in f["text_sessions"]:
            out["text"] = load_text(self.text_root, p.feature_split, p.school, p.cls, p.pid,
                                    f["text_tag"], session, int(f["max_text_tokens"]))
        return out

    def load_participant(self, p: Participant) -> list[dict | None]:
        sessions: list[dict | None] = []
        for s in SESSIONS:
            if s not in p.sessions:
                sessions.append(None)
                continue
            try:
                sessions.append(self.load_session(p, s))
            except FeatureDimError:
                raise
            except Exception as exc:
                if self.strict_data:
                    raise RuntimeError(f"Feature loading failed for {p.file_id}/{s}") from exc
                self._failures += 1
                if self._failures <= _MAX_LOGGED_FAILURES:
                    log.warning(f"Failed to load {p.file_id}/{s}: {exc!r}")
                sessions.append(None)
        return sessions

    def get_sessions(self, idx: int) -> list[dict | None]:
        if self._cache is not None:
            return self._cache[idx]
        return self.load_participant(self.participants[idx])

    def preload(self) -> None:
        cache = []
        for p in tqdm(self.participants, desc="Preload", dynamic_ncols=True):
            sessions = self.load_participant(p)
            cache.append([_to_half(s) if s is not None else None for s in sessions])
        self._cache = cache
        n_missing = sum(s is None for sess in cache for s in sess)
        log.info(f"Preloaded {len(cache)} participants; {n_missing} empty session slots; {self._failures} load failures")

    # ------------------------------------------------------------------ access
    def __len__(self) -> int:
        return len(self.participants)

    def __getitem__(self, idx: int) -> dict:
        p = self.participants[idx]
        sessions = [self._crop(s) if s is not None else None for s in self.get_sessions(idx)]
        if self.session_drop_prob > 0:
            valid = [i for i, s in enumerate(sessions) if s is not None]
            if len(valid) > 1 and np.random.rand() < self.session_drop_prob:
                sessions = list(sessions)
                sessions[int(np.random.choice(valid))] = None
        if self.strict_data and not any(
            s is not None and (s["mask_audio"].any() or s["mask_video"].any()) for s in sessions
        ):
            raise RuntimeError(f"No valid audio/video frames for participant {p.file_id}")
        return {"file_id": p.file_id, "sessions": sessions, "y_a1": p.y_a1, "y_a2": p.y_a2}

    def _crop(self, s: dict) -> dict:
        T = len(s["mask_audio"])
        L = self.max_frames
        if not L or T <= L:
            return s
        start = np.random.randint(0, T - L + 1) if self.train else (T - L) // 2
        sl = slice(start, start + L)
        return {
            **s,
            "audio": {k: v[sl] for k, v in s["audio"].items()},
            "video": {k: v[sl] for k, v in s["video"].items()},
            "group_masks": {k: v[sl] for k, v in s["group_masks"].items()},
            "mask_audio": s["mask_audio"][sl],
            "mask_video": s["mask_video"][sl],
            "vad": s["vad"][sl],
            "qc": s["qc"][sl],
        }

    def text_coverage(self) -> tuple[int, int]:
        """(sessions with text, sessions expected to have text) — cheap existence check."""
        if self.text_root is None:
            return 0, 0
        f = self.cfg_features
        found = expected = 0
        for p in self.participants:
            for s in p.sessions:
                if s in f["text_sessions"]:
                    expected += 1
                    found += text_path(self.text_root, p.feature_split, p.school, p.cls, p.pid,
                                       f["text_tag"], s).exists()
        return found, expected


def _to_half(s: dict) -> dict:
    return {
        **s,
        "audio": {k: v.astype(np.float16) for k, v in s["audio"].items()},
        "video": {k: v.astype(np.float16) for k, v in s["video"].items()},
        "text": None if s.get("text") is None else s["text"].astype(np.float16),
    }


def compute_feature_stats(ds: ParticipantDataset, max_participants: int, seed: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Per-dimension mean/std over valid frames of a random participant subset."""
    rng = np.random.default_rng(seed)
    n = min(max_participants, len(ds))
    idx = rng.choice(len(ds), size=n, replace=False)
    acc: dict[str, list] = {}

    def _add(key: str, x: np.ndarray) -> None:
        if x.size == 0:
            return
        x = x.astype(np.float64)
        if key not in acc:
            acc[key] = [np.zeros(x.shape[1]), np.zeros(x.shape[1]), 0]
        acc[key][0] += x.sum(0)
        acc[key][1] += (x ** 2).sum(0)
        acc[key][2] += x.shape[0]

    for i in tqdm(idx, desc="Feature stats", dynamic_ncols=True):
        for s in ds.get_sessions(int(i)):
            if s is None:
                continue
            for modality in ("audio", "video"):
                for name, arr in s[modality].items():
                    _add(f"{modality}/{name}", arr[s["group_masks"][f"{modality}/{name}"]])
            for name, vec in s["pooled"].items():
                if s["pooled_present"][name]:
                    _add(f"pooled/{name}", vec[None, :])

    stats = {}
    for key, (sm, sq, cnt) in acc.items():
        mean = sm / max(cnt, 1)
        std = np.sqrt(np.maximum(sq / max(cnt, 1) - mean ** 2, 0.0))
        std = np.where(std < 1e-6, 1.0, std)
        stats[key] = (mean.astype(np.float32), std.astype(np.float32))
    return stats


def participant_labels(participants: list[Participant]) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.stack([p.y_a1 for p in participants]) if participants else np.zeros((0, 3), np.float32),
        np.stack([p.y_a2 for p in participants]) if participants else np.zeros((0, 21), np.float32),
    )


def pooled_dir_exists(ds: ParticipantDataset, p: Participant, name: str, session: str) -> bool:
    return pooled_exists(ds._dir(p, "audio", name, session, pooled=True))
