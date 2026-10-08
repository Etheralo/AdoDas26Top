#!/usr/bin/env python3
"""Self-checks that need no data: metrics vs. scikit-learn, CORAL threshold
monotonicity, mask invariance, and a synthetic forward/backward pass.

    python tools/self_test.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.collate import make_collate  # noqa: E402
from models.heads import CoralThresholds  # noqa: E402
from models.model import build_model, compute_priors  # noqa: E402
from training.calibration import calibrate_a1, calibrate_a2, decode_a2  # noqa: E402
from training.config import DEFAULTS  # noqa: E402
from training.losses import LossComputer  # noqa: E402
from training.metrics import auroc, binary_f1, qwk  # noqa: E402

DIMS = {"audio": {"mel_mfcc": 93, "vad": 1}, "video": {"qc_stats": 4, "vision_ssl_embed": 32},
        "pooled": {"egemaps": 88}, "text": 24}


def check_metrics(rng: np.random.Generator) -> None:
    from sklearn.metrics import cohen_kappa_score, f1_score, roc_auc_score

    y = rng.integers(0, 2, 300)
    s = rng.random(300)
    assert abs(binary_f1(y, s >= 0.5) - f1_score(y, s >= 0.5)) < 1e-9
    s_tied = np.round(s, 1)
    assert abs(auroc(y, s_tied) - roc_auc_score(y, s_tied)) < 1e-9
    a, b = rng.integers(0, 4, 500), rng.integers(0, 4, 500)
    assert abs(qwk(a, b) - cohen_kappa_score(a, b, weights="quadratic")) < 1e-9
    print("metrics match scikit-learn")


def fake_session(rng: np.random.Generator, T: int) -> dict:
    s = {"audio": {n: rng.standard_normal((T, d)).astype(np.float32) for n, d in DIMS["audio"].items()},
         "video": {n: rng.standard_normal((T, d)).astype(np.float32) for n, d in DIMS["video"].items()}}
    s["mask_audio"] = rng.random(T) > 0.1
    s["mask_video"] = rng.random(T) > 0.3
    s["vad"] = (rng.random(T) > 0.5).astype(np.float32)
    s["qc"] = rng.random(T).astype(np.float32)
    s["pooled"] = {"egemaps": rng.standard_normal(88).astype(np.float32)}
    s["pooled_present"] = {"egemaps": True}
    s["text"] = rng.standard_normal((int(rng.integers(5, 20)), DIMS["text"])).astype(np.float32)
    return s


def fake_batch(rng: np.random.Generator, B: int) -> dict:
    items = []
    for b in range(B):
        sessions = [fake_session(rng, int(rng.integers(20, 60))) for _ in range(4)]
        sessions[0]["text"] = None
        if b == 0:
            sessions[2] = None  # a missing session
        items.append({"file_id": f"s_c_{b}", "sessions": sessions,
                      "y_a1": rng.integers(0, 2, 3).astype(np.float32),
                      "y_a2": rng.integers(0, 4, 21).astype(np.float32)})
    return make_collate(DIMS)(items)


def main() -> None:
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    check_metrics(rng)

    thr = CoralThresholds(21)
    thr.init_from_prior(np.sort(rng.random((21, 3)), axis=1)[:, ::-1] * 0.5)
    assert torch.all(thr()[:, 1:] > thr()[:, :-1]), "CORAL thresholds must increase"
    print("CORAL thresholds increase")

    m = dict(DEFAULTS["model"], use_text=True, d_model=64, d_session=64, d_text=32, conformer_layers=2)
    batch = fake_batch(rng, 3)
    priors = compute_priors(batch["y_a1"].numpy(), batch["y_a2"].numpy())
    model = build_model(m, DIMS, None, priors)
    model.eval()
    with torch.no_grad():
        out1 = model(batch)
        # Changing masked-out frames must not change the output.
        b2 = {**batch, "audio": {k: v.clone() for k, v in batch["audio"].items()}}
        for v in b2["audio"].values():
            v[~batch["mask_audio"]] = 1e3
        out2 = model(b2)
    for k in ("a1", "a2"):
        assert torch.allclose(out1[k], out2[k], atol=1e-4), f"masked frames leak into {k}"
    assert out1["a1"].shape == (3, 3) and out1["a2"].shape == (3, 21, 3)
    print("mask invariance and output shapes OK")

    model.train()
    losses = LossComputer(DEFAULTS["loss"], batch["y_a1"].numpy(), batch["y_a2"].numpy(), torch.device("cpu"))
    loss, parts = losses(model(batch), batch)
    loss.backward()
    assert torch.isfinite(loss), parts
    n_grad = sum(p.grad is not None for p in model.parameters() if p.requires_grad)
    print(f"forward/backward OK: loss {float(loss):.4f} {parts}; {n_grad} parameters received gradients")

    model.freeze_encoder()
    trainable = {n.split(".")[0] for n, p in model.named_parameters() if p.requires_grad}
    assert trainable <= {"a1_head", "a2_head", "a1_guidance"}, trainable
    print(f"stage-2 freeze keeps only {sorted(trainable)} trainable")

    logits = rng.standard_normal((200, 3))
    labels = rng.integers(0, 2, (200, 3)).astype(np.float32)
    assert calibrate_a1(logits, labels)["f1"] >= 0
    a2_logits = rng.standard_normal((200, 21, 3))
    cal = calibrate_a2(a2_logits, rng.integers(0, 4, (200, 21)).astype(np.float32))
    assert decode_a2(a2_logits, cal["method"]).max() <= 3
    print(f"calibration OK (A2 decode selected: {cal['method']})")
    print("ALL SELF-TESTS PASSED")


if __name__ == "__main__":
    main()
