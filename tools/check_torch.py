#!/usr/bin/env python3
"""Verify the training interpreter and CUDA forward/backward operations."""
import json
import os
import sys

import torch
import torch.nn.functional as F


def main():
    info = {
        "interpreter": sys.executable,
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_path": torch.__file__,
        "cuda_runtime": torch.version.cuda,
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_available": torch.cuda.is_available(),
    }
    try:
        if not info["cuda_available"]:
            raise RuntimeError("CUDA is unavailable in the training interpreter")
        info["gpu"] = torch.cuda.get_device_name(0)
        info["capability"] = torch.cuda.get_device_capability(0)
        info["compiled_architectures"] = torch.cuda.get_arch_list()
        torch.manual_seed(2026)
        layer = torch.nn.Linear(64, 64, device="cuda")
        optimizer = torch.optim.AdamW(layer.parameters(), lr=1e-3)
        x = torch.randn(4, 16, 64, device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y = F.scaled_dot_product_attention(layer(x), x, x)
            loss = y.float().square().mean()
        loss.backward()
        if not torch.isfinite(loss) or not all(
            p.grad is not None and torch.isfinite(p.grad).all()
            for p in layer.parameters()
        ):
            raise RuntimeError("Nonfinite loss or gradients")
        optimizer.step()
        torch.cuda.synchronize()
        info.update(loss=loss.item(), cuda_forward_backward_adamw="PASS", status="PASS")
    except Exception as exc:
        info.update(status="FAIL", error=str(exc))
    print(json.dumps(info, indent=2))
    return int(info["status"] != "PASS")


if __name__ == "__main__":
    sys.exit(main())
