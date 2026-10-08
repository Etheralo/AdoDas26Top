#!/usr/bin/env python3
"""Check that parallel chunks preserve participant/session order and padding."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.utils.data import Dataset
from training.trainer import make_loader


class SyntheticParticipants(Dataset):
    dims = {"audio": {"a": 3}, "video": {"v": 2}, "pooled": {"p": 2}, "text": 4}

    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        sessions = []
        for s in range(4):
            if (i+s) % 4 == 0:
                sessions.append(None)
                continue
            n = 1 + (i+s) % 7
            sessions.append({"audio": {"a": np.full((n,3), i+s, np.float32)},
                             "video": {"v": np.ones((n,2), np.float32)},
                             "mask_audio": np.ones(n,bool), "mask_video": np.arange(n)%2==0,
                             "vad": np.ones(n,np.float32), "qc": np.zeros(n,np.float32),
                             "pooled": {"p": np.ones(2,np.float32)}, "pooled_present": {"p": i%2==0},
                             "text": np.ones((1+i%3,4),np.float32)})
        return {"file_id": f"synthetic_{i}", "sessions": sessions,
                "y_a1": np.full(3,i%2,np.float32), "y_a2": np.full(21,i%4,np.float32)}


def equal(a, b):
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a: equal(a[k], b[k])
    elif isinstance(a, torch.Tensor):
        assert torch.equal(a,b)
    else:
        assert a == b


def main():
    torch.set_num_threads(1)
    for n, drop_last in ((11,False),(11,True),(7,True)):
        ds = SyntheticParticipants(n)
        reference = make_loader(ds,8,False,0,42,drop_last=drop_last)
        chunked = make_loader(ds,8,False,2,42,drop_last=drop_last,prefetch_factor=2,worker_batch_size=2)
        expected, actual = list(reference), list(chunked)
        assert len(actual) == len(expected) == len(chunked)
        for a,b in zip(expected,actual):
            equal(a,b)
            if torch.cuda.is_available(): assert b["audio"]["a"].is_pinned()
        chunked.loader._iterator._shutdown_workers()
    print("Chunked loader equals reference: participant/session order, labels, masks, padding, text and short/drop-last tails")


if __name__ == "__main__":
    main()
