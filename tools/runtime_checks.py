#!/usr/bin/env python3
"""Regression checks for train-only splits and optimizer accumulation."""
from __future__ import annotations
import copy
import sys
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
from data.dataset import read_manifest
from data.splits import load_participant_sets
from training.config import DEFAULTS
from training.trainer import train_one_epoch

def check_splits(root):
    rows = []
    for i in range(20):
        for session in ('A01', 'B01', 'B02', 'B03'):
            rows.append(dict(anon_school='SCH', anon_class='CLS', anon_pid=f'P{i:03}', session=session,
                             y_D=i % 2, y_A=0, y_S=0, **{f'd{j:02}': i % 4 for j in range(1,22)}))
    df = pd.DataFrame(rows)
    df.to_csv(root / 'train.csv', index=False)
    cfg = copy.deepcopy(DEFAULTS)
    cfg['paths'].update(manifest_dir=str(root), output_dir=str(root / 'outputs'))
    cfg['split']['require_official_val'] = False
    sets, path = load_participant_sets(cfg)
    assert len(sets['train']) == 18 and len(sets['inner_dev']) == 2 and not sets['val']
    again, _ = load_participant_sets(cfg, path)
    assert [p.file_id for p in again['inner_dev']] == [p.file_id for p in sets['inner_dev']]
    assert not ({p.file_id for p in sets['train']} & {p.file_id for p in sets['inner_dev']})
    cfg['split']['require_official_val'] = True
    try: load_participant_sets(cfg)
    except FileNotFoundError: pass
    else: raise AssertionError('Required official val was silently ignored')
    pd.concat([df, df.iloc[:1]]).to_csv(root / 'duplicates.csv', index=False)
    try: read_manifest(root / 'duplicates.csv')
    except ValueError: pass
    else: raise AssertionError('Duplicate manifest row accepted')
    print('Train-only/disjoint/reused split and missing-val/duplicate guards passed')

def check_accumulation():
    torch.manual_seed(4)
    model = torch.nn.Linear(2, 1, bias=False)
    reference = copy.deepcopy(model)
    x, y = torch.randn(5, 2), torch.randn(5, 1)
    batches = [{'x': x[i:i+1], 'y': y[i:i+1]} for i in range(5)]
    class Loss:
        def __call__(self, out, batch):
            loss = ((out - batch['y']) ** 2).mean()
            return loss, {}
    class Model(torch.nn.Module):
        def __init__(self, layer):
            super().__init__(); self.layer = layer
        def forward(self, batch): return self.layer(batch['x'])
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1.0)
    train_one_epoch(Model(model), batches, opt, sched, Loss(), torch.device('cpu'), False,
                    10000.0, 'Accumulation test', grad_accum_steps=2)
    ref_opt = torch.optim.SGD(reference.parameters(), lr=0.1)
    for start in range(0, 5, 2):
        ref_opt.zero_grad()
        ((reference(x[start:start+2]) - y[start:start+2]) ** 2).mean().backward()
        ref_opt.step()
    assert torch.allclose(model.weight, reference.weight, atol=1e-6)
    assert sched.last_epoch == 3
    print('Accumulation matches larger batches, including final short window; optimizer/scheduler steps = 3')

def main():
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory() as d: check_splits(Path(d))
    check_accumulation()
    print('ALL RUNTIME CHECKS PASSED')
if __name__ == '__main__': main()
