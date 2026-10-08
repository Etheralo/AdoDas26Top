#!/usr/bin/env python3
"""CPU checks for per-track gradients, freezing, configs and selection objectives."""
from __future__ import annotations
import copy
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.self_test import DIMS, fake_batch
from models.model import build_model
from training.config import DEFAULTS, load_config
from training.losses import LossComputer
from training.trainer import fit


def main():
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[1]
    batch = fake_batch(np.random.default_rng(7), 2)
    for task in ('a1', 'a2'):
        cfg = load_config(root / f'configs/aligned_{task}.yaml')
        assert cfg['train']['init_checkpoint'] is None
        assert cfg['train']['selection_task'] == task
        m = dict(cfg['model'], d_model=32, d_session=32, d_text=32, conformer_layers=min(1, cfg['model']['conformer_layers']))
        model = build_model(m, DIMS)
        lc = LossComputer(cfg['loss'], batch['y_a1'].numpy(), batch['y_a2'].numpy(), torch.device('cpu'))
        loss, _ = lc(model(batch), batch)
        loss.backward()
        assert torch.isfinite(loss)
        assert any(p.grad is not None for p in model.cross_session.parameters())
        if task == 'a2':
            assert all(p.grad is None for p in model.a1_head.parameters())
            assert any(p.grad is not None for p in model.a2_head.parameters())
        else:
            assert model.use_text and len(model.audio_enc.blocks) == 0
            assert any(p.grad is not None for p in model.text.parameters())
            assert any(p.grad is not None for p in model.a1_guidance.parameters())
            model.eval()
            tokens = torch.randn(2, 6, DIMS['text'])
            mask = torch.ones(2, 6, dtype=torch.bool)
            present = torch.ones(2, dtype=torch.bool)
            session = torch.tensor([1, 2])
            with torch.no_grad():
                ordered = model.text(tokens, mask, present, session)
                reversed_text = model.text(tokens.flip(1), mask, present, session)
            assert not torch.allclose(ordered, reversed_text, atol=1e-6), 'Text word order lost'
            model.zero_grad(set_to_none=True)
            model.freeze_encoder('a1')
            model.train()
            lc2_cfg = load_config(root / 'configs/aligned_a1_stage2.yaml')['loss']
            lc2 = LossComputer(lc2_cfg, batch['y_a1'].numpy(), batch['y_a2'].numpy(), torch.device('cpu'))
            lc2(model(batch), batch)[0].backward()
            trainable = {n.split('.')[0] for n,p in model.named_parameters() if p.requires_grad}
            assert trainable == {'a1_head'}
            assert not model.cross_session.training and model.a1_head.training
            assert all(p.grad is None for p in model.a2_head.parameters())
        print(f'{task}: finite forward/backward, expected active gradients and freeze OK')

    # A1 improves while A2 and their sum decrease: A1 early stopping must not
    # follow A2, the sum, or the contradictory calibrated scores.
    rows = []
    for i in range(3):
        rows.append({'a1_raw': {'f1': .2+.1*i, 'auroc': .6},
                     'a1_cal': {'f1': .8-.1*i},
                     'a2_raw': {'qwk': .6-.2*i},
                     'a2_cal': {'qwk': .1+.1*i, 'mae': .5}})
    for task, metric, expected_len in [('a1', 'raw', 3), ('a2', 'raw', 2), ('a2', 'calibrated', 3)]:
        cfg = copy.deepcopy(DEFAULTS)
        cfg['train'].update(selection_task=task, selection_metric=metric, epochs=3,
                            warmup_epochs=0, min_epochs=1, patience=1)
        pred = {'a1_logits': np.zeros((2,3)), 'a2_logits': np.zeros((2,21,3)),
                'y_a1': np.zeros((2,3)), 'y_a2': np.zeros((2,21))}
        with tempfile.TemporaryDirectory() as td:
            run = Path(td); (run/'checkpoints').mkdir()
            with patch('training.trainer.train_one_epoch', return_value={'loss': 1.}), \
                 patch('training.trainer.predict', return_value=pred), \
                 patch('training.trainer.calibrate_a1', return_value={}), \
                 patch('training.trainer.calibrate_a2', return_value={}), \
                 patch('training.trainer.score_predictions', side_effect=rows):
                out = fit(torch.nn.Linear(1,1), {'train':[None], 'inner_dev':[None]}, None,
                          cfg, torch.device('cpu'), False, run, {})
            assert len(out['history']) == expected_len
            assert out['history'][0]['selection']['metric'] == metric
        print(f'{task}: checkpoint/early stopping follows requested {metric} objective OK')
    print('ALL ALIGNMENT CHECKS PASSED')

if __name__ == '__main__': main()
