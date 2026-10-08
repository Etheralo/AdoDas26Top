#!/usr/bin/env python3
"""Regression checks for official-val selection and test-label isolation."""
from __future__ import annotations
import copy
import json
import runpy
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
import yaml
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.splits import load_participant_sets
from training.config import DEFAULTS
from training.trainer import fit
from infer import finalize_run


def manifest(start, count, labeled=True):
    rows = []
    for i in range(start, start+count):
        for session in ('A01', 'B01', 'B02', 'B03'):
            row = dict(anon_school='SCH', anon_class='CLS', anon_pid=f'P{i:03}', session=session)
            if labeled:
                row.update(y_D=i%2, y_A=0, y_S=0, **{f'd{j:02}': i%4 for j in range(1,22)})
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        manifest(0,20).to_csv(root/'train.csv',index=False)
        manifest(20,8).to_csv(root/'val.csv',index=False)
        manifest(28,4,False).to_csv(root/'test_hidden.csv',index=False)
        cfg = copy.deepcopy(DEFAULTS)
        cfg['paths'].update(manifest_dir=str(root), output_dir=str(root/'outputs'))
        cfg['split']['development_source'] = 'official_val'
        sets, _ = load_participant_sets(cfg, include_test=True)
        assert len(sets['train']) == 20 and len(sets['val']) == 8 and not sets['inner_dev']
        assert all((p.y_a1 == -1).all() and (p.y_a2 == -1).all() for p in sets['test_hidden'])
        bad = copy.deepcopy(cfg); bad['split']['final'] = True
        try: load_participant_sets(bad)
        except ValueError: pass
        else: raise AssertionError('Validation allowed into training')
        manifest(28,4).to_csv(root/'test_hidden.csv',index=False)
        try: load_participant_sets(cfg, include_test=True)
        except ValueError: pass
        else: raise AssertionError('Test labels entered training/inference')
        manifest(28,4,False).to_csv(root/'test_hidden.csv',index=False)

        rng = np.random.default_rng(9)
        pred = dict(file_ids=np.array([f'SCH_CLS_P{i:03}' for i in range(20,28)]),
                    a1_logits=rng.normal(size=(8,3)).astype('float32'),
                    a2_logits=np.sort(rng.normal(size=(8,21,3)),axis=-1)[...,::-1].astype('float32'),
                    y_a1=rng.integers(0,2,(8,3)), y_a2=rng.integers(0,4,(8,21)))
        cfg['train'].update(selection_task='a1', selection_metric='raw', epochs=1, warmup_epochs=0)
        run=root/'run'; (run/'checkpoints').mkdir(parents=True)
        with patch('training.trainer.train_one_epoch',return_value={'loss':1.}), \
             patch('training.trainer.predict',return_value=pred) as predict:
            result=fit(torch.nn.Linear(1,1), {'train':[None],'val':[None]}, None,
                       cfg,torch.device('cpu'),False,run,{})
        assert predict.call_count == 1 and result['history'][0]['selection']['split'] == 'val'
        assert 'inner_dev' not in result['history'][0]

        members=[]
        for index in range(2):
            member=root/f'member{index}'; (member/'predictions').mkdir(parents=True)
            (member/'submissions').mkdir(); (member/'logs').mkdir()
            (member/'config_used.yaml').write_text(yaml.safe_dump(cfg))
            for tag in ('a1','a2'):
                np.savez(member/'predictions'/f'val_{tag}.npz',**pred)
                test={k:v[:4].copy() for k,v in pred.items()}
                test['file_ids']=np.array([f'SCH_CLS_P{i:03}' for i in range(28,32)])
                test['y_a1']=np.full((4,3),-1); test['y_a2']=np.full((4,21),-1)
                np.savez(member/'predictions'/f'test_hidden_{tag}.npz',**test)
            report=finalize_run(member,cfg)
            assert report['calibration_split']=='val' and 'val_held_out' not in report
            assert 'val_used_for_selection' in report and not any(k.startswith('test') for k in report)
            members.append(str(member))
        ensemble_cfg={'members':members,'output_dir':str(root/'ensemble'), 'primary_task':'a2',
                      'a2':{'expert_selection':True,'min_gain':0.0}}
        path=root/'ensemble.yaml'; path.write_text(yaml.safe_dump(ensemble_cfg))
        script=Path(__file__).resolve().parents[1]/'ensemble.py'
        # Reuse the interpreter: two concurrent torch imports exceed a 2-GiB container.
        with patch.object(sys, 'argv', [str(script), '--config', str(path)]):
            runpy.run_path(str(script), run_name='__main__')
        report=json.loads(next((root/'ensemble').glob('runs/*/report.json')).read_text())
        assert report['calibration_split']=='val' and 'val_used_for_selection' in report
        assert len(report['a2_items'])==21 and 'val_held_out' not in report
    print('PASS: official-val selection, full training split, validation exclusion, test-label isolation, exports and 21-item expert selection')

if __name__=='__main__': main()
