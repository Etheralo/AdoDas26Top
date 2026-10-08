#!/usr/bin/env python3
"""Check extraction completion, manifests and CUDA before allocating a run."""
from __future__ import annotations
import argparse
import json
import sys
from io import BytesIO
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from data.dataset import read_manifest
from data.splits import development_split
from training.config import load_config, check_required

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/server_stage1.yaml')
    ap.add_argument('--allow-cpu', action='store_true', help='Diagnostic checks only')
    ap.add_argument('--allow-partial-extraction', action='store_true', help='Diagnostic checks only')
    ap.add_argument('--set', nargs='*', default=[])
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    check_required(cfg)
    problems, info = [], {}
    manifest_dir = Path(cfg['paths']['manifest_dir'])
    status_path = manifest_dir / 'extract.status'
    status = status_path.read_text().strip() if status_path.exists() else 'MISSING'
    info['extraction_status'] = status
    if not args.allow_partial_extraction and 'COMPLETE_ARCHIVES_REMOVED' not in status:
        problems.append('Extraction and verification have not completed')
    official = development_split(cfg) == 'val'
    if official:
        p = manifest_dir / 'extract_evaluation.status.json'
        if not p.exists() or json.loads(p.read_text()).get('state') != 'COMPLETE_ARCHIVES_REMOVED':
            problems.append('Official validation/test extraction is incomplete')
    for split in ('train', 'val', 'test_hidden'):
        path = manifest_dir / f'{split}.csv'
        if path.exists():
            try:
                df = read_manifest(path)
                info[split + '_rows'] = len(df)
                info[split + '_participants'] = len(df[['anon_school', 'anon_class', 'anon_pid']].drop_duplicates())
                if split != 'test_hidden' and df.filter(regex=r'^(y_[DAS]|d\d\d)$').isna().any().any():
                    problems.append(f'{split}: missing labels')
            except Exception as exc:
                problems.append(f'{split}: {exc}')
        elif split == 'train' or (split == 'val' and cfg['split']['require_official_val']) or (official and split == 'test_hidden'):
            problems.append(f'{split}: missing manifest')
        else:
            info[split] = 'unavailable; no official result can be reported'
    info.update(python=sys.version.split()[0], torch=str(torch.__version__), cuda_runtime=torch.version.cuda,
                cuda_available=torch.cuda.is_available())
    # Exercise the actual parquet engine after torch has loaded its C++ runtime.
    # A standalone pyarrow import or pip check can miss an import-order conflict.
    try:
        import pandas as pd
        stream = BytesIO()
        pd.DataFrame({'probe': [1]}).to_parquet(stream, engine='pyarrow', index=False)
        stream.seek(0)
        if pd.read_parquet(stream, engine='pyarrow')['probe'].tolist() != [1]:
            raise RuntimeError('Parquet roundtrip differs')
        info['parquet_after_torch'] = 'passed'
    except Exception as exc:
        problems.append(f'Parquet engine unavailable after torch import: {exc}')
    info['recommended_environment'] = 'Python 3.10 / PyTorch 2.10.0; environment.yaml'
    if torch.cuda.is_available():
        try:
            info['gpu'] = torch.cuda.get_device_name(0)
            info['gpu_memory_GiB'] = torch.cuda.get_device_properties(0).total_memory / 2**30
            x = torch.ones(16, 16, device='cuda')
            assert torch.isfinite(x @ x).all()
            torch.cuda.synchronize()
            info['cuda_compute_test'] = 'passed'
        except Exception as exc:
            problems.append(f'CUDA compute test failed: {exc}')
    elif not args.allow_cpu:
        problems.append('No CUDA GPU available; formal training cannot start')
    for name in ('memory.max', 'cpu.max'):
        p = Path('/sys/fs/cgroup') / name
        if p.exists(): info['container_' + name] = p.read_text().strip()
    if not args.allow_cpu:
        memory = info.get('container_memory.max', 'max')
        quota = info.get('container_cpu.max', 'max 100000').split()
        if memory != 'max' and int(memory) < 16 * 2**30:
            problems.append('Container memory below 16 GiB; allocate more before multi-worker training')
        if quota[0] != 'max' and int(quota[0]) / int(quota[1]) < 4:
            problems.append('Container CPU quota below 4 cores; allocate more before training')
    info['problems'] = problems
    print(json.dumps(info, indent=2, ensure_ascii=False))
    return bool(problems)
if __name__ == '__main__':
    sys.exit(main())
