#!/usr/bin/env python3
"""Prepare a small private manifest from an already verified school archive.

Only diagnostic outputs are produced. This never changes official manifests.
"""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import yaml
from data.dataset import read_manifest, build_participants
from data.feature_io import feature_dir, pooled_exists
from training.config import load_config

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/server_stage1.yaml')
    ap.add_argument('--participants', type=int, default=8)
    args = ap.parse_args()
    cfg = load_config(args.config)
    root = Path(cfg['paths']['manifest_dir'])
    if ' DONE SCH_001.zip' not in (root / 'extract.log').read_text():
        raise RuntimeError('SCH_001.zip has not finished verification')
    df = read_manifest(root / 'train.csv')
    candidates = build_participants(df[df.anon_school == 'SCH_001'], 'train')
    f = cfg['features']
    tags = {'ssl_embed': f['audio_ssl_tag'], 'vision_ssl_embed': f['video_ssl_tag']}
    selected = []
    for p in candidates:
        ok = True
        for session in p.sessions:
            for modality in ('audio', 'video'):
                for feature in f[modality]:
                    d = feature_dir(cfg['paths']['feature_root'], 'train', p.school, p.cls, p.pid,
                                    modality, feature, session, tags.get(feature))
                    ok = ok and (d / 'sequence.npz').is_file()
            for feature in f['audio_pooled']:
                d = feature_dir(cfg['paths']['feature_root'], 'train', p.school, p.cls, p.pid,
                                'audio', feature, session)
                ok = ok and pooled_exists(d)
        if ok and len(p.sessions) == 4: selected.append((p.school, p.cls, p.pid))
        if len(selected) == args.participants: break
    if len(selected) < 4:
        raise RuntimeError('Too few complete participants for a meaningful pipeline smoke')
    out = Path(cfg['paths']['output_dir']) / 'readiness'
    manifest_dir = out / 'smoke_manifests'
    manifest_dir.mkdir(parents=True, exist_ok=True)
    selected_set = set(selected)
    rows = df.apply(lambda r: (r.anon_school, r.anon_class, r.anon_pid) in selected_set, axis=1)
    df[rows].to_csv(manifest_dir / 'train.csv', index=False)
    cfg['paths'].update(manifest_dir=str(manifest_dir), output_dir=str(out / 'cpu_smoke'))
    cfg['features'].update(max_frames=64, eval_max_frames=64)
    cfg['model'].update(d_model=64, d_session=64, conformer_layers=1,
                        cross_modal_layers=1, cross_session_layers=1, query_layers=1)
    cfg['split'].update(inner_dev_ratio=0.25, require_official_val=False)
    cfg['train'].update(batch_size=1, eval_batch_size=1, grad_accum_steps=2, num_workers=0,
                        require_cuda=False, stats_participants=2, epochs=1, min_epochs=1, amp=False)
    path = out / 'cpu_smoke.yaml'
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f'Prepared diagnostic config {path}: {len(selected)} complete participants; reduced CPU-only architecture')
if __name__ == '__main__': main()
