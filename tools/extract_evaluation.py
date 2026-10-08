#!/usr/bin/env python3
"""Sequential evaluation ZIP extraction, CRC via unzip, size verification, deletion."""
import argparse
import fcntl
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import time
import zipfile


def save(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='/root/autodl-tmp/dataset-ai')
    ap.add_argument('--password-file', required=True)
    args = ap.parse_args()
    root = Path(args.root)
    lock = (root / 'extract_evaluation.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    password_file = Path(args.password_file)
    password = password_file.read_text().strip()
    status = root / 'extract_evaluation.status.json'
    plan_path = root / 'extract_evaluation.plan.json'
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
    else:
        plan = []
        for folder in ('Val', 'Test'):
            for archive in sorted((root / folder).rglob('*.zip')):
                target = root / ('transcripts' if 'transcript' in archive.name else
                                 'features' if archive.name.startswith('SCH_') else 'references')
                with zipfile.ZipFile(archive) as z:
                    entries = [i for i in z.infolist() if not i.is_dir()]
                    for i in z.infolist():
                        parts = PurePosixPath(i.filename)
                        if parts.is_absolute() or '..' in parts.parts or stat.S_ISLNK(i.external_attr >> 16):
                            raise RuntimeError(f'Unsafe ZIP entry: {i.filename}')
                    plan.append({'archive': str(archive), 'target': str(target),
                                 'compressed_bytes': archive.stat().st_size,
                                 'expanded_bytes': sum(i.file_size for i in entries),
                                 'files': len(entries), 'verified': False, 'deleted': False})
        if not plan: raise RuntimeError('No evaluation archives found')
        save(plan_path, plan)
    try:
        for index, item in enumerate(plan):
            archive, target = Path(item['archive']), Path(item['target'])
            if item['deleted']: continue
            if item['verified']:
                archive.unlink(missing_ok=True)
                item['deleted'] = True
                save(plan_path, plan)
                continue
            free = shutil.disk_usage(root).free
            # Add generous file/directory overhead and reserve 5 GiB.
            required = item['expanded_bytes'] + item['files'] * 8192 + 5 * 2**30
            if free < required: raise RuntimeError(f'Insufficient space for {archive.name}')
            target.mkdir(parents=True, exist_ok=True)
            save(status, {'state': 'EXTRACTING', 'index': index+1, 'total': len(plan),
                          'archive': str(archive), 'started_at': time.time(),
                          'expected_bytes': item['expanded_bytes'], 'free_bytes': free,
                          'completed_archives': sum(p['deleted'] for p in plan)})
            print(f'[{index+1}/{len(plan)} EXTRACT {archive} -> {target}', flush=True)
            with zipfile.ZipFile(archive) as z:
                entries = [i for i in z.infolist() if not i.is_dir()]
                # unzip checks each entry CRC while writing. Nonzero exit retains ZIP.
                result = subprocess.run(['unzip', '-o', '-q', '-P', password, str(archive), '-d', str(target)])
                if result.returncode:
                    raise RuntimeError(f'unzip failed for {archive.name}: exit {result.returncode}; ZIP retained')
                save(status, {'state': 'VERIFYING', 'index': index+1, 'total': len(plan), 'archive': str(archive)})
                for entry in entries:
                    output = target / entry.filename
                    if not output.is_file() or output.stat().st_size != entry.file_size:
                        raise RuntimeError(f'Extracted file missing or wrong size: {output}')
            item['verified'] = True
            save(plan_path, plan)
            archive.unlink()
            item['deleted'] = True
            save(plan_path, plan)
            print(f'[{index+1}/{len(plan)} VERIFIED_AND_DELETED free_GiB={shutil.disk_usage(root).free/2**30:.2f}', flush=True)
        val = root / 'Val/val.csv'
        manifest = root / 'val.csv'
        if val.exists():
            if manifest.exists() and manifest.read_bytes() != val.read_bytes():
                raise RuntimeError('Existing root val.csv differs; retained both copies')
            if not manifest.exists(): shutil.copy2(val, manifest)
        save(status, {'state': 'COMPLETE_ARCHIVES_REMOVED', 'completed_archives': len(plan),
                      'completed_at': time.time(), 'free_bytes': shutil.disk_usage(root).free})
        password_file.unlink(missing_ok=True)
        print('COMPLETE_ARCHIVES_REMOVED', flush=True)
    except BaseException as exc:
        save(status, {'state': 'FAILED', 'error': str(exc), 'updated_at': time.time()})
        raise

if __name__ == '__main__': main()
