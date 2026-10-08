#!/usr/bin/env python3
"""Monitor aggregate extraction progress without reading participant contents."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import time

TOTAL_BYTES = 813587749895  # Sum of uploaded ZIP central-directory file sizes.

def snapshot(root):
    paths = [root / name for name in ('features', 'transcripts') if (root / name).exists()]
    result = subprocess.run(['du', '-sb', *map(str, paths)], capture_output=True, text=True, check=True)
    written = sum(int(line.split('\t', 1)[0]) for line in result.stdout.splitlines())
    status = (root / 'extract.status').read_text().strip()
    done = []
    for line in (root / 'extract.log').read_text().splitlines():
        if ' DONE ' in line: done.append(line.rsplit(' ', 1)[-1])
    info = dict(time=datetime.now().astimezone().isoformat(), status=status, verified_archives=len(done),
                total_archives=11, extracted_GiB=round(written / 2**30, 2),
                expected_GiB=round(TOTAL_BYTES / 2**30, 2),
                approximate_percent=round(min(100, written / TOTAL_BYTES * 100), 2))
    output = root / 'progress.json'
    tmp = output.with_suffix('.tmp')
    tmp.write_text(json.dumps(info, indent=2) + '\n')
    tmp.replace(output)
    return info

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=Path('/root/autodl-tmp/dataset-ai'))
    ap.add_argument('--watch', action='store_true')
    ap.add_argument('--interval', type=int, default=300)
    args = ap.parse_args()
    while True:
        info = snapshot(args.root)
        print(json.dumps(info), flush=True)
        if not args.watch or 'FAILED' in info['status'] or 'COMPLETE_ARCHIVES_REMOVED' in info['status']:
            break
        time.sleep(max(30, args.interval))
if __name__ == '__main__': main()
