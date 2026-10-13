#!/usr/bin/env python3
"""Step 0: fetch the raw METR data into data/raw/.

  python pipeline/download_data.py [--force] [--raw-dir DIR]

Two sources (recorded in data/raw/MANIFEST.json):

1. METR's eval-analysis-public GitHub repo, pinned to the commit this
   archive was built from: the TH 1.1 and TH 1.0 runs.jsonl and
   release_dates.yaml.  A file already present with the pinned SHA-256
   is kept (--force re-downloads it); a download that does not match the
   pin is an error.
2. METR's live results file, https://metr.org/assets/task_results_1_1.yaml.
   This one is not pinned.  When the fetch succeeds it replaces the
   local copy and the data step uses it in preference to the TH 1.1
   runs.  The script compares it with the copy it replaces and warns
   when AIs have been added (the fits, scores, figures and paper then
   need rerunning) or removed.  When the fetch fails, the local copy
   stays (in a fresh checkout, the archived snapshot of 2026-10-06);
   with no local copy at all, the data step falls back to the TH 1.1
   runs.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import sys
import tempfile
import urllib.request
from pathlib import Path

import yaml

import _paths as P

MANIFEST = 'MANIFEST.json'
RAW_URL = 'https://raw.githubusercontent.com/{repo}/{commit}/{path}'
TIMEOUT_S = 60


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def fetch(url: str, dest_dir: Path):
    """Download `url` to a temporary file in `dest_dir`; (path, etag)."""
    req = urllib.request.Request(url, headers={'User-Agent': 'horizon_irt-reproduction'})
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
        etag = resp.headers.get('ETag')
        with tempfile.NamedTemporaryFile(dir=dest_dir, delete=False) as tmp:
            for chunk in iter(lambda: resp.read(1 << 20), b''):
                tmp.write(chunk)
    return Path(tmp.name), etag


def agents_of(path: Path) -> set:
    return set(yaml.safe_load(path.read_text())['agents'])


def github_files(manifest, raw_dir: Path, force: bool) -> bool:
    gh = manifest['github']
    ok = True
    for name, spec in gh['files'].items():
        dest = raw_dir / name
        if dest.exists() and not force and sha256(dest) == spec['sha256']:
            print(f'  {name}: present, matches the pin')
            continue
        url = RAW_URL.format(repo=gh['repo'], commit=gh['commit'],
                             path=spec['path'])
        print(f'  {name}: downloading {url}', flush=True)
        tmp, _ = fetch(url, raw_dir)
        got = sha256(tmp)
        if got != spec['sha256']:
            tmp.unlink()
            print(f'  ERROR {name}: SHA-256 {got} does not match the pin '
                  f'{spec["sha256"]}', file=sys.stderr)
            ok = False
            continue
        tmp.replace(dest)
        print(f'  {name}: ok ({spec["bytes"]:,} bytes)')
    return ok


def live_file(manifest, raw_dir: Path) -> None:
    live = manifest['live']
    dest = raw_dir / live['file']
    print(f"  {live['file']}: fetching {live['url']}", flush=True)
    tmp = None
    try:
        tmp, etag = fetch(live['url'], raw_dir)
        new_agents = agents_of(tmp)          # also checks it parses
    except Exception as exc:                 # network, HTTP or YAML error
        if tmp is not None and tmp.exists():
            tmp.unlink()
        if dest.exists():
            print(f'  WARNING: live fetch failed ({exc}); keeping the local '
                  f'copy retrieved {live["retrieved"]}.')
        else:
            print(f'  WARNING: live fetch failed ({exc}) and there is no '
                  'local copy; the data step will fall back to the TH 1.1 '
                  'runs.jsonl.')
        return

    digest = sha256(tmp)
    if dest.exists():
        old_agents = agents_of(dest)
        added = sorted(new_agents - old_agents)
        removed = sorted(old_agents - new_agents)
        if added:
            print(f'  WARNING: {len(added)} AI(s) new since the local copy '
                  f'of {live["retrieved"]}: {", ".join(added)}.  Rerun the '
                  'pipeline (make data fits eval scores figures paper); '
                  'results will differ from the archived ones.')
        if removed:
            print(f'  WARNING: {len(removed)} AI(s) no longer in the live '
                  f'file: {", ".join(removed)}.')
        if not added and not removed and digest != live['sha256']:
            print('  NOTE: same AIs, but the file content changed; results '
                  'for existing AIs may have been revised.')
        if digest == live['sha256'] and sha256(dest) == digest:
            tmp.unlink()
            print(f'  {live["file"]}: unchanged since {live["retrieved"]}')
            return
    tmp.replace(dest)
    live.update(sha256=digest, bytes=dest.stat().st_size,
                retrieved=datetime.date.today().isoformat(), etag=etag,
                n_agents=len(new_agents))
    print(f'  {live["file"]}: ok ({live["bytes"]:,} bytes, '
          f'{len(new_agents)} AIs)')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--raw-dir', type=Path, default=P.RAW)
    ap.add_argument('--force', action='store_true',
                    help='re-download the pinned GitHub files')
    args = ap.parse_args()
    raw_dir = args.raw_dir
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = raw_dir / MANIFEST
    if not manifest_path.exists():
        manifest_path.write_text((P.RAW / MANIFEST).read_text())
    manifest = json.loads(manifest_path.read_text())

    print(f"METR GitHub data at commit {manifest['github']['commit'][:10]}:")
    ok = github_files(manifest, raw_dir, args.force)
    print('METR live results:')
    live_file(manifest, raw_dir)
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
