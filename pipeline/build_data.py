#!/usr/bin/env python3
"""Step 1: raw METR results -> the cell-level dataset in data/processed/.

  python pipeline/build_data.py

Writes data/processed/canonical.csv (one row per task x AI cell) and
canonical_info.json (counts, sources, grafted models, metadata
disagreements).  The dataset is built from METR's live results file
(pipeline/download_data.py) and falls back to the TH 1.1 runs.jsonl if
that file is missing.  See horizon_irt.data for what it contains.
"""
from __future__ import annotations

import argparse
import json

import _paths as P
from horizon_irt import data


def main():
    argparse.ArgumentParser(description=__doc__.split('\n')[0]).parse_args()
    P.PROCESSED.mkdir(parents=True, exist_ok=True)
    name = 'canonical'
    cells, info = data.build_dataset(name, P.RAW)
    if info['primary_source'] != data.RAW_FILES['task_results']:
        print('WARNING: data/raw/task_results_1_1.yaml is missing; '
              'falling back to the TH 1.1 runs.jsonl.')
    out = P.cells_path(name)
    if out.exists():
        before = set(data.read_cells(out)['alias'])
        after = set(cells['alias'])
        for verb, ais in (('added', after - before),
                          ('removed', before - after)):
            if ais:
                print(f'WARNING: AIs {verb} since the previous build: '
                      f'{", ".join(sorted(ais))}')
    cells.to_csv(out, index=False)
    (P.PROCESSED / f'{name}_info.json').write_text(
        json.dumps(info, indent=2) + '\n')
    print(f"{name}: {info['n_models']} AIs, {info['n_tasks']} tasks, "
          f"{info['n_families']} families, {info['n_runs']} runs "
          f"(grafted {', '.join(info['grafted_models'])}; "
          f"{len(info['metadata_diffs'])} metadata disagreements)")


if __name__ == '__main__':
    main()
