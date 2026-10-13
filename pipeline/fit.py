#!/usr/bin/env python3
"""Step 2: fit specifications on a dataset -> artifacts/canonical/fits/.

  python pipeline/fit.py [--specs paper|all|ablations|name,...] [--njob 8]

`paper` (default) is the paper's four specifications plus `linear`;
`all` is the full 52-spec grid (about 20 minutes on 8 cores);
`ablations` is the named ablation variants (specs.ablation_specs()).
--specs also takes a comma-separated list mixing these keywords and
spec names, e.g. `paper,ablations`.  Fits run in parallel, slowest
first.
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor

import _paths as P
from horizon_irt import data, fitting
from horizon_irt.specs import (ABLATION_SPECS, PIPELINE_SPECS, all_specs,
                               fit_order, registry)

KEYWORDS = {'all': lambda: list(all_specs()),
            'paper': lambda: list(PIPELINE_SPECS),
            'ablations': lambda: list(ABLATION_SPECS)}


def select_specs(arg):
    names = []
    for item in (s.strip() for s in arg.split(',')):
        for n in (KEYWORDS[item]() if item in KEYWORDS else [item]):
            if n not in names:
                names.append(n)
    reg = registry()
    unknown = [n for n in names if n not in reg]
    if unknown:
        raise SystemExit(f'unknown specs: {unknown}')
    return fit_order(names)


def _worker(job):
    name, cells_path, out_dir, kw = job
    try:
        idx = data.load_cells(cells_path)
        s = fitting.fit_and_save(name, idx, out_dir, **kw)
        return name, 'ok', s['logl'], s['wall_s']
    except Exception:
        return name, 'FAIL', traceback.format_exc(), 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--specs', default='paper')
    ap.add_argument('--out', default=None,
                    help='output dir (default artifacts/canonical/fits)')
    ap.add_argument('--k', type=int, default=1000,
                    help='Gauss-Hermite nodes for the task effect')
    ap.add_argument('--kg', type=int, default=80,
                    help='Gauss-Hermite nodes for the family effect')
    ap.add_argument('--em-max-iter', type=int, default=300,
                    help='cap on EM updates per IRT fit')
    ap.add_argument('--em-accel', choices=('squarem', 'none'),
                    default='squarem',
                    help="EM driver: SQUAREM, or 'none' for plain EM")
    ap.add_argument('--njob', type=int, default=8)
    args = ap.parse_args()

    names = select_specs(args.specs)
    out = P.fits_dir() if args.out is None else args.out
    kw = dict(K=args.k, Kg=args.kg, max_iter=args.em_max_iter,
              accel=args.em_accel)
    jobs = [(n, str(P.cells_path()), str(out), kw) for n in names]
    print(f'fitting {len(jobs)} specs on {P.DATASET} -> {out} '
          f'(njob={args.njob}, K={args.k}, Kg={args.kg}, '
          f'EM accel={args.em_accel})', flush=True)
    t0 = time.perf_counter()
    n_fail = 0
    # classic specs read zm_homo's fit, so they must not start before it
    # is saved; fit_order puts them last, but a pool may still overlap.
    early = [j for j in jobs if not j[0].startswith('classic')]
    late = [j for j in jobs if j[0].startswith('classic')]
    for batch in (early, late):
        if not batch:
            continue
        with ProcessPoolExecutor(max_workers=args.njob) as ex:
            for name, status, info, wall in ex.map(_worker, batch):
                if status == 'ok':
                    print(f'  [ok] {name:34s} logL={info:12.2f} '
                          f'({wall:6.1f}s)', flush=True)
                else:
                    n_fail += 1
                    print(f'  [FAIL] {name}\n{info}', flush=True)
    print(f'done in {time.perf_counter() - t0:.0f}s; '
          f'{len(jobs) - n_fail} ok, {n_fail} failed')
    sys.exit(1 if n_fail else 0)


if __name__ == '__main__':
    main()
