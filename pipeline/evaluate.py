#!/usr/bin/env python3
"""Step 3: evaluation battery -> artifacts/canonical/cv/.

  python pipeline/evaluate.py [--mode cv|insample] [--specs paper|all|...]

--mode cv (the paper's evaluation) refits every spec on each training
fold of 5-fold task-grouped and family-grouped CV (seed 42) and scores
the held-out cells.  For the full grid this takes hours; for the
paper's specs it is dominated by Model 2's ten fold refits.

--mode insample is a fast stand-in for when the CV has not been run on
the current data yet.  It uses the same folds, held-out cells, scoring
and aggregation code, but scores every fold with the FULL-data fit, so
the "held-out" tasks were part of the fit's training data.  The
numbers it produces are in-sample, not cross-validated.  provenance.json
records the mode, and the figure scripts watermark anything drawn from
a stand-in.

Outputs: cv_preds_<scheme>.csv, cv_joint_ll.csv, cv_metrics.csv,
bayle_ci_<scheme>.csv, reliability_cv.csv, insample_metrics.csv,
reliability_insample.csv, provenance.json.
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

import _paths as P
from fit import select_specs
from horizon_irt import evaluate as EV
from horizon_irt.specs import registry


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--mode', choices=('cv', 'insample'), default='cv')
    ap.add_argument('--specs', default='paper')
    ap.add_argument('--out', default=None,
                    help='output dir (default artifacts/canonical/cv)')
    ap.add_argument('--schemes', default='task,family')
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--k', type=int, default=1000)
    ap.add_argument('--kg', type=int, default=80)
    ap.add_argument('--em-max-iter', type=int, default=300,
                    help='cap on EM updates per IRT fold fit')
    ap.add_argument('--em-accel', choices=('squarem', 'none'),
                    default='squarem',
                    help="EM driver: SQUAREM, or 'none' for plain EM")
    ap.add_argument('--njob', type=int, default=8)
    ap.add_argument('--ref', default='linear',
                    help='reference spec of the paired-contrast CI rows')
    ap.add_argument('--save-cv-fits', action='store_true',
                    help='save every fold fit under <out>/cv_fits/')
    args = ap.parse_args()

    names = select_specs(args.specs)
    if args.ref not in names:
        raise SystemExit(f'--ref {args.ref} must be among the specs')
    schemes = [s.strip() for s in args.schemes.split(',')]
    out = Path(P.cv_dir() if args.out is None else args.out)
    out.mkdir(parents=True, exist_ok=True)
    fits_dir = P.fits_dir()
    missing = [n for n in names if not (fits_dir / n).exists()]
    if missing:
        raise SystemExit(f'fit these first (pipeline/fit.py): {missing}')
    print(f'evaluating {len(names)} specs on {P.DATASET}, mode={args.mode} '
          f'-> {out}', flush=True)

    EV.stage_insample(names, fits_dir, out, P.cells_path())
    ok = EV.stage_cv(names, registry(), out, P.cells_path(), schemes,
                     args.folds, args.seed, args.k, args.kg,
                     args.em_max_iter, args.njob,
                     save_fits_dir=(out / 'cv_fits' if args.save_cv_fits
                                    else None),
                     refit=(args.mode == 'cv'), fits_dir=fits_dir,
                     accel=args.em_accel)
    EV.stage_cv_aggregate(names, out, schemes, ref=args.ref)
    (out / 'provenance.json').write_text(json.dumps({
        'mode': 'cv' if args.mode == 'cv' else 'insample-standin',
        'dataset': P.DATASET, 'specs': names, 'schemes': schemes,
        'folds': args.folds, 'seed': args.seed, 'K': args.k, 'Kg': args.kg,
        'em_max_iter': args.em_max_iter, 'em_accel': args.em_accel,
        'ref': args.ref,
        'created': datetime.date.today().isoformat(),
    }, indent=2) + '\n')
    print('evaluate done.')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
