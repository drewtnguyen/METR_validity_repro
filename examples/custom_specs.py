#!/usr/bin/env python3
"""Fit your own variants of the paper's Model 2.

  python examples/custom_specs.py [--quick]

Fits Model 2 and three variants built with fitting.make_spec, then
prints log-likelihood, parameter count, AIC and 50% time horizons:

  zero mean     fit_mean=False: the prior mean of task difficulty is
                log2 human time itself, f(t) = t, instead of a spline
  mean 10 df    mean_kint=8: a more flexible difficulty curve f
                (the paper's has 4 interior knots, 6 df)
  separate      var_kint=4, tau_kint=1: a larger spline for the task
                spread sigma(t) and a smaller one for the family spread
                tau(t_bar), which the paper sizes alike (4 df each)

Each full fit takes a minute or two.  --quick cuts the quadrature
nodes and EM updates so the whole script runs in about a minute; its
numbers are then rough.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from horizon_irt import data, fitting, model_api            # noqa: E402

MODEL2 = 's1_shslope_hetero_famspline_bb'
SHOW_AIS = ('GPT-4o (Inspect)', 'Claude 3.5 Sonnet (Old) (Inspect)',
            'GPT-5.2')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--quick', action='store_true')
    args = ap.parse_args()
    kw = (dict(K=200, Kg=40, max_iter=40) if args.quick
          else dict(K=1000, Kg=80, max_iter=300))

    idx = data.load_cells(ROOT / 'data' / 'processed' / 'canonical.csv')
    variants = {
        'Model 2 (paper)': MODEL2,
        'zero mean': fitting.make_spec(MODEL2, fit_mean=False),
        'mean 10 df': fitting.make_spec(MODEL2, mean_kint=8),
        'separate': fitting.make_spec(MODEL2, var_kint=4, tau_kint=1),
    }
    names = list(idx['model_names'])
    show = [a for a in SHOW_AIS if a in names]

    head = f"{'variant':16s} {'logL':>10s} {'k':>4s} {'AIC':>10s}"
    head += ''.join(f' {a[:18]:>18s}' for a in show)
    print(f'{idx["n_models"]} AIs, {idx["n_tasks"]} tasks; '
          f'50% horizons in minutes\n')
    print(head)
    for label, spec in variants.items():
        t0 = time.perf_counter()
        fit = fitting.fit_spec(spec, idx, **kw)
        name, cfg = fitting.resolve(spec)
        k_raw, k_eff = fitting.k_counts(name, cfg, idx['n_models'])
        th50 = model_api.time_horizons(fit, 0.5)
        row = (f"{label:16s} {fit['logl']:10.1f} {k_eff:4d} "
               f"{2 * k_eff - 2 * fit['logl']:10.1f}")
        row += ''.join(f' {th50[names.index(a)]:18.2f}' for a in show)
        print(row + f'   ({time.perf_counter() - t0:.0f}s)', flush=True)
    assert np.isfinite(th50).any()


if __name__ == '__main__':
    main()
