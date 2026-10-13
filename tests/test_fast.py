#!/usr/bin/env python3
"""Fast checks (about two minutes): data, basis, quick fits against the
shipped artifacts, fit format, SQUAREM against plain EM.

  python tests/test_fast.py
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from _common import FITS, ROOT, check, compare_fit_dir, finish
from horizon_irt import basis, data, fitting, io, model_api
from horizon_irt.specs import PAPER_SPECS, PIPELINE_SPECS, all_specs

CELLS = ROOT / 'data' / 'processed' / 'canonical.csv'


def test_data():
    print('processed data rebuild from data/raw:')
    cells, _ = data.build_dataset('canonical', ROOT / 'data' / 'raw')
    shipped = data.read_cells(CELLS)
    cells = cells.astype({'release_date': str})
    check(cells.reset_index(drop=True).equals(shipped),
          'rebuilt cells equal data/processed/canonical.csv')
    idx = data.load_cells(CELLS)
    check((idx['n_tasks'], idx['n_models'], idx['n_runs'])
          == (228, 26, 30211), '228 tasks, 26 AIs, 30,211 runs')


def test_basis():
    print('spline basis:')
    rng = np.random.default_rng(0)
    x = rng.uniform(-2.5, 10.5, 300)
    for k_int in (2, 4, 6):
        nb = basis.build_monotone_basis(x, k_int)   # asserts internally
        check(nb.design(x).shape == (300, k_int + 2),
              f'K_int={k_int}: identity in span, monotone, natural ends')


def test_specs():
    print('spec registry:')
    reg = all_specs()
    check(len(reg) == 52, '52 specifications')
    check(set(PIPELINE_SPECS) <= set(reg) and set(PAPER_SPECS) <= set(reg),
          'paper and pipeline specs are registered')


def test_quick_fits_and_format():
    print('quick fits vs the shipped artifacts, and the fit file format:')
    idx = data.load_cells(CELLS)
    xg = np.linspace(-6, 11, 97)
    with tempfile.TemporaryDirectory() as d:
        for name in ('linear', 'linear_shslope_sqrtfamily',
                     'shared_shslope_sqrtfamily'):
            fitting.fit_and_save(name, idx, Path(d))
            compare_fit_dir(Path(d) / name, FITS / name, name)
            fit = fitting.fit_spec(name, idx)
            back = io.load_fit(Path(d) / name)
            same = (np.array_equal(model_api.marginal_p(fit, xg),
                                   model_api.marginal_p(back, xg))
                    and np.array_equal(model_api.time_horizons(fit, 0.5),
                                       model_api.time_horizons(back, 0.5),
                                       equal_nan=True))
            check(same, f'{name}: saved fit predicts bit-for-bit')


def test_squarem():
    # plain EM must converge for the comparison to mean anything: at
    # K=200 it converges in 187 updates on the canonical data
    print('SQUAREM vs plain EM (zm_homo, K=200):')
    idx = data.load_cells(CELLS)
    plain = fitting.fit_spec('zm_homo', idx, K=200, accel='none')
    sq = fitting.fit_spec('zm_homo', idx, K=200)
    check(sq['accel'] == 'squarem' and sq['converged'] and sq['ascent_ok'],
          f"SQUAREM converged in {sq['iterations']} EM updates "
          f"(plain EM: {plain['iterations']}), logL never fell")
    check(sq['iterations'] < plain['iterations'],
          'SQUAREM needs fewer EM updates')
    check(sq['logl'] >= plain['logl'] - 1e-6 * abs(plain['logl']),
          f"SQUAREM logL {sq['logl']:.6f} >= plain EM {plain['logl']:.6f}")
    rel = float(np.nanmax(np.abs(model_api.time_horizons(sq, 0.5)
                                 / model_api.time_horizons(plain, 0.5)
                                 - 1.0)))
    check(rel < 1e-2, f'50% horizons agree (max rel diff {rel:.1e})')


def test_shipped_artifacts():
    print('shipped artifacts load and predict:')
    for name in PIPELINE_SPECS:
        d = FITS / name
        ok = d.exists()
        if ok:
            p = model_api.marginal_p(io.load_fit(d), np.array([0.0, 4.0]))
            ok = np.all(np.isfinite(p)) and np.all((p > 0) & (p < 1))
        check(ok, name)


if __name__ == '__main__':
    test_data()
    test_basis()
    test_specs()
    test_quick_fits_and_format()
    test_squarem()
    test_shipped_artifacts()
    finish()
