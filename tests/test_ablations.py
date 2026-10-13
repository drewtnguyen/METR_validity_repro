#!/usr/bin/env python3
"""Checks of the ablation options (about two minutes): explicit knots, spec
validation, anchored bumps and the JML-shape route.

  python tests/test_ablations.py
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from _common import ROOT, check, finish
from horizon_irt import basis, bumps, data, fitting, io, jml, model_api
from horizon_irt.specs import ABLATION_SPECS, all_specs, registry

CANONICAL = ROOT / 'data' / 'processed' / 'canonical.csv'
QUICK = dict(K=200, Kg=20, max_iter=60)

# Full-data results of jml_shape_pilot/ (surrounding repository) on the
# canonical data, which this package's JML route must reproduce.
PILOT_JML = {('invvar', 'extreme', True): -8913.210854916444,
             ('totvar', 'quasi', False): -8802.235288413225}


def test_knots():
    print('explicit spline knots:')
    rng = np.random.default_rng(0)
    x = rng.uniform(-2.5, 10.5, 300)
    nb = basis.build_monotone_basis(x, [0.0, 1.5, 7.0])   # asserts inside
    check(nb.df == 5 and np.allclose(nb.interior, [0.0, 1.5, 7.0]),
          'explicit knots: identity in span, monotone, natural ends')
    try:
        basis.build_monotone_basis(x, [11.0])
        check(False, 'a knot outside the data range is rejected')
    except ValueError:
        check(True, 'a knot outside the data range is rejected')
    idx = data.load_cells(CANONICAL)
    m1 = fitting.fit_spec('shared_shslope', idx)
    same = fitting.make_spec('shared_shslope',
                             mean_knots=tuple(2.0 ** m1['basis'].interior))
    m1k = fitting.fit_spec(same, idx)
    check(m1k['logl'] == m1['logl'],
          'knots at the quantile positions reproduce Model 1 exactly')
    sp = fitting.fit_spec(fitting.make_spec('spline', mean_knots=(15, 240)),
                          idx)
    check(sp['basis'].df == 4 and sp['n_params'] == 4 * idx['n_models'],
          'per-AI splines take explicit knots (2 knots -> 4 df per AI)')


def test_spec_validation():
    print('spec validation and the ablation registry:')
    reg = registry()
    check(len(all_specs()) == 52 and set(ABLATION_SPECS) <= set(reg)
          and len(reg) == 52 + len(ABLATION_SPECS),
          f'52 grid specs + {len(ABLATION_SPECS)} named ablations')
    for name, cfg in reg.items():
        fitting.resolve(dict(cfg, name=name))       # raises if invalid
    check(True, 'every registered spec validates')
    bad = [('spline', dict(sqrtfamily=True)), ('linear', dict(bb=True)),
           ('linear', dict(bumps=(bumps.bump(60, 240),))),
           ('classic', dict(bumps=(bumps.bump(60, 240),))),
           ('shared_shslope', dict(mean_kint=3, mean_knots=(60,))),
           ('shared_shslope', dict(jml_weights='invvar')),
           ('s1_homo', dict(shape_source='jml')),
           ('shared_shslope', dict(bumps=(bumps.bump(240, 60),)))]
    rejected = 0
    for base, kw in bad:
        try:
            fitting.make_spec(base, **kw)
        except (ValueError, KeyError):
            rejected += 1
    check(rejected == len(bad), f'{rejected}/{len(bad)} invalid combinations '
          'rejected instead of silently ignored')
    name = fitting.make_spec(
        'shared_shslope', bumps=(bumps.bump(60, 240),
                                 bumps.bump(240, 960, ai_specific=True)),
        mean_knots=(30, 120))['name']
    check(name == 'shared_shslope__bumps=60-240,240-960ai'
                  '__mean_knots=30-120', f'readable generated name: {name}')


def _gradient_error(core, idx, bump_list, **kw):
    """Max |analytic - central-difference| gradient of the bump objective
    at a random point."""
    cap = {}
    real = bumps._optimize

    def grab(obj, layout, fit, monotone):
        cap['obj'], cap['n'] = obj, layout.n
        return real(obj, layout, fit, monotone)

    bumps._optimize = grab
    try:
        bumps.fit_bumps(core, idx, bump_list, **kw)
    finally:
        bumps._optimize = real
    obj = cap['obj']
    z = np.random.default_rng(1).normal(0.0, 0.3, cap['n'])
    g = obj(z)[1]
    eye = np.eye(len(z)) * 1e-5
    fd = np.array([(obj(z + e)[0] - obj(z - e)[0]) / 2e-5 for e in eye])
    return float(np.max(np.abs(g - fd)) / max(1.0, np.max(np.abs(g))))


def _first_crossing_ok(fit, q=0.5):
    """With general bumps only the curves are monotone, so the first
    down-crossing must equal the usual inversion."""
    lo, hi = bumps._range(fit)
    xg = np.linspace(lo, hi, 481)
    P = model_api.marginal_p(fit, xg)
    ref = np.array([2.0 ** np.interp(q, P[::-1, j], xg[::-1])
                    if P[-1, j] <= q <= P[0, j] else np.nan
                    for j in range(P.shape[1])])
    th = model_api.time_horizons(fit, q)
    return np.allclose(th, ref, rtol=1e-9, equal_nan=True)


def test_bumps():
    print('anchored bumps:')
    idx = data.load_cells(CANONICAL)
    general = [bumps.bump(60, 240), bumps.bump(240, 960)]
    with_ai = general + [bumps.bump(240, 960, ai_specific=True)]
    for base, kw in (('shared_shslope', {}),
                     ('shared_bb', {}),
                     ('zm_homo_fam_bb', QUICK)):
        core = fitting.fit_spec(base, idx, **kw)
        bk = dict(K=kw.get('K', 1000), Kg=kw.get('Kg', 80))
        err = _gradient_error(core, idx, with_ai, **bk)
        check(err < 1e-5, f'{base}: analytic gradient matches central '
              f'differences (rel. err {err:.1e})')
        fit = bumps.fit_bumps(core, idx, with_ai, **bk)
        info = fit['bump_fit']
        check(info['gain'] > 0 and info['curve_monotone'],
              f"{base}: bumps raise logL by {info['gain']:.2f} and keep the "
              'difficulty curve monotone')
        ai = [b for b in fit['bumps'] if b['ai_specific']][0]
        check(abs(ai['exposure'] @ ai['amplitudes']) < 1e-8,
              f'{base}: AI-specific amplitudes have exposure-weighted mean 0')
        gen = bumps.fit_bumps(core, idx, general, **bk)
        check(_first_crossing_ok(gen), f'{base}: horizons of a general-bump '
              'fit equal the usual inversion')
    # the spec route, with a save/load round trip
    xg = np.linspace(-4, 10, 57)
    spec = fitting.make_spec('shared_shslope', bumps=with_ai)
    with tempfile.TemporaryDirectory() as d:
        fitting.fit_and_save(spec, idx, Path(d))
        back = io.load_fit(Path(d) / spec['name'])
    fit = fitting.fit_spec(spec, idx)
    check(np.array_equal(model_api.marginal_p(fit, xg),
                         model_api.marginal_p(back, xg))
          and np.array_equal(model_api.time_horizons(fit, 0.5),
                             model_api.time_horizons(back, 0.5),
                             equal_nan=True),
          'a saved bump fit predicts bit-for-bit')
    jll = model_api.joint_ll(fit, idx)
    check(abs(jll - fit['logl']) < 1e-6,
          'Model 1 + bumps: joint_ll equals the fitted logL')


def test_jml():
    print('JML-shape route (canonical data):')
    idx = data.load_cells(CANONICAL)
    for (wt, drop, shslope), ref in PILOT_JML.items():
        fit = fitting.fit_spec(fitting.make_spec(
            'shared_shslope', shape_source='jml', jml_weights=wt,
            jml_drop=drop, shslope=shslope), idx)
        check(abs(fit['logl'] - ref) < 1e-6,
              f'{wt}/{drop}/shslope={shslope}: logL {fit["logl"]:.4f} '
              'matches jml_shape_pilot')
    st = jml.theta_hat(idx)
    check(int(np.isnan(st['theta']).sum()) == 15 and not st['flipped'],
          '15 tasks without a finite JML difficulty dropped')
    # the frozen-shape fit at Model 1's own shape cannot do worse than
    # Model 1's coordinate ascent, and lands within 0.1 of it
    m1 = fitting.fit_spec('shared_shslope', idx)
    frozen = fitting.basic.fit_shared(idx, shslope=True, fixed_w=m1['w'])
    gap = frozen['logl'] - m1['logl']
    check(-1e-6 <= gap < 0.1, f'Model 1 shape frozen: logL gap {gap:+.2e}')


if __name__ == '__main__':
    test_knots()
    test_spec_validation()
    test_bumps()
    test_jml()
    finish()
