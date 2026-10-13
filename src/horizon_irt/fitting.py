"""Fit one specification of the grid and write its outputs.

  fit_spec(name, idx, ...)          -> fit dict (IRT fits anchored)
  save_outputs(name, cfg, fit, ...) -> writes <out>/<name>/:
      state.json + arrays.npz  the full fit (see horizon_irt.io)
      params.csv               per-AI alpha/beta and marginal th50/th80
      fit.json                 summary: logL, parameter counts, AIC, ...
      curves.csv               dense-grid fitted curves
      history.csv              EM history (IRT): one row per EM
                               iteration, or per SQUAREM cycle

`fit_spec` accepts a registry codename or a config dict of the same
shape as horizon_irt.specs.all_specs() values, so new variants can be
fit without registering them; `make_spec` builds such a dict from a
codename plus overrides, e.g.

  make_spec('s1_shslope_hetero_famspline_bb', mean_kint=8)  # 10-df f
  make_spec('s1_shslope_hetero_famspline_bb', fit_mean=False)  # f(t) = t

Config keys (SPEC_KEYS): cls ('basic' | 'irt'); for basic specs kind
('linear' | 'spline' | 'shared'); for IRT specs fit_mean (False is the
zero-mean prior f(t) = t), hetero (task spread sigma(t) vs constant),
family ('none' | 'fam' | 'famspline'), classic; for both bb
(beta-binomial overdispersion), probit, shslope (one discrimination
shared by all AIs), sqrtfamily (basic only: METR's sqrt-family
weighting).  Spline sizes, each defaulting to the paper's: mean_kint
(f, or the shared shape; 4 interior knots = 6 df), var_kint (sigma(t);
2 = 4 df) and tau_kint (tau(t_bar); 2 = 4 df).

Ablation options (docs/API.md has the details):
  mean_knots, var_knots, tau_knots  explicit interior knot positions in
      MINUTES, instead of a count at quantiles (*_kint), e.g.
      make_spec('shared_shslope', mean_knots=(60, 240, 960))
  bumps, bump_monotone  anchored bumps (horizon_irt.bumps), fitted on
      top of the fixed core fit, e.g.
      make_spec(M2, bumps=(bump(60, 240), bump(240, 960)))
  shape_source='jml', jml_weights, jml_drop, jml_beta_ridge
      kind='shared' only: freeze the shape at a monotone spline of JML
      difficulties (horizon_irt.jml)
specs.registry() adds named ablation variants (specs.ablation_specs())
to the 52-spec grid; fit_spec and make_spec accept either.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import basic, bumps, em, io, jml, model_api
from .specs import registry

SPEC_KEYS = {'cls', 'kind', 'fit_mean', 'hetero', 'family', 'bb', 'probit',
             'shslope', 'classic', 'sqrtfamily', 'mean_kint', 'var_kint',
             'tau_kint', 'mean_knots', 'var_knots', 'tau_knots', 'bumps',
             'bump_monotone', 'shape_source', 'jml_weights', 'jml_drop',
             'jml_beta_ridge'}
SPLINES = ('mean', 'var', 'tau')
DEFAULT_KINT = {'mean': em.MEAN_KINT, 'var': em.VAR_KINT,
                'tau': em.VAR_KINT}


def resolve(spec):
    """(name, cfg) from a registry codename or a config dict."""
    if isinstance(spec, str):
        reg = registry()
        if spec not in reg:
            raise KeyError(f'unknown spec {spec!r}')
        return spec, reg[spec]
    cfg = {k: v for k, v in spec.items() if k != 'name'}
    unknown = set(cfg) - SPEC_KEYS
    if unknown:
        raise KeyError(f'unknown spec keys {sorted(unknown)}; '
                       f'expected a subset of {sorted(SPEC_KEYS)}')
    _validate(cfg)
    return spec.get('name', 'custom'), cfg


def _validate(cfg):
    """Reject combinations a fitter would otherwise silently ignore."""
    for which in SPLINES:
        if (cfg.get(f'{which}_knots') is not None
                and cfg.get(f'{which}_kint') is not None):
            raise ValueError(f'give {which}_kint or {which}_knots, not both')
    kind = cfg.get('kind') if cfg['cls'] == 'basic' else None
    if kind == 'spline' and any(cfg.get(k) for k in
                                ('bb', 'sqrtfamily', 'shslope')):
        raise ValueError("kind='spline' supports probit only: bb, "
                         'sqrtfamily and shslope are not implemented')
    if kind == 'linear' and cfg.get('bb'):
        raise ValueError("kind='linear' has no beta-binomial version")
    source = cfg.get('shape_source', 'fit')
    if source not in ('fit', 'jml'):
        raise ValueError("shape_source must be 'fit' or 'jml'")
    if source == 'jml':
        if kind != 'shared':
            raise ValueError("shape_source='jml' needs kind='shared'")
        if cfg.get('jml_weights', 'invvar') not in jml.WEIGHTS:
            raise ValueError(f'jml_weights must be one of {jml.WEIGHTS}')
        if cfg.get('jml_drop', 'extreme') not in jml.DROP_RULES:
            raise ValueError(f'jml_drop must be one of {jml.DROP_RULES}')
    elif any(k.startswith('jml_') for k in cfg):
        raise ValueError("jml_* keys need shape_source='jml'")
    if cfg.get('bumps'):
        bumps.check_supported(cfg)
        bumps.normalize(cfg['bumps'])


def spline_size(cfg, which):
    """The size argument of spline `which` ('mean', 'var' or 'tau'):
    its interior-knot count, or its explicit knots in log2 minutes."""
    knots = cfg.get(f'{which}_knots')
    if knots is not None:
        return np.log2(np.asarray(knots, dtype=float))
    kint = cfg.get(f'{which}_kint')
    return DEFAULT_KINT[which] if kint is None else kint


def spline_df(cfg, which):
    """Degrees of freedom of spline `which`: interior knots + 2."""
    size = spline_size(cfg, which)
    n_int = len(np.unique(size)) if np.ndim(size) else int(size)
    return n_int + 2


def _name_part(k, v):
    """`k=v` for make_spec's names, compact for knots and bumps."""
    if k == 'bumps':
        v = ','.join(f"{b['start']:g}-{b['end']:g}"
                     + ('ai' if b['ai_specific'] else '')
                     for b in bumps.normalize(v))
    elif k.endswith('_knots'):
        v = '-'.join(f'{float(x):g}' for x in v)
    return f'__{k}={v}'


def make_spec(base, name=None, **overrides):
    """Config dict: registry spec `base` with `overrides` applied.

    Knots are given in minutes and bumps as horizon_irt.bumps.bump(...)
    entries; the generated name abbreviates both, e.g.
    'shared_shslope__bumps=60-240,240-960ai'."""
    cfg = dict(registry()[base], **overrides)
    cfg['name'] = name or base + ''.join(
        _name_part(k, v) for k, v in sorted(overrides.items()))
    resolve(cfg)                              # validates the keys
    return cfg


def fit_spec(spec, idx, K=1000, Kg=80, max_iter=300, tau0=0.0,
             fits_dir=None, anchor=True, classic_prior=None,
             accel='squarem'):
    """Fit `spec` on the cells in `idx`.

    K, Kg, max_iter, tau0 and accel go to em.fit_irt (IRT specs only);
    accel='none' runs plain EM instead of SQUAREM.

    The classic specs pin their prior at (t-bar, zm_homo's sigma-hat):
    pass those constants as `classic_prior`, or a `fits_dir` holding the
    zm_homo fit to compute them from.  anchor=False skips the
    identity-at-endpoints anchoring of IRT fits (CV fold fits are scored
    unanchored; the anchoring is a reparametrisation, so predictions
    agree up to rounding)."""
    name, cfg = resolve(spec)
    mean_kint = spline_size(cfg, 'mean')
    if cfg['cls'] == 'basic':
        if cfg['kind'] == 'linear':
            fit = basic.fit_linear(idx, probit=cfg['probit'],
                                   shslope=cfg.get('shslope', False),
                                   sqrtfamily=cfg.get('sqrtfamily', False))
        elif cfg['kind'] == 'spline':
            fit = basic.fit_pmspline(idx, probit=cfg['probit'],
                                     kint=mean_kint)
        elif cfg.get('shape_source', 'fit') == 'jml':
            fit = jml.fit_jml_shared(
                idx, weights=cfg.get('jml_weights', 'invvar'),
                drop=cfg.get('jml_drop', 'extreme'),
                beta_ridge=cfg.get('jml_beta_ridge', jml.BETA_RIDGE),
                probit=cfg['probit'], bb=cfg['bb'],
                shslope=cfg.get('shslope', False),
                sqrtfamily=cfg.get('sqrtfamily', False), kint=mean_kint)
        else:
            fit = basic.fit_shared(idx, probit=cfg['probit'], bb=cfg['bb'],
                                   shslope=cfg.get('shslope', False),
                                   sqrtfamily=cfg.get('sqrtfamily', False),
                                   kint=mean_kint)
    elif cfg['cls'] == 'irt':
        cm = cs = None
        if cfg.get('classic'):
            cm, cs = (classic_prior if classic_prior is not None else
                      em.classic_prior_constants(idx, fits_dir=fits_dir))
        fit = em.fit_irt(idx, fit_mean=cfg['fit_mean'],
                         hetero=cfg['hetero'], family=cfg['family'],
                         bb=cfg['bb'], probit=cfg['probit'],
                         shslope=cfg.get('shslope', False),
                         classic=cfg.get('classic', False),
                         classic_mean=cm, classic_sigma=cs, K=K, Kg=Kg,
                         tau0=tau0, max_iter=max_iter, accel=accel,
                         mean_kint=mean_kint,
                         var_kint=spline_size(cfg, 'var'),
                         tau_kint=spline_size(cfg, 'tau'))
        if anchor:
            em.anchor_identity(fit, idx)
    else:
        raise ValueError(f"unsupported spec class {cfg['cls']!r}")
    fit['cls'] = cfg['cls']
    fit['model_names'] = list(idx['model_names'])
    if cfg.get('bumps'):
        # anchored: the core above stays fixed, only amplitudes are fitted
        fit = bumps.fit_bumps(fit, idx, cfg['bumps'],
                              monotone=cfg.get('bump_monotone', True),
                              K=K, Kg=Kg)
    return fit


def fit_and_save(spec, idx, out_dir, **kw):
    name, cfg = resolve(spec)
    t0 = time.perf_counter()
    fit = fit_spec(spec, idx, fits_dir=out_dir, **kw)
    return save_outputs(name, cfg, fit, idx, Path(out_dir),
                        time.perf_counter() - t0)


def k_counts(name, cfg, n_models):
    if cfg['cls'] == 'basic':
        return None, None       # filled from the fit dict
    k_raw = 2 * n_models
    if cfg.get('shslope'):
        k_raw -= n_models - 1        # one shared beta, not n_models betas
    if cfg['fit_mean']:
        k_raw += spline_df(cfg, 'mean')
    # variance: hetero spline, or 1 for a fitted homo sigma; classic
    # freezes sigma=1 so it contributes NO parameter.
    if cfg['hetero']:
        k_raw += spline_df(cfg, 'var')
    elif not cfg.get('classic'):
        k_raw += 1
    k_raw += {'none': 0, 'fam': 1,
              'famspline': spline_df(cfg, 'tau')}[cfg['family']]
    if cfg['bb']:
        k_raw += 1
    # anchored bumps: one amplitude per general bump, n_models - 1 per
    # AI-specific bump (zero-mean); all shrunk by their priors
    for b in bumps.normalize(cfg.get('bumps') or ()):
        k_raw += n_models - 1 if b['ai_specific'] else 1
    k_eff = k_raw - 2 if cfg['fit_mean'] else k_raw
    return k_raw, k_eff


def save_outputs(name, cfg, fit, idx, out_dir, wall_s):
    d = out_dir / name
    d.mkdir(parents=True, exist_ok=True)
    n_models = idx['n_models']
    fit['cls'] = cfg['cls']
    fit['model_names'] = list(idx['model_names'])

    th50 = model_api.time_horizons(fit, 0.5)
    th80 = model_api.time_horizons(fit, 0.8)

    rows = {'model': idx['model_names'],
            'release_date': idx['release_date'],
            'th50_min': th50, 'th80_min': th80}
    if cfg['cls'] == 'irt':
        rows['alpha'] = fit['params'][:, 0]
        rows['beta'] = fit['params'][:, 1]
    elif cfg['kind'] == 'linear':
        rows['alpha'] = fit['params'][:, 0]
        rows['beta'] = fit['params'][:, 1]
    elif cfg['kind'] == 'shared':
        rows['alpha'] = fit['a']
        rows['beta'] = fit['b']
        if cfg['bb']:
            rows['rho'] = fit['rho']
    for b in fit.get('bumps') or ():
        if b['ai_specific']:
            rows[f"bump_{b['start']:g}-{b['end']:g}"] = b['amplitudes']
    pd.DataFrame(rows).to_csv(d / 'params.csv', index=False)

    k_raw, k_eff = k_counts(name, cfg, n_models)
    if k_raw is None:
        k_raw = k_eff = fit['n_params']
    summary = {
        'spec': name, 'config': {k: v for k, v in cfg.items()},
        'logl': fit['logl'], 'k_raw': int(k_raw), 'k_eff': int(k_eff),
        'aic': 2 * k_eff - 2 * fit['logl'],
        'align_c': fit.get('align_c', 1.0),
        'align_d': fit.get('align_d', 0.0),
        'phi': fit.get('phi'),
        'rho_global': (1.0 / (1.0 + fit['phi'])
                       if fit.get('phi') else None),
        'classic_mean': fit.get('classic_mean'),
        'classic_sigma': fit.get('classic_sigma'),
        'tau0': fit.get('tau0', 0.0),
        'tau_floor': fit.get('tau_floor'),
        'min_ll_diff': fit.get('min_ll_diff'),
        'ascent_ok': fit.get('ascent_ok'),
        'tau_min': (float(np.min(fit['tau_vec']))
                    if fit.get('tau_vec') is not None else None),
        'tau_max': (float(np.max(fit['tau_vec']))
                    if fit.get('tau_vec') is not None else None),
        'tau': fit.get('tau'),
        'accel': fit.get('accel'),
        'iterations': fit.get('iterations'),
        'cycles': fit.get('cycles'),
        'converged': fit.get('converged'),
        'wall_s': wall_s,
        'n_tasks': idx['n_tasks'], 'n_models': n_models,
        'n_runs': idx['n_runs'],
    }
    if cfg['cls'] == 'basic' and cfg.get('bb'):
        summary['rho_by_model_median'] = float(np.median(fit['rho']))
    if fit.get('bumps'):
        summary['core_logl'] = fit['core_logl']
        summary['bumps'] = [
            {k: v for k, v in b.items()
             if k not in ('amplitudes', 'exposure')} for b in fit['bumps']]
        summary['bump_fit'] = fit['bump_fit']
    if fit.get('shape_source') == 'jml':
        summary['jml'] = {k: v for k, v in fit['jml'].items()
                          if np.ndim(v) == 0}
    (d / 'fit.json').write_text(json.dumps(summary, indent=2, default=str))

    if 'history' in fit and fit['history']:
        pd.DataFrame(fit['history']).to_csv(d / 'history.csv', index=False)

    # dense curves
    if cfg['cls'] == 'irt':
        lo, hi = em._range_of(fit)
        xg = np.linspace(lo, hi, 241)
        h, s_tot = em.prior_curves(fit, xg)
        cur = {'x_log2min': xg, 'f': h, 's_marginal': s_tot}
        if fit.get('bumps'):
            cur['f_bumped'] = h + bumps.general_shift(fit, xg)
        pd.DataFrame(cur).to_csv(d / 'curves.csv', index=False)
    elif cfg['kind'] == 'shared':
        nb = fit['basis']
        xg = np.linspace(nb.lo, nb.hi, 241)
        f = (nb.raw_merged(xg) / fit['col_hi']) @ fit['w']
        f_scaled = nb.lo + (nb.hi - nb.lo) * f
        cur = {'x_log2min': xg, 'f01': f, 'f_scaled': f_scaled}
        if fit.get('bumps'):
            cur['f01_bumped'] = f + bumps.general_shift(fit, xg)
        pd.DataFrame(cur).to_csv(d / 'curves.csv', index=False)

    io.save_fit(fit, d)
    return summary
