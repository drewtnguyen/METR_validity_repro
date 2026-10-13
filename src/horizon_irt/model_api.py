"""Uniform prediction API over the fit classes (basic / irt).

Every saved fit is a dict with a 'cls' key.  This module dispatches:
  marginal_p(fit, x)        -> (len(x), n_models) marginal success probs
  time_horizons(fit, q)     -> per-model th_q in minutes (marginal inversion)
  joint_ll(fit, idx, ...)   -> prior-predictive joint log-lik (sequence
                               convention) of the cells in idx
  marginal_p_for_idx(fit, idx) -> per-cell marginal p for scoring

Fits with anchored bumps (a 'bumps' key, see horizon_irt.bumps) are
routed to that module, which adds the bumps to the core's predictions.

The latent-class mixtures of the original grid (cls 'mix') are not part
of this package; their fitter lives outside the paper's code.
"""
from __future__ import annotations

import numpy as np
from scipy.special import gammaln

from . import em
from . import basic
from . import bumps


# ----------------------------------------------------------------------
# dispatch
# ----------------------------------------------------------------------

def marginal_p(fit, x, model_indices=None, fam_tbar=None):
    if bumps.has_bumps(fit):
        return bumps.marginal_p(fit, x, model_indices, fam_tbar=fam_tbar)
    if fit['cls'] == 'basic':
        return basic.predict_p(fit, x, model_indices)
    if fit['cls'] == 'irt':
        return em.marginal_p(fit, x, model_indices, fam_tbar=fam_tbar)
    raise ValueError(f"unsupported fit class {fit['cls']!r}")


def time_horizons(fit, q, n_grid=481):
    if bumps.has_bumps(fit):
        return bumps.time_horizons(fit, q, n_grid)
    if fit['cls'] == 'basic':
        return basic.time_horizon(fit, q, n_grid)
    if fit['cls'] == 'irt':
        return em.time_horizon_marginal(fit, q, n_grid)
    raise ValueError(f"unsupported fit class {fit['cls']!r}")


def marginal_p_for_idx(fit, idx):
    """Per-cell marginal p (n_cells,), fam-aware for famspline."""
    cell_task = np.repeat(np.arange(idx['n_tasks']),
                          np.diff(idx['cell_off']))
    t = idx['t_array']
    fam_tbar = None
    if fit['cls'] == 'irt' and fit.get('family') == 'famspline':
        fam_tbar = idx['fam_mean_t'][idx['fam_of_task']]
    # p_j(t_i) for every (task, model) pair we need: compute per unique
    # task once for all models, then pick.
    P = marginal_p(fit, t, fam_tbar=fam_tbar)     # (n_tasks, n_models)
    return P[cell_task, idx['cell_midx']]


def _bb_cell_ll(p, s, n, phi, p_eps=1e-6):
    p = np.clip(p, p_eps, 1 - p_eps)
    a = p * phi
    b = (1.0 - p) * phi
    return (gammaln(s + a) + gammaln(n - s + b) - gammaln(n + phi)
            - gammaln(a) - gammaln(b) + gammaln(phi))


def joint_ll(fit, idx, family_scheme=False):
    """Prior-predictive joint log-lik of idx's cells, sequence scale."""
    if fit['cls'] == 'irt':
        if bumps.has_bumps(fit):
            return bumps.joint_ll(fit, idx, family_scheme=family_scheme)
        return em.joint_loglik_eval(fit, idx, family_scheme=family_scheme)
    # basic: cells independent given parameters
    p = marginal_p_for_idx(fit, idx)
    s, n = idx['cell_s'], idx['cell_n']
    if fit.get('bb'):
        rho = np.asarray(fit['rho'])[idx['cell_midx']]
        phi = (1.0 - rho) / rho
        return float(np.sum(_bb_cell_ll(p, s, n, phi)))
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(np.sum(s * np.log(p) + (n - s) * np.log(1.0 - p)))


def n_models_of(fit):
    if fit['cls'] == 'basic':
        return (len(fit['params']) if fit['kind'] == 'linear'
                else (len(fit['coefs']) if fit['kind'] == 'spline'
                      else len(fit['a'])))
    return fit['params'].shape[0]
