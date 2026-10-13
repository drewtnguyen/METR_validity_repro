"""Loading, rescaling and posterior summaries for Model 2
(`s1_shslope_hetero_famspline_bb`).

Model 2's likelihood is invariant under the affine relabelling
theta -> (theta - d)/c, absorbed by f, sigma(t), tau(t_bar), alpha and
the shared beta, so the latent scale is fixed only by convention.  The
stored fit uses endpoint anchoring f(a)=a, f(b)=b; both figures instead
re-resolve it by least squares of f(t_i) against t_i over the 228
tasks -- the affine map that brings f as close to the identity as
possible.  That is what licenses reading 2^theta as an equivalent task
length in minutes.

Shared by make_headline.py and make_difficulty_vs_length.py (difficulty vs
length), so the two figures are guaranteed to be on the same scale.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
from scipy.special import roots_hermite

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'pipeline'))
import _paths as P                                           # noqa: E402
from horizon_irt import data, io                             # noqa: E402
from horizon_irt import kernels as KK                        # noqa: E402
from horizon_irt import em                                   # noqa: E402
from horizon_irt.em import SIGMA_FLOOR, P_EPS                # noqa: E402

FITS = P.fits_dir()
SPEC = 's1_shslope_hetero_famspline_bb'   # Model 2 of Section 2.3


def load():
    """(fit, idx) for Model 2 at the stored, endpoint-anchored scale."""
    return io.load_fit(FITS / SPEC), data.load_cells(P.cells_path())


def prior_mean_sigma(fit, x):
    """(f(x), sigma(x)) at the fit's own scale.

    sigma(x) is the TASK-level spread only; the family effect tau is not
    included.  For the effective prior scale of a fresh task, use
    prior_mean_total_sigma."""
    x = np.atleast_1d(np.asarray(x, dtype=float))
    f = fit['mean_basis'].design(x) @ fit['coefs'] if fit['fit_mean'] else x
    if fit['hetero']:
        s = np.maximum(np.exp(fit['var_basis'].design(x) @ fit['gammas']),
                       SIGMA_FLOOR)
    else:
        s = np.full(len(x), max(math.exp(fit['gammas'][0]), SIGMA_FLOOR))
    return f, s


def prior_mean_total_sigma(fit, x):
    """(f(x), s(x)) with the family effect marginalized into the scale:
    s(x)^2 = sigma(x)^2 + tau(x)^2, tau evaluated at t_bar = x -- the
    prior for a fresh task from a fresh family whose mean length is x.
    This is em.prior_curves, the same convention the marginal horizons
    use."""
    return em.prior_curves(fit, np.atleast_1d(np.asarray(x, dtype=float)))


def ols_identity_align(fit, t):
    """(c, d) minimizing ||f(t_i) - (c t_i + d)||^2 over the tasks.

    Applying theta -> (theta - d)/c then makes the mean curve as close to
    the identity as least squares allows."""
    f_task, _ = prior_mean_sigma(fit, t)
    c, d = np.polyfit(t, f_task, 1)
    return float(c), float(d)


def rescale_latent(fit, c, d):
    """Copy of `fit` reparameterized by theta -> (theta - d)/c.

    Same transform as em.anchor_identity, but with (c, d) supplied rather
    than read off the boundary knots.  Likelihood-invariant.  `gammas` is
    the log-sigma vector, a single intercept under homoskedasticity."""
    out = dict(fit)
    e0 = np.zeros_like(fit['coefs'])
    e0[0] = 1.0
    out['coefs'] = (fit['coefs'] - d * e0) / c
    out['gammas'] = fit['gammas'].copy()
    out['gammas'][0] -= np.log(c)
    if fit['family'] == 'fam':
        out['tau'] = fit['tau'] / c
        out['tau_vec'] = fit['tau_vec'] / c
    elif fit['family'] == 'famspline':
        out['gam_tau'] = fit['gam_tau'].copy()
        out['gam_tau'][0] -= np.log(c)
        out['tau_vec'] = fit['tau_vec'] / c
    p = fit['params'].copy()
    p[:, 0] = p[:, 0] + p[:, 1] * d
    p[:, 1] = c * p[:, 1]
    out['params'] = p
    return out


def check_invariance(fit_aligned, params_df, model_names, tol=1e-6):
    """Marginal horizons must survive the realignment untouched -- the
    check that catches a sign or direction error in the map."""
    th50 = em.time_horizon_marginal(fit_aligned, 0.5)
    stored = params_df.set_index('model').loc[list(model_names), 'th50_min']
    rel = np.abs(th50 - stored.to_numpy()) / stored.to_numpy()
    assert np.nanmax(rel) < tol, \
        f'th50 not invariant: max rel {np.nanmax(rel):.2e}'
    return float(np.nanmax(rel))


def posterior_theta(fit, idx):
    """theta_hat_i = E[theta_i | data]: one E-step at the stored
    parameters, on the fit's own scale.

    The posterior mean is linear in theta, so the aligned value is just
    (theta_hat - d)/c -- no need to redo the quadrature after rescaling,
    and it avoids disturbing the fixed proposal grid."""
    t = idx['t_array']
    means, sigmas = prior_mean_sigma(fit, t)
    roots, wts = roots_hermite(fit['K'])
    theta_grid = np.sqrt(2.0) * fit['sigma0'] * roots[None, :] + t[:, None]
    with np.errstate(divide='ignore'):
        log_qw = np.log(wts / np.sqrt(np.pi))
    phi = fit['phi'] if fit['bb'] else 1.0
    if fit['family'] == 'none':
        _, _, S1, _ = KK.estep_c1(
            theta_grid, log_qw, t, means, sigmas, fit['sigma0'],
            fit['params'], idx['cell_midx'], idx['cell_s'], idx['cell_n'],
            idx['cell_off'], fit['probit'], fit['bb'], phi, P_EPS)
    else:
        # Family E-step: theta_i = f(t_i) + g_F + eps_i, with the family
        # effect g integrated over its own GH grid at the fitted
        # per-family tau_vec.  S1 is E[theta_i | data] as before.
        g_roots, g_w = roots_hermite(fit['Kg'])
        with np.errstate(divide='ignore'):
            log_gw = np.log(g_w / np.sqrt(np.pi))
        _, _, S1, _, _, _, _ = KK.estep_c1_family(
            theta_grid, log_qw, t, means, sigmas, fit['sigma0'],
            fit['params'], idx['cell_midx'], idx['cell_s'], idx['cell_n'],
            idx['cell_off'], idx['fam_offsets'], idx['fam_task_idx'],
            fit['tau_vec'], g_roots, log_gw, fit['probit'], fit['bb'],
            phi, P_EPS)
    return S1
