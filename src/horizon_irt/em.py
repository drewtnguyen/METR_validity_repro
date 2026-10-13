"""Unified C=1 random-effects 2PL IRT driver (headline-model-grid skill).

Covers the full C=1 crossing with one EM loop:

  spec dims : fit_mean (s1 vs zm)  x  hetero  x  family {none, fam,
              famspline}  x  sampling {bern, bb}  x  link {logit, probit}

Model:
  link P(Y=1 | theta_i) = alpha_j + beta_j * theta_i
  theta_i = m_i + [g_fam] + eps_i,  m_i = f(t_i) (s1) or t_i (zm)
  eps_i ~ N(0, sigma(t_i)^2)  (homo: constant)
  g_fam ~ N(0, tau^2) or N(0, tau(tbar_fam)^2)
  cells:  bern sequence  or  beta-binomial(phi) sequence likelihood

Conventions (see the skill): fixed GH proposal grid at sigma0 centred at
t_i, importance-reweighted; monotone natural-cubic mean (K_int=4, 6 df,
box bounds); natural-cubic log-variance (K_int=2, 4 df, free); sigma
floor 0.3; identity-at-endpoints anchoring of f after the fit resolves
the affine ridge (s1 specs only).

The EM update (one E-step, then every M-step) is wrapped in SQUAREM
(Varadhan & Roland 2008) by default, which extrapolates along the path
of two consecutive updates; accel='none' runs the plain EM loop.  Both
drivers reach the same fixed points; see run_squarem.
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
from scipy.special import roots_hermite
from scipy.optimize import lsq_linear

from .basis import NaturalBasis, build_monotone_basis
from . import kernels as KK
from .data import index_cells, load_cells   # noqa: F401  (re-exported)

MEAN_KINT = 4        # mean spline: natural cubic, 6 df
VAR_KINT = 2         # log-variance spline: natural cubic, 4 df
SIGMA_FLOOR = 0.3
TAU_FLOOR = 1e-3
P_EPS = 1e-6
LOG_PHI_BOUNDS = (-4.0, 20.0)
ACCEL_METHODS = ('squarem', 'none')
# SQUAREM's step-length cap starts at 1 (a plain-EM cycle) and is
# multiplied by this factor each time the step reaches it, and divided by
# it when a step at the cap is rejected (the SQUAREM package's `mstep`).
SQUAREM_MSTEP = 4.0
# SQUAREM keeps an extrapolation unless it lowers logL by more than this
# fraction of |logL|.  Without the slack it stalls near the optimum: the
# (alpha, beta) M-step clips p to [P_EPS, 1 - P_EPS] in its gradient but
# not in the logit-Bernoulli likelihood, so EM's fixed point sits a hair
# (~3e-8 in logL on a zm_homo fit) below the logL maximum, and
# from the maximum each EM step lowers logL slightly.  A strict safeguard
# then rejects every extrapolation and falls back to plain EM's slow
# final approach.  1e-9 of |logL| (~5e-6) is the size of change the
# default ftol already treats as no change.
SQUAREM_LL_SLACK = 1e-9


# ----------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------

def zm_sigma(fits_dir):
    """zm_homo's fitted homoscedastic sigma-hat, from its saved fit in
    `fits_dir` (so classic specs must be fit after zm_homo)."""
    from .io import load_fit
    p = Path(fits_dir) / 'zm_homo'
    if not p.exists():
        raise FileNotFoundError(
            f'classic prior needs the zm_homo fit ({p}); fit zm_homo first')
    zm = load_fit(p)
    return max(math.exp(float(zm['gammas'][0])), SIGMA_FLOOR)


def classic_prior_constants(idx, fits_dir):
    """(mean, sigma) for the pinned classic prior.

    mean  = t-bar, the average log2(human_minutes) over tasks;
    sigma = zm_homo's fitted homoscedastic sigma-hat -- so classic must be
            fit AFTER zm_homo.
    Both are FIXED constants of the spec (computed once from the full
    data / full-data zm_homo fit; CV folds reuse them unchanged)."""
    return float(idx['t_array'].mean()), zm_sigma(fits_dir)


def init_alpha_beta(idx, probit: bool) -> np.ndarray:
    """Per-model 2-param GLM on theta = t_i, via the Newton kernel with
    a single-node grid and unit responsibilities (Bernoulli init even
    for BB specs)."""
    n = idx['n_tasks']
    grid = idx['t_array'].reshape(n, 1)
    resp = np.ones((n, 1))
    params = np.zeros((idx['n_models'], 2))
    params[:, 1] = -0.5
    return KK.mstep_alpha_beta(params, grid, resp,
                               idx['mc_task'], idx['mc_s'], idx['mc_n'],
                               idx['mc_off'], probit, False, 1.0, P_EPS)


def monotone_wls(X, y, w, coefs_prev):
    """Box-bounded WLS on the merged natural basis with a tiny ridge
    toward the previous coefficients (augmented rows)."""
    sw = np.sqrt(w)
    A = X * sw[:, None]
    b = y * sw
    d = X.shape[1]
    eps = 1e-8 * (np.trace(A.T @ A) / d + 1.0)
    A_aug = np.vstack([A, math.sqrt(eps) * np.eye(d)])
    b_aug = np.concatenate([b, math.sqrt(eps) * coefs_prev])
    lb = np.full(d, -np.inf)
    lb[1:] = 0.0
    res = lsq_linear(A_aug, b_aug, bounds=(lb, np.inf), tol=1e-12)
    return res.x


def _rel_ll_change(prev, curr):
    return (curr - prev) / max(abs(prev), abs(curr), 1.0)


def _mstep_phi(u, theta_grid, resp, params, idx, probit, p_eps,
               max_steps=25):
    """Damped Newton in u = log(phi), backtracking on the EM objective."""
    lo, hi = LOG_PHI_BOUNDS
    for _ in range(max_steps):
        phi = math.exp(u)
        d1, d2 = KK.phi_score(theta_grid, resp, None, params,
                              idx['cell_midx'], idx['cell_s'],
                              idx['cell_n'], idx['cell_off'],
                              probit, phi, p_eps)
        gu = phi * d1
        hu = phi * phi * d2 + phi * d1
        if abs(gu) < 1e-8:
            break
        du = -gu / hu if hu < 0 else math.copysign(0.5, gu)
        du = max(-2.0, min(2.0, du))
        F0 = KK.weighted_ll(theta_grid, resp, params, idx['cell_midx'],
                            idx['cell_s'], idx['cell_n'], idx['cell_off'],
                            probit, True, phi, p_eps)
        t = 1.0
        accepted = False
        for _ in range(20):
            u_try = max(lo, min(hi, u + t * du))
            F1 = KK.weighted_ll(theta_grid, resp, params,
                                idx['cell_midx'], idx['cell_s'],
                                idx['cell_n'], idx['cell_off'],
                                probit, True, math.exp(u_try), p_eps)
            if F1 >= F0 - 1e-9:
                u = u_try
                accepted = True
                break
            t *= 0.5
        if not accepted or abs(t * du) < 1e-10:
            break
    return u


# ----------------------------------------------------------------------
# EM drivers: plain EM, and SQUAREM around the same update
# ----------------------------------------------------------------------
#
# Both drivers take `update(state) -> (new_state, logL, max_change)`: one
# E-step at `state` (which yields logL at `state`) followed by every
# M-step.  max_change is the largest absolute change the update makes to
# (alpha, beta), the prior means, sigma, tau and phi.  Both stop on the
# same two tests, applied to the update that starts each iteration (plain
# EM) or cycle (SQUAREM): max_change < tol, or a relative logL change
# below ftol since the previous one.  max_iter caps the number of
# updates, so it bounds the work, which is mostly E-steps, either way.
# Like the original loop, both return the last update's output together
# with the logL of its input.

def _history_row(it, ll, max_change, rel, st):
    rec = {'iter': it, 'logl': float(ll), 'max_change': max_change,
           'rel_ll_change': rel}
    if st['tau_vec'] is not None:
        rec['tau_min'] = float(st['tau_vec'].min())   # pure recording
        rec['tau_max'] = float(st['tau_vec'].max())
    return rec


def run_plain_em(update, st, max_iter, tol, ftol, verbose=False):
    """Plain EM: iterate the update until it stops moving."""
    history = []
    ll_prev = -np.inf
    converged = False
    for it in range(max_iter):
        st, ll, max_change = update(st)
        rel = _rel_ll_change(ll_prev, ll)
        history.append(_history_row(it + 1, ll, max_change, rel, st))
        if verbose and ((it + 1) % 25 == 0 or it == 0):
            print(f'    iter {it+1:4d} logL={ll:.4f} maxD={max_change:.2e}')
        if max_change < tol or (it > 0 and 0 <= rel < ftol):
            converged = True
            break
        ll_prev = ll
    return st, ll, history, it + 1, converged


def run_squarem(update, pack, unpack, st0, max_iter, tol, ftol,
                verbose=False):
    """SQUAREM (Varadhan & Roland 2008, step scheme SqS3) around `update`.

    Write F for the EM update and x for the parameters as one vector
    (`pack`; `unpack` maps a vector back to a state and projects it onto
    the parameter space).  Each cycle starts at x0 and takes two updates,
    x1 = F(x0) and x2 = F(x1).  With r = x1 - x0 and v = x2 - 2 x1 + x0,
    it then extrapolates to

        x' = x0 + 2 a r + a^2 v,   a = |r| / |v|,  clipped to [1, a_max].

    a = 1 gives x' = x2, two plain EM updates.  Larger a follows the EM
    path further: when EM crawls along a direction it shrinks by a factor
    lam per update, |r| / |v| ~ 1 / (1 - lam) and x' jumps most of the
    remaining distance (with one such direction, all of it).  A third
    update x'' = F(x') stabilizes the jump, and its E-step gives
    logL(x') at no extra cost.  The cycle keeps x'' only if logL(x') is
    at least logL(x0), less a slack of SQUAREM_LL_SLACK * |logL(x0)|;
    otherwise it falls back to x2.  Since EM itself does not lower logL,
    the logL at successive cycle starts never falls by more than the
    slack.

    a_max starts at 1 and is multiplied by SQUAREM_MSTEP each time a
    reaches it, so the steps lengthen only while they keep succeeding;
    a rejected extrapolation at a = a_max divides it back down.

    Returns (state, logL, history, updates, converged, cycles).  history
    has one row per cycle: logL and max_change are those of the update
    that starts the cycle, em_updates counts updates so far, alpha is the
    step length tried (1: the cycle was plain EM) and extrapolated says
    whether x'' was kept (False with alpha > 1: rejected).
    """
    history = []
    ll_prev = -np.inf
    converged = False
    n_upd = 0
    a_max = 1.0
    cycle = 0
    while True:
        cycle += 1
        st1, ll0, dx0 = update(st0)
        n_upd += 1
        st, ll = st1, ll0            # what is returned if EM stops here
        rel = _rel_ll_change(ll_prev, ll0)
        rec = _history_row(cycle, ll0, dx0, rel, st1)
        if dx0 < tol or (cycle > 1 and 0 <= rel < ftol):
            converged = True
        elif n_upd < max_iter:
            st2, ll1, _ = update(st1)
            n_upd += 1
            st, ll = st2, ll1
            alpha, extrapolated = 1.0, False
            if n_upd < max_iter:
                x0, x1, x2 = pack(st0), pack(st1), pack(st2)
                r = x1 - x0
                v = x2 - 2.0 * x1 + x0
                sv2 = float(v @ v)
                alpha = math.sqrt(float(r @ r) / sv2) if sv2 > 0 else 1.0
                alpha = min(max(alpha, 1.0), a_max)
                # within 1% of a = 1 the jump is x2 to rounding: skip it
                if alpha > 1.01:
                    x_ext = x0 + 2.0 * alpha * r + alpha * alpha * v
                    try:
                        st3, ll_ext, _ = update(unpack(x_ext))
                    except (ArithmeticError, ValueError,
                            np.linalg.LinAlgError):
                        st3, ll_ext = None, -np.inf
                    n_upd += 1
                    slack = SQUAREM_LL_SLACK * max(1.0, abs(ll0))
                    if np.isfinite(ll_ext) and ll_ext >= ll0 - slack:
                        st, ll = st3, ll_ext
                        extrapolated = True
                rec['alpha'] = alpha
                if alpha > 1.01 and not extrapolated:
                    # rejected: shorten the cap if the step was at it
                    if alpha == a_max:
                        a_max = max(1.0, a_max / SQUAREM_MSTEP)
                    alpha = 1.0
                if alpha == a_max:
                    a_max *= SQUAREM_MSTEP
            else:
                rec['alpha'] = alpha
            rec['extrapolated'] = extrapolated
            st0 = st
        rec['em_updates'] = n_upd
        history.append(rec)
        if verbose and (cycle % 10 == 0 or cycle == 1):
            print(f'    cycle {cycle:4d} updates {n_upd:4d} '
                  f'logL={ll0:.4f} maxD={dx0:.2e} '
                  f"a={rec.get('alpha', float('nan')):.1f}")
        if converged or n_upd >= max_iter:
            break
        ll_prev = ll0
    return st, ll, history, n_upd, converged, cycle


# ----------------------------------------------------------------------
# the unified C=1 EM
# ----------------------------------------------------------------------

def fit_irt(idx, fit_mean=True, hetero=True, family='none', bb=False,
            probit=False, shslope=False, classic=False,
            classic_mean=None, classic_sigma=None, K=1000, sigma0=2.2,
            Kg=80, tau0=0.0, sigma_init=2.0, tau_init=1.0, phi_init=5.0,
            fit_phi=True, max_iter=300, tol=1e-5, ftol=1e-9, verbose=False,
            init=None, mean_kint=MEAN_KINT, var_kint=VAR_KINT,
            tau_kint=VAR_KINT, accel='squarem'):
    """Fit one C=1 spec. Returns a result dict (unanchored scale).

    accel: 'squarem' (default) runs the EM update inside SQUAREM
    (run_squarem); 'none' iterates it plainly (run_plain_em).  Either
    way max_iter caps the number of EM updates, and the result's
    'iterations' counts them.

    mean_kint, var_kint, tau_kint: interior knots of the mean spline f,
    the log task-spread spline sigma(t) and the log family-spread spline
    tau(t_bar); each spline has kint + 2 degrees of freedom.

    classic: standard IRT with one FIXED prior for every task,
    theta_i ~ N(classic_mean, classic_sigma^2) (see
    classic_prior_constants: the mean log2 length and zm_homo's sigma),
    so the prior carries no task-length info.  Only (alpha_j, beta_j)[,
    phi] are estimated.  The quadrature grid is centred at classic_mean
    (not t_i) so the importance weights don't underflow.  Marginal p is
    constant in task length => time horizons are undefined (NaN).
    Requires fit_mean=False, hetero=False, family='none'.

    shslope: share a single discrimination beta across all models (the
    (alpha, beta) M-step pools beta over every model instead of fitting
    one per model).  Every params[:, 1] then holds the same beta.

    init: optional warm-start dict with any of
      params (n_models,2), coefs, gammas, gam_tau, tau, phi
    (e.g. the full-data fit, for bootstrap replicates). Dimensions must
    match the current bases; entries that don't match are ignored.
    """
    t0 = time.perf_counter()
    if accel not in ACCEL_METHODS:
        raise ValueError(f'accel must be one of {ACCEL_METHODS}, '
                         f'not {accel!r}')
    # classic pins the prior at N(classic_mean, classic_sigma^2), the same
    # for every task (task-length-independent): incompatible with a fitted
    # mean spline or a variance spline (both mean and sigma are frozen).
    # Pinning on the t-scale (mean = t-bar, sigma = zm_homo's sigma-hat,
    # see classic_prior_constants) is affinely equivalent to theta~N(0,1)
    # but keeps beta O(1) and posterior widths resolvable by the fixed
    # sigma0 grid -- the unit-scale variant had posteriors ~5x narrower
    # than the node spacing (quantized E-step, ~400-iter EM crawls,
    # init-dependent optima).
    assert not (classic and (fit_mean or hetero)), \
        'classic requires fit_mean=False, hetero=False'
    assert not classic or (classic_mean is not None
                           and classic_sigma is not None), \
        'classic needs classic_mean and classic_sigma (classic_prior_constants)'
    n_tasks = idx['n_tasks']
    t_array = idx['t_array']

    roots, wts = roots_hermite(K)
    # The grid is a PROPOSAL; the E-step importance-reweights it against
    # the true prior via d0 = theta - grid_center.  For classic the grid
    # must be centred at the (constant) prior mean, not at t_i.
    grid_center = (np.full(n_tasks, float(classic_mean)) if classic
                   else t_array)
    theta_grid = np.sqrt(2.0) * sigma0 * roots[None, :] + grid_center[:, None]
    # extreme GH weights underflow to 0 at large K; log -> -inf is the
    # correct (zero-weight) behavior, so silence the warning.
    with np.errstate(divide='ignore'):
        log_qw = np.log(wts / np.sqrt(np.pi))

    mean_basis = build_monotone_basis(t_array, mean_kint) if fit_mean else None
    X_mean = mean_basis.design(t_array) if fit_mean else None
    if hetero:
        var_basis = NaturalBasis(t_array, var_kint)
        Xv = var_basis.design(t_array)
    else:
        var_basis = None
        Xv = np.ones((n_tasks, 1))
    Btau = None
    tau_basis = None
    if family == 'famspline':
        tau_basis = NaturalBasis(idx['fam_mean_t'], tau_kint)
        Btau = tau_basis.design(idx['fam_mean_t'])

    params = init_alpha_beta(idx, probit)
    if shslope:
        # start every model from the mean per-model slope so the first
        # shared-beta M-step reads a consistent shared value.
        params[:, 1] = float(params[:, 1].mean())
    if fit_mean:
        lb = np.full(X_mean.shape[1], -np.inf)
        lb[1:] = 0.0
        coefs = lsq_linear(X_mean, t_array, bounds=(lb, np.inf),
                           tol=1e-12).x
    else:
        coefs = None
    gammas = np.zeros(Xv.shape[1])
    # classic: sigma frozen at classic_sigma; else start at sigma_init.
    gammas[0] = (math.log(classic_sigma) if classic
                 else math.log(sigma_init))
    tau_floor = TAU_FLOOR
    if family != 'none':
        g_roots, g_w = roots_hermite(Kg)
        with np.errstate(divide='ignore'):
            log_gw = np.log(g_w / np.sqrt(np.pi))
        if tau0 > 0.0:
            # Fixed xi atoms (see kernels.estep_c1_family).  The grid
            # cannot represent a prior narrower than its node spacing:
            # below that the importance weights stop normalizing and
            # harvest spurious logL (Z -> dxi / (tau*sqrt(2pi))), so
            # floor tau at the spacing.  Exactly the SIGMA_FLOOR = 0.3
            # vs theta-spacing 0.2185 fence on the other integral.
            gs = np.sort(g_roots)
            dxi = math.sqrt(2.0) * tau0 * (gs[Kg // 2 + 1] - gs[Kg // 2])
            tau_floor = max(TAU_FLOOR, 0.85 * dxi)
            assert Kg % 2 == 1, \
                'fixed xi grid needs odd Kg for an exact 0 node'
            assert np.isfinite(log_gw).all(), \
                'GH weights underflowed to 0; use Kg <= 321'
        fam_of = idx['fam_of_task']

    n_fams = len(idx['fam_names'])
    if family == 'famspline':
        gam_tau = np.zeros(Btau.shape[1])
        gam_tau[0] = math.log(max(tau_init, tau_floor))
        tau_vec = np.maximum(np.exp(Btau @ gam_tau), tau_floor)
    elif family == 'fam':
        gam_tau = None
        tau_vec = np.full(n_fams, max(tau_init, tau_floor))
    else:
        gam_tau = None
        tau_vec = None
    u_phi = math.log(phi_init)

    if init:
        p0 = init.get('params')
        if p0 is not None and np.shape(p0) == params.shape:
            params = np.array(p0, dtype=float).copy()
            if shslope:
                params[:, 1] = float(params[:, 1].mean())
        c0 = init.get('coefs')
        if fit_mean and c0 is not None and len(c0) == len(coefs):
            coefs = np.maximum(np.array(c0, dtype=float),
                               [-np.inf] + [0.0] * (len(coefs) - 1))
        g0 = init.get('gammas')
        if g0 is not None and len(g0) == len(gammas):
            gammas = np.array(g0, dtype=float).copy()
        gt0 = init.get('gam_tau')
        if (family == 'famspline' and gt0 is not None
                and len(gt0) == len(gam_tau)):
            gam_tau = np.array(gt0, dtype=float).copy()
            tau_vec = np.maximum(np.exp(Btau @ gam_tau), tau_floor)
        if family == 'fam' and init.get('tau'):
            tau_vec = np.full(n_fams, max(float(init['tau']), tau_floor))
        if bb and init.get('phi'):
            u_phi = min(max(math.log(float(init['phi'])),
                            LOG_PHI_BOUNDS[0]), LOG_PHI_BOUNDS[1])

    def em_update(st):
        """One EM update from state `st`: the E-step there, then every
        M-step.  Returns (new state, logL at st, max_change)."""
        # the (alpha, beta) kernels update params in place; SQUAREM still
        # needs the input state after the update, so work on a copy
        params = st['params'].copy()
        coefs, gammas = st['coefs'], st['gammas']
        gam_tau, tau_vec, u_phi = st['gam_tau'], st['tau_vec'], st['u_phi']
        if fit_mean:
            means = X_mean @ coefs
        elif classic:
            means = np.full(n_tasks, float(classic_mean))  # pinned constant
        else:
            means = t_array                     # zm: prior mean = t_i
        sig = np.maximum(np.exp(Xv @ gammas), SIGMA_FLOOR)
        phi = math.exp(u_phi) if bb else 1.0

        # grid_center is the proposal centre passed to the E-step as the
        # d0 reference (t_i normally, 0 for classic) -- must match the grid.
        if family == 'none':
            resp, ll, S1, S2 = KK.estep_c1(
                theta_grid, log_qw, grid_center, means, sig, sigma0, params,
                idx['cell_midx'], idx['cell_s'], idx['cell_n'],
                idx['cell_off'], probit, bb, phi, P_EPS)
            Sthg = None
            Eg = Eg2 = None
        else:
            resp, ll, S1, S2, Sthg, Eg, Eg2 = KK.estep_c1_family(
                theta_grid, log_qw, grid_center, means, sig, sigma0, params,
                idx['cell_midx'], idx['cell_s'], idx['cell_n'],
                idx['cell_off'], idx['fam_offsets'], idx['fam_task_idx'],
                tau_vec, g_roots, log_gw, probit, bb, phi, P_EPS, tau0)

        old = np.concatenate([
            params.ravel(), means, sig,
            tau_vec if tau_vec is not None else np.empty(0),
            np.array([phi]) if bb else np.empty(0)])

        if shslope:
            params = KK.mstep_alpha_shared_beta(
                params, theta_grid, resp, idx['mc_task'], idx['mc_s'],
                idx['mc_n'], idx['mc_off'], probit, bb, phi, P_EPS)
        else:
            params = KK.mstep_alpha_beta(
                params, theta_grid, resp, idx['mc_task'], idx['mc_s'],
                idx['mc_n'], idx['mc_off'], probit, bb, phi, P_EPS)

        if fit_mean:
            y = S1.copy()
            if family != 'none':
                y -= Eg[fam_of]
            w = 1.0 / np.maximum(sig, 1e-12) ** 2
            coefs = monotone_wls(X_mean, y, w, coefs)
            means = X_mean @ coefs

        # variance M-step (skipped for classic: sigma is frozen at 1).
        if not classic:
            # residual second moment about the UPDATED mean (and g)
            if family == 'none':
                sres = S2 - 2.0 * means * S1 + means ** 2
            else:
                sres = (S2 - 2.0 * means * S1 - 2.0 * Sthg
                        + 2.0 * means * Eg[fam_of] + means ** 2
                        + Eg2[fam_of])
            sres = np.maximum(sres, 1e-12)
            gammas = KK.mstep_log_variance(Xv, np.ones(n_tasks), sres,
                                           gammas)

        if family == 'fam':
            tau_vec = np.full(n_fams,
                              max(math.sqrt(float(Eg2.mean())), tau_floor))
        elif family == 'famspline':
            gam_tau = KK.mstep_log_variance(
                Btau, np.ones(n_fams), np.maximum(Eg2, 1e-12), gam_tau)
            tau_vec = np.maximum(np.exp(Btau @ gam_tau), tau_floor)

        if bb and fit_phi:
            u_phi = _mstep_phi(u_phi, theta_grid, resp, params, idx,
                               probit, P_EPS)

        sig = np.maximum(np.exp(Xv @ gammas), SIGMA_FLOOR)
        new = np.concatenate([
            params.ravel(), means, sig,
            tau_vec if tau_vec is not None else np.empty(0),
            np.array([math.exp(u_phi)]) if bb else np.empty(0)])
        max_change = float(np.max(np.abs(new - old)))
        return ({'params': params, 'coefs': coefs, 'gammas': gammas,
                 'gam_tau': gam_tau, 'tau_vec': tau_vec, 'u_phi': u_phi},
                ll, max_change)

    # SQUAREM extrapolates on one flat vector of the free parameters,
    # with the constrained ones on scales where any real value is
    # admissible or a simple projection restores admissibility.
    n_params = params.size

    def pack(st):
        parts = [st['params'].ravel()]
        if fit_mean:
            parts.append(st['coefs'])
        parts.append(st['gammas'])               # log sigma(t): free
        if family == 'famspline':
            parts.append(st['gam_tau'])          # log tau(t_bar): free
        elif family == 'fam':
            parts.append([math.log(st['tau_vec'][0])])
        if bb:
            parts.append([st['u_phi']])
        return np.concatenate(parts)

    def unpack(x):
        st = {'params': x[:n_params].reshape(params.shape).copy(),
              'coefs': None, 'gam_tau': None, 'tau_vec': None,
              'u_phi': u_phi}
        i = n_params
        if fit_mean:
            # coefficients past the intercept must stay >= 0 for f to be
            # increasing (monotone_wls's box): project onto the box
            c = x[i:i + len(coefs)].copy()
            c[1:] = np.maximum(c[1:], 0.0)
            st['coefs'] = c
            i += len(coefs)
        st['gammas'] = x[i:i + len(gammas)].copy()
        i += len(gammas)
        if family == 'famspline':
            st['gam_tau'] = x[i:i + len(gam_tau)].copy()
            st['tau_vec'] = np.maximum(np.exp(Btau @ st['gam_tau']),
                                       tau_floor)
            i += len(gam_tau)
        elif family == 'fam':
            st['tau_vec'] = np.full(n_fams, max(math.exp(x[i]), tau_floor))
            i += 1
        if bb:
            st['u_phi'] = min(max(float(x[i]), LOG_PHI_BOUNDS[0]),
                              LOG_PHI_BOUNDS[1])
        return st

    state = {'params': params, 'coefs': coefs, 'gammas': gammas,
             'gam_tau': gam_tau, 'tau_vec': tau_vec, 'u_phi': u_phi}
    if accel == 'squarem':
        state, ll, history, n_updates, converged, n_cycles = run_squarem(
            em_update, pack, unpack, state, max_iter, tol, ftol, verbose)
    else:
        state, ll, history, n_updates, converged = run_plain_em(
            em_update, state, max_iter, tol, ftol, verbose)
        n_cycles = None
    params, coefs, gammas = state['params'], state['coefs'], state['gammas']
    gam_tau, tau_vec, u_phi = state['gam_tau'], state['tau_vec'], \
        state['u_phi']

    # EM ascent sentinel.  Nothing here ever looked at the SIGN of the
    # logl increments, though they were already in history -- which is
    # why a broken-monotonicity bug in the family E-step had to be found
    # by hand from the penalized pilot.  Two lines make it standing.
    # Under SQUAREM the history holds the logL at each cycle start, which
    # the safeguard keeps from falling wherever EM itself would not.
    _lls = np.array([h['logl'] for h in history], dtype=float)
    min_ll_diff = float(np.min(np.diff(_lls))) if len(_lls) > 1 else 0.0

    return {
        'fit_mean': fit_mean, 'hetero': hetero, 'family': family,
        'bb': bb, 'probit': probit, 'shslope': shslope, 'classic': classic,
        'classic_mean': (float(classic_mean) if classic else None),
        'classic_sigma': (float(classic_sigma) if classic else None),
        'K': K, 'sigma0': sigma0, 'Kg': Kg,
        'tau0': float(tau0), 'tau_floor': float(tau_floor),
        'min_ll_diff': min_ll_diff,
        'ascent_ok': bool(min_ll_diff > -1e-6 * max(1.0, abs(ll))),
        'params': params, 'coefs': coefs, 'gammas': gammas,
        'gam_tau': gam_tau,
        'tau': (float(tau_vec[0]) if family == 'fam' else None),
        'tau_vec': tau_vec,
        'phi': (math.exp(u_phi) if bb else None),
        'mean_basis': mean_basis, 'var_basis': var_basis,
        'tau_basis': tau_basis,
        'logl': float(ll), 'history': history, 'accel': accel,
        'iterations': n_updates, 'cycles': n_cycles,
        'converged': converged, 'align_c': 1.0, 'align_d': 0.0,
        'x_lo': float(t_array.min()), 'x_hi': float(t_array.max()),
        'wall_s': time.perf_counter() - t0,
    }


def anchor_identity(fit, idx):
    """Resolve the affine ridge by identity-at-endpoints anchoring of the
    mean curve: f(a)=a, f(b)=b.  In-place; no-op for zm specs."""
    if not fit['fit_mean']:
        return fit
    nb = fit['mean_basis']
    X_ab = nb.design(np.array([nb.lo, nb.hi]))
    fa, fb = X_ab @ fit['coefs']
    c = (fb - fa) / (nb.hi - nb.lo)
    d = fa - c * nb.lo
    assert c > 0, 'anchoring requires f(b) > f(a)'
    e0 = np.zeros_like(fit['coefs'])
    e0[0] = 1.0
    fit['coefs'] = (fit['coefs'] - d * e0) / c
    fit['gammas'] = fit['gammas'].copy()
    fit['gammas'][0] -= math.log(c)
    if fit['family'] == 'fam':
        fit['tau'] = fit['tau'] / c
        fit['tau_vec'] = fit['tau_vec'] / c
    elif fit['family'] == 'famspline':
        fit['gam_tau'] = fit['gam_tau'].copy()
        fit['gam_tau'][0] -= math.log(c)
        fit['tau_vec'] = fit['tau_vec'] / c
    # tau_floor lives on the same scale as tau, so anchoring must carry
    # it along -- otherwise prediction-time flooring (prior_curves,
    # joint_loglik_eval) applies a PRE-anchor floor to POST-anchor taus
    # and silently evaluates a different model than the one fitted.
    if fit.get('tau_floor') is not None:
        fit['tau_floor'] = float(fit['tau_floor']) / c
    p = fit['params']
    p[:, 0] = p[:, 0] + p[:, 1] * d
    p[:, 1] = c * p[:, 1]
    fit['align_c'], fit['align_d'] = float(c), float(d)
    return fit


# ----------------------------------------------------------------------
# prediction-scale helpers (marginal p, joint loglik, horizons)
# ----------------------------------------------------------------------

def prior_curves(fit, x, fam_tbar=None):
    """(h(x), s(x)) of the effective single-Gaussian prior at lengths x.

    Family specs marginalize the (unknown-family) effect into the scale:
    s^2 = sigma(x)^2 + tau^2, with tau evaluated at fam_tbar (defaults to
    x itself -- 'a fresh family whose mean length is x') for famspline."""
    x = np.asarray(x, dtype=float)
    if fit['fit_mean']:
        h = fit['mean_basis'].design(x) @ fit['coefs']
    elif fit.get('classic'):
        # pinned constant prior mean -- no task-length dependence
        h = np.full(len(x), fit['classic_mean'])
    else:
        h = x.copy()                    # zm: prior mean = t_i
    if fit['hetero']:
        s = np.maximum(np.exp(fit['var_basis'].design(x) @ fit['gammas']),
                       SIGMA_FLOOR)
    else:
        s = np.full(len(x), max(math.exp(fit['gammas'][0]), SIGMA_FLOOR))
    if fit['family'] == 'fam':
        s = np.sqrt(s ** 2 + fit['tau'] ** 2)
    elif fit['family'] == 'famspline':
        tb = x if fam_tbar is None else np.asarray(fam_tbar, dtype=float)
        tau = np.maximum(
            np.exp(fit['tau_basis'].design(tb) @ fit['gam_tau']),
            fit.get('tau_floor', TAU_FLOOR))
        s = np.sqrt(s ** 2 + tau ** 2)
    return h, s


_GH_EVAL = roots_hermite(200)


def marginal_p(fit, x, model_indices=None, fam_tbar=None):
    """Marginal success probabilities p_j(x): (len(x), n_models_sel)."""
    h, s = prior_curves(fit, x, fam_tbar=fam_tbar)
    gr, gw = _GH_EVAL
    gwn = gw / np.sqrt(np.pi)
    params = fit['params']
    sel = (np.arange(params.shape[0]) if model_indices is None
           else np.asarray(model_indices))
    out = np.empty((len(h), len(sel)))
    for c, j in enumerate(sel):
        out[:, c] = KK.marginal_p_grid(params[j, 0], params[j, 1], h, s,
                                       gr, gwn, fit['probit'])
    return out


def time_horizon_marginal(fit, q, n_grid=481):
    """th_q per model in MINUTES by inverting the marginal curve p_j(t)
    on a dense length grid (NaN when q is not crossed in range)."""
    nb_lo, nb_hi = _range_of(fit)
    xg = np.linspace(nb_lo, nb_hi, n_grid)
    P = marginal_p(fit, xg)
    out = np.full(P.shape[1], np.nan)
    for j in range(P.shape[1]):
        p = P[:, j]
        # p is decreasing in x (beta < 0); interpolate the q crossing.
        ps = p[::-1]
        xs = xg[::-1]
        if q < ps[0] or q > ps[-1]:
            continue
        out[j] = 2.0 ** float(np.interp(q, ps, xs))
    return out


def _range_of(fit):
    if 'x_lo' in fit:
        return fit['x_lo'], fit['x_hi']
    for key in ('mean_basis', 'var_basis'):
        if fit.get(key) is not None:
            return fit[key].lo, fit[key].hi
    raise ValueError('fit has no basis range')


def joint_loglik_eval(fit, idx, n_nodes=200, family_scheme=False):
    """Prior-predictive joint log-lik of `idx` under `fit`.

    family_scheme=True and a family fit: integrate g at family level
    (whole families held out together, tau at the family's known tbar).
    Otherwise family fits marginalize g per task into the prior scale."""
    gr, gw = roots_hermite(n_nodes)
    log_gw = np.log(gw / np.sqrt(np.pi))
    phi = fit['phi'] if fit['bb'] else 1.0
    t = idx['t_array']
    if fit['family'] != 'none' and family_scheme:
        h, _ = prior_curves(fit, t)
        if fit['hetero']:
            s = np.maximum(
                np.exp(fit['var_basis'].design(t) @ fit['gammas']),
                SIGMA_FLOOR)
        else:
            s = np.full(len(t), max(math.exp(fit['gammas'][0]),
                                    SIGMA_FLOOR))
        if fit['family'] == 'fam':
            tau_vec = np.full(len(idx['fam_names']), fit['tau'])
        else:
            tau_vec = np.maximum(
                np.exp(fit['tau_basis'].design(idx['fam_mean_t'])
                       @ fit['gam_tau']), fit.get('tau_floor', TAU_FLOOR))
        g_roots, g_w = roots_hermite(80)
        per_fam = KK.joint_ll_family(
            t, h, s, fit['params'], idx['cell_midx'], idx['cell_s'],
            idx['cell_n'], idx['cell_off'], idx['fam_offsets'],
            idx['fam_task_idx'], tau_vec, gr, log_gw,
            g_roots, np.log(g_w / np.sqrt(np.pi)),
            fit['probit'], fit['bb'], phi, P_EPS)
        return float(per_fam.sum())
    h, s = prior_curves(fit, t, fam_tbar=(
        idx['fam_mean_t'][idx['fam_of_task']]
        if fit['family'] == 'famspline' else None))
    per_task = KK.joint_ll_pertask(
        t, h, s, fit['params'], idx['cell_midx'], idx['cell_s'],
        idx['cell_n'], idx['cell_off'], gr, log_gw,
        fit['probit'], fit['bb'], phi, P_EPS)
    return float(per_task.sum())
