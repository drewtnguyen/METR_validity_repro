"""Anchored bumps: compact corrections fitted on top of a fixed core fit.

A bump is a smooth, compactly supported correction over one window of
task lengths [start, end] (in minutes).  On x = log2 minutes its shape is

    B(x) = 16 u^2 (1 - u)^2,   u = (x - log2 start) / (log2 end - log2 start),

zero outside the window, C1 at both ends, and 1 at the window's
log-midpoint.  This is the shoulder kernel of late_fork_pilot and
canonical_bump_screen in the surrounding repository.

There are two kinds:

- A **general** bump shifts the core's difficulty curve, the same for
  every AI.  For an IRT spec it is added to the prior mean of theta,
  m(t) = f(t) + a B(t), so its amplitude a is in theta units.  For a
  `shared` spec (Model 1) it is added to the shape, f(x) + c B(x), so
  its amplitude c is in units of the shape's 0 -> 1 range.  A positive
  amplitude makes tasks in the window harder in both cases.
- An **AI-specific** bump adds d_j B(x) to AI j's success logit.  The
  d_j have an exposure-weighted zero mean, sum_j w_j d_j = 0 with
  w_j = sum over AI j's cells of n_ij B(t_i) (runs in the window), so
  they describe how the AIs DEPART from the common correction.  To also
  fit the common part, add a general bump over the same window.

A spec lists its bumps explicitly; the "double shoulder" of the pilots
is two general bumps, over [1 h, 4 h] and [4 h, 16 h]:

    from horizon_irt.bumps import bump
    bumps = (bump(60, 240), bump(240, 960))
    bumps += (bump(240, 960, ai_specific=True),)   # + AI departures

**Anchored fit.**  The core spec is fitted first, as usual.  Its
parameters are then held fixed and only the bump amplitudes are fitted,
by maximizing the core's own likelihood (the IRT marginal likelihood
with theta and the family effect integrated out, or the basic spec's
cell likelihood) plus a Gaussian log prior on the amplitudes:

- general: N(0, (prior_sd / |beta|)^2) in the amplitude's own units,
  with |beta| the core's mean |beta_j| (IRT) or mean b_j (`shared`),
  i.e. prior_sd on the logit scale.  Default prior_sd 2.
- AI-specific: N(0, prior_sd^2) on each d_j (logit scale), a ridge.
  Default prior_sd 0.25.

With `monotone=True` (the default) the general bumps must keep the
difficulty curve non-decreasing: m(t) for IRT, the shape for `shared`,
checked on a 600-point grid over the core's length range (a linear
constraint, so the solver is SLSQP).  AI-specific bumps are not
constrained; `bump_fit['max_marginal_rise']` reports the largest rise
of any AI's marginal success curve with length.

The IRT likelihood uses the EM's own quadrature: a fixed Gauss-Hermite
grid of K nodes per task (scale sigma0 = 2.2, centred at the core's
prior mean), importance-reweighted to the bumped prior, nested inside Kg
family nodes for family specs.  All bumps zero reproduces the core fit
exactly, so `bump_fit['gain']` is the log-likelihood the bumps add.
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit
from scipy.linalg import null_space
from scipy.optimize import minimize
from scipy.special import expit, log_ndtr, ndtr, roots_hermite

from . import basic, em
from . import kernels as KK

GENERAL_PRIOR_SD = 2.0        # logit scale; divided by |beta| for theta
AI_PRIOR_SD = 0.25            # logit scale, per AI
MONO_GRID = 600               # grid for the monotone-curve constraint
SIGMA0 = 2.2                  # quadrature scale, as in em.fit_irt
N_EVAL_NODES = 200            # prediction-time GH nodes, as in em


# ----------------------------------------------------------------------
# bump specifications
# ----------------------------------------------------------------------

def bump(start, end, ai_specific=False, prior_sd=None):
    """One bump over [start, end] minutes, as a spec entry."""
    out = {'start': float(start), 'end': float(end),
           'ai_specific': bool(ai_specific)}
    if prior_sd is not None:
        out['prior_sd'] = float(prior_sd)
    return out


def normalize(bumps):
    """Validated bump dicts with defaults filled in.

    Each entry is a dict with keys start, end (minutes), optional
    ai_specific (default False) and prior_sd, or a tuple
    (start, end[, ai_specific])."""
    out = []
    for b in bumps:
        if isinstance(b, (tuple, list)):
            b = bump(*b)
        unknown = set(b) - {'start', 'end', 'ai_specific', 'prior_sd'}
        if unknown:
            raise KeyError(f'unknown bump keys {sorted(unknown)}')
        start, end = float(b['start']), float(b['end'])
        if not 0 < start < end:
            raise ValueError(f'bump window must satisfy 0 < start < end '
                             f'(minutes), got [{start}, {end}]')
        ai = bool(b.get('ai_specific', False))
        sd = float(b.get('prior_sd',
                         AI_PRIOR_SD if ai else GENERAL_PRIOR_SD))
        if not (math.isfinite(sd) and sd > 0):
            raise ValueError('bump prior_sd must be finite and positive')
        out.append({'start': start, 'end': end, 'ai_specific': ai,
                    'prior_sd': sd})
    return out


def check_supported(cfg):
    """Raise unless the spec can carry bumps."""
    if cfg['cls'] == 'irt':
        if cfg.get('classic'):
            raise ValueError('bumps need a length-dependent prior; '
                             'classic specs have none')
        return
    if cfg.get('kind') != 'shared':
        raise ValueError("bumps are implemented for IRT specs and "
                         "kind='shared' (Model 1), not "
                         f"kind={cfg.get('kind')!r}")


def kernel(x, start, end):
    """B(x) for a window [start, end] in minutes; x in log2 minutes."""
    lo, hi = math.log2(start), math.log2(end)
    u = (np.asarray(x, dtype=float) - lo) / (hi - lo)
    return np.where((u > 0) & (u < 1), 16.0 * u ** 2 * (1.0 - u) ** 2, 0.0)


def has_bumps(fit):
    return bool(fit.get('bumps'))


def general_shift(fit, x):
    """Sum of the fitted general bumps at x (theta or shape units)."""
    x = np.asarray(x, dtype=float)
    out = np.zeros(len(x))
    for b in fit.get('bumps') or ():
        if not b['ai_specific']:
            out += b['amplitude'] * kernel(x, b['start'], b['end'])
    return out


def ai_offsets(fit, x):
    """(len(x), n_AIs) logit offsets from the fitted AI-specific bumps."""
    x = np.asarray(x, dtype=float)
    J = len(fit['model_names'])
    out = np.zeros((len(x), J))
    for b in fit.get('bumps') or ():
        if b['ai_specific']:
            out += np.outer(kernel(x, b['start'], b['end']),
                            np.asarray(b['amplitudes'], dtype=float))
    return out


# ----------------------------------------------------------------------
# numba kernels for the anchored IRT likelihood
# ----------------------------------------------------------------------

@njit(cache=True)
def _ll_rows(theta_grid, rows, params, delta, cell_midx, cell_s, cell_n,
             cell_off, probit, use_bb, phi, p_eps):
    """Data log-lik table for the tasks `rows`, with per-cell logit
    offsets delta (indexed by global cell), and d ll / d eta for each of
    those tasks' cells at each node (cells in row order)."""
    K = theta_grid.shape[1]
    n_r = len(rows)
    n_c = 0
    for q in range(n_r):
        n_c += cell_off[rows[q] + 1] - cell_off[rows[q]]
    ll = np.zeros((n_r, K))
    dll = np.empty((n_c, K))
    pos = 0
    for q in range(n_r):
        i = rows[q]
        for c in range(cell_off[i], cell_off[i + 1]):
            m = cell_midx[c]
            a = params[m, 0] + delta[c]
            b = params[m, 1]
            for k in range(K):
                eta = a + b * theta_grid[i, k]
                ll[q, k] += KK._cell_ll(cell_s[c], cell_n[c], eta, probit,
                                        use_bb, phi, p_eps)
                dll[pos, k] = KK._cell_dll_deta(cell_s[c], cell_n[c], eta,
                                                probit, use_bb, phi,
                                                p_eps)[0]
            pos += 1
    return ll, dll


@njit(cache=True)
def _marginal_ll(theta_grid, log_qw, center, sigma0, ll_tk, means, sigmas,
                 fam_offsets, fam_task_idx, tau_vec, g_roots, log_gw,
                 use_family):
    """Marginal log-lik on the fixed importance-weighted grid, as in
    kernels.estep_c1[_family] (tau-scaled family atoms, tau0 = 0).

    Returns (total, dm, P): dm_i = d total / d (prior mean of task i),
    exact for this quadrature since the grid does not move, and P the
    posterior weight of each (task, node), marginal over the family
    effect."""
    n_tasks, K = theta_grid.shape
    dm = np.zeros(n_tasks)
    P = np.zeros((n_tasks, K))
    total = 0.0
    log_s0 = math.log(sigma0)
    h0 = 0.5 / (sigma0 * sigma0)
    if not use_family:
        lw = np.empty(K)
        for i in range(n_tasks):
            s_i = sigmas[i]
            log_s = math.log(s_i)
            half_inv = 0.5 / (s_i * s_i)
            mx = -np.inf
            for k in range(K):
                th = theta_grid[i, k]
                d0 = th - center[i]
                z = th - means[i]
                v = (log_qw[k] + ll_tk[i, k] + log_s0 + d0 * d0 * h0
                     - log_s - z * z * half_inv)
                lw[k] = v
                if v > mx:
                    mx = v
            acc = 0.0
            for k in range(K):
                acc += math.exp(lw[k] - mx)
            logden = mx + math.log(acc)
            total += logden
            for k in range(K):
                r = math.exp(lw[k] - logden)
                P[i, k] = r
                dm[i] += r * (theta_grid[i, k] - means[i]) / (s_i * s_i)
        return total, dm, P

    Kg = len(g_roots)
    n_fams = len(fam_offsets) - 1
    for fi in range(n_fams):
        f0 = fam_offsets[fi]
        n_in = fam_offsets[fi + 1] - f0
        g_scale = math.sqrt(2.0) * tau_vec[fi]
        lw = np.empty((Kg, n_in, K))
        tden = np.empty((Kg, n_in))
        log_pf = np.empty(Kg)
        for a in range(Kg):
            g = g_scale * g_roots[a]
            tot = 0.0
            for jj in range(n_in):
                i = fam_task_idx[f0 + jj]
                s_i = sigmas[i]
                log_s = math.log(s_i)
                half_inv = 0.5 / (s_i * s_i)
                mu = means[i] + g
                mx = -np.inf
                for k in range(K):
                    th = theta_grid[i, k]
                    d0 = th - center[i]
                    z = th - mu
                    v = (log_qw[k] + ll_tk[i, k] + log_s0 + d0 * d0 * h0
                         - log_s - z * z * half_inv)
                    lw[a, jj, k] = v
                    if v > mx:
                        mx = v
                acc = 0.0
                for k in range(K):
                    acc += math.exp(lw[a, jj, k] - mx)
                tden[a, jj] = mx + math.log(acc)
                tot += tden[a, jj]
            log_pf[a] = log_gw[a] + tot
        mxg = -np.inf
        for a in range(Kg):
            if log_pf[a] > mxg:
                mxg = log_pf[a]
        accg = 0.0
        for a in range(Kg):
            accg += math.exp(log_pf[a] - mxg)
        log_den = mxg + math.log(accg)
        total += log_den
        for a in range(Kg):
            wg = math.exp(log_pf[a] - log_den)
            if wg < 1e-300:
                continue
            g = g_scale * g_roots[a]
            for jj in range(n_in):
                i = fam_task_idx[f0 + jj]
                s2 = sigmas[i] * sigmas[i]
                for k in range(K):
                    w_ik = wg * math.exp(lw[a, jj, k] - tden[a, jj])
                    P[i, k] += w_ik
                    dm[i] += w_ik * (theta_grid[i, k] - means[i] - g) / s2
    return total, dm, P


# ----------------------------------------------------------------------
# shared pieces
# ----------------------------------------------------------------------

def _irt_prior_parts(fit, t, fam_mean_t):
    """Core prior mean h(t), task sd sigma(t) (no family part) and the
    per-family tau, mirroring em.joint_loglik_eval."""
    h, _ = em.prior_curves(fit, t)
    if fit['hetero']:
        sig = np.maximum(np.exp(fit['var_basis'].design(t) @ fit['gammas']),
                         em.SIGMA_FLOOR)
    else:
        sig = np.full(len(t), max(math.exp(fit['gammas'][0]),
                                  em.SIGMA_FLOOR))
    tau_vec = None
    if fit['family'] == 'fam':
        tau_vec = np.full(len(fam_mean_t), float(fit['tau']))
    elif fit['family'] == 'famspline':
        tau_vec = np.maximum(
            np.exp(fit['tau_basis'].design(fam_mean_t) @ fit['gam_tau']),
            fit.get('tau_floor', em.TAU_FLOOR))
    return h, sig, tau_vec


def _range(fit):
    if fit['cls'] == 'irt':
        return em._range_of(fit)
    return fit['basis'].lo, fit['basis'].hi


def _core_curve(fit, x):
    """The core's difficulty curve on x: IRT prior mean or shared shape."""
    if fit['cls'] == 'irt':
        return em.prior_curves(fit, x)[0]
    nb = fit['basis']
    return (nb.raw_merged(x) / fit['col_hi']) @ fit['w']


def _slope_scale(fit):
    """Mean |logit slope| of the core: converts logit-scale prior sds to
    the general amplitude's units."""
    if fit['cls'] == 'irt':
        return float(np.mean(np.abs(fit['params'][:, 1])))
    return float(np.mean(np.abs(fit['b'])))


class _Layout:
    """Parameter vector z = [general amplitudes, then for each AI-specific
    bump its J-1 coordinates in the exposure-weighted zero-mean
    subspace].  d_l = Z_l gamma_l with Z_l orthonormal, so the ridge on
    d_l is the ridge on gamma_l."""

    def __init__(self, bumps, idx, x_task):
        self.bumps = bumps
        self.gen = [b for b in bumps if not b['ai_specific']]
        self.ai = [b for b in bumps if b['ai_specific']]
        J = idx['n_models']
        cell_task = np.repeat(np.arange(idx['n_tasks']),
                              np.diff(idx['cell_off']))
        self.B_task = (np.column_stack([kernel(x_task, b['start'], b['end'])
                                        for b in self.gen])
                       if self.gen else np.zeros((len(x_task), 0)))
        self.R_cell = []          # per AI bump: B at each cell's length
        self.Z = []               # per AI bump: (J, J-1) basis
        self.w_exp = []
        for b in self.ai:
            r = kernel(x_task, b['start'], b['end'])[cell_task]
            w = np.bincount(idx['cell_midx'], idx['cell_n'] * r,
                            minlength=J)
            self.R_cell.append(r)
            self.w_exp.append(w)
            self.Z.append(null_space(w[None, :]) if w.sum() > 0
                          else np.eye(J))
        self.n_gen = len(self.gen)
        self.sizes = [Z.shape[1] for Z in self.Z]
        self.n = self.n_gen + sum(self.sizes)

    def split(self, z):
        a = z[:self.n_gen]
        gam, pos = [], self.n_gen
        for size in self.sizes:
            gam.append(z[pos:pos + size])
            pos += size
        return a, gam

    def deltas(self, gam, cell_midx):
        """Per-cell logit offsets and the per-bump AI amplitudes d_l."""
        d = [Z @ g for Z, g in zip(self.Z, gam)]
        delta = np.zeros(len(cell_midx))
        for r, dl in zip(self.R_cell, d):
            delta += r * dl[cell_midx]
        return delta, d

    def log_prior(self, a, gam, slope):
        """log prior and its gradient wrt (a, gamma)."""
        sd_a = np.array([b['prior_sd'] for b in self.gen]) / slope
        lp = -0.5 * float(np.sum((a / sd_a) ** 2))
        g_a = -a / sd_a ** 2
        g_gam = []
        for b, g in zip(self.ai, gam):
            lp -= 0.5 * float(np.sum(g ** 2)) / b['prior_sd'] ** 2
            g_gam.append(-g / b['prior_sd'] ** 2)
        return lp, g_a, g_gam

    def grad_gamma(self, g_delta, cell_midx, J):
        """Chain d/d delta_c -> d/d gamma_l."""
        out = []
        for r, Z in zip(self.R_cell, self.Z):
            g_d = np.bincount(cell_midx, g_delta * r, minlength=J)
            out.append(Z.T @ g_d)
        return out


def _monotone_constraint(fit, layout):
    """SLSQP constraint: the core curve plus the general bumps must not
    decrease between consecutive points of a dense length grid."""
    lo, hi = _range(fit)
    xg = np.linspace(lo, hi, MONO_GRID)
    dh = np.diff(_core_curve(fit, xg))
    dB = np.diff(np.column_stack([kernel(xg, b['start'], b['end'])
                                  for b in layout.gen]), axis=0)
    n_rest = layout.n - layout.n_gen
    jac = np.hstack([dB, np.zeros((len(dh), n_rest))])
    return dict(type='ineq',
                fun=lambda z: dh + dB @ z[:layout.n_gen],
                jac=lambda z: jac)


def _optimize(obj, layout, fit, monotone):
    z0 = np.zeros(layout.n)
    if monotone and layout.n_gen:
        res = minimize(obj, z0, jac=True, method='SLSQP',
                       constraints=[_monotone_constraint(fit, layout)],
                       options=dict(maxiter=500, ftol=1e-10))
    else:
        res = minimize(obj, z0, jac=True, method='L-BFGS-B',
                       options=dict(maxiter=5000, ftol=1e-14, gtol=1e-8))
    return res


# ----------------------------------------------------------------------
# anchored fits
# ----------------------------------------------------------------------

def fit_bumps(core, idx, bumps, monotone=True, K=1000, Kg=80):
    """Fit `bumps` on top of the fixed `core` fit; returns a new fit dict
    (the core's keys plus 'bumps' and 'bump_fit')."""
    bumps = normalize(bumps)
    if not bumps:
        return core
    if core['cls'] == 'irt':
        return _fit_irt(core, idx, bumps, monotone, K, Kg)
    return _fit_shared(core, idx, bumps, monotone)


def _finish(core, layout, z, res, monotone, logl, logl0, lp, extra):
    a, gam = layout.split(z)
    J = len(core['model_names'])
    d = [Z @ g for Z, g in zip(layout.Z, gam)]
    out_bumps, ia, il = [], 0, 0
    for b in layout.bumps:
        e = dict(b)
        if b['ai_specific']:
            e['amplitudes'] = d[il]
            e['exposure'] = layout.w_exp[il]
            il += 1
        else:
            e['amplitude'] = float(a[ia])
            ia += 1
        out_bumps.append(e)
    fit = dict(core)
    fit['bumps'] = out_bumps
    fit['core_logl'] = core['logl']
    fit['logl'] = float(logl)
    fit['n_params'] = core.get('n_params', 0) + layout.n_gen + sum(
        layout.sizes)
    info = {'monotone': bool(monotone), 'gain': float(logl - logl0),
            'log_prior': float(lp), 'success': bool(res.success),
            'message': str(res.message), 'nit': int(res.nit),
            'n_amplitudes': int(layout.n)}
    info.update(extra)
    fit['bump_fit'] = info
    # diagnostics on the bumped curves
    lo, hi = _range(fit)
    xg = np.linspace(lo, hi, MONO_GRID)
    curve = _core_curve(fit, xg) + general_shift(fit, xg)
    info['curve_monotone'] = bool(np.all(np.diff(curve) >= -1e-8))
    P = marginal_p(fit, xg)
    info['max_marginal_rise'] = float(np.max(
        P - np.minimum.accumulate(P, axis=0)))
    assert J == P.shape[1]
    return fit


def _fit_irt(core, idx, bumps, monotone, K, Kg):
    t = idx['t_array']
    J = idx['n_models']
    midx, s, n = idx['cell_midx'], idx['cell_s'], idx['cell_n']
    off = idx['cell_off']
    h, sig, tau_vec = _irt_prior_parts(core, t, idx['fam_mean_t'])
    use_family = tau_vec is not None
    if not use_family:
        tau_vec = np.ones(1)
    probit, use_bb = bool(core['probit']), bool(core['bb'])
    phi = float(core['phi']) if use_bb else 1.0
    params = np.ascontiguousarray(core['params'], dtype=float)

    # fixed grid centred at the core's prior mean; nodes whose GH weight
    # underflows carry exactly zero mass and are dropped
    roots, wts = roots_hermite(K)
    with np.errstate(divide='ignore'):
        log_qw = np.log(wts / np.sqrt(np.pi))
    keep = np.isfinite(log_qw)
    roots, log_qw = roots[keep], log_qw[keep]
    theta_grid = np.sqrt(2.0) * SIGMA0 * roots[None, :] + h[:, None]
    g_roots, g_w = roots_hermite(Kg)
    log_gw = np.log(g_w / np.sqrt(np.pi))

    layout = _Layout(bumps, idx, t)
    slope = _slope_scale(core)
    ll_base = KK._ll_table(theta_grid, params, midx, s, n, off, probit,
                           use_bb, phi, em.P_EPS)
    # tasks touched by an AI-specific bump, and their cells in row order
    cell_task = np.repeat(np.arange(idx['n_tasks']), np.diff(off))
    touched = np.zeros(len(midx), dtype=bool)
    for r in layout.R_cell:
        touched |= r > 0
    rows = np.unique(cell_task[touched]).astype(np.int64)
    row_cells = (np.concatenate([np.arange(off[i], off[i + 1])
                                 for i in rows])
                 if len(rows) else np.zeros(0, dtype=np.int64))
    fam_off = idx['fam_offsets'].astype(np.int64)
    fam_idx = idx['fam_task_idx'].astype(np.int64)

    def loglik(z, need_grad=True):
        a, gam = layout.split(z)
        means = h + layout.B_task @ a
        delta, _ = layout.deltas(gam, midx)
        ll_tk = ll_base
        dll = None
        if len(rows):
            ll_tk = ll_base.copy()
            ll_rows, dll = _ll_rows(theta_grid, rows, params, delta, midx,
                                    s, n, off, probit, use_bb, phi,
                                    em.P_EPS)
            ll_tk[rows] = ll_rows
        tot, dm, P = _marginal_ll(theta_grid, log_qw, h, SIGMA0, ll_tk,
                                  means, sig, fam_off, fam_idx, tau_vec,
                                  g_roots, log_gw, use_family)
        if not need_grad:
            return tot
        g_a = layout.B_task.T @ dm
        g_delta = np.zeros(len(midx))
        if len(rows):
            g_delta[row_cells] = np.sum(P[cell_task[row_cells]] * dll,
                                        axis=1)
        return tot, g_a, layout.grad_gamma(g_delta, midx, J)

    def obj(z):
        tot, g_a, g_gam = loglik(z)
        a, gam = layout.split(z)
        lp, pa, pg = layout.log_prior(a, gam, slope)
        grad = np.concatenate([g_a + pa] + [x + y for x, y in zip(g_gam, pg)])
        return -(tot + lp), -grad

    z0 = np.zeros(layout.n)
    logl0 = loglik(z0, need_grad=False)
    res = _optimize(obj, layout, core, monotone)
    z = res.x
    logl = loglik(z, need_grad=False)
    a, gam = layout.split(z)
    lp = layout.log_prior(a, gam, slope)[0]
    return _finish(core, layout, z, res, monotone, logl, logl0, lp,
                   {'K': int(K), 'Kg': int(Kg)})


def _fit_shared(core, idx, bumps, monotone):
    cv = basic.cell_view(idx)
    J = idx['n_models']
    midx, s, n = cv['midx'], cv['s'], cv['n']
    probit, bb = bool(core['probit']), bool(core.get('bb'))
    f_cells = _core_curve(core, cv['x'])
    a_j, b_j = np.asarray(core['a']), np.asarray(core['b'])
    r_cells = np.asarray(core['r'])[midx] if bb else None
    # sqrtfamily: the core's weighted objective, rescaled to one unit
    # of weight per run so the priors weigh the same as when unweighted
    if core.get('sqrtfamily'):
        w = basic.sqrt_family_weights(idx)
        w = w * n.sum() / np.sum(w * n)
    else:
        w = np.ones(len(midx))

    layout = _Layout(bumps, idx, idx['t_array'])
    B_cells = layout.B_task[cv['task']]
    slope = _slope_scale(core)

    def cell_terms(z):
        a, gam = layout.split(z)
        delta, _ = layout.deltas(gam, midx)
        eta = a_j[midx] - b_j[midx] * (f_cells + B_cells @ a) + delta
        if bb:
            l, g, _ = basic._bb_ll_grads(eta, s, n, r_cells, probit)
        else:
            l, g = basic._bern_ll_grad(eta, s, n, probit)
        return l, g

    def obj(z):
        l, g = cell_terms(z)
        a, gam = layout.split(z)
        wg = w * g
        g_a = -(B_cells * (b_j[midx] * wg)[:, None]).sum(axis=0)
        g_gam = layout.grad_gamma(wg, midx, J)
        lp, pa, pg = layout.log_prior(a, gam, slope)
        grad = np.concatenate([g_a + pa] + [x + y for x, y in zip(g_gam, pg)])
        return -(float(np.sum(w * l)) + lp), -grad

    z0 = np.zeros(layout.n)
    wlogl0 = float(np.sum(w * cell_terms(z0)[0]))
    res = _optimize(obj, layout, core, monotone)
    z = res.x
    l = cell_terms(z)[0]
    wlogl = float(np.sum(w * l))
    a, gam = layout.split(z)
    lp = layout.log_prior(a, gam, slope)[0]
    extra = {}
    if core.get('sqrtfamily'):
        # gain on the (rescaled) weighted objective the bumps maximized
        extra['weighted_gain'] = wlogl - wlogl0
    fit = _finish(core, layout, z, res, monotone, float(np.sum(l)),
                  float(np.sum(cell_terms(z0)[0])), lp, extra)
    if core.get('sqrtfamily'):
        basic_w = basic.sqrt_family_weights(idx)
        fit['wlogl'] = float(np.sum(basic_w * l))
    return fit


# ----------------------------------------------------------------------
# predictions
# ----------------------------------------------------------------------

_GH = roots_hermite(N_EVAL_NODES)


def marginal_p(fit, x, model_indices=None, fam_tbar=None):
    """Marginal success probabilities with the bumps: (len(x), n_sel)."""
    x = np.asarray(x, dtype=float)
    J = len(fit['model_names'])
    sel = np.arange(J) if model_indices is None else np.asarray(
        model_indices)
    off = ai_offsets(fit, x)[:, sel]                  # (len(x), n_sel)
    shift = general_shift(fit, x)
    probit = bool(fit['probit'])
    if fit['cls'] == 'basic':
        f = _core_curve(fit, x) + shift
        a, b = np.asarray(fit['a'])[sel], np.asarray(fit['b'])[sel]
        eta = a[None, :] - b[None, :] * f[:, None] + off
        return ndtr(eta) if probit else expit(eta)
    h, s = em.prior_curves(fit, x, fam_tbar=fam_tbar)
    h = h + shift
    alpha = fit['params'][sel, 0][None, :] + off      # (len(x), n_sel)
    beta = fit['params'][sel, 1][None, :]
    if probit:
        z = (alpha + beta * h[:, None]) / np.sqrt(
            1.0 + beta ** 2 * s[:, None] ** 2)
        return np.exp(log_ndtr(z))
    gr, gw = _GH
    gwn = gw / np.sqrt(np.pi)
    out = np.zeros((len(x), len(sel)))
    for k in range(len(gr)):
        th = h + np.sqrt(2.0) * s * gr[k]
        out += gwn[k] * expit(alpha + beta * th[:, None])
    return out


def time_horizons(fit, q, n_grid=481):
    """th_q per AI in minutes: the shortest length at which the marginal
    curve falls to q.  With AI-specific bumps a curve need not be
    monotone, so this is the first down-crossing; without them it equals
    the usual inversion.  NaN when the curve starts below q or never
    reaches it within the fit's range."""
    lo, hi = _range(fit)
    xg = np.linspace(lo, hi, n_grid)
    P = marginal_p(fit, xg)
    out = np.full(P.shape[1], np.nan)
    for j in range(P.shape[1]):
        p = P[:, j]
        if p[0] < q:
            continue
        below = np.flatnonzero(p <= q)
        if len(below) == 0:
            continue
        k = below[0]
        if k == 0:
            out[j] = 2.0 ** xg[0]
            continue
        frac = (p[k - 1] - q) / (p[k - 1] - p[k])
        out[j] = 2.0 ** (xg[k - 1] + frac * (xg[k] - xg[k - 1]))
    return out


def joint_ll(fit, idx, family_scheme=False, n_nodes=N_EVAL_NODES):
    """Prior-predictive joint log-lik of idx's cells for a bumped IRT
    fit, on em.joint_loglik_eval's fresh grids.  AI offsets enter through
    one (alpha, beta) row per cell."""
    gr, gw = roots_hermite(n_nodes)
    log_gw = np.log(gw / np.sqrt(np.pi))
    t = idx['t_array']
    midx = idx['cell_midx']
    cell_task = np.repeat(np.arange(idx['n_tasks']),
                          np.diff(idx['cell_off']))
    delta = ai_offsets(fit, t)[cell_task, midx]
    params_c = np.column_stack([fit['params'][midx, 0] + delta,
                                fit['params'][midx, 1]])
    cell_c = np.arange(len(midx), dtype=midx.dtype)
    phi = float(fit['phi']) if fit['bb'] else 1.0
    shift = general_shift(fit, t)
    if fit['family'] != 'none' and family_scheme:
        h, sig, tau_vec = _irt_prior_parts(fit, t, idx['fam_mean_t'])
        g_roots, g_w = roots_hermite(80)
        per_fam = KK.joint_ll_family(
            t, h + shift, sig, params_c, cell_c, idx['cell_s'],
            idx['cell_n'], idx['cell_off'], idx['fam_offsets'],
            idx['fam_task_idx'], tau_vec, gr, log_gw,
            g_roots, np.log(g_w / np.sqrt(np.pi)),
            fit['probit'], fit['bb'], phi, em.P_EPS)
        return float(per_fam.sum())
    h, s_tot = em.prior_curves(fit, t, fam_tbar=(
        idx['fam_mean_t'][idx['fam_of_task']]
        if fit['family'] == 'famspline' else None))
    per_task = KK.joint_ll_pertask(
        t, h + shift, s_tot, params_c, cell_c, idx['cell_s'],
        idx['cell_n'], idx['cell_off'], gr, log_gw,
        fit['probit'], fit['bb'], phi, em.P_EPS)
    return float(per_task.sum())
