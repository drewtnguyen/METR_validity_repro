"""Direct (marginal) fits for the headline grid basic models.

  linear     : per-model  link p = a_j + b_j x          (logit == METR)
  spline     : per-model  link p = c0_j - S_j(x), S monotone up
  shared     : link p = a_j - b_j f(x), ONE monotone f: 0 -> 1, b_j >= 0
  shared_bb  : shared mean + beta-binomial cells, per-model rho_j

All on the merged natural-cubic basis (global knots from all unique task
lengths).  Optimization is bounded L-BFGS-B with analytic gradients; the
shared shape uses block coordinate ascent (exact per-model step + softmax
simplex shape step) -- joint optimization is documented to under-converge
and break the linear-nesting property, so never fit it jointly.

Likelihoods are always the run-SEQUENCE convention (no log C(n,s)), so
Bernoulli and BB fits are directly comparable with the IRT joint logliks.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import minimize
from scipy.special import log_ndtr, digamma

from .basis import build_monotone_basis
from . import kernels as KK
from . import em

R_BOUND = 20.0
RIDGE = 1e-4
LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)


def cell_view(idx):
    """Cell arrays + per-cell x for the basic fits."""
    cell_task = np.repeat(np.arange(idx['n_tasks']),
                          np.diff(idx['cell_off']))
    return dict(task=cell_task, midx=idx['cell_midx'], s=idx['cell_s'],
                n=idx['cell_n'], x=idx['t_array'][cell_task])


def sqrt_family_weights(idx):
    """Per-run METR 'diversity' weight for each cell: 1/(sqrt(n_F)*n_ij).

    Same convention as eval_all.cell_weights('sqrt_family', ...): a cell's
    total mass is 1/sqrt(n_F), a family's total mass sqrt(n_F).  Family
    sizes n_F (#tasks in the cell's family) come from the data in `idx`,
    so CV folds recompute them from their own training fold -- the weights
    are part of the fitting recipe, not fixed constants.
    """
    fam_size = np.diff(idx['fam_offsets'])[idx['fam_of_task']]
    cell_task = np.repeat(np.arange(idx['n_tasks']),
                          np.diff(idx['cell_off']))
    return 1.0 / (np.sqrt(fam_size[cell_task].astype(float))
                  * idx['cell_n'])


# ----------------------------------------------------------------------
# link-generic log-lik pieces (vectorized numpy, sequence convention)
# ----------------------------------------------------------------------

def _bern_ll_grad(eta, s, n, probit):
    if probit:
        ll = s * log_ndtr(eta) + (n - s) * log_ndtr(-eta)
        logphi = -0.5 * eta ** 2 - LOG_SQRT_2PI
        m_pos = np.exp(logphi - log_ndtr(eta))
        m_neg = np.exp(logphi - log_ndtr(-eta))
        d = s * m_pos - (n - s) * m_neg
        return ll, d
    sp = np.logaddexp(0.0, eta)
    p = 1.0 / (1.0 + np.exp(-eta))
    return s * eta - n * sp, s - n * p


def _p_of_eta(eta, probit):
    if probit:
        return np.exp(log_ndtr(eta))
    return 1.0 / (1.0 + np.exp(-eta))


def _bb_ll_grads(eta, s, n, r, probit, p_eps=1e-6):
    """Sequence BB: returns (ll, dll/deta, dll/dr) with rho = sigmoid(r),
    phi = (1-rho)/rho  (so log phi = -r + log(1-rho)... we use the
    a,b = p*phi, (1-p)*phi parameterization directly)."""
    from scipy.special import gammaln
    rho = 1.0 / (1.0 + np.exp(-r))
    phi = (1.0 - rho) / rho
    p = np.clip(_p_of_eta(eta, probit), p_eps, 1.0 - p_eps)
    a = p * phi
    b = (1.0 - p) * phi
    ll = (gammaln(s + a) + gammaln(n - s + b) - gammaln(n + phi)
          - gammaln(a) - gammaln(b) + gammaln(phi))
    dsa = digamma(s + a) - digamma(a)
    dnb = digamma(n - s + b) - digamma(b)
    dl_dp = phi * (dsa - dnb)
    if probit:
        dp = np.exp(-0.5 * eta ** 2) / math.sqrt(2 * math.pi)
    else:
        dp = p * (1.0 - p)
    dl_deta = dl_dp * dp
    dl_dphi = (p * digamma(s + a) + (1 - p) * digamma(n - s + b)
               - digamma(n + phi) - p * digamma(a) - (1 - p) * digamma(b)
               + digamma(phi))
    dphi_dr = -phi * (1.0 - rho)   # d/dr[(1-rho)/rho], rho = sigmoid(r)
    return ll, dl_deta, dl_dphi * dphi_dr


# ----------------------------------------------------------------------
# per-model fits
# ----------------------------------------------------------------------

def fit_linear(idx, probit=False, shslope=False, sqrtfamily=False):
    """Per-model 2-param GLM on x (== the METR model for logit).

    shslope: share a single slope b across all models (per-model
    intercepts a_j only) -- the "does the slope really vary by model?"
    null nested inside the per-model METR fit.  The result is packaged
    in the SAME (n_models, 2) params shape with the shared slope
    broadcast into every row, so predict_p/time_horizon/eval all work
    unchanged.

    sqrtfamily: maximize the sqrt_family-WEIGHTED log-lik (the METR
    paper's diversity weighting; see sqrt_family_weights) instead of the
    unweighted one.  Same model, different estimator: the reported
    'logl' is the UNWEIGHTED log-lik realized at the weighted optimum
    (one scale with every other spec; NOT a maximized likelihood, so
    AIC on it is not meaningful), and 'wlogl' is the weighted objective."""
    if shslope:
        w = sqrt_family_weights(idx) if sqrtfamily else None
        return _fit_linear_shslope(idx, probit, w)
    if sqrtfamily:
        return _fit_linear_weighted(idx, probit, sqrt_family_weights(idx))
    params = em.init_alpha_beta(idx, probit)
    cv = cell_view(idx)
    ll = np.zeros(idx['n_models'])
    for j in range(idx['n_models']):
        sel = cv['midx'] == j
        eta = params[j, 0] + params[j, 1] * cv['x'][sel]
        ll[j] = float(_bern_ll_grad(eta, cv['s'][sel], cv['n'][sel],
                                    probit)[0].sum())
    return {'kind': 'linear', 'probit': probit, 'params': params,
            'll_by_model': ll, 'logl': float(ll.sum()),
            'n_params': 2 * idx['n_models']}


def _fit_linear_weighted(idx, probit, w_cells):
    """Weighted per-model 2-param GLM: maximize sum_cells w * ll_cell.

    Concave in (a_j, b_j) for both links (weights >= 0), so a single
    L-BFGS-B solve per model with analytic gradients converges; warm
    start at the unweighted MLE."""
    cv = cell_view(idx)
    J = idx['n_models']
    init = fit_linear(idx, probit)['params']
    params = np.empty((J, 2))
    ll = np.zeros(J)       # unweighted log-lik at the weighted optimum
    wll = np.zeros(J)      # the weighted objective actually maximized
    for j in range(J):
        sel = cv['midx'] == j
        x, s, n = cv['x'][sel], cv['s'][sel], cv['n'][sel]
        w = w_cells[sel]

        def nll(th):
            eta = th[0] + th[1] * x
            l, g = _bern_ll_grad(eta, s, n, probit)
            wg = w * g
            return -float(np.sum(w * l)), -np.array([wg.sum(),
                                                     np.sum(wg * x)])

        res = minimize(nll, init[j], jac=True, method='L-BFGS-B',
                       options=dict(maxiter=10000, ftol=1e-14, gtol=1e-10,
                                    maxfun=100000))
        params[j] = res.x
        eta = res.x[0] + res.x[1] * x
        l = _bern_ll_grad(eta, s, n, probit)[0]
        ll[j] = float(l.sum())
        wll[j] = float(np.sum(w * l))
    return {'kind': 'linear', 'probit': probit, 'sqrtfamily': True,
            'params': params, 'll_by_model': ll, 'logl': float(ll.sum()),
            'wll_by_model': wll, 'wlogl': float(wll.sum()),
            'n_params': 2 * J}


def _fit_linear_shslope(idx, probit=False, w_cells=None):
    """Shared-slope linear fit: link p = a_j + b*x, one global slope b.

    Joint convex GLM over [a_0..a_{J-1}, b] (J+1 params).  Ordinary
    logistic/probit regression with a single continuous covariate and
    per-model intercept dummies -- concave, so a single L-BFGS-B solve
    with analytic gradients converges (no shared-shape simplex here, so
    the fit_shared joint-optimization caveat does not apply).

    w_cells: optional per-run cell weights; the objective becomes the
    weighted log-lik (sqrtfamily estimator), reported per fit_linear's
    logl/wlogl convention."""
    cv = cell_view(idx)
    J = idx['n_models']
    midx = cv['midx']
    x = cv['x']
    s = cv['s']
    n = cv['n']
    weighted = w_cells is not None
    w = np.ones(len(x)) if w_cells is None else w_cells

    # warm start from the matching per-model linear fit: intercepts
    # as-is, slope at the mean of the per-model slopes.
    lin = fit_linear(idx, probit, sqrtfamily=weighted)
    a0 = lin['params'][:, 0].copy()
    b0 = float(lin['params'][:, 1].mean())
    z0 = np.concatenate([a0, [b0]])

    def nll(z):
        a = z[:J]
        b = z[J]
        eta = a[midx] + b * x
        l, g = _bern_ll_grad(eta, s, n, probit)
        wg = w * g
        # grad of -ll: d(-ll)/da_j = -sum_{cells of j} wg ; d(-ll)/db = -sum wg*x
        grad = np.empty(J + 1)
        grad[:J] = -np.bincount(midx, weights=wg, minlength=J)
        grad[J] = -float(np.sum(wg * x))
        return -float(np.sum(w * l)), grad

    res = minimize(nll, z0, jac=True, method='L-BFGS-B',
                   options=dict(maxiter=10000, ftol=1e-14, gtol=1e-10,
                                maxfun=100000))
    a = res.x[:J]
    b = float(res.x[J])
    params = np.empty((J, 2))
    params[:, 0] = a
    params[:, 1] = b                      # shared slope broadcast per model
    ll = np.zeros(J)
    wll = np.zeros(J)
    for j in range(J):
        sel = midx == j
        eta = a[j] + b * x[sel]
        lj = _bern_ll_grad(eta, s[sel], n[sel], probit)[0]
        ll[j] = float(lj.sum())
        wll[j] = float(np.sum(w[sel] * lj))
    out = {'kind': 'linear', 'probit': probit, 'shslope': True,
           'params': params, 'll_by_model': ll, 'logl': float(ll.sum()),
           'n_params': J + 1}
    if weighted:
        out['sqrtfamily'] = True
        out['wll_by_model'] = wll
        out['wlogl'] = float(wll.sum())
    return out


def fit_pmspline(idx, probit=False, ridge=RIDGE, kint=em.MEAN_KINT):
    """Per-model monotone-decreasing natural cubic:
    link p = c0 - S(x), coefficients (c0 free, rest <= 0)."""
    nb = build_monotone_basis(idx['t_array'], kint)
    cv = cell_view(idx)
    X_all = nb.design(cv['x'])
    d = X_all.shape[1]
    J = idx['n_models']
    coefs = np.zeros((J, d))
    ll = np.zeros(J)
    lin = fit_linear(idx, probit)
    for j in range(J):
        sel = cv['midx'] == j
        X = X_all[sel]
        s = cv['s'][sel]
        n = cv['n'][sel]

        def nll(b):
            eta = X @ b
            l, g = _bern_ll_grad(eta, s, n, probit)
            pen = ridge * np.sum(b[1:] ** 2)
            grad = -(X.T @ g)
            grad[1:] += 2 * ridge * b[1:]
            return -float(l.sum()) + pen, grad

        # init at the model's own linear fit (exactly representable:
        # c0 + w*(x-lo) with equal raw weights; slope<=0 => weights<=0)
        a_l, b_l = lin['params'][j]
        b0 = np.zeros(d)
        b0[0] = a_l + b_l * nb.lo
        b0[1:] = b_l          # equal raw weights == linear slope
        if b_l > 0:           # increasing-in-x model: cone boundary
            b0[1:] = 0.0
            b0[0] = a_l + b_l * np.mean(cv['x'][sel])
        ub = np.full(d, np.inf)
        ub[1:] = 0.0
        res = minimize(nll, b0, jac=True, method='L-BFGS-B',
                       bounds=list(zip(np.full(d, -np.inf), ub)),
                       options=dict(maxiter=10000, ftol=1e-14,
                                    gtol=1e-10, maxfun=100000))
        coefs[j] = res.x
        eta = X @ res.x
        ll[j] = float(_bern_ll_grad(eta, s, n, probit)[0].sum())
    return {'kind': 'spline', 'probit': probit, 'coefs': coefs,
            'basis': nb, 'll_by_model': ll, 'logl': float(ll.sum()),
            'n_params': d * J}


# ----------------------------------------------------------------------
# shared shape (block coordinate ascent)
# ----------------------------------------------------------------------

def _softmax(u):
    e = np.exp(u - u.max())
    return e / e.sum()


def fit_shared(idx, probit=False, bb=False, shslope=False, sqrtfamily=False,
               max_outer=200, tol=1e-7, verbose=False, kint=em.MEAN_KINT,
               fixed_w=None):
    """Shared-shape marginal fit: link p = a_j - b_j f(x), one monotone
    f: 0->1 across models.  shslope shares a single slope b across all
    models (per-model a_j[, rho_j] only), stored broadcast into b[:] so
    predict_p/time_horizon work unchanged.

    sqrtfamily: maximize the sqrt_family-WEIGHTED log-lik (see
    fit_linear's docstring for the estimator/reporting convention:
    'logl' = unweighted log-lik at the weighted optimum, 'wlogl' = the
    weighted objective the coordinate ascent actually climbs).

    kint: interior knots of the shared shape f (kint + 2 df): a count,
    or explicit positions in log2 minutes (basis.interior_knots).

    fixed_w: freeze the shape at these simplex weights over the
    normalized basis columns (f = sum_m fixed_w_m M_m / M_m(hi)) and fit
    only the AI parameters.  The shape then comes from elsewhere (the
    JML route, horizon_irt.jml); its df still count in n_params, since
    it was estimated from the same data."""
    nb = build_monotone_basis(idx['t_array'], kint)
    cv = cell_view(idx)
    M_raw = nb.raw_merged(cv['x'])
    col_hi = nb.raw_merged(np.array([nb.hi]))[0]
    Mn = M_raw / col_hi                      # each column 0 -> 1
    J = idx['n_models']
    m = Mn.shape[1]
    w_all = sqrt_family_weights(idx) if sqrtfamily else np.ones(len(cv['x']))
    if fixed_w is not None:
        fixed_w = np.asarray(fixed_w, dtype=float)
        if fixed_w.shape != (m,) or np.any(fixed_w < 0) or \
                abs(fixed_w.sum() - 1.0) > 1e-8:
            raise ValueError(f'fixed_w must be {m} simplex weights')

    # init: f = linear shape (weights prop. to column ends == simplex
    # point that reproduces (x-lo)/(hi-lo)), (a,b) from the matching
    # (weighted or unweighted) linear fit.
    u = np.log(col_hi / col_hi.sum())
    lin = fit_linear(idx, probit, sqrtfamily=sqrtfamily)
    a = lin['params'][:, 0] + lin['params'][:, 1] * nb.lo
    b = np.maximum(-lin['params'][:, 1] * (nb.hi - nb.lo), 1e-6)
    if shslope:
        b[:] = float(b.mean())               # one shared slope, broadcast
    r = np.full(J, 2.0) if bb else None

    sel_by_model = [cv['midx'] == j for j in range(J)]
    w_by_model = [w_all[sel] for sel in sel_by_model]

    def cell_ll_grad(eta, s, n, rj):
        if bb:
            return _bb_ll_grads(eta, s, n, rj, probit)
        l, g = _bern_ll_grad(eta, s, n, probit)
        return l, g, None

    def total_nll(f_cells):
        """Negative WEIGHTED objective (w == 1 when not sqrtfamily)."""
        tot = 0.0
        for j in range(J):
            sel = sel_by_model[j]
            eta = a[j] - b[j] * f_cells[sel]
            l = cell_ll_grad(eta, cv['s'][sel], cv['n'][sel],
                             r[j] if bb else None)[0]
            tot -= float(np.sum(w_by_model[j] * l))
        return tot

    def model_step(f_cells):
        """Per-model exact step.  With shslope, b is held fixed here (a_j
        [and r_j] only); the shared b is updated by shared_slope_step."""
        for j in range(J):
            sel = sel_by_model[j]
            fj = f_cells[sel]
            s = cv['s'][sel]
            n = cv['n'][sel]
            wj = w_by_model[j]

            if shslope and bb:
                bj = b[j]
                def nll(th):                 # th = [a_j, r_j], b fixed
                    eta = th[0] - bj * fj
                    l, g_eta, g_r = _bb_ll_grads(eta, s, n, th[1], probit)
                    return -float(np.sum(wj * l)), -np.array([
                        np.sum(wj * g_eta), np.sum(wj * g_r)])
                x0 = np.array([a[j], r[j]])
                bounds = [(-np.inf, np.inf), (-R_BOUND, R_BOUND)]
                res = minimize(nll, x0, jac=True, method='L-BFGS-B',
                               bounds=bounds,
                               options=dict(maxiter=2000, ftol=1e-13,
                                            gtol=1e-10))
                a[j], r[j] = res.x[0], res.x[1]
            elif shslope:
                bj = b[j]
                def nll(th):                 # th = [a_j], b fixed
                    eta = th[0] - bj * fj
                    l, g_eta = _bern_ll_grad(eta, s, n, probit)
                    return -float(np.sum(wj * l)), -np.array([
                        np.sum(wj * g_eta)])
                res = minimize(nll, np.array([a[j]]), jac=True,
                               method='L-BFGS-B',
                               options=dict(maxiter=2000, ftol=1e-13,
                                            gtol=1e-10))
                a[j] = res.x[0]
            elif bb:
                def nll(th):
                    eta = th[0] - th[1] * fj
                    l, g_eta, g_r = _bb_ll_grads(eta, s, n, th[2], probit)
                    return -float(np.sum(wj * l)), -np.array([
                        np.sum(wj * g_eta), -np.sum(wj * g_eta * fj),
                        np.sum(wj * g_r)])
                x0 = np.array([a[j], b[j], r[j]])
                bounds = [(-np.inf, np.inf), (0.0, np.inf),
                          (-R_BOUND, R_BOUND)]
                res = minimize(nll, x0, jac=True, method='L-BFGS-B',
                               bounds=bounds,
                               options=dict(maxiter=2000, ftol=1e-13,
                                            gtol=1e-10))
                a[j], b[j], r[j] = res.x[0], res.x[1], res.x[2]
            else:
                def nll(th):
                    eta = th[0] - th[1] * fj
                    l, g_eta = _bern_ll_grad(eta, s, n, probit)
                    return -float(np.sum(wj * l)), -np.array([
                        np.sum(wj * g_eta), -np.sum(wj * g_eta * fj)])
                x0 = np.array([a[j], b[j]])
                bounds = [(-np.inf, np.inf), (0.0, np.inf)]
                res = minimize(nll, x0, jac=True, method='L-BFGS-B',
                               bounds=bounds,
                               options=dict(maxiter=2000, ftol=1e-13,
                                            gtol=1e-10))
                a[j], b[j] = res.x[0], res.x[1]

    def shared_slope_step(f_cells):
        """Global 1-D step for the shared slope b (a_j, r_j fixed); pools
        the objective/gradient over ALL models.  b >= 0, concave in b."""
        def nll(bv):
            bb_ = bv[0]
            tot = 0.0
            g = 0.0
            for j in range(J):
                sel = sel_by_model[j]
                fj = f_cells[sel]
                eta = a[j] - bb_ * fj
                out = cell_ll_grad(eta, cv['s'][sel], cv['n'][sel],
                                   r[j] if bb else None)
                wj = w_by_model[j]
                tot -= float(np.sum(wj * out[0]))
                # d(-ll)/db = +sum w*g_eta*f
                g += float(np.sum(wj * out[1] * fj))
            return tot, np.array([g])
        res = minimize(nll, np.array([b[0]]), jac=True, method='L-BFGS-B',
                       bounds=[(0.0, np.inf)],
                       options=dict(maxiter=2000, ftol=1e-13, gtol=1e-10))
        b[:] = res.x[0]

    def shape_nll_grad(u_):
        sw = _softmax(u_)
        f_cells = Mn @ sw
        nll = 0.0
        g_f = np.empty(len(f_cells))
        for j in range(J):
            sel = sel_by_model[j]
            eta = a[j] - b[j] * f_cells[sel]
            out = cell_ll_grad(eta, cv['s'][sel], cv['n'][sel],
                               r[j] if bb else None)
            wj = w_by_model[j]
            nll -= float(np.sum(wj * out[0]))
            g_f[sel] = b[j] * wj * out[1]   # d(-ll)/df = +b * w * dll/deta
        g_w = Mn.T @ g_f
        g_u = sw * (g_w - np.dot(sw, g_w))
        return nll, g_u

    def joint_ai_step(f_cells):
        """All AI parameters (a, b or the shared b, r) in one bounded
        L-BFGS-B solve at a FIXED shape.  Concave in (a, b) for both
        links, so this lands on the exact optimum that alternating a_j
        and shared-b steps only approach slowly.  Used for fixed_w only:
        with a free shape the blocks must stay separate (see the module
        docstring)."""
        nb_ = 1 if shslope else J
        midx = cv['midx']
        s, n = cv['s'], cv['n']

        def unpack(th):
            b_ = np.full(J, th[J]) if shslope else th[J:J + nb_]
            return th[:J], b_, (th[J + nb_:] if bb else None)

        def nll(th):
            a_, b_, r_ = unpack(th)
            eta = a_[midx] - b_[midx] * f_cells
            if bb:
                l, g_eta, g_r = _bb_ll_grads(eta, s, n, r_[midx], probit)
            else:
                l, g_eta = _bern_ll_grad(eta, s, n, probit)
            wg = w_all * g_eta
            ga = np.bincount(midx, wg, minlength=J)
            gb = np.bincount(midx, -wg * f_cells, minlength=J)
            parts = [ga, np.array([gb.sum()]) if shslope else gb]
            if bb:
                parts.append(np.bincount(midx, w_all * g_r, minlength=J))
            return -float(np.sum(w_all * l)), -np.concatenate(parts)

        th0 = np.concatenate([a, b[:1] if shslope else b]
                             + ([r] if bb else []))
        bounds = ([(None, None)] * J + [(0.0, None)] * nb_
                  + ([(-R_BOUND, R_BOUND)] * J if bb else []))
        res = minimize(nll, th0, jac=True, method='L-BFGS-B', bounds=bounds,
                       options=dict(maxiter=20000, ftol=1e-15, gtol=1e-10,
                                    maxfun=200000))
        a_, b_, r_ = unpack(res.x)
        a[:] = a_
        b[:] = b_
        if bb:
            r[:] = r_

    if fixed_w is not None:
        u = np.log(np.maximum(fixed_w, 1e-300))
        f_fixed = Mn @ fixed_w
        model_step(f_fixed)                  # warm start, then exact
        joint_ai_step(f_fixed)
        max_outer = 0                        # skip the shape loop
    prev = np.inf
    for it in range(max_outer):
        f_cells = Mn @ _softmax(u)
        model_step(f_cells)
        if shslope:
            shared_slope_step(f_cells)      # global b after per-model a_j
        res = minimize(shape_nll_grad, u, jac=True, method='L-BFGS-B',
                       options=dict(maxiter=100, ftol=1e-12, gtol=1e-9))
        u = res.x
        cur = total_nll(Mn @ _softmax(u))
        if verbose and it % 10 == 0:
            print(f'    outer {it}: nll={cur:.4f}')
        if abs(prev - cur) < tol * max(1.0, abs(cur)):
            break
        prev = cur
    if fixed_w is None:
        f_final = Mn @ _softmax(u)
        model_step(f_final)                  # final exact (a[,r]) step
        if shslope:
            shared_slope_step(f_final)       # and shared b

    w = _softmax(u) if fixed_w is None else fixed_w
    f_fin = Mn @ w
    # UNWEIGHTED log-lik at the optimum (fit_linear's reporting
    # convention; equals the maximized objective when not sqrtfamily)
    logl = 0.0
    for j in range(J):
        sel = sel_by_model[j]
        eta = a[j] - b[j] * f_fin[sel]
        l = cell_ll_grad(eta, cv['s'][sel], cv['n'][sel],
                         r[j] if bb else None)[0]
        logl += float(np.sum(l))
    # shslope: one shared slope replaces J per-model slopes (J-1 fewer).
    n_params = (m - 1) + (3 if bb else 2) * J - ((J - 1) if shslope else 0)
    out = {'kind': 'shared_bb' if bb else 'shared', 'probit': probit,
           'bb': bb, 'shslope': shslope, 'basis': nb, 'u': u, 'w': w,
           'col_hi': col_hi,
           'a': a, 'b': b, 'r': (r.copy() if bb else None),
           'rho': (1 / (1 + np.exp(-r)) if bb else None),
           'logl': float(logl), 'n_params': n_params}
    if sqrtfamily:
        out['sqrtfamily'] = True
        out['wlogl'] = -total_nll(f_fin)
    return out


# ----------------------------------------------------------------------
# predictions & horizons for basic fits
# ----------------------------------------------------------------------

def predict_p(fit, x, model_indices=None):
    """Marginal p_j(x): (len(x), n_sel). (BB per-run marginal == p.)"""
    x = np.asarray(x, dtype=float)
    J = (len(fit['params']) if fit['kind'] == 'linear'
         else (len(fit['coefs']) if fit['kind'] == 'spline'
               else len(fit['a'])))
    sel = np.arange(J) if model_indices is None else np.asarray(model_indices)
    out = np.empty((len(x), len(sel)))
    if fit['kind'] == 'linear':
        for c, j in enumerate(sel):
            eta = fit['params'][j, 0] + fit['params'][j, 1] * x
            out[:, c] = _p_of_eta(eta, fit['probit'])
        return out
    nb = fit['basis']
    if fit['kind'] == 'spline':
        X = nb.design(x)
        for c, j in enumerate(sel):
            out[:, c] = _p_of_eta(X @ fit['coefs'][j], fit['probit'])
        return out
    f = (nb.raw_merged(x) / fit['col_hi']) @ fit['w']
    for c, j in enumerate(sel):
        out[:, c] = _p_of_eta(fit['a'][j] - fit['b'][j] * f, fit['probit'])
    return out


def time_horizon(fit, q, n_grid=481):
    """th_q in minutes per model by inverting the fitted curve."""
    lo, hi = ((fit['basis'].lo, fit['basis'].hi)
              if fit['kind'] != 'linear' else (None, None))
    if fit['kind'] == 'linear':
        from scipy.special import ndtri
        inv = ndtri(q) if fit['probit'] else math.log(q / (1 - q))
        a_, b_ = fit['params'][:, 0], fit['params'][:, 1]
        with np.errstate(divide='ignore', invalid='ignore'):
            xq = (inv - a_) / b_
        return 2.0 ** xq
    xg = np.linspace(lo, hi, n_grid)
    P = predict_p(fit, xg)
    out = np.full(P.shape[1], np.nan)
    for j in range(P.shape[1]):
        ps = P[::-1, j]
        xs = xg[::-1]
        if q < ps[0] or q > ps[-1]:
            continue
        out[j] = 2.0 ** float(np.interp(q, ps, xs))
    return out
