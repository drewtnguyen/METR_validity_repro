"""Numba kernels for the unified C=1 IRT EM (headline-model-grid skill).

One kernel family covers the full C=1 crossing:
  link      : logit / probit           (probit flag)
  sampling  : Bernoulli / beta-binomial (use_bb flag, global phi)
  family    : none / scalar tau / tau(t_fam)  (separate family kernel,
              per-family tau_vec with the ll_tk table hoisted ONCE -- the
              famspline speed fix pattern)

Data are stored at the (task, model) CELL level with (s, n) counts; the
cell log-likelihood is always the run-SEQUENCE convention:
  bern: s log p + (n-s) log(1-p)
  bb  : logB(s + p*phi, n-s + (1-p)*phi) - logB(p*phi, (1-p)*phi)
which the BB reproduces exactly as phi -> inf, so Bernoulli and BB joint
log-liks are on one scale (skill: evaluation comparability convention).
"""
import math

import numpy as np
from numba import njit

SQRT2 = math.sqrt(2.0)
LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)


# ----------------------------------------------------------------------
# scalar special functions (numba-safe)
# ----------------------------------------------------------------------

@njit(cache=True, inline='always')
def _log_ndtr(z):
    """log Phi(z), stable for very negative z."""
    if z > -20.0:
        return math.log(0.5 * math.erfc(-z / SQRT2))
    z2 = z * z
    return (-0.5 * z2 - LOG_SQRT_2PI - math.log(-z)
            + math.log1p(-1.0 / z2 + 3.0 / (z2 * z2)))


@njit(cache=True, inline='always')
def _norm_pdf(z):
    return math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


@njit(cache=True, inline='always')
def _digamma(x):
    """psi(x) for x > 0: shift to x >= 6, then asymptotic series."""
    acc = 0.0
    while x < 6.0:
        acc -= 1.0 / x
        x += 1.0
    inv = 1.0 / x
    inv2 = inv * inv
    return (acc + math.log(x) - 0.5 * inv
            - inv2 * (1.0 / 12.0 - inv2 * (1.0 / 120.0 - inv2 / 252.0)))


@njit(cache=True, inline='always')
def _trigamma(x):
    """psi'(x) for x > 0: shift to x >= 6, then asymptotic series."""
    acc = 0.0
    while x < 6.0:
        acc += 1.0 / (x * x)
        x += 1.0
    inv = 1.0 / x
    inv2 = inv * inv
    return (acc + inv * (1.0 + 0.5 * inv
            + inv2 * (1.0 / 6.0 - inv2 * (1.0 / 30.0 - inv2 / 42.0))))


@njit(cache=True, inline='always')
def _lbeta(a, b):
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


@njit(cache=True, inline='always')
def _inv_link(eta, probit):
    if probit:
        if eta >= 0.0:
            return 1.0 - 0.5 * math.erfc(eta / SQRT2)
        return 0.5 * math.erfc(-eta / SQRT2)
    if eta >= 0.0:
        return 1.0 / (1.0 + math.exp(-eta))
    e = math.exp(eta)
    return e / (1.0 + e)


@njit(cache=True, inline='always')
def _cell_ll(s, n, eta, probit, use_bb, phi, p_eps):
    """Sequence-convention cell log-likelihood at linear predictor eta."""
    if use_bb:
        p = _inv_link(eta, probit)
        if p < p_eps:
            p = p_eps
        elif p > 1.0 - p_eps:
            p = 1.0 - p_eps
        a = p * phi
        b = (1.0 - p) * phi
        return _lbeta(s + a, n - s + b) - _lbeta(a, b)
    if probit:
        return s * _log_ndtr(eta) + (n - s) * _log_ndtr(-eta)
    # logit Bernoulli: s*eta - n*softplus(eta), stable both tails
    if eta >= 0.0:
        sp = eta + math.log1p(math.exp(-eta))
    else:
        sp = math.log1p(math.exp(eta))
    return s * eta - n * sp


@njit(cache=True, inline='always')
def _cell_dll_deta(s, n, eta, probit, use_bb, phi, p_eps):
    """(d/deta, d2/deta2) of the cell log-likelihood."""
    p = _inv_link(eta, probit)
    if p < p_eps:
        p = p_eps
    elif p > 1.0 - p_eps:
        p = 1.0 - p_eps
    if probit:
        pdf = _norm_pdf(eta)
        dp = pdf
        d2p = -eta * pdf
    else:
        dp = p * (1.0 - p)
        d2p = dp * (1.0 - 2.0 * p)
    if use_bb:
        a = p * phi
        b = (1.0 - p) * phi
        dsa = _digamma(s + a) - _digamma(a)
        dnb = _digamma(n - s + b) - _digamma(b)
        dl_dp = phi * (dsa - dnb)
        d2l_dp2 = phi * phi * (_trigamma(s + a) + _trigamma(n - s + b)
                               - _trigamma(a) - _trigamma(b))
    else:
        dl_dp = s / p - (n - s) / (1.0 - p)
        d2l_dp2 = -s / (p * p) - (n - s) / ((1.0 - p) * (1.0 - p))
    g = dl_dp * dp
    h = d2l_dp2 * dp * dp + dl_dp * d2p
    return g, h


@njit(cache=True, inline='always')
def _cell_dll_dphi(s, n, eta, probit, phi, p_eps):
    """(d/dphi, d2/dphi2) of the BB sequence cell log-likelihood."""
    p = _inv_link(eta, probit)
    if p < p_eps:
        p = p_eps
    elif p > 1.0 - p_eps:
        p = 1.0 - p_eps
    a = p * phi
    b = (1.0 - p) * phi
    q = 1.0 - p
    d1 = (p * _digamma(s + a) + q * _digamma(n - s + b)
          - _digamma(n + phi) - p * _digamma(a) - q * _digamma(b)
          + _digamma(phi))
    d2 = (p * p * _trigamma(s + a) + q * q * _trigamma(n - s + b)
          - _trigamma(n + phi) - p * p * _trigamma(a) - q * q * _trigamma(b)
          + _trigamma(phi))
    return d1, d2


# ----------------------------------------------------------------------
# hoisted per-(task, node) data log-likelihood table
# ----------------------------------------------------------------------

@njit(cache=True)
def _ll_table(theta_grid, params, cell_midx, cell_s, cell_n, cell_off,
              probit, use_bb, phi, p_eps):
    n_tasks, K = theta_grid.shape
    ll_tk = np.empty((n_tasks, K))
    for i in range(n_tasks):
        c0 = cell_off[i]
        c1 = cell_off[i + 1]
        for k in range(K):
            th = theta_grid[i, k]
            acc = 0.0
            for c in range(c0, c1):
                m = cell_midx[c]
                eta = params[m, 0] + params[m, 1] * th
                acc += _cell_ll(cell_s[c], cell_n[c], eta,
                                probit, use_bb, phi, p_eps)
            ll_tk[i, k] = acc
    return ll_tk


# ----------------------------------------------------------------------
# E-steps
# ----------------------------------------------------------------------

@njit(cache=True)
def estep_c1(theta_grid, log_qw, t_array, means, sigmas, sigma0,
             params, cell_midx, cell_s, cell_n, cell_off,
             probit, use_bb, phi, p_eps):
    """No-family E-step. Returns (resp, total_ll, S1, S2)."""
    n_tasks, K = theta_grid.shape
    ll_tk = _ll_table(theta_grid, params, cell_midx, cell_s, cell_n,
                      cell_off, probit, use_bb, phi, p_eps)
    resp = np.empty((n_tasks, K))
    S1 = np.empty(n_tasks)
    S2 = np.empty(n_tasks)
    total_ll = 0.0
    log_s0 = math.log(sigma0)
    half_inv_s02 = 0.5 / (sigma0 * sigma0)
    for i in range(n_tasks):
        s_i = sigmas[i]
        log_s = math.log(s_i)
        half_inv = 0.5 / (s_i * s_i)
        mu = means[i]
        mx = -np.inf
        for k in range(K):
            th = theta_grid[i, k]
            d0 = th - t_array[i]
            z = th - mu
            v = (log_qw[k] + ll_tk[i, k] + log_s0 + d0 * d0 * half_inv_s02
                 - log_s - z * z * half_inv)
            resp[i, k] = v
            if v > mx:
                mx = v
        acc = 0.0
        for k in range(K):
            acc += math.exp(resp[i, k] - mx)
        logden = mx + math.log(acc)
        total_ll += logden
        s1 = 0.0
        s2 = 0.0
        for k in range(K):
            r = math.exp(resp[i, k] - logden)
            resp[i, k] = r
            th = theta_grid[i, k]
            s1 += r * th
            s2 += r * th * th
        S1[i] = s1
        S2[i] = s2
    return resp, total_ll, S1, S2


@njit(cache=True)
def estep_c1_family(theta_grid, log_qw, t_array, means, sigmas, sigma0,
                    params, cell_midx, cell_s, cell_n, cell_off,
                    fam_offsets, fam_task_idx, tau_vec, g_roots, log_gw,
                    probit, use_bb, phi, p_eps, tau0=0.0):
    """Family E-step: theta_i = f(t_i) + g_fam + eps_i.

    Per-family tau (tau_vec) with the ll_tk table hoisted ONCE -- the
    famspline speed-fix pattern.  Returns
    (resp, total_ll, S1, S2, Sthg, Eg, Eg2) with Sthg_i = E[theta_i * g].

    xi quadrature.  tau0 > 0 puts the xi atoms at a FIXED base scale and
    lets the fitted tau enter by importance ratio -- the same convention
    the theta integral below already uses, and the one the skills
    require ("do not update or recenter the quadrature grid inside the
    EM iterations").  This matters because with tau-SCALED atoms the
    discretized prior over atoms is just log_gw, independent of tau, so
    the M-step tau^2 <- E[xi^2] (derived for the continuous model) is
    NOT the maximizer of the discretized surrogate, and EM ascent is
    not guaranteed.  With fixed atoms the ratio carries tau and that
    M-step is exact again.

    tau0 <= 0 keeps the legacy tau-scaled atoms, bit-for-bit (the two
    added terms are then exactly 0.0).  Callers must floor tau at the
    node spacing when tau0 > 0 -- see em.fit_irt; a prior narrower than
    the grid can resolve stops normalizing and harvests spurious logL,
    exactly the SIGMA_FLOOR story on the theta side.
    """
    n_tasks, K = theta_grid.shape
    Kg = len(g_roots)
    n_fams = len(fam_offsets) - 1
    ll_tk = _ll_table(theta_grid, params, cell_midx, cell_s, cell_n,
                      cell_off, probit, use_bb, phi, p_eps)
    resp = np.zeros((n_tasks, K))
    S1 = np.zeros(n_tasks)
    S2 = np.zeros(n_tasks)
    Sthg = np.zeros(n_tasks)
    Eg = np.zeros(n_fams)
    Eg2 = np.zeros(n_fams)
    total_ll = 0.0
    log_s0 = math.log(sigma0)
    half_inv_s02 = 0.5 / (sigma0 * sigma0)
    # guarded: numba's python error_model raises on 1/0 at tau0 = 0
    fixed_xi = tau0 > 0.0
    log_t0 = 0.0
    inv_t02 = 0.0
    if fixed_xi:
        log_t0 = math.log(tau0)
        inv_t02 = 1.0 / (tau0 * tau0)

    for fi in range(n_fams):
        f0 = fam_offsets[fi]
        f1 = fam_offsets[fi + 1]
        n_in = f1 - f0
        tau_f = tau_vec[fi]
        if fixed_xi:
            g_scale = SQRT2 * tau0
            log_rat0 = log_t0 - math.log(tau_f)
            half_dinv = 0.5 * (1.0 / (tau_f * tau_f) - inv_t02)
        else:
            g_scale = SQRT2 * tau_f
            log_rat0 = 0.0
            half_dinv = 0.0

        log_pf = np.empty(Kg)
        lw = np.empty((Kg, n_in, K))
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
                    d0 = th - t_array[i]
                    z = th - mu
                    v = (log_qw[k] + ll_tk[i, k] + log_s0
                         + d0 * d0 * half_inv_s02
                         - log_s - z * z * half_inv)
                    lw[a, jj, k] = v
                    if v > mx:
                        mx = v
                acc = 0.0
                for k in range(K):
                    acc += math.exp(lw[a, jj, k] - mx)
                tot += mx + math.log(acc)
            log_pf[a] = log_gw[a] + log_rat0 - g * g * half_dinv + tot

        mxg = -np.inf
        for a in range(Kg):
            if log_pf[a] > mxg:
                mxg = log_pf[a]
        accg = 0.0
        for a in range(Kg):
            accg += math.exp(log_pf[a] - mxg)
        log_den = mxg + math.log(accg)
        total_ll += log_den

        for a in range(Kg):
            wg = math.exp(log_pf[a] - log_den)
            if wg < 1e-300:
                continue
            g_a = g_scale * g_roots[a]
            Eg[fi] += wg * g_a
            Eg2[fi] += wg * g_a * g_a
            for jj in range(n_in):
                i = fam_task_idx[f0 + jj]
                mx = -np.inf
                for k in range(K):
                    if lw[a, jj, k] > mx:
                        mx = lw[a, jj, k]
                acc = 0.0
                for k in range(K):
                    acc += math.exp(lw[a, jj, k] - mx)
                for k in range(K):
                    w_ik = wg * math.exp(lw[a, jj, k] - mx) / acc
                    resp[i, k] += w_ik
                    th = theta_grid[i, k]
                    S1[i] += w_ik * th
                    S2[i] += w_ik * th * th
                    Sthg[i] += w_ik * th * g_a
    return resp, total_ll, S1, S2, Sthg, Eg, Eg2


# ----------------------------------------------------------------------
# M-steps
# ----------------------------------------------------------------------

@njit(cache=True)
def mstep_alpha_beta(params, theta_grid, resp,
                     mc_task, mc_s, mc_n, mc_off,
                     probit, use_bb, phi, p_eps,
                     max_steps=40, grad_tol=1e-9, step_cap=5.0):
    """Per-model damped Newton on (alpha, beta) over the quadrature grid.

    mc_* arrays list each model's cells: mc_task[c] = task index,
    mc_s/mc_n the counts, sliced by mc_off (n_models+1,)."""
    n_models = params.shape[0]
    K = theta_grid.shape[1]
    for m in range(n_models):
        a0 = params[m, 0]
        b0 = params[m, 1]
        F0 = _model_obj(a0, b0, m, theta_grid, resp, mc_task, mc_s, mc_n,
                        mc_off, probit, use_bb, phi, p_eps)
        for _ in range(max_steps):
            g0 = 0.0
            g1 = 0.0
            h00 = 0.0
            h01 = 0.0
            h11 = 0.0
            for c in range(mc_off[m], mc_off[m + 1]):
                i = mc_task[c]
                s = mc_s[c]
                n = mc_n[c]
                for k in range(K):
                    r = resp[i, k]
                    if r < 1e-14:
                        continue
                    th = theta_grid[i, k]
                    eta = a0 + b0 * th
                    g, h = _cell_dll_deta(s, n, eta, probit, use_bb,
                                          phi, p_eps)
                    g0 += r * g
                    g1 += r * g * th
                    h00 += r * h
                    h01 += r * h * th
                    h11 += r * h * th * th
            if abs(g0) < grad_tol and abs(g1) < grad_tol:
                break
            det = h00 * h11 - h01 * h01
            if abs(det) > 1e-12:
                da = -(h11 * g0 - h01 * g1) / det
                db = -(-h01 * g0 + h00 * g1) / det
            else:
                # near-singular Hessian: gradient-ascent fallback
                gn = math.sqrt(g0 * g0 + g1 * g1)
                da = g0 / gn
                db = g1 / gn
            if da > step_cap:
                da = step_cap
            elif da < -step_cap:
                da = -step_cap
            if db > step_cap:
                db = step_cap
            elif db < -step_cap:
                db = -step_cap
            # backtracking line search on the weighted log-likelihood:
            # concave objective + ascent check => globally convergent.
            t = 1.0
            moved = False
            for _ in range(30):
                F1 = _model_obj(a0 + t * da, b0 + t * db, m, theta_grid,
                                resp, mc_task, mc_s, mc_n, mc_off,
                                probit, use_bb, phi, p_eps)
                if F1 >= F0 - 1e-12:
                    a0 += t * da
                    b0 += t * db
                    F0 = F1
                    moved = True
                    break
                t *= 0.5
            if not moved or (abs(t * da) < 1e-10 and abs(t * db) < 1e-10):
                break
        params[m, 0] = a0
        params[m, 1] = b0
    return params


@njit(cache=True, inline='always')
def _model_obj(a0, b0, m, theta_grid, resp, mc_task, mc_s, mc_n, mc_off,
               probit, use_bb, phi, p_eps):
    K = theta_grid.shape[1]
    F = 0.0
    for c in range(mc_off[m], mc_off[m + 1]):
        i = mc_task[c]
        s = mc_s[c]
        n = mc_n[c]
        for k in range(K):
            r = resp[i, k]
            if r < 1e-14:
                continue
            eta = a0 + b0 * theta_grid[i, k]
            F += r * _cell_ll(s, n, eta, probit, use_bb, phi, p_eps)
    return F


@njit(cache=True, inline='always')
def _total_obj_shared(alphas, b, theta_grid, resp, mc_task, mc_s, mc_n,
                      mc_off, probit, use_bb, phi, p_eps):
    """Responsibility-weighted total log-lik with a single shared slope b
    and per-model intercepts alphas[m] -- the shslope line-search
    objective (concave in (alphas, b))."""
    n_models = alphas.shape[0]
    K = theta_grid.shape[1]
    F = 0.0
    for m in range(n_models):
        a0 = alphas[m]
        for c in range(mc_off[m], mc_off[m + 1]):
            i = mc_task[c]
            s = mc_s[c]
            n = mc_n[c]
            for k in range(K):
                r = resp[i, k]
                if r < 1e-14:
                    continue
                eta = a0 + b * theta_grid[i, k]
                F += r * _cell_ll(s, n, eta, probit, use_bb, phi, p_eps)
    return F


@njit(cache=True)
def mstep_alpha_shared_beta(params, theta_grid, resp,
                            mc_task, mc_s, mc_n, mc_off,
                            probit, use_bb, phi, p_eps,
                            max_steps=40, grad_tol=1e-9, step_cap=5.0):
    """Damped Newton on (alpha_1..alpha_J, b) with ONE slope b shared
    across all models (the shslope constraint).  The Hessian is an
    arrowhead: block-diagonal in the alphas with a shared b row/column;
    solved by block elimination of the alpha diagonal.  The objective is
    concave (weighted logistic/BB log-lik), so backtracking line search
    on the total gives a globally convergent ascent -- the same story as
    mstep_alpha_beta, just coupled through b.  On exit every params[m, 1]
    holds the same b (broadcast per model)."""
    n_models = params.shape[0]
    K = theta_grid.shape[1]
    alphas = params[:, 0].copy()
    b0 = params[0, 1]                     # shared slope (rows are equal)

    F0 = _total_obj_shared(alphas, b0, theta_grid, resp, mc_task, mc_s,
                           mc_n, mc_off, probit, use_bb, phi, p_eps)
    ga = np.empty(n_models)               # dL/dalpha_m
    Haa = np.empty(n_models)              # d2L/dalpha_m^2  (< 0)
    Hab = np.empty(n_models)              # d2L/(dalpha_m db)
    for _ in range(max_steps):
        gb = 0.0                          # dL/db (pooled over all models)
        Hbb = 0.0                         # d2L/db^2 (pooled)
        for m in range(n_models):
            gam = 0.0
            haa = 0.0
            hab = 0.0
            for c in range(mc_off[m], mc_off[m + 1]):
                i = mc_task[c]
                s = mc_s[c]
                n = mc_n[c]
                for k in range(K):
                    r = resp[i, k]
                    if r < 1e-14:
                        continue
                    th = theta_grid[i, k]
                    eta = alphas[m] + b0 * th
                    g, h = _cell_dll_deta(s, n, eta, probit, use_bb,
                                          phi, p_eps)
                    gam += r * g
                    haa += r * h
                    hab += r * h * th
                    gb += r * g * th
                    Hbb += r * h * th * th
            ga[m] = gam
            Haa[m] = haa
            Hab[m] = hab

        # convergence: all gradient components small
        gmax = abs(gb)
        for m in range(n_models):
            if abs(ga[m]) > gmax:
                gmax = abs(ga[m])
        if gmax < grad_tol:
            break

        # arrowhead Newton solve  H @ delta = -g  (H negative definite).
        # eliminate the alpha diagonal, scalar-solve for db, back-sub.
        # schur = Hbb - sum_m Hab_m^2 / Haa_m ;  rhs = gb - sum_m Hab_m*ga_m/Haa_m
        schur = Hbb
        rhs_b = gb
        ok = True
        for m in range(n_models):
            if abs(Haa[m]) < 1e-12:
                ok = False
                break
            schur -= Hab[m] * Hab[m] / Haa[m]
            rhs_b -= Hab[m] * ga[m] / Haa[m]
        if ok and abs(schur) > 1e-12:
            db = -rhs_b / schur
            da = np.empty(n_models)
            for m in range(n_models):
                # back-substitution: Haa*da + Hab*db = -ga
                da[m] = -(ga[m] + Hab[m] * db) / Haa[m]
        else:
            # near-singular: normalized gradient ascent fallback
            gn = gb * gb
            for m in range(n_models):
                gn += ga[m] * ga[m]
            gn = math.sqrt(gn)
            db = gb / gn
            da = ga / gn

        # step caps (per-component, matching mstep_alpha_beta)
        if db > step_cap:
            db = step_cap
        elif db < -step_cap:
            db = -step_cap
        for m in range(n_models):
            if da[m] > step_cap:
                da[m] = step_cap
            elif da[m] < -step_cap:
                da[m] = -step_cap

        # backtracking line search on the concave total objective
        t = 1.0
        moved = False
        for _ in range(30):
            a_try = alphas + t * da
            b_try = b0 + t * db
            F1 = _total_obj_shared(a_try, b_try, theta_grid, resp, mc_task,
                                   mc_s, mc_n, mc_off, probit, use_bb, phi,
                                   p_eps)
            if F1 >= F0 - 1e-12:
                alphas = a_try
                b0 = b_try
                F0 = F1
                moved = True
                break
            t *= 0.5
        if not moved:
            break
        smax = abs(t * db)
        for m in range(n_models):
            if abs(t * da[m]) > smax:
                smax = abs(t * da[m])
        if smax < 1e-10:
            break

    for m in range(n_models):
        params[m, 0] = alphas[m]
        params[m, 1] = b0                 # broadcast shared slope
    return params


@njit(cache=True)
def mstep_log_variance(Bv, w, s, gamma, max_steps=30, grad_tol=1e-10):
    """Concave damped-Newton maximizer of
        F(gamma) = sum_i [ -w_i * eta_i - 0.5 * s_i * exp(-2 eta_i) ],
    eta = Bv @ gamma  (log sigma).  Same objective as the residual_specs
    machinery; with a single constant column it reduces to the closed
    form sigma^2 = sum(s)/sum(w)."""
    n, d = Bv.shape
    g = gamma.copy()
    for _ in range(max_steps):
        eta = Bv @ g
        grad = np.zeros(d)
        hess = np.zeros((d, d))
        F0 = 0.0
        for i in range(n):
            e2 = math.exp(-2.0 * eta[i])
            gi = -w[i] + s[i] * e2
            hi = -2.0 * s[i] * e2
            F0 += -w[i] * eta[i] - 0.5 * s[i] * e2
            for a in range(d):
                grad[a] += gi * Bv[i, a]
                for b in range(d):
                    hess[a, b] += hi * Bv[i, a] * Bv[i, b]
        gn = 0.0
        for a in range(d):
            if abs(grad[a]) > gn:
                gn = abs(grad[a])
        if gn < grad_tol:
            break
        step = np.linalg.solve(hess, -grad)
        t = 1.0
        for _ in range(30):
            g_try = g + t * step
            eta_t = Bv @ g_try
            F1 = 0.0
            for i in range(n):
                F1 += (-w[i] * eta_t[i]
                       - 0.5 * s[i] * math.exp(-2.0 * eta_t[i]))
            if F1 >= F0 - 1e-12:
                break
            t *= 0.5
        g = g + t * step
    return g


@njit(cache=True)
def phi_score(theta_grid, resp, means_dummy, params,
              cell_midx, cell_s, cell_n, cell_off, probit, phi, p_eps):
    """Responsibility-weighted (dL/dphi, d2L/dphi2) over all cells."""
    n_tasks, K = theta_grid.shape
    d1 = 0.0
    d2 = 0.0
    for i in range(n_tasks):
        for c in range(cell_off[i], cell_off[i + 1]):
            m = cell_midx[c]
            s = cell_s[c]
            n = cell_n[c]
            for k in range(K):
                r = resp[i, k]
                if r < 1e-14:
                    continue
                eta = params[m, 0] + params[m, 1] * theta_grid[i, k]
                a, b = _cell_dll_dphi(s, n, eta, probit, phi, p_eps)
                d1 += r * a
                d2 += r * b
    return d1, d2


@njit(cache=True)
def weighted_ll(theta_grid, resp, params, cell_midx, cell_s, cell_n,
                cell_off, probit, use_bb, phi, p_eps):
    """sum_{i,k} resp[i,k] * ll_ik(phi) -- the phi line-search objective."""
    n_tasks, K = theta_grid.shape
    tot = 0.0
    for i in range(n_tasks):
        for c in range(cell_off[i], cell_off[i + 1]):
            m = cell_midx[c]
            s = cell_s[c]
            n = cell_n[c]
            for k in range(K):
                r = resp[i, k]
                if r < 1e-14:
                    continue
                eta = params[m, 0] + params[m, 1] * theta_grid[i, k]
                tot += r * _cell_ll(s, n, eta, probit, use_bb, phi, p_eps)
    return tot


# ----------------------------------------------------------------------
# evaluation helpers (marginal probabilities, joint loglik)
# ----------------------------------------------------------------------

@njit(cache=True)
def marginal_p_grid(alpha, beta, h, s, gh_roots, gh_w, probit):
    """p_j(x_g) = E_{theta ~ N(h_g, s_g^2)}[inv_link(alpha + beta*theta)]
    for one model over a grid of (h, s) pairs.  Probit uses the exact
    closed form Phi((alpha + beta*h)/sqrt(1 + beta^2 s^2))."""
    G = len(h)
    out = np.empty(G)
    if probit:
        for g in range(G):
            z = (alpha + beta * h[g]) / math.sqrt(
                1.0 + beta * beta * s[g] * s[g])
            out[g] = _inv_link(z, True)
        return out
    Kq = len(gh_roots)
    for g in range(G):
        acc = 0.0
        for k in range(Kq):
            th = h[g] + SQRT2 * s[g] * gh_roots[k]
            acc += gh_w[k] * _inv_link(alpha + beta * th, False)
        out[g] = acc
    return out


@njit(cache=True)
def joint_ll_pertask(t_array, means, sigmas, params,
                     cell_midx, cell_s, cell_n, cell_off,
                     gh_roots, log_gh_w, probit, use_bb, phi, p_eps):
    """Per-task joint marginal log-lik with a FRESH GH grid centred at
    the prior mean (evaluation only -- not the EM proposal grid)."""
    n_tasks = len(t_array)
    Kq = len(gh_roots)
    out = np.empty(n_tasks)
    for i in range(n_tasks):
        mx = -np.inf
        vals = np.empty(Kq)
        for k in range(Kq):
            th = means[i] + SQRT2 * sigmas[i] * gh_roots[k]
            acc = 0.0
            for c in range(cell_off[i], cell_off[i + 1]):
                m = cell_midx[c]
                eta = params[m, 0] + params[m, 1] * th
                acc += _cell_ll(cell_s[c], cell_n[c], eta,
                                probit, use_bb, phi, p_eps)
            v = log_gh_w[k] + acc
            vals[k] = v
            if v > mx:
                mx = v
        s = 0.0
        for k in range(Kq):
            s += math.exp(vals[k] - mx)
        out[i] = mx + math.log(s)
    return out


@njit(cache=True)
def joint_ll_family(t_array, means, sigmas, params,
                    cell_midx, cell_s, cell_n, cell_off,
                    fam_offsets, fam_task_idx, tau_vec,
                    gh_roots, log_gh_w, g_roots, log_gw,
                    probit, use_bb, phi, p_eps):
    """Per-FAMILY joint marginal log-lik (g integrated at family level,
    theta_i integrated per task inside).  Fresh grids, evaluation only."""
    n_fams = len(fam_offsets) - 1
    Kq = len(gh_roots)
    Kg = len(g_roots)
    out = np.empty(n_fams)
    for fi in range(n_fams):
        f0 = fam_offsets[fi]
        f1 = fam_offsets[fi + 1]
        g_scale = SQRT2 * tau_vec[fi]
        log_pf = np.empty(Kg)
        for a in range(Kg):
            g = g_scale * g_roots[a]
            tot = 0.0
            for jj in range(f0, f1):
                i = fam_task_idx[jj]
                mx = -np.inf
                vals = np.empty(Kq)
                for k in range(Kq):
                    th = means[i] + g + SQRT2 * sigmas[i] * gh_roots[k]
                    acc = 0.0
                    for c in range(cell_off[i], cell_off[i + 1]):
                        m = cell_midx[c]
                        eta = params[m, 0] + params[m, 1] * th
                        acc += _cell_ll(cell_s[c], cell_n[c], eta,
                                        probit, use_bb, phi, p_eps)
                    v = log_gh_w[k] + acc
                    vals[k] = v
                    if v > mx:
                        mx = v
                s = 0.0
                for k in range(Kq):
                    s += math.exp(vals[k] - mx)
                tot += mx + math.log(s)
            log_pf[a] = log_gw[a] + tot
        mxg = -np.inf
        for a in range(Kg):
            if log_pf[a] > mxg:
                mxg = log_pf[a]
        s = 0.0
        for a in range(Kg):
            s += math.exp(log_pf[a] - mxg)
        out[fi] = mxg + math.log(s)
    return out
