"""The JML-shape route to Model 1: a difficulty curve from fixed-effect IRT.

Three stages, all on the data being fitted (so CV folds redo all three):

1. **JML fit of the classic 2PL.**  logit p_ij = alpha_j + beta_j theta_i
   with every theta_i a free parameter (no prior): maximize the JOINT
   likelihood over (theta, alpha, beta), Bernoulli runs, logit link.  A
   task no AI solved, or every AI always solved, has no finite MLE and is
   dropped (`drop='extreme'`); `drop='quasi'` also drops tasks whose
   failures, or whose successes, all come from a single AI, whose
   theta-hat only that one AI pins down.  theta is oriented as
   difficulty (beta < 0), and each kept task gets the observed-
   information standard error SE_i = 1 / sqrt(sum_j beta_j^2 n_ij p_ij
   (1 - p_ij)), the AI side held fixed.
2. **Monotone spline of theta-hat on x = log2 minutes**, by box-bounded
   weighted least squares on the merged natural-cubic basis (the shape's
   own basis, so `mean_kint` sets its size and knots).  Task weights:
   'invvar' 1/SE^2; 'unweighted' 1; 'sqrtfamily' 1/sqrt(n_F); 'totvar'
   1/(SE^2 + tau^2), with the between-task variance tau^2 estimated by
   Paule-Mandel.
3. **Model 1 with that shape frozen.**  The spline g is normalized to the
   shared shape f: 0 -> 1 and only the AI parameters of
   logit p_ij = a_j - b_j f(x_i) are fitted, by basic.fit_shared with
   `fixed_w` (so `shslope`, `bb`, `probit` and `sqrtfamily` mean what
   they mean for any `shared` spec).

Ported from jml_shape_pilot/ and classic_vs_model2/jml.py in the
surrounding repository, which validated it on the canonical data (no
variant beat Model 1 under family CV).  Known caveats: JML theta-hats
carry the incidental-parameter bias (Neyman-Scott), with ~20-26 AIs per
task; and in some CV folds the joint maximum does not exist (a weak AI
becomes separable on theta).  `beta_ridge` (default 0.01, a prior sd of
10 on the standardized theta scale) keeps such slopes finite; if the
sweeps still fail to converge the fit raises, and `drop='quasi'` is the
usual fix.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq, lsq_linear

from . import basic, em
from .basis import build_monotone_basis

WEIGHTS = ('invvar', 'unweighted', 'sqrtfamily', 'totvar')
DROP_RULES = ('extreme', 'quasi')
BETA_RIDGE = 0.01
STEP_CAP = 1.0        # max |Newton step| per coordinate, guards early sweeps


def _sigmoid(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))   # overflow-free logistic


def _cell_task(idx):
    return np.repeat(np.arange(idx['n_tasks']), np.diff(idx['cell_off']))


def mixed_tasks(idx):
    """Tasks with at least one success and one failure."""
    off = idx['cell_off']
    s = np.add.reduceat(idx['cell_s'], off[:-1])
    n = np.add.reduceat(idx['cell_n'], off[:-1])
    return (s > 0) & (s < n)


def quasi_extreme(idx):
    """Tasks whose failures, or whose successes, come from at most one
    AI (this includes the extreme tasks, where it is zero AIs)."""
    ct = _cell_task(idx)
    fail_ai = np.bincount(ct, idx['cell_n'] > idx['cell_s'], idx['n_tasks'])
    succ_ai = np.bincount(ct, idx['cell_s'] > 0, idx['n_tasks'])
    return (fail_ai <= 1) | (succ_ai <= 1)


def fit_jml(idx, theta_init, keep=None, max_sweeps=20000, tol=1e-10,
            beta_ridge=BETA_RIDGE):
    """Joint ML of the classic 2PL on the tasks `keep` (default: the
    mixed-outcome tasks).

    Block coordinate ascent: one Newton step for each AI's (alpha_j,
    beta_j) given theta, then one for each theta_i given the AIs; both
    blocks are concave.  Each sweep re-standardizes theta to mean 0, sd 1
    over the kept tasks, which pins the likelihood's affine ridge.
    beta_ridge adds -beta_ridge/2 * sum_j beta_j^2.  Returns theta (NaN
    for dropped tasks), params (alpha, beta) and convergence info."""
    keep = mixed_tasks(idx) if keep is None else (keep & mixed_tasks(idx))
    n_tasks, n_models = idx['n_tasks'], idx['n_models']
    cell_task = _cell_task(idx)
    use = keep[cell_task]
    new_id = np.cumsum(keep) - 1           # kept tasks renumbered 0..K-1
    ti = new_id[cell_task[use]]
    mj = idx['cell_midx'][use]
    s, n = idx['cell_s'][use], idx['cell_n'][use]
    K = int(keep.sum())

    th = np.asarray(theta_init, float)[keep].copy()
    th = (th - th.mean()) / th.std()
    alpha = np.zeros(n_models)
    beta = np.full(n_models, -1.0)
    converged = False
    ll = -np.inf
    for sweep in range(max_sweeps):
        old = np.concatenate([th, alpha, beta])

        # AI block: one Newton step of each AI's logistic regression on theta
        p = _sigmoid(alpha[mj] + beta[mj] * th[ti])
        r, w = s - n * p, n * p * (1.0 - p)
        g_a = np.bincount(mj, r, n_models)
        g_b = np.bincount(mj, r * th[ti], n_models) - beta_ridge * beta
        h_aa = np.bincount(mj, w, n_models)
        h_ab = np.bincount(mj, w * th[ti], n_models)
        h_bb = np.bincount(mj, w * th[ti] ** 2, n_models) + beta_ridge
        det = h_aa * h_bb - h_ab ** 2
        alpha += np.clip((h_bb * g_a - h_ab * g_b) / det, -STEP_CAP, STEP_CAP)
        beta += np.clip((h_aa * g_b - h_ab * g_a) / det, -STEP_CAP, STEP_CAP)

        # task block: one Newton step for each theta_i given the AIs
        p = _sigmoid(alpha[mj] + beta[mj] * th[ti])
        r, w = s - n * p, n * p * (1.0 - p)
        g_t = np.bincount(ti, beta[mj] * r, K)
        h_t = np.bincount(ti, beta[mj] ** 2 * w, K)
        th += np.clip(g_t / h_t, -STEP_CAP, STEP_CAP)

        # pin the affine ridge: theta mean 0, sd 1 (likelihood-invariant)
        m, sd = th.mean(), th.std()
        th = (th - m) / sd
        alpha, beta = alpha + beta * m, beta * sd

        eta = alpha[mj] + beta[mj] * th[ti]
        ll = float(np.sum(s * eta - n * np.logaddexp(0.0, eta)))
        change = float(np.max(np.abs(np.concatenate([th, alpha, beta])
                                     - old)))
        if not np.isfinite(change):
            break
        if change < tol:
            converged = True
            break

    theta = np.full(n_tasks, np.nan)
    theta[keep] = th
    return {'theta': theta, 'params': np.column_stack([alpha, beta]),
            'keep': keep, 'logl_joint': ll, 'sweeps': sweep + 1,
            'converged': converged}


def theta_hat(idx, drop='extreme', beta_ridge=BETA_RIDGE):
    """Stage 1: JML difficulties theta-hat and their SEs (NaN for
    dropped tasks), oriented so that larger theta is harder."""
    if drop not in DROP_RULES:
        raise ValueError(f'jml_drop must be one of {DROP_RULES}')
    keep = ~quasi_extreme(idx) if drop == 'quasi' else None
    with np.errstate(all='ignore'):
        r = fit_jml(idx, idx['t_array'], keep=keep, beta_ridge=beta_ridge)
    if not r['converged']:
        raise RuntimeError(
            f"JML did not converge after {r['sweeps']} sweeps (drop="
            f"{drop!r}, beta_ridge={beta_ridge}): the joint maximum may not "
            "exist on these data; try jml_drop='quasi' or a larger "
            'jml_beta_ridge')
    th = r['theta'].copy()
    alpha, beta = r['params'][:, 0], r['params'][:, 1].copy()
    # the joint likelihood is invariant under theta -> -theta, beta ->
    # -beta; the fit starts at theta = t, beta = -1, so this rarely fires
    flipped = bool(beta.mean() > 0)
    if flipped:
        th, beta = -th, -beta
    ct = _cell_task(idx)
    mj = idx['cell_midx']
    p = _sigmoid(alpha[mj] + beta[mj] * np.nan_to_num(th)[ct])
    info = np.bincount(ct, beta[mj] ** 2 * idx['cell_n'] * p * (1 - p),
                       idx['n_tasks'])
    with np.errstate(divide='ignore'):
        se = np.where(r['keep'], 1.0 / np.sqrt(info), np.nan)
    return {'theta': th, 'se': se, 'keep': r['keep'],
            'params': np.column_stack([alpha, beta]), 'flipped': flipped,
            'sweeps': r['sweeps'], 'logl_joint': r['logl_joint']}


def fit_monotone_g(x, y, w, nb):
    """Box-bounded WLS: g(x) = c0 + sum_m w_m M_m(x), w_m >= 0."""
    X = nb.design(x)
    sw = np.sqrt(w)
    lb = np.full(X.shape[1], -np.inf)
    lb[1:] = 0.0
    return lsq_linear(X * sw[:, None], y * sw, bounds=(lb, np.inf),
                      tol=1e-12).x


def paule_mandel_tau2(x, y, se, nb):
    """Between-task variance tau^2 of theta_hat_i = g(x_i) + u_i + e_i,
    u_i ~ (0, tau^2), e_i ~ (0, SE_i^2): the root of Q(tau^2) = k - df,
    Q the weighted residual sum of squares of the monotone WLS fit at
    weights 1/(SE^2 + tau^2).  Q decreases in tau^2, so the root is
    unique; tau^2 = 0 if Q(0) <= k - df already."""
    X = nb.design(x)
    target = len(y) - nb.df

    def excess(tau2):
        w = 1.0 / (se ** 2 + tau2)
        r = y - X @ fit_monotone_g(x, y, w, nb)
        return float(np.sum(w * r ** 2)) - target

    if excess(0.0) <= 0:
        return 0.0
    hi = float(np.var(y))
    while excess(hi) > 0:
        hi *= 2.0
    return brentq(excess, 0.0, hi, xtol=1e-10, rtol=1e-10)


def task_weights(kind, families, se, x=None, y=None, nb=None):
    """Stage-2 weight of each regression task, and tau^2 ('totvar' only,
    else NaN).  'sqrtfamily' counts n_F among the regression's tasks, so
    each family's total weight is sqrt(n_F)."""
    if kind == 'unweighted':
        return np.ones(len(se)), np.nan
    if kind == 'invvar':
        return 1.0 / se ** 2, np.nan
    if kind == 'totvar':
        tau2 = paule_mandel_tau2(x, y, se, nb)
        return 1.0 / (se ** 2 + tau2), tau2
    if kind == 'sqrtfamily':
        _, inv, cnt = np.unique(families, return_inverse=True,
                                return_counts=True)
        return 1.0 / np.sqrt(cnt[inv].astype(float)), np.nan
    raise ValueError(f'jml_weights must be one of {WEIGHTS}')


def fit_jml_shared(idx, weights='invvar', drop='extreme',
                   beta_ridge=BETA_RIDGE, probit=False, bb=False,
                   shslope=False, sqrtfamily=False, kint=None):
    """All three stages; returns a `shared` fit dict with the stage-1/2
    results under 'jml'."""
    kint = em.MEAN_KINT if kint is None else kint
    x = idx['t_array']
    nb = build_monotone_basis(x, kint)
    st1 = theta_hat(idx, drop, beta_ridge)
    kept = ~np.isnan(st1['theta'])
    w_task = np.full(idx['n_tasks'], np.nan)
    w_task[kept], tau2 = task_weights(
        weights, idx['task_family'][kept], st1['se'][kept], x=x[kept],
        y=st1['theta'][kept], nb=nb)
    g_coef = fit_monotone_g(x[kept], st1['theta'][kept], w_task[kept], nb)
    col_hi = nb.raw_merged(np.array([nb.hi]))[0]
    mass = g_coef[1:] * col_hi
    if mass.sum() <= 1e-10:
        raise RuntimeError('JML shape: the fitted difficulty curve is flat')
    fit = basic.fit_shared(idx, probit=probit, bb=bb, shslope=shslope,
                           sqrtfamily=sqrtfamily, kint=kint,
                           fixed_w=mass / mass.sum())
    fit['shape_source'] = 'jml'
    fit['jml'] = {'weights': weights, 'drop': drop,
                  'beta_ridge': float(beta_ridge),
                  'theta_hat': st1['theta'], 'theta_se': st1['se'],
                  'task_weight': w_task, 'g_coefs': g_coef,
                  'tau2': float(tau2), 'params': st1['params'],
                  'n_dropped': int((~kept).sum()), 'sweeps': st1['sweeps'],
                  'flipped': st1['flipped'],
                  'logl_joint': st1['logl_joint']}
    return fit
