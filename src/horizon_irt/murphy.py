"""Elementary scores and Murphy curves for held-out marginal forecasts.

The elementary (cost-loss) score at level q for a probability forecast
p of a binary outcome Y is

    S_q = q*(1-Y)*1{p > q} + (1-q)*Y*1{p < q},

and the Murphy curve q -> mean S_q (Ehm, Gneiting, Jordan & Kruger
2016) shows every proper score at once.  Means are weighted over
held-out runs with the evaluation targets of horizon_irt.evaluate, and
paired differences between specifications get cluster-robust (Bayle)
intervals clustered on held-out task or family.

Computed from the CV prediction tables (cv_preds_<scheme>.csv): one row
per held-out (fold, task_id, task_family, model) cell with `s`
successes out of `n` runs and one probability column per spec.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from .evaluate import cell_weights, family_sizes_for, TARGETS

Q_MARKS = (0.5, 0.8)
REF_SPEC = 'linear'
KEYS = ['fold', 'task_id', 'task_family', 'model', 's', 'n']


# ---------------------------------------------------------------------------
# elementary score & Murphy curve
# ---------------------------------------------------------------------------

def elementary_score_cell(p, s, n, q):
    """Total elementary score S_q summed over the runs of each cell.

    A cell has `s` successes (Y=1) and `n-s` failures (Y=0) at forecast p:
        Y=1 run:  (1-q) * 1{p < q}
        Y=0 run:   q    * 1{p > q}
    so the cell total is
        s*(1-q)*1{p<q} + (n-s)*q*1{p>q}.

    `p` may be a 1-D array of cells; `q` a scalar or a 1-D grid.  Returns an
    array broadcast over (cells, q) when q is a grid, else over cells.
    """
    p = np.asarray(p, dtype=float)[:, None] if np.ndim(q) else np.asarray(p, float)
    s = np.asarray(s, dtype=float)
    n = np.asarray(n, dtype=float)
    q = np.asarray(q, dtype=float)
    if q.ndim:                                  # (cells, q)
        s = s[:, None]
        n = n[:, None]
    below = (p < q).astype(float)
    above = (p > q).astype(float)
    return s * (1.0 - q) * below + (n - s) * q * above


def murphy_curve(preds, spec, target, q_grid):
    """Weighted mean elementary score S_hat(q) for one spec over a q-grid.

        S_hat(q) = sum_cells w * S_q_cell / sum_cells w * n

    `preds` is a held-out prediction frame (one CV scheme) with columns
    task_id, task_family, s, n and a `spec` column of marginal probabilities.
    `w` is the per-run target weight (a cell's mass is w*n); the normaliser
    sum(w*n) matches eval_all.cell_metrics so the y-axis is a per-run mean.
    """
    p = preds[spec].to_numpy(dtype=float)
    s = preds['s'].to_numpy(dtype=float)
    n = preds['n'].to_numpy(dtype=float)
    fam_sizes = family_sizes_for(preds['task_id'], preds['task_family'])
    w = cell_weights(target, n, fam_sizes)
    denom = float(np.sum(w * n))
    sc = elementary_score_cell(p, s, n, q_grid)         # (cells, q)
    return (w @ sc) / denom                              # (q,)


def all_curves(preds, specs, target, q_grid):
    """{spec: S_hat(q) array} for every spec present in `preds`."""
    return {sp: murphy_curve(preds, sp, target, q_grid)
            for sp in specs if sp in preds.columns}


# ---------------------------------------------------------------------------
# paired contrast CIs on the difference curve  S_q,spec - S_q,linear
# ---------------------------------------------------------------------------

# two-sided normal quantiles for the confidence levels we draw
Z = {0.80: 1.2815515594607444,
     0.90: 1.6448536269514722,
     0.95: 1.959963984540054}


def _run_expand(preds, target, cluster_col):
    """Per-RUN fold, cluster and weight arrays (cells expanded by n).

    Mirrors eval_all.stage_cv_aggregate: a cell of n runs contributes n
    identical fold/cluster/weight entries (the per-run target weight), ordered
    s ones then n-s zeros -- matching how the run-level Y is laid out below.
    """
    n = preds['n'].to_numpy(int)
    fam = family_sizes_for(preds['task_id'], preds['task_family'])
    w = cell_weights(target, preds['n'].to_numpy(float), fam)
    folds_r = np.repeat(preds['fold'].to_numpy(), n)
    clus_r = np.repeat(preds[cluster_col].to_numpy(), n)
    w_r = np.repeat(w, n)
    return folds_r, clus_r, w_r


def _run_y(preds):
    """Per-run outcomes Y in the cell order used everywhere (s ones, then 0s)."""
    s = preds['s'].to_numpy(int)
    n = preds['n'].to_numpy(int)
    return np.concatenate([np.r_[np.ones(si), np.zeros(ni - si)]
                           for si, ni in zip(s, n)])


def _run_loss(p_cell, y_r, n, q):
    """Per-run elementary loss S_q for a single spec at scalar q.

        Y=1 run: (1-q)*1{p<q};   Y=0 run: q*1{p>q}
    `p_cell` is the per-cell forecast; expanded to runs to align with y_r.
    """
    p_r = np.repeat(p_cell, n.astype(int))
    below = (p_r < q).astype(float)
    above = (p_r > q).astype(float)
    return y_r * (1.0 - q) * below + (1.0 - y_r) * q * above


def contrast_var(d_r, folds_r, clus_r, w_r):
    """Cluster-robust (Bayle) variance of the weighted mean of d_r.

    Identical construction to eval_all.ci_tiers' `cluster` tier: weighted,
    fold-mean-centred residuals summed within held-out clusters.  Returns
    (estimate, variance); half-width at level L is Z[L]*sqrt(variance).
    """
    d_r = np.asarray(d_r, float)
    w_r = np.asarray(w_r, float)
    sw = float(w_r.sum())
    est = float(np.sum(w_r * d_r) / sw)
    df_ = pd.DataFrame({'wd': w_r * d_r, 'w': w_r, 'fold': folds_r})
    g = df_.groupby('fold')[['wd', 'w']].transform('sum')
    fold_mean = (g['wd'] / g['w']).to_numpy()
    resid = w_r * (d_r - fold_mean)
    t_c = pd.DataFrame({'r': resid, 'cl': clus_r}).groupby('cl')['r'].sum()
    var_cl = float(np.sum(t_c.to_numpy() ** 2)) / sw ** 2
    return est, var_cl


def diff_ci_curve(preds, spec, target, cluster_col, q_grid, level=0.95,
                  ref=REF_SPEC):
    """Cluster-robust CI half-width for (S_q,spec - S_q,ref) over q_grid.

    Returns (est, half) arrays of length len(q_grid); est is the paired
    difference at each q (equals murphy_curve(spec) - murphy_curve(ref))
    and [est-half, est+half] is the two-sided `level` interval.  `ref`
    defaults to REF_SPEC ('linear') but any spec column works -- e.g. a
    strengthened baseline such as linear_shslope_sqrtfamily.
    """
    folds_r, clus_r, w_r = _run_expand(preds, target, cluster_col)
    y_r = _run_y(preds)
    n = preds['n'].to_numpy()
    p_spec = preds[spec].to_numpy(float)
    p_lin = preds[ref].to_numpy(float)
    z = Z[level]
    est = np.empty(len(q_grid))
    half = np.empty(len(q_grid))
    for i, q in enumerate(q_grid):
        d_r = _run_loss(p_spec, y_r, n, q) - _run_loss(p_lin, y_r, n, q)
        e, v = contrast_var(d_r, folds_r, clus_r, w_r)
        est[i] = e
        half[i] = z * math.sqrt(max(v, 0.0))
    return est, half


def diff_ci_points(preds, spec, target, cluster_col, q_points, levels,
                   ref=REF_SPEC):
    """CIs at specific q (e.g. 0.5, 0.8) at several confidence levels.

    Returns {q: {'est': e, level: half, ...}} -- one variance per q reused
    across levels (only the z-multiplier changes).  `ref` is the comparison
    spec (default REF_SPEC).
    """
    folds_r, clus_r, w_r = _run_expand(preds, target, cluster_col)
    y_r = _run_y(preds)
    n = preds['n'].to_numpy()
    p_spec = preds[spec].to_numpy(float)
    p_lin = preds[ref].to_numpy(float)
    out = {}
    for q in q_points:
        d_r = _run_loss(p_spec, y_r, n, q) - _run_loss(p_lin, y_r, n, q)
        e, v = contrast_var(d_r, folds_r, clus_r, w_r)
        rec = {'est': e, 'var': v}
        for L in levels:
            rec[L] = Z[L] * math.sqrt(max(v, 0.0))
        out[q] = rec
    return out


def smooth_curve(c, window):
    """Centred rolling mean over the q-grid for DISPLAY only.

    The population elementary score E[S_q] is smooth in q; the empirical curve
    is a step function that jumps as q crosses each observed p, so a small
    moving average removes threshold jitter without shifting the curve.  Raw
    (unsmoothed) values are what get written to murphy_curves.csv.
    """
    if not window or window <= 1:
        return c
    w = int(window) | 1                       # force odd
    k = np.ones(w) / w
    pad = w // 2
    cp = np.pad(c, pad, mode='edge')
    return np.convolve(cp, k, mode='valid')


def load_preds(preds_dir, scheme):
    """The CV prediction table cv_preds_<scheme>.csv from `preds_dir`."""
    return pd.read_csv(Path(preds_dir) / f'cv_preds_{scheme}.csv')


def spec_columns(preds):
    """The spec (probability) columns = all columns after the key block."""
    return [c for c in preds.columns if c not in KEYS]
