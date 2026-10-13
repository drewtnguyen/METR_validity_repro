"""Evaluation battery for the headline grid (headline-model-grid skill).

Metrics (sequence convention throughout, so Bernoulli and BB are on one
scale):
  joint_ll        prior-predictive joint log-lik (theta -- and xi at
                  family level for the family scheme -- integrated out)
  marg_logscore   per-RUN MARGINAL Bernoulli log loss at p_j(t_i)
  brier           per-RUN Brier at p_j(t_i)
  ece             15-bin expected calibration error

`marg_logscore` is a sum of per-run Bernoulli log scores at the marginal
probability.  Under dependence (runs within a cell, tasks within a
family) that is a MARGINAL/composite log score -- proper for the
marginals -- and NOT the joint log-likelihood; `joint_ll` is the joint
one.  The two answer different questions and are reported separately.

Weighting.  Every metric is a weighted mean over held-out runs, with the
weight defining the target population -- i.e. which prediction problems
count how much (see the designing-marginal-scoring-rules skill).  Four
targets are computed for every metric:

  equal_run     w_run = 1.  A uniformly drawn RUN.  This is what the
                benchmark hands you, but it lets big rerun-heavy
                families dominate: the top 5 of 79 families are 38% of
                all runs.
  equal_task    w_run = 1/n_ij.  A uniformly drawn (task, model) cell.
  equal_family  w_run = 1/(n_F * n_ij), n_F = #tasks in the family.  A
                uniformly drawn task TYPE.  Headline under family CV.
  sqrt_family   w_run = 1/(sqrt(n_F) * n_ij); total family weight scales
                as sqrt(n_F).  An intermediate TARGET, reported as a
                sensitivity analysis.  It is NOT a correlation
                correction -- the 1/2 exponent is not implied by any
                within-family correlation we estimate.

Training is unweighted in every fold (each fold reproduces the full-data
recipe); only the held-out aggregation varies.  Training and evaluation
weights answer different questions and need not match.

In-sample: from the saved full-data fits.
CV: 5-fold task-grouped AND family-grouped (seed 42), refit per fold at
the fixed config, prior-predictive scoring on identical held-out cells
(models with <2 distinct train outcomes are dropped from that fold for
ALL specs).  CIs: naive SEM / Bayle iid-row / Bayle cluster-robust at
the held-out group level, plus paired contrasts vs `linear`.

On fold overlap: CV fits share most of their training data, so held-out
losses are dependent across folds.  No general variance correction for
this exists (Bengio & Grandvalet 2004).  The cluster-robust interval is
the reported default; its estimand is the average test error over the K
training sets actually used (Bayle et al., arXiv:2007.12671), under
which fold overlap is part of the estimand rather than missing variance.
"""
from __future__ import annotations

import json
import math
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from . import data, em, fitting, io, model_api

Z95 = 1.959963984540054
N_BINS = 15

# target distributions over held-out runs; see module docstring
TARGETS = ('equal_run', 'equal_task', 'equal_family', 'sqrt_family')


# ----------------------------------------------------------------------
# evaluation weights (target distribution)
# ----------------------------------------------------------------------

def cell_weights(target, n, fam_sizes):
    """Per-RUN weight for each cell under `target`.

    `n` is runs per cell, `fam_sizes` the number of held-out TASKS in
    each cell's family.  Returned weights are per-run, so a cell's total
    mass is w * n; equal_run therefore gives cells mass proportional to
    their run count (the historical behaviour) and the others normalise
    it away.  Scale is irrelevant -- every consumer normalises by the
    total weight -- but equal_run is left as exactly 1.0 so that it
    reproduces the unweighted arithmetic bit-for-bit.
    """
    n = np.asarray(n, dtype=float)
    if target == 'equal_run':
        return np.ones_like(n)
    if target == 'equal_task':
        return 1.0 / n
    fam = np.asarray(fam_sizes, dtype=float)
    if target == 'equal_family':
        return 1.0 / (fam * n)
    if target == 'sqrt_family':
        return 1.0 / (np.sqrt(fam) * n)
    raise ValueError(f'unknown target {target!r}')


def family_sizes_for(task_ids, families):
    """#distinct tasks per family, broadcast to one value per cell."""
    d = pd.DataFrame({'task_id': task_ids, 'fam': families})
    sizes = d.groupby('fam')['task_id'].nunique()
    return d['fam'].map(sizes).to_numpy(dtype=float)


# ----------------------------------------------------------------------
# metric helpers (cell-level, weighted)
# ----------------------------------------------------------------------

def cell_metrics(p, s, n, w=None):
    """Weighted mean log loss and Brier from cell-level (p, s, n).

    `w` is the per-run weight of each cell (default 1 = equal_run); a
    cell contributes w * n total mass.
    """
    p = np.clip(p, 1e-12, 1 - 1e-12)
    w = np.ones_like(np.asarray(n, dtype=float)) if w is None else w
    tot = float(np.sum(w * n))
    logloss = -float(np.sum(w * (s * np.log(p)
                                 + (n - s) * np.log(1 - p)))) / tot
    brier = float(np.sum(w * (s * (1 - p) ** 2 + (n - s) * p ** 2))) / tot
    return logloss, brier


def ece_weighted(p, s, n, w=None, n_bins=N_BINS):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    w = np.ones_like(np.asarray(n, dtype=float)) if w is None else w
    wn = w * n
    edges = np.linspace(0, 1, n_bins + 1)
    b = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    tot = float(wn.sum())
    ece = 0.0
    rows = []
    for k in range(n_bins):
        m = b == k
        wk = float(wn[m].sum())
        if wk == 0:
            continue
        conf = float(np.sum(p[m] * wn[m]) / wk)
        acc = float(np.sum(w[m] * s[m]) / wk)
        ece += (wk / tot) * abs(acc - conf)
        rows.append((k, conf, acc, wk))
    return ece, rows


def run_level_losses(p, s, n):
    """Expand cells to per-run loss arrays (logloss, brier)."""
    p = np.clip(p, 1e-12, 1 - 1e-12)
    reps = n.astype(int)
    p_r = np.repeat(p, reps)
    y_r = np.concatenate([np.r_[np.ones(int(si)), np.zeros(int(ni - si))]
                          for si, ni in zip(s, n)])
    ll = -(y_r * np.log(p_r) + (1 - y_r) * np.log(1 - p_r))
    br = (p_r - y_r) ** 2
    return ll, br


# ----------------------------------------------------------------------
# CI machinery (three tiers, per bayle_ci.py conventions)
# ----------------------------------------------------------------------

def ci_tiers(d, folds, clusters, w=None, center='fold'):
    """Three CI tiers for a weighted mean of per-run losses.

    With weights the estimate is sum(w*d)/sum(w) and every variance is
    formed from the weighted contributions w*e, normalised by sum(w)
    (which plays the role N did in the unweighted case).  With
    center='fold', fold means are weighted too, so the centring matches
    the repeated-fold estimand.  center='global' is used for
    leave-one-cluster-out CV, where each fold is one cluster and fold
    centring would make every cluster contribution identically zero.
    w=None (all weights 1) reproduces the unweighted formulas exactly.
    """
    d = np.asarray(d, dtype=float)
    N = len(d)
    w = np.ones(N) if w is None else np.asarray(w, dtype=float)
    sw = float(w.sum())
    est = float(np.sum(w * d) / sw)
    if center == 'fold':
        df_ = pd.DataFrame({'wd': w * d, 'w': w, 'fold': folds})
        g = df_.groupby('fold')[['wd', 'w']].transform('sum')
        center_value = (g['wd'] / g['w']).to_numpy()
    elif center == 'global':
        center_value = est
    else:
        raise ValueError("center must be 'fold' or 'global'")
    resid = w * (d - center_value)
    t_c = pd.DataFrame({'r': resid, 'cl': clusters}).groupby('cl')['r'].sum()
    var_cl = float(np.sum(t_c.to_numpy() ** 2)) / sw ** 2
    var_row = float(np.sum(resid ** 2)) / sw ** 2
    # naive tier: weighted-mean sampling variance treating rows as iid
    dev = w * (d - est)
    var_sem = float(np.sum(dev ** 2)) * N / ((N - 1) * sw ** 2)
    out = {'est': est, 'n_rows': N, 'n_clusters': int(t_c.shape[0]),
           'sum_w': sw}
    for tag, v in [('cluster', var_cl), ('bayle', var_row),
                   ('sem', var_sem)]:
        half = Z95 * math.sqrt(v)
        out[f'{tag}_lo'] = est - half
        out[f'{tag}_hi'] = est + half
    return out


# ----------------------------------------------------------------------
# folds
# ----------------------------------------------------------------------

def make_folds(df, scheme, n_folds, seed):
    if scheme == 'task':
        ids = np.sort(df['task_id'].unique())
        ordered = (ids if seed is None
                   else np.random.default_rng(seed).permutation(ids))
        blocks = np.array_split(ordered, n_folds)
        return [set(b) for b in blocks]
    fams = np.sort(df['task_family'].unique())
    ordered = (fams if seed is None
               else np.random.default_rng(seed).permutation(fams))
    blocks = np.array_split(ordered, n_folds)
    fam_tasks = df.groupby('task_family')['task_id'].agg(set)
    return [set().union(*(fam_tasks[f] for f in b)) for b in blocks]


def valid_models(train_df):
    """Models with both outcome classes present in the training fold."""
    g = train_df.groupby('alias')[['n_successes', 'n_runs']].sum()
    return set(g.index[(g['n_successes'] > 0)
                       & (g['n_successes'] < g['n_runs'])])


# ----------------------------------------------------------------------
# CV worker
# ----------------------------------------------------------------------

_DF = None
_FITS_DIR = None
_CLASSIC_PRIOR = None


def _init_worker(data_path, fits_dir):
    global _DF, _FITS_DIR
    _DF = data.read_cells(data_path)
    _FITS_DIR = fits_dir


def _classic_prior():
    """Pinned (t-bar, sigma_zm) from the FULL data / full-data zm_homo fit
    -- fixed constants of the classic spec, identical in every fold."""
    global _CLASSIC_PRIOR
    if _CLASSIC_PRIOR is None:
        tm = float(np.log2(
            _DF.groupby('task_id')['human_minutes'].first()).mean())
        _CLASSIC_PRIOR = (tm, em.zm_sigma(_FITS_DIR))
    return _CLASSIC_PRIOR


def _fit_for_cv(cfg, train_df, K, Kg, max_iter, accel):
    """Refit on a training fold.  Fold fits are scored unanchored, and
    the classic specs keep their full-data prior constants."""
    return fitting.fit_spec(
        cfg, em.index_cells(train_df), K=K, Kg=Kg, max_iter=max_iter,
        accel=accel, anchor=False,
        classic_prior=_classic_prior() if cfg.get('classic') else None)


def _cv_unit(job):
    (name, cfg, scheme, fold_i, test_task_ids, keep_models,
     K, Kg, max_iter, accel, save_fits_dir, refit) = job
    try:
        df = _DF
        in_test = df['task_id'].isin(test_task_ids)
        train_df = df[~in_test]
        test_df = df[in_test & df['alias'].isin(keep_models)]
        if refit:
            fit = _fit_for_cv(cfg, train_df, K, Kg, max_iter, accel)
        else:
            # in-sample stand-in: every fold is scored with the full-data
            # fit, so the held-out tasks were part of its training data
            fit = io.load_fit(Path(_FITS_DIR) / name)
        if save_fits_dir is not None and refit:
            io.save_fit(fit, Path(save_fits_dir) / scheme / name
                        / f'fold{fold_i}')
        idx_te = em.index_cells(test_df)
        assert list(idx_te['model_names']) == [
            m for m in fit['model_names'] if m in set(idx_te['model_names'])
        ], 'model ordering mismatch'
        # map test model indices onto the fit's model ordering
        pos_in_fit = {m: j for j, m in enumerate(fit['model_names'])}
        remap = np.array([pos_in_fit[m] for m in idx_te['model_names']],
                         dtype=np.int64)
        idx_scored = dict(idx_te)
        idx_scored['cell_midx'] = remap[idx_te['cell_midx']]
        # per-cell marginal p + joint loglik
        p = model_api.marginal_p_for_idx(fit, idx_scored)
        jll = model_api.joint_ll(fit, idx_scored,
                                 family_scheme=(scheme == 'family'))
        cell_task = np.repeat(np.arange(idx_te['n_tasks']),
                              np.diff(idx_te['cell_off']))
        out = pd.DataFrame({
            'scheme': scheme, 'fold': fold_i, 'spec': name,
            'task_id': idx_te['task_ids'][cell_task],
            'task_family': idx_te['task_family'][cell_task],
            'model': idx_te['model_names'][idx_te['cell_midx']],
            's': idx_te['cell_s'], 'n': idx_te['cell_n'], 'p': p,
        })
        return name, scheme, fold_i, 'ok', out, float(jll)
    except Exception:
        return name, scheme, fold_i, 'FAIL', traceback.format_exc(), np.nan


# ----------------------------------------------------------------------
# stages
# ----------------------------------------------------------------------

def stage_insample(names, fits_dir, out_dir, data_path):
    idx = data.load_cells(Path(data_path))
    cell_task = np.repeat(np.arange(idx['n_tasks']),
                          np.diff(idx['cell_off']))
    fam_sizes = family_sizes_for(idx['task_ids'][cell_task],
                                 idx['task_family'][cell_task])
    weights = {t: cell_weights(t, idx['cell_n'], fam_sizes) for t in TARGETS}
    rows, rel_rows = [], []
    for name in names:
        fit = io.load_fit(Path(fits_dir) / name)
        p = model_api.marginal_p_for_idx(fit, idx)
        jll = model_api.joint_ll(
            fit, idx, family_scheme=(fit.get('family', 'none') != 'none'))
        drift = (jll - fit['logl']) if fit['cls'] != 'basic' else 0.0
        for target in TARGETS:
            w = weights[target]
            logloss, brier = cell_metrics(p, idx['cell_s'], idx['cell_n'], w)
            ece, bins = ece_weighted(p, idx['cell_s'], idx['cell_n'], w)
            rows.append(dict(spec=name, scheme='insample', target=target,
                             joint_ll=jll, fit_logl=fit['logl'],
                             quad_drift=drift, marg_logscore=logloss,
                             brier=brier, ece=ece,
                             n_runs=int(idx['cell_n'].sum())))
            for k, conf, acc, wk in bins:
                rel_rows.append(dict(spec=name, target=target, bin=k,
                                     conf=conf, acc=acc, weight=wk))
            if target == 'equal_run':
                print(f'  insample {name:34s} jll={jll:12.2f} '
                      f'marg_logscore={logloss:.4f} brier={brier:.4f} '
                      f'ece={ece:.4f}', flush=True)
    pd.DataFrame(rows).to_csv(out_dir / 'insample_metrics.csv', index=False)
    pd.DataFrame(rel_rows).to_csv(out_dir / 'reliability_insample.csv',
                                  index=False)


def stage_cv(names, registry, out_dir, data_path, schemes, n_folds, seed,
             K, Kg, max_iter, njob, save_fits_dir=None, refit=True,
             fits_dir=None, accel='squarem'):
    """Score every (spec, scheme, fold) unit and write cv_preds_<scheme>.csv
    and cv_joint_ll.csv.  refit=False is the in-sample stand-in: no fold
    refits, every fold scored with the full-data fit from `fits_dir`."""
    df = data.read_cells(data_path)
    jobs = []
    for scheme in schemes:
        folds = make_folds(df, scheme, n_folds, seed)
        for fold_i, test_ids in enumerate(folds):
            train_df = df[~df['task_id'].isin(test_ids)]
            keep = valid_models(train_df)
            for name in names:
                jobs.append((name, registry[name], scheme, fold_i,
                             test_ids, keep, K, Kg, max_iter, accel,
                             None if save_fits_dir is None
                             else str(save_fits_dir), refit))
    print(f'CV: {len(jobs)} (spec x scheme x fold) units, njob={njob}')
    preds, jlls, fails = [], [], []
    t0 = time.perf_counter()
    done = 0
    with ProcessPoolExecutor(max_workers=njob, initializer=_init_worker,
                             initargs=(str(data_path),
                                       None if fits_dir is None
                                       else str(fits_dir))) as ex:
        for name, scheme, fold_i, status, payload, jll in ex.map(
                _cv_unit, jobs, chunksize=1):
            done += 1
            if status == 'ok':
                preds.append(payload)
                jlls.append(dict(spec=name, scheme=scheme, fold=fold_i,
                                 joint_ll=jll,
                                 n_cells=int(payload.shape[0])))
            else:
                fails.append((name, scheme, fold_i, payload))
                print(f'  [FAIL] {name} {scheme} fold{fold_i}\n{payload}',
                      flush=True)
            if done % 40 == 0:
                print(f'  ... {done}/{len(jobs)} units '
                      f'({time.perf_counter() - t0:.0f}s)', flush=True)
    all_preds = pd.concat(preds, ignore_index=True)
    for scheme in schemes:
        sub = all_preds[all_preds['scheme'] == scheme]
        wide = sub.pivot_table(
            index=['fold', 'task_id', 'task_family', 'model', 's', 'n'],
            columns='spec', values='p').reset_index()
        wide.to_csv(out_dir / f'cv_preds_{scheme}.csv', index=False)
    pd.DataFrame(jlls).to_csv(out_dir / 'cv_joint_ll.csv', index=False)
    if fails:
        print(f'{len(fails)} CV units FAILED')
    return len(fails) == 0


def stage_cv_aggregate(names, out_dir, schemes, ref='linear',
                       ci_center='fold'):
    """cv_metrics.csv + bayle_ci_<scheme>.csv + reliability files.

    Every metric and CI is computed under all four target distributions
    in TARGETS; `target` is a column in each output.  `ref` names the
    reference spec for the paired-contrast CI rows (kind = 'vs_<ref>');
    it must be among `names`.  `ci_center='global'` supports
    leave-one-cluster-out CV, where fold centring would degenerate.
    """
    jll = pd.read_csv(out_dir / 'cv_joint_ll.csv')
    met_rows, rel_rows = [], []
    for scheme in schemes:
        wide = pd.read_csv(out_dir / f'cv_preds_{scheme}.csv')
        cluster_col = 'task_id' if scheme == 'task' else 'task_family'
        s = wide['s'].to_numpy(float)
        n = wide['n'].to_numpy(float)
        # family size = #held-out tasks of that family (per fold, the
        # held-out set is all this scheme ever scores)
        fam_sizes = family_sizes_for(wide['task_id'].to_numpy(),
                                     wide['task_family'].to_numpy())
        weights = {t: cell_weights(t, n, fam_sizes) for t in TARGETS}
        # run-level expansion shared across specs (identical rows)
        reps = n.astype(int)
        folds_r = np.repeat(wide['fold'].to_numpy(), reps)
        clus_r = np.repeat(wide[cluster_col].to_numpy(), reps)
        w_r = {t: np.repeat(weights[t], reps) for t in TARGETS}
        losses = {}
        for name in names:
            p = wide[name].to_numpy(float)
            tot_jll = float(jll[(jll.spec == name)
                                & (jll.scheme == scheme)]['joint_ll'].sum())
            for target in TARGETS:
                w = weights[target]
                lg, br = cell_metrics(p, s, n, w)
                ece, bins = ece_weighted(p, s, n, w)
                met_rows.append(dict(spec=name, scheme=scheme,
                                     target=target, marg_logscore=lg,
                                     brier=br, ece=ece, joint_ll=tot_jll,
                                     n_runs=int(n.sum())))
                for k, conf, acc, wk in bins:
                    rel_rows.append(dict(spec=name, scheme=scheme,
                                         target=target, bin=k, conf=conf,
                                         acc=acc, weight=wk))
            losses[name] = run_level_losses(p, s, n)
        ci_rows = []
        for name in names:
            for metric, mi in [('marg_logscore', 0), ('brier', 1)]:
                for target in TARGETS:
                    wt = w_r[target]
                    row = dict(spec=name, scheme=scheme, target=target,
                               metric=metric, kind='absolute')
                    row.update(ci_tiers(losses[name][mi], folds_r, clus_r,
                                        wt, center=ci_center))
                    ci_rows.append(row)
                    if name != ref:
                        d = losses[name][mi] - losses[ref][mi]
                        row = dict(spec=name, scheme=scheme, target=target,
                                   metric=metric, kind=f'vs_{ref}')
                        row.update(ci_tiers(d, folds_r, clus_r, wt,
                                            center=ci_center))
                        ci_rows.append(row)
        pd.DataFrame(ci_rows).to_csv(out_dir / f'bayle_ci_{scheme}.csv',
                                     index=False)
    pd.DataFrame(met_rows).to_csv(out_dir / 'cv_metrics.csv', index=False)
    pd.DataFrame(rel_rows).to_csv(out_dir / 'reliability_cv.csv',
                                  index=False)
