#!/usr/bin/env python3
"""Step 4: the paper's scoring-rule suite -> artifacts/canonical/scores/.

  python pipeline/score.py [--check] [--out DIR]
  python pipeline/score.py --cv DIR --models name,name --out DIR

Reads the family-grouped CV predictions (artifacts/canonical/cv/
cv_preds_family.csv, from pipeline/evaluate.py) and compares Models 1
and 2 with the baseline, `linear_shslope_sqrtfamily`, on identical
held-out runs.  Intervals are cluster-robust (Bayle), clustered on the
held-out task family.

Scores (Section "Evaluation with proper scoring rules" of the paper):
marginal log score, Brier score, and the smoothed elementary scores at
q = 0.5 and 0.8 -- the elementary score averaged over q +/- 0.05.
Weightings ("Evaluation weights"): equal-task, equal-family and
sqrt-family.

Outputs:
  score_suite.csv    one row per reported number, keyed by `figure`:
                     scoregrid      paired difference (model - baseline)
                                    per model x weighting x score: est,
                                    se, 80/95% half-widths, per-fold means
                     murphy_points  the elementary-score difference at
                                    q = 0.5 and 0.8 (equal-family)
                     perai_gain     the scoregrid contrasts within each AI
  murphy_curves.csv  Murphy curves (mean elementary score on a 299-point
                     q grid, equal-family) of the baseline and both
                     models, with each model's difference from the
                     baseline and its 95% half-width

--check first verifies the per-run losses against the evaluate step's
stored Bayle CIs and cv_metrics.

--cv and --models score other specs (e.g. the ablation variants) with
the same suite: --cv reads the predictions of a separate evaluate run
(pipeline/evaluate.py --out DIR, which must include the baseline), and
--models replaces Models 1 and 2 by the listed specs.  Either one needs
--out, so the paper's scores are never overwritten.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

import _paths as P
from horizon_irt import data
from horizon_irt import murphy as MP
from horizon_irt.evaluate import run_level_losses

EVALS = P.cv_dir()

BASELINE = 'linear_shslope_sqrtfamily'
BASELINE_LABEL = 'baseline'
MODELS = [('shared_shslope_sqrtfamily', 'Model 1', 'm1'),
          ('s1_shslope_hetero_famspline_bb', 'Model 2', 'm2')]

SCHEME = 'family'
CLUSTER = 'task_family'

# The three main-text weightings (equal-run is appendix-only).
TARGETS3 = [('equal_task', 'equal-task'),
            ('equal_family', 'equal-family'),
            ('sqrt_family', 'sqrt-family')]
HEADLINE_TARGET = 'equal_family'      # for the Murphy curves

# The four scores of the suite.
SCORES = [('logscore', 'Marg. log score'),
          ('brier', 'Brier score'),
          ('elem50', r'$\tilde S_{q=0.5}$'),
          ('elem80', r'$\tilde S_{q=0.8}$')]
Q_OF = {'elem50': 0.5, 'elem80': 0.8}
Q_GRID = np.linspace(0.0, 1.0, 301)[1:-1]

# Half-width of the smoothing window for the elementary scores: the
# paper's smoothed score integrates S_{q+eps} over eps in [-0.05, 0.05].
# We report the uniform AVERAGE over that window (the integral divided by
# its 0.1 width) so magnitudes stay commensurate with S_q itself; the
# constant factor changes no comparison.
ELEM_HW = 0.05


def release_dates():
    """AI release dates, keyed by the aliases in CV predictions."""
    df = data.read_cells(P.cells_path()).drop_duplicates('alias')
    return dict(zip(df['alias'], pd.to_datetime(df['release_date'])))


def smoothed_elem_loss(p_cell, y_r, n, q0, hw=ELEM_HW):
    """Per-run smoothed elementary loss: the uniform average over
    q in [q0-hw, q0+hw] of murphy._run_loss's integrand
    (y=1: (1-q)1{p<q};  y=0: q 1{p>q}), in closed form.

    Ties p == q now sit under an integral of measure zero, so the
    no-ties caveat of the raw score disappears."""
    a, b = q0 - hw, q0 + hw
    w = b - a
    p_r = np.repeat(np.asarray(p_cell, float), n.astype(int))
    pc = np.clip(p_r, a, b)
    loss1 = ((1.0 - pc) ** 2 - (1.0 - b) ** 2) / (2.0 * w)   # y = 1
    loss0 = (pc ** 2 - a ** 2) / (2.0 * w)                   # y = 0
    return y_r * loss1 + (1.0 - y_r) * loss0


def run_losses(preds, spec, score):
    """Per-run loss array for one spec under one score, in the canonical
    cell-expansion order (s ones then n-s zeros -- run_level_losses and
    murphy._run_y lay runs out identically).  The elementary scores are
    the SMOOTHED ones (window average, see smoothed_elem_loss)."""
    p = preds[spec].to_numpy(float)
    s = preds['s'].to_numpy(float)
    n = preds['n'].to_numpy(float)
    if score in ('logscore', 'brier'):
        ll, br = run_level_losses(p, s, n)
        return ll if score == 'logscore' else br
    y_r = MP._run_y(preds)
    return smoothed_elem_loss(p, y_r, n, Q_OF[score])


def paired_cell(preds, spec, target, score):
    """One grid cell: est, se and per-fold means of the weighted paired
    difference (spec - BASELINE), cluster-robust over held-out families."""
    folds_r, clus_r, w_r = MP._run_expand(preds, target, CLUSTER)
    d_r = run_losses(preds, spec, score) - run_losses(preds, BASELINE, score)
    est, var = MP.contrast_var(d_r, folds_r, clus_r, w_r)
    df = pd.DataFrame({'wd': w_r * d_r, 'w': w_r, 'fold': folds_r})
    g = df.groupby('fold')[['wd', 'w']].sum()
    folds = (g['wd'] / g['w']).to_numpy()
    return est, math.sqrt(max(var, 0.0)), folds


def check(preds):
    """The pipeline must reproduce the precomputed numbers exactly.

    (1) vs-linear paired contrasts for logscore/brier match the stored
    Bayle CIs; (2) absolute weighted means match cv_metrics."""
    ci = pd.read_csv(EVALS / 'bayle_ci_family.csv')
    cv = pd.read_csv(EVALS / 'cv_metrics.csv')
    cv = cv[cv.scheme == SCHEME]
    worst = 0.0
    for spec, _, _ in MODELS:
        for target, _ in TARGETS3:
            folds_r, clus_r, w_r = MP._run_expand(preds, target, CLUSTER)
            for score, col in (('logscore', 'marg_logscore'),
                               ('brier', 'brier')):
                d_r = (run_losses(preds, spec, score)
                       - run_losses(preds, 'linear', score))
                est, var = MP.contrast_var(d_r, folds_r, clus_r, w_r)
                half = MP.Z[0.95] * math.sqrt(max(var, 0.0))
                r = ci[(ci.spec == spec) & (ci.target == target)
                       & (ci.metric == col) & (ci.kind == 'vs_linear')]
                r = r.iloc[0]
                worst = max(worst, abs(est - r.est),
                            abs((r.cluster_hi - r.cluster_lo) / 2 - half))
                # absolute weighted mean vs cv_metrics
                a = float(np.sum(w_r * run_losses(preds, spec, score))
                          / np.sum(w_r))
                m = cv[(cv.spec == spec) & (cv.target == target)].iloc[0]
                worst = max(worst, abs(a - float(m[col])))
    assert worst < 1e-9, f'check failed: max discrepancy {worst:.3e}'
    print(f'check passed: max discrepancy vs canonical CSVs {worst:.2e}')


def compute_perai_gains(preds):
    """Canonical four-score contrasts for every AI and weighting.

    Filtering before calling paired_cell deliberately recomputes the
    evaluation weights and family sizes within each AI.  The result is one
    row per (AI, model specification, weighting, score), with the same loss
    definitions used by the main score grid -- in particular, elem50 and
    elem80 are the smoothed band averages, not pointwise elementary scores.
    """
    dates = release_dates()
    aliases = list(preds['model'].unique())
    missing = sorted(set(aliases) - set(dates))
    assert not missing, f'aliases without release dates: {missing}'
    aliases.sort(key=dates.__getitem__)

    rows = []
    for alias in aliases:
        sub = preds[preds['model'] == alias].reset_index(drop=True)
        fold_ids = np.sort(sub['fold'].unique())
        for spec, name, _ in MODELS:
            for target, _ in TARGETS3:
                for score, _ in SCORES:
                    est, se, folds = paired_cell(sub, spec, target, score)
                    fold_full = dict.fromkeys(range(5), np.nan)
                    fold_full.update(dict(zip(fold_ids, folds)))
                    rows.append(dict(
                        figure='perai_gain', spec=spec, model=name,
                        ai=alias, release_date=dates[alias], target=target,
                        score=score, q=Q_OF.get(score, np.nan),
                        est=est, se=se, half80=MP.Z[0.80] * se,
                        half95=MP.Z[0.95] * se, n_rows=len(sub),
                        n_clusters=sub['task_family'].nunique(),
                        **{f'fold{i}': fold_full[i] for i in range(5)}))
    return pd.DataFrame(rows)


def verify_perai_gains(res, preds):
    """Structural and numerical checks for the appendix result table."""
    n_ai = preds['model'].nunique()
    expected_n = n_ai * len(MODELS) * len(TARGETS3) * len(SCORES)
    assert len(res) == expected_n, (len(res), expected_n)
    assert res['ai'].nunique() == n_ai
    assert set(res['spec']) == {m[0] for m in MODELS}
    assert set(res['target']) == {t[0] for t in TARGETS3}
    assert set(res['score']) == {s[0] for s in SCORES}
    assert np.isfinite(res[['est', 'se', 'half95']].to_numpy()).all()
    counts = res.groupby(['spec', 'ai', 'target', 'score']).size()
    assert (counts == 1).all()
    print(f'per-AI check passed: {expected_n} rows, {n_ai} AIs, '
          f'{len(MODELS)} models, 3 weightings, 4 scores')


def scoregrid_rows(preds):
    """Paired differences for every model x weighting x score."""
    rows = []
    for spec, name, _ in MODELS:
        for target, _ in TARGETS3:
            for score, _ in SCORES:
                est, se, folds = paired_cell(preds, spec, target, score)
                rows.append(dict(
                    figure='scoregrid', model=name, target=target,
                    score=score, q=Q_OF.get(score, np.nan), est=est, se=se,
                    half80=MP.Z[0.80] * se, half95=MP.Z[0.95] * se,
                    **{f'fold{i}': v for i, v in enumerate(folds)}))
    return rows


def murphy(preds):
    """(rows, curves): pointwise CIs at q = 0.5, 0.8 and the full curves."""
    rows, curves = [], []
    curve_b = MP.murphy_curve(preds, BASELINE, HEADLINE_TARGET, Q_GRID)
    curves.append(pd.DataFrame({
        'spec': BASELINE, 'model': BASELINE_LABEL,
        'target': HEADLINE_TARGET, 'q': Q_GRID, 'mean_score': curve_b,
        'diff_est': np.nan, 'diff_half95': np.nan}))
    for spec, name, _ in MODELS:
        curve_k = MP.murphy_curve(preds, spec, HEADLINE_TARGET, Q_GRID)
        est, half = MP.diff_ci_curve(preds, spec, HEADLINE_TARGET, CLUSTER,
                                     Q_GRID, level=0.95, ref=BASELINE)
        pts = MP.diff_ci_points(preds, spec, HEADLINE_TARGET, CLUSTER,
                                MP.Q_MARKS, (0.80, 0.95), ref=BASELINE)
        for q, rec in pts.items():
            rows.append(dict(
                figure='murphy_points', model=name,
                target=HEADLINE_TARGET, score='elementary', q=q,
                est=rec['est'], se=math.sqrt(max(rec['var'], 0.0)),
                half80=rec[0.80], half95=rec[0.95]))
        curves.append(pd.DataFrame({
            'spec': spec, 'model': name, 'target': HEADLINE_TARGET,
            'q': Q_GRID, 'mean_score': curve_k, 'diff_est': est,
            'diff_half95': half}))
    return rows, pd.concat(curves, ignore_index=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--check', action='store_true',
                    help='verify against bayle_ci_family/cv_metrics first')
    ap.add_argument('--out', type=Path, default=None,
                    help='output dir (default artifacts/canonical/scores)')
    ap.add_argument('--cv', type=Path, default=None,
                    help='CV predictions dir (default artifacts/canonical/cv)')
    ap.add_argument('--models', default=None,
                    help='comma-separated specs to score instead of '
                         'Models 1 and 2')
    args = ap.parse_args()
    if (args.cv is not None or args.models) and args.out is None:
        raise SystemExit('--cv and --models need --out, so the paper\'s '
                         'scores are not overwritten')
    global EVALS
    if args.cv is not None:
        EVALS = args.cv
    if args.models:
        MODELS[:] = [(s.strip(), s.strip(), s.strip())
                     for s in args.models.split(',')]
    out = P.scores_dir() if args.out is None else args.out
    out.mkdir(parents=True, exist_ok=True)

    preds = MP.load_preds(EVALS, SCHEME)
    missing = [s for s in [BASELINE] + [m[0] for m in MODELS]
               if s not in preds.columns]
    assert not missing, f'CV predictions lack specs {missing}'
    print(f'scoring {P.DATASET} ({P.cv_mode()}) -> {out}', flush=True)
    if args.check:
        check(preds)

    rows = scoregrid_rows(preds)
    murphy_rows, curves = murphy(preds)
    rows += murphy_rows
    perai = compute_perai_gains(preds)
    verify_perai_gains(perai, preds)
    rows.extend(perai.to_dict('records'))

    pd.DataFrame(rows).to_csv(out / 'score_suite.csv', index=False)
    curves.to_csv(out / 'murphy_curves.csv', index=False)
    grid = [r for r in rows if r['figure'] == 'scoregrid']
    sig = sum(1 for r in grid if abs(r['est']) > r['half95'])
    print(f'wrote {out / "score_suite.csv"} and murphy_curves.csv; '
          f'scoregrid cells with 95% interval excluding zero: '
          f'{sig} of {len(grid)}')


if __name__ == '__main__':
    main()
