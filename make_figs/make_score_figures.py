#!/usr/bin/env python3
"""Figures of the scoring-rule suite, drawn from artifacts/canonical/scores/.

  score_grid               paper Figure 3: paired differences (model -
                           baseline) for Models 1 and 2; rows are the
                           equal-task, equal-family and sqrt-family
                           weightings, columns the marginal log score,
                           Brier score and smoothed elementary scores at
                           q = 0.5 and 0.8, with nested 80/95% intervals
                           and the five per-fold differences as pale dots
  murphy_diagrams          paper Figure 4: Murphy curves of the baseline
                           and both models (left) and each model's
                           difference with its 95% band (right)
  per_ai_contrasts_model1  paper Figures 8 and 9: the score_grid contrasts
  per_ai_contrasts_model2  within each AI, by release date, with the six
                           2-15 min band AIs highlighted

Every number comes from pipeline/score.py; this script only draws.
Murphy curves are smoothed for display (centred rolling mean, window 9
of the 299-point q grid).  When the CV outputs are the in-sample
stand-in (cv/provenance.json), each figure is watermarked as such.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'pipeline'))
import _paths as P                                           # noqa: E402
import score as SC                                           # noqa: E402
from horizon_irt import murphy as MP                         # noqa: E402
import horizon_panels as hp                                  # noqa: E402

FIGS = P.FIGS
SCORES_DIR = P.scores_dir()
STANDIN = P.cv_mode() != 'cv'
# (spec, name, colour, tag) for Models 1 and 2
MODELS = [(spec, name, hp.MODEL_COLORS[spec], tag)
          for spec, name, tag in SC.MODELS]

# 80% thick, 95% thin -- two levels only (no 90%).
LEVELS = [(0.95, 0.8), (0.80, 3.0)]
SMOOTH = 9
SCL = 1e3                              # difference axes drawn in 1e-3 units


def save_cv_figure(fig, stem, outdir=None):
    """hp.save for a figure of held-out scores, watermarked if the CV
    outputs are the in-sample stand-in."""
    if STANDIN:
        hp.watermark(fig)
    hp.save(fig, FIGS if outdir is None else outdir, stem)


def fig_scoregrid(suite):
    """The combined 3x4 grid: both models per panel, horizontal intervals.

    Rows = weightings, columns = scores; within a panel Model 1 sits on
    the upper line and Model 2 on the lower, in their own colours, each
    with nested 80/95 intervals and its five per-fold differences as
    pale dots.  Columns share the difference (x) axis, so only the
    bottom row carries tick labels and the whole grid stays short."""
    from scipy.stats import norm
    z95 = norm.ppf(0.975)
    g = suite[suite['figure'] == 'scoregrid'].set_index(
        ['model', 'target', 'score'])
    fold_cols = [c for c in suite.columns if c.startswith('fold')]
    cells = {}                        # (ti, si, mi) -> (est, se, folds), x1e3
    for mi, (spec, name, color, _) in enumerate(MODELS):
        for ti, (target, _) in enumerate(SC.TARGETS3):
            for si, (score, _) in enumerate(SC.SCORES):
                r = g.loc[(name, target, score)]
                folds = r[fold_cols].to_numpy(float)
                folds = folds[~np.isnan(folds)]
                cells[ti, si, mi] = (r['est'] * SCL, r['se'] * SCL,
                                     folds * SCL)

    Y = {0: 0.72, 1: 0.28}            # Model 1 upper line, Model 2 lower
    fig, axes = plt.subplots(3, 4, figsize=(5.5, 2.0), sharex='col',
                             sharey=True)
    for si in range(4):
        vals = [v for ti in range(3) for mi in range(2) for v in
                (cells[ti, si, mi][0] - z95 * cells[ti, si, mi][1],
                 cells[ti, si, mi][0] + z95 * cells[ti, si, mi][1],
                 *cells[ti, si, mi][2])]
        lo, hi = min(vals + [0.0]), max(vals + [0.0])
        pad = 0.10 * (hi - lo or 1.0)
        axes[0, si].set_xlim(lo - pad, hi + pad)
        for ti in range(3):
            ax = axes[ti, si]
            ax.axvline(0.0, color='0.55', linewidth=0.8, zorder=2)
            for mi, (_, _, color, _) in enumerate(MODELS):
                est, se, folds = cells[ti, si, mi]
                y = Y[mi]
                # a sorted vertical fan: near-coincident fold values sit
                # at adjacent heights instead of overprinting (several
                # cells have two folds within a marker width -- with a
                # flat jitter the five dots read as four)
                offs = np.linspace(-0.16, 0.16, len(folds))
                ax.scatter(np.sort(folds), y + offs,
                           s=5, color=color, alpha=0.4, linewidth=0,
                           zorder=3)
                for level, lw in LEVELS:
                    z = norm.ppf(0.5 + level / 2.0)
                    ax.plot([est - z * se, est + z * se], [y, y],
                            color=color, linewidth=lw,
                            solid_capstyle='butt', zorder=4)
                ax.scatter([est], [y], s=11, color=color, zorder=5,
                           edgecolor='white', linewidth=0.4)
            ax.set_ylim(0.0, 1.0)
            ax.set_yticks([])
            hp.style(ax, fs=5.4)
            ax.grid(False, axis='y')
            if ti == 0:
                ax.set_title(SC.SCORES[si][1], fontsize=7)
            if si == 0:
                # horizontal row labels: rotated ones are taller than
                # these squeezed rows and collide with one another
                ax.set_ylabel(SC.TARGETS3[ti][1], fontsize=6, rotation=0,
                              ha='right', va='center', labelpad=3)
    fig.supxlabel(r'difference from baseline  ($\times 10^{-3}$, '
                  'left of 0 favours the model)', fontsize=6.5, y=0.065)
    handles = [plt.Line2D([], [], color=c, linewidth=2.2, label=n)
               for _, n, c, _ in MODELS]
    fig.legend(handles=handles, fontsize=6, frameon=False, ncol=2,
               loc='lower right', bbox_to_anchor=(0.995, 0.0),
               handlelength=1.2, handletextpad=0.4, columnspacing=1.0)
    fig.tight_layout(pad=0.35, w_pad=0.7, h_pad=0.45,
                     rect=(0, 0.075, 1, 1))
    save_cv_figure(fig, 'score_grid')


def fig_murphy(curves):
    """The combined 1x2 Murphy figure: overlay left, differences right."""
    sm = lambda c: MP.smooth_curve(c, SMOOTH)               # noqa: E731
    base = curves[curves['spec'] == SC.BASELINE]
    q_grid = base['q'].to_numpy()

    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.1))
    ax = axes[0]
    ax.plot(q_grid, sm(base['mean_score'].to_numpy()), color=hp.C_BASE,
            linewidth=1.1, label='baseline')
    for spec, name, color, _ in MODELS:
        k = curves[curves['spec'] == spec]
        ax.plot(q_grid, sm(k['mean_score'].to_numpy()), color=color,
                linewidth=1.1, label=name)
    for q in MP.Q_MARKS:
        ax.axvline(q, color='0.75', linewidth=0.6, linestyle=':', zorder=1)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel('$q$', fontsize=7)
    ax.set_ylabel(r'mean elementary score $\bar S(q)$', fontsize=6.5)
    ax.legend(fontsize=5.6, frameon=False, loc='upper right',
              handlelength=1.3, labelspacing=0.3)
    hp.style(ax)

    ax = axes[1]
    ax.axhline(0.0, color='0.45', linewidth=0.8, zorder=2)
    for q in MP.Q_MARKS:
        ax.axvline(q, color='0.8', linewidth=0.5, linestyle=':', zorder=1)
    for spec, name, color, _ in MODELS:
        k = curves[curves['spec'] == spec]
        d = k['diff_est'].to_numpy() * SCL
        h = k['diff_half95'].to_numpy() * SCL
        ax.fill_between(q_grid, sm(d - h), sm(d + h), color=color,
                        alpha=0.14, lw=0, zorder=2)
        ax.plot(q_grid, sm(d), color=color, linewidth=1.0, zorder=3)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel('$q$', fontsize=7)
    ax.set_ylabel('model $-$ baseline  ($\\times 10^{-3}$)', fontsize=6.5)
    hp.style(ax)
    fig.tight_layout(pad=0.4, w_pad=1.2)
    save_cv_figure(fig, 'murphy_diagrams')


def perai_gain_limits(res):
    """Shared 95%-inclusive y limits by score, used by both figures."""
    out = {}
    for score, _ in SC.SCORES:
        cell = res[res['score'] == score]
        lo = min(0.0, float((cell['est'] - cell['half95']).min())) * SCL
        hi = max(0.0, float((cell['est'] + cell['half95']).max())) * SCL
        pad = 0.06 * (hi - lo or 1.0)
        out[score] = (lo - pad, hi + pad)
    return out


def fig_perai_gain(res, spec, name, color, tag, limits, outdir=None,
                   stem=None):
    """One paper-width 4x3 per-AI gain grid for a headline model."""
    sub = res[res['spec'] == spec]
    fig, axes = plt.subplots(4, 3, figsize=(5.5, 5.0), sharex=True,
                             sharey='row')
    band_set = set(hp.BAND_AIS)
    date_lo = sub['release_date'].min() - pd.Timedelta(days=120)
    date_hi = sub['release_date'].max() + pd.Timedelta(days=120)

    for si, (score, score_label) in enumerate(SC.SCORES):
        for ti, (target, target_label) in enumerate(SC.TARGETS3):
            ax = axes[si, ti]
            cell = sub[(sub['target'] == target)
                       & (sub['score'] == score)].sort_values('release_date')
            in_band = cell['ai'].isin(band_set).to_numpy()
            for mask, mark_color, alpha, zorder, ms in (
                    (~in_band, '0.62', 0.72, 3, 2.1),
                    (in_band, color, 1.0, 4, 2.7)):
                if not np.any(mask):
                    continue
                part = cell.loc[mask]
                ax.errorbar(
                    part['release_date'], part['est'] * SCL,
                    yerr=part['half95'] * SCL, fmt='o',
                    color=mark_color, ecolor=mark_color, alpha=alpha,
                    markersize=ms, markeredgecolor='white',
                    markeredgewidth=0.3, elinewidth=0.55, capsize=1.4,
                    capthick=0.55, linestyle='none', zorder=zorder)
            ax.axhline(0.0, color='0.48', linewidth=0.7, zorder=2)
            ax.set_xlim(date_lo, date_hi)
            ax.set_ylim(*limits[score])
            ax.grid(True, axis='y', alpha=0.22, linewidth=0.45)
            ax.set_axisbelow(True)
            ax.tick_params(labelsize=5.5, length=2.2, pad=1.3)
            for side in ('top', 'right'):
                ax.spines[side].set_visible(False)
            if si == 0:
                ax.set_title(target_label, fontsize=7, pad=3)
            if ti == 0:
                ax.set_ylabel(f'{score_label}\n$\\times 10^{{-3}}$',
                              fontsize=6.2)

    for ax in axes[-1]:
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    handles = [
        plt.Line2D([], [], marker='o', linestyle='none', color=color,
                   markersize=3.1, label='six AIs with 2–15 min horizons'),
        plt.Line2D([], [], marker='o', linestyle='none', color='0.62',
                   markersize=2.7, label='other AIs')]
    fig.supxlabel('AI release date', fontsize=6.5, y=0.047)
    fig.legend(handles=handles, fontsize=5.8, frameon=False, ncol=2,
               loc='lower center', bbox_to_anchor=(0.5, 0.002),
               handletextpad=0.4, columnspacing=1.2)
    fig.tight_layout(pad=0.4, w_pad=0.8, h_pad=0.8,
                     rect=(0, 0.065, 1, 1))
    if stem is None:
        stem = f'perai_gain_{tag}'
    save_cv_figure(fig, stem, outdir)


def main():
    suite = pd.read_csv(SCORES_DIR / 'score_suite.csv')
    curves = pd.read_csv(SCORES_DIR / 'murphy_curves.csv')
    fig_scoregrid(suite)
    fig_murphy(curves)

    res = suite[suite['figure'] == 'perai_gain'].copy()
    res['release_date'] = pd.to_datetime(res['release_date'])
    missing = [a for a in hp.BAND_AIS if a not in set(res['ai'])]
    assert not missing, f'band AIs missing from the scores: {missing}'
    limits = perai_gain_limits(res)
    for spec, name, color, tag in MODELS:
        stem = 'per_ai_contrasts_' + name.lower().replace(' ', '')
        fig_perai_gain(res, spec, name, color, tag, limits, stem=stem)


if __name__ == '__main__':
    main()
