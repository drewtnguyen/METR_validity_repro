#!/usr/bin/env python3
"""The headline figure (paper Figure 1): METR's fit and Model 2, 3 x 2.

Rows: the METR baseline (blue) | diagnostics comparing the two fits |
Model 2 (purple).  Rows 1 and 3 hold the same pair of panels, time
horizons and conditional success, so the two fits read against each
other down the page.

Model 2 is `s1_shslope_hetero_famspline_bb`.  The "METR baseline" is
`linear_shslope_sqrtfamily`, the paper's baseline: METR's sqrt-family
weighted logistic regression with one slope shared across AIs.  It
scores better than METR's per-AI fits (`linear_sqrtfamily`) and its
50% horizons are close to theirs, so it stands in for them here; the
script prints how close.  Everything reads already-fitted models from
artifacts/canonical/fits/ -- nothing is refit here.

  (a) The baseline's 50% time horizons by release date, frontier AIs
      only (those beating every earlier AI).
  (b) The baseline's p_j(t*) at a ladder of fixed human times t*,
      against release date on a logit axis, in shades of METR's blue,
      for the same AIs.  With one shared slope the curves are parallel,
      evenly spaced copies of each other: the log-linear assumption
      made visible.
  (c) GPT-4o's in-sample p_j(t) under the fits of (a) (blue) and (e)
      (purple), against observed frequencies in factor-4 human-time
      bins (black, 95% Wald bars), with the 50% line.  Dashed drop
      lines in each fit's colour mark where its curve crosses 50%,
      GPT-4o's horizon under that fit, labelled with its value; grey
      dashed lines in (a) and (e) mark the same values on the horizon
      axis.  Data and binning come from
      make_success_curves.load_insample and binned_freq, so the
      construction matches the fit-results success-curve figures.
  (d) Model 2's time-to-difficulty conversion (as in
      make_difficulty_vs_length.py): each task's posterior difficulty
      theta_hat (light blue), the prior mean f(t) with its spread
      f +/- s (purple), and the baseline's assumption that difficulty is
      linear in log human time (blue line: the identity of the
      least-squares alignment in model2.py, i.e. the straight line
      closest to f).  The latent scale is defined only up to an affine
      map -- under any such relabelling of theta the fit is unchanged,
      and so is every horizon -- so the axis shows theta_hat standardized
      over the tasks (mean 0, SD 1).  Two scale-free references: a
      dashed line through the mean task at the SNR = 1 slope (a straight
      line whose explained spread equals its residual spread rises
      1/sqrt 2 SDs of theta_hat per SD of log time), and a short bar
      spanning one SD of log human time.
  (e) Model 2's 50% time horizons, frontier AIs only.  These are
      MARGINAL horizons: a task of human time t has an uncertain
      difficulty, so p_j(t) integrates theta over the fitted prior
      N(f(t), sigma(t)^2 + tau(t)^2) before inverting.  fit_all.py
      already computed these into params.csv.
  (f) As (b), for Model 2's p_j(t*), in shades of purple.

(a) and (e) share their axes, as do (b) and (f).  GPT-4o, the AI of
(c), is ringed in (a) and (e).  The flat stretch of the conversion in
(d), BAND_MIN, is shaded in (a), (d) and (e).

The difficulty horizon D_j = -alpha_j / beta is no longer plotted, but
is still written to the CSV; computing it uses the least-squares
identity alignment in model2.py, and model2.check_invariance asserts
the marginal horizons survive that realignment untouched.

Paper Figure 1 (label fig:headline).  Outputs: figs/headline.pdf,
.png, headline.csv and headline_difficulty.csv (panel (d)'s
standardized scale).  headline.csv keeps METR's own per-AI horizons
(`linear_sqrtfamily`) beside Model 2's, since the text quotes them; the
baseline
horizons that panel (a) plots are in its fit's params.csv.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import model2                                                # noqa: E402
import horizon_panels as hp                                  # noqa: E402
import make_success_curves as sc                            # noqa: E402
import make_difficulty_vs_length as dvl                     # noqa: E402
from horizon_irt import io, model_api                        # noqa: E402

FIGS = model2.P.FIGS

# The flat stretch of the time-to-difficulty conversion: where Model 2's
# f rises more slowly than the baseline's straight line, i.e. where its
# local slope on the aligned scale is below 1.  On the canonical fit
# that runs from 2.8 to 28.5 min (flattest, slope 0.31, at 10 min).
# The shaded window is drawn a little wider, 2-30 min, so its lower edge
# takes in the 2-minute bin where the flattening begins.  Shaded in the
# horizon panels and, along the human-time axis, in (d).
BAND_MIN = (2.0, 30.0)
BAND_LABEL = f'{BAND_MIN[0]:g}–{BAND_MIN[1]:g} min'

# Two AIs the caption quotes horizons for.  No longer ringed on the plot,
# but kept so the numbers behind that sentence stay one run away.
HIGHLIGHT = ['Claude 3 Opus (Inspect)', 'Claude 3.5 Sonnet (Old) (Inspect)']

# The AI singled out in panel (c), and ringed in (a) and (e).
PFIT_AI = 'GPT-4o (Inspect)'
# The fit drawn in METR's colour in (a), (b) and (c): the shared-slope
# baseline.  METR's own per-AI fit is still read, for the CSV and the
# printed comparison.
METR_SPEC = 'linear_shslope_sqrtfamily'
METR_OWN = 'linear_sqrtfamily'

# Fixed task lengths for (b) and (f), in minutes: the ladder of the
# difficulty figure's success-by-length panel.
T_STARS = dvl.T_STARS

# Panel (d)'s y-axis is theta_hat standardized over the tasks (sample
# SD); the limits are set from the points, the band and the lines, so
# every task lies inside the range.

# Panel (d)'s task dots: a light tint of the baseline's blue, so they
# stay behind both fitted lines.
C_TASKS = '#88aadd'

# Task-length ticks for (c) and (d), in log2 minutes: one per natural
# unit, which spaces them about evenly (factors of 60, 60, 24) and keeps
# the labels apart at a third of the text width.
LENGTH_TICKS = [(np.log2(1 / 60), '1 sec'), (0.0, '1 min'),
                (np.log2(60), '1 hr'), (np.log2(1440), '1 day')]


def finish(ax, title, fs=6.5):
    """Title, light grid and open top/right: the small-multiple style."""
    ax.set_title(title, fontsize=7.5, pad=3.5)
    hp.style(ax, fs=fs)


def length_axis(ax):
    ax.set_xticks([t for t, _ in LENGTH_TICKS])
    ax.set_xticklabels([l for _, l in LENGTH_TICKS])


def fmt_horizon(minutes):
    """GPT-4o's horizons are a few minutes, where hp.fmt_minutes'
    whole-minute rounding would hide the gap between the two fits, so
    keep one decimal under ten minutes."""
    return f'{minutes:.1f} min' if minutes < 10 else hp.fmt_minutes(minutes)


def ring_ai(ax, df, col, alias, label):
    """Ring one AI's point in black and name it: the cue that ties a
    horizon panel to panel (c)."""
    row = df[df['model'] == alias].iloc[0]
    ax.scatter([row['release_date']], [row[col]], s=30, marker='o',
               facecolor='none', edgecolor='0.1', linewidth=0.7, zorder=6)
    ax.annotate(label, (row['release_date'], row[col]),
                textcoords='offset points', xytext=(4.5, -7.5),
                ha='left', fontsize=5.6, color='0.1', zorder=6)


def mark_horizon(ax, df, alias):
    """Grey dashed line from the y-axis to one AI's horizon, labelled
    with its value: the value that panel (c)'s drop line marks on its
    x-axis.  Call once the x-limits are final, since the line starts at
    the left edge."""
    row = df[df['model'] == alias].iloc[0]
    left = ax.get_xlim()[0]
    ax.hlines(row['th50_min'], left, row['release_date'], colors='0.45',
              linestyles='--', linewidth=0.8, zorder=1)
    # Value just above the line's left end, clear of the data, which
    # sits far below a few minutes there.
    ax.annotate(fmt_horizon(row['th50_min']), (left, row['th50_min']),
                textcoords='offset points', xytext=(3.0, 1.5),
                ha='left', va='bottom', fontsize=5.6, color='0.3',
                zorder=6)
    ax.set_xlim(left=left)


def horizon_panel(ax, df, sota, color, title):
    """(a) or (e): 50% horizons by release date, GPT-4o ringed."""
    hp.panel(ax, df, 'th50_min', sota, color, title, band=BAND_MIN,
             annotate=False)
    ring_ai(ax, df, 'th50_min', PFIT_AI, 'GPT-4o')
    ax.set_ylabel('50% horizon (log scale)', fontsize=7, labelpad=1.5)
    ax.set_xlabel('Release date', fontsize=7)


def panel_pfit(ax, th50):
    """(c): GPT-4o's in-sample p_j(t) under both fits, observed binned
    frequencies, and the 50% line the horizons are read off.  `th50`
    maps each spec to GPT-4o's 50% horizon in minutes, from its
    params.csv: a dashed drop line in the fit's colour marks it."""
    # load_insample always includes the baseline, i.e. METR_SPEC
    cells, xg, curves, _, edges = sc.load_insample()
    ax.axhline(0.5, color='0.5', lw=0.7, ls='--', alpha=0.7, zorder=1)
    ax.plot(xg, curves[METR_SPEC][PFIT_AI], color=hp.C_METR,
            linewidth=1.0, zorder=3, label='METR baseline')
    ax.plot(xg, curves[model2.SPEC][PFIT_AI], color=hp.C_M2,
            linewidth=1.0, zorder=3, label='Model 2')
    # Drop lines from the 50% crossings to the axis.  They are drawn at
    # params.csv's horizons, the numbers (a) and (e) plot; the printed
    # check confirms the curves cross 50% there.
    for spec, color in ((METR_SPEC, hp.C_METR), (model2.SPEC, hp.C_M2)):
        x = np.log2(th50[spec])
        p_at = np.interp(x, xg, curves[spec][PFIT_AI])
        print(f'(c) {spec}: GPT-4o th50 {th50[spec]:.2f} min, '
              f'curve there {p_at:.4f}')
        ax.vlines(x, -0.03, 0.5, colors=color, linestyles='--',
                  linewidth=0.8, alpha=0.9, zorder=2)
        # Values run up the lines' outer sides (Model 2's left, the
        # baseline's right): the two lines are too close for side-by-side
        # labels, and the strip below the curves is empty.
        left = th50[spec] < max(th50.values())
        ax.annotate(fmt_horizon(th50[spec]), (x, 0.03),
                    textcoords='offset points',
                    xytext=(-1.5 if left else 1.5, 0), rotation=90,
                    ha='right' if left else 'left', va='bottom',
                    fontsize=5.6, color=color, zorder=6)
    sub = cells[cells['model'] == PFIT_AI]
    for k, (xbar, ph, lo, hi, _) in enumerate(sc.binned_freq(sub, edges)):
        ax.errorbar([xbar], [ph], yerr=[[max(ph - lo, 0.0)],
                                        [max(hi - ph, 0.0)]], fmt='o',
                    ms=2.0, color='0.15', elinewidth=0.7, capsize=0,
                    zorder=4, label='observed' if k == 0 else None)
    ax.set_xlim(xg[0] - 0.3, xg[-1] + 0.3)
    ax.set_ylim(-0.03, 1.03)
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_yticklabels(['0', '.5', '1'])
    length_axis(ax)
    ax.set_ylabel('P(success)', fontsize=7, labelpad=1.5)
    ax.set_xlabel('Human time (log scale)', fontsize=7)
    finish(ax, '(c) GPT-4o: fit comparison')
    ax.legend(fontsize=5.6, loc='lower left', frameon=False,
              handlelength=1.4, handletextpad=0.5, borderpad=0.1,
              labelspacing=0.28)


def standardized_scale(dc):
    """(to_y, mean, sd): the affine map from the aligned log2-minutes
    scale to panel (d)'s, which standardizes theta_hat over the tasks
    (sample SD).  Any affine relabelling of theta leaves the fit and
    every horizon unchanged, so this is a choice of units only."""
    th = dc['theta_hat']
    mean, sd = float(th.mean()), float(th.std(ddof=1))
    return (lambda v: (np.asarray(v, dtype=float) - mean) / sd), mean, sd


def snr_line(ax, dc, to_y, lo, hi):
    """Dashed reference through the mean task at the slope where a
    straight line's signal equals its residual noise, SNR 1: it rises
    1/sqrt(2) SDs of theta_hat per SD of log time, a correlation of
    0.71.  Steeper lines carry more signal than noise.  Defined from the
    SDs of the plotted theta_hat and of log time, so any affine choice
    of latent scale draws the same line."""
    t, th = dc['t'], dc['theta_hat']
    k = th.std(ddof=1) / t.std(ddof=1) / np.sqrt(2.0)
    xs = np.array([lo, hi])
    line, = ax.plot(xs, to_y(th.mean() + k * (xs - t.mean())), '--',
                    color='0.3', lw=0.8, dashes=(3.5, 2), zorder=5)
    return line


def sd_bar(ax, dc, hi):
    """Scale bar in the empty lower right: one SD of log human time
    across the tasks, labelled in doublings."""
    t_sd = dc['t'].std(ddof=1)
    x1 = hi - 0.2
    y_lo, y_hi = ax.get_ylim()
    y0 = y_lo + 0.08 * (y_hi - y_lo)
    cap = 0.02 * (y_hi - y_lo)
    ax.plot([x1 - t_sd, x1], [y0, y0], color='k', lw=0.8, zorder=6)
    for x in (x1 - t_sd, x1):
        ax.plot([x, x], [y0 - cap, y0 + cap], color='k', lw=0.8, zorder=6)
    ax.text(x1 - 0.5 * t_sd, y0 + 1.5 * cap,
            f'1 SD of log time ({t_sd:.1f} doublings)', fontsize=4.8,
            ha='center', va='bottom', zorder=6)


def panel_difficulty(ax, dc, to_y):
    """(d): Model 2's difficulty vs length, theta_hat standardized over
    the tasks, with METR's linear-in-log-length assumption for
    comparison."""
    t, xg, f, s = dc['t'], dc['xg'], dc['f'], dc['s']
    lo, hi = float(t.min()), float(t.max())
    ax.axvspan(*np.log2(BAND_MIN), color=hp.C_BAND, alpha=0.13, lw=0,
               zorder=0)
    metr, = ax.plot([lo, hi], to_y([lo, hi]), '-', color=hp.C_METR,
                    lw=1.1, zorder=2)
    band = ax.fill_between(xg, to_y(f - s), to_y(f + s), color=hp.C_M2,
                           alpha=0.15, lw=0, zorder=1)
    curve, = ax.plot(xg, to_y(f), color=hp.C_M2, lw=1.4, zorder=4)
    dots = ax.scatter(t, to_y(dc['theta_hat']), s=5, color=C_TASKS,
                      alpha=0.75, linewidth=0, zorder=3)
    ref = snr_line(ax, dc, to_y, lo, hi)
    # Limits from everything drawn, so no task is cut off; ticks every
    # SD, minor ticks every half.
    plotted = np.concatenate([to_y(f - s), to_y(f + s), to_y([lo, hi]),
                              to_y(dc['theta_hat'])])
    y_lo, y_hi = plotted.min() - 0.08, plotted.max() + 0.08
    ax.set_ylim(y_lo, y_hi)
    ax.set_yticks(np.arange(np.ceil(y_lo), y_hi + 1e-9))
    ax.set_yticks(np.arange(np.ceil(2 * y_lo) / 2, y_hi, 0.5), minor=True)
    ax.set_xlim(lo - 0.4, hi + 0.4)
    length_axis(ax)
    ax.set_xlabel('Human time (log scale)', fontsize=7)
    ax.set_ylabel('AI difficulty (SDs)', fontsize=7, labelpad=1.5)
    finish(ax, '(d) Time-to-difficulty conversion')
    sd_bar(ax, dc, hi)
    # Four short entries fit the empty upper left, above the short
    # tasks; the band's colour is keyed in (a).  Model 2's curve and
    # band share one entry, drawn as the line over its band.
    ax.legend([(band, curve), metr, dots, ref],
              [r'Model 2: $f(t) \pm s(t)$', 'METR baseline: linear',
               r'task $\hat\theta_i$', 'SNR = 1'],
              handler_map={tuple: HandlerTuple(ndivide=None, pad=0)},
              fontsize=5.0, loc='upper left', frameon=False,
              handlelength=1.4, handletextpad=0.45, borderpad=0.2,
              labelspacing=0.3)


def by_date(params, fit, P, keep):
    """Release dates in ascending order and P's columns to match, for
    the AIs in `keep` only (the fit's frontier).  marginal_p returns
    columns in fit order, which is the alphabetical params.csv order, so
    one argsort reorders both alike."""
    assert list(params['model']) == list(fit['model_names'])
    order = np.argsort(params['release_date'].to_numpy(), kind='stable')
    order = order[params['model'].iloc[order].isin(keep).to_numpy()]
    return params['release_date'].iloc[order].reset_index(drop=True), \
        P[:, order]


def main():
    fit, idx = model2.load()
    fit_metr = io.load_fit(model2.FITS / METR_SPEC)
    t = idx['t_array']

    m2 = pd.read_csv(model2.FITS / model2.SPEC / 'params.csv')
    lin = pd.read_csv(model2.FITS / METR_SPEC / 'params.csv')
    own = pd.read_csv(model2.FITS / METR_OWN / 'params.csv')

    c, d = model2.ols_identity_align(fit, t)
    aligned = model2.rescale_latent(fit, c, d)
    rel = model2.check_invariance(aligned, m2, fit['model_names'])
    print(f'latent realignment: theta -> (theta - {d:.4f}) / {c:.4f}  '
          f'(th50 invariant to {rel:.1e} rel.)')

    # Difficulty horizon: alpha_j + beta D = 0 on the aligned scale.
    # params.csv rows are alphabetical, the fit arrays follow model_names.
    pos = {m: k for k, m in enumerate(fit['model_names'])}
    a_j = np.array([aligned['params'][pos[m], 0] for m in m2['model']])
    b_j = np.array([aligned['params'][pos[m], 1] for m in m2['model']])
    m2['dh50_min'] = 2.0 ** (-a_j / b_j)

    for df in (lin, m2):
        df['release_date'] = pd.to_datetime(df['release_date'])
    hp.fit_ylim(lin['th50_min'], m2['th50_min'])

    sota_a = hp.sota_models(lin, 'th50_min')
    sota_e = hp.sota_models(m2, 'th50_min')
    print(f'frontier sizes: (a) {len(sota_a)}  (e) {len(sota_e)}; '
          f'identical: {set(sota_a) == set(sota_e)}')

    # Success by length on the t* ladder, both fits on one logit range.
    x_star = np.log2(np.array(T_STARS))
    # Each row shows its own fit's frontier AIs only, here and in the
    # horizon panels.
    dates_m, P_m = by_date(lin, fit_metr,
                           model_api.marginal_p(fit_metr, x_star), sota_a)
    dates_2, P_2 = by_date(m2, fit, model_api.marginal_p(fit, x_star),
                           sota_e)
    ylim_p = hp.logit_window(P_m, P_2)

    dc = dvl.difficulty_curves(fit, idx)
    to_y, th_mean, th_sd = standardized_scale(dc)
    tid = idx['task_ids']
    t_sd = float(dc['t'].std(ddof=1))
    r = float(np.corrcoef(dc['t'], dc['theta_hat'])[0, 1])
    print(f'(d) theta_hat standardized: mean {th_mean:.3f}, SD {th_sd:.3f} '
          f'log2-min-equivalent ({th_sd:.2f} doublings); SD of log2 '
          f'minutes {t_sd:.2f}; corr(theta_hat, t) = {r:.3f}; blue line '
          f'slope {t_sd / th_sd:.3f} SD per SD, SNR {r / np.sqrt(1 - r * r):.2f}')

    # 3 x 2: METR's row, the diagnostics row, Model 2's row.  Rows 1
    # and 3 are the same pair of panels, so their axes line up and the
    # two fits can be read against each other down the page.
    fig = plt.figure(figsize=(5.5, 5.2))
    gs = fig.add_gridspec(3, 2, wspace=0.30, hspace=0.45, left=0.085,
                          right=0.985, top=0.965, bottom=0.06)
    ax = {k: fig.add_subplot(gs[r, col]) for k, (r, col) in
          zip('abcdef', [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)])}

    horizon_panel(ax['a'], lin[lin['model'].isin(sota_a)], sota_a,
                  hp.C_METR, '(a) METR baseline: time horizons')
    horizon_panel(ax['e'], m2[m2['model'].isin(sota_e)], sota_e, hp.C_M2,
                  '(e) Model 2: time horizons')
    ax['e'].set_xlim(ax['a'].get_xlim())
    mark_horizon(ax['a'], lin, PFIT_AI)
    mark_horizon(ax['e'], m2, PFIT_AI)
    # Every plotted AI is a frontier AI, so only the trend and the band
    # need a key.
    ax['a'].legend(handles=[
        plt.Line2D([], [], linestyle='--', color='0.45', linewidth=1.0,
                   label='exp. fit'),
        plt.Rectangle((0, 0), 1, 1, color=hp.C_BAND, alpha=0.13,
                      label=BAND_LABEL),
    ], fontsize=5.6, loc='lower right', frameon=False, handlelength=1.4,
        handletextpad=0.5, borderpad=0.1, labelspacing=0.28)

    hp.pcurve_panel(ax['b'], dates_m, P_m, T_STARS,
                    '(b) METR baseline: cond. success',
                    label_pad_days=420,
                    cmap=hp.shade_ramp(hp.C_METR, light=0.45), ylim=ylim_p,
                    ylabel='P(success), logit scale')
    hp.pcurve_panel(ax['f'], dates_2, P_2, T_STARS,
                    '(f) Model 2: cond. success',
                    label_pad_days=420,
                    cmap=hp.shade_ramp(hp.C_M2, light=0.45), ylim=ylim_p,
                    ylabel='P(success), logit scale')
    for k in 'bf':
        ax[k].set_xlabel('Release date', fontsize=7)
        ax[k].set_xlim(left=ax['a'].get_xlim()[0])
    panel_pfit(ax['c'], {
        spec: float(df.set_index('model').loc[PFIT_AI, 'th50_min'])
        for spec, df in ((METR_SPEC, lin), (model2.SPEC, m2))})
    panel_difficulty(ax['d'], dc, to_y)
    hp.save(fig, FIGS, 'headline')

    tidy = m2[['model', 'release_date', 'th50_min', 'dh50_min']].copy()
    tidy = tidy.rename(columns={'th50_min': 'th50_model2_min'})
    tidy['th50_metr_min'] = tidy['model'].map(
        own.set_index('model')['th50_min'])
    tidy['sota'] = tidy['model'].isin(sota_e)
    tidy = tidy[['model', 'release_date', 'sota', 'th50_metr_min',
                 'th50_model2_min', 'dh50_min']].sort_values('release_date')
    tidy.to_csv(FIGS / 'headline.csv', index=False)
    print(f'wrote {FIGS / "headline.csv"}')

    pd.DataFrame({
        'task_id': tid, 't_log2min': dc['t'],
        'difficulty_sd': to_y(dc['theta_hat']),
        'f_sd': to_y(np.interp(dc['t'], dc['xg'], dc['f'])),
    }).to_csv(FIGS / 'headline_difficulty.csv', index=False)
    print(f'wrote {FIGS / "headline_difficulty.csv"}')

    ratio = (lin.set_index('model')['th50_min']
             / own.set_index('model')['th50_min'])
    worst = ratio.sub(1.0).abs().sort_values(ascending=False)
    print(f'(a) baseline / METR per-AI th50: median '
          f'{ratio.median():.3f}; largest gaps: '
          + ', '.join(f'{m} {ratio[m]:.2f}' for m in worst.index[:3]))
    for name, df, sota in (('(a) baseline th50', lin, sota_a),
                           ('(e) Model 2 th50', m2, sota_e)):
        doubling, r2, _ = hp.exp_trend(df[df['model'].isin(sota)],
                                       'th50_min')
        print(f'  {name:18s} doubling {doubling:5.2f} mo   R^2 {r2:.3f}')


if __name__ == '__main__':
    main()
