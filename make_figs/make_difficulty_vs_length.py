#!/usr/bin/env python3
"""Figure 2: AI task difficulty against human task length, under Model 2.

The estimated relationship behind Model 2's horizons; Figure 1(d)
draws the same curve and tasks on a task-anchored scale.  Nothing is
refit; this reads `s1_shslope_hetero_famspline_bb` from
artifacts/canonical/fits/.

  - dots: posterior difficulty of each of the 228 tasks,
    theta_hat_i = E[theta_i | data], one E-step at the fitted parameters
    (the family E-step: theta_i = f(t_i) + xi_F + eps_i).
  - curve: the monotone spline prior mean f, i.e. the difficulty a task
    of a given length is expected to have.
  - band: f +/- s, the prior spread of difficulty at a given length,
    with s(t)^2 = sigma(t)^2 + tau(t)^2 -- task-level spread plus the
    family effect at t_bar = t, the same fresh-task convention the
    marginal horizons use.  Both components vary with t under Model 2,
    so the band is NOT of constant width.
  - dashed line: the identity, where difficulty equals length.

Both axes are in log2 minutes, drawn with time tick labels, so the
identity is the 45-degree line and vertical distance from it reads as
"this task is N doublings harder than its length suggests".  The latent
scale is the least-squares-to-identity alignment of model2.py, which
also yields the difficulty horizons in headline.csv; Figure 1(d)
maps it affinely onto its task-anchored scale.

The shaded 2-16 minute band marks where f is flattest -- the range in
which METR's tasks are all of roughly equal difficulty to an AI, which
is what drives the wedge between time and difficulty horizons.

Paper Figure 2.  Outputs: figs/difficulty_vs_length.pdf, .png, and
difficulty_vs_length.csv.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import model2                                                # noqa: E402
import horizon_panels as hp                                  # noqa: E402

FIGS = model2.P.FIGS
BAND_MIN = (1.0, 4.0)              # log2 minutes: the 2-16 minute window

C_CURVE = '#ee7733'
C_PTS = '#4477aa'

# Wider than the horizon figures' grid: difficulties run past the longest
# task at the top end and below the shortest at the bottom.
TICKS_MIN = np.array([1 / 60, 15 / 60, 1, 4, 15, 60, 240, 960, 2400])
TICK_LABELS = ['1 sec', '15 sec', '1 min', '4 min', '15 min', '1 hr',
               '4 hr', '16 hr', '40 hr']


def _time_axis(ax, which, lo, hi, step=1):
    """`step` > 1 keeps every step'th tick; the poster render thins the x
    axis this way, since at poster type sizes nine time labels collide."""
    ticks = np.log2(TICKS_MIN)
    keep = (ticks >= lo) & (ticks <= hi)
    setter = ax.set_xticks if which == 'x' else ax.set_yticks
    labeller = ax.set_xticklabels if which == 'x' else ax.set_yticklabels
    setter(ticks[keep][::step])
    labeller([l for l, k in zip(TICK_LABELS, keep) if k][::step])


def difficulty_curves(fit, idx):
    """Everything the difficulty panel plots, on the least-squares
    identity scale (log2 minutes): each task's theta_hat, and the prior
    mean f with its task-level spread sigma and total spread s on a grid
    of lengths.  Figure 1(d) re-expresses the same values on a
    task-anchored scale, so both figures draw from this one function."""
    t = idx['t_array']
    c, d = model2.ols_identity_align(fit, t)
    print(f'latent realignment: theta -> (theta - {d:.4f}) / {c:.4f}')

    # Posterior means are linear in theta, so they realign by the same map.
    th_hat = (model2.posterior_theta(fit, idx) - d) / c
    xg = np.linspace(float(t.min()), float(t.max()), 400)
    f_raw, s_task_raw = model2.prior_mean_sigma(fit, xg)
    _, s_tot_raw = model2.prior_mean_total_sigma(fit, xg)
    return {'t': t, 'theta_hat': th_hat, 'xg': xg, 'f': (f_raw - d) / c,
            'sigma_task': s_task_raw / c, 's': s_tot_raw / c}


def draw(ax, fit, idx, scale=1.0, xtick_step=1):
    """Draw the difficulty-vs-length panel; return the plotted values.

    Every type and marker size is quoted at the paper's figure width and
    multiplied by `scale`; the poster passes scale > 1 (see
    make_poster_figs.py).  Returns theta_hat and the residual so the
    caller can write the CSV without redoing the E-step."""
    s = scale
    dc = difficulty_curves(fit, idx)
    t, th_hat, xg, fg = dc['t'], dc['theta_hat'], dc['xg'], dc['f']
    s_task, sg = dc['sigma_task'], dc['s']
    lo, hi = float(t.min()), float(t.max())
    s_fam = np.sqrt(np.maximum(sg ** 2 - s_task ** 2, 0.0))
    print(f'  sigma(t): {s_task.min():.3f}-{s_task.max():.3f}, '
          f'tau(t): {s_fam.min():.3f}-{s_fam.max():.3f}, '
          f'total s(t): {sg.min():.3f}-{sg.max():.3f} log2-minutes')
    resid = th_hat - t
    print(f'  theta_hat - t: mean {resid.mean():+.3f}, sd {resid.std():.3f}, '
          f'{(resid > 0).mean() * 100:.0f}% harder than nominal')
    fb = np.interp(BAND_MIN, xg, fg)
    slope = (fb[1] - fb[0]) / 3.0
    print(f'  f over 2-16 min: {fb[0]:.2f} -> {fb[1]:.2f} '
          f'(slope {slope:.2f} vs identity 1.0)')

    ax.axvspan(*BAND_MIN, color='0.85', alpha=0.55, lw=0, zorder=0,
               label='2–16 min')
    ax.plot([lo, hi], [lo, hi], '--', color='0.45', linewidth=0.9 * s,
            zorder=2, label='difficulty = human time')
    ax.fill_between(xg, fg - sg, fg + sg, color=C_CURVE, alpha=0.20, lw=0,
                    zorder=1, label=r'$f(t) \pm s(t)$')
    ax.plot(xg, fg, color=C_CURVE, linewidth=1.8 * s, zorder=4,
            label=r'prior mean $f(t)$')
    ax.scatter(t, th_hat, s=9 * s ** 2, color=C_PTS, alpha=0.55, linewidth=0,
               zorder=3, label=r'task $\hat\theta_i$')

    ax.set_xlim(lo - 0.4, hi + 0.4)
    ax.set_ylim(min(th_hat.min(), (fg - sg).min()) - 0.5,
                max(th_hat.max(), (fg + sg).max()) + 0.5)
    _time_axis(ax, 'x', *ax.get_xlim(), step=xtick_step)
    _time_axis(ax, 'y', *ax.get_ylim())
    ax.set_xlabel('Human time', fontsize=8 * s)
    ax.set_ylabel('Task difficulty for an AI\n(minutes-equivalent)',
                  fontsize=8 * s)
    ax.tick_params(labelsize=7 * s, length=2.5 * s, pad=1.5 * s)
    ax.grid(True, alpha=0.22, linewidth=0.5 * s)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.legend(fontsize=5.9 * s, loc='upper left', frameon=True,
              framealpha=0.85, edgecolor='0.85', handlelength=1.2,
              handletextpad=0.45, borderpad=0.3, labelspacing=0.22)
    return {'theta_hat': th_hat, 'resid': resid,
            'sigma_task': s_task, 'sigma_total': sg, 'band_slope': slope}


# Fixed task lengths for panel (a), in minutes: roughly factor-4 steps,
# rounded to wall-clock durations that read cleanly as labels.  (Moved
# here from the Figure 1 script when its panel (c) became the GPT-4o fit.)
T_STARS = [0.25, 1.0, 4.0, 15.0, 60.0, 240.0, 960.0]


def main():
    fit, idx = model2.load()

    # Panel (a): marginal p_j(t*) on the t* ladder, models by date.
    # em.marginal_p returns columns in fit order = alphabetical = the
    # params.csv order, so sorting by release date reorders both alike.
    m2 = pd.read_csv(model2.FITS / model2.SPEC / 'params.csv')
    m2['release_date'] = pd.to_datetime(m2['release_date'])
    P = model2.em.marginal_p(fit, np.log2(np.array(T_STARS)))
    order = np.argsort(m2['release_date'].to_numpy())
    p_dates = m2['release_date'].iloc[order].reset_index(drop=True)
    P = P[:, order]

    # Side by side, but not evenly: the difficulty panel (now first)
    # carries a task scatter, a band and a two-line y label, so it gets
    # ~40% more width than the p-curve panel, whose series read fine
    # when compressed.
    fig = plt.figure(figsize=(5.5, 2.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.4, 1.0], wspace=0.38,
                          left=0.095, right=0.99, top=0.90, bottom=0.16)
    axes = [fig.add_subplot(gs[0, i]) for i in range(2)]
    # every other time tick: nine labels need more than this width
    out = draw(axes[0], fit, idx, xtick_step=2)
    axes[0].set_title('(a) Model 2: difficulty vs human time',
                      fontsize=7.5, pad=3.5)
    # wider label margin than the default: this panel is narrower than
    # Figure 1's was, so the duration labels need more days per point of text
    hp.pcurve_panel(axes[1], p_dates, P, T_STARS,
                    '(b) Model 2: success by human time',
                    label_pad_days=560)
    axes[1].set_xlabel('Release date', fontsize=7)
    hp.save(fig, FIGS, 'difficulty_vs_length')

    cross = {t: p_dates.iloc[int(np.argmax(P[k] >= 0.5))].date()
             for k, t in enumerate(T_STARS) if (P[k] >= 0.5).any()}
    print('  (a) first model with p(t*) >= 1/2:',
          {hp.fmt_minutes(t): str(d) for t, d in cross.items()})

    pd.DataFrame({
        'task_id': idx['task_ids'], 'task_family': idx['task_family'],
        't_log2min': idx['t_array'], 'theta_hat_log2min': out['theta_hat'],
        'residual': out['resid'],
    }).to_csv(FIGS / 'difficulty_vs_length.csv', index=False)
    print(f'wrote {FIGS / "difficulty_vs_length.csv"}')


if __name__ == '__main__':
    main()
