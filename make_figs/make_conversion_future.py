#!/usr/bin/env python3
"""The time-to-difficulty conversion beyond one day: a hypothetical.

Figure 1(d)'s panel, redrawn on a human-time axis that runs on past the
current task suite to one month, for the discussion of how time
horizons should be read as the benchmark is extended.  Nothing is
refit; this reads Model 2 (`s1_shslope_hetero_famspline_bb`) from
artifacts/canonical/fits/ and draws on the same standardized scale as
Figure 1(d) (theta_hat over the tasks, mean 0, SD 1).

Up to one day it is Figure 1(d): task theta_hat (light blue), Model 2's
f +/- s (purple) and the baseline's straight line (blue).  Only one
task is longer than a day, so beyond that the data say little, and the
figure shows two continuations of the conversion from 1 day to 1 month:

  - the baseline's straight line, extended (blue, dotted): difficulty
    keeps rising at the same rate per doubling of human time;
  - a hypothetical extreme flattening (purple, dashed): f leaves 1 day
    with its fitted slope there, which then decays exponentially over
    FLAT_SCALE doublings, so f levels off at FLAT_SCALE times that
    slope above its 1-day value.  This is a scenario, not an estimate;
    it joins the fitted curve with matching value and slope.

Brackets at the right edge give the difficulty each continuation gains
between 1 day and 1 month.  Under the flat one, a 30x jump in time
horizon over that stretch takes almost no gain in difficulty.

Paper figure fig:future-conversion.  Outputs:
figs/conversion_future.pdf, .png, and conversion_future.csv (the two
continuations on a grid).
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import model2                                                # noqa: E402
import horizon_panels as hp                                  # noqa: E402
import make_difficulty_vs_length as dvl                     # noqa: E402
import make_headline as mh                                   # noqa: E402

FIGS = model2.P.FIGS

# Where the hypothetical takes over from the fit, and how far it runs,
# in log2 minutes: one day and one (30-day) month of human time.
X_DAY = np.log2(1440.0)
X_END = np.log2(30 * 1440.0)

# The flattening's decay length, in doublings: f's slope falls by a
# factor e every FLAT_SCALE doublings past one day.  One doubling makes
# the curve all but level by 1 week.
FLAT_SCALE = 1.0

# Ticks one per natural unit, as in Figure 1(d), plus the month the
# extension reaches.  A week sits too close to a month for both labels.
LENGTH_TICKS = mh.LENGTH_TICKS + [(X_END, '1 mo')]

C_HYPO = '0.93'       # shading behind the hypothetical stretch


def flattening(x, f0, g0):
    """The hypothetical f past one day: value f0 and slope g0 at X_DAY,
    slope decaying as exp(-(x - X_DAY) / FLAT_SCALE), so it levels off
    at f0 + g0 * FLAT_SCALE."""
    u = np.asarray(x, dtype=float) - X_DAY
    return f0 + g0 * FLAT_SCALE * (1.0 - np.exp(-u / FLAT_SCALE))


def bracket(ax, x, y0, y1, color, label):
    """Vertical bracket from y0 to y1 at x, labelled to its right: the
    difficulty a continuation gains between 1 day and 1 month."""
    cap = 0.12
    ax.plot([x, x], [y0, y1], color=color, lw=0.8, zorder=6,
            solid_capstyle='butt')
    for y in (y0, y1):
        ax.plot([x - cap, x + cap], [y, y], color=color, lw=0.8, zorder=6)
    ax.annotate(label, (x, 0.5 * (y0 + y1)), textcoords='offset points',
                xytext=(3.0, 0), ha='left', va='center', fontsize=5.4,
                color=color, zorder=6)


def main():
    fit, idx = model2.load()
    dc = dvl.difficulty_curves(fit, idx)
    to_y, _, th_sd = mh.standardized_scale(dc)
    t, xg = dc['t'], dc['xg']
    f, s = to_y(dc['f']), dc['s'] / th_sd
    lo, hi = float(t.min()), float(t.max())

    # The baseline's line on this scale: to_y of the identity.
    def line(x):
        return to_y(np.asarray(x, dtype=float))

    # The flattening joins f at one day with matching value and slope.
    f0 = float(np.interp(X_DAY, xg, f))
    g0 = float(np.interp(X_DAY, xg, np.gradient(f, xg)))
    xe = np.linspace(X_DAY, X_END, 200)
    flat = flattening(xe, f0, g0)
    gain_lin = float(line(X_END) - line(X_DAY))
    gain_flat = float(flat[-1] - f0)
    print(f'f at 1 day {f0:.3f} SD, slope {g0:.3f} SD/doubling '
          f'(baseline line {float(line(1) - line(0)):.3f})')
    print(f'1 day -> 1 month ({X_END - X_DAY:.2f} doublings): linear '
          f'+{gain_lin:.2f} SD, flattening +{gain_flat:.2f} SD')
    print(f'tasks longer than 1 day: {(t > X_DAY).sum()} of {len(t)}')

    fig, ax = plt.subplots(figsize=(3.3, 2.35))
    xmax = X_END + 2.0           # room for the bracket labels
    ax.axvspan(X_DAY, xmax, color=C_HYPO, lw=0, zorder=0)
    ax.axvline(X_DAY, color='0.55', lw=0.6, zorder=0.5)

    # Figure 1(d)'s panel, cut off at one day: the fitted f runs on to
    # the longest task (30 hr), but the hypothetical replaces it there.
    upto = xg <= X_DAY
    xd = np.append(xg[upto], X_DAY)
    fd, sd = np.interp(xd, xg, f), np.interp(xd, xg, s)
    band = ax.fill_between(xd, fd - sd, fd + sd, color=hp.C_M2, alpha=0.15,
                           lw=0, zorder=1)
    curve, = ax.plot(xd, fd, color=hp.C_M2, lw=1.4, zorder=4)
    metr, = ax.plot([lo, X_DAY], line([lo, X_DAY]), '-', color=hp.C_METR,
                    lw=1.1, zorder=2)
    dots = ax.scatter(t, to_y(dc['theta_hat']), s=5, color=mh.C_TASKS,
                      alpha=0.75, linewidth=0, zorder=3)

    # The two continuations.
    ext, = ax.plot([X_DAY, X_END], line([X_DAY, X_END]), ':',
                   color=hp.C_METR, lw=1.3, zorder=2)
    hyp, = ax.plot(xe, flat, '--', color=hp.C_M2, lw=1.4,
                   dashes=(3.5, 1.8), zorder=4)

    bracket(ax, X_END + 0.35, float(line(X_DAY)), float(line(X_END)),
            hp.C_METR, f'+{gain_lin:.1f} SD')
    bracket(ax, X_END + 0.35, f0, float(flat[-1]), hp.C_M2,
            f'+{gain_flat:.1f} SD')

    y_all = np.concatenate([f - s, f + s, to_y(dc['theta_hat']),
                            line([lo, X_END])])
    y_lo, y_hi = y_all.min() - 0.08, y_all.max() + 0.25
    ax.set_ylim(y_lo, y_hi)
    ax.set_yticks(np.arange(np.ceil(y_lo), y_hi + 1e-9))
    ax.set_xlim(lo - 0.4, xmax)
    ax.set_xticks([x for x, _ in LENGTH_TICKS])
    ax.set_xticklabels([l for _, l in LENGTH_TICKS])
    ax.text(X_DAY + 0.25, y_hi - 0.06, 'hypothetical', fontsize=5.6,
            color='0.35', ha='left', va='top', style='italic')
    ax.set_xlabel('Human time (log scale)', fontsize=7)
    ax.set_ylabel('AI difficulty (SDs)', fontsize=7, labelpad=1.5)
    hp.style(ax, fs=6.5)
    ax.legend([(band, curve), metr, dots, ext, hyp],
              [r'Model 2: $f(t) \pm s(t)$', 'METR baseline: linear',
               r'task $\hat\theta_i$', 'linear, extended',
               'extreme flattening'],
              handler_map={tuple: HandlerTuple(ndivide=None, pad=0)},
              fontsize=5.0, loc='upper left', frameon=False,
              handlelength=1.6, handletextpad=0.45, borderpad=0.2,
              labelspacing=0.3)
    hp.save(fig, FIGS, 'conversion_future')

    xs = np.linspace(X_DAY, X_END, 41)
    pd.DataFrame({'t_log2min': xs, 'linear_sd': line(xs),
                  'flattening_sd': flattening(xs, f0, g0)}).to_csv(
        FIGS / 'conversion_future.csv', index=False)
    print(f'wrote {FIGS / "conversion_future.csv"}')


if __name__ == '__main__':
    main()
