"""Shared drawing code for the paper's figures.

The timeline panels (panel, exp_trend, fit_ylim, decorate) draw time
horizons on a shared log-minutes y-axis with an exponential trend fit
to the frontier, for make_headline.py and
make_model2_over_metr_plot.py; pcurve_panel draws success probability
by task length for make_headline.py and
make_difficulty_vs_length.py.  The palette, the six
2-15 minute band AIs, the small-multiple axis style and save/watermark
are shared by every figure script.
"""
from __future__ import annotations

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.patheffects as patheffects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Tol-style qualitative palette, as used elsewhere in the repo.
C_METR, C_TH, C_DH = '#4477aa', '#228833', '#ee6677'
# Model 2's colour across the score/Murphy/pfits figures.
C_M2 = '#aa3377'
C_OTHER = '#bbbbbb'
# Score-suite and success-curve figures: the baseline's colour and each
# model's, keyed by spec (names and the spec list live in pipeline/score.py).
C_BASE = '#4477aa'
MODEL_COLORS = {'shared_shslope_sqrtfamily': '#ee7733',
                's1_shslope_hetero_famspline_bb': C_M2}
# The six AIs whose horizons fall in the 2-15 min band, release order.
BAND_AIS = ['GPT-4 0314', 'GPT-4 1106 (Inspect)', 'Claude 3 Opus (Inspect)',
            'GPT-4 Turbo (Inspect)', 'GPT-4o (Inspect)',
            'Claude 3.5 Sonnet (Old) (Inspect)']
# Highlight for the 2-15 minute window, where the difficulty curve f is
# flattest; deliberately outside the panel palette above.
C_BAND = '#cc3311'

# Every type and marker size below is quoted at the paper's figure width
# and multiplied by SCALE.  Poster renders set SCALE > 1 (see
# make_poster_figs.py); at the default of 1.0 the arithmetic is a no-op,
# so the paper figures are unaffected.
SCALE = 1.0

# y ticks in minutes; the same grid serves every panel of both figures,
# which is what lets the difficulty panel be read against the others.
TICKS_MIN = np.array([1 / 60, 15 / 60, 1, 4, 15, 60, 240])
TICK_LABELS = ['1 sec', '15 sec', '1 min', '4 min', '15 min', '1 hr', '4 hr']
YLIM = (0.02, 900.0)


def fit_ylim(*series):
    """Raise the top of the shared y-range when a plotted horizon comes
    within 25% of it, in factor-4 steps, adding the 16 hr / 64 hr ticks
    it then needs.  The default range fits the 21 AIs the paper was first
    fit on; the current data's newest AI is past 15 hours."""
    global YLIM, TICKS_MIN, TICK_LABELS
    v = np.concatenate([np.asarray(s, dtype=float).ravel() for s in series])
    v = v[np.isfinite(v)]
    top = YLIM[1]
    while v.size and 1.25 * v.max() > top:
        top *= 4.0
    if top == YLIM[1]:
        return
    YLIM = (YLIM[0], top)
    for tick, label in ((960.0, '16 hr'), (3840.0, '64 hr')):
        if tick < top and tick not in TICKS_MIN:
            TICKS_MIN = np.append(TICKS_MIN, tick)
            TICK_LABELS = TICK_LABELS + [label]


def sota_models(df, col):
    """Record-holders on `col` by release date (scripts/sota.py rule:
    a model joins the frontier iff it strictly beats every earlier one)."""
    out, running_max = [], -np.inf
    for _, r in df.sort_values('release_date', kind='stable').iterrows():
        if r[col] > running_max:
            out.append(r['model'])
            running_max = r[col]
    return out


def frac_year(dates):
    return dates.dt.year + (dates.dt.dayofyear - 1) / 365.25


def exp_trend(df, col):
    """OLS of log2(col) on fractional year -> (doubling months, R^2, fn)."""
    fy = frac_year(df['release_date']).to_numpy()
    y = np.log2(df[col].to_numpy())
    b, a = np.polyfit(fy, y, 1)
    resid = y - (a + b * fy)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1.0 - np.sum(resid ** 2) / ss_tot if ss_tot > 0 else np.nan
    doubling = 12.0 / b if b > 0 else np.inf
    return doubling, r2, (lambda g: 2.0 ** (a + b * g))


def panel(ax, df, col, sota, color, title, band=None, highlight=None,
          annotate=True):
    """One timeline panel; returns (doubling months, R^2) of its trend.

    The models' horizons are joined in release order (same line idiom as
    pcurve_panel), with an exponential trend fit to the FRONTIER alone
    dashed over the top and annotated with its doubling time and R^2.
    `annotate=False` keeps the dashed trend but drops the text box, for
    figures whose panels repeat the same base series and would otherwise
    carry the identical box twice.
    Frontier models -- those beating every earlier one -- are marked with
    squares, the rest with small grey circles.

    `band` is an (lo, hi) pair in the units of `col`, shaded to mark the
    window where f is flat.  `highlight` names AIs to ring
    in C_BAND -- the same AIs in every panel, so the three can be read
    against one another."""
    s = SCALE
    is_sota = df['model'].isin(sota)
    if band is not None:
        ax.axhspan(band[0], band[1], color=C_BAND, alpha=0.13, lw=0,
                   zorder=0)
    if highlight:
        sel = df['model'].isin(highlight)
        # Ring rather than recolour: the square/circle distinction still
        # has to read through, and so does the panel's own colour.
        ax.scatter(df.loc[sel, 'release_date'], df.loc[sel, col],
                   s=34 * s ** 2, marker='o', facecolor='none',
                   edgecolor=C_BAND, linewidth=0.8 * s, zorder=5)
    by_date = df.sort_values('release_date', kind='stable')
    ax.plot(by_date['release_date'], by_date[col], '-', color=color,
            linewidth=0.9 * s, zorder=2, solid_capstyle='round')
    # Marker areas scale as the square, so type and markers keep their
    # relative weight as the figure grows.
    ax.scatter(df.loc[~is_sota, 'release_date'], df.loc[~is_sota, col],
               s=7 * s ** 2, marker='o', color=C_OTHER, zorder=3,
               linewidth=0)
    ax.scatter(df.loc[is_sota, 'release_date'], df.loc[is_sota, col],
               s=9 * s ** 2, marker='s', color=color, zorder=4,
               linewidth=0)

    doubling, r2, curve = exp_trend(df[is_sota], col)
    fy = frac_year(df['release_date'])
    gx = np.linspace(fy.min() - 0.15, fy.max() + 0.15, 100)
    gd = pd.to_datetime((gx - 1970.0) * 365.25, unit='D',
                        origin=pd.Timestamp('1970-01-01'))
    ax.plot(gd, curve(gx), '--', color=color, linewidth=1.1 * s, alpha=0.9,
            zorder=5)
    if annotate:
        ax.text(0.04, 0.965, f'doubling {doubling:.1f} mo\n$R^2$ = {r2:.3f}',
                transform=ax.transAxes, va='top', ha='left',
                fontsize=6.2 * s,
                bbox=dict(boxstyle='round,pad=0.28', fc='white', alpha=0.8,
                          ec='0.8', linewidth=0.4 * s))

    ax.set_yscale('log')
    ax.set_ylim(*YLIM)
    ax.set_yticks(TICKS_MIN)
    ax.set_yticklabels(TICK_LABELS)
    ax.set_yticks([], minor=True)      # decade minor ticks read as clutter here
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.tick_params(labelsize=6.5 * s, length=2.5 * s, pad=1.5 * s)
    ax.grid(True, which='major', alpha=0.22, linewidth=0.5 * s)
    ax.set_axisbelow(True)
    ax.set_title(title, fontsize=7.5 * s, pad=3.5 * s)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    return doubling, r2


def fmt_minutes(m):
    """Human-readable duration label for a value in minutes."""
    if m < 1:
        return f'{m * 60:.0f} s'
    if m < 60:
        return f'{m:.0f} min'
    h = m / 60
    return f'{h:.0f} h' if abs(h - round(h)) < 0.05 else f'{h:.1f} h'


def _stagger(ys, min_gap):
    """Push label heights apart top-down so none overlap; input must be
    descending (it is: with a shared slope the curves never cross, so the
    right-edge values are ordered by task length)."""
    out = list(ys)
    for i in range(1, len(out)):
        out[i] = min(out[i], out[i - 1] - min_gap)
    return out


def shade_ramp(color, dark=0.55, light=0.6):
    """Single-hue colormap through `color`: darkened toward black at 0,
    `color` itself mid-way, tinted toward white at 1.  Lets a p-curve
    panel take its figure column's colour while its series stay
    distinguishable by shade."""
    rgb = np.array(matplotlib.colors.to_rgb(color))
    return matplotlib.colors.LinearSegmentedColormap.from_list(
        'ramp', [rgb * (1.0 - dark), rgb, rgb + (1.0 - rgb) * light])


def logit_window(*Ps):
    """(lo, hi) logit range holding every value of the probability arrays
    `Ps`, never narrower than p = 0.0001 to 0.9999.  Pass several arrays
    to put p-curve panels on one shared axis."""
    L = np.concatenate([np.log(P / (1.0 - P)).ravel() for P in Ps])
    return (min(np.log(0.0001 / 0.9999), L.min() - 0.3),
            max(np.log(0.9999 / 0.0001), L.max() + 0.3))


def pcurve_panel(ax, dates, P, t_stars, title, label_pad_days=330,
                 band_mask=None, cmap=None, ylim=None,
                 ylabel='marginal P(success), logit scale'):
    """Fitted marginal p_j(t*) against release date on a LOGIT y-axis,
    one line per fixed task length t* -- the level-set transpose of a
    horizon panel.

    The logit stretches the tails, where a p-scale plot flattens the
    early and late models against 0 and 1.  The lines join consecutive
    releases rather than fitting a trend: on this scale they are close to
    straight, but only approximately so -- the conditional response is
    logistic in theta, yet p_j(t) marginalizes that logistic over the
    prior and the mixture is not itself a logistic.

    `dates` must be sorted ascending and `P` of shape
    (len(t_stars), len(dates)) in the same order.  Duration labels sit at
    the right edge of each series (color alone would hide the exponential
    spacing of t*); `label_pad_days` widens the axis to make room.

    `band_mask` is a boolean over `dates` selecting the AIs to ring in
    C_BAND -- the same ones the horizon panels highlight, so all three
    panels track one pair of AIs across the figure.

    `cmap` replaces the default viridis (e.g. a shade_ramp of the
    panel's model colour), spanning the shortest t* at 0 to the longest
    at 1.  `ylim` fixes the logit range, so that two panels can share
    it (see logit_window); by default it fits this panel's values."""
    s = SCALE
    lx = np.log2(np.asarray(t_stars, dtype=float))
    if cmap is None:
        # vmax past the longest t* keeps viridis off its pale yellow end
        norm = matplotlib.colors.Normalize(vmin=lx.min(), vmax=lx.max() + 2)
        cmap = matplotlib.colormaps['viridis']
    else:
        norm = matplotlib.colors.Normalize(vmin=lx.min(), vmax=lx.max())
    x_end = mdates.date2num(dates.iloc[-1])
    # The fitted marginals never saturate (see the assert), so the logit
    # is finite throughout and needs no clipping.
    assert (P > 0).all() and (P < 1).all(), 'p at 0 or 1: logit undefined'
    L = np.log(P / (1.0 - P))

    # Window wide enough to hold every fitted value (the extremes are
    # p = 0.00016 and 0.99994), so nothing is clipped, while still
    # bounding the axis independently of the data.  Label anchors are
    # clamped into it before staggering: a series running past the top
    # would otherwise put its label off-figure and out of the key.
    y_lo, y_hi = logit_window(P) if ylim is None else ylim
    y_labs = _stagger(np.clip(L[:, -1], y_lo, y_hi - 0.35), min_gap=0.9)
    for k, t in enumerate(t_stars):
        color = cmap(norm(lx[k]))
        ax.plot(dates, L[k], marker='o', ms=1.7 * s, lw=0.9 * s,
                color=color, zorder=3)
        if band_mask is not None:
            ax.scatter(dates[band_mask], L[k][band_mask], s=13 * s ** 2,
                       marker='o', facecolor='none', edgecolor=C_BAND,
                       linewidth=0.55 * s, zorder=5)
        ax.annotate(fmt_minutes(t), (x_end, y_labs[k]),
                    textcoords='offset points', xytext=(2.5 * s, -1.4 * s),
                    ha='left', fontsize=4.8 * s, color=color,
                    fontweight='bold', zorder=4,
                    # white halo: the 50% line runs under the labels
                    path_effects=[patheffects.withStroke(
                        linewidth=1.6 * s, foreground='white')])
    ax.axhline(0.0, color='0.5', lw=0.7 * s, ls='--', alpha=0.5, zorder=1)
    ax.set_xlim(right=x_end + label_pad_days)
    # Ticks at round probabilities, so the logit scale stays readable as
    # a probability axis.
    p_ticks = [0.0001, 0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999, 0.9999]
    p_labels = ['.0001', '.001', '.01', '.1', '.5', '.9', '.99', '.999',
                '.9999']
    ax.set_yticks([np.log(p / (1 - p)) for p in p_ticks])
    ax.set_yticklabels(p_labels)
    ax.set_ylim(y_lo, y_hi)
    ax.set_ylabel(ylabel, fontsize=7 * s, labelpad=1.5 * s)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.tick_params(labelsize=6.5 * s, length=2.5 * s, pad=1.5 * s)
    ax.grid(True, which='major', alpha=0.22, linewidth=0.5 * s)
    ax.set_axisbelow(True)
    ax.set_title(title, fontsize=7.5 * s, pad=3.5 * s)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)


def decorate(fig, axes, vertical=False, dh_note=True, layout=True):
    """Shared axis labels, the marker key, and final layout.

    `vertical` is for the poster's stacked layout: the panels share an x
    axis, so only the last one carries the date label, and the y label
    belongs on the middle panel rather than the first.  `dh_note` adds
    the minutes-equivalent annotation on the third panel; pass False when
    that panel is not a difficulty-horizon axis.  `layout=False` skips
    tight_layout, for callers that set their own gridspec geometry --
    tight_layout's single w_pad cannot give the two gaps different
    widths."""
    s = SCALE
    ylabel = '50% horizon\n(minutes, log scale)'
    if vertical:
        axes[1].set_ylabel(ylabel, fontsize=7 * s)
        axes[-1].set_xlabel('Release date', fontsize=7 * s)
    else:
        axes[0].set_ylabel(ylabel, fontsize=7 * s)
        axes[1].set_xlabel('Release date', fontsize=7 * s)
    if dh_note:
        axes[2].text(0.97, 0.05, 'minutes-\nequivalent',
                     transform=axes[2].transAxes, ha='right', va='bottom',
                     fontsize=6 * s, color='0.35', style='italic')
    handles = [
        plt.Line2D([], [], marker='s', linestyle='none', color=C_METR,
                   markersize=2.4 * s, label='frontier'),
        plt.Line2D([], [], marker='o', linestyle='none', color=C_OTHER,
                   markersize=2.2 * s, label='other'),
        plt.Line2D([], [], linestyle='--', color='0.45', linewidth=1.0 * s,
                   label='exp. fit'),
        plt.Rectangle((0, 0), 1, 1, color=C_BAND, alpha=0.13,
                      label='2–15 min'),
    ]
    axes[0].legend(handles=handles, fontsize=5.6 * s, loc='lower right',
                   frameon=False, handlelength=1.4, handletextpad=0.5,
                   borderpad=0.1, labelspacing=0.28)
    if layout:
        fig.tight_layout(pad=0.35, w_pad=0.7)


def save(fig, figs_dir, stem):
    figs_dir.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'png'):
        out = figs_dir / f'{stem}.{ext}'
        # no CreationDate, so an unchanged figure re-renders byte-identical
        meta = {'CreationDate': None} if ext == 'pdf' else None
        fig.savefig(out, dpi=400, bbox_inches='tight', metadata=meta)
        print(f'wrote {out}')
    plt.close(fig)


STANDIN_TEXT = 'IN-SAMPLE STAND-IN, NOT CROSS-VALIDATED'


def watermark(fig, text=STANDIN_TEXT):
    """Diagonal banner across the whole figure, for figures drawn from
    the in-sample stand-in of pipeline/evaluate.py --mode insample."""
    fig.text(0.5, 0.5, text, transform=fig.transFigure, ha='center',
             va='center', rotation=20, fontsize=13 * SCALE,
             color='#cc3311', alpha=0.35, fontweight='bold', zorder=100)


def short_ai(alias):
    return alias.replace(' (Inspect)', '')


def style(ax, fs=6.5):
    """Light grid, no top/right spines: the small-multiple panel style."""
    ax.grid(True, alpha=0.22, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=fs, length=2.5, pad=1.5)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
