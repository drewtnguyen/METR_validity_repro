#!/usr/bin/env python3
"""Model 2 over the METR plot: time horizons, then Rasch abilities.

Two panels, each overlaying a Model 2 series on METR's own per-AI
time horizons (`linear_sqrtfamily`; Figure 1(a) instead plots the
shared-slope baseline, whose horizons are close; same panel styling
via horizon_panels).  As in Figure 1, only frontier AIs are drawn, each
series on its own frontier: METR's record-holders for the blue base,
Model 2's for the overlays (with one shared beta, alpha_j and Model 2's
horizons rank the AIs identically, so both overlays keep the same AIs).
The affine fit of (b) still uses every AI, frontier or not.

  (a) Model 2's marginal time horizons (Figure 1(e)'s series, in Model
      2's purple).  The correct estimate of the time-horizon construct
      bends away from METR's most visibly in the 2-30 minute band.
  (b) Model 2's Rasch abilities alpha_j, affinely rescaled (green --
      a different colour, because it is a different quantity: ability,
      not a horizon).  Model 2 shares one discrimination beta, so its
      intercepts alpha_1..alpha_J are its one-dimensional capability
      scale, identified only up to a positive affine map; we display
      them through the map that best fits METR's log2 horizons,

        (u, v) = argmin_{u,v} sum_j (u + v alpha_j - log2 th50_j^METR)^2,

      plotting A_j = 2^(u + v alpha_j) minutes.  Any latent realignment
      theta -> (theta - d)/c moves alpha by its own affine map
      (alpha -> alpha + beta d), so the overlay is invariant to the
      anchoring convention stored in params.csv.

Read together, the panels show the abilities agreeing with METR's
series more closely than Model 2's own horizons do: the warp the
horizon construct applies to ability (panel a's band departures) is
the same warp METR's misspecified linear fit cannot express.

model2 is imported only for the caption's side note: v is compared with
the model's own minutes-equivalent conversion -1/beta, whose value DOES
depend on the latent-scale convention (beta -> c beta), so it is quoted
at the OLS identity alignment used by every displayed Model 2 quantity
(Figures 1-2), not at the stored endpoint anchoring.

Paper Figure 5.  Outputs: figs/model2_over_metr_plot.{pdf,png} and
model2_over_metr_plot.csv (plotted values plus the fit numbers quoted in
the caption).
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import horizon_panels as hp                                  # noqa: E402
import model2                                                # noqa: E402

FITS = model2.FITS
SPEC = 's1_shslope_hetero_famspline_bb'   # Model 2 of Section 2.3
METR = 'linear_sqrtfamily'                # METR's own per-AI fit
FIGS = model2.P.FIGS

# Rescaled abilities are a different quantity from any horizon, so they
# get a colour no spec owns (blue = METR/baseline, orange = Model 1,
# purple = Model 2 across the score figures): Tol bright green.
C_ALPHA = '#228833'

# The 2-30 minute window shaded in Figure 1, where f is flat.
BAND_MIN = (2.0, 30.0)


def overlay(ax, df, col, color):
    """One Model 2 series over a METR base panel: diamonds joined in
    release order, the same line idiom as every horizon series."""
    by_date = df.sort_values('release_date', kind='stable')
    ax.plot(by_date['release_date'], by_date[col], '-', color=color,
            linewidth=0.9, zorder=2, solid_capstyle='round')
    ax.scatter(df['release_date'], df[col], s=9, marker='D', color=color,
               zorder=5, linewidth=0)


def main():
    m2 = pd.read_csv(FITS / SPEC / 'params.csv')
    lin = pd.read_csv(FITS / METR / 'params.csv')
    for df in (m2, lin):
        df['release_date'] = pd.to_datetime(df['release_date'])

    df = m2[['model', 'release_date', 'alpha', 'th50_min']].rename(
        columns={'th50_min': 'th50_model2_min'}).merge(
        lin[['model', 'th50_min']].rename(
            columns={'th50_min': 'th50_metr_min'}), on='model')
    alpha = df['alpha'].to_numpy()
    y = np.log2(df['th50_metr_min'].to_numpy())

    # The affine rescaling that best fits METR's log2 horizons.
    v, u = np.polyfit(alpha, y, 1)
    yhat = u + v * alpha
    df['alpha_rescaled_min'] = 2.0 ** yhat
    r2 = 1.0 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2)
    r = float(np.corrcoef(alpha, y)[0, 1])
    rmse = float(np.sqrt(np.mean((y - yhat) ** 2)))

    fit, idx = model2.load()
    c, _ = model2.ols_identity_align(fit, idx['t_array'])
    beta_aligned = c * m2['beta'].iloc[0]
    print(f'affine map: A_j = 2^({u:.3f} + {v:.3f} alpha_j) min '
          f'(Model 2 minutes-equivalent -1/beta = {-1.0 / beta_aligned:.3f} '
          f'at the OLS-aligned scale)')
    print(f'  R^2 = {r2:.4f}   r = {r:.4f}   '
          f'rmse = {rmse:.3f} log2-min (factor {2 ** rmse:.2f}x)')
    # The same affine-deviation measure for Model 2's own horizons: with
    # a shared beta they are an exact monotone function of alpha, so this
    # residual is purely the horizon construct's warp of the ability
    # scale -- the quantity panel (a) shows and panel (b) lacks.
    y2 = np.log2(df['th50_model2_min'].to_numpy())
    res2 = y2 - np.polyval(np.polyfit(alpha, y2, 1), alpha)
    print(f'  vs Model 2 own th50: rmse = {np.sqrt(np.mean(res2 ** 2)):.3f} '
          f'log2-min (factor {2 ** np.sqrt(np.mean(res2 ** 2)):.2f}x)')

    # Frontier AIs only, each series on its own frontier as in Figure 1.
    sota = hp.sota_models(lin, 'th50_min')
    sota_m2 = hp.sota_models(df, 'th50_model2_min')
    assert set(sota_m2) == set(hp.sota_models(df, 'alpha_rescaled_min')), \
        'alpha and Model 2 horizons should share a frontier'
    lin_f = lin[lin['model'].isin(sota)]
    df_f = df[df['model'].isin(sota_m2)]
    hp.fit_ylim(df['th50_model2_min'], df['alpha_rescaled_min'],
                lin['th50_min'])
    lo = df[['th50_model2_min', 'alpha_rescaled_min']].min().min()
    hi = df[['th50_model2_min', 'alpha_rescaled_min']].max().max()
    assert hp.YLIM[0] < lo and hi < hp.YLIM[1], \
        f'overlay outside panel y-limits: {lo:.3g}..{hi:.3g}'

    # Full text width (5.5 in, as Figure 1), so type stays at 1:1 in print.
    fig = plt.figure(figsize=(5.5, 2.5))
    gs = fig.add_gridspec(1, 2, wspace=0.08, left=0.088, right=0.995,
                          top=0.915, bottom=0.12)
    axes = [fig.add_subplot(gs[0, i]) for i in (0, 1)]

    doubling, tr2 = hp.panel(axes[0], lin_f, 'th50_min', sota, hp.C_METR,
                             '(a) Model 2 time horizons', band=BAND_MIN)
    overlay(axes[0], df_f, 'th50_model2_min', hp.C_M2)
    handles_a = [
        plt.Line2D([], [], marker='s', linestyle='none', color=hp.C_METR,
                   markersize=2.4, label='METR th50'),
        plt.Line2D([], [], marker='D', linestyle='-', color=hp.C_M2,
                   markersize=2.4, linewidth=0.9, label='Model 2 th50'),
        plt.Rectangle((0, 0), 1, 1, color=hp.C_BAND, alpha=0.13,
                      label='2–30 min'),
    ]
    axes[0].legend(handles=handles_a, fontsize=5.6, loc='lower right',
                   frameon=False, handlelength=1.4, handletextpad=0.5,
                   borderpad=0.1, labelspacing=0.28)

    # (b) repeats the identical METR base, so no second trend box; the
    # alpha-fit box takes its top-left spot instead.
    hp.panel(axes[1], lin_f, 'th50_min', sota, hp.C_METR,
             '(b) Model 2 abilities $\\alpha_j$, rescaled', band=BAND_MIN,
             annotate=False)
    overlay(axes[1], df_f, 'alpha_rescaled_min', C_ALPHA)
    axes[1].text(0.04, 0.965, f'$\\alpha$ fit: $R^2$ = {r2:.3f}',
                 transform=axes[1].transAxes, va='top', ha='left',
                 fontsize=6.2, color=C_ALPHA,
                 bbox=dict(boxstyle='round,pad=0.28', fc='white', alpha=0.8,
                           ec='0.8', linewidth=0.4))
    handles_b = [
        plt.Line2D([], [], marker='D', linestyle='-', color=C_ALPHA,
                   markersize=2.4, linewidth=0.9,
                   label=r'Model 2 $\alpha_j$, rescaled'),
    ]
    axes[1].legend(handles=handles_b, fontsize=5.6, loc='lower right',
                   frameon=False, handlelength=1.4, handletextpad=0.5,
                   borderpad=0.1, labelspacing=0.28)

    axes[1].set_xlim(axes[0].get_xlim())
    axes[1].set_yticklabels([])
    axes[1].tick_params(axis='y', length=0, pad=0)
    axes[0].set_ylabel('50% horizon\n(minutes, log scale)', fontsize=7)
    for ax in axes:
        ax.set_xlabel('Release date', fontsize=7)
    hp.save(fig, FIGS, 'model2_over_metr_plot')

    tidy = df.copy()
    tidy['sota'] = tidy['model'].isin(sota)
    tidy['sota_model2'] = tidy['model'].isin(sota_m2)
    tidy = tidy[['model', 'release_date', 'sota', 'sota_model2',
                 'th50_metr_min',
                 'th50_model2_min', 'alpha',
                 'alpha_rescaled_min']].sort_values('release_date')
    tidy.to_csv(FIGS / 'model2_over_metr_plot.csv', index=False)
    print(f'wrote {FIGS / "model2_over_metr_plot.csv"}')
    print(f'  METR frontier: doubling {doubling:.2f} mo   R^2 {tr2:.3f}')


if __name__ == '__main__':
    main()
