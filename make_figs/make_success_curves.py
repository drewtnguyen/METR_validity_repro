#!/usr/bin/env python3
"""In-sample success curves p_j(t) against binned observed frequencies.

  success_curves_band_ais  paper Figure 5: the six AIs whose horizons
                           fall in the 2-15 minute band, 2x3
  success_curves_all_ais   paper Figure 10: every AI in release order,
                           three columns

Each panel draws the full-data fits of the baseline, Model 1 and
Model 2 (artifacts/canonical/fits) over the observed success frequency
of the AI's runs, pooled in factor-4 length bins with 95% Wald
intervals.  make_headline.py reuses load_insample and binned_freq
for Figure 1(c).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'pipeline'))
import _paths as P                                           # noqa: E402
import score as SC                                           # noqa: E402
from horizon_irt import data, io, model_api                  # noqa: E402
from horizon_irt import murphy as MP                         # noqa: E402
import horizon_panels as hp                                  # noqa: E402

FIGS = P.FIGS
FITS = P.fits_dir()
# (spec, name, colour, tag) for Models 1 and 2
MODELS = [(spec, name, hp.MODEL_COLORS[spec], tag)
          for spec, name, tag in SC.MODELS]


# Observed-frequency bins: factor-4 in length (width 2 in log2 minutes),
# edges placed so bin centres fall on 1 min, 4 min, 16 min, ... -- the
# ladder used elsewhere in the paper.  Points sit at the run-weighted
# mean length within the bin, so with bins this wide their spacing is
# UNEVEN by design: the mean goes where the tasks are.  (We tried 12-16
# equal-width bins, which space evenly, but prefer the coarser familiar
# ladder; the captions note the wander instead.)
BIN_EDGES = np.arange(-7.0, 12.0, 2.0)
XTICKS = [(-4, '4 s'), (0, '1 min'), (4, '16 min'), (8, '4 hr')]


def load_insample(extra_specs=()):
    """Full-data fits and the observed cell table for the p_j(t) panels.

    Returns (cells, xg, curves, order, edges) with curves[spec][alias]
    = p_j on xg.  The cell (s, n) table and task lengths come from the
    processed dataset.
    `extra_specs` appends further spec names to the curve dict (e.g.
    Figure 1's panel (c) wants linear_sqrtfamily) without touching the
    trio the pfits figures draw."""
    idx = data.load_cells(P.cells_path())
    xmap = dict(zip(idx['task_ids'], idx['t_array']))
    raw = data.read_cells(P.cells_path())
    cells = pd.DataFrame({'task_id': raw['task_id'], 'model': raw['alias'],
                          's': raw['n_successes'], 'n': raw['n_runs']})
    cells['x'] = cells['task_id'].map(xmap)
    assert not cells['x'].isna().any()
    xg = np.linspace(cells['x'].min(), cells['x'].max(), 400)
    edges = BIN_EDGES
    curves = {}
    for spec in [SC.BASELINE] + [m[0] for m in MODELS] + list(extra_specs):
        fit = io.load_fit(FITS / spec)
        Pg = model_api.marginal_p(fit, xg)     # (len(xg), n_models)
        curves[spec] = {m: Pg[:, k] for k, m in
                        enumerate(fit['model_names'])}
    order = (pd.read_csv(FITS / SC.BASELINE / 'params.csv')
             .sort_values('release_date')['model'].tolist())
    return cells, xg, curves, order, edges


def binned_freq(sub, edges):
    """(x_mean, freq, lo, hi, n_runs) per non-empty equal-width bin.

    freq pools every run of the AI's cells in the bin (sum s / sum n)
    and sits at the RUN-WEIGHTED MEAN length within the bin -- the
    honest x position, which inside these wide factor-4 bins wanders to
    wherever the bin's tasks cluster.  [lo, hi] is the
    95% Wald interval p +/- z*sqrt(p(1-p)/N) treating the bin's runs as
    independent Bernoulli (degenerate at unanimous bins, and wrong
    anyway given within-cell correlation -- kept simple on purpose)."""
    z = MP.Z[0.95]
    x = sub['x'].to_numpy()
    b = np.clip(np.digitize(x, edges) - 1, 0, len(edges) - 2)
    out = []
    for k in np.unique(b):
        m = b == k
        S = float(sub.loc[m, 's'].sum())
        N = float(sub.loc[m, 'n'].sum())
        xbar = float(np.average(x[m], weights=sub.loc[m, 'n']))
        ph = S / N
        half = z * math.sqrt(ph * (1.0 - ph) / N)
        out.append((xbar, ph, max(ph - half, 0.0),
                    min(ph + half, 1.0), N))
    return out


def _pfit_panel(ax, cells, xg, curves, alias, edges, fs=5.6):
    sub = cells[cells['model'] == alias]
    ax.plot(xg, curves[SC.BASELINE][alias], color=hp.C_BASE, lw=0.9,
            zorder=3)
    for spec, _, color, _ in MODELS:
        ax.plot(xg, curves[spec][alias], color=color, lw=0.9, zorder=3)
    for xbar, ph, lo, hi, _ in binned_freq(sub, edges):
        # at ph = 0 or 1 the Wilson bound equals ph up to float error;
        # clamp so errorbar never sees a negative extent
        ax.errorbar([xbar], [ph], yerr=[[max(ph - lo, 0.0)],
                                        [max(hi - ph, 0.0)]], fmt='o',
                    ms=1.9, color='0.15', elinewidth=0.6, capsize=0,
                    zorder=4)
    ax.set_xlim(xg[0] - 0.3, xg[-1] + 0.3)
    ax.set_ylim(-0.03, 1.03)
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_yticklabels(['0', '.5', '1'])
    ax.set_xticks([t for t, _ in XTICKS])
    ax.set_xticklabels([l for _, l in XTICKS])
    hp.style(ax, fs=fs)
    ax.set_title(hp.short_ai(alias), fontsize=fs + 1.2)


def _pfit_legend(fig, y=0.0):
    handles = ([plt.Line2D([], [], color=hp.C_BASE, lw=1.4,
                           label='baseline')]
               + [plt.Line2D([], [], color=c, lw=1.4, label=n)
                  for _, n, c, _ in MODELS]
               + [plt.Line2D([], [], marker='o', linestyle='none',
                             color='0.15', markersize=2.6,
                             label='observed (binned)')])
    fig.legend(handles=handles, fontsize=6, frameon=False, ncol=4,
               loc='lower center', bbox_to_anchor=(0.5, y),
               handlelength=1.3, handletextpad=0.4, columnspacing=1.2)


def fig_pfits(cells, xg, curves, edges):
    """Main-text version: the six band AIs, 2x3."""
    fig, axes = plt.subplots(2, 3, figsize=(5.5, 2.7), sharex=True,
                             sharey=True)
    for ax, alias in zip(axes.flat, hp.BAND_AIS):
        _pfit_panel(ax, cells, xg, curves, alias, edges)
        # keep the shared limits but label every row's x ticks, not just
        # the bottom one -- with the panels this far apart the eye loses
        # the bottom row's labels
        ax.tick_params(labelbottom=True)
    for ax in axes[:, 0]:
        ax.set_ylabel('P(success)', fontsize=6)
    _pfit_legend(fig)
    fig.tight_layout(pad=0.35, w_pad=0.7, h_pad=1.0,
                     rect=(0, 0.055, 1, 1))
    hp.save(fig, FIGS, 'success_curves_band_ais')


def fig_pfits_all(cells, xg, curves, order, edges):
    """Appendix version: every AI, release order, 3 columns (9x3 for
    the 26 canonical AIs)."""
    nrows = math.ceil(len(order) / 3)
    fig, axes = plt.subplots(nrows, 3, figsize=(5.5, 1.2 * nrows),
                             sharex=True,
                             sharey=True)
    for ax, alias in zip(axes.flat, order):
        _pfit_panel(ax, cells, xg, curves, alias, edges)
        ax.tick_params(labelbottom=True)      # label every row (see fig_pfits)
    for ax in axes.flat[len(order):]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel('P(success)', fontsize=6)
    _pfit_legend(fig, y=0.005)
    fig.tight_layout(pad=0.35, w_pad=0.7, h_pad=1.1,
                     rect=(0, 0.025, 1, 1))
    hp.save(fig, FIGS, 'success_curves_all_ais')


def main():
    cells, xg, curves, order, edges = load_insample()
    missing = [a for a in hp.BAND_AIS if a not in order]
    assert not missing, f'band AIs missing from the fits: {missing}'
    fig_pfits(cells, xg, curves, edges)
    fig_pfits_all(cells, xg, curves, order, edges)


if __name__ == '__main__':
    main()
