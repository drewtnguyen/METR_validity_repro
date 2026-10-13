"""Spline bases for the headline model grid (headline-model-grid skill).

One construction serves every spline in the grid:

  1. Build the degree-2 (quadratic) clamped B-spline basis Q_1..Q_{K_int+3}
     on knots [a, interior knots, b].  The interior knots sit at
     quantiles of the unique task lengths, or at explicit positions.
  2. Take RAW antiderivatives A_m(x) = int_a^x Q_m (cubic I-splines).
  3. Merge (sum) the first two and the last two raw columns -> K_int+1
     columns M_m.  The merge ties the end coefficients of f', which is
     equivalent to f''(a) = f''(b) = 0: the natural end conditions hold
     by construction, with no equality constraints.

  f(x) = c0 + sum_m w_m M_m(x)  is then
    - a MONOTONE natural cubic when w >= 0 (box bounds only -- this is
      why lsq_linear / bounded L-BFGS-B suffice), used for mean splines
      and the shared shape;
    - an arbitrary natural cubic when w is free, used for log sigma(t)
      and log tau(t_fam).

Identity property: sum_m Q_m == 1 on [a, b], so equal weights on the raw
merged columns give exactly c0 + w*(x - a) -- the identity is in the
span.  `build_monotone_basis` asserts this.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import BSpline


def _quantile_knots(x: np.ndarray, n_interior: int) -> np.ndarray:
    """Interior knots at quantiles of the UNIQUE values of x."""
    ux = np.unique(np.asarray(x, dtype=float))
    qs = np.linspace(0.0, 1.0, n_interior + 2)[1:-1]
    kn = np.quantile(ux, qs)
    return np.unique(kn)


def interior_knots(x: np.ndarray, knots) -> np.ndarray:
    """Interior knots from a count or from explicit positions.

    knots: an int, the number of interior knots, placed at quantiles of
    the unique values of x; or a sequence of interior knot positions on
    x's scale (log2 minutes for every basis in this package), which must
    lie strictly inside (min x, max x).  Every downstream property (the
    identity in the span, the natural end conditions, monotone columns)
    holds for any knot positions."""
    if np.ndim(knots) == 0:
        return _quantile_knots(x, int(knots))
    lo, hi = float(np.min(x)), float(np.max(x))
    kn = np.unique(np.asarray(knots, dtype=float))
    if len(kn) == 0:
        raise ValueError('explicit knots: need at least one interior knot')
    if kn[0] <= lo or kn[-1] >= hi:
        raise ValueError(
            f'explicit knots {np.round(2.0 ** kn, 2).tolist()} min must lie '
            f'strictly inside the data range '
            f'({2.0 ** lo:.4g}, {2.0 ** hi:.4g}) min')
    return kn


def _quadratic_bspline_antiderivatives(x, lo, hi, interior,
                                        return_splines=False):
    """Raw antiderivative columns A_m(x) of the clamped quadratic
    B-spline basis on [lo, hi] with the given interior knots.

    Returns (A, n_basis[, antis]) where A has shape (len(x), n_basis)."""
    deg = 2
    t = np.concatenate([[lo] * (deg + 1), interior, [hi] * (deg + 1)])
    n_basis = len(t) - (deg + 1)
    xc = np.clip(np.asarray(x, dtype=float), lo, hi)
    A = np.empty((len(xc), n_basis))
    antis = []
    for m in range(n_basis):
        c = np.zeros(n_basis)
        c[m] = 1.0
        anti = BSpline(t, c, deg, extrapolate=False).antiderivative()
        antis.append(anti)
        A[:, m] = anti(xc) - anti(lo)
    A = np.nan_to_num(A)
    if return_splines:
        return A, n_basis, antis
    return A, n_basis


class NaturalBasis:
    """Merged-column natural cubic basis on [lo, hi].

    design(x) -> (len(x), 1 + n_cols) matrix [1, M_1..M_n].
    Coefficients beta = (c0, w).  Monotone iff w >= 0 (sufficient).
    df = K_int + 2 = 1 + n_cols.

    n_interior: a knot count (quantile placement) or explicit interior
    knot positions; see interior_knots.
    """

    def __init__(self, x_ref: np.ndarray, n_interior):
        x_ref = np.asarray(x_ref, dtype=float)
        self.lo = float(np.min(x_ref))
        self.hi = float(np.max(x_ref))
        self.interior = interior_knots(x_ref, n_interior)
        self.n_interior = len(self.interior)

    def raw_merged(self, x: np.ndarray) -> np.ndarray:
        """The merged raw antiderivative columns M_m(x), no intercept."""
        A, nb = _quadratic_bspline_antiderivatives(
            x, self.lo, self.hi, self.interior)
        # Merge first two and last two raw columns: ties the end
        # coefficients of f' => f''(lo) = f''(hi) = 0 by construction.
        cols = [A[:, 0] + A[:, 1]]
        for m in range(2, nb - 2):
            cols.append(A[:, m])
        cols.append(A[:, nb - 2] + A[:, nb - 1])
        return np.column_stack(cols)

    def design(self, x: np.ndarray) -> np.ndarray:
        M = self.raw_merged(x)
        return np.column_stack([np.ones(len(M)), M])

    @property
    def n_cols(self) -> int:
        return self.n_interior + 1

    @property
    def df(self) -> int:
        return self.n_interior + 2


def build_monotone_basis(x_ref: np.ndarray, n_interior) -> NaturalBasis:
    """Natural basis + the identity/natural-condition self-checks."""
    nb = NaturalBasis(x_ref, n_interior)
    grid = np.linspace(nb.lo, nb.hi, 4001)
    M = nb.raw_merged(grid)

    # Identity in span: equal raw weights give exactly (x - lo).
    ident = M.sum(axis=1)
    assert np.max(np.abs(ident - (grid - nb.lo))) < 1e-8, \
        'identity not in span of merged columns'

    # Monotone columns.
    assert np.all(np.diff(M, axis=0) > -1e-12), 'columns not monotone'

    # Natural end conditions: exact f'' of every merged column at both
    # boundary knots, via the antiderivative splines' second derivatives.
    _, nbn, antis = _quadratic_bspline_antiderivatives(
        grid[:2], nb.lo, nb.hi, nb.interior, return_splines=True)
    groups = [[0, 1]] + [[m] for m in range(2, nbn - 2)] + [[nbn - 2, nbn - 1]]
    eps = 1e-9 * (nb.hi - nb.lo)
    scale = 1.0 / (nb.hi - nb.lo) ** 2 + max(
        abs(float(sum(antis[m].derivative(2)(0.5 * (nb.lo + nb.hi))
                      for m in g))) for g in groups)
    for g in groups:
        d2_lo = float(sum(antis[m].derivative(2)(nb.lo + eps) for m in g))
        d2_hi = float(sum(antis[m].derivative(2)(nb.hi - eps) for m in g))
        assert abs(d2_lo) / scale < 1e-6, ('f\'\'(lo) != 0', g, d2_lo)
        assert abs(d2_hi) / scale < 1e-6, ('f\'\'(hi) != 0', g, d2_hi)
    return nb


if __name__ == '__main__':
    rng = np.random.default_rng(0)
    x = rng.uniform(-2.5, 10.5, 300)
    for k_int in (2, 4):
        nb = build_monotone_basis(x, k_int)
        assert nb.df == k_int + 2, (nb.df, k_int)
        X = nb.design(x)
        assert X.shape == (300, k_int + 2)
        # Weighted monotone LS on noisy monotone data stays monotone.
        from scipy.optimize import lsq_linear
        y = 0.4 * x + np.sin(x / 3) + rng.normal(0, .1, len(x))
        lbnd = np.full(X.shape[1], -np.inf)
        lbnd[1:] = 0.0
        res = lsq_linear(X, y, bounds=(lbnd, np.inf), tol=1e-12)
        g = np.linspace(nb.lo, nb.hi, 800)
        fit = nb.design(g) @ res.x
        assert np.all(np.diff(fit) > -1e-10)
        print(f'  K_int={k_int}: df={nb.df}, identity/natural/monotone OK, '
              f'lsq_linear fit monotone OK')
    print('basis.py self-tests passed.')
