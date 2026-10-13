"""Helpers shared by the test scripts (plain scripts, no pytest needed)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'pipeline'))
FITS = ROOT / 'artifacts' / 'canonical' / 'fits'
PY = sys.executable

# The shipped artifacts were made on macOS/arm64 with the PyPI wheels uv
# installs.  Other platforms' BLAS and compilers move these numbers at
# the 1e-7 level (relative; a few 1e-6 minutes on the longest horizons).
# 1e-6 absorbs that drift; a real change to the fitting code moves them
# by far more.
RTOL = ATOL = 1e-6

_failures = []


def check(cond, what):
    print(f"  [{'ok' if cond else 'FAIL'}] {what}", flush=True)
    if not cond:
        _failures.append(what)


def max_frame_diff(a: pd.DataFrame, b: pd.DataFrame) -> float:
    """Largest absolute difference over numeric columns; inf if the
    frames' shapes, columns or non-numeric values differ."""
    if list(a.columns) != list(b.columns) or a.shape != b.shape:
        return float('inf')
    num = a.select_dtypes('number').columns
    other = [c for c in a.columns if c not in num]
    if not a[other].astype(str).equals(b[other].astype(str)):
        return float('inf')
    if len(num) == 0:
        return 0.0
    x, y = a[num].to_numpy(float), b[num].to_numpy(float)
    if not np.array_equal(np.isnan(x), np.isnan(y)):
        return float('inf')
    d = np.abs(np.nan_to_num(x) - np.nan_to_num(y))
    return float(d.max()) if d.size else 0.0


def frames_close(a, b):
    num = a.select_dtypes('number').columns
    if max_frame_diff(a, b) == float('inf'):
        return False
    return np.allclose(a[num].to_numpy(float), b[num].to_numpy(float),
                       rtol=RTOL, atol=ATOL, equal_nan=True)


def compare_fit_dir(new: Path, ref: Path, name: str):
    for f in ('params.csv', 'curves.csv'):
        if (ref / f).exists():
            a, b = pd.read_csv(new / f), pd.read_csv(ref / f)
            check(frames_close(a, b),
                  f'{name}/{f} matches reference '
                  f'(max |diff| {max_frame_diff(a, b):.1e})')
    ja = json.loads((new / 'fit.json').read_text())
    jb = json.loads((ref / 'fit.json').read_text())
    check(np.isclose(ja['logl'], jb['logl'], rtol=RTOL, atol=ATOL)
          and ja['k_eff'] == jb['k_eff'],
          f"{name}/fit.json logL {ja['logl']:.6f} vs {jb['logl']:.6f}")


def finish():
    if _failures:
        print(f'\n{len(_failures)} check(s) FAILED')
        sys.exit(1)
    print('\nall checks passed')
