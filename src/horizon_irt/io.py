"""Saving and loading fits without pickle.

A fit is a dict of numpy arrays, scalars, strings, lists, None and
spline bases.  It is stored as two files in the fit's directory:

  state.json   everything except arrays; a NaturalBasis is stored as its
               defining knots, {"__basis__": {"lo", "hi", "interior"}}
  arrays.npz   every ndarray, referenced from state.json as
               {"__array__": "<key>"}

Python's json writes floats with repr, which round-trips float64
exactly, and the basis is a pure function of its knots, so load_fit
returns a fit that predicts bit-for-bit like the one that was saved.
Pickle would tie the files to this package's module layout; this format
only ties them to numpy.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .basis import NaturalBasis

FORMAT_VERSION = 1
STATE_FILE = 'state.json'
ARRAYS_FILE = 'arrays.npz'


def _is_basis(v) -> bool:
    # duck-typed so bases built by other copies of basis.py convert too
    return type(v).__name__ == 'NaturalBasis' and hasattr(v, 'interior')


def _encode(v, key, arrays):
    if isinstance(v, np.ndarray):
        if v.dtype == object:
            raise TypeError(f'{key}: object arrays are not supported')
        arrays[key] = v
        return {'__array__': key}
    if _is_basis(v):
        return {'__basis__': {'lo': float(v.lo), 'hi': float(v.hi),
                              'interior': [float(k) for k in v.interior]}}
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    if isinstance(v, dict):
        return {str(k): _encode(x, f'{key}.{k}', arrays)
                for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_encode(x, f'{key}[{i}]', arrays) for i, x in enumerate(v)]
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    raise TypeError(f'{key}: cannot serialise {type(v).__name__}')


def _basis_from_knots(d) -> NaturalBasis:
    nb = NaturalBasis.__new__(NaturalBasis)
    nb.lo = float(d['lo'])
    nb.hi = float(d['hi'])
    nb.interior = np.asarray(d['interior'], dtype=float)
    nb.n_interior = len(nb.interior)
    return nb


def _decode(v, arrays):
    if isinstance(v, dict):
        if set(v) == {'__array__'}:
            return arrays[v['__array__']]
        if set(v) == {'__basis__'}:
            return _basis_from_knots(v['__basis__'])
        return {k: _decode(x, arrays) for k, x in v.items()}
    if isinstance(v, list):
        return [_decode(x, arrays) for x in v]
    return v


def save_fit(fit: dict, out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays = {}
    state = {'format_version': FORMAT_VERSION,
             'fit': _encode(fit, 'fit', arrays)}
    (out_dir / STATE_FILE).write_text(json.dumps(state, indent=1) + '\n')
    np.savez(out_dir / ARRAYS_FILE, **arrays)


def load_fit(fit_dir: Path) -> dict:
    fit_dir = Path(fit_dir)
    state = json.loads((fit_dir / STATE_FILE).read_text())
    if state.get('format_version') != FORMAT_VERSION:
        raise ValueError(f'{fit_dir}: unsupported fit format '
                         f'{state.get("format_version")!r}')
    with np.load(fit_dir / ARRAYS_FILE, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    return _decode(state['fit'], arrays)
