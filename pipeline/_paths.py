"""Shared locations for the pipeline and figure scripts.

Everything is relative to the repository root, so the folder works the
same wherever it is checked out.  Outputs live under
artifacts/canonical/; the V4_DATASET environment variable renames that
folder (and the cell table read) for experiments on other data.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

RAW = ROOT / 'data' / 'raw'
PROCESSED = ROOT / 'data' / 'processed'
DATASET = os.environ.get('V4_DATASET', 'canonical')


def cells_path(dataset: str = DATASET) -> Path:
    return PROCESSED / f'{dataset}.csv'


def fits_dir(dataset: str = DATASET) -> Path:
    return ROOT / 'artifacts' / dataset / 'fits'


def cv_dir(dataset: str = DATASET) -> Path:
    return ROOT / 'artifacts' / dataset / 'cv'


# Figure outputs; V4_FIGS redirects them, e.g. to a temporary directory
# when checking the figure scripts without touching figs/.
FIGS = Path(os.environ.get('V4_FIGS', ROOT / 'figs'))


def cv_mode(dataset: str = DATASET) -> str:
    """'cv' or 'insample-standin', from the CV outputs' provenance.json."""
    import json
    prov = cv_dir(dataset) / 'provenance.json'
    return json.loads(prov.read_text())['mode'] if prov.exists() else 'missing'


def scores_dir(dataset: str = DATASET) -> Path:
    return ROOT / 'artifacts' / dataset / 'scores'
