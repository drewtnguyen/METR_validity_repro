"""Raw METR results -> cell-level datasets -> numpy index for fitting.

Every model in this package sees the data only through CELLS: one row
per (task, AI) pair holding the number of runs and of successes, plus
the task's human length and family and the AI's release date.  That is
all the likelihoods use, and it is exactly the granularity METR's
canonical results file publishes, so the processed datasets are stored
at cell level (`CELL_COLUMNS`).

One dataset is built (see `build_dataset`):

  canonical  METR's live TH 1.1 results, task_results_1_1.yaml
             (metr.org/assets/task_results_1_1.yaml), plus the pre-2024
             anchor models grafted from TH 1.0 runs.  If that file is
             missing, the TH 1.1 runs.jsonl stand in for it.

The graft (`augment_cells`) exists because TH 1.1 only starts at GPT-4
0314.  It adds every TH 1.0 model that TH 1.1 lacks (matching aliases
after stripping " (Inspect)", which marks the same model under a
different harness) and that was released before the cutoff, restricted
to tasks TH 1.1 already has.  TH 1.1's per-task metadata wins wherever
the two disagree.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

CELL_COLUMNS = ['task_id', 'task_family', 'alias', 'release_date',
                'human_minutes', 'n_successes', 'n_runs']

AUGMENT_CUTOFF = '2024-01-01'
INSPECT_SUFFIX = ' (Inspect)'

RAW_FILES = {
    'th1_1_runs': 'th1_1_runs.jsonl',
    'th1_0_runs': 'th1_0_runs.jsonl',
    'task_results': 'task_results_1_1.yaml',
    'release_dates': 'release_dates.yaml',
}
DATASETS = ('canonical',)


# ----------------------------------------------------------------------
# raw readers
# ----------------------------------------------------------------------

def read_release_dates(path: Path) -> dict:
    """alias -> 'YYYY-MM-DD' from release_dates.yaml (a `date:` mapping)."""
    raw = yaml.safe_load(Path(path).read_text())['date']
    return {str(k): str(v) for k, v in raw.items()}


def read_runs_jsonl(path: Path) -> pd.DataFrame:
    """AI runs from a METR runs.jsonl (human baseline runs dropped)."""
    keep = ('task_id', 'task_family', 'alias', 'score_binarized',
            'human_minutes')
    rows = []
    with open(path) as fh:
        for line in fh:
            r = json.loads(line)
            if r['alias'] != 'human':
                rows.append({k: r[k] for k in keep})
    runs = pd.DataFrame(rows)
    assert runs['score_binarized'].isin([0, 1]).all(), \
        'score_binarized must be 0/1 for every AI run'
    return runs


def runs_to_cells(runs: pd.DataFrame, release_dates: dict) -> pd.DataFrame:
    """Aggregate runs to (task, AI) cells.  Task metadata comes from the
    task's first run, as the original cleaning did."""
    missing = sorted(set(runs['alias']) - set(release_dates))
    assert not missing, f'aliases without a release date: {missing}'
    meta = runs.drop_duplicates('task_id').set_index('task_id')
    g = (runs.groupby(['task_id', 'alias'], sort=True)['score_binarized']
         .agg(['sum', 'size']).reset_index())
    return pd.DataFrame({
        'task_id': g['task_id'],
        'task_family': g['task_id'].map(meta['task_family']),
        'alias': g['alias'],
        'release_date': g['alias'].map(release_dates),
        'human_minutes': g['task_id'].map(meta['human_minutes']),
        'n_successes': g['sum'].astype(np.int64),
        'n_runs': g['size'].astype(np.int64),
    })[CELL_COLUMNS]


def read_task_results_yaml(path: Path) -> pd.DataFrame:
    """Cells from METR's published task_results YAML.

    The file has no family field; a task's family is its task_id prefix
    (`family/task`), which matches the runs.jsonl families exactly on
    TH 1.1."""
    doc = yaml.safe_load(Path(path).read_text())
    rows = []
    for alias, agent in doc['agents'].items():
        for t in agent['tasks']:
            rows.append({
                'task_id': t['task_id'],
                'task_family': t['task_id'].split('/')[0],
                'alias': alias,
                'release_date': str(agent['release_date']),
                'human_minutes': float(t['human_minutes']),
                'n_successes': int(t['n_successes']),
                'n_runs': int(t['n_runs']),
            })
    cells = pd.DataFrame(rows)[CELL_COLUMNS]
    assert (cells['n_successes'] <= cells['n_runs']).all()
    return sort_cells(cells)


def sort_cells(cells: pd.DataFrame) -> pd.DataFrame:
    return (cells.sort_values(['task_id', 'alias'], kind='stable')
            .reset_index(drop=True))


# ----------------------------------------------------------------------
# augmentation (TH 1.0 pre-2024 anchors onto a TH 1.1 primary)
# ----------------------------------------------------------------------

def augment_cells(primary: pd.DataFrame, secondary_runs: pd.DataFrame,
                  release_dates: dict, cutoff: str = AUGMENT_CUTOFF):
    """(cells, diffs): primary cells plus the grafted secondary models.

    `diffs` lists every (task_id, field, primary, secondary) metadata
    disagreement on a grafted task; the primary value is the one kept.
    """
    present = {a.removesuffix(INSPECT_SUFFIX) for a in primary['alias']}
    candidates = set(secondary_runs['alias']) - present
    add = sorted(a for a in candidates
                 if release_dates.get(a, '') and release_dates[a] < cutoff)
    sec = secondary_runs[secondary_runs['alias'].isin(add)
                         & secondary_runs['task_id'].isin(primary['task_id'])]

    meta = primary.drop_duplicates('task_id').set_index('task_id')
    diffs = []
    sec_meta = sec.drop_duplicates('task_id').set_index('task_id')
    for field in ('task_family', 'human_minutes'):
        mine = meta.loc[sec_meta.index, field].to_numpy()
        theirs = sec_meta[field].to_numpy()
        for k in np.flatnonzero(mine != theirs):
            diffs.append((sec_meta.index[k], field, mine[k], theirs[k]))

    # reconcile task metadata to the primary before aggregating
    sec = sec.assign(task_family=sec['task_id'].map(meta['task_family']),
                     human_minutes=sec['task_id'].map(meta['human_minutes']))
    grafted = runs_to_cells(sec, release_dates)
    return sort_cells(pd.concat([primary, grafted], ignore_index=True)), \
        sorted(diffs)


def build_dataset(name: str, raw_dir: Path):
    """(cells, info) for dataset `name` from the files in `raw_dir`."""
    raw_dir = Path(raw_dir)
    dates = read_release_dates(raw_dir / RAW_FILES['release_dates'])
    live = raw_dir / RAW_FILES['task_results']
    if name == 'canonical' and live.exists():
        primary = read_task_results_yaml(live)
        source = RAW_FILES['task_results']
    elif name == 'canonical':
        primary = runs_to_cells(
            read_runs_jsonl(raw_dir / RAW_FILES['th1_1_runs']), dates)
        source = RAW_FILES['th1_1_runs']
    else:
        raise ValueError(f'unknown dataset {name!r}; expected {DATASETS}')
    secondary = read_runs_jsonl(raw_dir / RAW_FILES['th1_0_runs'])
    cells, diffs = augment_cells(primary, secondary, dates)
    grafted = sorted(set(cells['alias']) - set(primary['alias']))
    info = {
        'dataset': name, 'primary_source': source,
        'augment_source': RAW_FILES['th1_0_runs'],
        'augment_cutoff': AUGMENT_CUTOFF, 'grafted_models': grafted,
        'n_tasks': int(cells['task_id'].nunique()),
        'n_families': int(cells['task_family'].nunique()),
        'n_models': int(cells['alias'].nunique()),
        'n_cells': int(len(cells)), 'n_runs': int(cells['n_runs'].sum()),
        'metadata_diffs': [list(map(str, d)) for d in diffs],
    }
    return cells, info


# ----------------------------------------------------------------------
# numpy index used by every fitter
# ----------------------------------------------------------------------

def index_cells(cells: pd.DataFrame) -> dict:
    """Cell table -> numpy index (tasks and models sorted by name).

    Accepts the cell schema (`n_successes`, `n_runs`) or, for
    convenience, a run-level table with `score_binarized`."""
    if 'score_binarized' in cells.columns:
        cells = (cells.groupby(['task_id', 'alias'], sort=True)
                 .agg(task_family=('task_family', 'first'),
                      release_date=('release_date', 'first'),
                      human_minutes=('human_minutes', 'first'),
                      n_successes=('score_binarized', 'sum'),
                      n_runs=('score_binarized', 'size'))
                 .reset_index())
    task_ids = np.sort(cells['task_id'].unique())
    model_names = np.sort(cells['alias'].unique())
    t_pos = {t: i for i, t in enumerate(task_ids)}
    m_pos = {m: j for j, m in enumerate(model_names)}

    per_task = cells.drop_duplicates('task_id').set_index('task_id')
    t_array = np.log2(per_task.loc[task_ids, 'human_minutes']
                      .astype(float)).to_numpy(float)
    task_family = per_task.loc[task_ids, 'task_family'].to_numpy(object)
    release = (cells.drop_duplicates('alias').set_index('alias')
               .loc[model_names, 'release_date'].astype(str)
               .to_numpy(object))

    g = (cells.groupby(['task_id', 'alias'], sort=True)
         [['n_successes', 'n_runs']].sum().reset_index())
    ti = g['task_id'].map(t_pos).to_numpy(np.int64)
    mi = g['alias'].map(m_pos).to_numpy(np.int64)
    s = g['n_successes'].to_numpy(np.float64)
    n = g['n_runs'].to_numpy(np.float64)

    order = np.argsort(ti, kind='stable')
    cell_task, cell_midx = ti[order], mi[order]
    cell_s, cell_n = s[order], n[order]
    cell_off = np.zeros(len(task_ids) + 1, dtype=np.int64)
    np.add.at(cell_off, cell_task + 1, 1)
    cell_off = np.cumsum(cell_off)

    morder = np.argsort(mi, kind='stable')
    mc_task, mc_s, mc_n = ti[morder], s[morder], n[morder]
    mc_off = np.zeros(len(model_names) + 1, dtype=np.int64)
    np.add.at(mc_off, mi[morder] + 1, 1)
    mc_off = np.cumsum(mc_off)

    fams = {}
    for i, f in enumerate(task_family):
        fams.setdefault(f, []).append(i)
    fam_names = sorted(fams)
    fam_offsets = [0]
    fam_task_idx = []
    fam_mean_t = []
    for f in fam_names:
        fam_task_idx.extend(fams[f])
        fam_offsets.append(len(fam_task_idx))
        fam_mean_t.append(float(np.mean(t_array[fams[f]])))
    fam_of_task = np.empty(len(task_ids), dtype=np.int64)
    for fi, f in enumerate(fam_names):
        fam_of_task[fams[f]] = fi

    return {
        'task_ids': task_ids, 'model_names': model_names,
        'release_date': release,
        't_array': t_array, 'task_family': task_family,
        'cell_midx': cell_midx, 'cell_s': cell_s, 'cell_n': cell_n,
        'cell_off': cell_off,
        'mc_task': mc_task, 'mc_s': mc_s, 'mc_n': mc_n, 'mc_off': mc_off,
        'fam_offsets': np.asarray(fam_offsets, dtype=np.int64),
        'fam_task_idx': np.asarray(fam_task_idx, dtype=np.int64),
        'fam_names': fam_names, 'fam_mean_t': np.asarray(fam_mean_t),
        'fam_of_task': fam_of_task,
        'n_tasks': len(task_ids), 'n_models': len(model_names),
        'n_runs': int(n.sum()),
    }


def read_cells(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={'release_date': str})


def load_cells(path: Path) -> dict:
    return index_cells(read_cells(path))
