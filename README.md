# On the estimation and validity of AI time horizons: reproduction package

This repository holds the code, data and fitted artifacts behind *On the
estimation and validity of AI time horizons: a statistical look at the
METR plot* ([arXiv:2610.12466](https://arxiv.org/abs/2610.12466); the
built PDF is `paper/main.pdf`). It is self-contained.

It does two things:

- **Reproduce the paper.** One command goes from METR's published data
  to the PDF. Each step can also be run on its own.
- **Fit your own variants.** The models live in a small library,
  `src/horizon_irt`. [`docs/API.md`](docs/API.md) describes it.

## Contents

- [Reproduce the paper](#reproduce-the-paper)
- [Step 1: Download the data](#step-1-download-the-data)
- [Step 2: Process the data](#step-2-process-the-data)
- [Step 3: Fit the models](#step-3-fit-the-models)
- [Step 4: Compute the metrics](#step-4-compute-the-metrics)
- [Step 5: Generate the figures](#step-5-generate-the-figures)
- [Step 6: Generate the paper](#step-6-generate-the-paper)
- [Ablation options](#ablation-options)
- [Tests](#tests)
- [Folder layout](#folder-layout)

## Reproduce the paper

```bash
uv sync
make reproduce
```

You need [uv](https://docs.astral.sh/uv/), internet access (for
`uv sync`, which installs Python 3.12 and the pinned packages from
`uv.lock` into `.venv`, and for step 1), and a LaTeX installation with
`latexmk` for step 6. LaTeX is not part of the environment. Every
`make` target runs Python as `uv run python`; set `PY` to use another
interpreter, e.g. `make test PY=python`.

`make reproduce` just runs the six steps in order:

```bash
make download data fits eval scores figures paper
```

| Step | Command | Script | Reads | Writes | Time (8 cores) |
| --- | --- | --- | --- | --- | --- |
| 1. Download the data | `make download` | `pipeline/download_data.py` | METR's GitHub repo and website | `data/raw/` | seconds |
| 2. Process the data | `make data` | `pipeline/build_data.py` | `data/raw/` | `data/processed/` | seconds |
| 3. Fit the models | `make fits` | `pipeline/fit.py` | `data/processed/` | `artifacts/canonical/fits/` | ~1 min |
| 4. Compute the metrics | `make eval`, then `make scores` | `pipeline/evaluate.py`, `pipeline/score.py` | processed data, fits | `artifacts/canonical/cv/`, `artifacts/canonical/scores/` | ~3 min, then ~10 s |
| 5. Generate the figures | `make figures` | `make_figs/make_*.py` | processed data, fits, scores | `figs/` | ~30 s |
| 6. Generate the paper | `make paper` | `latexmk` | `figs/`, `paper/main.tex` | `paper/main.pdf` | seconds |

Each step reads only what earlier steps wrote, and no target reruns the
steps before it. Every output of steps 1–4 is committed, so any step can
be run on its own. In particular, plain `make` runs only steps 5 and 6,
redrawing the figures and the PDF from the shipped artifacts.

Three variables change what the steps do:

| Variable | Default | Meaning |
| --- | --- | --- |
| `SPECS` | `paper` | which specifications steps 3 and 4 fit: `paper`, `all` (the 52-spec grid) or a comma-separated list of codenames |
| `NJOB` | `8` | parallel worker processes |
| `EM_ACCEL` | `squarem` | how steps 3 and 4 run EM for the IRT specs: `squarem` (accelerated) or `none` (plain EM). See [`docs/API.md`](docs/API.md#acceleration-squarem). |

For example, `make fits eval NJOB=16`.

**Exact reproduction.** METR's live results file (step 1) is not pinned.
If METR has changed it since 2026-10-06, `make download` brings in the
new version and every later result changes. The shipped canonical fits,
CV, scores, figures and PDF were made with the default settings from
the committed `data/raw/`, so to reproduce them exactly, skip the
download:

```bash
make data fits eval scores figures paper
```

With plain EM (`EM_ACCEL=none`), Model 2 stops at a slightly different
point near the same optimum. Its time horizons then differ from the
shipped ones by at most 0.04%, and its CV log scores by at most 1e-4.

## Step 1: Download the data

```bash
make download        # uv run python pipeline/download_data.py [--force]
```

The data come from two sources. `data/raw/MANIFEST.json` records both,
and `data/raw/PROVENANCE.md` gives the details.

1. **METR's GitHub repository**, pinned to commit `571a941b` (2026-02-13)
   of [METR/eval-analysis-public](https://github.com/METR/eval-analysis-public).
   Three files come from there:
   - `th1_1_runs.jsonl`: run-level Time Horizon 1.1 results
   - `th1_0_runs.jsonl`: run-level Time Horizon 1.0 results
   - `release_dates.yaml`: AI release dates

   Each file is checked against its pinned SHA-256. A download that
   doesn't match is an error. A file that is already present and
   matches is not fetched again, unless you pass `--force`.

2. **METR's live results file**, <https://metr.org/assets/task_results_1_1.yaml>.
   This is METR's current published version of the TH 1.1 results, and
   it is not pinned. The download fetches it every time.
   - **If the fetch succeeds**, the new file replaces the local copy,
     and in step 2 it **takes precedence over the TH 1.1 `runs.jsonl`**
     from GitHub. The step also compares the new file with the copy it
     replaces:
     - It warns when AIs have been added. Every later step then needs
       rerunning, and the results will differ from the shipped ones.
     - It warns when AIs have been removed.
     - It notes when the AIs are the same but the contents changed.
   - **If the fetch fails**, the existing copy is kept. A fresh checkout
     has the copy retrieved on 2026-10-06, which has 23 AIs. With no
     copy at all, step 2 falls back to the TH 1.1 `runs.jsonl`.

The TH 1.0 runs supply only the three oldest AIs (see step 2).

## Step 2: Process the data

```bash
make data            # uv run python pipeline/build_data.py
```

This writes:

- `data/processed/canonical.csv`: the cleaned table, with one row per
  (task, AI) cell
- `data/processed/canonical_info.json`: counts, sources, the AIs added
  from TH 1.0, and every task-metadata disagreement between TH 1.0 and
  TH 1.1

Every model is fit on this cell table alone.

| Column | Meaning |
| --- | --- |
| `task_id` | METR task identifier, `family/task` |
| `task_family` | METR task family |
| `alias` | AI name as METR labels it. A " (Inspect)" suffix marks runs under the Inspect harness. |
| `release_date` | AI release date, `YYYY-MM-DD` |
| `human_minutes` | the task's human completion time, in minutes |
| `n_successes` | this AI's successful runs on this task (METR's binarized score) |
| `n_runs` | this AI's runs on this task |

The dataset is built in three steps:

1. **Take the TH 1.1 cells.** These come from the live results file,
   which is already at cell level. It has no family field, so a task's
   family is its `task_id` prefix. On TH 1.1 this matches the families
   in METR's `runs.jsonl` exactly. If the live file is missing, the
   cells come from the TH 1.1 runs instead:
   - human baseline runs are dropped
   - successes and runs are counted per (task, AI)
   - each task's length and family come from its first run
2. **Add the pre-2024 AIs from TH 1.0.** TH 1.1 starts at GPT-4 0314, so
   GPT-2, davinci-002 (GPT-3) and gpt-3.5-turbo-instruct are taken from
   the TH 1.0 runs. The rule is to add every TH 1.0 AI that:
   - TH 1.1 lacks, comparing names with the " (Inspect)" suffix removed,
     and
   - was released before 2024-01-01.

   Only their runs on tasks TH 1.1 already has are kept. Where TH 1.0 and
   TH 1.1 disagree on a task's length or family, TH 1.1 wins. All the
   disagreements are in task length and are listed in the info file.
3. **Write the outputs.** The step writes the table and info file, and
   warns when AIs have appeared or disappeared since the previous build.

The result has 26 AIs, 228 tasks, 79 families, 5,643 cells and 30,211
runs. The models use task length on a log scale, t = log2(`human_minutes`).

## Step 3: Fit the models

```bash
make fits            # uv run python pipeline/fit.py --specs paper
make fits SPECS=all  # the full 52-specification grid (~20 min)
```

Each specification ("spec") is a model plus the procedure that fits it,
named by a codename. `paper` fits the four specs of Section 2.3, plus
plain `linear`, which the stored CV contrasts use as their reference:

| Codename | In the paper (Section 2.3) |
| --- | --- |
| `linear_sqrtfamily` | logistic regression: METR's model, fit with METR's sqrt-family weighting |
| `linear_shslope_sqrtfamily` | logistic regression with a shared slope: the baseline |
| `shared_shslope_sqrtfamily` | Model 1: shared monotone spline with a shared slope, fit with the same sqrt-family weighting as the baseline |
| `s1_shslope_hetero_famspline_bb` | Model 2: explanatory IRT with family effects and overdispersion |
| `linear` | unweighted logistic regression (CV reference only) |

[`docs/API.md`](docs/API.md) defines every spec and describes how the
fitting works.

### The fitted model objects

Each fit is written to its own directory,
`artifacts/canonical/fits/<codename>/`:

| File | Contents |
| --- | --- |
| `state.json`, `arrays.npz` | The complete fitted model. `horizon_irt.io.load_fit(dir)` returns it as a Python dict that predicts exactly like the original fit. |
| `params.csv` | One row per AI: `model`, `release_date`, the 50% and 80% time horizons `th50_min` and `th80_min` (minutes), and the AI's parameters `alpha` and `beta` (plus `rho` for overdispersed shared-shape fits) |
| `fit.json` | Summary: the spec's configuration, log-likelihood `logl`, parameter counts `k_raw` and `k_eff`, `aic`, the EM driver `accel` (`squarem` or `none`), the number of EM updates `iterations` and of SQUAREM `cycles`, `converged`, `ascent_ok` (the EM log-likelihood never fell), the latent-scale anchoring `align_c` and `align_d`, the overdispersion `phi` and `rho_global`, the run time `wall_s`, and the data size. Fits made before SQUAREM have no `accel` or `cycles`; they used plain EM. |
| `curves.csv` | Fitted curves on a 241-point grid of log2 minutes (`x_log2min`). For IRT fits, these are the difficulty prior mean `f` and its total spread `s_marginal`. For shared-shape fits, they are the shape on [0, 1] (`f01`) and on the log2-minutes scale (`f_scaled`). |
| `history.csv` | IRT fits only: log-likelihood and largest parameter change at each plain EM iteration or SQUAREM cycle. SQUAREM adds the step length tried and whether it was kept (see [`docs/API.md`](docs/API.md#stopping-and-diagnostics)). |

The time horizons in `params.csv` invert each AI's marginal success
curve p_j(t). For Model 2, that curve integrates the task's latent
difficulty over its fitted prior (Section 2.3, equation (1)). The
`alpha` and `beta` columns follow the library's sign conventions, which
can differ from the paper's (see
[`docs/API.md`](docs/API.md#sign-and-scale-conventions)).

To use a fit in Python:

```python
import sys; sys.path.insert(0, 'src')   # not needed under `uv run python`
from horizon_irt import io, model_api

fit = io.load_fit('artifacts/canonical/fits/s1_shslope_hetero_famspline_bb')
model_api.time_horizons(fit, 0.5)               # 50% horizons, minutes, per AI
model_api.marginal_p(fit, [0.0, 4.0])           # p_j at 1 and 16 minutes
```

## Step 4: Compute the metrics

```bash
make eval            # uv run python pipeline/evaluate.py --mode cv
make scores          # uv run python pipeline/score.py --check
```

**Cross-validation (`make eval`).** Every spec is evaluated under two
5-fold schemes (seed 42):

- the **task-grouped** scheme holds out whole tasks
- the **family-grouped** scheme holds out whole task families. This is
  the one the paper reports (Section 2.4.2).

On each training fold, every spec is refit with its full-data recipe.
The sqrt-family specs recompute their weights from the fold. Each refit
then predicts the held-out cells. An AI is dropped from a fold if its
training data lacks a success or a failure. It is dropped for every
spec, so all specs are scored on exactly the same cells.

**The scoring-rule suite (`make scores`).** This step reads the
family-grouped CV predictions. It compares Models 1 and 2 with the
baseline on the same held-out runs, as paired differences. `--check`
first confirms that it reproduces the numbers `make eval` stored.

### Metrics

Section 2.4 of the paper defines and motivates the metrics. Briefly,
with p the predicted success probability of AI j at task i's length,
and y a run's outcome:

- **Marginal log score** (Section 2.4): −[y log p + (1 − y) log(1 − p)]
  per run.
- **Brier score** (Section 2.4): (y − p)² per run.
- **Elementary score at level q** (Section 2.4):
  S_q = q(1 − y)·1{p > q} + (1 − q)·y·1{p < q} per run. It depends on the
  model only through its q time horizons. The suite reports the smoothed
  score S̃_q at q = 0.5 and 0.8. Here S̃_q is S_q averaged over
  [q − 0.05, q + 0.05], which is the paper's integral divided by the
  window's width of 0.1.
- **Murphy curve** (Section 2.4): q ↦ mean S_q, over a grid of 299
  levels.
- **Joint log-likelihood** (Section 2.4): the log-probability of all
  held-out runs together, with the IRT latent effects integrated out.
  Under the family scheme, each family's effect is integrated over that
  family's held-out tasks together. The other metrics are means, but
  this is a total. For the non-IRT specs, cells are independent given
  the fitted curve, so their joint log-likelihood is a sum over cells.
- **Expected calibration error**: 15 bins. It is computed but not
  reported in the paper.

Each mean is weighted (Section 2.4.1). A run of AI j on task i in family
F gets weight w_ij, where n_ij is the AI's number of runs on the task,
and n_F is the number of scored tasks in the family:

| Weighting | w_ij | In the paper |
| --- | --- | --- |
| equal-run | 1 | no (computed only) |
| equal-task | 1 / n_ij | yes |
| equal-family | 1 / (n_F · n_ij) | yes; also used for the Murphy curves |
| sqrt-family | 1 / (√n_F · n_ij) | yes |

The paper's suite (Section 2.4.2) crosses four scores (marginal log
score, Brier score, S̃_0.5 and S̃_0.8) with the last three weightings,
which gives 12 metrics. Intervals are cluster-robust (Bayle et al.,
2020), clustered on the held-out unit: the task or the family. Row-level
and naive intervals are stored too.

### Outputs

`artifacts/canonical/cv/` (from `make eval`):

| File | Contents |
| --- | --- |
| `cv_preds_{task,family}.csv` | Every held-out cell (`fold`, `task_id`, `task_family`, `model`, successes `s`, runs `n`), with one column per spec holding its predicted success probability. Everything else is computed from these. |
| `cv_metrics.csv` | Marginal log score, Brier score, ECE and joint log-likelihood for each spec, scheme and weighting |
| `bayle_ci_{task,family}.csv` | 95% intervals for the log and Brier scores, both for each spec and as contrasts with `linear`. Columns `cluster_*` are cluster-robust, `bayle_*` treat runs as independent, and `sem_*` are naive. |
| `cv_joint_ll.csv` | Joint log-likelihood for each spec, scheme and fold |
| `insample_metrics.csv` | The same metrics for the full-data fits |
| `reliability_{cv,insample}.csv` | The calibration bins behind the ECE |
| `provenance.json` | How the outputs were produced: the mode, dataset, specs, folds, seed and quadrature sizes |

`artifacts/canonical/scores/` (from `make scores`):

| File | Contents |
| --- | --- |
| `score_suite.csv` | One row per number the paper reports, keyed by `figure`. `scoregrid` rows are the paired differences (model − baseline) of Figure 3. `murphy_points` rows are the differences at q = 0.5 and 0.8. `perai_gain` rows are the same contrasts within each AI (Figures 8 and 9). Each row gives the estimate, its standard error, its 80% and 95% half-widths, and the five per-fold values. |
| `murphy_curves.csv` | The Murphy curves of the baseline and both models (Figure 4), with each model's difference from the baseline and its 95% half-width |

**In-sample stand-in.** `make eval-standin` (`--mode insample`) is a fast
substitute for when the CV has not been rerun on new data. It uses the
same folds and scoring code, but scores every fold with the full-data
fit, so its numbers are in-sample, not cross-validated. `provenance.json`
records the mode. Every figure drawn from a stand-in carries an
"IN-SAMPLE STAND-IN, NOT CROSS-VALIDATED" watermark.

## Step 5: Generate the figures

```bash
make figures         # runs the six make_figs/make_*.py scripts
```

| Paper | File in `figs/` | Script | Built from |
| --- | --- | --- | --- |
| Figure 1 | `headline` | `make_headline.py` | fits of `linear_shslope_sqrtfamily`, `linear_sqrtfamily` and Model 2; cells |
| Figure 2 | `difficulty_vs_length` | `make_difficulty_vs_length.py` | Model 2 fit; cells |
| Figure 3 | `score_grid` | `make_score_figures.py` | `scores/score_suite.csv` |
| Figure 4 | `murphy_diagrams` | `make_score_figures.py` | `scores/murphy_curves.csv` |
| Figure 5 | `success_curves_band_ais` | `make_success_curves.py` | fits of the baseline and Models 1 and 2; cells |
| Figure 6 | `model2_over_metr_plot` | `make_model2_over_metr_plot.py` | fits of `linear_sqrtfamily` and Model 2 |
| Discussion figure (`fig:future-conversion`) | `conversion_future` | `make_conversion_future.py` | Model 2 fit; cells |
| Figure 8 | `per_ai_contrasts_model1` | `make_score_figures.py` | `scores/score_suite.csv` |
| Figure 9 | `per_ai_contrasts_model2` | `make_score_figures.py` | `scores/score_suite.csv` |
| Figure 10 | `success_curves_all_ais` | `make_success_curves.py` | fits of the baseline and Models 1 and 2; cells |

Notes on the figure files:

- **PDF and PNG.** Each figure is written as a PDF, which the paper
  includes and the repository commits, and a PNG, which is not
  committed.
- **Plotted values.** `headline.csv`, `difficulty_vs_length.csv`
  and `model2_over_metr_plot.csv` hold the values behind the numbers the
  text quotes.
- **Hypothetical extension.** `conversion_future.csv` holds the two
  continuations of the conversion past one day that the discussion
  figure draws.
- **No Figure 7.** The paper has none: `main.tex` sets the figure
  counter to 7 just before the appendix.
- **Shared code.** `make_figs/horizon_panels.py` and `make_figs/model2.py`
  hold drawing and Model 2 helpers that the scripts share.

## Step 6: Generate the paper

```bash
make paper           # cd paper && latexmk -pdf main.tex
```

`paper/main.tex` includes its figures from `../figs/`. For arXiv,
upload the generated `paper/main.bbl` along with the source. It is
gitignored, so take it from your build.

## Ablation options

Beyond the paper's specifications, the library has options for
ablations of Models 1 and 2:

- a separate spline for each AI
- spline sizes, with knots at quantiles or at positions you choose
- anchored bumps on the difficulty curve, general or AI-specific
- Model 1 with a difficulty curve from fixed-effect (JML) IRT

[`docs/API.md`](docs/API.md#ablation-options) describes them. Five named
variants run from the pipeline scripts, for example:

```bash
make fits SPECS=ablations
uv run python pipeline/evaluate.py --specs paper,ablations --out artifacts/canonical/cv_ablations
uv run python pipeline/score.py --cv artifacts/canonical/cv_ablations --models shared_shslope_jml,shared_shslope_bump2_aibump --out artifacts/canonical/scores_ablations
```

`uv run python tests/test_ablations.py` checks these options in about two
minutes.

## Tests

```bash
make test            # about a minute
```

`make test` checks six things:

- the dataset rebuilds exactly from `data/raw/`
- the spline bases are built correctly
- the spec registry is complete
- three quick fits (`linear`, the baseline and Model 1) match the
  shipped `artifacts/canonical/fits/` to 1e-6, and their saved files
  predict exactly like the fits themselves
- SQUAREM and plain EM fit a small IRT spec (`zm_homo` with K = 200)
  to the same optimum, and SQUAREM takes fewer EM updates
- the shipped fits of the `paper` specs load and give valid
  probabilities

## Folder layout

```
Makefile               one target per step (see "Reproduce the paper")
pyproject.toml         project metadata and the pinned dependencies
uv.lock                the exact resolved versions `uv sync` installs
data/raw/              METR's data as downloaded (MANIFEST.json, PROVENANCE.md)
data/processed/        the cell-level dataset, canonical.csv, and its info file
pipeline/              step scripts: download_data, build_data, fit,
                       evaluate, score
src/horizon_irt/       the fitting and evaluation library (docs/API.md)
artifacts/canonical/   fits/, cv/ and scores/
make_figs/             one script per figure group, plus shared drawing code
figs/                  figure PDFs and plotted-value CSVs
paper/                 main.tex, bibliography and style files, main.pdf
examples/              custom_specs.py: fitting your own variants of Model 2
tests/                 test_fast.py and test_ablations.py
docs/                  API.md
```
