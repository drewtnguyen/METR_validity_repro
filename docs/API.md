# horizon_irt: API

`src/horizon_irt` is the library behind every fit in the paper. The
pipeline scripts call it, and you can call it directly to fit variants
of the paper's models. This page covers three things:

1. [the interface for fitting models](#fitting-a-model)
2. [which models are possible](#which-models-are-possible)
3. [options for ablations](#ablation-options)
4. [how the optimization works and how it is called](#how-the-optimization-works)

The scripts in this folder put `src/` on the Python path themselves.
Anywhere else, run Python as `uv run python` from the repository root
(`uv sync` installs `horizon_irt` into `.venv`), or add `src/` to
`PYTHONPATH`.

| Module | Contents |
| --- | --- |
| `data` | building the processed datasets; loading cells into the index every fitter takes |
| `specs` | the registry of 52 named specifications |
| `fitting` | `fit_spec` and friends: the single entry point for fitting |
| `em`, `kernels` | the SQUAREM-accelerated EM fitter for IRT specs, and its numba E- and M-step kernels |
| `basic`, `basis` | direct fitters for non-IRT specs; the monotone spline bases |
| `model_api` | predictions from any fit: success curves, time horizons, joint log-likelihood |
| `io` | saving and loading fits without pickle |
| `evaluate`, `murphy` | cross-validation, scoring rules, intervals, Murphy curves |
| `bumps`, `jml` | ablation options: anchored bumps, and Model 1 with a difficulty curve from fixed-effect IRT |

## Fitting a model

```python
from horizon_irt import data, fitting, io, model_api

idx = data.load_cells('data/processed/canonical.csv')
fit = fitting.fit_spec('s1_shslope_hetero_famspline_bb', idx)   # Model 2, ~1 min

model_api.time_horizons(fit, 0.5)       # 50% horizon of each AI, in minutes
model_api.marginal_p(fit, [0.0, 4.0])   # success probabilities at 1 and 16 minutes
io.save_fit(fit, 'my_fit/')             # writes state.json + arrays.npz
fit = io.load_fit('my_fit/')
```

### Data

| Function | Does |
| --- | --- |
| `data.load_cells(path)` | Reads a processed dataset into the **index**: a dict of numpy arrays that every fitter takes. Tasks and AIs are sorted by name. It holds task lengths `t_array` (log2 minutes), families, and each cell's successes and runs. |
| `data.read_cells(path)` | Reads a processed dataset as a DataFrame with the cell columns listed in the README. |
| `data.index_cells(df)` | Builds the index from such a DataFrame. Use it to fit on a subset of the data. |

For example, to fit on tasks of at least one minute:

```python
df = data.read_cells('data/processed/canonical.csv')
idx = data.index_cells(df[df['human_minutes'] >= 1])
```

The index lists AIs in `idx['model_names']`. Every per-AI output follows
that order.

### Fitting

| Function | Does |
| --- | --- |
| `fitting.fit_spec(spec, idx, K=1000, Kg=80, max_iter=300, accel='squarem')` | Fits one specification and returns the fit (a dict). `spec` is a registry codename or a configuration dict. IRT fits are anchored (see [Anchoring](#anchoring)). |
| `fitting.fit_and_save(spec, idx, out_dir, **kw)` | Fits, then writes `out_dir/<name>/` with the files the README describes (`params.csv`, `fit.json`, ...). Returns the `fit.json` summary. |
| `fitting.make_spec(base, name=None, **overrides)` | Returns a configuration dict: registry spec `base` with some keys changed. |
| `specs.all_specs()` | The registry: `{codename: configuration dict}` for all 52 specs. `specs.PAPER_SPECS` names the paper's four. |

`K` and `Kg` are the quadrature sizes for the task and family effects,
`max_iter` caps the number of EM updates, and `accel` chooses how EM is
run: `'squarem'` accelerates it with SQUAREM, and `'none'` runs plain EM
(see [Acceleration](#acceleration-squarem)). Non-IRT specs ignore all
four.

### Predicting

These work for any fit, whatever its class.

| Function | Returns |
| --- | --- |
| `model_api.marginal_p(fit, x)` | Each AI's success probability p_j(x) at log2 lengths `x`, with shape `(len(x), n_AIs)` |
| `model_api.time_horizons(fit, q)` | Each AI's q time horizon: the length in minutes where p_j crosses q. It is NaN if p_j does not cross q within the data's range. |
| `model_api.marginal_p_for_idx(fit, idx)` | p for every cell of `idx`, for scoring. The AIs in `idx` must be the fit's AIs, in the same order. |
| `model_api.joint_ll(fit, idx, family_scheme=False)` | The joint log-likelihood of `idx`'s cells, with any latent effects integrated out. `family_scheme=True` integrates each family effect over the family's tasks together. |

### From the command line

`pipeline/fit.py --specs name,name` fits registered specs into
`artifacts/canonical/fits/`, and `pipeline/evaluate.py --specs name,name`
cross-validates them. Both scripts look specs up in the registry, so to
cross-validate a spec of your own, add it to `ablation_specs()` in
`specs.py` (see [Named ablation specs](#named-ablation-specs)). Both
take `--em-accel none` to run plain EM instead of SQUAREM.

## Which models are possible

Notation: AI j, task i, task family F(i), and task length
t = log2(human minutes). Every specification gives each AI a success
curve p_j(t), the probability that one run of AI j succeeds on a task of
length t. Time horizons invert that curve. There are two classes of
specification.

### Basic specifications (`cls='basic'`)

A basic spec models the curve directly, and treats the runs as
independent given it. It has three kinds:

- **`kind='linear'`**: logit p_j(t) = α_j + β_j t. This is METR's
  logistic regression.
- **`kind='spline'`**: logit p_j(t) = c_j − S_j(t), with a separate
  increasing spline S_j for each AI.
- **`kind='shared'`**: logit p_j(t) = a_j − b_j f(t), with b_j ≥ 0. Here
  f is one increasing spline shared by all AIs, which rises from 0 to 1
  over the data's range of lengths. With `shslope` (one b for all AIs),
  this is **Model 1**.

### IRT specifications (`cls='irt'`)

An IRT spec is a generative model. Each task has a latent difficulty
θ_i, and

  logit P(run succeeds | θ_i) = α_j + β_j θ_i,
  θ_i = m(t_i) + ξ_F(i) + ε_i,  ε_i ~ N(0, σ(t_i)²),  ξ_F ~ N(0, τ²).

- The prior mean m is a fitted increasing spline f (`fit_mean=True`), or
  the task length itself, m(t) = t (`fit_mean=False`, the "zero-mean"
  or `zm` specs).
- The task spread σ is either one constant or a spline in t
  (`hetero`).
- The family effect ξ_F is either absent, or has one τ, or has τ
  varying with the family's mean log length t̄_F (`family`).
- With `bb`, a cell's successes are beta-binomial instead of binomial:
  its success probability is drawn from
  Beta(qφ, (1 − q)φ) around q = P(success | θ_i). Runs of one AI on one
  task are then correlated, with ρ = 1/(1 + φ).

**Model 2** is the IRT spec with every option on: a fitted f, σ(t),
τ(t̄_F), beta-binomial cells and one β shared by all AIs. Its success
curve integrates the difficulty out (paper equation (1)):

  p_j(t) = E[logit⁻¹(α_j + β θ)],  θ ~ N(f(t), σ(t)² + τ(t)²).

### Configuration keys

A specification is a dict of these keys (`fitting.SPEC_KEYS`):

| Key | Values | Meaning | Used by |
| --- | --- | --- | --- |
| `cls` | `'basic'`, `'irt'` | model class | all |
| `kind` | `'linear'`, `'spline'`, `'shared'` | curve form | basic |
| `fit_mean` | bool | a fitted spline f, or f(t) = t | IRT |
| `hetero` | bool | σ(t) a spline in t, or one constant σ | IRT |
| `family` | `'none'`, `'fam'`, `'famspline'` | no family effect; ξ_F with one τ; or τ a spline in t̄_F | IRT |
| `classic` | bool | textbook IRT: one fixed prior for every task, ignoring length, so time horizons are undefined | IRT |
| `bb` | bool | beta-binomial cells: one φ for IRT, one ρ_j per AI for `shared` | IRT, `shared` |
| `probit` | bool | probit link instead of logit | all |
| `shslope` | bool | one slope (β, or b) shared by all AIs | IRT, `linear`, `shared` |
| `sqrtfamily` | bool | fit by maximizing METR's sqrt-family-weighted likelihood: the same model, a different estimator | `linear`, `shared` |
| `mean_kint` | int, default 4 | interior knots of f (degrees of freedom = kint + 2) | IRT with `fit_mean`, `spline`, `shared` |
| `var_kint` | int, default 2 | interior knots of log σ(t) | IRT with `hetero` |
| `tau_kint` | int, default 2 | interior knots of log τ(t̄) | IRT with `family='famspline'` |
| `mean_knots`, `var_knots`, `tau_knots` | knot positions in minutes | explicit interior knots, instead of `*_kint` knots at quantiles | as the matching `*_kint` |
| `bumps` | tuple of `bumps.bump(start, end, ai_specific=False)` | anchored bumps on the difficulty curve | IRT except `classic`, `shared` |
| `bump_monotone` | bool, default True | general bumps must keep the difficulty curve increasing | with `bumps` |
| `shape_source` | `'fit'` (default) or `'jml'` | `'jml'` freezes the shape at a spline of JML difficulties | `shared` |
| `jml_weights`, `jml_drop`, `jml_beta_ridge` | see [the JML route](#model-1-with-a-jml-difficulty-curve) | settings of the JML route | `shape_source='jml'` |

[Ablation options](#ablation-options) describes the last five rows.

The fitter ignores keys that don't apply to a spec's class or kind.
Combinations that it can't fit raise an error instead: `bb`,
`sqrtfamily` or `shslope` with `kind='spline'`, `bb` with
`kind='linear'`, a `*_kint` together with the matching `*_knots`, and
bumps on `linear`, `spline` or `classic` specs.
`classic` requires `fit_mean=False`, `hetero=False` and `family='none'`.
It pins its prior at the mean log length and at `zm_homo`'s fitted σ, so
pass `fits_dir=` a directory that holds a `zm_homo` fit.

The paper's four specs in this form:

| Codename | Configuration |
| --- | --- |
| `linear_sqrtfamily` | `cls='basic', kind='linear', sqrtfamily=True` |
| `linear_shslope_sqrtfamily` | `cls='basic', kind='linear', shslope=True, sqrtfamily=True` |
| `shared_shslope_sqrtfamily` (Model 1) | `cls='basic', kind='shared', shslope=True, sqrtfamily=True` |
| `s1_shslope_hetero_famspline_bb` (Model 2) | `cls='irt', fit_mean=True, hetero=True, family='famspline', bb=True, shslope=True` |

Every key not shown is False or at its default.

### The registry

The 52 registered specs are the tested set. Their codenames follow the
configuration:

| Codenames | Count |
| --- | --- |
| `linear`, `spline`, `shared`, `shared_bb`, each also with `_probit` | 8 |
| `zm_homo`, `zm_hetero`, each also with `_probit`: IRT with f(t) = t | 4 |
| `zm_homo_fam`, with optional `_bb` and `_probit` | 4 |
| `s1_{homo,hetero}[_fam \| _famspline][_bb][_probit]`: IRT with a fitted f | 24 |
| `linear_shslope`, `shared_shslope`, `shared_shslope_bb`, `s1_shslope_homo_bb`, `s1_shslope_homo_famspline_bb`, `s1_shslope_hetero_famspline_bb` | 6 |
| `classic`, `classic_bb` | 2 |
| `linear_sqrtfamily`, `linear_shslope_sqrtfamily`, `shared_sqrtfamily`, `shared_shslope_sqrtfamily` | 4 |

`make_spec` builds combinations outside the registry. They are checked
only against the constraints above. Some examples:

```python
from horizon_irt.fitting import make_spec
M2 = 's1_shslope_hetero_famspline_bb'
make_spec(M2, fit_mean=False)           # prior mean f(t) = t
make_spec(M2, mean_kint=8)              # a 10-df difficulty curve f
make_spec(M2, var_kint=4, tau_kint=1)   # size σ(t) and τ(t̄) separately
```

A dict made by `make_spec` gets a name such as
`s1_shslope_hetero_famspline_bb__mean_kint=8`, which `fit_and_save` uses
as its directory. `examples/custom_specs.py` fits these three variants
next to Model 2 and compares them.

### Sign and scale conventions

The library's parameters don't always match the paper's notation:

- **`linear`.** The library fits logit p = α_j + β_j t, so β_j < 0 for
  decreasing curves. The paper writes α_j − β_j log2 t with β_j > 0.
- **IRT.** The library fits α_j + β_j θ, so β < 0 for Model 2. The paper
  writes α_j − βθ. The θ scale is fixed by [anchoring](#anchoring).
- **`shared`.** The library's f runs from 0 to 1 over the range
  [lo, hi] of log lengths, and `params.csv` stores a_j and b on that
  scale. The paper instead anchors f to the identity at the ends of the
  range. Its f is `curves.csv`'s `f_scaled` = lo + (hi − lo)·f, its
  β = b / (hi − lo), and its α_j = a_j + b·lo / (hi − lo).

Time horizons and success probabilities don't depend on any of these
choices.

### The fitted object

A fit is a plain dict. `cls` and `model_names` are always present.
Which other keys a fit has depends on its class:

| Key | Content |
| --- | --- |
| `cls`, `model_names`, `logl` | class; AI names in fit order; log-likelihood |
| `params` | `(n_AIs, 2)`: each AI's intercept and slope. Shared slopes are repeated in every row. |
| `coefs`, `mean_basis` | IRT: the coefficients and spline basis of f |
| `gammas`, `var_basis` | IRT: the coefficients of log σ(t). Without `hetero`, there is one coefficient. |
| `gam_tau`, `tau_basis`, `tau_vec`, `tau` | IRT families: the coefficients and basis of log τ(t̄), τ for each family, and the single τ for `fam` |
| `phi` | IRT with `bb`: the beta-binomial precision φ |
| `align_c`, `align_d` | IRT: the affine map that anchoring applied |
| `accel`, `iterations`, `cycles`, `converged`, `ascent_ok`, `history` | IRT: EM diagnostics (see [Stopping and diagnostics](#stopping-and-diagnostics)) |
| `a`, `b`, `rho`, `w`, `basis` | `shared`: intercepts, slopes, ρ_j per AI (with `bb`), shape weights, and spline basis |
| `coefs`, `basis` | `spline`: coefficients for each AI, and the spline basis |
| `wlogl` | `sqrtfamily`: the weighted objective. `logl` is then the unweighted log-likelihood at the weighted optimum, so an AIC based on it is not meaningful. |
| `bumps`, `bump_fit`, `core_logl` | fits with bumps: each bump with its fitted `amplitude` (general) or per-AI `amplitudes` and run `exposure` (AI-specific), the bump fit's diagnostics, and the core's log-likelihood. `logl` is then the log-likelihood with the bumps. |
| `shape_source`, `jml` | the JML route: `'jml'`, and the first two stages' results (θ̂ and its SEs, task weights, the spline of θ̂) |

## Ablation options

Three options change parts of Models 1 and 2 for ablations. All are
configuration keys, so `make_spec`, `fit_spec`, cross-validation and
the pipeline scripts treat them like any other key.
`tests/test_ablations.py` checks them.

### Separate splines and spline sizes

- **Separate splines.** `kind='spline'` (`spline` and `spline_probit` in
  the registry) gives each AI its own increasing spline:
  logit p_j(t) = c_j − S_j(t). `mean_kint` or `mean_knots` sets its size.
- **Number of knots.** `mean_kint`, `var_kint` and `tau_kint` set the
  number of interior knots. They sit at quantiles of the unique task
  lengths (for τ, of the families' mean lengths). Each spline has
  knots + 2 degrees of freedom.
- **Knot positions.** `mean_knots`, `var_knots` and `tau_knots` place
  the interior knots explicitly, in minutes. Every knot must lie
  strictly inside the data's range of lengths, or the fit raises. In
  cross-validation that range is the training fold's, so keep knots
  away from the shortest and longest tasks.

```python
from horizon_irt.fitting import make_spec
make_spec('shared_shslope', mean_kint=8)                    # Model 1, a 10-df shape
make_spec('shared_shslope', mean_knots=(15, 60, 240, 960))  # knots at 1/4, 1, 4 and 16 h
make_spec('spline', mean_knots=(60, 240, 960))              # a 5-df spline for each AI
```

### Bumps

A bump is a smooth correction over one window [start, end] of task
lengths, in minutes. Its shape is B(t) = 16u²(1 − u)², where u is t's
position in the window on the log scale. B is zero outside the window
and 1 at the window's log-midpoint. Each bump is one of two kinds:

- **General.** It moves the difficulty curve for every AI. For IRT
  specs it is added to the prior mean of θ, m(t) = f(t) + a·B(t), with a
  in θ units. For `shared` specs (Model 1) it is added to the shape,
  f(t) + c·B(t), with c in units of the shape's 0–1 range. A positive
  amplitude makes tasks in the window harder.
- **AI-specific.** It adds d_j·B(t) to AI j's success logit. The d_j
  are departures from the common correction: their mean, weighted by
  each AI's runs in the window, is zero. To fit the common part as
  well, add a general bump over the same window.

A spec lists every bump it has. The "double shoulder" of
`late_fork_pilot` is two general bumps:

```python
from horizon_irt.bumps import bump
M2 = 's1_shslope_hetero_famspline_bb'
make_spec(M2, bumps=(bump(60, 240), bump(240, 960)))
make_spec(M2, bumps=(bump(60, 240), bump(240, 960),
                     bump(240, 960, ai_specific=True)))
make_spec('shared_shslope', bumps=(bump(120, 480, ai_specific=True,
                                        prior_sd=0.5),))
```

**The fit is anchored.** `fit_spec` first fits the core spec as usual
(and anchors it, for IRT). It then holds every core parameter fixed and
fits only the amplitudes, by maximizing the core's own likelihood plus
Gaussian priors:

- general: N(0, (prior_sd / |β|)²) on a, with |β| the core's mean
  |β_j| (mean b_j for `shared`). So prior_sd, default 2, is on the
  logit scale.
- AI-specific: N(0, prior_sd²) on each d_j, a ridge. The default
  prior_sd is 0.25.

For IRT specs the likelihood is the marginal likelihood on the EM's
quadrature, with `fit_spec`'s `K` and `Kg`, and the gradients are
exact. With every amplitude at zero the model is the core itself, and
`bump_fit['gain']` is the log-likelihood the bumps add. For `shared`
specs it is the core's cell likelihood. With `sqrtfamily` that
likelihood is weighted, with the weights rescaled to average one per
run so that the priors carry the same weight as in an unweighted fit.

With `bump_monotone=True`, the default, the general bumps must keep the
difficulty curve non-decreasing on a 600-point grid of lengths. This is
a linear constraint, solved with SLSQP. AI-specific bumps are not
constrained, so an AI's success curve can rise with length.
`bump_fit['max_marginal_rise']` reports the largest such rise. A time
horizon is then the shortest length at which the curve falls to q.

Bumps work with every IRT spec except `classic`, and with
`kind='shared'`. They add to `k_raw` one parameter per general bump and
n_AIs − 1 per AI-specific bump. Their priors shrink them, so an AIC
overstates their cost.

### Model 1 with a JML difficulty curve

`shape_source='jml'`, with `kind='shared'`, replaces Model 1's fitted
shape by one estimated from fixed-effect IRT (`horizon_irt.jml`):

1. **JML.** Fit the classic 2PL, logit p_ij = α_j + β_j θ_i, with every
   θ_i a fixed parameter: maximize the joint likelihood over (θ, α, β),
   with Bernoulli runs. Tasks that no AI solved, or that every AI always
   solved, have no finite θ̂ and are dropped (15 on the canonical
   data). `jml_drop='quasi'` also drops tasks whose failures, or whose
   successes, all come from one AI. A weak ridge on the β_j
   (`jml_beta_ridge`, default 0.01) keeps a separable AI's slope finite.
   Each θ̂_i gets the standard error 1/√(observed information), with the
   AIs held fixed.
2. **Spline.** Fit an increasing spline g of θ̂ on t by weighted least
   squares, in the shape's own basis, so `mean_kint` and `mean_knots`
   apply. `jml_weights` sets the task weights: `'invvar'` 1/SE² (the
   default), `'unweighted'`, `'sqrtfamily'` 1/√n_F, or `'totvar'`
   1/(SE² + τ²), with the between-task variance τ² estimated by
   Paule–Mandel.
3. **Model 1.** Rescale g to the shape f, which runs from 0 to 1,
   freeze it, and fit only the AIs' a_j and b (b_j without `shslope`).

Cross-validation reruns all three stages on each training fold. If the
JML sweeps don't converge, the fit raises; `jml_drop='quasi'` usually
fixes that. `jml_shape_pilot/`, in the surrounding repository, found no
variant better than Model 1 under family CV.

### Named ablation specs

`specs.ablation_specs()` names five variants outside the 52-spec grid:

| Codename | Variant |
| --- | --- |
| `shared_shslope_jml` | `shared_shslope` (Model 1's shape, fit without the sqrt-family weights) with the JML shape and 1/SE² weights |
| `shared_shslope_bump2` | `shared_shslope` plus general bumps over [1 h, 4 h] and [4 h, 16 h] |
| `shared_shslope_bump2_aibump` | the same plus an AI-specific bump over [4 h, 16 h] |
| `s1_shslope_hetero_famspline_bb_bump2` | Model 2 plus the same two general bumps |
| `s1_shslope_hetero_famspline_bb_bump2_aibump` | the same plus the AI-specific bump |

`specs.registry()` holds the grid and these five. `fit_spec`,
`make_spec` and the pipeline scripts accept any name in it. In the
scripts, `--specs ablations` selects the five, and keywords and names
can be mixed:

```bash
uv run python pipeline/fit.py --specs ablations
uv run python pipeline/evaluate.py --specs paper,ablations --out artifacts/canonical/cv_ablations
uv run python pipeline/score.py --cv artifacts/canonical/cv_ablations --models shared_shslope_jml,shared_shslope_bump2_aibump --out artifacts/canonical/scores_ablations
```

Give `evaluate.py` an `--out`: its default directory holds the paper's
cross-validation, which a run with other specs would overwrite.
`evaluate.py` needs every spec's full-data fit in
`artifacts/canonical/fits/` (from `fit.py`) and `linear` among the
specs. `score.py` needs the baseline `linear_shslope_sqrtfamily`;
`paper` includes both. To run other variants from the command line, add
them to `ablation_specs()`.

## How the optimization works

### How it is called

`fitting.fit_spec` reads the spec's configuration and calls one fitter:

```
fit_spec(spec, idx, K, Kg, max_iter, accel)
├── basic, linear  →  basic.fit_linear(idx, probit, shslope, sqrtfamily)
├── basic, spline  →  basic.fit_pmspline(idx, probit, kint)
├── basic, shared  →  basic.fit_shared(idx, probit, bb, shslope, sqrtfamily, kint)
└── irt            →  em.fit_irt(idx, fit_mean, hetero, family, bb, probit,
                                 shslope, classic, K, Kg, max_iter, accel, ...)
                      then em.anchor_identity(fit, idx)
```

It then adds `cls` and `model_names`, which `model_api` uses to dispatch.
`em.fit_irt` has further settings that `fit_spec` leaves at their
defaults:

- `sigma0=2.2`: the quadrature scale
- `sigma_init`, `tau_init`, `phi_init`: starting values
- `tol=1e-5`, `ftol=1e-9`: stopping tolerances
- `init=`: a warm start from another fit's parameters
- `verbose=True`: prints progress every 10 SQUAREM cycles, or every 25
  plain EM iterations

To use these settings, call `em.fit_irt` directly and finish the fit as
`fit_spec` does:

```python
from horizon_irt import em
fit = em.fit_irt(idx, fit_mean=True, hetero=True, family='famspline',
                 bb=True, shslope=True, verbose=True)          # Model 2
em.anchor_identity(fit, idx)
fit.update(cls='irt', model_names=list(idx['model_names']))
```

### IRT specifications: EM with quadrature (`em.fit_irt`)

The fit maximizes the marginal likelihood: the probability of all the
data, with θ and ξ integrated out. It uses EM, computing the integrals
by Gauss–Hermite quadrature, and accelerates EM with SQUAREM.

**Quadrature.** Each task gets a grid of K = 1000 nodes for θ_i,
centred on its own log length t_i with scale σ0 = 2.2. The grid never
moves. Each iteration reweights the nodes against the current prior
N(m(t_i), σ(t_i)²) by importance sampling. The family effect ξ_F gets a
second rule of Kg = 80 nodes, which are scaled by the current τ (the
default `tau0=0`). `tau0 > 0` fixes these nodes at a base scale instead,
which makes the τ M-step exact.

**E-step** (numba kernels in `kernels.py`). For each task, the E-step
weighs every node by the likelihood of the task's cells there, times the
prior-to-grid importance ratio. Normalizing these weights gives the
posterior over the nodes; the normalizing constants sum to the marginal
log-likelihood. Specs with family effects do this jointly over all the
tasks of a family, nested inside the ξ nodes. The E-step returns
posterior moments of θ_i, and with families of ξ_F and θ_i·ξ_F.

**M-steps**, in this order, each given the latest values of the others:

1. **(α_j, β_j)**: damped Newton with backtracking, separately for each
   AI, on the posterior-weighted log-likelihood. With `shslope`, one
   Newton system covers every α_j and the shared β. Its Hessian has an
   arrowhead shape, which block elimination solves cheaply.
2. **f**: weighted least squares of the posterior mean of θ_i (minus
   the family effect) on the spline basis, with weights 1/σ(t_i)². The
   basis is built so that non-negative coefficients give an increasing
   f, so a box-constrained solver (`scipy.optimize.lsq_linear`) keeps f
   monotone.
3. **log σ(t)**: concave damped Newton on the expected log-density of
   the residuals θ_i − m(t_i) − ξ_F(i). For a constant σ this has a
   closed form.
4. **τ**: τ² is the mean posterior E[ξ_F²] for `fam`. For `famspline`,
   log τ(t̄) gets the same Newton step as log σ.
5. **φ**: damped Newton on log φ, with a backtracking line search on the
   EM objective, and log φ kept within [−4, 20].

**Floors.** σ ≥ 0.3, so the prior never gets narrower than the node
spacing can resolve. τ ≥ 1e-3.

**Start.**

- (α_j, β_j) start from a logistic fit of each AI on θ = t.
- f starts from the least-squares fit to the identity.
- σ, τ and φ start at 2, 1 and 5.

#### Acceleration: SQUAREM

One E-step followed by the M-steps is one **EM update**: a map F from
the current parameters to new ones. Plain EM just iterates F. Near the
optimum, each update shrinks the remaining error by a roughly constant
factor. When that factor is close to 1, plain EM crawls. In Model 2 this
happens to the family spread τ(t̄) at its low end, which is still
falling when plain EM stops.

SQUAREM (Varadhan and Roland, *Scandinavian Journal of Statistics* 35,
2008) extrapolates along the path that F traces. `em.run_squarem` runs
it in cycles:

1. From the current parameters x0, take two EM updates:
   x1 = F(x0) and x2 = F(x1).
2. With r = x1 − x0 and v = x2 − 2·x1 + x0, jump to
   x' = x0 + 2a·r + a²·v. The step length a = |r| / |v| (the SqS3
   scheme) is clipped to [1, a_max]. a = 1 gives x' = x2, which is
   plain EM.
3. Take one more update from the jump, x'' = F(x'). Its E-step also
   gives the log-likelihood at x'. The cycle keeps x'' if logL(x') is at
   least logL(x0), less a slack of 1e-9·|logL(x0)|
   (`SQUAREM_LL_SLACK`). Otherwise it keeps x2.

Details:

- **Step cap.** a_max starts at 1, so the first cycle is plain EM. It
  quadruples each time a reaches it, so steps lengthen only while they
  keep succeeding. A rejected jump at a = a_max divides it by 4.
- **The parameter vector.** x holds (α_j, β_j), the coefficients of f,
  the coefficients of log σ(t), log τ (or the coefficients of log τ(t̄)
  for `famspline`) and log φ. After a jump, f's coefficients are
  projected back onto the box that keeps f increasing, and log φ onto
  [−4, 20]. The σ and τ floors apply as in plain EM.
- **Why the slack.** The (α_j, β_j) M-step clips success probabilities
  to [1e-6, 1 − 1e-6] in its gradient, but the logit-Bernoulli
  likelihood does not clip them. So EM's fixed point sits slightly below
  the log-likelihood maximum (by about 3e-8 on a `zm_homo` fit of
  the earlier 21-AI data), and from the maximum each EM update lowers the log-likelihood a
  little. Without the slack, SQUAREM reached the maximum and then
  rejected every jump.
- **Cost.** A cycle costs three EM updates, or two when a = 1. The jump
  itself costs nothing by comparison.
- **Plain EM.** `accel='none'` runs plain EM (`em.run_plain_em`). The
  shipped artifacts were made with SQUAREM, the default.

#### Stopping and diagnostics

Both drivers apply the same tests to the EM update that starts each
plain EM iteration or SQUAREM cycle. They stop at the first of three
conditions:

- no parameter changes by more than `tol` = 1e-5 in that update. The
  parameters checked are α_j, β_j, the prior means m(t_i), σ(t_i), τ
  and φ.
- the relative change in log-likelihood since the previous iteration or
  cycle falls below `ftol` = 1e-9
- `max_iter` EM updates have run

The fit records:

- `accel`: `'squarem'` or `'none'`
- `iterations`: the number of EM updates. Each update is one E-step,
  so this number sets the run time.
- `cycles`: the number of SQUAREM cycles (`None` for plain EM)
- `converged`: whether it stopped before the cap
- `ascent_ok`: whether the log-likelihood never fell by more than
  rounding error (1e-6 of |logL|). Under SQUAREM, this checks the
  log-likelihood at the start of each cycle.
- `history`: the log, written as `history.csv`. It has one row per plain
  EM iteration, or per SQUAREM cycle. Its columns are `iter`, `logl`,
  `max_change` and `rel_ll_change`, plus `tau_min` and `tau_max` for
  family specs. Under SQUAREM, `iter` counts cycles, and three more
  columns record `em_updates` (updates so far), `alpha` (the step length
  tried; 1 for a plain cycle) and `extrapolated` (whether the jump was
  kept; False with `alpha` > 1 means it was rejected).

Model 2 with each driver, on the canonical data and on the 21-AI data
the paper was first fit on (the TH 1.1 runs.jsonl plus the same TH 1.0
graft; not shipped here, called legacy below):

| Data | Plain EM | SQUAREM |
| --- | --- | --- |
| canonical | 240 updates, about 5 minutes | 66 updates (23 cycles), about a minute |
| legacy | stops at the 300-update cap without converging | 69 updates (24 cycles), about a minute |

Both drivers stop while the low end of τ(t̄) is still falling: plain EM
on the legacy data at the cap, and otherwise on the `ftol` test. So the
two fits differ slightly. SQUAREM's log-likelihood is higher, by 1.4e-4
on the canonical data and 3.1e-4 on the legacy data. The 50% and 80%
time horizons differ by at most 0.04% on the canonical data and 0.2% on
the legacy data. In the canonical 5-fold CV, Model 2's scores move by at
most 1e-4 (log score), 4e-5 (Brier) and 7e-4 (ECE), and no spec changes
rank on any of them.

Across the 37 IRT specs of the full grid on the legacy data, SQUAREM
converged on all 37, and plain EM on 26 (the rest reached the cap).
SQUAREM took 2–4 times fewer EM updates and about half the total run
time. It reached a higher log-likelihood on 35 specs, and the 50% and
80% horizons of the logit specs moved by at most 1.3%.

The other two are probit specs, `zm_homo_probit` and
`zm_homo_fam_probit`. There SQUAREM converged to a point about 0.05
below plain EM's log-likelihood, with horizons up to 2.9% apart. The
cause is a fault in the probit (α_j, β_j) M-step, not in SQUAREM. That
M-step clips p to [1e-6, 1 − 1e-6] in its derivatives, and in the
probit tails this makes them far too small. The Newton direction it
builds can then fail to raise the true objective. When that happens,
the line search rejects every step and the M-step returns its input
unchanged, so EM stops there although the log-likelihood still has a
clear gradient. SQUAREM's jumps reached such a point; plain EM's path
did not. With exact probit derivatives (inverse Mills ratios), the two
drivers agree on both specs, and both gain about 2 in log-likelihood
over the plain-EM fits. That fix is not applied yet, so treat
the probit IRT fits with either driver as approximate. The logit and
beta-binomial M-steps, which every paper spec uses, did not stall in
any of these fits.

SQUAREM pays off at the default tolerances. With `tol=1e-8` and
`ftol=0` on the legacy `zm_homo` fit, it reached the log-likelihood
maximum in 50 updates. Its jumps then amplified the M-steps' small
inexactness, and it needed 558 updates to bring every parameter change
below 1e-8, as many as plain EM.

#### Anchoring

The likelihood doesn't change under any increasing affine relabelling
θ → (θ − d)/c, because f, σ, τ, α_j and β absorb it. `em.anchor_identity`
fixes the scale after fitting by requiring f(lo) = lo and f(hi) = hi at
the ends of the data's length range. It records the map it applied in
`align_c` and `align_d`. CV fold fits are scored unanchored, since
anchoring changes no prediction. The figures re-anchor Model 2 by least
squares of f(t_i) against t_i instead (`make_figs/model2.py`), which
also leaves the time horizons unchanged.

### Basic specifications (`basic.py`)

These fits maximize the likelihood of independent runs, or its
sqrt-family-weighted version.

- **Unweighted `linear`**: damped Newton for each AI. It uses the same
  kernel as the IRT (α, β) step, on a single node at θ = t.
- **Weighted (`sqrtfamily`) or shared-slope `linear`**: L-BFGS-B
  (`scipy.optimize.minimize`) with analytic gradients. Both objectives
  are concave, so one solve suffices.
- **`spline`**: L-BFGS-B for each AI on the spline coefficients, bounded
  so the curve decreases, with a small ridge penalty (1e-4).
- **`shared`** (Model 1): block coordinate ascent.
  - f is a convex combination of the spline basis columns, each
    normalized to rise from 0 to 1. Its weights are a softmax of free
    parameters, so f is increasing and runs from 0 to 1 by
    construction.
  - Each round has three parts:
    - It fits each AI's own parameters by L-BFGS-B: a_j, b_j unless
      `shslope`, and ρ_j with `bb`.
    - With `shslope`, it then fits the shared b.
    - It then fits the shape weights.
  - Rounds stop when the objective changes by less than 1e-7
    (relative), or after 200 rounds.
  - Optimizing all of these jointly converges poorly, so the blocks are
    deliberately kept separate.

### Predictions (`model_api`)

- **Basic fits.** These evaluate their curve directly.
- **IRT fits.** These integrate θ over the fitted prior with 200
  Gauss–Hermite nodes, folding the family effect into the prior's
  spread. For `famspline`, `marginal_p` evaluates τ at t̄ = t by default
  (pass `fam_tbar` to change this). This is the paper's convention for a
  fresh task from a fresh family. Scoring held-out cells uses each
  task's actual family mean length instead.
- **Time horizons.** Linear fits invert in closed form. All others
  invert p_j(t) numerically on a 481-point grid spanning the data's
  range.

### Cost

- The numba kernels compile on first use and are cached in
  `src/horizon_irt/__pycache__`, so the first fit in a fresh checkout
  takes longer.
- The kernels within a fit run on one thread. The pipeline parallelizes
  across specs and CV folds instead (`NJOB`).
- Run time grows with K and Kg. `examples/custom_specs.py --quick` uses
  K = 200, Kg = 40 and 40 EM updates to finish in about a minute,
  with rougher numbers.
