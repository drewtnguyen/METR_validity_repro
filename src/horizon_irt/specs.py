"""The headline model grid, by codename.

The original grid also had four latent-class mixture specs
(s{2,3}_homo_{shared,free}); they are not in the paper and are not part
of this package, which leaves 52 specifications.

The paper's four:
  linear_sqrtfamily               METR's per-AI logistic regression
                                  (sqrt-family weighted, as METR fits it)
  linear_shslope_sqrtfamily       shared-slope logistic: the baseline
  shared_shslope_sqrtfamily       Model 1 (sqrt-family weighted, like
                                  the baseline)
  s1_shslope_hetero_famspline_bb  Model 2
"""

PAPER_SPECS = ('linear_sqrtfamily', 'linear_shslope_sqrtfamily',
               'shared_shslope_sqrtfamily', 's1_shslope_hetero_famspline_bb')
# What the pipeline fits and evaluates by default: the paper's four plus
# unweighted `linear`, the reference spec of the stored CV contrasts.
PIPELINE_SPECS = ('linear',) + PAPER_SPECS
# ablation_specs() below names a few variants outside the grid.


def all_specs() -> dict:
    specs = {}
    # basic (marginal) models
    for kind, bb in [('linear', False), ('spline', False),
                     ('shared', False), ('shared', True)]:
        for probit in (False, True):
            name = kind + ('_bb' if bb else '') + \
                ('_probit' if probit else '')
            specs[name] = dict(cls='basic', kind=kind, bb=bb, probit=probit)
    # zm IRT
    for het in (False, True):
        for probit in (False, True):
            name = ('zm_hetero' if het else 'zm_homo') + \
                ('_probit' if probit else '')
            specs[name] = dict(cls='irt', fit_mean=False, hetero=het,
                               family='none', bb=False, probit=probit)
    # zm + scalar family effect: theta_i = t_i + xi_F + z_i,
    # xi_F ~ N(0, tau^2) -- identity mean still frozen (no ridge).
    # Bespoke homo cells only (+ BB), not a zm x family cross-product.
    for bb in (False, True):
        for probit in (False, True):
            name = 'zm_homo_fam' + ('_bb' if bb else '') + \
                ('_probit' if probit else '')
            specs[name] = dict(cls='irt', fit_mean=False, hetero=False,
                               family='fam', bb=bb, probit=probit)
    # s1 IRT crossing
    for het in (False, True):
        for fam in ('none', 'fam', 'famspline'):
            for bb in (False, True):
                for probit in (False, True):
                    name = 's1_hetero' if het else 's1_homo'
                    if fam != 'none':
                        name += f'_{fam}'
                    if bb:
                        name += '_bb'
                    if probit:
                        name += '_probit'
                    specs[name] = dict(cls='irt', fit_mean=True,
                                       hetero=het, family=fam, bb=bb,
                                       probit=probit)
    # shslope: a single discrimination/slope shared across all AI models
    # (basic: one shared slope b on x; IRT: one shared beta on theta).
    # Bespoke additions -- only these two cells, not the full cross-product.
    specs['linear_shslope'] = dict(cls='basic', kind='linear', bb=False,
                                   probit=False, shslope=True)
    specs['s1_shslope_homo_bb'] = dict(cls='irt', fit_mean=True, hetero=False,
                                       family='none', bb=True, probit=False,
                                       shslope=True)
    # shared-beta IRT with the famspline family effect (shared tau(t_bar))
    specs['s1_shslope_homo_famspline_bb'] = dict(
        cls='irt', fit_mean=True, hetero=False, family='famspline',
        bb=True, probit=False, shslope=True)
    # heteroskedastic variant: shared beta + sigma(t) + tau(t_bar) + BB
    specs['s1_shslope_hetero_famspline_bb'] = dict(
        cls='irt', fit_mean=True, hetero=True, family='famspline',
        bb=True, probit=False, shslope=True)
    # shared-shape mean with a single shared slope b (per-model a_j[, rho_j])
    specs['shared_shslope'] = dict(cls='basic', kind='shared', bb=False,
                                   probit=False, shslope=True)
    specs['shared_shslope_bb'] = dict(cls='basic', kind='shared', bb=True,
                                      probit=False, shslope=True)
    # classic: standard 2PL IRT with a FIXED task-length-independent prior
    # theta ~ N(t-bar, sigma_zm^2), pinned constants (t-bar = mean log2
    # minutes, sigma_zm = zm_homo's fitted sigma -- so classic fits AFTER
    # zm_homo; see em.classic_prior_constants).  Affinely equivalent to
    # theta~N(0,1) but numerically matched to the fixed sigma0 grid.
    # Prior carries no task-length info => marginal p constant in x =>
    # time horizons undefined (blank horizon plot).  classic_bb adds
    # global beta-binomial overdispersion.
    specs['classic'] = dict(cls='irt', fit_mean=False, hetero=False,
                            family='none', bb=False, probit=False,
                            classic=True)
    specs['classic_bb'] = dict(cls='irt', fit_mean=False, hetero=False,
                               family='none', bb=True, probit=False,
                               classic=True)
    # sqrtfamily: the SAME linear models fit by maximizing the
    # sqrt_family-WEIGHTED log-lik (the METR paper's diversity
    # weighting, w_run = 1/(sqrt(n_F)*n_ij)) -- a different estimator,
    # not a different model.  fit.json's 'logl' is the unweighted
    # log-lik realized at the weighted optimum (comparable scale; its
    # AIC is not meaningful), 'wlogl' in fit.pkl is the objective.
    specs['linear_sqrtfamily'] = dict(cls='basic', kind='linear',
                                      bb=False, probit=False,
                                      sqrtfamily=True)
    specs['linear_shslope_sqrtfamily'] = dict(cls='basic', kind='linear',
                                              bb=False, probit=False,
                                              shslope=True,
                                              sqrtfamily=True)
    specs['shared_sqrtfamily'] = dict(cls='basic', kind='shared',
                                      bb=False, probit=False,
                                      sqrtfamily=True)
    # Model 1 of the paper: the shared-shape, shared-slope fit with the
    # same sqrt-family weighting as the baseline.
    specs['shared_shslope_sqrtfamily'] = dict(cls='basic', kind='shared',
                                              bb=False, probit=False,
                                              shslope=True,
                                              sqrtfamily=True)
    assert len(specs) == 52, len(specs)
    return specs


# ----------------------------------------------------------------------
# ablation variants: named, but not part of the 52-spec grid
# ----------------------------------------------------------------------
# The bump windows are late_fork_pilot's: its "double shoulder"
# (family_double_shoulder_no_fork) is two general bumps over [1 h, 4 h]
# and [4 h, 16 h], and family_r1_shoulder adds AI-specific departures
# over [4 h, 16 h].  See horizon_irt.bumps.
DOUBLE_SHOULDER = ({'start': 60.0, 'end': 240.0, 'ai_specific': False},
                   {'start': 240.0, 'end': 960.0, 'ai_specific': False})
AI_SHOULDER = ({'start': 240.0, 'end': 960.0, 'ai_specific': True},)


def ablation_specs() -> dict:
    """Named ablation variants of Models 1 and 2.

      shared_shslope_jml          unweighted Model 1 with its shape frozen at a
                                  monotone spline of JML difficulties,
                                  weighted 1/SE^2 (horizon_irt.jml)
      <model>_bump2               the double shoulder, anchored on <model>
      <model>_bump2_aibump        the double shoulder plus AI-specific
                                  departures over [4 h, 16 h]

    for <model> in shared_shslope (Model 1's shape, fit without the
    sqrt-family weights) and s1_shslope_hetero_famspline_bb (Model 2).
    Other variants are
    built with fitting.make_spec; add them here to run them from the
    pipeline scripts."""
    grid = all_specs()
    model1 = grid['shared_shslope']
    model2 = grid['s1_shslope_hetero_famspline_bb']
    specs = {'shared_shslope_jml': dict(model1, shape_source='jml',
                                        jml_weights='invvar',
                                        jml_drop='extreme')}
    for name, base in (('shared_shslope', model1),
                       ('s1_shslope_hetero_famspline_bb', model2)):
        specs[name + '_bump2'] = dict(base, bumps=DOUBLE_SHOULDER)
        specs[name + '_bump2_aibump'] = dict(
            base, bumps=DOUBLE_SHOULDER + AI_SHOULDER)
    return specs


ABLATION_SPECS = tuple(ablation_specs())


def registry() -> dict:
    """Every named spec: the 52-spec grid plus the ablation variants."""
    return {**all_specs(), **ablation_specs()}


# famspline cells are the slow path -- fit them last (skill note).
# classic cells sort last too: their pinned prior sigma is read from the
# zm_homo fit, which must exist first.  (With parallel workers this
# ordering is dispatch order, not a hard barrier -- zm_homo is an early,
# fast spec, so in practice it is long finished before classic starts.)
def fit_order(names):
    return sorted(names, key=lambda n: ('famspline' in n
                                        or n.startswith('classic'), n))


if __name__ == '__main__':
    s = all_specs()
    print(len(s), 'specs:')
    for n in fit_order(s):
        print(' ', n)
