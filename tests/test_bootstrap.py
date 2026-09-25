"""
The vectorized bootstrap reproduces the sequential implementation: same draws, same estimates, same
failure accounting.
"""
import numpy as np
import pytest

from pyprocessmacro import Process
from pyprocessmacro import bootstrap as bs

pytestmark = pytest.mark.smoke


def _spec(model):
    return bs.BootstrapSpec(
        model._ind_y, model._exog_inds_y, model._inds_m, model._exog_inds_m,
        model._options["logit"], model._options["iterate"], model._options["convergence"],
    )


def _sequential(data, spec, n_boots, seed):
    """The 1.x loop, as a reference: one resample at a time with the scalar estimators."""
    sampler = bs.bootstrap_sampler(data.shape[0], seed)
    betas_y, betas_m, n_fail = [], [], 0
    while len(betas_y) < n_boots:
        sample = np.asarray(data, dtype=float)[next(sampler)]
        by, bm, failed = bs._fit_one_by_one(sample[None], spec)
        if failed[0]:
            n_fail += 1
            continue
        betas_y.append(by[0])
        betas_m.append(bm[:, 0])
    return np.array(betas_y), np.transpose(np.array(betas_m), (1, 0, 2)), n_fail


@pytest.mark.parametrize("kwargs", [
    dict(model=7, x="effort", w="motiv", m=["med1", "med2"], y="outcome"),
    dict(model=4, x="effort", m=["med1"], y="binary", logit=True),
])
def test_vectorized_bootstrap_matches_the_sequential_loop(fit, kwargs):
    p = fit(boot=300, seed=11, **kwargs)
    model = p.indirect_model
    spec = _spec(model)
    ref_y, ref_m, ref_fail = _sequential(model._data, spec, 300, 11)
    np.testing.assert_allclose(model._boot_betas_y, ref_y, rtol=1e-7, atol=1e-10)
    np.testing.assert_allclose(model._boot_betas_m, ref_m, rtol=1e-7, atol=1e-10)
    assert model._n_fail_samples == ref_fail == 0
    # Chunking does not change the result either.
    small_y, small_m, _, _ = bs.bootstrap_parameters(model._data, spec, 300, 11, chunk_size=7)
    np.testing.assert_allclose(small_y, model._boot_betas_y, rtol=1e-10)
    np.testing.assert_allclose(small_m, model._boot_betas_m, rtol=1e-10)


def test_failed_resamples_are_counted_like_before(fit, data):
    # A control that is 1 for a single row: resamples without that row have a constant column,
    # so their normal equations are singular and the resample is discarded.
    df = data.iloc[:40].copy()
    df["flag"] = 0.0
    df.iloc[0, df.columns.get_loc("flag")] = 1.0
    p = fit(4, df=df, x="effort", m=["med1"], y="outcome", controls=["flag"], boot=200, seed=3)
    model = p.indirect_model
    ref_y, ref_m, ref_fail = _sequential(model._data, _spec(model), 200, 3)
    assert model._n_fail_samples == ref_fail > 0
    np.testing.assert_allclose(model._boot_betas_y, ref_y, rtol=1e-7, atol=1e-8)
    np.testing.assert_allclose(model._boot_betas_m, ref_m, rtol=1e-7, atol=1e-8)


def test_same_seed_same_draws_as_2_0(fit):
    # The first resample is the first draw of RandomState(seed).randint(n, size=n), as in every earlier version.
    p = fit(4, x="effort", m=["med1"], y="outcome", boot=5, seed=2024)
    model = p.indirect_model
    first = np.random.RandomState(2024).randint(model._n_obs, size=model._n_obs)
    sample = np.asarray(model._data, dtype=float)[first]
    expected = bs.fast_OLS(sample[:, model._ind_y], sample[:, model._exog_inds_y])
    np.testing.assert_allclose(model._boot_betas_y[0], expected, rtol=1e-10)


def test_separable_resample_is_flagged_as_failed():
    # Columns: Cons, x, binary y, m. The batched solver hits a singular Hessian on the separable sample,
    # _fit_chunk falls back to the one-by-one path, and the resample is flagged.
    n = 50
    x = np.linspace(-3, 3, n)
    spec = bs.BootstrapSpec(ind_y=2, exog_inds_y=[0, 1], inds_m=[3], exog_inds_m=[0, 1],
                            logit=True, max_iter=200, tolerance=1e-10)
    separable = np.column_stack([np.ones(n), x, (x > 0).astype(float), x + 1.0])[None]
    _, _, failed = bs._fit_chunk(separable, spec)
    assert failed.tolist() == [True]
    rng = np.random.default_rng(0)
    gx = rng.normal(size=n)
    gy = (rng.random(n) < 1 / (1 + np.exp(-gx))).astype(float)
    fine = np.column_stack([np.ones(n), gx, gy, gx + rng.normal(size=n)])[None]
    params, _, failed = bs._fit_chunk(fine, spec)
    assert failed.tolist() == [False] and np.isfinite(params).all()





def test_perfect_separation_is_reported_even_when_the_score_vanishes():
    from pyprocessmacro.utils import ConvergenceError, fast_optimize

    n = 60
    x = np.linspace(-3, 3, n)
    exog = np.column_stack([np.ones(n), x])
    endog = (x > 0).astype(float)
    with pytest.raises(ConvergenceError):
        fast_optimize(endog, exog, n_obs=n, n_vars=2, max_iter=5000, tolerance=1e-10)
    # A saturated but "converged" batch is flagged too.
    params = np.array([[0.0, 1000.0]])  # predicts every outcome to within 1e-8 on this grid
    fitted = bs._logit_cdf(np.einsum("cnk,ck->cn", exog[None], params))
    assert np.abs(endog[None] - fitted).max() < 1e-8
    _, failed = bs._batch_logit(endog[None], exog[None], max_iter=0, tolerance=1e-10)
    assert failed.tolist() == [True]
