"""
Negative binomial outcome (#25): family="negbin", a PyProcessMacro extension that PROCESS does not offer.

The estimator is checked against statsmodels' NegativeBinomial and against numerical derivatives; the
mediation machinery is checked to carry the negative binomial b path through the point estimates, the
bootstrap, the summaries and the standardized exports.
"""
import numpy as np
import pandas as pd
import pytest
from scipy import optimize

from pyprocessmacro import Process, bootstrap as bs, negbin

M4 = dict(x="effort", m=["med1"], y="count")


def _design(data, columns):
    return np.column_stack([np.ones(len(data))] + [data[c].to_numpy(dtype=float) for c in columns])


# --- the estimator -------------------------------------------------------------------------------------

def test_fit_matches_statsmodels(fit, data):
    sm = pytest.importorskip("statsmodels.api")
    p = fit(4, family="negbin", **M4)
    res = p.outcome_models["count"].estimation_results
    exog = pd.DataFrame(dict(Cons=1.0, effort=data["effort"], med1=data["med1"]))
    theirs = sm.NegativeBinomial(data["count"], exog).fit(disp=0, maxiter=500)
    np.testing.assert_allclose(res["betas"], theirs.params.iloc[:-1].to_numpy(), rtol=1e-4, atol=1e-6)
    assert res["alpha"] == pytest.approx(theirs.params.iloc[-1], rel=1e-3)
    np.testing.assert_allclose(res["se"], theirs.bse.iloc[:-1].to_numpy(), rtol=1e-3)
    assert res["alpha_se"] == pytest.approx(theirs.bse.iloc[-1], rel=1e-2)
    assert res["llf"] == pytest.approx(theirs.llf, rel=1e-6)
    assert res["llnull"] == pytest.approx(theirs.llnull, rel=1e-4)
    assert res["names"] == ["Cons", "effort", "med1"]


def test_score_and_hessian_match_numerical_derivatives(data):
    y = data["count"].to_numpy(dtype=float)[None]
    x = _design(data, ["effort", "med1"])[None]
    theta = np.array([0.3, 0.2, 0.1, -0.5])

    def ll(t):
        return negbin.loglike(y, x, t[None])[0]

    def grad(t):
        return negbin.score_hessian(y, x, t[None])[0][0]

    analytic_grad, analytic_hess = negbin.score_hessian(y, x, theta[None])
    np.testing.assert_allclose(analytic_grad[0], optimize.approx_fprime(theta, ll, 1e-6), rtol=1e-4, atol=1e-3)
    numerical_hess = np.column_stack([optimize.approx_fprime(theta, lambda t, j=j: grad(t)[j], 1e-6) for j in range(4)])
    np.testing.assert_allclose(analytic_hess[0], numerical_hess, rtol=1e-4, atol=1e-2)


def test_batch_fit_matches_single_fits(data):
    rng = np.random.default_rng(0)
    n = len(data)
    idx = rng.integers(0, n, size=(5, n))
    y = data["count"].to_numpy(dtype=float)[idx]
    x = _design(data, ["effort", "med1"])[idx]
    params, failed = negbin.batch_fit(y, x, 500, 1e-10)
    assert not failed.any()
    for i in range(5):
        np.testing.assert_allclose(params[i], negbin.fit(y[i], x[i], 500, 1e-10), rtol=1e-6, atol=1e-8)
        assert negbin.loglike(y[i:i + 1], x[i:i + 1], params[i:i + 1])[0] >= negbin.loglike(
            y[i:i + 1], x[i:i + 1], params[i:i + 1] + np.array([[0.01, 0, 0, 0]]))[0]


def test_scipy_fallback_agrees_with_newton(data):
    y = data["count"].to_numpy(dtype=float)
    x = _design(data, ["effort", "med1"])
    np.testing.assert_allclose(negbin._fit_scipy(y, x), negbin.fit(y, x), rtol=1e-4, atol=1e-5)


def test_poisson_like_counts_fit_with_a_small_dispersion(data):
    rng = np.random.default_rng(1)
    df = data.copy()
    df["pois"] = rng.poisson(np.exp(0.5 + 0.3 * df["effort"]))
    p = Process(df, 4, x="effort", m=["med1"], y="pois", family="negbin", boot=50, seed=1, suppr_init=True)
    res = p.outcome_models["pois"].estimation_results
    assert np.isfinite(res["betas"]).all() and 0 < res["alpha"] < 0.05


def test_bootstrap_chunk_fits_the_negative_binomial(data):
    rng = np.random.default_rng(2)
    n = len(data)
    array = np.column_stack([np.ones(n), data["effort"], data["count"], data["med1"]])
    chunk = array[rng.integers(0, n, size=(4, n))]
    spec = bs.BootstrapSpec(ind_y=2, exog_inds_y=[0, 1, 3], inds_m=[3], exog_inds_m=[0, 1],
                            family="negbin", max_iter=500, tolerance=1e-10)
    betas_y, betas_m, failed = bs._fit_chunk(chunk, spec)
    assert not failed.any() and betas_y.shape == (4, 3) and betas_m.shape == (1, 4, 2)
    for i in range(4):
        np.testing.assert_allclose(betas_y[i], negbin.fit_betas(chunk[i][:, 2], chunk[i][:, [0, 1, 3]]), rtol=1e-6, atol=1e-8)
    one_by_one = bs._fit_one_by_one(chunk, spec)
    np.testing.assert_allclose(one_by_one[0], betas_y, rtol=1e-6, atol=1e-8)


# --- the mediation machinery ---------------------------------------------------------------------------

def test_model_4_indirect_effect_is_a_times_b(fit):
    p = fit(4, family="negbin", **M4)
    a = p.outcome_models["med1"].coeff_summary().loc["effort", "coeff"]
    b = p.outcome_models["count"].coeff_summary().loc["med1", "coeff"]
    table = p.indirect_model.coeff_summary()
    assert table["Effect"].iloc[0] == pytest.approx(a * b)
    assert np.isfinite(table[["Boot SE", "BootLLCI", "BootULCI"]].to_numpy(dtype=float)).all()
    assert table["BootLLCI"].iloc[0] < a * b < table["BootULCI"].iloc[0]
    direct = p.direct_model.coeff_summary()
    assert list(direct.columns) == ["Effect", "SE", "Z", "p", "LLCI", "ULCI"]
    assert direct["Effect"].iloc[0] == pytest.approx(p.outcome_models["count"].coeff_summary().loc["effort", "coeff"])
    draws = p.get_bootstrap_estimates()
    assert list(draws.columns) == ["BootSample", "OutcomeName", "Cons", "effort", "med1"]
    assert len(draws) == 2 * p.options["boot"]


def test_summary_names_the_estimator_and_the_extension(fit):
    p = fit(4, family="negbin", **M4)
    text = p.summary()
    assert "Negative Binomial Regression Summary" in text
    assert "alpha" in text
    assert "negative binomial regression" in text.splitlines()[0] and "PROCESS does not offer" in text.splitlines()[0]
    assert "Negative binomial" in p._repr_html_() or "negative binomial" in p._repr_html_()


def test_tidy_glance_augment_and_statsmodels(fit, data):
    sm = pytest.importorskip("statsmodels.api")
    p = fit(4, family="negbin", **M4)
    g = p.glance().set_index("outcome")
    assert g.loc["count", "estimator"] == "negbin"
    assert g.loc["count", "alpha"] > 0 and np.isnan(g.loc["med1", "alpha"])
    assert g.loc["count", "mcfadden"] == pytest.approx(p.outcome_models["count"].estimation_results["mcfadden"])
    rows = p.tidy(component="outcome")
    assert set(rows.loc[rows["outcome"] == "count", "method"]) == {"negbin"}
    aug = p.augment("count")
    res = p.outcome_models["count"].estimation_results
    fitted = np.exp(_design(data, ["effort", "med1"]) @ res["betas"])
    np.testing.assert_allclose(aug[".fitted_count"].to_numpy(), fitted)
    np.testing.assert_allclose(aug[".resid_count"].to_numpy(), data["count"].to_numpy() - fitted)
    refit = p.to_statsmodels()["count"]
    assert type(refit.model).__name__ == "NegativeBinomial"
    np.testing.assert_allclose(refit.params.iloc[:-1].to_numpy(), res["betas"], rtol=1e-4, atol=1e-6)


def test_moderated_paths_use_z_tests_and_a_likelihood_ratio_probe(fit):
    one = fit(1, x="effort", m="qual", y="count", family="negbin")
    assert "Z" in one.direct_model.coeff_summary().columns
    assert 0 <= one.direct_model.probe_p <= 1
    five = fit(5, x="effort", w="motiv", m=["med1"], y="count", family="negbin")
    assert "Z" in five.direct_model.coeff_summary().columns
    assert len(five.direct_model.coeff_summary()) == 3
    seven = fit(7, x="effort", w="motiv", m=["med1"], y="count", family="negbin")
    assert seven.indirect_model._analysis_list == ["MM"]
    assert np.isfinite(seven.indirect_model.MM_index_summary()["Index"].to_numpy(dtype=float)).all()


def test_serial_model_with_a_count_outcome(fit):
    p = fit(6, x="effort", m=["med1", "med2"], y="count", family="negbin", total=True)
    table = p.indirect_model.coeff_summary()
    assert np.isfinite(table[["Effect", "BootLLCI", "BootULCI"]].to_numpy(dtype=float)).all()
    assert "Negative Binomial Regression Summary" in p.summary()


# --- the option ----------------------------------------------------------------------------------------

def test_family_resolves_and_validates(fit, data):
    assert fit(4, **M4).options["family"] == "ols"
    assert fit(4, logit=True, x="effort", m=["med1"], y="binary").options["family"] == "logit"
    q = fit(4, family="logit", x="effort", m=["med1"], y="binary")
    assert q.options["logit"] is True and "Logistic Regression Summary" in q.summary()
    assert fit(4, family="negbin", **M4).options["logit"] is False
    with pytest.raises(ValueError, match="'family'"):
        fit(4, family="poisson", **M4)
    with pytest.raises(ValueError, match="disagree"):
        fit(4, logit=True, family="negbin", **M4)
    with pytest.raises(ValueError, match="non-negative integers"):
        fit(4, family="negbin", x="effort", m=["med1"], y="outcome")
    df = data.copy()
    df["zeros"] = 0
    with pytest.raises(ValueError, match="positive count"):
        fit(4, df=df, family="negbin", x="effort", m=["med1"], y="zeros")
    with pytest.raises(ValueError, match="effsize"):
        fit(4, family="negbin", effsize=True, **M4)
