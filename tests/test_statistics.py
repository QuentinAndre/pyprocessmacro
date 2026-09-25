"""
Statistics cross-checked against statsmodels on synthetic data.

The accuracy suite compares to PROCESS output at n = 1000 with loose
tolerances, which cannot tell t from z or catch degree-of-freedom slips.
These tests use tight tolerances and small samples.
"""
import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

pytestmark = pytest.mark.smoke


SPEC4 = dict(x="effort", m=["med1"], y="outcome")


def design(df, columns):
    return sm.add_constant(df[list(columns)])


# --- #40: OLS intervals use t critical values ----------------------------------------------


@pytest.mark.parametrize("n", [20, 300])
def test_ols_coefficient_intervals_match_statsmodels(fit, data, n):
    df = data.iloc[:n]
    p = fit(4, df=df, x="effort", m=["med1"], y="outcome", boot=50)
    ours = p.outcome_models["outcome"].coeff_summary()
    ref = sm.OLS(df["outcome"], design(df, ["effort", "med1"])).fit()
    ci = ref.conf_int(alpha=0.05)
    for name, ref_name in [("Cons", "const"), ("effort", "effort"), ("med1", "med1")]:
        assert ours.loc[name, "coeff"] == pytest.approx(ref.params[ref_name], rel=1e-8)
        assert ours.loc[name, "se"] == pytest.approx(ref.bse[ref_name], rel=1e-8)
        assert ours.loc[name, "p"] == pytest.approx(ref.pvalues[ref_name], rel=1e-6)
        assert ours.loc[name, "LLCI"] == pytest.approx(ci.loc[ref_name, 0], rel=1e-8)
        assert ours.loc[name, "ULCI"] == pytest.approx(ci.loc[ref_name, 1], rel=1e-8)
    # Without moderation the direct effect is the coefficient on X, interval included.
    direct = p.direct_model.coeff_summary().iloc[0]
    assert direct["LLCI"] == pytest.approx(ci.loc["effort", 0], rel=1e-8)
    assert direct["ULCI"] == pytest.approx(ci.loc["effort", 1], rel=1e-8)


def test_conditional_direct_effect_interval_matches_statsmodels_contrast(fit, data):
    df = data.iloc[:40]
    p = fit(5, df=df, x="effort", w="motiv", m=["med1"], y="outcome", boot=50)
    w0 = 0.7
    ours = p.spotlight_direct_effect(spotval={"motiv": [w0]}).iloc[0]
    # Outcome equation terms in the order the package uses: Cons, x, m1, w, x*w
    frame = df.assign(**{"effort*motiv": df["effort"] * df["motiv"]})
    ref = sm.OLS(df["outcome"], design(frame, ["effort", "med1", "motiv", "effort*motiv"])).fit()
    contrast = ref.t_test(np.array([[0.0, 1.0, 0.0, 0.0, w0]]))
    low, high = contrast.conf_int(alpha=0.05)[0]
    assert ours["Effect"] == pytest.approx(float(np.ravel(contrast.effect)[0]), rel=1e-8)
    assert ours["SE"] == pytest.approx(float(np.ravel(contrast.sd)[0]), rel=1e-8)
    assert ours["LLCI"] == pytest.approx(low, rel=1e-8)
    assert ours["ULCI"] == pytest.approx(high, rel=1e-8)


# --- #41: OLS model summary ---------------------------------------------------------------------


def test_ols_model_summary_matches_statsmodels(fit, data):
    p = fit(4, x="effort", m=["med1", "med2"], y="outcome", boot=50)
    r = p.outcome_models["outcome"].estimation_results
    ref = sm.OLS(data["outcome"], design(data, ["effort", "med1", "med2"])).fit()
    assert r["R2"] == pytest.approx(ref.rsquared, rel=1e-10)
    assert r["adjR2"] == pytest.approx(ref.rsquared_adj, rel=1e-10)
    assert r["mse"] == pytest.approx(ref.mse_resid, rel=1e-10)
    assert r["F"] == pytest.approx(ref.fvalue, rel=1e-10)
    assert r["F_pval"] == pytest.approx(ref.f_pvalue, rel=1e-6)
    assert r["F_pval"] > 0
    assert (r["df_r"], r["df_e"]) == (ref.df_model, ref.df_resid)


# --- #42: logit fit statistics -------------------------------------------------------------------


@pytest.mark.parametrize("n", [300, 3000])
def test_logit_fit_statistics_match_statsmodels(fit, n):
    rng = np.random.default_rng(42)
    x = rng.normal(size=n)
    m = 0.5 * x + rng.normal(size=n)
    y = (rng.random(n) < 1 / (1 + np.exp(-(0.8 * m + 0.5 * x)))).astype(int)
    df = pd.DataFrame(dict(effort=x, med1=m, binary=y))
    p = fit(4, df=df, x="effort", m=["med1"], y="binary", logit=True, boot=50)
    model = p.outcome_models["binary"]
    r = model.estimation_results
    ref = sm.Logit(df["binary"], design(df, ["effort", "med1"])).fit(disp=0)
    llf, llnull = ref.llf, ref.llnull
    assert r["minus2ll"] == pytest.approx(-2 * llf, rel=1e-6)
    assert r["d"] == pytest.approx(2 * (llf - llnull), rel=1e-6)
    assert r["pvalue"] == pytest.approx(ref.llr_pvalue, rel=1e-4, abs=1e-12)
    assert r["mcfadden"] == pytest.approx(ref.prsquared, rel=1e-6)
    coxsnell = 1 - np.exp(2 * (llnull - llf) / n)
    assert r["coxsnell"] == pytest.approx(coxsnell, rel=1e-6)
    assert r["nagelkerke"] == pytest.approx(coxsnell / (1 - np.exp(2 * llnull / n)), rel=1e-6)
    assert np.isfinite([r["coxsnell"], r["nagelkerke"]]).all()
    coeffs = model.coeff_summary()
    ci = ref.conf_int(alpha=0.05)
    for name, ref_name in [("Cons", "const"), ("effort", "effort"), ("med1", "med1")]:
        assert coeffs.loc[name, "coeff"] == pytest.approx(ref.params[ref_name], rel=1e-5)
        assert coeffs.loc[name, "se"] == pytest.approx(ref.bse[ref_name], rel=1e-4)
        assert coeffs.loc[name, "LLCI"] == pytest.approx(ci.loc[ref_name, 0], rel=1e-4)
        assert coeffs.loc[name, "ULCI"] == pytest.approx(ci.loc[ref_name, 1], rel=1e-4)


# --- #52: covariance estimators ------------------------------------------------------------------


@pytest.mark.parametrize("cov_type", ["standard", "HC0", "HC1", "HC2", "HC3"])
def test_ols_covariance_estimators_match_statsmodels(fit, data, cov_type):
    df = data.iloc[:80]
    p = fit(4, df=df, x="effort", m=["med1"], y="outcome", boot=50, cov_type=cov_type)
    ours = p.outcome_models["outcome"].coeff_summary()
    ref = sm.OLS(df["outcome"], design(df, ["effort", "med1"])).fit(
        cov_type="nonrobust" if cov_type == "standard" else cov_type
    )
    for name, ref_name in [("Cons", "const"), ("effort", "effort"), ("med1", "med1")]:
        assert ours.loc[name, "coeff"] == pytest.approx(ref.params[ref_name], rel=1e-8)
        assert ours.loc[name, "se"] == pytest.approx(ref.bse[ref_name], rel=1e-8), cov_type


def test_hc3_flag_is_shorthand_for_cov_type(fit):
    a = fit(4, boot=50, hc3=True, **SPEC4).outcome_models["outcome"].coeff_summary()
    b = fit(4, boot=50, cov_type="HC3", **SPEC4).outcome_models["outcome"].coeff_summary()
    pd.testing.assert_frame_equal(a, b)
    with pytest.raises(ValueError, match="cov_type"):
        fit(4, boot=50, hc3=True, cov_type="HC1", **SPEC4)
    with pytest.raises(ValueError, match="cov_type"):
        fit(4, boot=50, cov_type="robust", **SPEC4)


# --- #65: log-likelihood, AIC and BIC ---------------------------------------------------------------


def test_glance_information_criteria_match_statsmodels(fit, data):
    p = fit(4, x="effort", m=["med1"], y="outcome", boot=50)
    g = p.glance().set_index("outcome")
    ols = sm.OLS(data["outcome"], design(data, ["effort", "med1"])).fit()
    assert g.loc["outcome", "log_likelihood"] == pytest.approx(ols.llf, rel=1e-10)
    assert g.loc["outcome", "aic"] == pytest.approx(ols.aic, rel=1e-10)
    assert g.loc["outcome", "bic"] == pytest.approx(ols.bic, rel=1e-10)
    q = fit(4, x="effort", m=["med1"], y="binary", logit=True, boot=50)
    g = q.glance().set_index("outcome")
    logit = sm.Logit(data["binary"], design(data, ["effort", "med1"])).fit(disp=0)
    assert g.loc["binary", "log_likelihood"] == pytest.approx(logit.llf, rel=1e-6)
    assert g.loc["binary", "ll_null"] == pytest.approx(logit.llnull, rel=1e-6)
    assert g.loc["binary", "aic"] == pytest.approx(logit.aic, rel=1e-6)
    assert g.loc["binary", "bic"] == pytest.approx(logit.bic, rel=1e-6)
    assert g.loc["binary", "df_model"] == logit.df_model


# --- #66: augment against statsmodels -----------------------------------------------------------------


def test_augment_matches_statsmodels_fitted_values(fit, data):
    p = fit(4, x="effort", m=["med1"], y="outcome", boot=50)
    a = p.augment(outcome="outcome")
    ols = sm.OLS(data["outcome"], design(data, ["effort", "med1"])).fit()
    np.testing.assert_allclose(a[".fitted_outcome"].to_numpy(), ols.fittedvalues.to_numpy(), rtol=1e-10)
    np.testing.assert_allclose(a[".resid_outcome"].to_numpy(), ols.resid.to_numpy(), rtol=1e-8, atol=1e-10)
    q = fit(4, x="effort", m=["med1"], y="binary", logit=True, boot=50)
    a = q.augment(outcome="binary")
    logit = sm.Logit(data["binary"], design(data, ["effort", "med1"])).fit(disp=0)
    np.testing.assert_allclose(a[".fitted_binary"].to_numpy(), np.asarray(logit.predict()), rtol=1e-5)


# --- #67: to_statsmodels() ---------------------------------------------------------------------------------


@pytest.mark.parametrize("cov_type", ["standard", "HC3"])
def test_to_statsmodels_ols_is_a_faithful_twin(fit, data, cov_type):
    p = fit(4, x="effort", m=["med1"], y="outcome", boot=50, cov_type=cov_type)
    model = p.outcome_models["outcome"]
    twin = model.to_statsmodels()
    ours = model.coeff_summary()
    ci = twin.conf_int(alpha=0.05)
    assert list(twin.params.index) == list(ours.index)
    for name in ours.index:
        assert ours.loc[name, "coeff"] == pytest.approx(twin.params[name], rel=1e-8)
        assert ours.loc[name, "se"] == pytest.approx(twin.bse[name], rel=1e-8)
        assert ours.loc[name, "p"] == pytest.approx(twin.pvalues[name], rel=1e-6)
        assert ours.loc[name, "LLCI"] == pytest.approx(ci.loc[name, 0], rel=1e-8)
        assert ours.loc[name, "ULCI"] == pytest.approx(ci.loc[name, 1], rel=1e-8)
    assert twin.cov_type == ("nonrobust" if cov_type == "standard" else cov_type)
    assert "OLS Regression Results" in str(twin.summary())


def test_to_statsmodels_logit_is_a_faithful_twin(fit):
    p = fit(4, x="effort", m=["med1"], y="binary", logit=True, boot=50)
    model = p.outcome_models["binary"]
    twin = model.to_statsmodels()
    ours = model.coeff_summary()
    for name in ours.index:
        assert ours.loc[name, "coeff"] == pytest.approx(twin.params[name], rel=1e-5)
        assert ours.loc[name, "se"] == pytest.approx(twin.bse[name], rel=1e-4)
    assert "Logit Regression Results" in str(twin.summary())


def test_process_to_statsmodels_covers_every_outcome(fit):
    p = fit(7, x="effort", w="motiv", m=["med1", "med2"], y="outcome", boot=50)
    fits = p.to_statsmodels()
    assert list(fits) == list(p.outcome_models)
    contrast = fits["outcome"].t_test("effort + med1 = 0")
    assert np.isfinite(np.ravel(contrast.effect)[0])


def test_to_statsmodels_explains_the_missing_dependency(fit, monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "statsmodels.api", None)
    p = fit(4, x="effort", m=["med1"], y="outcome", boot=50)
    with pytest.raises(ImportError, match=r"pyprocessmacro\[statsmodels\]"):
        p.outcome_models["outcome"].to_statsmodels()
