"""
Model 6, serial mediation: point estimates against products of statsmodels coefficients, and the
bootstrap against a reference resampler that uses the same draws.
"""
from itertools import combinations

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from pyprocessmacro import Process
from pyprocessmacro.utils import bias_corrected_ci

pytestmark = pytest.mark.smoke


@pytest.fixture(scope="module")
def serial_data(data):
    rng = np.random.default_rng(6)
    df = data.copy()
    df["med2"] = 0.4 * df["effort"] + 0.5 * df["med1"] + rng.normal(size=len(df))
    df["med3"] = 0.2 * df["effort"] + 0.3 * df["med2"] + rng.normal(size=len(df))
    df["outcome"] = 0.3 * df["effort"] + 0.4 * df["med1"] + 0.3 * df["med2"] + 0.2 * df["med3"] + rng.normal(size=len(df))
    return df


def ols_coefficients(df, endog, exog):
    return sm.OLS(df[endog], sm.add_constant(df[list(exog)])).fit().params


def test_paths_follow_process_order_and_bounds(fit, serial_data):
    p2 = fit(6, df=serial_data, x="effort", m=["med1", "med2"], y="outcome")
    assert p2.indirect_model.path_labels == [
        "effort -> med1 -> outcome", "effort -> med1 -> med2 -> outcome", "effort -> med2 -> outcome",
    ]
    p3 = fit(6, df=serial_data, x="effort", m=["med1", "med2", "med3"], y="outcome")
    assert p3.indirect_model.path_labels == [
        "effort -> med1 -> outcome", "effort -> med1 -> med2 -> outcome", "effort -> med1 -> med3 -> outcome",
        "effort -> med1 -> med2 -> med3 -> outcome", "effort -> med2 -> outcome", "effort -> med2 -> med3 -> outcome",
        "effort -> med3 -> outcome",
    ]
    with pytest.raises(ValueError, match="two to four"):
        fit(6, df=serial_data, x="effort", m=["med1"], y="outcome")
    with pytest.raises(ValueError, match="two to four"):
        fit(6, df=serial_data, x="effort", m=["med1", "med2", "med3", "motiv", "skill"], y="outcome")


def test_equations_form_a_chain(fit, serial_data):
    p = fit(6, df=serial_data, x="effort", m=["med1", "med2", "med3"], y="outcome", controls=["ctrl"], controls_in="x_to_m")
    terms = {name: list(model.coeff_summary().index) for name, model in p.outcome_models.items()}
    assert terms["med1"] == ["Cons", "effort", "ctrl"]
    assert terms["med2"] == ["Cons", "effort", "ctrl", "med1"]
    assert terms["med3"] == ["Cons", "effort", "ctrl", "med1", "med2"]
    assert terms["outcome"] == ["Cons", "effort", "med1", "med2", "med3"]


def test_point_estimates_are_products_of_ols_coefficients(fit, serial_data):
    p = fit(6, df=serial_data, x="effort", m=["med1", "med2", "med3"], y="outcome", total=True, contrast=True)
    a = {1: ols_coefficients(serial_data, "med1", ["effort"]),
         2: ols_coefficients(serial_data, "med2", ["effort", "med1"]),
         3: ols_coefficients(serial_data, "med3", ["effort", "med1", "med2"])}
    b = ols_coefficients(serial_data, "outcome", ["effort", "med1", "med2", "med3"])
    expected = [
        a[1]["effort"] * b["med1"],
        a[1]["effort"] * a[2]["med1"] * b["med2"],
        a[1]["effort"] * a[3]["med1"] * b["med3"],
        a[1]["effort"] * a[2]["med1"] * a[3]["med2"] * b["med3"],
        a[2]["effort"] * b["med2"],
        a[2]["effort"] * a[3]["med2"] * b["med3"],
        a[3]["effort"] * b["med3"],
    ]
    table = p.indirect_model.coeff_summary()
    effects = table["Effect"].to_numpy()
    assert table[""].iloc[0] == "TOTAL"
    assert effects[0] == pytest.approx(sum(expected), rel=1e-10)
    np.testing.assert_allclose(effects[1:8], expected, rtol=1e-10)
    contrasts = [expected[i] - expected[j] for i, j in combinations(range(7), 2)]
    np.testing.assert_allclose(effects[8:], contrasts, rtol=1e-10)
    assert table[""].iloc[8] == "Contrast: effort -> med1 -> outcome vs. effort -> med1 -> med2 -> outcome"
    # The total indirect effect equals the total effect minus the direct effect.
    total_effect = ols_coefficients(serial_data, "outcome", ["effort"])["effort"]
    direct = p.direct_model.coeff_summary()["Effect"].iloc[0]
    assert effects[0] == pytest.approx(total_effect - direct, rel=1e-8)


def _reference_bootstrap(model, seed, n_boots, logit):
    """Resample with the same draws and refit every equation with numpy or statsmodels."""
    data = np.asarray(model._data, dtype=float)
    sampler = np.random.RandomState(seed)
    draws = []
    while len(draws) < n_boots:
        idx = sampler.randint(data.shape[0], size=data.shape[0])
        sample = data[idx]
        if logit:
            by = sm.Logit(sample[:, model._ind_y], sample[:, model._exog_inds_y]).fit(disp=0, tol=1e-12).params
        else:
            by = np.linalg.lstsq(sample[:, model._exog_inds_y], sample[:, model._ind_y], rcond=None)[0]
        bm = [np.linalg.lstsq(sample[:, exog], sample[:, ind], rcond=None)[0]
              for ind, exog in zip(model._inds_m, model._exog_inds_m_list)]
        draws.append(np.array([model._path_effect(path, np.asarray(by), bm) for path in model._paths]))
    return np.array(draws).T  # (n_paths, n_boots)


def test_bootstrap_matches_a_reference_resampler(fit, serial_data):
    p = fit(6, df=serial_data, x="effort", m=["med1", "med2"], y="outcome", boot=300, seed=17, total=True)
    model = p.indirect_model
    draws = _reference_bootstrap(model, 17, 300, logit=False)
    ours = model.estimation_results
    for k, path in enumerate(model._paths):
        estimate = model._path_effect(path, model._true_betas_y, model._true_betas_m)
        low, high = bias_corrected_ci(estimate, draws[k], 95)
        assert ours["effect"][k + 1] == pytest.approx(estimate, rel=1e-10)
        assert ours["se"][k + 1] == pytest.approx(draws[k].std(ddof=1), rel=1e-7)
        assert (ours["llci"][k + 1], ours["ulci"][k + 1]) == pytest.approx((low, high), rel=1e-7)
    total = draws.sum(axis=0)
    assert ours["se"][0] == pytest.approx(total.std(ddof=1), rel=1e-7)
    assert model._n_fail_samples == 0


def test_logistic_outcome_bootstrap_matches_a_reference_resampler(fit, serial_data):
    df = serial_data.copy()
    rng = np.random.default_rng(66)
    df["binary"] = (rng.random(len(df)) < 1 / (1 + np.exp(-(0.5 * df["med1"] + 0.4 * df["med2"] + 0.3 * df["effort"])))).astype(int)
    p = fit(6, df=df, x="effort", m=["med1", "med2"], y="binary", logit=True, boot=100, seed=5)
    model = p.indirect_model
    draws = _reference_bootstrap(model, 5, 100, logit=True)
    for k in range(len(model._paths)):
        assert model.estimation_results["se"][k] == pytest.approx(draws[k].std(ddof=1), rel=1e-4)


def test_summary_tidy_and_export_cover_the_paths(fit, serial_data, capsys):
    p = fit(6, df=serial_data, x="effort", m=["med1", "med2"], y="outcome", total=True, contrast=True)
    text = p.summary()
    assert "serial mediators" in text and "effort -> med1 -> med2 -> outcome" in text
    rows = p.tidy(["total", "indirect", "contrast"])
    assert list(rows["component"]) == ["total"] + ["indirect"] * 3 + ["contrast"] * 3
    assert list(rows["term"][1:4]) == p.indirect_model.path_labels
    assert p.tidy("outcome")["outcome"].tolist().count("med2") == 3  # Cons, effort, med1
    boots = p.get_bootstrap_estimates()
    assert boots.loc[boots["OutcomeName"] == "med2", "med1"].notna().all()
    assert boots.loc[boots["OutcomeName"] == "med1", "med1"].isna().all()
    with pytest.raises(ValueError):
        p.floodlight_indirect_effect(med_name="med1", mod_name="motiv")
    assert "<table" in p._repr_html_()



