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
