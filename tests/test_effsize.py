"""
Standardized indirect effects (effsize): point estimates by formula, bootstrap against a reference
resampler that uses the same draws and standardizes within each resample.
"""
import numpy as np
import pytest

from pyprocessmacro.utils import bias_corrected_ci

pytestmark = pytest.mark.smoke


def test_effect_sizes_follow_the_formulas(fit, data):
    p = fit(4, x="effort", m=["med1", "med2"], y="outcome", effsize=True, total=True, contrast=True)
    table = p.indirect_model.effect_size_summary()
    raw = p.indirect_model.coeff_summary()["Effect"].to_numpy()[:3]  # TOTAL, med1, med2 (no contrasts)
    sd_x, sd_y = data["effort"].std(ddof=1), data["outcome"].std(ddof=1)
    partial = table[table["Standardization"] == "partially"]
    complete = table[table["Standardization"] == "completely"]
    assert list(partial[""]) == list(complete[""]) == ["TOTAL", "med1", "med2"]
    np.testing.assert_allclose(partial["Effect"].to_numpy(), raw / sd_y, rtol=1e-10)
    np.testing.assert_allclose(complete["Effect"].to_numpy(), raw * sd_x / sd_y, rtol=1e-10)


def test_effect_size_bootstrap_matches_a_reference_resampler(fit, data):
    p = fit(4, x="effort", m=["med1"], y="outcome", effsize=True, boot=300, seed=21)
    model = p.indirect_model
    arr = np.asarray(model._data, dtype=float)
    ix, iy, im = model._symb_to_ind["x"], model._ind_y, model._inds_m[0]
    sampler = np.random.RandomState(21)
    ps, cs = [], []
    for _ in range(300):
        s = arr[sampler.randint(arr.shape[0], size=arr.shape[0])]
        b = np.linalg.lstsq(s[:, model._exog_inds_y], s[:, iy], rcond=None)[0]
        a = np.linalg.lstsq(s[:, model._exog_inds_m], s[:, im], rcond=None)[0]
        ab = a[model._exog_terms_m.index("x")] * b[model._exog_terms_y.index("m1")]
        sd_x, sd_y = s[:, ix].std(ddof=1), s[:, iy].std(ddof=1)
        ps.append(ab / sd_y)
        cs.append(ab * sd_x / sd_y)
    sizes = model.effect_sizes()
    for kind, draws in (("ps", np.array(ps)), ("cs", np.array(cs))):
        assert sizes[kind]["se"][0] == pytest.approx(draws.std(ddof=1), rel=1e-7)
        low, high = bias_corrected_ci(sizes[kind]["effect"][0], draws, 95)
        assert (sizes[kind]["llci"][0], sizes[kind]["ulci"][0]) == pytest.approx((low, high), rel=1e-7)


def test_effect_sizes_for_serial_paths(fit, data):
    rng = np.random.default_rng(9)
    df = data.copy()
    df["med2"] = 0.4 * df["effort"] + 0.5 * df["med1"] + rng.normal(size=len(df))
    p = fit(6, df=df, x="effort", m=["med1", "med2"], y="outcome", effsize=True)
    table = p.indirect_model.effect_size_summary()
    partial = table[table["Standardization"] == "partially"]
    assert list(partial[""]) == p.indirect_model.path_labels
    raw = p.indirect_model.coeff_summary()["Effect"].to_numpy()
    np.testing.assert_allclose(partial["Effect"].to_numpy(), raw / df["outcome"].std(ddof=1), rtol=1e-10)


@pytest.mark.parametrize("kwargs, message", [
    (dict(model=7, x="effort", w="motiv", m=["med1"], y="outcome"), "unmoderated"),
    (dict(model=4, x="effort", m=["med1"], y="binary", logit=True), "continuous outcome"),
    (dict(model=1, x="effort", m="motiv", y="outcome"), "mediat"),
])
def test_effect_sizes_are_only_for_unmoderated_continuous_mediation(fit, kwargs, message):
    with pytest.raises(ValueError, match=message):
        fit(effsize=True, **kwargs)
    with pytest.raises(ValueError, match="effsize"):
        fit(4, x="effort", m=["med1"], y="outcome", effsize="yes")


def test_effect_sizes_in_summary_tidy_and_html(fit):
    p = fit(4, x="effort", m=["med1", "med2"], y="outcome", effsize=True, total=True)
    text = p.summary()
    assert "Partially standardized indirect effect(s) of effort on outcome" in text
    assert "Completely standardized indirect effect(s) of effort on outcome" in text
    rows = p.tidy(["indirect_ps", "indirect_cs"])
    assert list(rows["component"]) == ["indirect_ps"] * 3 + ["indirect_cs"] * 3
    assert list(rows["term"][:3]) == ["total", "med1", "med2"]
    assert rows["n_boot"].eq(p.options["boot"]).all()
    assert "Standardized indirect effect(s)" in p._repr_html_()
    q = fit(4, x="effort", m=["med1"], y="outcome")
    assert "standardized" not in q.summary()



