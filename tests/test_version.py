"""
The version argument (#87): the conventions of PROCESS 2.16 (the default) and of PROCESS 5.0.

The 2.16 output files in tests/Results and the 5.0 files in tests/Results/v5 are the reference for each
version's numbers (test_models_accuracy.py, test_process5.py). These tests cover the resolution of the
defaults, the spotlight rules, the probing threshold of models 1 to 3, and the note for the models a later
PROCESS release retired, which every version estimates.
"""
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from pyprocessmacro import Process

M1 = dict(x="effort", m="motiv", y="outcome")
M1_NULL = dict(x="effort", m="qual", y="outcome")  # qual plays no part in the outcome: a weak interaction
M4 = dict(x="effort", m=["med1"], y="outcome")
RETIRED = {
    23: dict(x="effort", m=["med1"], w="motiv", z="skill", v="value", y="outcome"),
    30: dict(x="effort", m=["med1"], w="motiv", z="skill", v="value", y="outcome"),
    57: dict(x="effort", m=["med1"], w="motiv", z="skill", v="value", q="qual", y="outcome"),
    74: dict(x="effort", m=["med1"], y="outcome"),
}


def type6(values, q):
    """PROCESS's percentile: sorted values interpolated at position q(n + 1), R's quantile type 6."""
    v = np.sort(np.asarray(values, dtype=float))
    pos = q * (len(v) + 1)
    low = int(pos)
    return v[low - 1] + (v[low] - v[low - 1]) * (pos - low)


def moments(values):
    v = np.asarray(values, dtype=float)
    return [v.mean() - v.std(ddof=1), v.mean(), v.mean() + v.std(ddof=1)]


# --- defaults -----------------------------------------------------------------------------------------

def test_default_version_is_2_16(fit):
    p = fit(1, **M1)
    assert p.version == "2.16"
    assert p.options["percent"] is False
    assert p.options["intprobe"] == 1.0
    assert p.options["spotlight"] == "moments"
    np.testing.assert_allclose(p._spotlight_values["m"], moments(p._data["m"]))


def test_version_5_defaults(fit):
    p = fit(1, version="5.0", **M1)
    assert p.version == "5.0"
    assert p.options["percent"] is True
    assert p.options["intprobe"] == 0.10
    assert p.options["spotlight"] == "percentiles"
    np.testing.assert_allclose(p._spotlight_values["m"], [type6(p._data["m"], q) for q in (0.16, 0.50, 0.84)])


def test_version_5_intervals_are_percentile(fit):
    assert set(fit(4, version="5.0", **M4).tidy(component="indirect")["method"]) == {"bootstrap_percentile"}
    assert set(fit(4, **M4).tidy(component="indirect")["method"]) == {"bootstrap_bc"}


@pytest.mark.parametrize("version", [5.0, "5", 5, 2.16, 2])
def test_version_aliases(fit, version):
    assert fit(1, version=version, **M1).version == ("5.0" if str(version).startswith("5") else "2.16")


# --- explicit arguments win ---------------------------------------------------------------------------

def test_explicit_arguments_override_the_version(fit):
    assert fit(4, version="5.0", percent=False, **M4).options["percent"] is False
    assert fit(4, version="2.16", percent=True, **M4).options["percent"] is True
    assert fit(1, version="5.0", intprobe=1, **M1).options["intprobe"] == 1.0
    quantiles = fit(1, version="5.0", quantile=True, **M1)
    assert quantiles.options["spotlight"] == "quantiles"
    np.testing.assert_allclose(quantiles._spotlight_values["m"], np.percentile(quantiles._data["m"], [10, 25, 50, 75, 90]))
    with_moments = fit(1, version="5.0", moments=True, **M1)
    assert with_moments.options["spotlight"] == "moments"
    np.testing.assert_allclose(with_moments._spotlight_values["m"], moments(with_moments._data["m"]))
    custom = fit(1, version="5.0", modval={"motiv": [-1.0, 1.0]}, **M1)
    assert list(custom._spotlight_values["m"]) == [-1.0, 1.0]


# --- validation ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("version", ["3.0", "4.2", "five", None])
def test_unknown_version_raises(fit, version):
    with pytest.raises(ValueError, match="'version'"):
        fit(1, version=version, **M1)


def test_quantile_and_moments_conflict(fit):
    with pytest.raises(ValueError, match="'quantile' and 'moments'"):
        fit(1, quantile=True, moments=True, **M1)


@pytest.mark.parametrize("intprobe", [-0.1, 1.5, "0.1", True])
def test_invalid_intprobe_raises(fit, intprobe):
    with pytest.raises(ValueError, match="intprobe"):
        fit(1, intprobe=intprobe, **M1)


def test_unknown_model_number_raises(fit):
    with pytest.raises(ValueError, match="Model 99"):
        fit(99, **M1)


@pytest.mark.parametrize("model", sorted(RETIRED))
@pytest.mark.parametrize("version", ["2.16", "5.0"])
def test_retired_models_are_estimated_under_every_version_with_a_note(fit, model, version):
    release = "4.0" if model == 74 else "3.0"
    with pytest.warns(UserWarning, match=f"Model {model} was retired in PROCESS {release}"):
        p = fit(model, version=version, **RETIRED[model])
    assert p.version == version
    assert p.retired_note.startswith(f"Note: Model {model} was retired in PROCESS {release}")
    assert p.retired_note in p.summary()
    assert len(p.indirect_model.coeff_summary()) > 0


@pytest.mark.parametrize("version", ["2.16", "5.0"])
def test_shared_models_fit_under_both_versions(fit, version):
    p = fit(7, version=version, x="effort", w="motiv", m=["med1"], y="outcome")
    assert p.version == version and p.indirect_model._analysis_list == ["MM"]
    assert p.retired_note is None and "retired" not in p.summary()


# --- spotlight rules for discrete moderators ----------------------------------------------------------

def test_discrete_moderators_follow_each_version(fit, data):
    df = data.copy()
    df["three"] = np.repeat([0, 1, 2], [10, 280, 10])
    df["two"] = np.tile([0, 1], 150)
    # 2.16: every value of a moderator with at most five (PyProcessMacro's historical rule)
    assert list(fit(1, df=df, x="effort", m="three", y="outcome")._spotlight_values["m"]) == [0, 1, 2]
    # 5.0: the percentiles unless the moderator is dichotomous (PROCESS's rule)
    five = fit(1, df=df, version="5.0", x="effort", m="three", y="outcome")
    np.testing.assert_allclose(five._spotlight_values["m"], [type6(df["three"], q) for q in (0.16, 0.50, 0.84)])
    for version in ("2.16", "5.0"):
        assert list(fit(1, df=df, version=version, x="effort", m="two", y="outcome")._spotlight_values["m"]) == [0, 1]


# --- probing of the moderation-only models ------------------------------------------------------------

def test_probe_p_is_the_test_of_the_highest_order_interaction(fit):
    one = fit(1, **M1_NULL)
    assert one.direct_model.probe_terms == ["effort*qual"]
    assert one.direct_model.probe_p == pytest.approx(
        one.outcome_models["outcome"].coeff_summary().loc["effort*qual", "p"]
    )
    two = fit(2, x="effort", m="qual", w="ctrl", y="outcome")
    coeffs = two.outcome_models["outcome"].coeff_summary()
    assert two.direct_model.probe_terms == ["effort*qual", "effort*ctrl"]
    assert two.direct_model.probe_p == pytest.approx(min(coeffs.loc["effort*qual", "p"], coeffs.loc["effort*ctrl", "p"]))
    three = fit(3, x="effort", m="qual", w="ctrl", y="outcome")
    assert three.direct_model.probe_terms == ["effort*qual*ctrl"]
    assert three.direct_model.probe_p == pytest.approx(
        three.outcome_models["outcome"].coeff_summary().loc["effort*qual*ctrl", "p"]
    )
    assert fit(8, x="effort", w="motiv", m=["med1"], y="outcome").direct_model.probe_p is None  # mediation: always reported


def test_logit_probe_is_a_likelihood_ratio_test(fit, data):
    sm = pytest.importorskip("statsmodels.api")
    p = fit(1, x="effort", m="qual", y="binary", logit=True)
    exog = pd.DataFrame(dict(const=1.0, effort=data["effort"], qual=data["qual"], product=data["effort"] * data["qual"]))
    full = sm.Logit(data["binary"], exog).fit(disp=0)
    reduced = sm.Logit(data["binary"], exog.drop(columns="product")).fit(disp=0)
    expected = stats.chi2.sf(2 * (full.llf - reduced.llf), 1)
    assert p.direct_model.probe_p == pytest.approx(expected, rel=1e-4)


def test_intprobe_gates_the_conditional_effects_of_models_1_to_3(fit):
    always = fit(1, intprobe=1, **M1_NULL)
    never = fit(1, intprobe=0, **M1_NULL)
    assert always.direct_model.probed and not never.direct_model.probed
    assert "Conditional effect(s) of effort on outcome at values of the moderator(s)" in always.summary()
    text = never.summary()
    assert "not reported" in text and "intprobe = 0" in text
    assert "not reported" in never._repr_html_()
    pd.testing.assert_frame_equal(always.direct_model.coeff_summary(), never.direct_model.coeff_summary())


# --- the conventions are stated ----------------------------------------------------------------------

def test_banner_and_summary_state_the_conventions(data, capsys):
    p = Process(data, 4, boot=50, seed=1, version="5.0", **M4)
    assert "PROCESS version: 5.0. Bootstrap intervals: percentile." in capsys.readouterr().out
    assert p.summary().startswith("PROCESS version: 5.0. Bootstrap intervals: percentile.")
    assert "PROCESS version: 5.0" in p._repr_html_()
    q = Process(data, 1, version="5.0", modval={"motiv": [0.0]}, suppr_init=True, **M1)
    assert q.summary().startswith(
        "PROCESS version: 5.0. Moderators at the 16th, 50th and 84th percentiles (custom values for motiv). "
        "Conditional effects reported: when the interaction's p is at most 0.1."
    )
    assert Process(data, 1, suppr_init=True, **M1).summary().startswith(
        "PROCESS version: 2.16. Moderators at the mean and one SD either side. Conditional effects reported: always."
    )
