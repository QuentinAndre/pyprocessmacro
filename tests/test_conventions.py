"""
The conventions PROCESS 3.0 changed (#87, #90): the bootstrap interval type (`percent`), the spotlight values
of continuous moderators (`spotlight`) and the probing threshold of models 1 to 3 (`intprobe`), the line that
states them at the top of the output, and the note for the models a later PROCESS release retired (#91).

The 2.16 output files in tests/Results and the 5.0 files in tests/Results/v5 are the reference for the numbers
under each set of conventions (test_models_accuracy.py, test_process5.py).
"""
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from pyprocessmacro import Process

M1 = dict(x="effort", m="motiv", y="outcome")
M1_NULL = dict(x="effort", m="qual", y="outcome")  # qual plays no part in the outcome: a weak interaction
M4 = dict(x="effort", m=["med1"], y="outcome")
PROCESS3 = dict(percent=True, spotlight="percentiles")
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


# --- defaults are those of PROCESS 2 ------------------------------------------------------------------

def test_defaults_are_the_process_2_conventions(fit):
    p = fit(1, **M1)
    assert p.options["percent"] is False
    assert p.options["intprobe"] == 1.0
    assert p.options["spotlight"] == "moments"
    np.testing.assert_allclose(p._spotlight_values["m"], moments(p._data["m"]))
    assert set(fit(4, **M4).tidy(component="indirect")["method"]) == {"bootstrap_bc"}


def test_process_3_conventions(fit):
    p = fit(1, **PROCESS3, **M1)
    assert p.options["spotlight"] == "percentiles"
    np.testing.assert_allclose(p._spotlight_values["m"], [type6(p._data["m"], q) for q in (0.16, 0.50, 0.84)])
    assert set(fit(4, **PROCESS3, **M4).tidy(component="indirect")["method"]) == {"bootstrap_percentile"}


def test_quantiles_rule_and_its_alias(fit):
    for kwargs in (dict(quantile=True), dict(spotlight="quantiles"), dict(quantile=True, spotlight="quantiles")):
        p = fit(1, **kwargs, **M1)
        assert p.options["spotlight"] == "quantiles"
        np.testing.assert_allclose(p._spotlight_values["m"], np.percentile(p._data["m"], [10, 25, 50, 75, 90]))


def test_custom_values_win(fit):
    custom = fit(1, spotlight="percentiles", modval={"motiv": [-1.0, 1.0]}, **M1)
    assert list(custom._spotlight_values["m"]) == [-1.0, 1.0]


# --- validation ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("spotlight", ["mean", "16/50/84", 3, ["moments"]])
def test_unknown_spotlight_rule_raises(fit, spotlight):
    with pytest.raises(ValueError, match="'spotlight'"):
        fit(1, spotlight=spotlight, **M1)


def test_quantile_and_spotlight_conflict(fit):
    with pytest.raises(ValueError, match="'quantile' and 'spotlight'"):
        fit(1, quantile=True, spotlight="moments", **M1)


@pytest.mark.parametrize("percent", [None, "yes", 1.5])
def test_invalid_percent_raises(fit, percent):
    with pytest.raises(ValueError, match="'percent'"):
        fit(4, percent=percent, **M4)


@pytest.mark.parametrize("intprobe", [-0.1, 1.5, "0.1", True, None])
def test_invalid_intprobe_raises(fit, intprobe):
    with pytest.raises(ValueError, match="intprobe"):
        fit(1, intprobe=intprobe, **M1)


def test_unknown_model_number_raises(fit):
    with pytest.raises(ValueError, match="Model 99"):
        fit(99, **M1)


# --- retired models -----------------------------------------------------------------------------------

@pytest.mark.parametrize("model", sorted(RETIRED))
def test_retired_models_are_estimated_with_a_note(fit, model):
    release = "4.0" if model == 74 else "3.0"
    with pytest.warns(UserWarning, match=f"Model {model} was retired in PROCESS {release} and does not exist in PROCESS 5"):
        p = fit(model, **RETIRED[model])
    assert p.retired_note.startswith(f"Note: Model {model} was retired in PROCESS {release}")
    assert p.retired_note in p.summary()
    assert len(p.indirect_model.coeff_summary()) > 0


def test_current_models_carry_no_note(fit):
    p = fit(7, x="effort", w="motiv", m=["med1"], y="outcome")
    assert p.retired_note is None and "retired" not in p.summary()


# --- discrete moderators ------------------------------------------------------------------------------

def test_discrete_moderators_follow_the_rule(fit, data):
    df = data.copy()
    df["three"] = np.repeat([0, 1, 2], [10, 280, 10])
    df["two"] = np.tile([0, 1], 150)
    # moments and quantiles: every value of a moderator with at most five (PyProcessMacro's historical rule)
    assert list(fit(1, df=df, x="effort", m="three", y="outcome")._spotlight_values["m"]) == [0, 1, 2]
    assert list(fit(1, df=df, quantile=True, x="effort", m="three", y="outcome")._spotlight_values["m"]) == [0, 1, 2]
    # percentiles: the percentiles unless the moderator is dichotomous (PROCESS's rule)
    five = fit(1, df=df, spotlight="percentiles", x="effort", m="three", y="outcome")
    np.testing.assert_allclose(five._spotlight_values["m"], [type6(df["three"], q) for q in (0.16, 0.50, 0.84)])
    for rule in ("moments", "percentiles", "quantiles"):
        assert list(fit(1, df=df, spotlight=rule, x="effort", m="two", y="outcome")._spotlight_values["m"]) == [0, 1]


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


# --- the conventions line -----------------------------------------------------------------------------

def test_banner_and_summary_state_the_conventions(data, capsys):
    p = Process(data, 4, boot=50, seed=1, **M4)
    assert "Bootstrap intervals: bias-corrected (PROCESS 2 default)." in capsys.readouterr().out
    assert p.summary().startswith("Bootstrap intervals: bias-corrected (PROCESS 2 default).")
    assert "PROCESS 2 default" in p._repr_html_()
    three = Process(data, 4, boot=50, seed=1, percent=True, suppr_init=True, **M4)
    assert three.summary().startswith("Bootstrap intervals: percentile (PROCESS 3 and later default).")
    one = Process(data, 1, spotlight="percentiles", intprobe=0.10, suppr_init=True, **M1)
    assert one.summary().startswith(
        "Moderators at the 16th, 50th and 84th percentiles (PROCESS 3 and later default). "
        "Conditional effects reported: when the interaction's p is at most 0.1 (PROCESS 3 and later default)."
    )
    custom = Process(data, 1, modval={"motiv": [0.0]}, suppr_init=True, **M1)
    assert custom.summary().startswith(
        "Moderators at the mean and one SD either side (PROCESS 2 default); custom values for motiv. "
        "Conditional effects reported: always (PROCESS 2 default)."
    )
    quantiles = Process(data, 1, quantile=True, suppr_init=True, **M1)
    assert quantiles.summary().startswith("Moderators at the 10th, 25th, 50th, 75th and 90th percentiles. ")
