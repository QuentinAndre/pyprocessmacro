"""
Fast checks of the public API on synthetic data.

These exercise the code paths the accuracy suite never touches (summaries,
index tables, bootstrap export, plotting, floodlight, controls), so that
compatibility breakages such as pandas API removals are caught before release.
"""
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.smoke

# Variable names deliberately differ from the model symbols (x, w, z, ...) so
# that a mix-up between names and symbols shows up.
SPEC = {
    1: dict(x="effort", m="motiv", y="outcome"),
    4: dict(x="effort", m=["med1", "med2"], y="outcome"),
    7: dict(x="effort", w="motiv", m=["med1"], y="outcome"),
    8: dict(x="effort", w="motiv", m=["med1", "med2"], y="outcome"),
    10: dict(x="effort", w="motiv", z="skill", m=["med1"], y="outcome"),
    12: dict(x="effort", w="motiv", z="skill", m=["med1"], y="outcome"),
    14: dict(x="effort", v="value", m=["med1"], y="outcome"),
    21: dict(x="effort", w="motiv", v="value", m=["med1"], y="outcome"),
    58: dict(x="effort", w="motiv", m=["med1"], y="outcome"),
    74: dict(x="effort", m=["med1"], y="outcome"),
    75: dict(x="effort", w="motiv", z="skill", m=["med1", "med2"], y="outcome"),
}


# --- #32: summaries and index tables on current pandas ---------------------------------

INDEX_NAMES = {
    "MM": "MODERATED MEDIATION",
    "PMM": "PARTIAL MODERATED MEDIATION",
    "MMM": "MODERATED MODERATED MEDIATION",
    "CMM": "CONDITIONAL MODERATED MEDIATION",
}

# Which index tables a model reports. Models with a moderator on both paths (58, 75, ...) report none (#43).
INDEX_MODELS = {
    7: ["MM"],  # one moderator on x -> m
    10: ["PMM"],  # two moderators on x -> m, no three-way term
    12: ["MMM", "CMM"],  # two moderators on x -> m with a three-way term
    14: ["MM"],  # one moderator on m -> y
    21: ["MMM", "CMM"],  # one moderator on each path
    58: [],  # the moderator sits on both paths: no index
    75: [],  # both moderators sit on both paths: no index
}


@pytest.mark.parametrize("model", sorted(SPEC))
def test_summary_runs(fit, capsys, model):
    p = fit(model, **SPEC[model])
    p.summary()
    out = capsys.readouterr().out
    assert "OUTCOME MODELS" in out
    if model > 3:
        assert "DIRECT AND INDIRECT EFFECTS" in out
    else:
        assert "CONDITIONAL EFFECTS" in out
    for analysis in INDEX_MODELS.get(model, []):
        assert f"INDEX OF {INDEX_NAMES[analysis]}" in out
    if model in INDEX_MODELS and not INDEX_MODELS[model]:
        assert "INDEX OF" not in out


@pytest.mark.parametrize("model", sorted(INDEX_MODELS))
def test_index_tables_are_numeric(fit, model):
    p = fit(model, **SPEC[model])
    if not INDEX_MODELS[model]:
        with pytest.raises(NotImplementedError):
            p.indirect_model.MM_index_summary()
    for analysis in INDEX_MODELS[model]:
        table = getattr(p.indirect_model, f"{analysis}_index_summary")()
        assert len(table) > 0
        numeric = [c for c in table.columns if c in ("Index", "Boot SE", "LLCI", "ULCI", "BootLLCI", "BootULCI")]
        assert len(numeric) == 4, table.columns
        for column in numeric:
            assert pd.api.types.is_numeric_dtype(table[column]), (analysis, column, table[column].dtype)
        assert pd.api.types.is_object_dtype(table["Mediator"]) or table["Mediator"].dtype == "string"


@pytest.mark.parametrize("kwargs", [dict(model=4, total=True, contrast=True), dict(model=7)])
def test_indirect_coeff_summary_is_numeric(fit, kwargs):
    model = kwargs.pop("model")
    table = fit(model, **SPEC[model], **kwargs).indirect_model.coeff_summary()
    for column in ("Effect", "Boot SE", "BootLLCI", "BootULCI"):
        assert pd.api.types.is_numeric_dtype(table[column]), (column, table[column].dtype)


# --- #33: bootstrap export ---------------------------------------------------------------


def test_get_bootstrap_estimates(fit):
    p = fit(8, **SPEC[8])
    boots = p.get_bootstrap_estimates()
    n_boot = p.options["boot"]
    assert list(boots.columns[:2]) == ["BootSample", "OutcomeName"]
    assert len(boots) == n_boot * (1 + p.n_meds)
    assert boots.groupby("OutcomeName").size().to_dict() == {"outcome": n_boot, "med1": n_boot, "med2": n_boot}
    for term in ("Cons", "effort", "motiv", "effort*motiv", "med1", "med2"):
        assert term in boots.columns, term
        assert pd.api.types.is_numeric_dtype(boots[term]), term
    # Mediator equations do not contain the other mediator, so those cells are empty.
    assert boots.loc[boots["OutcomeName"] == "med1", "med2"].isna().all()
    assert boots.loc[boots["OutcomeName"] == "outcome", "med2"].notna().all()


def test_get_bootstrap_estimates_requires_mediation(fit):
    with pytest.raises(NotImplementedError):
        fit(1, **SPEC[1]).get_bootstrap_estimates()


# --- #34: controls_in ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "mode, in_outcome, in_mediator",
    [("all", True, True), ("x_to_m", False, True), ("all_to_y", True, False)],
)
def test_controls_in_modes(fit, mode, in_outcome, in_mediator):
    p = fit(7, controls=["ctrl"], controls_in=mode, **SPEC[7])
    outcome_terms = p.outcome_models["outcome"].coeff_summary().index
    mediator_terms = p.outcome_models["med1"].coeff_summary().index
    assert ("ctrl" in outcome_terms) is in_outcome
    assert ("ctrl" in mediator_terms) is in_mediator


def test_controls_in_rejects_unknown_mode(fit):
    with pytest.raises(ValueError, match="controls_in"):
        fit(7, controls=["ctrl"], controls_in="both", **SPEC[7])


# --- #35: a single mediator given as a string ----------------------------------------------


def test_single_mediator_as_string_matches_list(fit, data):
    df = data.rename(columns={"med1": "mediator"})
    as_str = fit(4, df=df, x="effort", m="mediator", y="outcome")
    as_list = fit(4, df=df, x="effort", m=["mediator"], y="outcome")
    assert as_str.n_meds == as_list.n_meds == 1
    assert as_str.mediators == as_list.mediators == ["mediator"]
    assert as_str.iv == "outcome"
    pd.testing.assert_frame_equal(
        as_str.indirect_model.coeff_summary(), as_list.indirect_model.coeff_summary()
    )


def test_moderation_only_models_have_no_mediators(fit):
    p = fit(1, **SPEC[1])
    assert p.mediators == []
    assert p.n_meds == 0


# --- #36: col and row facets ---------------------------------------------------------------


def test_parse_moderator_values_uses_col_and_row(fit):
    p = fit(10, **SPEC[10])
    spot_skill = list(p._spotlight_values["z"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        by_col = p._parse_moderator_values(x="motiv", hue=None, row=None, col="skill", modval={}, path="x_direct")
        by_row = p._parse_moderator_values(x="motiv", hue=None, row="skill", col=None, modval={}, path="x_direct")
    assert list(by_col["skill"]) == spot_skill
    assert list(by_row["skill"]) == spot_skill
    assert len(by_col["motiv"]) == 100  # the x-axis moderator is evaluated on a fine grid


@pytest.mark.parametrize("facet, shape", [("col", (1, 3)), ("row", (3, 1))])
def test_plot_facets_by_second_moderator(fit, facet, shape):
    p = fit(10, **SPEC[10])
    grid = p.plot_conditional_direct_effects(x="motiv", **{facet: "skill"})
    assert grid.axes.shape == shape
    plt.close("all")


# --- #37: floodlight with an unknown moderator name ----------------------------------------


def test_floodlight_rejects_unknown_other_moderator(fit):
    p = fit(10, **SPEC[10])
    with pytest.raises(ValueError, match="nonexistent"):
        p.floodlight_direct_effect(mod_name="motiv", other_modval={"nonexistent": 1})
    with pytest.raises(ValueError, match="nonexistent"):
        p.floodlight_indirect_effect(med_name="med1", mod_name="motiv", other_modval={"nonexistent": 1})


def test_floodlight_runs(fit):
    p = fit(10, **SPEC[10])
    direct = p.floodlight_direct_effect(mod_name="motiv", other_modval={"skill": 0.5})
    indirect = p.floodlight_indirect_effect(med_name="med1", mod_name="motiv", other_modval={"skill": 0.5})
    for analysis in (direct, indirect):
        regions = analysis.get_significance_regions()
        assert set(regions) == {"Negative on", "Positive on"}
        assert "FLOODLIGHT ANALYSIS" in repr(analysis)


# --- #38: hue_format keys ------------------------------------------------------------------


@pytest.mark.parametrize("fmt", ["{var1} = {val1:.2f}", "{var1} at {hue1:.2f}"])
def test_hue_format_accepts_documented_and_legacy_keys(fit, fmt):
    p = fit(10, **SPEC[10])
    grid = p.plot_conditional_direct_effects(x="motiv", hue="skill", hue_format=fmt)
    labels = list(grid.hue_names)
    assert len(labels) == 3 and all(label.startswith("skill") for label in labels), labels
    plt.close("all")


def test_hue_accepts_at_most_two_moderators(fit):
    p = fit(10, **SPEC[10])
    with pytest.raises(ValueError, match="hue"):
        p.plot_conditional_direct_effects(x="motiv", hue=["skill", "skill", "skill"])


# --- #44: sample size with missing data ----------------------------------------------------


def test_missing_rows_are_counted(fit, data):
    df = data.copy()
    df.loc[df.index[:10], "med1"] = np.nan
    p = fit(4, df=df, x="effort", m=["med1"], y="outcome")
    assert p.n_obs == len(df) - 10
    assert p.n_obs_null == 10
    assert p.outcome_models["outcome"].estimation_results["n"] == len(df) - 10
    assert "index" not in p._data.columns


def test_column_named_index_is_allowed(fit, data):
    df = data.rename(columns={"med1": "index"})
    p = fit(4, df=df, x="effort", m=["index"], y="outcome")
    assert p.mediators == ["index"]
    assert "index" in p.outcome_models


# --- #45: seed handling ---------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, None, 2**32 - 1])
def test_seed_accepts_zero_none_and_the_full_range(fit, seed):
    p = fit(4, seed=seed, **SPEC[4])
    assert p.get_bootstrap_estimates().shape[0] == p.options["boot"] * 3


@pytest.mark.parametrize("seed", [-1, 2**32, 1.5, "12"])
def test_seed_rejects_invalid_values(fit, seed):
    with pytest.raises(ValueError, match="seed"):
        fit(4, seed=seed, **SPEC[4])


def test_same_seed_reproduces_bootstrap(fit):
    a = fit(4, seed=7, **SPEC[4]).get_bootstrap_estimates()
    b = fit(4, seed=7, **SPEC[4]).get_bootstrap_estimates()
    pd.testing.assert_frame_equal(a, b)


# --- #46: modval names are validated ---------------------------------------------------------


def test_modval_rejects_unknown_and_non_moderator_names(fit):
    with pytest.raises(ValueError, match="nonexistent"):
        fit(7, modval={"nonexistent": [1, 2]}, **SPEC[7])
    with pytest.raises(ValueError, match="med1"):
        fit(7, modval={"med1": [1, 2]}, **SPEC[7])
    with pytest.raises(ValueError, match="not moderators"):
        fit(4, modval={"effort": [1]}, **SPEC[4])


def test_modval_values_are_used(fit):
    p = fit(7, modval={"motiv": [-2.0, 2.0]}, **SPEC[7])
    assert list(p._spotlight_values["w"]) == [-2.0, 2.0]
    assert sorted(p.indirect_model.coeff_summary()["motiv"].unique()) == [-2.0, 2.0]


def test_plot_modval_rejects_unknown_names(fit):
    p = fit(10, **SPEC[10])
    with pytest.raises(ValueError, match="nonexistent"):
        p.plot_conditional_direct_effects(x="motiv", modval={"nonexistent": [1]})


# --- #47: unsupported options warn, unknown ones raise -----------------------------------------


@pytest.mark.parametrize("option", ["jn", "effsize", "mc", "normal", "varorder", "coeffci", "plot", "save"])
def test_unsupported_options_warn(fit, option):
    with pytest.warns(UserWarning, match=option):
        fit(4, **{option: True}, **SPEC[4])


def test_unknown_keyword_arguments_raise(fit):
    with pytest.raises(TypeError, match="boots"):
        fit(4, boots=10, **SPEC[4])


# --- #48: importing the package leaves the warning filters alone -------------------------------


def test_import_does_not_change_warning_filters():
    import subprocess
    import sys

    # The dependencies register filters of their own, so import them first and check that
    # importing the package on top of them changes nothing.
    code = (
        "import warnings, numpy, scipy.stats, scipy.special, pandas, matplotlib.pyplot, seaborn; "
        "before = list(warnings.filters); import pyprocessmacro; "
        "assert warnings.filters == before, (before, warnings.filters)"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


# --- #49: non-convergence is reported, bootstrap failures are capped ---------------------------


def test_separated_logit_raises_convergence_error(fit):
    from pyprocessmacro import ConvergenceError

    n = 60
    rng = np.random.default_rng(5)
    effort = np.linspace(-3, 3, n)
    df = pd.DataFrame(dict(effort=effort, med1=0.5 * effort + rng.normal(size=n), binary=(effort > 0).astype(int)))
    with pytest.raises(ConvergenceError):
        fit(4, df=df, x="effort", m=["med1"], y="binary", logit=True, boot=10, iterate=300)


def test_bootstrap_gives_up_after_too_many_failures(fit, monkeypatch):
    from numpy.linalg import LinAlgError

    import pyprocessmacro.models as models

    real = models.fast_OLS
    calls = {"n": 0}

    def flaky(endog, exog):
        calls["n"] += 1
        if calls["n"] > 3:  # after the three true fits (outcome, med1, med2) every resample fails
            raise LinAlgError("singular")
        return real(endog, exog)

    monkeypatch.setattr(models, "fast_OLS", flaky)
    with pytest.raises(RuntimeError, match="bootstrap samples failed"):
        fit(4, boot=20, **SPEC[4])


def test_bias_corrected_ci_is_finite_when_draws_fall_on_one_side():
    from pyprocessmacro.utils import bias_corrected_ci

    samples = np.linspace(1.0, 2.0, 200)
    low, high = bias_corrected_ci(0.5, samples, conf=95)  # every draw is above the estimate
    assert np.isfinite([low, high]).all()
    assert 1.0 <= low <= high <= 2.0


# --- #50: removed API ---------------------------------------------------------------------------


def test_deprecated_plot_methods_and_stubs_are_gone():
    import glob
    import os

    import pyprocessmacro
    from pyprocessmacro import Process

    assert not hasattr(Process, "plot_direct_effects")
    assert not hasattr(Process, "plot_indirect_effects")
    assert glob.glob(os.path.join(os.path.dirname(pyprocessmacro.__file__), "*.pyi")) == []


# --- #52: dv alias ------------------------------------------------------------------------------


def test_dv_names_the_outcome(fit):
    p = fit(4, **SPEC[4])
    assert p.dv == p.iv == "outcome"


# --- #71: summary() returns its text; notebook representation ---------------------------------------


def test_summary_returns_the_printed_text(fit, capsys):
    p = fit(7, **SPEC[7])
    text = p.summary()
    printed = capsys.readouterr().out
    assert isinstance(text, str) and text.strip() == printed.strip()
    assert str(p) == text
    assert "DIRECT AND INDIRECT EFFECTS" in text


def test_repr_html_has_every_table(fit):
    p = fit(7, **SPEC[7])
    html = p._repr_html_()
    assert html.count("<table") == 2 * len(p.outcome_models) + 3  # model + coefficients per outcome, direct, indirect, MM index
    assert "Index of moderated mediation" in html
    q = fit(1, **SPEC[1])
    assert "Conditional effect" in q._repr_html_()


# --- #72: tables are built from typed columns ---------------------------------------------------------


def test_tables_have_typed_columns(fit):
    p = fit(12, **SPEC[12])
    tables = [p.indirect_model.coeff_summary(), p.indirect_model.MMM_index_summary(), p.indirect_model.CMM_index_summary()]
    for table in tables:
        for column in table.columns:
            if column in ("Mediator", "Moderator", "Focal Mod", ""):
                assert not pd.api.types.is_numeric_dtype(table[column]), column
            else:
                assert pd.api.types.is_float_dtype(table[column]), (column, table[column].dtype)
    import pyprocessmacro.models as models
    assert not hasattr(models, "_coerce_numeric")
