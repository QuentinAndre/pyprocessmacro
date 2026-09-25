"""
Fast checks of the public API on synthetic data.

These exercise the code paths the accuracy suite never touches (summaries,
index tables, bootstrap export, plotting, floodlight, controls), so that
compatibility breakages such as pandas API removals are caught before release.
"""
import warnings

import matplotlib.pyplot as plt
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

# Which index tables a model reports today (2.0 revisits the both-path cases, see #43).
INDEX_MODELS = {
    7: ["MM"],  # one moderator on x -> m
    10: ["PMM"],  # two moderators on x -> m, no three-way term
    12: ["MMM", "CMM"],  # two moderators on x -> m with a three-way term
    14: ["MM"],  # one moderator on m -> y
    21: ["MMM", "CMM"],  # one moderator on each path
    75: ["PMM"],  # two moderators on both paths, no three-way term
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


@pytest.mark.parametrize("model", sorted(INDEX_MODELS))
def test_index_tables_are_numeric(fit, model):
    p = fit(model, **SPEC[model])
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
