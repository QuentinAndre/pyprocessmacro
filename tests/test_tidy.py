"""
tidy(), glance() and augment() carry exactly the numbers of the printed tables under fixed names.
"""
import numpy as np
import pandas as pd
import pytest

from pyprocessmacro.tidy import GLANCE_COLUMNS, META_COLUMNS, STAT_COLUMNS

pytestmark = pytest.mark.smoke

SPEC = {
    1: dict(x="effort", m="motiv", y="outcome"),
    4: dict(x="effort", m=["med1", "med2"], y="outcome", total=True, contrast=True),
    7: dict(x="effort", w="motiv", m=["med1", "med2"], y="outcome"),
    10: dict(x="effort", w="motiv", z="skill", m=["med1"], y="outcome"),
    12: dict(x="effort", w="motiv", z="skill", m=["med1", "med2"], y="outcome"),
    14: dict(x="effort", v="value", m=["med1"], y="binary", logit=True),
    21: dict(x="effort", w="motiv", v="value", m=["med1"], y="outcome"),
    75: dict(x="effort", w="motiv", z="skill", m=["med1", "med2"], y="outcome"),
}


def _stats(frame, cols):
    return frame[list(cols)].to_numpy(dtype=float)


@pytest.mark.parametrize("model", sorted(SPEC))
def test_tidy_columns_and_types(fit, model):
    p = fit(model, **SPEC[model])
    t = p.tidy()
    moderators = sorted(p._symb_to_var[s] for s in p._moderators["all"])
    assert list(t.columns) == ["component", "outcome", "term", "moderator"] + moderators + STAT_COLUMNS + META_COLUMNS
    for c in STAT_COLUMNS + ["conf_level", "n_boot"] + moderators:
        assert pd.api.types.is_float_dtype(t[c]) or pd.api.types.is_integer_dtype(t[c]), c
    assert (t["conf_level"] == 95).all()
    assert set(t["component"]) <= {"outcome", "direct", "indirect", "total", "contrast",
                                   "index_mm", "index_pmm", "index_mmm", "index_cmm"}
    boot = t["component"].isin(["indirect", "total", "contrast"]) | t["component"].str.startswith("index")
    assert (t.loc[boot, "n_boot"] == p.options["boot"]).all()
    assert t.loc[boot, "method"].isin(["bootstrap_bc", "bootstrap_percentile"]).all()
    assert t.loc[boot, ["statistic", "p_value"]].isna().all().all()
    assert t.loc[~boot, "n_boot"].isna().all()


@pytest.mark.parametrize("model", sorted(SPEC))
def test_tidy_outcome_rows_match_coeff_summary(fit, model):
    p = fit(model, **SPEC[model])
    t = p.tidy("outcome")
    for outcome, om in p.outcome_models.items():
        table = om.coeff_summary()
        rows = t[t["outcome"] == outcome]
        assert list(rows["term"]) == list(table.index)
        stat = "Z" if "Z" in table.columns else "t"
        np.testing.assert_allclose(
            _stats(rows, STAT_COLUMNS), table[["coeff", "se", stat, "p", "LLCI", "ULCI"]].to_numpy(dtype=float)
        )
        assert set(rows["method"]) == {"logit" if stat == "Z" else "ols"}


@pytest.mark.parametrize("model", sorted(SPEC))
def test_tidy_direct_rows_match_direct_table(fit, model):
    p = fit(model, **SPEC[model])
    t = p.tidy("direct")
    table = p.direct_model.coeff_summary()
    stat = "Z" if "Z" in table.columns else "t"
    assert len(t) == len(table)
    np.testing.assert_allclose(_stats(t, STAT_COLUMNS), table[["Effect", "SE", stat, "p", "LLCI", "ULCI"]].to_numpy(dtype=float))
    for mod in [c for c in table.columns if c not in ("Effect", "SE", stat, "p", "LLCI", "ULCI")]:
        np.testing.assert_allclose(t[mod].to_numpy(dtype=float), table[mod].to_numpy(dtype=float))
    assert set(t["term"]) == {"effort"}


@pytest.mark.parametrize("model", [m for m in sorted(SPEC) if m > 3])
def test_tidy_indirect_rows_match_indirect_table(fit, model):
    p = fit(model, **SPEC[model])
    t = p.tidy(["indirect", "total", "contrast"])
    table = p.indirect_model.coeff_summary()
    assert len(t) == len(table)
    np.testing.assert_allclose(
        _stats(t, ["estimate", "std_error", "conf_low", "conf_high"]),
        table[["Effect", "Boot SE", "BootLLCI", "BootULCI"]].to_numpy(dtype=float),
    )
    if p.indirect_model._has_moderation:
        assert list(t["term"]) == list(table["Mediator"])
        for mod in [c for c in table.columns if c not in ("Mediator", "Effect", "Boot SE", "BootLLCI", "BootULCI")]:
            np.testing.assert_allclose(t[mod].to_numpy(dtype=float), table[mod].to_numpy(dtype=float))
    else:
        labels = list(table[""])
        expected = ["total" if l == "TOTAL" else l.replace("Contrast: ", "") for l in labels]
        assert list(t["term"]) == expected
        assert list(t["component"]) == ["total"] + ["indirect"] * 2 + ["contrast"]


@pytest.mark.parametrize("model, codes", [(7, ["MM"]), (10, ["PMM"]), (12, ["MMM", "CMM"]), (21, ["MMM", "CMM"]), (75, [])])
def test_tidy_index_rows_match_index_tables(fit, model, codes):
    p = fit(model, **SPEC[model])
    t = p.tidy()
    assert sorted(set(t.loc[t["component"].str.startswith("index"), "component"])) == sorted(f"index_{c.lower()}" for c in codes)
    for code in codes:
        rows = t[t["component"] == f"index_{code.lower()}"]
        table = getattr(p.indirect_model, f"{code}_index_summary")()
        assert len(rows) == len(table)
        low, high = ("LLCI", "ULCI") if "LLCI" in table.columns else ("BootLLCI", "BootULCI")
        np.testing.assert_allclose(
            _stats(rows, ["estimate", "std_error", "conf_low", "conf_high"]),
            table[["Index", "Boot SE", low, high]].to_numpy(dtype=float),
        )
        assert list(rows["term"]) == list(table["Mediator"])
        if code in ("MM", "PMM"):
            assert list(rows["moderator"]) == list(table["Moderator"])
        if code == "CMM":
            assert list(rows["moderator"]) == list(table["Focal Mod"])
            other = [c for c in rows.columns if c in ("motiv", "skill", "value")]
            at = rows[other].bfill(axis=1).iloc[:, 0].to_numpy(dtype=float)
            np.testing.assert_allclose(at, table["Other Mod At"].to_numpy(dtype=float))
        if code == "MMM":
            assert rows["moderator"].isna().all()


def test_tidy_component_filter(fit):
    p = fit(7, **SPEC[7])
    assert set(p.tidy("outcome")["component"]) == {"outcome"}
    assert set(p.tidy(["direct", "indirect"])["component"]) == {"direct", "indirect"}
    with pytest.raises(ValueError, match="Unknown component"):
        p.tidy("nope")


@pytest.mark.parametrize("model", [4, 14])
def test_glance_matches_model_summary(fit, model):
    p = fit(model, **SPEC[model])
    g = p.glance()
    assert list(g.columns) == GLANCE_COLUMNS
    assert list(g["outcome"]) == list(p.outcome_models)
    for _, row in g.iterrows():
        res = p.outcome_models[row["outcome"]].estimation_results
        assert row["n"] == res["n"]
        assert np.isfinite([row["log_likelihood"], row["aic"], row["bic"]]).all()
        if row["estimator"] == "ols":
            assert row["r_squared"] == res["R2"] and row["adj_r_squared"] == res["adjR2"]
            assert row["f_statistic"] == res["F"] and row["p_value"] == res["F_pval"]
            assert (row["df_model"], row["df_resid"]) == (res["df_r"], res["df_e"])
            assert np.isnan(row["mcfadden"])
        else:
            assert row["mcfadden"] == res["mcfadden"] and row["lr_statistic"] == res["d"]
            assert row["p_value"] == res["pvalue"] and row["cov_type"] == "hessian"
            assert np.isnan(row["r_squared"])
