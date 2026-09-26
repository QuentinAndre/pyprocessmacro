"""
Multicategorical X and moderators (#17): PROCESS's mcx and mcw options, compared to the output of PROCESS for R 5.0
on tests/Data/Data_MC.csv (tests/Results/v5/mc, made by tests/fixtures/regenerate.py --suite mc).

The comparison cases carry the `v5` marker like the other PROCESS 5 comparisons; the unit tests at the end run
with the fast suite.
"""
import os
import warnings

import numpy as np
import pandas as pd
import pytest

from pyprocessmacro import Process
from pyprocessmacro.categorical import Coding, code_matrix, expand_terms
from tests.v5_output_mc import parse_mc

HERE = os.path.dirname(os.path.abspath(__file__))
MC_DIR = os.path.join(HERE, "Results", "v5", "mc")
DATA = pd.read_csv(os.path.join(HERE, "Data", "Data_MC.csv"))

OUTCOME_TOL = dict(rtol=2e-4, atol=1e-6)
BOOT_TOL = 5e-2

# id -> Process arguments; every case adds boot=5000, seed=123456, percentile intervals, PROCESS 3 spotlights,
# HC3 unless hc3=False, total=True. PyProcessMacro's model 14 takes its moderator as v (2.16 numbering).
CASES = {
    "m4_mcx1": dict(model=4, x="xcat", m=["m"], y="y", mcx=1),
    "m4_mcx2": dict(model=4, x="xcat", m=["m"], y="y", mcx=2),
    "m4_mcx3": dict(model=4, x="xcat", m=["m"], y="y", mcx=3),
    "m4_mcx4": dict(model=4, x="xcat", m=["m"], y="y", mcx=4),
    "m4_mcx1_hc0": dict(model=4, x="xcat", m=["m"], y="y", mcx=1, hc3=False),
    "m4_2m_mcx1": dict(model=4, x="xcat", m=["m", "m2"], y="y", mcx=1),
    "m4_logit_mcx1": dict(model=4, x="xcat", m=["m"], y="y2", mcx=1, logit=True),
    "m6_mcx1": dict(model=6, x="xcat", m=["m", "m2"], y="y", mcx=1),
    "m1_mcx1": dict(model=1, x="xcat", m="w", y="y", mcx=1),
    "m1_mcx3": dict(model=1, x="xcat", m="w", y="y", mcx=3),
    "m1_mcw1": dict(model=1, x="x", m="wcat", y="y", mcw=1),
    "m1_mcw3": dict(model=1, x="x", m="wcat", y="y", mcw=3),
    "m1_mcx1_mcw1": dict(model=1, x="xcat", m="wcat", y="y", mcx=1, mcw=1),
    "m5_mcx1": dict(model=5, x="xcat", w="w", m=["m"], y="y", mcx=1),
    "m7_mcx1": dict(model=7, x="xcat", w="w", m=["m"], y="y", mcx=1),
    "m7_mcx3": dict(model=7, x="xcat", w="w", m=["m"], y="y", mcx=3),
    "m7_mcw1": dict(model=7, x="x", w="wcat", m=["m"], y="y", mcw=1),
    "m7_mcw3": dict(model=7, x="x", w="wcat", m=["m"], y="y", mcw=3),
    "m8_mcx1": dict(model=8, x="xcat", w="w", m=["m"], y="y", mcx=1),
    "m8_mcw1": dict(model=8, x="x", w="wcat", m=["m"], y="y", mcw=1),
    "m14_mcx1": dict(model=14, x="xcat", v="w", m=["m"], y="y", mcx=1),
    "m14_mcw1": dict(model=14, x="x", v="wcat", m=["m"], y="y", mcv=1),
    "m58_mcx1": dict(model=58, x="xcat", w="w", m=["m"], y="y", mcx=1),
}


def available():
    return [c for c in CASES if os.path.exists(os.path.join(MC_DIR, f"{c}.txt"))]


@pytest.fixture(scope="module")
def cache():
    return {}


def fitted(case, cache):
    """The parsed PROCESS 5 output and a Process fitted with matching options."""
    if case not in cache:
        with open(os.path.join(MC_DIR, f"{case}.txt"), encoding="utf-8") as f:
            parsed = parse_mc(f.read())
        kwargs = dict(CASES[case])
        hc3 = kwargs.pop("hc3", True)
        p = Process(DATA, boot=5000, seed=123456, conf=95, total=True, hc3=hc3, percent=True,
                    spotlight="percentiles", suppr_init=True, **kwargs)
        cache[case] = (parsed, p)
    return cache[case]


def their_label(label, p):
    """PROCESS labels every moderator's codes W1, W2, ...; PyProcessMacro labels them by the symbol (V1, V2 for v)."""
    for coding in p.categorical.values():
        if coding.letter not in ("X", "W") and label.startswith(coding.letter):
            return "W" + label[len(coding.letter):]
    return label


def their_term(term, p):
    return "*".join(their_label(f, p) for f in term.split("*"))


def _by_values(frame, columns):
    if not columns:
        return frame.reset_index(drop=True)
    return frame.sort_values(list(columns)).reset_index(drop=True)


# --- comparisons with PROCESS 5 -----------------------------------------------------------------------

pytestmark = pytest.mark.v5


@pytest.mark.parametrize("case", available())
def test_coding_tables_match(case, cache):
    parsed, p = fitted(case, cache)
    ours_by_letter = {("W" if c.letter in ("W", "V") else c.letter): c for c in p.categorical.values()}
    assert set(parsed["codings"]) == set(ours_by_letter)
    for letter, theirs in parsed["codings"].items():
        ours = ours_by_letter[letter].table()
        np.testing.assert_allclose(ours.iloc[:, 0].to_numpy(dtype=float), theirs.iloc[:, 0].to_numpy(dtype=float))
        np.testing.assert_allclose(ours.iloc[:, 1:].to_numpy(dtype=float), theirs.iloc[:, 1:].to_numpy(dtype=float), atol=1e-6)


@pytest.mark.parametrize("case", available())
def test_outcome_models_match(case, cache):
    parsed, p = fitted(case, cache)
    assert set(parsed["outcomes"]) == set(p.outcome_models)
    for name, block in parsed["outcomes"].items():
        ours = p.outcome_models[name].coeff_summary()
        ours.index = [their_term(t, p) for t in ours.index]
        theirs = block["coefficients"]
        assert set(theirs.index) == set(ours.index), (name, list(theirs.index), list(ours.index))
        theirs = theirs.loc[ours.index]
        stat = "Z" if "Z" in ours.columns else "t"
        np.testing.assert_allclose(
            ours[["coeff", "se", stat, "p", "LLCI", "ULCI"]].to_numpy(dtype=float),
            theirs[["coeff", "se", "stat", "p", "LLCI", "ULCI"]].to_numpy(dtype=float), **OUTCOME_TOL,
        )


@pytest.mark.parametrize("case", available())
def test_direct_effects_match(case, cache):
    parsed, p = fitted(case, cache)
    theirs = parsed["direct"]
    if theirs is None:
        pytest.skip("no direct-effect table in this file (moderation-only model)")
    ours = p.direct_model.coeff_summary()
    stat = "Z" if "Z" in ours.columns else "t"
    keys = [c for c in theirs.columns if c not in ("effect", "se", "stat", "p", "LLCI", "ULCI")]
    assert set(keys) <= set(ours.columns), (keys, list(ours.columns))
    theirs, ours = _by_values(theirs, keys), _by_values(ours, keys)
    assert len(theirs) == len(ours)
    for k in keys:
        if k == "X":
            assert list(ours[k]) == list(theirs[k])
        else:
            np.testing.assert_allclose(ours[k].to_numpy(dtype=float), theirs[k].to_numpy(dtype=float), atol=2e-5)
    np.testing.assert_allclose(
        ours[["Effect", "SE", stat, "p", "LLCI", "ULCI"]].to_numpy(dtype=float),
        theirs[["effect", "se", "stat", "p", "LLCI", "ULCI"]].to_numpy(dtype=float), **OUTCOME_TOL,
    )


@pytest.mark.parametrize("case", available())
def test_omnibus_test_matches(case, cache):
    parsed, p = fitted(case, cache)
    theirs = parsed["omnibus"]
    if theirs is None:
        pytest.skip("no omnibus test in this file")
    ours = p.direct_model.omnibus_test()
    assert list(ours.columns) == list(theirs.columns), (list(ours.columns), list(theirs.columns))
    np.testing.assert_allclose(ours.to_numpy(dtype=float), theirs.to_numpy(dtype=float), rtol=2e-4, atol=1e-6)


@pytest.mark.parametrize("case", [c for c in available() if CASES[c]["model"] <= 3])
def test_conditional_effects_of_moderation_models_match(case, cache):
    parsed, p = fitted(case, cache)
    ours = p.direct_model.coeff_summary()
    blocks = parsed["conditional"]
    if isinstance(blocks, dict):  # continuous X, categorical W: one table with the groups
        theirs = blocks["table"]
        mod = [c for c in theirs.columns if c not in ("effect", "se", "stat", "p", "LLCI", "ULCI")][0]
        theirs, ours = _by_values(theirs, [mod]), _by_values(ours, [mod])
        np.testing.assert_allclose(ours[mod].to_numpy(dtype=float), theirs[mod].to_numpy(dtype=float))
        np.testing.assert_allclose(ours[["Effect", "SE", "t", "p", "LLCI", "ULCI"]].to_numpy(dtype=float),
                                   theirs[["effect", "se", "stat", "p", "LLCI", "ULCI"]].to_numpy(dtype=float), **OUTCOME_TOL)
        return
    for block in blocks:  # categorical X: one table of relative effects per moderator value, with an equality test
        (mod, value), = block["at"].items()
        rows = ours[np.isclose(ours[mod].to_numpy(dtype=float), value, atol=2e-5)]
        theirs = block["effects"]
        assert list(rows["X"]) == list(theirs["X"])
        np.testing.assert_allclose(rows[["Effect", "SE", "t", "p", "LLCI", "ULCI"]].to_numpy(dtype=float),
                                   theirs[["effect", "se", "stat", "p", "LLCI", "ULCI"]].to_numpy(dtype=float), **OUTCOME_TOL)
        equality = p.direct_model.omnibus_test(at={mod: rows[mod].iloc[0]})
        np.testing.assert_allclose(equality[["F", "df1", "df2", "p"]].to_numpy(dtype=float),
                                   block["equality"][["F", "df1", "df2", "p"]].to_numpy(dtype=float), rtol=2e-4, atol=1e-6)


@pytest.mark.parametrize("case", [c for c in available() if CASES[c]["model"] > 3])
def test_indirect_effects_match(case, cache):
    parsed, p = fitted(case, cache)
    ours_all = p.indirect_model.coeff_summary()
    close = []
    for path, theirs in parsed["indirect"].items():
        if "Mediator" in ours_all.columns:  # conditional indirect effects
            ours = ours_all[ours_all["Mediator"] == path.split(" -> ")[1]]
        elif CASES[case]["model"] == 6:
            ours = ours_all[ours_all[""] == path]
        else:
            ours = ours_all[ours_all[""] == path.split(" -> ")[1]]
        keys = [c for c in theirs.columns if c not in ("Effect", "BootSE", "BootLLCI", "BootULCI")]
        theirs, ours = _by_values(theirs, keys), _by_values(ours, keys)
        assert len(theirs) == len(ours) > 0, (path, len(theirs), len(ours))
        for k in keys:
            if k == "X":
                assert list(ours[k]) == list(theirs[k])
            else:
                np.testing.assert_allclose(ours[k].to_numpy(dtype=float), theirs[k].to_numpy(dtype=float), atol=2e-5)
        np.testing.assert_allclose(ours["Effect"].to_numpy(dtype=float), theirs["Effect"].to_numpy(dtype=float), **OUTCOME_TOL)
        close.append(np.isclose(ours[["Boot SE", "BootLLCI", "BootULCI"]].to_numpy(dtype=float),
                                theirs[["BootSE", "BootLLCI", "BootULCI"]].to_numpy(dtype=float), rtol=BOOT_TOL, atol=BOOT_TOL))
    assert np.mean(np.concatenate([c.ravel() for c in close])) > 0.8


@pytest.mark.parametrize("case", [c for c in available() if CASES[c]["model"] > 3])
def test_indices_match(case, cache):
    parsed, p = fitted(case, cache)
    im = p.indirect_model
    if not parsed["indices"]:
        assert im._analysis_list == [], im._analysis_list
        return
    assert im._analysis_list == ["MM"]
    ours_all = im.MM_index_summary()
    ours_all["Moderator"] = [their_label(m, p) for m in ours_all["Moderator"]]
    for path, blocks in parsed["indices"].items():
        mediator = path.split(" -> ")[1]
        for label, theirs in blocks.items():
            ours = ours_all[ours_all["Mediator"] == mediator]
            if label:
                ours = ours[ours["X"] == label]
            ours = ours.set_index("Moderator")
            assert list(ours.index) == list(theirs.index), (list(ours.index), list(theirs.index))
            np.testing.assert_allclose(ours["Index"].to_numpy(dtype=float), theirs["Index"].to_numpy(dtype=float), **OUTCOME_TOL)
            assert np.isclose(ours[["Boot SE", "LLCI", "ULCI"]].to_numpy(dtype=float),
                              theirs[["BootSE", "BootLLCI", "BootULCI"]].to_numpy(dtype=float), rtol=BOOT_TOL, atol=BOOT_TOL).mean() > 0.6


# --- unit tests (fast) --------------------------------------------------------------------------------

def test_code_matrices_are_processs():
    np.testing.assert_allclose(code_matrix(4, "indicator"), [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    np.testing.assert_allclose(code_matrix(4, "sequential"), [[0, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1]])
    np.testing.assert_allclose(code_matrix(4, "helmert"),
                               [[-0.75, 0, 0], [0.25, -2 / 3, 0], [0.25, 1 / 3, -0.5], [0.25, 1 / 3, 0.5]])
    np.testing.assert_allclose(code_matrix(4, "effect"), [[-1, -1, -1], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    assert expand_terms(["Cons", "x", "w", "x*w"], {"x": Coding("c", [1, 1, 2, 2, 3, 3], "indicator", "X", "x")}) == \
        ["Cons", "x1", "x2", "w", "x1*w", "x2*w"]


@pytest.mark.parametrize("scheme", [1, 2, 3, 4, "indicator", "sequential", "helmert", "effect"])
def test_scheme_names_and_numbers(scheme):
    p = Process(DATA, 4, x="xcat", m=["m"], y="y", mcx=scheme, boot=50, seed=1, suppr_init=True)
    assert p.categorical["x"].scheme in ("indicator", "sequential", "helmert", "effect")
    assert list(p.outcome_models["m"].coeff_summary().index) == ["Cons", "X1", "X2", "X3"]


def test_option_validation():
    with pytest.raises(ValueError, match="'mcx'"):
        Process(DATA, 4, x="xcat", m=["m"], y="y", mcx=7, boot=50, suppr_init=True)
    with pytest.raises(ValueError, match="at least three categories"):
        Process(DATA, 4, x="y2", m=["m"], y="y", mcx=1, boot=50, suppr_init=True)
    with pytest.raises(ValueError, match="not available with a multicategorical X"):
        Process(DATA, 4, x="xcat", m=["m", "m2"], y="y", mcx=1, contrast=True, boot=50, suppr_init=True)
    with pytest.raises(ValueError, match="does not have"):
        Process(DATA, 4, x="xcat", m=["m"], y="y", mcw=1, boot=50, suppr_init=True)
    with pytest.raises(ValueError, match="Model 74"):
        Process(DATA, 74, x="xcat", m=["m"], y="y", mcx=1, boot=50, suppr_init=True)
    few = DATA.copy()
    few.loc[few["xcat"] == 4, "xcat"] = 3
    few.loc[few.index[:1], "xcat"] = 4  # one case in group 4
    with pytest.raises(ValueError, match="at least two cases"):
        Process(few, 4, x="xcat", m=["m"], y="y", mcx=1, boot=50, suppr_init=True)


def test_string_and_categorical_levels():
    df = DATA.copy()
    df["group"] = df["xcat"].map({1: "control", 2: "low", 3: "medium", 4: "high"})
    p = Process(df, 4, x="group", m=["m"], y="y", mcx=1, boot=50, seed=1, suppr_init=True)
    assert p.categorical["x"].levels == ["control", "high", "low", "medium"]  # sorted, as PROCESS orders numbers
    assert list(p.categorical["x"].table()["group"]) == ["control", "high", "low", "medium"]
    df["group"] = pd.Categorical(df["group"])
    q = Process(df, 4, x="group", m=["m"], y="y", mcx=1, boot=50, seed=1, suppr_init=True)
    np.testing.assert_allclose(q.outcome_models["m"].estimation_results["betas"], p.outcome_models["m"].estimation_results["betas"])
    assert "control" in q.summary()


def test_tidy_and_bootstrap_exports_carry_the_codes():
    p = Process(DATA, 7, x="xcat", w="w", m=["m"], y="y", mcx=1, boot=60, seed=1, suppr_init=True)
    t = p.tidy()
    assert "x_code" in t.columns
    direct = t[t["component"] == "direct"]
    assert list(direct["term"]) == ["X1", "X2", "X3"]
    indirect = t[t["component"] == "indirect"]
    assert set(indirect["x_code"]) == {"X1", "X2", "X3"} and len(indirect) == 9
    index = t[t["component"] == "index_mm"]
    assert len(index) == 3 and set(index["moderator"]) == {"w"}
    assert list(p.get_bootstrap_estimates().columns)[:6] == ["BootSample", "OutcomeName", "Cons", "X1", "X2", "X3"]
    q = Process(DATA, 4, x="x", m=["m"], y="y", boot=60, seed=1, suppr_init=True)
    assert "x_code" not in q.tidy().columns


def test_effect_sizes_are_partially_standardized_only():
    p = Process(DATA, 4, x="xcat", m=["m"], y="y", mcx=1, effsize=True, boot=60, seed=1, suppr_init=True)
    table = p.indirect_model.effect_size_summary()
    assert set(table["Standardization"]) == {"partially"} and set(table["X"]) == {"X1", "X2", "X3"}
    text = p.summary()
    assert "Partially standardized" in text and "Completely standardized" not in text
    sd_y = DATA["y"].std(ddof=1)
    raw = p.indirect_model.coeff_summary()
    np.testing.assert_allclose(table["Effect"].to_numpy(dtype=float), raw["Effect"].to_numpy(dtype=float) / sd_y)


def test_categorical_moderator_probes_its_groups_and_refuses_other_values():
    p = Process(DATA, 7, x="x", w="wcat", m=["m"], y="y", mcw=1, boot=60, seed=1, suppr_init=True)
    assert list(p._spotlight_values["w"]) == [1, 2, 3]
    assert list(p.direct_model.coeff_summary().columns) == ["Effect", "SE", "t", "p", "LLCI", "ULCI"]
    assert list(p.indirect_model.coeff_summary()["wcat"]) == [1, 2, 3]
    assert list(p.indirect_model.MM_index_summary()["Moderator"]) == ["W1", "W2"]
    sub = p.spotlight_indirect_effect("m", spotval={"wcat": [1, 3]})
    assert list(sub["wcat"]) == [1, 3]
    with pytest.raises(ValueError, match="not groups"):
        Process(DATA, 7, x="x", w="wcat", m=["m"], y="y", mcw=1, modval={"wcat": [1.5]}, boot=60, suppr_init=True)
    with pytest.raises(NotImplementedError):
        p.floodlight_indirect_effect("m", "wcat")


def test_two_moderator_indices_are_not_reported_with_a_categorical_moderator():
    with pytest.warns(UserWarning, match="multicategorical moderator"):
        p = Process(DATA, 9, x="x", w="wcat", z="w", m=["m"], y="y", mcw=1, boot=60, seed=1, suppr_init=True)
    assert p.indirect_model._analysis_list == []
    with pytest.raises(NotImplementedError):
        p.indirect_model._PMM_index()


def test_relative_effects_summary_and_spotlight():
    p = Process(DATA, 1, x="xcat", m="w", y="y", mcx=1, boot=50, seed=1, suppr_init=True)
    text = p.summary()
    assert "Coding of the multicategorical X variable xcat (indicator coding)" in text
    assert "Relative conditional effect(s) of xcat on y" in text
    assert "Test of equality of the conditional means of y across the groups of xcat" in text
    spot = p.spotlight_direct_effect(spotval={"w": [0.0]})
    assert list(spot["X"]) == ["X1", "X2", "X3"]
    with pytest.raises(NotImplementedError):
        p.plot_conditional_direct_effects(x="w")
    q = Process(DATA, 4, x="xcat", m=["m"], y="y", mcx=1, boot=50, seed=1, suppr_init=True)
    assert "Relative direct effect of xcat on y" in q.summary() and "Omnibus test of direct effect" in q.summary()
    assert q.direct_model.probe_p is None
    r = Process(DATA, 1, x="xcat", m="wcat", y="y", mcx=1, mcw=1, boot=50, seed=1, suppr_init=True)
    assert 0 <= r.direct_model.probe_p <= 1 and len(r.direct_model.probe_terms) == 6
