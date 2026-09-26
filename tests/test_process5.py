"""
Comparison with PROCESS for R version 5 (tests/Results/v5), the reference for the 3.x parity work.

Feature parity with PROCESS 5 is a 3.x target (#82). These tests therefore assert only what the 2.x
releases already claim to reproduce, fitted at PROCESS 5's own moderator values with percentile
intervals:

* outcome models: coefficients, standard errors, t or Z, p-values and intervals;
* direct and conditional direct effects;
* indirect, conditional indirect and total effects and contrasts (point estimates, and bootstrap
  statistics within a Monte Carlo tolerance, since PROCESS resamples with R's generator);
* every index of moderated mediation PROCESS 5 prints, and none where it prints none;
* the completely standardized indirect effects of models 4 and 6.

Deliberate differences between 2.x and PROCESS 5, to be revisited for 3.0, are listed in
DIFFERENCES and are not asserted here.
"""
import glob
import os
import re

import numpy as np
import pandas as pd
import pytest

from pyprocessmacro import Process
from tests.test_index_reporting import load
from tests.v5_output import moderator_values, parse

pytestmark = pytest.mark.v5

V5_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Results", "v5")

DIFFERENCES = {
    "intervals": "PROCESS 5 reports percentile bootstrap intervals by default; 2.x bias-corrected (#74).",
    "spotlight values": "PROCESS 5 probes continuous moderators at the 16th, 50th and 84th percentiles; "
                        "2.x at the mean and plus or minus one standard deviation.",
    "conditional contrasts": "PROCESS 5 prints pairwise contrasts between conditional indirect effects; 2.x does not.",
    "interaction tests": "PROCESS 5 prints R-squared-change tests of the highest-order interaction; 2.x does not.",
    "total effect model": "PROCESS 5 prints a total-effect model for the unmoderated mediation models; 2.x does not.",
    "standardized direct": "PROCESS 5 prints standardized total and direct effects (c_cs, c'_cs) with effsize; 2.x does not.",
    "partially standardized": "PROCESS 5 no longer prints partially standardized indirect effects; 2.x does (#70).",
    "model 6 path order": "PROCESS 5 orders serial paths by length before first mediator; 2.x follows 2.16.",
    "probing": "PROCESS 5 probes an interaction only when its p-value is below intprobe (0.10 by default); 2.x always "
               "reports conditional effects. The files were generated with intprobe=1.",
    "logit three-way probing": "PROCESS 5.0 fails (object dfres not found) when probing a three-way interaction on a "
                               "logistic outcome, so the Logit Model 3 file has no conditional table (default probing).",
    "model 74": "PROCESS 5 has no model 74 (invalid model number); 2.x keeps it as defined in 2.16.",
    "models 23 to 27 and 30 to 57": "PROCESS 5 has no models with three or four moderators; 2.x keeps them as in 2.16.",
}

OUTCOME_TOL = dict(rtol=2e-4, atol=1e-6)   # six printed decimals; different linear algebra
BOOT_TOL = {"OLS": 5e-2, "Logit": 1e-1}   # different resampler: agreement within Monte Carlo error


def available():
    cases = []
    for path in sorted(glob.glob(os.path.join(V5_DIR, "Results_*_Model*.txt"))):
        kind, model = re.search(r"Results_(OLS|Logit)_Model(\d+)\.txt", path).groups()
        cases.append((int(model), kind))
    return sorted(cases)


@pytest.fixture(scope="module")
def cache():
    return {}


def fitted(model, kind, cache):
    """The parsed PROCESS 5 output and a Process fitted with matching options at PROCESS's moderator values."""
    key = (model, kind)
    if key not in cache:
        txt, data, kwargs = load(model, kind)
        with open(os.path.join(V5_DIR, f"Results_{kind}_Model{model}.txt"), encoding="utf-8") as f:
            parsed = parse(f.read())
        moderators = [v for v in ("m", "w", "z", "v", "q") if v in data.columns and (model <= 3 or v != "m")]
        modval = moderator_values(parsed, moderators)
        effsize = model in (4, 6) and kind == "OLS"
        p = Process(data, model, boot=5000, seed=123456, conf=95, total=True, contrast=True, hc3=True,
                    percent=True, logit=(kind == "Logit"), modval=modval, effsize=effsize, suppr_init=True, **kwargs)
        cache[key] = (parsed, p)
    return cache[key]


def _by_values(frame, columns):
    """Sort a table by its moderator columns so two tables with the same values align row by row."""
    if not columns:
        return frame.reset_index(drop=True)
    return frame.sort_values(list(columns)).reset_index(drop=True)


@pytest.mark.parametrize("model, kind", available())
def test_outcome_models_match(model, kind, cache):
    parsed, p = fitted(model, kind, cache)
    assert set(parsed["outcomes"]) == set(p.outcome_models)
    for name, block in parsed["outcomes"].items():
        ours = p.outcome_models[name].coeff_summary()
        theirs = block["coefficients"]
        assert set(theirs.index) == set(ours.index), (name, list(theirs.index), list(ours.index))
        theirs = theirs.loc[ours.index]  # PROCESS orders product terms differently
        stat = "Z" if "Z" in ours.columns else "t"  # mediator equations are OLS even with a binary Y
        np.testing.assert_allclose(
            ours[["coeff", "se", stat, "p", "LLCI", "ULCI"]].to_numpy(dtype=float),
            theirs[["coeff", "se", "stat", "p", "LLCI", "ULCI"]].to_numpy(dtype=float), **OUTCOME_TOL,
        )


@pytest.mark.parametrize("model, kind", available())
def test_direct_effects_match(model, kind, cache):
    parsed, p = fitted(model, kind, cache)
    theirs = parsed["direct"]
    if theirs is None:
        pytest.skip(DIFFERENCES["logit three-way probing"])
    ours = p.direct_model.coeff_summary()
    stat = "Z" if kind == "Logit" else "t"
    mods = [c for c in theirs.columns if c not in ("effect", "se", "stat", "p", "LLCI", "ULCI", "c'_cs")]
    theirs, ours = _by_values(theirs, mods), _by_values(ours, mods)
    assert len(theirs) == len(ours)
    for m in mods:
        np.testing.assert_allclose(ours[m].to_numpy(dtype=float), theirs[m].to_numpy(dtype=float), atol=2e-5)
    np.testing.assert_allclose(
        ours[["Effect", "SE", stat, "p", "LLCI", "ULCI"]].to_numpy(dtype=float),
        theirs[["effect", "se", "stat", "p", "LLCI", "ULCI"]].to_numpy(dtype=float), **OUTCOME_TOL,
    )


def _our_label(component, term, stv_paths):
    if component == "total":
        return "TOTAL"
    if component == "contrast":
        return term.replace(" vs. ", " minus ")
    return term


@pytest.mark.parametrize("model, kind", [c for c in available() if c[0] > 3])
def test_indirect_effects_match(model, kind, cache):
    parsed, p = fitted(model, kind, cache)
    im = p.indirect_model
    close = []
    if "unmoderated" in parsed["indirect"]:
        theirs = parsed["indirect"]["unmoderated"]
        ours = im.coeff_summary()
        rows_theirs, rows_ours = [], []
        for (component, term), (_, row) in zip(im.effect_labels, ours.iterrows()):
            if component == "contrast":  # PROCESS may order a pair the other way round: match the pair, flip the sign
                a, b = term.split(" vs. ")
                if f"{a} minus {b}" in theirs.index:
                    their, sign = theirs.loc[f"{a} minus {b}"], 1.0
                else:
                    their, sign = theirs.loc[f"{b} minus {a}"], -1.0
                bounds = sorted([sign * their["BootLLCI"], sign * their["BootULCI"]])
                rows_theirs.append([sign * their["Effect"], their["BootSE"], bounds[0], bounds[1]])
            else:
                their = theirs.loc["TOTAL" if component == "total" else term]
                rows_theirs.append([their["Effect"], their["BootSE"], their["BootLLCI"], their["BootULCI"]])
            rows_ours.append([row["Effect"], row["Boot SE"], row["BootLLCI"], row["BootULCI"]])
        rows_theirs, rows_ours = np.array(rows_theirs), np.array(rows_ours)
        np.testing.assert_allclose(rows_ours[:, 0], rows_theirs[:, 0], **OUTCOME_TOL)
        close = np.isclose(rows_ours[:, 1:], rows_theirs[:, 1:], rtol=BOOT_TOL[kind], atol=BOOT_TOL[kind])
    else:
        ours_all = im.coeff_summary()
        for path, theirs in parsed["indirect"]["conditional"].items():
            mediator = path.split(" -> ")[1]
            ours = ours_all[ours_all["Mediator"] == mediator]
            mods = [c for c in theirs.columns if c not in ("Effect", "BootSE", "BootLLCI", "BootULCI")]
            theirs, ours = _by_values(theirs, mods), _by_values(ours, mods)
            assert len(theirs) == len(ours), path
            for m in mods:
                np.testing.assert_allclose(ours[m].to_numpy(dtype=float), theirs[m].to_numpy(dtype=float), atol=2e-5)
            np.testing.assert_allclose(ours["Effect"].to_numpy(dtype=float), theirs["Effect"].to_numpy(dtype=float), **OUTCOME_TOL)
            close.append(np.isclose(ours[["Boot SE", "BootLLCI", "BootULCI"]].to_numpy(dtype=float),
                                    theirs[["BootSE", "BootLLCI", "BootULCI"]].to_numpy(dtype=float),
                                    rtol=BOOT_TOL[kind], atol=BOOT_TOL[kind]))
        close = np.concatenate([c.ravel() for c in close])
    assert np.mean(close) > 0.8, np.mean(close)


@pytest.mark.parametrize("model, kind", [c for c in available() if c[0] > 3])
def test_indices_match(model, kind, cache):
    parsed, p = fitted(model, kind, cache)
    im = p.indirect_model
    their_codes = set()
    for blocks in parsed["indices"].values():
        their_codes |= set(blocks)
    assert their_codes == set(im._analysis_list), (their_codes, im._analysis_list)
    stv = p._symb_to_var
    for path, blocks in parsed["indices"].items():
        mediator = path.split(" -> ")[1]
        for code, table in blocks.items():
            if code in ("MM", "PMM"):
                ours = getattr(im, f"{code}_index_summary")()
                ours = ours[ours["Mediator"] == mediator].set_index("Moderator")
                for moderator, row in table.iterrows():
                    assert ours.loc[moderator, "Index"] == pytest.approx(row["Index"], rel=2e-4, abs=1e-6), (code, path, moderator)
            elif code == "MMM":
                ours = im.MMM_index_summary()
                value = ours.loc[ours["Mediator"] == mediator, "Index"].iloc[0]
                assert value == pytest.approx(table["Index"].iloc[0], rel=2e-4, abs=1e-6), (code, path)
            elif code == "CMM":
                ours = im.CMM_index_summary()
                indirect_mods = [c for c in parsed["indirect"]["conditional"][path].columns
                                 if c not in ("Effect", "BootSE", "BootLLCI", "BootULCI")]
                for focal, sub in table.items():
                    other = [c for c in sub.columns if c not in ("Index", "BootSE", "BootLLCI", "BootULCI")][0]
                    focal_name = [m for m in indirect_mods if m != other][0]  # PROCESS's W or Z, by variable name
                    rows = ours[(ours["Mediator"] == mediator) & (ours["Focal Mod"] == focal_name)]
                    rows = _by_values(rows, ["Other Mod At"])
                    sub = _by_values(sub, [other])
                    np.testing.assert_allclose(rows["Index"].to_numpy(dtype=float), sub["Index"].to_numpy(dtype=float), rtol=2e-4, atol=1e-6)


@pytest.mark.parametrize("model", [4, 6])
def test_completely_standardized_effects_match(model, cache):
    parsed, p = fitted(model, "OLS", cache)
    theirs = parsed["effsize"]
    ours = p.indirect_model.effect_size_summary()
    ours = ours[ours["Standardization"] == "completely"].set_index("")
    for label, row in theirs.iterrows():
        if "minus" in label:
            continue  # contrasts of standardized effects: not reported by 2.x
        assert ours.loc[label, "Effect"] == pytest.approx(row["Effect"], rel=2e-4, abs=1e-6), label
