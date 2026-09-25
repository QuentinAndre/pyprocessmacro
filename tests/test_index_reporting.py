"""
Which moderated-mediation indices are reported, and their values, checked against the
PROCESS 2.16 output in tests/Results.
"""
import os
import re

import numpy as np
import pandas as pd
import pytest

from pyprocessmacro import Process

TEST_DIR = os.path.dirname(os.path.abspath(__file__))

HEADINGS = {
    "MODERATED MEDIATION": "MM",
    "PARTIAL MODERATED MEDIATION": "PMM",
    "MODERATED MODERATED MEDIATION": "MMM",
    "CONDITIONAL MODERATED MEDIATION": "CMM",
}

# PROCESS 2.16 predates the indices of moderated moderated mediation and conditional moderated
# mediation (Hayes, 2018). These models report them following PROCESS 3; every other model
# matches the 2.16 output exactly.
PROCESS3_EXTENSION = {11, 12, 13, 18, 19, 20, 21, 22, 28, 29}

MEDIATION_MODELS = [n for n in range(4, 77) if n != 6]

# Models for which PROCESS 2.16 prints an index table, by index.
INDEX_MODELS = {"MM": [7, 8, 14, 15, 74], "PMM": [9, 10, 16, 17]}


def load(model, kind="OLS"):
    with open(os.path.join(TEST_DIR, f"Results/Results_{kind}_Model{model}.txt"), encoding="utf-8-sig") as f:
        txt = f.read()
    data = pd.read_csv(os.path.join(TEST_DIR, f"Data/Data_Model{model}.csv"))
    with open(os.path.join(TEST_DIR, f"Data/Varlist_Model{model}.txt")) as f:
        varlist = f.read().split(",")
    kwargs = {v: v for v in varlist if "m" not in v}
    kwargs["m"] = ["m1", "m2"] if ("m1" in varlist and model > 3) else "m"
    if kind == "Logit":
        kwargs["y"] = "y2"
    return txt, data, kwargs


def fixture_headings(txt):
    return {HEADINGS[h.strip()] for h in re.findall(r"\*{3,}\s*INDEX OF ([A-Z ]+?)\s*\*{3,}", txt)}


def parse_index_tables(txt, code):
    """Return {(moderator or None, mediator): [Index, SE(Boot), BootLLCI, BootULCI]} for one section."""
    name = {v: k for k, v in HEADINGS.items()}[code]
    section = txt.split(f"INDEX OF {name}")[1].split("ANALYSIS NOTES")[0]
    rows, moderator, in_table = {}, None, False
    lines = iter(section.splitlines())
    for line in lines:
        s = line.strip()
        if not s or s.startswith("*"):
            in_table = False
            continue
        if s == "Moderator:":
            moderator = next(lines).strip()
            continue
        if s == "Mediator":
            in_table = True
            next(lines)  # column headings
            continue
        if in_table:
            parts = s.split()
            rows[(moderator, parts[0])] = np.array(parts[1:], dtype=float)
    return rows


@pytest.mark.parametrize("model", MEDIATION_MODELS)
def test_index_sections_match_process(model):
    txt, data, kwargs = load(model)
    p = Process(data, model, boot=10, suppr_init=True, **kwargs)
    ours = set(p._gen_analysis_list())
    expected = fixture_headings(txt)
    if model in PROCESS3_EXTENSION:
        assert expected == set()
        assert ours == {"MMM", "CMM"}
    else:
        assert ours == expected


@pytest.mark.parametrize("kind", ["OLS", "Logit"])
@pytest.mark.parametrize("code, model", [(c, m) for c, models in INDEX_MODELS.items() for m in models])
def test_index_values_match_process(code, model, kind):
    txt, data, kwargs = load(model, kind)
    p = Process(
        data, model, precision=4, conf=95, modval={}, quantile=False, logit=(kind == "Logit"),
        seed=123456, suppr_init=True, total=True, hc3=True, **kwargs
    )
    ours = getattr(p.indirect_model, f"{code}_index_summary")()
    expected = parse_index_tables(txt, code)
    assert len(ours) == len(expected)
    boot_tol = 1e-2 if kind == "OLS" else 5e-2
    close = []
    for _, row in ours.iterrows():
        key = (row["Moderator"] if code == "PMM" else None, row["Mediator"])
        ref = expected[key]
        assert row["Index"] == pytest.approx(ref[0], rel=1e-3, abs=1e-3), key
        close.extend(np.isclose([row["Boot SE"], row["LLCI"], row["ULCI"]], ref[1:], rtol=boot_tol, atol=boot_tol))
    # Bootstrap statistics come from different random draws than PROCESS used.
    assert np.mean(close) > 0.8, np.mean(close)
