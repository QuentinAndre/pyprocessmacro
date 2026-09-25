import numpy as np
import pandas as pd
import pytest
from pyprocessmacro import Process
from io import StringIO
import os

MODELS_LIST = [4, 5] + [i for i in range(7, 73)] + [75, 76]
MODERATION_MODELS = [1, 2, 3]

# Get the directory where this test file is located
TEST_DIR = os.path.dirname(os.path.abspath(__file__))

def get_direct_effect(txt):
    dir = txt.split("Direct effect of X on Y")[1].split("\n\n")[0]
    dirdata = pd.read_csv(StringIO(dir), sep=r"\s{2,}", engine="python")
    return dirdata

def get_direct_effect_mod(txt):
    dirmod = txt.split("Conditional direct effect(s) of X on Y at values of the moderator(s):")[1].split("\n\n")[0]
    dirmoddata = pd.read_csv(StringIO(dirmod), sep=r"\s{2,}", engine="python")
    return dirmoddata

def get_indirect_effect_mod(txt):
    indirs = txt.split("Conditional indirect effect(s) of X on Y at values of the moderator(s):")[1].split("Mediator")
    indir1 = indirs[1]
    indir2 = indirs[2].split("\n\n")[0]
    indir1data = pd.read_csv(StringIO(indir1), sep=r"\s{2,}", engine="python").reset_index()
    indir2data = pd.read_csv(StringIO(indir2), sep=r"\s{2,}", engine="python").reset_index()
    indirdata = pd.concat([indir1data, indir2data])
    cols = indirdata.columns.values
    cols[0] = "Mediator"
    indirdata.columns = cols
    return indirdata

def get_indirect_effect(txt):
    indir = txt.split("Indirect effect of X on Y")[1].split("\n\n")[0]
    indirdata = pd.read_csv(StringIO(indir), sep=r"\s{2,}", engine="python").reset_index()
    cols = indirdata.columns.values
    cols[0] = ""
    indirdata.columns = cols
    return indirdata

def get_ols_accuracy(model_num):
    with open(os.path.join(TEST_DIR, "Results/Results_OLS_Model{}.txt".format(model_num))) as f:
        spssoutput = f.read().split("******************** DIRECT AND INDIRECT EFFECTS *************************\n")[1]

    data = pd.read_csv(os.path.join(TEST_DIR, "Data/Data_Model{}.csv".format(model_num)))
    with open(os.path.join(TEST_DIR, "Data/Varlist_Model{}.txt".format(model_num))) as f:
        varlist = f.read().split(",")

    kwargs = {i: i for i in varlist if "m" not in i}
    if ("m1" in varlist) & (model_num > 3):
        kwargs["m"] = ["m1", "m2"]
    else:
        kwargs["m"] = "m"

    process = Process(data, model_num,precision=4, conf=95, modval={},
                      quantile=False, logit=False, seed=123456, suppr_init=True,
                      total=True, hc3=True, **kwargs)

    dir = process.direct_model
    indir = process.indirect_model

    if dir._has_moderation:
        truedir = get_direct_effect_mod(spssoutput)
    else:
        truedir = get_direct_effect(spssoutput)
    estdir = dir.coeff_summary()

    if indir._has_moderation:
        trueindir = get_indirect_effect_mod(spssoutput)
    else:
        trueindir = get_indirect_effect(spssoutput)
    estindir = indir.coeff_summary()

    # Sort datasets
    coldir = ["Effect", "SE", "t", "p", "LLCI", "ULCI"]
    colindir = ["Effect", "Boot SE", "BootLLCI", "BootULCI"]
    sortcols_dir = [i for i in truedir.columns if i not in coldir + [""]]
    sortcols_indir = [i for i in trueindir.columns if i not in colindir]
    if sortcols_dir:
        estdir = estdir.sort_values(by=sortcols_dir)
    if sortcols_indir:
        estindir = estindir.sort_values(by=sortcols_indir)

    direffects = np.isclose(truedir[coldir].values, estdir[coldir].values, rtol=1e-3, atol=1e-3).mean()
    indireffects = np.isclose(trueindir["Effect"].values, estindir["Effect"].values, rtol=1e-3, atol=1e-3).mean()
    indirboots = np.isclose(trueindir[colindir].values, estindir[colindir].values, rtol=1e-2, atol=1e-2).mean()
    return direffects, indireffects, indirboots

def get_logit_accuracy(model_num):
    with open(os.path.join(TEST_DIR, "Results/Results_Logit_Model{}.txt".format(model_num))) as f:
        spssoutput = f.read().split("******************** DIRECT AND INDIRECT EFFECTS *************************\n")[1]

    data = pd.read_csv(os.path.join(TEST_DIR, "Data/Data_Model{}.csv".format(model_num)))
    with open(os.path.join(TEST_DIR, "Data/Varlist_Model{}.txt".format(model_num))) as f:
        varlist = f.read().split(",")
    varlist = [i if i != "y" else "y2" for i in varlist]
    kwargs = {i: i for i in varlist if "m" not in i}
    if ("m1" in varlist) & (model_num > 3):
        kwargs["m"] = ["m1", "m2"]
    else:
        kwargs["m"] = "m"
    kwargs["y"] = "y2"


    process = Process(data, model_num, precision=4, conf=95, modval={},
                      quantile=False, logit=True, seed=123456,
                      total=True, suppr_init=True, hc3=True, **kwargs)

    dir = process.direct_model
    indir = process.indirect_model

    if dir._has_moderation:
        truedir = get_direct_effect_mod(spssoutput)
    else:
        truedir = get_direct_effect(spssoutput)
    estdir = dir.coeff_summary()

    if indir._has_moderation:
        trueindir = get_indirect_effect_mod(spssoutput)
    else:
        trueindir = get_indirect_effect(spssoutput)
    estindir = indir.coeff_summary()

    # Sort datasets
    coldir = ["Effect", "SE", "Z", "p", "LLCI", "ULCI"]
    colindir = ["Effect", "Boot SE", "BootLLCI", "BootULCI"]
    sortcols_dir = [i for i in truedir.columns if i not in coldir + [""]]
    sortcols_indir = [i for i in trueindir.columns if i not in colindir]
    if sortcols_dir:
        estdir = estdir.sort_values(by=sortcols_dir)
    if sortcols_indir:
        estindir = estindir.sort_values(by=sortcols_indir)
    direffects = np.isclose(truedir[coldir].values, estdir[coldir].values, rtol=1e-3, atol=1e-3).mean()
    indireffects = np.isclose(trueindir["Effect"].values, estindir["Effect"].values, rtol=1e-3, atol=1e-3).mean()
    indirboots = np.isclose(trueindir[colindir].values, estindir[colindir].values, rtol=5e-2, atol=5e-2).mean()
    return direffects, indireffects, indirboots

@pytest.fixture(params=MODELS_LIST)
def get_model_number(request):
    model = request.param
    return model

def test_logit_accuracy(get_model_number):
    d, i, b = get_logit_accuracy(get_model_number)
    assert (d == 1) & (i == 1) & (b > 0.9)

def test_ols_accuracy(get_model_number):
    d, i, b = get_ols_accuracy(get_model_number)
    assert (d == 1) & (i == 1) & (b > 0.9)



def get_conditional_effects(txt):
    block = txt.split("Conditional effect of X on Y at values of the moderator(s):")[1].split("\n\n")[0]
    return pd.read_csv(StringIO(block), sep=r"\s{2,}", engine="python").rename(columns={"se": "SE"})


def get_moderation_accuracy(model_num, logit):
    kind = "Logit" if logit else "OLS"
    with open(os.path.join(TEST_DIR, "Results/Results_{}_Model{}.txt".format(kind, model_num))) as f:
        txt = f.read()
    data = pd.read_csv(os.path.join(TEST_DIR, "Data/Data_Model{}.csv".format(model_num)))
    with open(os.path.join(TEST_DIR, "Data/Varlist_Model{}.txt".format(model_num))) as f:
        varlist = f.read().split(",")
    kwargs = {i: i for i in varlist}
    if logit:
        kwargs["y"] = "y2"
    process = Process(data, model_num, precision=4, conf=95, modval={}, quantile=False, logit=logit,
                      seed=123456, suppr_init=True, hc3=True, **kwargs)
    truth = get_conditional_effects(txt)
    est = process.direct_model.coeff_summary()
    stat_cols = ["Effect", "SE", "Z" if logit else "t", "p", "LLCI", "ULCI"]
    mod_cols = [c for c in truth.columns if c not in stat_cols]
    truth = truth.sort_values(by=mod_cols)
    est = est.sort_values(by=mod_cols)
    return np.isclose(truth[stat_cols].values, est[stat_cols].values, rtol=1e-3, atol=1e-3).mean()


@pytest.mark.parametrize("model", MODERATION_MODELS)
def test_ols_moderation_accuracy(model):
    assert get_moderation_accuracy(model, logit=False) == 1


@pytest.mark.parametrize("model", MODERATION_MODELS)
def test_logit_moderation_accuracy(model):
    assert get_moderation_accuracy(model, logit=True) == 1


def get_model74_accuracy(logit):
    """Model 74 (X moderates its own indirect effect): PROCESS prints conditional indirect effects only."""
    kind = "Logit" if logit else "OLS"
    with open(os.path.join(TEST_DIR, "Results/Results_{}_Model74.txt".format(kind))) as f:
        txt = f.read().split("******************** DIRECT AND INDIRECT EFFECTS *************************\n")[1]
    data = pd.read_csv(os.path.join(TEST_DIR, "Data/Data_Model74.csv"))
    kwargs = dict(x="x", m=["m1", "m2"], y="y2" if logit else "y")
    process = Process(data, 74, precision=4, conf=95, modval={}, quantile=False, logit=logit,
                      seed=123456, suppr_init=True, total=True, hc3=True, **kwargs)
    truth = get_indirect_effect_mod(txt)
    est = process.indirect_model.coeff_summary()
    colindir = ["Effect", "Boot SE", "BootLLCI", "BootULCI"]
    sortcols = [c for c in truth.columns if c not in colindir]
    truth = truth.sort_values(by=sortcols)
    est = est.sort_values(by=sortcols)
    tol = 5e-2 if logit else 1e-2
    effects = np.isclose(truth["Effect"].values, est["Effect"].values, rtol=1e-3, atol=1e-3).mean()
    boots = np.isclose(truth[colindir].values, est[colindir].values, rtol=tol, atol=tol).mean()
    return effects, boots


@pytest.mark.parametrize("logit", [False, True], ids=["ols", "logit"])
def test_model74_indirect_accuracy(logit):
    effects, boots = get_model74_accuracy(logit)
    assert (effects == 1) & (boots > 0.9)
