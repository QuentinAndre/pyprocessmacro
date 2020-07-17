import numpy as np
import pandas as pd
import pytest
from ..pyprocessmacro import Process
from io import StringIO

MODELS_LIST = [5] + [i for i in range(7, 73)] + [75, 76]


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
    with open("Results/Results_OLS_Model{}.txt".format(model_num)) as f:
        spssoutput = f.read().split("******************** DIRECT AND INDIRECT EFFECTS *************************\n")[1]

    data = pd.read_csv("Data/Data_Model{}.csv".format(model_num))
    varlist = open("Data/Varlist_Model{}.txt".format(model_num)).read().split(",")

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
    with open("Results/Results_Logit_Model{}.txt".format(model_num)) as f:
        spssoutput = f.read().split("******************** DIRECT AND INDIRECT EFFECTS *************************\n")[1]

    data = pd.read_csv("Data/Data_Model{}.csv".format(model_num))
    varlist = open("Data/Varlist_Model{}.txt".format(model_num)).read().split(",")
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

