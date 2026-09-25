# -*- coding: utf-8 -*-
"""
Standardized indirect effects (#70), the `effsize` option.

For an unmoderated indirect path and a continuous outcome, PROCESS reports the partially standardized
indirect effect (the indirect effect divided by the standard deviation of Y) and the completely
standardized indirect effect (further multiplied by the standard deviation of X). Both are bootstrapped
by standardizing within each resample with that resample's standard deviations, as PROCESS does.
"""
from functools import partial

import numpy as np

from .models import _summary_table
from .utils import bias_corrected_ci, percentile_ci

KINDS = {"ps": "Partially standardized", "cs": "Completely standardized"}
TABLE_COLUMNS = ["Effect", "Boot SE", "BootLLCI", "BootULCI"]


def _display(label):
    """Row label as the indirect-effect table prints it."""
    return "TOTAL" if label == "total" else label


def standardized_effects(model):
    """
    :param model: a ParallelMediationModel or SerialMediationModel with `_raw_indirect_draws()` and `_boot_sds`
    :return: {"ps": {...}, "cs": {...}} where each value holds "labels" and the arrays "effect", "se",
        "llci", "ulci", one entry per row (the total if requested, then one per mediator or path)
    """
    labels, effects, draws = model._raw_indirect_draws()
    ind_x, ind_y = model._symb_to_ind["x"], model._ind_y
    sd_x, sd_y = model._data[:, ind_x].std(ddof=1), model._data[:, ind_y].std(ddof=1)
    boot_sd_x, boot_sd_y = model._boot_sds[:, 0], model._boot_sds[:, 1]
    scales = {"ps": (1 / sd_y, 1 / boot_sd_y), "cs": (sd_x / sd_y, boot_sd_x / boot_sd_y)}
    conf = model._options["conf"]
    interval = percentile_ci if model._options["percent"] else None
    out = {}
    for kind, (scale, boot_scale) in scales.items():
        estimate = effects * scale
        boot = draws * boot_scale
        rows = {"labels": list(labels), "effect": estimate, "se": boot.std(axis=1, ddof=1)}
        bounds = np.array([
            interval(boot[i], conf) if interval else bias_corrected_ci(estimate[i], boot[i], conf)
            for i in range(len(labels))
        ])
        rows["llci"], rows["ulci"] = bounds[:, 0], bounds[:, 1]
        out[kind] = rows
    return out


def effect_size_table(model):
    """One table with a Standardization column: partial rows first, then complete rows."""
    results = standardized_effects(model)
    levels, stats = [], []
    for kind in ("ps", "cs"):
        rows = results[kind]
        for i, label in enumerate(rows["labels"]):
            levels.append([_display(label), KINDS[kind].split()[0].lower()])
            stats.append([rows["effect"][i], rows["se"][i], rows["llci"][i], rows["ulci"][i]])
    return _summary_table(levels, ["", "Standardization"], np.array(stats), TABLE_COLUMNS)


def effect_size_text(model, float_format):
    """The two sections PROCESS prints, one per kind of standardization."""
    stv = model._symb_to_var
    results = standardized_effects(model)
    text = ""
    for kind in ("ps", "cs"):
        rows = results[kind]
        stats = np.array([rows["effect"], rows["se"], rows["llci"], rows["ulci"]]).T
        table = _summary_table([[_display(label)] for label in rows["labels"]], [""], stats, TABLE_COLUMNS)
        text += "{kind} indirect effect(s) of {x} on {y}:\n\n{table}\n\n".format(
            kind=KINDS[kind], x=stv["x"], y=stv["y"], table=table.to_string(float_format=float_format)
        )
    return text
