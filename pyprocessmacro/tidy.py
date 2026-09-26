# -*- coding: utf-8 -*-
"""
Standardized result tables, modelled on R's broom package.

tidy() returns one long DataFrame with one row per estimate and fixed column names, glance() one row of
fit statistics per outcome model, and augment() the analysis data with fitted values and residuals.
All three read the numeric estimation results directly, so they carry exactly the numbers of the
printed tables under consistent names.
"""
from itertools import product

import numpy as np
import pandas as pd

STAT_COLUMNS = ["estimate", "std_error", "statistic", "p_value", "conf_low", "conf_high"]
META_COLUMNS = ["method", "conf_level", "n_boot"]

GLANCE_COLUMNS = [
    "outcome",
    "estimator",
    "cov_type",
    "n",
    "r_squared",
    "adj_r_squared",
    "mse",
    "f_statistic",
    "df_model",
    "df_resid",
    "p_value",
    "log_likelihood",
    "ll_null",
    "lr_statistic",
    "mcfadden",
    "cox_snell",
    "nagelkerke",
    "alpha",
    "aic",
    "bic",
]


def tidy(process, component=None):
    """
    One row per estimate of a fitted Process model.

    :param process: a fitted Process instance
    :param component: None for every row, or one component name or a list of names to keep, among
        "outcome", "direct", "indirect", "total", "contrast", "index_mm", "index_pmm", "index_mmm",
        "index_cmm", "indirect_ps", "indirect_cs".
    :return: DataFrame with the columns component, outcome, term, moderator, one column per moderator
        of the model holding the spotlight value the row is evaluated at, estimate, std_error,
        statistic, p_value, conf_low, conf_high, method, conf_level, n_boot.
    """
    stv = process._symb_to_var
    moderators = sorted(stv[s] for s in process._moderators["all"])  # by variable name
    x_code = ["x_code"] if getattr(process.direct_model, "_categorical_x", False) else []  # #17
    columns = ["component", "outcome", "term", "moderator"] + x_code + moderators + STAT_COLUMNS + META_COLUMNS
    conf = process.options["conf"]
    n_boot = process.options["boot"]
    boot_method = "bootstrap_percentile" if process.options["percent"] else "bootstrap_bc"
    dv = process.dv
    rows = []

    def add(comp, outcome, term, estimate, std_error, conf_low, conf_high, method,
            statistic=np.nan, p_value=np.nan, moderator=None, at=None, boot=False):
        row = {"component": comp, "outcome": outcome, "term": term, "moderator": moderator}
        row.update({m: np.nan for m in moderators + x_code})
        if at:
            row.update(at)
        row.update(
            estimate=float(estimate), std_error=float(std_error), statistic=float(statistic),
            p_value=float(p_value), conf_low=float(conf_low), conf_high=float(conf_high),
            method=method, conf_level=conf, n_boot=(n_boot if boot else np.nan),
        )
        rows.append(row)

    # Outcome models: one row per coefficient.
    for outcome, model in process.outcome_models.items():
        res = model.estimation_results
        statistic = res["z" if "z" in res else "t"]
        for i, term in enumerate(res["names"]):
            add("outcome", outcome, term, res["betas"][i], res["se"][i], res["llci"][i], res["ulci"][i],
                _estimator(res), statistic[i], res["p"][i])

    # Direct (or, for models 1 to 3, conditional) effects of X on Y; one row per code of a multicategorical X,
    # named by the code (#17).
    direct = process.direct_model
    res = direct._estimation_results
    direct_mods = [stv[s] for s in direct._moderators_symb]
    combos = list(product(*direct._moderators_values))
    i = 0
    for label in direct._x_labels:
        for combo in combos:
            add("direct", dv, label, res["betas"][i], res["se"][i], res["llci"][i], res["ulci"][i],
                _estimator(direct._model.estimation_results), res["t"][i], res["p"][i],
                at=dict(zip(direct_mods, combo)))
            i += 1

    if not process.has_mediation:
        return _finish(rows, columns, component)

    # With a multicategorical X, every indirect row is repeated per code, labelled by the code in `x_code` (#17).
    parent = process.indirect_model
    mediators = [stv[f"m{i + 1}"] for i in range(parent._n_meds)]
    for indirect, x_label in zip(parent._views(), parent._x_labels):
        _indirect_rows(indirect, add, dv, mediators, stv, boot_method, process.options,
                       x_label if parent._categorical_x else None)

    return _finish(rows, columns, component)


def _indirect_rows(indirect, add, dv, mediators, stv, boot_method, options, x_code):
    """Rows of the indirect effects, the standardized effects and the indices of one (code-)view."""
    extra = {"x_code": x_code} if x_code is not None else {}
    res = indirect.estimation_results
    if indirect._has_moderation:
        ind_mods = [stv[s] for s in indirect._moderators_symb]
        combos = list(product(*indirect._moderators_values))
        k = 0
        for med in mediators:
            for combo in combos:
                add("indirect", dv, med, res["effect"][k], res["se"][k], res["llci"][k], res["ulci"][k],
                    boot_method, at={**dict(zip(ind_mods, combo)), **extra}, boot=True)
                k += 1
    else:  # parallel mediators or the serial paths of model 6, labelled by the model
        for k, (comp, term) in enumerate(indirect.effect_labels):
            add(comp, dv, term, res["effect"][k], res["se"][k], res["llci"][k], res["ulci"][k],
                boot_method, at=extra or None, boot=True)

    if options.get("effsize"):
        sizes = indirect.effect_sizes()
        for kind, comp in (("ps", "indirect_ps"), ("cs", "indirect_cs")):
            if kind not in sizes:
                continue
            rows_k = sizes[kind]
            for k, term in enumerate(rows_k["labels"]):
                add(comp, dv, term, rows_k["effect"][k], rows_k["se"][k], rows_k["llci"][k], rows_k["ulci"][k],
                    boot_method, at=extra or None, boot=True)

    # Indices of moderated mediation, in the layouts of the corresponding index methods.
    mod_symbols = list(indirect._moderators_symb)
    mod_names = [stv[s] for s in mod_symbols]
    for code in indirect._analysis_list:
        res = getattr(indirect, f"_{code}_index")()
        comp = f"index_{code.lower()}"
        if code == "MM":
            k = 0
            for mod_label in indirect.mm_moderator_labels():
                for med in mediators:
                    add(comp, dv, med, res["effect"][k], res["se"][k], res["llci"][k], res["ulci"][k],
                        boot_method, moderator=mod_label, at=extra or None, boot=True)
                    k += 1
        elif code == "PMM":
            k = 0
            for name in mod_names:
                for med in mediators:
                    add(comp, dv, med, res["effect"][k], res["se"][k], res["llci"][k], res["ulci"][k],
                        boot_method, moderator=name, at=extra or None, boot=True)
                    k += 1
        elif code == "MMM":
            for i, med in enumerate(mediators):
                add(comp, dv, med, res["effect"][i], res["se"][i], res["llci"][i], res["ulci"][i],
                    boot_method, at=extra or None, boot=True)
        elif code == "CMM":
            mod1, mod2 = mod_names
            values1, values2 = indirect._moderators_values
            k = 0
            for med in mediators:
                for value in values1:  # index of moderated mediation by mod2, conditional on mod1
                    add(comp, dv, med, res["effect"][k], res["se"][k], res["llci"][k], res["ulci"][k],
                        boot_method, moderator=mod2, at={mod1: value, **extra}, boot=True)
                    k += 1
            for med in mediators:
                for value in values2:  # index of moderated mediation by mod1, conditional on mod2
                    add(comp, dv, med, res["effect"][k], res["se"][k], res["llci"][k], res["ulci"][k],
                        boot_method, moderator=mod1, at={mod2: value, **extra}, boot=True)
                    k += 1


def _estimator(res):
    """The name of the estimator behind an outcome model's results: ols, logit or negbin (#25)."""
    return "negbin" if "alpha" in res else ("logit" if "z" in res else "ols")


def _finish(rows, columns, component):
    frame = pd.DataFrame(rows, columns=columns)
    if component is not None:
        wanted = [component] if isinstance(component, str) else list(component)
        unknown = sorted(set(wanted) - set(frame["component"].unique()) - {
            "outcome", "direct", "indirect", "total", "contrast", "index_mm", "index_pmm", "index_mmm", "index_cmm",
            "indirect_ps", "indirect_cs",
        })
        if unknown:
            raise ValueError(f"Unknown component(s): {', '.join(unknown)}.")
        frame = frame[frame["component"].isin(wanted)]
    return frame.reset_index(drop=True)


def glance(process):
    """
    One row of fit statistics per outcome model.

    OLS rows fill r_squared, adj_r_squared, mse, f_statistic, df_model, df_resid and p_value; logistic
    rows fill ll_null, lr_statistic, df_model, p_value, mcfadden, cox_snell and nagelkerke; negative
    binomial rows fill ll_null, lr_statistic, df_model, p_value, mcfadden and alpha (#25). All fill n,
    log_likelihood, aic and bic.
    """
    rows = []
    for outcome, model in process.outcome_models.items():
        res = model.estimation_results
        row = {c: np.nan for c in GLANCE_COLUMNS}
        row.update(outcome=outcome, cov_type=res["cov_type"], n=res["n"], log_likelihood=res["llf"],
                   aic=res["aic"], bic=res["bic"])
        if "alpha" in res:
            row.update(estimator="negbin", ll_null=res["llnull"], lr_statistic=res["d"], df_model=res["df_model"],
                       p_value=res["pvalue"], mcfadden=res["mcfadden"], alpha=res["alpha"])
        elif "z" in res:
            row.update(estimator="logit", ll_null=res["llnull"], lr_statistic=res["d"], df_model=res["df_model"],
                       p_value=res["pvalue"], mcfadden=res["mcfadden"], cox_snell=res["coxsnell"],
                       nagelkerke=res["nagelkerke"])
        else:
            row.update(estimator="ols", r_squared=res["R2"], adj_r_squared=res["adjR2"], mse=res["mse"],
                       f_statistic=res["F"], df_model=res["df_r"], df_resid=res["df_e"], p_value=res["F_pval"])
        rows.append(row)
    return pd.DataFrame(rows, columns=GLANCE_COLUMNS)


def augment(process, outcome=None):
    """
    The analysis data (rows kept after listwise deletion, user variable names) with, for each outcome
    model, a `.fitted_<outcome>` and a `.resid_<outcome>` column. For a logistic outcome the fitted
    value is the predicted probability, for a negative binomial outcome the predicted count, and the
    residual is the response residual. A logistic outcome appears as the 0/1 recoding the model was
    fitted on.

    :param outcome: None for every outcome model, or the name of one outcome.
    """
    stv = process._symb_to_var
    data = process._data
    keep = [c for c in data.columns if c != "Cons" and "*" not in c]
    frame = data[keep].rename(columns=lambda c: stv.get(c, c)).copy()
    models = process.outcome_models if outcome is None else {outcome: process.outcome_models[outcome]}
    for name, model in models.items():
        res = model.estimation_results
        linear = model._exog @ res["betas"]
        fitted = model._cdf(linear) if "z" in res else linear
        frame[f".fitted_{name}"] = fitted
        frame[f".resid_{name}"] = model._endog - fitted
    return frame
