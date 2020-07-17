# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from numpy import dot
from numpy.linalg import inv, LinAlgError
from scipy.stats import norm
from seaborn import FacetGrid


def z_score(conf):
    """
    :param conf: Desired level of confidence
    :return: The Z-score corresponding to the level of confidence desired.
    """
    return norm.ppf((100 - (100 - conf) / 2) / 100)


def bias_corrected_ci(estimate, samples, conf=95):
    """
    Return the bias-corrected bootstrap confidence interval for an estimate
    :param estimate: Numerical estimate in the original sample
    :param samples: Nx1 array of bootstrapped estimates
    :param conf: Level of the desired confidence interval
    :return: Bias-corrected bootstrapped LLCI and ULCI for the estimate.
    """
    # noinspection PyUnresolvedReferences
    ptilde = ((samples < estimate) * 1).mean()
    Z = norm.ppf(ptilde)
    Zci = z_score(conf)
    Zlow, Zhigh = -Zci + 2 * Z, Zci + 2 * Z
    plow, phigh = norm._cdf(Zlow), norm._cdf(Zhigh)
    llci = np.percentile(samples, plow * 100, interpolation="lower")
    ulci = np.percentile(samples, phigh * 100, interpolation="higher")
    return llci, ulci


def percentile_ci(samples, conf):
    """
    Based on an array of values, returns the lower and upper percentile bound for a desired level of confidence
    :param samples: NxK array of samples
    :param conf: Desired level of confidence
    :return: 2xK array corresponding to the lower and upper percentile bounds for K estimates.
    """
    lower = (100 - conf) / 2
    upper = 100 - lower
    return np.percentile(samples, [lower, upper])


def fast_OLS(endog, exog):
    """
    A simple function for (X'X)^(-1)X'Y
    :return: The Kx1 array of estimated coefficients.
    """
    try:
        return dot(dot(inv(dot(exog.T, exog)), exog.T), endog).squeeze()
    except LinAlgError:
        raise LinAlgError


def logit_cdf(X):
    """
    The CDF of the logistic function.
    :param X: Value at which to estimate the CDF
    :return: The logistic function CDF, evaluated at X
    """
    idx = X > 0
    out = np.empty(X.size, dtype=float)
    out[idx] = 1 / (1 + np.exp(-X[idx]))
    exp_X = np.exp(X[~idx])
    out[~idx] = exp_X / (1 + exp_X)
    return out


def logit_score(endog, exog, params, n_obs):
    """
    The score of the logistic function.
    :param endog: Nx1 vector of endogenous predictions
    :param exog: NxK vector of exogenous predictors
    :param params: Kx1 vector of parameters for the predictors
    :param n_obs: Number of observations
    :return: The score, a Kx1 vector, evaluated at `params'
    """
    return dot(endog - logit_cdf(dot(exog, params)), exog) / n_obs


def logit_hessian(exog, params, n_obs):
    """
    The hessian of the logistic function.
    :param exog: NxK vector of exogenous predictors
    :param params: Kx1 vector of parameters for the predictors
    :param n_obs: Number of observations
    :return: The Hessian, a KxK matrix, evaluated at `params'
    """
    L = logit_cdf(np.dot(exog, params))
    return -dot(L * (1 - L) * exog.T, exog) / n_obs


def fast_optimize(endog, exog, n_obs=0, n_vars=0, max_iter=10000, tolerance=1e-10):
    """
    A convenience function for the Newton-Raphson method to evaluate a logistic model.
    :param endog: Nx1 vector of endogenous predictions
    :param exog: NxK vector of exogenous predictors
    :param n_obs: Number of observations N
    :param n_vars: Number of exogenous predictors K
    :param max_iter: Maximum number of iterations
    :param tolerance: Margin of error for convergence
    :return: The error-minimizing parameters for the model.
    """
    iterations = 0
    oldparams = np.inf
    newparams = np.repeat(0, n_vars)
    while iterations < max_iter and np.any(np.abs(newparams - oldparams) > tolerance):
        oldparams = newparams
        try:
            H = logit_hessian(exog, oldparams, n_obs)
            newparams = oldparams - dot(
                inv(H), logit_score(endog, exog, oldparams, n_obs)
            )
        except LinAlgError:
            raise LinAlgError
        iterations += 1
    return newparams


def bootstrap_sampler(n_obs, seed=None):
    """
    A generator of bootstrapped indices. Samples with repetition a list of indices.
    :param n_obs: Number of observations
    :param seed: The seed to use for the random number generator
    :return: Bootstrapped indices of size n_obs
    """
    seeder = np.random.RandomState(seed)
    seeder.seed(seed)
    while True:
        yield seeder.randint(n_obs, size=n_obs)


def eigvals(exog):
    """
    Return the eigenvalues of a matrix of endogenous predictors.
    :param exog: NxK matrix of exogenous predictors.
    :return: Kx1 vector of eigenvalues, sorted in decreasing order of magnitude.
    """
    return np.sort(np.linalg.eigvalsh(dot(exog.T, exog)))[::-1]


def eval_expression(expr, values=None):
    """
    Evaluate a symbolic expression and returns a numerical array.
    :param expr: A symbolic expression to evaluate, in the form of a N_terms * N_Vars matrix
    :param values: None, or a dictionary of variable:value pairs, to substitute in the symbolic expression.
    :return: An evaled expression, in the form of an N_terms array.
    """
    n_coeffs = expr.shape[0]
    evaled_expr = np.zeros(n_coeffs)
    for (i, term) in enumerate(expr):
        if values:
            evaled_term = np.array(
                [
                    values.get(elem, 0) if isinstance(elem, str) else elem
                    for elem in term
                ]
            )
        else:
            evaled_term = np.array(
                [0 if isinstance(elem, str) else elem for elem in term]
            )  # All variables at 0
        evaled_expr[i] = np.product(
            evaled_term.astype(float)
        )  # Gradient is the product of values
    return evaled_expr


def gen_moderators(raw_equations, raw_varlist):
    terms_y = raw_equations["all_to_y"]
    terms_m = raw_equations["x_to_m"]
    # Moderators of X in the path to Y
    mod_x_direct = set(filter(lambda v: f"x*{v}" in terms_y, raw_varlist))

    # Moderators of X in the path to M
    mod_x_indirect = set(filter(lambda v: f"x*{v}" in terms_m, raw_varlist))

    # Moderators of M in the path to Y
    mod_m = set(filter(lambda v: f"m*{v}" in terms_y, raw_varlist))

    moderators = {
        "x_direct": mod_x_direct,
        "x_indirect": mod_x_indirect,
        "m": mod_m,
        "all": mod_x_indirect | mod_x_direct | mod_m,
        "indirect": mod_x_indirect | mod_m,
    }
    return moderators


def plot_errorbars(
    x, y, yerrlow, yerrhigh, plot_kws=None, err_kws=None, *args, **kwargs
):
    yerr = [yerrlow, yerrhigh]
    err_kws_final = kwargs.copy()
    err_kws_final.update(err_kws)
    err_kws_final.update({"marker": "", "fmt": "none", "label": "", "zorder": 3})
    plot_kws_final = kwargs.copy()
    plot_kws_final.update(plot_kws)
    plt.plot(x, y, *args, **plot_kws_final)
    plt.errorbar(x, y, yerr, *args, **err_kws_final)
    return None


def plot_errorbands(x, y, llci, ulci, plot_kws=None, err_kws=None, *args, **kwargs):
    err_kws_final = kwargs.copy()
    err_kws_final.update(err_kws)
    err_kws_final.update({"label": ""})
    plot_kws_final = kwargs.copy()
    plot_kws_final.update(plot_kws)
    plt.plot(x, y, *args, **plot_kws_final)
    plt.fill_between(x, llci, ulci, *args, **err_kws_final)
    return None


def plot_conditional_effects(
    df_effects, x, hue, row, col, errstyle, hue_format, facet_kws, plot_kws, err_kws
):
    if isinstance(hue, list):
        huename = "Hue"
        if len(hue) == 2:
            if hue_format is None:
                hue_format = "{var1} at {hue1:.2f},  {var2} at {hue2:.2f}"
            df_effects["Hue"] = df_effects[hue].apply(
                lambda d: hue_format.format(
                    var1=hue[0], var2=hue[1], hue1=d[hue[0]], hue2=d[hue[1]]
                ),
                axis=1,
            )
        else:
            if hue_format is None:
                hue_format = "{var1} at {hue1:.2f}"
            df_effects["Hue"] = df_effects[hue[0]].apply(
                lambda d: hue_format.format(var1=hue[0], hue1=d)
            )
    elif isinstance(hue, str):
        huename = "Hue"
        if hue_format is None:
            hue_format = "{var1} at {hue1:.2f}"
        df_effects["Hue"] = df_effects[hue].apply(
            lambda d: hue_format.format(var1=hue, hue1=d)
        )
    else:
        huename = None

    if not facet_kws:
        facet_kws = {}
    if not plot_kws:
        plot_kws = {}

    g = FacetGrid(hue=huename, data=df_effects, col=col, row=row, **facet_kws)

    if errstyle == "band":
        if not err_kws:
            err_kws = {"alpha": 0.2}
        g.map(
            plot_errorbands,
            x,
            "Effect",
            "LLCI",
            "ULCI",
            plot_kws=plot_kws,
            err_kws=err_kws,
        )
    elif errstyle == "ci":
        if not err_kws:
            err_kws = {"alpha": 1, "capthick": 1, "capsize": 3}
        df_effects["yerr_low"] = df_effects["Effect"] - df_effects["LLCI"]
        df_effects["yerr_high"] = df_effects["ULCI"] - df_effects["Effect"]
        g.map(
            plot_errorbars,
            x,
            "Effect",
            "yerr_low",
            "yerr_high",
            plot_kws=plot_kws,
            err_kws=err_kws,
        )
    elif errstyle == "none":
        g.map(plt.plot, x, "Effect", **plot_kws)

    if facet_kws.get("margin_titles"):
        for ax in g.axes.flat:
            plt.setp(ax.texts, text="")

    if row and col:
        g.set_titles(
            row_template="{row_var} at {row_name:.2f}",
            col_template="{col_var} at {col_name:.2f}",
        )

    return g


# noinspection PyTypeChecker
def find_significance_region(
    spotlight_func, mod_symb, modval_min, modval_max, modval_other_symb, atol, rtol
):
    mos = modval_other_symb.copy()
    dict_modval_min = {**dict([[mod_symb, modval_min]]), **mos}
    dict_modval_max = {**dict([[mod_symb, modval_max]]), **mos}
    b_min, _, llci_min, ulci_min = spotlight_func(dict_modval_min)
    b_max, _, llci_max, ulci_max = spotlight_func(dict_modval_max)

    slope = "positive" if (b_min < b_max) else "negative"

    if slope == "negative":
        # Flip the values to facilitate the computations.
        b_min, llci_min, ulci_min, b_max, llci_max, ulci_max = (
            b_max,
            llci_max,
            ulci_max,
            b_min,
            llci_min,
            ulci_min,
        )

    # Cases 1 and 2: The effect is always significantly negative/positive:
    if ulci_max < 0:
        return [[modval_min, modval_max], []]
    if llci_min > 0:
        return [[], [modval_min, modval_max]]

    # Case 3: The effect is negative and sig. in one region, and becomes non-significant at some critical value:
    if (ulci_min < 0) and (llci_max < 0 < ulci_max):
        critical_value_neg = search_critical_values(
            spotlight_func,
            modval_min,
            modval_max,
            mod_symb,
            mos,
            slope,
            region="negative",
            atol=atol,
            rtol=rtol,
        )
        return (
            [[modval_min, critical_value_neg], []]
            if slope == "positive"
            else [[critical_value_neg, modval_max], []]
        )

    # Case 4: The is positive and significant in one region, and becomes non-significant at some critical value:
    if (llci_min < 0 < ulci_min) and (llci_max > 0):
        critical_value_pos = search_critical_values(
            spotlight_func,
            modval_min,
            modval_max,
            mod_symb,
            mos,
            slope,
            region="positive",
            atol=atol,
            rtol=rtol,
        )
        return (
            [[], [critical_value_pos, modval_max]]
            if slope == "positive"
            else [[], [modval_min, critical_value_pos]]
        )

    # Case 5: The effect is negative and significant in one region, and crossover to positive and sig. in another:
    if (ulci_min < 0) and (llci_max > 0):
        modval_diff = modval_max - modval_min
        dist_to_zero = 1 - (b_max / (b_max - b_min))
        if slope == "positive":
            modval_zero = modval_min + modval_diff * dist_to_zero
            critical_value_neg = search_critical_values(
                spotlight_func,
                modval_min,
                modval_zero,
                mod_symb,
                mos,
                slope,
                region="negative",
                atol=atol,
                rtol=rtol,
            )
            critical_value_pos = search_critical_values(
                spotlight_func,
                modval_zero,
                modval_max,
                mod_symb,
                mos,
                slope,
                region="positive",
                atol=atol,
                rtol=rtol,
            )
            return [[modval_min, critical_value_neg], [critical_value_pos, modval_max]]
        else:
            modval_zero = modval_max - modval_diff * dist_to_zero
            critical_value_neg = search_critical_values(
                spotlight_func,
                modval_min,
                modval_zero,
                mod_symb,
                mos,
                slope,
                region="positive",
                atol=atol,
                rtol=rtol,
            )
            critical_value_pos = search_critical_values(
                spotlight_func,
                modval_zero,
                modval_max,
                mod_symb,
                mos,
                slope,
                region="negative",
                atol=atol,
                rtol=rtol,
            )
            return [[critical_value_pos, modval_max], [modval_min, critical_value_neg]]

    # Case 6: The effect is not significant on the bounds of the region, but can still be significant in some middle
    # range:
    if (llci_min < 0 < ulci_min) and (llci_max < 0 < ulci_max):
        return search_mid_range(
            spotlight_func, modval_min, modval_max, mod_symb, mos, atol=atol, rtol=rtol
        )


def search_mid_range(
    spotlight_func, min_val, max_val, mod_symb, mod_dict, atol=1e-8, rtol=1e-5
):
    cvals = np.linspace(min_val, max_val, 1000)  # Construct a grid of 1000 points.
    arr_ci = np.empty((1000, 2))
    # noinspection PyRedundantParentheses
    arr_b = np.empty(shape=(1000))
    for i, cval in enumerate(cvals):
        mod_dict[mod_symb] = cval
        arr_b[i], _, arr_ci[i][0], arr_ci[i][1] = spotlight_func(mod_dict)

    non_sig = list(
        map(lambda x: x[0] < 0 < x[1], arr_ci)
    )  # Check if there is at least one point where the CI does
    # not include 0
    if all(non_sig):  # If not, no significant region.
        return [[], []]

    # Otherwise, we identify the effect at the point at which the CI is the most narrow.
    effect_at_tightest_ci = arr_b[np.argmin(arr_ci[:, 1] - arr_ci[:, 0])]

    if effect_at_tightest_ci > 0:  # Significance region will be positive
        # Slope here is the slope of the CI: the slope of the effect itself is not significant.
        mid_val = cvals[np.argmax(arr_ci[:, 0])]
        lval = search_critical_values(
            spotlight_func,
            min_val,
            mid_val,
            mod_symb,
            mod_dict,
            slope="positive",
            region="positive",
            atol=atol,
            rtol=rtol,
        )
        uval = search_critical_values(
            spotlight_func,
            mid_val,
            max_val,
            mod_symb,
            mod_dict,
            slope="negative",
            region="positive",
            atol=atol,
            rtol=rtol,
        )
        return [[lval, uval], []]

    else:
        # Slope here is the slope of the CI: the slope of the effect itself is not significant.
        mid_val = cvals[np.argmin(arr_ci[:, 1])]
        lval = search_critical_values(
            spotlight_func,
            min_val,
            mid_val,
            mod_symb,
            mod_dict,
            slope="negative",
            region="negative",
            atol=atol,
            rtol=rtol,
        )
        uval = search_critical_values(
            spotlight_func,
            mid_val,
            max_val,
            mod_symb,
            mod_dict,
            slope="positive",
            region="negative",
            atol=atol,
            rtol=rtol,
        )
        return [[], [lval, uval]]


def search_critical_values(
    spotlight_func,
    min_val,
    max_val,
    mod_symb,
    mod_dict,
    slope="positive",
    region="positive",
    atol=1e-8,
    rtol=1e-5,
):
    cval = (max_val + min_val) / 2
    mod_dict[mod_symb] = cval
    b, se, llci, ulci = spotlight_func(mod_dict)
    if region == "positive":
        while not np.isclose(
            llci, 0, rtol, atol
        ):  # b > 0, we are looking at when LLCI intersects 0
            if (
                llci < 0
            ):  # If LLCI has crossed zero, moderator value is too small (large if decreasing slope)
                min_val, max_val = (
                    (cval, max_val) if slope == "positive" else (min_val, cval)
                )
            else:  # If it has not crossed zero yet, moderator value is too large (small if decreasing slope)
                min_val, max_val = (
                    (min_val, cval) if slope == "positive" else (cval, max_val)
                )
            prev_cval, cval = (cval, (max_val + min_val) / 2)
            mod_dict[mod_symb] = cval
            b, se, llci, ulci = spotlight_func(mod_dict)
            if (
                prev_cval == cval
            ):  # If we cannot reach 0 with the desired level of precision
                break
    else:
        while not np.isclose(
            ulci, 0, rtol, atol
        ):  # b < 0, we are looking at when ULCI intersects 0
            if ulci < 0:  # If ULCI has not crossed zero, moderator value is too small.
                min_val, max_val = (
                    (cval, max_val) if slope == "positive" else (min_val, cval)
                )
            else:  # If ULCI has crossed zero, moderator value is too large.
                min_val, max_val = (
                    (min_val, cval) if slope == "positive" else (cval, max_val)
                )
            prev_cval, cval = (cval, (max_val + min_val) / 2)
            mod_dict[mod_symb] = cval
            b, se, llci, ulci = spotlight_func(mod_dict)
            if (
                prev_cval == cval
            ):  # If we cannot reach 0 with the desired level of precision
                break
    return cval
