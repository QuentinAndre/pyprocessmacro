# -*- coding: utf-8 -*-
import copy
import warnings
from functools import partial
from itertools import product, combinations

import numpy as np
import pandas as pd
import scipy.stats as stats
from numpy import dot
from numpy.linalg import inv, LinAlgError

from . import negbin
from .bootstrap import BootstrapSpec, bootstrap_parameters, family_of
from .categorical import code_symbols, mod_dict as _mod_dict
from .utils import (
    fast_OLS,
    fast_optimize,
    bootstrap_sampler,
    eval_expression,
    bias_corrected_ci,
    z_score,
    t_score,
    percentile_ci,
    find_significance_region,
    ConvergenceError,
)


def _with_code_column(frames, labels):
    """Concatenate one table per X code, with the code label in a first column "X" (#17)."""
    out = []
    for label, frame in zip(labels, frames):
        frame = frame.copy()
        frame.insert(0, "X", label)
        out.append(frame)
    table = pd.concat(out)
    table.index = [""] * len(table)
    return table


def _summary_table(levels, level_columns, stats, stat_columns):
    """
    Assemble a summary table from label columns and numeric statistics, keeping each column's type (#72).
    :param levels: list of rows of labels and moderator values
    :param level_columns: their column names
    :param stats: (n_rows x n_stats) array of statistics
    :param stat_columns: their column names
    """
    labels = pd.DataFrame(list(levels), columns=level_columns)
    numbers = pd.DataFrame(np.asarray(stats, dtype=float), columns=stat_columns)
    table = pd.concat([labels, numbers], axis=1)
    table.index = [""] * len(table)
    return table


class BaseLogit(object):
    """
    A convenience parent class for the methods used in Logistic models.
    """

    def __init__(self, endog: np.array, exog: np.array, options: dict) -> None:
        self._endog = endog
        self._exog = exog
        self._n_obs = exog.shape[0]
        self._n_vars = exog.shape[1]
        if not options:
            options = {}
        self._options = options

    @staticmethod
    def _cdf(X: np.array) -> np.array:
        """
        The CDF of the logistic function.
        :param X: Values at which to evaluate the CDF
        :return: The CDF of the logistic function, evaluated at X
        """
        idx = X > 0
        out = np.empty(X.size, dtype=float)
        with warnings.catch_warnings():
            warnings.filterwarnings("error")
            try:
                out[idx] = 1 / (1 + np.exp(-X[idx]))
                exp_X = np.exp(X[~idx])
                out[~idx] = exp_X / (1 + exp_X)
                return out
            except RuntimeWarning:
                raise RuntimeError

    def _loglike(self, params):
        return np.sum(self._loglikeobs(params))

    def _loglikeobs(self, params):
        q = 2 * self._endog - 1
        X = self._exog
        return np.log(self._cdf(q * dot(X, params)))

    def _score(self, params):
        z = dot(self._exog, params)
        L = self._cdf(z)
        return dot(self._endog - L, self._exog)

    def _hessian(self, params):
        X = self._exog
        L = self._cdf(dot(X, params))
        return dot(L * (1 - L) * X.T, X)

    def _optimize(self):
        max_iter = self._options["iterate"]
        tolerance = self._options["convergence"]
        iterations = 0

        def score(params):
            return self._score(params) / self._n_obs

        def hess(params):
            return -self._hessian(params) / self._n_obs

        oldparams = np.inf
        newparams = np.zeros(self._n_vars)
        while iterations < max_iter and np.any(
                np.abs(newparams - oldparams) > tolerance
        ):
            oldparams = newparams
            try:
                H = hess(oldparams)
                newparams = oldparams - dot(inv(H), score(oldparams))
            except LinAlgError:
                raise ConvergenceError("The Hessian of the logistic regression is singular.")
            iterations += 1
            if not np.all(np.isfinite(newparams)):
                raise ConvergenceError(
                    "The logistic regression diverged (the outcome may be perfectly separated)."
                )
        if np.any(np.abs(newparams - oldparams) > tolerance):
            raise ConvergenceError(
                f"The logistic regression did not converge in {max_iter} iterations "
                "(increase 'iterate', relax 'convergence', or check the outcome for separation)."
            )
        if np.all(np.abs(self._endog - self._cdf(dot(self._exog, newparams))) < 1e-8):
            # Saturated fit: the score is exactly zero, so the loop above "converges" to huge coefficients.
            raise ConvergenceError("The logistic regression is perfectly separated: every outcome is predicted exactly, so the coefficients are not identified.")
        return newparams


class NullLogitModel(BaseLogit):
    def __init__(self, endog, options=None):
        n_obs = endog.shape[0]
        exog = np.ones((n_obs, 1))
        if not options:
            options = {}
        super().__init__(endog, exog, options)


class BaseOutcomeModel(object):
    """
    A statistical model reflecting the path from independent predictors (X, or X and M)
    to an endogenous outcome (Y, or M).
    """

    def __init__(
            self, data, endogvar, exogvars, symb_to_ind, symb_to_var, options=None
    ):
        """
        Instantiate the model.
        :param data: np.array
            A NxK array of data
        :param endogvar: string
            The name of the endogenous variable.
        :param exogvars: list of strings
            The names of the exogenous variables.
        :param symb_to_ind: dict of int
            A dictionary mapping variable symbols to indices.
        :param symb_to_var: dict of strings
            A dictionary mapping variable symbols to names.
        :param options: dict
            A dictionary of options.
        """

        if options is None:
            options = {}
        self._data = data
        self._endogvar = endogvar
        self._exogvars = exogvars
        self._symb_to_ind = symb_to_ind
        self._symb_to_var = symb_to_var
        if not options:
            options = {}
        self._options = options

        endog_ind = self._symb_to_ind[self._endogvar]
        exog_ind = [self._symb_to_ind[var] for var in self._exogvars]
        self._endog = data[:, endog_ind].astype(float)
        self._exog = data[:, exog_ind].astype(float)

        self._n_obs = self._exog.shape[0]
        self._n_vars = self._exog.shape[1]

        self._varnames = [i for i in self._exogvars if (("*" not in i) & (i != "Cons"))]
        self._derivative = self._gen_derivative(wrt="x")

        self.estimation_results = self._estimate()

    def _gen_derivative(self, wrt):
        """
        Generate a symbolic derivative of the equation with respect to the variable 'wrt', and stores it in a matrix.

        For instance (Model 21), we consider the equation aConstant + bX + cW + dX*W, that we derivate wrt to X:
            * The rearranged equation for X is: 1*(aConstant + cW) + X*(b + dW).
            * The derivative of this expression is: (b + dW), or in matrix form: [0, 1, 0, W] * [a, b, c, d]

        The first vector depends on the value of the moderator W: therefore, it cannot be represented numerically.
        Instead, we express derivative using the following technique:
            * Each term in the equation (i.e. Constant, X, W, X*W) is represented by a row.
            * Each variable is represented by a column.
            * The column for X (the variable with respect to which the equation is derivated) is equal to 0 if the
                term does not contain X, and 1 otherwise
            * The other columns are equal to the variable if the term contains the variable, and to 1 otherwise.

        That way, the product of the columns is equal to the value of each term in the derivative:
           X  W
        [[ 0, 1 ], # Value of the Constant term : 0*1 = 0
         [ 1, 1 ], # Value of X term : 1*1 = 1
         [ 0, W ], # Value of the W term: 0*W = 0
         [ 1, W ]] # Value of the X*W: 1*W = W

        The advantage of this matrix is that it is a symbolic expression, in which we can substitute for the values of
        the moderators, and then take the product of columns to obtain the numerical representation of the derivative
        as a vector.

        :return: A matrix of size (n_terms x n_vars)
        """
        deriv = np.empty((len(self._varnames), len(self._exogvars)), dtype="object")
        for i, var in enumerate(self._varnames):
            if var == wrt:
                deriv[i] = [1 if var in term else 0 for term in self._exogvars]
            else:
                deriv[i] = [var if var in term else 1 for term in self._exogvars]
        return deriv.T

    def coeff_summary(self):
        """
        Get the estimates of the terms in the model.
        :return: A DataFrame of betas, se, t (or z), p, llci, ulci for all variables of the model.
        """
        results = self.estimation_results
        if results:
            if "t" in results.keys():  # Model has t-stats rather than z-stats
                coeffs = np.array(
                    [
                        results["betas"],
                        results["se"],
                        results["t"],
                        results["p"],
                        results["llci"],
                        results["ulci"],
                    ]
                ).T
                df = pd.DataFrame(
                    coeffs,
                    index=results["names"],
                    columns=["coeff", "se", "t", "p", "LLCI", "ULCI"],
                )
            else:  # Model has z-stats.
                coeffs = np.array(
                    [
                        results["betas"],
                        results["se"],
                        results["z"],
                        results["p"],
                        results["llci"],
                        results["ulci"],
                    ]
                ).T
                df = pd.DataFrame(
                    coeffs,
                    index=results["names"],
                    columns=["coeff", "se", "Z", "p", "LLCI", "ULCI"],
                )
        else:
            raise NotImplementedError(
                "The model has not been estimated yet. Please estimate the model first."
            )
        return df

    def _estimate(self):
        pass

    def to_statsmodels(self):
        """
        Refit this outcome model with statsmodels and return the results object (#67).

        OLS models use statsmodels.OLS with the same covariance estimator (t-based inference, as here);
        logistic models use statsmodels.Logit and negative binomial models statsmodels.NegativeBinomial,
        whose default covariance is the inverse Hessian, as here.
        statsmodels is an optional dependency: pip install pyprocessmacro[statsmodels].
        """
        try:
            import statsmodels.api as sm
        except ImportError as error:  # pragma: no cover - exercised with a stubbed module
            raise ImportError(
                "to_statsmodels() needs statsmodels: pip install pyprocessmacro[statsmodels]"
            ) from error
        results = self.estimation_results
        exog = pd.DataFrame(self._exog, columns=results["names"])
        endog = pd.Series(self._endog, name=self._symb_to_var[self._endogvar])
        if "alpha" in results:  # negative binomial (#25): statsmodels estimates alpha as its last parameter
            return sm.NegativeBinomial(endog, exog).fit(disp=0, maxiter=500)
        if "z" in results:
            return sm.Logit(endog, exog).fit(disp=0)
        cov_type = results["cov_type"]
        if cov_type == "standard":
            return sm.OLS(endog, exog).fit()
        return sm.OLS(endog, exog).fit(cov_type=cov_type, use_t=True)


class OLSOutcomeModel(BaseOutcomeModel):
    """
    An OLS subclass for OutcomeModels. Implement methods specific to the OLS estimation.
    """

    def __init__(
            self, data, endogvar, exogvars, symb_to_ind, symb_to_var, options=None
    ):
        super().__init__(data, endogvar, exogvars, symb_to_ind, symb_to_var, options)

    def _estimate(self):
        """
        Estimate the coefficients and statistics of the OLS model, and store the results in a dictionary of
        estimation_results.
        :return: self
        """
        y = self._endog
        x = self._exog
        n_obs = self._n_obs
        n_vars = self._n_vars
        inv_xx = inv(dot(x.T, x))
        xy = dot(x.T, y)
        betas = dot(inv_xx, xy)
        df_e = n_obs - n_vars
        df_r = n_vars - 1
        resid = y - dot(x, betas)
        mse = (resid ** 2).sum() / df_e
        sse = dot(resid.T, resid) / df_e
        errortype = self._options.get("cov_type") or ("HC3" if self._options.get("hc3") else "standard")
        if errortype == "standard":
            vcv = np.true_divide(1, n_obs - n_vars) * dot(resid.T, resid) * inv_xx
        elif errortype == "HC0":
            sq_resid = (resid ** 2).squeeze()
            vcv = dot(dot(dot(inv_xx, x.T) * sq_resid, x), inv_xx)
        elif errortype == "HC1":
            sq_resid = (resid ** 2).squeeze()
            vcv = np.true_divide(n_obs, n_obs - n_vars) * dot(  # n_vars counts the constant (#52)
                dot(dot(inv_xx, x.T) * sq_resid, x), inv_xx
            )
        elif errortype == "HC2":
            sq_resid = (resid ** 2).squeeze()
            H = (x.dot(inv_xx) * x).sum(axis=-1)
            vcv = dot(dot(dot(inv_xx, x.T) * (sq_resid / (1 - H)), x), inv_xx)
        elif errortype == "HC3":
            sq_resid = (resid ** 2).squeeze()
            H = (x.dot(inv_xx) * x).sum(axis=-1)
            vcv = dot(dot(dot(inv_xx, x.T) * (sq_resid / ((1 - H) ** 2)), x), inv_xx)
        else:
            raise ValueError(
                "The covariance type {} is not supported. Please specify 'standard', 'HC0'"
                "'HC1', 'HC2', or 'HC3'".format(errortype)
            )

        betas = betas.squeeze()
        se = np.sqrt(np.diagonal(vcv)).squeeze()
        t = betas / se
        p = stats.t.sf(np.abs(t), df_e) * 2
        conf = self._options["conf"]
        tcrit = t_score(conf, df_e)  # OLS intervals use the t distribution, as PROCESS does (#40)
        R2 = 1 - resid.var() / y.var()
        adjR2 = 1 - (1 - R2) * ((n_obs - 1) / df_e)  # n_vars already counts the constant (#41)
        F = (R2 / df_r) / ((1 - R2) / df_e)
        F_pval = stats.f.sf(F, df_r, df_e)
        rss = float(dot(resid.T, resid))
        llf = -n_obs / 2 * (np.log(2 * np.pi) + np.log(rss / n_obs) + 1)
        aic = 2 * n_vars - 2 * llf
        bic = n_vars * np.log(n_obs) - 2 * llf
        llci = betas - (se * tcrit)
        ulci = betas + (se * tcrit)
        names = [self._symb_to_var.get(x, x) for x in self._exogvars]
        estimation_results = {
            "betas": betas,
            "se": se,
            "vcv": vcv,
            "t": t,
            "p": p,
            "R2": R2,
            "adjR2": adjR2,
            "df_e": int(df_e),
            "df_r": int(df_r),
            "mse": mse,
            "F": F,
            "sse": sse,
            "F_pval": F_pval,
            "llci": llci,
            "ulci": ulci,
            "names": names,
            "n": int(n_obs),
            "llf": llf,
            "aic": aic,
            "bic": bic,
            "cov_type": errortype,
        }
        return estimation_results

    def model_summary(self):
        """
        The summary of the model statistics: R², F-stats, etc...
        :return: A DataFrame of model statistics
        """
        results = self.estimation_results
        statistics = ["R2", "adjR2", "mse", "F", "df_r", "df_e", "F_pval"]
        row = [[results[s] for s in statistics]]
        df = pd.DataFrame(
            row,
            index=[""],
            columns=["R²", "Adj. R²", "MSE", "F", "df1", "df2", "p-value"],
        )
        return df

    def coeff_summary(self):
        """
        The summary of the OLS estimates for the model: betas, se, t, p-values, etc...
        :return: A DataFrame of coefficient statistics
        """
        return super().coeff_summary()

    def summary(self):
        """
        Pretty-print the summary with text. Used by Process to display the model and coefficients in a nicer way.
        :return: A string to display.
        """
        prec = self._options["precision"]
        float_format = partial("{:.{prec}f}".format, prec=prec)
        basestr = (
            "Outcome = {} \n"
            "OLS Regression Summary\n\n{}\n\n"
            "Coefficients\n\n{}".format(
                self._symb_to_var[self._endogvar],
                self.model_summary().to_string(float_format=float_format),
                self.coeff_summary().to_string(float_format=float_format),
            )
        )
        return basestr

    def __str__(self):
        return self.summary()


class LogitOutcomeModel(BaseOutcomeModel, BaseLogit):
    """
    A Logit subclass for OutcomeModels. Implement methods specific to the Logistic estimation.
    """

    def __init__(
            self, data, endogvar, exogvars, symb_to_ind, symb_to_var, options=None
    ):
        super().__init__(data, endogvar, exogvars, symb_to_ind, symb_to_var, options)

    def _estimate(self):
        """
        Estimate the coefficients and statistics of the Logistic model, and store the results in a dictionary of
        estimation_results.
        :return: self
        """
        betas = self._optimize()
        vcv = inv(self._hessian(betas))

        se = np.sqrt(np.diagonal(vcv)).squeeze()
        z = betas / se
        p = stats.norm.sf(np.abs(z)) * 2
        conf = self._options["conf"]
        zscore = z_score(conf)
        llci = betas - (se * zscore)
        ulci = betas + (se * zscore)

        # GOF statistics
        llmodel = self._loglike(betas)
        minus2ll = -2 * llmodel

        null_model = NullLogitModel(self._endog, self._options)
        betas_null = null_model._optimize()
        llnull = null_model._loglike(betas_null)

        d = 2 * (llmodel - llnull)
        pvalue = stats.chi2.sf(d, self._n_vars - 1)
        mcfadden = 1 - llmodel / llnull
        # Likelihood ratios are taken in log space: exp(llnull) underflows to 0 beyond about a
        # thousand observations, which turned both pseudo R-squared into NaN (#42).
        coxsnell = 1 - np.exp(2 * (llnull - llmodel) / self._n_obs)
        nagelkerke = coxsnell / (1 - np.exp(2 * llnull / self._n_obs))
        aic = 2 * self._n_vars - 2 * llmodel
        bic = self._n_vars * np.log(self._n_obs) - 2 * llmodel
        names = [self._symb_to_var.get(x, x) for x in self._exogvars]
        estimation_results = {
            "betas": betas,
            "se": se,
            "vcv": vcv,
            "z": z,
            "p": p,
            "llci": llci,
            "ulci": ulci,
            "mcfadden": mcfadden,
            "coxsnell": coxsnell,
            "nagelkerke": nagelkerke,
            "d": d,
            "minus2ll": minus2ll,
            "pvalue": pvalue,
            "n": int(self._n_obs),
            "names": names,
            "llf": llmodel,
            "llnull": llnull,
            "aic": aic,
            "bic": bic,
            "df_model": int(self._n_vars - 1),
            "cov_type": "hessian",
        }
        return estimation_results

    def model_summary(self):
        """
        The summary of the model statistics: Model LL, pseudo R², etc...
        :return: A DataFrame of model statistics
        """
        results = self.estimation_results
        row = [
            [
                results[i]
                for i in [
                "minus2ll",
                "d",
                "pvalue",
                "mcfadden",
                "coxsnell",
                "nagelkerke",
                "n",
            ]
            ]
        ]
        return pd.DataFrame(
            row,
            index=[""],
            columns=[
                "-2LL",
                "Model LL",
                "p-value",
                "McFadden",
                "Cox-Snell",
                "Nagelkerke",
                "n",
            ],
        )

    def coeff_summary(self):
        """
        The summary of the OLS estimates for the model: betas, se, t, p-values, etc...
        :return: A DataFrame of coefficient statistics
        """
        return super().coeff_summary()

    def summary(self):
        """
        Pretty-print the summary with text. Used by Process to display the model and coefficients in a nicer way.
        :return: A string to display.
        """
        prec = self._options["precision"]
        float_format = partial("{:.{prec}f}".format, prec=prec)
        basestr = (
            "\n**************************************************************************\n"
            "Outcome = {} \n"
            "Logistic Regression Summary\n\n{}\n\n"
            "Coefficients\n\n{}".format(
                self._symb_to_var[self._endogvar],
                self.model_summary().to_string(float_format=float_format),
                self.coeff_summary().to_string(float_format=float_format),
            )
        )
        return basestr

    def __str__(self):
        return self.summary()


class NegBinOutcomeModel(BaseOutcomeModel):
    """
    Negative binomial regression (NB2, log link) for a count outcome Y (#25): a PyProcessMacro extension that
    PROCESS does not offer. The coefficients are on the log-count scale with Wald z tests, and the dispersion
    alpha (Var(Y) = mu + alpha mu^2) is estimated by maximum likelihood and reported in the model summary.
    """

    @staticmethod
    def _cdf(linear):
        """The mean function, exp. Named as the logistic model's so that augment() finds the fitted values."""
        return np.exp(linear)

    def _estimate(self):
        max_iter, tolerance = self._options["iterate"], self._options["convergence"]
        params = negbin.fit(self._endog, self._exog, max_iter, tolerance)
        betas, log_alpha = params[:-1], params[-1]
        try:
            vcv_full = inv(-negbin.hessian(self._endog, self._exog, params))
        except LinAlgError:
            raise ConvergenceError("The Hessian of the negative binomial regression is singular.")
        vcv = vcv_full[:-1, :-1]
        se = np.sqrt(np.diagonal(vcv))
        z = betas / se
        p = stats.norm.sf(np.abs(z)) * 2
        zscore = z_score(self._options["conf"])
        llci = betas - se * zscore
        ulci = betas + se * zscore
        alpha = float(np.exp(log_alpha))
        alpha_se = alpha * float(np.sqrt(max(vcv_full[-1, -1], 0.0)))  # delta method from log alpha
        llmodel = negbin.loglike_sum(self._endog, self._exog, params)
        constant = np.ones((self._n_obs, 1))
        llnull = negbin.loglike_sum(self._endog, constant, negbin.fit(self._endog, constant, max_iter, tolerance))
        d = 2 * (llmodel - llnull)
        pvalue = stats.chi2.sf(d, self._n_vars - 1)
        n_params = self._n_vars + 1  # alpha counts as a parameter
        names = [self._symb_to_var.get(x, x) for x in self._exogvars]
        return {
            "betas": betas,
            "se": se,
            "vcv": vcv,
            "z": z,
            "p": p,
            "llci": llci,
            "ulci": ulci,
            "alpha": alpha,
            "alpha_se": alpha_se,
            "d": d,
            "minus2ll": -2 * llmodel,
            "pvalue": pvalue,
            "mcfadden": 1 - llmodel / llnull,
            "n": int(self._n_obs),
            "names": names,
            "llf": llmodel,
            "llnull": llnull,
            "aic": 2 * n_params - 2 * llmodel,
            "bic": n_params * np.log(self._n_obs) - 2 * llmodel,
            "df_model": int(self._n_vars - 1),
            "cov_type": "hessian",
        }

    def model_summary(self):
        """The model statistics: -2LL, the likelihood-ratio test against the intercept-only model, alpha."""
        results = self.estimation_results
        row = [[results[i] for i in ["minus2ll", "d", "pvalue", "mcfadden", "alpha", "n"]]]
        return pd.DataFrame(row, index=[""], columns=["-2LL", "Model LL", "p-value", "McFadden", "alpha", "n"])

    def summary(self):
        prec = self._options["precision"]
        float_format = partial("{:.{prec}f}".format, prec=prec)
        return (
            "\n**************************************************************************\n"
            "Outcome = {} \n"
            "Negative Binomial Regression Summary\n\n{}\n\n"
            "Coefficients\n\n{}".format(
                self._symb_to_var[self._endogvar],
                self.model_summary().to_string(float_format=float_format),
                self.coeff_summary().to_string(float_format=float_format),
            )
        )

    def __str__(self):
        return self.summary()


class ParallelMediationModel(object):
    """
    A class describing a parallel mediation model between an endogenous variable Y, one or several mediators M, and a
    set of exogenous predictors for the endogenous variable and the mediators.
    """

    ANALYSIS_NAMES = {
        "MM": "MODERATED MEDIATION",
        "PMM": "PARTIAL MODERATED MEDIATION",
        "MMM": "MODERATED MODERATED MEDIATION",
        "CMM": "CONDITIONAL MODERATED MEDIATION",
    }

    def __init__(
            self,
            data,
            exog_terms_y,
            exog_terms_m,
            mod_symb,
            spot_values,
            n_meds,
            analysis_list,
            symb_to_ind,
            symb_to_var,
            options=None,
            x_symbs=("x",),
            x_labels=None,
            mod_codes=None,
    ):
        """
        :param data: array
            NxK array of data
        :param exog_terms_y: list of strings
            Symbols of exogenous terms for the estimation of the outcome Y
        :param exog_terms_m: list of strings
            Symbols of exogenous terms for the estimation of the mediator(s) M (same for all mediators)
        :param mod_symb: list of strings
            Symbols of the moderator(s) of the path from X to the mediator(s) M and of the path from M to Y
        :param spot_values: dict of lists
            The spotlight values of the moderator(s)
        :param n_meds: int
            Number of mediator(s)
        :param analysis_list: list of ["MM", "PMM", "CMM", "MMM"]
            The list of additional analysis to conduct.
        :param symb_to_ind: dict of int
            Dictionary mapping the symbols to the indices of the variable in the data
        :param symb_to_var:
            Dictionary mapping the symbols to the actual names of the variable in the data
        :param options: dict
            Dictionary of options, from the Process object
        :param x_symbs: the symbols of X: ("x",) or the codes of a multicategorical X (#17)
        :param x_labels: the labels of the codes (X1, X2, ...) when X is multicategorical
        :param mod_codes: {moderator symbol: {level: {code symbol: value}}} for multicategorical moderators
        """
        self._data = data
        self._exog_terms_y = exog_terms_y
        self._exog_terms_m = exog_terms_m
        self._n_meds = n_meds
        self._symb_to_ind = symb_to_ind
        self._symb_to_var = symb_to_var
        self._n_obs = data.shape[0]
        if not options:
            options = {}
        self._options = options
        self._x_symbs = list(x_symbs)
        self._x_symb = self._x_symbs[0]
        self._x_labels = list(x_labels) if x_labels else [symb_to_var.get("x", "x")]
        self._categorical_x = len(self._x_symbs) > 1
        self._mod_codes = mod_codes or {}
        self._code_views = None

        self._vars_y = [
            i for i in self._exog_terms_y if (("*" not in i) & (i != "Cons"))
        ]
        self._ind_y = self._symb_to_ind["y"]
        self._exog_inds_y = [self._symb_to_ind[var] for var in self._exog_terms_y]

        self._vars_m = [
            i for i in self._exog_terms_m if (("*" not in i) & (i != "Cons"))
        ]
        self._endog_vars_m = ["m{}".format(i + 1) for i in range(self._n_meds)]
        self._inds_m = [self._symb_to_ind[m] for m in self._endog_vars_m]
        self._exog_inds_m = [self._symb_to_ind[var] for var in self._exog_terms_m]

        self._compute_betas_m = fast_OLS
        family = family_of(self._options)
        max_iter = self._options["iterate"]
        tolerance = self._options["convergence"]
        if family == "logit":
            self._compute_betas_y = partial(
                fast_optimize,
                n_obs=self._n_obs,
                n_vars=len(self._exog_inds_y),
                max_iter=max_iter,
                tolerance=tolerance,
            )
        elif family == "negbin":  # #25
            self._compute_betas_y = partial(negbin.fit_betas, max_iter=max_iter, tolerance=tolerance)
        else:
            self._compute_betas_y = fast_OLS

        self._true_betas_y, self._true_betas_m = self._estimate_true_params()
        self._boot_betas_y, self._boot_betas_m, self._n_fail_samples = (
            self._estimate_bootstrapped_params()
        )

        self._base_derivs_by_x = {xs: self._gen_derivatives(xs) for xs in self._x_symbs}
        self._base_derivs = self._base_derivs_by_x[self._x_symb]

        self._moderators_symb = mod_symb
        self._moderators_values = [
            spot_values.get(i, [0]) for i in self._moderators_symb
        ]
        self._has_moderation = True if mod_symb else False
        self._analysis_list = analysis_list
        if self._has_moderation:
            self.estimation_results = self._cond_ind_effects()
        else:
            self.estimation_results = self._simple_ind_effects()

    def _estimate_true_params(self):
        """
        Compute the true parameters for:
            * The path from the predictors to Y (computed using OLS/Logit, depending on the nature of Y)
            * The path(s) from the mediator(s) to Y (computed using OLS)
        :return: A tuple of (true_betas_y, true_betas_m)
            * true_betas_y is a vector of size n_params_y
            * true_betas_m is a list of vectors of size n_params_m
        """
        # True betas of the path from Ms to Y
        endog_y = self._data[:, self._ind_y]
        exog_y = self._data[:, self._exog_inds_y]
        true_betas_y = self._compute_betas_y(endog_y, exog_y)

        # For each mediator Mi, true betas from X to Mi
        true_betas_m = []
        m_exog = self._data[:, self._exog_inds_m]
        for m_ind in self._inds_m:
            m_endog = self._data[:, m_ind]
            betas = self._compute_betas_m(m_endog, m_exog)
            true_betas_m.append(betas)

        return true_betas_y, true_betas_m

    def _estimate_bootstrapped_params(self):
        """
        Compute the bootstrapped parameters for:
            * The path from the predictors to Y (computed using OLS/Logit, depending on the nature of Y)
            * The path(s) from the mediator(s) to Y (computed using OLS)
        :return: A tuple of (boot_betas_y, boot_betas_m, n_fail_samples)
            * boot_betas_y is a matrix of size n_boots x n_params_y
            * boot_betas_m is an array of size n_meds x n_boots x n_params_m
            * n_fail_samples is the number of resamples discarded because an equation could not be estimated
        """
        spec = BootstrapSpec(
            self._ind_y, self._exog_inds_y, self._inds_m, self._exog_inds_m,
            family=family_of(self._options), max_iter=self._options["iterate"],
            tolerance=self._options["convergence"],
        )
        # Batched estimation with the same draws as the sequential loop (#68).
        boot_betas_y, boot_betas_m, n_fail_samples, self._boot_sds = bootstrap_parameters(
            self._data, spec, self._options["boot"], self._options["seed"],
            sd_inds=[self._symb_to_ind["x"], self._ind_y],  # per-resample SDs for the standardized effects (#70)
        )
        return boot_betas_y, boot_betas_m, n_fail_samples

    def _gen_derivatives(self, x_symb="x"):
        """
        Generate the list of symbolic derivatives for the indirect path(s) from X to Y. The derivative of the path from
        X to M is taken with respect to X (or to one code of a multicategorical X), and the derivative of the path
        to Y is taken with respect to M.

        For instance (Model 21), we consider the equation of x_to_m:
            * The equation of x_to_m is: aConstant + bX + cW + dX*W. Rearranging for X: 1*(aConstant + cW) + X*(b + dW).
            * The derivative of this expression is: (b + dW), or in matrix form: [0, 1, 0, W] * [a, b, c, d]

        The first vector depends on the value of the moderator W: therefore, it cannot be represented numerically.
        Instead, we express derivative using the following technique:
            * Each term in the equation (i.e. Constant, X, W, X*W) is represented by a row.
            * Each variable is represented by a column.
            * The column for X (the variable with respect to which the equation is derivated) is equal to 0 if the
                term does not contain X, and 1 otherwise
            * The other columns are equal to the variable if the term contains the variable, and to 1 otherwise.
        That way, the product of the columns is equal to the value of each term in the derivative:

           X  W
        [[ 0, 1 ], # Value of the Constant term : 0*1 = 0
         [ 1, 1 ], # Value of X term : 1*1 = 1
         [ 0, W ], # Value of the W term: 0*W = 0
         [ 1, W ]] # Value of the X*W term: 1*W = W

        The advantage of this matrix is that it is a symbolic expression, in which we can substitute for the values of
        the moderators, and then take the product of columns to obtain the numerical representation of the derivative
        as a vector.

        :return: dict of matrices
            A dictionary with keys 'x_to_m' and 'm_to_y':
                'x_to_m' is the symbolic derivative of X to the mediator(s) M (one derivative)
                'm_to_y' is the list of symbolic derivative(s) from the mediator(s) M to Y (n_meds derivative(s))
        """
        derivs = {}

        # Derivative of X to M
        vars_m = self._vars_m
        exog_terms_m = self._exog_terms_m
        x_to_m = np.empty((len(vars_m), len(exog_terms_m)), dtype="object")
        for j, var in enumerate(vars_m):
            if var == x_symb:
                x_to_m[j] = [1 if var in term.split("*") else 0 for term in exog_terms_m]
            else:
                x_to_m[j] = [var if var in term.split("*") else 1 for term in exog_terms_m]
        derivs["x_to_m"] = x_to_m.T

        list_m_to_y = []
        for i in range(self._n_meds):  # For all the mediators...
            # ... derivate the path from M to Y (unique to each mediator)
            vars_y = self._vars_y
            exog_terms_y = self._exog_terms_y
            m_to_y = np.empty((len(vars_y), len(exog_terms_y)), dtype="object")
            for j, var in enumerate(vars_y):
                if var == "m{}".format(i + 1):
                    m_to_y[j] = [1 if var in term.split("*") else 0 for term in exog_terms_y]
                else:
                    m_to_y[j] = [var if var in term.split("*") else 1 for term in exog_terms_y]
            list_m_to_y.append(m_to_y.T)

        derivs["m_to_y"] = list_m_to_y
        return derivs

    def _indirect_effect_at(self, med_index, mod_dict):
        """
        Compute the indirect effect through a specific mediator at specific value(s) of the moderator(s)
        :param med_index: int
            Index of the mediator.
        :param mod_dict: dict
            None, or a mod_name:mod_value dictionary of moderator values.
        :return: e: scalar
                    Effect at the moderator values
                 be: array
                    Effects for all bootstrap samples (N_Boots x 1)
                 se: scalar
                    Standard error based on bootstrap samples
                 llci: scalar
                    Lower level of CI based on bootstrap samples
                 ulci: scalar
                    Upper level of CI based on bootstrap samples
        """
        conf = self._options["conf"]
        der_x_to_m = self._base_derivs["x_to_m"]
        der_m_to_y = self._base_derivs["m_to_y"][med_index]
        expr_x_to_m = eval_expression(der_x_to_m, mod_dict)
        expr_m_to_y = eval_expression(der_m_to_y, mod_dict)

        # Generation of the effects and bootstrapped effects: product of m_der and y_der
        e = dot(self._true_betas_y, expr_m_to_y) * dot(
            self._true_betas_m[med_index], expr_x_to_m
        )
        be = dot(self._boot_betas_y, expr_m_to_y) * dot(
            self._boot_betas_m[med_index], expr_x_to_m
        )
        se = be.std(ddof=1)
        if self._options["percent"]:
            llci, ulci = percentile_ci(be, conf)
        else:
            llci, ulci = bias_corrected_ci(e, be, conf)
        return e, be, se, llci, ulci

    def _get_conditional_indirect_effects(self, med_index, mod_symb, mod_values):
        """
        Return the indirect effects for all combinations of the moderators mod_symb specified in mod_values.
        :param med_index: int
            Index of the mediator.
        :param mod_values: matrix
            A (N_Comb x N_Mods) matrix of all combinations of values for all moderator(s)
        :return: e: array
                    Effects for all combinations of the moderator values (N_Comb x 1)
                 be: matrix
                    Effects for all combinations of the moderator values for all bootstrap samples (N_Comb x N_Boots)
                 se: array
                    SE based on bootstrap samples for all combinations of the moderator values (N_Comb x 1)
                 llci: array
                    LLCI based on bootstrap samples for all combinations of the moderator values (N_Comb x 1)
                 ulci: array
                    ULCI based on bootstrap samples for all combinations of the moderator values (N_Comb x 1)
        """

        n_boots = self._options["boot"]
        n_comb = len(mod_values)
        e, se, llci, ulci = np.empty((4, n_comb))
        be = np.empty((n_comb, n_boots))

        for i, vals in enumerate(mod_values):
            mod_dict = _mod_dict(mod_symb, vals, self._mod_codes)
            e[i], be[i], se[i], llci[i], ulci[i] = self._indirect_effect_at(
                med_index, mod_dict
            )

        return e, be, se, llci, ulci

    def _simple_ind_effects(self):
        """
        Generate the indirect effects.
        This is done only if the indirect path from X to Y through M is not moderated.
        If the option "total"  is set to 1, then the total indirect effect is estimated.
        If the option "contrast" is set to 1, then the pairwise contrasts between the different mediators are estimated.
        :return: dict
            A dictionary of lists "effect", "se", "llci", and "ulci".
        """
        conf = self._options["conf"]
        n_boots = self._options["boot"]
        e = np.empty(self._n_meds)
        be = np.empty((self._n_meds, n_boots))

        for i in range(self._n_meds):
            e[i], be[i], *_ = self._indirect_effect_at(i, {})

        effects = []
        se = []
        llci, ulci = [], []

        if self._options["total"]:
            total_e = e.sum()
            boot_total_e = be.sum(axis=0)
            total_se = boot_total_e.std(ddof=1)
            if self._options["percent"]:
                total_ci = percentile_ci(boot_total_e, conf)
            else:
                total_ci = bias_corrected_ci(total_e, boot_total_e, conf)
            effects.append(total_e)
            se.append(total_se)
            llci.append(total_ci[0])
            ulci.append(total_ci[1])

        for i in range(self._n_meds):
            effects.append(e[i])
            se.append(be[i].std(ddof=1))
            if self._options["percent"]:
                ci = percentile_ci(be[i], conf)
            else:
                ci = bias_corrected_ci(e[i], be[i], conf)
            llci.append(ci[0])
            ulci.append(ci[1])

        if self._options["contrast"]:
            inds = [i for i in range(self._n_meds)]
            contrasts = combinations(inds, 2)
            for i1, i2 in contrasts:
                cont_e = e[i1] - e[i2]
                boot_cont_e = be[i1] - be[i2]
                cont_se = boot_cont_e.std(ddof=1)
                if self._options["percent"]:
                    cont_ci = percentile_ci(boot_cont_e, conf)
                else:
                    cont_ci = bias_corrected_ci(cont_e, boot_cont_e, conf)
                effects.append(cont_e)
                se.append(cont_se)
                llci.append(cont_ci[0])
                ulci.append(cont_ci[1])

        statistics = [np.array(i).flatten() for i in [effects, se, llci, ulci]]

        return {k: v for k, v in zip(["effect", "se", "llci", "ulci"], statistics)}

    def _cond_ind_effects(self):
        """
        Generate the conditional indirect effects for all mediators.
        :return: dict
                    A dictionary "effect", "se", "llci", and "ulci" of (N_Meds x N_Comb) matrices, corresponding to the
                    statistics for the N_Meds mediators at the N_Comb different levels of the moderator(s).
        """
        mod_values = [i for i in product(*self._moderators_values)]
        mod_symb = self._moderators_symb

        n_cond_effects = len(mod_values)
        effects, se, llci, ulci = np.empty((4, self._n_meds, n_cond_effects))

        for i in range(self._n_meds):
            effects[i], _, se[i], llci[i], ulci[
                i
            ] = self._get_conditional_indirect_effects(i, mod_symb, mod_values)

        statistics = [i.flatten() for i in [effects, se, llci, ulci]]
        return {k: v for k, v in zip(["effect", "se", "llci", "ulci"], statistics)}

    def _MM_index(self):
        """
        The Moderated Mediation (MM) index is computed when exactly one moderator is present on the
        mediation path.
        It represents the marginal impact of one moderator (i.e. the impact of an increase in one unit for this
        moderator on the indirect effect).
        """
        if "MM" not in self._analysis_list:
            raise ValueError(
                "This model does not report the Index for Moderated Mediation."
            )

        conf = self._options["conf"]
        (mod,) = self._moderators_symb  # Only one moderator

        # A unit increase of the moderator; for a multicategorical moderator, one code at a time (#17): the
        # index is then the difference between the conditional indirect effects the code defines.
        if mod in self._mod_codes:
            codes = code_symbols(self._mod_codes, mod)
            steps = [({c: 0.0 for c in codes}, {c: float(c == on) for c in codes}) for on in codes]
        else:
            steps = [({mod: 0}, {mod: 1})]

        effects, se, llci, ulci = np.empty((4, len(steps), self._n_meds))
        for s, (dict_baseline, dict_effect) in enumerate(steps):
            for i in range(self._n_meds):  # ... For all the mediators
                e_baseline, be_baseline, *_ = self._indirect_effect_at(i, dict_baseline)
                e_effect, be_effect, *_ = self._indirect_effect_at(i, dict_effect)
                e_mm = e_effect - e_baseline  # Moderator at 1 vs. Moderator at 0
                be_mm = be_effect - be_baseline
                effects[s, i] = e_mm
                se[s, i] = be_mm.std(ddof=1)
                if self._options["percent"]:
                    llci[s, i], ulci[s, i] = percentile_ci(be_mm, conf)
                else:
                    llci[s, i], ulci[s, i] = bias_corrected_ci(e_mm, be_mm, conf)

        statistics = [i.flatten() for i in [effects, se, llci, ulci]]

        return {k: v for k, v in zip(["effect", "se", "llci", "ulci"], statistics)}

    def _PMM_index(self):
        """
        The Partial Moderated Mediation (PMM) index is only computed when exactly two moderators are present on the
        mediation path.
        It represents the marginal impact of one moderator (i.e. the impact of an increase in one unit for this
        moderator on the indirect effect), conditional on a value of zero for the other moderator.
        """
        self._refuse_categorical_moderators("partial moderated mediation")
        if "PMM" not in self._analysis_list:
            raise ValueError(
                "This model does not report the Index for Partial Moderated Mediation."
            )

        conf = self._options["conf"]
        n_boots = self._options["boot"]
        mod1, mod2 = self._moderators_symb  # Only two moderators

        # noinspection PyTypeChecker
        dict_baseline = dict([[mod1, 0], [mod2, 0]])
        e_baseline, be_baseline = (
            np.empty(self._n_meds),
            np.empty((self._n_meds, n_boots)),
        )

        # noinspection PyTypeChecker
        dict_mod1 = dict([[mod1, 1], [mod2, 0]])
        e_mod1, be_mod1 = np.empty(self._n_meds), np.empty((self._n_meds, n_boots))

        # noinspection PyTypeChecker
        dict_mod2 = dict([[mod1, 0], [mod2, 1]])
        e_mod2, be_mod2 = np.empty(self._n_meds), np.empty((self._n_meds, n_boots))

        effects, se, llci, ulci = np.empty((4, 2, self._n_meds))
        for i in range(self._n_meds):
            e_baseline[i], be_baseline[i], *_ = self._indirect_effect_at(
                i, dict_baseline
            )
            e_mod1[i], be_mod1[i], *_ = self._indirect_effect_at(i, dict_mod1)
            e_mod2[i], be_mod2[i], *_ = self._indirect_effect_at(i, dict_mod2)

            e_pmm1 = (
                    e_mod1[i] - e_baseline[i]
            )  # Effect of Moderator1 at 1 vs. Moderator1 at 0
            e_pmm2 = (
                    e_mod2[i] - e_baseline[i]
            )  # Effect of Moderator2 at 1 vs. Moderator2 at 0

            be_pmm1 = be_mod1[i] - be_baseline[i]  # Bootstrapped effects of...
            be_pmm2 = be_mod2[i] - be_baseline[i]

            effects[0][i] = e_pmm1  # PMM of first moderator
            se[0][i] = be_pmm1.std(ddof=1)

            effects[1][i] = e_pmm2  # PMM of second moderator
            se[1][i] = be_pmm2.std(ddof=1)

            if self._options["percent"]:
                llci[0][i], ulci[0][i] = percentile_ci(be_pmm1, conf)
                llci[1][i], ulci[1][i] = percentile_ci(be_pmm2, conf)
            else:
                llci[0][i], ulci[0][i] = bias_corrected_ci(e_pmm1, be_pmm1, conf)
                llci[1][i], ulci[1][i] = bias_corrected_ci(e_pmm2, be_pmm2, conf)

        statistics = [i.flatten() for i in [effects, se, llci, ulci]]

        return {k: v for k, v in zip(["effect", "se", "llci", "ulci"], statistics)}

    def _MMM_index(self):
        """
        The Moderated Moderated Mediation (MMM) index is only computed when exactly two moderators are present on the
        mediation path.
        It represents the marginal impact of one moderator (i.e. the impact of an increase in one unit for this
        moderator on the indirect effect) on the marginal impact of the other moderator.
        """
        self._refuse_categorical_moderators("moderated moderated mediation")
        if "MMM" not in self._analysis_list:
            raise ValueError(
                "This model does not report the Index for Moderated Moderated Mediation."
            )

        conf = self._options["conf"]
        n_boots = self._options["boot"]
        mod1, mod2 = self._moderators_symb  # Only two moderators

        # noinspection PyTypeChecker
        dict_both_on = dict([[mod1, 1], [mod2, 1]])  # Both moderators are on
        e_both_on, be_both_on = (
            np.empty(self._n_meds),
            np.empty((self._n_meds, n_boots)),
        )

        # noinspection PyTypeChecker
        dict_mod1_on = dict([[mod1, 2], [mod2, 0]])  # Only the first moderator is on
        e_mod1_on, be_mod1_on = (
            np.empty(self._n_meds),
            np.empty((self._n_meds, n_boots)),
        )

        # noinspection PyTypeChecker
        dict_mod2_on = dict([[mod1, 0], [mod2, 2]])  # Only the second moderator is on
        e_mod2_on, be_mod2_on = (
            np.empty(self._n_meds),
            np.empty((self._n_meds, n_boots)),
        )

        effects, se, llci, ulci = np.empty((4, 1, self._n_meds))
        for i in range(self._n_meds):
            e_both_on[i], be_both_on[i], *_ = self._indirect_effect_at(i, dict_both_on)
            e_mod1_on[i], be_mod1_on[i], *_ = self._indirect_effect_at(i, dict_mod1_on)
            e_mod2_on[i], be_mod2_on[i], *_ = self._indirect_effect_at(i, dict_mod2_on)
            e_mmm = e_both_on[i] - (e_mod1_on[i] + e_mod2_on[i]) / 2
            be_mmm = be_both_on[i] - (be_mod1_on[i] + be_mod2_on[i]) / 2

            effects[0][i] = e_mmm
            se[0][i] = be_mmm.std(ddof=1)
            if self._options["percent"]:
                llci[0][i], ulci[0][i] = percentile_ci(be_mmm, conf)
            else:
                llci[0][i], ulci[0][i] = bias_corrected_ci(e_mmm, be_mmm, conf)

        statistics = [i.flatten() for i in [effects, se, llci, ulci]]
        return {k: v for k, v in zip(["effect", "se", "llci", "ulci"], statistics)}

    def _floodlight_analysis(
            self, med_index, mod_symb, modval_range, other_modval_symb, atol=1e-8, rtol=1e-5
    ):
        """
        Conduct a floodlight analysis of the indirect effect for a specific mediator.
        Search the critical values of mod_symb, at specific value(s) mod_dict of the other moderators.
        :param med_index: int
            The index of the mediator for which to conduct the spotlight analysis.
        :param mod_symb: str
            The symbol of the moderator
        :param modval_range: list of float
            The minimum and maximum values of the moderator.
        :param other_modval_symb: dict
            A mod_symb:mod_value dictionary of values for the other moderators of the direct path.
        """

        def spotlight_wrapper(f, i):
            def wrapped(dict_modval):
                b, be, se, llci, ulci = f(i, dict_modval)
                return b, se, llci, ulci

            return wrapped

        spotlight_func = spotlight_wrapper(self._indirect_effect_at, med_index)
        modval_min, modval_max = modval_range
        sig_region = find_significance_region(
            spotlight_func,
            mod_symb,
            modval_min,
            modval_max,
            other_modval_symb,
            atol=atol,
            rtol=rtol,
        )
        return sig_region

    def _CMM_index(self):
        """
        The Conditional Moderated Mediation (CMM) index is only computed when exactly two moderators are present on the
        mediation path.
        It represents the marginal impact of one moderator (i.e. the impact of an increase in one unit for this
        moderator on the indirect effect) at various levels of the other moderator.
        """
        self._refuse_categorical_moderators("conditional moderated mediation")
        if "CMM" not in self._analysis_list:
            raise ValueError(
                "This model does not report the Index for Conditional Moderated Mediation."
            )

        conf = self._options["conf"]
        mod1, mod2 = self._moderators_symb
        mod1_val, mod2_val = self._moderators_values

        n_levels_mod1 = len(mod1_val)
        n_levels_mod2 = len(mod2_val)

        effects_mod1, se_mod1, llci_mod1, ulci_mod1 = np.empty(
            (4, self._n_meds, n_levels_mod1)
        )
        effects_mod2, se_mod2, llci_mod2, ulci_mod2 = np.empty(
            (4, self._n_meds, n_levels_mod2)
        )

        for i in range(self._n_meds):
            for j, val in enumerate(
                    mod1_val
            ):  # Conditional moderated mediation effects for Moderator 2
                # noinspection PyTypeChecker
                dict_off = dict([[mod1, val], [mod2, 0]])
                # noinspection PyTypeChecker
                dict_on = dict([[mod1, val], [mod2, 1]])
                e_off, be_off, *_ = self._indirect_effect_at(i, dict_off)
                e_on, be_on, *_ = self._indirect_effect_at(i, dict_on)
                e_cmm = e_on - e_off
                be_cmm = be_on - be_off

                effects_mod1[i][j] = e_cmm
                se_mod1[i][j] = be_cmm.std(ddof=1)
                if self._options["percent"]:
                    llci_mod1[i][j], ulci_mod1[i][j] = percentile_ci(be_cmm, conf)
                else:
                    llci_mod1[i][j], ulci_mod1[i][j] = bias_corrected_ci(
                        e_cmm, be_cmm, conf
                    )

            for j, val in enumerate(
                    mod2_val
            ):  # Conditional moderated mediation effects for Moderator 1
                # noinspection PyTypeChecker
                dict_off = dict([[mod2, val], [mod1, 0]])
                # noinspection PyTypeChecker
                dict_on = dict([[mod2, val], [mod1, 1]])
                e_off, be_off, *_ = self._indirect_effect_at(i, dict_off)
                e_on, be_on, *_ = self._indirect_effect_at(i, dict_on)
                e_cmm = e_on - e_off
                be_cmm = be_on - be_off

                effects_mod2[i][j] = e_cmm
                se_mod2[i][j] = be_cmm.std(ddof=1)
                if self._options["percent"]:
                    llci_mod2[i][j], ulci_mod2[i][j] = percentile_ci(be_cmm, conf)
                else:
                    llci_mod2[i][j], ulci_mod2[i][j] = bias_corrected_ci(
                        e_cmm, be_cmm, conf
                    )

        stats_mod1 = [
            i.flatten() for i in [effects_mod1, se_mod1, llci_mod1, ulci_mod1]
        ]
        stats_mod2 = [
            i.flatten() for i in [effects_mod2, se_mod2, llci_mod2, ulci_mod2]
        ]
        statistics = np.concatenate([stats_mod1, stats_mod2], axis=1)

        return {k: v for k, v in zip(["effect", "se", "llci", "ulci"], statistics)}

    def _cond_ind_effects_wrapper(self):
        """
        A wrapper for the conditional indirect effects.
        :return: pd.DataFrame
            A DataFrame of effects, se, llci, and ulci, for the conditional indirect effects.
        """
        symb_to_var = self._symb_to_var
        results = self.estimation_results
        rows_stats = np.array(
            [results["effect"], results["se"], results["llci"], results["ulci"]]
        ).T
        cols_stats = ["Effect", "Boot SE", "BootLLCI", "BootULCI"]

        mod_values = self._moderators_values
        med_values = [
            [
                symb_to_var.get("m{}".format(i + 1), "m{}".format(i + 1))
                for i in range(self._n_meds)
            ]
        ]
        values = med_values + mod_values

        cols_levels = ["Mediator"] + [
            symb_to_var.get(x, x) for x in self._moderators_symb
        ]
        return _summary_table(product(*values), cols_levels, rows_stats, cols_stats)

    def _simple_ind_effects_wrapper(self):
        """
        A wrapper for the indirect effects (and for total/contrast effects if specified)
        :return: pd.DataFrame
            A DataFrame of effects, se, llci, and ulci, for the simple/total/constrasts of indirect effects.
        """
        symb_to_var = self._symb_to_var
        results = self.estimation_results
        rows_stats = np.array(
            [results["effect"], results["se"], results["llci"], results["ulci"]]
        ).T

        med_names = [
            symb_to_var.get("m{}".format(i + 1), "m{}".format(i + 1))
            for i in range(self._n_meds)
        ]
        rows_levels = []
        if self._options["total"]:
            rows_levels += ["TOTAL"]
        rows_levels += med_names
        if self._options["contrast"]:
            contrasts = [
                "Contrast: {} vs. {}".format(a, b)
                for a, b in combinations(med_names, 2)
            ]
            rows_levels += contrasts
        return _summary_table(
            [[label] for label in rows_levels], [""], rows_stats, ["Effect", "Boot SE", "BootLLCI", "BootULCI"]
        )

    def _MM_index_wrapper(self):
        """
        A wrapper for the Moderated Mediation index.
        :return: pd.DataFrame
            A DataFrame of effects, se, llci, and ulci, for the PMM index.
        """
        symb_to_var = self._symb_to_var
        results = self._MM_index()
        rows_stats = np.array(
            [results["effect"], results["se"], results["llci"], results["ulci"]]
        ).T
        cols_stats = ["Index", "Boot SE", "LLCI", "ULCI"]

        mod_names = [self.mm_moderator_labels()]
        med_names = [
            [
                symb_to_var.get("m{}".format(i + 1), "m{}".format(i + 1))
                for i in range(self._n_meds)
            ]
        ]
        return _summary_table(product(*(mod_names + med_names)), ["Moderator", "Mediator"], rows_stats, cols_stats)

    def mm_moderator_labels(self):
        """The rows of the index of moderated mediation: the moderator, or each code of a categorical one (#17)."""
        (mod,) = self._moderators_symb
        if mod in self._mod_codes:
            return [self._symb_to_var.get(c, c) for c in code_symbols(self._mod_codes, mod)]
        return [self._symb_to_var.get(mod, mod)]

    def _refuse_categorical_moderators(self, what):
        categorical = [self._symb_to_var.get(m, m) for m in self._moderators_symb if m in self._mod_codes]
        if categorical:
            raise NotImplementedError(
                f"The index of {what} is not available with a multicategorical moderator ({', '.join(categorical)})."
            )

    def _PMM_index_wrapper(self):
        """
        A wrapper for the Partial Moderated Mediation index.
        :return: pd.DataFrame
            A DataFrame of effects, se, llci, and ulci, for the PMM index.
        """
        symb_to_var = self._symb_to_var
        results = self._PMM_index()
        rows_stats = np.array(
            [results["effect"], results["se"], results["llci"], results["ulci"]]
        ).T
        cols_stats = ["Index", "Boot SE", "LLCI", "ULCI"]

        mod_names = [[symb_to_var.get(i, i) for i in self._moderators_symb]]
        med_names = [
            [
                symb_to_var.get("m{}".format(i + 1), "m{}".format(i + 1))
                for i in range(self._n_meds)
            ]
        ]
        return _summary_table(product(*(mod_names + med_names)), ["Moderator", "Mediator"], rows_stats, cols_stats)

    def _CMM_index_wrapper(self):
        """
        A wrapper for the Conditional Moderated Mediation index.
        :return: pd.DataFrame
            A DataFrame of effects, se, llci, and ulci, for the CMM index.
        """
        symb_to_var = self._symb_to_var
        results = self._CMM_index()

        rows_stats = np.array(
            [results["effect"], results["se"], results["llci"], results["ulci"]]
        ).T
        cols_stats = ["Index", "Boot SE", "BootLLCI", "BootULCI"]

        mod1_name, mod2_name = [symb_to_var.get(i, i) for i in self._moderators_symb]
        mod1_values, mod2_values = self._moderators_values
        med_names = [
            symb_to_var.get("m{}".format(i + 1), "m{}".format(i + 1))
            for i in range(self._n_meds)
        ]

        rows_modname = [mod2_name] * len(mod1_values) * self._n_meds + [
            mod1_name
        ] * len(mod2_values) * self._n_meds
        rows_medname = list(np.repeat(med_names, len(mod1_values))) + list(np.repeat(med_names, len(mod2_values)))
        rows_modvalues = list(np.tile(mod1_values, self._n_meds)) + list(np.tile(mod2_values, self._n_meds))
        return _summary_table(
            zip(rows_modname, rows_medname, [float(v) for v in rows_modvalues]),
            ["Focal Mod", "Mediator", "Other Mod At"],
            rows_stats,
            cols_stats,
        )

    def _MMM_index_wrapper(self):
        """
        A wrapper for the Moderated Moderated Mediation index.
        :return: pd.DataFrame
            A DataFrame of effects, se, llci, and ulci, for the CMM index.
        """
        symb_to_var = self._symb_to_var
        results = self._MMM_index()
        rows_stats = np.array(
            [results["effect"], results["se"], results["llci"], results["ulci"]]
        ).T
        cols_stats = ["Index", "Boot SE", "BootLLCI", "BootULCI"]

        med_names = [
            [
                symb_to_var.get("m{}".format(i + 1), "m{}".format(i + 1))
                for i in range(self._n_meds)
            ]
        ]
        return _summary_table(product(*med_names), ["Mediator"], rows_stats, cols_stats)

    def MM_index_summary(self):  # noqa: N802 (PROCESS names)
        """The index of moderated mediation; per code of a multicategorical X (#17)."""
        return self._per_code(lambda m: m._MM_index_summary_single())

    def _MM_index_summary_single(self):
        if "MM" in self._analysis_list:
            return self._MM_index_wrapper()
        else:
            raise NotImplementedError(
                "This model does not report the Moderated Mediation index."
            )

    def MMM_index_summary(self):  # noqa: N802
        return self._per_code(lambda m: m._MMM_index_summary_single())

    def _MMM_index_summary_single(self):
        if "MMM" in self._analysis_list:
            return self._MMM_index_wrapper()
        else:
            raise NotImplementedError(
                "This model does not report the Moderated Moderated Mediation index."
            )

    def PMM_index_summary(self):  # noqa: N802
        return self._per_code(lambda m: m._PMM_index_summary_single())

    def _PMM_index_summary_single(self):
        if "PMM" in self._analysis_list:
            return self._PMM_index_wrapper()
        else:
            raise NotImplementedError(
                "This model does not report the Partial Moderated Mediation index."
            )

    def CMM_index_summary(self):  # noqa: N802
        return self._per_code(lambda m: m._CMM_index_summary_single())

    def _CMM_index_summary_single(self):
        if "CMM" in self._analysis_list:
            return self._CMM_index_wrapper()
        else:
            raise NotImplementedError(
                "This model does not report the Conditional Moderated Mediation index."
            )

    def _raw_indirect_draws(self):
        """Labels, estimates and bootstrap draws of the total (if requested) and of each mediator's indirect effect."""
        e = np.empty(self._n_meds)
        be = np.empty((self._n_meds, self._options["boot"]))
        for i in range(self._n_meds):
            e[i], be[i], *_ = self._indirect_effect_at(i, {})
        labels = [term for component, term in self.effect_labels if component != "contrast"]
        if self._options["total"]:
            e = np.concatenate([[e.sum()], e])
            be = np.concatenate([be.sum(axis=0, keepdims=True), be])
        return labels, e, be

    def effect_sizes(self):
        """Partially and completely standardized indirect effects; see effsize.standardized_effects (#70)."""
        from . import effsize as _effsize

        return _effsize.standardized_effects(self)

    def effect_size_summary(self):
        """The standardized indirect effects as one table with a Standardization column (#70)."""
        from . import effsize as _effsize

        return self._per_code(_effsize.effect_size_table)

    @property
    def effect_labels(self):
        """(component, term) for every row of unmoderated estimation_results, in order."""
        mediators = [self._symb_to_var.get(f"m{i + 1}", f"m{i + 1}") for i in range(self._n_meds)]
        labels = []
        if self._options["total"]:
            labels.append(("total", "total"))
        labels += [("indirect", med) for med in mediators]
        if self._options["contrast"]:
            labels += [("contrast", f"{a} vs. {b}") for a, b in combinations(mediators, 2)]
        return labels

    def _views(self):
        """
        One model per code of a multicategorical X, sharing the estimates and the bootstrap draws and
        differing in the derivative with respect to X (#17); [self] when X is continuous.
        """
        if not self._categorical_x:
            return [self]
        if self._code_views is None:
            views = []
            for xs, label in zip(self._x_symbs, self._x_labels):
                view = copy.copy(self)
                view._x_symbs, view._x_symb, view._x_labels = [xs], xs, [label]
                view._categorical_x, view._code_views = False, None
                view._base_derivs = self._base_derivs_by_x[xs]
                view.estimation_results = (
                    view._cond_ind_effects() if view._has_moderation else view._simple_ind_effects()
                )
                views.append(view)
            self._code_views = views
        return self._code_views

    def _per_code(self, method):
        """A table from `method` for X, or the concatenation over the codes of a multicategorical X."""
        if not self._categorical_x:
            return method(self)
        return _with_code_column([method(view) for view in self._views()], self._x_labels)

    def coeff_summary(self):
        """
        Get the summary of the indirect effect(s); one block per code, labelled in a first column "X", when X
        is multicategorical (#17).
        :return: The appropriate moderated/unmoderated effect(s).
        """
        return self._per_code(
            lambda m: m._cond_ind_effects_wrapper() if m._has_moderation else m._simple_ind_effects_wrapper()
        )

    def summary(self):
        """
        Pretty-print the summary with text. Used by Process to display the coefficients in a nicer way.
        :return: A string to display.
        """
        prec = self._options["precision"]
        float_format = partial("{:.{prec}f}".format, prec=prec)
        analysis_func = {
            "MM": ("MODERATED MEDIATION", self._MM_index_wrapper),
            "PMM": ("PARTIAL MODERATED MEDIATION", self._PMM_index_wrapper),
            "MMM": ("MODERATED MODERATED MEDIATION", self._MMM_index_wrapper),
            "CMM": ("CONDITIONAL MODERATED MEDIATION", self._CMM_index_wrapper),
        }
        symb_to_var = self._symb_to_var
        relative = "Relative " if self._categorical_x else ""
        if self._has_moderation:
            basestr = (
                "{rel}onditional indirect effect(s) of {x} on {y} at values of the moderator(s):\n\n"
                "{coeffs}\n\n".format(
                    rel="Relative c" if self._categorical_x else "C",
                    x=symb_to_var["x"],
                    y=symb_to_var["y"],
                    coeffs=self.coeff_summary().to_string(float_format=float_format),
                )
            )
        else:
            basestr = "{rel}ndirect effect of {x} on {y}:\n\n" "{coeffs}\n\n".format(
                rel="Relative i" if self._categorical_x else "I",
                x=symb_to_var["x"],
                y=symb_to_var["y"],
                coeffs=self.coeff_summary().to_string(float_format=float_format),
            )
        for a in self._analysis_list:
            name, _ = analysis_func[a]
            results = getattr(self, f"{a}_index_summary")()
            basestr += (
                "**************** INDEX OF {name} ******************\n\n"
                "{results}\n\n".format(
                    name=name, results=results.to_string(float_format=float_format)
                )
            )
        if self._options.get("effsize"):
            from . import effsize as _effsize

            for view, label in zip(self._views(), self._x_labels):
                if self._categorical_x:
                    basestr += f"{relative}effects for {label}:\n\n"
                basestr += _effsize.effect_size_text(view, float_format)
        return basestr

    def __str__(self):
        return self.summary()


class DirectEffectModel(object):
    def __init__(
            self, model, mod_symb, spot_values, has_mediation, symb_to_var, options=None,
            x_symbs=("x",), x_labels=None, mod_codes=None,
    ):
        """
        A container for the direct effect of the variable X on the outcome Y. If the model includes one or several
        moderators of X, this container returns the conditional direct effects. With a multicategorical X, one
        relative effect per code (#17).
        :param model: process.OutcomeModel
            The OutcomeModel object of the outcome Y.
        :param mod_symb: list of string
            The symbols of the moderators of the direct effect.
        :param symb_to_var: dict of string
            The dictionary mapping each symbol to a variable name.
        :param options: dict
            The options of the model.

        For the moderation-only models (1 to 3), probe_p holds the p-value of the test of the highest-order
        interaction of X, which PROCESS 3 and later compare to intprobe before reporting the conditional effects,
        and probe_terms the names of the terms tested (#87). Both are None or empty for mediation models, which
        always report their conditional direct effects.
        """
        self._model = model
        self._is_logit = isinstance(model, (LogitOutcomeModel, NegBinOutcomeModel))  # z-based inference
        self._symb_to_var = symb_to_var
        self._x_symbs = list(x_symbs)
        self._x_labels = list(x_labels) if x_labels else [symb_to_var.get("x", "x")]
        self._categorical_x = len(self._x_symbs) > 1
        self._mod_codes = mod_codes or {}
        self._derivatives = {xs: self._model._gen_derivative(wrt=xs) for xs in self._x_symbs}
        self._derivative = self._derivatives[self._x_symbs[0]]
        self._has_mediation = has_mediation
        self._moderators_symb = mod_symb
        self._moderators_values = [
            spot_values.get(i, [0]) for i in self._moderators_symb
        ]
        self._has_moderation = True if self._moderators_symb else False
        if not options:
            options = {}
        self._options = options
        self._estimation_results = self._estimate()
        if self._has_moderation and not self._has_mediation:
            self.probe_p, self.probe_terms = self._probe()
        else:
            self.probe_p, self.probe_terms = None, []

    def _probe(self):
        """
        The p-value PROCESS 3 and later compare to intprobe before reporting the conditional effects of X (#87):
        the test of the highest-order product term(s) of X with its moderator(s), the smallest p-value when two
        additive moderators give two terms of that order. OLS: the coefficient's test, which is the F test of the
        change in R-squared PROCESS prints for a single term under the same covariance estimator. Logit and
        negative binomial: the likelihood-ratio test of the term, which PROCESS prints for a binary outcome.
        :return: (p-value, the names of the terms tested)
        """
        model = self._model
        exog = list(model._exogvars)
        x_set = set(self._x_symbs)
        with_x = [t for t in exog if "*" in t and (set(t.split("*")) & x_set)]
        if not with_x:
            return None, []
        order = max(len(t.split("*")) for t in with_x)
        terms = [t for t in with_x if len(t.split("*")) == order]
        # With multicategorical variables, the codes of one interaction form a single joint test (#17): group the
        # terms by the interaction they belong to, ignoring the code digits.
        code_of = {}
        for symb, codes in self._code_groups().items():
            for c in codes:
                code_of[c] = symb
        groups = {}
        for term in terms:
            key = "*".join(code_of.get(f, f) for f in term.split("*"))
            groups.setdefault(key, []).append(term)
        pvalues = []
        for members in groups.values():
            if self._is_logit:
                reduced = type(model)(
                    model._data, model._endogvar, [t for t in exog if t not in members],
                    model._symb_to_ind, model._symb_to_var, model._options,
                )
                chi2 = 2 * (model.estimation_results["llf"] - reduced.estimation_results["llf"])
                pvalues.append(float(stats.chi2.sf(max(chi2, 0.0), len(members))))
            elif len(members) == 1:
                pvalues.append(float(np.asarray(model.estimation_results["p"]).ravel()[exog.index(members[0])]))
            else:  # Wald F test of the joint interaction under the model's covariance estimator
                idx = [exog.index(t) for t in members]
                b = np.asarray(model.estimation_results["betas"])[idx]
                v = np.asarray(model.estimation_results["vcv"])[np.ix_(idx, idx)]
                f = float(b @ np.linalg.solve(v, b)) / len(idx)
                pvalues.append(float(stats.f.sf(f, len(idx), model.estimation_results["df_e"])))
        return min(pvalues), [self._symb_to_var.get(t, t) for t in terms]

    def _code_groups(self):
        """{symbol: [code symbols]} for X and for the categorical moderators (#17)."""
        groups = {}
        if self._categorical_x:
            groups["x"] = list(self._x_symbs)
        for mod in self._mod_codes:
            groups[mod] = code_symbols(self._mod_codes, mod)
        return groups

    @property
    def probed(self):
        """
        Whether the conditional effects of X are reported: always for mediation models and unmoderated effects,
        otherwise when the p-value of the highest-order interaction is at most intprobe (#87).
        """
        if self.probe_p is None:
            return True
        intprobe = self._options.get("intprobe", 1.0)
        return intprobe >= 1 or self.probe_p <= intprobe

    def _not_probed_text(self):
        x, y = self._symb_to_var["x"], self._symb_to_var["y"]
        return (
            f"Conditional effect(s) of {x} on {y} are not reported: the highest-order interaction "
            f"({', '.join(self.probe_terms)}) has p = {self.probe_p:.4f}, above intprobe = "
            f"{self._options.get('intprobe', 1.0):g}. They remain available from direct_model.coeff_summary().\n"
        )

    def _estimate(self):
        """
        Estimates the direct effect of X on Y, and return the results into as a dictionary.
        :return: dict
            A dictionary of parameters and model estimates.
        """
        mod_values = [i for i in product(*self._moderators_values)]
        mod_symb = self._moderators_symb
        per_code = [self._get_conditional_direct_effects(mod_symb, mod_values, xs) for xs in self._x_symbs]
        betas, se, llci, ulci = (np.concatenate([stats[i] for stats in per_code]) for i in range(4))
        t = betas / se
        if self._is_logit:
            p = stats.norm.sf(np.abs(t)) * 2
        else:
            df_e = self._model.estimation_results["df_e"]
            p = stats.t.sf(np.abs(t), df_e) * 2
        estimation_results = {
            "betas": betas,
            "se": se,
            "t": t,
            "p": p,
            "llci": llci,
            "ulci": ulci,
        }
        return estimation_results

    def _get_conditional_direct_effects(self, mod_symb, mod_values, x_symb=None):
        """
        Estimates the conditional direct effects of X on Y, at different values of the moderator(s)
        :param mod_symb: list of string
            A list of moderator symbols
        :param mod_values: array of int/float
            A list of lists of spotlight values for each moderator (groups for a categorical moderator).
        :param x_symb: the symbol of X, or of one of its codes (#17)
        :return:
        """
        betas, se, llci, ulci = np.zeros((4, len(mod_values)))
        for i, val in enumerate(
                mod_values
        ):  # All possible products of level(s) of moderator(s)
            betas[i], se[i], llci[i], ulci[i] = self._direct_effect_at(_mod_dict(mod_symb, val, self._mod_codes), x_symb)
        return betas, se, llci, ulci

    def _gradient(self, mod_dict, x_symb=None):
        """The gradient of the (relative) direct effect with respect to the coefficients, at moderator values."""
        return eval_expression(self._derivatives[x_symb or self._x_symbs[0]], mod_dict)

    def omnibus_test(self, at=None):
        """
        Joint test that the relative (conditional) direct effects of a multicategorical X are all zero (#17).
        Without moderators this is PROCESS's "Omnibus test of direct effect of X on Y": for OLS the Wald F test
        under the model's covariance estimator with the change in R-squared from dropping the codes, for a
        logistic or negative binomial outcome the likelihood-ratio test. At values of the moderators (`at`, a
        {name: value} dict) it is PROCESS's "Test of equality of conditional means": a Wald F or chi-square.
        :return: one-row DataFrame
        """
        if not self._categorical_x:
            raise ValueError("The omnibus test needs a multicategorical X (mcx).")
        results = self._model.estimation_results
        b, vcv = results["betas"], results["vcv"]
        q = len(self._x_symbs)
        names = {v: k for k, v in self._symb_to_var.items()}
        at = {names.get(k, k): v for k, v in (at or {}).items()}
        mod_dict = _mod_dict(self._moderators_symb, [at.get(m, 0) for m in self._moderators_symb], self._mod_codes)
        if at is None or not at:
            mod_dict = _mod_dict(self._moderators_symb, [0] * len(self._moderators_symb), self._mod_codes)
        gradients = np.array([self._gradient(mod_dict, xs) for xs in self._x_symbs])
        estimate = gradients @ b
        wald = float(estimate @ np.linalg.solve(gradients @ vcv @ gradients.T, estimate))
        if self._is_logit:
            if not at:  # likelihood-ratio test of the codes, as PROCESS prints for a binary outcome
                model = self._model
                keep = [t for t in model._exogvars if not (set(t.split("*")) & set(self._x_symbs))]
                reduced = type(model)(model._data, model._endogvar, keep, model._symb_to_ind, model._symb_to_var, model._options)
                chi2 = 2 * (results["llf"] - reduced.estimation_results["llf"])
                df = len(model._exogvars) - len(keep)
            else:
                chi2, df = wald, q
            return pd.DataFrame([[chi2, df, stats.chi2.sf(chi2, df)]], index=[""], columns=["Chi-sq", "df", "p"])
        df2 = results["df_e"]
        f = wald / q
        p = stats.f.sf(f, q, df2)
        if at:
            return pd.DataFrame([[f, q, df2, p]], index=[""], columns=["F", "df1", "df2", "p"])
        model = self._model
        keep = [i for i, t in enumerate(model._exogvars) if not (set(t.split("*")) & set(self._x_symbs))]
        reduced_betas = fast_OLS(model._endog, model._exog[:, keep])
        sse = np.sum((model._endog - model._exog[:, keep] @ reduced_betas) ** 2)
        sst = np.sum((model._endog - model._endog.mean()) ** 2)
        r2_change = results["R2"] - (1 - sse / sst)
        return pd.DataFrame([[r2_change, f, q, df2, p]], index=[""], columns=["R2-chng", "F", "df1", "df2", "p"])

    def _floodlight_analysis(
            self, mod_symb, modval_range, other_modval_symb, atol=1e-8, rtol=1e-5
    ):
        """
        Conduct a floodlight analysis of the direct effect. Search the critical values of mod_symb,
        at specific value(s) mod_dict of the other moderators.
        :param mod_symb: str
            The symbol of the moderator
        :param modval_range: list of float
            The minimum and maximum values of the moderator.
        :param other_modval_symb: dict
            A mod_symb:mod_value dictionary of values for the other moderators of the direct path.
        """
        modval_min, modval_max = modval_range
        sig_region = find_significance_region(
            self._direct_effect_at,
            mod_symb,
            modval_min,
            modval_max,
            other_modval_symb,
            atol=atol,
            rtol=rtol,
        )
        return sig_region

    def _direct_effect_at(self, mod_dict, x_symb=None):
        """
        Compute the direct effect at specific value(s) of the moderator(s)
        :param mod_dict: dict
            None, or a mod_symb:mod_value dictionary of moderator values.
        :param x_symb: the symbol of X or of one of its codes (#17); the first one by default
        :return: e: scalar
                    Effect at the moderator values
                 se: scalar
                    Standard error
                 llci: scalar
                    Lower level of CI based on normal theory
                 ulci: scalar
                    Upper level of CI based on normal theory
        """
        conf = self._options["conf"]
        b = self._model.estimation_results["betas"]
        vcv = self._model.estimation_results["vcv"]
        grad = self._gradient(mod_dict, x_symb)  # Gradient at level(s) of the moderator(s)
        betas = dot(grad, b)  # Estimate is dot product of gradient and coefficients
        var = dot(
            dot(grad, vcv), np.transpose(grad)
        )  # V(Grad(X)) = Grad(X).V(X).Grad'(X)
        se = np.sqrt(var)
        if self._is_logit:
            crit = z_score(conf)
        else:  # OLS intervals use the t distribution, as PROCESS does (#40)
            crit = t_score(conf, self._model.estimation_results["df_e"])
        llci = betas - (se * crit)
        ulci = betas + (se * crit)
        return betas, se, llci, ulci

    def coeff_summary(self):
        """
        The summary of the direct effect(s): betas, se, t, p-values, etc...
        :return: pd.DataFrame
            A DataFrame of coefficient statistics
        """
        if self._estimation_results:
            symb_to_var = self._symb_to_var
            results = self._estimation_results
            statistics = [
                results["betas"],
                results["se"],
                results["t"],
                results["p"],
                results["llci"],
                results["ulci"],
            ]
            coeffs_rows = np.array([i.flatten() for i in statistics]).T
            if self._is_logit:
                coeffs_columns = ["Effect", "SE", "Z", "p", "LLCI", "ULCI"]
            else:
                coeffs_columns = ["Effect", "SE", "t", "p", "LLCI", "ULCI"]
            combos = [list(i) for i in product(*self._moderators_values)]
            mod_columns = [symb_to_var.get(x, x) for x in self._moderators_symb]
            levels = pd.DataFrame(combos * len(self._x_symbs), columns=mod_columns)
            numbers = pd.DataFrame(coeffs_rows, columns=coeffs_columns)
            df = pd.concat([levels, numbers], axis=1)
            df.index = [""] * len(df)
            if self._categorical_x:  # one block per code, labelled in a first column (#17)
                df.insert(0, "X", np.repeat(self._x_labels, len(combos)))
            return df
        else:
            raise NotImplementedError(
                "The model has not been estimated yet. Please estimate the model first."
            )

    def summary(self):
        """
        Pretty-print the summary with text. Used by Process to display the coefficients in a nicer way.
        :return: string
            The text summary of the model.
        """
        symb_to_var = self._symb_to_var
        prec = self._options["precision"]
        float_format = partial("{:.{prec}f}".format, prec=prec)
        rel = "Relative " if self._categorical_x else ""
        if self._has_mediation:
            if self._has_moderation:
                basestr = (
                    "{rel}onditional direct effect(s) of {x} on {y} at values of the moderator(s):\n\n"
                    "{coeffs}\n".format(
                        rel="Relative c" if self._categorical_x else "C",
                        x=symb_to_var["x"],
                        y=symb_to_var["y"],
                        coeffs=self.coeff_summary().to_string(
                            float_format=float_format
                        ),
                    )
                )
            else:
                basestr = "{rel}irect effect of {x} on {y}:\n\n" "{coeffs}\n".format(
                    rel="Relative d" if self._categorical_x else "D",
                    x=symb_to_var["x"],
                    y=symb_to_var["y"],
                    coeffs=self.coeff_summary().to_string(float_format=float_format),
                )
        elif self.probed:
            basestr = (
                "{rel}onditional effect(s) of {x} on {y} at values of the moderator(s):\n\n"
                "{coeffs}\n".format(
                    rel="Relative c" if self._categorical_x else "C",
                    x=symb_to_var["x"],
                    y=symb_to_var["y"],
                    coeffs=self.coeff_summary().to_string(float_format=float_format),
                )
            )
        else:  # PROCESS 3 and later probe only below intprobe (#87)
            basestr = self._not_probed_text()
        if self._categorical_x and (self._has_mediation or self.probed):
            basestr += "\n" + self.omnibus_text(float_format)
        return basestr

    def omnibus_text(self, float_format):
        """The omnibus test(s) of a multicategorical X as PROCESS prints them (#17)."""
        x, y = self._symb_to_var["x"], self._symb_to_var["y"]
        if not self._has_moderation:
            kind = "likelihood ratio test" if self._is_logit else "test"
            return "Omnibus {kind} of direct effect of {x} on {y}:\n\n{table}\n".format(
                kind=kind, x=x, y=y, table=self.omnibus_test().to_string(float_format=float_format)
            )
        mod_names = [self._symb_to_var.get(m, m) for m in self._moderators_symb]
        rows = []
        for combo in product(*self._moderators_values):
            test = self.omnibus_test(at=dict(zip(mod_names, combo)))
            rows.append(list(combo) + test.iloc[0].tolist())
        table = pd.DataFrame(rows, columns=mod_names + list(test.columns), index=[""] * len(rows))
        return "Test of equality of the conditional means of {y} across the groups of {x}:\n\n{table}\n".format(
            x=x, y=y, table=table.to_string(float_format=float_format)
        )

    def __str__(self):
        return self.summary()


class BaseFloodlightAnalysis:
    def __init__(
            self,
            med_name,
            mod_name,
            sig_regions,
            modval_range,
            other_modval_name,
            precision,
    ):
        """
        A container for a spotlight analysis of the direct effect of the variable X on the outcome Y.
        :param mod_name: string
            The name of the moderator.
        :param sig_regions: list of two lists
            The regions of significance found for the moderator
        :param modval_range: list
            The range of the moderator mod_name.
        :param other_modval_name: dict
            A dictionnary of the values taken by the other moderators.
        :param precision: int
            The decimal precision at which to display the results.
        """
        if med_name is None:
            self._path = "direct"
        else:
            self._path = "indirect"
            self.med_name = med_name
        self.mod_name = mod_name
        self.sig_regions = sig_regions
        self.modval_range = modval_range
        self.other_modval_name = other_modval_name
        self.precision = precision

    def __repr__(self):
        mod_name = self.mod_name
        modval_min, modval_max = self.modval_range
        prec_format = self.precision
        other_modval_name = self.other_modval_name
        sig_regions = self.sig_regions
        effect_label = self._path

        if effect_label == "direct":
            ret_str = """*********************** FLOODLIGHT ANALYSIS OF THE DIRECT EFFECT ***********************\n"""
        else:
            ret_str = """********************** FLOODLIGHT ANALYSIS OF THE INDIRECT EFFECT **********************\n"""
        ret_str += "\n----------------------------------- Analysis Details -----------------------------------\n\n"
        if effect_label == "indirect":
            ret_str += f"Mediator:\n    {self.med_name}\n\n"
        ret_str += f"Focal Moderator:\n    {mod_name}, Range = "
        ret_str += f"[{modval_min:.{prec_format}}, {modval_max:.{prec_format}}]\n\n"

        if other_modval_name:
            ret_str += "Spotlight value for other moderators:\n"
            for k, v in other_modval_name.items():
                if isinstance(v, int):
                    ret_str += f"    {k} = {v}\n"
                else:
                    ret_str += f"    {k} = {v:.{prec_format}}\n"

        ret_str += "\n----------------------------------- Analysis Results -----------------------------------\n\n"

        if sig_regions == [[], []]:
            ret_str += f"The {effect_label} effect is never significant on the range."
        else:
            if sig_regions[0]:
                lb, ub = sig_regions[0]
                ret_str += f"The {effect_label} effect is significantly negative on the interval "
                ret_str += f"[{lb:.{prec_format}}, {ub:.{prec_format}}]\n"
            if sig_regions[1]:
                lb, ub = sig_regions[1]
                ret_str += f"The {effect_label} effect is significantly positive on the interval "
                ret_str += f"[{lb:.{prec_format}}, {ub:.{prec_format}}]\n"
        ret_str += """\n\n****************************************************************************************\n"""
        return ret_str

    def get_significance_regions(self):
        return {"Negative on": self.sig_regions[0], "Positive on": self.sig_regions[1]}


class DirectFloodlightAnalysis(BaseFloodlightAnalysis):
    def __init__(
            self, mod_name, sig_regions, modval_range, other_modval_name, precision
    ):
        """
        A container for a spotlight analysis of the direct effect of the variable X on the outcome Y.
        :param mod_name: string
            The name of the moderator.
        :param sig_regions: list of two lists
            The regions of significance found for the moderator
        :param modval_range: list
            The range of the moderator mod_name.
        :param other_modval_name: dict
            A dictionnary of the values taken by the other moderators.
        :param precision: int
            The decimal precision at which to display the results.
        """
        super().__init__(
            None, mod_name, sig_regions, modval_range, other_modval_name, precision
        )


class IndirectFloodlightAnalysis(BaseFloodlightAnalysis):
    def __init__(
            self,
            med_name,
            mod_name,
            sig_regions,
            modval_range,
            other_modval_name,
            precision,
    ):
        """
        A container for a spotlight analysis of the direct effect of the variable X on the outcome Y.
        :param mod_name: string
            The name of the moderator.
        :param sig_regions: list of two lists
            The regions of significance found for the moderator
        :param modval_range: list
            The range of the moderator mod_name.
        :param other_modval_name: dict
            A dictionnary of the values taken by the other moderators.
        :param precision: int
            The decimal precision at which to display the results.
        """
        super().__init__(
            med_name, mod_name, sig_regions, modval_range, other_modval_name, precision
        )
