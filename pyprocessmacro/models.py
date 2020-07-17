# -*- coding: utf-8 -*-
import warnings
from functools import partial
from itertools import product, combinations

import numpy as np
import pandas as pd
import scipy.stats as stats
from numpy import dot
from numpy.linalg import inv, LinAlgError

from .utils import (
    fast_OLS,
    fast_optimize,
    bootstrap_sampler,
    eval_expression,
    bias_corrected_ci,
    z_score,
    percentile_ci,
    find_significance_region,
)


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
        newparams = np.repeat(0, self._n_vars)
        while iterations < max_iter and np.any(
            np.abs(newparams - oldparams) > tolerance
        ):
            oldparams = newparams
            H = hess(oldparams)
            newparams = oldparams - dot(inv(H), score(oldparams))
            iterations += 1
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
        errortype = "standard" if self._options["hc3"] is False else "HC3"
        if errortype == "standard":
            vcv = np.true_divide(1, n_obs - n_vars) * dot(resid.T, resid) * inv_xx
        elif errortype == "HC0":
            sq_resid = (resid ** 2).squeeze()
            vcv = dot(dot(dot(inv_xx, x.T) * sq_resid, x), inv_xx)
        elif errortype == "HC1":
            sq_resid = (resid ** 2).squeeze()
            vcv = np.true_divide(n_obs, n_obs - n_vars - 1) * dot(
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
                "'HC1', 'HC2', or 'HC3".format(errortype)
            )

        betas = betas.squeeze()
        se = np.sqrt(np.diagonal(vcv)).squeeze()
        t = betas / se
        p = stats.t.sf(np.abs(t), df_e) * 2
        conf = self._options["conf"]
        zscore = z_score(conf)
        R2 = 1 - resid.var() / y.var()
        adjR2 = 1 - (1 - R2) * ((n_obs - 1) / (n_obs - n_vars - 1))
        F = (R2 / df_r) / ((1 - R2) / df_e)
        F_pval = 1 - stats.f.cdf(F, df_r, df_e)
        llci = betas - (se * zscore)
        ulci = betas + (se * zscore)
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
        lmodel = np.exp(llmodel)
        minus2ll = -2 * llmodel

        null_model = NullLogitModel(self._endog, self._options)
        betas_null = null_model._optimize()
        llnull = null_model._loglike(betas_null)
        lnull = np.exp(llnull)

        d = 2 * (llmodel - llnull)
        pvalue = stats.chi2.sf(d, self._n_vars - 1)
        mcfadden = 1 - llmodel / llnull
        coxsnell = 1 - (lnull / lmodel) ** (2 / self._n_obs)
        nagelkerke = coxsnell / (1 - lnull ** (2 / self._n_obs))
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


class ParallelMediationModel(object):
    """
    A class describing a parallel mediation model between an endogenous variable Y, one or several mediators M, and a
    set of exogenous predictors for the endogenous variable and the mediators.
    """

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
        if self._options["logit"]:
            max_iter = self._options["iterate"]
            tolerance = self._options["convergence"]
            self._compute_betas_y = partial(
                fast_optimize,
                n_obs=self._n_obs,
                n_vars=len(self._exog_inds_y),
                max_iter=max_iter,
                tolerance=tolerance,
            )
        else:
            self._compute_betas_y = fast_OLS

        self._true_betas_y, self._true_betas_m = self._estimate_true_params()
        self._boot_betas_y, self._boot_betas_m, self._n_fail_samples = (
            self._estimate_bootstrapped_params()
        )

        self._base_derivs = self._gen_derivatives()

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
        :return: A tuple of (true_betas_y, true_betas_m)
            * true_betas_y is a matrix of size n_boots x n_params_y
            * true_betas_m is a list of matrices of size n_boots x n_params_y
        """
        n_boots = self._options["boot"]
        seed = self._options["seed"]
        boot_betas_y = np.empty((n_boots, len(self._exog_terms_y)))
        boot_betas_m = np.empty((self._n_meds, n_boots, len(self._exog_terms_m)))
        n_fail_samples = 0
        boot_ind = 0
        sampler = bootstrap_sampler(self._n_obs, seed)
        while boot_ind < n_boots:
            ind = next(sampler)
            data_boot = self._data[ind, :]
            y_e = data_boot[:, self._ind_y]
            y_x = data_boot[:, self._exog_inds_y]
            try:
                y_b = self._compute_betas_y(y_e, y_x)
                m_x = data_boot[:, self._exog_inds_m]
                boot_betas_y[boot_ind] = y_b
                for j, m_ind in enumerate(self._inds_m):
                    m_e = data_boot[:, m_ind]
                    m_b = self._compute_betas_m(m_e, m_x)
                    boot_betas_m[j][boot_ind] = m_b
                boot_ind += 1
            except LinAlgError:  # Hessian (Logit) or X'X (OLS) cannot be inverted
                n_fail_samples += 1

        return boot_betas_y, boot_betas_m, n_fail_samples

    def _gen_derivatives(self):
        """
        Generate the list of symbolic derivatives for the indirect path(s) from X to Y. The derivative of the path from
        X to M is taken with respect to X, and the derivative of the path to Y is taken with respect to M.

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
            if var == "x":
                x_to_m[j] = [1 if var in term else 0 for term in exog_terms_m]
            else:
                x_to_m[j] = [var if var in term else 1 for term in exog_terms_m]
        derivs["x_to_m"] = x_to_m.T

        list_m_to_y = []
        for i in range(self._n_meds):  # For all the mediators...
            # ... derivate the path from M to Y (unique to each mediator)
            vars_y = self._vars_y
            exog_terms_y = self._exog_terms_y
            m_to_y = np.empty((len(vars_y), len(exog_terms_y)), dtype="object")
            for j, var in enumerate(vars_y):
                if var == "m{}".format(i + 1):
                    m_to_y[j] = [1 if var in term else 0 for term in exog_terms_y]
                else:
                    m_to_y[j] = [var if var in term else 1 for term in exog_terms_y]
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
            mod_dict = {k: v for k, v in zip(mod_symb, vals)}
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
        n_boots = self._options["boot"]
        (mod,) = self._moderators_symb  # Only one moderator

        # noinspection PyTypeChecker
        dict_baseline = dict([[mod, 0]])  # Only moderator at 0
        e_baseline, be_baseline = (
            np.empty(self._n_meds),
            np.empty((self._n_meds, n_boots)),
        )

        # noinspection PyTypeChecker
        dict_effect = dict([[mod, 1]])  # Only moderator at 1
        e_effect, be_effect = np.empty(self._n_meds), np.empty((self._n_meds, n_boots))

        effects, se, llci, ulci = np.empty((4, self._n_meds))

        for i in range(self._n_meds):  # ... For all the mediators
            e_baseline[i], be_baseline[i], *_ = self._indirect_effect_at(
                i, dict_baseline
            )
            e_effect[i], be_effect[i], *_ = self._indirect_effect_at(i, dict_effect)

            e_mm = e_effect[i] - e_baseline[i]  # Moderator at 1 vs. Moderator at 0
            be_mm = be_effect[i] - be_baseline[i]

            effects[i] = e_mm
            se[i] = be_mm.std(ddof=1)
            if self._options["percent"]:
                llci[i], ulci[i] = percentile_ci(be_mm, conf)
            else:
                llci[i], ulci[i] = bias_corrected_ci(e_mm, be_mm, conf)

        statistics = [i.flatten() for i in [effects, se, llci, ulci]]

        return {k: v for k, v in zip(["effect", "se", "llci", "ulci"], statistics)}

    def _PMM_index(self):
        """
        The Partial Moderated Mediation (PMM) index is only computed when exactly two moderators are present on the
        mediation path.
        It represents the marginal impact of one moderator (i.e. the impact of an increase in one unit for this
        moderator on the indirect effect), conditional on a value of zero for the other moderator.
        """
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

        rows_levels = np.array([i for i in product(*values)])
        cols_levels = ["Mediator"] + [
            symb_to_var.get(x, x) for x in self._moderators_symb
        ]

        rows = np.concatenate([rows_levels, rows_stats], axis=1)
        cols = cols_levels + cols_stats
        df = pd.DataFrame(rows, columns=cols, index=[""] * rows.shape[0])
        # noinspection PyTypeChecker
        return df.apply(pd.to_numeric, args=["ignore"])

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
        rows_levels = np.array(rows_levels).reshape(-1, 1)

        rows = np.concatenate([rows_levels, rows_stats], axis=1)
        cols = ["", "Effect", "Boot SE", "BootLLCI", "BootULCI"]
        df = pd.DataFrame(rows, columns=cols, index=[""] * rows.shape[0])
        # noinspection PyTypeChecker
        return df.apply(pd.to_numeric, args=["ignore"])

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

        mod_names = [[symb_to_var.get(i, i) for i in self._moderators_symb]]
        med_names = [
            [
                symb_to_var.get("m{}".format(i + 1), "m{}".format(i + 1))
                for i in range(self._n_meds)
            ]
        ]
        values = mod_names + med_names
        rows_levels = np.array([i for i in product(*values)])
        cols_levels = ["Moderator", "Mediator"]

        rows = np.concatenate([rows_levels, rows_stats], axis=1)
        cols = cols_levels + cols_stats
        df = pd.DataFrame(rows, columns=cols, index=[""] * rows.shape[0])
        # noinspection PyTypeChecker
        return df.apply(pd.to_numeric, args=["ignore"])

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
        values = mod_names + med_names
        rows_levels = np.array([i for i in product(*values)])
        cols_levels = ["Moderator", "Mediator"]

        rows = np.concatenate([rows_levels, rows_stats], axis=1)
        cols = cols_levels + cols_stats
        df = pd.DataFrame(rows, columns=cols, index=[""] * rows.shape[0])
        # noinspection PyTypeChecker
        return df.apply(pd.to_numeric, args=["ignore"])

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
        rows_modname = np.reshape(rows_modname, (-1, 1))

        rows_medname = np.concatenate(
            [
                np.repeat(med_names, len(mod1_values)),
                np.repeat(med_names, len(mod2_values)),
            ]
        )
        rows_medname = np.reshape(rows_medname, (-1, 1))

        rows_modvalues = np.concatenate(
            [np.tile(mod1_values, self._n_meds), np.tile(mod2_values, self._n_meds)]
        )
        rows_modvalues = np.reshape(rows_modvalues, (-1, 1))

        cols_levels = ["Focal Mod", "Mediator", "Other Mod At"]
        rows_levels = np.concatenate(
            [rows_modname, rows_medname, rows_modvalues], axis=1
        )
        rows = np.concatenate([rows_levels, rows_stats], axis=1)
        cols = cols_levels + cols_stats
        df = pd.DataFrame(rows, columns=cols, index=[""] * rows.shape[0])

        # noinspection PyTypeChecker
        return df.apply(pd.to_numeric, args=["ignore"])

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
        rows_levels = np.array([i for i in product(*med_names)])
        cols_levels = ["Mediator"]

        rows = np.concatenate([rows_levels, rows_stats], axis=1)
        cols = cols_levels + cols_stats
        df = pd.DataFrame(rows, columns=cols, index=[""] * rows.shape[0])
        # noinspection PyTypeChecker
        return df.apply(pd.to_numeric, args=["ignore"])

    def MM_index_summary(self):
        if "MMM" in self._analysis_list:
            return self._MM_index_wrapper()
        else:
            raise NotImplementedError(
                "This model does not report the Moderated Mediation index."
            )

    def MMM_index_summary(self):
        if "MMM" in self._analysis_list:
            return self._MMM_index_wrapper()
        else:
            raise NotImplementedError(
                "This model does not report the Moderated Moderated Mediation index."
            )

    def PMM_index_summary(self):
        if "PMM" in self._analysis_list:
            return self._PMM_index_wrapper()
        else:
            raise NotImplementedError(
                "This model does not report the Partial Moderated Mediation index."
            )

    def CMM_index_summary(self):
        if "CMM" in self._analysis_list:
            return self._CMM_index_wrapper()
        else:
            raise NotImplementedError(
                "This model does not report the Conditional Moderated Mediation index."
            )

    def coeff_summary(self):
        """
        Get the summary of the indirect effect(s).
        :return: The appropriate moderated/unmoderated effect(s).
        """
        return (
            self._cond_ind_effects_wrapper()
            if self._has_moderation
            else self._simple_ind_effects_wrapper()
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
        if self._has_moderation:
            basestr = (
                "Conditional indirect effect(s) of {x} on {y} at values of the moderator(s):\n\n"
                "{coeffs}\n\n".format(
                    x=symb_to_var["x"],
                    y=symb_to_var["y"],
                    coeffs=self.coeff_summary().to_string(float_format=float_format),
                )
            )
        else:
            basestr = "Indirect effect of {x} on {y}:\n\n" "{coeffs}\n\n".format(
                x=symb_to_var["x"],
                y=symb_to_var["y"],
                coeffs=self.coeff_summary().to_string(float_format=float_format),
            )
        for a in self._analysis_list:
            name, get_results = analysis_func[a]
            results = get_results()
            basestr += (
                "**************** INDEX OF {name} ******************\n\n"
                "{results}\n\n".format(
                    name=name, results=results.to_string(float_format=float_format)
                )
            )
        return basestr

    def __str__(self):
        return self.summary()


class DirectEffectModel(object):
    def __init__(
        self, model, mod_symb, spot_values, has_mediation, symb_to_var, options=None
    ):
        """
        A container for the direct effect of the variable X on the outcome Y. If the model includes one or several
        moderators of X, this container returns the conditional direct effects.
        :param model: process.OutcomeModel
            The OutcomeModel object of the outcome Y.
        :param mod_symb: list of string
            The symbols of the moderators of the direct effect.
        :param symb_to_var: dict of string
            The dictionary mapping each symbol to a variable name.
        :param options: dict
            The options of the model.
        """
        self._model = model
        self._is_logit = isinstance(model, LogitOutcomeModel)
        self._symb_to_var = symb_to_var
        self._derivative = self._model._derivative
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

    def _estimate(self):
        """
        Estimates the direct effect of X on Y, and return the results into as a dictionary.
        :return: dict
            A dictionary of parameters and model estimates.
        """
        mod_values = [i for i in product(*self._moderators_values)]
        mod_symb = self._moderators_symb
        betas, se, llci, ulci = self._get_conditional_direct_effects(
            mod_symb, mod_values
        )
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

    def _get_conditional_direct_effects(self, mod_symb, mod_values):
        """
        Estimates the conditional direct effects of X on Y, at different values of the moderator(s)
        :param mod_symb: list of string
            A list of moderator symbols
        :param mod_values: array of int/float
            A list of lists of spotlight values for each moderator.
        :return:
        """
        betas, se, llci, ulci = np.zeros((4, len(mod_values)))
        for i, val in enumerate(
            mod_values
        ):  # All possible products of level(s) of moderator(s)
            mod_dict = {n: v for n, v in zip(mod_symb, val)}
            betas[i], se[i], llci[i], ulci[i] = self._direct_effect_at(mod_dict)
        return betas, se, llci, ulci

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

    def _direct_effect_at(self, mod_dict):
        """
        Compute the direct effect at specific value(s) of the moderator(s)
        :param mod_dict: dict
            None, or a mod_symb:mod_value dictionary of moderator values.
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
        deriv = self._derivative
        grad = eval_expression(
            deriv, mod_dict
        )  # Gradient at level(s) of the moderator(s)
        betas = dot(grad, b)  # Estimate is dot product of gradient and coefficients
        var = dot(
            dot(grad, vcv), np.transpose(grad)
        )  # V(Grad(X)) = Grad(X).V(X).Grad'(X)
        se = np.sqrt(var)
        zscore = z_score(conf)
        llci = betas - (se * zscore)
        ulci = betas + (se * zscore)
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
            mod_rows = np.array([i for i in product(*self._moderators_values)])
            mod_columns = [symb_to_var.get(x, x) for x in self._moderators_symb]
            rows = np.concatenate([mod_rows, coeffs_rows], axis=1)
            columns = mod_columns + coeffs_columns
            df = pd.DataFrame(rows, columns=columns, index=[""] * rows.shape[0])
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
        if self._has_mediation:
            if self._has_moderation:
                basestr = (
                    "Conditional direct effect(s) of {x} on {y} at values of the moderator(s):\n\n"
                    "{coeffs}\n".format(
                        x=symb_to_var["x"],
                        y=symb_to_var["y"],
                        coeffs=self.coeff_summary().to_string(
                            float_format=float_format
                        ),
                    )
                )
            else:
                basestr = "Direct effect of {x} on {y}:\n\n" "{coeffs}\n".format(
                    x=symb_to_var["x"],
                    y=symb_to_var["y"],
                    coeffs=self.coeff_summary().to_string(float_format=float_format),
                )
        else:
            basestr = (
                "Conditional effect(s) of {x} on {y} at values of the moderator(s):\n\n"
                "{coeffs}\n".format(
                    x=symb_to_var["x"],
                    y=symb_to_var["y"],
                    coeffs=self.coeff_summary().to_string(float_format=float_format),
                )
            )
        return basestr

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
        print(self.__class__.__bases__)
