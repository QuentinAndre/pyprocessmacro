from numpy.linalg import inv, LinAlgError
from numpy import dot
from scipy.stats import norm
import numpy as np
import warnings
import matplotlib.pyplot as plt


def z_score(conf):
    """
    :param conf: The level of confidence desired
    :return: The Z-score corresponding to the level of confidence desired.
    """
    return norm.ppf((100 - (100 - conf) / 2) / 100)


def bias_corrected_ci(true_coeff, samples, conf=95):
    """
    Return the bias-corrected bootstrap confidence interval, using the method from the book.
    :param true_coeff: The estimates in the original sample
    :param samples: The bootstrapped estimates
    :param conf: The level of the desired confidence interval
    :return: The bias-corrected LLCI and ULCI for the bootstrapped estimates.
    """
    ptilde = (samples < true_coeff).mean()
    Z = norm.ppf(ptilde)
    Zci = z_score(conf)
    Zlow, Zhigh = -Zci + 2 * Z, Zci + 2 * Z
    plow, phigh = norm._cdf(Zlow), norm._cdf(Zhigh)
    llci = np.percentile(samples, plow * 100, interpolation='lower')
    ulci = np.percentile(samples, phigh * 100, interpolation='higher')
    return llci, ulci


def percentile_ci(samples, conf=95):
    """
    Based on an array of values, returns the lower and upper percentile bound for a desired level of confidence
    :param samples: A NxK array of samples
    :param conf: A desired level of confidence
    :return: A 2xK array corresponding the lower and upper percentile bounds for K estimates.
    """
    lower = (100 - conf) / 2
    upper = 100 - lower
    return np.percentile(samples, [lower, upper])


def fast_OLS(endog, exog):
    """
    A simple function for (X'X)^(-1)X'Y
    :return: A K-length array of estimated coefficients.
    """
    try:
        return dot(dot(inv(dot(exog.T, exog)), exog.T), endog).squeeze()
    except LinAlgError:
        raise LinAlgError


def logit_cdf(X):
    """
    The CDF of the logistic function.
    :param X: A scalar
    :return: A scalar
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
    :param endog: An Nx1 vector of endogenous predictions
    :param exog: An NxK vector of exogenous predictors
    :param params: A Kx1 vector of parameters for the predictors
    :param n_obs: The number of observations N
    :return: The score, a Kx1 vector, evaluated at `params'
    """
    return dot(endog - logit_cdf(dot(exog, params)), exog) / n_obs


def logit_hessian(exog, params, n_obs):
    """
    The hessian of the logistic function.
    :param exog: An NxK vector of exogenous predictors
    :param params: A Kx1 vector of parameters for the predictors
    :param n_obs: The number of observations N
    :return: The Hessian, a KxK matrix, evaluated at `params'
    """
    L = logit_cdf(np.dot(exog, params))
    return -dot(L * (1 - L) * exog.T, exog) / n_obs


def fast_optimize(endog, exog, n_obs=0, n_vars=0, max_iter=10000, tolerance=1e-10):
    """
    A convenience function for the Newton-Raphson method to evaluate a logistic model.
    :param endog: An Nx1 vector of endogenous predictions
    :param exog: An NxK vector of exogenous predictors
    :param n_obs: The number of observations N
    :param n_vars: The number of exogenous predictors K
    :param max_iter: The maximum number of iterations
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
            newparams = oldparams - dot(inv(H), logit_score(endog, exog, oldparams, n_obs))
        except LinAlgError:
            raise LinAlgError
        iterations += 1
    return newparams


def bootstrap_sampler(n_obs, seed=None):
    """
    A generator of bootstrapped indices. Samples with repetition a list of indices.
    :return: A bootstrapped sample of size NxK
    """
    seeder = np.random.RandomState(seed)
    seeder.seed(seed)
    while True:
        yield seeder.randint(n_obs, size=n_obs)


def eigvals(exog):
    """
    Return the eigenvalues of a matrix of engogeneous predictors.
    :param exog: A NxK matrix of exogenous predictors.
    :return: A Kx1 vector of eigenvalues, sorted in decreasing order of size.
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
            evaled_term = np.array([values.get(elem, 0) if isinstance(elem, str) else elem for elem in term])
        else:
            evaled_term = np.array(
                [0 if isinstance(elem, str) else elem for elem in term])  # All variables at 0
        evaled_expr[i] = np.product(evaled_term.astype(float))  # Gradient is the product of values
    return evaled_expr

def plot_errorbars(x, y, yerrlow, yerrhigh, plot_kws=None, err_kws=None, *args, **kwargs):
    yerr = [yerrlow, yerrhigh]
    err_kws = {**kwargs, **err_kws, 'marker': "", 'fmt':'none', 'label':'', "zorder":3}
    plot_kws = {**kwargs, **plot_kws}
    plt.plot(x, y, **plot_kws)
    plt.errorbar(x, y, yerr, **err_kws)
    return None

def plot_errorbands(x, y, llci, ulci, plot_kws=None, err_kws=None, *args, **kwargs):
    err_kws = {**kwargs, **err_kws, 'label':''}
    plot_kws = {**kwargs, **plot_kws}
    plt.plot(x, y, **plot_kws)
    plt.fill_between(x, llci, ulci, **err_kws)
    return None