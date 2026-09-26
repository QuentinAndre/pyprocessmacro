# -*- coding: utf-8 -*-
"""
Negative binomial regression for a count outcome Y (#25): the family="negbin" option.

A PyProcessMacro extension; PROCESS estimates continuous outcomes by OLS and binary outcomes by logistic
regression only. The model is the NB2 parametrization: y ~ NegBin(mu, alpha) with mu = exp(X beta) and
Var(y) = mu + alpha mu^2, estimated by maximum likelihood over theta = (beta, log alpha) with a damped Newton
method. The estimator is batched over bootstrap resamples the way the logistic fits are (#68); a single fit
that Newton cannot finish falls back to scipy's L-BFGS-B.
"""
import numpy as np
from numpy.linalg import LinAlgError
from scipy import optimize, special

from .utils import ConvergenceError

# Bounds on log(alpha). Below exp(-15) the model is Poisson to working precision and the likelihood is flat in
# alpha, so the bound keeps the Newton steps finite when the counts are not overdispersed.
LOG_ALPHA_MIN, LOG_ALPHA_MAX = -15.0, 10.0


def loglike(endog, exog, params):
    """
    Sum of the log-likelihood of a batch of regressions.
    :param endog: (c, n) counts
    :param exog: (c, n, k) design matrices
    :param params: (c, k + 1) parameters, beta then log alpha
    :return: (c,) log-likelihoods
    """
    linear = np.einsum("cnk,ck->cn", exog, params[:, :-1])
    log_r = -params[:, -1:]  # r = 1 / alpha, (c, 1)
    r = np.exp(log_r)
    log_r_mu = np.logaddexp(log_r, linear)  # log(r + mu)
    ll = (
        special.gammaln(endog + r) - special.gammaln(r) - special.gammaln(endog + 1)
        + r * (log_r - log_r_mu) + endog * (linear - log_r_mu)
    )
    return ll.sum(axis=1)


def score_hessian(endog, exog, params):
    """
    Gradient and Hessian of the log-likelihood in (beta, log alpha) for a batch.
    :return: (c, k + 1) gradients and (c, k + 1, k + 1) Hessians
    """
    count, n_obs, k = exog.shape
    linear = np.einsum("cnk,ck->cn", exog, params[:, :-1])
    log_r = -params[:, -1:]
    r = np.exp(log_r)  # (c, 1)
    mu = np.exp(linear)
    log_r_mu = np.logaddexp(log_r, linear)
    r_mu = np.exp(log_r_mu)  # r + mu
    w = np.exp(log_r - log_r_mu)  # r / (r + mu)
    resid = endog - mu
    # s = d loglike / d r per observation, and its derivative in r
    s = special.digamma(endog + r) - special.digamma(r) + (log_r - log_r_mu) + (mu - endog) / r_mu
    s_prime = (
        special.polygamma(1, endog + r) - special.polygamma(1, r) + mu / (r * r_mu) - (mu - endog) / r_mu ** 2
    )
    g_beta = np.einsum("cn,cnk->ck", w * resid, exog)
    g_eta = -(r * s).sum(axis=1)  # d r / d log alpha = -r
    h_bb = -np.einsum("cnk,cn,cnj->ckj", exog, w * mu * (r + endog) / r_mu, exog)
    h_be = -np.einsum("cn,cnk->ck", w * mu * resid / r_mu, exog)
    h_ee = (r * s + r ** 2 * s_prime).sum(axis=1)
    grad = np.concatenate([g_beta, g_eta[:, None]], axis=1)
    hess = np.empty((count, k + 1, k + 1))
    hess[:, :k, :k] = h_bb
    hess[:, :k, k] = h_be
    hess[:, k, :k] = h_be
    hess[:, k, k] = h_ee
    return grad, hess


def _start(endog, exog):
    """Intercept at the log of the mean count, other slopes at zero, alpha from the method of moments."""
    count, n_obs, k = exog.shape
    params = np.zeros((count, k + 1))
    mean = endog.mean(axis=1)
    var = endog.var(axis=1, ddof=1) if n_obs > 1 else mean
    constant = [j for j in range(k) if np.all(exog[:, :, j] == 1)]
    if constant:
        params[:, constant[0]] = np.log(np.maximum(mean, 1e-8))
    alpha = (var - mean) / np.maximum(mean, 1e-8) ** 2
    params[:, k] = np.log(np.clip(alpha, 1e-2, 1e2))
    return params


def _clip(params):
    params[:, -1] = np.clip(params[:, -1], LOG_ALPHA_MIN, LOG_ALPHA_MAX)
    return params


def batch_fit(endog, exog, max_iter, tolerance):
    """
    Damped Newton for a batch: endog (c, n), exog (c, n, k) -> params (c, k + 1) as beta then log alpha, and
    failed (c,). Samples leave the active set when they converge; a sample whose parameters stop being finite,
    whose step cannot raise the likelihood, or that has not converged after max_iter updates is marked failed.
    A singular Hessian raises LinAlgError for the caller to fall back to one-by-one fits.
    """
    count = exog.shape[0]
    params = _start(endog, exog)
    ll = loglike(endog, exog, params)
    failed = np.zeros(count, dtype=bool)
    active = np.ones(count, dtype=bool)
    for _ in range(max_iter):
        idx = np.flatnonzero(active)
        if idx.size == 0:
            break
        x, y, p = exog[idx], endog[idx], params[idx]
        grad, hess = score_hessian(y, x, p)
        step = -np.linalg.solve(hess, grad[..., None])[..., 0]
        downhill = np.einsum("ck,ck->c", step, grad) <= 0  # Newton points downhill where ll is not concave
        step[downhill] = grad[downhill]
        scale = np.ones(idx.size)
        new = _clip(p + step)
        ll_new = loglike(y, x, new)
        for _ in range(20):  # step halving until the likelihood does not decrease
            worse = ~np.isfinite(ll_new) | (ll_new < ll[idx] - 1e-10)
            if not worse.any():
                break
            scale[worse] /= 2
            new[worse] = _clip(p[worse] + scale[worse, None] * step[worse])
            ll_new[worse] = loglike(y[worse], x[worse], new[worse])
        worse = ~np.isfinite(ll_new) | (ll_new < ll[idx] - 1e-10)
        accept = ~worse & np.isfinite(new).all(axis=1)
        params[idx[accept]] = new[accept]
        ll[idx[accept]] = ll_new[accept]
        small = np.abs(scale[:, None] * step).max(axis=1) <= tolerance
        converged = (accept & (np.abs(new - p).max(axis=1) <= tolerance)) | (~accept & small)
        bad = ~accept & ~small
        failed[idx[bad]] = True
        active[idx[bad | converged]] = False
    failed |= active  # still active after max_iter updates: not converged
    return params, failed


def fit(endog, exog, max_iter=10000, tolerance=1e-10):
    """
    Parameters (beta then log alpha) of one negative binomial regression.
    :raises ConvergenceError: when neither Newton nor L-BFGS-B converges
    """
    endog = np.asarray(endog, dtype=float)
    exog = np.asarray(exog, dtype=float)
    try:
        params, failed = batch_fit(endog[None], exog[None], max_iter, tolerance)
        if not failed[0]:
            return params[0]
    except LinAlgError:
        pass
    return _fit_scipy(endog, exog)


def fit_betas(endog, exog, max_iter=10000, tolerance=1e-10):
    """The regression coefficients only, for the mediation machinery that reads the Y equation's betas."""
    return fit(endog, exog, max_iter, tolerance)[:-1]


def _fit_scipy(endog, exog):
    y, x = endog[None], exog[None]

    def objective(theta):
        t = theta[None]
        grad, _ = score_hessian(y, x, t)
        return -loglike(y, x, t)[0], -grad[0]

    bounds = [(None, None)] * exog.shape[1] + [(LOG_ALPHA_MIN, LOG_ALPHA_MAX)]
    result = optimize.minimize(objective, _start(y, x)[0], jac=True, method="L-BFGS-B", bounds=bounds)
    if not result.success or not np.all(np.isfinite(result.x)):
        raise ConvergenceError(f"The negative binomial regression did not converge: {result.message}")
    return result.x


def hessian(endog, exog, params):
    """Hessian of one regression's log-likelihood in (beta, log alpha)."""
    return score_hessian(
        np.asarray(endog, dtype=float)[None], np.asarray(exog, dtype=float)[None], np.asarray(params, dtype=float)[None]
    )[1][0]


def loglike_sum(endog, exog, params):
    """Log-likelihood of one regression."""
    return float(loglike(
        np.asarray(endog, dtype=float)[None], np.asarray(exog, dtype=float)[None], np.asarray(params, dtype=float)[None]
    )[0])


def check_counts(values, name):
    """Raise ValueError unless the values are non-negative integers with at least one positive count."""
    values = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(values)) or np.any(values < 0) or np.any(values != np.round(values)):
        raise ValueError(
            f"family='negbin' needs a count outcome: '{name}' must hold non-negative integers."
        )
    if not np.any(values > 0):
        raise ValueError(f"family='negbin' needs a count outcome with at least one positive count in '{name}'.")
