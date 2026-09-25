# -*- coding: utf-8 -*-
"""
Vectorized bootstrap of the outcome and mediator equations (#68).

Resample indices are still drawn one sample at a time from the same generator as before, so the random
stream is identical to the sequential implementation; only the estimation is batched. Chunks of resamples
are fitted with stacked linear algebra, and a chunk that hits a singular system falls back to fitting its
samples one by one so that failures are detected per sample exactly as before.
"""
import numpy as np
from numpy.linalg import LinAlgError

from .utils import ConvergenceError, bootstrap_sampler, fast_OLS, fast_optimize

# Upper bound on the number of floating-point values held by one chunk of resampled data.
CHUNK_ELEMENTS = 10_000_000


class BootstrapSpec:
    """Which columns of the data array are which, and how the outcome equation is estimated."""

    def __init__(self, ind_y, exog_inds_y, inds_m, exog_inds_m, logit, max_iter, tolerance):
        self.ind_y = ind_y
        self.exog_inds_y = list(exog_inds_y)
        self.inds_m = list(inds_m)
        self.exog_inds_m = list(exog_inds_m)
        self.logit = bool(logit)
        self.max_iter = max_iter
        self.tolerance = tolerance


def bootstrap_parameters(data, spec, n_boots, seed, chunk_size=None):
    """
    Estimate the outcome and mediator equations on n_boots resamples of the data.

    :param data: (n_obs x n_cols) array of the analysis data
    :param spec: BootstrapSpec
    :param n_boots: number of successful resamples wanted
    :param seed: seed of the resampler (None for fresh entropy)
    :param chunk_size: resamples fitted per batch; None picks one from the data size
    :return: (betas_y, betas_m, n_fail): (n_boots x k_y) array, (n_meds x n_boots x k_m) array, and the
        number of resamples discarded because an equation could not be estimated on them.
    """
    data = np.asarray(data, dtype=float)
    n_obs, n_cols = data.shape
    if chunk_size is None:
        chunk_size = int(max(1, min(n_boots, CHUNK_ELEMENTS // max(1, n_obs * n_cols))))
    sampler = bootstrap_sampler(n_obs, seed)
    k_y, k_m, n_meds = len(spec.exog_inds_y), len(spec.exog_inds_m), len(spec.inds_m)
    betas_y = np.empty((n_boots, k_y))
    betas_m = np.empty((n_meds, n_boots, k_m))
    filled, n_fail, max_failures = 0, 0, n_boots
    while filled < n_boots:
        count = min(chunk_size, n_boots - filled)
        indices = np.stack([next(sampler) for _ in range(count)])  # one draw per sample: same stream as before
        chunk_y, chunk_m, failed = _fit_chunk(data[indices], spec)
        ok = ~failed
        n_ok = int(ok.sum())
        betas_y[filled:filled + n_ok] = chunk_y[ok]
        betas_m[:, filled:filled + n_ok] = chunk_m[:, ok]
        filled += n_ok
        n_fail += int(failed.sum())
        if n_fail > max_failures:
            raise RuntimeError(
                f"{n_fail} bootstrap samples failed to estimate before {n_boots} succeeded. "
                "The model is probably not estimable on resamples of this data (check for separation, "
                "collinearity, or a very small sample)."
            )
    return betas_y, betas_m, n_fail


def _fit_chunk(chunk, spec):
    """
    Fit every resample of a chunk. Returns (betas_y (c x k_y), betas_m (n_meds x c x k_m), failed (c,)).
    A singular system anywhere in the batch sends the whole chunk to the one-by-one path.
    """
    y = chunk[:, :, spec.ind_y]
    exog_y = chunk[:, :, spec.exog_inds_y]
    exog_m = chunk[:, :, spec.exog_inds_m]
    endog_m = chunk[:, :, spec.inds_m]
    try:
        if spec.logit:
            betas_y, failed = _batch_logit(y, exog_y, spec.max_iter, spec.tolerance)
        else:
            betas_y = _batch_ols(y[..., None], exog_y)[..., 0]
            failed = ~np.isfinite(betas_y).all(axis=1)
        betas_m = _batch_ols(endog_m, exog_m)  # (c, k_m, n_meds)
    except LinAlgError:
        return _fit_one_by_one(chunk, spec)
    failed = failed | ~np.isfinite(betas_m).all(axis=(1, 2))
    return betas_y, np.transpose(betas_m, (2, 0, 1)), failed


def _fit_one_by_one(chunk, spec):
    """The sequential path, used when a batch contains a singular system: same estimators as 1.x."""
    count = chunk.shape[0]
    betas_y = np.zeros((count, len(spec.exog_inds_y)))
    betas_m = np.zeros((len(spec.inds_m), count, len(spec.exog_inds_m)))
    failed = np.zeros(count, dtype=bool)
    for i in range(count):
        sample = chunk[i]
        try:
            if spec.logit:
                betas_y[i] = fast_optimize(
                    sample[:, spec.ind_y], sample[:, spec.exog_inds_y], n_obs=sample.shape[0],
                    n_vars=len(spec.exog_inds_y), max_iter=spec.max_iter, tolerance=spec.tolerance,
                )
            else:
                betas_y[i] = fast_OLS(sample[:, spec.ind_y], sample[:, spec.exog_inds_y])
            for j, ind in enumerate(spec.inds_m):
                betas_m[j, i] = fast_OLS(sample[:, ind], sample[:, spec.exog_inds_m])
        except (LinAlgError, ConvergenceError):
            failed[i] = True
    return betas_y, betas_m, failed


def _batch_ols(endog, exog):
    """(X'X)^-1 X'Y for a batch: endog (c x n x r), exog (c x n x k) -> (c x k x r)."""
    xtx = np.einsum("cnk,cnj->ckj", exog, exog)
    xty = np.einsum("cnk,cnr->ckr", exog, endog)
    return np.linalg.solve(xtx, xty)


def _logit_cdf(z):
    out = np.empty_like(z)
    positive = z > 0
    out[positive] = 1 / (1 + np.exp(-z[positive]))
    expz = np.exp(z[~positive])
    out[~positive] = expz / (1 + expz)
    return out


def _batch_logit(endog, exog, max_iter, tolerance):
    """
    Newton-Raphson for a batch of logistic regressions: endog (c x n), exog (c x n x k) -> (c x k), failed (c,).
    Samples leave the active set when they converge; a sample whose parameters stop being finite, or that has
    not converged after max_iter updates, is marked failed.
    """
    count, n_obs, k = exog.shape
    params = np.zeros((count, k))
    failed = np.zeros(count, dtype=bool)
    active = np.ones(count, dtype=bool)
    for _ in range(max_iter):
        idx = np.flatnonzero(active)
        if idx.size == 0:
            break
        x, y, p = exog[idx], endog[idx], params[idx]
        fitted = _logit_cdf(np.einsum("cnk,ck->cn", x, p))
        score = np.einsum("cn,cnk->ck", y - fitted, x) / n_obs
        hessian = -np.einsum("cnk,cn,cnj->ckj", x, fitted * (1 - fitted), x) / n_obs
        new = p - np.linalg.solve(hessian, score[..., None])[..., 0]
        finite = np.isfinite(new).all(axis=1)
        converged = finite & (np.abs(new - p).max(axis=1) <= tolerance)
        params[idx] = np.where(finite[:, None], new, p)
        failed[idx[~finite]] = True
        active[idx[~finite | converged]] = False
    failed |= active  # still active after max_iter updates: not converged
    return params, failed


FAILURE_MESSAGE = (
    "{n_fail} bootstrap samples failed to estimate before {n_boots} succeeded. The model is probably not "
    "estimable on resamples of this data (check for separation, collinearity, or a very small sample)."
)


def bootstrap_equations(data, equations, n_boots, seed, max_iter=10000, tolerance=1e-10, chunk_size=None):
    """
    Estimate several equations, each with its own design matrix, on n_boots resamples (serial mediation).

    :param equations: list of (endog_ind, exog_inds, logit) triples
    :return: (list of (n_boots x k_i) arrays, one per equation, n_fail)
    """
    data = np.asarray(data, dtype=float)
    n_obs, n_cols = data.shape
    if chunk_size is None:
        chunk_size = int(max(1, min(n_boots, CHUNK_ELEMENTS // max(1, n_obs * n_cols))))
    sampler = bootstrap_sampler(n_obs, seed)
    betas = [np.empty((n_boots, len(exog_inds))) for _, exog_inds, _ in equations]
    filled, n_fail, max_failures = 0, 0, n_boots
    while filled < n_boots:
        count = min(chunk_size, n_boots - filled)
        indices = np.stack([next(sampler) for _ in range(count)])
        chunk_betas, failed = _fit_equations_chunk(data[indices], equations, max_iter, tolerance)
        ok = ~failed
        n_ok = int(ok.sum())
        for store, estimates in zip(betas, chunk_betas):
            store[filled:filled + n_ok] = estimates[ok]
        filled += n_ok
        n_fail += int(failed.sum())
        if n_fail > max_failures:
            raise RuntimeError(FAILURE_MESSAGE.format(n_fail=n_fail, n_boots=n_boots))
    return betas, n_fail


def _fit_equations_chunk(chunk, equations, max_iter, tolerance):
    failed = np.zeros(chunk.shape[0], dtype=bool)
    estimates = []
    try:
        for endog_ind, exog_inds, logit in equations:
            endog, exog = chunk[:, :, endog_ind], chunk[:, :, exog_inds]
            if logit:
                betas, bad = _batch_logit(endog, exog, max_iter, tolerance)
            else:
                betas = _batch_ols(endog[..., None], exog)[..., 0]
                bad = ~np.isfinite(betas).all(axis=1)
            estimates.append(betas)
            failed |= bad
    except LinAlgError:
        return _fit_equations_one_by_one(chunk, equations, max_iter, tolerance)
    return estimates, failed


def _fit_equations_one_by_one(chunk, equations, max_iter, tolerance):
    count = chunk.shape[0]
    estimates = [np.zeros((count, len(exog_inds))) for _, exog_inds, _ in equations]
    failed = np.zeros(count, dtype=bool)
    for i in range(count):
        sample = chunk[i]
        try:
            for store, (endog_ind, exog_inds, logit) in zip(estimates, equations):
                if logit:
                    store[i] = fast_optimize(
                        sample[:, endog_ind], sample[:, exog_inds], n_obs=sample.shape[0],
                        n_vars=len(exog_inds), max_iter=max_iter, tolerance=tolerance,
                    )
                else:
                    store[i] = fast_OLS(sample[:, endog_ind], sample[:, exog_inds])
        except (LinAlgError, ConvergenceError):
            failed[i] = True
    return estimates, failed
