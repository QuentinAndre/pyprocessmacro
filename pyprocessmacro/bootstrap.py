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

from . import negbin
from .utils import ConvergenceError, bootstrap_sampler, fast_OLS, fast_optimize

# Upper bound on the number of floating-point values held by one chunk of resampled data. Chunks of this size
# stay in cache: measured on 2026-09-26 (#96), the batched estimators are then 1.5 to 3.5 times faster than
# one-by-one fits, whereas the 10 million bound shipped in 2.1 made them memory-bound and no faster.
CHUNK_ELEMENTS = 100_000


def family_of(options):
    """The estimator of the outcome equation named by the options: 'ols', 'logit' or 'negbin' (#25)."""
    return options.get("family") or ("logit" if options.get("logit") else "ols")


def _as_family(kind):
    """A family name from a family name or from the pre-2.2 logit boolean."""
    if kind is True:
        return "logit"
    if kind is False or kind is None:
        return "ols"
    return kind


class BootstrapSpec:
    """
    Which columns of the data array are which, and how the outcome equation is estimated: family "ols",
    "logit" or "negbin" (#25). The pre-2.2 logit boolean is still accepted.
    """

    def __init__(self, ind_y, exog_inds_y, inds_m, exog_inds_m, family="ols", max_iter=10000, tolerance=1e-10,
                 logit=None):
        self.ind_y = ind_y
        self.exog_inds_y = list(exog_inds_y)
        self.inds_m = list(inds_m)
        self.exog_inds_m = list(exog_inds_m)
        self.family = _as_family(family if logit is None else logit)
        self.logit = self.family == "logit"
        self.max_iter = max_iter
        self.tolerance = tolerance


def bootstrap_parameters(data, spec, n_boots, seed, chunk_size=None, sd_inds=None):
    """
    Estimate the outcome and mediator equations on n_boots resamples of the data.

    :param data: (n_obs x n_cols) array of the analysis data
    :param spec: BootstrapSpec
    :param n_boots: number of successful resamples wanted
    :param seed: seed of the resampler (None for fresh entropy)
    :param chunk_size: resamples fitted per batch; None picks one from the data size
    :param sd_inds: columns whose standard deviation is wanted for every successful resample (#70)
    :return: (betas_y, betas_m, n_fail, sds): (n_boots x k_y) array, (n_meds x n_boots x k_m) array, the
        number of resamples discarded because an equation could not be estimated on them, and the
        (n_boots x len(sd_inds)) array of standard deviations, or None.
    """
    data = np.asarray(data, dtype=float)
    n_obs, n_cols = data.shape
    if chunk_size is None:
        chunk_size = int(max(1, min(n_boots, CHUNK_ELEMENTS // max(1, n_obs * n_cols))))
    sampler = bootstrap_sampler(n_obs, seed)
    k_y, k_m, n_meds = len(spec.exog_inds_y), len(spec.exog_inds_m), len(spec.inds_m)
    betas_y = np.empty((n_boots, k_y))
    betas_m = np.empty((n_meds, n_boots, k_m))
    sds = None if sd_inds is None else np.empty((n_boots, len(sd_inds)))
    filled, n_fail, max_failures = 0, 0, n_boots
    while filled < n_boots:
        count = min(chunk_size, n_boots - filled)
        indices = np.stack([next(sampler) for _ in range(count)])  # one draw per sample: same stream as before
        chunk = data[indices]
        chunk_y, chunk_m, failed = _fit_chunk(chunk, spec)
        ok = ~failed
        n_ok = int(ok.sum())
        betas_y[filled:filled + n_ok] = chunk_y[ok]
        betas_m[:, filled:filled + n_ok] = chunk_m[:, ok]
        if sd_inds is not None:
            sds[filled:filled + n_ok] = chunk[ok][:, :, sd_inds].std(axis=1, ddof=1)
        filled += n_ok
        n_fail += int(failed.sum())
        if n_fail > max_failures:
            raise RuntimeError(FAILURE_MESSAGE.format(n_fail=n_fail, n_boots=n_boots))
    return betas_y, betas_m, n_fail, sds


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
        betas_y, failed = batch_outcome(y, exog_y, spec.family, spec.max_iter, spec.tolerance)
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
            betas_y[i] = fit_outcome(
                sample[:, spec.ind_y], sample[:, spec.exog_inds_y], spec.family, spec.max_iter, spec.tolerance
            )
            for j, ind in enumerate(spec.inds_m):
                betas_m[j, i] = fast_OLS(sample[:, ind], sample[:, spec.exog_inds_m])
        except (LinAlgError, ConvergenceError):
            failed[i] = True
    return betas_y, betas_m, failed


def batch_outcome(endog, exog, family, max_iter, tolerance):
    """The outcome equation on a batch, by family: (c x k) betas and (c,) failed flags."""
    if family == "logit":
        return _batch_logit(endog, exog, max_iter, tolerance)
    if family == "negbin":
        params, failed = negbin.batch_fit(endog, exog, max_iter, tolerance)
        return params[:, :-1], failed  # the last parameter is log alpha
    betas = _batch_ols(endog[..., None], exog)[..., 0]
    return betas, ~np.isfinite(betas).all(axis=1)


def fit_outcome(endog, exog, family, max_iter, tolerance):
    """One outcome equation, by family. Raises LinAlgError or ConvergenceError when it cannot be estimated."""
    if family == "logit":
        return fast_optimize(
            endog, exog, n_obs=exog.shape[0], n_vars=exog.shape[1], max_iter=max_iter, tolerance=tolerance
        )
    if family == "negbin":
        return negbin.fit_betas(endog, exog, max_iter, tolerance)
    return fast_OLS(endog, exog)


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
    # Saturated fits: every outcome predicted exactly means separation, not convergence.
    fitted = _logit_cdf(np.einsum("cnk,ck->cn", exog, params))
    failed |= np.abs(endog - fitted).max(axis=1) < 1e-8
    return params, failed


FAILURE_MESSAGE = (
    "{n_fail} bootstrap samples failed to estimate before {n_boots} succeeded. The model is probably not "
    "estimable on resamples of this data (check for separation, collinearity, or a very small sample)."
)


def bootstrap_equations(data, equations, n_boots, seed, max_iter=10000, tolerance=1e-10, chunk_size=None,
                        sd_inds=None):
    """
    Estimate several equations, each with its own design matrix, on n_boots resamples (serial mediation).

    :param equations: list of (endog_ind, exog_inds, family) triples; family is "ols", "logit" or "negbin"
        (a boolean is read as the pre-2.2 logit flag)
    :param sd_inds: columns whose standard deviation is wanted for every successful resample (#70)
    :return: (list of (n_boots x k_i) arrays, one per equation, n_fail, sds or None)
    """
    data = np.asarray(data, dtype=float)
    n_obs, n_cols = data.shape
    if chunk_size is None:
        chunk_size = int(max(1, min(n_boots, CHUNK_ELEMENTS // max(1, n_obs * n_cols))))
    sampler = bootstrap_sampler(n_obs, seed)
    betas = [np.empty((n_boots, len(exog_inds))) for _, exog_inds, _ in equations]
    sds = None if sd_inds is None else np.empty((n_boots, len(sd_inds)))
    filled, n_fail, max_failures = 0, 0, n_boots
    while filled < n_boots:
        count = min(chunk_size, n_boots - filled)
        indices = np.stack([next(sampler) for _ in range(count)])
        chunk = data[indices]
        chunk_betas, failed = _fit_equations_chunk(chunk, equations, max_iter, tolerance)
        ok = ~failed
        n_ok = int(ok.sum())
        for store, estimates in zip(betas, chunk_betas):
            store[filled:filled + n_ok] = estimates[ok]
        if sd_inds is not None:
            sds[filled:filled + n_ok] = chunk[ok][:, :, sd_inds].std(axis=1, ddof=1)
        filled += n_ok
        n_fail += int(failed.sum())
        if n_fail > max_failures:
            raise RuntimeError(FAILURE_MESSAGE.format(n_fail=n_fail, n_boots=n_boots))
    return betas, n_fail, sds


def _fit_equations_chunk(chunk, equations, max_iter, tolerance):
    failed = np.zeros(chunk.shape[0], dtype=bool)
    estimates = []
    try:
        for endog_ind, exog_inds, kind in equations:
            endog, exog = chunk[:, :, endog_ind], chunk[:, :, exog_inds]
            betas, bad = batch_outcome(endog, exog, _as_family(kind), max_iter, tolerance)
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
            for store, (endog_ind, exog_inds, kind) in zip(estimates, equations):
                store[i] = fit_outcome(sample[:, endog_ind], sample[:, exog_inds], _as_family(kind), max_iter, tolerance)
        except (LinAlgError, ConvergenceError):
            failed[i] = True
    return estimates, failed
