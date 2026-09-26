# Changelog

All notable changes to PyProcessMacro are documented in this file. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the
project uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- `version` argument: `"2.16"` (the default, what PyProcessMacro has always produced) or `"5.0"`, the conventions of PROCESS for R 5.0: percentile bootstrap intervals, spotlight values at the 16th, 50th and 84th percentiles computed as PROCESS does, conditional effects of models 1 to 3 reported only when the highest-order interaction has p at most `intprobe=0.10`, and refusal of models 23 to 27, 30 to 57 and 74, which PROCESS 5 no longer defines and PyProcessMacro keeps estimating under `"2.16"`. An argument passed explicitly wins over the version's default. The 2.16 conventions stay the default throughout 2.x; 3.0 switches to 5.0 (#87, #74).
- `intprobe` and `moments` options, PROCESS's own names for the probing threshold of models 1 to 3 and for spotlight values at the mean and one SD either side. `direct_model.probe_p` and `direct_model.probed` expose the test PROCESS compares to `intprobe`: the coefficient's test for OLS, a likelihood-ratio test for a binary outcome (#87).
- The initialization banner, `summary()` and the notebook display start with the PROCESS version emulated and the conventions in force (#87).
- The PROCESS 5 comparison test fits with `version="5.0"` and lets PyProcessMacro compute the spotlight values, which checks its percentile rule against PROCESS's (#87).
- Comparison files produced with PROCESS for R version 5.0 for every model PROCESS 5 defines (`tests/Results/v5`), the generator that makes them from Hayes's `process.R` (`tests/fixtures/regenerate.py`), and a `v5`-marked test that checks what 2.x already claims to reproduce against them and lists the deliberate differences for 3.0. The 2.16 files remain the accuracy reference for 2.x (#82).
- Model 6 is compared to the PROCESS 2.16 output file that was already in the repository (#69).

### Changed

- `percent` defaults to None and takes the version's value: bias-corrected intervals under `"2.16"`, as before (#87).
- The error for an invalid `quantile` value named the wrong option.

## [2.1.0] - 2026-09-25

Additive release, tracked in the
[2.1.0 milestone](https://github.com/QuentinAndre/pyprocessmacro/milestone/3).
Nothing written against 2.0 stops working.

### Added

- `Process.tidy()`: every estimate in one long DataFrame with fixed column names (component, outcome, term, moderator, one column per moderator, estimate, std_error, statistic, p_value, conf_low, conf_high, method, conf_level, n_boot), modelled on R's broom (#64).
- `Process.glance()`: one row of fit statistics per outcome model, including log-likelihood, AIC and BIC, which are also stored in `estimation_results` (#65).
- `Process.augment()`: the analysis data with fitted values and residuals per outcome model (#66).
- `Process.summary()` returns the text it prints, `str(process)` gives the same text, and a `Process` object displays its tables as HTML in notebooks (#71).
- README section on `tidy()`, `glance()`, `augment()` and the `summary()` text.
- `OutcomeModel.to_statsmodels()` and `Process.to_statsmodels()` refit each outcome model with statsmodels (same design matrix, same covariance estimator) and return the statsmodels results object, giving access to its summaries, contrasts, diagnostics and table formatters. statsmodels is an optional extra: `pip install pyprocessmacro[statsmodels]` (#67).
- Model 6, serial mediation with two to four mediators in causal order: specific indirect effects through every ordered subset of mediators, labelled by path, plus total and contrasts. Point estimates are checked against products of statsmodels coefficients and the bootstrap against an independent resampler; a PROCESS 2.16 output file for model 6 is not part of the fixtures yet (#69).
- `effsize=True` reports the partially and completely standardized indirect effects with bootstrap intervals, standardized within each resample as PROCESS does, for unmoderated indirect paths with a continuous outcome (models 4 and 6). The option used to warn that it was unsupported (#70).

### Changed

- The printed tables are assembled from typed columns instead of a string array coerced back to numbers; same content, no more `to_numeric` round trip (#72).
- The bootstrap fits resamples in batches with stacked linear algebra instead of one Python iteration per resample, which makes mediation models several times faster. The resample indices are drawn exactly as before, so a given seed still reproduces the same draws and, up to floating-point rounding, the same estimates as 2.0 (#68, resolves #75 for this release).

### Fixed

- A perfectly separated logistic regression raises `ConvergenceError` on every platform. On numpy 1.26 the Newton-Raphson loop could reach a saturated fit, where the score is exactly zero, and return huge coefficients as if it had converged.

## [2.0.0] - 2026-09-25

Major release, tracked in the
[2.0.0 milestone](https://github.com/QuentinAndre/pyprocessmacro/milestone/2).
Reported values change in this release; see "Upgrading to 2.0" in
`README.md`.

### Changed

- Confidence intervals for OLS coefficients and for direct effects now use t critical values with the residual degrees of freedom, as PROCESS does. Intervals were based on z, which made them too narrow in small samples (#40).
- No index of moderated mediation is reported when a moderator sits on both the X-to-M and the M-to-Y paths (models 58 to 73, 75 and 76), matching PROCESS. The indirect effect is quadratic in such a moderator and the previously reported values were not Hayes's indices (#43).
- `modval` raises a `ValueError` naming any key that is not a moderator of the model, in the constructor and in the plotting methods; misspelled names were silently ignored (#46).
- Passing `jn=True`, `effsize=True` or `mc=True` now warns that the option is not supported; the warnings never fired. Unsupported PROCESS options such as `normal` warn with a visible `UserWarning` instead of a hidden `DeprecationWarning`, and an unknown keyword argument raises a `TypeError` instead of being ignored (#47).
- A logistic regression that diverges or does not converge raises `pyprocessmacro.ConvergenceError` instead of returning garbage silently; failed bootstrap resamples are counted for that reason too, and the bootstrap gives up with a clear error once more resamples failed than were requested. Bias-corrected intervals stay finite when every draw falls on one side of the estimate (#49).

### Fixed

- Adjusted R² of the OLS outcome models used one degree of freedom too many; the F p-value is computed with the survival function so it no longer rounds to exactly zero (#41).
- Cox-Snell and Nagelkerke pseudo R² of logistic outcome models are computed in log space and no longer become NaN beyond about a thousand observations (#42).
- The sample size reported after dropping rows with missing values is the number of rows kept; the number of dropped rows was always reported as zero (#44).
- `seed=0` and `seed=None` are accepted; any integer up to 2**32 - 1 works, and `None` draws a different bootstrap sample on every run (#45).
- Importing the package no longer resets Python's global warning filters (#48).
- The HC1 covariance estimator scaled by n/(n-k-1) instead of n/(n-k); it was unreachable before `cov_type` existed (#52).

### Removed

- `plot_direct_effects()` and `plot_indirect_effects()`, which raised a `DeprecationWarning` since 1.0.0. Use `plot_conditional_direct_effects()` and `plot_conditional_indirect_effects()` (#50).
- The `.pyi` stub files, which were inaccurate and unmaintained; the inline type hints remain (#50).

### Added

- `statsmodels` is a test dependency (`pip install -e .[test]`).
- The accuracy suite against PROCESS now also covers model 4, the conditional effects of the moderation-only models 1 to 3, and the index tables of every model where PROCESS 2.16 prints one.
- `cov_type` option selecting the OLS covariance estimator: `"standard"` (default), `"HC0"`, `"HC1"`, `"HC2"` or `"HC3"`; `hc3=True` remains as shorthand for `"HC3"`. `Process.dv` names the outcome variable; `iv`, which held it under a misleading name, is kept for compatibility (#52).
- README: an "Upgrading to 2.0" section listing every change in reported values and behaviour, and documentation of `cov_type`; the 1.0.4 note no longer calls the default estimator HC0 (#51).

## [1.0.14] - 2026-09-25

Hotfix release, tracked in the
[1.0.14 milestone](https://github.com/QuentinAndre/pyprocessmacro/milestone/1).

### Changed

- Packaging moved from `setup.py` to `pyproject.toml`. The version is now read
  from `pyprocessmacro.__version__` only (#30).
- Supported Python versions are 3.11 and newer. Release 1.0.13 declared
  `>=3.14` by mistake, which made `pip` fall back to 1.0.12 on older
  interpreters (#28).
- Minimum dependency versions are declared and exercised in CI: numpy 1.26,
  pandas 2.0, scipy 1.10, matplotlib 3.7, seaborn 0.13.

### Fixed

- The wheel no longer installs a top-level `tests` package (#29).
- `Process.summary()` and the four index summaries no longer fail on pandas 3, where `to_numeric(errors="ignore")` was removed (#32).
- `get_bootstrap_estimates()` no longer fails on pandas 2 and newer, where `DataFrame.append` was removed (#33).
- `controls_in="x_to_m"` is accepted again; a typo in the validator rejected it (#34).
- A single mediator can be passed as a string; it was treated as a list of characters (#35).
- `plot_conditional_direct_effects()` and `plot_conditional_indirect_effects()` facet `col` and `row` by the moderator asked for, not by the x-axis moderator (#36).
- `floodlight_direct_effect()` and `floodlight_indirect_effect()` raise a clear error for an unknown variable in `other_modval` (#37).
- `hue_format` accepts the documented `val1` and `val2` keys (`hue1` and `hue2` still work), and `hue` rejects more than two moderators (#38).
- README examples use the current method names, the floodlight example calls the floodlight method, and the install section states the Python floor (#39).

### Added

- Continuous integration on GitHub Actions across Python 3.11 to 3.14 and
  pandas 2 and 3, plus a build-and-check job (#31).
- This changelog (#30).
- Smoke tests of the public API in `tests/test_api_smoke.py`, marked `smoke`.

## [1.0.13] - 2026-03-31

### Changed

- Refactored for numpy 2.x compatibility (#26).

### Known issues

- Declares `Requires-Python >=3.14` by mistake. Use 1.0.14 instead (#28).
- `Process.summary()` fails on pandas 3 for every model with a moderator on
  the indirect path (#32), and `get_bootstrap_estimates()` fails on pandas 2
  and newer (#33). Both are fixed in 1.0.14.

## [1.0.12] - 2022-03-08

### Fixed

- Crash in `summary()` after a pandas change to display options (#22).

## [1.0.11] and earlier

See the version history section of `README.md`.

[Unreleased]: https://github.com/QuentinAndre/pyprocessmacro/compare/2.1.0...HEAD
[2.1.0]: https://github.com/QuentinAndre/pyprocessmacro/compare/2.0.0...2.1.0
[2.0.0]: https://github.com/QuentinAndre/pyprocessmacro/compare/1.0.14...2.0.0
[1.0.14]: https://github.com/QuentinAndre/pyprocessmacro/compare/1.0.13...1.0.14
[1.0.13]: https://github.com/QuentinAndre/pyprocessmacro/releases/tag/1.0.13
