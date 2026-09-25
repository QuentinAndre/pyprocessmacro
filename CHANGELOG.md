# Changelog

All notable changes to PyProcessMacro are documented in this file. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the
project uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

Work towards 1.0.14, a hotfix release. Tracked in the
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

[Unreleased]: https://github.com/QuentinAndre/pyprocessmacro/compare/1.0.13...HEAD
[1.0.13]: https://github.com/QuentinAndre/pyprocessmacro/releases/tag/1.0.13
