# -*- coding: utf-8 -*-
"""
Serial mediation, PROCESS model 6 (#69).

The mediators form a causal chain: M1 depends on X, M2 on X and M1, M3 on X, M1 and M2, and so on, and
Y depends on X and every mediator. A specific indirect effect runs through any ordered subset of the
mediators and equals the product of the coefficients along the path; with k mediators there are 2**k - 1
of them, ordered as PROCESS orders them (by first mediator, then by length).
"""
from functools import partial
from itertools import combinations

import numpy as np

from .bootstrap import bootstrap_equations
from .models import _summary_table
from .utils import bias_corrected_ci, fast_OLS, fast_optimize, percentile_ci


class SerialMediationModel(object):
    MIN_MEDIATORS = 2
    MAX_MEDIATORS = 4  # as in PROCESS

    ANALYSIS_NAMES = {}

    def __init__(self, data, equations, n_meds, symb_to_ind, symb_to_var, options=None):
        """
        :param data: (n_obs x n_cols) array of the analysis data
        :param equations: list of (endogenous symbol, exogenous terms); the first is the outcome equation,
            the next ones the mediator equations in causal order
        :param n_meds: number of mediators
        :param symb_to_ind: symbol -> column index
        :param symb_to_var: symbol -> variable name
        :param options: the Process options
        """
        self._data = data
        self._n_meds = n_meds
        self._symb_to_ind = symb_to_ind
        self._symb_to_var = symb_to_var
        self._n_obs = data.shape[0]
        self._options = options or {}
        self._equations = equations

        self._exog_terms_y = list(equations[0][1])
        self._exog_terms_m_list = [list(terms) for _, terms in equations[1:]]
        self._exog_terms_m = None  # each mediator has its own design; see _exog_terms_m_list
        self._ind_y = symb_to_ind["y"]
        self._exog_inds_y = [symb_to_ind[t] for t in self._exog_terms_y]
        self._endog_vars_m = [f"m{i + 1}" for i in range(n_meds)]
        self._inds_m = [symb_to_ind[m] for m in self._endog_vars_m]
        self._exog_inds_m_list = [[symb_to_ind[t] for t in terms] for terms in self._exog_terms_m_list]

        # Positions of the path coefficients in each equation.
        self._pos_x_in_m = [terms.index("x") for terms in self._exog_terms_m_list]
        self._pos_m_in_m = [
            {j: terms.index(f"m{j + 1}") for j in range(i)} for i, terms in enumerate(self._exog_terms_m_list)
        ]
        self._pos_m_in_y = [self._exog_terms_y.index(f"m{i + 1}") for i in range(n_meds)]

        # No moderation and no index in model 6; these attributes keep the interface of the parallel model.
        self._moderators_symb = []
        self._moderators_values = []
        self._has_moderation = False
        self._analysis_list = []

        self._paths = self._gen_paths(n_meds)
        self._true_betas_y, self._true_betas_m = self._estimate_true_params()
        self._boot_betas_y, self._boot_betas_m, self._n_fail_samples = self._estimate_bootstrapped_params()
        self.estimation_results = self._indirect_effects()

    @staticmethod
    def _gen_paths(n_meds):
        """Ordered subsets of mediators, in PROCESS order: by first mediator, then by length, then lexicographic."""
        paths = []
        for first in range(n_meds):
            later = range(first + 1, n_meds)
            for size in range(0, n_meds - first):
                for rest in combinations(later, size):
                    paths.append((first,) + rest)
        return paths

    def path_label(self, path):
        stv = self._symb_to_var
        return " -> ".join([stv["x"]] + [stv[f"m{i + 1}"] for i in path] + [stv["y"]])

    @property
    def path_labels(self):
        return [self.path_label(p) for p in self._paths]

    @property
    def effect_labels(self):
        """(component, term) for every row of estimation_results, in order."""
        labels = []
        if self._options["total"]:
            labels.append(("total", "total"))
        labels += [("indirect", label) for label in self.path_labels]
        if self._options["contrast"]:
            labels += [("contrast", f"{a} vs. {b}") for a, b in combinations(self.path_labels, 2)]
        return labels

    def _estimate_true_params(self):
        endog_y = self._data[:, self._ind_y]
        exog_y = self._data[:, self._exog_inds_y]
        if self._options["logit"]:
            betas_y = fast_optimize(
                endog_y, exog_y, n_obs=self._n_obs, n_vars=len(self._exog_inds_y),
                max_iter=self._options["iterate"], tolerance=self._options["convergence"],
            )
        else:
            betas_y = fast_OLS(endog_y, exog_y)
        betas_m = [
            fast_OLS(self._data[:, ind], self._data[:, exog_inds])
            for ind, exog_inds in zip(self._inds_m, self._exog_inds_m_list)
        ]
        return betas_y, betas_m

    def _estimate_bootstrapped_params(self):
        equations = [(self._ind_y, self._exog_inds_y, bool(self._options["logit"]))] + [
            (ind, exog_inds, False) for ind, exog_inds in zip(self._inds_m, self._exog_inds_m_list)
        ]
        betas, n_fail = bootstrap_equations(
            self._data, equations, self._options["boot"], self._options["seed"],
            max_iter=self._options["iterate"], tolerance=self._options["convergence"],
        )
        return betas[0], betas[1:], n_fail

    def _path_effect(self, path, betas_y, betas_m):
        """Product of the coefficients along a path; betas may be 1-D (estimates) or 2-D (bootstrap draws)."""
        first, last = path[0], path[-1]
        effect = betas_m[first][..., self._pos_x_in_m[first]]
        for previous, following in zip(path[:-1], path[1:]):
            effect = effect * betas_m[following][..., self._pos_m_in_m[following][previous]]
        return effect * betas_y[..., self._pos_m_in_y[last]]

    def _interval(self, estimate, draws):
        conf = self._options["conf"]
        if self._options["percent"]:
            return percentile_ci(draws, conf)
        return bias_corrected_ci(estimate, draws, conf)

    def _indirect_effects(self):
        """Specific indirect effects along every path, plus the total and the pairwise contrasts if asked."""
        e = np.array([self._path_effect(p, self._true_betas_y, self._true_betas_m) for p in self._paths])
        be = np.array([self._path_effect(p, self._boot_betas_y, self._boot_betas_m) for p in self._paths])
        effects, se, llci, ulci = [], [], [], []

        def add(estimate, draws):
            low, high = self._interval(estimate, draws)
            effects.append(estimate)
            se.append(draws.std(ddof=1))
            llci.append(low)
            ulci.append(high)

        if self._options["total"]:
            add(e.sum(), be.sum(axis=0))
        for i in range(len(self._paths)):
            add(e[i], be[i])
        if self._options["contrast"]:
            for i, j in combinations(range(len(self._paths)), 2):
                add(e[i] - e[j], be[i] - be[j])
        statistics = [np.array(v, dtype=float) for v in (effects, se, llci, ulci)]
        return dict(zip(["effect", "se", "llci", "ulci"], statistics))

    def coeff_summary(self):
        """
        The specific indirect effects (and the total and contrasts if asked), one row per path.
        :return: DataFrame with columns "", Effect, Boot SE, BootLLCI, BootULCI
        """
        results = self.estimation_results
        rows_stats = np.array([results["effect"], results["se"], results["llci"], results["ulci"]]).T
        labels = []
        for component, term in self.effect_labels:
            if component == "total":
                labels.append("TOTAL")
            elif component == "contrast":
                labels.append(f"Contrast: {term}")
            else:
                labels.append(term)
        return _summary_table([[label] for label in labels], [""], rows_stats, ["Effect", "Boot SE", "BootLLCI", "BootULCI"])

    def summary(self):
        prec = self._options["precision"]
        float_format = partial("{:.{prec}f}".format, prec=prec)
        stv = self._symb_to_var
        return "Indirect effect(s) of {x} on {y} through the serial mediators:\n\n{coeffs}\n\n".format(
            x=stv["x"], y=stv["y"], coeffs=self.coeff_summary().to_string(float_format=float_format)
        )

    def __str__(self):
        return self.summary()
