# -*- coding: utf-8 -*-
"""
Multicategorical X and W (#17): PROCESS's mcx and mcw options.

A categorical variable with g groups is represented by g - 1 codes built with one of PROCESS's coding systems
(indicator, sequential, Helmert or effect coding, the same matrices as PROCESS's makdummy), labelled X1..X{g-1}
or W1..W{g-1}. Every term of a model equation that involves the variable is expanded into one term per code,
so that the outcome models, the bootstrap and the symbolic derivatives work unchanged on the codes; the
effects of X then come out per code ("relative" effects) and a categorical moderator is probed at its groups.
"""
from itertools import product

import numpy as np
import pandas as pd

SCHEMES = {"indicator": 1, "sequential": 2, "helmert": 3, "effect": 4}
MAX_LEVELS = 9


def scheme_name(value, option):
    """Normalize an mcx or mcw value: None or 0 is off; 1 to 4 or a scheme name selects the coding."""
    if value is None or value is False or (isinstance(value, (int, np.integer)) and value == 0):
        return None
    if isinstance(value, str) and value.lower() in SCHEMES:
        return value.lower()
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool) and 1 <= value <= 4:
        return {v: k for k, v in SCHEMES.items()}[int(value)]
    raise ValueError(
        f"The option '{option}' must be None, one of 1 (indicator), 2 (sequential), 3 (helmert), 4 (effect), "
        f"or one of the names {', '.join(repr(s) for s in SCHEMES)}."
    )


def code_matrix(n_levels, scheme):
    """The (g x g-1) matrix of codes for g groups in ascending order, as PROCESS builds it."""
    g, k = n_levels, n_levels - 1
    matrix = np.zeros((g, k))
    if scheme == "indicator":
        matrix[1:, :] = np.eye(k)
    elif scheme == "sequential":
        for i in range(1, g):
            matrix[i, :i] = 1.0
    elif scheme == "effect":
        matrix[1:, :] = np.eye(k)
        matrix[0, :] = -1.0
    elif scheme == "helmert":
        for j in range(k):
            matrix[j, j] = -(g - 1 - j) / (g - j)
            matrix[j + 1:, j] = 1.0 / (g - j)
    else:
        raise ValueError(f"Unknown coding scheme {scheme!r}.")
    return matrix


class Coding:
    """
    The codes of one multicategorical variable.

    :param name: the variable name in the data
    :param values: its values on the analysis rows
    :param scheme: "indicator", "sequential", "helmert" or "effect"
    :param letter: "X" or "W", the prefix of the code labels
    :param symbol: the model symbol the codes replace ("x", "w", or "m" for the moderator of models 1 to 3)
    """

    def __init__(self, name, values, scheme, letter, symbol):
        series = pd.Series(np.asarray(values))
        if series.isna().any():
            raise ValueError(f"The multicategorical variable '{name}' has missing values on the analysis rows.")
        counts = series.value_counts()
        levels = sorted(counts.index.tolist())
        if len(levels) < 3:
            raise ValueError(
                f"The multicategorical variable '{name}' must have at least three categories (it has "
                f"{len(levels)}); a two-level variable is a plain dichotomous predictor."
            )
        if len(levels) > MAX_LEVELS:
            raise ValueError(f"The multicategorical variable '{name}' has more than {MAX_LEVELS} categories.")
        if counts.min() < 2:
            raise ValueError(f"Every category of the multicategorical variable '{name}' must contain at least two cases.")
        self.name = name
        self.scheme = scheme
        self.letter = letter
        self.symbol = symbol
        self.levels = levels
        self.matrix = code_matrix(len(levels), scheme)
        self.labels = [f"{letter}{j + 1}" for j in range(len(levels) - 1)]
        self.symbols = [f"{symbol}{j + 1}" for j in range(len(levels) - 1)]
        self._positions = {level: i for i, level in enumerate(levels)}

    def codes_for(self, values):
        """(n x g-1) array of codes for the given values."""
        positions = np.array([self._positions[v] for v in np.asarray(values)])
        return self.matrix[positions]

    def numeric(self, values):
        """The values as floats when they are numbers, else their 1-based position among the sorted levels."""
        try:
            return np.asarray(values, dtype=float)
        except (TypeError, ValueError):
            return np.array([self._positions[v] + 1 for v in np.asarray(values)], dtype=float)

    def level_codes(self):
        """{level: {code symbol: code value}} for substituting a group into the symbolic derivatives."""
        return {level: dict(zip(self.symbols, self.matrix[i])) for i, level in enumerate(self.levels)}

    def table(self):
        """The mapping of categories to codes, as PROCESS prints it at the top of its output."""
        frame = pd.DataFrame(self.matrix, columns=self.labels)
        frame.insert(0, self.name, self.levels)
        frame.index = [""] * len(frame)
        return frame


def expand_terms(terms, codings):
    """
    Replace every term that involves a coded symbol by one term per code (all combinations when a product
    involves two coded symbols), keeping the order of the terms.
    :param terms: list of terms such as "Cons", "x", "x*w"
    :param codings: {symbol: Coding}
    """
    if not codings:
        return list(terms)
    expanded = []
    for term in terms:
        factors = term.split("*")
        choices = [codings[f].symbols if f in codings else [f] for f in factors]
        for combo in product(*choices):
            expanded.append("*".join(combo))
    return expanded


def mod_dict(mod_symb, values, mod_codes):
    """
    The substitution dict for eval_expression: a categorical moderator's group expands to its code values.
    :param mod_symb: moderator symbols
    :param values: one value per moderator (a group for a categorical moderator)
    :param mod_codes: {moderator symbol: {level: {code symbol: value}}}
    """
    out = {}
    for symb, value in zip(mod_symb, values):
        if mod_codes and symb in mod_codes:
            out.update(mod_codes[symb][value])
        else:
            out[symb] = value
    return out


def code_symbols(mod_codes, symb):
    """The code symbols of a categorical moderator, in order."""
    first = next(iter(mod_codes[symb].values()))
    return list(first)
