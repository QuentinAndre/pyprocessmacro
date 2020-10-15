# -*- coding: utf-8 -*-
import copy
import warnings
from itertools import product

import numpy as np
import pandas as pd

from .models import (
    OLSOutcomeModel,
    DirectEffectModel,
    ParallelMediationModel,
    LogitOutcomeModel,
    DirectFloodlightAnalysis,
    IndirectFloodlightAnalysis,
)
from .utils import plot_conditional_effects, gen_moderators

warnings.simplefilter("default")


class Process(object):
    __var_kws__ = {"x", "m", "w", "z", "v", "q", "y"}

    __options_kws__ = {
        "cluster",
        "contrast",
        "boot",
        "mc",
        "conf",
        "effsize",
        "jn",
        "hc3",
        "controls_in",
        "total",
        "center",
        "quantile",
        "detail",
        "plot",
        "seed",
        "percent",
        "iterate",
        "convergence",
        "precision",
        "logit",
        "modval",
        "controls",
    }

    __models_vars__ = {
        1: {"x", "m", "y"},
        2: {"x", "m", "w", "y"},
        3: {"x", "m", "w", "y"},
        4: {"x", "m", "y"},
        5: {"x", "m", "w", "y"},
        6: {"x", "m", "y"},
        7: {"x", "m", "w", "y"},
        8: {"x", "m", "w", "y"},
        9: {"x", "m", "w", "z", "y"},
        10: {"x", "m", "w", "z", "y"},
        11: {"x", "m", "w", "z", "y"},
        12: {"x", "m", "w", "z", "y"},
        13: {"x", "m", "w", "z", "y"},
        14: {"x", "m", "v", "y"},
        15: {"x", "m", "v", "y"},
        16: {"x", "m", "v", "q", "y"},
        17: {"x", "m", "v", "q", "y"},
        18: {"x", "m", "v", "q", "y"},
        19: {"x", "m", "v", "q", "y"},
        20: {"x", "m", "v", "q", "y"},
        21: {"x", "m", "w", "v", "y"},
        22: {"x", "m", "w", "v", "y"},
        23: {"x", "m", "w", "z", "v", "y"},
        24: {"x", "m", "w", "z", "v", "y"},
        25: {"x", "m", "w", "z", "v", "y"},
        26: {"x", "m", "w", "z", "v", "y"},
        27: {"x", "m", "w", "z", "v", "y"},
        28: {"x", "m", "w", "v", "y"},
        29: {"x", "m", "w", "v", "y"},
        30: {"x", "m", "w", "z", "v", "y"},
        31: {"x", "m", "w", "z", "v", "y"},
        32: {"x", "m", "w", "z", "v", "y"},
        33: {"x", "m", "w", "z", "v", "y"},
        34: {"x", "m", "w", "z", "v", "y"},
        35: {"x", "m", "w", "v", "q", "y"},
        36: {"x", "m", "w", "v", "q", "y"},
        37: {"x", "m", "w", "v", "q", "y"},
        38: {"x", "m", "w", "v", "q", "y"},
        39: {"x", "m", "w", "v", "q", "y"},
        40: {"x", "m", "w", "v", "q", "y"},
        41: {"x", "m", "w", "v", "q", "y"},
        42: {"x", "m", "w", "v", "q", "y"},
        43: {"x", "m", "w", "v", "q", "y"},
        44: {"x", "m", "w", "v", "q", "y"},
        45: {"x", "m", "w", "z", "v", "q", "y"},
        46: {"x", "m", "w", "z", "v", "q", "y"},
        47: {"x", "m", "w", "z", "v", "q", "y"},
        48: {"x", "m", "w", "z", "v", "q", "y"},
        49: {"x", "m", "w", "z", "v", "q", "y"},
        50: {"x", "m", "w", "z", "v", "q", "y"},
        51: {"x", "m", "w", "z", "v", "q", "y"},
        52: {"x", "m", "w", "z", "v", "q", "y"},
        53: {"x", "m", "w", "z", "v", "q", "y"},
        54: {"x", "m", "w", "z", "v", "q", "y"},
        55: {"x", "m", "w", "z", "v", "q", "y"},
        56: {"x", "m", "w", "z", "v", "q", "y"},
        57: {"x", "m", "w", "z", "v", "q", "y"},
        58: {"x", "m", "w", "y"},
        59: {"x", "m", "w", "y"},
        60: {"x", "m", "w", "z", "y"},
        61: {"x", "m", "w", "z", "y"},
        62: {"x", "m", "w", "z", "y"},
        63: {"x", "m", "w", "z", "y"},
        64: {"x", "m", "w", "v", "y"},
        65: {"x", "m", "w", "v", "y"},
        66: {"x", "m", "w", "v", "y"},
        67: {"x", "m", "w", "v", "y"},
        68: {"x", "m", "w", "z", "y"},
        69: {"x", "m", "w", "z", "y"},
        70: {"x", "m", "w", "v", "y"},
        71: {"x", "m", "w", "v", "y"},
        72: {"x", "m", "w", "z", "y"},
        73: {"x", "m", "w", "z", "y"},
        74: {"x", "m", "y"},
        75: {"x", "m", "w", "z", "y"},
        76: {"x", "m", "w", "z", "y"},
    }

    __models_eqs__ = {
        1: {"all_to_y": ["x", "m", "x*m"], "x_to_m": []},
        2: {"all_to_y": ["x", "m", "w", "x*m", "x*w"], "x_to_m": []},
        3: {"all_to_y": ["x", "m", "w", "x*m", "x*w", "m*w", "x*m*w"], "x_to_m": []},
        4: {"all_to_y": ["x", "m"], "x_to_m": ["x"]},
        5: {"all_to_y": ["x", "m", "w", "x*w"], "x_to_m": ["x"]},
        6: {"all_to_y": ["x", "m"], "x_to_m": ["x"]},
        7: {"all_to_y": ["x", "m"], "x_to_m": ["x", "w", "x*w"]},
        8: {"all_to_y": ["x", "m", "w", "x*w"], "x_to_m": ["x", "w", "x*w"]},
        9: {"all_to_y": ["x", "m"], "x_to_m": ["x", "w", "z", "x*w", "x*z"]},
        10: {
            "all_to_y": ["x", "m", "w", "z", "x*w", "x*z"],
            "x_to_m": ["x", "w", "z", "x*w", "x*z"],
        },
        11: {
            "all_to_y": ["x", "m"],
            "x_to_m": ["x", "w", "z", "x*w", "x*z", "w*z", "x*w*z"],
        },
        12: {
            "all_to_y": ["x", "m", "w", "z", "x*w", "x*z", "w*z", "x*w*z"],
            "x_to_m": ["x", "w", "z", "x*w", "x*z", "w*z", "x*w*z"],
        },
        13: {
            "all_to_y": ["x", "m", "w", "x*w"],
            "x_to_m": ["x", "w", "z", "x*w", "x*z", "w*z", "x*w*z"],
        },
        14: {"all_to_y": ["x", "m", "v", "m*v"], "x_to_m": ["x"]},
        15: {"all_to_y": ["x", "m", "v", "x*v", "m*v"], "x_to_m": ["x"]},
        16: {"all_to_y": ["x", "m", "v", "q", "m*v", "m*q"], "x_to_m": ["x"]},
        17: {
            "all_to_y": ["x", "m", "v", "q", "x*v", "x*q", "m*v", "m*q"],
            "x_to_m": ["x"],
        },
        18: {
            "all_to_y": ["x", "m", "v", "q", "m*v", "m*q", "v*q", "m*v*q"],
            "x_to_m": ["x"],
        },
        19: {
            "all_to_y": [
                "x",
                "m",
                "v",
                "q",
                "x*v",
                "x*q",
                "m*v",
                "m*q",
                "v*q",
                "x*v*q",
                "m*v*q",
            ],
            "x_to_m": ["x"],
        },
        20: {
            "all_to_y": ["x", "m", "v", "q", "x*v", "m*v", "m*q", "v*q", "m*v*q"],
            "x_to_m": ["x"],
        },
        21: {"all_to_y": ["x", "m", "v", "m*v"], "x_to_m": ["x", "w", "x*w"]},
        22: {
            "all_to_y": ["x", "m", "w", "v", "x*w", "m*v"],
            "x_to_m": ["x", "w", "x*w"],
        },
        23: {
            "all_to_y": ["x", "m", "v", "m*v"],
            "x_to_m": ["x", "w", "z", "x*w", "x*z"],
        },
        24: {
            "all_to_y": ["x", "m", "w", "z", "v", "x*w", "x*z", "m*v"],
            "x_to_m": ["x", "w", "z", "x*w", "x*z"],
        },
        25: {
            "all_to_y": ["x", "m", "v", "m*v"],
            "x_to_m": ["x", "w", "z", "x*w", "x*z", "w*z", "x*w*z"],
        },
        26: {
            "all_to_y": ["x", "m", "w", "z", "v", "x*w", "x*z", "m*v", "w*z", "x*w*z"],
            "x_to_m": ["x", "w", "z", "x*w", "x*z", "w*z", "x*w*z"],
        },
        27: {
            "all_to_y": ["x", "m", "w", "v", "x*w", "m*v"],
            "x_to_m": ["x", "w", "z", "x*w", "x*z", "w*z", "x*w*z"],
        },
        28: {"all_to_y": ["x", "m", "v", "x*v", "m*v"], "x_to_m": ["x", "w", "x*w"]},
        29: {
            "all_to_y": ["x", "m", "w", "v", "x*w", "x*v", "m*v"],
            "x_to_m": ["x", "w", "x*w"],
        },
        30: {
            "all_to_y": ["x", "m", "v", "x*v", "m*v"],
            "x_to_m": ["x", "w", "z", "x*w", "x*z"],
        },
        31: {
            "all_to_y": ["x", "m", "w", "z", "v", "x*w", "x*z", "x*v", "m*v"],
            "x_to_m": ["x", "w", "z", "x*w", "x*z"],
        },
        32: {
            "all_to_y": ["x", "m", "v", "x*v", "m*v"],
            "x_to_m": ["x", "w", "z", "x*w", "x*z", "w*z", "x*w*z"],
        },
        33: {
            "all_to_y": [
                "x",
                "m",
                "w",
                "z",
                "v",
                "x*w",
                "x*z",
                "x*v",
                "m*v",
                "w*z",
                "x*w*z",
            ],
            "x_to_m": ["x", "w", "z", "x*w", "x*z", "w*z", "x*w*z"],
        },
        34: {
            "all_to_y": ["x", "m", "w", "v", "x*w", "x*v", "m*v"],
            "x_to_m": ["x", "w", "z", "x*w", "x*z", "w*z", "x*w*z"],
        },
        35: {
            "all_to_y": ["x", "m", "v", "q", "m*v", "m*q"],
            "x_to_m": ["x", "w", "x*w"],
        },
        36: {
            "all_to_y": ["x", "m", "v", "q", "x*v", "x*q", "m*v", "m*q"],
            "x_to_m": ["x", "w", "x*w"],
        },
        37: {
            "all_to_y": ["x", "m", "v", "q", "m*v", "m*q", "v*q", "m*v*q"],
            "x_to_m": ["x", "w", "x*w"],
        },
        38: {
            "all_to_y": [
                "x",
                "m",
                "v",
                "q",
                "x*v",
                "x*q",
                "m*v",
                "m*q",
                "v*q",
                "x*v*q",
                "m*v*q",
            ],
            "x_to_m": ["x", "w", "x*w"],
        },
        39: {
            "all_to_y": ["x", "m", "v", "q", "x*v", "m*v", "m*q", "v*q", "m*v*q"],
            "x_to_m": ["x", "w", "x*w"],
        },
        40: {
            "all_to_y": ["x", "m", "w", "v", "q", "x*w", "m*v", "m*q"],
            "x_to_m": ["x", "w", "x*w"],
        },
        41: {
            "all_to_y": ["x", "m", "w", "v", "q", "x*w", "x*v", "x*q", "m*v", "m*q"],
            "x_to_m": ["x", "w", "x*w"],
        },
        42: {
            "all_to_y": ["x", "m", "w", "v", "q", "x*w", "m*v", "m*q", "v*q", "m*v*q"],
            "x_to_m": ["x", "w", "x*w"],
        },
        43: {
            "all_to_y": [
                "x",
                "m",
                "w",
                "v",
                "q",
                "x*w",
                "x*v",
                "x*q",
                "m*v",
                "m*q",
                "v*q",
                "x*v*q",
                "m*v*q",
            ],
            "x_to_m": ["x", "w", "x*w"],
        },
        44: {
            "all_to_y": [
                "x",
                "m",
                "w",
                "v",
                "q",
                "x*w",
                "x*v",
                "m*v",
                "m*q",
                "v*q",
                "m*v*q",
            ],
            "x_to_m": ["x", "w", "x*w"],
        },
        45: {
            "all_to_y": ["x", "m", "v", "q", "m*v", "m*q"],
            "x_to_m": ["x", "w", "z", "x*w", "x*z"],
        },
        46: {
            "all_to_y": ["x", "m", "v", "q", "m*v", "m*q"],
            "x_to_m": ["x", "w", "z", "x*w", "x*z", "w*z", "x*w*z"],
        },
        47: {
            "all_to_y": ["x", "m", "v", "q", "m*v", "m*q", "v*q", "m*v*q"],
            "x_to_m": ["x", "w", "z", "x*w", "x*z"],
        },
        48: {
            "all_to_y": ["x", "m", "v", "q", "m*v", "m*q", "v*q", "m*v*q"],
            "x_to_m": ["x", "w", "z", "x*w", "x*z", "w*z", "x*w*z"],
        },
        49: {
            "all_to_y": ["x", "m", "w", "z", "v", "q", "x*w", "x*z", "m*v", "m*q"],
            "x_to_m": ["x", "w", "z", "x*w", "x*z"],
        },
        50: {
            "all_to_y": ["x", "m", "v", "q", "x*v", "x*q", "m*v", "m*q"],
            "x_to_m": ["x", "w", "z", "x*w", "x*z"],
        },
        51: {
            "all_to_y": [
                "x",
                "m",
                "w",
                "z",
                "v",
                "q",
                "x*w",
                "x*z",
                "m*v",
                "m*q",
                "w*z",
                "x*w*z",
            ],
            "x_to_m": ["x", "w", "z", "x*w", "x*z", "w*z", "x*w*z"],
        },
        52: {
            "all_to_y": ["x", "m", "v", "q", "x*v", "x*q", "m*v", "m*q"],
            "x_to_m": ["x", "w", "z", "x*w", "x*z", "w*z", "x*w*z"],
        },
        53: {
            "all_to_y": [
                "x",
                "m",
                "w",
                "z",
                "v",
                "q",
                "x*w",
                "x*z",
                "m*v",
                "m*q",
                "v*q",
                "m*v*q",
            ],
            "x_to_m": ["x", "w", "z", "x*w", "x*z"],
        },
        54: {
            "all_to_y": [
                "x",
                "m",
                "v",
                "q",
                "x*v",
                "x*q",
                "m*v",
                "m*q",
                "v*q",
                "x*v*q",
                "m*v*q",
            ],
            "x_to_m": ["x", "w", "z", "x*w", "x*z"],
        },
        55: {
            "all_to_y": [
                "x",
                "m",
                "w",
                "z",
                "v",
                "q",
                "x*w",
                "x*z",
                "m*v",
                "m*q",
                "w*z",
                "v*q",
                "x*w*z",
                "m*v*q",
            ],
            "x_to_m": ["x", "w", "z", "x*w", "x*z", "w*z", "x*w*z"],
        },
        56: {
            "all_to_y": [
                "x",
                "m",
                "v",
                "q",
                "x*v",
                "x*q",
                "m*v",
                "m*q",
                "v*q",
                "x*v*q",
                "m*v*q",
            ],
            "x_to_m": ["x", "w", "z", "x*w", "x*z", "w*z", "x*w*z"],
        },
        57: {
            "all_to_y": [
                "x",
                "m",
                "w",
                "z",
                "v",
                "q",
                "x*w",
                "x*z",
                "x*v",
                "x*q",
                "m*v",
                "m*q",
                "w*z",
                "v*q",
                "x*v*q",
                "m*v*q",
                "x*w*z",
            ],
            "x_to_m": ["x", "w", "z", "x*w", "x*z", "w*z", "x*w*z"],
        },
        58: {"all_to_y": ["x", "m", "w", "m*w"], "x_to_m": ["x", "w", "x*w"]},
        59: {"all_to_y": ["x", "m", "w", "x*w", "m*w"], "x_to_m": ["x", "w", "x*w"]},
        60: {
            "all_to_y": ["x", "m", "w", "m*w"],
            "x_to_m": ["x", "w", "z", "x*w", "x*z"],
        },
        61: {
            "all_to_y": ["x", "m", "w", "x*w", "m*w"],
            "x_to_m": ["x", "w", "z", "x*w", "x*z"],
        },
        62: {
            "all_to_y": ["x", "m", "w", "z", "x*z", "m*w"],
            "x_to_m": ["x", "w", "z", "x*w", "x*z"],
        },
        63: {
            "all_to_y": ["x", "m", "w", "z", "x*w", "x*z", "m*w"],
            "x_to_m": ["x", "w", "z", "x*w", "x*z"],
        },
        64: {
            "all_to_y": ["x", "m", "w", "v", "m*w", "m*v"],
            "x_to_m": ["x", "w", "x*w"],
        },
        65: {
            "all_to_y": ["x", "m", "w", "v", "x*w", "m*w", "m*v"],
            "x_to_m": ["x", "w", "x*w"],
        },
        66: {
            "all_to_y": ["x", "m", "w", "v", "x*v", "m*w", "m*v"],
            "x_to_m": ["x", "w", "x*w"],
        },
        67: {
            "all_to_y": ["x", "m", "w", "v", "x*w", "x*v", "m*w", "m*v"],
            "x_to_m": ["x", "w", "x*w"],
        },
        68: {
            "all_to_y": ["x", "m", "w", "m*w"],
            "x_to_m": ["x", "w", "z", "x*w", "x*z", "w*z", "x*w*z"],
        },
        69: {
            "all_to_y": ["x", "m", "w", "z", "x*w", "x*z", "m*w", "w*z", "x*w*z"],
            "x_to_m": ["x", "w", "z", "x*w", "x*z", "w*z", "x*w*z"],
        },
        70: {
            "all_to_y": ["x", "m", "w", "v", "m*w", "m*v", "w*v", "m*w*v"],
            "x_to_m": ["x", "w", "x*w"],
        },
        71: {
            "all_to_y": [
                "x",
                "m",
                "w",
                "v",
                "x*w",
                "x*v",
                "m*w",
                "m*v",
                "w*v",
                "x*w*v",
                "m*w*v",
            ],
            "x_to_m": ["x", "w", "x*w"],
        },
        72: {
            "all_to_y": ["x", "m", "w", "z", "m*w", "m*z", "w*z", "m*w*z"],
            "x_to_m": ["x", "w", "z", "x*w", "x*z", "w*z", "x*w*z"],
        },
        73: {
            "all_to_y": [
                "x",
                "m",
                "w",
                "z",
                "x*w",
                "x*z",
                "m*w",
                "m*z",
                "w*z",
                "x*w*z",
                "m*w*z",
            ],
            "x_to_m": ["x", "w", "z", "x*w", "x*z", "w*z", "x*w*z"],
        },
        74: {"all_to_y": ["x", "m", "m*x"], "x_to_m": ["x"]},
        75: {
            "all_to_y": ["x", "m", "w", "z", "m*w", "m*z"],
            "x_to_m": ["x", "w", "z", "x*z", "x*w"],
        },
        76: {
            "all_to_y": ["x", "m", "w", "z", "x*z", "x*w", "m*w", "m*z"],
            "x_to_m": ["x", "w", "z", "x*z", "x*w"],
        },
    }

    def __init__(
            self,
            data,
            model=3,
            modval=None,
            cluster=None,
            boot=5000,
            seed=12345,
            mc=None,
            conf=95,
            effsize=False,
            jn=False,
            hc3=False,
            controls=None,
            controls_in="all",
            total=False,
            contrast=False,
            center=False,
            quantile=False,
            detail=True,
            percent=False,
            logit=False,
            iterate=10000,
            convergence=0.000_000_01,
            precision=4,
            suppr_init=False,
            **kwargs,
    ):
        """
        Initialize a Process model, as defined in Andrew F. Hayes's book 'Introduction to Mediation, Moderation,
        and Conditional Process Analysis'.
        :param data: pd.DataFrame
            A pandas DataFrame containing the data. The name of variables supplied to Process instance must correspond
            to columns in the data. Variable names are case-sensitive.
        :param model: int
            The number of the statistical model to estimate. The full list of models can be found at the
            following address: "http://afhayes.com/public/templates.pdf"
        :param kwargs: dict
            A dictionary with key m, v, w, q, ... mapping each symbol to a variable in the data
        :param modval: dict
            The keys of the dictionary are the name of the variables specified as moderators in the model, and the value
            should be a list of spotlight values for this moderator.
            If a key does not exist in the dictionary, the default spotlight values (quantile or +/- 1SD from the mean,
            see the 'quantile' argument) will be used.
        :param cluster: string
            A variable name that uniquely identifies each of the individuals in the data. Used to generate fixed effects
            for repeated-measures designs.
        :param boot: int
            The number of bootstrap repetitions for the estimation of the SE and CI in indirect effects.
        :param seed: int
            The seed to use for bootstrap samples. Specify an integer between 0 and 1e10 for a replicable seed.
        :param conf: int
            A value between 51 and 99, representing the desired level of confidence for the confidence intervals
        :param effsize: bool
            If True, an estimate of the effect size will be reported.
        :param jn: bool
            If True, the Johnson-Neymann region of significance will be reported.
        :param hc3: bool
            If True, the HC3 estimator will be used for the variance/covariance matrix of the parameters.
        :param controls: list of string
            A list of control variables to include to the model(s).
        :param controls_in: "all", "x_to_m", "all_to_y"
            If 'x_to_m", the controls are added to all the equations from X to the mediator(s) M.
            If 'all_to_y", the controls are added to the equation from X and M to Y.
            if 'all", the controls are added to all equations.
        :param contrast: bool
            If True, pairwise contrasts between all the existing mediators will be generated for Model 6.
        :param total:  bool
            If True, the total effect of all the mediators will be generated for Model 6.
        :param center: bool
            If True, the moderator(s) and variable(s) they moderate will be mean-centered.
        :param quantile: bool
            If True, the spotlight values for each of the moderators will be the [10, 25, 50, 75, 90]
            percentile values of the variable. If False, the spotlight values are equal to
            [M - SD, M, M + SD] for the variable.
        :param detail: bool
            If True, a summary will be printed for each outcome model. If False, this summary will be omitted, and
            only the direct/indirect effects will be printed.
        :param percent: bool
            If True, percentile-based CIs are returned for the indirect effect, instead of bias-corrected CIs.
        :param logit: bool
            If True, and if the outcome is a binary variable, then a logistic regression will be used to
            estimate the parameters of its equation.
        :param iterate: int
            The maximum number of iterations for the Newton-Raphson algorithm of the logistic regression.
        :param convergence: float
            The larger acceptable delta between the old and new parameters before convergence in the
            Newton-Raphson algorithm of the logistic regression.
        :param precision:
            The number of decimal places to display in the summary of the model results.
        """
        if kwargs.pop("mc", None):
            warnings.warn(
                "The argument 'mc' for Monte-Carlo simulations is not supported",
                DeprecationWarning,
            )
        if kwargs.pop("normal", None):
            warnings.warn(
                "The argument 'normal' for normal theory tests is not supported. "
                "Bootstrapped CI are recommended.",
                DeprecationWarning,
            )
        if kwargs.pop("varorder", None):
            warnings.warn(
                "The argument 'varorder' for normal theory tests is not supported. "
                "Bootstrapped CI are recommended.",
                DeprecationWarning,
            )
        if kwargs.pop("varlist", None):
            warnings.warn(
                "The 'varlist' is not required. To specify controls, use the 'controls' arguments",
                DeprecationWarning,
            )
        if kwargs.pop("coeffci", None):
            warnings.warn(
                "The argument 'coeffci' is not supported.", DeprecationWarning
            )
        if kwargs.pop("plot", None):
            warnings.warn(
                "The argument 'plot' is not supported. Check the 'plot_conditional_direct_effects() and"
                "'plot_conditional_indirect_effects()' methods instead.",
                DeprecationWarning,
            )
        if kwargs.pop("save", None):
            warnings.warn(
                "The argument 'save' is not supported. Call the 'get_bootstrap_estimates() method to recover"
                "the bootstrap samples instead.",
                DeprecationWarning,
            )
        if kwargs.pop("effsize", None):
            warnings.warn(
                "The argument 'effsize' for effect sizes is not supported yet."
                "It is coming in future versions of PyProcessMacro.",
                SyntaxWarning,
            )
        if kwargs.pop("jn", None):
            warnings.warn(
                "The argument 'jn' for the Johnson-Neyman region of significance is not supported."
                "Call the 'floodlight_direct_effect()' and 'floodlight_indirect_effect()' methods instead.",
                DeprecationWarning,
            )

        if model == 6:
            raise NotImplementedError(
                "The model 6 for serial mediation is not supported yet."
                "It is coming in future versions of PyProcessMacro."
            )

        if not isinstance(data, pd.DataFrame):
            raise ValueError(
                "The variable provided for data is not a valid pd.DataFrame object. Please provide a"
                "valid Pandas DataFrame as the 'data' argument."
            )
        if not controls:
            controls = []
        if not modval:
            modval = {}

        self.model_num = model

        self.controls = controls
        self._data = data
        arguments = locals()

        # Validate the arguments supplied as options
        self.options = self._gen_valid_options(arguments)

        # Check the congruence between the model specifications, the model number, and the data, and store the final
        # list of variables used
        var_kwargs = {k: v for k, v in kwargs.items() if k in self.__var_kws__}

        self.mediators = var_kwargs.get("m")
        self.iv = var_kwargs.get("y")

        self.varlist = self._gen_valid_varlist(var_kwargs)

        # Check the number of mediators supplied to the model:
        if self.model_num > 3:
            self.n_meds = len(self.mediators)
            self.has_mediation = True
        else:
            self.n_meds = 0  # 1, 2 and 3 are moderation-only
            self.has_mediation = False

        # Creating the dictionary mapping each variable to its symbol (y, x, m, w, ...)
        self._var_to_symb, self._symb_to_var = self._gen_var_mapping(var_kwargs)

        # Identify the moderators of the different variables for the different paths
        raw_equations = copy.deepcopy(self.__models_eqs__[model])
        raw_varlist = copy.deepcopy(self.__models_vars__[model])

        self._moderators = gen_moderators(raw_equations, raw_varlist)

        # Generating the equations used for estimation: a list of (exog, endog) tuples.
        self._equations = self._gen_equations(
            raw_equations["all_to_y"], raw_equations["x_to_m"], controls_in=controls_in
        )

        # Prepare the data: drop NaN, rename the columns, add a constant, and mean-center moderators if needed.
        self._data, self.n_obs, self.n_obs_null, self.centered_vars = (
            self._prepare_data()
        )

        # Now that the data is ready, create the dictionary mapping each symbol to its column index.
        self._symb_to_ind = {v: i for i, v in enumerate(self._data.columns.values)}

        # Generate the statistical models that will be estimated
        self.outcome_models = self._gen_outcome_models()

        # Rename the dictionary of custom spotlight values, and generating the spotlight values.
        modval_symb = {self._var_to_symb.get(k): v for k, v in modval.items()}
        self._spotlight_values = self._gen_spotlight_values(modval_symb)

        # Generate the direct model.
        self.direct_model = self._gen_direct_effect_model()

        # Generate the indirect model (if there is one).
        if self.has_mediation:
            self.indirect_model = self._gen_indirect_effect_model()
        else:
            self.indirect_model = None

        # Show the initialized Process instance.
        if not suppr_init:
            self._print_init()

    def _gen_valid_options(self, arguments):
        """
        Validate the arguments specified for the different options used in Process.
        :return:
        """
        options = {k: arguments.get(k) for k in self.__options_kws__}
        errstr = ""
        conf = options["conf"]
        seed = options["seed"]
        if not isinstance(conf, int) or ((conf <= 50) or (conf >= 100)):
            errstr += (
                "The option 'conf' must be an integer between 50 and 100, exclusive.\n"
            )

        if not isinstance(seed, int) or ((seed <= 0) or (seed >= 1e9)):
            errstr += "The option 'seed' must be  an integer between 0 and 1 000 000 000, exclusive.\n"

        if options["contrast"] not in [True, False]:
            errstr += "The option 'contrast' must be 'True' or 'False'.\n"
        if options["total"] not in [True, False]:
            errstr += "The option 'total' must be 'True' or 'False'.\n"
        if options["jn"] not in [True, False]:
            errstr += "The option 'jn' must be 'True' or 'False'.\n"
        if options["hc3"] not in [True, False]:
            errstr += "The option 'hc3' must be 'True' or 'False'.\n"
        if options["center"] not in [True, False]:
            errstr += "The option 'center' must be 'True' or 'False'.\n"
        if options["quantile"] not in [True, False]:
            errstr += "The option 'jn' must be 'True' or 'False'.\n"
        if options["jn"] not in [True, False]:
            errstr += "The option 'jn' must be 'True' or 'False'.\n"
        if options["logit"] not in [True, False]:
            errstr += "The option 'logit' must be 'True' or 'False'.\n"
        if options["controls_in"] not in ["all", "x_to_tm", "all_to_y"]:
            errstr += "The option 'controls_in' should be one of 'all', 'x_to_m', 'all_to_y'\n"
        if not isinstance(options["modval"], dict):
            errstr += "The option 'modval' must be a dictionary.\n"
        else:
            for k, v in options["modval"].items():
                if not isinstance(v, list):
                    errstr += f"The value associated with {k} in the dictionary 'modval' is not a list.\n"
                else:
                    try:
                        [float(i) for i in v]
                    except (ValueError, TypeError):
                        errstr += f"The value of {k} in the dictionary 'modval' is not a list of numbers.\n"

        if errstr != "":
            raise ValueError(
                f"""Some errors were found in the options specified. Please correct the following error(s):
            {errstr}
            """
            )
        return options

    def _gen_valid_varlist(self, var_kwargs):
        """
        Validate the arguments and data supplied to the Process Model, by checking:
            1. That the arguments match the model specification
            2. That the arguments are present in the list of variables supplied to the model
            3. That the variables supplied as arguments are columns in the DataFrame
        The complete list of variables used in the model is then returned.
        :return: list
            The list of variables used in Process.
        """
        # Check the number of variables supplied as argument, and convert them to lists if needed.
        for k, v in var_kwargs.items():
            if isinstance(v, str):
                var_kwargs[k] = [v]
            if isinstance(v, list):
                if ((k != "m") & (len(v) != 1)) or (
                        (self.model_num <= 3) & (len(v) != 1)
                ):
                    raise ValueError(
                        f"Several variable names have been specified for '{k}'. \n"
                        f"The Model {self.model_num} accepts only one variable for '{k}'"
                    )

        var_kwargs_set = set(var_kwargs.keys())
        expected_var_kwargs = self.__models_vars__[self.model_num]

        # Validate the Model specification.
        if var_kwargs_set != expected_var_kwargs:
            miss_vars = expected_var_kwargs - var_kwargs_set
            exced_vars = var_kwargs_set - expected_var_kwargs
            err_str = f"""The variables supplied do not match the definition of Model {self.model_num}
            """
            if miss_vars:
                err_str += f"Expected variable(s) not supplied: {', '.join(miss_vars)}"
            if exced_vars:
                err_str += f"Variable(s) supplied not supported by the model: {', '.join(exced_vars)}"
            raise ValueError(err_str)

        var_kwargs_names = []
        for _, v in var_kwargs.items():
            var_kwargs_names.extend(v)

        varlist = list(set(self.controls) | set(var_kwargs_names))
        # Validate the presence of variables in the data.
        miss_var_names = set(varlist) - set(self._data.columns)
        if miss_var_names:
            raise ValueError(
                f"""One or several of the variables supplied to the model is/are not present in the
            DataFrame supplied. Please double check that '{", ".join(miss_var_names)}' is/are in the DataFrame."""
            )
        return varlist

    def _gen_var_mapping(self, var_kwargs):
        """
        Map each dependent variable, independent variable, moderator, and mediator supplied to the model
        to a unique symbol:
            The dependent variable is mapped to y, and dependent variable is mapped to x.
            The mediators are coded m1, m2, m3...
            The moderators are coded by their respective letters (w, z, v, q). When m is a moderator, it is coded m1.
            The control variables are coded c1, c2, c3...
        :return: tuple (var_to_symb, symb_to_var)
            The dictionaries mapping the variable name to their symbols, and vice versa
        """
        var_to_symb = {
            "Cons": "Cons"
        }  # Map each variable name to its symbolic representation.
        symb_to_var = {"Cons": "Cons"}

        for key, varnames in var_kwargs.items():
            if (key == "m") & (self.model_num > 3):  # Variable m is a mediator
                for i, v in enumerate(varnames):
                    symb_to_var[f"{key}{i + 1}"] = v
                    var_to_symb[v] = f"{key}{i + 1}"
            else:  # All other variables are moderators_all, IVs, or DVs.
                symb_to_var[f"{key}"] = varnames[0]
                var_to_symb[varnames[0]] = f"{key}"

        for i, v in enumerate(self.controls):
            var_to_symb[v] = f"c{i}"
            symb_to_var[f"c{i}"] = v

        return var_to_symb, symb_to_var

    def _gen_equations(self, all_to_y, x_to_m, controls_in="all"):
        """
        Generate a list of tuple (exogterm, endogterms) that define the equations to estimate the full path to Y
        and the full path to the mediators (if relevant)
            Model 1, 2, 3: No mediator, and therefore only one equation to estimate the full path to Y.
                eqlist[0]: (y, full_path_to_y)
            Model 6: Serial mediation. One equation to estimate the full path to Y, and N equations to estimate
            each of the serial mediation paths.
                eqlist[0] = (y, full_path_to_y)
                eqlist[1:] = (Mi, full_path_to_Mi) for i in M1, M2, ... MN
            Model 4, 5, and 7+: Parallel mediation. One equation to estimate the full path to Y, and N identical
            equations to estimate each of the parralel mediators.
                eqlist[0] = (y, full_path_to_y)
                eqlist[1:] = (Mi, full_path_to_M) for i in M1, M2, ... MN
        """

        all_to_y_base = all_to_y.copy()
        x_to_m_base = x_to_m.copy()

        eqlist = []
        control_vars = [v for v in self._symb_to_var.keys() if v[0] == "c"]

        if controls_in == "all":
            all_to_y_base += control_vars
            x_to_m_base += control_vars
        elif controls_in == "x_to_m":
            x_to_m_base += control_vars
        elif controls_in == "all_to_y":
            all_to_y_base += control_vars

        if self.model_num <= 3:  # No mediators, just the equation for Y to estimate.
            eqlist.append(("y", ["Cons"] + all_to_y_base))
            return eqlist

        elif self.model_num != 6:  # Parallel mediators
            all_to_y_full = ["Cons"] + [i for i in all_to_y_base if "m" not in i]
            x_to_m_full = ["Cons"] + [i for i in x_to_m_base if "m" not in i]

            # Substitute the individual mediation terms to the general term "m" in the equations
            for i in range(self.n_meds):
                for term in all_to_y_base:
                    if "m" in term:
                        all_to_y_full.append(term.replace("m", f"m{i + 1}"))
                    else:
                        pass

            eqlist.append(("y", all_to_y_full))  # Full path to Y
            eqlist += [
                (f"m{i + 1}", x_to_m_full) for i in range(self.n_meds)
            ]  # Same equation for all mediators
            return eqlist

        else:  # Serial mediators are only found in model 6
            eq_y_full = ["Cons"] + [i for i in all_to_y if "m" not in i]
            eq_x_to_meds_basis = ["Cons"] + [i for i in x_to_m]

            # Substitute the individual mediation terms to the general term "m" in the equations
            for i in range(self.n_meds):
                for term in all_to_y:
                    if "m" in term:
                        eq_y_full.append(term.replace("m", f"m{i + 1}"))
                    else:
                        pass

            eqlist.append(("y", eq_y_full))  # Equation for Y
            for i in range(self.n_meds):  # One equation per mediator
                # Add the previous mediators as predictors
                eq_x_to_med = eq_x_to_meds_basis + [f"m{i + 1}" for _ in range(i)]
                eqlist.append((f"m{i + 1}", eq_x_to_med))
        return eqlist

    def _prepare_data(self):
        """
        Subset the dataframe to the columns needed for estimation purposes, add a constant, and drop missing
        observations from the dataset.
        :return: pd.DataFrame
        """
        # Subset the data to the columns used in the model
        data = self._data[self.varlist].copy()
        n_obs_before = self._data.shape[0]
        data = data.dropna().reset_index()
        n_obs_after = self._data.shape[0]
        n_obs_null = n_obs_before - n_obs_after

        # Map each variable name to a unique variable code, and rename the columns in the data.)
        data.rename(columns=self._var_to_symb, inplace=True)

        # Adding a constant to the data.
        data["Cons"] = 1

        if self.options["logit"]:
            endog = data["y"]
            uniques = np.unique(endog)
            if len(uniques) != 2:
                raise ValueError(
                    "The dependent variable does not have exactly two distinct outcomes."
                    "Please provide another dataset or change the 'logit' option to 0"
                )
            else:
                endog_logit = [0 if i == uniques[0] else 1 for i in endog]
            data["y"] = endog_logit

        centered_vars = []
        if self.options["center"]:
            mod_m = self._moderators["m"]
            mod_x = self._moderators["x_indirect"] | self._moderators["x_direct"]
            if mod_m:
                centered_vars += list(mod_m) + list(
                    {c for c in data.columns if "m" in c}
                )
            if mod_x:
                centered_vars += list(mod_x) + ["x"]
            data.loc[:, centered_vars] = data.loc[:, centered_vars].subtract(
                data.loc[:, centered_vars].mean()
            )

        # Now we add the interaction columns
        eqs = self._equations
        intterms = [
            term for _, eq in eqs for term in eq if "*" in term
        ]  # All interaction terms in all equations.
        for term in intterms:
            termlist = term.split("*")  # List of terms in interaction
            data[term] = data.loc[:, termlist].product(axis=1)
            termname = "*".join(
                [self._symb_to_var[i] for i in termlist]
            )  # Full name of interaction
            self._var_to_symb[termname] = term
            self._symb_to_var[term] = termname

        return data, n_obs_after, n_obs_null, centered_vars

    def _gen_outcome_models(self):
        """
        Instantiate the OLS/Logit models for all outcomes (Y, and the mediators when they exist)
        :return: dict
            A dictionary of outcome_models, with keys corresponding to the name(s) of independent variable
            and mediator(s)
        """

        data_array = self._data.values
        models = {}

        # Generating the model for Y
        y_equation = self._equations[0]
        y_endog = y_equation[0]
        y_exogs = y_equation[1]
        if self.options["logit"]:
            model_yfull = LogitOutcomeModel(
                data_array,
                y_endog,
                y_exogs,
                self._symb_to_ind,
                self._symb_to_var,
                self.options,
            )
        else:
            model_yfull = OLSOutcomeModel(
                data_array,
                y_endog,
                y_exogs,
                self._symb_to_ind,
                self._symb_to_var,
                self.options,
            )
        models[self.iv] = model_yfull

        if self.model_num > 3:  # Generating the model "Direct path to Y" (if it exists)
            for eq, med_name in zip(self._equations[1:], self.mediators):
                m_endog = eq[0]
                m_exogs = eq[1]
                model_m = OLSOutcomeModel(
                    data_array,
                    m_endog,
                    m_exogs,
                    self._symb_to_ind,
                    self._symb_to_var,
                    self.options,
                )
                models[med_name] = model_m
        return models

    def _gen_spotlight_values(self, modval_symb):
        """
        Generate the spotlight values of the moderators.
        The function first search the dictionary 'modval' for custom spotlight values.
        If they cannot be found, it uses discrete values if the moderator is discrete.
        If the moderator is not discrete, is uses quantiles or mean +/- 1SD, depending on the options specified
        in the Process object.
        :param modval_symb: dict
            A dictionary {'mod_symb':[mod_values]} of custom spotlight values for the moderator(s)
        :return: dict
            A dictionary {'mod_symb':[spot_values]} of spotlight values for all the moderator(s) of the model
        """
        spot_values = {}
        data = self._data.values
        for mod in self._moderators["all"]:
            index = self._symb_to_ind[mod]
            val = data[:, index]
            spotvals = modval_symb.get(mod, [])
            if not spotvals:
                if len(np.unique(val)) <= 5:
                    spot_values[mod] = np.unique(val)
                elif self.options["quantile"]:
                    spot_values[mod] = np.percentile(val, [10, 25, 50, 75, 90])
                else:
                    m = val.mean()
                    std = val.std(ddof=1)
                    spot_values[mod] = [m - std, m, m + std]
            else:
                spot_values[mod] = spotvals
        return spot_values

    def _gen_direct_effect_model(self):
        """
        Generate the Direct Effect Model for the direct path.
        :return: a DirectEffectModel object, properly initialized.
        """
        mod_names = self._moderators["x_direct"]
        model_iv = self.outcome_models[self.iv]
        dem = DirectEffectModel(
            model_iv,
            mod_names,
            self._spotlight_values,
            self.has_mediation,
            self._symb_to_var,
            self.options,
        )
        return dem

    def _gen_indirect_effect_model(self):
        """
        Generate the Parallel Mediation Model for the indirect path.
        :return: a ParallelMediationModel object, properly initialized.
        """
        y_exogvars = self._equations[0][1]
        m_exogvars = self._equations[1][1]
        data_array = self._data.values
        mod_symb = self._moderators["indirect"]
        spot_values = self._spotlight_values
        analysis_list = self._gen_analysis_list()
        iem = ParallelMediationModel(
            data_array,
            y_exogvars,
            m_exogvars,
            mod_symb,
            spot_values,
            self.n_meds,
            analysis_list,
            self._symb_to_ind,
            self._symb_to_var,
            self.options,
        )
        return iem

    def _print_init(self):
        """
        Print the initialization information.
        :return: None
        """
        initstr = (
            "Process successfully initialized.\n"
            "Based on the Process Macro by Andrew F. Hayes, Ph.D. (www.afhayes.com)\n\n"
            "\n****************************** SPECIFICATION ****************************\n\n"
            f"Model = {self.model_num}\n\n"
            "Variables:"
        )
        parameters = [
            (symb, name)
            for (symb, name) in self._symb_to_var.items()
            if "*" not in symb and "c" not in symb
        ]
        for symb, name in parameters:
            if "*" in symb or "c" in symb:
                pass
            else:
                initstr += f"\n    {symb} = {name}"
        controls = [name for (symb, name) in self._symb_to_var.items() if "c" in symb]
        if controls:
            initstr += f"\nStatistical Controls:\n {', '.join(controls)}\n\n"

        meancentered = self.centered_vars
        if meancentered:
            initstr += f"\nMean-centered variables:\n {', '.join([self._symb_to_var.get(v) for v in meancentered])}\n\n"

        initstr += f"\n\nSample size:\n{self.n_obs}"
        if self.n_obs_null:
            initstr += f" ({self.n_obs_null} observations removed due to missingness)"
        if self.has_mediation:
            initstr += (
                "\n\nBootstrapping information for indirect effects:\n"
                f"Final number of bootstrap samples: {self.options['boot']}\n"
                f"Number of samples discarded due to convergence issues: {self.indirect_model._n_fail_samples}"
            )

        print(initstr)

    def _gen_analysis_list(self):
        """
        Process always report the indirect effect/conditional indirect effect(s).
        If one moderator is present on the indirect path, Process will report the Moderated Mediation Index (MM).
        If exactly two moderators are present on the indirect path, additional analysis could be reported: the
        Conditional Moderated Mediation (CMM), the Moderated Moderated Mediation (MMM), and the
        Partial Moderated Mediation (PMM). This function establishes which of those statistics should be reported:
          1. If the two moderators are on two different paths (X to M, or M to Y): both CMM and MMM are reported.
          2. If the two moderators are on the same path and form a 3-way interaction: both CMM and MMM are reported.
          3. If the two moderators are on the same path and do not form a 3-way: the PMM is reported.
          4. If at least one of the two moderators is present on both paths: no analysis is reported.

        This function returns the list of additional analysis to report. If no additional analysis must be performed,
         this list is empty.
        :return: list
            ["MM"], ["MMM", "CMM"], ["PMM"] or []
        """
        n_mods_m = len(self._moderators["m"])
        n_mods_ind = len(self._moderators["indirect"])

        y_exogvars = self._equations[0][1]
        m_exogvars = self._equations[1][1]

        if n_mods_ind == 0:  # No moderators on indirect path, so no additional analysis.
            return []

        terms = y_exogvars + m_exogvars
        threeway = any(
            [1 if len(term.split("*")) == 3 else 0 for term in terms]
        )  # Find if there is any three-way interaction

        if n_mods_ind == 1:  # One single moderator, moderated mediation analysis
            return ["MM"]
        elif n_mods_ind == 2:
            if (n_mods_m == 1) or threeway:  # Moderators on two different paths
                return ["MMM", "CMM"]
            else:
                return ["PMM"]
        else:
            return []

    def _parse_moderator_values(self, x, hue, row, col, modval, path):
        """
        Parse the moderator values supplied to a direct/indirect effect plot. Used in `plot_conditional_direct_effects`
        and `plot_conditional_indirect_effects`
        :param x: string
            Name of the moderator to plot on x-axis
        :param hue: string
            Name of the moderator to color-code
        :param row:
        :param col:
        :param modval:
        :param path:
        :return:
        """
        modval_symb = {self._var_to_symb[k]: v for k, v in modval.items()}
        spotlight_values_symb = self._spotlight_values.copy()

        # Values for x-axis
        x_symb = self._var_to_symb[x]
        x_values = modval_symb.get(x_symb)

        if x_values is None:
            xdata = self._data[x_symb]
            if len(np.unique(xdata)) == 2:
                x_values = np.unique(xdata)
            else:
                x_values = np.linspace(xdata.min(), xdata.max(), 100)

        # Values for hue
        if isinstance(hue, str):
            huevar1 = hue
            huesymb1 = self._var_to_symb[huevar1]
            huevar2 = None
            hue1_values = modval_symb.get(huesymb1,
                                          spotlight_values_symb[huesymb1])
            hue2_values = [0]

        elif isinstance(hue, list):
            if len(hue) == 2:
                huevar1 = hue[0]
                huevar2 = hue[1]
                huesymb1 = self._var_to_symb[huevar1]
                huesymb2 = self._var_to_symb[huevar2]
                hue1_values = modval_symb.get(huesymb1,
                                              spotlight_values_symb[huesymb1])
                hue2_values = modval_symb.get(huesymb2,
                                              spotlight_values_symb[huesymb2])
            elif len(hue) == 1:
                huevar1 = hue[0]
                huesymb1 = self._var_to_symb[huevar1]
                huevar2 = None
                hue1_values = modval_symb.get(huesymb1,
                                              spotlight_values_symb[huesymb1])
                hue2_values = [0]
        else:
            huevar1 = None
            huevar2 = None
            hue1_values = [0]
            hue2_values = [0]

        col_symb = self._var_to_symb[x]
        col_values = modval_symb.get(col_symb, spotlight_values_symb.get(col_symb, [0]))

        row_symb = self._var_to_symb[x]
        row_values = modval_symb.get(row_symb, spotlight_values_symb.get(row_symb, [0]))

        mod_names = [x, huevar1, huevar2, col, row]
        mod_values = [x_values, hue1_values, hue2_values, col_values, row_values]

        modval_vars = {n: v for (n, v) in zip(mod_names, mod_values) if n is not None}
        modval_others = {k: v for k, v in modval.items() if k not in modval_vars}
        modval_others_invalid = any([len(v) > 1 for k, v in modval_others.items()])

        if modval_others_invalid:
            raise SyntaxError(
                "You cannot specify more than one focal value for moderator that is not displayed in the graph."
            )

        modval_parsed = {**modval_others, **modval_vars}

        for m in self._moderators[path]:
            m_var = self._symb_to_var[m]
            if modval_parsed.get(m_var) is None:
                warnings.warn(
                    f"The moderator {m_var} exerts an influence on the effect, but is not specified as a factor on\
                     the graph. Its value has been explicitely set to 0.",
                    SyntaxWarning,
                )
                modval_parsed[m_var] = [0]
        return modval_parsed

    # API
    def summary(self):
        """
        Print the summary of the Process model.
        :return: None
        """
        with pd.option_context("precision", self.options["precision"]):
            full_model = self.outcome_models[self.iv]
            m_models = [
                self.outcome_models.get(med_name) for med_name in self.mediators
            ]
            if self.options["detail"]:
                print(
                    "\n***************************** OUTCOME MODELS ****************************\n"
                )
                print(full_model)
                print(
                    "\n-------------------------------------------------------------------------\n"
                )
                if self.model_num > 3:
                    for med_model in m_models:
                        print(med_model)
                        print(
                            "\n-------------------------------------------------------------------------\n"
                        )
            if self.indirect_model:
                print(
                    "\n********************** DIRECT AND INDIRECT EFFECTS **********************\n"
                )
                print(self.direct_model)
                print(self.indirect_model)
            else:
                print(
                    "\n********************** CONDITIONAL EFFECTS **********************\n"
                )
                print(self.direct_model)

    def get_bootstrap_estimates(self):
        if not self.has_mediation:
            raise NotImplementedError(
                "This model does not have a mediation. As such, no bootstrap samples were "
                "generated."
            )
        iem = self.indirect_model
        stv = self._symb_to_var

        boot_betas_y = iem._boot_betas_y
        boot_betas_m = iem._boot_betas_m

        cols_y = [stv[t] for t in iem._exog_terms_y]
        df = pd.DataFrame(boot_betas_y, columns=cols_y)
        df["___"] = stv["y"]
        for i in range(self.n_meds):
            cols_m = [stv[t] for t in iem._exog_terms_m]
            df_m = pd.DataFrame(boot_betas_m[i], columns=cols_m)
            df_m["___"] = stv[f"m{i + 1}"]
            df = df.append(df_m)
        df.index.name = "BootSample"
        df.insert(0, "OutcomeName", df["___"].values)
        del df["___"]
        return df.reset_index()

    def floodlight_indirect_effect(
            self, med_name, mod_name, other_modval=None, atol=1e-8, rtol=1e-5
    ):
        """
        Generate a floodlight analysis to find the Johnson-Neyman point(s) of significance of the indirect effect
        :param med_name: string
            The name of the mediator on which to perform the floodlight analysis
        :param mod_name: string
            The name of the moderator on which to perform the floodlight analysis
        :param other_modval: dict of floats or None
            The conditional value of the other moderator(s). Only a single value can be given for each moderator
        :param atol: float
            The absolute error tolerance for convergence
        :param rtol: float
            The relative error tolerance for convergence
        :return: IndirectFloodlightAnalysis
            An IndirectFloodlightAnalysis object
        """
        mod_symb = self._var_to_symb.get(mod_name)
        if not mod_symb:
            raise ValueError(f"The variable {mod_name} is not a variable in the model.")

        other_modval_symb = {}
        if other_modval:
            for k, v in other_modval.items():
                symb = self._var_to_symb.get(k)
                if not mod_symb:
                    raise ValueError(
                        f"The variable {mod_name} is not a variable in the model."
                    )
                if isinstance(v, list):
                    raise ValueError(
                        f"The spotlight value for the variable {k} should be a number, not a list."
                    )
                else:
                    other_modval_symb[symb] = v

        if not self.has_mediation:
            raise ValueError(f"Model {self.model_num} does not include a mediator.")

        if mod_symb not in self._moderators["indirect"]:
            raise ValueError(
                f"The variable {mod_name} does not moderate the indirect path in Model {self.model_num}."
            )

        for ms in self._moderators["indirect"]:
            if (ms != mod_symb) and (ms not in other_modval_symb.keys()):
                other_modval_symb[ms] = 0

        if not med_name:
            raise ValueError(
                "You must specify the name of the mediator for which to plot the indirect effects"
            )

        try:
            med_index = self.mediators.index(med_name)
        except ValueError:
            raise ValueError(f"The variable {med_name} is not a mediator in the model.")

        mod_values = self._data[mod_symb]
        modval_range = [min(mod_values), max(mod_values)]
        sig_regions = self.indirect_model._floodlight_analysis(
            med_index, mod_symb, modval_range, other_modval_symb, atol=atol, rtol=rtol
        )
        prec = self.options["precision"]
        other_modval_name = {
            self._symb_to_var[k]: v for k, v in other_modval_symb.items()
        }
        return IndirectFloodlightAnalysis(
            med_name, mod_name, sig_regions, modval_range, other_modval_name, prec
        )

    def floodlight_direct_effect(
            self, mod_name, other_modval=None, atol=1e-8, rtol=1e-5
    ):
        """
        Generate a floodlight analysis to find the Johnson-Neyman point(s) of significance of the direct effect
        :param mod_name: string
            The name of the moderator on which to perform the floodlight analysis
        :param other_modval: dict of floats or None
            The conditional value of the other moderator(s). Only a single value can be given for each moderator
        :param atol: float
            The absolute error tolerance for convergence
        :param rtol: float
            The relative error tolerance for convergence
        :return: DirectFloodlightAnalysis
            A DirectFloodlightAnalysis object
        """
        mod_symb = self._var_to_symb.get(mod_name)
        if not mod_symb:
            raise ValueError(f"The variable {mod_name} is not a variable in the model.")

        other_modval_symb = {}
        if other_modval:
            for k, v in other_modval.items():
                symb = self._var_to_symb.get(k)
                if not mod_symb:
                    raise ValueError(
                        f"The variable {mod_name} is not a variable in the model."
                    )
                if isinstance(v, list):
                    raise ValueError(
                        f"The spotlight value for the variable {k} should be a number, not a list."
                    )
                else:
                    other_modval_symb[symb] = v

        if mod_symb not in self._moderators["x_direct"]:
            raise ValueError(
                f"The variable {mod_name} does not moderate the direct path in Model {self.model_num}."
            )

        for ms in self._moderators["x_direct"]:
            if (ms != mod_symb) and (ms not in other_modval_symb.keys()):
                other_modval_symb[ms] = 0

        mod_values = self._data[mod_symb]
        modval_range = [min(mod_values), max(mod_values)]
        sig_regions = self.direct_model._floodlight_analysis(
            mod_symb, modval_range, other_modval_symb, atol=atol, rtol=rtol
        )
        other_modval_name = {
            self._symb_to_var[k]: v for k, v in other_modval_symb.items()
        }
        prec = self.options["precision"]
        return DirectFloodlightAnalysis(
            mod_name, sig_regions, modval_range, other_modval_name, prec
        )

    def spotlight_indirect_effect(self, med_name, spotval=None):
        """
        Conduct a spotlight analysis of an indirect effect, at different level(s) of its moderator(s).
        :param med_name: string
            The name of the mediator for which to compute the conditional indirect effect(s).
        :param spotval: dict or None
            A dictionary of {'mod_name': [spot_val_1, ..., spot_val_2]} of spotlight levels for the moderators.
            By default, the spotlight values of the moderators are the same as the ones specified when initializing
            Process.
        :return:
            A DataFrame of indirect Effects/SE/LLCI/ULCI, at various levels of the moderators.
        """
        if not self.has_mediation:
            raise ValueError(f"Model {self.model_num} does not include a mediator.")

        if len(self._moderators["indirect"]) == 0:
            raise ValueError(
                f"The indirect path of Model {self.model_num} does not include a moderator."
            )

        if not med_name:
            raise ValueError(
                "You must specify the name of the mediator for which to plot the indirect effects"
            )

        try:
            med_index = self.mediators.index(med_name)
        except ValueError:
            raise ValueError(f"The variable {med_name} is not a mediator in the model.")

        spotval_symb = self._spotlight_values.copy()

        if spotval:
            for mod_name, mod_val in spotval.items():
                mod_symb = self._var_to_symb.get(mod_name)
                if not mod_symb:
                    raise ValueError(
                        f"The variable {mod_name} is not a variable in the model."
                    )
                else:
                    spotval_symb[mod_symb] = mod_val

        names, v = zip(*spotval_symb.items())
        values = np.array([i for i in product(*v)])
        effect, _, se, llci, ulci = self.indirect_model._get_conditional_indirect_effects(
            med_index, names, values
        )

        rows = np.array([effect, se, llci, ulci]).T

        df1 = pd.DataFrame(rows, columns=["Effect", "Boot SE", "LLCI", "ULCI"])
        df2 = pd.DataFrame(values, columns=names)
        df = df1.join(df2, how="outer")

        stv = self._symb_to_var
        df.rename(columns=lambda c: stv.get(c, c), inplace=True)
        return df

    def spotlight_direct_effect(self, spotval=None):
        """
        Conduct a spotlight analysis of the direct effect, at different level(s) of its moderator(s).
        :param spotval: dict or None
            A dictionary of {'mod_name': [mod_val_1, ..., mod_val_2]} for custom levels of the moderators. By default,
            the values of the moderators are the same as the ones specified when initializing Process.
        :return:
            A DataFrame of direct Effects/SE/LLCI/ULCI, at various levels of the moderators.
        """
        if not self._moderators["x_direct"]:
            raise ValueError(
                f"The direct path of Model {self.model_num} does not include a moderator."
            )

        spotval_symb = self._spotlight_values.copy()

        if spotval:
            for mod_name, mod_val in spotval.items():
                mod_symb = self._var_to_symb.get(mod_name)
                if not mod_symb:
                    raise ValueError(
                        f"The variable {mod_name} is not a variable in the model."
                    )
                else:
                    spotval_symb[mod_symb] = mod_val

        names, v = zip(*spotval_symb.items())
        values = np.array([i for i in product(*v)])

        effect, se, llci, ulci = self.direct_model._get_conditional_direct_effects(
            names, values
        )

        rows = np.array([effect, se, llci, ulci]).T

        df1 = pd.DataFrame(rows, columns=["Effect", "SE", "LLCI", "ULCI"])
        df2 = pd.DataFrame(values, columns=names)
        df = df1.join(df2, how="outer")

        stv = self._symb_to_var
        df.rename(columns=lambda c: stv.get(c, c), inplace=True)
        return df

    def plot_conditional_direct_effects(
            self,
            x=None,
            hue=None,
            row=None,
            col=None,
            modval=None,
            errstyle="band",
            hue_format=None,
            facet_kws=None,
            plot_kws=None,
            err_kws=None,
    ):
        """
        Plot the conditional direct effects of the IV, at specified values of the moderator(s).
        The functions relies on Seaborn's FacetGrid object, to represent complex interactions between up to 5 different
        moderators.

        :param x: string
            The name of the moderator which levels should be represented on the x-axis of the plot
        :param hue: string or list or None
            The name(s) (up to two) of the moderators which (pairs of) levels should be color-coded on the plot.
        :param row: string or None
            If not None, multiple plots will be created on the horizontal axis, as many as there are levels of the
            moderator 'row'.
        :param col: string or None
            If not None, multiple plots will be created on the vertical axis, as many as there are levels of the
            moderator 'row'.
        :param modval: dict or None
            A dictionary of {'mod_name': [mod_val_1, ..., mod_val_2]} for custom levels of the moderators. By default,
            the values of the moderators are the same as the ones specified when initializing Process.
        :param errstyle: 'band', 'ci', or 'none'
            How to represent the confidence interval for the indirect effect. The type of CI corresponds to the type
            specified at the initialization of the Process object.
                If 'band', a confidence band is drawn on the plot.
                If 'ci', confidence intervals are drawn at each discrete value of the moderator on the x-axis
                If 'none', no confidence interval is drawn.
        :param hue_format: string or None
            By default, the color-code are labeled:
                'Mod1 at val1' if there is one moderator for 'hue'.
                'Mod2 at val1, Mod2 at val2' if there are two moderators for 'hue'.
            Alternatively, a string that should be formatted can be passed. The string will receive as arguments:
                var1 (the name of the first moderator)
                var2 (the name of the second moderator, if it exists)
                val1 (the value of the first moderator)
                val2 (the value of the second moderator)
            A valid string would for instance look like this: '{var1} = {val1:.4f}, {var2} = {val2:.4f}'
        :param facet_kws: dict
            A dictionary of arguments that should be passed to the FacetGrid object (such as sharex, sharey, size,
            aspect...)
        :param plot_kws:
            A dictionary of arguments to be passed to the 'plt.scatter' function (such as linestyle, linewidth...)
        :param err_kws:
            A dictionary of arguments to be passed to the 'plt.fill_between' (if errstyle='band') or to the
            'plt.errorbar' (if errstyle='ci') (such as alpha, capthick, capsize...)
        :return: a FacetGrid object
        """

        if not x:
            raise ValueError("You must specify at least one moderator for 'x'")
        if modval is None:
            modval = {}

        modval_parsed = self._parse_moderator_values(
            x, hue, row, col, modval, "x_direct"
        )
        df_effects = self.spotlight_direct_effect(spotval=modval_parsed)
        return plot_conditional_effects(
            df_effects,
            x,
            hue,
            row,
            col,
            errstyle,
            hue_format,
            facet_kws,
            plot_kws,
            err_kws,
        )

    def plot_conditional_indirect_effects(
            self,
            med_name=None,
            x=None,
            hue=None,
            row=None,
            col=None,
            modval=None,
            errstyle="band",
            hue_format=None,
            facet_kws=None,
            plot_kws=None,
            err_kws=None,
    ):
        """
        Plot the conditional indirect effect for a given mediator, at specified values of the moderator(s).
        The functions relies on Seaborn's FacetGrid object, to represent complex interactions between up to 5 different
        moderators.

        :param med_name: string
            The name of the mediator for which to plot the indirect effect(s).
        :param x: string
            The name of the moderator which levels should be represented on the x-axis of the plot
        :param hue: string or list or None
            The name(s) (up to two) of the moderators which (pairs of) levels should be color-coded on the plot.
        :param row: string or None
            If not None, multiple plots will be created on the horizontal axis, as many as there are levels of the
            moderator 'row'.
        :param col: string or None
            If not None, multiple plots will be created on the vertical axis, as many as there are levels of the
            moderator 'row'.
        :param modval: dict or None
            A dictionary of {'mod_name': [mod_val_1, ..., mod_val_2]} for custom levels of the moderators. By default,
            the values of the moderators are the same as the ones specified when initializing Process.
        :param errstyle: 'band', 'ci', or 'none'
            How to represent the confidence interval for the indirect effect. The type of CI corresponds to the type
            specified at the initialization of the Process object.
                If 'band', a confidence band is drawn on the plot.
                If 'ci', confidence intervals are drawn at each discrete value of the moderator on the x-axis
                If 'none', no confidence interval is drawn.
        :param hue_format: string or None
            By default, the color-code are labeled:
                'Mod1 at val1' if there is two moderator for 'hue'.
                'Mod2 at val1, Mod2 at val2' if there are two moderators for 'hue'.
            Alternatively, a string that should be formatted can be passed. The string will receive as arguments:
                var1 (the name of the first moderator)
                var2 (the name of the second moderator, if it exists)
                val1 (the value of the first moderator)
                val2 (the value of the second moderator)
            A valid string would for instance look like this: '{var1} = {val1:.4f}, {var2} = {val2:.4f}'
        :param facet_kws: dict
            A dictionary of arguments that should be passed to the FacetGrid object (such as sharex, sharey, size,
            aspect...)
        :param plot_kws:
            A dictionary of arguments to be passed to the 'plt.scatter' function (such as linestyle, linewidth...)
        :param err_kws:
            A dictionary of arguments to be passed to the 'plt.fill_between' (if errstyle='band') or to the
            'plt.errorbar' (if errstyle='ci') (such as alpha, capthick, capsize...)
        :return: a FacetGrid object
        """
        if not x:
            raise ValueError("You must specify at least one moderator for 'x'")
        if modval is None:
            modval = {}

        modval_parsed = self._parse_moderator_values(
            x, hue, row, col, modval, path="indirect"
        )
        df_effects = self.spotlight_indirect_effect(
            med_name=med_name, spotval=modval_parsed
        )

        return plot_conditional_effects(
            df_effects,
            x,
            hue,
            row,
            col,
            errstyle,
            hue_format,
            facet_kws,
            plot_kws,
            err_kws,
        )

    # DEPRECATED METHODS
    def plot_indirect_effects(self, *args, **kwargs):
        raise DeprecationWarning(
            "The method 'plot_indirect_effects' has been deprecated. Please use the equivalent method named \
            'plot_conditional_indirect_effects."
        )

    def plot_direct_effects(self, *args, **kwargs):
        raise DeprecationWarning(
            "The method 'plot_direct_effects' has been deprecated. Please use the equivalent method named \
            'plot_conditional_direct_effects."
        )
