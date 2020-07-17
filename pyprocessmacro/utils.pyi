from typing import Callable, Union, Iterable, Dict, List, Optional, Any, Set

import numpy as np
from pandas import DataFrame
from seaborn import FacetGrid

def z_score(conf: float) -> float: ...
def bias_corrected_ci(
    estimate: np.array, samples: np.array, conf: Union[float, int]
) -> (float, float): ...
def percentile_ci(samples: np.array, conf: Union[float, int]) -> np.array: ...
def fast_OLS(endog: np.array, exog: np.array) -> np.array: ...
def logit_cdf(X: np.array) -> np.array: ...
def logit_score(
    endog: np.array, exog: np.array, params: np.array, n_obs: int
) -> np.array: ...
def logit_hessian(exog: np.array, params: np.array, n_obs: int) -> np.array: ...
def fast_optimize(
    endog: np.array,
    exog: np.array,
    n_obs: int,
    n_vars: int,
    max_iter: int,
    tolerance: float = 1e-10,
) -> np.array: ...
def bootstrap_sampler(n_obs: int, seed: int) -> np.array: ...
def eigvals(exog: np.array) -> np.array: ...
def eval_expression(expr: np.array, values: Dict) -> np.array: ...
def gen_moderators(
    raw_equations: Dict[str, List[str]], raw_varlist: List[str]
) -> Dict[str, Set[str]]: ...
def plot_errorbars(
    x: np.array,
    y: np.array,
    yerrlow: np.array,
    yerrhigh: np.array,
    plot_kws: Optional[Dict[str, Any]],
    err_kws: Optional[Dict[str, Any]],
    *args,
    **kwargs
) -> None: ...
def plot_errorbands(
    x: np.array,
    y: np.array,
    llci: np.array,
    ulci: np.array,
    plot_kws: Optional[Dict[str, Any]],
    err_kws: Optional[Dict[str, Any]],
    *args,
    **kwargs
) -> None: ...
def plot_conditional_effects(
    df_effects: DataFrame,
    x: str,
    hue: Optional[Union[str, List[str]]],
    row: Optional[str],
    col: Optional[str],
    errstyle: Optional[str],
    hue_format: Optional[str],
    facet_kws: Optional[Dict[str, Any]],
    plot_kws: Optional[Dict[str, Any]],
    err_kws: Optional[Dict[str, Any]],
) -> FacetGrid: ...
def find_significance_region(
    spotlight_func: Callable[[Dict], Iterable],
    mod_symb: str,
    modval_min: float,
    modval_max: float,
    modval_other_symb: Dict,
    atol: float,
    rtol: float,
) -> List[List[float]]: ...
def search_mid_range(
    spotlight_func: Callable[[Dict], Iterable],
    min_val: float,
    max_val: float,
    mod_symb: str,
    mod_dict: Dict,
    atol: float,
    rtol: float,
) -> List[List[float]]: ...
def search_critical_values(
    spotlight_func: Callable[[Dict], Iterable],
    min_val: float,
    max_val: float,
    mod_symb: str,
    mod_dict: Dict,
    slope: str,
    region: str,
    atol: float,
    rtol: float,
) -> float: ...
