from numpy import ndarray
from typing import Any, Dict, List, Optional, Tuple, Union, Iterable, Callable

from pandas import DataFrame

class BaseLogit:
    _endog: ndarray
    _exog: ndarray
    _n_obs: int
    _n_vars: int
    _options: Dict[str, Any]
    def __init__(
        self, endog: ndarray, exog: ndarray, options: Dict[str, Any]
    ) -> None: ...
    @staticmethod
    def _cdf(X: ndarray) -> ndarray: ...
    def _hessian(self, params: ndarray) -> ndarray: ...
    def _loglike(self, params: ndarray) -> float: ...
    def _loglikeobs(self, params: ndarray) -> ndarray: ...
    def _optimize(self) -> ndarray: ...
    def _score(self, params: ndarray) -> ndarray: ...

class NullLogitModel(BaseLogit):
    def __init__(self, endog: ndarray, options: Optional[Dict[str, Any]]) -> None: ...

class BaseOutcomeModel:
    _data: ndarray
    _derivative: ndarray
    _endog: ndarray
    _endogvar: str
    _exog: ndarray
    _exogvars: List[str]
    _n_obs: int
    _n_vars: int
    _options: Dict[Union[None, str], Any]
    _symb_to_ind: Dict[str, int]
    _symb_to_var: Dict[str, str]
    _varnames: List[str]
    estimation_results: Dict[str, Union[ndarray, float, int, List[str]]]
    def __init__(
        self,
        data: ndarray,
        endogvar: str,
        exogvars: List[str],
        symb_to_ind: Dict[str, int],
        symb_to_var: Dict[str, str],
        options: Optional[Dict[str, Any]] = ...,
    ) -> None: ...
    def _gen_derivative(self, wrt: str) -> ndarray: ...
    def coeff_summary(self) -> DataFrame: ...
    def _estimate(self) -> Dict[str, Union[ndarray, float, int, List[str]]]: ...

class ParallelMediationModel:
    _analysis_list: List[str]
    _base_derivs: Dict[str, Union[ndarray, List[ndarray]]]
    _boot_betas_m: ndarray
    _boot_betas_y: ndarray
    _compute_betas_m: Callable[[ndarray, ndarray], ndarray]
    _compute_betas_y: Callable[[ndarray, ndarray], ndarray]
    _data: ndarray
    _endog_vars_m: List[str]
    _exog_inds_m: List[int]
    _exog_inds_y: List[int]
    _exog_terms_m: List[str]
    _exog_terms_y: List[str]
    _has_moderation: bool
    _ind_y: int
    _inds_m: List[int]
    _moderators_symb: List[str]
    _moderators_values: List[List[float]]
    _n_fail_samples: int
    _n_meds: int
    _n_obs: int
    _options: Dict[Union[None, str], Any]
    _symb_to_ind: Dict[str, int]
    _symb_to_var: Dict[str, str]
    _true_betas_m: ndarray
    _true_betas_y: ndarray
    _vars_m: List[str]
    _vars_y: List[str]
    estimation_results: Dict[str, ndarray]
    def __init__(
        self,
        data: ndarray,
        exog_terms_y: List[str],
        exog_terms_m: List[str],
        mod_symb: Iterable[str],
        spot_values: Dict[str, List[float]],
        n_meds: int,
        analysis_list: List[str],
        symb_to_ind: Dict[str, int],
        symb_to_var: Dict[str, str],
        options: Optional[Dict[str, Any]] = ...,
    ) -> None: ...
    def _simple_ind_effects(self) -> Dict[str, ndarray]: ...
    def _simple_ind_effects_wrapper(self) -> DataFrame: ...
    def _MM_index(self) -> Dict[str, ndarray]: ...
    def _MMM_index(self) -> Dict[str, ndarray]: ...
    def _CMM_index(self) -> Dict[str, ndarray]: ...
    def _PMM_index(self) -> Dict[str, ndarray]: ...
    def _MM_index_wrapper(self) -> DataFrame: ...
    def _MMM_index_wrapper(self) -> DataFrame: ...
    def _CMM_index_wrapper(self) -> DataFrame: ...
    def _PMM_index_wrapper(self) -> DataFrame: ...
    def MM_index_summary(self) -> DataFrame: ...
    def MMM_index_summary(self) -> DataFrame: ...
    def CMM_index_summary(self) -> DataFrame: ...
    def PMM_index_summary(self) -> DataFrame: ...
    def _cond_ind_effects(self) -> DataFrame: ...
    def _cond_ind_effects_wrapper(self) -> DataFrame: ...
    def _estimate_bootstrapped_params(self) -> Tuple[ndarray, ndarray, int]: ...
    def _estimate_true_params(self) -> Tuple[ndarray, List[ndarray]]: ...
    def _floodlight_analysis(
        self,
        med_index: int,
        mod_symb: str,
        modval_range: List[float],
        other_modval_symb: Dict[str, int],
        atol: float,
        rtol: float,
    ) -> Union[List[List[float]]]: ...
    def _gen_derivatives(self) -> Dict[str, Union[ndarray, List[ndarray]]]: ...
    def _get_conditional_indirect_effects(
        self,
        med_index: int,
        mod_symb: Iterable[str],
        mod_values: Union[ndarray, List[Iterable[float]]],
    ) -> Tuple[ndarray, ndarray, ndarray, ndarray, ndarray]: ...
    def _indirect_effect_at(
        self, med_index: int, mod_dict: Dict[str, float]
    ) -> Tuple[float, ndarray, float, float, float]: ...
    def coeff_summary(self) -> DataFrame: ...
    def summary(self) -> str: ...

class LogitOutcomeModel(BaseOutcomeModel, BaseLogit):
    def __init__(
        self,
        data: ndarray,
        endogvar: str,
        exogvars: List[str],
        symb_to_ind: Dict[str, int],
        symb_to_var: Dict[str, str],
        options: Optional[Dict[str, Any]] = ...,
    ) -> None: ...
    def _estimate(self) -> Dict[str, Union[ndarray, float, int, List[str]]]: ...

class OLSOutcomeModel(BaseOutcomeModel):
    def __init__(
        self,
        data: ndarray,
        endogvar: str,
        exogvars: List[str],
        symb_to_ind: Dict[str, int],
        symb_to_var: Dict[str, str],
        options: Optional[Dict[str, Any]] = ...,
    ) -> None: ...
    def _estimate(self) -> Dict[str, Union[ndarray, float, int, List[str]]]: ...

class DirectEffectModel:
    _derivative: ndarray
    _estimation_results: Dict[str, ndarray]
    _has_mediation: bool
    _has_moderation: bool
    _is_logit: bool
    _model: Union[LogitOutcomeModel, OLSOutcomeModel]
    _moderators_symb: List[str]
    _moderators_values: List[List[float]]
    _options: Dict[Union[None, str], Any]
    _symb_to_var: Dict[str, str]
    def __init__(
        self,
        model: OLSOutcomeModel,
        mod_symb: Iterable[str],
        spot_values: Dict[str, List[float]],
        has_mediation: bool,
        symb_to_var: Dict[str, str],
        options: Optional[Dict[str, Any]] = ...,
    ) -> None: ...
    def _direct_effect_at(
        self, mod_dict: Dict[str, float]
    ) -> Tuple[float, float, float, float]: ...
    def _estimate(self) -> Dict[str, ndarray]: ...
    def _floodlight_analysis(
        self,
        mod_symb: str,
        modval_range: List[float],
        other_modval_symb: Dict[str, int],
        atol: float,
        rtol: float,
    ) -> List[List[float]]: ...
    def _get_conditional_direct_effects(
        self, mod_symb: List[str], mod_values: Union[ndarray, List[Iterable[float]]]
    ) -> Tuple[ndarray, ndarray, ndarray, ndarray]: ...

class BaseFloodlightAnalysis:
    sig_regions: List[List[float]]
    def __init__(
        self,
        med_name: Optional[str],
        mod_name: str,
        sig_regions: List[List[float]],
        modval_range: List[float],
        other_modval_name: Dict[str, float],
        precision: int,
    ) -> None: ...
    def get_significance_regions(self) -> Dict[str, List[float]]: ...

class DirectFloodlightAnalysis(BaseFloodlightAnalysis):
    def __init__(
        self,
        mod_name: str,
        sig_regions: List[List[float]],
        modval_range: List[float],
        other_modval_name: Dict[str, int],
        precision: int,
    ) -> None: ...


class IndirectFloodlightAnalysis(BaseFloodlightAnalysis):
    def __init__(
        self,
        med_name: str,
        mod_name: str,
        sig_regions: List[List[float]],
        modval_range: List[float],
        other_modval_name: Dict[str, int],
        precision: int,
    ) -> None: ...
