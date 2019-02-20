# -*- coding: utf-8 -*-
from typing import Union, Iterable, Dict, List, Any, Set, Tuple, Optional

from numpy import (
    ndarray,
)
from pandas.core.frame import DataFrame
from seaborn.axisgrid import FacetGrid
from typing_extensions import NoReturn

from .models import OLSOutcomeModel, DirectEffectModel, ParallelMediationModel, LogitOutcomeModel, \
    DirectFloodlightAnalysis, IndirectFloodlightAnalysis


class Process(object):
    __var_kws__: Set[str]

    __options_kws__: Set[str]

    __models_vars__: Dict[str, Set[str]]

    __models_eqs__: Dict[str, Dict[List[str]]]

    model_num: int
    controls: Iterable
    data: DataFrame
    options = Dict[str, Any]
    varlist = List[str]
    mediators = List[str]
    moderators: Dict[str, Set[str]]
    iv: str
    n_obs: int
    n_obs_null: int
    n_meds: int
    has_mediation: bool
    outcome_models: Dict[str, Union[OLSOutcomeModel, LogitOutcomeModel]]
    spotlight_values: Dict[str, List[float]]
    direct_model: DirectEffectModel
    indirect_model: Union[None, ParallelMediationModel]
    centered_vars: Union[List, None]

    _var_to_symb: Dict[str, str]
    _symb_to_var: Dict[str, str]
    _symb_to_ind: Dict[str, int]

    _equations: List[Tuple[Union[str, List[str]]]]

    def __init__(
            self,
            data: DataFrame,
            model: int,
            modval: Optional[Dict[str, List[float]]],
            cluster: Optional[str],
            boot: Optional[int],
            seed: Optional[int],
            mc: Optional[bool],
            conf: Optional[int],
            effsize: Optional[bool],
            jn: Optional[bool],
            hc3: Optional[bool],
            controls: Optional[Iterable[str]],
            controls_in: str,
            total: Optional[bool],
            contrast: Optional[bool],
            center: Optional[bool],
            quantile: Optional[bool],
            detail: Optional[bool],
            percent: Optional[bool],
            logit: Optional[bool],
            iterate: Optional[int],
            convergence: Optional[float],
            precision: Optional[int],
            suppr_init: Optional[bool],
            **kwargs
    ) -> None: ...

    # PRIVATE METHODS
    def _gen_valid_options(
            self,
            argument: Dict[str, Any]
    ) -> Dict[str, Any]: ...

    def _gen_valid_varlist(
            self,
            var_kwargs: Dict[str, Union[str, List]]
    ) -> List[str]: ...

    def _gen_analysis_list(self) -> List[str]:
        ...

    def _gen_spotlight_values(
            self,
            modval: Optional[Dict[str, List[float]]]
    ) -> Dict[str, Iterable[float]]: ...

    def _gen_var_mapping(
            self,
            var_kwargs: Dict[str, str]
    ) -> Tuple[Dict[str, str], Dict[str, str]]: ...

    def _prepare_data(self) -> Tuple[DataFrame, int, int, List[str]]:
        ...

    def _gen_equations(
            self,
            all_to_y: List[str],
            x_to_m: List[str],
            controls_in: str
    ) -> List[Tuple[str, List[str]]]: ...

    def _gen_outcome_models(self) -> Dict[str, Union[OLSOutcomeModel, LogitOutcomeModel]]: ...

    def _gen_direct_effect_model(self) -> DirectEffectModel: ...

    def _gen_indirect_effect_model(self) -> ParallelMediationModel: ...

    def _print_init(self) -> None: ...

    def _parse_moderator_values(
            self,
            x: str,
            hue: Optional[Union[str, List[str]]],
            row: Optional[str],
            col: Optional[str],
            modval: Optional[Dict[str, Union[ndarray, List[float]]]],
            path: str
    ) -> Dict[str, Union[ndarray, List[float]]]: ...

    # API METHODS

    def summary(self) -> None: ...

    def get_bootstrap_estimates(self) -> DataFrame: ...

    def floodlight_indirect_effect(
            self,
            med_name: str,
            mod_name: str,
            other_modval: Optional[Dict[str, float]],
            atol: Optional[float],
            rtol: Optional[float]
    ) -> IndirectFloodlightAnalysis: ...

    def floodlight_direct_effect(
            self,
            mod_name: str,
            other_modval: Optional[Dict[str, float]],
            atol: Optional[float],
            rtol: Optional[float]
    ) -> DirectFloodlightAnalysis: ...

    def spotlight_indirect_effect(
            self,
            med_name: str,
            spotval: Optional[Dict[str, Union[ndarray, List[float]]]]
    ) -> DataFrame: ...

    def spotlight_direct_effect(
            self,
            spotval: Optional[Dict[str, Union[ndarray, List[float]]]]
    ) -> DataFrame: ...

    def plot_conditional_direct_effects(
            self,
            x: str,
            hue: Optional[Union[str, List[str]]],
            row: Optional[str],
            col: Optional[str],
            modval: Optional[Dict[str, Union[ndarray, List[float]]]],
            errstyle: Optional[str],
            hue_format: Optional[str],
            facet_kws: Optional[Dict[str, Any]],
            plot_kws: Optional[Dict[str, Any]],
            err_kws: Optional[Dict[str, Any]]
    ) -> FacetGrid: ...

    def plot_conditional_indirect_effects(
            self,
            med_name: str,
            x: str,
            hue: Optional[Union[str, List[str]]],
            row: Optional[str],
            col: Optional[str],
            modval: Optional[Dict[str, Union[ndarray, List[float]]]],
            errstyle: Optional[str],
            hue_format: Optional[str],
            facet_kws: Optional[Dict[str, Any]],
            plot_kws: Optional[Dict[str, Any]],
            err_kws: Optional[Dict[str, Any]]
    ) -> FacetGrid: ...

    # DEPRECATED METHODS
    def plot_indirect_effects(self, *args, **kwargs) -> NoReturn: ...

    def plot_direct_effects(self, *args, **kwargs) -> NoReturn: ...
