from __future__ import annotations
from functools import wraps
import logging
import warnings
import numpy as np
from scipy import stats
from typing import Literal, Any, Sequence, TYPE_CHECKING
from .types import ConfInterval, DynamicTesting
if TYPE_CHECKING:
    from ._Forecaster_parent import Forecaster_parent

class _developer_utils:
    @staticmethod
    def log_warnings(func:callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with warnings.catch_warnings(record=True) as warn_list:
                warnings.simplefilter("always")
                result = func(*args, **kwargs)
                for warn in warn_list:
                    logging.warning(warn.message)
            return result
        return wrapper
    
    @staticmethod
    def descriptive_assert(statement:bool, ErrorType:Exception, error_message:str):
        try:
            assert statement
        except AssertionError:
            raise ErrorType(error_message)

    @staticmethod      
    def _return_na_if_len_zero(y:Sequence[float],pred:Sequence[float],func:callable):
        if len(pred) == 0:
            return np.nan
        else:
            return func(y,pred[-len(y):])

    @staticmethod
    def _set_ci_step(f:'Forecaster_parent',s:int):
        return stats.norm.ppf(1 - (1 - f.cilevel) / 2) * s

    @staticmethod
    def _check_train_only_arg(f:'Forecaster_parent', train_only:bool):
        _developer_utils.descriptive_assert(isinstance(train_only, bool), ValueError, f"train_only must be True or False, got {train_only} of type {type(train_only)}.")
        _developer_utils.descriptive_assert(not train_only or f.test_length > 0, ValueError, "train_only cannot be True when test_length is 0.")

    @staticmethod
    def _warn_about_not_finding_cis(m:str):
        warnings.warn(
            f'Confidence intervals not found for {m}. '
            'To turn on confidence intervals for future evaluated models, call the eval_cis() method.'
            ,category=Warning
        )

    @staticmethod
    def _reshape_func_input(x:Sequence[float],func:callable):
        x = np.array(x).reshape(-1,1)
        if x.shape[0] == 0:
            return []
        else:
            return func(x)[:,0]

    @staticmethod
    def _select_reg_for_direct_forecasting(f:'Forecaster_parent'):
        return {
            k:v.to_list() 
            for k, v in f.current_xreg.items() 
            if (
                np.isnan(f.future_xreg[k]).sum() == 0 
                and len(f.future_xreg[k]) == len(f.future_dates)
            )
        }

class NamedBoxCox:
    def __init__(self,name:str,transform:bool):
        self.name = name
        self.transform = transform

    def __call__(self,x,lmbda):
        if self.transform:
            return [(i**lmbda - 1) / lmbda for i in x] if lmbda != 0 else [np.log(i) for i in x]
        else:
            return [(i*lmbda + 1)**(1/lmbda) for i in x] if lmbda != 0 else [np.exp(i) for i in x]

    def __repr__(self):
        return self.name
    
def _tune_test_forecast(
    f:"Forecaster_parent",
    models:list[str],
    cross_validate:bool,
    dynamic_tuning:DynamicTesting,
    dynamic_testing:DynamicTesting,
    limit_grid_size:int|ConfInterval,
    suffix:str,
    error:Literal['raise','warn','ignore'],
    min_grid_size:int = 1,
    feature_importance:bool = False,
    fi_try_order:list[str] = None,
    use_progress_bar:bool = False,
    **cvkwargs:Any,
):
    if use_progress_bar: # notebooks only get progress bar
        from tqdm.notebook import tqdm
    else:
        tqdm = list

    for m in tqdm(models):
        call_me = m if suffix is None else m + suffix
        f.set_estimator(m)
        if limit_grid_size is not None:
            f.ingest_grid(m)
            f.limit_grid_size(n=limit_grid_size,min_grid_size=min_grid_size)
        if cross_validate:
            f.cross_validate(dynamic_tuning=dynamic_tuning, **cvkwargs)
        else:
            f.tune(dynamic_tuning=dynamic_tuning)
        try:
            f.auto_forecast(
                dynamic_testing=dynamic_testing,
                call_me=call_me,
            )
        except Exception as e:
            if error == 'raise':
                raise
            elif error == 'warn':
                warnings.warn(
                    f"{m} model could not be evaluated. "
                    f"Here's the error: {e}",
                    category=Warning,
                )
                continue
            elif error == 'ignore':
                continue
            else:
                raise ValueError(f'Value passed to error arg not recognized: {error}')
        if feature_importance:
            if fi_try_order is None:
                f.save_feature_importance()
            else:
                f.save_feature_importance(try_order=fi_try_order)

boxcox_tr = NamedBoxCox(name='BoxcoxTransform',transform=True)
boxcox_re = NamedBoxCox(name='BoxcoxRevert',transform=False)