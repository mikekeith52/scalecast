from __future__ import annotations
from functools import wraps
import logging
import warnings
import numpy as np
from scipy import stats
from statsmodels.tsa.tsatools import freq_to_period
from typing import Literal, Any, TYPE_CHECKING
from .types import ConfInterval, DynamicTesting
if TYPE_CHECKING:
    from ._Forecaster_parent import _Forecaster_parent

class _developer_utils:
    @staticmethod
    def log_warnings(func):
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
    def descriptive_assert(statement, ErrorType, error_message):
        # descriptive assert statement for descriptive exception raising
        try:
            assert statement
        except AssertionError:
            raise ErrorType(error_message)

    @staticmethod      
    def _return_na_if_len_zero(y,pred,func):
        return (
            np.nan 
            if len(pred) == 0 else func(y,pred) 
            if len(y) == len(pred) else func(y,pred[-len(y):])
        )

    @staticmethod
    def _set_ci_step(f,s):
        return stats.norm.ppf(1 - (1 - f.cilevel) / 2) * s

    @staticmethod
    def _check_train_only_arg(f, train_only):
        _developer_utils.descriptive_assert(
            isinstance(train_only, bool), ValueError, f"train_only must be True or False, got {train_only} of type {type(train_only)}."
        )
        _developer_utils.descriptive_assert(
            not train_only or f.test_length > 0, ValueError, "train_only cannot be True when test_length is 0."
        )

    @staticmethod
    def _check_if_correct_estimator(estimator,possible_estimators):
        _developer_utils.descriptive_assert(
            estimator in possible_estimators,
            ValueError,
            f"estimator must be one of {possible_estimators}, got {estimator}.",
        )

    @staticmethod
    def _warn_about_not_finding_cis(m):
        warnings.warn(
            f'Confidence intervals not found for {m}. '
            'To turn on confidence intervals for future evaluated models, call the eval_cis() method.'
            ,category=Warning
        )

    @staticmethod
    def _convert_m(m,freq):
        if m == 'auto':
            if freq is not None:
                if freq.startswith('M'):
                    return 12
                elif freq.startswith('Q'):
                    return 4
                elif freq.startswith('H'):
                    return 24
                else:
                    try:
                        return freq_to_period(freq)
                    except:
                        return 1
            else:
                return 1
        return m

    @staticmethod
    def _determine_best_by(metrics):
        return [
            'TestSet' + m.upper() for m in metrics
        ] + [
            'InSample' + m.upper() for m in metrics
        ] + ['ValidationMetricValue']

    @staticmethod
    def _reshape_func_input(x,func):
        x = np.array(x).reshape(-1,1)
        if x.shape[0] == 0:
            return []
        else:
            return func(x)[:,0]

    @staticmethod
    def _select_reg_for_direct_forecasting(f):
        return {
            k:v.to_list() 
            for k, v in f.current_xreg.items() 
            if (
                np.isnan(f.future_xreg[k]).sum() == 0 
                and len(f.future_xreg[k]) == len(f.future_dates)
            )
        }

class NamedBoxCox:
    def __init__(self,name,transform):
        self.name = name
        self.transform = transform

    def __call__(self,x,lmbda):
        if self.transform:
            return [(i**lmbda - 1) / lmbda for i in x] if lmbda != 0 else [np.log(i) for i in x]
        else:
            return [(i*lmbda + 1)**(1/lmbda) for i in x] if lmbda != 0 else [np.exp(i) for i in x]

    def __repr__(self):
        return self.name
    
def _none(x): # need it like this so can be pickled
    return x
    
def _tune_test_forecast(
    f:"_Forecaster_parent",
    models:list[str],
    cross_validate:bool,
    dynamic_tuning:DynamicTesting,
    dynamic_testing:DynamicTesting,
    limit_grid_size:int|ConfInterval,
    suffix:str,
    error:Literal['raise','warn','ignore'],
    min_grid_size:int = 1,
    summary_stats:bool = False,
    feature_importance:bool = False,
    fi_try_order:list[str] = None,
    use_progress_bar:bool = False,
    **cvkwargs:Any,
):
    if use_progress_bar: # notebooks only get progress bar
        from tqdm.notebook import tqdm
    else:
        tqdm = list

    [_developer_utils._check_if_correct_estimator(m,f.can_be_tuned) for m in models]
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
        if summary_stats:
            f.save_summary_stats()
        if feature_importance:
            if fi_try_order is None:
                f.save_feature_importance()
            else:
                f.save_feature_importance(try_order=fi_try_order)

boxcox_tr = NamedBoxCox(name='BoxcoxTransform',transform=True)
boxcox_re = NamedBoxCox(name='BoxcoxRevert',transform=False)