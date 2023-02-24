from functools import wraps
import logging
import warnings
import numpy as np
from scipy import stats

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
        return np.nan if len(pred) == 0 else func(y,pred)

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
        ] + [
            'LevelTestSet' + m.upper() for m in metrics
        ] + [
            'LevelInSample' + m.upper() for m in metrics
        ] + ['ValidationMetricValue']