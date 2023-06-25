from .__init__ import (
    __sklearn_imports__,
    __sklearn_estimators__,
    __non_sklearn_estimators__,
    __estimators__,
    __cannot_be_tuned__,
    __can_be_tuned__,
    __metrics__,
    __normalizer__,
    __colors__,
    __not_hyperparams__,
)
from ._utils import _developer_utils
import copy
import pandas as pd
import numpy as np
import importlib
import warnings
import logging
import inspect
import datetime
import math
import random
from sklearn.preprocessing import PowerTransformer

# descriptive errors

class ForecastError(Exception):
    pass

class Forecaster_parent:
    def __init__(
        self,
        y,
        test_length,
        cis,
        metrics,
        **kwargs,
    ):
        self._logging()
        self.y = y
        self.sklearn_imports = __sklearn_imports__
        self.sklearn_estimators = __sklearn_estimators__
        self.estimators = __sklearn_estimators__ # this will be overwritten with Forecaster but maintained in MVForecaster
        self.can_be_tuned = __sklearn_estimators__ # this will be overwritten with Forecaster but maintained in MVForecaster
        self.normalizer = __normalizer__
        self.set_test_length(test_length)
        self.validation_length = 1
        self.validation_metric = metrics[0]
        self.cis = cis
        self.cilevel = 0.95
        self.current_xreg = {} # Series
        self.future_xreg = {} # lists
        self.history = {}
        self.set_metrics(metrics)
        self.set_estimator("mlr")
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __copy__(self):
        if hasattr(self,'tf_model'):
            delattr(self,'tf_model')
        obj = type(self).__new__(self.__class__)
        obj.__dict__.update(self.__dict__)
        return obj

    def __deepcopy__(self):
        if hasattr(self,'tf_model'):
            delattr(self,'tf_model')
        obj = type(self).__new__(self.__class__)
        obj.__dict__ = copy.deepcopy(self.__dict__)
        return obj

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __getstate__(self):
        self._logging()
        state = self.__dict__.copy()
        return state

    def _check_right_test_length_for_cis(self,cilevel):
        min_test_length = round(1/(1-cilevel))
        if self.test_length < min_test_length:
            raise ValueError(
                'Cannot evaluate confidence intervals at the '
                '{:.0%} level when test_length is set to less than {} observations. '
                'The test length is currently set to {} observations. '
                'The test length must be at least 1/(1-cilevel) in length for conformal intervals to work.'.format(
                    cilevel,
                    int(min_test_length),
                    self.test_length,
                )
            )

    def _logging(self):
        logging.basicConfig(filename="warnings.log", level=logging.WARNING)

    def _clear_the_deck(self):
        """ deletes the following attributes to prepare a new forecast:
            'univariate','fitted_values','regr','X','feature_importance','summary_stats','models','weights'
        """
        for attr in (
            "fitted_values",
            "regr",
            "X",
            "Xvars",
            "feature_importance",
            "perm",
            "summary_stats",
            "models",
            "weights",
            "tf_model",
            "scaler",
        ):
            try:
                delattr(self, attr)
            except AttributeError:
                pass

    def copy(self):
        """ Creates an object copy.
        """
        return self.__copy__()

    def deepcopy(self):
        """ Creates an object deepcopy.
        """
        return self.__deepcopy__()

    def add_seasonal_regressors(
        self, 
        *args, 
        raw=True, 
        sincos=False, 
        dummy=False, 
        drop_first=False,
        cycle_lens=None,
    ):
        """ Adds seasonal regressors. 
        Can be in the form of Fourier transformed, dummy, or integer values.

        Args:
            *args (str): Values that return a series of int type from pandas.dt or pandas.dt.isocalendar().
                See https://pandas.pydata.org/docs/reference/api/pandas.Series.dt.year.html.
            raw (bool): Default True.
                Whether to use the raw integer values.
            sincos (bool): Default False.
                Whether to use a Fourier transformation of the raw integer values.
                The length of the cycle is derived from the max observed value unless cycle_lens is specified.
            dummy (bool): Default False.
                Whether to use dummy variables from the raw int values.
            drop_first (bool): Default False.
                Whether to drop the first observed dummy level.
                Not relevant when dummy = False.
            cycle_lens (dict): Optional. A dictionary that specifies a cycle length for each selected seasonality.
                If this is not specified or a selected seasonality is not added to the dictionary as a key, the
                cycle length will be selected automatically as the maximum value observed for the given seasonality.
                Not relevant when sincos = False.

        Returns:
            None

        >>> f.add_seasonal_regressors('year')
        >>> f.add_seasonal_regressors(
        >>>     'dayofyear',
        >>>     'month',
        >>>     'week',
        >>>     'quarter',
        >>>     raw=False,
        >>>     sincos=True,
        >>>     cycle_lens={'dayofyear':365.25},
        >>> )
        >>> f.add_seasonal_regressors('dayofweek',raw=False,dummy=True,drop_first=True)
        """
        #self._validate_future_dates_exist()
        if not (raw | sincos | dummy):
            raise ValueError("at least one of raw, sincos, dummy must be True")
        for s in args:
            try:
                if s in ("week", "weekofyear"):
                    _raw = getattr(self.current_dates.dt.isocalendar(), s)
                else:
                    _raw = getattr(self.current_dates.dt, s)
            except AttributeError:
                raise ValueError(
                    f'cannot set "{s}". see possible values here: https://pandas.pydata.org/docs/reference/api/pandas.Series.dt.year.html'
                )

            try:
                _raw.astype(int)
            except ValueError:
                raise ValueError(
                    f"{s} must return an int. use dummy = True to get dummies"
                )

            if s in ("week", "weekofyear"):
                _raw_fut = getattr(self.future_dates.dt.isocalendar(), s)
            else:
                _raw_fut = getattr(self.future_dates.dt, s)
            if raw:
                self.current_xreg[s] = _raw
                self.future_xreg[s] = _raw_fut.to_list()
            if sincos:
                _cycles = (
                    _raw.max() if cycle_lens is None 
                    else _raw.max()  if s not in cycle_lens 
                    else cycle_lens[s]
                )
                self.current_xreg[f"{s}sin"] = np.sin(np.pi * _raw / (_cycles / 2))
                self.current_xreg[f"{s}cos"] = np.cos(np.pi * _raw / (_cycles / 2))
                self.future_xreg[f"{s}sin"] = np.sin(
                    np.pi * _raw_fut / (_cycles / 2)
                ).to_list()
                self.future_xreg[f"{s}cos"] = np.cos(
                    np.pi * _raw_fut / (_cycles / 2)
                ).to_list()
            if dummy:
                all_dummies = []
                stg_df = pd.DataFrame({s: _raw.astype(str)})
                stg_df_fut = pd.DataFrame({s: _raw_fut.astype(str)})
                for c, v in (
                    pd.get_dummies(stg_df, drop_first=drop_first)
                    .to_dict(orient="series")
                    .items()
                ):
                    self.current_xreg[c] = v
                    all_dummies.append(c)
                for c, v in (
                    pd.get_dummies(stg_df_fut, drop_first=drop_first)
                    .to_dict(orient="list")
                    .items()
                ):
                    if c in all_dummies:
                        self.future_xreg[c] = v
                for c in [d for d in all_dummies if d not in self.future_xreg.keys()]:
                    self.future_xreg[c] = [0] * len(self.future_dates)

    def add_time_trend(self, called="t"):
        """ Adds a time trend from 1 to length of the series + the forecast horizon as a current and future Xvar.

        Args:
            Called (str): Default 't'.
                What to call the resulting variable.

        Returns:
            None

        >>> f.add_time_trend() # adds time trend called 't'
        """
        #self._validate_future_dates_exist()
        self.current_xreg[called] = pd.Series(range(1, len(self.y) + 1))
        self.future_xreg[called] = list(
            range(len(self.y) + 1, len(self.y) + len(self.future_dates) + 1)
        )

    def add_cycle(self, cycle_length, called=None):
        """ Adds a regressor that acts as a seasonal cycle.
        Use this function to capture non-normal seasonality.

        Args:
            cycle_length (int): How many time steps make one complete cycle.
            called (str): Optional. What to call the resulting variable.
                Two variables will be created--one for a sin transformation and the other for cos
                resulting variable names will have "sin" or "cos" at the end.
                Example, called = 'cycle5' will become 'cycle5sin', 'cycle5cos'.
                If left unspecified, 'cycle{cycle_length}' will be used as the name.

        Returns:
            None

        >>> f.add_cycle(13) # adds a seasonal effect that cycles every 13 observations called 'cycle13'
        """
        #self._validate_future_dates_exist()
        if called is None:
            called = f"cycle{cycle_length}"
        full_sin = pd.Series(range(1, len(self.y) + len(self.future_dates) + 1)).apply(
            lambda x: np.sin(np.pi * x / (cycle_length / 2))
        )
        full_cos = pd.Series(range(1, len(self.y) + len(self.future_dates) + 1)).apply(
            lambda x: np.cos(np.pi * x / (cycle_length / 2))
        )
        self.current_xreg[called + "sin"] = pd.Series(full_sin.values[: len(self.y)])
        self.current_xreg[called + "cos"] = pd.Series(full_cos.values[: len(self.y)])
        self.future_xreg[called + "sin"] = list(full_sin.values[len(self.y) :])
        self.future_xreg[called + "cos"] = list(full_cos.values[len(self.y) :])

    def add_other_regressor(self, called, start, end):
        """ Adds a dummy variable that is 1 during the specified time period, 0 otherwise.

        Args:
            called (str):
                What to call the resulting variable.
            start (str, datetime.datetime, or pd.Timestamp): Start date.
                Must be parsable by pandas' Timestamp function.
            end (str, datetime.datetime, or pd.Timestamp): End date.
                Must be parsable by pandas' Timestamp function.

        Returns:
            None

        >>> f.add_other_regressor('january_2021','2021-01-01','2021-01-31')
        """
        #self._validate_future_dates_exist()
        self.current_xreg[called] = pd.Series(
            [1 if (x >= pd.Timestamp(start)) & (x <= pd.Timestamp(end)) else 0 for x in self.current_dates]
        )
        self.future_xreg[called] = [
            1 if (x >= pd.Timestamp(start)) & (x <= pd.Timestamp(end)) else 0 for x in self.future_dates
        ]

    def add_covid19_regressor(
        self,
        called="COVID19",
        start=datetime.datetime(2020, 3, 15),
        end=datetime.datetime(2021, 5, 13),
    ):
        """ Adds a dummy variable that is 1 during the time period that COVID19 effects are present for the series, 0 otherwise.
        The default dates are selected to be optimized for the time-span where the economy was most impacted by COVID.

        Args:
            called (str): Default 'COVID19'.
               What to call the resulting variable.
            start (str, datetime.datetime, or pd.Timestamp): Default datetime.datetime(2020,3,15).
                The start date (default is day Walt Disney World closed in the U.S.).
                Must be parsable by pandas' Timestamp function.
            end: (str, datetime.datetime, or pd.Timestamp): Default datetime.datetime(2021,5,13).
               The end date (default is day the U.S. CDC first dropped the mask mandate/recommendation for vaccinated people).
               Must be parsable by pandas' Timestamp function.

        Returns:
            None
        """
        #self._validate_future_dates_exist()
        self.add_other_regressor(called=called, start=start, end=end)

    def add_combo_regressors(self, *args, sep="_"):
        """ Combines all passed variables by multiplying their values together.

        Args:
            *args (str): Names of Xvars that aleady exist in the object.
            sep (str): Default '_'.
                The separator between each term in arg to create the final variable name.

        Returns:
            None

        >>> f.add_combo_regressors('t','monthsin') # multiplies these two together (called 't_monthsin')
        >>> f.add_combo_regressors('t','monthcos') # multiplies these two together (called 't_monthcos')
        """
        #self._validate_future_dates_exist()
        _developer_utils.descriptive_assert(
            len(args) > 1,
            ForecastError,
            "Need to pass at least two variables to combine regressors.",
        )
        for i, a in enumerate(args):
            _developer_utils.descriptive_assert(
                not a.startswith("AR"),
                ForecastError,
                "No combining AR terms at this time -- it confuses the forecasting mechanism.",
            )
            if i == 0:
                self.current_xreg[sep.join(args)] = self.current_xreg[a]
                self.future_xreg[sep.join(args)] = self.future_xreg[a]
            else:
                self.current_xreg[sep.join(args)] = pd.Series(
                    [
                        a * b
                        for a, b in zip(
                            self.current_xreg[sep.join(args)], self.current_xreg[a]
                        )
                    ]
                )
                self.future_xreg[sep.join(args)] = [
                    a * b
                    for a, b in zip(
                        self.future_xreg[sep.join(args)], self.future_xreg[a]
                    )
                ]

    def add_poly_terms(self, *args, pwr=2, sep="^"):
        """ raises all passed variables (no AR terms) to exponential powers (ints only).

        Args:
            *args (str): Names of Xvars that aleady exist in the object
            pwr (int): Default 2.
                The max power to add to each term in args (2 to this number will be added).
            sep (str): default '^'.
                The separator between each term in arg to create the final variable name.

        Returns:
            None

        >>> f.add_poly_terms('t','year',pwr=3) # raises t and year to 2nd and 3rd powers (called 't^2', 't^3', 'year^2', 'year^3')
        """
        #self._validate_future_dates_exist()
        for a in args:
            _developer_utils.descriptive_assert(
                not a.startswith("AR"),
                ForecastError,
                "No polynomial AR terms at this time -- it confuses the forecasting mechanism.",
            )
            for i in range(2, pwr + 1):
                self.current_xreg[f"{a}{sep}{i}"] = self.current_xreg[a] ** i
                self.future_xreg[f"{a}{sep}{i}"] = [x ** i for x in self.future_xreg[a]]

    def add_exp_terms(self, *args, pwr, sep="^", cutoff=2, drop=False):
        """ Raises all passed variables (no AR terms) to exponential powers (ints or floats).

        Args:
            *args (str): Names of Xvars that aleady exist in the object.
            pwr (float): 
                The power to raise each term to in args.
                Can use values like 0.5 to perform square roots, etc.
            sep (str): default '^'.
                The separator between each term in arg to create the final variable name.
            cutoff (int): default 2.
                The resulting variable name will be rounded to this number based on the passed pwr.
                For instance, if pwr = 0.33333333333 and 't' is passed as an arg to *args, the resulting name will be t^0.33 by default.
            drop (bool): Default False.
                Whether to drop the regressors passed to *args.

        Returns:
            None

        >>> f.add_exp_terms('t',pwr=.5) # adds square root t called 't^0.5'
        """
        #self._validate_future_dates_exist()
        pwr = float(pwr)
        for a in args:
            _developer_utils.descriptive_assert(
                not a.startswith("AR"),
                ForecastError,
                "No exponent AR terms at this time -- it confuses the forecasting mechanism.",
            )
            self.current_xreg[f"{a}{sep}{round(pwr,cutoff)}"] = (
                self.current_xreg[a] ** pwr
            )
            self.future_xreg[f"{a}{sep}{round(pwr,cutoff)}"] = [
                x ** pwr for x in self.future_xreg[a]
            ]

        if drop:
            self.drop_Xvars(*args)

    def add_logged_terms(self, *args, base=math.e, sep="", drop=False):
        """ Logs all passed variables (no AR terms).

        Args:
            *args (str): Names of Xvars that aleady exist in the object.
            base (float): Default math.e (natural log). The log base.
                Must be math.e or int greater than 1.
            sep (str): Default ''.
                The separator between each term in arg to create the final variable name.
                Resulting variable names will be like "log2t" or "lnt" by default.
            drop (bool): Default False.
                Whether to drop the regressors passed to *args.

        Returns:
            None

        >>> f.add_logged_terms('t') # adds natural log t callend 'lnt'
        """
        #self._validate_future_dates_exist()
        for a in args:
            _developer_utils.descriptive_assert(
                not a.startswith("AR"),
                ForecastError,
                "No logged AR terms at this time -- it confuses the forecasting mechanism.",
            )
            if base == math.e:
                pass
            elif not (isinstance(base, int)):
                raise ValueError(
                    f"base must be math.e or an int greater than 1, got {base}."
                )
            elif base <= 1:
                raise ValueError(
                    f"base must be math.e or an int greater than 1, got {base}."
                )

            b = "ln" if base == math.e else "log" + str(base)
            self.current_xreg[f"{b}{sep}{a}"] = pd.Series(
                [math.log(x, base) for x in self.current_xreg[a]]
            )
            self.future_xreg[f"{b}{sep}{a}"] = [
                math.log(x, base) for x in self.future_xreg[a]
            ]

        if drop:
            self.drop_Xvars(*args)

    def add_pt_terms(self, *args, method="box-cox", sep="_", drop=False):
        """ Applies a box-cox or yeo-johnson power transformation to all passed variables (no AR terms).

        Args:
            *args (str): Names of Xvars that aleady exist in the object.
            method (str): One of {'box-cox','yeo-johnson'}, default 'box-cox'.
                The type of transformation.
                box-cox works for positive values only.
                yeo-johnson is like a box-cox but can be used with 0s or negatives.
                https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PowerTransformer.html.
            sep (str): Default ''.
                The separator between each term in arg to create the final variable name.
                Resulting variable names will be like "box-cox_t" or "yeo-johnson_t" by default.
            drop (bool): Default False.
                Whether to drop the regressors passed to *args.

        Returns:
            None

        >>> f.add_pt_terms('t') # adds box cox of t called 'box-cox_t'
        """
        #self._validate_future_dates_exist()
        pt = PowerTransformer(method=method, standardize=False)
        for a in args:
            _developer_utils.descriptive_assert(
                not a.startswith("AR"),
                ForecastError,
                "no power-transforming AR terms at this time -- it confuses the forecasting mechanism",
            )
            reshaped_current = np.reshape(self.current_xreg[a].values, (-1, 1))
            reshaped_future = np.reshape(self.future_xreg[a], (-1, 1))

            self.current_xreg[f"{method}{sep}{a}"] = pd.Series(
                [x[0] for x in pt.fit_transform(reshaped_current)]
            )
            self.future_xreg[f"{method}{sep}{a}"] = [
                x[0] for x in pt.fit_transform(reshaped_future)
            ]

        if drop:
            self.drop_Xvars(*args)

    def drop_regressors(self, *args, error = 'raise'):
        """ Drops regressors.

        Args:
            *args (str): The names of regressors to drop.
            error (str): One of 'ignore','raise'. Default 'raise'.
                What to do with the error if the Xvar is not found in the object.

        Returns:
            None

        >>> f.add_time_trend()
        >>> f.add_exp_terms('t',pwr=.5)
        >>> f.drop_regressors('t','t^0.5')
        """
        for a in args:
            if a not in self.current_xreg:
                if error == 'raise':
                    raise ForecastError(f'Cannot find {a} in Forecaster object.')
                elif error == 'ignore':
                    continue
                else:
                    raise ValueError(f'arg passed to error not recognized: {error}')
            self.current_xreg.pop(a)
            self.future_xreg.pop(a)

    def drop_Xvars(self, *args, error = 'raise'):
        """ Drops regressors.

        Args:
            *args (str): The names of regressors to drop.
            error (str): One of 'ignore','raise'. Default 'raise'.
                What to do with the error if the Xvar is not found in the object.

        Returns:
            None

        >>> f.add_time_trend()
        >>> f.add_exp_terms('t',pwr=.5)
        >>> f.drop_Xvars('t','t^0.5')
        """
        self.drop_regressors(*args)

    def drop_all_Xvars(self):
        """ drops all regressors.
        """
        self.drop_Xvars(*self.get_regressor_names())

    def pop(self, *args):
        """ Deletes evaluated forecasts from the object's memory.

        Args:
            *args (str): Names of models matching what was passed to call_me when model was evaluated.

        >>> models = ('mlr','mlp','lightgbm')
        >>> f.tune_test_forecast(models,dynamic_testing=False,feature_importance=True)
        >>> f.pop('mlr')
        """
        for a in args:
            self.history.pop(a)


    def add_sklearn_estimator(self, imported_module, called):
        """ Adds a new estimator from scikit-learn not built-in to the forecaster object that can be called using set_estimator().
        Only regression models are accepted.
        
        Args:
            imported_module (scikit-learn regression model):
                The model from scikit-learn to add. Must have already been imported locally.
                Supports models from sklearn and sklearn APIs.
            called (str):
                The name of the estimator that can be called using set_estimator().

        Returns:
            None

        >>> from sklearn.ensemble import StackingRegressor
        >>> f.add_sklearn_estimator(StackingRegressor,called='stacking')
        >>> f.set_estimator('stacking')
        >>> f.manual_forecast(...)
        """
        self.sklearn_imports[called] = imported_module
        self.sklearn_estimators.append(called)
        self.estimators.append(called)
        self.can_be_tuned.append(called)

    def add_metric(self, func, called = None):
        """ Add a metric to be evaluated when validating and testing models.
        The function should accept two arguments where the first argument is an array of actual values
        and the second is an array of predicted values. The function returns a float.

        Args:
            func (function): The function used to calculate the metric.
            called (str): Optional. The name that can be used to reference the metric function
                within the object. If not specified, will use the function's name.

        >>> from scalecast.util import metrics
        >>> def rmse_mae(a,f):
        >>>     # average of rmse and mae
        >>>     return (metrics.rmse(a,f) + metrics.mae(a,f)) / 2
        >>> f.add_metric(rmse_mae)
        >>> f.set_validation_metric('rmse_mae') # optimize models using this metric
        """
        _developer_utils.descriptive_assert(
            len(inspect.signature(func).parameters) == 2,
            ValueError,
            "The passed function must take exactly two arguments."
        )
        called = self._called(func,called)
        self.metrics[called] = func

    def _called(self,func,called):
        return func.__name__ if called is None else called

    def auto_forecast(
        self,
        call_me=None,
        dynamic_testing=True,
        test_again=True,
    ):
        """ Auto forecasts with the best parameters indicated from the tuning process.

        Args:
            call_me (str): Optional.
                What to call the model when storing it in the object's history dictionary.
                If not specified, the model's nickname will be assigned the estimator value ('mlp' will be 'mlp', etc.).
                Duplicated names will be overwritten with the most recently called model.
            dynamic_testing (bool or int): Default True.
                Whether to dynamically/recursively test the forecast (meaning AR terms will be propagated with predicted values).
                If True, evaluates dynamically over the entire out-of-sample slice of data.
                If int, window evaluates over that many steps (2 for 2-step dynamic forecasting, 12 for 12-step, etc.).
                Setting this to False or 1 means faster performance, 
                but gives a less-good indication of how well the forecast will perform more than one period out.
                The model will skip testing if the test_length attribute is set to 0.

        >>> f.set_estimator('xgboost')
        >>> f.tune()
        >>> f.auto_forecast()
        """
        if not hasattr(self, "best_params"):
            warnings.warn(
                "Since tune() or cross_validate() has not been called,"
                f" {self.estimator} model will be run with default hyperparameters.",
                category = Warning,
            )
            self.best_params = {}
            self.validation_metric_value = np.nan
        self.manual_forecast(
            call_me=call_me,
            dynamic_testing=dynamic_testing,
            test_again = test_again,
            **self.best_params,
        )

    def manual_forecast(
        self, call_me=None, dynamic_testing=True, test_again = True, bank_history = True, **kwargs
    ):
        """ Manually forecasts with the hyperparameters, Xvars, and normalizer selection passed as keywords.
        See https://scalecast.readthedocs.io/en/latest/Forecaster/_forecast.html.

        Args:
            call_me (str): Optional.
                What to call the model when storing it in the object's history.
                If not specified, the model's nickname will be assigned the estimator value ('mlp' will be 'mlp', etc.).
                Duplicated names will be overwritten with the most recently called model.
            dynamic_testing (bool or int): Default True.
                Whether to dynamically/recursively test the forecast (meaning AR terms will be propagated with predicted values).
                If True, evaluates dynamically over the entire out-of-sample slice of data.
                If int, window evaluates over that many steps (2 for 2-step dynamic forecasting, 12 for 12-step, etc.).
                Setting this to False or 1 means faster performance, 
                but gives a less-good indication of how well the forecast will perform more than one period out.
                The model will skip testing if the test_length attribute is set to 0.
            test_again (bool): Default True.
                Whether to test the model before forecasting to a future horizon.
                If test_length is 0, this is ignored. Set this to False if you tested the model manually by calling f.test()
                and don't want to waste resources testing it again.
            **kwargs: passed to the _forecast_{estimator}() method and can include such parameters as Xvars, 
                normalizer, cap, and floor, in addition to any given model's specific hyperparameters.
                See https://scalecast.readthedocs.io/en/latest/Forecaster/_forecast.html.

        >>> f.set_estimator('lasso')
        >>> f.manual_forecast(alpha=.5)
        """
        _developer_utils.descriptive_assert(
            isinstance(call_me, str) | (call_me is None),
            ValueError,
            "call_me must be a str type or None.",
        )

        _developer_utils.descriptive_assert(
            len(self.future_dates) > 0,
            ForecastError,
            "Before calling a model, please make sure you have generated future dates"
            " by calling generate_future_dates(), set_last_future_date(), or ingest_Xvars_df(use_future_dates=True).",
        )

        self._clear_the_deck() # delete some attributes from last model
        
        self.dynamic_testing = dynamic_testing
        self.call_me = self.estimator if call_me is None else call_me

        if test_again and self.test_length > 0 and self.estimator != 'combo':
            self.test(
                **kwargs,
                dynamic_testing=dynamic_testing,
                call_me=call_me,
            )
        elif bank_history:
            if self.call_me not in self.history.keys():
                self.history[self.call_me] = {}

        preds = (
            self._forecast_sklearn(
                fcster=self.estimator,
                dynamic_testing=dynamic_testing,
                **kwargs,
            )
            if self.estimator in self.sklearn_estimators
            else getattr(self, f"_forecast_{self.estimator}")(
                dynamic_testing=dynamic_testing, **kwargs
            )
        )
        self.forecast = preds
        if bank_history:
            self._bank_history(**kwargs)

    def eval_cis(self,mode=True,cilevel=.95):
        """ Call this function to change whether or not the Forecaster sets confidence intervals on all evaluated models.
        Beginning 0.17.0, only conformal confidence intervals are supported. Conformal intervals need a test set to be configured soundly.
        Confidence intervals cannot be evaluated when there aren't at least 1/(1-cilevel) observations in the test set.

        Args:
            mode (bool): Default True. Whether to set confidence intervals on or off for models.
            cilevel (float): Default .95. Must be greater than 0, less than 1. The confidence level
                to use to set intervals.
        """
        if mode is True:
            self._check_right_test_length_for_cis(cilevel)
        
        self.cis=mode
        self.set_cilevel(cilevel)

    def ingest_grid(self, grid):
        """ Ingests a grid to tune the estimator.

        Args:
            grid (dict or str):
                If dict, must be a user-created grid.
                If str, must match the name of a dict grid stored in a grids file.

        Returns:
            None

        >>> f.set_estimator('mlr')
        >>> f.ingest_grid({'normalizer':['scale','minmax']})
        """
        from itertools import product

        def expand_grid(grid):
            # returns a list of dictionaries
            keys = grid.keys()
            values = grid.values()

            # Generate all possible combinations of hyperparameter values
            combinations = product(*values)

            # Convert combinations to list of dictionaries
            hyperparameter_combinations = [dict(zip(keys, combination)) for combination in combinations]
            return hyperparameter_combinations

        try:
            if isinstance(grid, str):
                Grids = importlib.import_module(self.grids_file)
                importlib.reload(Grids)
                grid = getattr(Grids, grid)
        except SyntaxError:
            raise
        except:
            raise ForecastError(
                f"Tried to load a grid called {self.estimator} from {self.grids_file}.py, "
                "but either the file could not be found in the current directory, "
                "there is no grid with that name, or the dictionary values are not list-like. "
                "Try the ingest_grid() method with a dictionary grid passed manually."
            )
        grid = expand_grid(grid)
        self.grid = grid

    def limit_grid_size(self, n, min_grid_size = 1, random_seed=None):
        """ Makes a grid smaller randomly.

        Args:
            n (int or float):
                If int, randomly selects that many parameter combinations.
                If float, must be less than 1 and greater 0, randomly selects that percentage of parameter combinations.
            min_grid_size (int): Default 1.
                The min number of hyperparameters to keep from the original grid if a float is passed to n.
            random_seed (int): Optional.
                Set a seed to make results consistent.

        Returns:
            None

        >>> from scalecast import GridGenerator
        >>> GridGenerator.get_example_grids()
        >>> f.set_estimator('mlp')
        >>> f.ingest_grid('mlp')
        >>> f.limit_grid_size(10,random_seed=20) # limits grid to 10 iterations
        >>> f.limit_grid_size(.5,random_seed=20) # limits grid to half its original size
        """
        if random_seed is not None:
            random.seed(random_seed)

        if (n < 1) & (n > 0):
            n = len(self.grid) * n

        n = int(
            min(
                len(self.grid),
                max(min_grid_size,n),
            )
        )
        self.grid = random.sample(self.grid,n)

    def set_metrics(self,metrics):
        """ Set or change the evaluated metrics for all model testing and validation.

        Args:
            metrics (list): The metrics to evaluate when validating
                and testing models. Each element must exist in utils.metrics and take only two arguments: a and f.
                See https://scalecast.readthedocs.io/en/latest/Forecaster/Util.html#metrics.
                For each metric and model that is tested, the test-set and in-sample metrics will be evaluated and can be
                exported. Level test-set and in-sample metrics are also currently available, but will be removed in a future version.
        """
        bad_metrics = [met for met in metrics if isinstance(met,str) and met not in __metrics__]
        if len(bad_metrics) > 0:
            raise ValueError(
                f'Each element in metrics must be one of {list(__metrics__.keys())} or be a function.'
                f' Got the following invalid values: {bad_metrics}.'
            )
        self.metrics = {}
        for met in metrics:
            if isinstance(met,str):
                self.metrics[met] = __metrics__[met]
            else:
                self.add_metric(met)
        self.determine_best_by = _developer_utils._determine_best_by(self.metrics)

    def set_validation_metric(self, metric):
        """ Sets the metric that will be used to tune all subsequent models.

        Args:
            metric: One of Forecaster.metrics.
                The metric to optimize the models with using the validation set.
                Although model testing will evaluate all metrics in Forecaster.metrics,
                model optimization with tuning and cross validation only uses one of these.

        Returns:
            None

        >>> f.set_validation_metric('mae')
        """
        _developer_utils.descriptive_assert(
            metric in self.metrics,
            ValueError,
            f"metric must be one of {list(self.metrics)}, got {metric}.",
        )
        self.validation_metric = metric

    def set_test_length(self, n=1):
        """ Sets the length of the test set. As of version 0.16.0, 0-length test sets are supported.

        Args:
            n (int or float): Default 1.
                The length of the resulting test set.
                Pass 0 to skip testing models.
                Fractional splits are supported by passing a float less than 1 and greater than 0.

        Returns:
            None

        >>> f.set_test_length(12) # test set of 12
        >>> f.set_test_length(.2) # 20% test split
        """
        float(n)
        if n == 0:
            self.test_length = 0
        if n >= 1:
            n = int(n)
            _developer_utils.descriptive_assert(
                isinstance(n, int),
                ValueError,
                f"n must be an int of at least 0 or float greater than 0 and less than 1, got {n} of type {type(n)}.",
            )
            self.test_length = n
        else:
            _developer_utils.descriptive_assert(
                n >= 0,
                ValueError,
                f"n must be an int of at least 0 or float greater than 0 and less than 1, got {n} of type {type(n)}.",
            )
            leny = len(self.y[self.names[0]]) if isinstance(self.y,dict) else len(self.y) 
            self.test_length = int(leny * n)

    def set_validation_length(self, n=1):
        """ Sets the length of the validation set. This will never matter for models that are not tuned.

        Args:
            n (int): Default 1.
                The length of the resulting validation set.

        Returns:
            None

        >>> f.set_validation_length(6) # validation length of 6
        """
        n = int(n)
        _developer_utils.descriptive_assert(n > 0, ValueError, f"n must be greater than 0, got {n}.")
        if (self.validation_metric == "r2") & (n == 1):
            raise ValueError(
                "Can only set a validation_length of 1 if validation_metric is not r2. Try calling set_validation_metric() with a different metric."
                f"Possible values are: {_metrics_}."
            )
        self.validation_length = n

    def set_cilevel(self, n):
        """ Sets the level for the resulting confidence intervals (95% default).

        Args:
            n (float): Greater than 0 and less than 1.

        Returns:
            None

        >>> f.set_cilevel(.80) # next forecast will get 80% confidence intervals
        """
        _developer_utils.descriptive_assert(
            n < 1 and n > 0, ValueError, "n must be a float greater than 0 and less than 1."
        )
        self.cilevel = n

    def set_estimator(self, estimator):
        """ Sets the estimator to forecast with.

        Args:
            estimator (str): One of Forecaster.estimators.

        Returns:
            None

        >>> f.set_estimator('lasso')
        >>> f.manual_forecast(alpha = .5)
        """
        _developer_utils._check_if_correct_estimator(estimator,self.estimators)
        if hasattr(self, "estimator"):
            if estimator != self.estimator:
                for attr in (
                    "grid",
                    "grid_evaluated",
                    "best_params",
                    "validation_metric_value",
                    "actuals",
                ):
                    if hasattr(self, attr):
                        delattr(self, attr)
                self._clear_the_deck()
                self.estimator = estimator
        else:
            self.estimator = estimator

    def set_grids_file(self,name='Grids'):
        """ Sets the name of the file where the object will look automatically for grids when calling 
        `tune()`, `cross_validate()`, `tune_test_forecast()`, or similar function.
        If the grids file does not exist in the working directory, the error will only be raised once tuning is called.
        
        Args:
            name (str): Default 'Grids'.
                The name of the file to look for.
                This file must exist in the working directory.
                The default will look for a file called "Grids.py".

        >>> f.set_grids_file('ModGrids') # expects to find a file called ModGrids.py in working directory.
        """
        _developer_utils.descriptive_assert(
            isinstance(name,str),
            ValueError,
            f'name argument expected str type, got {type(name)}.'
        )
        self.grids_file = name

    def generate_future_dates(self, n):
        """ Generates a certain amount of future dates in same frequency as current_dates.

        Args:
            n (int): Greater than 0.
                Number of future dates to produce.
                This will also be the forecast length.

        Returns:
            None

        >>> f.generate_future_dates(12) # 12 future dates to forecast out to
        """
        self.future_dates = pd.Series(
            pd.date_range(
                start=self.current_dates.values[-1], periods=n + 1, freq=self.freq
            ).values[1:]
        )

    def set_last_future_date(self, date):
        """ Generates future dates in the same frequency as current_dates that ends on a specified date.

        Args:
            date (datetime.datetime, pd.Timestamp, or str):
                The date to end on. Must be parsable by pandas' Timestamp() function.

        Returns:
            None

        >>> f.set_last_future_date('2021-06-01') # creates future dates up to this one in the expected frequency
        """
        self.future_dates = pd.Series(
            pd.date_range(
                start=self.current_dates.values[-1], end=date, freq=self.freq
            ).values[1:]
        )

    def add_lagged_terms(self, *args, lags=1, upto=True, sep="_", drop = False):
        """ Lags all passed variables (no AR terms) 1 or more times.

        Args:
            *args (str): Names of Xvars that aleady exist in the object.
            lags (int): Greater than 0, default 1.
                The number of times to lag each passed variable.
            upto (bool): Default True.
                Whether to add all lags up to the number passed to lags.
                If you pass 6 to lags and upto is True, lags 1, 2, 3, 4, 5, 6 will all be added.
                If you pass 6 to lags and upto is False, lag 6 only will be added.
            sep (str): Default '_'.
                The separator between each term in arg to create the final variable name.
                Resulting variable names will be like "tlag_1" or "tlag_2" by default.
            drop (bool): Default False.
                Whether to drop the regressors passed to *args.

        Returns:
            None

        >>> add_lagged_terms('t',lags=3) # adds first, second, and third lag of t called 'tlag_1' - 'tlag_3'
        >>> add_lagged_terms('t',lags=6,upto=False) # adds 6th lag of t only called 'tlag_6'
        """
        #self._validate_future_dates_exist()
        lags = int(lags)
        _developer_utils.descriptive_assert(
            lags >= 1,
            ValueError,
            f"lags must be an int type greater than 0, got {lags}.",
        )
        for a in args:
            _developer_utils.descriptive_assert(
                not a.startswith("AR"),
                ForecastError,
                "Adding lagged AR terms makes no sense, add more AR terms instead.",
            )
            if upto:
                for i in range(1, lags + 1):
                    self.current_xreg[f"{a}lag{sep}{i}"] = self.current_xreg[a].shift(i)
                    fx = (
                        pd.Series(self.current_xreg[a].to_list() + self.future_xreg[a])
                        .shift(i)
                        .to_list()
                    )
                    self.future_xreg[f"{a}lag{sep}{i}"] = fx[-len(self.future_dates) :]
            else:
                self.current_xreg[f"{a}lag{sep}{lags}"] = self.current_xreg[a].shift(
                    lags
                )
                fx = (
                    pd.Series(self.current_xreg[a].to_list() + self.future_xreg[a])
                    .shift(lags)
                    .to_list()
                )
                self.future_xreg[f"{a}lag{sep}{lags}"] = fx[-len(self.future_dates) :]

        ars = [int(x[2:]) for x in self.current_xreg if x.startswith('AR')]
        ars = max(ars) if len(ars) else 0
        if lags > ars:
            if not isinstance(self.y,dict): # Forecaster
                warnings.warn(
                    f'With the introduction of {lags} lags, there are now more n/a observations than are introduced with {ars} '
                    f'autoregressive terms. This can cause failures in models. Try running chop_from_back({lags-ars}) but be careful '
                    'if series transformations were taken, especially, DiffTransform().',
                    category = Warning,
                )
            else: # MVForecaster
                warnings.warn(
                    f'If models are run with fewer than {lags} lags, models can fail.',
                    category = Warning,
                )

        if drop:
            self.drop_Xvars(*args)

    def add_series(
        self,
        series,
        called,
        first_date=None,
        forward_pad=True,
        back_pad=True
    ):
        """ Adds other series to the object as regressors. 
        If the added series is less than the length of Forecaster.y + len(Forecaster.future_dates), 
        it will padded with 0s by default.

        Args:
            series (list-like): The series to add as a regressor to the object.
            called (str): Required. What to call the resulting regressor in the Forecaster object.
            first_date (Datetime): Optional. The first date that corresponds with the added series.
                If left unspecified, will assume its first date is the same as the first
                date in the Forecaster object.
                Must be datetime or otherwise able to be parsed by the pandas.Timestamp() function.
            pad (bool): Default True. 
                Whether to put 0s before and/or after the series if the series is too short.

        >>> x = [1,2,3,4,5,6]
        >>> f.add_series(series = x,called='x') # assumes first date is same as what is in f.current_dates
        """
        if first_date is None:
            first_date = self.current_dates.min()

        df = pd.DataFrame({
            'Date':pd.date_range(start=first_date,periods=len(series),freq=self.freq).to_list(),
            called:list(series),
        })

        self.ingest_Xvars_df(df,pad=True)

    def ingest_Xvars_df(
        self, df, date_col="Date", drop_first=False, use_future_dates=False, pad = False,
    ):
        """ Ingests a dataframe of regressors and saves its Xvars to the object.
        The user must specify a date column name in the dataframe being ingested.
        All non-numeric values are dummied.
        The dataframe should cover the entire future horizon stored within the Forecaster object, but can be padded with 0s
        if testing only is desired.
        Any columns in the dataframe that begin with "AR" will be confused with autoregressive terms and could cause errors.

        Args:
            df (DataFrame): The dataframe that is at least the length of the y array stored in the object plus the forecast horizon.
            date_col (str): Default 'Date'.
                The name of the date column in the dataframe.
                This column must have the same frequency as the dates stored in the Forecaster object.
            drop_first (bool): Default False.
                Whether to drop the first observation of any dummied variables.
                Irrelevant if passing all numeric values.
            use_future_dates (bool): Default False.
                Whether to use the future dates in the dataframe as the resulting future_dates attribute in the Forecaster object.
            pad (bool): Default False.
                Whether to pad any missing values with 0s.

        Returns:
            None
        """
        _developer_utils.descriptive_assert(
            df.shape[0] == len(df[date_col].unique()),
            ValueError,
            "Each date supplied must be unique.",
        )
        df_copy = df.copy()
        df_copy[date_col] = pd.to_datetime(df_copy[date_col])
        df_copy = df_copy.loc[df_copy[date_col] >= self.current_dates.values[0]]
        df_copy = pd.get_dummies(df_copy, drop_first=drop_first)
        current_df = df_copy.loc[df_copy[date_col].isin(self.current_dates)]
        future_df = pd.DataFrame({date_col: self.future_dates.to_list()})
        future_df = df_copy.loc[df_copy[date_col] > self.current_dates.values[-1]]

        fpad_len = 0
        bpad_len = 0
        if current_df.shape[0] < len(self.y):
            if pad:
                fpad_len = len(self.y) - current_df.shape[0]
            else:
                raise ForecastError(
                "Could not ingest Xvars dataframe."
                " Make sure the dataframe spans the entire date range as y and is at least one observation to the future."
                " Specify the date column name in the `date_col` argument.",
                )
        if not use_future_dates:
            if future_df.shape[0] < len(self.future_dates):
                if pad:
                    bpad_len = (len(self.future_dates)) - future_df.shape[0]
                else:
                    raise ForecastError(
                        "The future dates in the dataframe should be at least the same"
                        " length as the future dates in the Forecaster object." 
                        " If you want to use the dataframe to set the future dates for the object,"
                        " pass True to the use_future_dates argument.",
                    )
        else:
            self.future_dates = future_df[date_col]
        for c in [c for c in future_df if c != date_col]:
            self.future_xreg[c] = future_df[c].to_list()[: len(self.future_dates)] + [0] * bpad_len
            self.current_xreg[c] = pd.Series([0] * fpad_len + current_df[c].to_list(),dtype=float)

    def export_validation_grid(self, model) -> pd.DataFrame:
        """ Exports the validation grid from a model, converted to a pandas dataframe.
        Raises an error if the model was not tuned.

        Args:
            model (str):
                The name of them model to export for.
                Matches what was passed to call_me when evaluating the model.
        Returns:
            (DataFrame): The resulting validation grid of the evaluated model passed to model arg.
        """
        hist = self.history[model]
        df = pd.DataFrame(columns = list(hist['grid'][0].keys()))
        for i, grid in enumerate(hist['grid']):
            for k,v in grid.items():
                df.loc[i,k] = v
        for j, h in enumerate(hist['grid_evaluated'].T):
            df[f'Fold{j}Metric'] = h
        df['AverageMetric'] = np.mean(hist['grid_evaluated'],axis=1)
        df['MetricEvaluated'] = hist['ValidationMetric']
        if hasattr(self,'optimize_on'):
            df['Optimized On'] = self.optimize_on
        return df

    def test(
        self,
        dynamic_testing=True,
        call_me=None,
        **kwargs,
    ):
        """ Tests the forecast estimator out-of-sample. Uses the test_length attribute to determine on how-many observations.
        All test-set splits maintain temporal order.

        Args:
            dynamic_testing (bool or int): Default True.
                Whether to dynamically/recursively test the forecast (meaning AR terms will be propagated with predicted values).
                If True, evaluates dynamically over the entire out-of-sample slice of data.
                If int, window evaluates over that many steps (2 for 2-step dynamic forecasting, 12 for 12-step, etc.).
                Setting this to False or 1 means faster performance, 
                but gives a less-good indication of how well the forecast will perform more than one period out.
                This will fail if the test_length attribute is 0.
            call_me (str): Optional.
                What to call the model when storing it in the object's history.
                If not specified, the model's nickname will be assigned the estimator value ('mlp' will be 'mlp', etc.).
                Duplicated names will be overwritten with the most recently called model.
            **kwargs: passed to the _forecast_{estimator}() method and can include such parameters as Xvars, 
                normalizer, cap, and floor, in addition to any given model's specific hyperparameters.
                See https://scalecast.readthedocs.io/en/latest/Forecaster/_forecast.html.

        >>> f.set_estimator('lasso')
        >>> f.test(alpha=.5)
        """
        if self.test_length == 0:
            raise ValueError(
                'Cannot test models when test_length is 0.'
                ' Call f.set_test_length() to generate a test-set for this object.'
            )

        is_Forecaster = not isinstance(self.y,dict)
        call_me = self.estimator if call_me is None else call_me
        if call_me not in self.history:
            self.history[call_me] = {}
            already_existed = False
        else:
            already_existed = True

        if is_Forecaster:
            actuals = self.y.to_list()[-self.test_length:]
        else:
            actuals = [
                self.y[k].to_list()[-self.test_length:]
                for k in self.y
            ]

        f1 = self.deepcopy()
        f1.chop_from_front(
            self.test_length,
            fcst_length = self.test_length,
        )
        f1.set_test_length(0)
        f1.eval_cis(False)
        f1.actuals = actuals
        f1.manual_forecast(
            dynamic_testing = dynamic_testing,
            test_again = False, 
            call_me = call_me, 
            **kwargs,
        )
        fcst = f1.forecast

        if not already_existed:
            attrs_to_copy = (
                'Estimator',
                'Xvars',
                'HyperParams',
                'Lags',
                'regr',
                'X',
            )
            self.history[call_me] = {
                k:(v if k in attrs_to_copy else None) for 
                k, v in f1.history[call_me].items()
            }
        self.call_me = call_me
        self.history[call_me]['TestSetLength'] = self.test_length
        self.history[call_me]['TestSetPredictions'] = fcst
        self.history[call_me]['TestSetActuals'] = ( 
            actuals if is_Forecaster
            else {k :actuals[i] for i, k in enumerate(fcst.keys())}
        )

        for met, func in self.metrics.items():
            if is_Forecaster:
                self.history[call_me]['TestSet' + met.upper()] = func(actuals,fcst)
            else:
                self.history[call_me]['TestSet' + met.upper()] = {
                    k:func(actuals[i],fcst[k]) for i, k in enumerate(fcst.keys())
                } # little awkward, but necessary for cross_validate to work more efficiently

    def tune(self, dynamic_tuning=False,set_aside_test_set=True):
        """ Tunes the specified estimator using an ingested grid (ingests a grid from Grids.py with same name as 
        the estimator by default). This is akin to cross-validation with one fold and a test_length equal to f.validation_length.
        Any parameters that can be passed as arguments to manual_forecast() can be tuned with this process.
        The chosen parameters are stored in the best_params attribute.
        The evaluated validation grid can be exported to a dataframe using f.export_validation_grid().

        Args:
            dynamic_tuning (bool or int): Default False.
                Whether to dynamically/recursively test the forecast during the tuning process 
                (meaning AR terms will be propagated with predicted values).
                If True, evaluates recursively over the entire out-of-sample slice of data.
                If int, window evaluates over that many steps (2 for 2-step recurvie testing, 12 for 12-step, etc.).
                Setting this to False or 1 means faster performance, 
                but gives a less-good indication of how well the forecast will perform more than one period out.
            set_aside_test_set (bool): Default True. Whether to separate the test set specified in f.test_length during this process.

        Returns:
            None

        >>> f.set_estimator('xgboost')
        >>> f.tune()
        >>> f.auto_forecast()
        """
        self.cross_validate(
            k=1,
            dynamic_tuning=dynamic_tuning,
            test_length=self.validation_length,
            set_aside_test_set = set_aside_test_set,
        )

    def cross_validate(
        self, 
        k=5, 
        test_length = None,
        train_length = None,
        space_between_sets = None,
        rolling=False, 
        dynamic_tuning=False,
        set_aside_test_set=True,
        verbose=False,
    ):
        """ Tunes a model's hyperparameters using time-series cross validation. 
        Monitors the metric specified in the valiation_metric attribute. 
        Set an estimator before calling. 
        Reads a grid for the estimator from a grids file unless a grid is ingested manually. 
        The chosen parameters are stored in the best_params attribute.
        All metrics from each iteration are stored in grid_evaluated. The rows in this matrix correspond to the element index in f.grid (a hyperparameter combo)
        and the columns are the derived metrics across the k folds. Any hyperparameters that ever failed to evaluate will return N/A and are not considered.
        The best parameter combo is determined by the best average derived matrix across all folds.
        The temporal order of the series is always maintained in this process.
        If a test_length is specified in the object, it will be set aside by default.
        (Default) Normal cv diagram: https://scalecast-examples.readthedocs.io/en/latest/misc/validation/validation.html#5-Fold-Time-Series-Cross-Validation.
        (Default) Rolling cv diagram: https://scalecast-examples.readthedocs.io/en/latest/misc/validation/validation.html#5-Fold-Rolling-Time-Series-Cross-Validation. 

        Args:
            k (int): Default 5. 
                The number of folds. 
                If 1, behaves as if the model were being tuned on a single held out set.
            test_length (int): Optional. The size of each held-out sample. 
                By default, determined such that the last test set and train set are the same size.
            train_length (int): Optional.  The size of each training set.
                By default, all available observations before each test set are used.
            space_between_sets (int): Optional. The space between each training set.
                By default, uses the test_length.
            rolling (bool): Default False. Whether to use a rolling method, meaning every train and test size is the same. 
                This is ignored when either of train_length or test_length is specified.
            dynamic_tuning (bool or int): Default False.
                Whether to dynamically/recursively test the forecast during the tuning process 
                (meaning AR terms will be propagated with predicted values).
                If True, evaluates recursively over the entire out-of-sample slice of data.
                If int, window evaluates over that many steps (2 for 2-step recurvie testing, 12 for 12-step, etc.).
                Setting this to False or 1 means faster performance, 
                but gives a less-good indication of how well the forecast will perform more than one period out.
            set_aside_test_set (bool): Default True. Whether to separate the test set specified in f.test_length during this process.
            verbose (bool): Default False. Whether to print out information about the test size, train size, and date ranges for each fold.

        Returns:
            None

        >>> f.set_estimator('xgboost')
        >>> f.cross_validate() # tunes hyperparam values
        >>> f.auto_forecast() # forecasts with the best params
        """
        if not hasattr(self, "grid"):
            self.ingest_grid(self.estimator)

        is_Forecaster = not isinstance(self.y,dict)
        rolling = bool(rolling)
        k = int(k)
        _developer_utils.descriptive_assert(k >= 1, ValueError, f"k must be at least 1, got {k}.")
        f1 = self.deepcopy()
        f1.eval_cis(False)
        if not set_aside_test_set:
            f1.set_test_length(0)
        if is_Forecaster:
            usable_obs = len(f1.y) - f1.test_length
        else:
            usable_obs = len(f1.y[f1.names[0]]) - f1.test_length
        if test_length is None:
            test_length = usable_obs // (k + 1)
        if train_length is None:
            train_length = test_length if rolling else train_length
        if space_between_sets is None:
            space_between_sets = test_length

        grid = self.grid
        func = self.metrics[self.validation_metric] # function to evaluate metric for each grid try
        iters = len(grid)
        metrics = np.zeros((iters,k)) # each row is a hyperparam try and column is a fold
        for i in range(k):
            val_chop = test_length + space_between_sets * i
            ttl_chop = val_chop + f1.test_length
            if -ttl_chop + test_length == 0 and i == 0: # -ttl_chop + test_length can equal 0 on the first iteration, in which case, we want the following behavior
                if is_Forecaster:
                    actuals = f1.y.to_list()[-ttl_chop:]
                else:
                    actuals = [
                        f1.y[k].to_list()[-ttl_chop:]
                        for k in f1.y
                    ]
            else: 
                if is_Forecaster:
                    actuals = f1.y.to_list()[-ttl_chop:(-ttl_chop + test_length)]
                else:
                    actuals = [
                        f1.y[k].to_list()[-ttl_chop:(-ttl_chop + test_length)]
                        for k in f1.y
                    ]
            err_message = f'Something went wrong with determining set lengths. Should be {test_length}.'
            if is_Forecaster:
                assert len(actuals) == test_length,err_message + f' Got {len(actuals)}.'
            else:
                assert np.all([len(a) == test_length for a in actuals]),err_message
            f2 = f1.deepcopy()
            f2.chop_from_front(ttl_chop,fcst_length=test_length)
            f2.actuals = actuals
            if train_length is not None:
                f2.keep_smaller_history(train_length)
            if verbose:
                if i == 0:
                    print(f'Num hyperparams to try for the {self.estimator} model: {len(grid)}.')
                if is_Forecaster:
                    print(
                        f'Fold {i}: Train size: {len(f2.y)} ({f2.current_dates.min()} - {f2.current_dates.max()}). '
                        f'Test Size: {len(f2.actuals)} ({f2.future_dates.min()} - {f2.future_dates.max()}). '
                    )
                else:
                    print(
                        f'Fold {i}: Train size: {len(f2.y[f2.names[0]])} ({f2.current_dates.min()} - {f2.current_dates.max()}). '
                        f'Test Size: {len(f2.actuals[0])} ({f2.future_dates.min()} - {f2.future_dates.max()}). '
                    )

            for h, hp in enumerate(grid):
                try:
                    f2.manual_forecast(
                        **hp,
                        dynamic_testing=dynamic_tuning,
                        test_again=False,
                        bank_history=False,
                    )
                    if is_Forecaster:
                        fcst = f2.forecast[:]
                        evaluated_metric = func(actuals,fcst)
                    else:
                        fcst = [v[:] for k, v in f2.forecast.items()]
                        evaluated_metrics = [func(a,f) for a, f in zip(actuals,fcst)]
                        evaluated_metric = self.optimizer_funcs[self.optimize_on](evaluated_metrics) # mean, particular series, etc.
                except (TypeError,ForecastError):
                    raise
                except Exception as e:
                    #raise # good to uncomment when debugging
                    evaluated_metric = np.nan
                    logging.warning(
                        f"Could not evaluate the paramaters: {hp}. error: {e}",
                    )
                metrics[h,i] = evaluated_metric

        self.grid_evaluated = metrics
        if np.all(np.isnan(metrics[:,-1])):
            warnings.warn(
                f'The last CV fold will not be considered when choosing hyperparemeters for the {self.estimator} model, '
                'as all parameters failed to return a metric. '
                'This most frequently happens when default CV parameters were used with an RNN model.',
                category=Warning,
            )
            metrics = metrics[:,:-1]
        avg_mets = np.mean(metrics,axis=1) # any hps that ever evaluated na not in consideration
        if np.all(np.isnan(avg_mets)):
            warnings.warn(
                f"None of the keyword/value combos stored in the grid could be evaluated for the {self.estimator} model."
                " See the errors in warnings.log.",
                category=Warning,
            )
            self.best_params = {}
            self.validation_metric_value = np.nan
        else:
            best_hp_idx = np.nanargmin(avg_mets) if self.validation_metric != 'r2' else np.nanargmax(avg_mets)
            self.best_params = grid[best_hp_idx]
            self.validation_metric_value = avg_mets[best_hp_idx]
        if verbose:
            print(f'Chosen paramaters: {self.best_params}.')

    def _fit_normalizer(self, X, normalizer):
        _developer_utils.descriptive_assert(
            normalizer in self.normalizer,
            ValueError,
            f"normalizer must be one of {list(self.normalizer.keys())}, got {normalizer}.",
        )
        
        if normalizer is None:
            return None

        scaler = self.normalizer[normalizer]()
        scaler.fit(X)
        return scaler

    def _scale(self, scaler,X):
        return scaler.transform(X) if scaler is not None else X

def _tune_test_forecast(
    f,
    models,
    cross_validate,
    dynamic_tuning,
    dynamic_testing,
    limit_grid_size,
    suffix,
    error,
    min_grid_size = 1,
    summary_stats = False,
    feature_importance = False,
    fi_method=None,
    tqdm = False,
    **cvkwargs,
):
    if tqdm: # notebooks only get progress bar
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
            f.save_feature_importance(method=fi_method)