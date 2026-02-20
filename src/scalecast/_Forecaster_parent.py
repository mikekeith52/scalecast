from __future__ import annotations
from .cfg import (
    METRICS,
    NORMALIZERS,
    ESTIMATORS,
    CLEAR_ATTRS_ON_ESTIMATOR_CHANGE,
)
from ._utils import _developer_utils
from .types import (
    DatetimeLike, 
    ConfInterval,
    PositiveInt,
    NonNegativeInt, 
    DynamicTesting, 
    EvaluatedModel, 
    AvailableXvar, 
    DefaultMetric, 
    AvailableModel,
    AvailableNormalizer,
    XvarValues,
)
from .typing_utils import ScikitLike, NormalizerLike
from .classes import AR, Estimator, EvaluatedMetric, DetermineBestBy, MetricStore, ValidatedList, DefaultNormalizer
from .models import SKLearnUni
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
from typing import Sequence, Self, Optional, Literal, Any, TYPE_CHECKING
if TYPE_CHECKING:
    from ._Forecaster_parent import _Forecaster_parent

# descriptive errors

class ForecastError(Exception):
    pass

class Forecaster_parent:
    def __init__(
        self,
        y:Sequence[float|int],
        test_length:NonNegativeInt,
        cis:bool,
        metrics:ValidatedList[MetricStore]=METRICS[:4]
    ):
        self._logging()
        self.y = y
        self.estimators = ESTIMATORS # this will be overwritten with Forecaster but maintained in MVForecaster
        self.normalizer = NORMALIZERS # TODO make this its own class
        self.set_test_length(test_length)
        self.validation_length = 1
        self.validation_metric = metrics[0]
        self.cilevel = 0.95
        self.current_xreg = {} # Series
        self.future_xreg = {} # lists
        self.history = {}
        self.metrics = metrics
        self.determine_best_by = DetermineBestBy(metrics,metrics[0])
        self.set_estimator("mlr")
        self.eval_cis(mode = cis, cilevel = self.cilevel)

    def __copy__(self):
        if hasattr(self,'tf_model'):
            delattr(self,'tf_model')
        obj = type(self).__new__(self.__class__)
        obj.__dict__.update(self.__dict__)
        return obj

    def __deepcopy__(self, memo):
        if id(self) in memo:
            return memo[id(self)]
        
        if hasattr(self,'tf_model'):
            delattr(self,'tf_model')

        # Create new instance without calling __init__
        cls = self.__class__
        result = cls.__new__(cls)

        # Store in memo before copying attributes
        memo[id(self)] = result

        # Deep copy attributes
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memo))

        return result

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __getstate__(self):
        self._logging()
        state = self.__dict__.copy()
        return state

    def _check_right_test_length_for_cis(self,cilevel:ConfInterval):
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
            "ymin",
            "ymax",
        ):
            try:
                delattr(self, attr)
            except AttributeError:
                pass

    def copy(self):
        """ Creates an object copy.
        """
        return self.__copy__()

    def add_seasonal_regressors(
        self, 
        *args:str, 
        raw:bool=True, 
        sincos:bool=False, 
        dummy:bool=False, 
        drop_first:bool=False,
        cycle_lens:dict[str,int]=None,
        fourier_order:float = 2.0,
    ) -> Self:
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
                Each key should match a value passed to *args.
                If this is not specified or a selected seasonality is not added to the dictionary as a key, the
                cycle length will be selected automatically as the maximum value observed for the given seasonality.
                Not relevant when sincos = False.
            fourier_order (float): Default 2.0. The fourier order to apply to terms that are added using `sincos = True`.
                This number is the number of complete cycles in that given seasonal period. 2 captures the fundamental frequency
                and its first harmonic. Higher orders will capture more complex seasonality, but may lead to overfitting.

        Returns:
            Self

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
                self.current_xreg[f"{s}sin"] = np.sin(np.pi * _raw / (_cycles / fourier_order))
                self.current_xreg[f"{s}cos"] = np.cos(np.pi * _raw / (_cycles / fourier_order))
                self.future_xreg[f"{s}sin"] = np.sin(
                    np.pi * _raw_fut / (_cycles / fourier_order)
                ).to_list()
                self.future_xreg[f"{s}cos"] = np.cos(
                    np.pi * _raw_fut / (_cycles / fourier_order)
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

    def add_time_trend(self, called:str="t") -> Self:
        """ Adds a time trend from 1 to length of the series + the forecast horizon as a current and future Xvar.

        Args:
            Called (str): Default 't'.
                What to call the resulting variable.

        Returns:
            Self

        >>> f.add_time_trend() # adds time trend called 't'
        """
        #self._validate_future_dates_exist()
        self.current_xreg[called] = pd.Series(range(1, len(self.y) + 1))
        self.future_xreg[called] = list(
            range(len(self.y) + 1, len(self.y) + len(self.future_dates) + 1)
        )

        return self

    def transfer_cis(
        self,
        transfer_from:"_Forecaster_parent",
        model:str,
        transfer_to_model:str=None,
        transfer_test_set_cis:Optional[bool]=None,
    ) -> Self:
        """ Transfers the confidence intervals from a model forecast in a passed `Forecaster` or `MVForecaster` object.

        Args:
            transfer_from (Forecaster or MVForecaster): The object that contains the model from which intervals
                should be transferred.
            model (str): The model nickname of the already-evaluated model stored in `transfer_from`.
            transfer_to_model (str): Optional. The nickname of the model to which the intervals should be transferred.
                If not specified, inherits the name passed to `model`.
            transfer_test_set_cis (bool): Optional. Whether to pass intervals for test-set predictions.
                If left unspecified, the decision is made based on whether the inheriting object has
                test-set predictions evaluated.

        Returns:
            Self

        >>> f.manual_forecast(call_me='mlr')
        >>> f_new.transfer_predict(transfer_from=f,model='mlr')
        >>> f_new.transfer_cis(transfer_from=f,model='mlr')
        """
        transfer_to_model = model if transfer_to_model is None else transfer_to_model
        self.history[transfer_to_model]['CILevel'] = transfer_from.history[model]['CILevel']
        try:
            ci_range = (
                transfer_from.history[model]['UpperCI'][-1] - 
                transfer_from.history[model]['Forecast'][-1]
            ) # assume default naive confidence interval
        except KeyError: # mvforecaster
            ci_range = {
                k:v[-1] - transfer_from.history[model]['Forecast'][k][-1]
                for k, v in transfer_from.history[model]['UpperCI'].items()
            }
        self._set_cis(
            "UpperCI",
            "LowerCI",
            m = transfer_to_model,
            ci_range = ci_range,
            preds = self.history[transfer_to_model]["Forecast"],
        )
        
        if transfer_test_set_cis == 'infer':
            transfer_test_set_cis = (
                'TestSetPredictions' in self.history[transfer_to_model] 
                and len(self.history[transfer_to_model]['TestSetPredictions'])
            )
        
        if transfer_test_set_cis:
            self._set_cis(
                "TestSetUpperCI",
                "TestSetLowerCI",
                m = transfer_to_model,
                ci_range = ci_range,
                preds = self.history[transfer_to_model]["TestSetPredictions"],
            )
            if not len(self.history[transfer_to_model]["TestSetPredictions"]):
                warnings.warn(
                    'Test set predictions could not be found in the Forecaster object receiving the transfer.'
                    ' Therefore, no test-set confidence intervals were transferred.',
                    category=Warning,
                )
        return self
    
    def list_stored_ar_terms(self):
        """ Returns a list of all stored autoregressive (AR) terms.

        Returns:
            list: All stored AR terms.
        """
        return [x for x in self.current_xreg if isinstance(x,AR)]
    
    def get_max_lag_order(self):
        """ Returns the highest lag order variable stored in the object. Returns 0 if none were found.

        Returns:
            int: The max order found.
        """
        all_ars = self.list_stored_ar_terms()
        if all_ars:
            return max([x.lag_order for x in all_ars])
        else:
            return 0

    def add_cycle(self, cycle_length:PositiveInt, fourier_order:float = 2.0, called:Optional[str]=None) -> Self:
        """ Adds a regressor that acts as a seasonal cycle. Use this function to capture non-normal seasonality.

        Args:
            cycle_length (int): How many time steps make one complete cycle.
            fourier_order (float): Default 2.0. The fourier order to apply.
                This number is the number of complete cycles in that given seasonal period. 2 captures the fundamental frequency
                and its first harmonic. Higher orders will capture more complex seasonality, but may lead to overfitting.
            called (str): Optional. What to call the resulting variable.
                Two variables will be created--one for a sin transformation and the other for cos
                resulting variable names will have "sin" or "cos" at the end.
                Example, called = 'cycle5' will become 'cycle5sin', 'cycle5cos'.
                If left unspecified, 'cycle{cycle_length}' will be used as the name.

        Returns:
            Self

        >>> f.add_cycle(13) # adds a seasonal effect that cycles every 13 observations called 'cycle13'
        """
        #self._validate_future_dates_exist()
        if called is None:
            called = f"cycle{cycle_length}"
        full_sin = pd.Series(range(1, len(self.y) + len(self.future_dates) + 1)).apply(
            lambda x: np.sin(np.pi * x / (cycle_length / fourier_order))
        )
        full_cos = pd.Series(range(1, len(self.y) + len(self.future_dates) + 1)).apply(
            lambda x: np.cos(np.pi * x / (cycle_length / fourier_order))
        )
        self.current_xreg[called + "sin"] = pd.Series(full_sin.values[: len(self.y)])
        self.current_xreg[called + "cos"] = pd.Series(full_cos.values[: len(self.y)])
        self.future_xreg[called + "sin"] = list(full_sin.values[len(self.y) :])
        self.future_xreg[called + "cos"] = list(full_cos.values[len(self.y) :])

        return self

    def add_other_regressor(self, called:str, start:DatetimeLike, end:DatetimeLike) -> Self:
        """ Adds a dummy variable that is 1 during the specified time period, 0 otherwise.

        Args:
            called (str):
                What to call the resulting variable.
            start (str, datetime.datetime, or pd.Timestamp): Start date.
                Must be parsable by pandas' Timestamp function.
            end (str, datetime.datetime, or pd.Timestamp): End date.
                Must be parsable by pandas' Timestamp function.

        Returns:
            Self

        >>> f.add_other_regressor('january_2021','2021-01-01','2021-01-31')
        """
        #self._validate_future_dates_exist()
        self.current_xreg[called] = pd.Series(
            [1 if (x >= pd.Timestamp(start)) & (x <= pd.Timestamp(end)) else 0 for x in self.current_dates]
        )
        self.future_xreg[called] = [
            1 if (x >= pd.Timestamp(start)) & (x <= pd.Timestamp(end)) else 0 for x in self.future_dates
        ]

        return self

    def add_covid19_regressor(
        self,
        called:str="COVID19",
        start:DatetimeLike=datetime.datetime(2020, 3, 15),
        end:DatetimeLike=datetime.datetime(2021, 5, 13),
    ) -> Self:
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
        return self

    def add_combo_regressors(self, *args:AvailableXvar, sep:str="_") -> Self:
        """ Combines all passed variables by multiplying their values together.

        Args:
            *args (str): Names of Xvars that aleady exist in the object.
            sep (str): Default '_'.
                The separator between each term in arg to create the final variable name.

        Returns:
            Self

        >>> f.add_combo_regressors('t','monthsin') # multiplies these two together (called 't_monthsin')
        >>> f.add_combo_regressors('t','monthcos') # multiplies these two together (called 't_monthcos')
        """
        if len(args) < 2:
            raise ForecastError("Need to pass at least two regressor names to form the combination(s).")
        for i, a in enumerate(args):
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
        return self

    def add_poly_terms(self, *args:AvailableXvar, pwr:NonNegativeInt=2, sep:str="^") -> Self:
        """ raises all passed variables (no AR terms) to exponential powers (ints only).

        Args:
            *args (str): Names of Xvars that aleady exist in the object
            pwr (int): Default 2.
                The max power to add to each term in args (2 to this number will be added).
            sep (str): default '^'.
                The separator between each term in arg to create the final variable name.

        Returns:
            Self

        >>> f.add_poly_terms('t','year',pwr=3) # raises t and year to 2nd and 3rd powers (called 't^2', 't^3', 'year^2', 'year^3')
        """
        #self._validate_future_dates_exist()
        for a in args:
            for i in range(2, pwr + 1):
                self.current_xreg[f"{a}{sep}{i}"] = self.current_xreg[a] ** i
                self.future_xreg[f"{a}{sep}{i}"] = [x ** i for x in self.future_xreg[a]]

        return self

    def add_exp_terms(self, *args:AvailableXvar, pwr:float, sep:str="^", cutoff:NonNegativeInt=2, drop:bool=False) -> Self:
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
            Self

        >>> f.add_exp_terms('t',pwr=.5) # adds square root t called 't^0.5'
        """
        #self._validate_future_dates_exist()
        pwr = float(pwr)
        for a in args:
            self.current_xreg[f"{a}{sep}{round(pwr,cutoff)}"] = (
                self.current_xreg[a] ** pwr
            )
            self.future_xreg[f"{a}{sep}{round(pwr,cutoff)}"] = [
                x ** pwr for x in self.future_xreg[a]
            ]

        if drop:
            self.drop_Xvars(*args)

        return self

    def add_logged_terms(self, *args:AvailableXvar, base:float=math.e, sep:str="", drop:bool=False) -> Self:
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
            Self

        >>> f.add_logged_terms('t') # adds natural log t callend 'lnt'
        """
        #self._validate_future_dates_exist()
        for a in args:
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

        return self

    def add_pt_terms(
        self, 
        *args:AvailableXvar, 
        method:Literal['box-cox','yeo-johnson']="box-cox", 
        sep:str="_", 
        drop:bool=False
    ) -> Self:
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
            Self

        >>> f.add_pt_terms('t') # adds box cox of t called 'box-cox_t'
        """
        #self._validate_future_dates_exist()
        pt = PowerTransformer(method=method, standardize=False)
        for a in args:
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

        return self

    def drop_regressors(self, *args:AvailableXvar, raise_error:bool = True):
        """ Drops regressors.

        Args:
            *args (str): The names of regressors to drop.
            raise_error (bool): Whether to raise an error if regressors not found. Default raises. False ignores.

        Returns:
            Self

        >>> f.add_time_trend()
        >>> f.add_exp_terms('t',pwr=.5)
        >>> f.drop_regressors('t','t^0.5')
        """
        for a in args:
            if a not in self.current_xreg:
                if raise_error:
                    raise ForecastError(f'Cannot find {a} in Forecaster object.')
            self.current_xreg.pop(a)
            self.future_xreg.pop(a)

        return self

    def drop_Xvars(self, *args:AvailableXvar, raise_error:bool = True) -> Self:
        """ Drops regressors.

        Args:
            *args (str): The names of regressors to drop.
            raise_error (bool): Whether to raise an error if regressors not found. Default raises. False ignores.

        Returns:
            Self

        >>> f.add_time_trend()
        >>> f.add_exp_terms('t',pwr=.5)
        >>> f.drop_Xvars('t','t^0.5')
        """
        self.drop_regressors(*args,raise_error=raise_error)

    def drop_all_Xvars(self) -> Self:
        """ Drops all regressors.

        Returns:
            Self
        """
        self.drop_Xvars(*self.get_regressor_names())
        return Self

    def pop(self, *args:EvaluatedModel) -> Self:
        """ Deletes evaluated forecasts from the object's memory.

        Args:
            *args (str): Names of models matching what was passed to call_me when model was evaluated.

        Returns:
            Self

        >>> models = ('mlr','mlp','lightgbm')
        >>> f.tune_test_forecast(models,dynamic_testing=False,feature_importance=True)
        >>> f.pop('mlr')
        """
        for a in args:
            self.history.pop(a)

        return self


    def add_sklearn_estimator(self, imported_module:ScikitLike, called:str, mv:bool=False) -> Self:
        """ Adds a new estimator from scikit-learn not built-in to the forecaster object that can be called using set_estimator().
        Only regression models are accepted.
        
        Args:
            imported_module (scikit-learn regression model):
                The model from scikit-learn to add. Must have already been imported locally.
                Supports models from sklearn and sklearn APIs.
            called (str):
                The name of the estimator that can be called using set_estimator().
            mv (bool):
                Whether the add is for Multivariate forecasting.

        Returns:
            Self

        >>> from sklearn.ensemble import StackingRegressor
        >>> f.add_sklearn_estimator(StackingRegressor,called='stacking')
        >>> f.set_estimator('stacking')
        >>> f.manual_forecast(...)
        """
        if mv:
            interpreted_model = SKLearnUni
        else:
            pass # TODO: implement
        
        self.estimators.estimator_list.append(
            Estimator(name=called,imported_model=imported_module,interpreted_model=interpreted_model)
        )
        return self

    def _called(self,func:callable,called:str):
        return func.__name__ if called is None else called

    def auto_forecast(
        self,
        call_me:Optional[str]=None,
        test_model:bool=True,
        predict_fitted:bool=True,
        dynamic_testing:DynamicTesting=True,
    ) -> Self:
        """ Auto forecasts with the best parameters indicated from the tuning process.

        Args:
            call_me (str): Optional.
                What to call the model when storing it in the object's history dictionary.
                If not specified, the model's nickname will be assigned the estimator value ('mlp' will be 'mlp', etc.).
                Duplicated names will be overwritten with the most recently called model.
            test_model (bool): Default True.
                Whether to test the model before forecasting to a future horizon.
                If test_length is 0, this is ignored. Set this to False if you tested the model manually by calling f.test()
                and don't want to waste resources testing it again.
            dynamic_testing (bool or int): Default True.
                Whether to dynamically/recursively test the forecast (meaning AR terms will be propagated with predicted values).
                If True, evaluates dynamically over the entire out-of-sample slice of data.
                If int, window evaluates over that many steps (2 for 2-step dynamic forecasting, 12 for 12-step, etc.).
                Setting this to False or 1 means faster performance, 
                but gives a less-good indication of how well the forecast will perform more than one period out.
                The model will skip testing if the test_length attribute is set to 0.
            predict_fitted (bool): Whether to predict fitted values.

        Returns:
            Self

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
            test_model = test_model,
            predict_fitted=predict_fitted,
            **self.best_params,
        )

    def init_estimator(
        self,
        dynamic_testing:Optional[DynamicTesting]=None,
        **kwargs:Any
    ) -> Self:
        """
        Docstring for init_estimator
        
        :param self: Description
        :param dynamic_testing: Description
        :type dynamic_testing: DynamicTesting
        :param Xvars: Description
        :type Xvars: XvarValues
        :param kwargs: Description
        :type kwargs: any
        :return: Description
        :rtype: Self
        """
        call_estimator = self.estimators.lookup_item(self.estimator)
        all_kwargs = dict(
            f=self,
            model=self.estimators.lookup_item(self.estimator).imported_model,
            **kwargs,
        )
        accepted_params = inspect.signature(call_estimator.interpreted_model.__init__).parameters
        if "dynamic_testing" in accepted_params:
            all_kwargs['dynamic_testing'] = dynamic_testing
        
        self.call_estimator = call_estimator.interpreted_model(**all_kwargs)
        return Self
    
    def fit(self,**fit_params:Any) -> Self:
        """
        Docstring for fit
        
        :param self: Description
        :param fit_params: Description
        :type fit_params: Any
        :return: Description
        :rtype: Self
        """
        X = self.call_estimator.generate_current_X()
        y = self.y.values
        self.call_estimator.fit(X,y,**fit_params)
        return self

    def predict(self,**predict_params:Any):
        """
        Docstring for predict
        
        :param self: Description
        :param X: Description
        :type X: np.ndarray
        """
        X = self.call_estimator.generate_future_X()
        return self.call_estimator.predict(X,**predict_params)
    
    def predict_fitted_vals(self,**predict_params:Any):
        """
        Docstring for predict_fitted_vals
        
        :param self: Description
        :param predict_params: Description
        :type predict_params: Any
        """
        X = self.call_estimator.generate_current_X()
        return self.call_estimator.predict(X,in_sample=True,**predict_params)

    def manual_forecast(
        self, 
        call_me:Optional[str]=None, 
        test_model:bool = True, 
        dynamic_testing:DynamicTesting=True, 
        bank_history:bool = True,
        predict_fitted:bool=True,
        **kwargs:Any,
    ) -> list[float]:
        """ Manually forecasts with the hyperparameters, Xvars, and normalizer selection passed as keywords.
        See https://scalecast.readthedocs.io/en/latest/Forecaster/_forecast.html.

        Args:
            call_me (str): Optional.
                What to call the model when storing it in the object's history.
                If not specified, the model's nickname will be assigned the estimator value ('mlp' will be 'mlp', etc.).
                Duplicated names will be overwritten with the most recently called model.
            test_model (bool): Default True.
                Whether to test the model before forecasting to a future horizon.
                If test_length is 0, this is ignored. Set this to False if you tested the model manually by calling f.test()
                and don't want to waste resources testing it again.
            dynamic_testing (bool or int): Default True.
                Whether to dynamically/recursively test the forecast (meaning AR terms will be propagated with predicted values).
                If True, evaluates dynamically over the entire out-of-sample slice of data.
                If int, window evaluates over that many steps (2 for 2-step dynamic forecasting, 12 for 12-step, etc.).
                Setting this to False or 1 means faster performance, 
                but gives a less-good indication of how well the forecast will perform more than one period out.
                The model will skip testing if the test_length attribute is set to 0.
            predict_fitted (bool): Whether to predict fitted values.
            **kwargs: passed to the _forecast_{estimator}() method and can include such parameters as Xvars, 
                normalizer, cap, and floor, in addition to any given model's specific hyperparameters.
                See https://scalecast.readthedocs.io/en/latest/Forecaster/_forecast.html.

        Returns:
            List[float]: The forecasted predictions.

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

        if test_model and self.test_length:
            self.test(
                **kwargs,
                dynamic_testing=dynamic_testing,
                call_me=call_me,
            )
        elif bank_history:
            if self.call_me not in self.history.keys():
                self.history[self.call_me] = {}

        try:
            self.init_estimator(**kwargs,dynamic_testing=dynamic_testing)
            self.fit()
            preds = self.predict()
            if predict_fitted:
                fvs = self.predict_fitted_vals()
            else:
                fvs = []
        except:
            if self.call_me in self.history:
                self.history.pop(self.call_me)
            raise

        self.forecast = preds
        self.fitted_values = fvs
        if bank_history:
            self._bank_history(**kwargs)

        return preds

    def eval_cis(self,mode:bool=True,cilevel:ConfInterval=.95) -> Self:
        """ Call this function to change whether or not the Forecaster sets confidence intervals on all evaluated models.
        Beginning 0.17.0, only conformal confidence intervals are supported. Conformal intervals need a test set to be configured soundly.
        Confidence intervals cannot be evaluated when there aren't at least 1/(1-cilevel) observations in the test set.

        Args:
            mode (bool): Default True. Whether to set confidence intervals on or off for models.
            cilevel (float): Default .95. Must be greater than 0, less than 1. The confidence level
                to use to set intervals.
        """
        if mode:
            self._check_right_test_length_for_cis(cilevel)
        
        self.cis=mode
        self.set_cilevel(cilevel)
        return self

    def ingest_grid(self, grid:str|dict[str,Any]) -> Self:
        """ Ingests a grid to tune the estimator.

        Args:
            grid (dict or str):
                If dict, must be a user-created grid.
                If str, must match the name of a dict grid stored in a grids file.

        Returns:
            Self

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
        return self

    def limit_grid_size(self, n:PositiveInt|ConfInterval, min_grid_size:PositiveInt = 1, random_seed:Optional[int]=None) -> Self:
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
            Self

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
        return self

    def set_metrics(self,metrics:list[MetricStore|DefaultMetric],keep_existing:bool=True) -> Self:
        """ Set or change the evaluated metrics for all model testing and validation.

        Args:
            metrics (list[MetricStore|str]): The metrics to evaluate when validating and testing models. 
                If str, each element must exist as a name in scalecast.Metrics.Metrics and can only accept two arguments: a and f.
                Otherwise use the MetricStore class from scalecast.Classes to specify a custom metric.
                For each metric and model that is tested, the test-set and in-sample metrics will be evaluated and can be
                exported. Level test-set and in-sample metrics are also currently available, but will be removed in a future version.
            keep_existing (bool): Whether to keep evaluating all existing metrics already in the object.
        
        Returns:
            Self
        """
        add_mets = []
        for met in metrics:
            match met:
                case MetricStore():
                    add_mets.append(met)
                case str():
                    add_mets.append(METRICS.lookup_item(met))
                case _:
                    raise ValueError(f'cannot use value passed to metrics: {met}')
                
        if keep_existing:
            self.metrics = [met for met in self.metrics if met not in add_mets]
        else:
            self.metrics = add_mets

        self.determine_best_by = DetermineBestBy(self.metrics,self.validation_metric)

        return self
    
    def parse_determine_best_by(self,determine_best_by:DetermineBestBy) -> MetricStore:
        """
        Docstring for parse_determine_best_by
        
        :param self: Description
        :param determine_best_by: Description
        :type determine_best_by: DetermineBestBy
        :return: Description
        :rtype: MetricStore
        """
        label = self.determine_best_by.lookup_label(determine_best_by)
        return self.determine_best_by.lookup_metric(label)

    def set_validation_metric(self, metric:str) -> Self:
        """ Sets the metric that will be used to tune all subsequent models.

        Args:
            metric (str): One of the names in Forecaster.metrics.
                The metric to optimize the models with using the validation set.
                Although model testing will evaluate all metrics in Forecaster.metrics,
                model optimization with tuning and cross validation only uses one of these.

        Returns:
            Self

        >>> f.set_validation_metric('mae')
        """
        self.validation_metric = [m for m in self.metrics if m.name == metric][0]
        self.set_validation_length(max(self.validation_metric.min_obs_required,self.validation_length))
        return self

    def set_test_length(self, n:NonNegativeInt|ConfInterval=1) -> Self:
        """ Sets the length of the test set. As of version 0.16.0, 0-length test sets are supported.

        Args:
            n (int or float): Default 1.
                The length of the resulting test set.
                Pass 0 to skip testing models.
                Fractional splits are supported by passing a float less than 1 and greater than 0.

        Returns:
            Self

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

        return self

    def set_validation_length(self, n:PositiveInt=1) -> Self:
        """ Sets the length of the validation set. This will never matter for models that are not tuned.

        Args:
            n (int): Default 1.
                The length of the resulting validation set.

        Returns:
            Self

        >>> f.set_validation_length(6) # validation length of 6
        """
        n = int(n)
        _developer_utils.descriptive_assert(n > 0, ValueError, f"n must be greater than 0, got {n}.")
        if self.validation_metric.min_obs_required > n:
            raise ValueError(f'The chosen validation length of {n} is too small for the validation metric: {self.validation_metric.name}')
        self.validation_length = n
        return self

    def set_cilevel(self, n:ConfInterval) -> Self:
        """ Sets the level for the resulting confidence intervals (95% default).

        Args:
            n (float): Greater than 0 and less than 1.

        Returns:
            Self
        
        >>> f.set_cilevel(.80) # next forecast will get 80% confidence intervals
        """
        _developer_utils.descriptive_assert(
            n < 1 and n > 0, ValueError, "n must be a float greater than 0 and less than 1."
        )
        self.cilevel = n
        return self

    def set_estimator(self, estimator:AvailableModel) -> Self:
        """ Sets the estimator to forecast with.

        Args:
            estimator (str): One of Forecaster.estimators.

        Returns:
            Self

        >>> f.set_estimator('lasso')
        >>> f.manual_forecast(alpha = .5)
        """
        if hasattr(self, "estimator"):
            if estimator != self.estimator:
                for attr in CLEAR_ATTRS_ON_ESTIMATOR_CHANGE:
                    if hasattr(self, attr):
                        delattr(self, attr)
                self._clear_the_deck()
                self.estimator = estimator
        else:
            self.estimator = estimator

        return self

    def set_grids_file(self,name:str='Grids') -> Self:
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
        return self

    def generate_future_dates(self, n:PositiveInt) -> Self:
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

        return self

    def set_last_future_date(self, date:DatetimeLike) -> Self:
        """ Generates future dates in the same frequency as current_dates that ends on a specified date.

        Args:
            date (datetime-like):
                The date to end on. Must be parsable by pandas' Timestamp() function.

        Returns:
            Self

        >>> f.set_last_future_date('2021-06-01') # creates future dates up to this one in the expected frequency
        """
        self.future_dates = pd.Series(
            pd.date_range(
                start=self.current_dates.values[-1], end=date, freq=self.freq
            ).values[1:]
        )

        return self

    def add_lagged_terms(
        self, 
        *args:AvailableXvar, 
        lags:PositiveInt=1, 
        upto:bool=True, 
        sep:str="_", 
        drop:bool = False
    ) -> Self:
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
            Self

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

        ars = self.get_max_lag_order()
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

        return self

    def add_series(
        self,
        series:Sequence[int|float],
        called:str,
        first_date:Optional[DatetimeLike]=None,
        pad:bool=True,
    ) -> Self:
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

        self.ingest_Xvars_df(df,pad=pad)
        return self

    def ingest_Xvars_df(
        self, 
        df:pd.DataFrame, 
        date_col:str="Date", 
        drop_first:bool=False, 
        use_future_dates:bool=False, 
        pad:bool = False,
    ) -> Self:
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
            Self
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

        return self

    def export_validation_grid(self, model:EvaluatedModel) -> pd.DataFrame:
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
        df['MetricEvaluated'] = hist['ValidationMetric'].name
        if hasattr(self,'optimize_on'):
            df['Optimized On'] = self.optimize_on
        return df

    def test(
        self,
        dynamic_testing:DynamicTesting=True,
        call_me:Optional[str]=None,
        **kwargs:Any,
    ) -> Self:
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

        is_Forecaster = not self.determine_if_MVForecaster()
        call_me = self.estimator if call_me is None else call_me
        if call_me not in self.history:
            self.history[call_me] = {}
            already_existed = False
        else:
            already_existed = True

        if is_Forecaster:
            actuals:list[float] = self.y.to_list()[-self.test_length:]
        else:
            actuals:list[list[float]] = [
                self.y[k].to_list()[-self.test_length:]
                for k in self.y
            ]

        f1 = copy.deepcopy(self)
        f1.chop_from_front(
            self.test_length,
            fcst_length = self.test_length,
        )
        f1.set_test_length(0)
        f1.eval_cis(False)
        fcst = f1.manual_forecast(
            dynamic_testing = dynamic_testing,
            test_model = False, 
            call_me = call_me, 
            test_set_actuals = actuals,
            predict_fitted = False,
            **kwargs,
        )

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

        for met, value in zip(self.determine_best_by.metrics,self.determine_best_by.values):
            if value.startswith('TestSet'):
                if is_Forecaster:
                    self.history[call_me][value] = EvaluatedMetric(score=met.eval_func(actuals,fcst),store=met)
                else:
                    for i, k in enumerate(fcst.keys()):
                        self.history[call_me][value] = EvaluatedMetric(score=met.eval_func(actuals[i],fcst[k]),store=met)

        return self
    
    def determine_if_MVForecaster(self):
        """
        Docstring for determine_if_MVForecaster
        
        :param self: Description
        """
        return isinstance(self.y,dict)

    def tune(self, dynamic_tuning:DynamicTesting=False,set_aside_test_set:bool=True) -> Self:
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
            set_aside_test_set (bool): Default True. 
                Whether to separate the test set specified in f.test_length during this process.

        Returns:
            Self

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

        return self

    def cross_validate(
        self, 
        k:PositiveInt=5, 
        test_length:Optional[int] = None,
        train_length:Optional[int] = None,
        space_between_sets:Optional[int] = None,
        rolling:bool=False, 
        dynamic_tuning:DynamicTesting=False,
        set_aside_test_set:bool=True,
        verbose:bool=False,
    ) -> Self:
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
            Self

        >>> f.set_estimator('xgboost')
        >>> f.cross_validate() # tunes hyperparam values
        >>> f.auto_forecast() # forecasts with the best params
        """
        if not hasattr(self, "grid"):
            self.ingest_grid(self.estimator)

        is_Forecaster = not self.determine_if_MVForecaster()
        rolling = bool(rolling)
        k = int(k)
        _developer_utils.descriptive_assert(k >= 1, ValueError, f"k must be at least 1, got {k}.")
        f1 = copy.deepcopy(self)
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
        func = self.validation_metric.eval_func # function to evaluate metric for each grid try
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
            f2 = copy.deepcopy(f1)
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
                        test_model=False,
                        bank_history=False,
                        predict_fitted=False,
                    )
                    if is_Forecaster:
                        fcst = f2.forecast[:]
                        evaluated_metric = func(actuals,fcst)
                    else:
                        fcst = [v[:] for _, v in f2.forecast.items()]
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
            self.validation_metric_value = EvaluatedMetric(score=np.nan,store=self.validation_metric)
        else:
            if self.validation_metric.lower_is_better:
                best_hp_idx = np.nanargmin(avg_mets)
            else:
                best_hp_idx = np.nanargmax(avg_mets)
            self.best_params = grid[best_hp_idx]
            self.validation_metric_value = EvaluatedMetric(score=avg_mets[best_hp_idx],store=self.validation_metric)
        if verbose:
            print(f'Chosen paramaters: {self.best_params}.')

        return self

    def lookup_normalizer(self,normalizer:AvailableNormalizer=None) -> NormalizerLike:
        """ Returns the normalizing object (i.e. StandardScaler) with fit/transform methods.
        
        Args:
            normalizer (str): Optional. The name of the normalizer in the object's normalizer attribute.
                Default returns a function that does nothing.

        Returns:
            NormalizerLike: An object with the fit/transform methods.
        """
        return self.normalizer[normalizer]
