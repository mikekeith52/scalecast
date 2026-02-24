from __future__ import annotations
from .cfg import ESTIMATORS, IGNORE_AS_HYPERPARAMS, COLORS
from ._utils import _developer_utils, _tune_test_forecast
from ._Forecaster_parent import Forecaster_parent,ForecastError
from .types import (
    DatetimeLike, 
    PositiveInt, 
    NonNegativeInt, 
    ConfInterval, 
    DynamicTesting, 
    ModelValues,
    XvarValues,
    DetermineBestBy,
    EvaluatedModel,
    FIMethod,
    SKLearnModel,
    AvailableModel,
    ExportOptions,
)
from .classes import AR, EvaluatedMetric
from .models import SKLearnUni
from typing import Literal, Optional, Collection, Sequence, Any, Self, TYPE_CHECKING
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import seaborn as sns
import datetime
import warnings
import os
from pathlib import Path
from itertools import cycle
import copy
from collections.abc import Sized
from scipy import stats
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose, STL, DecomposeResult
if TYPE_CHECKING:
    from .Forecaster import Forecaster
    import shap

class Forecaster(Forecaster_parent):
    """ Docstring

    Args:
        y (collection): An array of all observed values. Must match the order of elements passed to current_dates.
        current_dates (collection): An array of all observed dates.
            Must be same length as y and in the same sequence.
            Can pass any numerical index if dates are unknown; in this case, 
            It will act as if dates are in nanosecond frequency.
        future_dates (int): Optional. The future dates to add to the model upon initialization.
            If not added when object is initialized, can be added later.
        test_length (int or float): Default 0. The test length that all models will use to test all models out of sample.
            If float, must be between 0 and 1 and will be treated as a fractional split.
            By default, models will not be tested.
        cis (bool): Default False. Whether to evaluate naive conformal confidence intervals for every model evaluated.
            If setting to True, ensure you also set a test_length of at least 20 observations for 95% confidence intervals.
            See eval_cis() and set_cilevel() methods and docstrings for more information.
        carry_fit_models (bool): Default True.
            Whether to store the regression model for each fitted model in history.
            Setting this to False can save memory.
    """
    def __init__(
        self, 
        y:Sequence[float|int], 
        current_dates:Sequence[DatetimeLike], 
        future_dates:Optional[int]=None,
        test_length:NonNegativeInt = 0,
        cis:bool = False,
        carry_fit_models:bool = True,
    ):
        super().__init__(
            y = y,
            test_length = test_length,
            cis = cis,
        )
        self.estimators = ESTIMATORS
        self.current_dates = current_dates
        self.future_dates = pd.Series([], dtype="datetime64[ns]")
        self.init_dates = list(current_dates)
        self.grids_file = 'Grids'
        self.carry_fit_models = carry_fit_models
        self._typ_set()  # ensures that the passed values are the right types
        if future_dates is not None:
            self.generate_future_dates(future_dates)

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return """Forecaster(
    DateStartActuals={}
    DateEndActuals={}
    Freq={}
    N_actuals={}
    ForecastLength={}
    Xvars={}
    TestLength={}
    ValidationMetric={}
    ForecastsEvaluated={}
    CILevel={}
    CurrentEstimator={}
    GridsFile={}
)""".format(
        self.current_dates.values[0].astype(str),
        self.current_dates.values[-1].astype(str),
        self.freq,
        self.n_actuals(),
        len(self.future_dates),
        list(self.current_xreg.keys()),
        self.test_length,
        self.validation_metric.name,
        list(self.history.keys()),
        self.cilevel if self.cis is True else None,
        None if not hasattr(self, "estimator") else self.estimator,
        self.grids_file,
    )

    def __getstate__(self):
        self._logging()
        state = self.__dict__.copy()
        return state

    def _validate_future_dates_exist(self):
        """ makes sure future periods have been specified before adding regressors
        """
        _developer_utils.descriptive_assert(
            len(self.future_dates) > 0,
            ForecastError,
            "Before adding regressors, please make sure you have generated future dates by calling generate_future_dates(),"
            " set_last_future_date(), or ingest_Xvars_df(use_future_dates=True).",
        )
    
    def _set_cis(self,*attrs,m,ci_range,preds):
        for i, attr in enumerate(attrs):
            self.history[m][attr] = [p + (ci_range if i%2 == 0 else (ci_range*-1)) for p in preds]

    def _bank_history(self, **kwargs:Any):
        """ places all relevant information from the last evaluated forecast into the history dictionary attribute
            **kwargs: passed from each model, depending on how that model uses Xvars, normalizer, and other args
        """
        # since test only, what gets saved to history is relevant to the train set only, the genesis of the next line
        call_me = self.call_me
        self.history[call_me]['Estimator'] = self.estimator
        self.history[call_me]['Xvars'] = self.call_estimator.Xvars if hasattr(self.call_estimator,'Xvars') else None
        self.history[call_me]['HyperParams'] = {k: v for k, v in kwargs.items() if k not in IGNORE_AS_HYPERPARAMS}
        self.history[call_me]['Observations'] = len(self.y)
        self.history[call_me]['Forecast'] = self.forecast[:]
        self.history[call_me]['FittedVals'] = self.fitted_values[:]
        self.history[call_me]['DynamicallyTested'] = self.dynamic_testing
        self.history[call_me]['CILevel'] = self.cilevel if self.cis else np.nan
        for attr in ('TestSetPredictions','TestSetActuals'):
            if attr not in self.history[call_me]:
                self.history[call_me][attr] = []

        if hasattr(self,'best_params'):
            if np.all([k in self.best_params for k in self.history[call_me]['HyperParams']]):
                self.history[call_me]['ValidationMetric'] = self.validation_metric
                self.history[call_me]['ValidationMetricValue'] = self.validation_metric_value
                self.history[call_me]['grid'] = self.grid
                self.history[call_me]['grid_evaluated'] = self.grid_evaluated

        for met in self.metrics:
            for met, value in zip(self.determine_best_by.metrics,self.determine_best_by.values):
                if value.startswith('InSample'):
                    self.history[call_me][value] = EvaluatedMetric(
                        score = _developer_utils._return_na_if_len_zero(
                            self.y.iloc[-len(self.fitted_values) :], self.fitted_values, met.eval_func
                        ),
                        store = met,
                    )

        for attr in ("call_estimator", "models", "weights", "ymin", "ymax"):
            if hasattr(self, attr):
                if attr == 'call_estimator' and not self.carry_fit_models:
                    continue
                self.history[call_me][attr] = getattr(self, attr)
        
        if self.cis is True and len(self.history[call_me]['TestSetPredictions']) > 0: #and 'TestSetPredictions' in self.history[call_me].keys():
            self._check_right_test_length_for_cis(self.cilevel)
            fcst = self.history[call_me]['Forecast']
            test_preds = self.history[call_me]['TestSetPredictions']
            test_actuals = self.history[call_me]['TestSetActuals']
            test_resids = np.array([p - a for p, a in zip(test_preds,test_actuals)])
            #test_resids = correct_residuals(test_resids)
            ci_range = np.percentile(np.abs(test_resids), 100 * self.cilevel)
            self._set_cis(
                "TestSetUpperCI",
                "TestSetLowerCI",
                m = call_me,
                ci_range = ci_range,
                preds = test_preds,
            )
            self._set_cis(
                "UpperCI",
                "LowerCI",
                m = call_me,
                ci_range = ci_range,
                preds = fcst,            )

    def _bank_fi_to_history(self):
        """ for every model where feature importance can be extracted, saves that info to a pandas dataframe wehre index is the regressor name
        """
        call_me = self.call_me
        self.history[call_me]['feature_importance_explainer'] = self.explainer
        self.history[call_me]["feature_importance"] = self.feature_importance

    def synthesize_models(
        self,
        models:ModelValues='all',
        determine_best_by:Optional[DetermineBestBy]=None,
        call_me:str='synthesized',
        cilevel:ConfInterval=.95,
        verbose:bool = False,
    ) -> Self:
        """ Creates a model that is an average of other models with confidence intervals determined by forming normal distributions around each point prediction.

        Args:
            models (list-like or str): Default 'all'. Which models to consider.
                Can start with top ('top_5').
            determine_best_by (str): Optional. Combine with `call_me = 'top_{n}'`. One of Forecaster.determine_best_by.
            call_me (str): The name of the resulting model. Default 'synthesized'.
            cilevel (float): The confidence level for the resulting confidence interval. Default .95.
            verbose (bool): Whether to print successful completion of the function. Default False.
        """
        models = self._parse_models(models,determine_best_by)
        if len(models) < 3:
            raise ValueError('Please pass at least three models to the synthesize() method.')

        z_score = stats.norm.ppf((1 + cilevel) / 2)

        f = copy.deepcopy(self)
        f.eval_cis(mode=False)
        f.set_estimator('combo')
        f.manual_forecast(models=models,call_me=call_me)
        st_errors = np.array([self.history[m]['Forecast'] for m in models]).std(axis=0) / np.sqrt(len(models))
        
        f.history[call_me]['UpperCI'] = np.array(f.history[call_me]['Forecast']) + z_score * st_errors
        f.history[call_me]['LowerCI'] = np.array(f.history[call_me]['Forecast']) - z_score * st_errors
        f.history[call_me]['CILevel'] = cilevel

        test_st_errors = np.array([self.history[m]['TestSetPredictions'] for m in models]).std(axis=0) / np.sqrt(len(models))
        if len(test_st_errors):
            f.history[call_me]['TestSetUpperCI'] = np.array(f.history[call_me]['TestSetPredictions']) + z_score * test_st_errors
            f.history[call_me]['TestSetLowerCI'] = np.array(f.history[call_me]['TestSetPredictions']) - z_score * test_st_errors

        self.history[call_me] = f.history[call_me]
        if verbose:
            print('⚡Models Synthesized⚡')
        return self

    def _parse_models(self, models:ModelValues, determine_best_by:DetermineBestBy) -> list[str]:
        """ Takes a collection of models and orders them best-to-worst based on a given metric and returns the ordered list (of str type).

        Args:
            models (ModelValues): See scalecast.types.ModelValues
            determine_best_by (str): one of Forecaster.determine_best_by.
                if a model does not have the metric specified here (i.e. one of the passed models wasn't tuned and this is 'ValidationMetricValue'), it will be ignored silently, so be careful

        Returns:
            (list) The ordered evaluated models.
        """
        if determine_best_by is None:
            match models:
                case "all":
                    models = list(self.history.keys())
                case str():
                    if models.startswith("top_"):
                        raise ValueError('Cannot use models starts with "top_" unless the determine_best_by or order_by argument is specified.')
                    else:
                        models = [models]
                case _:
                    models = list(models)
        else:
            all_models = [m for m, d in self.history.items() if determine_best_by in d]
            all_models = self.order_fcsts(all_models, determine_best_by)
            match models:
                case 'all':
                    models = all_models[:]
                case str():
                    if models.startswith("top_"):
                        models = all_models[: int(models.split("_")[1])]
                    else:
                        models = [models]
                case _:
                    models = [m for m in all_models if m in models]
        
        if not models:
            raise ValueError(f"models argument with determine_best_by={determine_best_by} returns no evaluated forecasts.")
        
        return models

    def _diffy(self, n):
        """ parses the argument fed to a diffy parameter

        Args:
            n (bool or int): one of {True,False,0,1}.
                If False or 0, does not difference.
                If True or 1, differences 1 time.

        Returns:
            (Series): The differenced array.
        """
        n = int(n)
        _developer_utils.descriptive_assert(
            (n <= 1) & (n >= 0),
            ValueError,
            "diffy cannot be less than 0 or greater than 1.",
        )
        y = self.y.copy()
        for i in range(n):
            y = y.diff().dropna()
        return y

    def infer_freq(self) -> Self:
        """ Uses the pandas library to infer the frequency of the loaded dates.
        
        Returns:
            Self
        """
        if not hasattr(self, "freq"):
            self.freq = pd.infer_freq(self.current_dates)
            self.current_dates.freq = self.freq
            self.future_dates.freq = self.freq

        return self

    def _typ_set(self):
        """ Converts all objects in y, current_dates, future_dates, current_xreg, and future_xreg to appropriate types if possible.
        Automatically gets called when object is initiated to minimize potential errors.

        >>> f._typ_set() # sets all arrays to the correct format
        """
        self.y = pd.Series(self.y).dropna().astype(np.float64)
        self.current_dates = pd.to_datetime(pd.Series(list(self.current_dates)[-len(self.y) :]))
        _developer_utils.descriptive_assert(
            len(self.y) == len(self.current_dates),
            ValueError,
            f"y and current_dates must be same size -- y is {len(self.y)} and current_dates is {len(self.current_dates)}.",
        )
        self.future_dates = pd.to_datetime(pd.Series(self.future_dates))
        for k, v in self.current_xreg.items():
            self.current_xreg[k] = pd.Series(list(v)[-len(self.y) :]).astype(np.float64)
            _developer_utils.descriptive_assert(
                len(self.current_xreg[k]) == len(self.y),
                ForecastError,
                "Length of array representing the input '{}' does match the length of the stored observed values: ({} vs. {})."
                " If you do not know how this happened, consider raising an issue: https://github.com/mikekeith52/scalecast/issues/new.".format(
                    k,
                    len(self.current_xreg[k]),
                    len(self.y),
                ),
            )
            self.future_xreg[k] = [float(x) for x in self.future_xreg[k]]

        self.infer_freq()

    def add_signals(
        self, 
        model_nicknames:Collection[EvaluatedModel], 
        fill_strategy:Optional[Literal['actuals','bfill']] = 'actuals', 
        train_only:bool = False
    ):
        """ Adds the predictions from already-evaluated models as covariates that can be used for future evaluated models.
        The names of the added variables will all begin with "signal_" and end with the given model nickname.

        Args:
            model_nicknames (list): The names of already-evaluated models with information stored in the history attribute.
            fill_strategy (str or None): The strategy to fill NA values that are present at the beginning of a given model's fitted values.
                Available options are: 'actuals' (default) which will replace nulls with actuals;
                'bfill' which will backfill null values;
                or None which will leave null values alone, which can cause errors in future evaluated models.
            train_only (bool): Default False. Whether to add fitted values from the training set only.
                The test-set predictions will be out-of-sample if this is True. The future unknown values are always out-of-sample.
                Even when this is True, the future unknown values are taken from a model trained on the full set of
                known observations.

        >>> f.set_estimator('lstm')
        >>> f.manual_forecast(call_me='lstm')
        >>> f.add_signals(model_nicknames = ['lstm']) # adds a regressor called 'signal_lstm'
        """
        for m in model_nicknames:
            fcst = self.history[m]['Forecast'][:]
            fvs = self.history[m]['FittedVals'][:]
            num_fvs = len(fvs)
            pad = (
                [
                    np.nan if fill_strategy is None
                    else fvs[0]
                ] * (len(self.y) - num_fvs) 
                if fill_strategy != 'actuals' 
                else self.y.to_list()[:-num_fvs]
            )
            self.current_xreg[f'signal_{m}'] = pd.Series(pad + fvs)
            if train_only:
                tsp = self.history[m]['TestSetPredictions'][:]
                self.current_xreg[f'signal_{m}'].iloc[-len(tsp):] = tsp
            self.future_xreg[f'signal_{m}'] = fcst

    def add_ar_terms(self, n:PositiveInt|list[PositiveInt]) -> Self:
        """ Adds auto-regressive terms.

        Args:
            n (int or list[int]): If int, the number of lags to add to the object (1 to this number will be added by default).
                If list, will add the lags specified in the collection (`[2,4]` will add lags 2 and 4).
                To add only lag 10, pass `[10]`. To add 10 lags, pass `10`.

        Returns:
            Self

        >>> f.add_ar_terms(4) # adds four lags of y called 'AR1' - 'AR4' to predict with
        >>> f.add_ar_terms([4]) # adds the fourth lag called 'AR4' to predict with
        """
        #self._validate_future_dates_exist()
        match n:
            case 0:
                return
            case Sized():
                iterable = [int(i) for i in n]
            case _:
                iterable = range(1,int(n)+1)

        fcst_length = len(self.future_dates)
        for i in iterable:
            self.current_xreg[AR(i)] = pd.Series(self.y).shift(i)
            self.future_xreg[AR(i)] = (self.y.to_list()[-i:] + ([np.nan] * (fcst_length - i)))[:fcst_length]
        return self

    def add_AR_terms(self, N:tuple[int,int]) -> Self:
        """ Adds seasonal auto-regressive terms.
            
        Args:
            N (tuple): First element is the number of lags to add and the second element is the space between lags.

        Returns:
            Self

        >>> f.add_AR_terms((2,12)) # adds 12th and 24th lags called 'AR12', 'AR24'
        """
        return self.add_ar_terms(list(range(N[1], N[1] * N[0] + 1, N[1])))

    def reduce_Xvars(
        self,
        method:FIMethod="PermutationExplainer",
        estimator:SKLearnModel="lasso",
        keep_at_least:PositiveInt=1,
        keep_this_many:PositiveInt|Literal['auto','sqrt']="auto",
        grid_search:bool=True,
        use_loaded_grid:bool=False,
        dynamic_tuning:DynamicTesting=False,
        monitor:DetermineBestBy="ValidationMetricValue",
        overwrite:bool=True,
        cross_validate:bool=False,
        masker:Optional["shap.maskers.Masker"] = None,
        cvkwargs:dict[str,Any]={},
        **kwargs:Any,
    ) -> Self:
        """ Requires the optional shap library. Reduces the regressor variables stored in the object. Any feature importance type available with
        `f.save_feature_importance()` can be used to rank features in this process.
        Features are reduced one-at-a-time, according to which one ranked the lowest.
        After each variable reduction, the model is re-run and feature importance re-evaluated. 
        By default, the validation-set error is used to avoid leakage 
        and the variable set that most reduced the error is selected. The pfi_error_values attr is one greater in length than pfi_dropped_vars attr because 
        The first error is the initial error before any variables were dropped. The following attributes:
        `pfi_dropped_vars` and `pfi_error_values`, which are lists representing the error change with the 
        corresponding dropped variable, are created and stored in the `Forecaster` object.
        See the example:
        https://scalecast-examples.readthedocs.io/en/latest/misc/feature-selection/feature_selection.html.

        Args:
            method (Literal): See scalecast.types.FIMethod.
                The method for scoring features.
            estimator (str): One of Forecaster.sklearn_estimators. Default 'lasso'.
                The estimator to use to determine the best set of variables.
            keep_at_least (str or int): Default 1.
                The fewest number of Xvars to keep..
                'sqrt' keeps at least the sqare root of the number of Xvars rounded down.
                This exists so that the keep_this_many keyword can use 'auto' as an argument.
            keep_this_many (str or int): Default 'auto'.
                The number of Xvars to keep if method == 'pfi' or 'shap'.
                "auto" keeps the number of xvars that returned the best error using the 
                metric passed to `monitor`, but it is the most computationally expensive.
                "sqrt" keeps the square root of the total number of observations rounded down.
            gird_search (bool): Default True.
                Whether to run a grid search for optimal hyperparams on the validation set.
                If use_loaded_grid is False, uses a grids file currently available in the working directory 
                or creates a new grids file called Grids.py with default values if none available to determine the grid to use.
                The grid search is only run once and then those hyperparameters are used for
                all subsequent pfi runs when method == 'pfi'.
                In any utilized grid, do not include 'Xvars' as a key.
                If you want to access the chosen hyperparams after the fact, they are stored in the reduction_hyperparams
                attribute.
            use_loaded_grid (bool): Default False.
                Whether to use the currently loaded grid in the object instead of using a grid from a file.
                In any utilized grid, do not include 'Xvars' as a key.
            dynamic_tuning (bool or int): Default False.
                Whether to dynamically tune the model or, if int, how many forecast steps to dynamically tune it.
            monitor (str): One of Forecaster.determine_best_by. Default 'ValidationSetMetric'.
                The metric to be monitored when making reduction decisions. 
            overwrite (bool): Default True.
                If False, the list of selected Xvars are stored in an attribute called reduced_Xvars.
                If True, this list of regressors overwrites the current Xvars in the object.
            cross_validate (bool): Default False.
                Whether to tune the model with cross validation. 
                If False, uses the validation slice of data to tune.
                If not monitoring ValidationMetricValue, you will want to leave this False.
            masker (shap.maskers): Optional.
                Pass your own masker to this function if desired. Default will use shap.maskers.Independent with default arguments.
            cvkwargs (dict): Default {}. Passed to the `cross_validate()` method.
            **kwargs: Passed to the `manual_forecast()` method and can include arguments related to 
                a given model's hyperparameters or dynamic_testing.
                Do not pass hyperparameters if grid_search is True.
                Do not pass Xvars.

        Returns:
            Self

        >>> f.add_ar_terms(24)
        >>> f.add_seasonal_regressors('month',raw=False,sincos=True,dummy=True)
        >>> f.add_seasonal_regressors('year')
        >>> f.add_time_trend()
        >>> f.set_validation_length(12)
        >>> f.reduce_Xvars(overwrite=False) # reduce with lasso (but don't overwrite Xvars)
        >>> print(f.reduced_Xvars) # view results
        >>> f.reduce_Xvars(
        >>>     method='TreeExplainer',
        >>>     estimator='gbt',
        >>>     keep_at_least=10,
        >>>     keep_this_many='auto',
        >>>     dynamic_testing=False,
        >>>     dynamic_tuning=True,
        >>>     cross_validate=True,
        >>>     cvkwargs={'rolling':True},
        >>> ) # reduce with gradient boosted tree estimator and overwrite with result
        >>> print(f.reduced_Xvars) # view results
        """
        if self.estimators.lookup_item(estimator).interpreted_model != SKLearnUni:
            raise ValueError('This function only works with base models that use a scikit-learn API.')

        f = copy.deepcopy(self)
        f.set_estimator(estimator)
        if grid_search:
            if not use_loaded_grid and f.grids_file + ".py" not in os.listdir('.'):
                from . import GridGenerator
                GridGenerator.get_example_grids(out_name = f.grids_file + ".py")
            f.ingest_grid(f.estimator)
        else:
            f.ingest_grid({k: [v] for k, v in kwargs.items()})

        if not cross_validate:
            f.tune(dynamic_tuning=dynamic_tuning, set_aside_test_set = False)
        else:
            f.cross_validate(dynamic_tuning=dynamic_tuning, set_aside_test_set = False, **cvkwargs)
        f.ingest_grid({k: [v] for k, v in f.best_params.items()})
        f.auto_forecast(test_again=False)

        self.reduction_hyperparams = f.best_params.copy()

        f.save_feature_importance(
            method='shap',
            try_order=[method],
            on_error="raise",
            masker=masker,
        )

        fi_df = f.export_feature_importance(estimator)

        features = fi_df.index.to_list()
        init_metric = f.history[estimator][monitor]

        dropped = []
        metrics = [init_metric]

        sqrt = int(len(f.y) ** 0.5)

        match keep_this_many:
            case 'auto':
                keep_this_many_new = 1
            case 'sqrt':
                keep_this_many_new = sqrt
            case _:
                keep_this_many_new = keep_this_many

        match keep_at_least:
            case 'sqrt':
                keep_at_least_new = 'sqrt'
            case _:
                keep_at_least_new = keep_at_least

        stop_at = max(keep_this_many_new, keep_at_least_new)
        drop_this_many = len(f.current_xreg) - stop_at

        for _ in range(drop_this_many):
            dropped.append(features[-1])
            features = features[:-1]
            f.grid[0]["Xvars"] = features
            if not cross_validate:
                f.tune(dynamic_tuning=dynamic_tuning, set_aside_test_set = False)
            else:
                f.cross_validate(dynamic_tuning=dynamic_tuning, set_aside_test_set = False, **cvkwargs)
            f.auto_forecast(test_again=False)
            new_metric = f.history[estimator][monitor]
            metrics.append(new_metric)

            f.save_feature_importance(try_order=[method], on_error="raise")
            fi_df = f.export_feature_importance(estimator)
            features = fi_df.index.to_list()

        match keep_this_many:
            case 'auto':
                optimal_drop = metrics.index(max(metrics))
            case _:
                optimal_drop = drop_this_many

        self.reduced_Xvars = [x for x in self.current_xreg.keys() if x not in dropped[:optimal_drop]]
        self.pfi_dropped_vars = dropped
        self.pfi_error_values = metrics

        if overwrite:
            self.current_xreg = {x: v for x, v in self.current_xreg.items() if x in self.reduced_Xvars}
            self.future_xreg = {x: v for x, v in self.future_xreg.items() if x in self.reduced_Xvars}
        return self

    def _Xvar_select_forecast(
        self,
        f:'Forecaster',
        estimator:AvailableModel,
        monitor:DetermineBestBy,
        cross_validate:bool,
        dynamic_tuning:DynamicTesting,
        cvkwargs:dict[str,Any],
        kwargs:Any,
        Xvars:XvarValues = 'all',
    ) -> dict[str,EvaluatedMetric]:
        f.set_estimator(estimator)
        if monitor == 'ValidationMetricValue':
            grid = {k:[v] for k, v in kwargs.items()}
            grid['Xvars'] = [Xvars]
            f.ingest_grid(grid)
            if not cross_validate:
                f.tune(dynamic_tuning=dynamic_tuning)
            else:
                f.cross_validate(**cvkwargs,dynamic_tuning=dynamic_tuning)
            return f.validation_metric_value
        else:
            f.test(**kwargs,Xvars=Xvars)
            return f.history[estimator][monitor]

    def auto_Xvar_select(
        self,
        estimator:SKLearnModel = 'mlr',
        try_trend:bool = True,
        trend_estimator:SKLearnModel = 'mlr',
        trend_estimator_kwargs:dict[str,Any] = {},
        decomp_trend:bool = True,
        decomp_method:Literal['additive','multiplicative'] = 'additive',
        try_ln_trend:bool = True,
        max_trend_poly_order:PositiveInt = 2,
        try_seasonalities:bool = True,
        seasonality_repr:list[str]|dict[list[str]] = ['sincos'],
        exclude_seasonalities:list[str] = [],
        irr_cycles:Optional[list[PositiveInt]] = None, # list of cycles
        max_ar:Literal['auto']|NonNegativeInt = 'auto', # set to 0 to not test
        test_already_added:bool = True,
        must_keep:list[str] = [],
        monitor:DetermineBestBy = 'ValidationMetricValue',
        cross_validate:bool = False,
        dynamic_tuning:bool =False,
        cvkwargs:dict[str,Any]={},
        **kwargs:Any,
    ):
        """ Attempts to find the ideal trend, seasonality, and look-back representations for the stored series by 
        systematically adding regressors to the object and monintoring a passed metric value.
        Searches for trend first, then seasonalities, then optimal lag order, then the best 
        combination of all of the above, along with irregular cycles (if specified) and any regressors already added to the object.
        The function offers flexibility around setting Xvars it must add to the object by letting the user add these regressors before calling the function, 
        telling the function not to re-search for them, and telling the function not to drop them when considering the optimal combination of regressors.
        The final optimal combination of regressors is determined by grouping all extracted regressors into trends, seasonalities, irregular cycles, 
        ar terms, and regressors already added, and tying all combinations of all these groups.
        See the example: https://scalecast-examples.readthedocs.io/en/latest/misc/auto_Xvar/auto_Xvar.html.
        
        Args:
            estimator (str): Default 'mlr'.
                The estimator to use to determine the best seasonal and lag regressors.
            try_trend (bool): Default True.
                Whether to search for trend representations of the series.
            trend_estimator (str): One of Forecaster.sklearn_estimators. Default 'mlr'.
                Ignored if try_trend is False.
                The estimator to use to determine the best trend representation.
            trend_estimator_kwargs (dict): Default {}. The model parameters to pass to the trend_estimator model.
            decomp_trend (bool): Default True. Whether to decompose the series to estimate the trend.
                Ignored if try_trend is False.
                The idea is there can be many seasonalities represented by scalecast, but only one trend,
                so using a decomposition method for trend could lead to finding a better trend representation.
            decomp_method (str): One of 'additive','multiplicative'. Default 'additive'.
                The decomp method used to represent the trend.
                Ignored if try_trend is False. Ignored if decomp_trend is False.
            try_ln_trend (bool): Default True.
                Ignored if try_trend is False.
                Whether to search logged trend representations using a natural log.
            max_trend_poly_order (int): Default 2.
                The highest order trend representation that will be searched.
            try_seasonalities (bool): Default True.
                Whether to search for seasonal representations.
                This function uses a hierachical approach from secondly --> quarterly representations.
                Secondly will search all seasonal representations up to quarterly to find the best hierarchy of seasonalities.
                Anything lower than second and higher than quarter will not receive a seasonality with this method.
                Day seasonality and lower will try, 'day' (of month), 'dayofweek', and 'dayofyear' seasonalities.
                Everything else will try cycles that reset yearly, so to search for intermitent seasonal fluctuations, 
                use the irr_cycles argument.
            seasonality_repr (list or dict[str,list]): Default ['sincos'].
                How to represent the extracted seasonalties. the default will use fourier representations only.
                Ignored if try_seasonalities is False.
                Other elements to add to the list: 'dummy','raw','drop_first'. Can add multiple or one of these.
                If dict, the key needs to be the seasonal representation ('quarter' for quarterly, 'month' for monthly)
                and the value a list. If a seasonal representation is not found in this dictionary, it will default to
                ['sincos'], i.e. a fourier representation. 'drop_first' ignored when 'dummy' is not present.
            exclude_seasonalities (list): Default []. 
                Ignored if try_seasonalities is False.
                Add in this list any seasonal representations to skip searching.
                If you have day frequency and you only want to search dayofweek, you should specify this as:
                ['dayofweek','week','month','quarter'].
            irr_cycles (list[int]): Optional. 
                Add any irregular cycle lengths to a list as integers to search for using this method.
            max_ar ('auto' or int): The highest lag order to search for.
                If 'auto', will use the greater of the forecast length or the test-set length as the lag order.
                If a larger number than available observations is placed here, the AR search will stop early.
                Set to 0 to skip searching for lag terms.
            test_already_added (bool): Default True.
                If there are already regressors added to the series, you can either always keep them in the object
                by setting this to False, or by default, it is possible they will be dropped when looking for the
                optimal combination of regressors in the object.
            must_keep (list-like): Default []. The names of any regressors that must be kept in the object.
                All regressors here must already be added to the Forecaster object before calling the function.
                This is ignored if test_already_added is False since it becomes redundant.
            monitor (str): One of Forecaster.determine_best_by. Default 'ValidationMetricValue'.
                The metric to be monitored when making reduction decisions. 
            cross_validate (bool): Default False.
                Whether to tune the model with cross validation. 
                If False, uses the validation slice of data to tune.
                If not monitoring ValidationMetricValue, you will want to leave this False.
            dynamic_tuning (bool or int): Default False.
                Whether to dynamically tune the model or, if int, how many forecast steps to dynamically tune it.
            cvkwargs (dict): Default {}. Passed to the cross_validate() method.
            **kwargs: Passed to manual_forecast() method and can include arguments related to 
                a given model's hyperparameters or dynamic_testing.
                Do not pass Xvars.

        Returns:
            (dict[tuple[float]]): A dictionary where each key is a tuple of variable combinations 
            and the value is the derived metric (based on value passed to monitor argument).

        >>> f.add_covid19_regressor()
        >>> f.auto_Xvar_select(cross_validate=True)
        """
        if self.estimators.lookup_item(estimator).interpreted_model != SKLearnUni:
            raise ValueError('This function only works with base models that use a scikit-learn API.')
        
        def get_Xvar_combos(
            f,
            best_trend,
            best_seasonality,
            best_ar_order,
            regressors_already_added,
            seas_to_try,
        ):
            trend_regressors = []
            if best_trend is not None:
                f.add_time_trend()
                if best_trend.startswith('ln'):
                    f.add_logged_terms('t',drop=True)
                    trend_regressors.append('lnt')
                else:
                    trend_regressors.append('t')
                if '^' in best_trend:
                    pwr = int(best_trend.split('^')[-1])
                    if best_trend.startswith('ln'):
                        f.add_poly_terms('lnt',pwr=pwr)
                        trend_regressors += ['lnt^' + str(pwr) for i in range(2,pwr+1)]
                    else:
                        f.add_poly_terms('t',pwr=pwr)
                        trend_regressors += ['t^' + str(pwr) for i in range(2,pwr+1)]

            seas_regressors = []
            if best_seasonality is not None:
                seas_to_try = seas_to_try[:(seas_to_try.index(best_seasonality) + 1)]
                if isinstance(seasonality_repr,list):
                    f.add_seasonal_regressors(
                        *seas_to_try,
                        raw='raw' in seasonality_repr,
                        sincos='sincos' in seasonality_repr,
                        dummy='dummy' in seasonality_repr,
                        drop_first='drop_first' in seasonality_repr,
                    )
                elif isinstance(seasonality_repr,dict):
                    for s in seas_to_try:
                        if s in seasonality_repr:
                            f.add_seasonal_regressors(
                                s,
                                raw='raw' in seasonality_repr[s],
                                sincos='sincos' in seasonality_repr[s],
                                dummy='dummy' in seasonality_repr[s],
                                drop_first='drop_first' in seasonality_repr[s],
                            )
                        else:
                            f.add_seasonal_regressors(
                                s,
                                raw=False,
                                sincos=True,
                            )
                for s in seas_to_try:
                    seas_regressors += [
                        x for x in f.get_regressor_names() 
                        if (x == s + 'sin') 
                        or (x == s + 'cos') 
                        or x.startswith(s + '_') 
                        or x == s 
                    ]
            if best_ar_order is not None:
                f.add_ar_terms(best_ar_order)
                ar_regressors = [x for x in f.get_regressor_names() if isinstance(x,str)]
            else:
                ar_regressors = []
            if irr_cycles is not None:
                for i in irr_cycles:
                    f.add_cycle(i)
                irr_regressors = (
                    ['cycle' + str(i) + 'sin' for i in irr_cycles] + 
                    ['cycle' + str(i) + 'cos' for i in irr_cycles]
                )
            else:
                irr_regressors = []

            Xvars = [
                regressors_already_added, # just already added
                trend_regressors + seas_regressors + regressors_already_added,
                trend_regressors + ar_regressors + regressors_already_added,
                trend_regressors + irr_regressors + regressors_already_added,
                trend_regressors + seas_regressors + ar_regressors + regressors_already_added,
                trend_regressors + seas_regressors + ar_regressors + irr_regressors + regressors_already_added,
                seas_regressors + ar_regressors + regressors_already_added,
                seas_regressors + ar_regressors + irr_regressors + regressors_already_added,
                ar_regressors + irr_regressors + regressors_already_added,
            ]
            if test_already_added: # if this is False, user has to take a combo that includes those already added
                Xvars += [
                    trend_regressors, # just trend
                    seas_regressors, # just seasonality
                    ar_regressors, # just ar
                    irr_regressors, # just irr
                    trend_regressors + seas_regressors,
                    trend_regressors + ar_regressors,
                    trend_regressors + irr_regressors,
                    trend_regressors + seas_regressors + ar_regressors,
                    trend_regressors + seas_regressors + ar_regressors + irr_regressors,
                    seas_regressors + ar_regressors,
                    seas_regressors + ar_regressors + irr_regressors,
                    ar_regressors + irr_regressors,
                    [], # potentially just must_keep
                ]
                Xvars = [xvar_list + must_keep for xvar_list in Xvars]
            
            # https://stackoverflow.com/questions/2213923/removing-duplicates-from-a-list-of-lists
            Xvars_deduped = []
            for xvar_set in Xvars:
                if xvar_set and xvar_set not in Xvars_deduped:
                    #https://stackoverflow.com/questions/60062818/difference-between-removing-duplicates-from-a-list-using-dict-and-set
                    Xvars_deduped.append(list(dict.fromkeys(xvar_set))) # dedupe and preserve order
            return Xvars_deduped
        
        match estimator:
            case None:
                estimator = self.estimator

        self.set_estimator(estimator)

        trend_metrics = {}
        seasonality_metrics = {}
        ar_metrics = {}
        final_metrics = {}
        seas_to_try = []

        regressors_already_added = self.get_regressor_names()
        
        if isinstance(must_keep,str):
            must_keep = [must_keep]
        must_keep = [x for x in must_keep if x in regressors_already_added]
        
        f = copy.deepcopy(self)
        f.drop_all_Xvars()

        if try_trend:
            if decomp_trend:
                try:
                    decomp = f.seasonal_decompose(
                        model=decomp_method,
                        extrapolate_trend='freq',
                    )
                    ft = Forecaster(
                        y = decomp.trend,
                        current_dates = decomp.trend.index,
                    )
                except Exception as e:
                    warnings.warn(
                        f'Trend decomposition did not work and raised this error: {e} '
                        'Switching to the non-decomp method.',
                        category=Warning,
                    )
                    decomp_trend = False
            if not decomp_trend:
                ft = copy.deepcopy(f)

            ft.add_time_trend()
            ft.set_test_length(f.test_length)
            ft.set_validation_length(f.validation_length)
            f1 = copy.deepcopy(ft)
            trend_metrics['t'] = self._Xvar_select_forecast(
                f=f1,
                estimator=trend_estimator,
                monitor=monitor,
                cross_validate=cross_validate,
                dynamic_tuning=dynamic_tuning,
                cvkwargs=cvkwargs,
                kwargs=trend_estimator_kwargs,
            )
            if max_trend_poly_order > 1:
                for i in range(2,max_trend_poly_order+1):
                    f1.add_poly_terms('t',pwr=i)
                    trend_metrics['t' + str(i)] = self._Xvar_select_forecast(
                        f=f1,
                        estimator=trend_estimator,
                        monitor=monitor,
                        cross_validate=cross_validate,
                        dynamic_tuning=dynamic_tuning,
                        cvkwargs=cvkwargs,
                        kwargs=trend_estimator_kwargs,
                    )
            if try_ln_trend:
                f2 = copy.deepcopy(ft)
                f2.add_logged_terms('t',drop=True)
                trend_metrics['lnt'] = self._Xvar_select_forecast(
                    f=f2,
                    estimator=trend_estimator,
                    monitor=monitor,
                    cross_validate=cross_validate,
                    dynamic_tuning=dynamic_tuning,
                    cvkwargs=cvkwargs,
                    kwargs=trend_estimator_kwargs,
                )
                if max_trend_poly_order > 1:
                    for i in range(2,max_trend_poly_order+1):
                        f2.add_poly_terms('lnt',pwr=i)
                        trend_metrics['lnt' + str(i)] = self._Xvar_select_forecast(
                            f=f2,
                            estimator=trend_estimator,
                            monitor=monitor,
                            cross_validate=cross_validate,
                            dynamic_tuning=dynamic_tuning,
                            cvkwargs=cvkwargs,
                            kwargs=trend_estimator_kwargs,
                        )
        trends = self.parse_labeled_metrics(trend_metrics)
        best_trend = [k for k in trends][0]
        
        if try_seasonalities:
            seasonalities = {
                (
                    'Q',
                    'BQ',
                    'QS',
                    'Q-DEC',
                    'Q-JAN',
                    'Q-FEB',
                    'Q-MAR',
                    'Q-APR',
                    'Q-MAY',
                    'Q-JUN',
                    'Q-JUL',
                    'Q-AUG',
                    'Q-SEP',
                    'Q-OCT',
                    'Q-NOV',
                    'BQ-DEC',
                    'BQ-JAN',
                    'BQ-FEB',
                    'BQ-MAR',
                    'BQ-APR',
                    'BQ-MAY',
                    'BQ-JUN',
                    'BQ-JUL',
                    'BQ-AUG',
                    'BQ-SEP',
                    'BQ-OCT',
                    'BQ-NOV',
                    'QS-DEC',
                    'QS-JAN',
                    'QS-FEB',
                    'QS-MAR',
                    'QS-APR',
                    'QS-MAY',
                    'QS-JUN',
                    'QS-JUL',
                    'QS-AUG',
                    'QS-SEP',
                    'QS-OCT',
                    'QS-NOV',
                ):['quarter'],
                (
                    'M',
                    'MS',
                    'SM',
                    'BM',
                ):['month'],
                (
                    'W',
                    'W-SUN',
                    'W-MON',
                    'W-TUE',
                    'W-WED',
                    'W-THU',
                    'W-FRI',
                    'W-SAT',
                ):['week'],
                ('B','D'):['dayofweek','day','dayofyear'],
                ('H',):['hour'],
                ('T',):['minute'],
                ('S',):['second'],
            }
                        
            i = 0
            for freq, seas in seasonalities.items():
                seas_to_try += [s for s in seas if s not in exclude_seasonalities]
                if f.freq in freq:
                    i+=1
                    break
            if not i:
                warnings.warn(
                    f'No seasonalities are currently associated with the {f.freq} frequency.',
                    category=Warning,
                )
            else:
                seas_to_try.reverse() # lowest to highest order seasonality
                for i,s in enumerate(seas_to_try):
                    f1 = copy.deepcopy(f)
                    f1.set_estimator(estimator)
                    if isinstance(seasonality_repr,list):
                        f1.add_seasonal_regressors(
                            *seas_to_try[:(i+1)],
                            raw='raw' in seasonality_repr, # since this defaults to True, do it this way
                            sincos='sincos' in seasonality_repr,
                            dummy='dummy' in seasonality_repr,
                            drop_first='drop_first' in seasonality_repr,
                        )
                    elif isinstance(seasonality_repr,dict):
                        for s1 in seas_to_try[:(i+1)]:
                            if s1 in seasonality_repr:
                                f1.add_seasonal_regressors(
                                    s1,
                                    raw='raw' in seasonality_repr[s1],
                                    sincos='sincos' in seasonality_repr[s1],
                                    dummy='dummy' in seasonality_repr[s1],
                                    drop_first='drop_first' in seasonality_repr[s1],

                                )
                            else: # default to fourier
                                f1.add_seasonal_regressors(
                                    s1,
                                    raw=False,
                                    sincos=True,
                                )
                    else:
                        raise TypeError(f'seasonality_repr must be list or dict type, got {type(seasonality_repr)}.')
                    seasonality_metrics[s] = self._Xvar_select_forecast(
                        f=f1,
                        estimator=estimator,
                        monitor=monitor,
                        cross_validate=cross_validate,
                        dynamic_tuning=dynamic_tuning,
                        cvkwargs=cvkwargs,
                        kwargs=kwargs,
                    )
        
        seasonalities = self.parse_labeled_metrics(seasonality_metrics)
        best_seasonality = [k for k in seasonalities][0]
        
        if max_ar == 'auto' or max_ar > 0:
            max_ar = max(len(f.future_dates),f.test_length) if max_ar == 'auto' else max_ar
            for i in range(1,max_ar+1):
                f1 = copy.deepcopy(f)
                f1.add_ar_terms(i)
                ar_metrics[i] = self._Xvar_select_forecast(
                    f=f1,
                    estimator=estimator,
                    monitor=monitor,
                    cross_validate=cross_validate,
                    dynamic_tuning=dynamic_tuning,
                    cvkwargs=cvkwargs,
                    kwargs=kwargs,
                )
                if np.isnan(ar_metrics[i].score):
                    warnings.warn(
                        f'Cannot estimate {estimator} model with {i} AR terms.',
                        category=Warning,
                    )
                    ar_metrics.pop(i)
                    break

        ar_orders = self.parse_labeled_metrics(ar_metrics)
        best_ar_order = [k for k in ar_orders][0]

        f = copy.deepcopy(self)

        Xvars = get_Xvar_combos(
            f,
            best_trend,
            best_seasonality,
            best_ar_order,
            regressors_already_added,
            seas_to_try,
        )
        for xvar_set in Xvars:
            final_metrics[tuple(xvar_set)] = self._Xvar_select_forecast(
                f=f,
                estimator=estimator,
                monitor=monitor,
                cross_validate=cross_validate,
                dynamic_tuning=dynamic_tuning,
                cvkwargs=cvkwargs,
                kwargs=kwargs,
                Xvars=xvar_set,
            )
        combos = self.parse_labeled_metrics(final_metrics)
        best_combo = [k for k in combos][0]

        f.drop_Xvars(*[x for x in f.get_regressor_names() if x not in best_combo])
        self.current_xreg = f.current_xreg
        self.future_xreg = f.future_xreg
        if not final_metrics:
            warnings.warn(
                "auto_Xvar_select() did not add any regressors to the object."
                " Sometimes this happens when the object's test length is 0"
                " and the function's monitor argument is specified as a test set metric.",
                category = Warning,
            )
        return final_metrics

    def restore_series_length(self):
        """ Restores the original y values and current dates in the object from before `keep_smaller_history()`
        or `determine_best_series_length()` were called. If those methods were never called, this function
        does nothing. Restoring a series' length automatically drops all stored regressors in the object.
        """
        if not hasattr(self,'orig_attr'):
            return

        self.y = pd.Series(self.orig_attr['y'])
        self.current_dates = pd.Series(self.orig_attr['cd'])
        self.drop_all_Xvars()
        delattr(self,'orig_attr')

    def determine_best_series_length(
        self,
        estimator:AvailableModel = 'mlr',
        min_obs:PositiveInt = 100,
        max_obs:Optional[PositiveInt] = None,
        step:PositiveInt = 25,
        monitor:DetermineBestBy = 'ValidationMetricValue',
        cross_validate:bool = False,
        dynamic_tuning:DynamicTesting = False,
        cvkwargs:dict[str,Any] = {},
        chop:bool = True,
        **kwargs:Any,
    ) -> dict[int,float]:
        """ Attempts to find the optimal length for the series to produce accurate forecasts by systematically shortening the series, 
        running estimations, and monitoring a passed metric value.
        This should be run after Xvars have already been added to the object and all Xvars will be used in the iterative estimations.

        Args:
            estimator (str): One of Forecaster.estimators. Default 'mlr'.
                The estimator to use to determine the best series length.
            min_obs (int): Default 100.
                The shortest representation of the series to search.
            max_obs (int): Optional.
                The longest representation of the series to search.
                By default, the last estimation will be run on all available observations.
            step (int): Default 25.
                How big a step to take between iterations.
            monitor (str): One of Forecaster.determine_best_by. Default 'ValidationSetMetric'.
                The metric to be monitored when making reduction decisions. 
            cross_validate (bool): Default False.
                Whether to tune the model with cross validation. 
                If False, uses the validation slice of data to tune.
                If not monitoring ValidationMetricValue, you will want to leave this False.
            dynamic_tuning (bool or int): Default False.
                Whether to dynamically tune the model or, if int, how many forecast steps to dynamically tune it.
            cvkwargs (dict): Default {}. Passed to the cross_validate() method.
            chop (bool): Default True. Whether to shorten the series if a shorter length is found to be best.
            **kwargs: Passed to manual_forecast() method and can include arguments related to 
                a given model's hyperparameters, dynamic_testing, or Xvars.

        Returns:
            (dict[int,float]): A dictionary where each key is a series length and the value is the derived metric 
            (based on what was passed to the monitor argument).

        >>> f.auto_Xvar_select()
        >>> f.determine_best_series_length()
        """
        history_metrics = {}
        max_obs = len(self.y) if max_obs is None else max_obs
        i = len(self.y) - 1
        for i in np.arange(min_obs,max_obs,step):
            f = copy.deepcopy(self)
            f.keep_smaller_history(i)
            history_metrics[i] = self._Xvar_select_forecast(
                f=f,
                estimator=estimator,
                monitor=monitor,
                dynamic_tuning=dynamic_tuning,
                cross_validate=cross_validate,
                cvkwargs=cvkwargs,
                kwargs=kwargs,
            )
        if i < max_obs:
            f = copy.deepcopy(self)
            history_metrics[max_obs] = self._Xvar_select_forecast(
                f=f,
                estimator=estimator,
                monitor=monitor,
                dynamic_tuning=dynamic_tuning,
                cross_validate=cross_validate,
                cvkwargs=cvkwargs,
                kwargs=kwargs,
            )
        best_history_to_keep = self.parse_labeled_metrics(history_metrics)

        if chop:
            self.keep_smaller_history(list(best_history_to_keep.keys())[0])
        
        return history_metrics

    def adf_test(
        self, 
        critical_pval:ConfInterval=0.05, 
        full_res:bool=True, 
        train_only:bool=False, 
        diffy:bool=False,
        **kwargs:Any,
    ) -> bool|tuple[float,float,int,int,dict,float]:
        """ Tests the stationarity of the y series using augmented dickey fuller.
        Ports from statsmodels: https://www.statsmodels.org/dev/generated/statsmodels.tsa.stattools.adfuller.html.

        Args:
            critical_pval (float): Default 0.05.
                The p-value threshold in the statistical test to accept the alternative hypothesis.
            full_res (bool): Default True.
                If True, returns a dictionary with the pvalue, evaluated statistic, 
                and other statistical information (returns what the `adfuller()` function from statsmodels does).
                If False, returns a bool that matches whether the test indicates stationarity.
            train_only (bool): Default False.
                If True, will exclude the test set from the test (to avoid leakage).
            diffy (bool or int): One of {True,False,0,1}. Default False.
                Whether to difference the data before passing the values to the function.
                If False or 0, does not difference.
                If True or 1, differences 1 time.
            **kwargs: Passed to the `adfuller()` function from statsmodels. See
                https://www.statsmodels.org/dev/generated/statsmodels.tsa.stattools.adfuller.html.

        Returns:
            (bool or tuple): If bool (full_res = False), returns whether the test suggests stationarity.
            Otherwise, returns the full results (stat, pval, etc.) of the test.

        >>> stat, pval, _, _, _, _ = f.adf_test(full_res=True)
        """
        _developer_utils._check_train_only_arg(self,train_only)
        y = self._diffy(diffy)
        res = adfuller(
            (
                y.values
                if not train_only
                else y.values[: -self.test_length]
            ),
            **kwargs,
        )
        return res if full_res else True if res[1] < critical_pval else False
            

    def normality_test(self, train_only=False):
        """ Runs D'Agostino and Pearson's test for normality
        ported from scipy: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.normaltest.html.
        Holds the null hypothesis that the series is normally distributed.

        Args:
            train_only (bool): Default False.
                If True, will exclude the test set from the test (to avoid leakage).

        Returns: 
            (float, float): The derived statistic and pvalue.
        """
        _developer_utils._check_train_only_arg(self,train_only)
        y = self.y.dropna().values if not train_only else self.y.dropna().values[: -self.test_length]
        return stats.normaltest(y)

    def plot_acf(self, diffy:bool=False, train_only:bool=False, **kwargs:Any) -> Figure:
        """ Plots an autocorrelation function of the y values.

        Args:
            diffy (bool or int): One of {True,False,0,1}. default False.
                Whether to difference the data before passing the values to the function.
                If False or 0, does not difference.
                If True or 1, differences 1 time.
            train_only (bool): Default False.
                If True, will exclude the test set from the test (a measure added to avoid leakage).
            **kwargs: Passed to plot_acf() function from statsmodels.

        Returns:
            (Figure): If ax is None, the created figure. Otherwise the figure to which ax is connected.

        >>> import matplotlib.pyplot as plt
        >>> f.plot_acf(train_only=True)
        >>> plt.plot()
        """
        _developer_utils._check_train_only_arg(self,train_only)
        y = self._diffy(diffy)
        y = y.values if not train_only else y.values[: -self.test_length]
        return plot_acf(y, **kwargs)

    def plot_pacf(self, diff:bool=False, train_only:bool=False, **kwargs:Any) -> Figure:
        """ Plots a partial autocorrelation function of the y values.

        Args:
            diffy (bool or int): One of {True,False,0,1}. Default False.
                Whether to difference the data before passing the values to the function.
                If False or 0, does not difference.
                If True or 1, differences 1 time.
            train_only (bool): Default False.
                If True, will exclude the test set from the test (a measure added to avoid leakage).
            **kwargs: Passed to plot_pacf() function from statsmodels.

        Returns:
            (Figure): If ax is None, the created figure. Otherwise the figure to which ax is connected.

        >>> import matplotlib.pyplot as plt
        >>> f.plot_pacf(train_only=True)
        >>> plt.plot()
        """
        _developer_utils._check_train_only_arg(self,train_only)
        y = self._diffy(diffy)
        y = y.values if not train_only else y.values[: -self.test_length]
        return plot_pacf(y, **kwargs)

    def plot_periodogram(self, diffy:bool=False, train_only:bool=False) -> tuple[np.ndarray,np.ndarray]:
        """ Plots a periodogram of the y values (comes from scipy.signal).

        Args:
            diffy (bool or int): One of {True,False,0,1}. Default False.
                Whether to difference the data before passing the values to the function.
                If False or 0, does not difference.
                If True or 1, differences 1 time.
            train_only (bool): Default False.
                If True, will exclude the test set from the test (a measure added to avoid leakage).

        Returns:
            (ndarray,ndarray): Element 1: Array of sample frequencies. Element 2: Power spectral density or power spectrum of x.

        >>> import matplotlib.pyplot as plt
        >>> a, b = f.plot_periodogram(diffy=True,train_only=True)
        >>> plt.semilogy(a, b)
        >>> plt.show()
        """
        from scipy.signal import periodogram

        _developer_utils._check_train_only_arg(self,train_only)
        y = self._diffy(diffy)
        y = y.values if not train_only else y.values[: -self.test_length]
        return periodogram(y)

    def STL(self, diffy:bool=False, train_only:bool=False, **kwargs:Any) -> DecomposeResult:
        """ Returns a Season-Trend decomposition using LOESS of the y values.
        
        Args:
            diffy (bool): Default False.
                Whether to difference the data before passing the values to the function.
                If False or 0, does not difference.
                If True or 1, differences 1 time.
            train_only (bool): Default False.
                If True, will exclude the test set from the test (a measure added to avoid leakage).
            **kwargs: Passed to STL() function from statsmodels.
                See https://www.statsmodels.org/dev/generated/statsmodels.tsa.seasonal.STL.html.

        Returns:
            (DecomposeResult): An object with seasonal, trend, and resid attributes.

        >>> import matplotlib.pyplot as plt
        >>> f.STL(train_only=True).plot()
        >>> plt.show()
        """
        _developer_utils._check_train_only_arg(self,train_only)
        y = self._diffy(diffy)
        current_dates = (
            self.current_dates.values[-len(y) :]
            if not train_only
            else self.current_dates.values[-len(y) : -self.test_length]
        )
        y = y.values if not train_only else y.values[: -self.test_length]
        X = pd.DataFrame({"y": y}, index=current_dates)
        X.index.freq = self.freq
        return STL(X.dropna(), **kwargs)

    def seasonal_decompose(self, diffy:bool=False, train_only:bool=False, **kwargs:Any) -> DecomposeResult:
        """ Returns a signal/seasonal decomposition of the y values.

        Args:
            diffy (bool): Default False.
                Whether to difference the data before passing the values to the function.
                If False or 0, does not difference.
                If True or 1, differences 1 time.
            train_only (bool): Default False.
                If True, will exclude the test set from the test (a measure added to avoid leakage).
            **kwargs: Passed to seasonal_decompose() function from statsmodels.
                See https://www.statsmodels.org/dev/generated/statsmodels.tsa.seasonal.seasonal_decompose.html.

        Returns:
            (DecomposeResult): An object with seasonal, trend, and resid attributes.

        >>> import matplotlib.pyplot as plt
        >>> f.seasonal_decompose(train_only=True).plot()
        >>> plt.show()
        """
        _developer_utils._check_train_only_arg(self,train_only)
        y = self._diffy(diffy)
        current_dates = (
            self.current_dates.values[-len(y) :]
            if not train_only
            else self.current_dates.values[-len(y) : -self.test_length]
        )
        y = y.values if not train_only else y.values[: -self.test_length]
        X = pd.DataFrame({"y": y}, index=current_dates)
        X.index.freq = self.freq
        return seasonal_decompose(X.dropna(), **kwargs)

    def tune_test_forecast(
        self,
        models:AvailableModel,
        cross_validate:bool=False,
        dynamic_tuning:bool=False,
        dynamic_testing:bool=True,
        feature_importance:bool=False,
        fi_try_order:Optional[Sequence[FIMethod]]=None,
        limit_grid_size:Optional[PositiveInt|ConfInterval]=None,
        min_grid_size:PositiveInt=1,
        suffix:Optional[str]=None,
        error:Literal['ignore','raise','warn']='raise',
        **cvkwargs:dict[str,Any],
    ) -> Self:
        """ Iterates through a list of models, tunes them using grids in a grids file, forecasts them, and can save feature information.

        Args:
            models (list-like):
                Each element must be a name in Forecaster.estimators.
            cross_validate (bool): Default False.
                Whether to tune the model with cross validation. 
                If False, uses the validation slice of data to tune.
            dynamic_tuning (bool or int): Default False.
                whether to dynamically tune the forecast (meaning AR terms will be propagated with predicted values).
                if True, evaluates dynamically over the entire out-of-sample slice of data.
                if int, window evaluates over that many steps (2 for 2-step dynamic forecasting, 12 for 12-step, etc.).
                setting this to False or 1 means faster performance, 
                but gives a less-good indication of how well the forecast will perform out x amount of periods.
            dynamic_testing (bool or int): Default True.
                whether to dynamically test the forecast (meaning AR terms will be propagated with predicted values).
                if True, evaluates dynamically over the entire out-of-sample slice of data.
                if int, window evaluates over that many steps (2 for 2-step dynamic forecasting, 12 for 12-step, etc.).
                setting this to False or 1 means faster performance, 
                but gives a less-good indication of how well the forecast will perform out x amount of periods.
            feature_importance (bool): Default False.
                Whether to save feature importance information for the models that offer it.
            fi_try_order (list): Optional.
                If the `feature_importance` argument is `True`, what feature importance methods to try? If using a combination
                of tree-based and linear models, for example, it might be good to pass ['TreeExplainer','LinearExplainer']. The default
                will use whatever is specifiec by default in `Forecaster.save_feature_importance()`, which usually ends up being
                the `PermutationExplainer`.
            limit_grid_size (int or float): Optional. Pass an argument here to limit each of the grids being read.
                See https://scalecast.readthedocs.io/en/latest/Forecaster/Forecaster.html#src.scalecast.Forecaster.Forecaster.limit_grid_size.
            min_grid_size (int): Default 1. The smallest grid size to keep. Ignored if limit_grid_size is None.
            suffix (str): Optional. A suffix to add to each model as it is evaluated to differentiate them when called
                later. If unspecified, each model can be called by its estimator name.
            error (str): One of 'ignore','raise','warn'; default 'raise'.
                What to do with the error if a given model fails.
                'warn' prints a warning that the model could not be evaluated.
            **cvkwargs: Passed to the cross_validate() method.

        Returns:
            Self

        >>> models = ('mlr','mlp','lightgbm')
        >>> f.tune_test_forecast(models,dynamic_testing=False,feature_importance=True)
        """
        _tune_test_forecast(
            f = self,
            models = models,
            cross_validate=cross_validate,
            dynamic_tuning=dynamic_tuning,
            dynamic_testing=dynamic_testing,
            limit_grid_size=limit_grid_size,
            min_grid_size=min_grid_size,
            suffix=suffix,
            error=error,
            feature_importance=feature_importance,
            fi_try_order=fi_try_order,
            **cvkwargs,
        )
        return self

    def save_feature_importance(
            self, 
            method:Literal['shap']="shap", 
            on_error:Literal['warn','ignore','raise']="warn", 
            try_order:Sequence[FIMethod] = [
                'PermutationExplainer',
                'TreeExplainer',
                'LinearExplainer',
                'KernelExplainer',
                'SamplingExplainer',
            ], 
            masker:Optional["shap.maskers.Masker"] = None,
            verbose:bool = False,
    ):
        """ Requires shap. Saves feature info for models that offer it (sklearn models).
        Call after evaluating the model you want it for and before changing the estimator.
        This method saves a dataframe listing the feature as the index and its score. This dataframe can be recalled using
        the `export_feature_importance()` method. The importance scores
        are determined as the average shap score applied to each feature in each observation.

        Args:
            method (str): Default 'shap'.
                As of scalecast 0.19.4, shap is the only method available, as pfi is deprecated.
            on_error (str): One of {'warn','raise','ignore'}. Default 'warn'.
                If the last model called doesn't support feature importance,
                'warn' will log a warning. 'raise' will raise an error.
            try_order (list): The order of explainers to try. 
                If one fails, will try setting with the next one. This should be able to set feature importance on
                any sklearn model.
                What each Explainer does can be found in the shap documentation: 
                https://shap-lrjball.readthedocs.io/en/latest/index.html
            masker (shap.maskers): Optional.
                Pass your own masker if desired and you are using the PermutationExplainer or LinearExplainer. 
                Default will use shap.maskers.Independent masker with default arguments.
            verbose (bool): Default True.
                Whether to print out information about which explainers were tried/chosen.
                The chosen explainer is saved in Forecaster.history[estimator]['feature_importance_explainer'].

        >>> f.set_estimator('xgboost')
        >>> f.manual_forecast()
        >>> f.save_feature_importance()
        >>> fi = f.export_feature_importance('xgboost') # returns a dataframe
        """
        import shap
        
        _developer_utils.descriptive_assert(
            method in ("shap",),
            ValueError,
            f'`method` must be "shap", got "{method}".',
        )
        required_args = {
            # name of explainer: [required method, required arg]
            'PermutationExplainer':['predict','masker'],
            'TreeExplainer':[None,None],
            'LinearExplainer':[None,'masker'],
            'KernelExplainer':['predict','data'],
            'SamplingExplainer':['predict','data'],
            # fe:
            #'GPUTreeExplainer':[None,None],
            #'AdditiveExplainer':['predict','masker'],
            #'GradientExplainer':
            #'PartitionExplainer':
            'DeepExplainer':[None,None],
        }
        fail = False
        if not hasattr(self.call_estimator,'Xvars') or not self.call_estimator.Xvars:
            fail = True
            error = f'Feature importance only works for models that use external regressors.'
        else:
            try:
                regr = self.call_estimator
                X = self.call_estimator.generate_current_X()
                Xvars = self.call_estimator.Xvars

                if masker is None:
                    masker = shap.maskers.Independent(data = X)

                for t in try_order:
                    if t not in required_args:
                        raise ValueError(
                            f'{t} is not an available explainer and all explainers tried before it failed.'
                        )
                    if verbose:
                        print(f'Trying to set feature importance with {t}.')
                    
                    args = required_args[t]

                    if args[0] is None:
                        inp1 = regr
                    elif args[0] == 'predict':
                        inp1 = regr.predict
                    
                    if args[1] is None:
                        inp2 = []
                    elif args[1] == 'data':
                        inp2 = [X]
                    elif args[1] == 'masker':
                        inp2 = [masker]

                    try:
                        explainer = getattr(shap,t)(inp1,*inp2)
                        shap_values = explainer(X)
                        if verbose:
                            print(f'{t} successful!')
                    except Exception as e:
                        error = str(e)
                        continue
                    break
                
                if 'explainer' not in locals() or 'shap_values' not in locals():
                    raise TypeError(error)
                
                shap_df = pd.DataFrame(shap_values.values, columns=Xvars)
                shap_fi = (
                    pd.DataFrame(
                        {
                            "feature": shap_df.columns.to_list(),
                            "weight": np.abs(shap_df).mean(),
                            "std": np.abs(shap_df).std(),
                        }
                    )
                    .set_index("feature")
                    .sort_values("weight", ascending=False)
                )
                self.explainer = t
                self.feature_importance = shap_fi
            except Exception as e:
                fail = True
                error = e
        if fail:
            if on_error == "warn":
                warnings.warn(
                    f"Cannot set {method} feature importance on {self.call_me}."
                    f" Here is the error: {error}",
                    category = Warning,
                )
            elif on_error == "raise":
                raise Exception(error)
            elif on_error != 'ignore':
                raise ValueError(f"Value passed to on_error not recognized: {on_error}.")
            return

        self._bank_fi_to_history()

    def chop_from_front(self, n:PositiveInt, fcst_length:Optional[PositiveInt] = None) -> Self:
        """ Cuts the amount of y observations in the object from the front counting backwards.
        The current length of the forecast horizon will be maintained and all future regressors will be rewritten to the appropriate attributes.

        Args:
            n (int):
                The number of observations to cut from the front.
            fcst_length (int): Optional.
                The new length of the forecast length.
                By default, maintains the same forecast length currently in the object.

        Returns:
            Self

        >>> f.chop_from_front(10) # keeps all observations before the last 10
        """
        n = int(n)
        fcst_length = len(self.future_dates) if fcst_length is None else fcst_length
        self.y = self.y.iloc[:-n]
        self.current_dates = self.current_dates.iloc[:-n]
        self.generate_future_dates(fcst_length)
        self.future_xreg = {k:(self.current_xreg[k].to_list()[-n:] + v[:max(0,(fcst_length-n))])[-fcst_length:] for k, v in self.future_xreg.items()}
        self.future_xreg = (
            {k:v[:k.lag_order] + ([np.nan] * (len(self.future_dates) - k.lag_order)) for k, v in self.future_xreg.items() if k in self.list_stored_ar_terms()}|
            {k:v[:] for k, v in self.future_xreg.items() if k not in self.list_stored_ar_terms()}
        )
        self.current_xreg = {k:v.iloc[:-n].reset_index(drop=True) for k, v in self.current_xreg.items()}
        return self

    def chop_from_back(self,n:PositiveInt) -> Self:
        """ Cuts y observations in the object from the back by counting forward from the beginning.

        Args:
            n (int): The number of observations to cut from the back.

        Returns:
            Self

        >>> f.chop_from_back(10) # chops 10 observations off the back
        """
        n = int(n)
        to_chop = -(len(self.y) - n)
        self.y = self.y.iloc[to_chop:]
        self.current_dates = self.current_dates.iloc[to_chop:]
        self.current_xreg = {
            k:v.iloc[to_chop:].reset_index(drop=True) 
            for k, v in self.current_xreg.items()
        }
        return self

    def keep_smaller_history(self, n:PositiveInt|str|DatetimeLike) -> Self:
        """ Cuts y observations in the object by counting back from the beginning.

        Args:
            n (int, str, or datetime.datetime):
                If int, the number of observations to keep.
                Otherwise, the last observation to keep.
                Must be parsable by pandas' Timestamp function.

        Returns:
            Self

        >>> f.keep_smaller_history(500) # keeps last 500 observations
        >>> f.keep_smaller_history('2020-01-01') # keeps only observations on or later than 1/1/2020
        """
        if (type(n) is datetime.datetime) or (type(n) is pd.Timestamp) or isinstance(n,str):
            n = len([i for i in self.current_dates if i >= pd.Timestamp(n)])
        
        n = int(n)
        _developer_utils.descriptive_assert(n > 2,ValueError,"n must be an int, datetime object, or str and there must be more than 2 observations to keep.")
        self.orig_attr = {'y':self.y.values.copy(),'cd':self.current_dates.values.copy()} # for reverting later
        self.y = self.y.iloc[-n:]
        self.current_dates = self.current_dates.iloc[-n:]
        self.current_xreg = {k:v.iloc[-n:].reset_index(drop=True) for k, v in self.current_xreg.items()}
        return self

    def get_regressor_names(self):
        """ Gets the regressor names stored in the object.

        Returns:
            (list): Regressor names that have been added to the object.
        
        >>> f.add_time_trend()
        >>> f.get_regressor_names()
        """
        return [k for k in self.current_xreg.keys()]

    def get_freq(self):
        """ Gets the pandas inferred date frequency.
        
        Returns:
            (str): The inferred frequency of the current_dates array.

        >>> f.get_freq()
        """
        return self.freq

    def validate_regressor_names(self):
        """ Validates that all regressor names exist in both current_xregs and future_xregs.
        Raises an error if this is not the case.
        """
        try:
            assert sorted(self.current_xreg.keys()) == sorted(self.future_xreg.keys())
        except AssertionError:
            case1 = [
                k for k in self.current_xreg.keys() if k not in self.future_xreg.keys()
            ]
            case2 = [
                k for k in self.future_xreg.keys() if k not in self.current_xreg.keys()
            ]
            raise ValueError(
                f"The following regressors are in current_xreg but not future_xreg: {case1}\n"
                f"The following regressors are in future_xreg but not current_xreg: {case2}",
            )

    def plot(
        self, 
        models:ModelValues="all",
        exclude:list[EvaluatedModel] = [],
        order_by:Optional[DetermineBestBy]=None, 
        ci:bool=False,
        ax:Optional[Axes] = None,
        figsize:tuple[int,int]=(12,6),
        colors:Optional[list[str]] = COLORS,
    ) -> Axes:
        """ Plots all forecasts with the actuals, or just actuals if no forecasts have been evaluated or are selected.

        Args:
            models (list-like, str, or None): Default 'all'.
               The forecasted models to plot.
               Can start with "top_" and the metric specified in order_by will be used to order the models appropriately.
               If None or models/order_by combo invalid, will plot only actual values.
            exclude (collection): Default []. Pass any models here that you don't want displayed.
                Good to use in conjunction with models = 'top_{n}'. 
            order_by (str): Optional. One of Forecaster.determine_best_by.  
                How to order the display of forecasts on the plots (from best-to-worst according to the selected metric).
                Default doesn't order.
            ci (bool): Default False.
                Whether to display the confidence intervals.
            ax (Axis): Optional. The existing axis to write the resulting figure to.
            figsize (tuple): Default (12,6). The size of the resulting figure. Ignored when ax is not None.
            colors (list[str]): Optional. The colors to use when making the plot.

        Returns:
            (Axis): The figure's axis.

        >>> models = ('mlr','mlp','lightgbm')
        >>> f.tune_test_forecast(models,dynamic_testing=False,feature_importance=True)
        >>> f.plot(order_by='TestSetRMSE') # plots all forecasts
        >>> plt.show()
        """
        try:
            models = [m for m in self._parse_models(models, order_by) if m not in exclude]
        except (ValueError, TypeError):
            models = None

        colors = cycle(colors)

        if ax is None:
            _, ax = plt.subplots(figsize=figsize)

        if models is None:
            sns.lineplot(
                x=self.current_dates.values, 
                y=self.y.values, 
                label="actuals", 
                ax=ax
            )
        else:
            y = self.y.dropna().values
            plot = {
                "date": self.current_dates.values[-len(y) :],
                "actuals": y,
            }
            plot["actuals_len"] = min(len(plot["date"]), len(plot["actuals"]))

            sns.lineplot(
                x=plot["date"][-plot["actuals_len"] :],
                y=plot["actuals"][-plot["actuals_len"] :],
                label="actuals",
                ax=ax,
            )
            for m in models:
                color = next(colors)
                plot[m] = (self.history[m]["Forecast"])
                if plot[m] is None or not len(plot[m]):
                    continue
                sns.lineplot(
                    x=self.future_dates.to_list(),
                    y=plot[m],
                    color=color,
                    label=m,
                    ax=ax,
                )
                if ci:
                    try:
                        ax.fill_between(
                            x=self.future_dates.values,
                            y1=self.history[m]["UpperCI"],
                            y2=self.history[m]["LowerCI"],
                            alpha=0.2,
                            color=color,
                            label="{} {:.0%} CI".format(m, self.history[m]["CILevel"]),
                        )
                    except KeyError:
                        _developer_utils._warn_about_not_finding_cis(m)
        
        ax.legend()
        ax.set_xlabel("Date")
        ax.set_ylabel("Values")
        return ax

    def plot_test_set(
        self, 
        models:ModelValues="all",
        exclude:list[EvaluatedModel] = [],
        order_by:Optional[DetermineBestBy]=None, 
        include_train:bool|NonNegativeInt=True, 
        ci:bool=False,
        ax:Optional[Axes] = None,
        figsize:tuple[int,int]=(12,6),
        colors:Optional[list[str]] = COLORS,
    ):
        """ Plots all test-set predictions with the actuals.

        Args:
            models (list-like or str): Default 'all'.
               The forecated models to plot.
               Can start with "top_" and the metric specified in order_by will be used to order the models appropriately.
            exclude (collection): Default []. Pass any models here that you don't want displayed.
                Good to use in conjunction with models = 'top_{n}'. 
            order_by (str): Optional. One of Forecaster.determine_best_by.
                How to order the display of forecasts on the plots (from best-to-worst according to the selected metric).
            include_train (bool or int): Default True.
                Use to zoom into testing results.
                If True, plots the test results with the entire history in y.
                If False, matches y history to test results and only plots this.
                If int, plots that length of y to match to test results.
            ci (bool): Default False.
                Whether to display the confidence intervals.
                Default is 100 boostrapped samples and a 95% confidence interval.
            ax (Axis): Optional. The existing axis to write the resulting figure to.
            figsize (tuple): Default (12,6). Size of the resulting figure. 
                Ignored when ax is not None.
            colors (list[str]): Optional. The colors to use when making the plot.

        Returns:
            (Axis): The figure's axis.

        >>> models = ('mlr','mlp','lightgbm')
        >>> f.tune_test_forecast(models,dynamic_testing=False,feature_importance=True)
        >>> f.plot(order_by='TestSetRMSE') # plots all test-set results
        >>> plt.show()
        """
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)

        colors = cycle(colors)

        models = [m for m in self._parse_models(models, order_by) if m not in exclude]
        _developer_utils.descriptive_assert(
            np.all(['TestSetPredictions' in self.history[m].keys() for m in models]),
            ForecastError,
            'plot_test_set() does not work when models were not tested (test_length set to 0).',
        )
        y = self.y.dropna().values
        plot = {
            "date": self.current_dates.to_list()[-len(y) :],
            "actuals": y
        }
        plot["actuals_len"] = min(len(plot["date"]), len(plot["actuals"]))

        if not isinstance(include_train,bool):
            include_train = int(include_train)
            _developer_utils.descriptive_assert(
                include_train > 1,
                ValueError,
                f"include_train must be a bool type or an int greater than 1, got {include_train}.",
            )
            plot["actuals"] = plot["actuals"][-include_train:]
            plot["date"] = plot["date"][-include_train:]
        else:
            if not include_train:
                plot["actuals"] = plot["actuals"][-self.test_length :]
                plot["date"] = plot["date"][-self.test_length :]

        sns.lineplot(
            x=plot["date"][-plot["actuals_len"] :],
            y=plot["actuals"][-plot["actuals_len"] :],
            label="actuals",
            ax=ax,
        )

        for m in models:
            color = next(colors)
            plot[m] = (self.history[m]["TestSetPredictions"])
            test_dates = self.current_dates.values[-len(plot[m]) :]
            sns.lineplot(
                x=test_dates,
                y=plot[m],
                linestyle="--",
                color=color,
                alpha=0.7,
                label=m,
                ax=ax,
            )
            if ci:
                try:
                    ax.fill_between(
                        x=test_dates,
                        y1=self.history[m]["TestSetUpperCI"],
                        y2=self.history[m]["TestSetLowerCI"],
                        alpha=0.2,
                        color=color,
                        label="{} {:.0%} CI".format(m, self.history[m]["CILevel"]),
                    )
                except KeyError:
                    _developer_utils._warn_about_not_finding_cis(m)

        ax.legend()
        ax.set_xlabel("Date")
        ax.set_ylabel("Values")
        return ax

    def plot_fitted(
        self, 
        models:ModelValues="all",
        exclude:list[EvaluatedModel] = [], 
        order_by:Optional[DetermineBestBy]=None, 
        ax:Optional[Axes] = None,
        figsize:tuple[int,int]=(12,6),
        colors:Optional[list[str]] = COLORS,
    ):
        """ Plots all fitted values with the actuals. Does not support level fitted values (for now).

        Args:
            models (list-like,str): Default 'all'.
               The forecated models to plot.
               Can start with "top_" and the metric specified in order_by will be used to order the models appropriately.
            exclude (collection): Default []. Pass any models here that you don't want displayed.
                Good to use in conjunction with models = 'top_{n}'. 
            order_by (str): Optional. One of Forecaster.determine_best_by.
                How to order the display of forecasts on the plots (from best-to-worst according to the selected metric).
            ax (Axis): Optional. The existing axis to write the resulting figure to.
            figsize (tuple): Default (12,6). Size of the resulting figure. Ignored when ax is not None.
            colors (list[str]): Optional. The colors to use when making the plot.

        Returns:
            (Axis): The figure's axis.

        >>> models = ('mlr','mlp','lightgbm')
        >>> f.tune_test_forecast(models,dynamic_testing=False,feature_importance=True)
        >>> f.plot_fitted(order_by='TestSetRMSE') # plots all fitted values
        >>> plt.show()
        """
        colors = cycle(colors)

        if ax is None:
            _, ax = plt.subplots(figsize=figsize)

        models = [m for m in self._parse_models(models, order_by) if m not in exclude]
        dates = self.current_dates.values
        actuals = self.y.values

        plot = {
            "date": dates,
            "actuals": actuals,
        }
        sns.lineplot(x=plot["date"], y=plot["actuals"], label="actuals", ax=ax)

        for m in models:
            color = next(colors)
            plot[m] = (self.history[m]["FittedVals"])
            if plot[m] is None or not len(plot[m]):
                continue
            sns.lineplot(
                x=plot["date"][-len(plot[m]) :],
                y=plot[m][-len(plot["date"]) :],
                linestyle="--",
                color=color,
                alpha=0.7,
                label=m,
                ax=ax,
            )

        ax.legend()
        ax.set_xlabel("Date")
        ax.set_ylabel("Values")
        return ax

    def export(
        self,
        dfs:ExportOptions|list[ExportOptions]=[
            "model_summaries",
            "lvl_test_set_predictions",
            "lvl_fcsts",
        ],
        models:ModelValues="all",
        best_model:Literal['auto']|EvaluatedModel="auto",
        determine_best_by:Optional[DetermineBestBy]=None,
        cis:bool=False,
        to_excel:bool=False,
        out_path:os.PathLike=Path.cwd(),
        excel_name:str="results.xlsx",
    ) -> pd.DataFrame|dict[str,pd.DataFrame]:
        """ Exports 1-all of 3 pandas DataFrames. Can write to excel with each DataFrame on a separate sheet.
        Will return either a dictionary with dataframes as values (df str arguments as keys) or a single dataframe if only one df is specified.

        Args:
            dfs (list-like or str): Default 
                ['model_summaries', 'lvl_test_set_predictions', 'lvl_fcsts'].
                A list or name of the specific dataframe(s) you want returned and/or written to excel.
                Must be one of or multiple of the elements in default.
                Exporting test set predictions only works if all exported models were tested using the same test length.
            models (list-like or str): Default 'all'.
                The models to write information for.
                Can start with "top_" and the metric specified in `determine_best_by` will be used to order the models appropriately.
            best_model (str): Default 'auto'.
                The name of the best model, if "auto", will determine this by the metric in determine_best_by.
                If not "auto", must match a model nickname of an already-evaluated model.
            determine_best_by (str): One of Forecaster.determine_best_by or None. Default 'TestSetRMSE'.
                If None and best_model is 'auto', the best model will be designated as the first-evaluated model.
            to_excel (bool): Default False.
                Whether to save to excel.
            out_path (PathLike): Default './'.
                The path to save the excel file to (ignored when `to_excel=False`).
            cis (bool): Default False.
                Whether to export confidence intervals for models in 
                "lvl_test_set_predictions", "lvl_fcsts"
                dataframes.
            excel_name (str): Default 'results.xlsx'.
                The name to call the excel file (ignored when `to_excel=False`).

        Returns:
            (DataFrame or Dict[str,DataFrame]): either a single pandas dataframe if one element passed to dfs 
            or a dictionary where the keys match what was passed to dfs and the values are dataframes. 

        >>> results = f.export(dfs=['model_summaries','lvl_fcsts'],to_excel=True) # returns a dict
        >>> model_summaries = results['model_summaries'] # returns a dataframe
        >>> lvl_fcsts = results['lvl_fcsts'] # returns a dataframe
        >>> ts_preds = f.export('lvl_test_set_predictions') # returns a dataframe
        """
        _developer_utils.descriptive_assert(
            isinstance(cis, bool),
            "ValueError",
            f"Argument passed to cis must be a bool type, got {type(cis)}.",
        )

        match dfs:
            case str():
                dfs = [dfs]
            case obj if hasattr(obj, "__len__") and not obj:
                raise ValueError("No values passed to the dfs argument.")
            case _:
                dfs = list(dfs)

        models = self._parse_models(models, determine_best_by)
        _dfs_ = ["model_summaries","lvl_test_set_predictions","lvl_fcsts"]
        _bad_dfs_ = [i for i in dfs if i not in _dfs_]
        if _bad_dfs_:
            raise ValueError(f"Values passed to the dfs list must be in {_dfs_}, not {_bad_dfs_}")
        
        match best_model:
            case 'auto':
                if determine_best_by is None:
                    best_fcst_name = list(self.history.keys())[0] # first evaluated model
                else:
                    best_fcst_name = self.order_fcsts(models, determine_best_by)[0]
            case _:
                best_fcst_name = best_model

        output = {}
        if "model_summaries" in dfs:
            cols = [
                "ModelNickname",
                "Estimator",
                "Xvars",
                "HyperParams",
                "Observations",
                "DynamicallyTested",
                "TestSetLength",
                "CILevel",
                "ValidationMetric",
                "ValidationMetricValue",
                "models",
                "weights",
                "best_model",
            ]

            model_summaries = []
            for m in models:
                model_summary_m = pd.DataFrame({"ModelNickname": [m]})
                for c in cols:
                    if c in self.history[m]:
                        model_summary_m[c] = [self.history[m][c]]
                    elif c == "best_model":
                        model_summary_m[c] = m == best_fcst_name
                    else:
                        model_summary_m[c] = pd.NA
                for c in self.determine_best_by:
                    if c in self.history[m]:
                        model_summary_m[c] = self.history[m][c].score
                model_summaries.append(model_summary_m)
            output["model_summaries"] = pd.concat(model_summaries)
        
        if "lvl_fcsts" in dfs:
            lvl_fcsts = pd.DataFrame({"DATE": self.future_dates.to_list()})
            for m in models:
                lvl_fcsts[m] = self.history[m]["Forecast"]
                if cis:
                    try:
                        lvl_fcsts[m + "_upperci"] = self.history[m]["UpperCI"]
                        lvl_fcsts[m + "_lowerci"] = self.history[m]["LowerCI"]
                    except KeyError:
                        _developer_utils._warn_about_not_finding_cis(m)
            output["lvl_fcsts"] = lvl_fcsts
        
        if "lvl_test_set_predictions" in dfs:
            if self.test_length == 0:
                output["lvl_test_set_predictions"] = pd.DataFrame()
            else:
                test_set_predictions = pd.DataFrame({"DATE": self.current_dates.values[-self.test_length :]})
                test_set_predictions["actual"] = self.y.values[-self.test_length :]
                for m in models:
                    test_set_predictions[m] = self.history[m]["TestSetPredictions"]
                    if cis:
                        try:
                            test_set_predictions[m + "_upperci"] = self.history[m]["TestSetUpperCI"]
                            test_set_predictions[m + "_lowerci"] = self.history[m]["TestSetLowerCI"]
                        except KeyError:
                            _developer_utils._warn_about_not_finding_cis(m)
                output["lvl_test_set_predictions"] = test_set_predictions

        if to_excel:
            out_path = Path(out_path)
            with pd.ExcelWriter(out_path/excel_name, engine="openpyxl") as writer:
                for k, df in output.items():
                    df.to_excel(writer, sheet_name=k, index=False)

        if len(output.keys()) == 1:
            return list(output.values())[0]
        else:
            return output

    def export_feature_importance(self, model:EvaluatedModel) -> pd.DataFrame:
        """ Exports the feature importance from a model.
        Raises an error if you never saved the model's feature importance.

        Args:
            model (str):
                The name of them model to export for.
                Matches what was passed to call_me when evaluating the model.

        Returns:
            (DataFrame): The resulting feature importances of the evaluated model passed to `model`.

        >>> fi = f.export_feature_importance('mlr')
        """
        return self.history[model]["feature_importance"]

    def all_feature_info_to_excel(self, out_path:os.PathLike=Path.cwd(), excel_name:str="feature_info.xlsx"):
        """ Saves all feature importance to excel.
        Each model where such info is available for gets its own tab.
        Be sure to have called save_feature_importance() before using this function.

        Args:
            out_path (PathLike): Default './'.
                The path to export to.
            excel_name (str): Default 'feature_info.xlsx'.
                The name of the resulting excel file.

        Returns:
            None
        """
        try:
            out_path = Path(out_path)
            with pd.ExcelWriter(out_path/excel_name, engine="openpyxl") as writer:
                for m in self.history.keys():
                    if "feature_importance" in self.history[m].keys():
                        self.history[m]["feature_importance"].to_excel(
                            writer, sheet_name=f"{m}_feature_importance"
                        )
        except IndexError:
            raise ForecastError( "No saved feature importance could be found.")

    def all_validation_grids_to_excel(
        self,
        out_path:os.PathLike=Path.cwd(),
        excel_name:str="validation_grids.xlsx",
    ):
        """ Saves all validation grids to excel.
        Each model where such info is available for gets its own tab.
        Be sure to have tuned at least model before calling this.

        Args:
            out_path (PathLike): Default uses current working directory.
                The path to export to.
            excel_name (str): Default 'feature_info.xlsx'.
                The name of the resulting excel file.

        Returns:
            None
        """
        try:
            out_path = Path(out_path)
            with pd.ExcelWriter(out_path/excel_name, engine="openpyxl") as writer:
                for m in self.history.keys():
                    if "grid_evaluated" in self.history[m].keys():
                        df = self.export_validation_grid(m)
                        df.to_excel(writer, sheet_name=m, index=False)
        except IndexError:
            raise ForecastError("No validation grids could be found.")

    def export_Xvars_df(self, dropna=False):
        """ Gets all utilized regressors and values.
            
        Args:
            dropna (bool): Default False.
                Whether to drop null values from the resulting dataframe

        Returns:
            (DataFrame): A dataframe of Xvars and names/values stored in the object.
        """
        self._typ_set() if not hasattr(self, "estimator") else None
        df = pd.concat(
            [
                pd.concat(
                    [
                        pd.DataFrame({"DATE": self.current_dates.to_list()}),
                        pd.DataFrame({k: v.to_list() for k, v in self.current_xreg.items()}),
                    ],
                    axis=1,
                ),
                pd.concat(
                    [
                        pd.DataFrame({"DATE": self.future_dates.to_list()}), 
                        pd.DataFrame({k: v[:] for k, v in self.future_xreg.items()})
                    ],
                    axis=1,
                ),
            ],
        )
        return df.dropna() if dropna else df

    def export_fitted_vals(self, model:EvaluatedModel) -> pd.DataFrame:
        """ Exports a single dataframe with dates, fitted values, actuals, and residuals for one model.

        Args:
            model (str):
                The model nickname.

        Returns:
            (DataFrame): A dataframe with dates, fitted values, actuals, and residuals.
        """
        df = pd.DataFrame(
            {
                "DATE": self.current_dates.values[-len(self.history[model]["FittedVals"]) :],
                "Actuals": self.y.values[-len(self.history[model]["FittedVals"]) :],
                "FittedVals": self.history[model]["FittedVals"],
            }
        )
        df["Residuals"] = df["Actuals"] - df["FittedVals"]
        return df

    def round(self,decimals:int=0) -> Self:
        """ Rounds the values saved to `Forecaster.y`.

        Args:
            decimals (int): The number of digits to round off to. Passed to `np.round(decimals)`.

        Returns: 
             Self
        """

        self.y = np.round(self.y,decimals=decimals)
        return self