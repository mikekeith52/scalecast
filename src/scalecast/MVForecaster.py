from __future__ import annotations
from .cfg import COLORS, SERIES_COLORS, IGNORE_AS_HYPERPARAMS, MV_ESTIMATORS
from ._utils import _developer_utils, _tune_test_forecast
from ._Forecaster_parent import Forecaster_parent, ForecastError
from .types import (
    PositiveInt, 
    NonNegativeInt, 
    ConfInterval, 
    ModelValues,
    DetermineBestBy,
    EvaluatedModel,
    AvailableModel,
    ExportOptions,
    SeriesName,
    SeriesValues,
)
from .classes import AR, EvaluatedMetric
from typing import Optional, Literal, Any, Self, TYPE_CHECKING
import warnings
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import seaborn as sns
from itertools import cycle
import warnings
import datetime
if TYPE_CHECKING:
    from .Forecaster import Forecaster

class MVForecaster(Forecaster_parent):
    """ MVForecaster is a class for forecasting multiple series at once with the same models and hyperparameters.

    Args:
        *fs (Forecaster): Forecaster objects
        names (list-like): Optional. An array with the same number of elements as *fs that can be used to map to each series.
            Ex. if names == ['UTUR','UNRATE'], the user must now refer to the series with the selected names. 
            If specific names are not supplied, refer to the series with y1, y2, etc.
            The order the series are supplied will be maintained.
        not_same_len_action (str): One of 'trim', 'fail'. default 'trim'.
            What to do with series that are different lengths.
            'trim' will trim each series so that all dates line up.
        merge_Xvars (str): One of 'union', 'u', 'intersection', 'i'. default 'union'.
            How to combine Xvars in each object.
            'union' or 'u' combines all regressors from each object.
            'intersection' or 'i' combines only regressors that all objects have in common.
        merge_future_dates (str): One of 'longest', 'shortest'. Default 'longest'.
            Which future dates to use in the various series. This can be changed later.
        test_length (int or float): Default 0. The test length that all models will use to test all models out of sample.
            If float, must be between 0 and 1 and will be treated as a fractional split.
            By default, models will not be tested.
        validation_length (int): The size of the validation set. Default sets to 1.
        metrics (list[str]): Optional. List of metrics to evaluate every model. 
        optimize_on (str): The way to aggregate the derived metrics when optimizing models across all series. 
            This can be a function: 'mean', 'min', 'max', a custom function that takes a list of objects and returns an aggregate function (such as a weighted average) 
            or a series name. Custom functions and weighted averages can also be added later
            by calling mvf.set_optimize_on().
        cis (bool): Default False. Whether to evaluate probabilistic confidence intervals for every model evaluated.
            If setting to True, ensure you also set a test_length of at least 20 observations for 95% confidence intervals.
            See eval_cis() and set_cilevel() methods and docstrings for more information.
        carry_fit_models (bool): Default False.
            Whether to store the regression model for each fitted model in history.
            Setting this to False can save memory.
    """
    def __init__(
        self,
        *fs:'Forecaster',
        names:Optional[str]=None,
        not_same_len_action:Literal["trim","fail"]="trim",
        merge_Xvars:Literal["u","union","intersection","i"]="union",
        merge_future_dates:Literal["longest","shortest"]="longest",
        test_length:NonNegativeInt = 0,
        validation_length:NonNegativeInt=1,
        metrics:Optional[list[str]]=None,
        optimize_on:Literal['mean','min','max']|callable|SeriesName = 'mean',
        cis:bool = False,
        carry_fit_models:bool = False,
    ):
        super().__init__(
            y = fs[0].y, # placeholder -- will be overwritten
            test_length = test_length,
            cis = cis,
            validation_length=validation_length,
        )
        for f in fs:
            f._typ_set()
        if (
            len(set([len(f.current_dates) for f in fs])) > 1
            or len(set([min(f.current_dates) for f in fs])) > 1
        ):
            if not_same_len_action == "fail":
                raise ValueError("All series must be same length.")
            elif not_same_len_action == "trim":
                from .multiseries import line_up_dates
                line_up_dates(*fs)
            else:
                raise ValueError(f'not_same_len_action must be one of ("trim","fail"), got {not_same_len_action}.')

        if len(set([f.freq for f in fs])) > 1:
            raise ValueError("All date frequencies in passed Forecaster objects must be equal.")
        if len(fs) < 2:
            raise ValueError("Must pass at least two series.")

        self.grids_file = 'MVGrids'
        self.freq = fs[0].freq
        self.n_series = len(fs)
        self.estimators = MV_ESTIMATORS

        self.y = {}
        if names is None:
            names = [f'y{i+1}' for i in range(self.n_series)]
        self.names = names
        for i, f in enumerate(fs):
            if i == 0:
                self.current_dates = f.current_dates.copy().reset_index(drop=True)
            self.y[names[i]] = f.y.copy()
            if merge_Xvars in ("union", "u"):
                if i == 0:
                    self.current_xreg = {
                        k: v.copy().reset_index(drop=True)
                        for k, v in f.current_xreg.items()
                        if not isinstance(k,AR)
                    }
                    self.future_xreg = {
                        k: v[:]
                        for k, v in f.future_xreg.items()
                        if not isinstance(k,AR)
                    }
                else:
                    for k, v in f.current_xreg.items():
                        if not isinstance(k,AR):
                            self.current_xreg[k] = v.copy().reset_index(drop=True)
                            self.future_xreg[k] = f.future_xreg[k][:]
            elif merge_Xvars in ("intersection", "i"):
                if i == 0:
                    self.current_xreg = {
                        k: v.copy().reset_index(drop=True) for k,v in f.current_xreg.items() if not isinstance(k,AR)
                    }
                    self.future_xreg = {k: v[:] for k,v in f.future_xreg.items() if not isinstance(k,AR)}
                else:
                    f.drop_Xvars(*[k for k in f.current_xreg if k not in self.current_xreg or isinstance(k,AR)])

            else:
                raise ValueError(
                    f"merge_Xvars must be one of ('union','u','intersection','i'), got {merge_Xvars}"
                )

        self.optimizer_funcs = {
            "mean": np.mean,
            "min": np.min,
            "max": np.max,
        }
        self.set_optimize_on(optimize_on)
        future_dates_lengths = {i: len(f.future_dates) for i, f in enumerate(fs)}
        if merge_future_dates == "longest":
            self.future_dates = [
                fs[i].future_dates
                for i, v in future_dates_lengths.items()
                if v == max(future_dates_lengths.values())
            ][0]
        elif merge_future_dates == "shortest":
            self.future_dates = [
                fs[i].future_dates
                for i, v in future_dates_lengths.items()
                if v == min(future_dates_lengths.values())
            ][0]
        else:
            raise ValueError(f"merge_future_dates must be one of ('longest','shortest'), got {merge_future_dates}.")
        if metrics:
            self.set_metrics(metrics)
        self.carry_fit_models = carry_fit_models

    def __repr__(self):
        return """MVForecaster(
    DateStartActuals={}
    DateEndActuals={}
    Freq={}
    N_actuals={}
    N_series={}
    SeriesNames={}
    ForecastLength={}
    Xvars={}
    TestLength={}
    ValidationLength={}
    ValidationMetric={}
    ForecastsEvaluated={}
    CILevel={}
    CurrentEstimator={}
    OptimizeOn={}
    GridsFile={}
)""".format(
        self.current_dates.values[0].astype(str),
        self.current_dates.values[-1].astype(str),
        self.freq,
        self.n_actuals(),
        self.n_series,
        self.names,
        len(self.future_dates),
        list(self.current_xreg.keys()),
        self.test_length,
        self.validation_length,
        self.validation_metric,
        list(self.history.keys()),
        self.cilevel if self.cis is True else None,
        self.estimator,
        self.optimize_on,
        self.grids_file,
    )

    def add_optimizer_func(self, func:callable, called:Optional[str] = None) -> Self:
        """ Add an optimizer function that can be used to determine the best-performing model.
        This is in addition to the 'mean', 'min', and 'max' functions that are available by default.

        Args:
            func (Function): The function to add.
            called (str): Optional. How to refer to the function when calling `optimize_on()`.
                If left unspecified, will use the name of the function.

        Returns:
            Self

        >>> def weighted(x):
        >>>     # weighted average of first two series in the object
        >>>     return x[0]*.25 + x[1]*.75
        >>> mvf.add_optimizer_func(weighted)
        >>> mvf.set_optimize_on('weighted') # optimize on that function
        >>> mvf.set_estimator('mlr')
        >>> mvf.tune() # best model now chosen based on the weighted average function you added; series2 gets 3x the weight of series 1
        """
        called = self._called(func,called)
        self.optimizer_funcs[called] = func
        return self

    def _typ_set(self):
        """ Placeholder function.
        """
        return

    def add_signals(
        self,
        model_nicknames:list[EvaluatedModel], 
        series:SeriesName|Literal['all'] = 'all', 
        fill_strategy:Literal['actuals','bfill']|None = 'actuals', 
        train_only:bool = False
    ) -> Self:
        """ Adds the predictions from already-evaluated models as covariates that can be used for future evaluated models.
        The names of the added variables will all begin with "signal_" and end with the given model nickname folowed by the series name.

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

        Returns:
            Self

        >>> mvf.set_estimator('xgboost')
        >>> mvf.manual_forecast()
        >>> mvf.add_signals(model_nicknames = ['xgboost']) # adds regressors called 'signal_xgboost_{series1name}', ..., 'signal_xgboost_{seriesNname}'
        """
        series = self._parse_series(series)
        for m in model_nicknames:
            for s in series:
                fcst = self.history[m]['Forecast'][s][:]
                fvs = self.history[m]['FittedVals'][s][:]
                num_fvs = len(fvs)
                match fill_strategy:
                    case 'actuals':
                        pad = [np.nan if fill_strategy is None else fvs[0]] * (len(self.y[s]) - num_fvs)
                    case _:
                        pad = self.y[s].to_list()[:-num_fvs]

                self.current_xreg[f'signal_{m}_{s}'] = pd.Series(pad + fvs)
                
                if train_only:
                    tsp = self.history[m]['TestSetPredictions'][s][:]
                    self.current_xreg[f'signal_{m}_{s}'].iloc[-len(tsp):] = tsp
                
                self.future_xreg[f'signal_{m}_{s}'] = fcst
        
        return self

    def chop_from_front(self,n:NonNegativeInt,fcst_length:Optional[NonNegativeInt]=None) -> Self:
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

        >>> mvf.chop_from_front(10) # keeps all observations before the last 10
        """
        n = int(n)
        fcst_length = len(self.future_dates) if fcst_length is None else fcst_length
        self.y = {k:v.iloc[:-n] for k, v in self.y.items()}
        self.current_dates = self.current_dates.iloc[:-n]
        self.generate_future_dates(fcst_length)
        self.future_xreg = {
            k:(self.current_xreg[k].to_list()[-n:] + v[:max(0,(fcst_length-n))])[-fcst_length:]
            for k, v in self.future_xreg.items()
        }
        self.current_xreg = {k:v.iloc[:-n].reset_index(drop=True) for k, v in self.current_xreg.items()}
        return self

    def keep_smaller_history(self, n:NonNegativeInt) -> Self:
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
        _developer_utils.descriptive_assert(
            isinstance(n, int),
            ValueError,
            "n must be an int, datetime object, or str and there must be more than 2 observations to keep.",
        )
        _developer_utils.descriptive_assert(
            n > 2,
            ValueError,
            "n must be an int, datetime object, or str and there must be more than 2 observations to keep.",
        )
        self.y = {k: v.iloc[-n:] for k, v in self.y.items()}
        self.current_dates = self.current_dates.iloc[-n:]
        self.current_xreg = {k:v.iloc[-n:].reset_index(drop=True) for k, v in self.current_xreg.items()}
        return self

    def tune_test_forecast(
        self,
        models:list[AvailableModel],
        cross_validate:bool=False,
        dynamic_tuning:bool=False,
        dynamic_testing:bool=True,
        limit_grid_size:Optional[NonNegativeInt|ConfInterval]=None,
        min_grid_size:NonNegativeInt=1,
        suffix:Optional[str]=None,
        error:Literal['ignore','raise','warn']='raise',
        **cvkwargs:Any,
    ) -> Self:
        """ Iterates through a list of models, tunes them using grids in a grids file, and forecasts them.

        Args:
            models (list-like): The models to iterate through.
            cross_validate (bool): Default False.
                Whether to tune the model with cross validation. 
                If False, uses the validation slice of data to tune.
            dynamic_tuning (bool or int): Default False.
                Whether to dynamically tune the model or, if int, how many forecast steps to dynamically tune it.
            dynamic_testing (bool or int): Default True.
                Whether to dynamically/recursively test the forecast (meaning AR terms will be propagated with predicted values).
                If True, evaluates dynamically over the entire out-of-sample slice of data.
                If int, window evaluates over that many steps (2 for 2-step dynamic forecasting, 12 for 12-step, etc.).
                Setting this to False or 1 means faster performance, 
                but gives a less-good indication of how well the forecast will perform more than one period out.
                The model will skip testing if the test_length attribute is set to 0.
            limit_grid_size (int or float): Optional. Pass an argument here to limit each of the grids being read.
                See https://scalecast.readthedocs.io/en/latest/Forecaster/MVForecaster.html#src.scalecast.MVForecaster.MVForecaster.limit_grid_size.
            min_grid_size (int): Default 1. The smallest grid size to keep. Ignored if limit_grid_size is None.
            suffix (str): Optional. A suffix to add to each model as it is evaluated to differentiate them when called later. 
                If unspecified, each model can be called by its estimator name.
            error (str): One of 'ignore','raise','warn'. Default 'raise'.
                What to do with the error if a given model fails.
                'warn' logs a warning that the model could not be evaluated.
            **cvkwargs: Passed to the cross_validate() method.

        Returns:
            Self

        >>> models = ('mlr','mlp','lightgbm')
        >>> mvf.tune_test_forecast(models,dynamic_testing=False)
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
            **cvkwargs,
        )
        return self

    def set_optimize_on(self, how:str) -> Self:
        """ Choose how to determine best models by choosing which series should be optimized or the aggregate function to apply on the derived metrics across all series.
        This is the decision that will be used for optimizing model hyperparameters.

        Args:
            how (str): One of MVForecaster.optimizer_funcs, a series name, or a function. 
                Only one series name will be in mvf.optimizer_funcs at a given time.
                mvf.optimize_on is set to 'mean' when the object is initiated.
        
        Returns:
            Self
        """
        if callable(how):
            self.add_optimizer_func(how)
            self.optimize_on = how.__name__
            return
        elif how in self.names: # assume it's a series name
            if self.optimize_on == how:
                return
            elif np.any([o in self.names for o in self.optimizer_funcs]):
                to_pop = [o for o in self.optimizer_funcs if o in self.names]
                for p in to_pop:
                    self.optimizer_funcs.pop(p)
            self.series_to_optimize = self.names.index(how)
            self.add_optimizer_func(func=lambda x: self.series_to_optimize,called=how)
        elif how not in self.optimizer_funcs:
            raise ValueError(
                f'Value passed to how cannot be used: {how}. '
                f'Possible values are: {list(self.optimizer_funcs.keys())} or a function.'
            )

        self.optimize_on = how
        return self

    def _set_cis(self,*attrs,m,ci_range,preds):
        for i, attr in enumerate(attrs):
            self.history[m][attr] = {
                k:p + (ci_range[k] if i%2 == 0 else (ci_range[k]*-1))
                for k,p in preds.items()
            }

    def _bank_history(self, **kwargs:Any):
        """ places all relevant information from the last evaluated forecast into the history dictionary attribute
            **kwargs: passed from each model, depending on how that model uses Xvars, normalizer, and other args
        """
        call_me = self.call_me
        fitted_val_actuals = {k: (v.to_list()[-len(self.fitted_values[k]):]) for k, v in self.y.items()}
        
        self.history[call_me]['Estimator'] = self.estimator
        if hasattr(self.call_estimator,'Xvars'):
            self.history[call_me]['Xvars'] = self.call_estimator.Xvars
        else:
            self.history[call_me]['Xvars'] = None
        self.history[call_me]['HyperParams'] = {k: v for k, v in kwargs.items() if k not in IGNORE_AS_HYPERPARAMS}
        self.history[call_me]['Lags'] =  self.call_estimator.lags
        self.history[call_me]['Forecast'] = self.forecast
        self.history[call_me]['Observations'] = len(self.current_dates)
        self.history[call_me]['FittedVals'] = self.fitted_values
        self.history[call_me]['DynamicallyTested'] = self.dynamic_testing
        self.history[call_me]['CILevel'] = self.cilevel if self.cis else np.nan
        if self.carry_fit_models:
            self.history[call_me]['regr'] = self.trained_models
        for attr in ('TestSetPredictions','TestSetActuals'):
            if attr not in self.history[call_me]:
                self.history[call_me][attr] = {n:[] for n in self.names}

        if hasattr(self,'best_params'):
            if np.all([k in self.best_params for k in self.history[call_me]['HyperParams']]):
                self.history[call_me]['ValidationMetric'] = self.validation_metric
                self.history[call_me]['ValidationMetricValue'] = self.validation_metric_value
                self.history[call_me]['grid'] = self.grid
                self.history[call_me]['grid_evaluated'] = self.grid_evaluated

        for met in self.metrics:
            self.history[call_me]["InSample" + met.name.upper()] = {
                series: EvaluatedMetric(store=met,score=_developer_utils._return_na_if_len_zero(a, self.fitted_values[series], met.eval_func))
                for series, a in fitted_val_actuals.items()
            }

        if self.cis and self.history[call_me]['TestSetPredictions'][self.names[0]]:
            self._check_right_test_length_for_cis(self.cilevel)
            fcst = self.history[call_me]['Forecast']
            test_preds = self.history[call_me]['TestSetPredictions']
            test_actuals = self.history[call_me]['TestSetActuals']
            test_resids = {k:np.array(p) - np.array(test_actuals[k]) for k, p in test_preds.items()}
            ci_range = {k:np.percentile(np.abs(r), 100 * self.cilevel) for k,r in test_resids.items()}
            self._set_cis(
                "UpperCI",
                "LowerCI",
                m = call_me,
                ci_range = ci_range,
                preds = fcst,
            )
            self._set_cis(
                "TestSetUpperCI",
                "TestSetLowerCI",
                m = call_me,
                ci_range = ci_range,
                preds = test_preds,
            )

    def set_best_model(self, model:Optional[EvaluatedModel]=None, determine_best_by:Optional[DetermineBestBy]=None) -> Self:
        """ Sets the best model to be referenced as "best".
        One of model or determine_best_by parameters must be specified.

        Args:
            model (str): The model to set as the best.
                Must match the estimator name or call_me if that was used when evaluating the model.
            determine_best_by (str): One of MVForecaster.determine_best_by.
                If model is specified, this will be ignored.

        Returns:
            Self
        """
        if model is not None:
            if model in self.history.keys():
                self.best_model = model
            else:
                raise ValueError(f"Cannot find {model} in history.")
        else:
            self.best_model = self.order_fcsts(determine_best_by=determine_best_by)[0]
        return self

    def _parse_series(self, series:SeriesValues) -> list[SeriesName]:
        match series:
            case 'all':
                return list(self.y.keys())
            case str():
                return [series]
            case _:
                return list(series) 

    def _parse_models(self, models:ModelValues, put_best_on_top:bool) -> list[EvaluatedModel]:
        match models:
            case 'all':
                models = list(self.history.keys())
            case str():
                models = [models]
            case _:
                models = list(models)

        if put_best_on_top:
            models = ([self.best_model] if self.best_model in models else []) + [
                m for m in models if m != self.best_model
            ]
        return models

    def plot(
        self, 
        models:ModelValues="all", 
        series:SeriesValues="all", 
        put_best_on_top:bool=False, 
        ci:bool=False, 
        ax:Axes=None,
        figsize:tuple[int,int]=(12,6),
        colors:Optional[list[str]] = COLORS,
        series_colors:Optional[list[str]] = SERIES_COLORS,
    ) -> Axes:
        """ Plots all forecasts with the actuals, or just actuals if no forecasts have been evaluated or are selected.

        Args:
            models (list-like or str): Default 'all'.
               The forecasted models to plot.
               Name of the model, 'all', or list-like of model names.
               'top_' and None not supported.
            series (list-like or str): Default 'all'.
               The series to plot.
               Name of the series, 'all', or list-like of series names.
            put_best_on_top (bool): Only set to True if you have previously called set_best_model().
                If False, ignored.
            ci (bool): Default False.
                Whether to display the confidence intervals.
            ax (Axis): Optional. The existing axis to write the resulting figure to.
            figsize (tuple): Default (12,6). Size of the resulting figure. Ignored when ax is not None.
            colors (list[str]): Optional. The colors to use when drawing the forecasts.
            series_colors (list[str]): Optional. The colors to use when drawing the actual series.

        Returns:
            (Axis): The figure's axis.

        >>> mvf.plot() # plots all forecasts and all series
        >>> plt.show()
        """
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)

        colors = cycle(colors)
        series_colors = cycle(series_colors)

        series = self._parse_series(series)
        models = self._parse_models(models, put_best_on_top)

        k = 0
        for s in series:
            series_color = next(series_colors)
            sns.lineplot(
                x=self.current_dates.to_list(),
                y=self.y[s].to_list()[-len(self.current_dates) :],
                label=f"{s} actuals",
                ax=ax,
                color=series_color,
            )
            for m in models:
                color = next(colors)
                sns.lineplot(
                    x=self.future_dates.to_list(),
                    y=self.history[m]["Forecast"][s],
                    label=f"{s} {m}",
                    color=color,
                    ax=ax,
                )

                if ci:
                    try:
                        ax.fill_between(
                            x=self.future_dates.to_list(),
                            y1=self.history[m]["UpperCI"][s],
                            y2=self.history[m]["LowerCI"][s],
                            alpha=0.2,
                            color=color,
                            label="{} {} {:.0%} CI".format(
                                s, m, self.history[m]["CILevel"]
                            ),
                        )
                    except KeyError:
                        _developer_utils._warn_about_not_finding_cis(m)
                k += 1

        ax.legend()
        ax.set_xlabel("Date")
        ax.set_ylabel("Values")
        return ax

    def plot_test_set(
        self,
        models:ModelValues="all",
        series:SeriesValues="all",
        put_best_on_top:bool=False,
        include_train:bool|PositiveInt=True,
        ci:bool=False,
        ax:Axes=None,
        figsize:tuple[int,int]=(12,6),
        colors:Optional[list[str]] = COLORS,
        series_colors:Optional[list[str]] = SERIES_COLORS,
    ) -> Axes:
        """  Plots all test-set predictions with the actuals.

        Args:
            models (list-like or str): Default 'all'.
               The forecasted models to plot.
               Name of the model, 'all', or list-like of model names.
               'top_' and None not supported.
            series (list-like or str): Default 'all'.
               The series to plot.
               Name of the series ('y1...', 'series1...' or user-selected name), 'all', or list-like of series names.
            put_best_on_top (bool): Only set to True if you have previously called set_best_model().
                If False, ignored.
            include_train (bool or int): Default True.
                Use to zoom into training results.
                If True, plots the test results with the entire history in y.
                If False, matches y history to test results and only plots this.
                If int, plots that length of y to match to test results.
            ci (bool): Default False.
                Whether to display the confidence intervals.
            ax (Axis): Optional. The existing axis to write the resulting figure to.
            figsize (tuple): Default (12,6). Size of the resulting figure. Ignored when ax is not None.
            colors (list[str]): Optional. The colors to use when drawing the forecasts.
            series_colors (list[str]): Optional. The colors to use when drawing the actual series.

        Returns:
            (Axis): The figure's axis.

        >>> mvf.plot_test_set() # plots all test set predictions on all series
        >>> plt.show()
        """
        _developer_utils.descriptive_assert(
            self.test_length > 0,
            ForecastError,
            'plot_test_set() does not work when models were not tested (test_length set to 0).',
        )

        if ax is None:
            _, ax = plt.subplots(figsize=figsize)

        colors = cycle(colors)
        series_colors = cycle(series_colors)

        series = self._parse_series(series)
        models = self._parse_models(models, put_best_on_top)

        match include_train:
            case True:
                include_train = self.n_actuals()
            case False:
                include_train = self.test_length

        k = 0
        for s in series:
            series_color = next(series_colors)
            y = self.y[s].to_list()[-len(self.current_dates) :]
            sns.lineplot(
                x=self.current_dates.to_list()[-include_train:],
                y=y[-include_train:],
                label=f"{s} actual",
                ax=ax,
                color=series_color,
            )
            for m in models:
                color = next(colors)
                sns.lineplot(
                    x=self.current_dates.to_list()[-self.test_length :],
                    y=self.history[m]["TestSetPredictions"][s],
                    label=f"{s} {m}",
                    color=color,
                    linestyle="--",
                    alpha=0.7,
                    ax=ax,
                )

                if ci:
                    try:
                        plt.fill_between(
                            x=self.current_dates.to_list()[-self.test_length :],
                            y1=self.history[m]["TestSetUpperCI"][s],
                            y2=self.history[m]["TestSetLowerCI"][s],
                            alpha=0.2,
                            color=color,
                            label="{} {} {:.0%} CI".format(
                                s, m, self.history[m]["CILevel"]
                            ),
                        )
                    except KeyError:
                        _developer_utils._warn_about_not_finding_cis(m)
                k += 1

        ax.legend()
        ax.set_xlabel("Date")
        ax.set_ylabel("Values")
        return ax

    def plot_fitted(
        self, 
        models:ModelValues="all", 
        series:SeriesValues="all", 
        ax:Axes=None,
        figsize:tuple[int,int]=(12,6),
        colors:Optional[list[str]] = COLORS,
        series_colors:Optional[list[str]] = SERIES_COLORS,
    ) -> Axes:
        """ Plots fitted values with the actuals.

        Args:
            models (list-like or str): Default 'all'.
               The forecasted models to plot.
               Name of the model, 'all', or list-like of model names.
               'top_' and None not supported.
            series (list-like or str): Default 'all'.
               The series to plot.
               Name of the series ('y1...', 'series1...' or user-selected name), 'all', or list-like of series names.
            ax (Axis): Optional. The existing axis to write the resulting figure to.
            figsize (tuple): Default (12,6). Size of the resulting figure. Ignored when ax is not None.
            colors (list[str]): Optional. The colors to use when drawing the forecasts.
            series_colors (list[str]): Optional. The colors to use when drawing the actual series.

        Returns:
            (Axis): The figure's axis.

        >>> mvf.plot_fitted() # plots all fitted values on all series
        >>> plt.show()
        """
        series = self._parse_series(series)
        models = self._parse_models(models, False)
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)

        colors = cycle(colors)
        series_colors = cycle(series_colors)

        k = 0
        for s in series:
            series_color = next(series_colors)
            act = self.y[s].to_list()
            sns.lineplot(
                x=self.current_dates.to_list(),
                y=act[-len(self.current_dates):],
                label=f"{s} actual",
                ax=ax,
                color=series_color,
            )
            for m in models:
                color = next(colors)
                fvs = (self.history[m]["FittedVals"][s])
                sns.lineplot(
                    x=self.current_dates.to_list()[-len(fvs) :],
                    y=fvs,
                    label=f"{s} {m}",
                    linestyle="--",
                    alpha=0.7,
                    color=color,
                    ax=ax,
                )
                k += 1

    def export(
        self,
        dfs:ExportOptions|list[ExportOptions]=[
            "model_summaries",
            "lvl_test_set_predictions",
            "lvl_fcsts",
        ],
        models:ModelValues="all",
        series:SeriesValues="all",
        cis:bool=False,
        to_excel:bool=False,
        out_path:os.PathLike=Path.cwd(),
        excel_name:str="results.xlsx",
    ) -> pd.DataFrame|dict[str,pd.DataFrame]:
        """ Exports 1-all of 3 pandas dataframes. Can write to excel with each dataframe on a separate sheet.
        Will return either a dictionary with dataframes as values (df str arguments as keys) or a single dataframe if only one df is specified.

        Args:
            dfs (list-like or str): Default 
                ['model_summaries', 'lvl_test_set_predictions', 'lvl_fcsts'].
                A list or name of the specific dataframe(s) you want returned and/or written to excel.
                Must be one of or multiple of the elements in default.
                Exporting test set predictions only works if all exported models were tested using the same test length.
            models (list-like or str): Default 'all'.
               The forecasted models to plot.
               Name of the model, 'all', or list-like of model names.
               'top_' and None not supported.
            series (list-like or str): Default 'all'.
               The series to plot.
               Name of the series ('y1...', 'series1...' or user-selected name), 'all', or list-like of series names.
            to_excel (bool): Default False.
                Whether to save to excel.
            cis (bool): Default False.
                Whether to export confidence intervals for models in 
                "all_fcsts", "test_set_predictions", "lvl_test_set_predictions", "lvl_fcsts"
                dataframes.
            out_path (str): Default './'.
                The path to save the excel file to (ignored when `to_excel=False`).
            excel_name (str): Default 'results.xlsx'.
                The name to call the excel file (ignored when `to_excel=False`).

        Returns:
            (DataFrame or Dict[str,DataFrame]): Either a single pandas dataframe if one element passed to dfs 
            or a dictionary where the keys match what was passed to dfs and the values are dataframes. 

        >>> results = mvf.export(dfs=['model_summaries','lvl_fcsts'],to_excel=True) # returns a dict
        >>> model_summaries = results['model_summaries'] # returns a dataframe
        >>> lvl_fcsts = results['lvl_fcsts'] # returns a dataframe
        >>> ts_preds = mvf.export('test_set_predictions') # returns a dataframe
        """
        _developer_utils.descriptive_assert(
            isinstance(cis, bool),
            "ValueError",
            f"argument passed to cis must be a bool type, not {type(cis)}",
        )

        match dfs:
            case str():
                dfs = [dfs]
            case obj if hasattr(obj, "__len__") and not obj:
                raise ValueError("No values passed to the dfs argument.")
            case _:
                dfs = list(dfs)

        _dfs_ = ["model_summaries","lvl_test_set_predictions","lvl_fcsts"]
        _bad_dfs_ = [i for i in dfs if i not in _dfs_]
        if _bad_dfs_:
            raise ValueError(f"Values passed to the dfs list must be in {_dfs_}, not {_bad_dfs_}")

        series = self._parse_series(series)
        models = self._parse_models(models, hasattr(self, "best_model"))
        output = {}
        if "model_summaries" in dfs:
            cols = [
                "Estimator",
                "Xvars",
                "HyperParams",
                "Lags",
                "Observations",
                "DynamicallyTested",
                "TestSetLength",
                "ValidationMetric",
                "ValidationMetricValue",
                "OptimizedOn",
                "MetricOptimized",
                "best_model",
            ]
            model_summaries = []
            for s in series:
                for m in models:
                    model_summary_sm = pd.DataFrame({"Series": [s], "ModelNickname": [m]})
                    for c in cols:
                        if c in self.history[m]:
                            attr = self.history[m][c]
                            match attr:
                                case dict():
                                    if c in ['HyperParams','Lags']:
                                        model_summary_sm[c] = [attr]
                                    else:
                                        model_summary_sm[c] = [attr.get(s,pd.NA)]
                                case _:
                                    model_summary_sm[c] = [attr]
                        elif c == "OptimizedOn" and hasattr(self, "best_model"):
                            if self.optimize_on in self.optimizer_funcs:
                                model_summary_sm["OptimizedOn"] = [self.optimize_on]
                            else:
                                series, label = self._parse_series(self.optimize_on)
                                model_summary_sm["OptimizedOn"] = [label]
                            if hasattr(self, "optimize_metric"):
                                model_summary_sm["MetricOptimized"] = self.optimize_metric
                            model_summary_sm["best_model"] = m == self.best_model
                        else:
                            model_summary_sm[c] = pd.NA
                    
                    for c in self.determine_best_by:
                        if c in self.history[m]:
                            match self.history[m][c]:
                                case dict():
                                    model_summary_sm[c] = self.history[m][c][s].score

                    model_summaries.append(model_summary_sm)
            model_summaries = pd.concat(model_summaries)
            output["model_summaries"] = model_summaries
        
        if "lvl_fcsts" in dfs:
            df = pd.DataFrame({"DATE": self.future_dates.to_list()})
            for s in series:
                for m in models:
                    df[f"{s}_{m}_lvl_fcst"] = self.history[m]["Forecast"][s][:]
                    if cis:
                        try:
                            df[f"{s}_{m}_lvl_fcst_upper"] = self.history[m]["UpperCI"][s][:]
                            df[f"{s}_{m}_lvl_fcst_lower"] = self.history[m]["LowerCI"][s][:]
                        except KeyError:
                            _developer_utils._warn_about_not_finding_cis(m)
            output["lvl_fcsts"] = df
        
        if "lvl_test_set_predictions" in dfs:
            if self.test_length > 0:
                df = pd.DataFrame({"DATE": self.current_dates.to_list()[-self.test_length :]})
                
                i = 0
                for s in series:
                    df[f"{s}_actuals"] = self.y[s].to_list()[-self.test_length :]
                    for m in models:
                        df[f"{s}_{m}_lvl_ts"] = self.history[m]["TestSetPredictions"][s][:]
                        if cis:
                            try:
                                df[f"{s}_{m}_lvl_ts_upper"] = self.history[m]["TestSetUpperCI"][s][:]
                                df[f"{s}_{m}_lvl_ts_lower"] = self.history[m]["TestSetLowerCI"][s][:]
                            except KeyError:
                                _developer_utils._warn_about_not_finding_cis(m)
                    i += 1
                output["lvl_test_set_predictions"] = df
            else:
                output["lvl_test_set_predictions"] = pd.DataFrame()
        if to_excel:
            out_path = Path(out_path)
            with pd.ExcelWriter(out_path/excel_name, engine="openpyxl") as writer:
                for k, df in output.items():
                    df.to_excel(writer, sheet_name=k, index=False)

        if len(output.keys()) == 1:
            return list(output.values())[0]
        else:
            return output

    def export_fitted_vals(self, series:SeriesValues="all", models:ModelValues="all") -> pd.DataFrame:
        """ Exports a dataframe of fitted values and actuals.

        Args:
            models (list-like or str): Default 'all'.
               Name of the model, 'all', or list-like of model names.
            series (list-like or str): Default 'all'.
               The series to plot.
               Name of the series ('y1...', 'series1...' or user-selected name), 'all', or list-like of series names.

        Returns:
            (DataFrame): The fitted values for all selected series and models.
        """
        series = self._parse_series(series)
        models = self._parse_models(models, False)
        dfd = {"DATE": self.current_dates.to_list()}
        length = len(dfd["DATE"])
        i = 0
        for s in series:
            dfd[f"{s}_actuals"] = (
                self.y[s].to_list()
            )
            for m in models:
                dfd[f"{s}_{m}_fvs"] = (
                    self.history[m]["FittedVals"][s][:]
                )
                length = min(length, len(dfd[f"{s}_{m}_fvs"]), len(dfd[f"{s}_actuals"]))
            i += 1
        return pd.DataFrame({c: v[-length:] for c, v in dfd.items()})

    def corr(self, train_only:bool=False, disp:Literal['matrix','heatmap']="matrix", df:Optional[pd.DataFrame]=None, **kwargs:Any) -> pd.DataFrame|Figure:
        """ Displays pearson correlation between all stored series in object.

        Args:
            train_only (bool): Default False.
                Whether to only include the training set (to avoid leakage).
            disp (str): One of {'matrix','heatmap'}. Default 'matrix'.
                How to display results.
            df (DataFrame): Optional. A dataframe to display correlation for.
                If specified, a dataframe will be created using all series with no lags.
                This argument exists to make the corr_lags() method work and it is not recommended to use it manually.
            **kwargs: Passed to seaborn.heatmap() function and are ignored if disp == 'matrix'.

        Returns:
            (DataFrame or Figure): The created dataframe if disp == 'matrix' else the heatmap fig.
        """
        if df is None:
            df = pd.DataFrame({k:v.values for k, v in self.y.items()})

        if train_only:
            df = df.iloc[: -self.test_length]

        corr = df.corr()
        if disp == "matrix":
            if len(kwargs) > 0:
                warnings.warn(
                    f"Keyword arguments: {kwargs} ignored when disp == 'matrix'.",
                    category=Warning,
                )
            return corr

        elif disp == "heatmap":
            _, ax = plt.subplots()
            sns.heatmap(corr, **kwargs, ax=ax)
            return ax

        else:
            raise ValueError(f'disp must be one of "matrix","heatmap", got {disp}')

    def corr_lags(self, y:Optional[SeriesName]=None, x:Optional[SeriesName]=None, lags:PositiveInt=1, **kwargs:Any) -> pd.DataFrame:
        """ Displays pearson correlation between one series and another series' lags.

        Args:
            y (str): The series to display as the dependent series. Default will take the first loaded series in the object.
            x (str): The series to display as the independent series. Default will take the second loaded series in the object.
            lags (int): Default 1. The number of lags to display in the independent series.
            **kwargs: Passed to the MVForecaster.corr() method. Will not pass the df arg.

        Returns:
            (DataFrame or Figure): The created dataframe if disp == 'matrix' else the heatmap fig.
        """
        y = self.names[0] if y is None else y
        x = self.names[0] if x is None else x
        series1 = self._parse_series(y)
        series2 = self._parse_series(x)

        df = pd.DataFrame({k:v.to_list() for k, v in self.y.items() if k in (series1[0],series2[0])})

        for i in range(lags):
            df[series2[0] + f"_lag{i+1}"] = df[series2[0]].shift(i + 1)

        df = df.dropna()
        return self.corr(df=df, **kwargs)
