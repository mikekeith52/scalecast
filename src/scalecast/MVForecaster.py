from .__init__ import __colors__, __series_colors__, __not_hyperparams__
from ._utils import _developer_utils
from ._Forecaster_parent import (
    Forecaster_parent,
    ForecastError,
    _tune_test_forecast,
)
import warnings
import typing
import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
import logging
from scipy import stats
import copy
import datetime

class MVForecaster(Forecaster_parent):
    def __init__(
        self,
        *fs,
        names=None,
        not_same_len_action="trim",
        merge_Xvars="union",
        merge_future_dates="longest",
        test_length = 0,
        optimize_on = 'mean',
        cis = False,
        metrics=['rmse','mape','mae','r2'],
        **kwargs,
    ):
        """ 
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
            optimize_on (str): The way to aggregate the derived metrics when optimizing models across all series. 
                This can be a function: 'mean', 'min', 'max', a custom function that takes a list of objects and returns an aggregate function (such as a weighted average) 
                or a series name. Custom functions and weighted averages can also be added later
                by calling mvf.set_optimize_on().
            cis (bool): Default False. Whether to evaluate probabilistic confidence intervals for every model evaluated.
                If setting to True, ensure you also set a test_length of at least 20 observations for 95% confidence intervals.
                See eval_cis() and set_cilevel() methods and docstrings for more information.
            metrics (list): Default ['rmse','mape','mae','r2']. The metrics to evaluate when validating
                and testing models. Each element must exist in utils.metrics and take only two arguments: a and f.
                Or the element should be a function that accepts two arguments that will be referenced later by its name.
                See https://scalecast.readthedocs.io/en/latest/Forecaster/Util.html#metrics.
                The first element of this list will be set as the default validation metric, but that can be changed.
                For each metric and model that is tested, the test-set and in-sample metrics will be evaluated and can be
                exported.
            **kwargs: Become attributes.
        """
        super().__init__(
            y = fs[0].y, # placeholder -- will be overwritten
            test_length = test_length,
            cis = cis,
            metrics = metrics,
            **kwargs,
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
                raise ValueError(
                    f'not_same_len_action must be one of ("trim","fail"), got {not_same_len_action}.'
                )

        if len(set([f.freq for f in fs])) > 1:
            raise ValueError("All date frequencies in passed Forecaster objects must be equal.")
        if len(fs) < 2:
            raise ValueError("Must pass at least two series.")

        self.grids_file = 'MVGrids'
        self.freq = fs[0].freq
        self.n_series = len(fs)
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
                        if not k.startswith("AR")
                    }
                    self.future_xreg = {
                        k: v[:]
                        for k, v in f.future_xreg.items()
                        if not k.startswith("AR")
                    }
                else:
                    for k, v in f.current_xreg.items():
                        if not k.startswith("AR"):
                            self.current_xreg[k] = v.copy().reset_index(drop=True)
                            self.future_xreg[k] = f.future_xreg[k][:]
            elif merge_Xvars in ("intersection", "i"):
                if i == 0:
                    self.current_xreg = {
                        k: v.copy().reset_index(drop=True) for k,v in f.current_xreg.items() if not k.startswith('AR')
                    }
                    self.future_xreg = {k: v[:] for k,v in f.future_xreg.items() if not k.startswith('AR')}
                else:
                    f.drop_Xvars(*[k for k in f.current_xreg if k not in self.current_xreg or k.startswith('AR')])

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
            raise ValueError(
                f"merge_future_dates must be one of ('longest','shortest'), got {merge_future_dates}."
            )
        self.set_test_length(test_length)
        for key, value in kwargs.items():
            setattr(self, key, value)

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
        len(self.current_dates),
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

    def add_optimizer_func(self, func, called = None):
        """ Add an optimizer function that can be used to determine the best-performing model.
        This is in addition to the 'mean', 'min', and 'max' functions that are available by default.

        Args:
            func (Function): The function to add.
            called (str): Optional. How to refer to the function when calling `optimize_on()`.
                If left unspecified, will use the name of the function.

        Returns:
            None

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

    def _typ_set(self):
        """ Placeholder function.
        """
        return

    def add_signals(
        self,
        model_nicknames, 
        series = 'all', 
        fill_strategy = 'actuals', 
        train_only = False
    ):
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
                pad = (
                    [
                        np.nan if fill_strategy is None
                        else fvs[0]
                    ] * (len(self.y[s]) - num_fvs) 
                    if fill_strategy != 'actuals' 
                    else self.y[s].to_list()[:-num_fvs]
                )
                self.current_xreg[f'signal_{m}_{s}'] = pd.Series(pad + fvs)
                if train_only:
                    tsp = self.history[m]['TestSetPredictions'][s][:]
                    self.current_xreg[f'signal_{m}_{s}'].iloc[-len(tsp):] = tsp
                self.future_xreg[f'signal_{m}_{s}'] = fcst

    def chop_from_front(self,n,fcst_length=None):
        """ Cuts the amount of y observations in the object from the front counting backwards.
        The current length of the forecast horizon will be maintained and all future regressors will be rewritten to the appropriate attributes.

        Args:
            n (int):
                The number of observations to cut from the front.
            fcst_length (int): Optional.
                The new length of the forecast length.
                By default, maintains the same forecast length currently in the object.

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
        self.current_xreg = {
            k:v.iloc[:-n].reset_index(drop=True)
            for k, v in self.current_xreg.items()
        }

    def keep_smaller_history(self, n):
        """ Cuts y observations in the object by counting back from the beginning.

        Args:
            n (int, str, or datetime.datetime):
                If int, the number of observations to keep.
                Otherwise, the last observation to keep.
                Must be parsable by pandas' Timestamp function.

        Returns:
            None

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

    def tune_test_forecast(
        self,
        models,
        cross_validate=False,
        dynamic_tuning=False,
        dynamic_testing=True,
        limit_grid_size=None,
        min_grid_size=1,
        suffix=None,
        error='raise',
        **cvkwargs,
    ):
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
            suffix (str): Optional. A suffix to add to each model as it is evaluate to differentiate them when called later. 
                If unspecified, each model can be called by its estimator name.
            error (str): One of 'ignore','raise','warn'. Default 'raise'.
                What to do with the error if a given model fails.
                'warn' logs a warning that the model could not be evaluated.
            **cvkwargs: Passed to the cross_validate() method.

        Returns:
            None

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

    def set_optimize_on(self, how):
        """ Choose how to determine best models by choosing which series should be optimized or the aggregate function to apply on the derived metrics across all series.
        This is the decision that will be used for optimizing model hyperparameters.

        Args:
            how (str): One of MVForecaster.optimizer_funcs, a series name, or a function. 
                Only one series name will be in mvf.optimizer_funcs at a given time.
                mvf.optimize_on is set to 'mean' when the object is initiated.
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
            globals()['series_to_optimize'] = self.names.index(how)
            self.add_optimizer_func(func=optimize_on_series,called=how)
        elif how not in self.optimizer_funcs:
            raise ValueError(
                f'Value passed to how cannot be used: {how}. '
                f'Possible values are: {list(self.optimizer_funcs.keys())} or a function.')

        self.optimize_on = how
    
    @_developer_utils.log_warnings
    def _forecast_sklearn(
        self, fcster, dynamic_testing = True, Xvars = 'all', normalizer="minmax", lags=1, **kwargs
    ):
        """ Runs the vector multivariate forecast start-to-finish. All Xvars stored in the object are used always. All sklearn estimators supported.
        See example1: https://scalecast-examples.readthedocs.io/en/latest/multivariate/multivariate.html
        and example2: https://scalecast-examples.readthedocs.io/en/latest/multivariate-beyond/mv.html.

        Args:
            fcster (str): One of `MVForecaster.estimators`. Scikit-learn estimators or APIs only. 
                Reads the estimator set to `set_estimator()` method.
            Xvars (str or list-like): Default 'all'. The exogenous/seasonal variables to use when forecasting.
                If None is passed, no Xvars will be used.
            dynamic_testing (bool or int): Default True.
                Whether to dynamically/recursively test the forecast (meaning AR terms will be propagated with predicted values).
                If True, evaluates dynamically over the entire out-of-sample slice of data.
                If int, window evaluates over that many steps (2 for 2-step dynamic forecasting, 12 for 12-step, etc.).
                Setting this to False or 1 means faster performance, 
                but gives a less-good indication of how well the forecast will perform more than one period out.
                The model will skip testing if the test_length attribute is set to 0.
            normalizer (str): Default 'minmax'.
                The scaling technique to apply to the input data and lags. One of `MVForecaster.normalizer`. 
            lags (int | list[int] | dict[str,(int | list[int])]): Default 1.
                The lags to add from each series to forecast with.
                Needs to use at least one lag for any sklearn model.
                Some models in the `scalecast.auxmodels` module require you to pass None or 0 to lags.
                If int, that many lags will be added for all series.
                If list, each element must be int types, and only those lags will be added for each series.
                If dict, the key must be a series name and the key is a list or int.
            **kwargs: Treated as model hyperparameters and passed to the applicable sklearn or other type of estimator.

        >>> mvf.set_estimator('gbt')
        >>> mvf.manual_forecast(lags=3) # adds three lags for each series
        >>> mvf.manual_forecast(lags=[1,3]) # first and third lags added for each series
        >>> mvf.manual_forecast(lags={'y1':2,'y2':3}) # 2 lags added for first series, 3 lags for second
        >>> mvf.manual_forecast(lags={'series1':[1,3],'series2':3}) # first and third lag for first series, 3 lags for second
        """

        def prepare_data(Xvars,lags):
            observed = np.array([self.current_xreg[x].values.copy() for x in Xvars]).T
            future = np.array([np.array(self.future_xreg[x][:]) for x in Xvars]).T

            ylen = len(self.y[self.names[0]])
            if len(observed.shape) > 1:
                no_other_xvars = False
                observed_future = np.concatenate([observed,future],axis=0)
            else:
                no_other_xvars = True
                observed_future = np.array([0]*(ylen + len(self.future_dates))).reshape(-1,1) # column of 0s
                observed = np.array([0]*ylen).reshape(-1,1)

            err_message = f'Cannot accept this lags argument: {lags}.'

            if nolags: # vecm
                observedy = np.array(
                    [v.to_list() for k, v in self.y.items()]
                ).T
                futurey = np.zeros((len(self.future_dates),self.n_series))
                if no_other_xvars:
                    observed = observedy
                    future = futurey
                else:
                    observed = np.concatenate([observedy,observed],axis=1)
                    future = np.concatenate([futurey,future],axis=1)
                return observed, future, None
            elif isinstance(lags, (float,int)):
                lags = int(lags)
                max_lag = lags
                lag_matrix = np.zeros((observed_future.shape[0],max_lag*self.n_series))
                pos = 0
                for i in range(self.n_series):
                    for j in range(lags):
                        Xvars.append('LAG_' + self.names[i] + "_" + str(j+1)) # UTUR_1 for first lag to keep track of position
                        lag_matrix[:,pos] = (
                            [np.nan] * (j+1)
                            + self.y[self.names[i]].to_list() 
                            + [np.nan] * (lag_matrix.shape[0] - ylen - (j+1)) # pad with nas
                        )[:lag_matrix.shape[0]] 
                        pos += 1
            elif isinstance(lags, dict):
                total_lags = 0
                for k, v in lags.items():
                    if hasattr(v,'__len__') and not isinstance(v,str):
                        total_lags += len(v)
                    elif isinstance(v,(float,int)):
                        total_lags += v
                    else:
                        raise ValueError(err_message)
                lag_matrix = np.zeros((observed_future.shape[0],total_lags))
                pos = 0
                max_lag = 1
                for k,v in lags.items():
                    if hasattr(v,'__len__') and not isinstance(v,str):
                        for i in v:
                            lag_matrix[:,pos] = (
                                [np.nan] * i
                                + self.y[k].to_list()
                                + [np.nan]
                                * (lag_matrix.shape[0] - ylen - i)
                            )[:lag_matrix.shape[0]] 
                            Xvars.append('LAG_' + k + "_" + str(i))
                            pos+=1
                        max_lag = max(max_lag,max(v))
                    elif isinstance(v,(float,int)):
                        for i in range(v):
                            lag_matrix[:,pos] = (
                                [np.nan] * (i+1)
                                + self.y[k].to_list()
                                + [np.nan]
                                * (lag_matrix.shape[0] - ylen - (i+1))
                            )[:lag_matrix.shape[0]] 
                            Xvars.append('LAG_' + k + "_" + str(i+1))
                            pos+=1
                        max_lag = max(max_lag,v)
            elif hasattr(lags,'__len__') and not isinstance(lags,str):
                lag_matrix = np.zeros((observed_future.shape[0],len(lags)*self.n_series))
                pos = 0
                max_lag = max(lags)
                for i in range(self.n_series):
                    for v in lags:
                        Xvars.append('LAG_' + self.names[i] + "_" + str(v))
                        lag_matrix[:,pos] = (
                            [np.nan] * v
                            + self.y[self.names[i]].to_list()
                            + [np.nan] * (lag_matrix.shape[0] - ylen - v)
                        )[:lag_matrix.shape[0]]
                        pos+=1
            else:
                raise ValueError(err_message)

            observed_future = np.concatenate([observed_future,lag_matrix],axis=1)
            start_col = 1 if no_other_xvars else 0
            future = observed_future[observed.shape[0]:,start_col:]
            observed = observed_future[max_lag:observed.shape[0],start_col:]
            return observed, future, Xvars

        def scale(scaler, X) -> np.ndarray:
            """ uses scaler parsed from _parse_normalizer() function to transform matrix passed to X.

            Args:
                scaler (MinMaxScaler, Normalizer, StandardScaler, PowerTransformer, or None): 
                    the fitted scaler or None type
                X (ndarray or DataFrame):
                    the matrix to transform

            Returns:
                (ndarray): The scaled x values.
            """
            if scaler is not None:
                return scaler.transform(X)
            else:
                return X

        def train(X, y, normalizer, **kwargs):
            self.scaler = self._parse_normalizer(X, normalizer)
            X = scale(self.scaler, X)
            regr = self.sklearn_imports[fcster](**kwargs)
            # below added for vecm model -- could be expanded for others as well
            extra_kws_map = {
                'dates':self.current_dates.values.copy(),
                'n_series':self.n_series,
            }
            if hasattr(regr,'_scalecast_set'):
                for att in regr._scalecast_set:
                    setattr(regr,att,extra_kws_map[att])
            
            regr.fit(X, y)
            return regr

        def evaluate(trained_models, future, dynamic_testing, Xvars = None):
            if nolags:
                future = scale(self.scaler, future)
                p = trained_models.predict(future)
                preds = {k:list(p[:,i]) for i, k in enumerate(self.y)}
            elif dynamic_testing is False:
                preds = {}
                future = scale(self.scaler, future)
                for series, regr in trained_models.items():
                    preds[series] = list(regr.predict(future))
            else:
                preds = {series: [] for series in trained_models.keys()}
                series = {
                    k:v.to_list()
                    for k,v in self.y.items()
                }
                for i in range(future.shape[0]):
                    fut = scale(self.scaler,future[i,:].reshape(1,-1))
                    for s, regr in trained_models.items():
                        snum = list(self.y.keys()).index(s)
                        pred = regr.predict(fut)[0]
                        preds[s].append(pred)
                        if (i < len(future) - 1):
                            if ((i+1) % dynamic_testing == 0) and (hasattr(self,'actuals')):
                                series[s].append(self.actuals[snum][i])
                            else:
                                series[s].append(pred)
                    if (i < len(future) - 1):
                        for x in Xvars:
                            if x.startswith('LAG_'):
                                idx = Xvars.index(x)
                                s = x.split('_')[1]
                                lagno = int(x.split('_')[-1])
                                future[i+1,idx] = series[s][-lagno]
            return preds

        steps = len(self.future_dates)
        dynamic_testing = (
            steps + 1 
            if dynamic_testing is True or not hasattr(self,'actuals') 
            else 1 if dynamic_testing is False 
            else dynamic_testing
        )
        Xvars = list(self.current_xreg.keys()) if Xvars == 'all' else [] if Xvars is None else list(Xvars)[:]
        self.Xvars = Xvars
        nolags = lags is None or not lags
        observed, future, Xvars = prepare_data(Xvars=Xvars[:],lags=lags)
        
        if nolags:
            trained_full = train(
                X=observed.copy(),
                y=None,
                normalizer=None,
                **kwargs,
            )
        else:
            trained_full = {}
            for k, v in self.y.items():
                trained_full[k] = train(
                    X=observed,
                    y=v.values[-observed.shape[0] :].copy(),
                    normalizer=normalizer,
                    **kwargs,
                )
        self.trained_models = trained_full # does not go to history
        if hasattr(self,'actuals'): # skip fitted values if only testing out-of-sample
            self.fitted_values = {k:[] for k in self.y}
        elif nolags:
            self.fitted_values = {
                k:list(trained_full.fittedvalues[:,i])
                for i, k in enumerate(self.y.keys())
            }
        else:
            self.fitted_values = evaluate(
                trained_models=trained_full, 
                future=observed, 
                dynamic_testing=False,
            )
        return evaluate(
            trained_models=trained_full, 
            future=future, 
            dynamic_testing=dynamic_testing,
            Xvars = Xvars,
        )

    def _parse_normalizer(self, X_train, normalizer):
        """ fits an appropriate scaler to training data that will then be applied to future data

        Args:
            X_train (DataFrame): The independent values.
            normalizer (str): One of MVForecaster.normalizer.
                if 'minmax', uses the MinMaxScaler from sklearn.preprocessing.
                if 'scale', uses the StandardScaler from sklearn.preprocessing.
                if 'normalize', uses the Normalizer from sklearn.preprocessing.
                if None, returns None.

        Returns:
            (scikit-learn preprecessing scaler/normalizer): The normalizer fitted on training data only.
        """
        _developer_utils.descriptive_assert(
            normalizer in self.normalizer,
            ValueError,
            f"normalizer must be one of {self.normalizer}, got {normalizer}.",
        )
        X_train = X_train if not hasattr(X_train,'values') else X_train.values
        if normalizer is None:
            return None

        scaler = scaler = self.normalizer[normalizer]()
        scaler.fit(X_train)
        return scaler

    def _set_cis(self,*attrs,m,ci_range,forecast,tspreds):
        for i, attr in enumerate(attrs):
            self.history[m][attr] = {
                k:p + (ci_range[k] if i%2 == 0 else (ci_range[k]*-1))
                for k,p in (
                    forecast.items() if i <= 1 else tspreds.items()
                )
            }

    def _bank_history(self, **kwargs):
        """ places all relevant information from the last evaluated forecast into the history dictionary attribute
            **kwargs: passed from each model, depending on how that model uses Xvars, normalizer, and other args
        """
        call_me = self.call_me
        fitted_val_actuals = {
            k: (
                self.y[k].to_list()[
                    -len(self.fitted_values[k]) :
                ] 
            ) for k in self.y
        }
        lags = kwargs["lags"] if "lags" in kwargs.keys() else 1
        
        self.history[call_me]['Estimator'] = self.estimator
        self.history[call_me]['Xvars'] = self.Xvars # self.Xvars
        self.history[call_me]['HyperParams'] = {k: v for k, v in kwargs.items() if k not in __not_hyperparams__}
        self.history[call_me]['Lags'] =  None if lags is None else int(lags) if not hasattr(lags,'__len__') else lags
        self.history[call_me]['Forecast'] = self.forecast
        self.history[call_me]['Observations'] = len(self.current_dates)
        self.history[call_me]['FittedVals'] = self.fitted_values
        self.history[call_me]['DynamicallyTested'] = self.dynamic_testing
        self.history[call_me]['CILevel'] = self.cilevel if self.cis else np.nan
        for attr in ('TestSetPredictions','TestSetActuals'):
            if attr not in self.history[call_me]:
                self.history[call_me][attr] = {n:[] for n in self.names}

        if hasattr(self,'best_params'):
            if np.all([k in self.best_params for k in self.history[call_me]['HyperParams']]):
                self.history[call_me]['ValidationMetric'] = self.validation_metric
                self.history[call_me]['ValidationMetricValue'] = self.validation_metric_value
                self.history[call_me]['grid'] = self.grid
                self.history[call_me]['grid_evaluated'] = self.grid_evaluated

        for met, func in self.metrics.items():
            self.history[call_me]["InSample" + met.upper()] = {
                series: _developer_utils._return_na_if_len_zero(a, self.fitted_values[series], func)
                for series, a in fitted_val_actuals.items()
            }

        if self.cis is True and len(self.history[call_me]['TestSetPredictions'][self.names[0]]) > 0:
            self._check_right_test_length_for_cis(self.cilevel)
            fcst = self.history[call_me]['Forecast']
            test_preds = self.history[call_me]['TestSetPredictions']
            test_actuals = self.history[call_me]['TestSetActuals']
            test_resids = {k:np.array(p) - np.array(test_actuals[k]) for k, p in test_preds.items()}
            #test_resids = {k:correct_residuals(r) for k, r in test_resids.items()}
            ci_range = {k:np.percentile(np.abs(r), 100 * self.cilevel) for k,r in test_resids.items()}
            self._set_cis(
                "UpperCI",
                "LowerCI",
                "TestSetUpperCI",
                "TestSetLowerCI",
                m = call_me,
                ci_range = ci_range,
                forecast = fcst,
                tspreds = test_preds,
            )

    def set_best_model(self, model=None, determine_best_by=None):
        """ Sets the best model to be referenced as "best".
        One of model or determine_best_by parameters must be specified.

        Args:
            model (str): The model to set as the best.
                Must match the estimator name or call_me if that was used when evaluating the model.
            determine_best_by (str): One of MVForecaster.determine_best_by.
                If model is specified, this will be ignored.

        Returns:
            None
        """
        if model is not None:
            if model in self.history.keys():
                self.best_model = model
            else:
                raise ValueError(f"Cannot find {model} in history.")
        else:
            _developer_utils.descriptive_assert(
                determine_best_by in self.determine_best_by,
                ValueError,
                f"determine_best_by must be one of {self.determine_best_by}, got {determine_best_by}.",
            )
            models_metrics = {m: v[determine_best_by] for m, v in self.history.items()}

            if determine_best_by != 'ValidationMetricValue':
                models_metrics = {
                    m: self.optimizer_funcs[self.optimize_on](list(v.values()))
                    for m, v in models_metrics.items()
                }

            x = [h[0] for h in Counter(models_metrics).most_common()]
            self.best_model = (
                x[0]
                if (determine_best_by.endswith("R2"))
                | (
                    (determine_best_by == "ValidationMetricValue")
                    & (self.validation_metric.upper() == "R2")
                )
                else x[-1]
            )
            self.optimize_metric = determine_best_by

    def _parse_series(self, series):
        if series == "all":
            series = list(self.y.keys())
        elif isinstance(series,str):
            series = [series]
        else:
            series = list(series)

        return series

    def _parse_models(self, models, put_best_on_top):
        if models == "all":
            models = list(self.history.keys())
        elif isinstance(models, str):
            models = [models]
        else:
            models = list(models)
        if put_best_on_top:
            models = ([self.best_model] if self.best_model in models else []) + [
                m for m in models if m != self.best_model
            ]
        return models

    def plot(
        self, 
        models="all", 
        series="all", 
        put_best_on_top=False, 
        ci=False, 
        ax=None,
        figsize=(12,6),
    ):
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


        Returns:
            (Axis): The figure's axis.

        >>> mvf.plot() # plots all forecasts and all series
        >>> plt.show()
        """
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)

        series = self._parse_series(series)
        models = self._parse_models(models, put_best_on_top)

        k = 0
        for i, s in enumerate(series):
            sns.lineplot(
                x=self.current_dates.to_list(),
                y=self.y[s].to_list()[-len(self.current_dates) :],
                label=f"{s} actuals",
                ax=ax,
                color=__series_colors__[i],
            )
            for m in models:
                sns.lineplot(
                    x=self.future_dates.to_list(),
                    y=self.history[m]["Forecast"][s],
                    label=f"{s} {m}",
                    color=__colors__[k],
                    ax=ax,
                )

                if ci:
                    try:
                        ax.fill_between(
                            x=self.future_dates.to_list(),
                            y1=self.history[m]["UpperCI"][s],
                            y2=self.history[m]["LowerCI"][s],
                            alpha=0.2,
                            color=__colors__[k],
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
        models="all",
        series="all",
        put_best_on_top=False,
        include_train=True,
        ci=False,
        ax=None,
        figsize=(12,6),
    ):
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

        
        series = self._parse_series(series)
        models = self._parse_models(models, put_best_on_top)
        include_train = (
            len(self.current_dates)
            if include_train is True
            else self.test_length
            if include_train is False
            else include_train
        )

        k = 0
        for i, s in enumerate(series):
            y = self.y[s].to_list()[-len(self.current_dates) :]
            sns.lineplot(
                x=self.current_dates.to_list()[-include_train:],
                y=y[-include_train:],
                label=f"{s} actual",
                ax=ax,
                color=__series_colors__[i],
            )
            for m in models:
                sns.lineplot(
                    x=self.current_dates.to_list()[-self.test_length :],
                    y=self.history[m]["TestSetPredictions"][s],
                    label=f"{s} {m}",
                    color=__colors__[k],
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
                            color=__colors__[k],
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
        models="all", 
        series="all", 
        ax=None,
        figsize=(12,6),
    ):
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


        Returns:
            (Axis): The figure's axis.

        >>> mvf.plot_fitted() # plots all fitted values on all series
        >>> plt.show()
        """
        series = self._parse_series(series)
        models = self._parse_models(models, False)
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)

        k = 0
        for i, s in enumerate(series):
            act = self.y[s].to_list()
            sns.lineplot(
                x=self.current_dates.to_list(),
                y=act[-len(self.current_dates):],
                label=f"{s} actual",
                ax=ax,
                color=__series_colors__[i],
            )
            for m in models:
                fvs = (self.history[m]["FittedVals"][s])
                sns.lineplot(
                    x=self.current_dates.to_list()[-len(fvs) :],
                    y=fvs,
                    label=f"{s} {m}",
                    linestyle="--",
                    alpha=0.7,
                    color=__colors__[k],
                    ax=ax,
                )
                k += 1

    def export(
        self,
        dfs=[
            "model_summaries",
            "lvl_test_set_predictions",
            "lvl_fcsts",
        ],
        models="all",
        series="all",
        cis=False,
        to_excel=False,
        out_path="./",
        excel_name="results.xlsx",
    ):
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
        if isinstance(dfs, str):
            dfs = [dfs]
        else:
            dfs = list(dfs)
        if len(dfs) == 0:
            raise ValueError("no dfs passed to dfs")
        _dfs_ = [
            "all_fcsts",
            "model_summaries",
            "test_set_predictions",
            "lvl_test_set_predictions",
            "lvl_fcsts",
        ]
        _bad_dfs_ = [i for i in dfs if i not in _dfs_]
        if len(_bad_dfs_) > 0:
            raise ValueError(
                f"values passed to the dfs list must be in {_dfs_}, not {_bad_dfs_}"
            )
        series = self._parse_series(series)
        models = self._parse_models(models, hasattr(self, "best_model"))
        output = {}
        if "model_summaries" in dfs:
            cols1 = [
                "Series",
                "ModelNickname",
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
            model_summaries = pd.DataFrame()
            for s in series:
                for m in models:
                    model_summary_sm = pd.DataFrame(
                        {"Series": [s], "ModelNickname": [m]}
                    )
                    cols = cols1  + [
                        k for k in self.history[m] if (
                            (
                                k.startswith('TestSet') & (k not in ( 
                                    'TestSetPredictions',
                                    'TestSetActuals',
                                    'TestSetLowerCI',
                                    'TestSetUpperCI',
                                )
                            ))
                            | k.startswith('InSample')
                        )
                    ]
                    for c in cols:
                        if c not in (
                            "Series",
                            "ModelNickname",
                            "LastTestSetPrediction",
                            "LastTestSetActual",
                            "OptimizedOn",
                            "MetricOptimized",
                            "best_model",
                        ):
                            attr = self.history[m][c] if c in self.history[m] else np.nan
                            if not isinstance(attr, dict) or c in (
                                "HyperParams",
                                "Lags",
                            ):
                                model_summary_sm[c] = [attr]
                            else:
                                try:
                                    model_summary_sm[c] = [attr[s]]
                                except KeyError:
                                    model_summary_sm[c] = np.nan
                        elif c == "OptimizedOn" and hasattr(self, "best_model"):
                            if self.optimize_on in self.optimizer_funcs:
                                model_summary_sm["OptimizedOn"] = [self.optimize_on]
                            else:
                                series, label = self._parse_series(self.optimize_on)
                                model_summary_sm["OptimizedOn"] = [label]
                            if hasattr(self, "optimize_metric"):
                                model_summary_sm[
                                    "MetricOptimized"
                                ] = self.optimize_metric
                            model_summary_sm["best_model"] = m == self.best_model
                    model_summaries = pd.concat(
                        [model_summaries, model_summary_sm], ignore_index=True
                    )
            output["model_summaries"] = model_summaries
        if "lvl_fcsts" in dfs:
            df = pd.DataFrame({"DATE": self.future_dates.to_list()})
            for s in series:
                for m in models:
                    df[f"{s}_{m}_lvl_fcst"] = self.history[m]["Forecast"][s][:]
                    if cis:
                        try:
                            df[f"{s}_{m}_lvl_fcst_upper"] = self.history[m]["UpperCI"][
                                s
                            ][:]
                            df[f"{s}_{m}_lvl_fcst_lower"] = self.history[m]["LowerCI"][
                                s
                            ][:]
                        except KeyError:
                            _developer_utils._warn_about_not_finding_cis(m)
            output["lvl_fcsts"] = df
        if "lvl_test_set_predictions" in dfs:
            if self.test_length > 0:
                df = pd.DataFrame(
                    {"DATE": self.current_dates.to_list()[-self.test_length :]}
                )
                i = 0
                for s in series:
                    df[f"{s}_actuals"] = self.y[s].to_list()[-self.test_length :]
                    for m in models:
                        df[f"{s}_{m}_lvl_ts"] = self.history[m]["TestSetPredictions"][s][:]
                        if cis:
                            try:
                                df[f"{s}_{m}_lvl_ts_upper"] = self.history[m]["TestSetUpperCI"][
                                    s
                                ][:]
                                df[f"{s}_{m}_lvl_ts_lower"] = self.history[m]["TestSetLowerCI"][
                                    s
                                ][:]
                            except KeyError:
                                _developer_utils._warn_about_not_finding_cis(m)
                    i += 1
                output["lvl_test_set_predictions"] = df
            else:
                output["lvl_test_set_predictions"] = pd.DataFrame()
        if to_excel:
            with pd.ExcelWriter(
                os.path.join(out_path, excel_name), engine="openpyxl"
            ) as writer:
                for k, df in output.items():
                    df.to_excel(writer, sheet_name=k, index=False)

        if len(output.keys()) == 1:
            return list(output.values())[0]
        else:
            return output

    def export_fitted_vals(self, series="all", models="all"):
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

    def corr(self, train_only=False, disp="matrix", df=None, **kwargs):
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
            series = self._parse_series("all")
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

    def corr_lags(self, y=None, x=None, lags=1, **kwargs):
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

def optimize_on_series(x):
    return x[globals()['series_to_optimize']]