from .__init__ import __colors__, __series_colors__
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

class MVForecaster(Forecaster_parent):
    def __init__(
        self,
        *fs,
        not_same_len_action="trim",
        merge_Xvars="union",
        merge_future_dates="longest",
        test_length = 0,
        cis = False,
        names=None,
        metrics=['rmse','mape','mae','r2'],
        **kwargs,
    ):
        """ 
        Args:
            *fs (Forecaster): Forecaster objects
            not_same_len_action (str): One of 'trim', 'fail'. default 'trim'.
                What to do with series that are different lengths.
                'trim' will trim based on the most recent first date in each series.
                If the various series have different end dates, this option will still fail the initialization.
            merge_Xvars (str): One of 'union', 'u', 'intersection', 'i'. default 'union'.
                How to combine Xvars in each object.
                'union' or 'u' combines all regressors from each object.
                'intersection' or 'i' combines only regressors that all objects have in common.
            merge_future_dates (str): One of 'longest', 'shortest'. Default 'longest'.
                Which future dates to use in the various series.
            test_length (int or float): Default 0. The test length that all models will use to test all models out of sample.
                If float, must be between 0 and 1 and will be treated as a fractional split.
                By default, models will not be tested.
            cis (bool): Default False. Whether to evaluate probabilistic confidence intervals for every model evaluated.
                If setting to True, ensure you also set a test_length of at least 20 observations for 95% confidence intervals.
                See eval_cis() and set_cilevel() methods and docstrings for more information.
            names (list-like): Optional. An array with the same number of elements as *fs that can be used to map to each series.
                Ex. if names == ['UTUR','UNRATE'], the user can now refer to series1 and series2 with the user-selected names later. 
                If specific names are not supplied, refer to the series with series1 and series2 or y1 and y2.
                'series...' and 'y...' notation to refer to series are used interchangeably throughout the object.
                Series can always be referred to with 'series...' and 'y...' notation, even if user-selected names are provided.
                The order the series are supplied will be maintained.
            metrics (list): Default ['rmse','mape','mae','r2']. The metrics to evaluate when validating
                and testing models. Each element must exist in utils.metrics and take only two arguments: a and f.
                See https://scalecast.readthedocs.io/en/latest/Forecaster/Util.html#metrics.
                The first element of this list will be set as the default validation metric, but that can be changed.
                For each metric and model that is tested, the test-set and in-sample metrics will be evaluated and can be
                exported. Level test-set and in-sample metrics are also currently available, but will be removed in a future version.
            **kwargs: Become attributes.
        """
        super().__init__(
            y = fs[0].y,
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
            _developer_utils.descriptive_assert(
                len(set([max(f.current_dates) for f in fs])) == 1,
                ForecastError,
                "Series cannot have different end dates.",
            )
            if not_same_len_action == "fail":
                raise ValueError("All series must be same length.")
            elif not_same_len_action == "trim":
                from .multiseries import keep_smallest_first_date
                keep_smallest_first_date(*fs)
            else:
                raise ValueError(
                    f'not_same_len_action must be one of ("trim","fail"), got {not_same_len_action}.'
                )
        if len(set([min(f.current_dates) for f in fs])) > 1:
            raise ValueError("All obs must begin in same time period.")
        if len(set([f.freq for f in fs])) > 1:
            raise ValueError("All date frequencies must be equal.")
        if len(fs) < 2:
            raise ValueError("Must pass at least two series.")

        self.optimizer_funcs = {
            "mean": np.mean,
            "min": np.min,
            "max": np.max,
        }
        self.optimize_on = list(self.optimizer_funcs.keys())[0]
        self.grids_file = 'MVGrids'
        self.freq = fs[0].freq
        self.n_series = len(fs)
        for i, f in enumerate(fs):
            setattr(
                self,
                f"series{i+1}",
                {
                    "y": f.y.copy().reset_index(drop=True),
                    "levely": f.levely.copy(),
                    "integration": f.integration,
                    "init_dates": f.init_dates,
                },
            )
            if i == 0:
                self.current_dates = f.current_dates.copy().reset_index(drop=True)
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

        self.integration = {
            f"y{i+1}": getattr(self, f"series{i+1}")["integration"]
            for i in range(self.n_series)
        }
        self.set_test_length(test_length)
        for key, value in kwargs.items():
            setattr(self, key, value)

        if (
            names is not None
        ):  # checking for these objects is how we know whether user supplied names later
            names = list(names)
            self.name_series_map = {
                names[i]: [f"series{i+1}", f"y{i+1}"] for i in range(self.n_series)
            }
            self.y_name_map = {f"y{i+1}": names[i] for i in range(self.n_series)}
            self.series_name_map = {
                f"series{i+1}": names[i] for i in range(self.n_series)
            }

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
        [f"series{i+1}" for i in range(self.n_series)]
        if not hasattr(self, "name_series_map")
        else list(self.name_series_map.keys()),
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

    def add_optimizer_func(self, func, called):
        """ Add an optimizer function that can be used to determine the best-performing model.
        This is in addition to the 'mean', 'min', and 'max' functions that are available by default.

        Args:
            func (Function): The function to add.
            called (str): How to refer to the function when calling `optimize_on()`.

        Returns:
            None

        >>> mvf = MVForecaster(...)
        >>> mvf.add_optimizer_func(lambda x: x[0]*.25 + x[1]*.75,'weighted') # adds a weighted average of first two series in the object
        >>> mvf.set_optimize_on('weighted')
        >>> mvf.set_estimator('mlr')
        >>> mvf.tune() # best model now chosen based on the weighted average function you added; series2 gets 3x the weight of series 1
        """
        self.optimizer_funcs[called] = func

    def _typ_set(self):
        """ Placeholder function.
        """
        return

    def tune(self, dynamic_tuning=False, cv=False):
        """ Tunes the specified estimator using an ingested grid (ingests a grid from a grids file with same name as the estimator by default).
        Any parameters that can be passed as arguments to manual_forecast() can be tuned with this process.
        The chosen parameters are stored in the best_params attribute.
        The full validation grid is stored in grid_evaluated.

        Args:
            dynamic_tuning (bool or int): Default False.
                Whether to dynamically/recursively tune the forecast using the series lags.
                Setting this to False means faster performance, but gives a less-good indication of how well the forecast will perform more than 1 period out.
            cv (bool): Default False.
                Whether the tune is part of a larger cross-validation process.
                This does not need to specified by the user and should be kept False when calling `tune()`.

        Returns:
            None

        >>> mvf.set_estimator('xgboost')
        >>> mvf.tune()
        >>> mvf.auto_forecast()
        """
        if not hasattr(self, "grid"):
            self.ingest_grid(self.estimator)

        series, labels = self._parse_series("all")
        labels = (
            series.copy()
            if (
                self.optimize_on.startswith("y")
                and self.optimize_on.split("y")[-1].isnumeric()
            )
            or self.optimize_on.startswith("series")
            else labels
        )
        metrics = {f"{label}_metric": [] for label in labels}
        iters = self.grid.shape[0]
        for i in range(iters):
            try:
                # returns a dict
                hp = {k: v[i] for k, v in self.grid.to_dict(orient="list").items()}
                val_preds, val_ac = self._forecast(
                    fcster=self.estimator,
                    tune=True,
                    dynamic_testing=dynamic_tuning,
                    **hp,
                )
                for s, l in zip(series, labels):
                    vp = val_preds[s]
                    va = val_ac[s][-len(vp) :]
                    metrics[l + "_metric"].append(
                        self.metrics[self.validation_metric](va, vp)
                    )
            except TypeError:
                raise
            except Exception as e:
                raise
                self.grid.drop(i, axis=0, inplace=True)
                logging.warning(f"Could not evaluate the paramaters: {hp}. error: {e}")
        metrics = pd.DataFrame(metrics)
        if metrics.shape[0] > 0:
            self.dynamic_tuning = dynamic_tuning
            self.grid.reset_index(drop=True, inplace=True)
            self.grid_evaluated = self.grid.copy()
            self.grid_evaluated["validation_length"] = self.validation_length
            self.grid_evaluated["validation_metric"] = self.validation_metric
            if self.optimize_on in self.optimizer_funcs:
                metrics["optimized_metric"] = metrics.apply(
                    self.optimizer_funcs[self.optimize_on], axis=1
                )
            else:
                metrics["optimized_metric"] = metrics[self.optimize_on + "_metric"]
            self.grid_evaluated = pd.concat([self.grid_evaluated, metrics], axis=1)
        else:
            self.grid_evaluated = pd.DataFrame()
        if not cv:
            self._find_best_params(self.grid_evaluated)

    def cross_validate(self, k=5, rolling=False, dynamic_tuning=False):
        """ Tunes a model's hyperparameters using time-series cross validation. 
        Monitors the metric specified in the valiation_metric attribute. 
        Set an estimator before calling. This is an alternative method to tune().
        Reads a grid for the estimator from a grids file unless a grid is ingested manually. 
        Each fold size is equal to one another and is determined such that the last fold's 
        training and validation sizes are the same (or close to the same). With `rolling` = True, 
        All train sizes will be the same for each fold. 
        The chosen parameters are stored in the best_params attribute.
        The full validation grid is stored in the grid_evaluated attribute.
        Normal CV diagram: https://scalecast-examples.readthedocs.io/en/latest/misc/validation/validation.html#5-Fold-Time-Series-Cross-Validation.
        Rolling CV diagram: https://scalecast-examples.readthedocs.io/en/latest/misc/validation/validation.html#5-Fold-Rolling-Time-Series-Cross-Validation. 

        Args:
            k (int): Default 5. The number of folds. Must be at least 2.
            rolling (bool): Default False. Whether to use a rolling method.
            dynamic_tuning (bool or int): Default False.
                Whether to dynamically tune the model or, if int, how many forecast steps to dynamically tune it.

        Returns:
            None
        """
        rolling = bool(rolling)
        k = int(k)
        _developer_utils.descriptive_assert(k >= 2, ValueError, f"k must be at least 2, got {k}")
        mvf = self.__deepcopy__()
        usable_obs = len(mvf.series1["y"]) - mvf.test_length
        val_size = usable_obs // (k + 1)
        _developer_utils.descriptive_assert(
            val_size > 0,
            ForecastError,
            f'not enough observations in sample to cross validate.'
        )
        mvf.set_validation_length(val_size)
        grid_evaluated_cv = pd.DataFrame()
        for i in range(k):
            if i > 0:
                mvf.current_xreg = {
                    k: pd.Series(v.values[:-val_size])
                    for k, v in mvf.current_xreg.items()
                }
                mvf.current_dates = pd.Series(mvf.current_dates.values[:-val_size])
                for s in range(mvf.n_series):
                    setattr(
                        mvf,
                        f"series{s+1}",
                        {
                            "y": pd.Series(
                                getattr(mvf, f"series{s+1}")["y"].values[:-val_size]
                            ),
                            "levely": getattr(mvf, f"series{s+1}")["levely"][
                                :-val_size
                            ],
                            "integration": getattr(mvf, f"series{s+1}")["integration"],
                        },
                    )

            mvf2 = mvf.__deepcopy__()
            if rolling:
                n = val_size * 2 + mvf2.test_length
                mvf2.current_dates = pd.Series(mvf2.current_dates.values[-n:])
                mvf2.current_xreg = {
                    k: pd.Series(v.values[-n:]) for k, v in mvf2.current_xreg.items()
                }
                for s in range(mvf2.n_series):
                    setattr(
                        mvf2,
                        f"series{s+1}",
                        {
                            "y": pd.Series(
                                getattr(mvf2, f"series{s+1}")["y"].values[-n:]
                            ),
                            "levely": getattr(mvf2, f"series{s+1}")["levely"][-n:],
                            "integration": getattr(mvf2, f"series{s+1}")["integration"],
                        },
                    )

            mvf2.tune(dynamic_tuning=dynamic_tuning,cv=True)
            orig_grid = mvf2.grid.copy()
            if mvf2.grid_evaluated.shape[0] == 0:
                self.grid = pd.DataFrame()
                self._find_best_params(mvf2.grid_evaluated)
                return  # writes a warning and moves on

            mvf2.grid_evaluated["fold"] = i
            mvf2.grid_evaluated["rolling"] = rolling
            mvf2.grid_evaluated["train_length"] = (
                len(mvf2.series1["y"]) - val_size - mvf2.test_length
            )
            grid_evaluated_cv = pd.concat([grid_evaluated_cv, mvf2.grid_evaluated])

        # convert into str for the group or it fails
        if "lags" in grid_evaluated_cv:
            grid_evaluated_cv["lags"] = grid_evaluated_cv["lags"].apply(
                lambda x: str(x)
            )

        grid_evaluated = grid_evaluated_cv.groupby(
            grid_evaluated_cv.columns.to_list()[: -(4 + mvf2.n_series)],
            dropna=False,
            as_index=False,
            sort=False,
        )["optimized_metric"].mean()

        # convert back to whatever it was before or it fails when calling the hyperparam vals
        # contributions welcome for a more elegant solution
        if "lags" in grid_evaluated:
            grid_evaluated["lags"] = grid_evaluated["lags"].apply(lambda x: eval(x))

        self.grid = grid_evaluated.iloc[:, :-3]
        self.dynamic_tuning = mvf2.dynamic_tuning
        self._find_best_params(grid_evaluated)
        self.grid_evaluated = grid_evaluated_cv.reset_index(drop=True)
        self.grid = orig_grid  # because None changes to np.nan otherwise
        self.cross_validated = True

    def _find_best_params(self, grid_evaluated):
        self.cross_validated = (
            False  # will be changed to True appropriately if cv was called
        )
        if grid_evaluated.shape[0] > 0:
            if self.validation_metric == "r2":
                best_params_idx = self.grid.loc[
                    grid_evaluated["optimized_metric"]
                    == grid_evaluated["optimized_metric"].max()
                ].index.to_list()[0]
            elif self.validation_metric == 'mape' and grid_evaluated['metric_value'].isna().all():
                raise ValueError('validation metric cannot be mape when 0s are in the validation set.')
            else:
                best_params_idx = self.grid.loc[
                    grid_evaluated["optimized_metric"]
                    == grid_evaluated["optimized_metric"].min()
                ].index.to_list()[0]

            self.best_params = {
                k: v[best_params_idx]
                for k, v in self.grid.to_dict(orient="series").items()
            }
            self.best_params = {
                k: (
                    v
                    if not isinstance(v, float)
                    else int(v)
                    if str(v).endswith(".0")
                    else None
                    if np.isnan(v)
                    else v
                )
                for k, v in self.best_params.items()
            }
            self.validation_metric_value = grid_evaluated.loc[
                best_params_idx, "optimized_metric"
            ]
        else:
            warnings.warn(
                f"None of the keyword/value combos stored in the grid could be evaluated for the {self.estimator} model."
                " See the errors in warnings.log.",
                category=Warning,
            )
            self.best_params = {}


    def manual_forecast(self, call_me=None, dynamic_testing=True, test_only = None, **kwargs):
        """ Manually forecasts with the hyperparameters, normalizer, and lag selections passed as keywords.

        Args:
            call_me (str): Optional.
                What to call the model when storing it in the object's history dictionary.
                If not specified, the model's nickname will be assigned the estimator value ('mlp' will be 'mlp', etc.).
                Duplicated names will be overwritten with the most recently called model.
            dynamic_testing (bool or int): Default True.
                Whether to dynamically/recursively test the forecast (meaning AR terms will be propagated with predicted values).
                If True, evaluates recursively over the entire out-of-sample slice of data.
                If int, window evaluates over that many steps (2 for 2-step recurvie testing, 12 for 12-step, etc.).
                Setting this to False or 1 means faster performance, 
                but gives a less-good indication of how well the forecast will perform more than one period out.
            **kwargs: Passed to the _forecast_{estimator}() method.
                Can include lags and normalizer in addition to any given model's specific hyperparameters.

        Returns:
            None

        >>> mvf.set_estimator('mlr')
        >>> mvf.manual_forecast(normalizer='scale')
        """
        _developer_utils.descriptive_assert(
            isinstance(call_me, str) | (call_me is None),
            ValueError,
            "call_me must be a str type or None",
        )

        if "tune" in kwargs.keys():
            kwargs.pop("tune")
            warnings.warn(
                "tune argument will be ignored.",
                category=Warning,
            )

        self.call_me = self.estimator if call_me is None else call_me
        self.forecast = self._forecast(
            fcster=self.estimator, dynamic_testing=dynamic_testing, **kwargs
        )
        self._bank_history(**kwargs)

    def tune_test_forecast(
        self,
        models,
        cross_validate=False,
        dynamic_tuning=False,
        dynamic_testing=True,
        limit_grid_size=None,
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
            suffix=suffix,
            error=error,
            **cvkwargs,
        )
        

    def set_optimize_on(self, how):
        """ Choose how to determine best models by choosing which series should be optimized.
        This is the decision that will be used for tuning models as well.

        Args:
            how (str): One of MVForecaster.optimizer_funcs, 'series{n}', 'y{n}', or the series name.
                If in MVForecaster.optimizer_funcs, will optimize based on that metric.
                If 'series{n}', 'y{n}', or the series name, will choose the model that performs best on that series.
                By default, optimize_on is set to 'mean' when the object is initiated.
        """
        if how in self.optimizer_funcs:
            self.optimize_on = how
        else:
            series, labels = self._parse_series(how)
            self.optimize_on = labels[0]
    
    @_developer_utils.log_warnings
    def _forecast(
        self, fcster, dynamic_testing, tune=False, normalizer="minmax", lags=1, **kwargs
    ):
        """ Runs the vector multivariate forecast start-to-finish. All Xvars stored in the object are used always. All sklearn estimators supported.
        See example: https://scalecast-examples.readthedocs.io/en/latest/multivariate/multivariate.html.

        Args:
            fcster (str): One of `MVForecaster.estimators`. Reads the estimator set to `set_estimator()` method.
            dynamic_testing (bool or int): Default True.
                Whether to dynamically/recursively test the forecast (meaning AR terms will be propagated with predicted values).
                If True, evaluates dynamically over the entire out-of-sample slice of data.
                If int, window evaluates over that many steps (2 for 2-step dynamic forecasting, 12 for 12-step, etc.).
                Setting this to False or 1 means faster performance, 
                but gives a less-good indication of how well the forecast will perform more than one period out.
                The model will skip testing if the test_length attribute is set to 0.
            tune (bool): Default False.
                Whether the model is being tuned.
                It does not need to be specified by the user.
            normalizer (str): The scaling technique to apply to the data. One of `MVForecaster.normalizer`. 
                Default 'minmax'.
                If not None and a test length is specified greater than 0, the normalizer is fit on the training data only.
            lags (int | list[int] | dict[str,(int | list[int])]): Default 1.
                The lags to add from each series to forecast with.
                Needs to use at least one lag for any sklearn model.
                Some models in the `scalecast.auxmodels` module require you to pass None or 0 to lags.
                If int, that many lags will be added for all series.
                If list, each element must be int types, and only those lags will be added for each series.
                If dict, the key must be the user-selected series name, 'series{n}' or 'y{n}', and key is list or int.
            **kwargs: Treated as model hyperparameters and passed to the applicable sklearn or other type of estimator.

        >>> mvf.set_estimator('gbt')
        >>> mvf.manual_forecast(lags=3) # adds three lags for each series
        >>> mvf.manual_forecast(lags=[1,3]) # first and third lags added for each series
        >>> mvf.manual_forecast(lags={'y1':2,'y2':3}) # 2 lags added for first series, 3 lags for second
        >>> mvf.manual_forecast(lags={'series1':[1,3],'series2':3}) # first and third lag for first series, 3 lags for second
        """

        def prepare_data(lags):
            observed = pd.DataFrame(self.current_xreg)
            future = pd.DataFrame(self.future_xreg, index=range(len(self.future_dates)))
            
            if lags is None or not lags:
                observedy = pd.DataFrame(
                    {f'y{i+1}':getattr(self,f'series{i+1}')['y'].to_list() for i in range(self.n_series)}
                )
                observed = pd.concat([observedy,observed],axis=1)
                return observed, future

            for i in range(self.n_series):
                if str(lags).isnumeric() or isinstance(lags, float):
                    lags = int(lags)
                    for j in range(lags):
                        col = f"y{i+1}_lag{j+1}"
                        observed[col] = (
                            getattr(self, "series" + str(i + 1))["y"]
                            .shift(j + 1)
                            .values
                        )
                        future.loc[0, col] = getattr(self, "series" + str(i + 1))[
                            "y"
                        ].values[-(j + 1)]
                elif isinstance(lags, dict):
                    series, labels = self._parse_series(lags.keys())
                    if "y" + str(i + 1) in series:
                        idx = series.index(f"y{i+1}")
                        lag = lags[labels[idx]]
                    else:
                        continue
                    if str(lag).isnumeric() or isinstance(lag, float):
                        lag = int(lag)
                        for j in range(lag):
                            col = f"y{i+1}_lag{j+1}"
                            observed[col] = (
                                getattr(self, "series" + str(i + 1))["y"]
                                .shift(j + 1)
                                .values
                            )
                            future.loc[0, col] = getattr(self, "series" + str(i + 1))[
                                "y"
                            ].values[-(j + 1)]
                    elif isinstance(lag, str):
                        raise ValueError(f"Cannot use argument for lags: {lags}.")
                    else:
                        try:
                            lag = list(lag)
                        except TypeError:
                            raise ValueError(f"Cannot use argument for lags: {lags}.")
                        for j in lag:
                            col = f"y{i+1}_lag{j}"
                            observed[col] = (
                                getattr(self, "series" + str(i + 1))["y"]
                                .shift(j)
                                .values
                            )
                            future.loc[0, col] = getattr(self, "series" + str(i + 1))[
                                "y"
                            ].values[-j]
                elif isinstance(lags, str):
                    raise ValueError(f"lags cannot be str type, got {lags}.")
                else:
                    try:
                        lags = list(lags)
                    except TypeError:
                        raise ValueError(f"Cannot use argument for lags: {lags}.")
                    for j in lags:
                        col = f"y{i+1}_lag{j}"
                        observed[col] = (
                            getattr(self, "series" + str(i + 1))["y"].shift(j).values
                        )
                        future.loc[0, col] = getattr(self, "series" + str(i + 1))[
                            "y"
                        ].values[-j]
            return observed.dropna().reset_index(drop=True), future

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
                return scaler.transform(X if not hasattr(X,'values') else X.values)
            else:
                return X.values if hasattr(X, "values") else X

        def train(X, y, normalizer, true_forecast = False, **kwargs):
            if not true_forecast or self.test_length == 0:
                self.scaler = self._parse_normalizer(
                    X, 
                    normalizer,
                ) # only fit normalizer on training data or on full dataset if not testing
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

        def evaluate(trained_models, future, dynamic_testing):
            future = future.reset_index(drop=True)
            if lags is None or not lags:
                future = scale(self.scaler, future)
                p = trained_models.predict(future)
                preds = {
                    f'y{i+1}':list(p[:,i]) for i in range(self.n_series)
                }
            elif dynamic_testing is False:
                preds = {}
                future = scale(self.scaler, future)
                for series, regr in trained_models.items():
                    preds[series] = list(regr.predict(future))
            else:
                preds = {series: [] for series in trained_models.keys()}
                for i in range(len(future)):
                    fut = scale(self.scaler,future.iloc[i].values.reshape(1,-1))
                    for series, regr in trained_models.items():
                        preds[series].append(regr.predict(fut)[0])
                    if i < len(future) - 1:
                        for c in future.columns.to_list()[len(self.current_xreg):]:
                            ar = int(c.split('_lag')[-1])
                            series = c.split('_lag')[0]
                            s_num = int(series[1:])
                            idx = i + 1 - ar
                            if idx <= -1:
                                future.loc[i+1,c] = getattr(self, f'series{s_num}')['y'].to_list()[idx]
                            elif dynamic_testing is not True and (i+1) % dynamic_testing == 0:
                                pass
                            else:
                                future.loc[i+1,c] = preds[series][idx]
            return preds

        _developer_utils.descriptive_assert(
            isinstance(dynamic_testing, bool)
            | isinstance(dynamic_testing, int) & (dynamic_testing > -1),
            ValueError,
            f"dynamic_testing expected bool or non-negative int type, got {dynamic_testing}.",
        )
        dynamic_testing = (
            False
            if type(dynamic_testing) == int and dynamic_testing == 1
            else dynamic_testing
        )  # 1-step dynamic testing is same as no dynamic testing and no dynamic testing is more efficient
        observed, future = prepare_data(lags)

        if tune or self.test_length > 0:
            test_length = self.test_length + (self.validation_length if tune else 0)
            validation_length = self.validation_length
            if lags is None or not lags:
                trained = train(
                    X=observed.values[:-test_length].copy(),
                    y=None,
                    normalizer=None,
                    true_forecast=False,
                    **kwargs,
                )
            else:
                trained = {}
                for i in range(self.n_series):
                    trained[f"y{i+1}"] = train(
                        X=observed.values[:-test_length].copy(),
                        y=getattr(self, f"series{i+1}")["y"]
                        .values[-observed.shape[0] : -test_length]
                        .copy(),
                        normalizer=normalizer,
                        true_forecast=False,
                        **kwargs,
                    )
            preds = evaluate(
                trained,
                observed.iloc[-(test_length + validation_length) : -validation_length, :]
                if tune
                else observed.iloc[-test_length:],
                dynamic_testing,
            )
            if tune:
                return (
                    preds.copy(),
                    {
                        f"y{i+1}": getattr(self, f"series{i+1}")["y"]
                        .values[-(test_length + validation_length) : -validation_length]
                        .copy()
                        for i in range(self.n_series)
                    },
                )
            self.test_set_pred = preds.copy()
        else:
            self.test_set_pred = {}
        
        if lags is None or not lags:
            trained_full = train(
                X=observed.copy(),
                y=None,
                normalizer=None,
                true_forecast = True,
                **kwargs,
            )
        else:
            trained_full = {}
            for i in range(self.n_series):
                trained_full[f"y{i+1}"] = train(
                    X=observed.copy(),
                    y=getattr(self, f"series{i+1}")["y"]
                    .values[-observed.shape[0] :]
                    .copy(),
                    normalizer=normalizer,
                    true_forecast = True,
                    **kwargs,
                )
        self.dynamic_testing = dynamic_testing
        self.trained_models = trained_full # does not go to history
        self.fitted_values = evaluate(
            trained_models=trained_full, 
            future=observed.copy(), 
            dynamic_testing=False
        )
        return evaluate(
            trained_models=trained_full, 
            future=future.copy(), 
            dynamic_testing=True
        )

    def _parse_normalizer(self, X_train, normalizer):
        """ fits an appropriate scaler to training data that will then be applied to test and future data

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
        def undiff(preds_orig, test=False):
            # self.seriesx['levely']
            # self.seriesx['integration']
            preds = {series: l[:] for series, l in preds_orig.items()}
            actuals_all = {
                series: getattr(self, f"series{i+1}")["levely"][:]
                for i, series in enumerate(preds.keys())
            }
            integration = {
                series: getattr(self, f"series{i+1}")["integration"]
                for i, series in enumerate(preds.keys())
            }
            if test:
                actuals = {
                    series: ac[: -len(preds_orig[series])]
                    for series, ac in actuals_all.items()
                }
                test_set_actuals = {
                    series: ac[-len(preds_orig[series]) :]
                    for series, ac in actuals_all.items()
                }
            else:
                actuals = actuals_all.copy()
                test_set_actuals = None

            for series, p in preds.items():
                if integration[series] == 0:
                    continue
                first_obs = actuals[series][-1]
                preds[series].insert(0, first_obs)
                preds[series] = list(np.cumsum(preds[series]))[1:]
            return preds, test_set_actuals

        call_me = self.call_me
        test_set_preds = self.test_set_pred.copy()
        test_set_actuals = {} if self.test_length == 0 else {
            f"y{i+1}": (
                getattr(self, f"series{i+1}")["y"].to_list()[-self.test_length :]
            ) for i in range(self.n_series)
        }
        fitted_vals = self.fitted_values.copy()
        fitted_val_actuals = {
            f"y{i+1}": (
                getattr(self, f"series{i+1}")["y"].to_list()[
                    -len(fitted_vals[f"y{i+1}"]) :
                ] 
            ) for i in range(self.n_series)
        }
        resids = {
            series: (
                [fv - ac for fv, ac in zip(fitted_vals[series], act)]
            ) for series, act in fitted_val_actuals.items()
        }
        fcst = self.forecast.copy()
        lvl_fcst, _ = undiff(fcst.copy())
        lvl_tsp, lvl_tsa = undiff(test_set_preds, test=True)
        lvl_fv, lvl_fva = undiff(fitted_vals, test=True)
        lvl_resids = {
            series: [fv - ac for fv, ac in zip(lvl_fv[series], act)]
            for series, act in lvl_fva.items()
        }
        self.history[call_me] = {
            "Estimator": self.estimator,
            "Xvars": list(self.current_xreg.keys()),
            "HyperParams": {
                k: v for k, v in kwargs.items() if k not in ("normalizer", "lags", "mvf")
            },
            "Lags": kwargs["lags"] if "lags" in kwargs.keys() else 1,
            "Scaler": kwargs["normalizer"]
            if "normalizer" in kwargs.keys()
            else "minmax",
            "Integration": self.integration,
            "Forecast": fcst,
            "Observations": len(self.current_dates),
            "FittedVals": fitted_vals,
            "Resids": resids,
            "Tuned": None,
            "CrossValidated": False,
            "DynamicallyTested": self.dynamic_testing,
            "TestSetLength": self.test_length,
            "TestSetPredictions": test_set_preds,
            "TestSetActuals": test_set_actuals,
            "CILevel": self.cilevel if self.cis else np.nan,
            "ValidationSetLength": None,
            "ValidationMetric": None,
            "ValidationMetricValue": None,
            "grid_evaluated": None,
            "LevelForecast": lvl_fcst,
            "LevelTestSetPreds": lvl_tsp,
            "LevelTestSetActuals": lvl_tsa,
            "LevelFittedVals": lvl_fv,
        }
        for met, func in self.metrics.items():
            self.history[call_me]["TestSet" + met.upper()] = {
                series: _developer_utils._return_na_if_len_zero(a, test_set_preds[series], func)
                for series, a in test_set_actuals.items()   
            }
            self.history[call_me]["InSample" + met.upper()] = {
                series: _developer_utils._return_na_if_len_zero(a, fitted_vals[series], func)
                for series, a in fitted_val_actuals.items()
            }
            self.history[call_me]["LevelTestSet" + met.upper()] = {
                series: _developer_utils._return_na_if_len_zero(a, lvl_tsp[series], func) for series, a in lvl_tsa.items()
            }
            self.history[call_me]["LevelInSample" + met.upper()] = {
                series: _developer_utils._return_na_if_len_zero(a, lvl_fv[series], func) for series, a in lvl_fva.items()
            }
        if hasattr(self, "best_params"):
            self.history[call_me]["Tuned"] = True
            self.history[call_me]["CrossValidated"] = self.cross_validated
            self.history[call_me]["ValidationSetLength"] = (
                self.validation_length if not self.cross_validated else np.nan
            )
            self.history[call_me]["ValidationMetric"] = self.validation_metric
            self.history[call_me][
                "ValidationMetricValue"
            ] = self.validation_metric_value
            self.history[call_me]["grid_evaluated"] = self.grid_evaluated

        if self.cis is True:
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
            if self.integration == 1:
                fcst = self.history[call_me]['LevelForecast']
                test_preds = self.history[call_me]['LevelTestSetPreds']
                test_actuals = {
                    k: getattr(
                        self,
                        'series{i}'
                    )['levely'][
                        -self.test_length:
                    ] for i, k in enumerate(test_preds.keys())
                }
                test_resids = {k:np.abs(np.array(p) - np.array(test_actuals[k])) for k, p in test_preds.items()}
                ci_range = {k:np.percentile(r, 100 * self.cilevel) for k,r in test_resids.items()}
            self._set_cis(
                "LevelUpperCI",
                "LevelLowerCI",
                "LevelTSUpperCI",
                "LevelTSLowerCI",
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

            if self.optimize_on in self.optimizer_funcs and determine_best_by != 'ValidationMetricValue':
                models_metrics = {
                    m: self.optimizer_funcs[self.optimize_on](list(v.values()))
                    for m, v in models_metrics.items()
                }
            elif determine_best_by != 'ValidationMetricValue':
                series, label = self._parse_series(self.optimize_on)
                models_metrics = {m: v[series[0]] for m, v in models_metrics.items()}

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
        """ returns list (series), list (labels)
        """

        def parse_str_arg(arg):
            if arg.startswith("series"):
                series = ["y{}".format(arg.split("series")[-1])]
                labels = series.copy()
            elif arg.startswith("y"):
                series = [arg]
                labels = series.copy()
            else:
                labels = [arg]
                series = [self.name_series_map[arg][1]]
            return series, labels

        if series == "all":
            series = [f"y{i+1}" for i in range(self.n_series)]
            if hasattr(self, "name_series_map"):
                labels = list(self.name_series_map.keys())
            else:
                labels = series.copy()
        elif isinstance(series, str):
            series, labels = parse_str_arg(series)
        else:
            series1 = list(series)
            series = [parse_str_arg(s)[0][0] for s in series1]
            labels = [parse_str_arg(s)[1][0] for s in series1]
        return series, labels

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
        level=False, 
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
               Name of the series ('y1...', 'series1...' or user-selected name), 'all', or list-like of series names.
            put_best_on_top (bool): Only set to True if you have previously called set_best_model().
                If False, ignored.
            level (bool): Default False.
                If True, will always plot level forecasts.
                If False, will plot the forecasts at whatever level they were called on.
                If False and there are a mix of models passed with different integrations, will default to True.
                This argument will be removed from a future version of scalecast as all series transformations
                will be handled with the SeriesTransformer object.
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

        if level is True:
            warnings.warn(
                'The level argument will be removed from a future version of scalecast. '
                'All transformations will be handled by the SeriesTransformer object.',
                category = FutureWarning,
            )

        series, labels = self._parse_series(series)
        models = self._parse_models(models, put_best_on_top)
        integration = [v for s, v in self.integration.items() if s in series]
        level = True if len(set(integration)) > 1 else level

        k = 0
        for i, s in enumerate(series):
            sns.lineplot(
                x=self.current_dates.to_list(),
                y=list(
                    getattr(self, "series{}".format(s.split("y")[-1]))[
                        "y" if not level else "levely"
                    ]
                )[-len(self.current_dates) :],
                label=f"{labels[i]} actual",
                ax=ax,
                color=__series_colors__[i],
            )
            for m in models:
                sns.lineplot(
                    x=self.future_dates.to_list(),
                    y=self.history[m]["Forecast"][s]
                    if not level
                    else self.history[m]["LevelForecast"][s],
                    label=f"{labels[i]} {m}",
                    color=__colors__[k],
                    ax=ax,
                )

                if ci:
                    try:
                        plt.fill_between(
                            x=self.future_dates.to_list(),
                            y1=self.history[m]["UpperCI"][s]
                            if not level
                            else self.history[m]["LevelUpperCI"][s],
                            y2=self.history[m]["LowerCI"][s]
                            if not level
                            else self.history[m]["LevelLowerCI"][s],
                            alpha=0.2,
                            color=__colors__[k],
                            label="{} {} {:.0%} CI".format(
                                labels[i], m, self.history[m]["CILevel"]
                            ),
                        )
                    except KeyError:
                        _developer_utils._warn_about_not_finding_cis(m)
                k += 1

        plt.legend()
        plt.xlabel("Date")
        plt.ylabel("Values")
        return ax

    def plot_test_set(
        self,
        models="all",
        series="all",
        put_best_on_top=False,
        include_train=True,
        level=False,
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
            level (bool): Default False.
                If True, will always plot level forecasts.
                If False, will plot the forecasts at whatever level they were called on.
                If False and there are a mix of models passed with different integrations, will default to True.
                This argument will be removed from a future version of scalecast as all series transformations
                will be handled with the SeriesTransformer object.
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

        if level is True:
            warnings.warn(
                'The level argument will be removed from a future version of scalecast. '
                'All transformations will be handled by the SeriesTransformer object.',
                category = FutureWarning,
            )
        
        series, labels = self._parse_series(series)
        models = self._parse_models(models, put_best_on_top)
        integration = [v for s, v in self.integration.items() if s in series]
        level = True if len(set(integration)) > 1 else level
        include_train = (
            len(self.current_dates)
            if include_train is True
            else self.test_length
            if include_train is False
            else include_train
        )

        k = 0
        for i, s in enumerate(series):
            y = list(
                getattr(self, "series{}".format(s.split("y")[-1]))[
                    "y" if not level else "levely"
                ]
            )[-len(self.current_dates) :]
            sns.lineplot(
                x=self.current_dates.to_list()[-include_train:],
                y=y[-include_train:],
                label=f"{labels[i]} actual",
                ax=ax,
                color=__series_colors__[i],
            )
            for m in models:
                sns.lineplot(
                    x=self.current_dates.to_list()[-self.test_length :],
                    y=self.history[m]["TestSetPredictions"][s]
                    if not level
                    else self.history[m]["LevelTestSetPreds"][s],
                    label=f"{labels[i]} {m}",
                    color=__colors__[k],
                    linestyle="--",
                    alpha=0.7,
                    ax=ax,
                )

                if ci:
                    try:
                        plt.fill_between(
                            x=self.current_dates.to_list()[-self.test_length :],
                            y1=self.history[m]["TestSetUpperCI"][s]
                            if not level
                            else self.history[m]["LevelTSUpperCI"][s],
                            y2=self.history[m]["TestSetLowerCI"][s]
                            if not level
                            else self.history[m]["LevelTSLowerCI"][s],
                            alpha=0.2,
                            color=__colors__[k],
                            label="{} {} {:.0%} CI".format(
                                labels[i], m, self.history[m]["CILevel"]
                            ),
                        )
                    except KeyError:
                        _developer_utils._warn_about_not_finding_cis(m)
                k += 1

        plt.legend()
        plt.xlabel("Date")
        plt.ylabel("Values")
        return ax

    def plot_fitted(
        self, 
        models="all", 
        series="all", 
        level=False,
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
            level (bool): Default False.
                If True, will always plot level forecasts.
                If False, will plot the forecasts at whatever level they were called on.
                This argument will be removed from a future version of scalecast as all series transformations
                will be handled with the SeriesTransformer object.
            ax (Axis): Optional. The existing axis to write the resulting figure to.
            figsize (tuple): Default (12,6). Size of the resulting figure. Ignored when ax is not None.


        Returns:
            (Axis): The figure's axis.

        >>> mvf.plot_fitted() # plots all fitted values on all series
        >>> plt.show()
        """
        series, labels = self._parse_series(series)
        models = self._parse_models(models, False)
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)

        if level is True:
            warnings.warn(
                'The level argument will be removed from a future version of scalecast. '
                'All transformations will be handled by the SeriesTransformer object.',
                category = FutureWarning,
            )

        k = 0
        for i, s in enumerate(series):
            act = (
                getattr(self, "series{}".format(s.split("y")[-1]))["y"].to_list()
                if not level
                else getattr(self, "series{}".format(s.split("y")[-1]))["levely"][:]
            )
            sns.lineplot(
                x=self.current_dates.to_list(),
                y=act[-len(self.current_dates):],
                label=f"{labels[i]} actual",
                ax=ax,
                color=__series_colors__[i],
            )
            for m in models:
                fvs = (
                    self.history[m]["FittedVals"][s]
                    if not level
                    else self.history[m]["LevelFittedVals"][s]
                )
                sns.lineplot(
                    x=self.current_dates.to_list()[-len(fvs) :],
                    y=fvs,
                    label=f"{labels[i]} {m}",
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
        """ Exports 1-all of 5 pandas dataframes. Can write to excel with each dataframe on a separate sheet.
        Will return either a dictionary with dataframes as values (df str arguments as keys) or a single dataframe if only one df is specified.

        Args:
            dfs (list-like or str): Default 
                ['model_summaries', 'lvl_test_set_predictions', 'lvl_fcsts'].
                A list or name of the specific dataframe(s) you want returned and/or written to excel.
                Must be one of or multiple of the elements in default and can also include "all_fcsts" and "test_set_predictions", but those
                will be removed in a future version of scalecast. The distinction between level/not level
                will no longer exist within the Forecaster object. All transforming and reverting will be handled with the SeriesTransformer object.
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
        series, labels = self._parse_series(series)
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
                "Scaler",
                "Observations",
                "Tuned",
                "CrossValidated",
                "DynamicallyTested",
                "Integration",
                "TestSetLength",
                "ValidationSetLength",
                "ValidationMetric",
                "ValidationMetricValue",
                "OptimizedOn",
                "MetricOptimized",
                "best_model",
            ]
            model_summaries = pd.DataFrame()
            for l, s in zip(labels, series):
                for m in models:
                    model_summary_sm = pd.DataFrame(
                        {"Series": [l], "ModelNickname": [m]}
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
                            | (
                                k.startswith('LevelTestSet')
                                & (k not in (
                                    'LevelTestSetPreds',
                                    'LevelTestSetActuals',
                                )
                            ))
                            | k.startswith('LevelInSample')
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
                            attr = self.history[m][c]
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
        if "all_fcsts" in dfs:
            warnings.warn(
                'The "all_fcsts" DataFrame is will be removed in a future version of scalecast.'
                ' To extract point forecasts for evaluated models, use "lvl_fcsts".',
                category = FutureWarning,
            )
            df = pd.DataFrame({"DATE": self.future_dates.to_list()})
            for l, s in zip(labels, series):
                for m in models:
                    df[f"{l}_{m}_fcst"] = self.history[m]["Forecast"][s][:]
                    if cis:
                        try:
                            df[f"{l}_{m}_fcst_upper"] = self.history[m]["UpperCI"][s][:]
                            df[f"{l}_{m}_fcst_lower"] = self.history[m]["LowerCI"][s][:]
                        except KeyError:
                            _developer_utils._warn_about_not_finding_cis(m)
            output["all_fcsts"] = df
        if "test_set_predictions" in dfs:
            warnings.warn(
                'The "test_set_predictions" DataFrame will be removed in a future version of scalecast.'
                ' To extract test-set predictions for evaluated models, use "lvl_test_set_predictions".',
                category = FutureWarning,
            )
            if self.test_length > 0:
                df = pd.DataFrame(
                    {"DATE": self.current_dates.to_list()[-self.test_length :]}
                )
                i = 0
                for l, s in zip(labels, series):
                    df[f"{l}_actuals"] = getattr(self, f"series{i+1}")["y"].to_list()[
                        -self.test_length :
                    ]
                    for m in models:
                        df[f"{l}_{m}_test_preds"] = self.history[m]["TestSetPredictions"][
                            s
                        ][:]
                        if cis:
                            try:
                                df[f"{l}_{m}_test_preds_upper"] = self.history[m][
                                    "TestSetUpperCI"
                                ][s][:]
                                df[f"{l}_{m}_test_preds_lower"] = self.history[m][
                                    "TestSetLowerCI"
                                ][s][:]
                            except KeyError:
                                _developer_utils._warn_about_not_finding_cis(m)
                    i += 1
                output["test_set_predictions"] = df
            else:
                output["test_set_predictions"] = pd.DataFrame()
        if "lvl_fcsts" in dfs:
            df = pd.DataFrame({"DATE": self.future_dates.to_list()})
            for l, s in zip(labels, series):
                for m in models:
                    df[f"{l}_{m}_lvl_fcst"] = self.history[m]["LevelForecast"][s][:]
                    if cis:
                        try:
                            df[f"{l}_{m}_lvl_fcst_upper"] = self.history[m]["LevelUpperCI"][
                                s
                            ][:]
                            df[f"{l}_{m}_lvl_fcst_lower"] = self.history[m]["LevelLowerCI"][
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
                for l, s in zip(labels, series):
                    df[f"{l}_actuals"] = getattr(self, f"series{i+1}")["levely"][
                        -self.test_length :
                    ]
                    for m in models:
                        df[f"{l}_{m}_lvl_ts"] = self.history[m]["LevelTestSetPreds"][s][:]
                        if cis:
                            try:
                                df[f"{l}_{m}_lvl_ts_upper"] = self.history[m]["LevelTSUpperCI"][
                                    s
                                ][:]
                                df[f"{l}_{m}_lvl_ts_lower"] = self.history[m]["LevelTSLowerCI"][
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

    def export_fitted_vals(self, series="all", models="all", level=False):
        """ Exports a dataframe of fitted values and actuals.

        Args:
            models (list-like or str): Default 'all'.
               Name of the model, 'all', or list-like of model names.
            series (list-like or str): Default 'all'.
               The series to plot.
               Name of the series ('y1...', 'series1...' or user-selected name), 'all', or list-like of series names.
            level (bool): Default False.
                Whether to export level fitted values.
                This argument will be removed from a future version of scalecast as all series transformations
                will be handled with the SeriesTransformer object.
                This argument will be removed from a future version of scalecast as all series transformations
                will be handled with the SeriesTransformer object.

        Returns:
            (DataFrame): The fitted values for all selected series and models.
        """
        series, labels = self._parse_series(series)
        models = self._parse_models(models, False)
        dfd = {"DATE": self.current_dates.to_list()}
        length = len(dfd["DATE"])
        i = 0
        for l, s in zip(labels, series):
            dfd[f"{l}_actuals"] = (
                getattr(self, f"series{i+1}")["y"].to_list()
                if not level
                else getattr(self, f"series{i+1}")["levely"]
            )
            for m in models:
                dfd[f"{l}_{m}_fvs"] = (
                    self.history[m]["FittedVals"][s][:]
                    if not level
                    else self.history[m]["LevelFittedVals"][s][:]
                )
                length = min(length, len(dfd[f"{l}_{m}_fvs"]), len(dfd[f"{l}_actuals"]))
            i += 1
        return pd.DataFrame({c: v[-length:] for c, v in dfd.items()})

    def export_validation_grid(self, model):
        """ Exports a validation grid for a selected model.

        Args:
            model (str): The model to export the validation grid for.

        Returns:
            (DataFrame): The validation grid.
        """
        return self.history[model]["grid_evaluated"].copy()

    def backtest(self, model, fcst_length="auto", n_iter=10, jump_back=1):
        """ Runs a backtest of a selected evaluated model over a certain 
        amount of iterations to test the average error if that model were 
        implemented over the last so-many actual forecast intervals.
        All scoring is dynamic to give a true out-of-sample result.
        All metrics are specific to level data and will only work if the default metrics were maintained when the MVForecaster object was initiated.
        Two results are extracted: a dataframe of actuals and predictions across each iteration and
        a dataframe of test-set metrics across each iteration with a mean total as the last column.
        These results are stored in the Forecaster object's history and can be extracted by calling 
        `export_backtest_metrics()` and `export_backtest_values()`.

        Args:
            model (str): The model to run the backtest for. Use the model nickname.
            fcst_length (int or str): Default 'auto'. 
                If 'auto', uses the same forecast length as the number of future dates saved in the object currently.
                If int, uses that as the forecast length.
            n_iter (int): Default 10. The number of iterations to backtest.
                Models will iteratively train on all data before the fcst_length worth of values.
                Each iteration takes observations (this number is determined by the value passed to the jump_back arg)
                off the end to redo the cast until all of n_iter is exhausted.
            jump_back (int): Default 1. 
                The number of time steps between two consecutive training sets.

        Returns:
            None

        >>> mvf.set_estimator('mlr')
        >>> mvf.manual_forecast()
        >>> mvf.backtest('mlr')
        >>> backtest_metrics = mvf.export_backtest_metrics('mlr')
        >>> backetest_values = mvf.export_backtest_values('mlr')
        """
        warnings.warn(
            "The `MVForecaster.backtest()` method will be removed in a future version. Please use MVPipeline.backtest() instead.",
            category = FutureWarning,
        )
        if fcst_length == "auto":
            fcst_length = len(self.future_dates)
        fcst_length = int(fcst_length)
        series, labels = self._parse_series("all")
        mets = list(self.metrics.keys())
        tuples = []
        for s in labels:
            for m in mets:
                tuples.append(
                    (s, m)
                )  # no list comprehension to preserve order and because the lengths of labels and mets are different
        index = pd.MultiIndex.from_tuples(tuples, names=["series", "metric"])
        metric_results = pd.DataFrame(
            columns=[f"iter{i}" for i in range(1, n_iter + 1)], index=index
        )
        value_results = pd.DataFrame()
        f1 = self.deepcopy()
        f1.eval_cis(False)
        for i in range(n_iter):
            f = f1.deepcopy()
            if i > 0:
                f.current_xreg = {
                    k: pd.Series(v.values[: -i * jump_back])
                    for k, v in f.current_xreg.items()
                }
                f.current_dates = pd.Series(f.current_dates.values[: -i * jump_back])
                for k in range(f.n_series):
                    getattr(f, f"series{k+1}")["y"] = pd.Series(
                        getattr(f, f"series{k+1}")["y"].values[: -i * jump_back]
                    )
                    getattr(f, f"series{k+1}")["levely"] = getattr(f, f"series{k+1}")[
                        "levely"
                    ][: -i * jump_back]
                f.future_dates = pd.Series(f.future_dates.values[: -i * jump_back])
                f.future_xreg = {
                    k: v[: -i * jump_back] for k, v in f.future_xreg.items()
                }

            f.set_test_length(fcst_length)
            f.set_estimator(f.history[model]["Estimator"])
            params = f.history[model]["HyperParams"].copy()
            params["lags"] = f.history[model]["Lags"]
            params["normalizer"] = f.history[model]["Scaler"]
            f.history = {}
            f.manual_forecast(**params,call_me=model)
            test_mets = f.export("model_summaries",models=model)
            test_preds = f.export("lvl_test_set_predictions",models=model)
            for s in labels:
                for m in mets:
                    metric_results.loc[(s, m), f"iter{i+1}"] = test_mets.loc[
                        test_mets["Series"] == s, f"LevelTestSet{m.upper()}"
                    ].values[0]
                value_results[f"{s}_iter{i+1}dates"] = test_preds["DATE"]
                value_results[f"{s}_iter{i+1}actuals"] = test_preds[f"{s}_actuals"]
                value_results[f"{s}_iter{i+1}preds"] = test_preds[f"{s}_{model}_lvl_ts"]
        metric_results["mean"] = metric_results.mean(axis=1)
        self.history[model]["BacktestMetrics"] = metric_results
        self.history[model]["BacktestValues"] = value_results

    def export_backtest_metrics(self, model):
        """ Extracts the backtest metrics for a given model.
        Only works if `backtest()` has been called.

        Args:
            model (str): The model nickname to extract metrics for.

        Returns:
            (DataFrame): A copy of the backtest metrics.
        """
        return self.history[model]["BacktestMetrics"].copy()

    def export_backtest_values(self, model):
        """ Extracts the backtest values for a given model.
        Only works if `backtest()` has been called.

        Args:
            model (str): The model nickname to extract values for.

        Returns:
            (DataFrame): A copy of the backtest values.
        """
        return self.history[model]["BacktestValues"].copy()

    def corr(self, train_only=False, disp="matrix", df=None, **kwargs):
        """ Displays pearson correlation between all stored series in object, at whatever level they are stored.

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
            series, labels = self._parse_series("all")
            df = pd.DataFrame(
                {
                    labels[n]: getattr(self, f"series{n+1}")["y"]
                    for n in range(self.n_series)
                }
            )

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

    def corr_lags(self, y="series1", x="series2", lags=1, **kwargs):
        """ Displays pearson correlation between one series and another series' lags.

        Args:
            y (str): Default 'series1'. The series to display as the dependent series.
                Can use 'series{n}', 'y{n}' or user-selected name.
            x (str): Default 'series2'. The series to display as the independent series.
                Can use 'series{n}', 'y{n}' or user-selected name.
            lags (int): Default 1. The number of lags to display in the independent series.
            **kwargs: Passed to the MVForecaster.corr() method. Will not pass the df arg.

        Returns:
            (DataFrame or Figure): The created dataframe if disp == 'matrix' else the heatmap fig.
        """
        series1, label1 = self._parse_series(y)
        series2, label2 = self._parse_series(x)

        df = pd.DataFrame(
            {
                label1[0]: getattr(self, "series" + series1[0].split("y")[-1])["y"],
                label2[0]: getattr(self, "series" + series2[0].split("y")[-1])["y"],
            }
        )

        for i in range(lags):
            df[label2[0] + f"_lag{i+1}"] = df[label2[0]].shift(i + 1)

        df = df.dropna()
        return self.corr(df=df, **kwargs)