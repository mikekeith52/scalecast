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
from functools import wraps
from scipy import stats
import importlib
import copy

logging.basicConfig(filename="warnings.log", level=logging.WARNING)
#logging.captureWarnings(True)

from scalecast.Forecaster import (
    mape,
    rmse,
    mae,
    r2,
    log_warnings,
    _sklearn_imports_,
    _sklearn_estimators_,
    _metrics_,
    descriptive_assert,
    _normalizer_,
    _determine_best_by_,
    _colors_,
    _return_na_if_len_zero,
    _set_ci_step,
    _warn_about_not_finding_cis,
    ForecastError,
)

class MVForecaster:
    def __init__(
        self,
        *fs,
        not_same_len_action="trim",
        merge_Xvars="union",
        merge_future_dates="longest",
        test_length = 0,
        cis = False,
        names=None,
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
            **kwargs: Become attributes.
        """
        for f in fs:
            f.typ_set()
        if (
            len(set([len(f.current_dates) for f in fs])) > 1
            or len(set([min(f.current_dates) for f in fs])) > 1
        ):
            descriptive_assert(
                len(set([max(f.current_dates) for f in fs])) == 1,
                ForecastError,
                "Series cannot have different end dates.",
            )
            if not_same_len_action == "fail":
                raise ValueError("All series must be same length.")
            elif not_same_len_action == "trim":
                from scalecast.multiseries import keep_smallest_first_date
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

        self.optimize_on = "mean"
        self.estimator = "mlr"
        self.current_xreg = {}
        self.future_xreg = {}
        self.history = {}
        self.validation_length = 1
        self.validation_metric = "rmse"
        self.cis = cis
        self.cilevel = 0.95
        self.bootstrap_samples = 100
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
                f"merge_future_dates must be one of ('longest','shortest'), got {merge_future_dates}"
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
    BootstrapSamples={}
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
            self.bootstrap_samples,
            self.estimator,
            self.optimize_on,
            self.grids_file,
        )

    def __copy__(self):
        obj = type(self).__new__(self.__class__)
        obj.__dict__.update(self.__dict__)
        return obj

    def __deepcopy__(self):
        obj = type(self).__new__(self.__class__)
        obj.__dict__.update(self.__dict__)
        obj.__dict__ = copy.deepcopy(obj.__dict__)
        return obj

    def copy(self):
        """ Creates an object copy.
        """
        return self.__copy__()

    def deepcopy(self):
        """ Creates an object deepcopy.
        """
        return self.__deepcopy__()

    def add_sklearn_estimator(self, imported_module, called):
        """ Adds a new estimator from scikit-learn not built-in to the forecaster object that can be called using set_estimator().
        Only regression models supported.
        
        Args:
            imported_module (sklearn regression model):
                The model from sklearn to add. must have already been imported locally.
                Supports models from sklearn and sklearn APIs.
            called (str):
                The name of the estimator that can be called using set_estimator().

        Returns:
            None

        >>> from sklearn.ensemble import StackingRegressor
        >>> mvf.add_sklearn_estimator(StackingRegressor,called='stacking')
        >>> mvf.set_estimator('stacking')
        >>> mvf.manual_forecast(...)
        """
        globals()[called + "_"] = imported_module
        _sklearn_imports_[called] = globals()[called + "_"]
        _sklearn_estimators_.append(called)
        _sklearn_estimators_.sort()

    def set_test_length(self, n=1):
        """ Sets the length of the test set.

        Args:
            n (int or float): Default 1.
                The length of the resulting test set.
                Fractional splits are supported by passing a float less than 1 and greater than 0.
                Pass 0 to skip testing models.

        Returns:
            None

        >>> mvf.set_test_length(12) # test set of 12
        >>> mvf.set_test_length(.2) # 20% test split
        """
        if n == 0:
            self.test_length = 0
            return
        elif n >= 1:
            n = int(n)
            self.test_length = n
            return
        elif n < 1:
            n = float(n)
            self.test_length = int(len(self.current_dates) * n)
            return
        else:
            raise ValueError(f"n must be an int of at least 0 or float greater than 0 and less than 1, got {n} of type {type(n)}")

    def set_validation_length(self, n=1):
        """ Sets the length of the validation set. This will only matter when tuning a model.

        Args:
            n (int): Default 1.
                The length of the resulting validation set.

        Returns:
            None

        >>> mvf.set_validation_length(6) # validation length of 6
        """
        if n <= 0:
            raise ValueError(f"n must be greater than 0, got {n}")
        if (self.validation_metric == "r2") & (n == 1):
            raise ValueError(
                "can only set a validation_length of 1 if validation_metric is not r2. try set_validation_metric()"
            )
        self.validation_length = n

    def set_estimator(self, estimator):
        """ Sets the estimator to forecast with.

        Args:
            estimator (str): One of _sklearn_estimators_.

        Returns:
            None

        >>> mvf.set_estimator('mlr')
        """
        if estimator not in _sklearn_estimators_:
            raise ValueError(
                f"estimator must be one of {_sklearn_estimators_}, got {estimator}"
            )

        if hasattr(self, "estimator"):
            if estimator != self.estimator:
                for attr in (
                    "grid",
                    "grid_evaluated",
                    "best_params",
                    "validation_metric_value",
                ):
                    if hasattr(self, attr):
                        delattr(self, attr)
                self.estimator = estimator
        else:
            self.estimator = estimator

    def ingest_grid(self, grid):
        """ Ingests a grid to tune the estimator.

        Args:
            grid (dict or str):
                If dict, must be a user-created grid.
                If str, must match the name of a dict grid stored in a grids file.

        Returns:
            None

        >>> mvf.set_estimator('mlr')
        >>> mvf.ingest_grid({'normalizer':['scale','minmax']})
        """
        from itertools import product

        def expand_grid(d):
            return pd.DataFrame([row for row in product(*d.values())], columns=d.keys())

        try:
            if isinstance(grid, str):
                MVGrids = importlib.import_module(self.grids_file)
                importlib.reload(MVGrids)
                grid = getattr(MVGrids, grid)
        except SyntaxError:
            raise
        except:
            raise ForecastError.NoGrid(
                f"tried to load a grid called {self.estimator} from {self.grids_file}.py, "
                "but either the file could not be found in the current directory, "
                "there is no grid with that name, or the dictionary values are not list-like. "
                "try ingest_grid() with a dictionary grid passed manually."
            )
        grid = expand_grid(grid)
        self.grid = grid

    def set_validation_metric(self, metric="rmse"):
        """ Sets the metric that will be used to tune all subsequent models.

        Args:
            metric: One of _metrics_. Default 'rmse'.
                The metric to optimize the models with using the validation set.

        Returns:
            None

        >>> mvf.set_validation_metric('mae')
        """
        if metric not in _metrics_:
            raise ValueError(f"metric must be one of {_metrics_}, got {metric}")

        if (metric == "r2") & (self.validation_length < 2):
            raise ValueError(
                "can only validate with r2 if the validation length is at least 2, try calling set_validation_length()"
            )
        self.validation_metric = metric

    def limit_grid_size(self, n, random_seed=None):
        """ Makes a grid smaller randomly.

        Args:
            n (int or float):
                If int, randomly selects that many parameter combinations.
                If float, must be less than 1 and greater 0, randomly selects that percentage of parameter combinations.
            random_seed (int): Optional.
                Set a seed to make results consistent.

        Returns:
            None

        >>> from scalecast import GridGenerator
        >>> GridGenerator.get_example_grids()
        >>> mvf.set_estimator('mlp')
        >>> mvf.ingest_grid('mlp')
        >>> mvf.limit_grid_size(10,random_seed=20) # limits grid to 10 iterations
        >>> mvf.limit_grid_size(.5,random_seed=20) # limits grid to half its original size
        """
        if random_seed is not None:
            random.seed(random_seed)

        if n >= 1:
            self.grid = self.grid.sample(n=min(n, self.grid.shape[0])).reset_index(
                drop=True
            )
        elif (n < 1) & (n > 0):
            self.grid = self.grid.sample(frac=n).reset_index(drop=True)
        else:
            raise ValueError(f"argment passed to n not usable: {n}")

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
        _optimizer_funcs_[called] = func

    def set_grids_file(self,name='MVGrids'):
        """ Sets the name of the file where the object will look automatically for grids when calling 
        `tune()`, `cross_validate()`, `tune_test_forecast()`, or similar function.
        If the grids file does not exist in the working directory, the error will only be raised once tuning is called.
        
        Args:
            name (str): Default 'MVGrids.'
                The name of the file to look for.
                This file must exist in the working directory.
                The default will look for a file called "MVGrids.py".

        >>> mvf.set_grids_file('ModGrids') # expects to find a file called ModGrids.py in working directory.
        """
        descriptive_assert(isinstance(name,str),ValueError,f'name argument expected str type, got {type(name)}')
        self.grids_file = name

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
                        globals()[self.validation_metric](va, vp)
                    )
            except TypeError:
                raise
            except Exception as e:
                self.grid.drop(i, axis=0, inplace=True)
                logging.warning(f"Could not evaluate the paramaters: {hp}. error: {e}")
        metrics = pd.DataFrame(metrics)
        if metrics.shape[0] > 0:
            self.dynamic_tuning = dynamic_tuning
            self.grid.reset_index(drop=True, inplace=True)
            self.grid_evaluated = self.grid.copy()
            self.grid_evaluated["validation_length"] = self.validation_length
            self.grid_evaluated["validation_metric"] = self.validation_metric
            if self.optimize_on in _optimizer_funcs_:
                metrics["optimized_metric"] = metrics.apply(
                    _optimizer_funcs_[self.optimize_on], axis=1
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
        descriptive_assert(k >= 2, ValueError, f"k must be at least 2, got {k}")
        mvf = self.__deepcopy__()
        usable_obs = len(mvf.series1["y"]) - mvf.test_length
        val_size = usable_obs // (k + 1)
        descriptive_assert(
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
                f"None of the keyword/value combos stored in the grid could be evaluated for the {self.estimator} model.",
                category=Warning,
            )
            self.best_params = {}

    def auto_forecast(
        self, call_me=None, dynamic_testing=True,
    ):
        """ Auto forecasts with the best parameters indicated from the tuning process.

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

        Returns:
            None

        >>> mvf.set_estimator('xgboost')
        >>> mvf.tune()
        >>> mvf.auto_forecast()
        """
        if not hasattr(self, "best_params"):
            warnings.warn(
                f"Since tune() or cross_validate() has not been called, {self.estimator} model will be run with default hyperparameters.",
                category=Warning,
            )
            self.best_params = {}
        self.manual_forecast(
            call_me=call_me, dynamic_testing=dynamic_testing, **self.best_params
        )

    def manual_forecast(self, call_me=None, dynamic_testing=True, **kwargs):
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
        descriptive_assert(
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
        for m in models:
            call_me = m if suffix is None else m + suffix
            self.set_estimator(m)
            if limit_grid_size is not None:
                self.ingest_grid(m)
                self.limit_grid_size(limit_grid_size)
            if cross_validate:
                self.cross_validate(dynamic_tuning=dynamic_tuning, **cvkwargs)
            else:
                self.tune(dynamic_tuning=dynamic_tuning)
            try:
                self.auto_forecast(
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
                    raise ValueError(f'value passed to error arg not recognized: {error}')

    def set_optimize_on(self, how):
        """ Choose how to determine best models by choosing which series should be optimized.
        This is the decision that will be used for tuning models as well.

        Args:
            how (str): One of _optimizer_funcs_, 'series{n}', 'y{n}', or the series name.
                If in _optimizer_funcs_, will optimize based on that metric.
                If 'series{n}', 'y{n}', or the series name, will choose the model that performs best on that series.
                By default, optimize_on is set to 'mean' when the object is initiated.
        """
        if how in _optimizer_funcs_:
            self.optimize_on = how
        else:
            series, labels = self._parse_series(how)
            self.optimize_on = labels[0]
    
    @log_warnings
    def _forecast(
        self, fcster, dynamic_testing, tune=False, normalizer="minmax", lags=1, **kwargs
    ):
        """ Runs the vector multivariate forecast start-to-finish. All Xvars stored in the object are used always. All sklearn estimators supported.
        See example: https://scalecast-examples.readthedocs.io/en/latest/multivariate/multivariate.html.

        Args:
            fcster (str): One of _sklearn_estimators_. Reads the estimator set to `set_estimator()` method.
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
            normalizer (str): The scaling technique to apply to the data. One of _normalizer_. 
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
                        raise ValueError(f"cannot use argument for lags: {lags}")
                    else:
                        try:
                            lag = list(lag)
                        except TypeError:
                            raise ValueError(f"cannot use argument for lags: {lags}")
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
                    raise ValueError(f"lags cannot be str type, got {lags}")
                else:
                    try:
                        lags = list(lags)
                    except TypeError:
                        raise ValueError(f"cannot use argument for lags: {lags}")
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
            regr = _sklearn_imports_[fcster](**kwargs)
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

        descriptive_assert(
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
            X_train (DataFrame): the independent values.
            normalizer (str): one of _normalizer_.
                if 'minmax', uses the MinMaxScaler from sklearn.preprocessing.
                if 'scale', uses the StandardScaler from sklearn.preprocessing.
                if 'normalize', uses the Normalizer from sklearn.preprocessing.
                if None, returns None.

        Returns:
            (scikit-learn preprecessing scaler/normalizer): The normalizer fitted on training data only.
        """
        descriptive_assert(
            normalizer in _normalizer_,
            ValueError,
            f"normalizer must be one of {_normalizer_}, got {normalizer}",
        )
        X_train = X_train if not hasattr(X_train,'values') else X_train.values
        if normalizer == "minmax":
            from sklearn.preprocessing import MinMaxScaler as Scaler
        elif normalizer == "normalize":
            from sklearn.preprocessing import Normalizer as Scaler
        elif normalizer == "scale":
            from sklearn.preprocessing import StandardScaler as Scaler
        else:
            return None

        scaler = Scaler()
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
            "TestSetRMSE": {
                series: _return_na_if_len_zero(a, test_set_preds[series], rmse)
                for series, a in test_set_actuals.items()
            },
            "TestSetMAPE": {
                series: _return_na_if_len_zero(a, test_set_preds[series], mape)
                for series, a in test_set_actuals.items()
            },
            "TestSetMAE": {
                series: _return_na_if_len_zero(a, test_set_preds[series], mae)
                for series, a in test_set_actuals.items()
            },
            "TestSetR2": {
                series: _return_na_if_len_zero(a, test_set_preds[series], r2)
                for series, a in test_set_actuals.items()
            },
            "InSampleRMSE": {
                series: _return_na_if_len_zero(a, fitted_vals[series],rmse)
                for series, a in fitted_val_actuals.items()
            },
            "InSampleMAPE": {
                series: _return_na_if_len_zero(a, fitted_vals[series], mape)
                for series, a in fitted_val_actuals.items()
            },
            "InSampleMAE": {
                series: _return_na_if_len_zero(a, fitted_vals[series], mae)
                for series, a in fitted_val_actuals.items()
            },
            "InSampleR2": {
                series: _return_na_if_len_zero(a, fitted_vals[series], r2)
                for series, a in fitted_val_actuals.items()
            },
            "CILevel": self.cilevel,
            "ValidationSetLength": None,
            "ValidationMetric": None,
            "ValidationMetricValue": None,
            "grid_evaluated": None,
            "LevelForecast": lvl_fcst,
            "LevelTestSetPreds": lvl_tsp,
            "LevelTestSetActuals": lvl_tsa,
            "LevelTestSetRMSE": {
                series: _return_na_if_len_zero(a, lvl_tsp[series], rmse) for series, a in lvl_tsa.items()
            },
            "LevelTestSetMAPE": {
                series: _return_na_if_len_zero(a, lvl_tsp[series], mape) for series, a in lvl_tsa.items()
            },
            "LevelTestSetMAE": {
                series: _return_na_if_len_zero(a, lvl_tsp[series], mae) for series, a in lvl_tsa.items()
            },
            "LevelTestSetR2": {
                series: _return_na_if_len_zero(a, lvl_tsp[series], r2) for series, a in lvl_tsa.items()
            },
            "LevelFittedVals": lvl_fv,
            "LevelInSampleRMSE": {
                series: _return_na_if_len_zero(a, lvl_fv[series], rmse) for series, a in lvl_fva.items()
            },
            "LevelInSampleMAPE": {
                series: _return_na_if_len_zero(a, lvl_fv[series], mape) for series, a in lvl_fva.items()
            },
            "LevelInSampleMAE": {
                series: _return_na_if_len_zero(a, lvl_fv[series], mae) for series, a in lvl_fva.items()
            },
            "LevelInSampleR2": {
                series: _return_na_if_len_zero(a, lvl_fv[series], r2) for series, a in lvl_fva.items()
            },
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
            test_resids = {k:np.abs(np.array(p) - np.array(test_actuals[k])) for k, p in test_preds.items()}
            ci_range = {k:np.percentile(r, 100 * self.cilevel) for k,r in test_resids.items()}
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

    def _check_right_test_length_for_cis(self,cilevel):
        min_test_length = np.ceil(1/(1-cilevel))
        if self.test_length < min_test_length:
            raise ValueError(
                'Cannot evaluate confidence intervals at the '
                '{:.0%} level when test_length is set to less than {} observations. '
                'The test length must be at least 1/(1-cilevel) in length for conformal intervals to work.'.format(
                    cilevel,
                    int(min_test_length),
                )
            )

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

    def set_cilevel(self, n):
        """ Sets the level for the resulting confidence intervals (95% default).

        Args:
            n (float): Greater than 0 and less than 1.

        Returns:
            None

        >>> f.set_cilevel(.80) # next forecast will get 80% confidence intervals
        """
        descriptive_assert(
            n < 1 and n > 0, ValueError, "n must be a float greater than 0 and less than 1."
        )
        self.cilevel = n

    def set_best_model(self, model=None, determine_best_by=None):
        """ Sets the best model to be referenced as "best".
        One of model or determine_best_by parameters must be specified.

        Args:
            model (str): The model to set as the best.
                Must match the estimator name or call_me if that was used when evaluating the model.
            determine_best_by (str): One of _determine_best_by_.
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
            descriptive_assert(
                determine_best_by in _determine_best_by_,
                ValueError,
                f"determine_best_by must be one of {_determine_best_by_}, got {determine_best_by}.",
            )
            models_metrics = {m: v[determine_best_by] for m, v in self.history.items()}

            if self.optimize_on in _optimizer_funcs_ and determine_best_by != 'ValidationMetricValue':
                models_metrics = {
                    m: _optimizer_funcs_[self.optimize_on](list(v.values()))
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

    def _warn_about_not_finding_cis(self,m):
        warnings.warn(
            f'Confidence intervals not found for {m}.',
            category=Warning,
        )

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
                color=_series_colors_[i],
            )
            for m in models:
                sns.lineplot(
                    x=self.future_dates.to_list(),
                    y=self.history[m]["Forecast"][s]
                    if not level
                    else self.history[m]["LevelForecast"][s],
                    label=f"{labels[i]} {m}",
                    color=_colors_[k],
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
                            color=_colors_[k],
                            label="{} {} {:.0%} CI".format(
                                labels[i], m, self.history[m]["CILevel"]
                            ),
                        )
                    except KeyError:
                        _warn_about_not_finding_cis(m)
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
            ci (bool): Default False.
                Whether to display the confidence intervals.
            ax (Axis): Optional. The existing axis to write the resulting figure to.
            figsize (tuple): Default (12,6). Size of the resulting figure. Ignored when ax is not None.


        Returns:
            (Axis): The figure's axis.

        >>> mvf.plot_test_set() # plots all test set predictions on all series
        >>> plt.show()
        """
        descriptive_assert(
            self.test_length > 0,
            ForecastError,
            'plot_test_set() does not work when models were not tested (test_length set to 0).',
        )

        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
        
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
                color=_series_colors_[i],
            )
            for m in models:
                sns.lineplot(
                    x=self.current_dates.to_list()[-self.test_length :],
                    y=self.history[m]["TestSetPredictions"][s]
                    if not level
                    else self.history[m]["LevelTestSetPreds"][s],
                    label=f"{labels[i]} {m}",
                    color=_colors_[k],
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
                            color=_colors_[k],
                            label="{} {} {:.0%} CI".format(
                                labels[i], m, self.history[m]["CILevel"]
                            ),
                        )
                    except KeyError:
                        _warn_about_not_finding_cis(m)
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
                color=_series_colors_[i],
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
                    color=_colors_[k],
                    ax=ax,
                )
                k += 1

    def export(
        self,
        dfs=[
            "model_summaries",
            "all_fcsts",
            "test_set_predictions",
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
                ['all_fcsts','model_summaries','test_set_predictions','lvl_test_set_predictions','lvl_fcsts'].
                A list or name of the specific dataframe(s) you want returned and/or written to excel.
                Must be one of or multiple of default.
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
        descriptive_assert(
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
            cols = [
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
                "TestSetRMSE",
                "TestSetMAPE",
                "TestSetMAE",
                "TestSetR2",
                "LastTestSetPrediction",
                "LastTestSetActual",
                "CILevel",
                "InSampleRMSE",
                "InSampleMAPE",
                "InSampleMAE",
                "InSampleR2",
                "ValidationSetLength",
                "ValidationMetric",
                "ValidationMetricValue",
                "LevelTestSetRMSE",
                "LevelTestSetMAPE",
                "LevelTestSetMAE",
                "LevelTestSetR2",
                "LevelInSampleRMSE",
                "LevelInSampleMAPE",
                "LevelInSampleMAE",
                "LevelInSampleR2",
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
                                model_summary_sm[c] = [attr[s]]
                        elif c == "LastTestSetPrediction":
                            model_summary_sm[c] = [
                                self.history[m]["TestSetPredictions"][s][-1]
                            ]
                        elif c == "LastTestSetActual":
                            model_summary_sm[c] = [
                                self.history[m]["TestSetActuals"][s][-1]
                            ]
                        elif c == "OptimizedOn" and hasattr(self, "best_model"):
                            if self.optimize_on in _optimizer_funcs_:
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
            df = pd.DataFrame({"DATE": self.future_dates.to_list()})
            for l, s in zip(labels, series):
                for m in models:
                    df[f"{l}_{m}_fcst"] = self.history[m]["Forecast"][s][:]
                    if cis:
                        try:
                            df[f"{l}_{m}_fcst_upper"] = self.history[m]["UpperCI"][s][:]
                            df[f"{l}_{m}_fcst_lower"] = self.history[m]["LowerCI"][s][:]
                        except KeyError:
                            _warn_about_not_finding_cis(m)
            output["all_fcsts"] = df
        if "test_set_predictions" in dfs:
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
                            _warn_about_not_finding_cis(m)
                i += 1
            output["test_set_predictions"] = df
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
                            _warn_about_not_finding_cis(m)
            output["lvl_fcsts"] = df
        if "lvl_test_set_predictions" in dfs:
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
                            _warn_about_not_finding_cis(m)
                i += 1
            output["lvl_test_set_predictions"] = df
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
                Whether to export level fitted values

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
        All metrics are specific to level data.
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
        if fcst_length == "auto":
            fcst_length = len(self.future_dates)
        fcst_length = int(fcst_length)
        series, labels = self._parse_series("all")
        mets = ["RMSE", "MAE", "R2", "MAPE"]
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
                        test_mets["Series"] == s, f"LevelTestSet{m}"
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
            warnings.warn(f"{kwargs} ignored.",category=Warning)
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

_series_colors_ = [
    "#0000FF",
    "#00FFFF",
    "#7393B3",
    "#088F8F",
    "#0096FF",
    "#F0FFFF",
    "#00FFFF",
    "#5D3FD3",
    "#191970",
    "#9FE2BF",
] * 100

_optimizer_funcs_ = {
    "mean": np.mean,
    "min": np.min,
    "max": np.max,
}