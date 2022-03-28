import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import logging
from scipy import stats
from scalecast.Forecaster import (
    mape,
    rmse,
    mae,
    r2,
    _sklearn_imports_,
    _sklearn_estimators_,
    _metrics_,
    descriptive_assert,
    _normalizer_,
    _determine_best_by_,
    _colors_,
    ForecastError
)

# LOGGING
logging.basicConfig(filename="warnings.log", level=logging.WARNING)
logging.captureWarnings(True)

_series_colors_ = [
    '#0000FF',
    '#00FFFF', 
    '#7393B3', 
    '#088F8F', 
    '#0096FF', 
    '#F0FFFF', 
    '#00FFFF', 
    '#5D3FD3', 
    '#191970', 
    '#9FE2BF'
]*100

class MVForecaster:
    def __init__(self,
        *fs,
        not_same_len_action='trim',
        merge_Xvars='union',
        merge_future_dates='longest',
        names=None,
        **kwargs):
        """ __init__()

        Args:
            *fs (Forecaster): Forecaster objects
            not_same_len_action (str): one of 'trim', 'fail'. default 'trim'.
                what to do with series that are different lengths.
                'trim' will trim based on the most recent first date in each series.
                if the various series have different end dates, this option will still fail the initilization.
            merge_Xvars (str): one of 'union', 'u', 'intersection', 'i'. default 'union'.
                how to combine Xvars in each object.
                'union' or 'u' combines all regressors from each object.
                'intersection' or 'i' combines only regressors that all objects have in common.
            merge_future_dates (str): one of 'longest', 'shortest'. default 'longest'.
                which future dates to use in the various series.
            names (list-like): optional. an array with the same number of elements as *fs that can be used to map to each series.
                ex. names = ['UTUR','UNRATE']: can now refer to series1 and series2 with the user-selected names later. 
                if specific names are not supplied, refer to the series with series1 and series2 or y1 and y2.
                'series...' and 'y...' notation to refer to series are used interchangeably throughout the object.
                series can always be referred to with 'series...' and 'y...' notation, even if user-selected names are provided.
                the order the series are supplied will be maintained.
            **kwargs: become attributes.

        Returns:
            (MVForecaster): the object.
        """
        if len(set([len(f.current_dates) for f in fs])) > 1 or len(set([min(f.current_dates) for f in fs])) > 1:
            descriptive_assert(len(set([max(f.current_dates) for f in fs])) == 1,ForecastError,'series cannot have different end dates')
            if not_same_len_action == 'fail':
                raise ValueError('all series must be same length')
            elif not_same_len_action == 'trim':
                from scalecast.multiseries import keep_smallest_first_date
                keep_smallest_first_date(*fs)
            else:
                raise ValueError(f'not_same_len_action must be one of ("trim","fail"), got {not_same_len_action}')
        if len(set([min(f.current_dates) for f in fs])) > 1:
            raise ValueError('all obs must begin in same time period')
        if len(set([f.freq for f in fs])) > 1:
            raise ValueError('all date frequencies must be equal')
        if len(fs) < 2:
            raise ValueError('must pass at least two series')
        
        self.optimize_on = 'mean'
        self.estimator = 'mlr'
        self.current_xreg = {}
        self.future_xreg = {}
        self.history = {}
        self.test_length = 1
        self.validation_length = 1
        self.validation_metric = "rmse"
        self.cilevel = 0.95
        self.bootstrap_samples = 100
        self.freq = fs[0].freq
        self.n_series = len(fs)
        for i, f in enumerate(fs):
            setattr(self, f'series{i+1}', {'y':f.y.copy(),
                'levely':f.levely.copy(),
                'integration':f.integration})
            if i == 0:
                self.current_dates = f.current_dates.copy()
            if merge_Xvars in ('union','u'):
                if i == 0:
                    self.current_xreg = {k:v.copy() for k, v in f.current_xreg.items() if not k.startswith('AR')}
                    self.future_xreg = {k:v[:] for k, v in f.future_xreg.items() if not k.startswith('AR')}
                else:
                    for k, v in f.current_xreg.items():
                        if not k.startswith('AR'):
                            self.current_xreg[k] = v.copy()
                            self.future_xreg[k] = f.future_xreg[k][:]
            elif merge_Xvars in ('intersection','i'):
                if i == 0:
                    self.current_xreg = {k:v in f.current_xreg.items()}
                    self.future_xreg = {k:v in f.future_xreg.items()}
                else:
                    for k, v in f.current_xreg.items():
                        if k not in self.current_xreg.keys():
                            self.current_xreg.pop(k)
                            self.future_xreg.pop(k)
            else:
                raise ValueError(f"merge_Xvars must be one of ('union','u','intersection','i'), got {merge_Xvars}")

        future_dates_lengths = {i:len(f.future_dates) for i, f in enumerate(fs)}
        if merge_future_dates=='longest':
            self.future_dates = [fs[i].future_dates for i, v in future_dates_lengths.items() if v == max(future_dates_lengths.values())][0]
        elif merge_future_dates == 'shortest':
            self.future_dates = [fs[i].future_dates for i, v in future_dates_lengths.items() if v == min(future_dates_lengths.values())][0]
        else:
            raise ValueError(f"merge_future_dates must be one of ('longest','shortest'), got {merge_future_dates}")
        
        self.integration = {f'y{i+1}':getattr(self,f'series{i+1}')['integration'] for i in range(self.n_series)}
        for key, value in kwargs.items():
            setattr(self, key, value)

        if names is not None:
            names = list(names)
            globals()['name_series_map'] = {names[i]:[f'series{i+1}',f'y{i+1}'] for i in range(self.n_series)}
            globals()['y_name_map'] = {f'y{i+1}':names[i] for i in range(self.n_series)}
            globals()['series_name_map'] = {f'series{i+1}':names[i] for i in range(self.n_series)}

    def __repr__(self):
        return """MVForecaster(
    DateStartActuals={}
    DateEndActuals={}
    Freq={}
    N_actuals={}
    N_series={}
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
)""".format(self.current_dates.values[0].astype(str),
            self.current_dates.values[-1].astype(str),
            self.freq,
            len(self.current_dates),
            self.n_series,
            len(self.future_dates),
            list(self.current_xreg.keys()),
            self.test_length,
            self.validation_length,
            self.validation_metric,
            list(self.history.keys()),
            self.cilevel,
            self.bootstrap_samples,
            self.estimator,
            self.optimize_on)

    def add_sklearn_estimator(self,imported_module,called):
        """ adds a new estimator from scikit-learn not built-in to the forecaster object that can be called using set_estimator().
        be careful to choose regression models only.
        
        Args:
            imported_module (sklearn regression model):
                the model from sklearn to add. must have already been imported locally.
                supports models from sklearn and sklearn APIs.
            called (str):
                the name of the estimator that can be called using set_estimator().

        Returns:
            None

        >>> from sklearn.linear_model import Lasso
        >>> mvf.add_sklearn_estimator(Lasso,called='lasso')
        >>> mvf.set_estimator('lasso')
        >>> mvf.ingest_grid({'alpha':[.1,.5,1,1.5,2],'lags':[1,2,3]})
        >>> mvf.tune()
        >>> mvf.auto_forecast()
        """
        globals()[called + '_'] = imported_module
        _sklearn_imports_[called] = globals()[called + '_']
        _sklearn_estimators_.append(called)
        _sklearn_estimators_.sort()

    def set_test_length(self, n=1):
        """ sets the length of the test set.

        Args:
            n (int or float): default 1.
                the length of the resulting test set.
                fractional splits are supported by passing a float less than 1 and greater than 0.

        Returns:
            None

        >>> mvf.set_test_length(12) # test set of 12
        >>> mvf.set_test_length(.2) # 20% test split
        """
        float(n)
        if n >= 1:
            descriptive_assert(
                isinstance(n, int), 
                ValueError, 
                f"n must be an int of at least 1 or float greater than 0 and less than 1, got {n}"
            )
            self.test_length = n
        else:
            descriptive_assert(
                n > 0, 
                ValueError,
                f"n must be an int of at least 1 or float greater than 0 and less than 1, got {n}"
            )
            self.test_length = int(len(self.current_dates) * n)

    def set_validation_length(self, n=1):
        """ sets the length of the validation set.

        Args:
            n (int): default 1.
                the length of the resulting validation set.

        Returns:
            None

        >>> mvf.set_validation_length(6) # validation length of 6
        """
        if n <= 0:
            raise ValueError(f"n must be greater than 1, got {n}")
        if (self.validation_metric == "r2") & (n == 1):
            raise ValueError(
                "can only set a validation_length of 1 if validation_metric is not r2. try set_validation_metric()"
            )
        self.validation_length = n

    def set_estimator(self, estimator):
        """ sets the estimator to forecast with.

        Args:
            estimator (str): one of _sklearn_estimators_

        Returns:
            None

        >>> mvf.set_estimator('mlr')
        """
        if estimator not in _sklearn_estimators_:
            raise ValueError(f'estimator must be one of {_sklearn_estimators_}, got {estimator}')

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

    def ingest_grid(self,grid):
        """ ingests a grid to tune the estimator.

        Args:
            grid (dict or str):
                if dict, must be a user-created grid.
                if str, must match the name of a dict grid stored in MVGrids.py.

        Returns:
            None

        >>> mvf.set_estimator('mlr')
        >>> mvf.ingest_grid({'normalizer':['scale','minmax']})
        """
        from itertools import product

        def expand_grid(d):
            return pd.DataFrame([row for row in product(*d.values())], columns=d.keys())

        if isinstance(grid, str):
            import MVGrids
            grid = getattr(MVGrids, grid)
        
        grid = expand_grid(grid)
        self.grid = grid

    def set_validation_metric(self, metric="rmse"):
        """ sets the metric that will be used to tune all subsequent models.

        Args:
            metric: one of _metrics_, default 'rmse'.
                the metric to optimize the models with using the validation set.

        Returns:
            None

        >>> mvf.set_validation_metric('mae')
        """
        if metric not in _metrics_:
            raise ValueError(f'metric must be one of {_metrics_}, got {metric}')

        if (metric == "r2") & (self.validation_length < 2):
            raise ValueError(
                "can only validate with r2 if the validation length is at least 2, try calling set_validation_length()"
            )
        self.validation_metric = metric

    def limit_grid_size(self, n, random_seed=None):
        """ makes a grid smaller randomly.

        Args:
            n (int or float):
                if int, randomly selects that many parameter combinations.
                if float, must be less than 1 and greater 0, randomly selects that percentage of parameter combinations.
            random_seed (int): optional.
                set a seed to make results consistent.

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

    def tune(self, dynamic_tuning=False):
        """ tunes the specified estimator using an ingested grid (ingests a grid from MVGrids.py with same name as the estimator by default).
        any parameters that can be passed as arguments to manual_forecast() can be tuned with this process.

        Args:
            dynamic_tuning (bool): default False.
                whether to dynamically tune the forecast (meaning AR terms will be propogated with predicted values).
                setting this to False means faster performance, but gives a less-good indication of how well the forecast will perform out x amount of periods.
                when False, metrics effectively become an average of one-step forecasts.

        Returns:
            None

        >>> mvf.set_estimator('xgboost')
        >>> mvf.tune()
        >>> mvf.auto_forecast()
        """
        if not hasattr(self, "grid"):
            try:
                self.ingest_grid(self.estimator)
            except SyntaxError:
                raise
            except:
                raise ForecastError.NoGrid(f'to tune, a grid must be loaded. tried to load a grid called {self.estimator}, but either the MVGrids.py file could not be found in the current directory, there is no grid with that name, or the dictionary values are not list-like. try ingest_grid() with a dictionary grid passed manually.')

        metrics = {f'y{i+1}_metric':[] for i in range(self.n_series)}
        for i, v in self.grid.iterrows():
            try:
                # returns a dict
                val_preds, val_ac = self._forecast(fcster=self.estimator,tune=True,dynamic_testing=dynamic_tuning,**v)
                for series, a in val_ac.items():
                    metrics[series + '_metric'].append(globals()[self.validation_metric](a,val_preds[series]))
            except TypeError:
                raise
            except Exception as e:
                self.grid.drop(i, axis=0, inplace=True)
                logging.warning(
                    f"could not evaluate the paramaters: {dict(v)}. error: {e}"
                )
        metrics = pd.DataFrame(metrics)
        if metrics.shape[0] > 0:
            self.grid.reset_index(drop=True,inplace=True)
            self.grid_evaluated = self.grid.copy()
            self.grid_evaluated["validation_length"] = self.validation_length
            self.grid_evaluated["validation_metric"] = self.validation_metric            
            if self.optimize_on == 'mean':
                metrics['optimized_metric'] = metrics.mean(axis=1)
            else:
                metrics['optimized_metric'] = metrics.iloc[:,(int(self.optimize_on.split('series')[-1]) - 1)]
            self.grid_evaluated = pd.concat([self.grid_evaluated,metrics],axis=1)
            if self.validation_metric == "r2":
                best_params_idx = self.grid.loc[
                    self.grid_evaluated["optimized_metric"]
                    == self.grid_evaluated["optimized_metric"].max()
                ].index.to_list()[0]
                self.best_params = dict(self.grid.loc[best_params_idx])
            else:
                best_params_idx = self.grid.loc[
                    self.grid_evaluated["optimized_metric"]
                    == self.grid_evaluated["optimized_metric"].min()
                ].index.to_list()[0]
                self.best_params = dict(self.grid.loc[best_params_idx])

            self.validation_metric_value = self.grid_evaluated.loc[
                best_params_idx, "optimized_metric"
            ]
        else:
            logging.warning(
                f"none of the keyword/value combos stored in the grid could be evaluated for the {self.estimator} model"
            )
            self.best_params = {}
        self.dynamic_tuning = dynamic_tuning

    def auto_forecast(self, call_me=None, dynamic_testing=True):
        """ auto forecasts with the best parameters indicated from the tuning process.

        Args:
            call_me (str): optional.
                what to call the model when storing it in the object's history dictionary.
                if not specified, the model's nickname will be assigned the estimator value ('mlp' will be 'mlp', etc.).
                duplicated names will be overwritten with the most recently called model.
            dynamic_testing (bool): default True.
                whether to dynamically test the forecast (meaning AR terms will be propogated with predicted values).
                setting this to False means faster performance, but gives a less-good indication of how well the forecast will perform out x amount of periods.
                when False, test-set metrics effectively become an average of one-step forecasts.

        Returns:
            None

        >>> mvf.set_estimator('xgboost')
        >>> mvf.tune()
        >>> mvf.auto_forecast()
        """
        if not hasattr(self, "best_params"):
            logging.warning(
                f"since tune() has not been called, {self.estimator} model will be run with default hyperparameters"
            )
            self.best_params = {}
        self.manual_forecast(call_me=call_me, dynamic_testing=dynamic_testing, **self.best_params)
        call_me = self.estimator if call_me is None else call_me
        if len(self.best_params) > 0:
            self.history[call_me]['Tuned'] = True if not self.dynamic_tuning else 'Dynamically'
            self.history[call_me]['ValidationSetLength'] = self.validation_length
            self.history[call_me]['ValidationMetric'] = self.validation_metric
            self.history[call_me]['ValidationMetricValue'] = self.validation_metric_value
            self.history[call_me]['grid_evaluated'] = self.grid_evaluated

    def manual_forecast(self, call_me=None, dynamic_testing=True, **kwargs):
        """ manually forecasts with the hyperparameters, normalizer, and lag selections passed as keywords.

        Args:
            call_me (str): optional.
                what to call the model when storing it in the object's history dictionary.
                if not specified, the model's nickname will be assigned the estimator value ('mlp' will be 'mlp', etc.).
                duplicated names will be overwritten with the most recently called model.
            dynamic_testing (bool): default True.
                whether to dynamically test the forecast (meaning AR terms will be propogated with predicted values).
                setting this to False means faster performance, but gives a less-good indication of how well the forecast will perform out x amount of periods.
                when False, test-set metrics effectively become an average of one-step forecasts.
            **kwargs: passed to the _forecast_{estimator}() method.
                can include lags and normalizer in addition to any given model's specific hyperparameters.

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

        if 'tune' in kwargs.keys():
            kwargs.pop('tune')
            logging.warning('tune argument will be ignored')

        self.call_me = self.estimator if call_me is None else call_me
        self.forecast = self._forecast(fcster=self.estimator, dynamic_testing=dynamic_testing, **kwargs)
        self._bank_history(**kwargs)

    def tune_test_forecast(
        self,
        models,
        dynamic_tuning=False,
        dynamic_testing=True
    ):
        """ iterates through a list of models, tunes them using grids in MVGrids.py, and forecasts them.

        Args:
            models (list-like):
                each element must be in _can_be_tuned_.
            dynamic_tuning (bool): default False.
                whether to dynamically tune the forecast (meaning AR terms will be propogated with predicted values).
                setting this to False means faster performance, but gives a less-good indication of how well the forecast will perform out x amount of periods.
                when False, metrics effectively become an average of one-step forecasts.
            dynamic_testing (bool): default True.
                whether to dynamically test the forecast (meaning AR terms will be propogated with predicted values).
                setting this to False means faster performance, but gives a less-good indication of how well the forecast will perform out x amount of periods.
                when False, test-set metrics effectively become an average of one-step forecasts.

        Returns:
            None

        >>> models = ('mlr','mlp','lightgbm')
        >>> mvf.tune_test_forecast(models,dynamic_testing=False,feature_importance=True)
        """
        descriptive_assert(
            os.path.isfile("./MVGrids.py"),
            FileNotFoundError,
            "MVGrids.py not found in working directory",
        )
        for m in models:
            self.set_estimator(m)
            self.tune(dynamic_tuning=dynamic_tuning)
            self.auto_forecast(dynamic_testing=dynamic_testing)

    def set_optimize_on(self, how):
        """ choose how to determine best models by choosing which series should be optimized.
        this is the decision that will be used for tuning models as well.

        Args:
            how (str): one of 'mean', 'series1', 'series2', ... , 'seriesn' or 'y1', 'y2', ... , 'yn' or the series name.
                if 'mean', will optimize based on the mean metric evaluated on all series.
                if 'series...', 'y...', or the series name, will choose the model that did the best on that series.
                by default, this is set to 'mean' when the object is initiated.
        """
        if how == 'mean':
            self.optimize_on = 'mean'
            return
        if 'name_series_map' in globals():
            how = name_series_map[how][0]
        descriptive_assert(how.startswith('series') or how.startswith('y') or how == 'mean',ValueError,f'value passed to how not usable: {how}')
        self.optimize_on = how if how.startswith('series') else 'series{}'.format(how.split('y')[-1])

    def _forecast(self, 
        fcster,
        dynamic_testing,
        tune=False,
        normalizer="minmax",
        lags=1,
        **kwargs):
        """ runs the vector multi-variate forecast start-to-finish. all Xvars used always. all sklearn estimators supported.
        see example: https://scalecast-examples.readthedocs.io/en/latest/multivariate/multivariate.html

        Args:
            fcster (str): one of _sklearn_estimators_.
            dynamic_testing (bool):
                whether to dynamically test the forecast (meaning AR terms will be propogated with predicted values).
                setting this to False means faster performance, but gives a less-good indication of how well the forecast will perform out x amount of periods.
                when False, test-set metrics effectively become an average of one-step forecasts.
            tune (bool): default False.
                whether the model is being tuned.
            normalizer (str): one of _normalizer_.
                if not None, normalizer applied to training data only to not leak.
            lags (int, list[int], dict[str,int|list[int]]): default 1.
                the lags to add from each series to forecast with.
                needs to use at least one lag (otherwise, use a univariate approach).
                if int, that many lags will be added for all series
                if list, each element must be ints, and only those lags will be added for each series.
                if dict, the key must be the user-selected series name, 'series{n}' or 'y{n}' and key is list or int.
            **kwargs: treated as model hyperparameters and passed to _sklearn_imports_[model]()

        Returns:
            (float or list): The evaluated metric value on the validation set if tuning a model otherwise, the list of predictions.
        
        >>> mvf.set_estimator('gbt')
        >>> mvf.manual_forecast(lags=3) # adds three lags for each series
        >>> mvf.manual_forecast(lags=[1,3]) # first and third lags added for each series
        >>> mvf.manual_forecast(lags={'y1':2,'y2':3}) # 2 lags added for first series, 3 lags for second
        >>> mvf.manual_forecast(lags={'series1':[1,3],'series2':3}) # first and third lag for first series, 3 lags for second
        """
        def prepare_data(lags):
            observed = pd.DataFrame(self.current_xreg)
            future = pd.DataFrame(self.future_xreg)
            for i in range(self.n_series):
                if str(lags).isnumeric() or isinstance(lags,float):
                    lags = int(lags)
                    for j in range(lags):
                        col = f'y{i+1}_lag{j+1}'
                        observed[col] = getattr(self,'series'+str(i+1))['y'].shift(j+1).values
                        future.loc[0,col]  = getattr(self,'series'+str(i+1))['y'].values[-(j+1)]
                elif isinstance(lags,dict):
                    series, labels = self._parse_series(lags.keys())
                    if 'y' + str(i+1) in series:
                        idx = series.index(f'y{i+1}')
                        lag = lags[labels[idx]]
                    else:
                        continue
                    if str(lag).isnumeric() or isinstance(lag,float):
                        lag = int(lag)
                        for j in range(lag):
                            col = f'y{i+1}_lag{j+1}'
                            observed[col] = getattr(self,'series'+str(i+1))['y'].shift(j+1).values
                            future.loc[0,col]  = getattr(self,'series'+str(i+1))['y'].values[-(j+1)]
                    elif isinstance(lag,str):
                        raise ValueError(f'cannot use argument for lags: {lags}')
                    else:
                        try:
                            lag = list(lag)
                        except TypeError:
                            raise ValueError(f'cannot use argument for lags: {lags}')
                        for j in lag:
                            col = f'y{i+1}_lag{j}'
                            observed[col] = getattr(self,'series'+str(i+1))['y'].shift(j).values
                            future.loc[0,col]  = getattr(self,'series'+str(i+1))['y'].values[-j]
                elif isinstance(lags,str):
                    raise ValueError(f'lags cannot be str type, got {lags}')
                else:
                    try:
                        lags = list(lags)
                    except TypeError:
                        raise ValueError(f'cannot use argument for lags: {lags}')
                    for j in lags:
                        col = f'y{i+1}_lag{j}'
                        observed[col] = getattr(self,'series'+str(i+1))['y'].shift(j).values
                        future.loc[0,col]  = getattr(self,'series'+str(i+1))['y'].values[-j]
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
            if not scaler is None:
                return scaler.transform(X)
            else:
                return X.values if hasattr(X, "values") else X

        def train(X, y, normalizer, **kwargs):
            self.scaler = self._parse_normalizer(X, normalizer)
            X = scale(self.scaler,X)
            regr = _sklearn_imports_[fcster](**kwargs)
            regr.fit(X,y)
            return regr

        def evaluate(trained_models,future,dynamic_testing):
            future = future.reset_index(drop=True)
            if not dynamic_testing:
                preds = {}
                future = scale(self.scaler,future)
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
                            if idx > -1:
                                future.loc[i+1,c] = preds[series][idx]
                            else:
                                future.loc[i+1,c] = getattr(self, f'series{s_num}')['y'].to_list()[idx]
            return preds

        test_length = (self.test_length + (self.validation_length if tune else 0))
        validation_length = self.validation_length
        observed, future = prepare_data(lags)

        # test the model
        trained = {}
        for i in range(self.n_series):
            trained[f'y{i+1}'] = train(X=observed.values[:-test_length].copy(),
                y=getattr(self,f'series{i+1}')['y'].values[-observed.shape[0]:-test_length].copy(),
                normalizer=normalizer,
                **kwargs)

        preds = evaluate(trained,
                         observed.iloc[-(test_length + validation_length):-validation_length,:] 
                         if tune else 
                         observed.iloc[-test_length:],
                         dynamic_testing)

        if tune:
            return (
                preds.copy(),
                {f'y{i+1}':getattr(self,f'series{i+1}')['y'].values[-(test_length + validation_length):-validation_length].copy() for i in range(self.n_series)}
            )

        trained_full = {}
        for i in range(self.n_series):
            trained_full[f'y{i+1}'] = train(X=observed.copy(),
                y=getattr(self,f'series{i+1}')['y'].values[-observed.shape[0]:].copy(),
                normalizer=normalizer,
                **kwargs)

        self.dynamic_testing = dynamic_testing
        self.test_set_pred = preds.copy()
        self.trained_models = trained_full
        self.fitted_values = evaluate(trained_full,observed.copy(),False)
        return evaluate(trained_full,future.copy(),True)

    def _parse_normalizer(self, X_train, normalizer):
        """ fits an appropriate scaler to training data that will then be applied to test and future data

        Args:
            X_train (DataFrame): the independent values.
            normalizer (str): one of _normalizer_.
                if 'minmax', uses the MinMaxScaler from sklearn.preprocessing.
                if 'scale', uses the StandardScaler from sklearn.preprocessing.
                if 'normalize', uses the Normalizer from sklearn.preprocessing.
                if 'pt', uses the PowerTransformer from sklearn.preprocessing.
                if None, returns None.

        Returns:
            (scikit-learn preprecessing scaler/normalizer): The normalizer fitted on training data only.
        """
        descriptive_assert(
            normalizer in _normalizer_,
            ValueError,
            f"normalizer must be one of {_normalizer_}, got {normalizer}",
        )
        if normalizer == "minmax":
            from sklearn.preprocessing import MinMaxScaler as Scaler
        elif normalizer == "normalize":
            from sklearn.preprocessing import Normalizer as Scaler
        elif normalizer == "scale":
            from sklearn.preprocessing import StandardScaler as Scaler
        elif normalizer == "pt":  # fixing an issue with 0.3.7
            try:
                from sklearn.preprocessing import PowerTransformer as Scaler
                scaler = Scaler()
                scaler.fit(X_train)
                return scaler
            except ValueError:
                logging.warning(
                    f"the pt normalizer did not work for the {self.estimator} model, defaulting to a StandardScaler"
                )
                normalizer = "scale"
                from sklearn.preprocessing import StandardScaler as Scaler
        else:
            return None

        scaler = Scaler()
        scaler.fit(X_train)
        return scaler

    def _bank_history(self, **kwargs):
        """ places all relevant information from the last evaluated forecast into the history dictionary attribute
            **kwargs: passed from each model, depending on how that model uses Xvars, normalizer, and other args
        """
        def find_cis(resids):
            random.seed(20)
            bootstrapped_resids = {series: np.random.choice(r, size=self.bootstrap_samples) for series, r in resids.items()}
            bootstrap_mean = {series: np.mean(r) for series, r in bootstrapped_resids.items()}
            bootstrap_std = {series: np.std(r) for series, r in bootstrapped_resids.items()}
            return {
                series: stats.norm.ppf(1 - (1 - self.cilevel) / 2) * bootstrap_std[series] + bootstrap_mean[series] for series in resids.keys()
            }

        def undiff(preds_orig,test=False):
            # self.seriesx['levely']
            # self.seriesx['integration']
            preds = {series: l[:] for series, l in preds_orig.items()}
            actuals_all = {series:getattr(self,f'series{i+1}')['levely'][:] for i, series in enumerate(preds.keys())}
            integration = {series:getattr(self,f'series{i+1}')['integration'] for i, series in enumerate(preds.keys())}
            if test:
                actuals = {series:ac[:-self.test_length] for series, ac in actuals_all.items()}
                test_set_actuals = {series:ac[-self.test_length:] for series, ac in actuals_all.items()}
            else:
                actuals = actuals_all.copy()
                test_set_actuals = None

            for series, p in preds.items():
                if integration[series] == 0:
                    continue
                elif integration == 2:
                    first_obs = actuals[series][-1] - actuals[series][-2]
                    preds[series].insert(0,first_obs)
                    preds[series] = list(np.cumsum(preds[series]))[1:]
                first_obs = actuals[series][-1] 
                preds[series].insert(0,first_obs)
                preds[series] = list(np.cumsum(preds[series]))[1:]
            return preds, test_set_actuals

        test_set_preds = self.test_set_pred.copy()
        test_set_actuals = {f'y{i+1}':getattr(self,f'series{i+1}')['y'].to_list()[-self.test_length:] for i in range(self.n_series)}
        fitted_vals = self.fitted_values.copy()
        fitted_val_actuals = {f'y{i+1}':getattr(self,f'series{i+1}')['y'].to_list()[-len(fitted_vals[f'y{i+1}']):] for i in range(self.n_series)}
        resids = {series: [fv - ac for fv, ac in zip(fitted_vals[series],act)] for series, act in fitted_val_actuals.items()}
        cis = find_cis(resids)
        fcst = self.forecast.copy()
        lvl_fcst, _ = undiff(fcst.copy())
        lvl_tsp, lvl_tsa = undiff(test_set_preds,test=True)
        self.history[self.call_me] = {
            "Estimator": self.estimator,
            "Xvars": list(self.current_xreg.keys()),
            "HyperParams": {k: v for k, v in kwargs.items() if k not in ("normalizer","lags")},
            "Lags": kwargs['lags'] if 'lags' in kwargs.keys() else 1,
            "Scaler": kwargs["normalizer"] if "normalizer" in kwargs.keys() else 'minmax',
            "Integration":self.integration,
            "Forecast": fcst,
            "UpperCI": {series:p+cis[series] for series, p in fcst.items()},
            "LowerCI": {series:p-cis[series] for series, p in fcst.items()},
            "Observations": len(self.current_dates),
            "FittedVals": fitted_vals,
            "Resids": resids,
            "Tuned": None,
            "DynamicallyTested": self.dynamic_testing,
            "TestSetLength": self.test_length,
            "TestSetPredictions": test_set_preds,
            "TestSetActuals": test_set_actuals,
            "TestSetRMSE": {series:rmse(a,test_set_preds[series]) for series, a in test_set_actuals.items()},
            "TestSetMAPE": {series:mape(a,test_set_preds[series]) for series, a in test_set_actuals.items()},
            "TestSetMAE": {series:mae(a,test_set_preds[series]) for series, a in test_set_actuals.items()},
            "TestSetR2": {series:r2(a,test_set_preds[series]) for series, a in test_set_actuals.items()},
            "TestSetUpperCI": {series:p+cis[series] for series, p in test_set_preds.items()},
            "TestSetLowerCI": {series:p-cis[series] for series, p in test_set_preds.items()},
            "InSampleRMSE": {series:rmse(a,fitted_vals[series]) for series, a in fitted_val_actuals.items()},
            "InSampleMAPE": {series:mape(a,fitted_vals[series]) for series, a in fitted_val_actuals.items()},
            "InSampleMAE": {series:mae(a,fitted_vals[series]) for series, a in fitted_val_actuals.items()},
            "InSampleR2": {series:r2(a,fitted_vals[series]) for series, a in fitted_val_actuals.items()},
            "CILevel": self.cilevel,
            "CIPlusMinus": cis,
            "ValidationSetLength": None,
            "ValidationMetric": None,
            "ValidationMetricValue": None,
            "grid_evaluated": None,
            "LevelForecast": lvl_fcst,
            "LevelTestSetPreds": lvl_tsp,
            "LevelTestSetRMSE": {series:rmse(a,lvl_tsp[series]) for series, a in lvl_tsa.items()},
            "LevelTestSetMAPE": {series:mape(a,lvl_tsp[series]) for series, a in lvl_tsa.items()},
            "LevelTestSetMAE": {series:mae(a,lvl_tsp[series]) for series, a in lvl_tsa.items()},
            "LevelTestSetR2": {series:r2(a,lvl_tsp[series]) for series, a in lvl_tsa.items()},
        }

    def set_best_model(self, model=None, determine_best_by=None):
        """ sets the best model to be referenced as "best".
        one of model or determine_best_by parameters must be specified.

        Args:
            model (str): the model to set as the best.
                must match the estimator name or call_me if that was used when evaluating the model.
            determine_best_by (str): one of _determine_best_by_.
                if model is specified, this will be ignored.

        Returns:
            None
        """
        if model is not None:
            if model in self.history.keys():
                self.best_model = model
            else:
                raise ValueError(f'cannot find {model} in history')
        else:
            descriptive_assert(
                determine_best_by in _determine_best_by_,
                ValueError,
                f"determine_best_by must be one of {_determine_best_by_}, got {determine_best_by}",
            )
            models_metrics = {m: v[determine_best_by] for m, v in self.history.items()}

            if self.optimize_on == 'mean':
                models_metrics = {m:np.mean(list(v.values())) for m, v in models_metrics.items()}
            else:
                models_metrics = {m:v['y{}'.format(self.optimize_on.split('series')[-1])] for m, v in models_metrics.items()}

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

    def _parse_series(self,series):
        """ returns list (series), list (labels)
        """
        def parse_str_arg(arg):
            if arg.startswith('series'):
                series = [f'y{i}'.format(arg.split('series')[-1])]
                labels = arg.copy()
            elif arg.startswith('y'):
                series = [arg]
                labels = series.copy()
            else:
                labels = [arg]
                series = [name_series_map[arg][1]]
            return series, labels

        if series == 'all':
            series = [f'y{i+1}' for i in range(self.n_series)]
            if 'name_series_map' in globals():
                labels = list(name_series_map.keys())
            else:
                lables = series.copy()
        elif isinstance(series,str):
            series, labels = parse_str_arg(series)
        else:
            series1 = list(series)
            series = [parse_str_arg(s)[0][0] for s in series1]
            labels = [parse_str_arg(s)[1][0] for s in series1]
        return series, labels

    def _parse_models(self,models,put_best_on_top):
        if models == 'all':
            models = list(self.history.keys())
        elif isinstance(models,str):
            models = [models]
        else:
            models = list(models)
        if put_best_on_top:
            models = ([self.best_model] if self.best_model in models else []) + [m for m in models if m != self.best_model]
        return models

    def plot(self,
        models="all",
        series='all',
        put_best_on_top=False,
        level=False,
        ci=False):
        """ plots all forecasts with the actuals, or just actuals if no forecasts have been evaluated or are selected.

        Args:
            models (list-like or str): default 'all'.
               the forecasted models to plot.
               name of the model, 'all', or list-like of model names.
               'top_' and None not supported.
            series (list-like or str): default 'all'.
               the series to plot.
               name of the series ('y1...', 'series1...' or user-selected name), 'all', or list-like of model names.
            put_best_on_top (bool): only set to True if you have previously called set_best_model().
                if False, ignored.
            level (bool): default False.
                if True, will always plot level forecasts.
                if False, will plot the forecasts at whatever level they were called on.
                if False and there are a mix of models passed with different integrations, will default to True.
            ci (bool): default False.
                whether to display the confidence intervals.
                change defaults by calling `set_cilevel()` and `set_bootstrapped_samples()` before forecasting.
                ignored when level = True.

        Returns:
            (Figure): the created figure

        >>> mvf.plot() # plots all forecasts and all series
        >>> plt.show()
        """
        series, labels = self._parse_series(series)
        models = self._parse_models(models,put_best_on_top)
        integration = [v for s,v in self.integration.items() if s in series]
        level = True if len(set(integration)) > 1 else level
        _, ax = plt.subplots()

        k = 0
        for i, s in enumerate(series):
            sns.lineplot(x=self.current_dates.to_list(),
                y=getattr(self,'series{}'.format(s.split('y')[-1])).to_list() if not level else getattr(self,f'series{i+1}')['levely'][-len(self.current_dates):],
                label = f'{labels[i]} actual',
                ax=ax,
                color = _series_colors_[i])
            for m in models:
                sns.lineplot(x=self.future_dates.to_list(),
                    y = self.history[m]['Forecast'][s] if not level else self.history[m]['LevelForecast'][s],
                    label = f'{labels[i]} {m}',
                    color=_colors_[k],
                    ax=ax)

                if ci and not level:
                    plt.fill_between(
                        x=self.future_dates.to_list(),
                        y1=self.history[m]["UpperCI"][s],
                        y2=self.history[m]["LowerCI"][s],
                        alpha=0.2,
                        color=_colors_[k],
                        label="{} {} {:.0%} CI".format(labels[i], m, self.history[m]["CILevel"]),
                    )
                k += 1

        plt.legend()
        plt.xlabel("Date")
        plt.ylabel("Values")
        return ax

    def plot_test_set(self,
        models="all",
        series='all',
        put_best_on_top=False,
        include_train=True,
        level=False,
        ci=False):
        """  plots all test-set predictions with the actuals.

        Args:
            models (list-like or str): default 'all'.
               the forecasted models to plot.
               name of the model, 'all', or list-like of model names.
               'top_' and None not supported.
            series (list-like or str): default 'all'.
               the series to plot.
               name of the series ('y1...', 'series1...' or user-selected name), 'all', or list-like of model names.
            put_best_on_top (bool): only set to True if you have previously called set_best_model().
                if False, ignored.
            include_train (bool or int): default True.
                use to zoom into training resultsl
                if True, plots the test results with the entire history in y.
                if False, matches y history to test results and only plots this.
                if int, plots that length of y to match to test results.
            level (bool): default False.
                if True, will always plot level forecasts.
                if False, will plot the forecasts at whatever level they were called on.
                if False and there are a mix of models passed with different integrations, will default to True.
            ci (bool): default False.
                whether to display the confidence intervals.
                change defaults by calling `set_cilevel()` and `set_bootstrapped_samples()` before forecasting.
                ignored when level = True.

        Returns:
            (Figure): the created figure

        >>> mvf.plot_test_set() # plots all test set predictions on all series
        >>> plt.show()
        """
        series, labels = self._parse_series(series)
        models = self._parse_models(models,put_best_on_top)
        integration = [v for s,v in self.integration.items() if s in series]
        level = True if len(set(integration)) > 1 else level
        include_train = len(self.current_dates) if include_train is True else self.test_length if include_train is False else include_train
        _, ax = plt.subplots()

        k = 0
        for i, s in enumerate(series):
            y = getattr(self,'series{}'.format(s.split('y')[-1]))['y'].to_list() if not level else getattr(self,s)['levely'][-len(self.current_dates):]
            sns.lineplot(x=self.current_dates.to_list()[-include_train:],
                y=y[-include_train:],
                label = f'{labels[i]} actual',
                ax=ax,
                color = _series_colors_[i])
            for m in models:
                sns.lineplot(x=self.current_dates.to_list()[-self.test_length:],
                    y = self.history[m]['TestSetPredictions'][s] if not level else self.history[m]['LevelTestSetPreds'][s],
                    label = f'{labels[i]} {m}',
                    color=_colors_[k],
                    linestyle="--",
                    alpha=0.7,
                    ax=ax)

                if ci and not level:
                    plt.fill_between(
                        x=self.current_dates.to_list()[-self.test_length:],
                        y1=self.history[m]["TestSetUpperCI"][s],
                        y2=self.history[m]["TestSetLowerCI"][s],
                        alpha=0.2,
                        color=_colors_[k],
                        label="{} {} {:.0%} CI".format(labels[i], m, self.history[m]["CILevel"]),
                    )
                k += 1

        plt.legend()
        plt.xlabel("Date")
        plt.ylabel("Values")
        return ax

    def plot_fitted(self,models="all",series='all'):
        """ plots fitted values with the actuals.

        Args:
            models (list-like or str): default 'all'.
               the forecasted models to plot.
               name of the model, 'all', or list-like of model names.
               'top_' and None not supported.
            series (list-like or str): default 'all'.
               the series to plot.
               name of the series ('y1...', 'series1...' or user-selected name), 'all', or list-like of model names.

        Returns:
            (Figure): the created figure

        >>> mvf.plot_fitted() # plots all fitted values on all series
        >>> plt.show()
        """
        series, labels = self._parse_series(series)
        models = self._parse_models(models,False)
        _, ax = plt.subplots()
        k = 0
        for i, s in enumerate(series):
            sns.lineplot(x=self.current_dates.to_list(),
                y=getattr(self,'series{}'.format(s.split('y')[-1]))['y'].to_list(),
                label = f'{labels[i]} actual',
                ax=ax,
                color = _series_colors_[i])
            for m in models:
                sns.lineplot(x=self.current_dates.to_list()[-len(self.history[m]['FittedVals'][s]):],
                    y = self.history[m]['FittedVals'][s],
                    label = f'{labels[i]} {m}',
                    color=_colors_[k],
                    ax=ax)
                k += 1

    def export_model_summaries(self,models="all",series='all'):
        """ exports a dataframe with information about each model and its performance on each series.

        Args:
            models (list-like or str): default 'all'.
               the forecasted models to plot.
               name of the model, 'all', or list-like of model names.
               'top_' and None not supported.
            series (list-like or str): default 'all'.
               the series to plot.
               name of the series ('y1...', 'series1...' or user-selected name), 'all', or list-like of model names.

        Returns:
            (DataFrame): the resulting model summaries.
        """
        series, labels = self._parse_series(series)
        models = self._parse_models(models,hasattr(self,'best_model'))
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
            "CIPlusMinus",
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
            "OptimizedOn",
            "MetricOptimized",
            "best_model",
        ]
        model_summaries = pd.DataFrame()
        for l, s in zip(labels,series):
            for m in models:
                model_summary_sm = pd.DataFrame({"Series": [l], "ModelNickname": [m]})
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
                        if not isinstance(attr,dict) or c in ('HyperParams','Lags'):
                            model_summary_sm[c] = [attr]
                        else:
                            model_summary_sm[c] = [attr[s]]
                    elif c == "LastTestSetPrediction":
                        model_summary_sm[c] = [self.history[m]["TestSetPredictions"][s][-1]]
                    elif c == "LastTestSetActual":
                        model_summary_sm[c] = [self.history[m]["TestSetActuals"][s][-1]]
                    elif c == "OptimizedOn" and hasattr(self,'best_model'): 
                        if self.optimize_on == 'mean':
                            model_summary_sm['OptimizedOn'] = ['mean']
                        elif 'series_name_map' in globals():
                            model_summary_sm['OptimizedOn'] = [series_name_map[self.optimize_on]]
                        else:
                            model_summary_sm['OptimizedOn'] = [self.optimize_on]
                        if hasattr(self,'optimize_metric'):
                            model_summary_sm['MetricOptimized'] = self.optimize_metric
                        model_summary_sm["best_model"] = m == self.best_model
                model_summaries = pd.concat(
                    [model_summaries, model_summary_sm], ignore_index=True
                )
        return model_summaries

    def export_fitted_vals(self,series='all',models='all'):
        """ exports a dataframe of fitted values and actuals

        Args:
            models (list-like or str): default 'all'.
               name of the model, 'all', or list-like of model names.
            series (list-like or str): default 'all'.
               the series to plot.
               name of the series ('y1...', 'series1...' or user-selected name), 'all', or list-like of model names.

        Returns:
            (DataFrame): the fitted values for all selected series and models.
        """
        series, labels = self._parse_series(series)
        models = self._parse_models(models,False)
        dfd = {'DATE':self.current_dates.to_list()}
        length = len(dfd['DATE'])
        i = 0
        for l,s in zip(labels,series):
            dfd[f'{l}_actuals'] = getattr(self,f'series{i+1}')['y'].to_list()
            for m in models:
                dfd[f'{l}_{m}_fvs'] = self.history[m]['FittedVals'][s][:]
                length = min(length,len(dfd[f'{l}_{m}_fvs']),len(dfd[f'{l}_actuals'])) 
            i+=1
        return pd.DataFrame({c:v[-length:] for c,v in dfd.items()})

    def export_forecasts(self,series='all',models='all',cis=False):
        """ exports a dataframe of forecasts at whatever level the forecast was performed.

        Args:
            models (list-like or str): default 'all'.
               name of the model, 'all', or list-like of model names.
            series (list-like or str): default 'all'.
               the series to plot.
               name of the series ('y1...', 'series1...' or user-selected name), 'all', or list-like of model names.
            cis (bool): default False.
                whether to include confidence intervals for each series/model.
                ignored if not True.

        Returns:
            (DataFrame): the forecasts for all selected series and models.
        """
        series, labels = self._parse_series(series)
        models = self._parse_models(models,False)
        df = pd.DataFrame({'DATE':self.future_dates.to_list()})
        for l,s in zip(labels,series):
            for m in models:
                df[f'{l}_{m}_fcst'] = self.history[m]['Forecast'][s][:]
                if cis:
                    df[f'{l}_{m}_upper'] = self.history[m]['UpperCI'][s][:]
                    df[f'{l}_{m}_lower'] = self.history[m]['LowerCI'][s][:]
        return df

    def export_test_set_preds(self,series='all',models='all',cis=False):
        """ exports a dataframe of test set preds and actuals.

        Args:
            models (list-like or str): default 'all'.
               name of the model, 'all', or list-like of model names.
            series (list-like or str): default 'all'.
               the series to plot.
               name of the series ('y1...', 'series1...' or user-selected name), 'all', or list-like of model names.
            cis (bool): default False.
                whether to include confidence intervals for each series/model.
                ignored if not True.

        Returns:
            (DataFrame): the test set preds for all selected series and models.
        """
        series, labels = self._parse_series(series)
        models = self._parse_models(models,False)
        df = pd.DataFrame({'DATE':self.current_dates.to_list()[-self.test_length:]})
        i = 0
        for l,s in zip(labels,series):
            df[f'{l}_actuals'] = getattr(self,f'series{i+1}')['y'].to_list()[-self.test_length:]
            for m in models:
                df[f'{l}_{m}_test_preds'] = self.history[m]['TestSetPredictions'][s][:]
                if cis:
                    df[f'{l}_{m}_upper'] = self.history[m]['TestSetUpperCI'][s][:]
                    df[f'{l}_{m}_lower'] = self.history[m]['TestSetLowerCI'][s][:]
            i+=1
        return df

    def export_level_forecasts(self,series='all',models='all'):
        """ exports a dataframe of level forecasts.

        Args:
            models (list-like or str): default 'all'.
               name of the model, 'all', or list-like of model names.
            series (list-like or str): default 'all'.
               the series to plot.
               name of the series ('y1...', 'series1...' or user-selected name), 'all', or list-like of model names.

        Returns:
            (DataFrame): the level forecasts for all selected series and models.
        """
        series, labels = self._parse_series(series)
        models = self._parse_models(models,False)
        df = pd.DataFrame({'DATE':self.future_dates.to_list()})
        for l,s in zip(labels,series):
            for m in models:
                df[f'{l}_{m}_fcst'] = self.history[m]['LevelForecast'][s][:]
        return df

    def export_level_test_set_preds(self,series='all',models='all'):
        """ exports a dataframe of level test set preds and actuals.

        Args:
            models (list-like or str): default 'all'.
               name of the model, 'all', or list-like of model names.
            series (list-like or str): default 'all'.
               the series to plot.
               name of the series ('y1...', 'series1...' or user-selected name), 'all', or list-like of model names.

        Returns:
            (DataFrame): the level forecasts for all selected series and models.
        """
        series, labels = self._parse_series(series)
        models = self._parse_models(models,False)
        df = pd.DataFrame({'DATE':self.current_dates.to_list()[-self.test_length:]})
        i = 0
        for l,s in zip(labels,series):
            df[f'{l}_actuals'] = getattr(self,f'series{i+1}')['levely'][-self.test_length:]
            for m in models:
                df[f'{l}_{m}_test_preds'] = self.history[m]['LevelTestSetPreds'][s][:]
            i+=1
        return df

    def export_validation_grid(self,model):
        """ exports a validation grid for a selected model.

        Args:
            model (str): the model to export the validation grid for.

        Returns:
            (DataFrame): the validation grid
        """
        return self.history[model]['grid_evaluated'].copy()
        