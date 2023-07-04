from .__init__ import (
    __estimators__,
    __can_be_tuned__,
    __not_hyperparams__,
    __colors__,
)
from ._utils import _developer_utils
from ._Forecaster_parent import (
    Forecaster_parent,
    ForecastError,
    _tune_test_forecast,
)
import typing
from typing import Union, Tuple, Dict
import importlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import logging
import warnings
import os
import math
import random
from collections import Counter
from scipy import stats
import sklearn
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose, STL
from sklearn.model_selection import train_test_split

class Forecaster(Forecaster_parent):
    def __init__(
        self, 
        y, 
        current_dates, 
        future_dates=None,
        test_length = 0,
        cis = False,
        metrics = ['rmse','mape','mae','r2'],
        **kwargs
    ):
        """ 
        Args:
            y (collection): An array of all observed values.
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
            metrics (list): Default ['rmse','mape','mae','r2']. The metrics to evaluate when validating
                and testing models. Each element must exist in utils.metrics and take only two arguments: a and f.
                See https://scalecast.readthedocs.io/en/latest/Forecaster/Util.html#metrics.
                Or the element should be a function that accepts two arguments that will be referenced later by its name.
                The first element of this list will be set as the default validation metric, but that can be changed.
                For each metric and model that is tested, the test-set and in-sample metrics will be evaluated and can be
                exported.
            **kwargs: Become attributes. 
        """
        super().__init__(
            y = y,
            test_length = test_length,
            cis = cis,
            metrics = metrics,
            **kwargs,
        )

        self.estimators = __estimators__
        self.can_be_tuned = __can_be_tuned__
        self.current_dates = current_dates
        self.future_dates = pd.Series([], dtype="datetime64[ns]")
        self.init_dates = list(current_dates)
        self.grids_file = 'Grids'
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
        len(self.y),
        len(self.future_dates),
        list(self.current_xreg.keys()),
        self.test_length,
        self.validation_metric,
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
    
    def _set_cis(self,*attrs,m,ci_range,forecast,tspreds):
        for i, attr in enumerate(attrs):
            self.history[m][attr] = [
                p + (ci_range if i%2 == 0 else (ci_range*-1))
                for p in (
                    forecast if i <= 1 else tspreds
                )
            ]

    def _bank_history(
        self, 
        **kwargs
        ):
        """ places all relevant information from the last evaluated forecast into the history dictionary attribute
            **kwargs: passed from each model, depending on how that model uses Xvars, normalizer, and other args
        """
        # since test only, what gets saved to history is relevant to the train set only, the genesis of the next line
        call_me = self.call_me
        self.history[call_me]['Estimator'] = self.estimator
        self.history[call_me]['Xvars'] = self.Xvars if hasattr(self,'Xvars') else None
        self.history[call_me]['HyperParams'] = {k: v for k, v in kwargs.items() if k not in __not_hyperparams__}
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

        for met, func in self.metrics.items():
            self.history[call_me]['InSample' + met.upper()] = _developer_utils._return_na_if_len_zero(
                self.y.iloc[-len(self.fitted_values) :], self.fitted_values, func
            )

        for attr in ("regr", "X", "models", "weights"):
            if hasattr(self, attr):
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
                "UpperCI",
                "LowerCI",
                "TestSetUpperCI",
                "TestSetLowerCI",
                m = call_me,
                ci_range = ci_range,
                forecast = fcst,
                tspreds = test_preds,
            )

    def _set_summary_stats(self):
        """ for every model where summary stats are available, saves them to a pandas dataframe where index is the regressor name
        """
        results_summary = self.regr.summary()
        results_as_html = results_summary.tables[1].as_html()
        self.summary_stats = pd.read_html(results_as_html, header=0, index_col=0)[0]

    def _bank_fi_to_history(self):
        """ for every model where ELI5 permutation feature importance can be extracted, saves that info to a pandas dataframe wehre index is the regressor name
        """
        call_me = self.call_me
        self.history[call_me]["feature_importance"] = self.feature_importance

    def _bank_summary_stats_to_history(self):
        """ saves summary stats (where available) to history
        """
        call_me = self.call_me
        self.history[call_me]["summary_stats"] = self.summary_stats

    def _warn_about_dynamic_testing(self,dynamic_testing,does_not_use_lags=False,uses_direct=False):
        if dynamic_testing is not True:
            warning = f'The dynamic_testing arg is always set to True for the {self.estimator} model.'
            warning += " This model doesn't use lags to make predictions." if does_not_use_lags else ''
            warning += " This model uses direct forecasting with lags." if uses_direct else ''
            warnings.warn(warning,category=Warning)
            self.dynamic_testing = True
    
    @_developer_utils.log_warnings
    def _forecast_sklearn(
        self,
        fcster,
        dynamic_testing=True,
        Xvars=None,
        normalizer="minmax",
        **kwargs,
    ):
        """ Runs an sklearn forecast start-to-finish.
        See the example: https://scalecast-examples.readthedocs.io/en/latest/sklearn/sklearn.html.

        Args:
            fcster (str): One of `Forecaster.sklearn_estimators`. Reads the estimator passed to the estimator attribute.
            dynamic_testing (bool or int): Default True.
                Whether to dynamically/recursively test the forecast (meaning AR terms will be propagated with predicted values).
                If True, evaluates recursively over the entire out-of-sample slice of data.
                If int, window evaluates over that many steps (2 for 2-step recurvie testing, 12 for 12-step, etc.).
                Setting this to False or 1 gives a less-good indication of how well the forecast will perform more than one period out.
            Xvars (list-like, str, or None): The regressors to predict with. By default, all will be used.
                Be sure to have added them to the Forecaster object first.
                None means all Xvars added to the Forecaster object will be used for sklearn estimators.
            normalizer (str): Default 'minmax'.
                The scaling technique to apply to the data. One of `Forecaster.normalizer`. 
            **kwargs: Treated as model hyperparameters and passed to the applicable sklearn estimator.
        """
        def evaluate_model(
            regr,
            steps,
            y,
            scaler,
            current_X,
            future_X,
            dynamic_testing,
            Xvars, # list of names to get correct positions
        ):
            # apply the normalizer fit on training data only
            X = self._scale(scaler, current_X)
            self.X = X # for feature importance setting
            regr.fit(X, y)

            has_ars = len([x for x in Xvars if x.startswith('AR')]) > 0

            if has_ars:
                actuals = self.actuals if hasattr(self,'actuals') else []
                peeks = [a if (i+1)%dynamic_testing == 0 else np.nan for i, a in enumerate(actuals)]

                series = self.y.to_list() # this is used to add peeking to the models
                preds = [] # this is used to produce the real predictions
                for i in range(steps):
                    p = self._scale(scaler,future_X[i,:].reshape(1,len(Xvars)))
                    pred = regr.predict(p)[0]
                    preds.append(pred)
                    if hasattr(self,'actuals') and (i+1) % dynamic_testing == 0:
                        series.append(peeks[i])
                    else:
                        series.append(pred)

                    if i == (steps-1):
                        break

                    for pos, x in enumerate(Xvars):
                        if x.startswith('AR'):
                            ar = int(x[2:])
                            future_X[i+1,pos] = series[-ar]
            else:
                p = self._scale(scaler,future_X)
                preds = regr.predict(p)

            return list(preds)

        
        _developer_utils.descriptive_assert(
            len(self.current_xreg.keys()) > 0,
            ForecastError,
            f"Need at least 1 Xvar to forecast with the {self.estimator} model.",
        )
        _developer_utils.descriptive_assert(
            isinstance(dynamic_testing, bool)
            | isinstance(dynamic_testing, int) & (dynamic_testing > -1),
            ValueError,
            f"dynamic_testing expected bool or non-negative int type, got {dynamic_testing}.",
        )
        return_fitted = not hasattr(self,'actuals') # save resources by not generating fitted values when only testing out of sample
        steps = len(self.future_dates)
        dynamic_testing = (
            1 if dynamic_testing is False 
            else steps + 1 if dynamic_testing is True
            else dynamic_testing
        )
        Xvars = list(self.current_xreg.keys()) if Xvars == 'all' or Xvars is None else list(Xvars)[:]
        # list of integers, each one representing the n/a values in each AR term
        ars = [int(x[2:]) for x in Xvars if x.startswith("AR")]
        # if using ARs, instead of foregetting those obs, ignore them with sklearn forecasts (leaves them available for other types of forecasts)
        obs_to_drop = max(ars) if len(ars) > 0 else 0
        y = self.y.values[obs_to_drop:].copy()

        current_X = np.array([self.current_xreg[x].values[obs_to_drop:].copy() for x in Xvars]).T
        future_X = np.array([np.array(self.future_xreg[x][:]) for x in Xvars]).T

        # get a list of Xvars, the y array, the X matrix, and the test size (can be different depending on if tuning or testing)
        self.regr = self.sklearn_imports[fcster](**kwargs)
        self.scaler = self._fit_normalizer(current_X, normalizer)
        self.Xvars = Xvars
        
        preds = evaluate_model(
            regr = self.regr,
            steps = steps,
            y = y,
            current_X = current_X,
            future_X = future_X,
            scaler = self.scaler,
            dynamic_testing = dynamic_testing,
            Xvars = self.Xvars,
        )

        self.fitted_values = list(self.regr.predict(self.X)) if return_fitted else []
        return preds

    @_developer_utils.log_warnings
    def _forecast_theta(
        self, dynamic_testing=True, **kwargs
    ):
        """ Forecasts with Four Theta from darts.
        See the example: https://scalecast-examples.readthedocs.io/en/latest/theta/theta.html.

        Args:
            dynamic_testing (bool): Default True.
                Always set to True for theta like all scalecast models that don't use lags.
            **kwargs: passed to the FourTheta() function from darts.
                See https://unit8co.github.io/darts/generated_api/darts.models.forecasting.theta.html.
        """
        from darts import TimeSeries
        from darts.models.forecasting.theta import FourTheta

        return_fitted = not hasattr(self,'actuals') # save resources by not generating fitted values when only testing out of sample

        self._warn_about_dynamic_testing(
            dynamic_testing=dynamic_testing,
            does_not_use_lags=True,
            uses_direct=False,
        )
        sns.set_palette("tab10")  # darts changes this

        y = self.y.to_list()
        d = self.current_dates.to_list()

        y = pd.Series(y, index=d)
        ts = TimeSeries.from_series(y)

        regr = FourTheta(**kwargs)
        regr.fit(ts)
        pred = regr.predict(len(self.future_dates))
        pred = [p[0] for p in pred.values()]
        if return_fitted:
            resid = [r[0] for r in regr.residuals(ts).values()]
            actuals = y[-len(resid) :]
            fvs = [r + a for r, a in zip(resid, actuals)]
        else:
            fvs = []

        self.fitted_values = fvs
        return pred
    
    @_developer_utils.log_warnings
    def _forecast_hwes(
        self, dynamic_testing=True, **kwargs
    ):
        """ Forecasts with Holt-Winters exponential smoothing.
        See the example: https://scalecast-examples.readthedocs.io/en/latest/hwes/hwes.html.

        Args:
            dynamic_testing (bool): Default True.
                Always set to True for HWES like all scalecast models that don't use lags.
            **kwargs: Passed to the HWES() function from statsmodels. `endog` passed automatically.
                See https://www.statsmodels.org/dev/generated/statsmodels.tsa.holtwinters.ExponentialSmoothing.html.
        """
        from statsmodels.tsa.holtwinters import ExponentialSmoothing as HWES
        self._warn_about_dynamic_testing(
            dynamic_testing=dynamic_testing,
            does_not_use_lags=True,
            uses_direct=False,
        )
        y = self.y.to_list()

        regr = HWES(self.y, dates=self.current_dates, freq=self.freq, **kwargs).fit(
            optimized=True, use_brute=True
        )
        pred = regr.predict(start=len(y), end=len(y) + len(self.future_dates) - 1)
        
        self.fitted_values = list(regr.fittedvalues)
        self.regr = regr
        return list(pred)

    @_developer_utils.log_warnings
    def _forecast_tbats(
        self, dynamic_testing=True, random_seed = None, **kwargs,
    ):
        """ Forecasts with TBATS.

        Args:
            dynamic_testing (bool): Default True.
                Always set to True for HWES like all scalecast models that don't use lags.
            random_seed (int): Optonal.
                Set a random seed for consistent results.
            **kwargs: Passed to the TBATS() function. See https://github.com/intive-DataScience/tbats/blob/master/examples/detailed_tbats.py.
                show_warnings arg set to True in scalecast and warnings are logged.
        """
        from tbats import TBATS

        if random_seed is not None:
            np.random.seed(random_seed)
        
        y = self.y.dropna().values.copy()
        steps = len(self.future_dates)

        regr = TBATS(show_warnings=True,**kwargs)
        regr = regr.fit(y)

        self.regr = regr
        self.fitted_values = regr.y_hat
        return list(regr.forecast(steps=steps))
    
    @_developer_utils.log_warnings
    def _forecast_arima(
        self, Xvars=None, dynamic_testing=True, **kwargs
    ):
        """ Forecasts with ARIMA (or AR, ARMA, SARIMA, SARIMAX).
        See the example: https://scalecast-examples.readthedocs.io/en/latest/arima/arima.html.

        Args:
            Xvars (list-like, str, or None): Default None. The regressors to predict with.
                None means no Xvars used (unlike sklearn models).
            dynamic_testing (bool): Default True.
                Always ignored in ARIMA - ARIMA in scalecast dynamically tests all models over the full forecast horizon using statsmodels.
            **kwargs: Passed to the ARIMA() function from statsmodels. `endog` and `exog` are passed automatically. 
                See https://www.statsmodels.org/devel/generated/statsmodels.tsa.arima.model.ARIMA.html.
        """
        from statsmodels.tsa.arima.model import ARIMA

        self._warn_about_dynamic_testing(dynamic_testing=dynamic_testing)
        y = self.y.values.copy()
        Xvars = (
            [x for x in self.current_xreg.keys() if not x.startswith("AR")]
            if Xvars == "all"
            else [x for x in Xvars if not x.startswith("AR")]
            if Xvars is not None
            else Xvars
        )
        current_X = (
            np.array([self.current_xreg[x].values.copy() for x in Xvars]).T 
            if Xvars is not None 
            else None
        )
        future_X = (
            np.array([np.array(self.future_xreg[x][:]) for x in Xvars]).T 
            if Xvars is not None 
            else None
        )
        regr = ARIMA(
            y,
            exog=current_X,
            dates=self.current_dates,
            freq=self.freq,
            **kwargs,
        ).fit()
        fcst = regr.predict(
            exog=future_X,
            start=len(y),
            end=len(y) + len(self.future_dates) - 1,
            typ="levels",
            dynamic=True,
        )
        self.fitted_values = list(regr.fittedvalues)
        self.Xvars = Xvars
        self.regr = regr
        return list(fcst)
    
    @_developer_utils.log_warnings
    def _forecast_prophet(
        self,
        Xvars=None,
        dynamic_testing=True,
        cap=None,
        floor=None,
        **kwargs,
    ):
        """ Forecasts with the Prophet model from the prophet library.
        See example: https://scalecast-examples.readthedocs.io/en/latest/prophet/prophet.html.

        Args:
            Xvars (list-like, str, or None): Default None. The regressors to predict with.
                None means no Xvars used (unlike sklearn models).
            dynamic_testing (bool): Default True.
                Always set to True for Prophet like all scalecast models that don't use lags.
            cap (float): Optional.
                Specific to Prophet when using logistic growth -- the largest amount the model is allowed to evaluate to.
            floor (float): Optional.
                Specific to Prophet when using logistic growth -- the smallest amount the model is allowed to evaluate to.
            **kwargs: Passed to the Prophet() function from prophet. 
                See https://facebook.github.io/prophet/docs/quick_start.html#python-api.
        """
        from prophet import Prophet

        return_fitted = not hasattr(self,'actuals')

        self._warn_about_dynamic_testing(
            dynamic_testing=dynamic_testing,
            does_not_use_lags=True,
        )
        
        X = pd.DataFrame(
            {k: v.to_list() for k, v in self.current_xreg.items() if not k.startswith("AR")}
        )
        p = pd.DataFrame(
            {
                k: v[-len(self.future_dates) :]
                for k, v in self.future_xreg.items()
                if not k.startswith("AR")
            }
        )
        Xvars = (
            [x for x in self.current_xreg.keys() if not x.startswith("AR")]
            if Xvars == "all"
            else [x for x in Xvars if not x.startswith("AR")]
            if Xvars is not None
            else []
        )
        if cap is not None:
            X["cap"] = cap
        if floor is not None:
            X["floor"] = floor
        X["y"] = self.y.to_list()
        X["ds"] = self.current_dates.to_list()
        p["ds"] = self.future_dates.to_list()
        model = Prophet(**kwargs)
        for x in Xvars:
            model.add_regressor(x)
        regr = Prophet(**kwargs)
        regr.fit(X)
        fcst = regr.predict(p)

        self.fitted_values = regr.predict(X)["yhat"].to_list() if return_fitted else []
        self.Xvars = Xvars
        self.regr = regr

        return fcst["yhat"].to_list()
    
    @_developer_utils.log_warnings
    def _forecast_silverkite(
        self, dynamic_testing=True, Xvars=None, **kwargs
    ):
        """ Forecasts with the silverkite model from LinkedIn greykite library.
        See the example: https://scalecast-examples.readthedocs.io/en/latest/silverkite/silverkite.html.

        Args:
            dynamic_testing (bool): Default True.
                Always True for silverkite. It can use lags but they are always far enough in the past to allow a direct forecast.
            Xvars (list-like, str, or None): The regressors to predict with.
                None means no Xvars used (unlike sklearn models).
            **kwargs: Passed to the ModelComponentsParam function from greykite.framework.templates.autogen.forecast_config.
        """
        from greykite.framework.templates.autogen.forecast_config import (
            ForecastConfig,
            MetadataParam,
            ModelComponentsParam,
            EvaluationPeriodParam,
        )
        from greykite.framework.templates.forecaster import Forecaster as SKForecaster

        return_fitted = not hasattr(self,'actuals')

        self._warn_about_dynamic_testing(
            dynamic_testing=dynamic_testing,
            does_not_use_lags=False,
            uses_direct=True,
        )
        
        Xvars = (
            [x for x in self.current_xreg.keys() if not x.startswith("AR")]
            if Xvars == "all"
            else [x for x in Xvars if not x.startswith("AR")]
            if Xvars is not None
            else []
        )

        def _forecast_sk(df, Xvars, validation_length, test_length, forecast_length):
            test_length = test_length if test_length > 0 else -(df.shape[0] + 1)
            validation_length = (
                validation_length if validation_length > 0 else -(df.shape[0] + 1)
            )
            pred_df = df.iloc[:-test_length, :].dropna() if self.test_length > 0 else df.dropna()
            if validation_length > 0:
                pred_df.loc[:-validation_length, "y"] = None
            metadata = MetadataParam(time_col="ts", value_col="y", freq=self.freq)
            components = ModelComponentsParam(
                regressors={"regressor_cols": Xvars} if Xvars is not None else None,
                **kwargs,
            )
            forecaster = SKForecaster()
            result = forecaster.run_forecast_config(
                df=pred_df,
                config=ForecastConfig(
                    forecast_horizon=forecast_length,
                    metadata_param=metadata,
                    evaluation_period_param=EvaluationPeriodParam(
                        cv_max_splits=0
                    ),  # makes it very much faster
                ),
            )
            return (
                result.forecast.df["forecast"].to_list(),
                result.model[-1].summary().info_dict["coef_summary_df"],
            )

        fcst_length = len(self.future_dates)
        ts_df = pd.DataFrame(
            {
                "ts": self.current_dates.to_list() + self.future_dates.to_list(),
                "y": self.y.to_list() + [None] * fcst_length,
            }
        )
        reg_df = pd.DataFrame(
            {x: self.current_xreg[x].to_list() + self.future_xreg[x] for x in Xvars}
        )
        df = pd.concat([ts_df, reg_df], axis=1)
        result = _forecast_sk(df, Xvars, 0, 0, fcst_length)
        self.summary_stats = result[1].set_index("Pred_col")
        self.fitted_values = result[0][:-fcst_length] if return_fitted else []
        self.Xvars = Xvars
        return result[0][-fcst_length:]

    @_developer_utils.log_warnings
    def _forecast_lstm(
        self,
        dynamic_testing=True,
        lags=1,
        lstm_layer_sizes=(8,),
        dropout=(0.0,),
        loss="mean_absolute_error",
        activation="tanh",
        optimizer="Adam",
        learning_rate=0.001,
        random_seed=None,
        plot_loss=False,
        **kwargs,
    ):
        """ Forecasts with a long-short term memory neural network from TensorFlow.
        Only regressor options are the series' own history (specified in the `lags` argument).
        The rnn estimator can employ an LSTM and use exegenous regressors.
        Always uses a minmax scaler on the inputs and outputs. The resulting point forecasts are unscaled.
        The model is saved in the tf_model attribute and a summary can be called by calling Forecaster.tf_model.summary().
        See the example: https://scalecast-examples.readthedocs.io/en/latest/lstm/lstm.html.
            
        Args:
            dynamic_testing (bool): Default True.
                Always True for lstm. The model uses a direct forecast.
            lags (int): Must be greater than 0. Default 1.
                The number of lags to train the model with.
            lstm_layer_sizes (list-like): Default (8,).
                The size of each lstm layer to add.
                The first element is for the input layer.
                The size of this array minus 1 will equal the number of hidden layers in the resulting model.
            dropout (list-like): Default (0.0,).
                The dropout rate for each lstm layer.
                Must be the same size as lstm_layer_sizes.
            loss (str): Default 'mean_absolute_error'.
                The loss function to minimize while traning the model.
                See available options here: https://www.tensorflow.org/api_docs/python/tf/keras/losses.
                Be sure to choose one that is suitable for regression tasks.
            activation (str): Default "tanh".
                The activation function to use in each LSTM layer.
                See available values here: https://www.tensorflow.org/api_docs/python/tf/keras/activations.
            optimizer (str): default "Adam".
                The optimizer to use when compiling the model.
                See available values here: https://www.tensorflow.org/api_docs/python/tf/keras/optimizers.
            learning_rate (float): Default 0.001.
                The learning rate to use when compiling the model.
            random_seed (int): Optional.
                Set a seed for consistent results.
                With tensorflow networks, setting seeds does not guarantee consistent results.
            plot_loss (bool): Default False.
                Whether to plot the LSTM loss function stored in history for each epoch.
                If validation_split is passed to kwargs, it will plot the validation loss as well.
                The resulting plot looks better if epochs > 1 passed to **kwargs.
            **kwargs: Passed to fit() and can include epochs, verbose, callbacks, validation_split, and more.
        """
        def convert_lstm_args_to_rnn(**kwargs):
            new_kwargs = {
                k: v
                for k, v in kwargs.items()
                if k not in ("lstm_layer_sizes", "dropout", "activation")
            }
            new_kwargs["layers_struct"] = [
                (
                    "LSTM",
                    {
                        "units": v,
                        "activation": kwargs["activation"],
                        "dropout": kwargs["dropout"][i],
                    },
                )
                for i, v in enumerate(kwargs["lstm_layer_sizes"])
            ]
            return new_kwargs

        new_kwargs = convert_lstm_args_to_rnn(
            dynamic_testing=dynamic_testing,
            lags=lags,
            lstm_layer_sizes=lstm_layer_sizes,
            dropout=dropout,
            loss=loss,
            activation=activation,
            optimizer=optimizer,
            learning_rate=learning_rate,
            random_seed=random_seed,
            plot_loss=plot_loss,
            **kwargs,
        )
        return self._forecast_rnn(**new_kwargs)

    @_developer_utils.log_warnings
    def _forecast_rnn(
        self,
        dynamic_testing=True,
        Xvars=None,
        lags=None,
        layers_struct=[("SimpleRNN", {"units": 8, "activation": "tanh"})],
        loss="mean_absolute_error",
        optimizer="Adam",
        learning_rate=0.001,
        random_seed=None,
        plot_loss_test=False,
        plot_loss=False,
        scale_X = True,
        scale_y = True,
        **kwargs,
    ):
        """ Forecasts with a recurrent neural network from TensorFlow, such as LSTM or simple recurrent.
        Not all features from tensorflow are available, but many of the most common ones for time series models are.
        This function accepts lags and external regressors as inputs.
        The model is saved in the tf_model attribute and a summary can be called by calling Forecaster.tf_model.summary().
        See the univariate example: https://scalecast-examples.readthedocs.io/en/latest/rnn/rnn.html
        and the multivariate example: https://scalecast-examples.readthedocs.io/en/latest/multivariate-beyond/mv.html#8.-LSTM-Modeling.

        Args:
            dynamic_testing (bool): Default True.
                Always True for rnn. The model uses a direct forecast.
            Xvars (list-like): Default None. The Xvars to train the models with. 
                By default, all regressors added to the Forecaster object will be used.
            lags (int): Alternative to Xvars. If wanting to train with lags only, specify this argument. If specified,
                Xvars is ignored.
            layers_struct (list[tuple[str,dict[str,Union[float,str]]]]): Default [('SimpleRNN',{'units':8,'activation':'tanh'})].
                Each element in the list is a tuple with two elements.
                First element of the list is the input layer (input_shape set automatically).
                First element of the tuple in the list is the type of layer ('SimpleRNN','LSTM', or 'Dense').
                Second element is a dict.
                In the dict, key is a str representing hyperparameter name: 'units','activation', etc.
                Val is hyperparameter value.
                See here for options related to SimpleRNN: https://www.tensorflow.org/api_docs/python/tf/keras/layers/SimpleRNN.
                For LSTM: https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM.
                For Dense: https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense.
            loss (str or tf.keras.losses.Loss): Default 'mean_absolute_error'.
                The loss function to minimize.
                See available options here: https://www.tensorflow.org/api_docs/python/tf/keras/losses.
                Be sure to choose one that is suitable for regression tasks.
            optimizer (str or tf Optimizer): Default "Adam".
                The optimizer to use when compiling the model.
                See available values here: https://www.tensorflow.org/api_docs/python/tf/keras/optimizers.
                If str, it will use the optimizer with default args.
                If type Optimizer, will use the optimizer exactly as specified.
            learning_rate (float): Default 0.001.
                The learning rate to use when compiling the model.
                Ignored if you pass your own optimizer with a learning rate.
            random_seed (int): Optional.
                Set a seed for consistent results.
                With tensorflow networks, setting seeds does not guarantee consistent results.
            plot_loss_test (bool): Default False.
                Whether to plot the loss trend stored in history for each epoch on the test set.
                If validation_split passed to kwargs, will plot the validation loss as well.
                The resulting plot looks better if epochs > 1 passed to **kwargs.
            plot_loss (bool): default False.
                whether to plot the loss trend stored in history for each epoch on the full model.
                if validation_split passed to kwargs, will plot the validation loss as well.
                looks better if epochs > 1 passed to **kwargs.
            scale_X (bool): Default True.
                Whether to scale the exogenous inputs with a minmax scaler.
            scale_y (bool): Default True.
                Whether to scale the endogenous inputs (lags), as well as the model output, with a minmax scaler.
                The results will automatically return unscaled.
            **kwargs: Passed to fit() and can include epochs, verbose, callbacks, validation_split, and more.
        """
        def plot_loss_rnn(history, title):
            plt.plot(history.history["loss"], label="train_loss")
            if "val_loss" in history.history.keys():
                plt.plot(history.history["val_loss"], label="val_loss")
            plt.title(title)
            plt.xlabel("epoch")
            plt.legend(loc="upper right")
            plt.show()
        
        def get_compiled_model(y):
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Dense, LSTM, SimpleRNN
            import tensorflow.keras.optimizers
            if isinstance(optimizer, str):
                local_optimizer = eval(f"tensorflow.keras.optimizers.{optimizer}")(
                    learning_rate=learning_rate
                )
            else:
                local_optimizer = optimizer
            for i, kv in enumerate(layers_struct):
                layer = locals()[kv[0]]

                if i == 0:
                    if kv[0] in ("LSTM", "SimpleRNN"):
                        kv[1]["return_sequences"] = len(layers_struct) > 1
                    model = Sequential(
                        [layer(**kv[1], input_shape=(n_timesteps, 1),)]
                    )
                else:
                    if kv[0] in ("LSTM", "SimpleRNN"):
                        kv[1]["return_sequences"] = not i == (len(layers_struct) - 1)
                        if kv[1]["return_sequences"]:
                            kv[1]["return_sequences"] = (
                                layers_struct[i + 1][0] != "Dense"
                            )

                    model.add(layer(**kv[1]))
            model.add(Dense(y.shape[1]))  # output layer

            # compile model
            model.compile(optimizer=local_optimizer, loss=loss)
            return model

        def process_y(
            y,
            lags,
            total_period,
        ):
            ymin = y.min()
            ymax = y.max()

            if scale_y:
                ylist = [(yi - ymin) / (ymax - ymin) for yi in y]
            else:
                ylist = [yi for yi in y]

            idx_end = len(y)
            idx_start = idx_end - total_period
            y_new = []

            while idx_start > 0:
                y_line = ylist[idx_start + lags : idx_start + total_period]
                y_new.append(y_line)
                idx_start -= 1

            return (
                np.array(y_new[::-1]),
                ymin,                      # for scaling lags
                ymax,
            )

        def process_X(
            Xvars,
            lags,
            forecast_length,
            ymin,
            ymax,
        ):
            def minmax_scale(x):
                return (x - ymin) / (ymax - ymin)

            X_lags = np.array([
                v.to_list() + self.future_xreg[k][:1] 
                for k,v in self.current_xreg.items() 
                if k in Xvars and k.startswith('AR')
            ]).T
            X_other = np.array([
                v.to_list() + self.future_xreg[k][:1] 
                for k,v in self.current_xreg.items() 
                if k in Xvars and not k.startswith('AR')
            ]).T

            X_lags_new = X_lags[lags:]
            X_other_new = X_other[lags:]
            
            # scale lags
            if len(X_lags_new) > 0 and scale_y:
                X_lags_new = np.vectorize(minmax_scale)(X_lags_new)
            # scale other regressors
            if len(X_other_new) > 0 and scale_X:
                X_other_train = X_other_new[:-1]
                scaler = self._fit_normalizer(X_other_train,'minmax')
                X_other_new = self._scale(scaler,X_other_new)
                
            # combine
            if len(X_lags_new) > 0 and len(X_other_new) > 0:
                X = np.concatenate([X_lags_new,X_other_new],axis=1)
            elif len(X_lags_new) > 0:
                X = X_lags_new
            else:
                X = X_other_new

            fut = X[-1:]
            X = X[:-1]

            return X, fut

        _developer_utils.descriptive_assert(
            len(layers_struct) >= 1,
            ValueError,
            f"Must pass at least one layer to layers_struct, got {layers_struct}.",
        )

        return_fitted = not hasattr(self,'actuals')

        self._warn_about_dynamic_testing(
            dynamic_testing=dynamic_testing,
            does_not_use_lags=False,
            uses_direct=True,
        )
        if random_seed is not None:
            np.random.seed(random_seed)
        if lags is None:
            Xvars = list(self.current_xreg.keys()) if Xvars is None or Xvars == 'all' else Xvars
            lags = len([x for x in Xvars if x.startswith('AR')])
            if len(Xvars) == 0:
                raise ForecastError(f"Need at least 1 Xvar to forecast with the {self.estimator} model.")
        else:
            self.add_ar_terms(lags)
            Xvars = [k for k in self.current_xreg if k.startswith('AR') and int(k[2:]) <= lags]

        needed_obs = lags + len(self.future_dates) + 1
        if needed_obs > len(self.y):
            suggested_lags = len(self.y) - len(self.future_dates) - 1
            err_message = (
                f'Not enough observations to run the {self.estimator} model! ' 
                f'Need at least {needed_obs} obserations when using {lags} lags to predict a future horizon of {len(self.future_dates)}. '
            )
            if suggested_lags > 0:
                err_message += f'Try reducing the number of lags to {len(self.y) - len(self.future_dates) - 1}.'
            else:
                err_message += 'Remember, testing the model subtracts that many observations from the total, so reducing the size of the test set may also be necessary.'

            raise ValueError(err_message)
        
        forecast_length = len(self.future_dates)
        total_period = lags + forecast_length
        y, ymin, ymax = process_y(
            y = self.y.values.copy(),
            lags = lags,
            total_period = total_period, 
        )
        X, fut = process_X(
            Xvars = Xvars,
            lags = lags,
            forecast_length = forecast_length,
            ymin = ymin,
            ymax = ymax,
        )
        X = X[1:-(forecast_length-1)] if forecast_length > 1 else X[1:]
        #print('last X train:',X[-1])
        #print('last y train:',y[-1])
        #print('fut:',fut[-1])
        n_timesteps = X.shape[1]
        X = X.reshape(X.shape[0], X.shape[1], 1)
        fut = fut.reshape(fut.shape[0], fut.shape[1], 1)
        model = get_compiled_model(y)
        hist = model.fit(X, y, **kwargs)
        fcst = model.predict(fut)
        if return_fitted:
            fvs = model.predict(X)
            fvs =  [p[0] for p in fvs[:-1]] + [p for p in fvs[-1]] 
        else:
            fvs = []
        fcst = [p for p in fcst[0]]
        if scale_y:
            fvs = [p * (ymax - ymin) + ymin for p in fvs]
            fcst = [p * (ymax - ymin) + ymin for p in fcst]
        if plot_loss:
            plot_loss_rnn(hist, f"{self.estimator} model loss")
        self.tf_model = model
        self.fitted_values = fvs
        return fcst

    @_developer_utils.log_warnings
    def _forecast_naive(
            self,
            seasonal=False,
            m='auto',
            **kwargs,
        ):
            """ Forecasts with a naive estimator, meaning the last observed value is propagated forward for non-seasonal models
            or the last m-periods are propagated forward where m is the length of the seasonal cycle.

            Args:
                seasonal (bool): Default False. Whether to use a seasonal naive model.
                m (str or int): Default 'auto'. Ignored when seasonal is False.
                    The number of observations that counts one seasonal step.
                    When 'auto', uses the M4 competition values: 
                    for Hourly: 24, Monthly: 12, Quarterly: 4. everything else gets 1 (no seasonality assumed)
                    so pass your own values for other frequencies.
                **kwargs: Not used but added to the model so it doesn't fail.
            """
            self.dynamic_testing = True
            return_fitted = not hasattr(self,'actuals') # save resources by not generating fitted values when only testing out of sample

            m = _developer_utils._convert_m(m,self.freq) if seasonal else 1
            obs = self.y.values.copy()
            fcst = (pd.Series(obs).to_list()[-m:] * int(np.ceil(len(self.future_dates)/m)))[:len(self.future_dates)]
            fvs = pd.Series(obs).shift(m).dropna().to_list() if return_fitted else []
            
            self.fitted_values = fvs
            return fcst
    
    @_developer_utils.log_warnings
    def _forecast_combo(
        self,
        how="simple",
        models="all",
        dynamic_testing=True,
        determine_best_by="ValidationMetricValue",
        rebalance_weights=0.1,
        weights=None,
        splice_points=None,
    ):
        """ Combines at least previously evaluted forecasts to create a new model.
        One-model combinations are supported to facilitate auto-selecting models.
        This model is always applied to previously evaluated models' test sets and cannot be tuned. 
        It will fail if models in the combination used different test lengths. 
        See the following explanation for the weighted-average model:
        The weighted model in scalecast uses a weighted average of all selected models, 
        applying the same weights to the fitted values, test-set metrics, and predictions. 
        A user can supply their own weights or let the algorithm determine optimal weights 
        based on a passed error metric (such as "TestSetMAPE"). To avoid leakage, it is 
        recommended to use the default value, "ValidationMetricValue" to determine weights, 
        although this is not possible if the selected models have not all been tuned on the 
        validation set. The weighting uses a MaxMin scaler when an error metric is passed, 
        and a MinMax scaler when r-squared is selected as the metric to base weights on. 
        When this scaler is applied, the resulting values are then rebalanced to add to 1. 
        Since the worst-performing model in this case will always be weighted zero, 
        the user can select a factor to add to all scaled values before the rebalancing 
        is applied; by default, this is 0.1. The higher this factor is, the closer the weighted 
        average will be to a simple average and vice-versa.
        See the example: https://scalecast-examples.readthedocs.io/en/latest/combo/combo.html.

        Args:
            how (str): One of {'simple','weighted','splice'}. Default 'simple'.
                The type of combination.
                If 'simple', uses a simple average.
                If 'weighted', uses a weighted average.
                If 'splice', splices several forecasts together at specified splice points.
            models (list-like or str): Default 'all'.
                Which models to combine.
                Can start with top ('top_5').
            dynamic_testing (bool): Default True.
                Always set to True for combo.
            determine_best_by (str): One of Forecaster.determine_best_by, default 'ValidationMetricValue'.
                If models does not start with 'top_' and how is not 'weighted', this is ignored.
                If how is 'weighted' and manual weights are specified, this is ignored.
            rebalance_weights (float): Default 0.1.
                How to rebalance the weights when how = 'weighted'.
                The higher, the closer the weights will be to each other for each model.
                If 0, the worst-performing model will be weighted with 0.
                Must be greater than or equal to 0.
            weights (list-like): Optional.
                Only applicable when how='weighted'.
                Manually specifies weights.
                Must be the same size as models.
                If None and how='weighted', weights are set automatically.
                If manually passed weights do not add to 1, will rebalance them.
            splice_points (list-like): Optional.
                Only applicable when how='splice'.
                Elements in array must be parsable by pandas' Timestamp function.
                Must be exactly one less in length than the number of models.
                models[0] --> :splice_points[0]
                models[-1] --> splice_points[-1]:
        """
        self.dynamic_testing = True
        
        determine_best_by = (
            determine_best_by
            if (weights is None) & ((models[:4] == "top_") | (how == "weighted"))
            else None
            if how != "weighted" or len(weights) > 0
            else determine_best_by
        )
        minmax = (
            (str(determine_best_by).endswith("R2"))
            | (
                (determine_best_by == "ValidationMetricValue")
                & (self.validation_metric.upper() == "R2")
            )
            | (weights is not None)
        )
        models = self._parse_models(models, determine_best_by)
        models = [m for m in models if m != self.call_me]

        if len(models) == 1:
            how = 'simple'

        fcsts = pd.DataFrame({m: self.history[m]["Forecast"] for m in models})
        preds = pd.DataFrame({
            m: (
                self.history[m]["TestSetPredictions"] 
                if 'TestSetPredictions' in self.history[m] else []
            ) 
            for m in models
        })
        obs_to_keep = min(len(self.history[m]["FittedVals"]) for m in models)
        fvs = pd.DataFrame(
            {m: self.history[m]["FittedVals"][-obs_to_keep:] for m in models}
        )
        actuals = self.y.to_list()[-preds.shape[0] :]
        if how == "weighted":
            scale = True
            if weights is None:
                weights = pd.DataFrame(
                    {m: [self.history[m][determine_best_by]] for m in models}
                )
            else:
                _developer_utils.descriptive_assert(
                    len(weights) == len(models),
                    ForecastError,
                    f"Must pass as many weights as models. Got {len(weights)} weights and {len(models)} models."
                )
                _developer_utils.descriptive_assert(
                    not isinstance(weights, str),
                    TypeError,
                    f"Value passed to the weights argument cannot be used: {weights}.",
                )
                weights = pd.DataFrame(zip(models, weights)).set_index(0).transpose()
                if weights.sum(axis=1).values[0] == 1:
                    scale = False
                    rebalance_weights = 0
            try:
                _developer_utils.descriptive_assert(
                    rebalance_weights >= 0,
                    ValueError,
                    "When using a weighted average, rebalance_weights must be numeric and at least 0 in value.",
                )
                if scale:
                    if minmax:
                        weights = (weights - weights.min(axis=1).values[0]) / (
                            weights.max(axis=1).values[0]
                            - weights.min(axis=1).values[0]
                        )  # minmax scaler
                    else:
                        weights = (weights - weights.max(axis=1).values[0]) / (
                            weights.min(axis=1).values[0]
                            - weights.max(axis=1).values[0]
                        )  # maxmin scaler
                weights += rebalance_weights  # by default, add .1 to every value here so that every model gets some weight instead of 0 for the worst one
                weights = weights / weights.sum(axis=1).values[0]
                pred = (preds * weights.values[0]).sum(axis=1).to_list()
                fv = (fvs * weights.values[0]).sum(axis=1).to_list()
                fcst = (fcsts * weights.values[0]).sum(axis=1).to_list()
            except ZeroDivisionError:
                how = "simple"  # all models have the same test set metric value so it's a simple average (never seen this, but jic)
        if how in ("simple", "splice"):
            pred = preds.mean(axis=1).to_list()
            fv = fvs.mean(axis=1).to_list()
            if how == "simple":
                fcst = fcsts.mean(axis=1).to_list()
            elif how == "splice":
                _developer_utils.descriptive_assert(
                    len(models) == len(splice_points) + 1,
                    ForecastError,
                    "Must have exactly 1 more model passed to models as splice points.",
                )
                splice_points = pd.to_datetime(sorted(splice_points)).to_list()
                future_dates = self.future_dates.to_list()
                _developer_utils.descriptive_assert(
                    np.array([p in future_dates for p in splice_points]).all(),
                    TypeError,
                    "All elements in splice_points must be parsable by pandas' Timestamp function and be present in the future_dates attribute.",
                )
                fcst = [None] * len(future_dates)
                start = 0
                for i, _ in enumerate(splice_points):
                    end = [
                        idx
                        for idx, v in enumerate(future_dates)
                        if v == splice_points[i]
                    ][0]
                    fcst[start:end] = fcsts[models[i]].values[start:end]
                    start = end
                fcst[start:] = fcsts[models[-1]].values[start:]

        self._metrics(y = actuals, pred = pred, call_me = self.call_me)
        self.weights = tuple(weights.values[0]) if weights is not None else None
        self.models = models
        self.fitted_values = fv
        return fcst

    def _metrics(self, y, pred, call_me):
        """ Needed for combo modeling only.
        """
        self.history[call_me]['TestSetActuals'] = y
        self.history[call_me]['TestSetPredictions'] = pred
        for k, func in self.metrics.items():
            self.history[call_me]['TestSet' + k.upper()] = (
                _developer_utils._return_na_if_len_zero(y, pred, func)
            )

    def _parse_models(self, models, determine_best_by):
        """ takes a collection of models and orders them best-to-worst based on a given metric and returns the ordered list (of str type).

        Args:
            models (list-like): each element is one of Forecaster.estimators.
            determine_best_by (str): one of Forecaster.determine_best_by.
                if a model does not have the metric specified here (i.e. one of the passed models wasn't tuned and this is 'ValidationMetricValue'), it will be ignored silently, so be careful

        Returns:
            (list) The ordered evaluated models.
        """
        if determine_best_by is None:
            if models[:4] == "top_":
                raise ValueError(
                    'Cannot use models starts with "top_" unless the determine_best_by or order_by argument is specified.'
                )
            elif models == "all":
                models = list(self.history.keys())
            elif isinstance(models, str):
                models = [models]
            else:
                models = list(models)
            if len(models) == 0:
                raise ValueError(
                    f"models argument with determine_best_by={determine_best_by} returns no evaluated forecasts."
                )
        else:
            all_models = [
                m for m, d in self.history.items() if determine_best_by in d.keys()
            ]
            all_models = self.order_fcsts(all_models, determine_best_by)
            if models == "all":
                models = all_models[:]
            elif models[:4] == "top_":
                models = all_models[: int(models.split("_")[1])]
            elif isinstance(models, str):
                models = [models]
            else:
                models = [m for m in all_models if m in models]
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

    def infer_freq(self):
        """ uses pandas library to infer frequency of loaded dates.
        """
        if not hasattr(self, "freq"):
            self.freq = pd.infer_freq(self.current_dates)
            self.current_dates.freq = self.freq
            self.future_dates.freq = self.freq

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

    def add_signals(self, model_nicknames, fill_strategy = 'actuals', train_only = False):
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

    def add_ar_terms(self, n):
        """ Adds auto-regressive terms.

        Args:
            n (int): The number of lags to add to the object (1 to this number will be added).

        Returns:
            None

        >>> f.add_ar_terms(4) # adds four lags of y (called 'AR1' - 'AR4') to predict with
        """
        #self._validate_future_dates_exist()
        n = int(n)

        if n == 0:
            return

        _developer_utils.descriptive_assert(
            n >= 0, ValueError, f"n must be greater than or equal to 0, got {n}."
        )
        fcst_length = len(self.future_dates)
        for i in range(1, n + 1):
            self.current_xreg[f"AR{i}"] = pd.Series(self.y).shift(i)
            self.future_xreg[f"AR{i}"] = (
                self.y.to_list()[-i:]
                + ([np.nan] * (fcst_length - i))
            )[:fcst_length]

    def add_AR_terms(self, N):
        """ Adds seasonal auto-regressive terms.
            
        Args:
            N (tuple): First element is the number of lags to add and the second element is the space between lags.

        Returns:
            None

        >>> f.add_AR_terms((2,12)) # adds 12th and 24th lags called 'AR12', 'AR24'
        """
        #self._validate_future_dates_exist()
        _developer_utils.descriptive_assert(
            (len(N) == 2) & (not isinstance(N, str)),
            ValueError,
            f"N must be an array-like of length 2 (lags to add, observations between lags), got {N}.",
        )
        fcst_length = len(self.future_dates)
        for i in range(N[1], N[1] * N[0] + 1, N[1]):
            self.current_xreg[f"AR{i}"] = pd.Series(self.y).shift(i)
            self.future_xreg[f"AR{i}"] = (
                self.y.to_list()[-i:]
                + ([np.nan] * (fcst_length - i))
            )[:fcst_length]

    def reduce_Xvars(
        self,
        method="l1",
        estimator="lasso",
        keep_at_least=1,
        keep_this_many="auto",
        grid_search=True,
        use_loaded_grid=False,
        dynamic_tuning=False,
        monitor="ValidationMetricValue",
        overwrite=True,
        cross_validate=False,
        cvkwargs={},
        **kwargs,
    ):
        """ Reduces the regressor variables stored in the object. The following methods are available:
        l1 which uses a simple l1 penalty and Lasso regressor; pfi that stands for 
        permutation feature importance; and shap. pfi and shap offer more flexibility to view how removing
        variables one-at-a-time, according to which variable is evaluated as least helpful to the
        model after each model evaluation, affects a given error metric for any scikit-learn model.
        After each variable reduction, the model is re-run and feature importance re-evaluated. 
        When using pfi, feature scores are adjusted to account for colinearity, which is a known issue with this method, 
        by sorting by each feature's score and standard deviation, dropping variables first that have both a 
        low score and low standard deviation. By default, the validation-set error is used to avoid leakage 
        and the variable set that most reduced the error is selected.

        Args:
            method (str): One of {'l1','pfi','shap'}. Default 'l1'.
                The reduction method. 
                'l1' uses a lasso regressor and grid searches for the optimal alpha on the validation set
                unless an alpha value is passed to the hyperparams arg and grid_search arg is False.
                'pfi' uses permutation feature importance and is more computationally expensive
                but can use any sklearn estimator.
                'shap' uses shap feature importance, but it is not available for all sklearn models.
                Method "pfi" or "shap" creates attributes in the object called pfi_dropped_vars and pfi_error_values that are lists
                representing the error change with the corresponding dropped variable.
                The pfi_error_values attr is one greater in length than pfi_dropped_vars attr because 
                The first error is the initial error before any variables were dropped.
            estimator (str): One of Forecaster.sklearn_estimators. Default 'lasso'.
                The estimator to use to determine the best set of variables.
                If method == 'l1', the `estimator` arg is ignored and is always lasso.
            keep_at_least (str or int): Default 1.
                The fewest number of Xvars to keep if method == 'pfi'.
                'sqrt' keeps at least the sqare root of the number of Xvars rounded down.
                This exists so that the keep_this_many keyword can use 'auto' as an argument.
            keep_this_many (str or int): Default 'auto'.
                The number of Xvars to keep if method == 'pfi' or 'shap'.
                "auto" keeps the number of xvars that returned the best error using the 
                metric passed to monitor, but it is the most computationally expensive.
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
                Ignored when pfi == 'l1'.
                The metric to be monitored when making reduction decisions. 
            overwrite (bool): Default True.
                If False, the list of selected Xvars are stored in an attribute called reduced_Xvars.
                If True, this list of regressors overwrites the current Xvars in the object.
            cross_validate (bool): Default False.
                Whether to tune the model with cross validation. 
                If False, uses the validation slice of data to tune.
                IIf not monitoring ValidationMetricValue, you will want to leave this False.
            cvkwargs (dict): Default {}. Passed to the `cross_validate()` method.
            **kwargs: Passed to the `manual_forecast()` method and can include arguments related to 
                a given model's hyperparameters or dynamic_testing.
                Do not pass hyperparameters if grid_search is True.
                Do not pass Xvars.

        Returns:
            None

        >>> f.add_ar_terms(24)
        >>> f.add_seasonal_regressors('month',raw=False,sincos=True,dummy=True)
        >>> f.add_seasonal_regressors('year')
        >>> f.add_time_trend()
        >>> f.set_validation_length(12)
        >>> f.reduce_Xvars(overwrite=False) # reduce with lasso (but don't overwrite Xvars)
        >>> print(f.reduced_Xvars) # view results
        >>> f.reduce_Xvars(
        >>>     method='pfi',
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
        _developer_utils.descriptive_assert(
            method in ("l1", "pfi", "shap"),
            ValueError,
            f'method must be one of "pfi", "l1", "shap", got {method}.',
        )
        f = self.deepcopy()
        f.set_estimator(estimator if method in ("pfi", "shap") else "lasso")
        _developer_utils._check_if_correct_estimator(f.estimator,self.sklearn_estimators)
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

        if method == "l1":
            coef_fi_lasso = pd.DataFrame(
                {
                    x: [np.abs(co)]
                    for x, co in zip(f.history["lasso"]["Xvars"], f.history[estimator]['regr'].coef_,)
                },
                index=["feature"],
            ).T
            self.reduced_Xvars = coef_fi_lasso.loc[
                coef_fi_lasso["feature"] != 0
            ].index.to_list()
        else:
            f.save_feature_importance(method=method, on_error="raise")
            fi_df = f.export_feature_importance(estimator)
            using_r2 = monitor.endswith("R2") or (
                f.validation_metric == "r2" and monitor == "ValidationMetricValue"
            )

            if method == "pfi":
                fi_df["weight"] = np.abs(fi_df["weight"])
                fi_df.sort_values(["weight", "std"], ascending=False, inplace=True)

            features = fi_df.index.to_list()
            init_error = f.history[estimator][monitor]
            init_error = -init_error if using_r2 else init_error

            dropped = []
            errors = [init_error]

            sqrt = int(len(f.y) ** 0.5)
            keep_this_many_new = (
                1
                if keep_this_many == "auto"
                else sqrt
                if keep_this_many == "sqrt"
                else keep_this_many
            )
            keep_at_least_new = sqrt if keep_at_least == "sqrt" else keep_at_least

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
                new_error = f.history[estimator][monitor]
                new_error = -new_error if using_r2 else new_error
                errors.append(new_error)

                f.save_feature_importance(method=method, on_error="raise")
                fi_df = f.export_feature_importance(estimator)

                if method == "pfi":
                    fi_df["weight"] = np.abs(fi_df["weight"])
                    fi_df.sort_values(["weight", "std"], ascending=False, inplace=True)

                features = fi_df.index.to_list()

            optimal_drop = (
                errors.index(min(errors))
                if keep_this_many == "auto"
                else drop_this_many
            )
            self.reduced_Xvars = [
                x for x in self.current_xreg.keys() if x not in dropped[:optimal_drop]
            ]
            self.pfi_dropped_vars = dropped
            self.pfi_error_values = [-e for e in errors] if using_r2 else errors

        if overwrite:
            self.current_xreg = {
                x: v for x, v in self.current_xreg.items() if x in self.reduced_Xvars
            }
            self.future_xreg = {
                x: v for x, v in self.future_xreg.items() if x in self.reduced_Xvars
            }

    def _Xvar_select_forecast(
        self,
        f,
        estimator,
        monitor,
        cross_validate,
        dynamic_tuning,
        cvkwargs,
        kwargs,
        Xvars = 'all',
    ):
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
        estimator = 'mlr',
        try_trend = True,
        trend_estimator = 'mlr',
        trend_estimator_kwargs = {},
        decomp_trend = True,
        decomp_method = 'additive',
        try_ln_trend = True,
        max_trend_poly_order = 2,
        try_seasonalities = True,
        seasonality_repr = ['sincos'],
        exclude_seasonalities = [],
        irr_cycles = None, # list of cycles
        max_ar = 'auto', # set to 0 to not test
        test_already_added = True,
        must_keep = [],
        monitor = 'ValidationMetricValue',
        cross_validate = False,
        dynamic_tuning=False,
        cvkwargs={},
        **kwargs,
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
            estimator (str): One of Forecaster.sklearn_estimators. Default 'mlr'.
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
                If 'auto', will use the greater of 10 or the test-set length as the lag order.
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
            **kwargs: {assed to manual_forecast() method and can include arguments related to 
                a given model's hyperparameters or dynamic_testing.
                Do not pass Xvars.

        Returns:
            (dict[tuple[float]]): A dictionary where each key is a tuple of variable combinations 
            and the value is the derived metric (based on value passed to monitor argument).

        >>> f.add_covid19_regressor()
        >>> f.auto_Xvar_select(cross_validate=True)
        """
        def parse_best_metrics(metrics):
            x = [m[0] for m in Counter(metrics).most_common() if not np.isnan(m[1])]
            return None if not x else x[0] if using_r2 else x[-1]
        
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
                ar_regressors = [x for x in f.get_regressor_names() if x.startswith('AR')]
            else:
                ar_regressors = []
            if irr_cycles is not None:
                for i in irr_cycles:
                    f.add_cycle(i)
                irr_regressors = [
                    'cycle' + str(i) + 'sin' for i in irr_cycles
                ] + [
                    'cycle' + str(i) + 'cos' for i in irr_cycles
                ]
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
        
        estimator = self.estimator if estimator is None else estimator
        _developer_utils._check_if_correct_estimator(estimator,self.sklearn_estimators)

        using_r2 = monitor.endswith("R2") or (
            self.validation_metric == "r2" 
            and monitor == "ValidationMetricValue"
        )

        trend_metrics = {}
        seasonality_metrics = {}
        irr_cycles_metrics = {}
        ar_metrics = {}
        final_metrics = {}

        seas_to_try = []

        regressors_already_added = self.get_regressor_names()
        must_keep = [x for x in must_keep if x in regressors_already_added]
        f = self.deepcopy()
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
                ft = f.deepcopy()

            ft.add_time_trend()
            ft.set_test_length(f.test_length)
            ft.set_validation_length(f.validation_length)
            f1 = ft.deepcopy()
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
                f2 = ft.deepcopy()
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
        best_trend = parse_best_metrics(trend_metrics)
        
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
                    f1 = f.deepcopy()
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
        best_seasonality = parse_best_metrics(seasonality_metrics)
        
        if max_ar == 'auto' or max_ar > 0:
            max_ar = max(10,f.test_length) if max_ar == 'auto' else max_ar
            for i in range(1,max_ar+1):
                try:
                    f1 = f.deepcopy()
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
                    if np.isnan(ar_metrics[i]):
                        warnings.warn(
                            f'Cannot estimate {estimator} model with {i} AR terms.',
                            category=Warning,
                        )
                        ar_metrics.pop(i)
                        break
                except (IndexError,AttributeError,ForecastError):
                    raise
                    warnings.warn(
                        f'Cannot estimate {estimator} model with {i} AR terms.',
                        category=Warning,
                    )
                    break
        
        best_ar_order = parse_best_metrics(ar_metrics)

        f = self.deepcopy()

        Xvars = get_Xvar_combos(
            f,
            best_trend,
            best_seasonality,
            best_ar_order,
            regressors_already_added,
            seas_to_try,
        )
        for xvar_set in Xvars:
            final_metrics[tuple(xvar_set)] =  self._Xvar_select_forecast(
                f=f,
                estimator=estimator,
                monitor=monitor,
                cross_validate=cross_validate,
                dynamic_tuning=dynamic_tuning,
                cvkwargs=cvkwargs,
                kwargs=kwargs,
                Xvars=xvar_set,
            )
        best_combo = parse_best_metrics(final_metrics)

        f.drop_Xvars(*[x for x in f.get_regressor_names() if x not in best_combo])
        self.current_xreg = f.current_xreg
        self.future_xreg = f.future_xreg
        if len(final_metrics) == 0:
            warnings.warn(
                "auto_Xvar_select() did not add any regressors to the object."
                " Sometimes this happens when the object's test length is 0"
                " and the function's monitor argument is specified as a test set metric.",
                category = Warning,
            )
        return final_metrics

    def determine_best_series_length(
        self,
        estimator = 'mlr',
        min_obs = 100,
        max_obs = None,
        step = 25,
        monitor = 'ValidationMetricValue',
        cross_validate = False,
        dynamic_tuning = False,
        cvkwargs = {},
        chop = True,
        **kwargs,
    ):
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
            (dict[int[float]]): A dictionary where each key is a series length and the value is the derived metric 
            (based on what was passed to the monitor argument).

        >>> f.auto_Xvar_select()
        >>> f.determine_best_series_length()
        """
        def parse_best_metrics(metrics):
            x = [m[0] for m in Counter(metrics).most_common()]
            return None if not x else x[0] if using_r2 else x[-1]

        using_r2 = monitor.endswith("R2") or (
                self.validation_metric == "r2" and monitor == "ValidationMetricValue"
            )

        history_metrics = {}
        max_obs = len(self.y) if max_obs is None else max_obs
        i = len(self.y) - 1
        for i in np.arange(min_obs,max_obs,step):
            f = self.deepcopy()
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
            f = self.deepcopy()
            history_metrics[max_obs] = self._Xvar_select_forecast(
                f=f,
                estimator=estimator,
                monitor=monitor,
                dynamic_tuning=dynamic_tuning,
                cross_validate=cross_validate,
                cvkwargs=cvkwargs,
                kwargs=kwargs,
            )
        best_history_to_keep = parse_best_metrics(history_metrics)

        if chop:
            self.keep_smaller_history(best_history_to_keep)
        
        return history_metrics

    def adf_test(
        self, 
        critical_pval=0.05, 
        full_res=True, 
        train_only=False, 
        diffy=False,
        **kwargs
    ):
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

    def plot_acf(self, diffy=False, train_only=False, **kwargs):
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

    def plot_pacf(self, diffy=False, train_only=False, **kwargs):
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

    def plot_periodogram(self, diffy=False, train_only=False):
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

    def STL(self, diffy=False, train_only=False, **kwargs):
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

    def seasonal_decompose(self, diffy=False, train_only=False, **kwargs):
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
        models,
        cross_validate=False,
        dynamic_tuning=False,
        dynamic_testing=True,
        summary_stats=False,
        feature_importance=False,
        fi_method="pfi",
        limit_grid_size=None,
        min_grid_size=1,
        suffix=None,
        error='raise',
        **cvkwargs,
    ):
        """ Iterates through a list of models, tunes them using grids in a grids file, forecasts them, and can save feature information.

        Args:
            models (list-like):
                Each element must be in Forecaster.can_be_tuned.
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
            summary_stats (bool): Default False.
                Whether to save summary stats for the models that offer those.
            feature_importance (bool): Default False.
                Whether to save permutation feature importance information for the models that offer those.
            fi_method (str): One of {'pfi','shap'}. Default 'pfi'.
                The type of feature importance to save for the models that support it.
                Ignored if feature_importance is False.
            limit_grid_size (int or float): Optional. Pass an argument here to limit each of the grids being read.
                See https://scalecast.readthedocs.io/en/latest/Forecaster/Forecaster.html#src.scalecast.Forecaster.Forecaster.limit_grid_size.
            min_grid_size (int): Default 1. The smallest grid size to keep. Ignored if limit_grid_size is None.
            suffix (str): Optional. A suffix to add to each model as it is evaluate to differentiate them when called
                later. If unspecified, each model can be called by its estimator name.
            error (str): One of 'ignore','raise','warn'; default 'raise'.
                What to do with the error if a given model fails.
                'warn' prints a warning that the model could not be evaluated.
            **cvkwargs: Passed to the cross_validate() method.

        Returns:
            None

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
            summary_stats=summary_stats,
            feature_importance=feature_importance,
            fi_method=fi_method,
            **cvkwargs,
        )

    def save_feature_importance(self, model = None, method="pfi", on_error="warn"):
        """ Saves feature info for models that offer it (sklearn models).
        Call after evaluating the model you want it for.
        This method saves a dataframe listing the feature as the index and its score. This dataframe can be recalled using
        the `export_feature_importance()` method. Scores for the pfi method are the average decrease in accuracy
        over 10 permutations for each feature. For shap, it is determined as the average score applied to each
        feature in each observation.

        Args:
            model (str): Optional. The model's nickname to save information for.
                By default, uses the last evaluated or tested model.
            method (str): One of {'pfi','shap'}.
                The type of feature importance to set.
                'pfi' supported for all sklearn model types. 
                'shap' for xgboost, lightgbm and some others.
            on_error (str): One of {'warn','raise','ignore'}. Default 'warn'.
                If the last model called doesn't support feature importance,
                'warn' will log a warning. 'raise' will raise an error.

        >>> f.set_estimator('mlr')
        >>> f.manual_forecast()
        >>> f.save_feature_importance() # pfi
        >>> f.export_feature_importance('mlr')
        >>>
        >>> f.set_estimator('xgboost')
        >>> f.manual_forecast()
        >>> f.save_feature_importance('shap') # shap
        >>> f.export_feature_importance('xgboost')
        """
        _developer_utils.descriptive_assert(
            method in ("pfi", "shap"),
            ValueError,
            f'kind must be one of "pfi","shap", got {method}.',
        )
        fail = False
        try:
            model = self.call_me if model is None else model
            regr = self.history[model]['regr']
            X = self.history[model]['X']
            Xvars = self.history[model]['Xvars']
            if method == "pfi":
                import eli5
                from eli5.sklearn import PermutationImportance
                perm = PermutationImportance(regr).fit(
                    X, self.y.values[: X.shape[0]],
                )
                self.feature_importance = eli5.explain_weights_df(
                    perm, 
                    feature_names=Xvars,
                ).set_index("feature")
            else:
                import shap
                explainer = shap.TreeExplainer(regr)
                shap_values = explainer.shap_values(X)
                shap_df = pd.DataFrame(shap_values, columns=self.Xvars)
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
                raise TypeError(str(error))
            elif on_error != 'ignore':
                raise ValueError(f"Value passed to on_error not recognized: {on_error}.")
            return

        self._bank_fi_to_history()

    def save_summary_stats(self):
        """ Saves summary stats for models that offer it and will not raise errors if not available.
        Call after evaluating the model you want it for and before changing the estimator.

        >>> f.set_estimator('arima')
        >>> f.manual_forecast(order=(1,1,1))
        >>> f.save_summary_stats()
        """
        try:
            self._set_summary_stats()
        except:
            warnings.warn(
                f"{self.estimator} does not have summary stats.",
                category = Warning,
            )
            return
        self._bank_summary_stats_to_history()

    def chop_from_front(self, n, fcst_length = None):
        """ Cuts the amount of y observations in the object from the front counting backwards.
        The current length of the forecast horizon will be maintained and all future regressors will be rewritten to the appropriate attributes.

        Args:
            n (int):
                The number of observations to cut from the front.
            fcst_length (int): Optional.
                The new length of the forecast length.
                By default, maintains the same forecast length currently in the object.

        >>> f.chop_from_front(10) # keeps all observations before the last 10
        """
        n = int(n)
        fcst_length = len(self.future_dates) if fcst_length is None else fcst_length
        self.y = self.y.iloc[:-n]
        self.current_dates = self.current_dates.iloc[:-n]
        self.generate_future_dates(fcst_length)
        self.future_xreg = {
            k:(self.current_xreg[k].to_list()[-n:] + v[:max(0,(fcst_length-n))])[-fcst_length:]
            for k, v in self.future_xreg.items()
        }
        self.future_xreg = {
            k:v[:int(k[2:])] + ([np.nan] * (len(self.future_dates) - int(k[2:])))
            if k.startswith('AR') else v[:] 
            for k, v in self.future_xreg.items()
        }
        self.current_xreg = {
            k:v.iloc[:-n].reset_index(drop=True)
            for k, v in self.current_xreg.items()
        }

    def chop_from_back(self,n):
        """ Cuts y observations in the object from the back by counting forward from the beginning.

        Args:
            n (int): The number of observations to cut from the back.

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
        self.y = self.y.iloc[-n:]
        self.current_dates = self.current_dates.iloc[-n:]
        self.current_xreg = {k:v.iloc[-n:].reset_index(drop=True) for k, v in self.current_xreg.items()}

    def order_fcsts(self, models = 'all', determine_best_by="TestSetRMSE"):
        """ Gets estimated forecasts ordered from best-to-worst.
        
        Args:
            models (str or list-like): Default 'all'.
                If not 'all', each element must match an evaluated model's nickname.
                'all' will only consider models that have a non-null determine_best_by value in history.
            determine_best_by (str): Default 'TestSetRMSE'. One of Forecaster.determine_best_by.

        Returns:
            (list): The ordered models.

        >>> models = ('mlr','mlp','lightgbm')
        >>> f.tune_test_forecast(models,dynamic_testing=False,feature_importance=True)
        >>> ordered_models = f.order_fcsts(models,"TestSetRMSE")
        """
        _developer_utils.descriptive_assert(
            determine_best_by in self.determine_best_by,
            ValueError,
            f"determine_best_by must be one of {self.determine_best_by}, got {determine_best_by}.",
        )
        
        if models == "all":
            models = [
            m for m, v in self.history.items() 
            if determine_best_by in v and not np.isnan(v[determine_best_by])
        ]
        
        models_metrics = {m: self.history[m][determine_best_by] for m in models}
        x = [h[0] for h in Counter(models_metrics).most_common()]
        return (
            x
            if (determine_best_by.endswith("R2"))
            | (
                (determine_best_by == "ValidationMetricValue")
                & (self.validation_metric.upper() == "R2")
            )
            else x[::-1]
        )

    def get_regressor_names(self):
        """ Gets the regressor names stored in the object.

        Args:
            None

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
        models="all",
        exclude = [],
        order_by=None, 
        ci=False,
        ax = None,
        figsize=(12,6),
    ):
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
            ci (bool): Default False.
                Whether to display the confidence intervals.
            ax (Axis): Optional. The existing axis to write the resulting figure to.
            figsize (tuple): Default (12,6). The size of the resulting figure. Ignored when ax is not None.

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
            for i, m in enumerate(models):
                plot[m] = (self.history[m]["Forecast"])
                if plot[m] is None or not len(plot[m]):
                    continue
                sns.lineplot(
                    x=self.future_dates.to_list(),
                    y=plot[m],
                    color=__colors__[i],
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
                            color=__colors__[i],
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
        models="all",
        exclude = [],
        order_by=None, 
        include_train=True, 
        ci=False,
        ax = None,
        figsize=(12,6),
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

        Returns:
            (Axis): The figure's axis.

        >>> models = ('mlr','mlp','lightgbm')
        >>> f.tune_test_forecast(models,dynamic_testing=False,feature_importance=True)
        >>> f.plot(order_by='TestSetRMSE') # plots all test-set results
        >>> plt.show()
        """
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)

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

        for i, m in enumerate(models):
            plot[m] = (self.history[m]["TestSetPredictions"])
            test_dates = self.current_dates.values[-len(plot[m]) :]
            sns.lineplot(
                x=test_dates,
                y=plot[m],
                linestyle="--",
                color=__colors__[i],
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
                        color=__colors__[i],
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
        models="all",
        exclude = [], 
        order_by=None, 
        ax = None,
        figsize=(12,6),
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

        Returns:
            (Axis): The figure's axis.

        >>> models = ('mlr','mlp','lightgbm')
        >>> f.tune_test_forecast(models,dynamic_testing=False,feature_importance=True)
        >>> f.plot_fitted(order_by='TestSetRMSE') # plots all fitted values
        >>> plt.show()
        """
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

        for i, m in enumerate(models):
            plot[m] = (self.history[m]["FittedVals"])
            if plot[m] is None or not len(plot[m]):
                continue
            sns.lineplot(
                x=plot["date"][-len(plot[m]) :],
                y=plot[m][-len(plot["date"]) :],
                linestyle="--",
                color=__colors__[i],
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
        dfs=[
            "model_summaries",
            "lvl_test_set_predictions",
            "lvl_fcsts",
        ],
        models="all",
        best_model="auto",
        determine_best_by=None,
        cis=False,
        to_excel=False,
        out_path="./",
        excel_name="results.xlsx",
    ) -> Union[Dict[str, pd.DataFrame], pd.DataFrame]:
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
            out_path (str): Default './'.
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
        if isinstance(dfs, str):
            dfs = [dfs]
        else:
            dfs = list(dfs)
        if len(dfs) == 0:
            raise ValueError("No values passed to the dfs argument.")
        models = self._parse_models(models, determine_best_by)
        _dfs_ = [
            "model_summaries",
            "lvl_test_set_predictions",
            "lvl_fcsts",
        ]
        _bad_dfs_ = [i for i in dfs if i not in _dfs_]
        if len(_bad_dfs_) > 0:
            raise ValueError(
                f"Values passed to the dfs list must be in {_dfs_}, not {_bad_dfs_}"
            )
        if determine_best_by is not None:
            best_fcst_name = (
                self.order_fcsts(models, determine_best_by)[0]
                if best_model == "auto"
                else best_model
            )
        else:
            best_fcst_name = list(self.history.keys())[0] # first evaluated model
        output = {}
        if "model_summaries" in dfs:
            cols1 = [
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

            model_summaries = pd.DataFrame()
            for m in models:
                model_summary_m = pd.DataFrame({"ModelNickname": [m]})
                cols = cols1 + [
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
                        "ModelNickname",
                        "LastTestSetPrediction",
                        "LastTestSetActual",
                        "best_model",
                    ):
                        model_summary_m[c] = [
                            self.history[m][c] if c in self.history[m].keys() else np.nan
                        ]
                    elif c == "best_model":
                        model_summary_m[c] = m == best_fcst_name
                model_summaries = pd.concat(
                    [model_summaries, model_summary_m], ignore_index=True
                )
            output["model_summaries"] = model_summaries
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
                            test_set_predictions[m + "_upperci"] = self.history[m][
                                "TestSetUpperCI"
                            ]
                            test_set_predictions[m + "_lowerci"] = self.history[m][
                                "TestSetLowerCI"
                            ]
                        except KeyError:
                            _developer_utils._warn_about_not_finding_cis(m)
                output["lvl_test_set_predictions"] = test_set_predictions

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

    def export_summary_stats(self, model) -> pd.DataFrame:
        """ Exports the summary stats from a model.
        Raises an error if you never saved the model's summary stats.

        Args:
            model (str):
                The name of them model to export for.
                Matches what was passed to call_me when evaluating the model.

        Returns:
            (DataFrame): The resulting summary stats of the evaluated model passed to model arg.

        >>> ss = f.export_summary_stats('arima')
        """
        return self.history[model]["summary_stats"]

    def export_feature_importance(self, model) -> pd.DataFrame:
        """ Exports the feature importance from a model.
        Raises an error if you never saved the model's feature importance.

        Args:
            model (str):
                The name of them model to export for.
                Matches what was passed to call_me when evaluating the model.

        Returns:
            (DataFrame): The resulting feature importances of the evaluated model passed to model arg.

        >>> fi = f.export_feature_importance('mlr')
        """
        return self.history[model]["feature_importance"]

    def all_feature_info_to_excel(self, out_path="./", excel_name="feature_info.xlsx"):
        """ Saves all feature importance and summary stats to excel.
        Each model where such info is available for gets its own tab.
        Be sure to have called save_summary_stats() and/or save_feature_importance() before using this function.

        Args:
            out_path (str): Default './'.
                The path to export to.
            excel_name (str): Default 'feature_info.xlsx'.
                The name of the resulting excel file.

        Returns:
            None
        """
        try:
            with pd.ExcelWriter(
                os.path.join(out_path, excel_name), engine="openpyxl"
            ) as writer:
                for m in self.history.keys():
                    if "summary_stats" in self.history[m].keys():
                        self.history[m]["summary_stats"].to_excel(
                            writer, sheet_name=f"{m}_summary_stats"
                        )
                    elif "feature_importance" in self.history[m].keys():
                        self.history[m]["feature_importance"].to_excel(
                            writer, sheet_name=f"{m}_feature_importance"
                        )
        except IndexError:
            raise ForecastError(
                "No saved feature importance or summary stats could be found."
            )

    def all_validation_grids_to_excel(
        self,
        out_path="./",
        excel_name="validation_grids.xlsx",
        sort_by_metric_value=False,
        ascending=True,
    ):
        """ Saves all validation grids to excel.
        Each model where such info is available for gets its own tab.
        Be sure to have tuned at least model before calling this.

        Args:
            out_path (str): Default './'.
                The path to export to.
            excel_name (str): Default 'feature_info.xlsx'.
                The name of the resulting excel file.
            sort_by_metric_value (bool): Default False.
                Whether to sort the output by performance on validation set.
            ascending (bool): Default True.
                Whether to sort least-to-greatest.
                Ignored if sort_by_metric_value is False.

        Returns:
            None
        """
        try:
            with pd.ExcelWriter(
                os.path.join(out_path, excel_name), engine="openpyxl"
            ) as writer:
                for m in self.history.keys():
                    if "grid_evaluated" in self.history[m].keys():
                        df = self.export_validation_grid(m)
                        df.to_excel(writer, sheet_name=m, index=False)
        except IndexError:
            raise
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

    def export_fitted_vals(self, model):
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