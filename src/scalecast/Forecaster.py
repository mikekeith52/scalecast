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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_percentage_error,
    mean_absolute_error,
)
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose

import copy
logging.basicConfig(filename="warnings.log", level=logging.WARNING)
logging.captureWarnings(True)

# sklearn imports below
from sklearn.linear_model import LinearRegression as mlr_
from sklearn.neural_network import MLPRegressor as mlp_
from sklearn.ensemble import GradientBoostingRegressor as gbt_
from xgboost import XGBRegressor as xgboost_
from lightgbm import LGBMRegressor as lightgbm_
from sklearn.ensemble import RandomForestRegressor as rf_
from sklearn.linear_model import ElasticNet as elasticnet_
from sklearn.linear_model import Lasso as lasso_
from sklearn.linear_model import Ridge as ridge_
from sklearn.svm import SVR as svr_
from sklearn.neighbors import KNeighborsRegressor as knn_
from sklearn.linear_model import SGDRegressor as sgd_

# FUNCTIONS

# custom metrics - are what you expect except MAPE, which can return None
def mape(y, pred):
    return (np.nan if 0 in y else mean_absolute_percentage_error(y, pred))  # average o(1) worst-case o(n)


def rmse(y, pred):
    return mean_squared_error(y, pred) ** 0.5


def mae(y, pred):
    return mean_absolute_error(y, pred)


def r2(y, pred):
    return r2_score(y, pred)

def prepare_data(Xvars, y, current_xreg):
    if Xvars is None or Xvars == "all":
        Xvars = [x for x in current_xreg.keys()]
    elif isinstance(Xvars,str):
        Xvars = [Xvars]

    y = [i for i in y]
    X = pd.DataFrame(current_xreg)
    X = X[Xvars].values
    return Xvars, y, X

def prepare_rnn(yvals, lags, forecast_length):
    """ prepares and scales the data for rnn models.

    Args:
        yvals (ndarray): dependent variable values to prepare
        lags (int): the number of lags to use as predictors and to make the X matrix with
        forecast_length (int): the amount of time to forecast out

    Returns:
        (ndarray,ndarray): The new X matrix and y array.
    """
    ylist = [(y - yvals.min()) / (yvals.max() - yvals.min()) for y in yvals]

    n_future = forecast_length
    n_past = lags
    total_period = n_future + n_past
    
    idx_end = len(ylist)
    idx_start = idx_end - total_period
    
    X_new = []
    y_new = []

    while idx_start > 0:
        x_line = ylist[idx_start:idx_start+n_past]
        y_line = ylist[idx_start+n_past:idx_start+total_period]

        X_new.append(x_line)
        y_new.append(y_line)
        
        idx_start = idx_start - 1
        
    X_new = np.array(X_new)
    y_new = np.array(y_new)

    return X_new, y_new

def plot_loss_rnn(history,title):
    plt.plot(history.history['loss'],label='train_loss')
    if 'val_loss' in history.history.keys():
        plt.plot(history.history['val_loss'],label='val_loss')
    plt.title(title)
    plt.xlabel('epoch')
    plt.legend(loc='upper right')
    plt.show()

# descriptive assert statement for error catching
def descriptive_assert(statement, ErrorType, error_message):
    try:
        assert statement
    except AssertionError:
        raise ErrorType(error_message)

# KEY GLOBALS

# to add a new sklearn model, just add to this dict
_sklearn_imports_ = {
    "mlr": mlr_,
    "mlp": mlp_,
    "gbt": gbt_,
    "xgboost": xgboost_,
    "lasso": lasso_,
    "ridge": ridge_,
    "lightgbm": lightgbm_,
    "rf": rf_,
    "elasticnet": elasticnet_,
    "svr": svr_,
    "knn": knn_,
    "sgd": sgd_,
}
_sklearn_estimators_ = sorted(_sklearn_imports_.keys())

# to add non-sklearn models, add to the list below
_non_sklearn_estimators_ = ["arima", "hwes", "prophet", "silverkite", "rnn", "lstm", "combo"]
_estimators_ = sorted(_sklearn_estimators_ + _non_sklearn_estimators_)
_cannot_be_tuned_ = ['combo','rnn','lstm']
_can_be_tuned_ = [m for m in _estimators_ if m not in _cannot_be_tuned_]
_metrics_ = ["r2", "rmse", "mape", "mae"]
_determine_best_by_ = [
    "TestSetRMSE",
    "TestSetMAPE",
    "TestSetMAE",
    "TestSetR2",
    "InSampleRMSE",
    "InSampleMAPE",
    "InSampleMAE",
    "InSampleR2",
    "ValidationMetricValue",
    "LevelTestSetRMSE",
    "LevelTestSetMAPE",
    "LevelTestSetMAE",
    "LevelTestSetR2",
]
_normalizer_ = ["minmax", "normalize", "scale", None]
_colors_ = [
    "#FFA500",
    "#DC143C",
    "#00FF7F",
    "#808000",
    "#BC8F8F",
    "#A9A9A9",
    "#8B008B",
    "#FF1493",
    "#FFDAB9",
    "#20B2AA",
    "#7FFFD4",
    "#A52A2A",
    "#DCDCDC",
    "#E6E6FA",
    "#BDB76B",
    "#DEB887",
] * 10
_not_hyperparams_ = ["Xvars", "normalizer", "tuned", "plot_loss","plot_loss_test"]

# DESCRIPTIVE ERRORS
class ForecastError(Exception):
    class CannotDiff(Exception):
        pass

    class CannotUndiff(Exception):
        pass

    class NoGrid(Exception):
        pass

    class PlottingError(Exception):
        pass


# MAIN OBJECT
class Forecaster:
    def __init__(self, y, current_dates, **kwargs):
        self.y = y
        self.current_dates = current_dates
        self.future_dates = pd.Series([],dtype='datetime64[ns]')
        self.current_xreg = {}  # values should be pandas series (to make differencing work more easily)
        self.future_xreg = {}  # values should be lists (to make iterative forecasting work more easily)
        self.history = {}
        self.test_length = 1
        self.validation_length = 1
        self.validation_metric = "rmse"
        self.integration = 0
        self.levely = list(y)
        self.init_dates = list(current_dates)
        self.cilevel = 0.95
        self.bootstrap_samples = 100
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.typ_set()  # ensures that the passed values are the right types

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
        """ creates an object copy.
        """
        return self.__copy__() 

    def deepcopy(self):
        """ creates an object deepcopy.
        """
        return self.__deepcopy__()

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
    Differenced={}
    TestLength={}
    ValidationLength={}
    ValidationMetric={}
    ForecastsEvaluated={}
    CILevel={}
    BootstrapSamples={}
    CurrentEstimator={}
)""".format(
    self.current_dates.values[0].astype(str),
    self.current_dates.values[-1].astype(str),
    self.freq,
    len(self.y),
    len(self.future_dates),
    list(self.current_xreg.keys()),
    self.integration,
    self.test_length,
    self.validation_length,
    self.validation_metric,
    list(self.history.keys()),
    self.cilevel,
    self.bootstrap_samples,
    None if not hasattr(self,'estimator') else self.estimator,
    )

    def _split_data(self, X, y, test_length, tune):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_length, shuffle=False)
        X_test, y_test = (X_test, y_test) if not tune else (X_test[:-self.test_length], y_test[:-self.test_length])
        return X_train, X_test, y_train, y_test

    def _validate_no_test_only(self,models):
        descriptive_assert(
            max([int(self.history[m]['TestOnly']) for m in models]) == 0,
            ForecastError,
            'this method does not accept any models run test_only = True',
        )

    def _validate_future_dates_exist(self):
        """ makes sure future periods have been specified before adding regressors
        """
        descriptive_assert(
            len(self.future_dates) > 0,
            ForecastError,
            "before adding regressors, please make sure you have generated future dates by calling generate_future_dates(), set_last_future_date(), or ingest_Xvars_df(use_future_dates=True)",
        )

    def _find_cis(self,y,fvs):
        """ bootstraps the upper and lower forecast estimates using the info stored in cilevel and bootstrap_samples
        """
        random.seed(20)
        resids = [
            fv - ac
            for fv, ac in zip(fvs, y[-len(fvs):])
        ]
        bootstrapped_resids = np.random.choice(resids, size=self.bootstrap_samples)
        bootstrap_mean = np.mean(bootstrapped_resids)
        bootstrap_std = np.std(bootstrapped_resids)
        return (stats.norm.ppf(1 - (1 - self.cilevel) / 2) * bootstrap_std + bootstrap_mean)

    def _bank_history(self, **kwargs):
        """ places all relevant information from the last evaluated forecast into the history dictionary attribute
            **kwargs: passed from each model, depending on how that model uses Xvars, normalizer, and other args
        """
        # since test only, what gets saved to history is relevant to the train set only, the genesis of the next line
        y_use = self.y.values.copy() if not self.test_only else self.y.values[:-self.test_length].copy()
        fvs_use = self.fitted_values[:] if not self.test_only else self.fitted_values[:-self.test_length]
        call_me = self.call_me
        # originally, the ci_range used object attributes only, but test_only makes using args necessary
        ci_range = self._find_cis(y_use,fvs_use)
        self.history[call_me] = {
            "Estimator": self.estimator,
            "TestOnly": self.test_only,
            "Xvars": self.Xvars,
            "HyperParams": {k: v for k, v in kwargs.items() if k not in _not_hyperparams_},
            "Scaler": None
                      if self.estimator not in ['rnn','lstm'] + _sklearn_estimators_ 
                      else 'minmax' if 'normalizer' not in kwargs
                      else kwargs['normalizer'],
            "Observations": len(y_use),
            "Forecast": self.forecast[:],
            "UpperCI": [f + ci_range for f in self.forecast],
            "LowerCI": [f - ci_range for f in self.forecast],
            "FittedVals": self.fitted_values[:],
            "Tuned": False if not kwargs["tuned"] else "Dynamically" if self.dynamic_tuning else True,
            "DynamicallyTested": self.dynamic_testing,
            "Integration": self.integration,
            "TestSetLength": self.test_length,
            "TestSetRMSE": self.rmse,
            "TestSetMAPE": self.mape,
            "TestSetMAE": self.mae,
            "TestSetR2": self.r2,
            "TestSetPredictions": self.test_set_pred[:],
            "TestSetUpperCI": [f + ci_range for f in self.test_set_pred],  # not exactly right, but close enough with caveat
            "TestSetLowerCI": [f - ci_range for f in self.test_set_pred],  # not exactly right, but close enough with caveat
            "TestSetActuals": self.test_set_actuals[:],
            "InSampleRMSE": rmse(y_use[-len(fvs_use) :], fvs_use),
            "InSampleMAPE": mape(y_use[-len(fvs_use) :], fvs_use),
            "InSampleMAE": mae(y_use[-len(fvs_use) :], fvs_use),
            "InSampleR2": r2(y_use[-len(fvs_use) :], fvs_use),
            "CILevel":self.cilevel,
            "CIPlusMinus":ci_range,
        }

        if kwargs["tuned"]:
            self.history[call_me]["ValidationSetLength"] = self.validation_length
            self.history[call_me]["ValidationMetric"] = self.validation_metric
            self.history[call_me]["ValidationMetricValue"] = self.validation_metric_value

        for attr in ("grid_evaluated", "models", "weights"):
            if hasattr(self, attr):
                self.history[call_me][attr] = getattr(self, attr)

        self.history[call_me]["LevelY"] = self.levely[:]
        if self.integration > 0:
            integration = self.integration

            fcst = self.forecast[::-1]
            pred = self.history[call_me]["TestSetPredictions"][::-1]

            if integration == 2:
                fcst.append(y_use[-2] + y_use[-1])
                pred.append(
                    y_use[-(len(pred) + 2)] + y_use[-(len(pred) + 1)]
                )
            else:
                fcst.append(self.levely[-1])
                pred.append(self.levely[-(len(pred) + 1)])

            fcst = list(np.cumsum(fcst[::-1]))[1:]
            pred = list(np.cumsum(pred[::-1]))[1:]

            if integration == 2:
                fcst.reverse()
                fcst.append(self.levely[-1])
                fcst = list(np.cumsum(fcst[::-1]))[1:]

                pred.reverse()
                pred.append(self.levely[-(len(pred) + 1)])
                pred = list(np.cumsum(pred[::-1]))[1:]

            self.history[call_me]["LevelForecast"] = fcst[:]
            self.history[call_me]["LevelTestSetPreds"] = pred[:]
            self.history[call_me]["LevelTestSetRMSE"] = rmse(self.levely[-len(pred) :], pred)
            self.history[call_me]["LevelTestSetMAPE"] = mape(self.levely[-len(pred) :], pred)
            self.history[call_me]["LevelTestSetMAE"] = mae(self.levely[-len(pred) :], pred)
            self.history[call_me]["LevelTestSetR2"] = r2(self.levely[-len(pred) :], pred)
        else:  # better to have these attributes populated for all series
            self.history[call_me]["LevelForecast"] = self.forecast[:]
            self.history[call_me]["LevelTestSetPreds"] = self.test_set_pred[:]
            self.history[call_me]["LevelTestSetRMSE"] = self.rmse
            self.history[call_me]["LevelTestSetMAPE"] = self.mape
            self.history[call_me]["LevelTestSetMAE"] = self.mae
            self.history[call_me]["LevelTestSetR2"] = self.r2

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
        self.history[call_me]["per"] = self.perm

    def _bank_summary_stats_to_history(self):
        """ saves summary stats (where available) to history
        """
        call_me = self.call_me
        self.history[call_me]["summary_stats"] = self.summary_stats

    def _metrics(self, y, pred):
        """ creates the following attributes: test_set_actuals, test_set_pred, rmse, r2, mae, mape.

        Args:
            y (list-like):
                the actual observations
            pred (list-like):
                the predictions of y from the model

        Returns:
            None
        """
        self.test_set_actuals = list(y)
        self.test_set_pred = list(pred)
        self.rmse = rmse(y, pred)
        self.r2 = r2(y, pred)
        self.mae = mae(y, pred)
        self.mape = mape(y, pred)

    def _tune(self) -> float:
        """ reads which validation metric to use in _metrics_ and pulls that attribute value to return from function.
            deletes: 'r2','rmse','mape','mae','test_set_pred', and 'test_set_actuals' attributes if they exist.
        """
        metric = getattr(self, getattr(self, "validation_metric"))
        for attr in ("r2", "rmse", "mape", "mae", "test_set_pred", "test_set_actuals"):
            delattr(self, attr)
        return metric

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
        ):
            try:
                delattr(self, attr)
            except AttributeError:
                pass


    def _forecast_sklearn(
        self,
        fcster,
        dynamic_testing,
        tune=False,
        Xvars=None,
        normalizer="minmax",
        test_only=False,
        **kwargs,
    ):
        """ runs an sklearn forecast start-to-finish.
        see example: https://scalecast-examples.readthedocs.io/en/latest/sklearn/sklearn.html

        Args:
            fcster (str): one of _sklearn_estimators_. reads the estimator set to `set_estimator()` method.
            dynamic_testing (bool):
                whether to dynamically test the forecast (meaning AR terms will be propogated with predicted values).
                setting this to False means faster performance, but gives a less-good indication of how well the forecast will perform out x amount of periods.
                when False, test-set metrics effectively become an average of one-step forecasts.
            tune (bool): default False.
                whether the model is being tuned.
                does not need to be specified by user.
            Xvars (list-like, str, or None): the regressors to predict with.
                be sure to have added them to the Forecaster object first.
                None means all Xvars added to the Forecaster object will be used.
            normalizer (str): one of _normalizer_. default 'minmax'.
                if not None, normalizer applied to training data only to not leak.
            test_only (bool): default False:
                whether to stop your model after the testing process.
                all forecasts in history will be equivalent to test-set predictions.
                when forecasting to future periods, there is no way to skip the training process.
                any plot or export of forecasts into a future horizon will fail 
                and not all methods will raise descriptive errors.
            **kwargs: treated as model hyperparameters and passed to _sklearn_imports_[model]()
        """
        def fit_normalizer(X, normalizer):
            descriptive_assert(
                normalizer in _normalizer_,
                ForecastError,
                f"normalizer must be one of {_normalizer_}, got {normalizer}",
            )
            if normalizer == "minmax":
                from sklearn.preprocessing import MinMaxScaler as Scaler
            elif normalizer == "normalize":
                from sklearn.preprocessing import Normalizer as Scaler
            elif normalizer == "scale":
                from sklearn.preprocessing import StandardScaler as Scaler
            else:
                return None
            
            scaler = Scaler()
            scaler.fit(X)
            return scaler

        def evaluate_model(scaler,regr,X,y,Xvars,fcst_horizon,future_xreg,dynamic_testing,true_forecast):
            def scale(scaler,X):
                return scaler.transform(X) if scaler is not None else X

            # apply the normalizer fit on training data only
            X = scale(scaler, X)
            self.X = X # for feature importance setting
            regr.fit(X, y)
            # if not using any AR terms or not dynamically evaluating the forecast, use the below (faster but ends up being an average of one-step forecasts when AR terms are involved)
            if (len([x for x in Xvars if x.startswith("AR")]) == 0) | (not dynamic_testing):
                p = pd.DataFrame(future_xreg).values[:fcst_horizon]
                p = scale(scaler, p)
                return (regr.predict(p),regr.predict(X),Xvars,regr)
            # otherwise, use a dynamic process to propogate out-of-sample AR terms with predictions (slower but more indicative of a true forecast performance)
            fcst = []
            for i in range(fcst_horizon):
                p = pd.DataFrame({x: [future_xreg[x][i]] for x in Xvars}).values
                p = scale(scaler, p)
                fcst.append(regr.predict(p)[0])
                if not i == (fcst_horizon - 1):
                    for k, v in future_xreg.items():
                        if k.startswith("AR"):
                            ar = int(k[2:])
                            idx = i + 1 - ar
                            if idx > -1:
                                try:
                                    future_xreg[k][i + 1] = fcst[idx]
                                except IndexError:
                                    future_xreg[k].append(fcst[idx])
                            else:
                                try:
                                    future_xreg[k][i + 1] = y[idx]
                                except IndexError:
                                    future_xreg[k].append(y[idx])
            
            return (fcst,regr.predict(X),Xvars,regr)

        descriptive_assert(
            len(self.current_xreg.keys()) > 0,
            ForecastError,
            f"need at least 1 Xvar to forecast with the {self.estimator} model",
        )

        # list of integers, each one representing the n/a values in each AR term
        ars = [v.isnull().sum() for x, v in self.current_xreg.items() if x.startswith("AR")]
        # if using ARs, instead of foregetting those obs, ignore them with sklearn forecasts (leaves them available for other types of forecasts)
        obs_to_drop = max(ars) if len(ars) > 0 else 0
        y = self.y.values[obs_to_drop:].copy()
        current_xreg = {xvar: x.values[obs_to_drop:].copy() for xvar, x in self.current_xreg.items()}
        test_length = self.test_length if not tune else self.validation_length + self.test_length
        # get a list of Xvars, the y array, the X matrix, and the test size (can be different depending on if tuning or testing)
        Xvars, y, X = prepare_data(Xvars, y, current_xreg)
        # split the data
        X_train, X_test, y_train, y_test = self._split_data(X, y, test_length, tune)
        # fit the normalizer to training data only
        scaler = fit_normalizer(X_train, normalizer)
        # get the sklearn regressor
        regr = _sklearn_imports_[fcster](**kwargs)
        # train/test the model
        result = evaluate_model(
            scaler,
            regr,
            X_train,
            y_train,
            Xvars,
            test_length - self.test_length if tune else test_length, # fcst_horizon
            {x: current_xreg[x][-test_length:] for x in Xvars}, # for AR processing
            dynamic_testing,
            False
        )
        pred = [i for i in result[0]]
        # set the test-set metrics
        self._metrics(y_test, pred)
        if tune: 
            return self._tune()
        if test_only:
            # last fitted val has to match last date in current_dates or a bunch of stuff breaks
            result = (result[0],[i for i in result[1]] + [np.nan]*self.test_length,result[2],result[3])
            return result

        # run full model
        return evaluate_model(
            scaler,
            regr,
            X,
            y,
            Xvars,
            len(self.future_dates),
            {x: self.future_xreg[x][:] for x in Xvars},
            True,
            True,
        )

    def _forecast_hwes(self, tune=False, dynamic_testing=True, test_only=False, **kwargs):
        """ forecasts with holt-winters exponential smoothing.
        see example: https://scalecast-examples.readthedocs.io/en/latest/hwes/hwes.html

        Args:
            tune (bool): default False.
                whether the model is being tuned.
                does not need to be specified by user.
            dynamic_testing (bool): default True.
                always ignored in HWES (for now) - everything is set to be dynamic using statsmodels.
            test_only (bool): default False:
                whether to stop your model after the testing process.
                all forecasts in history will be equivalent to test-set predictions.
                when forecasting to future periods, there is no way to skip the training process.
                any plot or export of forecasts into a future horizon will fail for this model
                and not all methods will raise descriptive errors.
            **kwargs: passed to the HWES() function from statsmodels 
        """
        if not dynamic_testing: logging.warning("dynamic_testing is True always for hwes model")
        self.dynamic_testing = True
        from statsmodels.tsa.holtwinters import ExponentialSmoothing as HWES

        y = self.y.to_list()
        test_length = self.test_length if not tune else self.validation_length + self.test_length
        y_train = y[: -test_length]
        y_test = y[-test_length :]
        regr = HWES(
            y_train,
            dates=self.current_dates.values[: -self.test_length],
            freq=self.freq,
            **kwargs,
        ).fit(optimized=True, use_brute=True)
        pred = regr.predict(
            start=len(y_train), 
            end=len(y_train) + (len(y_test) if not tune else len(y_test) - self.test_length) - 1
        )
        self._metrics(y_test if not tune else y_test[:-self.test_length], pred)
        # tune
        if tune:
            return self._tune()
        
        # test only
        if test_only:
            return (pred,list(regr.fittedvalues) + [np.nan]*self.test_length,None,regr)

        # full
        regr = HWES(
            self.y, 
            dates=self.current_dates, 
            freq=self.freq, 
            **kwargs
        ).fit(optimized=True, use_brute=True)
        pred = regr.predict(start=len(y), end=len(y) + len(self.future_dates) - 1)
        return (pred,regr.fittedvalues,None,regr)

    def _forecast_arima(self, tune=False, Xvars=None, dynamic_testing=True, test_only=False, **kwargs):
        """ forecasts with ARIMA (or AR, ARMA, SARIMA, SARIMAX).
        see example: https://scalecast-examples.readthedocs.io/en/latest/arima/arima.html

        Args:
            tune (bool): default False.
                whether the model is being tuned.
                does not need to be specified by user.
            Xvars (list-like, str, or None): the regressors to predict with.
                be sure to have added them to the Forecaster object first.
                None means no Xvars used (unlike sklearn models).
            dynamic_testing (bool): default True.
                always ignored in ARIMA (for now) - everything is set to be dynamic using statsmodels.
            test_only (bool): default False:
                whether to stop your model after the testing process.
                all forecasts in history will be equivalent to test-set predictions.
                when forecasting to future periods, there is no way to skip the training process.
                any plot or export of forecasts into a future horizon will fail for this model
                and not all methods will raise descriptive errors.
            **kwargs: passed to the ARIMA() function from statsmodels.
        """
        if not dynamic_testing:
            logging.warning(
                "dynamic_testing argument will be ignored for the arima model"
            )
        self.dynamic_testing = True
        from statsmodels.tsa.arima.model import ARIMA

        Xvars = (
            [x for x in self.current_xreg.keys() if not x.startswith("AR")]
            if Xvars == "all"
            else [x for x in Xvars if not x.startswith("AR")]
            if Xvars is not None
            else Xvars
        )
        Xvars_orig = None if Xvars is None else None if len(Xvars) == 0 else Xvars
        test_length = self.test_length if not tune else self.validation_length + self.test_length
        Xvars, y, X = prepare_data(Xvars, self.y, self.current_xreg)
        if Xvars_orig is None:
            X, X_train, X_test, Xvars = None, None, None, None
            y_train = y[:-test_length]
            y_test = y[-test_length:]
        else:
            X_train, X_test, y_train, y_test = self._split_data(X, y, test_length, tune)
        regr = ARIMA(
            y_train,
            exog=X_train,
            dates=self.current_dates.values[: -test_length],
            freq=self.freq,
            **kwargs,
        ).fit()
        pred = regr.predict(
            exog=X_test,
            start=len(y_train),
            end=len(y_train) + len(y_test) - 1,
            typ="levels",
            dynamic=True,
        )
        self._metrics(y_test, pred)
        # tune
        if tune:
            return self._tune()
        # test only
        if test_only:
            return (pred,list(regr.fittedvalues) + [np.nan]*self.test_length,Xvars,regr)
        # full
        regr = ARIMA(
            self.y.to_list(),
            exog=X,
            dates=self.current_dates,
            freq=self.freq,
            **kwargs,
        ).fit()
        p = (
            pd.DataFrame({x: self.future_xreg[x] for x in Xvars})
            if Xvars is not None
            else None
        )
        fcst = regr.predict(
            exog=p,
            start=len(y),
            end=len(y) + len(self.future_dates) - 1,
            typ="levels",
            dynamic=True,
        )
        return (fcst,list(regr.fittedvalues),Xvars,regr)

    def _forecast_prophet(
        self,
        tune=False,
        Xvars=None,
        dynamic_testing=True,
        cap=None,
        floor=None,
        test_only=False,
        **kwargs,
    ) -> Union[float, list]:
        """ forecasts with the prophet model from facebook.
        see example: https://scalecast-examples.readthedocs.io/en/latest/prophet/prophet.html

        Args:
            tune (bool): default False.
                whether to tune the forecast.
                does not need to be specified by user.
            Xvars (list-like, str, or None): the regressors to predict with.
                be sure to have added them to the Forecaster object first.
                None means no Xvars used (unlike sklearn models).
            dynamic_testing (bool): default True.
                always ignored for Prophet (for now).
            cap (float): optional.
                specific to prophet when using logistic growth -- the largest amount the model is allowed to evaluate to.
            floor (float): optional.
                specific to prophet when using logistic growth -- the smallest amount the model is allowed to evaluate to.
            test_only (bool): default False:
                whether to stop your model after the testing process.
                all forecasts in history will be equivalent to test-set predictions.
                when forecasting to future periods, there is no way to skip the training process.
                any plot or export of forecasts into a future horizon will fail for this model
                and not all methods will raise descriptive errors.
            **kwargs: passed to the Prophet() function from fbprophet.
        """
        if not dynamic_testing:
            logging.warning(
                "dynamic_testing argument will be ignored for the prophet model"
            )
        self.dynamic_testing = True
        from fbprophet import Prophet

        X = pd.DataFrame({k: v for k, v in self.current_xreg.items() if not k.startswith("AR")})
        p = pd.DataFrame({k: v[-len(self.future_dates):] for k, v in self.future_xreg.items() if not k.startswith("AR")})
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
        # tune
        if tune:
            X_train = X.iloc[: -(self.test_length + self.validation_length)]
            X_test = X.iloc[-(self.test_length + self.validation_length) : -self.test_length]
            y_test = X["y"].values[-(self.test_length + self.validation_length) : -self.test_length]
            model.fit(X_train)
            pred = model.predict(X_test)
            self._metrics(y_test, pred["yhat"].to_list())
            return self._tune()

        model.fit(X.iloc[: -self.test_length])
        pred = model.predict(X.iloc[-self.test_length :])
        self._metrics(X["y"].values[-self.test_length :], pred["yhat"].values)
        
        # test only
        if test_only:
            return (
                pred['yhat'], 
                model.predict(X.iloc[: -self.test_length])["yhat"].to_list() + [np.nan]*self.test_length, 
                model, 
                Xvars
            )

        # full
        regr = Prophet(**kwargs)
        regr.fit(X)
        fcst = regr.predict(p)
        return (fcst["yhat"],regr.predict(X)["yhat"],Xvars,regr)

    def _forecast_silverkite(self, tune=False, dynamic_testing=True, Xvars=None, test_only=False, **kwargs):
        """ forecasts with the silverkite model from LinkedIn greykite library.
        see example: https://scalecast-examples.readthedocs.io/en/latest/silverkite/silverkite.html

        Args:
            tune (bool): default False.
                whether to tune the forecast.
                does not need to be specified by user.
            dynamic_testing (bool): default True.
                always ignored for silverkite (for now).
            Xvars (list-like, str, or None): the regressors to predict with.
                be sure to have added them to the Forecaster object first.
                None means no Xvars used (unlike sklearn models).
            test_only (bool): default False:
                whether to stop your model after the testing process.
                all forecasts in history will be equivalent to test-set predictions.
                when forecasting to future periods, there is no way to skip the training process.
                any plot or export of forecasts into a future horizon will fail for this model
                and not all methods will raise descriptive errors.
            **kwargs: passed to the ModelComponentsParam function from greykite.framework.templates.autogen.forecast_config.
        """
        if not dynamic_testing:
            logging.warning(
                "dynamic_testing argument will be ignored for the silverkite model"
            )
        self.dynamic_testing = True
        from greykite.framework.templates.autogen.forecast_config import (
            ForecastConfig,
            MetadataParam,
            ModelComponentsParam,
            EvaluationPeriodParam,
        )
        from greykite.framework.templates.forecaster import Forecaster as SKForecaster

        Xvars = (
            [x for x in self.current_xreg.keys() if not x.startswith("AR")]
            if Xvars == "all"
            else [x for x in Xvars if not x.startswith("AR")]
            if Xvars is not None
            else []
        )

        def _forecast_sk(df, Xvars, validation_length, test_length, forecast_length):
            test_length = test_length if test_length > 0 else -(df.shape[0] + 1)
            validation_length = (validation_length if validation_length > 0 else -(df.shape[0] + 1))
            pred_df = df.iloc[:-test_length, :].dropna()
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
        reg_df = pd.DataFrame({x: self.current_xreg[x].to_list() + self.future_xreg[x] for x in Xvars})
        df = pd.concat([ts_df, reg_df], axis=1)

        if tune:
            y_test = self.y.values[-(self.test_length + self.validation_length) : -self.test_length]
            result = _forecast_sk(
                df,
                Xvars,
                self.validation_length,
                self.test_length,
                self.validation_length,
            )
            pred = result[0]
            self._metrics(y_test, pred[-self.validation_length:])
            return self._tune()

        result = _forecast_sk(df, Xvars, self.test_length, 0, self.test_length)
        Xvars = Xvars if Xvars != [] else None
        pred = result[0]
        self._metrics(self.y.values[-self.test_length :], pred[-self.test_length :])        
        if test_only:
            self.summary_stats = result[1].set_index("Pred_col")
            return (pred,pred[:-self.test_length],Xvars,None)
        
        result = _forecast_sk(df, Xvars, 0, 0, fcst_length)
        self.summary_stats = result[1].set_index("Pred_col")
        return (result[0][-fcst_length:],result[0][:-fcst_length],Xvars,None)

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
        test_only=False,
        **kwargs,
    ): 
        """ forecasts with a long-short term memory neural network from TensorFlow.
        cannot be tuned.
        only xvar options are the series' own history (specify in lags argument).
        always uses minmax normalizer.
        fitted values are the last fcst_length worth of values only.
        anything this function can do, rnn can also do. 
        this function is simpler to set up than rnn.
        see example: https://scalecast-examples.readthedocs.io/en/latest/lstm/lstm.html
            
        Args:
            dynamic_testing (bool): default True.
                always ignored for lstm.
            lags (int): greater than 0, default 1.
                the number of y-variable lags to train the model with.
            lstm_layer_sizes (list-like): default (25,).
                the size of each lstm layer to add.
                the first element is for the input layer.
                the size of this array minus 1 will equal the number of hidden layers in the resulting model.
            dropout (list-like): default (0.0,).
                the dropout rate for each lstm layer.
                must be the same size as lstm_layer_sizes.
            loss (str): default 'mean_absolute_error'.
                the loss function to minimize.
                see available options here:
                  https://www.tensorflow.org/api_docs/python/tf/keras/losses.
                be sure to choose one that is suitable for regression tasks.
            activation (str): default "tanh".
                the activation function to use in each lstm layer.
                see available values here:
                  https://www.tensorflow.org/api_docs/python/tf/keras/activations.
            optimizer (str): default "Adam".
                the optimizer to use when compiling the model.
                see available values here:
                  https://www.tensorflow.org/api_docs/python/tf/keras/optimizers.
            learning_rate (float): default 0.001.
                the learning rate to use when compiling the model.
            random_seed (int): optional.
                set a seed for consistent results.
                with tensorflow networks, setting seeds does not guarantee consistent results.
            plot_loss (bool): default False.
                whether to plot the LSTM loss function stored in history for each epoch.
                if validation_split passed to kwargs, will plot the validation loss as well.
                looks better if epochs > 1 passed to **kwargs.
            test_only (bool): default False:
                whether to stop your model after the testing process.
                all forecasts in history will be equivalent to test-set predictions.
                when forecasting to future periods, there is no way to skip the training process.
                any plot or export of forecasts into a future horizon will fail for this model
                and not all methods will raise descriptive errors.
            **kwargs: passed to fit() and can include epochs, verbose, callbacks, validation_split, and more
        """
        def convert_lstm_args_to_rnn(**kwargs):
            new_kwargs = {k:v for k,v in kwargs.items() if k not in ('lstm_layer_sizes','dropout','activation')}
            new_kwargs['layers_struct'] = [('LSTM',
                {
                    'units':v,
                    'activation':kwargs['activation'],
                    'dropout':kwargs['dropout'][i]
                }) for i,v in enumerate(kwargs['lstm_layer_sizes'])]
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

    def _forecast_rnn(
        self,
        dynamic_testing=True,
        lags=1,
        layers_struct=[('SimpleRNN',{'units':8,'activation':'tanh'})],
        loss="mean_absolute_error",
        optimizer="Adam",
        learning_rate=0.001,
        random_seed=None,
        plot_loss_test=False,
        plot_loss=False,
        test_only=False,
        **kwargs,
    ):
        """ forecasts with a recurrent neural network from TensorFlow, such as lstm or simple recurrent.
        not all features from tensorflow are available, but it is possible that more features will be added in the future.
        cannot be tuned.
        only xvar options are the series' own history (specified in lags argument).
        always uses minmax normalizer.
        see example: https://scalecast-examples.readthedocs.io/en/latest/rnn/rnn.html

        Args:
            dynamic_testing (bool): default True.
                always ignored for rnn.
            lags (int): greater than 0, default 1.
                the number of y-variable lags to train the model with.
            layers_struct (list[tuple[str,dict[str,Union[float,str]]]]): default [('SimpleRNN',{'units':8,'activation':'tanh'})].
                each element in the list is a tuple with two elements.
                first element of the list is the input layer (input_shape set automatically).
                first element of the tuple in the list is the type of layer ('SimpleRNN','LSTM', or 'Dense').
                second element is a dict.
                in the dict, key is a str representing hyperparameter name: 'units','activation', etc.
                val is hyperparameter value.
                see here for options related to SimpleRNN: https://www.tensorflow.org/api_docs/python/tf/keras/layers/SimpleRNN
                for LSTM: https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM
                for Dense: https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense
            loss (str or tf.keras.losses.Loss): default 'mean_absolute_error'.
                the loss function to minimize.
                see available options here: 
                  https://www.tensorflow.org/api_docs/python/tf/keras/losses.
                be sure to choose one that is suitable for regression tasks.
            optimizer (str or tf Optimizer): default "Adam".
                the optimizer to use when compiling the model.
                see available values here: 
                  https://www.tensorflow.org/api_docs/python/tf/keras/optimizers.
                if str, will use the optimizer with default args
                if type Optimizer, will use the optimizer exactly as specified
            learning_rate (float): default 0.001.
                the learning rate to use when compiling the model.
                ignored if you pass your own optimizer with a learning rate
            random_seed (int): optional.
                set a seed for consistent results.
                with tensorflow networks, setting seeds does not guarantee consistent results.
            plot_loss_test (bool): default False.
                whether to plot the loss trend stored in history for each epoch on the test set.
                if validation_split passed to kwargs, will plot the validation loss as well.
                looks better if epochs > 1 passed to **kwargs.
            plot_loss (bool): default False.
                whether to plot the loss trend stored in history for each epoch on the full model.
                if validation_split passed to kwargs, will plot the validation loss as well.
                looks better if epochs > 1 passed to **kwargs.
            test_only (bool): default False:
                whether to stop your model after the testing process.
                all forecasts in history will be equivalent to test-set predictions.
                when forecasting to future periods, there is no way to skip the training process.
                any plot or export of forecasts into a future horizon will fail for this model
                and not all methods will raise descriptive errors.
            **kwargs: passed to fit() and can include epochs, verbose, callbacks, validation_split, and more.
        """
        def get_compiled_model(y):
            # build model
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Dense, LSTM, SimpleRNN
            import tensorflow.keras.optimizers
            if isinstance(optimizer,str):
                local_optimizer = eval(f"tensorflow.keras.optimizers.{optimizer}")(learning_rate=learning_rate)
            else:
                local_optimizer = optimizer
            for i, kv in enumerate(layers_struct):                
                layer = locals()[kv[0]]

                if i == 0:
                    if kv[0] in ('LSTM','SimpleRNN'):
                        kv[1]['return_sequences'] = len(layers_struct) > 1
                    model = Sequential(
                        [
                            layer(
                                **kv[1],
                                input_shape=(n_timesteps, n_features),
                            )
                        ]
                    )
                else:
                    if kv[0] in ('LSTM','SimpleRNN'):
                        kv[1]['return_sequences'] = (not i == (len(layers_struct) - 1))
                        if kv[1]['return_sequences']:
                            kv[1]['return_sequences'] = layers_struct[i+1][0] != 'Dense'

                    model.add(layer(**kv[1]))
            model.add(Dense(y.shape[1]))  # output layer

            # compile model
            model.compile(optimizer=local_optimizer, loss=loss)
            return model
        
        descriptive_assert(
            len(layers_struct) >= 1,
            ValueError,
            f"must pass at least one layer to layers_struct, got {layers_struct}",
        )
        if not dynamic_testing:
            logging.warning(
                "dynamic_testing argument will be ignored for the rnn model"
            )
        self.dynamic_testing = None

        if random_seed is not None:
            random.seed(random_seed)

        y_train = self.y.values[:-self.test_length].copy()
        y_test = self.y.values[-self.test_length:].copy()

        ymin = y_train.min()
        ymax = y_train.max()

        X_train, y_train_new = prepare_rnn(y_train,lags,self.test_length)
        X, y_new = prepare_rnn(self.y.values.copy(),lags,len(self.future_dates))

        X_test = np.array([[(i - ymin) / (ymax - ymin) for i in self.y.values[-(lags + self.test_length):-self.test_length].copy()]])
        y_test_new = np.array([[(i - ymin) / (ymax - ymin) for i in self.y.values[-self.test_length:].copy()]])
        fut = np.array([[(i - self.y.min()) / (self.y.max() - self.y.min()) for i in self.y.values[-lags:].copy()]])

        n_timesteps = X_train.shape[1]
        n_features = 1

        X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],1)
        X_test = X_test.reshape(X_test.shape[0],X_test.shape[1],1)
        X = X.reshape(X.shape[0],X.shape[1],1)
        fut = fut.reshape(fut.shape[0],fut.shape[1],1)

        test_model = get_compiled_model(y_train_new)
        hist = test_model.fit(X_train,y_train_new,**kwargs)
        pred = test_model.predict(X_test)

        pred = [p * (ymax-ymin) + ymin for p in pred[0]] # un-minmax

        # set the test-set metrics
        self._metrics(y_test, pred)
        if plot_loss_test:
            plot_loss_rnn(hist,'model loss - test')
        if test_only:
            fvs = test_model.predict(X_train)
            fvs = [p[0] * (ymax-ymin) + ymin for p in fvs[1:][::-1]] + [
                                  p * (ymax-ymin) + ymin for p in fvs[0]]
            return (pred, fvs + [np.nan]*self.test_length, None, None)

        model = get_compiled_model(y_new)
        hist = model.fit(X,y_new,**kwargs)
        if plot_loss:
            plot_loss_rnn(hist,'model loss - full')
        fcst = model.predict(fut)
        fvs = model.predict(X)
        fvs = [p[0] * (self.y.max()-self.y.min()) + self.y.min() for p in fvs[1:][::-1]] + [
                              p * (self.y.max()-self.y.min()) + self.y.min() for p in fvs[0]]
        fcst = [p * (self.y.max()-self.y.min()) + self.y.min() for p in fcst[0]] 

        return (fcst, fvs, None, None)

    def _forecast_combo(
        self,
        how="simple",
        models="all",
        dynamic_testing=True,
        determine_best_by="ValidationMetricValue",
        rebalance_weights=0.1,
        weights=None,
        splice_points=None,
        test_only=False,
    ):
        """ combines at least two previously evaluted forecasts to create a new model.
        this method cannot be run on models that were run test_only = True.
        see the following explanation for the weighted-average model:
        The weighted model in scalecast uses a weighted average of all selected models, 
        applying the same weights to the fitted values, test-set metrics, and predictions. 
        A user can supply their own weights or let the algorithm determine optimal weights 
        based on a passed error metric (such as "TestSetMAPE"). To avoid overfitting, it is 
        recommended to use the default value, "ValidationSetMetric" to determine weights, 
        although this is not possible if the selected models have not all been tuned on the 
        validation set. The weighting uses a MaxMin scaler when an error metric is passed, 
        and a MinMax scaler when r-squared is selected as the metric to base weights on. 
        When this scaler is applied, the resulting values are then rebalanced to add to 1. 
        Since the worst-performing model in this case will always be weighted zero, 
        the user can select a factor to add to all scaled values before the rebalancing 
        is applied; by default, this is 0.1. The higher this factor is, the closer the weighted 
        average will be to a simple average and vice-versa.
        see example: https://scalecast-examples.readthedocs.io/en/latest/combo/combo.html

        Args:
            how (str): one of {'simple','weighted','splice'}, default 'simple'.
                the type of combination.
                if 'simple', uses a simple average.
                if 'weighted', uses a weighted average.
                if 'splice', splices several forecasts together at specified splice points.
            models (list-like or str): default 'all'.
                which models to combine.
                can start with top ('top_5').
            dynamic_testing (bool): default True.
                always ignored for combo (for now and possibly forever).
            determine_best_by (str): one of _determine_best_by_, default 'ValidationMetricValue'.
                if models does not start with 'top_' and how is not 'weighted', this is ignored.
                if how is 'weighted' and manual weights are specified, this is ignored.
            rebalance_weights (float): default 0.1.
                how to rebalance the weights when how = 'weighted'.
                the higher, the closer the weights will be to each other for each model.
                if 0, the worst-performing model will be weighted with 0.
                must be greater than or equal to 0.
            weights (list-like): optional.
                only applicable when how='weighted'.
                manually specifies weights.
                must be the same size as models.
                if None and how='weighted', weights are set automatically.
                if manually passed weights do not add to 1, will rebalance them.
            splice_points (list-like): optional.
                only applicable when how='splice'.
                elements in array must be str in '%Y-%m-%d' or datetime object.
                must be exactly one less in length than the number of models.
                models[0] --> :splice_points[0]
                models[-1] --> splice_points[-1]:
            test_only (bool): default False:
                always ignored in combo model.
        """
        if not dynamic_testing:
            logging.warning(
                "dynamic_testing argument ignored for the combo model"
            )
        if test_only:
            logging.warning(
                "test_only argument ignored for the combo model"
            )
        self.dynamic_testing = None
        self.test_only = False
        determine_best_by = (
            determine_best_by
            if (weights is None) & ((models[:4] == "top_") | (how == "weighted"))
            else None
            if how != "weighted"
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
        self._validate_no_test_only(models)
        descriptive_assert(
            len(models) > 1,
            ForecastError,
            f"need at least two models to average, got {len(models)}",
        )
        fcsts = pd.DataFrame({m: self.history[m]["Forecast"] for m in models})
        preds = pd.DataFrame({m: self.history[m]["TestSetPredictions"] for m in models})
        obs_to_keep = min(len(self.history[m]["FittedVals"]) for m in models)
        fvs = pd.DataFrame(
            {m: self.history[m]["FittedVals"][-obs_to_keep:] for m in models}
        )
        actuals = self.y.values[-preds.shape[0] :]
        if how == "weighted":
            scale = True
            if weights is None:
                weights = pd.DataFrame(
                    {m: [self.history[m][determine_best_by]] for m in models}
                )
            else:
                descriptive_assert(
                    len(weights) == len(models),
                    ForecastError,
                    "must pass as many weights as models",
                )
                descriptive_assert(
                    not isinstance(weights, str),
                    TypeError,
                    f"weights argument not recognized: {weights}",
                )
                weights = pd.DataFrame(zip(models, weights)).set_index(0).transpose()
                if weights.sum(axis=1).values[0] == 1:
                    scale = False
                    rebalance_weights = 0
            try:
                descriptive_assert(
                    rebalance_weights >= 0,
                    ValueError,
                    "when using a weighted average, rebalance_weights must be numeric and at least 0 in value",
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
                descriptive_assert(
                    len(models) == len(splice_points) + 1,
                    ForecastError,
                    "must have exactly 1 more model passed to models as splice points",
                )
                splice_points = pd.to_datetime(sorted(splice_points)).to_list()
                future_dates = self.future_dates.to_list()
                descriptive_assert(
                    np.array([p in future_dates for p in splice_points]).all(),
                    TypeError,
                    "all elements in splice_points must be datetime objects or str in '%Y-%m-%d' format and must be present in future_dates attribute",
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

        self._metrics(actuals, pred)
        self.weights = tuple(weights.values[0]) if weights is not None else None
        self.models = models
        return (fcst, fv, None, None)

    def _parse_models(self, models, determine_best_by) -> list:
        """ takes a collection of models and orders them best-to-worst based on a given metric and returns the ordered list (of str type).

        Args:
            models (list-like): each element is one of _estimators_
            determine_best_by (str): one of _determine_best_by_
                if a model does not have the metric specified here (i.e. one of the passed models wasn't tuned and this is 'ValidationMetricValue'), it will be ignored silently, so be careful

        Returns:
            (list) The ordered evaluated models.
        """
        if determine_best_by is None:
            if models[:4] == "top_":
                raise ValueError(
                    'cannot use models starts with "top_" unless the determine_best_by or order_by argument is specified'
                )
            elif models == "all":
                models = list(self.history.keys())
            elif isinstance(models, str):
                models = [models]
            else:
                models = list(models)
            if len(models) == 0:
                raise ValueError(
                    f"models argument with determine_best_by={determine_best_by} returns no evaluated forecasts"
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

    def _diffy(self, n) -> pd.Series:
        """ parses the argument fed to a diffy parameter

        Args:
            n (bool or int): one of {True,False,0,1,2}.
                If False or 0, does not difference.
                If True or 1, differences 1 time.
                If 2, differences 2 times.

        Returns:
            (Series): The differenced array.
        """
        n = int(n)
        descriptive_assert(
            (n <= 2) & (n >= 0),
            ValueError,
            "diffy cannot be less than 0 or greater than 2",
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

    def fillna_y(self, how="ffill"):
        """ fills null values in the y attribute.

        Args:
            how (str): one of {'backfill', 'bfill', 'pad', 'ffill', 'midpoint'}.
                midpoint is unique to this library and only works if there is not more than two missing values sequentially.
                all other possible arguments are from pandas.DataFrame.fillna() method and will do the same.

        Returns:
            None
        """
        self.y = pd.Series(self.y)
        if (
            how != "midpoint"
        ):  # only works if there aren't more than 2 na one after another
            self.y = self.y.fillna(method=how)
        else:
            for i, val in enumerate(self.y.values):
                if val is None:
                    self.y.values[i] = (self.y.values[i - 1] + self.y.values[i + 1]) / 2

    def generate_future_dates(self, n):
        """ generates a certain amount of future dates in same frequency as current_dates.

        Args:
            n (int): greater than 0.
                number of future dates to produce.
                this will also be the forecast length.

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
        """ generates future dates in the same frequency as current_dates that ends on a specified date.

        Args:
            date (datetime.datetime, pd.Timestamp, or str):
                the date to end on. if str, must be in '%Y-%m-%d' format.

        Returns:
            None

        >>> f.set_last_future_date('2021-06-01') # creates future dates up to this one in the expected frequency
        """
        if isinstance(date, str):
            date = datetime.datetime.strptime(date, "%Y-%m-%d")
        self.future_dates = pd.Series(
            pd.date_range(
                start=self.current_dates.values[-1], end=date, freq=self.freq
            ).values[1:]
        )

    def typ_set(self):
        """ converts all objects in y, current_dates, future_dates, current_xreg, and future_xreg to appropriate types if possible.
        automatically gets called when object is initiated.

        >>> f.typ_set() # sets all arrays to the correct format
        """
        self.y = pd.Series(self.y).dropna().astype(np.float64)
        self.current_dates = pd.to_datetime(
            pd.Series(list(self.current_dates)[-len(self.y) :]),
            infer_datetime_format=True,
        )
        descriptive_assert(
            len(self.y) == len(self.current_dates),
            ValueError,
            f"y and current_dates must be same size -- y is {len(self.y)} and current_dates is {len(self.current_dates)}",
        )
        self.future_dates = pd.to_datetime(
            pd.Series(self.future_dates), infer_datetime_format=True
        )
        for k, v in self.current_xreg.items():
            self.current_xreg[k] = pd.Series(list(v)[-len(self.y) :]).astype(np.float64)
            descriptive_assert(
                len(self.current_xreg[k]) == len(self.y),
                ForecastError,
                "something went wrong when setting covariate values--try resetting the object and trying again",
            )
            self.future_xreg[k] = [float(x) for x in self.future_xreg[k]]

        self.infer_freq()

    def diff(self, i=1):
        """ differences the y attribute, as well as all AR values stored in current_xreg and future_xreg.

        Args:
            i (int): default 1.
                the number of differences to take.
                must be 1 or 2.

        Returns:
            None

        >>> f.diff(2) # differences y twice
        """
        descriptive_assert(
            self.integration == 0,
            ForecastError.CannotDiff,
            "series has already been differenced, if you want to difference again, use undiff() first, then diff(2)",
        )
        if i == 0:
            return

        descriptive_assert(
            i in (1, 2),
            ValueError,
            f"only 1st and 2nd order integrations supported for now, got i={i}",
        )
        self.integration = i
        for _ in range(i):
            self.y = self.y.diff()
        for k, v in self.current_xreg.items():
            if k.startswith("AR"):
                ar = int(k[2:])
                for _ in range(i):
                    self.current_xreg[k] = v.diff()
                self.future_xreg[k] = [self.y.values[-ar]]

        delattr(self, "adf_stationary") if hasattr(self, "adf_stationary") else None

    def integrate(
        self, critical_pval=0.05, train_only=False, max_integration=2
    ):
        """ differences the series 0, 1, or 2 times based on ADF test results.

        Args:
            critical_pval (float): default 0.05.
                the p-value threshold in the statistical test to accept the alternative hypothesis.
            train_only (bool): default False.
                if True, will exclude the test set from the ADF test (to avoid leakage).
            max_integration (int): one of {1,2}, default 2.
                if 1, will only difference data up to one time even if the results of the test indicate two integrations.
                if 2, behaves how you would expect.

        Returns:
            None

        >>> f.integrate(max_integration=1) # differences y only once if it is not stationarity
        >>> f.integrate() # differences y up to twice it is not stationarity and if its first difference is not stationary
        """
        descriptive_assert(
            self.integration == 0,
            ForecastError,
            "can only run integrate() when series hasn't been differenced",
        )
        descriptive_assert(
            isinstance(train_only, bool), ValueError, "train_only must be True or False"
        )
        res0 = adfuller(
            self.y.dropna()
            if not train_only
            else self.y.dropna().values[: -self.test_length]
        )
        if res0[1] <= critical_pval:
            return

        res1 = adfuller(
            self.y.diff().dropna()
            if not train_only
            else self.y.diff().dropna().values[: -self.test_length]
        )
        if (res1[1] <= critical_pval) | (max_integration == 1):
            self.diff()
            return

        self.diff(2)
        self.adf_stationary = True

    def add_ar_terms(self, n):
        """ adds auto-regressive terms.

        Args:
            n (int): the number of terms to add (1 to this number will be added).

        Returns:
            None

        >>> f.add_ar_terms(4) # adds four lags of y to predict with
        """
        self._validate_future_dates_exist()
        
        if n == 0:
            return

        descriptive_assert(isinstance(n, int), ValueError, f"n must be an int, got {n}")
        descriptive_assert(n >= 0, ValueError, f"n must be greater than or equal to 0, got {n}")
        descriptive_assert(
            self.integration == 0,
            ForecastError,
            "AR terms must be added before differencing (don't worry, they will be differenced too)",
        )
        for i in range(1, n + 1):
            self.current_xreg[f"AR{i}"] = pd.Series(self.y).shift(i)
            self.future_xreg[f"AR{i}"] = [self.y.values[-i]]

    def add_AR_terms(self, N):
        """ adds seasonal auto-regressive terms.
            
        Args:
            N (tuple): first element is the number of terms to add and the second element is the space between terms.

        Returns:
            None

        >>> f.add_AR_terms((2,12)) # adds 12th and 24th lags
        """
        self._validate_future_dates_exist()
        descriptive_assert(
            (len(N) == 2) & (not isinstance(N, str)),
            ValueError,
            f"n must be an array-like of length 2 (P,m), got {N}",
        )
        descriptive_assert(
            self.integration == 0,
            ForecastError,
            "AR terms must be added before differencing (don't worry, they will be differenced too)",
        )
        for i in range(N[1], N[1] * N[0] + 1, N[1]):
            self.current_xreg[f"AR{i}"] = pd.Series(self.y).shift(i)
            self.future_xreg[f"AR{i}"] = [self.y.values[-i]]

    def ingest_Xvars_df(
        self, df, date_col="Date", drop_first=False, use_future_dates=False
    ):
        """ ingests a dataframe of regressors and saves its contents to the Forecaster object.
        must specify a date column.
        all non-numeric values will be dummied.
        any columns in the dataframe that begin with "AR" will be confused with autoregressive terms and could cause errors.

        Args:
            df (DataFrame): the dataframe that is at least the length of len(current_dates) + len(future_dates)
            date_col (str): default 'Date'.
                the name of the date column in the dataframe.
                this column must have the same frequency as the dates in current_dates.
            drop_first (bool): default False.
                whether to drop the first observation of any dummied variables.
                irrelevant if passing all numeric values.
            use_future_dates (bool): default False.
                whether to use the future dates in the dataframe as the future_dates attribute in the object.

        Returns:
            None
        """
        descriptive_assert(
            df.shape[0] == len(df[date_col].unique()),
            ValueError,
            "each date supplied must be unique",
        )
        df[date_col] = pd.to_datetime(df[date_col]).to_list()
        df = df.loc[df[date_col] >= self.current_dates.values[0]]
        df = pd.get_dummies(df, drop_first=drop_first)
        current_df = df.loc[df[date_col].isin(self.current_dates)]
        future_df = df.loc[df[date_col] > self.current_dates.values[-1]]
        descriptive_assert(
            current_df.shape[0] == len(self.y),
            ForecastError,
            "something went wrong--make sure the dataframe spans the entire daterange as y and is at least one observation to the future and specify a date column in date_col parameter",
        )
        if not use_future_dates:
            descriptive_assert(
                future_df.shape[0] >= len(self.future_dates),
                ValueError,
                "the future dates in the dataframe should be at least the same length as the future dates in the Forecaster object. if you desire to use the dataframe to set the future dates for the object, use use_future_dates=True",
            )
        else:
            self.future_dates = future_df[date_col]

        for c in [c for c in future_df if c != date_col]:
            self.future_xreg[c] = future_df[c].to_list()[: len(self.future_dates)]
            self.current_xreg[c] = current_df[c]

        for x, v in self.future_xreg.items():
            self.future_xreg[x] = v[: len(self.future_dates)]
            if not len(v) == len(self.future_dates):
                logging.warning(
                    f"warning: {x} is not the correct length in the future_dates attribute and this can cause errors when forecasting. its length is {len(v)} and future_dates length is {len(self.future_dates)}"
                )

    def reduce_Xvars(
        self,
        method='l1',
        estimator='lasso',
        keep_at_least=1,
        keep_this_many='auto',
        grid_search=True,
        use_loaded_grid=False,
        dynamic_tuning=False,
        monitor='ValidationMetricValue',
        overwrite=True,
        **kwargs,
    ):
        """ reduces the regressor variables stored in the object. two methods are available:
        l1 which uses a simple l1 penalty and Lasso regressor; as well as pfi that stands for 
        permutation feature importance and offers more flexibility to view how removing
        variables one-at-a-time, according to which variable is evaluated as least helpful to the
        model after each model evaluation, affects a given error metric for any scikit-learn model.
        after each variable reduction, the model is re-run and pfi re-evaluated. feature scores
        are adjusted to account for colinearity, a limitation of pfi, by sorting by each feature's score
        and standard deviation, dropping variables first that have both a low score and low standard deviation.
        by default, the validation-set error is used to avoid leakage and the variable set that most 
        reduced the error is selected.

        Args:
            method (str): one of {'l1','pfi'}, default 'l1'.
                the reduction method. 
                l1 uses a lasso regressor and grid searches for the optimal alpha on the validation set
                unless an alpha value is passed to the hyperparams arg and grid_search arg is False.
                pfi uses permutation feature importance and is more computationally expensive
                but can use any sklearn estimator.
                method "pfi" creates attributes in object called pfi_dropped_vars and pfi_error_values that are two lists
                that represent the error change with the corresponding dropped var.
                the pfi_error_values attr is one greater in length than pfi_dropped_vars attr because 
                the first error is the intial error before any variables were dropped.
            estimator (str): one of _sklearn_estimators_ or 'lasso'. default 'lasso'.
                the estimator to use to determine the best set of vars.
                if method == 'l1', estimator arg is ignored and is always lasso.
            keep_at_least (str or int): default 1.
                the fewest number of Xvars to keep if method == 'pfi'.
                'sqrt' keeps at least the sqare root of the number of Xvars rounded down.
                this exists so that the keep_this_many keyword can use 'auto' as an argument.
            keep_this_many (str or int): default 'auto'.
                the number of Xvars to keep if method == 'pfi'.
                "auto" keeps the number of xvars that returned the best error using the 
                metric passed to monitor, but it is the most computationally expensive.
                "sqrt" keeps the square root of the total number of observations rounded down.
            gird_search (bool): default True.
                whether to run a grid search for optimal hyperparams on the validation set.
                if use_loaded_grid is False, uses the Grids.py file currently available in the working directory 
                or creates a new Grids.py file with default values if none available to determine the grid to use.
                the grid search is only run once and then those hyperparameters are used for
                all subsequent pfi runs when method == 'pfi'.
                in any utilized grid, do not include 'Xvars' as a key.
                if you want to access the chosen hyperparams after the fact, they are stored in the reduction_hyperparams
                attribute.
            use_loaded_grid (bool): default False.
                whether to use the currently loaded grid in the object instead of using a grid in Grids.py.
                in any utilized grid, do not include 'Xvars' as a key.
            dynamic_tuning (bool): default False.
                whether to dynamically tune the model.
            monitor (str): one of _determine_best_by_. default 'ValidationSetMetric'.
                ignored when pfi == 'l1'.
                the metric to be monitored when making reduction decisions. 
            overwrite (bool): default True.
                if False, the list of selected Xvars are stored in an attribute called reduced_Xvars.
                if True, this list of regressors overwrites the current Xvars in the object.
            **kwargs: passed to manual_forecast() method and can include arguments related to 
                a given model's hyperparameters or dynamic_testing.
                do not pass hyperparameters if grid_search is True.
                do not pass Xvars.

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
                method='pfi',
                estimator='gbt',
                keep_at_least=10,
                keep_this_many='auto',
                dynamic_testing=False,
                dynamic_tuning=True,
            ) # reduce with gradient boosted tree estimator
        >>> print(f.reduced_Xvars) # view results
        """
        descriptive_assert(
            method in ('l1','pfi'),
            ValueError,f'method must be one of "pfi", "l1", got {method}'
        )
        f = self.__deepcopy__()
        descriptive_assert(
            estimator in _sklearn_estimators_,
            ValueError,
            f'estimator must be one of {_sklearn_estimators_}, got {estimator}'
        )
        f.set_estimator(estimator if method == 'pfi' else 'lasso')
        if grid_search:
            if not use_loaded_grid:
                from scalecast import GridGenerator
                GridGenerator.get_example_grids()
                f.ingest_grid(estimator)
            else:
                descriptive_assert(
                    hasattr(f,'grid'),
                    ForecastError,
                    'grid not loaded, try setting use_loaded_grid to False'
                )
        else:
            f.ingest_grid({k:[v] for k,v in kwargs.items()})

        f.tune(dynamic_tuning=dynamic_tuning)
        f.ingest_grid({k:[v] for k,v in f.best_params.items()})
        f.auto_forecast(test_only=True)

        self.reduction_hyperparams = f.best_params.copy()

        if method == 'l1':
            coef_fi_lasso = pd.DataFrame(
                {x: [np.abs(co)] for x, co in zip(f.history["lasso"]["Xvars"], f.regr.coef_)},
                index=["feature"],
            ).T
            self.reduced_Xvars = coef_fi_lasso.loc[coef_fi_lasso["feature"] != 0].index.to_list()
        else:
            f.save_feature_importance()
            pfi_df = f.export_feature_importance(estimator)
            using_r2 = (
                monitor.endswith('R2') or (f.validation_metric == 'r2' and monitor == 'ValidationMetricValue')
            )
            pfi_df["weight"] = np.abs(pfi_df["weight"])
            pfi_df.sort_values(["weight", "std"], ascending=False, inplace=True)
            features = pfi_df.index.to_list()
            init_error = f.history[estimator][monitor]
            init_error = -init_error if using_r2 else init_error

            dropped = []
            errors = [init_error]

            sqrt = int(len(f.y) ** 0.5)
            keep_this_many_new = (
                1 if keep_this_many == 'auto'
                else sqrt if keep_this_many == 'sqrt'
                else keep_this_many
            )
            keep_at_least_new = (
                sqrt if keep_at_least == 'sqrt'
                else keep_at_least
            )

            stop_at = max(keep_this_many_new,keep_at_least_new)
            drop_this_many = len(f.current_xreg) - stop_at

            for _ in range(drop_this_many):
                dropped.append(features[-1])
                features = features[:-1]
                f.grid['Xvars'] = [features]
                f.tune(dynamic_tuning=dynamic_tuning)
                f.auto_forecast(test_only=True)
                new_error = f.history[estimator][monitor]
                new_error = -new_error if using_r2 else new_error
                errors.append(new_error)

                f.save_feature_importance()
                pfi_df = f.export_feature_importance(estimator)
                pfi_df['weight'] = np.abs(pfi_df['weight'])
                pfi_df.sort_values(["weight", "std"], ascending=False, inplace=True)
                features = pfi_df.index.to_list()

            optimal_drop = errors.index(min(errors)) if keep_this_many == 'auto' else drop_this_many
            self.reduced_Xvars = [x for x in self.current_xreg.keys() if x not in dropped[:optimal_drop]]
            self.pfi_dropped_vars = dropped
            self.pfi_error_values = [-e for e in errors] if using_r2 else errors

        if overwrite:
            self.current_xreg = {x:v for x,v in self.current_xreg.items() if x in self.reduced_Xvars}
            self.future_xreg = {x:v for x,v in self.future_xreg.items() if x in self.reduced_Xvars}

    def set_test_length(self, n=1):
        """ sets the length of the test set.

        Args:
            n (int or float): default 1.
                the length of the resulting test set.
                fractional splits are supported by passing a float less than 1 and greater than 0.

        Returns:
            None

        >>> f.set_test_length(12) # test set of 12
        >>> f.set_test_length(.2) # 20% test split
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
            self.test_length = int(len(self.y) * n)
        
    def set_validation_length(self, n=1):
        """ sets the length of the validation set.

        Args:
            n (int): default 1.
                the length of the resulting validation set.

        Returns:
            None

        >>> f.set_validation_length(6) # validation length of 6
        """
        descriptive_assert(isinstance(n, int), ValueError, f"n must be an int, got {n}")
        descriptive_assert(n > 0, ValueError, f"n must be greater than 1, got {n}")
        if (self.validation_metric == "r2") & (n == 1):
            raise ValueError(
                "can only set a validation_length of 1 if validation_metric is not r2. try set_validation_metric()"
            )
        self.validation_length = n

    def set_cilevel(self, n):
        """ sets the level for the resulting confidence intervals (95% default).

        Args:
            n (float): greater than 0 and less than 1.

        Returns:
            None

        >>> f.set_cilevel(.80) # next forecast will get 80% confidence intervals
        """
        descriptive_assert(
            n < 1 and n > 0, ValueError, "n must be greater than 0 and less than 1"
        )
        self.cilevel = n

    def set_bootstrap_samples(self, n):
        """ sets the number of bootstrap samples to set confidence intervals for each model (100 default).

        Args:
            n (int): greater than or equal to 30.
                30 because you need around there to satisfy central limit theorem.
                the lower this number, the faster the performance, but the less confident in the resulting intervals you should be.

        Returns:
            None

        >>> f.set_bootstrap_samples(1000) # next forecast will get confidence intervals with 1,000 bootstrap sample
        """
        descriptive_assert(n >= 30, ValueError, "n must be greater than or equal to 30")
        self.bootstrap_samples = n

    def adf_test(
        self, critical_pval=0.05, quiet=True, full_res=False, train_only=False, **kwargs
    ) -> Union[tuple, bool]:
        """ tests the stationarity of the y series using augmented dickey fuller.

        Args:
            critical_pval (float): default 0.05.
                the p-value threshold in the statistical test to accept the alternative hypothesis.
            quiet (bool): default True.
                if False, prints whether the tests suggests stationary or non-stationary data.
            full_res (bool): default False.
                if True, returns a dictionary with the pvalue, evaluated statistic, and other statistical information (returns what the adfuller() function from statsmodels does).
                if False, returns a bool that matches whether the test indicates stationarity.
            train_only (bool): default False.
                if True, will exclude the test set from the test (to avoid leakage).
            **kwargs: passed to adfuller() function from statsmodels.

        Returns:
            (bool or tuple): if bool (full_res = False), returns whether the test suggests stationarity.
                otherwise, returns the full results (stat, pval, etc.) of the test.

        >>> stat, pval, _, _, _, _ = f.adf_test(full_res=True)
        """
        descriptive_assert(
            isinstance(train_only, bool), ValueError, "train_only must be True or False"
        )
        res = adfuller(
            self.y.dropna()
            if not train_only
            else self.y.dropna().values[: -self.test_length],
            **kwargs,
        )
        if not full_res:
            if res[1] <= critical_pval:
                if not quiet:
                    print("series appears to be stationary")
                self.adf_stationary = True
                return True
            else:
                if not quiet:
                    print("series might not be stationary")
                self.adf_stationary = False
                return False
        else:
            return res

    def plot_acf(self, diffy=False, train_only=False, **kwargs):
        """ plots an autocorrelation function of the y values.

        Args:
            diffy (bool or int): one of {True,False,0,1,2}. default False.
                whether to difference the data and how many times before passing the values to the function.
                if False or 0, does not difference.
                if True or 1, differences 1 time.
                if 2, differences 2 times.
            train_only (bool): default False.
                if True, will exclude the test set from the test (a measure added to avoid leakage).
            **kwargs: passed to plot_acf() function from statsmodels.

        Returns:
            (Figure): If ax is None, the created figure. Otherwise the figure to which ax is connected.

        >>> import matplotlib.pyplot as plt
        >>> f.plot_acf(train_only=True)
        >>> plt.plot()
        """
        descriptive_assert(
            isinstance(train_only, bool), ValueError, "train_only must be True or False"
        )
        y = self._diffy(diffy)
        y = y.values if not train_only else y.values[: -self.test_length]
        return plot_acf(y, **kwargs)

    def plot_pacf(self, diffy=False, train_only=False, **kwargs):
        """ plots a partial autocorrelation function of the y values

        Args:
            diffy (bool or int): one of {True,False,0,1,2}. default False.
                whether to difference the data and how many times before passing the values to the function.
                if False or 0, does not difference.
                if True or 1, differences 1 time.
                if 2, differences 2 times.
            train_only (bool): default False.
                if True, will exclude the test set from the test (a measure added to avoid leakage).
            **kwargs: passed to plot_pacf() function from statsmodels.

        Returns:
            (Figure): If ax is None, the created figure. Otherwise the figure to which ax is connected.

        >>> import matplotlib.pyplot as plt
        >>> f.plot_pacf(train_only=True)
        >>> plt.plot()
        """
        descriptive_assert(
            isinstance(train_only, bool), ValueError, "train_only must be True or False"
        )
        y = self._diffy(diffy)
        y = y.values if not train_only else y.values[: -self.test_length]
        return plot_pacf(y, **kwargs)

    def plot_periodogram(self, diffy=False, train_only=False):
        """ plots a periodogram of the y values (comes from scipy.signal).

        Args:
            diffy (bool or int): one of {True,False,0,1,2}. default False.
                whether to difference the data and how many times before passing the values to the function.
                if False or 0, does not difference.
                if True or 1, differences 1 time.
                if 2, differences 2 times.
            train_only (bool): default False.
                if True, will exclude the test set from the test (a measure added to avoid leakage).

        Returns:
            (ndarray,ndarray): Element 1: Array of sample frequencies. Element 2: Power spectral density or power spectrum of x.

        >>> import matplotlib.pyplot as plt
        >>> a, b = f.plot_periodogram(diffy=True,train_only=True)
        >>> plt.semilogy(a, b)
        >>> plt.show()
        """
        from scipy.signal import periodogram

        descriptive_assert(
            isinstance(train_only, bool), ValueError, "train_only must be True or False"
        )
        y = self._diffy(diffy)
        y = y.values if not train_only else y.values[: -self.test_length]
        return periodogram(y)

    def seasonal_decompose(self, diffy=False, train_only=False, **kwargs):
        """ plots a signal/seasonal decomposition of the y values.

        Args:
            diffy (bool or int): one of {True,False,0,1,2}. default False.
                whether to difference the data and how many times before passing the values to the function.
                if False or 0, does not difference.
                if True or 1, differences 1 time.
                if 2, differences 2 times.
            train_only (bool): default False.
                If True, will exclude the test set from the test (a measure added to avoid leakage).
            **kwargs: passed to seasonal_decompose() function from statsmodels.

        Returns:
            (DecomposeResult): An object with seasonal, trend, and resid attributes.

        >>> import matplotlib.pyplot as plt
        >>> f.seasonal_decompose(train_only=True).plot()
        >>> plt.show()
        """
        descriptive_assert(
            isinstance(train_only, bool), ValueError, "train_only must be True or False"
        )
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

    def add_seasonal_regressors(
        self, *args, raw=True, sincos=False, dummy=False, drop_first=False
    ):
        """ adds seasonal regressors.

        Args:
            *args: each of str type.
                values that return a series of int type from pandas.dt and pandas.dt.isocalendar().
                see https://pandas.pydata.org/docs/reference/api/pandas.Series.dt.year.html.
            raw (bool): default True.
                whether to use the raw integer values
            sincos (bool): default False.
                whether to use a sin/cos transformation of the raw integer values (estimates the cycle based on the max observed value)
            dummy (bool): default False.
                whether to use dummy variables from the raw int values
            drop_first (bool): default False.
                whether to drop the first observed dummy level.
                not relevant when dummy = False

        Returns:
            None

        >>> f.add_seasonal_regressors('year')
        >>> f.add_seasonal_regressors('month','week','quarter',raw=False,sincos=True)
        >>> f.add_seasonal_regressors('dayofweek',raw=False,dummy=True,drop_first=True)
        """
        self._validate_future_dates_exist()
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
                    _raw.max()
                )  # does not always capture the complete cycle, but this is probably the best we can do
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
        """ adds a time trend from 1 to len(current_dates) + len(future_dates) in current_xreg and future_xreg.

        Args:
            called (str): default 't'.
                what to call the resulting variable

        Returns:
            None

        >>> f.add_time_trend()
        """
        self._validate_future_dates_exist()
        self.current_xreg[called] = pd.Series(range(1, len(self.y) + 1))
        self.future_xreg[called] = list(
            range(len(self.y) + 1, len(self.y) + len(self.future_dates) + 1)
        )

    def add_cycle(self,cycle_length,called=None):
        """ adds a regressor that acts as a seasonal cycle.
        use this function to capture non-normal seasonality.

        Args:
            cycle_length (int): how many time steps make one complete cycle.
            called (str): optional. what to call the resulting variable.
                two variables will be created--one for a sin transformation and the other for cos
                resulting variable names will have "sin" or "cos" at the end.
                example, called = 'cycle5' will become 'cycle5sin', 'cycle5cos'.
                if left unspecified, 'cycle{cycle_length}' will be used as the name.

        Returns:
            None

        >>> f.add_cycle(13) # adds a seasonal effect that cycles every 13 observations
        """
        self._validate_future_dates_exist()
        if called is None:
            called = f'cycle{cycle_length}'
        full_sin = pd.Series(range(1, len(self.y) + len(self.future_dates) + 1)).apply(lambda x: np.sin(np.pi * x / (cycle_length / 2)))
        full_cos = pd.Series(range(1, len(self.y) + len(self.future_dates) + 1)).apply(lambda x: np.cos(np.pi * x / (cycle_length / 2)))
        self.current_xreg[called + 'sin'] = pd.Series(full_sin.values[:len(self.y)])
        self.current_xreg[called + 'cos'] = pd.Series(full_cos.values[:len(self.y)])
        self.future_xreg[called + 'sin'] = list(full_sin.values[len(self.y):])
        self.future_xreg[called + 'cos'] = list(full_cos.values[len(self.y):])

    def add_other_regressor(self, called, start, end):
        """ adds dummy variable that is 1 during the specified time period, 0 otherwise.

        Args:
            called (str):
                what to call the resulting variable.
            start (str, datetime.datetime, or pd.Timestamp): start date.
                use format '%Y-%m-%d' when passing strings.
            end (str, datetime.datetime, or pd.Timestamp): end date.
                use format '%Y-%m-%d' when passing strings.

        Returns:
            None

        >>> f.add_other_regressor('january_2021','2021-01-01','2021-01-31')
        """
        self._validate_future_dates_exist()
        if isinstance(start, str):
            start = datetime.datetime.strptime(start, "%Y-%m-%d")
        if isinstance(end, str):
            end = datetime.datetime.strptime(end, "%Y-%m-%d")
        self.current_xreg[called] = pd.Series(
            [1 if (x >= start) & (x <= end) else 0 for x in self.current_dates]
        )
        self.future_xreg[called] = [
            1 if (x >= start) & (x <= end) else 0 for x in self.future_dates
        ]

    def add_covid19_regressor(
        self,
        called="COVID19",
        start=datetime.datetime(2020, 3, 15),
        end=datetime.datetime(2021, 5, 13),
    ):
        """ adds dummy variable that is 1 during the time period that covid19 effects are present for the series, 0 otherwise.
        the default dates are selected to be optimized for the time-span where the economy was most impacted by COVID.

        Args:
            called (str): default 'COVID19'.
               what to call the resulting variable.
            start (str, datetime.datetime, or pd.Timestamp): default datetime.datetime(2020,3,15).
                the start date (default is day Walt Disney World closed in the U.S.).
                use format '%Y-%m-%d' when passing strings.
            end: (str, datetime.datetime, or pd.Timestamp): default datetime.datetime(2021,5,13).
               the end date (default is day the U.S. CDC dropped mask mandate/recommendation for vaccinated people).
               use format '%Y-%m-%d' when passing strings.

        Returns:
            None
        """
        self._validate_future_dates_exist()
        self.add_other_regressor(called=called, start=start, end=end)

    def add_combo_regressors(self, *args, sep="_"):
        """ combines all passed variables by multiplying their values together.

        Args:
            *args (str): names of Xvars that aleady exist in the object.
            sep (str): default '_'.
                the separator between each term in arg to create the final variable name.

        Returns:
            None

        >>> f.add_combo_regressors('t','monthsin') # multiplies these two together
        >>> f.add_combo_regressors('t','monthcos') # multiplies these two together
        """
        self._validate_future_dates_exist()
        descriptive_assert(
            len(args) > 1,
            ForecastError,
            "need at least two variables to combine regressors",
        )
        for i, a in enumerate(args):
            descriptive_assert(
                not a.startswith("AR"),
                ForecastError,
                "no combining AR terms at this time -- it confuses the forecasting mechanism",
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
            *args (str): names of Xvars that aleady exist in the object
            pwr (int): default 2.
                the max power to add to each term in args (2 to this number will be added).
            sep (str): default '^'.
                the separator between each term in arg to create the final variable name.

        Returns:
            None

        >>> f.add_poly_terms('t','year',pwr=3) ### raises t and year to 2nd and 3rd powers
        """
        self._validate_future_dates_exist()
        for a in args:
            descriptive_assert(
                not a.startswith("AR"),
                ForecastError,
                "no polynomial AR terms at this time -- it confuses the forecasting mechanism",
            )
            for i in range(2, pwr + 1):
                self.current_xreg[f"{a}{sep}{i}"] = self.current_xreg[a] ** i
                self.future_xreg[f"{a}{sep}{i}"] = [x ** i for x in self.future_xreg[a]]

    def add_exp_terms(self, *args, pwr, sep="^", cutoff=2, drop=False):
        """ raises all passed variables (no AR terms) to exponential powers (ints or floats).

        Args:
            *args (str): names of Xvars that aleady exist in the object.
            pwr (float): 
                the power to raise each term to in args.
                can use values like 0.5 to perform square roots, etc.
            sep (str): default '^'.
                the separator between each term in arg to create the final variable name.
            cutoff (int): default 2.
                the resulting variable name will be rounded to this number based on the passed pwr.
                for instance, if pwr = 0.33333333333 and 't' is passed as an arg to *args, the resulting name will be t^0.33 by default.
            drop (bool): default False.
                whether to drop the regressors passed to *args.

        Returns:
            None

        >>> f.add_exp_terms('t',pwr=.5) # adds square root t
        """
        self._validate_future_dates_exist()
        pwr = float(pwr)
        for a in args:
            descriptive_assert(
                not a.startswith("AR"),
                ForecastError,
                "no exponent AR terms at this time -- it confuses the forecasting mechanism",
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
        """ logs all passed variables (no AR terms).

        Args:
            *args (str): names of Xvars that aleady exist in the object.
            base (float): default math.e. the log base.
                must be math.e or int greater than 1.
            sep (str): default ''.
                the separator between each term in arg to create the final variable name.
                resulting variable names will be like "log2t" or "lnt" by default
            drop (bool): default False.
                whether to drop the regressors passed to *args.

        Returns:
            None

        >>> f.add_logged_terms('t') # adds natural log t
        """
        self._validate_future_dates_exist()
        for a in args:
            descriptive_assert(
                not a.startswith("AR"),
                ForecastError,
                "no logged AR terms at this time -- it confuses the forecasting mechanism",
            )
            if base == math.e:
                pass
            elif not (isinstance(base, int)):
                raise ValueError(
                    f"base must be math.e or an int greater than 1, got {base}"
                )
            elif base <= 1:
                raise ValueError(
                    f"base must be math.e or an int greater than 1, got {base}"
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
        """ applies a box-cox or yeo-johnson power transformation to all passed variables (no AR terms).

        Args:
            *args (str): names of Xvars that aleady exist in the object
            method (str): one of {'box-cox','yeo-johnson'}, default 'box-cox'.
                the type of transformation.
                box-cox works for positive values only.
                yeo-johnson is like a box-cox but can be used with 0s or negatives.
                https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PowerTransformer.html.
            sep (str): default ''.
                the separator between each term in arg to create the final variable name.
                resulting variable names will be like "box-cox_t" or "yeo-johnson_t" by default.
            drop (bool): default False.
                whether to drop the regressors passed to *args.

        Returns:
            None

        >>> f.add_pt_terms('t') # adds box cox of t
        """
        self._validate_future_dates_exist()
        pt = PowerTransformer(method=method, standardize=False)
        for a in args:
            descriptive_assert(
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

    def add_diffed_terms(self, *args, diff=1, sep="_", drop=False):
        """ differences all passed variables (no AR terms) up to 2 times.

        Args:
            *args (str): names of Xvars that aleady exist in the object.
            diff (int): one of {1,2}, default 1.
                the number of times to difference each variable passed to args.
            sep (str): default '_'.
                the separator between each term in arg to create the final variable name.
                resulting variable names will be like "tdiff_1" or "tdiff_2" by default.
            drop (bool): default False.
                whether to drop the regressors passed to *args.

        Returns:
            None

        >>> add_diffed_terms('t') # adds first difference of t as regressor
        """
        self._validate_future_dates_exist()
        descriptive_assert(
            diff in (1, 2), ValueError, f"diff must be 1 or 2, got {diff}"
        )
        for a in args:
            descriptive_assert(
                not a.startswith("AR"),
                ForecastError,
                "AR terms can only be differenced by using diff() method",
            )
            for i in range(1, diff + 1):
                cx = self.current_xreg[a].diff()
                fx = (
                    pd.Series(self.current_xreg[a].to_list() + self.future_xreg[a])
                    .diff()
                    .to_list()
                )

            self.current_xreg[f"{a}diff{sep}{diff}"] = cx
            self.future_xreg[f"{a}diff{sep}{diff}"] = fx[-len(self.future_dates) :]

        obs_to_keep = len(self.y) - diff
        self.keep_smaller_history(obs_to_keep)

        if drop:
            self.drop_Xvars(*args)

    def add_lagged_terms(self, *args, lags=1, upto=True, sep="_"):
        """ lags all passed variables (no AR terms) 1 or more times.

        Args:
            *args (str): names of Xvars that aleady exist in the object.
            lags (int): greater than 0, default 1.
                the number of times to lag each passed variable.
            upto (bool): default True.
                whether to add all lags up to the number passed to lags.
                if you pass 6 to lags and upto is True, lags 1, 2, 3, 4, 5, 6 will all be added.
                if you pass 6 to lags and upto is False, lag 6 only will be added.
            sep (str): default '_'.
                the separator between each term in arg to create the final variable name.
                resulting variable names will be like "tlag_1" or "tlag_2" by default.

        Returns:
            None

        >>> add_lagged_terms('t',lags=3) # adds first, second, and third lag of t
        >>> add_lagged_terms('t',lags=6,upto=False) # adds 6th lag of t only
        """
        self._validate_future_dates_exist()
        descriptive_assert(
            isinstance(lags, int),
            ValueError,
            f"lags must be an int type greater than 0, got {lags}",
        )
        descriptive_assert(
            lags >= 1,
            ValueError,
            f"lags must be an int type greater than 0, got {lags}",
        )
        for a in args:
            descriptive_assert(
                not a.startswith("AR"),
                ForecastError,
                "adding lagged AR terms makes no sense, add more AR terms instead",
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

        obs_to_keep = len(self.y) - lags
        self.keep_smaller_history(obs_to_keep)

    def undiff(self, suppress_error=False):
        """ undifferences y to original level and drops all regressors (such as AR terms).

        Args:
            suppress_error (bool): default False.
                whether to suppress an error that gets raised if the series was never differenced.

        Returns:
            None

        >>> f.undiff()
        """
        self.typ_set()
        if self.integration == 0:
            if suppress_error:
                return
            else:
                raise ForecastError.CannotUndiff(
                    "cannot undiff a series that was never differenced"
                )

        self.current_xreg = {}
        self.future_xreg = {}

        self.current_dates = pd.Series(self.init_dates)
        self.y = pd.Series(self.levely)

        self.integration = 0
        if hasattr(self, "adf_stationary"):
            delattr(self, "adf_stationary")

    def set_estimator(self, estimator):
        """ sets the estimator to forecast with.

        Args:
            estimator (str): one of _estimators_

        Returns:
            None

        >>> f.set_estimator('mlr')
        """
        descriptive_assert(
            estimator in _estimators_,
            ValueError,
            f"estimator must be one of {_estimators_}, got {estimator}",
        )
        self.typ_set()
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
        """ ingests a grid to tune the estimator.

        Args:
            grid (dict or str):
                if dict, must be a user-created grid.
                if str, must match the name of a dict grid stored in Grids.py.

        Returns:
            None

        >>> f.set_estimator('mlr')
        >>> f.ingest_grid({'normalizer':['scale','minmax']})
        """
        from itertools import product

        def expand_grid(d):
            return pd.DataFrame([row for row in product(*d.values())], columns=d.keys())

        if isinstance(grid, str):
            import Grids

            importlib.reload(Grids)
            grid = getattr(Grids, grid)
        grid = expand_grid(grid)
        self.grid = grid

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
        >>> f.set_estimator('mlp')
        >>> f.ingest_grid('mlp')
        >>> f.limit_grid_size(10,random_seed=20) # limits grid to 10 iterations
        >>> f.limit_grid_size(.5,random_seed=20) # limits grid to half its original size
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

    def set_validation_metric(self, metric="rmse"):
        """ sets the metric that will be used to tune all subsequent models.

        Args:
            metric: one of _metrics_, default 'rmse'.
                the metric to optimize the models with using the validation set.

        Returns:
            None

        >>> f.set_validation_metric('mae')
        """
        descriptive_assert(
            metric in _metrics_,
            ValueError,
            f"metric must be one of {_metrics_}, got {metric}",
        )
        if (metric == "r2") & (self.validation_length < 2):
            raise ValueError(
                "can only validate with r2 if the validation length is at least 2, try calling set_validation_length()"
            )
        self.validation_metric = metric

    def tune(self, dynamic_tuning=False):
        """ tunes the specified estimator using an ingested grid (ingests a grid from Grids.py with same name as the estimator by default).
        any parameters that can be passed as arguments to manual_forecast() can be tuned with this process.

        Args:
            dynamic_tuning (bool): default False.
                whether to dynamically tune the forecast (meaning AR terms will be propogated with predicted values).
                setting this to False means faster performance, but gives a less-good indication of how well the forecast will perform out x amount of periods.
                when False, metrics effectively become an average of one-step forecasts.

        Returns:
            None

        >>> f.set_estimator('xgboost')
        >>> f.tune()
        >>> f.auto_forecast()
        """
        if not hasattr(self, "grid"):
            try:
                self.ingest_grid(self.estimator)
            except SyntaxError:
                raise
            except:
                raise ForecastError.NoGrid(f'to tune, a grid must be loaded. tried to load a grid called {self.estimator}, but either the Grids.py file could not be found in the current directory, there is no grid with that name, or the dictionary values are not list-like. try ingest_grid() with a dictionary grid passed manually.')

        if self.estimator in _cannot_be_tuned_:
            raise ForecastError(f"{self.estimator} models cannot be tuned")
            self.best_params = {}
            return

        metrics = []
        iters = self.grid.shape[0]
        for i in range(iters):
            try:
                hp = {k:v[i] for k,v in self.grid.to_dict(orient='list').items()}
                if self.estimator in _sklearn_estimators_:
                    metrics.append(
                        self._forecast_sklearn(
                            fcster=self.estimator,
                            tune=True,
                            dynamic_testing=dynamic_tuning,
                            **hp,
                        )
                    )
                else:
                    metrics.append(
                        getattr(self, f"_forecast_{self.estimator}")(tune=True,**hp)
                    )
            except TypeError:
                raise
            except Exception as e:
                self.grid.drop(i, axis=0, inplace=True)
                logging.warning(
                    f"could not evaluate the paramaters: {hp}. error: {e}"
                )

        if len(metrics) > 0:
            self.grid_evaluated = self.grid.copy()
            self.grid_evaluated["validation_length"] = self.validation_length
            self.grid_evaluated["validation_metric"] = self.validation_metric
            self.grid_evaluated["metric_value"] = metrics
            if self.validation_metric == "r2":
                best_params_idx = self.grid.loc[
                    self.grid_evaluated["metric_value"]
                    == self.grid_evaluated["metric_value"].max()
                ].index.to_list()[0]
            else:
                best_params_idx = self.grid.loc[
                    self.grid_evaluated["metric_value"]
                    == self.grid_evaluated["metric_value"].min()
                ].index.to_list()[0]
            self.best_params = {k: v[best_params_idx] for k, v in self.grid.to_dict(orient='series').items()}
            self.validation_metric_value = self.grid_evaluated.loc[
                best_params_idx, "metric_value"
            ]
        else:
            logging.warning(
                f"none of the keyword/value combos stored in the grid could be evaluated for the {self.estimator} model"
            )
            self.best_params = {}

        self.dynamic_tuning = dynamic_tuning

    def manual_forecast(self, call_me=None, dynamic_testing=True, test_only=False, **kwargs):
        """ manually forecasts with the hyperparameters, Xvars, and normalizer selection passed as keywords.

        Args:
            call_me (str): optional.
                what to call the model when storing it in the object's history dictionary.
                if not specified, the model's nickname will be assigned the estimator value ('mlp' will be 'mlp', etc.).
                duplicated names will be overwritten with the most recently called model.
            dynamic_testing (bool): default True.
                whether to dynamically test the forecast (meaning AR terms will be propogated with predicted values).
                setting this to False means faster performance, but gives a less-good indication of how well the forecast will perform out x amount of periods.
                when False, test-set metrics effectively become an average of one-step forecasts.
            test_only (bool): default False:
                whether to stop your model after the testing process.
                all forecasts in history will be equivalent to test-set predictions.
                when forecasting to future periods, there is no way to skip the training process.
                any plot or export of forecasts into a future horizon will fail for this model
                and not all methods will raise descriptive errors.
            **kwargs: passed to the _forecast_{estimator}() method and can include such parameters as Xvars, normalizer, cap, and floor, in addition to any given model's specific hyperparameters
                for sklearn models, can inlcude normalizer and Xvars.
                for ARIMA, Prophet and Silverkite models, can include Xvars but not normalizer.
                LSTM and RNN models have their own sets of possible keywords.
                see https://scalecast.readthedocs.io/en/latest/Forecaster/_forecast.html.

        Returns:
            None

        >>> f.set_estimator('mlr')
        >>> f.manual_forecast(normalizer='pt')
        """
        descriptive_assert(
            isinstance(call_me, str) | (call_me is None),
            ValueError,
            "call_me must be a str type or None",
        )

        descriptive_assert(
            len(self.future_dates) > 0,
            ForecastError,
            "before calling a model, please make sure you have generated future dates by calling generate_future_dates(), set_last_future_date(), or ingest_Xvars_df(use_future_dates=True)",
        )

        if 'tune' in kwargs.keys():
            kwargs.pop('tune')
            logging.warning('tune argument will be ignored')

        self._clear_the_deck()
        self.test_only = test_only
        self.dynamic_testing = dynamic_testing
        self.call_me = self.estimator if call_me is None else call_me
        result = (
            self._forecast_sklearn(fcster=self.estimator, dynamic_testing=dynamic_testing, test_only=test_only, **kwargs)
            if self.estimator in _sklearn_estimators_
            else getattr(self, f"_forecast_{self.estimator}")(dynamic_testing=dynamic_testing, test_only=test_only, **kwargs)
        ) # 0 - forecast, 1 - fitted vals, 2 - Xvars, 3 - regr
        self.forecast = [i for i in result[0]]
        self.fitted_values = [i for i in result[1]]
        self.Xvars = result[2]
        self.regr = result[3]
        if self.estimator in ('arima','hwes'):
            self._set_summary_stats()

        self._bank_history(
            tuned=False
            if not hasattr(self, "best_params")
            else len(self.best_params.keys()) > 0,
            **kwargs,
        )

    def auto_forecast(self, call_me=None, dynamic_testing=True, test_only=False):
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
            test_only (bool): default False:
                whether to stop your model after the testing process.
                all forecasts in history will be equivalent to test-set predictions.
                when forecasting to future periods, there is no way to skip the training process.
                any plot or export of forecasts into a future horizon will fail for this model
                and not all methods will raise descriptive errors.

        Returns:
            None

        >>> f.set_estimator('xgboost')
        >>> f.tune()
        >>> f.auto_forecast()
        """
        if not hasattr(self, "best_params"):
            logging.warning(
                f"since tune() has not been called, {self.estimator} model will be run with default hyperparameters"
            )
            self.best_params = {}
        self.manual_forecast(
            call_me=call_me, dynamic_testing=dynamic_testing, test_only=test_only, **self.best_params
        )

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

        >>> from sklearn.ensemble import StackingRegressor
        >>> f.add_sklearn_estimator(StackingRegressor,called='stacking')
        >>> f.set_estimator('stacking')
        >>> f.manual_forecast(...)
        """
        globals()[called + '_'] = imported_module
        _sklearn_imports_[called] = globals()[called + '_']
        _sklearn_estimators_.append(called)
        _sklearn_estimators_.sort()
        _estimators_.append(called)
        _estimators_.sort()
        _can_be_tuned_.append(called)
        _can_be_tuned_.sort()

    def tune_test_forecast(
        self,
        models,
        dynamic_tuning=False,
        dynamic_testing=True,
        summary_stats=False,
        feature_importance=False,
    ):
        """ iterates through a list of models, tunes them using grids in Grids.py, forecasts them, and can save feature information.

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
            summary_stats (bool): default False.
                whether to save summary stats for the models that offer those.
            feature_importance (bool): default False.
                whether to save permutation feature importance information for the models that offer those.

        Returns:
            None

        >>> models = ('mlr','mlp','lightgbm')
        >>> f.tune_test_forecast(models,dynamic_testing=False,feature_importance=True)
        """
        descriptive_assert(
            len([m for m in models if m not in _estimators_]) == 0,
            ValueError,
            f"all models passed to models argument most be one of {_estimators_}",
        )
        descriptive_assert(
            os.path.isfile("./Grids.py"),
            FileNotFoundError,
            "Grids.py not found in working directory",
        )
        descriptive_assert(
            "comob" not in models, ValueError, "combo models cannot be tuned"
        )
        for m in models:
            self.set_estimator(m)
            self.tune(dynamic_tuning=dynamic_tuning)
            self.auto_forecast(dynamic_testing=dynamic_testing)

            if summary_stats:
                self.save_summary_stats()
            if feature_importance:
                self.save_feature_importance()

    def save_feature_importance(self):
        """ saves feature info for models that offer it and will not raise errors if not available.
        call after evaluating the model you want it for and before changing the estimator.
        this is more reliable as a feature-importance scorer when models were run test_only = True
        because then all feature scores will be out-of-sample.

        >>> f.set_estimator('mlr')
        >>> f.manual_forecast()
        >>> f.save_feature_importance()
        """
        import eli5
        from eli5.sklearn import PermutationImportance

        try:
            perm = PermutationImportance(self.regr).fit(
                self.X[-self.test_length:], self.y.values[-self.test_length:]
            )
        except (TypeError,AttributeError):
            logging.warning(
                f"cannot set feature importance on the {self.estimator} model"
            )
            return
        self.feature_importance = eli5.explain_weights_df(
            perm, feature_names=self.history[self.call_me]["Xvars"]
        ).set_index("feature")
        self.perm = perm
        self._bank_fi_to_history()

    def save_summary_stats(self):
        """ saves summary stats for models that offer it and will not raise errors if not available.
        call after evaluating the model you want it for and before changing the estimator.

        >>> f.set_estimator('arima')
        >>> f.manual_forecast(order=(1,1,1))
        >>> f.save_summary_stats()
        """
        if not hasattr(self, "summary_stats"):
            logging.warning(
                f"last model run ({self.estimator}) does not have summary stats"
            )
            return
        self._bank_summary_stats_to_history()

    def keep_smaller_history(self, n):
        """ cuts the amount of y observations in the object.

        Args:
            n (int, str, or datetime.datetime):
                if int, the number of observations to keep.
                otherwise, the last observation to keep.
                if str, must be '%Y-%m-%d' format.

        Returns:
            None

        >>> f.keep_smaller_history(500) # keeps last 500 observations
        >>> f.keep_smaller_history('2020-01-01') # keeps only observations on or later than 1/1/2020
        """
        if isinstance(n, str):
            n = datetime.datetime.strptime(n, "%Y-%m-%d")
        if (type(n) is datetime.datetime) or (type(n) is pd.Timestamp):
            n = len([i for i in self.current_dates if i >= n])
        descriptive_assert(
            isinstance(n, int),
            ValueError,
            "n must be an int, datetime object, or str in '%Y-%m-%d' format and there must be more than 2 observations to keep",
        )
        descriptive_assert(
            n > 2,
            ValueError,
            "n must be an int, datetime object, or str in '%Y-%m-%d' format and there must be more than 2 observations to keep",
        )
        self.y = self.y[-n:]
        self.current_dates = self.current_dates[-n:]
        for k, v in self.current_xreg.items():
            self.current_xreg[k] = v[-n:]

    def order_fcsts(self, models, determine_best_by="TestSetRMSE") -> list:
        """ gets estimated forecasts ordered from best-to-worst.
        
        Args:
            models (list-like):
                each element must match an evaluated model's nickname (which is the same as its estimator name by default).
            determine_best_by (str): default 'TestSetRMSE'. one of _determine_best_by_.

        Returns:
            (list): The ordered models.

        >>> models = ('mlr','mlp','lightgbm')
        >>> f.tune_test_forecast(models,dynamic_testing=False,feature_importance=True)
        >>> ordered_models = f.order_fcsts(models,"LevelTestSetMAPE")
        """
        descriptive_assert(
            determine_best_by in _determine_best_by_,
            ValueError,
            f"determine_best_by must be one of {_determine_best_by_}, got {determine_best_by}",
        )
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

    def get_regressor_names(self) -> list:
        """ gets the regressor names stored in the object.

        Args:
            None

        Returns:
            (list): Regressor names that have been added to the object.
        
        >>> f.add_time_trend()
        >>> f.get_regressor_names()
        """
        return [k for k in self.current_xreg.keys()]

    def get_freq(self) -> str:
        """ gets the pandas inferred date frequency
        
        Returns:
            (str): The inferred frequency of the current_dates array.

        >>> f.get_freq()
        """
        return self.freq

    def validate_regressor_names(self):
        """ validates that all regressor names exist in both current_xregs and future_xregs.
        raises an error if this is not the case.
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
                f"the following regressors are in current_xreg but not future_xreg: {case1}\nthe following regressors are in future_xreg but not current_xreg {case2}"
            )

    def plot(
        self,
        models="all",
        order_by=None,
        level=False,
        print_attr=[],
        ci=False,
    ):
        """ plots all forecasts with the actuals, or just actuals if no forecasts have been evaluated or are selected.
        if any models passed to models were run test_only=True, will raise an error.

        Args:
            models (list-like, str, or None): default 'all'.
               the forecasted models to plot.
               can start with "top_" and the metric specified in order_by will be used to order the models appropriately.
               if None or models/order_by combo invalid, will plot only actual values.
            order_by (str): one of _determine_best_by_. optional.
            level (bool): default False.
                if True, will always plot level forecasts.
                if False, will plot the forecasts at whatever level they were called on.
                if False and there are a mix of models passed with different integrations, will default to True.
            print_attr (list-like): default [].
                attributes from history to print to console.
                if the attribute doesn't exist for a passed model, will not raise error, will just skip that element.
            ci (bool): default False.
                whether to display the confidence intervals.
                change defaults by calling `set_cilevel()` and `set_bootstrapped_samples()` before forecasting.
                ignored when level = True.

        Returns:
            (Figure): the created figure

        >>> models = ('mlr','mlp','lightgbm')
        >>> f.tune_test_forecast(models,dynamic_testing=False,feature_importance=True)
        >>> f.plot(order_by='LevelTestSetMAPE') # plots all forecasts
        >>> plt.show()
        """
        try:
            models = self._parse_models(models, order_by)
        except (ValueError, TypeError):
            models = None

        _, ax = plt.subplots()

        if models is None:
            sns.lineplot(x=self.current_dates.values, y=self.y.values, label="actuals", ax=ax)
            plt.legend()
            plt.xlabel("Date")
            plt.ylabel("Values")
            return ax

        integration = set(
            [d["Integration"] for m, d in self.history.items() if m in models]
        )
        if len(integration) > 1:
            level = True

        y = self.y.copy()
        if self.integration == 0:
            for _ in range(max(integration)):
                y = y.diff()
        self._validate_no_test_only(models)
        plot = {
            "date": self.current_dates.to_list()[-len(y.dropna()) :]
            if not level
            else self.current_dates.to_list()[
                -len(self.history[models[0]]["LevelY"]) :
            ],
            "actuals": y.dropna().to_list()
            if not level
            else self.history[models[0]]["LevelY"],
        }
        plot["actuals_len"] = min(len(plot["date"]), len(plot["actuals"]))

        print_attr_map = {}
        sns.lineplot(
            x=plot["date"][-plot["actuals_len"] :],
            y=plot["actuals"][-plot["actuals_len"] :],
            label="actuals",
            ax=ax,
        )
        for i, m in enumerate(models):
            plot[m] = (
                self.history[m]["Forecast"]
                if not level
                else self.history[m]["LevelForecast"]
            )
            sns.lineplot(
                x=self.future_dates.to_list(), 
                y=plot[m], 
                color=_colors_[i], 
                label=m, 
                ax = ax
            )
            if ci and not level:
                plt.fill_between(
                    x=self.future_dates.to_list(),
                    y1=self.history[m]["UpperCI"],
                    y2=self.history[m]["LowerCI"],
                    alpha=0.2,
                    color=_colors_[i],
                    label="{} {:.0%} CI".format(m, self.history[m]["CILevel"]),
                )
            print_attr_map[m] = {
                a: self.history[m][a] for a in print_attr if a in self.history[m].keys()
            }

        for m, d in print_attr_map.items():
            for k, v in d.items():
                print(f"{m} {k}: {v}")

        plt.legend()
        plt.xlabel("Date")
        plt.ylabel("Values")
        return ax

    def plot_test_set(
        self,
        models="all",
        order_by=None,
        include_train=True,
        level=False,
        ci=False,
    ):
        """ plots all test-set predictions with the actuals.

        Args:
            models (list-like or str): default 'all'.
               the forecated models to plot.
               can start with "top_" and the metric specified in order_by will be used to order the models appropriately.
            order_by (str): one of _determine_best_by_, optional.
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
                default is 100 boostrapped samples and a 95% confidence interval.
                change defaults by calling `set_cilevel()` and `set_bootstrapped_samples()` before forecasting.
                ignored when level = False.

        Returns:
            (Figure): the created figure

        >>> models = ('mlr','mlp','lightgbm')
        >>> f.tune_test_forecast(models,dynamic_testing=False,feature_importance=True)
        >>> f.plot(order_by='LevelTestSetMAPE') # plots all test-set results
        >>> plt.show()
        """
        _, ax = plt.subplots()
        models = self._parse_models(models, order_by)
        integration = set(
            [d["Integration"] for m, d in self.history.items() if m in models]
        )
        if len(integration) > 1:
            level = True

        y = self.y.copy()
        if self.integration == 0:
            for _ in range(max(integration)):
                y = y.diff()

        plot = {
            "date": self.current_dates.to_list()[-len(y.dropna()) :]
            if not level
            else self.current_dates.to_list()[
                -len(self.history[models[0]]["LevelY"]) :
            ],
            "actuals": y.dropna().to_list()
            if not level
            else self.history[models[0]]["LevelY"],
        }
        plot["actuals_len"] = min(len(plot["date"]), len(plot["actuals"]))

        if str(include_train).isnumeric():
            descriptive_assert(
                (include_train > 1) & isinstance(include_train, int),
                ValueError,
                f"include_train must be a bool type or an int greater than 1, got {include_train}",
            )
            plot["actuals"] = plot["actuals"][-include_train:]
            plot["date"] = plot["date"][-include_train:]
        elif isinstance(include_train, bool):
            if not include_train:
                plot["actuals"] = plot["actuals"][-self.test_length :]
                plot["date"] = plot["date"][-self.test_length :]
        else:
            raise ValueError(
                f"include_train argument not recognized: ({include_train})"
            )

        sns.lineplot(
            x=plot["date"][-plot["actuals_len"] :],
            y=plot["actuals"][-plot["actuals_len"] :],
            label="actuals",
            ax = ax,
        )

        for i, m in enumerate(models):
            plot[m] = (
                self.history[m]["TestSetPredictions"]
                if not level
                else self.history[m]["LevelTestSetPreds"]
            )
            test_dates = self.current_dates.to_list()[-len(plot[m]) :]
            sns.lineplot(
                x=test_dates,
                y=plot[m],
                linestyle="--",
                color=_colors_[i],
                alpha=0.7,
                label=m,
                ax=ax
            )
            if ci and not level:
                plt.fill_between(
                    x=test_dates,
                    y1=self.history[m]["TestSetUpperCI"],
                    y2=self.history[m]["TestSetLowerCI"],
                    alpha=0.2,
                    color=_colors_[i],
                    label="{} {:.0%} CI".format(m, self.history[m]["CILevel"]),
                )

        plt.legend()
        plt.xlabel("Date")
        plt.ylabel("Values")
        return ax

    def plot_fitted(
        self,
        models="all",
        order_by=None,
    ):
        """ plots all fitted values with the actuals.

        Args:
            models (list-like,str): default 'all'.
               the forecated models to plot.
               can start with "top_" and the metric specified in order_by will be used to order the models appropriately.
            order_by (str): one of _determine_best_by_, default None.

        Returns:
            (Figure): the created figure

        >>> models = ('mlr','mlp','lightgbm')
        >>> f.tune_test_forecast(models,dynamic_testing=False,feature_importance=True)
        >>> f.plot_fitted(order_by='LevelTestSetMAPE') # plots all fitted values
        >>> plt.show()
        """
        _, ax = plt.subplots()
        models = self._parse_models(models, order_by)
        integration = set(
            [d["Integration"] for m, d in self.history.items() if m in models]
        )
        if len(integration) > 1:
            raise ForecastError.PlottingError(
                "cannot plot fitted values when forecasts run at different levels"
            )

        y = self.y.copy()
        if self.integration == 0:
            for _ in range(max(integration)):
                y = y.diff()

        plot = {
            "date": self.current_dates.to_list()[-len(y.dropna()) :],
            "actuals": y.dropna().to_list(),
        }
        sns.lineplot(x=plot["date"], y=plot["actuals"], label="actuals", ax=ax)

        for i, m in enumerate(models):
            plot[m] = self.history[m]["FittedVals"]
            sns.lineplot(
                x=plot["date"][-len(plot[m]) :],
                y=plot[m][-len(plot["date"]) :],
                linestyle="--",
                color=_colors_[i],
                alpha=0.7,
                label=m,
                ax=ax,
            )

        plt.legend()
        plt.xlabel("Date")
        plt.ylabel("Values")
        return ax

    def drop_regressors(self, *args):
        """ drops regressors.

        Args:
            *args (str): the names of regressors to drop.

        Returns:
            None

        >>> f.add_time_trend()
        >>> f.add_exp_terms('t',pwr=.5)
        >>> f.drop_regressors('t','t^0.5')
        """
        for a in args:
            self.current_xreg.pop(a)
            self.future_xreg.pop(a)

    def drop_Xvars(self, *args):
        """ drops regressors.

        Args:
            *args (str): the names of regressors to drop.

        Returns:
            None

        >>> f.add_time_trend()
        >>> f.add_exp_terms('t',pwr=.5)
        >>> f.drop_Xvars('t','t^0.5')
        """
        self.drop_regressors(*args)

    def pop(self, *args):
        """ deletes evaluated forecasts from the object's memory.

        Args:
            *args (str): names of models matching what was passed to call_me.
            default for call_me in a given model is the same as the estimator name.


        >>> models = ('mlr','mlp','lightgbm')
        >>> f.tune_test_forecast(models,dynamic_testing=False,feature_importance=True)
        >>> f.pop('mlr')
        """
        for a in args:
            self.history.pop(a)

    def pop_using_criterion(self, metric, evaluated_as, threshold, delete_all=True):
        """ deletes all forecasts from history that meet a given criterion.

        Args:
            metric (str): one of _determine_best_by_ + ['AnyPrediction','AnyLevelPrediction'].
            evaluated_as (str): one of {"<","<=",">",">=","=="}.
            threshold (float): the threshold to compare the metric and operator to.
            delete_all (bool): default True.
                if the passed criterion deletes all forecasts, whether to actually delete all forecasts.
                if False and all forecasts meet criterion, will keep them all.

        Returns:
            None

        >>> models = ('mlr','mlp','lightgbm')
        >>> f.tune_test_forecast(models,dynamic_testing=False,feature_importance=True)
        >>> f.pop_using_criterion('LevelTestSetMAPE','>',2)
        >>> f.pop_using_criterion('AnyPrediction','<',0,delete_all=False)
        """
        descriptive_assert(
            evaluated_as in ("<", "<=", ">", ">=","=="),
            ValueError,
            f'evaluated_as must be one of ("<","<=",">",">=","=="), got {evaluated_as}',
        )
        threshold = float(threshold)
        if metric in _determine_best_by_:
            fcsts = (
                [m for m, v in self.history.items() if v[metric] > threshold]
                if "evaluated_as" == ">"
                else [m for m, v in self.history.items() if v[metric] >= threshold]
                if "evaluated_as" == ">="
                else [m for m, v in self.history.items() if v[metric] < threshold]
                if "evaluated_as" == "<"
                else [m for m, v in self.history.items() if v[metric] <= threshold]
                if "evaluated_as" == "<="
                else [m for m, v in self.history.items() if v[metric] == threshold]
            )
        elif metric not in ("AnyPrediction", "AnyLevelPrediction"):
            raise ValueError(
                "metric must be one of {}, got {}".format(
                    _determine_best_by_ + ["AnyPrediction", "AnyLevelPrediction"],
                    metric,
                )
            )
        else:
            metric = "Forecast" if metric == "AnyPrediction" else "LevelForecast"
            fcsts = (
                [m for m, v in self.history.items() if max(v[metric]) > threshold]
                if "evaluated_as" == ">"
                else [m for m, v in self.history.items() if max(v[metric]) >= threshold]
                if "evaluated_as" == ">="
                else [m for m, v in self.history.items() if min(v[metric]) < threshold]
                if "evaluated_as" == "<"
                else [m for m, v in self.history.items() if min(v[metric]) <= threshold]
                if "evaluated_as" == "<="
                else [m for m, v in self.history.items() if threshold in v[metric]]
            )

        if (len(fcsts) < len(self.history.keys())) | delete_all:
            self.pop(*fcsts)

    def export(
        self,
        dfs=[
            "all_fcsts",
            "model_summaries",
            "best_fcst",
            "test_set_predictions",
            "lvl_test_set_predictions",
            "lvl_fcsts",
        ],
        models="all",
        best_model="auto",
        determine_best_by="TestSetRMSE",
        to_excel=False,
        out_path="./",
        excel_name="results.xlsx",
    ) -> Union[Dict[str, pd.DataFrame], pd.DataFrame]:
        """ exports 1-all of 5 pandas dataframes, can write to excel with each dataframe on a separate sheet.
        will return either a dictionary with dataframes as values (df str arguments as keys) or a single dataframe if only one df is specified.

        Args:
            dfs (list-like or str): default ['all_fcsts','model_summaries','best_fcst','test_set_predictions','lvl_fcsts'].
                a list or name of the specific dataframe(s) you want returned and/or written to excel.
                must be one of or multiple of default.
            models (list-like or str): default 'all'.
                the models to write information for.
                can start with "top_" and the metric specified in `determine_best_by` will be used to order the models appropriately.
            best_model (str): default 'auto'.
                the name of the best model, if "auto", will determine this by the metric in determine_best_by.
                if not "auto", must match a model nickname of an already-evaluated model.
            determine_best_by (str): one of _determine_best_by_, default 'TestSetRMSE'.
            to_excel (bool): default False.
                whether to save to excel.
            out_path (str): default './'.
                the path to save the excel file to (ignored when `to_excel=False`).
            excel_name (str): default 'results.xlsx'.
                the name to call the excel file (ignored when `to_excel=False`).

        Returns:
            (DataFrame or Dict[str,DataFrame]): either a single pandas dataframe if one element passed to dfs 
            or a dictionary where the keys match what was passed to dfs and the values are dataframes. 

        >>> f.export(dfs=['model_summaries','lvl_fcsts'],to_excel=True)
        """
        if isinstance(dfs, str):
            dfs = [dfs]
        else:
            dfs = list(dfs)
        if len(dfs) == 0:
            raise ValueError("no dfs passed to dfs")
        determine_best_by = determine_best_by if best_model == "auto" else None
        models = self._parse_models(models, determine_best_by)
        _dfs_ = [
            "all_fcsts",
            "model_summaries",
            "best_fcst",
            "test_set_predictions",
            "lvl_test_set_predictions",
            "lvl_fcsts",
        ]
        _bad_dfs_ = [i for i in dfs if i not in _dfs_]

        if len(_bad_dfs_) > 0:
            raise ValueError(
                f"values passed to the dfs list must be in {_dfs_}, not {_bad_dfs_}"
            )

        best_fcst_name = (
            self.order_fcsts(models, determine_best_by)[0]
            if best_model == "auto"
            else best_model
        )
        output = {}
        if "all_fcsts" in dfs:
            all_fcsts = pd.DataFrame({"DATE": self.future_dates.to_list()})
            for m in self.history.keys():
                all_fcsts[m] = self.history[m]["Forecast"]
            output["all_fcsts"] = all_fcsts
        if "model_summaries" in dfs:
            cols = [
                "ModelNickname",
                "Estimator",
                "Xvars",
                "HyperParams",
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
                "models",
                "weights",
                "LevelTestSetRMSE",
                "LevelTestSetMAPE",
                "LevelTestSetMAE",
                "LevelTestSetR2",
                "best_model",
            ]

            model_summaries = pd.DataFrame()
            for m in models:
                model_summary_m = pd.DataFrame({"ModelNickname": [m]})
                for c in cols:
                    if c not in (
                        "ModelNickname",
                        "LastTestSetPrediction",
                        "LastTestSetActual",
                        "best_model",
                    ):
                        model_summary_m[c] = [
                            self.history[m][c] if c in self.history[m].keys() else None
                        ]
                    elif c == "LastTestSetPrediction":
                        model_summary_m[c] = [self.history[m]["TestSetPredictions"][-1]]
                    elif c == "LastTestSetActual":
                        model_summary_m[c] = [self.history[m]["TestSetActuals"][-1]]
                    elif c == "best_model":
                        model_summary_m[c] = m == best_fcst_name
                model_summaries = pd.concat(
                    [model_summaries, model_summary_m], ignore_index=True
                )
            output["model_summaries"] = model_summaries
        if "best_fcst" in dfs:
            self._validate_no_test_only(models)
            best_fcst = pd.DataFrame(
                {"DATE": self.current_dates.to_list() + self.future_dates.to_list()}
            )
            best_fcst["VALUES"] = (
                self.y.to_list() + self.history[best_fcst_name]["Forecast"]
            )
            best_fcst["MODEL"] = ["actual"] * len(self.current_dates) + [
                best_fcst_name
            ] * len(self.future_dates)
            output["best_fcst"] = best_fcst
        if "test_set_predictions" in dfs:
            test_set_predictions = pd.DataFrame(
                {"DATE": self.current_dates[-self.test_length :]}
            )
            test_set_predictions["actual"] = self.y.to_list()[-self.test_length :]
            for m in models:
                test_set_predictions[m] = self.history[m]["TestSetPredictions"]
            output["test_set_predictions"] = test_set_predictions
        if "lvl_test_set_predictions" in dfs:
            test_set_predictions = pd.DataFrame(
                {"DATE": self.current_dates[-self.test_length :]}
            )
            test_set_predictions["actual"] = self.levely[-self.test_length :]
            for m in models:
                test_set_predictions[m] = self.history[m]["LevelTestSetPreds"]
            output["lvl_test_set_predictions"] = test_set_predictions
        if "lvl_fcsts" in dfs:
            self._validate_no_test_only(models)
            lvl_fcsts = pd.DataFrame({"DATE": self.future_dates.to_list()})
            for m in models:
                if "LevelForecast" in self.history[m].keys():
                    lvl_fcsts[m] = self.history[m]["LevelForecast"]
            if lvl_fcsts.shape[1] > 1:
                output["lvl_fcsts"] = lvl_fcsts

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
        """ exports the summary stats from a model.
        raises an error if you never saved the model's summary stats.

        Args:
            model (str):
                the name of them model to export for.
                matches what was passed to call_me when calling the forecast (default is estimator name)

        Returns:
            (DataFrame): The resulting summary stats of the evaluated model passed to model arg.

        >>> ss = f.export_summary_stats('arima')
        """
        return self.history[model]["summary_stats"]

    def export_feature_importance(self, model) -> pd.DataFrame:
        """ exports the feature importance from a model.
        raises an error if you never saved the model's feature importance.

        Args:
            model (str):
                the name of them model to export for.
                matches what was passed to call_me when calling the forecast (default is estimator name)

        Returns:
            (DataFrame): The resulting feature importances of the evaluated model passed to model arg.

        >>> fi = f.export_feature_importance('mlr')
        """
        return self.history[model]["feature_importance"]

    def export_validation_grid(self, model) -> pd.DataFrame:
        """ exports the validation grid from a model.
        raises an error if the model was not tuned.

        Args:
            model (str):
                the name of them model to export for.
                matches what was passed to call_me when calling the forecast.

        Returns:
            (DataFrame): The resulting validation grid of the evaluated model passed to model arg.
        """
        return self.history[model]["grid_evaluated"]

    def all_feature_info_to_excel(
        self, out_path="./", excel_name="feature_info.xlsx"
    ):
        """ saves all feature importance and summary stats to excel.
        each model where such info is available for gets its own tab.
        be sure to have called save_summary_stats() and/or save_feature_importance() before using this function.

        Args:
            out_path (str): default './'
                the path to export to
            excel_name (str): default 'feature_info.xlsx'
                the name of the resulting excel file

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
                "no saved feature importance or summary stats could be found"
            )

    def all_validation_grids_to_excel(
        self,
        out_path="./",
        excel_name="validation_grids.xlsx",
        sort_by_metric_value=False,
        ascending=True,
    ):
        """ saves all validation grids to excel.
        each model where such info is available for gets its own tab.
        be sure to have tuned at least model before calling this.

        Args:
            out_path (str): default './'.
                the path to export to.
            excel_name (str): default 'feature_info.xlsx'.
                the name of the resulting excel file.
            sort_by_metric_value (bool): default False.
                whether to sort the output by performance on validation set
            ascending (bool): default True.
                whether to sort least-to-greatest.
                ignored if sort_by_metric_value is False.

        Returns:
            None
        """
        try:
            with pd.ExcelWriter(
                os.path.join(out_path, excel_name), engine="openpyxl"
            ) as writer:
                for m in self.history.keys():
                    if "grid_evaluated" in self.history[m].keys():
                        df = (
                            self.history[m]["grid_evaluated"].copy()
                            if not sort_by_metric_value
                            else self.history[m]["grid_evaluated"].sort_values(
                                ["metric_value"], ascending=ascending
                            )
                        )
                        df.to_excel(writer, sheet_name=m, index=False)
        except IndexError:
            raise ForecastError("no validation grids could be found")

    def reset(self):
        """ drops all regressors and reverts object to original (level) state when initiated.
        """
        self.undiff(suppress_error=True)
        self.current_xreg = {}
        self.future_xreg = {}

    def export_Xvars_df(self, dropna=False):
        """ gets all utilized regressors and values.
            
        Args:
            dropna (bool): default False.
                whether to drop null values from the resulting dataframe

        Returns:
            (DataFrame): A dataframe of Xvars and names/values stored in the object.
        """
        self.typ_set() if not hasattr(self, "estimator") else None
        fut_df = pd.DataFrame()
        for k, v in self.future_xreg.items():
            if len(v) < len(self.future_dates):
                fut_df[k] = v + [None] * (len(self.future_dates) - len(v))
            else:
                fut_df[k] = v[:]

        df = pd.concat(
            [
                pd.concat(
                    [
                        pd.DataFrame({"DATE": self.current_dates.to_list()}),
                        pd.DataFrame(self.current_xreg),
                    ],
                    axis=1,
                ),
                pd.concat(
                    [pd.DataFrame({"DATE": self.future_dates.to_list()}), fut_df],
                    axis=1,
                ),
            ],
            ignore_index=True,
        )

        return df.dropna() if dropna else df

    def export_forecasts_with_cis(self, model):
        """ exports a single dataframe with forecasts and upper and lower forecast bounds.

        Args:
            model (str):
                the model nickname (must exist in history.keys()).

        Returns:
            (DataFrame): A dataframe with forecasts to future dates and corresponding confidence intervals.
        """
        self._validate_no_test_only([model])
        return pd.DataFrame(
            {
                "DATE": self.future_dates.to_list(),
                "UpperForecast": self.history[model]["UpperCI"],
                "Forecast": self.history[model]["Forecast"],
                "LowerForecast": self.history[model]["LowerCI"],
                "ModelNickname": [model] * len(self.future_dates),
                "CILevel": [self.history[model]["CILevel"]] * len(self.future_dates),
            }
        )

    def export_test_set_preds_with_cis(self, model):
        """ exports a single dataframe with test-set predictions, actuals, and upper and lower prediction bounds.

        Args:
            model (str):
                the model nickname (must exist in history.keys()).

        Returns:
            (DataFrame): A dataframe with test-set predictions and actuals with corresponding confidence intervals.
        """
        return pd.DataFrame(
            {
                "DATE": self.current_dates.to_list()[-len(self.history[model]["TestSetPredictions"]):],
                "UpperPreds": self.history[model]["TestSetUpperCI"],
                "Preds": self.history[model]["TestSetPredictions"],
                "Actuals": self.history[model]["TestSetActuals"],
                "LowerPreds": self.history[model]["TestSetLowerCI"],
                "ModelNickname": [model] * len(self.history[model]["TestSetPredictions"]),
                "CILevel": [self.history[model]["CILevel"]] * len(self.history[model]["TestSetPredictions"]),
            }
        )

    def export_fitted_vals(self,model):
        """ exports a single dataframe with dates, fitted values, actuals, and residuals.

        Args:
            model (str):
                the model nickname (must exist in history.keys()).

        Returns:
            (DataFrame): A dataframe with dates, fitted values, actuals, and residuals.
        """
        df = pd.DataFrame({
                "DATE": self.current_dates.to_list()[-len(self.history[model]["FittedVals"]):],
                "Actuals": self.y.to_list()[-len(self.history[model]["FittedVals"]):],
                "FittedVals": self.history[model]["FittedVals"]})

        df['Residuals'] = df['Actuals'] - df['FittedVals']
        return df

    def backtest(self,model,fcst_length='auto',n_iter=10):
        """ runs a backtest of a selected evaluated model over a certain 
        amount of iterations to test the average error if that model were 
        implemented over the last so-many actual forecast intervals.
        all scoring is dynamic to give a true out-of-sample result.
        all metrics are specific to level data.
        two results are extracted: a dataframe of actuals and predictions across each iteration and
        a dataframe of test-set metrics across each iteration with a mean total as the last column.
        these results are stored in the Forecaster object's history and can be extracted by calling 
        `export_backtest_metrics()` and `export_backtest_values()`.
        combo models cannot be backtest and will raise an error if you attempt to do so.

        Args:
            model (str): the model to run the backtest for. use the model nickname.
            fcst_length (int or str): default 'auto'. 
                if 'auto', uses the same forecast length as saved in the object currently.
                if int, uses that as the forecast length.
            n_iter (int): default 10. the number of iterations to backtest.
                models will iteratively train on all data before the fcst_length worth of values.
                each iteration takes one observation off the end to redo the cast until all of n_iter is exhausted.

        Returns:
            None
        """
        if fcst_length == 'auto':
            fcst_length = len(self.future_dates)
        fcst_length = int(fcst_length)
        metric_results = pd.DataFrame(columns=[f'iter{i}' for i in range(1,n_iter+1)],index=['RMSE','MAE','R2','MAPE'])
        value_results = pd.DataFrame()
        for i in range(n_iter):
            f = self.__deepcopy__()
            if i > 0:
                f.current_xreg = {k:pd.Series(v.values[:-i]) for k, v in f.current_xreg.items()}
                f.current_dates = pd.Series(f.current_dates.values[:-i])
                f.y = pd.Series(f.y.values[:-i])
                f.levely = f.levely[:-i]
            
            f.set_test_length(fcst_length)
            f.set_estimator(f.history[model]['Estimator'])
            descriptive_assert(f.estimator!='combo',ValueError,'combo models cannot be backtest')
            params = f.history[model]['HyperParams'].copy()
            if f.history[model]['Xvars'] is not None:
                params['Xvars'] = f.history[model]['Xvars'][:]
            if f.estimator in _sklearn_estimators_:
                params['normalizer'] = f.history[model]['Scaler']
            f.history = {}
            f.manual_forecast(**params,test_only=True)
            test_mets = f.export('model_summaries')
            test_preds = f.export('lvl_test_set_predictions')
            metric_results.loc['RMSE',f'iter{i+1}'] = test_mets['LevelTestSetRMSE'].values[0]
            metric_results.loc['MAE',f'iter{i+1}'] = test_mets['LevelTestSetMAE'].values[0]
            metric_results.loc['R2',f'iter{i+1}'] = test_mets['LevelTestSetR2'].values[0]
            metric_results.loc['MAPE',f'iter{i+1}'] = test_mets['LevelTestSetMAPE'].values[0]
            value_results[f'iter{i+1}actuals'] = test_preds.iloc[:,1].values.copy()
            value_results[f'iter{i+1}preds'] = test_preds.iloc[:,2].values.copy()

        metric_results['mean'] = metric_results.mean(axis=1)
        self.history[model]['BacktestMetrics'] = metric_results
        self.history[model]['BacktestValues'] = value_results

    def export_backtest_metrics(self,model):
        """ extracts the backtest metrics for a given model.
        only works if `backtest()` has been called.

        Args:
            model (str): the model nickname to extract metrics for.

        Returns:
            (DataFrame): a copy of the backtest metrics.
        """
        return self.history[model]['BacktestMetrics'].copy()

    def export_backtest_values(self,model):
        """ extracts the backtest values for a given model.
        only works if `backtest()` has been called.

        Args:
            model (str): the model nickname to extract values for.

        Returns:
            (DataFrame): a copy of the backtest values.
        """
        return self.history[model]['BacktestValues'].copy()
