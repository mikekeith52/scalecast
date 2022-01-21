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

# LOGGING
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
from sklearn.svm import SVR as svr_
from sklearn.neighbors import KNeighborsRegressor as knn_

# FUNCTIONS

# custom metrics - are what you expect except MAPE, which can return None
def mape(y, pred):
    return (
        None if 0 in y else mean_absolute_percentage_error(y, pred)
    )  # average o(1) worst-case o(n)


def rmse(y, pred):
    return mean_squared_error(y, pred) ** 0.5


def mae(y, pred):
    return mean_absolute_error(y, pred)


def r2(y, pred):
    return r2_score(y, pred)


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
    "lightgbm": lightgbm_,
    "rf": rf_,
    "elasticnet": elasticnet_,
    "svr": svr_,
    "knn": knn_,
}
_sklearn_estimators_ = sorted(_sklearn_imports_.keys())

# to add non-sklearn models, add to the list below
_non_sklearn_estimators_ = ["arima", "hwes", "lstm", "prophet", "silverkite", "rnn", "combo"]
_estimators_ = sorted(_sklearn_estimators_ + _non_sklearn_estimators_)
_cannot_be_tuned_ = ['combo','lstm','rnn']
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
_normalizer_ = ["minmax", "normalize", "scale", "pt", None]
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
_adder_funcs_ = [
    'add_ar_terms',
    'add_AR_terms',
    'add_seasonal_regressors',
    'add_time_trend',
    'add_other_regressor',
    'add_covid19_regressor',
    'add_combo_regressors',
    'add_poly_terms',
    'add_exp_terms',
    'add_logged_terms',
    'dd_pt_terms',
    'add_diffed_terms',
    'add_lagged_terms',
]
_exporter_funcs_ = [
    'export',
    'export_summary_stats',
    'export_feature_importance',
    'export_validation_grid',
    'all_feature_info_to_excel',
    'all_validation_grids_to_excel',
    'export_Xvars_df',
    'export_forecasts_with_cis',
    'export_test_set_preds_with_cis',
    'export_fitted_vals',
]
_setter_funcs_ = [
    'set_last_future_date',
    'set_test_length',
    'set_validation_length',
    'set_cilevel',
    'set_bootstrap_samples',
    'set_estimator',
    'set_validation_metric',
]
_plotter_funcs_ = [
    'plot_acf',
    'plot_pacf',
    'plot_periodogram',
    'seasonal_decompose',
    'plot',
    'plot_test_set',
    'plot_fitted',
]
_getter_funcs_ = [
    'get_regressor_names',
    'get_freq',
]

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
        self.future_dates = pd.Series([])
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

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return """Forecaster(
    DateStartActuals={}
    DateEndActuals={}
    Freq={}
    ForecastLength={}
    Xvars={}
    Differenced={}
    TestLength={}
    ValidationLength={}
    ValidationMetric={}
    ForecastsEvaluated={}
    CILevel={}
    BootstrapSamples={}
)""".format(self.current_dates.values[0].astype(str),
            self.current_dates.values[-1].astype(str),
            self.freq,
            len(self.future_dates),
            list(self.current_xreg.keys()),
            self.integration,
            self.test_length,
            self.validation_length,
            self.validation_metric,
            list(self.history.keys()),
            self.cilevel,
            self.bootstrap_samples)

    def get_funcs(self,which):
        """ returns a group of functions based on what's passed to which
            which: str, one of {'adder','exporter','setter','plotter','getter'}
        """
        return globals()[f'_{which}_funcs_']

    def _adder(self) -> None:
        """ makes sure future periods have been specified before adding regressors
        """
        descriptive_assert(
            len(self.future_dates) > 0,
            ForecastError,
            "before adding regressors, please make sure you have generated future dates by calling generate_future_dates(), set_last_future_date(), or ingest_Xvars_df(use_future_dates=True)",
        )

    def _find_cis(self):
        """ bootsrapts the upper and lower forecast estimates using the info stored in cilevel and bootstrap_samples
        """
        random.seed(20)
        resids = [
            fv - ac
            for fv, ac in zip(self.fitted_values[:], self.y[-len(self.fitted_values):])
        ]
        bootstrapped_resids = np.random.choice(resids, size=self.bootstrap_samples)
        bootstrap_mean = np.mean(bootstrapped_resids)
        bootstrap_std = np.std(bootstrapped_resids)
        return (
            stats.norm.ppf(1 - (1 - self.cilevel) / 2) * bootstrap_std + bootstrap_mean
        )

    def _bank_history(self, **kwargs) -> None:
        """ places all relevant information from the last evaluated forecast into the history dictionary attribute
            **kwargs are passed from each model, depending on how that model uses Xvars and normalizer args
        """
        call_me = self.call_me
        ci_range = self._find_cis()
        self.history[call_me] = {
            "Estimator": self.estimator,
            "Xvars": self.Xvars,
            "HyperParams": {k: v for k, v in kwargs.items() if k not in ("Xvars", "normalizer", "auto", "plot_loss")},
            "Scaler": kwargs["normalizer"] if "normalizer" in kwargs.keys() else 'minmax' if self.estimator in ("lstm","rnn") else None if self.estimator in ("prophet", "combo") else None if hasattr(self, "univariate") else "minmax",
            "Observations": len(self.y),
            "Forecast": self.forecast[:],
            "UpperCI": [f + ci_range for f in self.forecast],
            "LowerCI": [f - ci_range for f in self.forecast],
            "FittedVals": self.fitted_values[:],
            "Tuned": False if not kwargs["auto"] else "Dynamically" if self.dynamic_tuning else True,
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
            "InSampleRMSE": rmse(self.y.values[-len(self.fitted_values) :], self.fitted_values),
            "InSampleMAPE": mape(self.y.values[-len(self.fitted_values) :], self.fitted_values),
            "InSampleMAE": mae(self.y.values[-len(self.fitted_values) :], self.fitted_values),
            "InSampleR2": r2(self.y.values[-len(self.fitted_values) :], self.fitted_values),
            "CILevel":self.cilevel,
            "CIPlusMinus":ci_range,
        }

        if kwargs["auto"]:
            self.history[call_me]["ValidationSetLength"] = self.validation_length
            self.history[call_me]["ValidationMetric"] = self.validation_metric
            self.history[call_me]["ValidationMetricValue"] = self.validation_metric_value

        for attr in ("univariate", "grid_evaluated", "models", "weights"):
            if hasattr(self, attr):
                self.history[call_me][attr] = getattr(self, attr)

        self.history[call_me]["LevelY"] = self.levely[:]
        if self.integration > 0:
            integration = self.integration

            fcst = self.forecast[::-1]
            # fcstuci = [f + ci_range for f in self.forecast][::-1]
            # fcstlci = [f - ci_range for f in self.forecast][::-1]
            pred = self.history[call_me]["TestSetPredictions"][::-1]
            # preduci = [f + ci_range for f in self.test_set_pred][::-1]
            # predlci = [f - ci_range for f in self.test_set_pred][::-1]

            if integration == 2:
                fcst.append(self.y.values[-2] + self.y.values[-1])
                # fcstuci.append(self.history[call_me]['UpperCI'][-2] + self.history[call_me]['UpperCI'][-1])
                # fcstlci.append(self.history[call_me]['LowerCI'][-2] + self.history[call_me]['LowerCI'][-1])
                pred.append(
                    self.y.values[-(len(pred) + 2)] + self.y.values[-(len(pred) + 1)]
                )
                # preduci.append(self.history[call_me]['TestSetUpperCI'][-2] + self.history[call_me]['TestSetUpperCI'][-1])
                # predlci.append(self.history[call_me]['TestSetLowerCI'][-2] + self.history[call_me]['TestSetLowerCI'][-1])
            else:
                fcst.append(self.levely[-1])
                # fcstuci.append(self.history[call_me]['UpperCI'][-1])
                # fcstlci.append(self.history[call_me]['LowerCI'][-1])
                pred.append(self.levely[-(len(pred) + 1)])
                # preduci.append(self.history[call_me]['TestSetUpperCI'][-1])
                # predlci.append(self.history[call_me]['TestSetLowerCI'][-1])

            fcst = list(np.cumsum(fcst[::-1]))[1:]
            # fcstuci = list(np.cumsum(fcstuci[::-1]))[1:]
            # fcstlci = list(np.cumsum(fcstlci[::-1]))[1:]
            pred = list(np.cumsum(pred[::-1]))[1:]
            # preduci = list(np.cumsum(preduci[::-1]))[1:]
            # predlci = list(np.cumsum(predlci[::-1]))[1:]

            if integration == 2:
                fcst.reverse()
                fcst.append(self.levely[-1])
                fcst = list(np.cumsum(fcst[::-1]))[1:]
                # fcstuci.reverse()
                # fcstuci.append(self.history[call_me]['UpperCI'][-1])
                # fcstuci = list(np.cumsum(fcstuci[::-1]))[1:]
                # fcstlci.reverse()
                # fcstlci.append(self.history[call_me]['LowerCI'][-1])
                # fcstlci = list(np.cumsum(fcstlci[::-1]))[1:]

                pred.reverse()
                pred.append(self.levely[-(len(pred) + 1)])
                pred = list(np.cumsum(pred[::-1]))[1:]
                # preduci.reverse()
                # preduci.append(self.history[call_me]['TestSetUpperCI'][-1])
                # preduci = list(np.cumsum(preduci[::-1]))[1:]
                # predlci.reverse()
                # predlci.append(self.history[call_me]['TestSetLowerCI'][-1])
                # predlci = list(np.cumsum(predlci[::-1]))[1:]

            self.history[call_me]["LevelForecast"] = fcst[:]
            # self.history[call_me]['LevelForecastUpperCI'] = fcstuci[:]
            # self.history[call_me]['LevelForecastLowerCI'] = fcstlci[:]
            self.history[call_me]["LevelTestSetPreds"] = pred[:]
            # self.history[call_me]['LevelTestSetPredsUpperCI'] = preduci[:]
            # self.history[call_me]['LevelTestSetPredsLowerCI'] = predlci[:]
            self.history[call_me]["LevelTestSetRMSE"] = rmse(self.levely[-len(pred) :], pred)
            self.history[call_me]["LevelTestSetMAPE"] = mape(self.levely[-len(pred) :], pred)
            self.history[call_me]["LevelTestSetMAE"] = mae(self.levely[-len(pred) :], pred)
            self.history[call_me]["LevelTestSetR2"] = r2(self.levely[-len(pred) :], pred)
        else:  # better to have these attributes populated for all series
            self.history[call_me]["LevelForecast"] = self.forecast[:]
            # self.history[call_me]['LevelForecastUpperCI'] = [f + ci_range for f in self.forecast]
            # self.history[call_me]['LevelForecastLowerCI'] = [f - ci_range for f in self.forecast]
            self.history[call_me]["LevelTestSetPreds"] = self.test_set_pred[:]
            # self.history[call_me]['LevelTestSetPredsUpperCI'] = [f + ci_range for f in self.test_set_pred]
            # self.history[call_me]['LevelTestSetPredsLowerCI'] = [f - ci_range for f in self.test_set_pred]
            self.history[call_me]["LevelTestSetRMSE"] = self.rmse
            self.history[call_me]["LevelTestSetMAPE"] = self.mape
            self.history[call_me]["LevelTestSetMAE"] = self.mae
            self.history[call_me]["LevelTestSetR2"] = self.r2

    def _set_summary_stats(self) -> None:
        """ for every model where summary stats are available, saves them to a pandas dataframe where index is the regressor name
        """
        results_summary = self.regr.summary()
        results_as_html = results_summary.tables[1].as_html()
        self.summary_stats = pd.read_html(results_as_html, header=0, index_col=0)[0]

    def _bank_fi_to_history(self) -> None:
        """ for every model where ELI5 permutation feature importance can be extracted, saves that info to a pandas dataframe wehre index is the regressor name
        """
        call_me = self.call_me
        self.history[call_me]["feature_importance"] = self.feature_importance

    def _bank_summary_stats_to_history(self) -> None:
        """ saves summary stats (where available) to history
        """
        call_me = self.call_me
        self.history[call_me]["summary_stats"] = self.summary_stats

    def _parse_normalizer(
        self, X_train, normalizer
    ) -> Union[
        sklearn.preprocessing.MinMaxScaler,
        sklearn.preprocessing.Normalizer,
        sklearn.preprocessing.StandardScaler,
        None,
    ]:
        """ fits an appropriate scaler to training data that will then be applied to test and future data
            X_train: pandas dataframe
            normalizer: one of {_normalizer_}
                if 'minmax', uses the MinMaxScaler from sklearn.preprocessing
                if 'scale', uses the StandardScaler from sklearn.preprocessing
                if 'normalize', uses the Normalizer from sklearn.preprocessing
                if 'pt', uses the PowerTransformer from sklearn.preprocessing
                if None, returns None
        """
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

    def _train_test_split(
        self, X, y, test_size
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """ splits data chronologically into training and testing set--the last observations in order will be used in test set
            X: numpy array or pandas dataframe
                regressor values
            y: numpy array or pandas series
                dependent-variable values
            test_size: int (no fractional splits)
                size of resulting test set
        """
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=False
        )
        return X_train, X_test, y_train, y_test

    def _metrics(self, y, pred) -> None:
        """ creates the following attributes: test_set_actuals, test_set_pred, rmse, r2, mae, mape
            y: list-like
                the actual observations
            pred: list-like
                the predictions of y
        """
        self.test_set_actuals = list(y)
        self.test_set_pred = list(pred)
        self.rmse = rmse(y, pred)
        self.r2 = r2(y, pred)
        self.mae = mae(y, pred)
        self.mape = mape(y, pred)

    def _tune(self) -> float:
        """ reads which validation metric to use in _metrics_ and pulls that attribute value to return from function
            deletes: 'r2','rmse','mape','mae','test_set_pred', and 'test_set_actuals' attributes if they exist
        """
        metric = getattr(self, getattr(self, "validation_metric"))
        for attr in ("r2", "rmse", "mape", "mae", "test_set_pred", "test_set_actuals"):
            delattr(self, attr)
        return metric

    def _scale(self, scaler, X) -> np.ndarray:
        """ uses scaler parsed from _parse_normalizer() function to transform matrix passed to X
            scaler: sklearn.preprocessing.MinMaxScaler | sklearn.preprocessing.Normalizer | sklearn.preprocessing.StandardScaler | None
            X: numpy array or pandas dataframe
                the matrix to transform
        """
        if not scaler is None:
            return scaler.transform(X)
        else:
            return X.values if hasattr(X, "values") else X

    def _clear_the_deck(self) -> None:
        """ deletes the following attributes to prepare a new forecast:
            'univariate','fitted_values','regr','X','feature_importance','summary_stats','models','weights'
        """
        for attr in (
            "univariate",
            "fitted_values",
            "regr",
            "X",
            "feature_importance",
            "summary_stats",
            "models",
            "weights",
        ):
            try:
                delattr(self, attr)
            except AttributeError:
                pass

    def _prepare_sklearn(
        self, tune, Xvars, y, current_xreg
    ) -> Tuple[str, list, pd.DataFrame, int]:
        """ returns objects specific to forecasting with sklearn
            tune: bool
                whether the forecasting interation is for tuning the model
            Xvars: str or None
                if None, uses all Xvars
                if str, uses only those Xvars
        """
        if Xvars is None or Xvars == "all":
            Xvars = list(current_xreg.keys())

        if tune:
            y = list(y)[: -self.test_length]
            X = pd.DataFrame({k: list(v) for k, v in current_xreg.items()}).iloc[
                : -self.test_length, :
            ]
            test_size = self.validation_length
        else:
            y = list(y)
            X = pd.DataFrame({k: list(v) for k, v in current_xreg.items()})
            test_size = self.test_length
        X = X[Xvars]
        self.Xvars = Xvars
        return Xvars, y, X, test_size

    def _evaluate_sklearn(
        self,
        scaler,
        regr,
        X,
        y,
        Xvars,
        future_dates,
        future_xreg,
        dynamic_testing,
        true_forecast=False,
        lstm=False,
        **kwargs,
    ) -> list:
        """ forecasts an sklearn model into the unknown
            beginning in 0.4.1, now supports lstm model evaluation (three arguments added to function)
            uses loops to dynamically plug in AR values without leaking in either a tune/test process or true forecast, unless dynamic_testing is False
            returns a list of forecasted values
            scaler: sklearn.preprocessing.MinMaxScaler | sklearn.preprocessing.Normalizer | sklearn.preprocessing.StandardScaler | None
                the scaling to use on the future xreg values if not None
            regr: sklearn model
                the regression model to forecast with
            X: np.ndarray
                a matrix of regressor values
            y: np.ndarray
                the known dependent-variable values
            Xvars: str or None
                the name of the regressors to use
                must be stored in the current_xreg and future_xreg attributes
            true_forecast: bool, default False
                False if testing or tuning
                if True, saves regr, X, and fitted_values attributes
            lstm: bool, default False
                Whether evaluating an LSTM model
            **kwargs passed to fit() method and ignored when lstm == False
        """
        # not tuning/testing
        if true_forecast:
            self._clear_the_deck()
            self.dynamic_testing = dynamic_testing

        # apply the normalizer fit on training data only
        X = self._scale(scaler, X)
        regr.fit(X, y) if not lstm else regr.fit(X, y, **kwargs)

        # not tuning/testing
        if true_forecast:
            self.regr = regr
            self.X = X
            self.fitted_values = (
                list(regr.predict(X)) if not lstm else [x[0] for x in regr.predict(X)]
            )

        # if not using any AR terms or not dynamically evaluating the forecast, use the below (faster but ends up being an average of one-step forecasts when AR terms are involved)
        if (len([x for x in Xvars if x.startswith("AR")]) == 0) | (
            (not true_forecast) & (not dynamic_testing)
        ):
            p = pd.DataFrame(future_xreg)
            p = self._scale(scaler, p)
            fcst = list(regr.predict(p))

        # otherwise, use a dynamic process to propogate out-of-sample AR terms with predictions (slower but more indicative of a true forecast performance)
        else:
            fcst = []
            for i, _ in enumerate(future_dates):
                p = pd.DataFrame(
                    {k: [v[i]] for k, v in future_xreg.items() if k in Xvars}
                )
                p = self._scale(scaler, p)
                fcst.append(
                    regr.predict(p)[0]
                    if not lstm
                    else regr.predict(p.reshape(p.shape[0], p.shape[1], 1))[0][0]
                )
                if not i == len(future_dates) - 1:
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
        return fcst

    def _forecast_sklearn(
        self,
        fcster,
        dynamic_testing,
        tune=False,
        Xvars=None,
        normalizer="minmax",
        **kwargs,
    ) -> Union[float, list]:
        """ runs an sklearn forecast start-to-finish
            fcster: str, one of {_sklearn_estimators_}
            dynamic_testing: bool
                whether to dynamically test the forecast (meaning AR terms will be propogated with predicted values)
                setting this to False means faster performance, but gives a less-good indication of how well the forecast will perform out x amount of periods
                when False, test-set metrics effectively become an average of one-step forecasts
            tune: bool, default False
                whether the model is being tuned
            Xvars: list where all elements are in current_xreg keys, 'all', or None
                if None and Xvars are required, None becomes equivalent to 'all'
            normalizer: one of {_normalizer_}
                if not None, normalizer applied to training data only to not leak
            **kwargs treated as model hyperparameters and passed to _sklearn_imports_[model]()
        """
        descriptive_assert(
            len(self.current_xreg.keys()) > 0,
            ForecastError,
            f"need at least 1 Xvar to forecast with the {self.estimator} model",
        )

        # list of integers, each one representing the n/a values in each AR term
        ars = [
            v.isnull().sum() for x, v in self.current_xreg.items() if x.startswith("AR")
        ]
        # if using ARs, instead of foregetting those obs, ignore them with sklearn forecasts (leaves them available for other types of forecasts)
        obs_to_drop = max(ars) if len(ars) > 0 else 0
        y = self.y.values[obs_to_drop:].copy()
        current_xreg = {
            xvar: x.values[obs_to_drop:].copy() for xvar, x in self.current_xreg.items()
        }
        # get a list of Xvars, the y array, the X matrix, and the test size (can be different depending on if tuning or testing)
        Xvars, y, X, test_size = self._prepare_sklearn(tune, Xvars, y, current_xreg)
        # split the data
        X_train, X_test, y_train, y_test = self._train_test_split(X, y, test_size)
        # fit the normalizer to training data only
        scaler = self._parse_normalizer(X_train, normalizer)
        # get the sklearn regressor
        regr = _sklearn_imports_[fcster](**kwargs)
        # train/test the model
        pred = self._evaluate_sklearn(
            scaler,
            regr,
            X_train,
            y_train,
            Xvars,
            self.current_dates.values[-test_size:],
            {x: v[-test_size:] for x, v in current_xreg.items() if x in Xvars},
            dynamic_testing,
        )
        # set the test-set metrics
        self._metrics(y_test, pred)

        # if we are tuning, return the relevant test-set metric only and delete other metrics out of memory
        # otherwise, get me my entire forecast into the future
        return (
            self._tune()
            if tune
            else self._evaluate_sklearn(
                scaler,
                regr,
                X,
                y,
                Xvars,
                self.future_dates.copy(),
                {x: v[:] for x, v in self.future_xreg.items() if x in Xvars},
                dynamic_testing,
                true_forecast=True,
            )
        )

    def _forecast_hwes(
        self, tune=False, dynamic_testing=True, **kwargs
    ) -> Union[float, list]:
        """ forecasts with holt-winters exponential smoothing
            tune: bool, default False
                whether to tune the forecast
                if True, returns a metric
                if False, returns a list of forecasted values
            dynamic_testing: bool, default True
                always ignored in HWES (for now) - everything is set to be dynamic using statsmodels
            **kwargs passed to the HWES() function from statsmodels
        """
        if not dynamic_testing:
            logging.warning(
                "dynamic_testing argument will be ignored for the hwes model"
            )
        self.dynamic_testing = True
        from statsmodels.tsa.holtwinters import ExponentialSmoothing as HWES

        y = self.y.to_list()
        if tune:
            y_train = y[: -(self.validation_length + self.test_length)]
            y_test = y[-(self.test_length + self.validation_length) : -self.test_length]
        else:
            y_train = y[: -self.test_length]
            y_test = y[-self.test_length :]
        self.Xvars = None
        hwes_train = HWES(
            y_train,
            dates=self.current_dates.values[: -self.test_length],
            freq=self.freq,
            **kwargs,
        ).fit(optimized=True, use_brute=True)
        pred = hwes_train.predict(
            start=len(y_train), end=len(y_train) + len(y_test) - 1
        )
        self._metrics(y_test, pred)
        if tune:
            return self._tune()
        else:  # forecast
            self._clear_the_deck()
            self.univariate = True
            self.X = None
            regr = HWES(self.y, dates=self.current_dates, freq=self.freq, **kwargs).fit(
                optimized=True, use_brute=True
            )
            self.fitted_values = list(regr.fittedvalues)
            self.regr = regr
            self._set_summary_stats()
            return list(
                regr.predict(start=len(y), end=len(y) + len(self.future_dates) - 1)
            )

    def _forecast_arima(
        self, tune=False, Xvars=None, dynamic_testing=True, **kwargs
    ) -> Union[float, list]:
        """ forecasts with ARIMA (or AR, ARMA, SARIMA, SARIMAX)
            tune: bool, default False
                whether to tune the forecast
                if True, returns a metric
                if False, returns a list of forecasted values
            Xvars: str or None, default None
                the names of the regressors to use -- must match names in current_xreg and future_xreg
                if None, unlike sklearn model, will use no regressors
            dynamic_testing: bool, default True
                always ignored in ARIMA (for now) - everything is set to be dynamic using statsmodels
            **kwargs passed to the ARIMA() function from statsmodels
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
        Xvars, y, X, test_size = self._prepare_sklearn(
            tune, Xvars, self.y, self.current_xreg
        )
        if len(self.current_xreg.keys()) > 0:
            X_train, X_test, y_train, y_test = self._train_test_split(X, y, test_size)
        else:
            y_train = self.y.values[:test_size]
            y_test = self.y.values[-test_size:]
        if Xvars_orig is None:
            X, X_train, X_test = None, None, None
            self.Xvars = None
        arima_train = ARIMA(
            y_train,
            exog=X_train,
            dates=self.current_dates.values[: -self.test_length],
            freq=self.freq,
            **kwargs,
        ).fit()
        pred = arima_train.predict(
            exog=X_test,
            start=len(y_train),
            end=len(y_train) + len(y_test) - 1,
            typ="levels",
        )
        self._metrics(y_test, pred)
        if tune:
            return self._tune()
        else:
            self._clear_the_deck()
            if Xvars_orig is None:
                self.univariate = True
            self.X = X
            regr = ARIMA(
                self.y.values[:],
                exog=X,
                dates=self.current_dates,
                freq=self.freq,
                **kwargs,
            ).fit()
            self.fitted_values = list(regr.fittedvalues)
            self.regr = regr
            self._set_summary_stats()
            p = (
                pd.DataFrame(
                    {k: v for k, v in self.future_xreg.items() if k in self.Xvars}
                )
                if self.Xvars is not None
                else None
            )
            fcst = regr.predict(
                exog=p,
                start=len(y),
                end=len(y) + len(self.future_dates) - 1,
                typ="levels",
                dynamic=True,
            )
            return list(fcst)

    def _forecast_prophet(
        self,
        tune=False,
        Xvars=None,
        dynamic_testing=True,
        cap=None,
        floor=None,
        **kwargs,
    ) -> Union[float, list]:
        """ forecasts with the prophet model from facebook
            tune: bool, default False
                whether to tune the forecast
                if True, returns a metric
                if False, returns a list of forecasted values
            Xvars: str or None, default None
                the names of the regressors to use -- must match names in current_xreg and future_xreg
                if None, unlike sklearn model, will use no regressors
            dynamic_testing: bool, default True
                always ignored for Prophet (for now)
            cap: float or None, default None
                specific to prophet when using logistic growth -- the largest amount the model is allowed to evaluate to
            floor: float or None, default None
                specific to prophet when using logistic growth -- the smallest amount the model is allowed to evaluate to
            **kwargs passed to the Prophet() function from fbprophet
        """
        if not dynamic_testing:
            logging.warning(
                "dynamic_testing argument will be ignored for the prophet model"
            )
        self.dynamic_testing = True
        from fbprophet import Prophet

        X = pd.DataFrame(
            {k: v for k, v in self.current_xreg.items() if not k.startswith("AR")}
        )
        p = pd.DataFrame(
            {k: v for k, v in self.future_xreg.items() if not k.startswith("AR")}
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
        if tune:
            X_train = X.iloc[: -(self.test_length + self.validation_length)]
            X_test = X.iloc[
                -(self.test_length + self.validation_length) : -self.test_length
            ]
            y_test = X["y"].values[
                -(self.test_length + self.validation_length) : -self.test_length
            ]
            model.fit(X_train)
            pred = model.predict(X_test)
            self._metrics(y_test, pred["yhat"].to_list())
            return self._tune()
        else:
            model.fit(X.iloc[: -self.test_length])
            pred = model.predict(X.iloc[-self.test_length :])
            self._metrics(X["y"].values[-self.test_length :], pred["yhat"].to_list())
            self._clear_the_deck()
            self.X = X[Xvars]
            if len(Xvars) == 0:
                self.univariate = True
                self.X = None
            self.Xvars = Xvars if Xvars != [] else None

            regr = Prophet(**kwargs)
            regr.fit(X)
            self.fitted_values = regr.predict(X)["yhat"].to_list()
            self.regr = regr
            fcst = regr.predict(p)
            return fcst["yhat"].to_list()

    def _forecast_silverkite(
        self, tune=False, dynamic_testing=True, Xvars=None, **kwargs
    ) -> Union[float, list]:
        """ forecasts with the silverkte model from LinkedIn greykite library
            tune: bool, default False
                whether to tune the forecast
                if True, returns a metric
                if False, returns a list of forecasted values
            dynamic_testing: bool, default True
                always ignored for silverkite (for now)
            Xvars: str or None, default None
                the names of the regressors to use -- must match names in current_xreg and future_xreg
                if None, unlike sklearn model, will use no regressors
            **kwargs passed to the ModelComponentsParam function from greykite.framework.templates.autogen.forecast_config
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
            validation_length = (
                validation_length if validation_length > 0 else -(df.shape[0] + 1)
            )
            pred_df = df.iloc[:-test_length, :]
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

        if tune:
            y_test = self.y.values[
                -(self.test_length + self.validation_length) : -self.test_length
            ]
            pred = _forecast_sk(
                df,
                Xvars,
                self.validation_length,
                self.test_length,
                self.validation_length,
            )[0]
            self._metrics(y_test, pred[-self.validation_length :])
            return self._tune()
        else:
            pred = _forecast_sk(df, Xvars, self.test_length, 0, self.test_length)[0]
            self._metrics(self.y.values[-self.test_length :], pred[-self.test_length :])
            self._clear_the_deck()
            self.X = df[Xvars]
            if len(Xvars) == 0:
                self.univariate = True
                self.X = None
            self.Xvars = Xvars if Xvars != [] else None
            self.regr = None  # placeholder to make feature importance work
            result = _forecast_sk(df, Xvars, 0, 0, fcst_length)
            self.summary_stats = result[1].set_index("Pred_col")
            self.fitted_values = result[0][:-fcst_length]
            return result[0][-fcst_length:]

    def _prepare_lstm(self, yvals, lags, forecast_length):
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

    def _forecast_rnn(
        self,
        dynamic_testing=True,
        lags=1,
        hidden_layers_struct={'simple':{'units':8,'activation':'tanh'}},
        loss="mean_absolute_error",
        optimizer="Adam",
        learning_rate=0.001,
        random_seed=None,
        plot_loss=False,
        **kwargs,
    ):
        """ forecasts with a recurrent neural network from TensorFlow, such as lstm or simple recurrent
            cannot be tuned
            only xvar options are the series' own history (specify in lags argument)
            always uses minmax normalizer
            this is a similar function to _forecast_lstm() but it is more complex to allow more flexibility
            fitted values are the last fcst_length worth of values only
            dynamic_testing: bool, default True
                always ignored for lstm because the model doesn't work like others
            lags: int greater than 0, default 1
                the number of y-variable lags to train the model with
            hidden_layers_struct: dict[str,dict[str,Union[float,str]]], default {'simple':{'size':8,'activation':'tanh'}}
                key is the type of each hidden layer, one of {'simple','lstm'}
                val is a dict
                    key is str representing hyperparameter value: 'units','activation', etc
                        see all possible here for simple rnn: https://www.tensorflow.org/api_docs/python/tf/keras/layers/SimpleRNN
                        here for lstm: https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM
                    val is the desired hyperparam value
                    do not pass return_sequences or input_shape as these will be set automatically
            loss: str, default 'mean_absolute_error'
                the loss function to minimize
                see available options here: https://www.tensorflow.org/api_docs/python/tf/keras/losses
                be sure to choose one that is suitable for regression tasks
            optimizer: str, default "Adam"
                the optimizer to use when compiling the model
                see available values here: https://www.tensorflow.org/api_docs/python/tf/keras/optimizers
                watch capitalization as that matters (can't be "adam", must be "Adam")
            learning_rate: float, default 0.001
                the learning rate to use when compiling the model
            random_seed: int, optional
                set a seed for consistent results
                with tensorflow networks, setting seeds does not guarantee consistent results
            plot_loss: bool, default False
                whether to plot the LSTM loss function stored in history for each epoch
                if validation_split passed to kwargs, will plot the validation loss as well
                looks better if epochs > 1 passed to **kwargs
            **kwargs passed to fit() and can include epochs, verbose, callbacks, validation_split, and more
        """
        descriptive_assert(
            len(hidden_layers_struct.keys()) >= 1,
            ValueError,
            f"must pass at least one layer to hidden_layers_struct, got {hidden_layers_struct}",
        )
        if not dynamic_testing:
            logging.warning(
                "dynamic_testing argument will be ignored for the rnn model"
            )
        self.dynamic_testing = True

        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, LSTM, SimpleRNN
        import tensorflow.keras.optimizers

        optimizer = eval(f"tensorflow.keras.optimizers.{optimizer}")

        if random_seed is not None:
            random.seed(random_seed)

        y_train = self.y.values[:-self.test_length].copy()
        y_test = self.y.values[-self.test_length:].copy()

        ymin = y_train.min()
        ymax = y_train.max()

        X_train, y_train_new = self._prepare_lstm(y_train,lags,self.test_length)
        X, y_new = self._prepare_lstm(self.y.values.copy(),lags,len(self.future_dates))

        X_test = np.array([[(i - ymin) / (ymax - ymin) for i in self.y.values[-(lags + self.test_length):-self.test_length].copy()]])
        y_test_new = np.array([[(i - ymin) / (ymax - ymin) for i in self.y.values[-self.test_length:].copy()]])
        fut = np.array([[(i - self.y.min()) / (self.y.max() - self.y.min()) for i in self.y.values[-lags:].copy()]])

        n_timesteps = X_train.shape[1]
        n_features = 1

        X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],1)
        X_test = X_test.reshape(X_test.shape[0],X_test.shape[1],1)
        X = X.reshape(X.shape[0],X.shape[1],1)
        fut = fut.reshape(fut.shape[0],fut.shape[1],1)

        def get_compiled_model(y):
            # build model
            for i, kv in enumerate(hidden_layers_struct.items()):
                descriptive_assert(
                    kv[0] in ('simple','lstm'),
                    ValueError,
                    f'each key in the hidden_layers_struct dict must be one of ("simple","lstm"), got {kv[0]}'
                )
                layer = SimpleRNN if kv[0] == 'simple' else LSTM
                if i == 0:
                    model = Sequential(
                        [
                            layer(
                                **kv[1],
                                input_shape=(n_timesteps, n_features),
                                return_sequences=len(hidden_layers_struct.keys()) > 1,
                            )
                        ]
                    )
                else:
                    model.add(
                        layer(
                            **kv[1],
                            return_sequences=(not i == (len(hidden_layers_struct.keys()) - 1))
                        )
                    )
            model.add(Dense(y.shape[1]))  # output layer

            # compile model
            model.compile(optimizer=optimizer(learning_rate=learning_rate), loss=loss)
            return model

        test_model = get_compiled_model(y_train_new)
        test_model.fit(X_train,y_train_new,**kwargs)
        pred = test_model.predict(X_test)

        pred = [p * (ymax-ymin) + ymin for p in pred[0]] # un-minmax

        # set the test-set metrics
        self._metrics(y_test, pred)

        model = get_compiled_model(y_new)
        hist = model.fit(X,y_new,**kwargs)

        if plot_loss:
            plt.plot(hist.history['loss'],label='train_loss')
            if 'val_loss' in hist.history.keys():
                plt.plot(hist.history['val_loss'],label='val_loss')
            plt.title('model loss')
            plt.xlabel('epoch')
            plt.legend(loc='upper right')
            plt.show()

        fcst = model.predict(fut)
        self.fitted_values = [p * (self.y.max()-self.y.min()) + self.y.min() for p in model.predict(X)[0]] # only last fcst amount of values
        self.Xvars = None
        self.univariate = True

        return [p * (self.y.max()-self.y.min()) + self.y.min() for p in fcst[0]]


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
        """ forecasts with a long-short term memory neural network from TensorFlow
            cannot be tuned
            only xvar options are the series' own history (specify in lags argument)
            always uses minmax normalizer
            fitted values are the last fcst_length worth of values only
            dynamic_testing: bool, default True
                always ignored for lstm because the model doesn't work like others
            lags: int greater than 0, default 1
                the number of y-variable lags to train the model with
            lstm_layer_sizes: list-like, default (25,)
                the size of each lstm layer to add
                the first element is for the input layer
                the size of this array minus 1 will equal the number of hidden layers in the resulting model
            dropout: list-like, default (0.0,)
                the dropout rate for each lstm layer
                must be the same size as lstm_layer_sizes
            loss: str, default 'mean_absolute_error'
                the loss function to minimize
                see available options here: https://www.tensorflow.org/api_docs/python/tf/keras/losses
                be sure to choose one that is suitable for regression tasks
            activation: str, default "tanh"
                the activation function to use in each lstm layer
                see available values here: https://www.tensorflow.org/api_docs/python/tf/keras/activations
            optimizer: str, default "Adam"
                the optimizer to use when compiling the model
                see available values here: https://www.tensorflow.org/api_docs/python/tf/keras/optimizers
                watch capitalization as that matters (can't be "adam", must be "Adam")
            learning_rate: float, default 0.001
                the learning rate to use when compiling the model
            random_seed: int, optional
                set a seed for consistent results
                with tensorflow networks, setting seeds does not guarantee consistent results
            plot_loss: bool, default False
                whether to plot the LSTM loss function stored in history for each epoch
                if validation_split passed to kwargs, will plot the validation loss as well
                looks better if epochs > 1 passed to **kwargs
            **kwargs passed to fit() and can include epochs, verbose, callbacks, validation_split, and more
        """
        descriptive_assert(
            len(lstm_layer_sizes) >= 1,
            ValueError,
            f"must pass at least one layer to lstm_layer_sizes, got {lstm_layer_sizes}",
        )
        descriptive_assert(
            len(dropout) == len(lstm_layer_sizes),
            ValueError,
            f"length of dropout must be the same size as lstm_layer_sizes, got {len(dropout)} and {len(lstm_layer_sizes)}",
        )
        if not dynamic_testing:
            logging.warning(
                "dynamic_testing argument will be ignored for the lstm model"
            )
        self.dynamic_testing = True

        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, LSTM
        import tensorflow.keras.optimizers

        optimizer = eval(f"tensorflow.keras.optimizers.{optimizer}")

        if random_seed is not None:
            random.seed(random_seed)

        y_train = self.y.values[:-self.test_length].copy()
        y_test = self.y.values[-self.test_length:].copy()

        ymin = y_train.min()
        ymax = y_train.max()

        X_train, y_train_new = self._prepare_lstm(y_train,lags,self.test_length)
        X, y_new = self._prepare_lstm(self.y.values.copy(),lags,len(self.future_dates))

        X_test = np.array([[(i - ymin) / (ymax - ymin) for i in self.y.values[-(lags + self.test_length):-self.test_length].copy()]])
        y_test_new = np.array([[(i - ymin) / (ymax - ymin) for i in self.y.values[-self.test_length:].copy()]])
        fut = np.array([[(i - self.y.min()) / (self.y.max() - self.y.min()) for i in self.y.values[-lags:].copy()]])

        n_timesteps = X_train.shape[1]
        n_features = 1

        X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],1)
        X_test = X_test.reshape(X_test.shape[0],X_test.shape[1],1)
        X = X.reshape(X.shape[0],X.shape[1],1)
        fut = fut.reshape(fut.shape[0],fut.shape[1],1)

        def get_compiled_model(y):
            # build model
            model = Sequential(
                [
                    LSTM(
                        lstm_layer_sizes[0],
                        activation=activation,
                        input_shape=(n_timesteps, n_features),
                        dropout=dropout[0],
                        return_sequences=len(lstm_layer_sizes) > 1,
                    )
                ]
            )
            for i, layer in enumerate(lstm_layer_sizes):
                if i > 0:  # since we already added the first layer
                    model.add(
                        LSTM(
                            layer,
                            activation=activation,
                            return_sequences=(not i == (len(lstm_layer_sizes) - 1)),
                            dropout=dropout[i],
                        )
                    )
            model.add(Dense(y.shape[1]))  # output layer

            # compile model
            model.compile(optimizer=optimizer(learning_rate=learning_rate), loss=loss)
            return model

        test_model = get_compiled_model(y_train_new)
        test_model.fit(X_train,y_train_new,**kwargs)
        pred = test_model.predict(X_test)

        pred = [p * (ymax-ymin) + ymin for p in pred[0]] # un-minmax

        # set the test-set metrics
        self._metrics(y_test, pred)

        model = get_compiled_model(y_new)
        hist = model.fit(X,y_new,**kwargs)

        if plot_loss:
            plt.plot(hist.history['loss'],label='train_loss')
            if 'val_loss' in hist.history.keys():
                plt.plot(hist.history['val_loss'],label='val_loss')
            plt.title('model loss')
            plt.xlabel('epoch')
            plt.legend(loc='upper right')
            plt.show()

        fcst = model.predict(fut)
        self.fitted_values = [p * (self.y.max()-self.y.min()) + self.y.min() for p in model.predict(X)[0]] # only last fcst amount of values
        self.Xvars = None
        self.univariate = True

        return [p * (self.y.max()-self.y.min()) + self.y.min() for p in fcst[0]]

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
        """ combines at least two previously evaluted forecasts to create a new estimator
            how: one of {'simple','weighted','splice'}, default 'simple'
                the type of combination
                all test lengths must be the same for all combined models
            models: 'all', starts with "top_", or list-like, default 'all'
                which models to combine
                must be at least 2 in length
                if using list-like object, elements must match model nicknames specified in call_me when forecasting
            dynamic_testing: bool, default True
                always ignored for combo (for now and possibly forever)
            determine_best_by: one of {'TestSetRMSE','TestSetMAPE','TestSetMAE','TestSetR2InSampleRMSE','InSampleMAPE','InSampleMAE','InSampleR2','ValidationMetricValue','LevelTestSetRMSE','LevelTestSetMAPE','LevelTestSetMAE','LevelTestSetR2',None}, default 'ValidationMetricValue'
                if (models does not start with 'top_' and how is not 'weighted') or (how is 'weighted' and manual weights are specified), this is ignored
                'TestSetRMSE','TestSetMAPE','TestSetMAE','TestSetR2InSampleRMSE','LevelTestSetRMSE','LevelTestSetMAPE','LevelTestSetMAE','LevelTestSetR2' will probably lead to overfitting (data leakage)
                'InSampleMAPE','InSampleMAE','InSampleR2' probably will lead to overfitting since in-sample includes the test set and overfitted models are weighted more highly
                'ValidationMetricValue' is the safest option to avoid overfitting, but only works if all combined models were tuned and the validation metric was the same for all models
            rebalance_weights: float, default 0.1
                a minmax/maxmin scaler is used to perform the weighted average, but this method means the worst performing model on the test set is always weighted 0
                to correct that so that all models have some weight in the final combo, you can rebalance the weights but specifying this parameter
                the higher this is, the closer to a simple average the weighted average becomes
                must be at least 0 -- 0 means the worst model is not given any weight
            weights: (optional) list-like
                only applicable when how='weighted'
                manually specifies weights
                must be the same size as models
                if None and how='weighted', weights are set automatically
                if manually passed weights do not add to 1, will rebalance them
            splice_points: (optional) list-like
                only applicable when how='splice'
                elements in array must be str in yyyy-mm-dd or datetime object
                must be exactly one less in length than the number of models
                    models[0] --> :splice_points[0]
                    models[-1] --> splice_points[-1]:
        """
        if not dynamic_testing:
            logging.warning(
                "dynamic_testing argument will be ignored for the combo model"
            )
        self.dynamic_testing = None
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
                )  # always use r2 since higher is better (could use maxmin scale for other metrics?)
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
                    "all elements in splice_points must be datetime objects or str in yyyy-mm-dd format and must be present in future_dates attribute",
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
        self._clear_the_deck()
        self.weights = tuple(weights.values[0]) if weights is not None else None
        self.models = models
        self.fitted_values = fv
        self.Xvars = None
        self.X = None
        self.regr = None
        return fcst

    def _parse_models(self, models, determine_best_by) -> list:
        """ takes a list-like of models and orders them best-to-worst based on a given metric and returns the ordered list (of str type)
            models: a list-like object where each element is one of _estimators_
            determine_best_by: one of {_determine_best_by_}
                if a model does not have the metric specified here (i.e. one of the passed models wasn't tuned and this is 'ValidationMetricValue'), it will be ignored silently, so be careful
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
            n: one of {True,False,0,1,2}
                If False or 0, does not difference
                If True or 1, differences 1 time
                If 2, differences 2 times
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

    def infer_freq(self) -> None:
        """ uses pandas library to infer frequency of loaded dates
        """
        if not hasattr(self, "freq"):
            self.freq = pd.infer_freq(self.current_dates)
            self.current_dates.freq = self.freq

    def fillna_y(self, how="ffill") -> None:
        """ fills null values in the y attribute
            how: {'backfill', 'bfill', 'pad', 'ffill', 'midpoint'}
                midpoint is unique to this library and only works if there is not more than two missing values sequentially
                all other possible arguments are from pandas.DataFrame.fillna() method and will do the same
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

    def generate_future_dates(self, n) -> None:
        """ generates a certain amount of future dates based on an inferred frequency
            n: int
                number of future dates to produce
                this will also be the forecast length
        """
        self.future_dates = pd.Series(
            pd.date_range(
                start=self.current_dates.values[-1], periods=n + 1, freq=self.freq
            ).values[1:]
        )

    def set_last_future_date(self, date) -> None:
        """ generates future dates that ends on the passed date
            date: datetime.datetime, pandas.Timestamp, or str ('%Y-%m-%d' format)
                the date to end on
                the number of future generated dates will be used as the forecast length
        """
        if isinstance(date, str):
            date = datetime.datetime.strptime(date, "%Y-%m-%d")
        self.future_dates = pd.Series(
            pd.date_range(
                start=self.current_dates.values[-1], end=date, freq=self.freq
            ).values[1:]
        )

    def typ_set(self) -> None:
        """ converts all objects in y, current_dates, future_dates, current_xreg, and future_xreg to appropriate types if possible
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

    def diff(self, i=1) -> None:
        """ differences the y attribute, as well as all regressor values stored in current_xreg and future_xreg
            call this after adding all desired ar terms and those terms will be differenced too
            if you add ar terms after differencing, an error will be raised
            i: one of {0,1,2}, default 1
                the number of differences to take
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
    ) -> None:
        """differences the series 0, 1, or 2 times based on ADF test
            critical_pval: float, default 0.05
                the p-value threshold in the statistical test to accept the alternative hypothesis
            train_only: bool, default False
                if True, will exclude the test set from the test (a measure added to avoid leakage)
            max_integration: int, one of {1,2}, default 2
                if 1, will only difference data up to one time even if the results of the test indicate two integrations
                if 2, behaves how you would expect
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

    def add_ar_terms(self, n) -> None:
        """ add auto-regressive terms to forecast with
            n: int
                the number of terms to add (1 to this number will be added)
        """
        self._adder()
        descriptive_assert(isinstance(n, int), ValueError, f"n must be an int, got {n}")
        descriptive_assert(n > 0, ValueError, f"n must be greater than 0, got {n}")
        descriptive_assert(
            self.integration == 0,
            ForecastError,
            "AR terms must be added before differencing (don't worry, they will be differenced too)",
        )
        for i in range(1, n + 1):
            self.current_xreg[f"AR{i}"] = pd.Series(self.y).shift(i)
            self.future_xreg[f"AR{i}"] = [self.y.values[-i]]

    def add_AR_terms(self, N) -> None:
        """ add seasonal AR terms
            N: tuple of len 2 (P,m)
                P: int
                    the number of terms to add
                m: int
                    the seasonal period (12 for monthly data, etc.)
        """
        self._adder()
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
    ) -> None:
        """ ingest a dataframe of regressors with names (don't start anything with AR!!!)
            all non-numeric values will be dummied
            df: pandas.DataFrame
            date_col: str, default 'Date'
                the name of the date column in the dataframe (use named index only if passing this column as an index)
            drop_first: bool, default False
                whether to drop the first observation of any dummied variables, irrelevant if passing all numeric values
            use_future_dates: bool, default False
                whether to use the future dates in the dataframe as the future_dates attribute in the object
                if False, the dataframe must have at least the same number of observations as len(future_dates)
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

    def set_test_length(self, n=1) -> None:
        """ set the length of the test set (no fractional splits)
            n: int, default 1
                the length of the resulting test set
        """
        descriptive_assert(isinstance(n, int), ValueError, f"n must be an int, got {n}")
        self.test_length = n

    def set_validation_length(self, n=1) -> None:
        """ set the length of the validation set (no fractional splits)
            n: int, default 1
                the length of the resulting validation set
        """
        descriptive_assert(isinstance(n, int), ValueError, f"n must be an int, got {n}")
        descriptive_assert(n > 0, ValueError, f"n must be greater than 1, got {n}")
        if (self.validation_metric == "r2") & (n == 1):
            raise ValueError(
                "can only set a validation_length of 1 if validation_metric is not r2. try set_validation_metric()"
            )
        self.validation_length = n

    def set_cilevel(self, n):
        """ sets the level for the resulting confidence intervals (95% default)
            n: float greater than 0 and less than 1
        """
        descriptive_assert(
            n < 1 and n > 0, ValueError, "n must be greater than 0 and less than 1"
        )
        self.cilevel = n

    def set_bootstrap_samples(self, n):
        """ sets the number of bootstrap samples to set confidence intervals for each model (default is 100)
            n: float greater than or equal to 30
                30 because you need around there to satisfy central limit theorem
                the lower this number, the faster the performance, but the less sure of the outcome you can be
        """
        descriptive_assert(n >= 30, ValueError, "n must be greater than or equal to 30")
        self.bootstrap_samples = n

    def adf_test(
        self, critical_pval=0.05, quiet=True, full_res=False, train_only=False, **kwargs
    ) -> Union[dict, bool]:
        """ tests the stationarity of the y series using augmented dickey fuller
            critical_pval: float, default 0.05
                the p-value threshold in the statistical test to accept the alternative hypothesis
            quiet: bool, default True
                if False, prints whether the tests suggests stationary or non-stationary data
            full_res: bool, default False
                if True, returns a dictionary with the pvalue, evaluated statistic, and other statistical information (returns what the adfuller() function from statsmodels does)
                if False, returns a bool that matches whether the test indicates stationarity
            train_only: bool, default False
                if True, will exclude the test set from the test (a measure added to avoid leakage)
            **kwargs passed to adfuller() function from statsmodels
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

    def plot_acf(self, diffy=False, train_only=False, **kwargs) -> plt:
        """ plots an autocorrelation function of the y values
            diffy: one of {True,False,0,1,2}, default False
                whether to difference the data and how many times before passing the values to the function
                If False or 0, does not difference
                If True or 1, differences 1 time
                If 2, differences 2 times
            train_only: bool, default False
                if True, will exclude the test set from the test (a measure added to avoid leakage)
            **kwargs passed to plot_acf() function from statsmodels
        """
        descriptive_assert(
            isinstance(train_only, bool), ValueError, "train_only must be True or False"
        )
        y = self._diffy(diffy)
        y = y.values if not train_only else y.values[: -self.test_length]
        return plot_acf(y, **kwargs)

    def plot_pacf(self, diffy=False, train_only=False, **kwargs) -> plt:
        """ plots a partial autocorrelation function of the y values
            diffy: one of {True,False,0,1,2}, default False
                whether to difference the data and how many times before passing the values to the function
                If False or 0, does not difference
                If True or 1, differences 1 time
                If 2, differences 2 times
            train_only: bool, default False
                if True, will exclude the test set from the test (a measure added to avoid leakage)
            **kwargs passed to plot_pacf() function from statsmodels
        """
        descriptive_assert(
            isinstance(train_only, bool), ValueError, "train_only must be True or False"
        )
        y = self._diffy(diffy)
        y = y.values if not train_only else y.values[: -self.test_length]
        return plot_pacf(y, **kwargs)

    def plot_periodogram(self, diffy=False, train_only=False):
        """ plots a periodogram of the y values (comes from scipy.signal)
            diffy: one of {True,False,0,1,2}, default False
                whether to difference the data and how many times before passing the values to the function
                If False or 0, does not difference
                If True or 1, differences 1 time
                If 2, differences 2 times
            train_only: bool, default False
                if True, will exclude the test set from the test (a measure added to avoid leakage)
        """
        from scipy.signal import periodogram

        descriptive_assert(
            isinstance(train_only, bool), ValueError, "train_only must be True or False"
        )
        y = self._diffy(diffy)
        y = y.values if not train_only else y.values[: -self.test_length]
        return periodogram(y)

    def seasonal_decompose(self, diffy=False, train_only=False, **kwargs):
        """ plots a signal/seasonal decomposition of the y values
            diffy: one of {True,False,0,1,2}, default False
                whether to difference the data and how many times before passing the values to the function
                If False or 0, does not difference
                If True or 1, differences 1 time
                If 2, differences 2 times
            train_only: bool, default False
                if True, will exclude the test set from the test (a measure added to avoid leakage)
            **kwargs passed to seasonal_decompose() function from statsmodels
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
    ) -> None:
        """ adds seasonal regressors to the object
            *args: each of str type
                values that return a series of int type from pandas.dt and pandas.dt.isocalendar()
            raw: bool, default True
                whether to use the raw integer values
            sincos: bool, default False
                whether to use a sin/cos transformation of the raw integer values (estimates the cycle based on the max observed value)
            dummy: bool, default False
                whether to use dummy variables from the raw int values
            drop_first: bool, default False
                whether to drop the first observed dummy level (saves a degree of freedom when model estimates an intercept)
                not relevant when dummy = False
        """
        self._adder()
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
                    f"{s} must return an int; use dummy = True to get dummies"
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

    def add_time_trend(self, called="t") -> None:
        """ adds a time trend from 0 to len(current_dates) + len(future_dates)
            called: str, default 't'
                what to call the resulting variable
        """
        self._adder()
        self.current_xreg[called] = pd.Series(range(1, len(self.y) + 1))
        self.future_xreg[called] = list(
            range(len(self.y) + 1, len(self.y) + len(self.future_dates) + 1)
        )

    def add_other_regressor(self, called, start, end) -> None:
        """ adds dummy variable that is 1 during the specified time period, 0 otherwise
            called: str
                what to call the resulting variable
            start: str, datetime, or pd.Timestamp object
            end: str, datetime, or pd.Timestamp object
        """
        self._adder()
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
    ) -> None:  # default is from when disney world closed to the end of the national (USA) mask mandate
        """ adds dummy variable that is 1 during the time period that covid19 effects are present for the series, 0 otherwise
            this function may be out of date as the pandemic has lasted longer than most expected, but we are keeping it for now
            called str, default 'COVID19'
               what to call the resulting variable
            start: str, datetime, or pd.Timestamp object, default datetime.datetime(2020,3,15)
                the start date (default is day Walt Disney World closed in the U.S.)
                use format yyyy-mm-dd when passing strings
           end: str, datetime, or pd.Timestamp object, default datetime.datetime(2021,5,13)
               the end date (default is day the U.S. CDC dropped mask mandate/recommendation for vaccinated people)
                use format yyyy-mm-dd when passing strings
        """
        self._adder()
        self.add_other_regressor(called=called, start=start, end=end)

    def add_combo_regressors(self, *args, sep="_") -> None:
        """ combines all passed variables by multiplying their values together
            *args: names of Xvars that aleady exist in the object
            sep: str, default '_'
                the separator between each term in arg to create the final variable name 
        """
        self._adder()
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

    def add_poly_terms(self, *args, pwr=2, sep="^") -> None:
        """ raises all passed variables (no AR terms) to exponential powers (ints only)
            *args: names of Xvars that aleady exist in the object
            pwr: int, default 2
                the max power to add to each term in args (2 to this number will be added)
            sep: str, default '^'
                the separator between each term in arg to create the final variable name 
        """
        self._adder()
        for a in args:
            descriptive_assert(
                not a.startswith("AR"),
                ForecastError,
                "no polynomial AR terms at this time -- it confuses the forecasting mechanism",
            )
            for i in range(2, pwr + 1):
                self.current_xreg[f"{a}{sep}{i}"] = self.current_xreg[a] ** i
                self.future_xreg[f"{a}{sep}{i}"] = [x ** i for x in self.future_xreg[a]]

    def add_exp_terms(self, *args, pwr, sep="^", cutoff=2):
        """ raises all passed variables (no AR terms) to exponential powers (ints or floats)
            *args: names of Xvars that aleady exist in the object
            pwr: float
                the power to raise each term to in args
                can use values like 0.5 to perform square roots, etc.
            sep: str, default '^'
                the separator between each term in arg to create the final variable name 
            cutoff: int, default 2
                the resulting variable name will be rounded to this number based on the passed pwr
                for instance, if pwr = 0.33333333333 and 't' is passed as an arg to *args, the resulting name will be t^0.33 by default
        """
        self._adder()
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

    def add_logged_terms(self, *args, base=math.e, sep=""):
        """ logs all passed variables (no AR terms)
            *args: names of Xvars that aleady exist in the object
            base: math.e or int greater than 1
                the log base
            sep: str, default ''
                the separator between each term in arg to create the final variable name
                resulting variable names will be like "log2t" or "lnt" by default
        """
        self._adder()
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

    def add_pt_terms(self, *args, method="box-cox", sep="_"):
        """ applies a box-cox or yeo-johnson power transformation to all passed variables (no AR terms)
            *args: names of Xvars that aleady exist in the object
            method: one of {'box-cox','yeo-johnson'}, default 'box-cox'
                the type of transformation
                box-cox works for positive values only
                yeo-johnson is like a box-cox but can be used with 0s or negatives (https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PowerTransformer.html)
            sep: str, default ''
                the separator between each term in arg to create the final variable name
                resulting variable names will be like "box-cox_t" or "yeo-johnson_t" by default 
        """
        self._adder()
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

    def add_diffed_terms(self, *args, diff=1, sep="_"):
        """ differences all passed variables (no AR terms) up to 2 times
            *args: names of Xvars that aleady exist in the object
            diff: one of {1,2}, default 1
                the number of times to difference each variable passed to args
            sep: str, default '_'
                the separator between each term in arg to create the final variable name
                resulting variable names will be like "tdiff_1" or "tdiff_2" by default

        """
        self._adder()
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

    def add_lagged_terms(self, *args, lags=1, upto=True, sep="_"):
        """ lags all passed variables (no AR terms) 1 or more times
            *args: names of Xvars that aleady exist in the object
            lags: int greater than 0, default 1
                the number of times to lag each passed variable
            upto: bool, default True
                whether to add all lags up to the number passed to lags
                if you pass 6 to lags and upto is True, lags 1, 2, 3, 4, 5, 6 will all be added
                if you pass 6 to lags and upto is False, lag 6 only will be added
            sep: str, default '_'
                the separator between each term in arg to create the final variable name
                resulting variable names will be like "tlag_1" or "tlag_2" by default
        """
        self._adder()
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

    def undiff(self, suppress_error=False) -> None:
        """ undifferences y to original level and drops all regressors (such as AR terms)
            suppress_error: bool, default False
                whether to suppress an error that gets raised if the series was never differenced
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

    def set_estimator(self, estimator) -> None:
        """ sets the estimator to forecast with
            estimator: one of _estimators_
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

    def ingest_grid(self, grid) -> None:
        """ ingests a grid to tune the estimator
            grid: dict or str
                if dict, must be a user-created grid
                if str, must match the name of a dict grid stored in Grids.py
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

    def limit_grid_size(self, n, random_seed=None) -> None:
        """ makes a grid smaller randomly
            n: int or float
                if int, randomly selects that many parameter combinations
                if float, must be less than 1 and greater 0, randomly selects that percentage of parameter combinations
            random_seed: int, optional
                set a seed to make results consistent
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

    def set_validation_metric(self, metric="rmse") -> None:
        """ sets the metric that will be used to tune all subsequent models
            not a good idea to change this if you are planning to combo model as weird things could happen
            metric: one of _metrics_, default 'rmse'
                the metric to optimize the models on using the validation set
        """
        descriptive_assert(
            metric in _metrics_,
            ValueError,
            f"metric must be one of {_metrics_}, got {metric}",
        )
        if (metric == "r2") & (self.validation_length < 2):
            raise ValueError(
                "can only validate with r2 if the validation length is at least 2, try set_validation_length()"
            )
        self.validation_metric = metric

    def tune(self, dynamic_tuning=False) -> None:
        """ tunes the specified estimator using an ingested grid (ingests a grid from Grids.py with same name as the estimator by default)
            any parameters you can pass as **kwargs to manual_forecast() can be tuned with this process
            dynamic_tuning: bool, default False
                whether to dynamically tune the forecast (meaning AR terms will be propogated with predicted values)
                setting this to False means faster performance, but gives a less-good indication of how well the forecast will perform out x amount of periods
                when False, metrics effectively become an average of one-step forecasts
        """
        if not hasattr(self, "grid"):
            try:
                self.ingest_grid(self.estimator)
            except SyntaxError:
                raise
            except:
                raise
                # raise ForecastError.NoGrid(f'to tune, a grid must be loaded. tried to load a grid called {self.estimator}, but either the Grids.py file could not be found in the current directory, there is no grid with that name, or the dictionary values are not list-like. try ingest_grid() with a dictionary grid passed manually.')

        if self.estimator in _cannot_be_tuned_:
            raise ForecastError(f"{self.estimator} models cannot be tuned")
            self.best_params = {}
            return

        metrics = []
        for i, v in self.grid.iterrows():
            try:
                if self.estimator in _sklearn_estimators_:
                    metrics.append(
                        getattr(self, "_forecast_sklearn")(
                            fcster=self.estimator,
                            tune=True,
                            dynamic_testing=dynamic_tuning,
                            **v,
                        )
                    )
                else:
                    metrics.append(
                        getattr(self, f"_forecast_{self.estimator}")(tune=True, **v)
                    )
            except TypeError:
                raise
            except Exception as e:
                self.grid.drop(i, axis=0, inplace=True)
                logging.warning(
                    f"could not evaluate the paramaters: {dict(v)}. error: {e}"
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
                self.best_params = dict(self.grid.loc[best_params_idx])
            else:
                best_params_idx = self.grid.loc[
                    self.grid_evaluated["metric_value"]
                    == self.grid_evaluated["metric_value"].min()
                ].index.to_list()[0]
                self.best_params = dict(self.grid.loc[best_params_idx])

            self.validation_metric_value = self.grid_evaluated.loc[
                best_params_idx, "metric_value"
            ]
        else:
            logging.warning(
                f"none of the keyword/value combos stored in the grid could be evaluated for the {self.estimator} model"
            )
            self.best_params = {}

        self.dynamic_tuning = dynamic_tuning

    def manual_forecast(self, call_me=None, dynamic_testing=True, **kwargs) -> None:
        """ manually forecasts with the hyperparameters, Xvars, and normalizer selection passed as keywoords
            call_me: str, optional
                what to call the model when storing it in the object's history dictionary
                if not specified, the model's nickname will be assigned the estimator value ('mlp' will be 'mlp', etc.)
                duplicated names will be overwritten with the most recently called model
            dynamic_testing: bool, default True
                whether to dynamically test the forecast (meaning AR terms will be propogated with predicted values)
                setting this to False means faster performance, but gives a less-good indication of how well the forecast will perform out x amount of periods
                when False, test-set metrics effectively become an average of one-step forecasts
            **kwargs are passed to the _forecast_{estimator}() method and can include such parameters as Xvars, normalizer, cap, and floor, in addition to any given model's specific hyperparameters
            **kwargs include for sklearn and lstm models:
                normalizer: one of {_normalizer_}, default 'minmax'
                    if not None, normalizer applied to training data only to not leak
                Xvars: list where all elements are in current_xreg keys, 'all', or None
                    if None and Xvars are required, None becomes equivalent to 'all'
        """
        descriptive_assert(
            isinstance(call_me, str) | (call_me is None),
            ValueError,
            "call_me must be a str type or None",
        )
        self.call_me = self.estimator if call_me is None else call_me
        self.forecast = (
            getattr(self, "_forecast_sklearn")(
                fcster=self.estimator, dynamic_testing=dynamic_testing, **kwargs
            )
            if self.estimator in _sklearn_estimators_
            else getattr(self, f"_forecast_{self.estimator}")(
                dynamic_testing=dynamic_testing, **kwargs
            )
        )
        self._bank_history(
            auto=False
            if not hasattr(self, "best_params")
            else len(self.best_params.keys()) > 0,
            **kwargs,
        )

    def auto_forecast(self, call_me=None, dynamic_testing=True) -> None:
        """ auto forecast with the best parameters indicated from the tuning process
            see manual_forecast() docstring 
        """
        if not hasattr(self, "best_params"):
            logging.warning(
                f"since tune() has not been called, {self.estimator} model will be run with default hyperparameters"
            )
            self.best_params = {}
        self.forecast = self.manual_forecast(
            call_me=call_me, dynamic_testing=dynamic_testing, **self.best_params
        )

    def tune_test_forecast(
        self,
        models,
        dynamic_tuning=False,
        dynamic_testing=True,
        summary_stats=False,
        feature_importance=False,
    ) -> None:
        """ iterates through a list of models, tunes them using grids in Grids.py, forecasts them, and can save feature information
            models: list-like
                each element must match an element in _estimators_ (except "combo", which cannot be tuned)
            dynamic_tuning: bool, default False
                whether to dynamically tune the forecast (meaning AR terms will be propogated with predicted values)
                setting this to False means faster performance, but gives a less-good indication of how well the forecast will perform out x amount of periods
                when False, metrics effectively become an average of one-step forecasts
            dynamic_testing: bool, default True
                whether to dynamically test the forecast (meaning AR terms will be propogated with predicted values)
                setting this to False means faster performance, but gives a less-good indication of how well the forecast will perform out x amount of periods
                when False, test-set metrics effectively become an average of one-step forecasts
            summary_stats: bool, default False
                whether to save summary stats for the models that offer those
            feature_importance: bool, default False
                whether to save permutation feature importance information for the models that offer those
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

    def save_feature_importance(self) -> None:
        """ save feature info for models that offer it
            will not raise errors if not available
        """
        import eli5
        from eli5.sklearn import PermutationImportance

        try:
            perm = PermutationImportance(self.regr).fit(
                self.X, self.y.values[-len(self.X) :]
            )
        except TypeError:
            logging.warning(
                f"cannot set feature importance on the {self.estimator} model"
            )
            return
        self.feature_importance = eli5.explain_weights_df(
            perm, feature_names=self.history[self.call_me]["Xvars"]
        ).set_index("feature")
        self._bank_fi_to_history()

    def save_summary_stats(self) -> None:
        """ save summary stats for models that offer it
            will not raise errors if not available
        """
        if not hasattr(self, "summary_stats"):
            logging.warning(
                f"last model run ({self.estimator}) does not have summary stats"
            )
            return
        self._bank_summary_stats_to_history()

    def keep_smaller_history(self, n) -> None:
        """ cuts the amount of observations in the object (trims the current_dates and current_xreg attributes as well)
            n: int, str in '%Y-%m-%d' format, or datetime object
                if int, the number of observations to keep
                otherwise, the last observation to keep 
        """
        if isinstance(n, str):
            n = datetime.datetime.strptime(n, "%Y-%m-%d")
        if (type(n) is datetime.datetime) or (type(n) is pd.Timestamp):
            n = len([i for i in self.current_dates if i >= n])
        descriptive_assert(
            isinstance(n, int),
            ValueError,
            "n must be an int, datetime object, or str in yyyy-mm-dd format and there must be more than 2 observations to keep",
        )
        descriptive_assert(
            n > 2,
            ValueError,
            "n must be an int, datetime object, or str in yyyy-mm-dd format and there must be more than 2 observations to keep",
        )
        self.y = self.y[-n:]
        self.current_dates = self.current_dates[-n:]
        for k, v in self.current_xreg.items():
            self.current_xreg[k] = v[-n:]

    def order_fcsts(self, models, determine_best_by="TestSetRMSE") -> list:
        """ returns a list of estiamated forecasts from best-to-worst
            models: list-like
                each element must match an element in _estimators_ (except "combo", which cannot be tuned)
            determine_best_by: one of _determine_best_by_
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
        """returns a lit of regressor names stored in the object"""
        return [k for k in self.current_xreg.keys()]

    def get_freq(self) -> str:
        """returns the pandas inferred date frequency"""
        return self.freq

    def validate_regressor_names(self) -> None:
        """validates that all regressor names exist in both current_xregs and future_xregs"""
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
        to_png=False,
        out_path="./",
        png_name="plot.png",
        ci=False,
    ) -> None:
        """ plots all forecasts with the actuals, or just actuals if no forecasts available
            models: list-like, str, or None; default 'all'
               the forecated models to plot
               can start with "top_" and the metric specified in order_by will be used to order the models appropriately
               if None or models/order_by combo invalid, will plot only actual values
            order_by: one of _determine_best_by_, default None
            level: bool, default False
                if True, will always plot level forecasts
                if False, will plot the forecasts at whatever level they were called on
                if False and there are a mix of models passed with different integrations, will default to True
            print_attr: list-like, default []
                attributes from history dict to print to console
                if the attribute doesn't exist for a passed model, will not raise error, will just skip that element
            to_png: bool, default False
                whether to save the resulting image to a png file
            out_path: str, default './'
                the path to save the png file to (ignored when `to_png=False`)
            png_name: str, default './plot.png'
                the name of the resulting png image (ignored when `to_png=False`)
            ci: bool, default False
                whether to display the confidence intervals
                default is 100 boostrapped samples and a 95% confidence interval
                change defaults by calling `set_cilevel()` and `set_bootstrapped_samples()` before forecasting
                ignored when level = False
        """
        try:
            models = self._parse_models(models, order_by)
        except (ValueError, TypeError):
            models = None

        if models is None:
            sns.lineplot(x=self.current_dates.values, y=self.y.values, label="actuals")
            plt.legend()
            plt.xlabel("Date")
            plt.ylabel("Values")
            plt.title("Plot of y Vals")
            plt.show()
            return

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

        print_attr_map = {}
        sns.lineplot(
            x=plot["date"][-plot["actuals_len"] :],
            y=plot["actuals"][-plot["actuals_len"] :],
            label="actuals",
        )
        for i, m in enumerate(models):
            plot[m] = (
                self.history[m]["Forecast"]
                if not level
                else self.history[m]["LevelForecast"]
            )
            sns.lineplot(
                x=self.future_dates.to_list(), y=plot[m], color=_colors_[i], label=m
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
        plt.title("Forecast Results")
        if to_png:
            plt.savefig(os.path.join(out_path, png_name))
        plt.show()

    def plot_test_set(
        self,
        models="all",
        order_by=None,
        include_train=True,
        level=False,
        to_png=False,
        out_path="./",
        png_name="./plot.png",
        ci=False,
    ) -> None:
        """ plots all test-set predictions with the actuals
            models: list-like or str, default 'all'
               the forecated models to plot
               can start with "top_" and the metric specified in order_by will be used to order the models appropriately
            order_by: one of _determine_best_by_, default None
            include_train: bool or int, default True
                use to zoom into training results
                if True, plots the test results with the entire history in y
                if False, matches y history to test results and only plots this
                if int, plots that length of y to match to test results
            level: bool, default False
                if True, will always plot level forecasts
                if False, will plot the forecasts at whatever level they were called on
                if False and there are a mix of models passed with different integrations, will default to True
            to_png: bool, default False
                whether to save the resulting image to a png file
            out_path: str, default './'
                the path to save the png file to (ignored when `to_png=False`)
            png_name: str, default './plot.png'
                the name of the resulting png image (ignored when `to_png=False`)
            ci: bool, default False
                whether to display the confidence intervals
                default is 100 boostrapped samples and a 95% confidence interval
                change defaults by calling `set_cilevel()` and `set_bootstrapped_samples()` before forecasting
                ignored when level = False
        """
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
        plt.title("Test Set Results")
        if to_png:
            plt.savefig(os.path.join(out_path, png_name))
        plt.show()

    def plot_fitted(
        self,
        models="all",
        order_by=None,
        to_png=False,
        out_path="./",
        png_name="./plot.png",
    ) -> None:
        """ plots all fitted values with the actuals
            models: list-like or str, default 'all'
               the forecated models to plot
               can start with "top_" and the metric specified in order_by will be used to order the models appropriately
            order_by: one of _determine_best_by_, default None
            to_png: bool, default False
                whether to save the resulting image to a png file
            out_path: str, default './'
                the path to save the png file to (ignored when `to_png=False`)
            png_name: str, default './plot.png'
                the name of the resulting png image (ignored when `to_png=False`)
        """
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
        sns.lineplot(x=plot["date"], y=plot["actuals"], label="actuals")

        for i, m in enumerate(models):
            plot[m] = self.history[m]["FittedVals"]
            sns.lineplot(
                x=plot["date"][-len(plot[m]) :],
                y=plot[m][-len(plot["date"]) :],
                linestyle="--",
                color=_colors_[i],
                alpha=0.7,
                label=m,
            )

        plt.legend()
        plt.xlabel("Date")
        plt.ylabel("Values")
        plt.title("Fitted Values")
        if to_png:
            plt.savefig(os.path.join(out_path, png_name))
        plt.show()

    def drop_regressors(self, *args) -> None:
        """ drops regressors
            *args are names of regressors to drop
        """
        for a in args:
            self.current_xreg.pop(a)
            self.future_xreg.pop(a)

    def drop_Xvars(self, *args):
        """ drops regressors
            *args are names of regressors to drop
        """
        self.drop_regressors(*args)

    def pop(self, *args) -> None:
        """ deletes evaluated forecasts from the history dictionary
            *args names of models matching what was passed to call_me (default for call_me in a given model is the same as the estimator name)
        """
        for a in args:
            self.history.pop(a)

    def pop_using_criterion(self, metric, evaluated_as, threshold, delete_all=True):
        """ deletes all forecasts from history that meet a given criterion
            metric: str, one of _determine_best_by_ + ['AnyPrediction','AnyLevelPrediction']
            evaluated_as: str, one of {"<","<=",">",">=","=="}
            threshold: float
            delete_all: bool, default True
                if the passed criterion deletes all forecasts, whether to actually delete all forecasts
                if False and all forecasts meet criterion, will keep them all
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
        self.pop(*fcsts) if len(fcsts) < len(
            self.history.keys()
        ) or delete_all else None

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
        """ exports 1-all of 5 pandas dataframes, can write to excel with each dataframe on a separate sheet
            will return either a dictionary with dataframes as values (df str arguments as keys) or a single dataframe if only one df is specified
            dfs: list-like or str, default ['all_fcsts','model_summaries','best_fcst','test_set_predictions','lvl_fcsts']
                a list or name of the specific dataframe(s) you want returned and/or written to excel
                must be one of or multiple of default
            models: list-like or str, default 'all'
                the models to write information for
                can start with "top_" and the metric specified in `determine_best_by` will be used to order the models appropriately
            best_model: str, default 'auto'
                the name of the best model, if "auto", will determine this by the metric in determine_best_by
                if not "auto", must match a model nickname of an already-evaluated model
            determine_best_by: one of _determine_best_by_, default 'TestSetRMSE'
            to_excel: bool, default False
                whether to save to excel
            out_path: str, default './'
                the path to save the excel file to (ignored when `to_excel=False`)
            excel_name: str, default 'results.xlsx'
                the name to call the excel file (ignored when `to_excel=False`)
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
                "univariate",
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
        """ exports the summary stats from a model
            raises an error if you never saved the model's summary stats
            model: str
                the name of them model to export for, matches what was passed to call_me when calling the forecast (default is estimator name)
        """
        return self.history[model]["summary_stats"]

    def export_feature_importance(self, model) -> pd.DataFrame:
        """ exports the feature importance from a model
            raises an error if you never saved the model's feature importance
            model: str
                the name of them model to export for, matches what was passed to call_me when calling the forecast (default is estimator name)
        """
        return self.history[model]["feature_importance"]

    def export_validation_grid(self, model) -> pd.DataFrame:
        """ exports the validation from a model
            raises an error if you never tuned the model
            model: str
                the name of them model to export for, matches what was passed to call_me when calling the forecast (default is estimator name)
        """
        return self.history[model]["grid_evaluated"]

    def all_feature_info_to_excel(
        self, out_path="./", excel_name="feature_info.xlsx"
    ) -> None:
        """ saves all feature importance and summary stats to excel
            each model where such info is available for gets its own tab
            be sure to call save_summary_stats() and save_feature_importance() before using this function
            out_path: str, default './'
                the path to export to
            excel_name: str, default 'feature_info.xlsx'
                the name of the resulting excel file
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
    ) -> None:
        """ saves all validation grids to excel
            each model where such info is available for gets its own tab
            out_path: str, default './'
                the path to export to
            excel_name: str, default 'feature_info.xlsx'
                the name of the resulting excel file
            sort_by_metric_value: bool, default False
            ascending: bool, default True
                whether to sort least-to-greatest
                ignored if sort_by_metric_value is False
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

    def reset(self) -> None:
        """ drops all regressors and reverts object to original (level) state when initiated
        """
        self.undiff(suppress_error=True)
        self.current_xreg = {}
        self.future_xreg = {}

    def export_Xvars_df(self, dropna=False):
        """ returns a pandas dataframe with all utilized regressors and values
            dropna: bool, default False
                whether to drop null values from the resulting dataframe
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
        """ exports a single dataframe with forecasts and upper and lower forecast bounds
            model: str
                the model nickname (must exist in history.keys())
        """
        return pd.DataFrame(
            {
                "DATE": self.future_dates.to_list(),
                "UpperForecast": self.history[model]["UpperCI"],
                "Forecast": self.history[model]["Forecast"],
                "LowerForecast": self.history[model]["LowerCI"],
                "ModelNickname": [model] * len(self.future_dates),
            }
        )

    def export_test_set_preds_with_cis(self, model):
        """ exports a single dataframe with test-set predictions, actuals, and upper and lower prediction bounds
            model: str
                the model nickname (must exist in history.keys())
        """
        return pd.DataFrame(
            {
                "DATE": self.current_dates.to_list()[
                    -len(self.history[model]["TestSetPredictions"]) :
                ],
                "UpperPreds": self.history[model]["TestSetUpperCI"],
                "Preds": self.history[model]["TestSetPredictions"],
                "Actuals": self.history[model]["TestSetActuals"],
                "LowerPreds": self.history[model]["TestSetLowerCI"],
                "ModelNickname": [model]
                * len(self.history[model]["TestSetPredictions"]),
            }
        )

    def export_fitted_vals(self,model):
        """ exports a single dataframe with fitted values and actuals
            model: str
                the model nickname (must exist in history.keys())
        """
        return pd.DataFrame(
            {
                "DATE": self.current_dates.to_list()[-len(self.history[model]["FittedVals"]):],
                "Actuals": self.y.to_list()[-len(self.history[model]["FittedVals"]):],
                "FittedVals": self.history[model]["FittedVals"]
            }
        )
