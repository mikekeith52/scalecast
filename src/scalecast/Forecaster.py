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
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_percentage_error,
    mean_absolute_error,
)
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

class Forecaster:
    def __init__(
        self, 
        y, 
        current_dates, 
        require_future_dates=True, 
        future_dates=None,
        test_length = 1, 
        **kwargs
    ):
        """ 
        Args:
            y (collection): An array of all observed values.
            current_dates (collection): An array of all observed dates.
                Must be same length as y and in the same sequence.
                Can pass any numerical index if dates are unknown; in this case, 
                It will act as if dates are in nanosecond frequency.
            require_future_dates (bool): Default True.
                If False, none of the models will forecast into future periods by default.
                If True, all models will forecast into future periods, 
                unless run with test_only = True. 
                When adding regressors, they will automatically be added into future periods.
            future_dates (int): Optional. The future dates to add to the model upon initialization.
                if not added when object is initialized, can be added later.
            test_length (int): Default 1. The test length that all models will use to test all models out of sample.
                Change to 0 to skip testing models. In a future version of scalecast, the default value will be changed to 0, but it
                is left 1 for now to maintain the behavior of Forecaster objects. Before version 0.16.0, skipping model
                testing was impossible.
            **kwargs: Become attributes. Life hack: specify the `test_length` arg here. 
                To skip model testing, pass `test_length = 0`.
        """
        self.y = y
        self.current_dates = current_dates
        self.require_future_dates = require_future_dates
        self.future_dates = pd.Series([], dtype="datetime64[ns]")
        self.current_xreg = (
            {}
        )  # values should be pandas series (to make differencing work more easily)
        self.future_xreg = (
            {}
        )  # values should be lists (to make iterative forecasting work more easily)
        self.history = {}
        self.test_length = test_length
        self.validation_length = 1
        self.validation_metric = "rmse"
        self.integration = 0
        self.levely = list(y)
        self.init_dates = list(current_dates)
        self.cilevel = 0.95
        self.bootstrap_samples = 100
        self.grids_file = 'Grids'
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.typ_set()  # ensures that the passed values are the right types

        if not require_future_dates:
            self.generate_future_dates(1)  # placeholder -- never used
        if future_dates is not None:
            self.generate_future_dates(future_dates)

        globals()["f_init_"] = self.__deepcopy__()

    def __copy__(self):
        if hasattr(self,'tf_model'):
            delattr(self,'tf_model')
        obj = type(self).__new__(self.__class__)
        obj.__dict__.update(self.__dict__)
        return obj

    def __deepcopy__(self):
        if hasattr(self,'tf_model'):
            delattr(self,'tf_model')
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
    GridsFile={}
)""".format(
            self.current_dates.values[0].astype(str),
            self.current_dates.values[-1].astype(str),
            self.freq,
            len(self.y),
            len(self.future_dates) if self.require_future_dates else "NA",
            list(self.current_xreg.keys()),
            self.integration,
            self.test_length,
            self.validation_length,
            self.validation_metric,
            list(self.history.keys()),
            self.cilevel,
            self.bootstrap_samples,
            None if not hasattr(self, "estimator") else self.estimator,
            self.grids_file,
        )

    def _split_data(self, X, y, test_length, tune):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_length, shuffle=False
        )
        if tune and self.test_length > 0:
            X_test, y_test = (X_test[: -self.test_length], y_test[: -self.test_length])
        return X_train, X_test, y_train, y_test

    def _validate_no_test_only(self, models):
        descriptive_assert(
            max([int(self.history[m]["TestOnly"]) for m in models]) == 0,
            ForecastError,
            "this method does not accept any models run test_only = True or when require_future_dates attr is False",
        )

    # this is used across a few different models
    def _prepare_model_data(self,Xvars, y, current_xreg):
        if Xvars is None or Xvars == "all":
            Xvars = [x for x in current_xreg.keys()]
        elif isinstance(Xvars, str):
            Xvars = [Xvars]

        y = [i for i in y]
        X = pd.DataFrame(current_xreg)
        X = X[Xvars].values
        return Xvars, y, X

    def _validate_future_dates_exist(self):
        """ makes sure future periods have been specified before adding regressors
        """
        descriptive_assert(
            len(self.future_dates) > 0,
            ForecastError,
            "before adding regressors, please make sure you have generated future dates by calling generate_future_dates(), set_last_future_date(), or ingest_Xvars_df(use_future_dates=True)",
        )

    def _find_cis(self, y, fvs):
        """ bootstraps the upper and lower forecast estimates using the info stored in cilevel and bootstrap_samples
        """
        random.seed(20)
        resids = [fv - ac for fv, ac in zip(fvs, y[-len(fvs) :])]
        bootstrapped_resids = np.random.choice(resids, size=self.bootstrap_samples)
        bootstrap_mean = np.mean(bootstrapped_resids)
        bootstrap_std = np.std(bootstrapped_resids)
        return _set_ci_step(
            f = self,
            s = bootstrap_std,
        ) + bootstrap_mean
        #return (
        #    stats.norm.ppf(1 - (1 - self.cilevel) / 2) * bootstrap_std + bootstrap_mean
        #)

    def _bank_history(self, **kwargs):
        """ places all relevant information from the last evaluated forecast into the history dictionary attribute
            **kwargs: passed from each model, depending on how that model uses Xvars, normalizer, and other args
        """
        # since test only, what gets saved to history is relevant to the train set only, the genesis of the next line
        y_use = (
            self.y.values.copy()
            if not self.test_only
            else self.y.values[: -self.test_length].copy()
        )
        fvs_use = (
            self.fitted_values[:]
            if not self.test_only
            else self.fitted_values[: -self.test_length]
        )
        call_me = self.call_me
        # originally, the ci_range used object attributes only, but test_only makes using args necessary
        ci_range = self._find_cis(y_use, fvs_use)
        self.history[call_me] = {
            "Estimator": self.estimator,
            "TestOnly": self.test_only,
            "Xvars": self.Xvars,
            "HyperParams": {
                k: v for k, v in kwargs.items() if k not in _not_hyperparams_
            },
            "Scaler": (
                None
                if self.estimator not in ["rnn", "lstm"] + _sklearn_estimators_
                else "minmax"
                if "normalizer" not in kwargs
                else kwargs["normalizer"]
            ),
            "Observations": len(y_use),
            "Forecast": self.forecast[:],
            "UpperCI": [f + ci_range for f in self.forecast],
            "LowerCI": [f - ci_range for f in self.forecast],
            "FittedVals": self.fitted_values[:],
            "Tuned": hasattr(self, "best_params"),
            "CrossValidated": (
                False
                if not hasattr(self, "best_params")
                else False
                if not hasattr(self, "grid_evaluated")
                else True
                if "fold" in self.grid_evaluated.columns
                else False
            ),
            "DynamicallyTested": self.dynamic_testing,
            "Integration": self.integration,
            "TestSetLength": self.test_length,
            "TestSetRMSE": self.rmse,
            "TestSetMAPE": self.mape,
            "TestSetMAE": self.mae,
            "TestSetR2": self.r2,
            "TestSetPredictions": self.test_set_pred[:],
            "TestSetUpperCI": [
                f + ci_range for f in self.test_set_pred
            ],  # not exactly right, but close enough with caveat
            "TestSetLowerCI": [
                f - ci_range for f in self.test_set_pred
            ],  # not exactly right, but close enough with caveat
            "TestSetActuals": self.test_set_actuals[:],
            "InSampleRMSE": _return_na_if_len_zero(y_use[-len(fvs_use) :], fvs_use, rmse),
            "InSampleMAPE": _return_na_if_len_zero(y_use[-len(fvs_use) :], fvs_use, mape),
            "InSampleMAE": _return_na_if_len_zero(y_use[-len(fvs_use) :], fvs_use, mae),
            "InSampleR2": _return_na_if_len_zero(y_use[-len(fvs_use) :], fvs_use,r2),
            "CILevel": self.cilevel,
            "CIPlusMinus": ci_range,
        }

        if hasattr(self, "best_params"):
            self.history[call_me]["ValidationSetLength"] = (
                self.validation_length
                if "fold" not in self.grid_evaluated.columns
                else None
            )
            self.history[call_me]["ValidationMetric"] = self.validation_metric
            self.history[call_me][
                "ValidationMetricValue"
            ] = self.validation_metric_value

        for attr in ("grid_evaluated", "models", "weights"):
            if hasattr(self, attr):
                self.history[call_me][attr] = getattr(self, attr)

        self.history[call_me]["LevelY"] = self.levely[:]
        
        if self.integration == 1:
            integration = self.integration

            fcst = self.forecast[::-1]
            pred = self.history[call_me]["TestSetPredictions"][::-1]
            fvs = fvs_use[:]

            fcst.append(self.levely[-1])
            pred.append(self.levely[-(len(pred) + 1)])
            fvs.insert(0, self.levely[-len(fvs) - 1])

            fcst = list(np.cumsum(fcst[::-1]))[1:]
            pred = list(np.cumsum(pred[::-1]))[1:]
            fvs = list(np.cumsum(fvs))[1:]

            ci_range = self._find_cis(self.levely[-len(fvs) :], fvs)
            self.history[call_me]["LevelForecast"] = fcst[:]
            self.history[call_me]["LevelTestSetPreds"] = pred[:]
            self.history[call_me]["LevelTestSetRMSE"] = _return_na_if_len_zero(
                self.levely[-len(pred) :], pred, rmse
            )
            self.history[call_me]["LevelTestSetMAPE"] = _return_na_if_len_zero(
                self.levely[-len(pred) :], pred, mape
            )
            self.history[call_me]["LevelTestSetMAE"] = _return_na_if_len_zero(
                self.levely[-len(pred) :], pred, mae
            )
            self.history[call_me]["LevelTestSetR2"] = _return_na_if_len_zero(
                self.levely[-len(pred) :], pred, r2
            )
            self.history[call_me]["LevelFittedVals"] = fvs[:]
            self.history[call_me]["LevelInSampleRMSE"] = _return_na_if_len_zero(
                self.levely[-len(fvs) :], fvs, rmse
            )
            self.history[call_me]["LevelInSampleMAPE"] = _return_na_if_len_zero(
                self.levely[-len(fvs) :], fvs, mape
            )
            self.history[call_me]["LevelInSampleMAE"] = _return_na_if_len_zero(
                self.levely[-len(fvs) :], fvs, mae
            )
            self.history[call_me]["LevelInSampleR2"] = _return_na_if_len_zero(
                self.levely[-len(fvs) :], fvs, r2
            )
            self.history[call_me]["LevelUpperCI"] = [f + ci_range for f in fcst]
            self.history[call_me]["LevelLowerCI"] = [f - ci_range for f in fcst]
            self.history[call_me]["LevelTSUpperCI"] = [f + ci_range for f in pred]
            self.history[call_me]["LevelTSLowerCI"] = [f - ci_range for f in pred]
        else:  # better to have these attributes populated for all series
            self.history[call_me]["LevelForecast"] = self.forecast[:]
            self.history[call_me]["LevelTestSetPreds"] = self.test_set_pred[:]
            self.history[call_me]["LevelTestSetRMSE"] = self.rmse
            self.history[call_me]["LevelTestSetMAPE"] = self.mape
            self.history[call_me]["LevelTestSetMAE"] = self.mae
            self.history[call_me]["LevelTestSetR2"] = self.r2
            self.history[call_me]["LevelFittedVals"] = self.history[call_me][
                "FittedVals"
            ][:]
            self.history[call_me]["LevelInSampleRMSE"] = self.history[call_me][
                "InSampleRMSE"
            ]
            self.history[call_me]["LevelInSampleMAPE"] = self.history[call_me][
                "InSampleMAPE"
            ]
            self.history[call_me]["LevelInSampleMAE"] = self.history[call_me][
                "InSampleMAE"
            ]
            self.history[call_me]["LevelInSampleR2"] = self.history[call_me][
                "InSampleR2"
            ]
            self.history[call_me]["LevelUpperCI"] = self.history[call_me]["UpperCI"][:]
            self.history[call_me]["LevelLowerCI"] = self.history[call_me]["LowerCI"][:]
            self.history[call_me]["LevelTSUpperCI"] = self.history[call_me][
                "TestSetUpperCI"
            ][:]
            self.history[call_me]["LevelTSLowerCI"] = self.history[call_me][
                "TestSetLowerCI"
            ][:]

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
        self.rmse = _return_na_if_len_zero(y, pred, rmse)
        self.r2 = _return_na_if_len_zero(y, pred, r2)
        self.mae = _return_na_if_len_zero(y, pred, mae)
        self.mape = _return_na_if_len_zero(y, pred, mape)

    def _tune(self):
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
            "tf_model",
        ):
            try:
                delattr(self, attr)
            except AttributeError:
                pass

    def _raise_test_only_error(self):
        raise ValueError('test_only cannot be True if test_length is 0.')

    def _warn_about_dynamic_testing(self,dynamic_testing,does_not_use_lags=False,uses_direct=False):
        if dynamic_testing is not True:
            warning = f'The dynamic_testing arg is always set to True for the {self.estimator} model.'
            warning += " This model doesn't use lags to make predictions." if does_not_use_lags else ''
            warning += " This model uses direct forecasting with lags." if uses_direct else ''
            warnings.warn(warning)
            self.dynamic_testing = True

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
        """ Runs an sklearn forecast start-to-finish.
        See the example: https://scalecast-examples.readthedocs.io/en/latest/sklearn/sklearn.html.

        Args:
            fcster (str): One of _sklearn_estimators_. Reads the estimator passed to the estimator attribute.
            dynamic_testing (bool or int):
                Whether to dynamically/recursively test the forecast (meaning AR terms will be propagated with predicted values).
                If True, evaluates recursively over the entire out-of-sample slice of data.
                If int, window evaluates over that many steps (2 for 2-step recurvie testing, 12 for 12-step, etc.).
                Setting this to False or 1 means faster performance, 
                but gives a less-good indication of how well the forecast will perform more than one period out.
            tune (bool): Default False.
                Whether the model is being tuned.
                It should not be specified by the user.
            Xvars (list-like, str, or None): The regressors to predict with.
                Be sure to have added them to the Forecaster object first.
                None means all Xvars added to the Forecaster object will be used for sklearn estimators (not so for other estimators).
            normalizer (str): The scaling technique to apply to the data. One of _normalizer_. 
                Default 'minmax'.
                If not None and a test length is specified greater than 0, the normalizer is fit on the training data only.
            test_only (bool): Default False.
                Whether to stop the model after the testing process and not forecast into future periods.
                The forecast info stored in the object's history will be equivalent to test-set predictions.
                When True, any plot or export of forecasts into a future horizon will fail 
                and not all methods will raise descriptive errors.
                Will automatically change to True if object was initiated with require_future_dates = False.
            **kwargs: Treated as model hyperparameters and passed to the applicable sklearn estimator.
        """
        def fit_normalizer(X, normalizer):
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
            if (
                (
                    not [x for x in Xvars if x.startswith("AR")]
                ) | (
                    dynamic_testing is False
                )
            ):
                p = pd.DataFrame(future_xreg).values[:fcst_horizon]
                p = scale(scaler, p)
                return (regr.predict(p),regr.predict(X),Xvars,regr)
            # otherwise, use a dynamic process to propagate out-of-sample AR terms with predictions (slower but more indicative of a true forecast performance)
            fcst = []
            actuals = {k:list(v)[:] for k, v in future_xreg.items() if k.startswith('AR')}
            for i in range(fcst_horizon):
                p = pd.DataFrame({x: [future_xreg[x][i]] for x in Xvars}).values
                p = scale(scaler, p)
                fcst.append(regr.predict(p)[0])
                if not i == (fcst_horizon - 1):
                    for k, v in future_xreg.items():
                        if k.startswith("AR"):
                            ar = int(k[2:])
                            idx = i + 1 - ar
                            # full dynamic horizon
                            if dynamic_testing is True:
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
                            # window forecasting
                            else:
                                if (i+1) % dynamic_testing  == 0:
                                    future_xreg[k][:(i + 1)] = actuals[k][:(i + 1)]
                                elif idx > -1:
                                    future_xreg[k][i + 1] = fcst[idx]
                                else:
                                    future_xreg[k][i + 1] = y[idx]

            return (fcst,regr.predict(X),Xvars,regr)

        descriptive_assert(
            len(self.current_xreg.keys()) > 0,
            ForecastError,
            f"Need at least 1 Xvar to forecast with the {self.estimator} model.",
        )
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
        )  # 1-step dynamic testing is same as no dynamic testing
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
        Xvars, y, X = self._prepare_model_data(Xvars, y, current_xreg)
        regr = _sklearn_imports_[fcster](**kwargs)
        # train/test the model
        if self.test_length > 0 or tune:
            test_length = (
                self.test_length if not tune 
                else self.validation_length + self.test_length
            )
            X_train, X_test, y_train, y_test = self._split_data(X, y, test_length, tune)
            scaler = fit_normalizer(X_train, normalizer)
            result = evaluate_model(
                scaler=scaler,
                regr=regr,
                X=X_train,
                y=y_train,
                Xvars=Xvars,
                fcst_horizon=test_length - self.test_length if tune else test_length,  # fcst_horizon
                future_xreg={x: current_xreg[x][-test_length:] for x in Xvars},  # for AR processing
                dynamic_testing=dynamic_testing,
                true_forecast=False,
            )
            pred = [i for i in result[0]]
            # set the test-set metrics
            self._metrics(y_test, pred)
            if tune:
                return self._tune()
            elif test_only:
                return (
                    result[0],
                    [i for i in result[1]] + [np.nan] * self.test_length,
                    result[2],
                    result[3],
                )
        elif test_only:
            self._raise_test_only_error()
        else:
            scaler = fit_normalizer(X, normalizer)
            self._metrics([],[])
        return evaluate_model(
            scaler=scaler,
            regr=regr,
            X=X,
            y=y,
            Xvars=Xvars,
            fcst_horizon=len(self.future_dates),
            future_xreg={x: self.future_xreg[x][:] for x in Xvars},
            dynamic_testing=True,
            true_forecast=True,
        )

    def _forecast_theta(
        self, tune=False, dynamic_testing=True, test_only=False, **kwargs
    ):
        """ Forecasts with Four Theta from darts.
        See the example: https://scalecast-examples.readthedocs.io/en/latest/theta/theta.html.

        Args:
            tune (bool): Default False.
                Whether the model is being tuned.
                It does not need to be specified by user.
            dynamic_testing (bool): Default True.
                Always set to True for theta like all scalecast models that don't use lags.
            test_only (bool): Default False.
                Whether to stop the model after the testing process and not forecast into future periods.
                The forecast info stored in the object's history will be equivalent to test-set predictions.
                When True, any plot or export of forecasts into a future horizon will fail 
                and not all methods will raise descriptive errors.
                Will automatically change to True if object was initiated with require_future_dates = False.
            **kwargs: passed to the FourTheta() function from darts.
                See https://unit8co.github.io/darts/generated_api/darts.models.forecasting.theta.html.
        """
        from darts import TimeSeries
        from darts.models.forecasting.theta import FourTheta

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

        if self.test_length > 0 or tune:
            test_length = (
                self.test_length if not tune 
                else self.validation_length + self.test_length
            )
            y_train = pd.Series(
                y[:-test_length], 
                index=d[:-test_length],
            )
            y_test = y[-test_length:]
            y_test = y_test[: -self.test_length] if tune and self.test_length > 0 else y_test
            ts_train = TimeSeries.from_series(y_train)
            regr = FourTheta(**kwargs)
            regr.fit(ts_train)
            pred = regr.predict(len(y_test))
            pred = [p[0] for p in pred.values()]
            self._metrics(y_test, pred)
            if tune:
                return self._tune()
            elif test_only:
                resid = [r[0] for r in regr.residuals(ts_train).values()]
                actuals = y_train[-len(resid) :]
                fvs = [r + a for r, a in zip(resid, actuals)]
                return (
                    pred,
                    fvs + [np.nan] * self.test_length,
                    None,
                    None,
                    # regr, # I would like to include this in the future but it breaks things for now
                )
        elif test_only:
            self._raise_test_only_error()
        else:
            self._metrics([], [])
        regr = FourTheta(**kwargs)
        regr.fit(ts)
        pred = regr.predict(len(self.future_dates))
        pred = [p[0] for p in pred.values()]
        resid = [r[0] for r in regr.residuals(ts).values()]
        actuals = y[-len(resid) :]
        fvs = [r + a for r, a in zip(resid, actuals)]
        return (pred, fvs, None, None)

    def _forecast_hwes(
        self, tune=False, dynamic_testing=True, test_only=False, **kwargs
    ):
        """ Forecasts with Holt-Winters exponential smoothing.
        See the example: https://scalecast-examples.readthedocs.io/en/latest/hwes/hwes.html.

        Args:
            tune (bool): Default False.
                Whether the model is being tuned.
                It does not need to be specified by the user.
            dynamic_testing (bool): Default True.
                Always set to True for HWES like all scalecast models that don't use lags.
            test_only (bool): Default False.
                Whether to stop the model after the testing process and not forecast into future periods.
                The forecast info stored in the object's history will be equivalent to test-set predictions.
                When True, any plot or export of forecasts into a future horizon will fail 
                and not all methods will raise descriptive errors.
                Will automatically change to True if object was initiated with require_future_dates = False.
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
        kwargs = {k:bool(v) if isinstance(v,np.bool_) else v for k, v in kwargs.items()} # issue #19
        if self.test_length > 0 or tune:
            test_length = (
                self.test_length if not tune 
                else self.validation_length + self.test_length
            )
            y_train = y[:-test_length] if self.test_length > 0 else y[:]
            y_test = y[-test_length:]
            regr = HWES(
                y_train,
                dates=self.current_dates.values[: -test_length],
                freq=self.freq,
                **kwargs,
            ).fit(optimized=True, use_brute=True)
            pred = regr.predict(
                start=len(y_train),
                end=(
                    len(y_train) + (
                    len(y_test) if not tune 
                    else len(y_test) - self.test_length
                    ) - 1
                ),
            )
            self._metrics(y_test if not tune or self.test_length == 0 else y_test[: -self.test_length], pred)
            if tune:
                return self._tune()
            elif test_only:
                return (
                    pred,
                    list(regr.fittedvalues) + [np.nan] * self.test_length,
                    None,
                    regr,
                )
        elif test_only:
            self._raise_test_only_error()
        else:
            self._metrics([],[])
        regr = HWES(self.y, dates=self.current_dates, freq=self.freq, **kwargs).fit(
            optimized=True, use_brute=True
        )
        pred = regr.predict(start=len(y), end=len(y) + len(self.future_dates) - 1)
        return (pred, regr.fittedvalues, None, regr)

    def _forecast_arima(
        self, tune=False, Xvars=None, dynamic_testing=True, test_only=False, **kwargs
    ):
        """ Forecasts with ARIMA (or AR, ARMA, SARIMA, SARIMAX).
        See the example: https://scalecast-examples.readthedocs.io/en/latest/arima/arima.html.

        Args:
            tune (bool): Default False.
                Whether the model is being tuned.
                It does not need to be specified by user.
            Xvars (list-like, str, or None): Default None. The regressors to predict with.
                None means no Xvars used (unlike sklearn models).
            dynamic_testing (bool): Default True.
                Always ignored in ARIMA - ARIMA in scalecast dynamically tests all models over the full forecast horizon using statsmodels.
            test_only (bool): Default False.
                Whether to stop the model after the testing process and not forecast into future periods.
                The forecast info stored in the object's history will be equivalent to test-set predictions.
                When True, any plot or export of forecasts into a future horizon will fail 
                and not all methods will raise descriptive errors.
                Will automatically change to True if object was initiated with require_future_dates = False.
            **kwargs: Passed to the ARIMA() function from statsmodels. `endog` and `exog` are passed automatically. 
                See https://www.statsmodels.org/devel/generated/statsmodels.tsa.arima.model.ARIMA.html.
        """
        from statsmodels.tsa.arima.model import ARIMA
        self._warn_about_dynamic_testing(dynamic_testing=dynamic_testing)
        Xvars = (
            [x for x in self.current_xreg.keys() if not x.startswith("AR")]
            if Xvars == "all"
            else [x for x in Xvars if not x.startswith("AR")]
            if Xvars is not None
            else Xvars
        )
        Xvars_orig = None if Xvars is None else None if not Xvars else Xvars
        Xvars, y, X = self._prepare_model_data(Xvars, self.y, self.current_xreg)
        if self.test_length > 0 or tune:
            test_length = (
                self.test_length if not tune else self.validation_length + self.test_length
            )
            if Xvars_orig is None:
                X_train, X_test = None, None
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
            if tune:
                return self._tune()
            elif test_only:
                return (
                    pred,
                    list(regr.fittedvalues) + [np.nan] * self.test_length,
                    Xvars,
                    regr,
                )
        elif test_only:
            self._raise_test_only_error()
        else:
            self._metrics([],[])
        Xvars = None if Xvars_orig is None else Xvars
        X = None if Xvars_orig is None else X
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
        return (fcst, list(regr.fittedvalues), Xvars, regr)

    def _forecast_prophet(
        self,
        tune=False,
        Xvars=None,
        dynamic_testing=True,
        cap=None,
        floor=None,
        test_only=False,
        **kwargs,
    ):
        """ Forecasts with the Prophet model from the prophet library.
        See example: https://scalecast-examples.readthedocs.io/en/latest/prophet/prophet.html.

        Args:
            tune (bool): Default False.
                Whether to tune the forecast.
                Does not need to be specified by user.
            Xvars (list-like, str, or None): Default None. The regressors to predict with.
                None means no Xvars used (unlike sklearn models).
            dynamic_testing (bool): Default True.
                Always set to True for Prophet like all scalecast models that don't use lags.
            cap (float): Optional.
                Specific to Prophet when using logistic growth -- the largest amount the model is allowed to evaluate to.
            floor (float): Optional.
                Specific to Prophet when using logistic growth -- the smallest amount the model is allowed to evaluate to.
            test_only (bool): Default False.
                Whether to stop the model after the testing process and not forecast into future periods.
                The forecast info stored in the object's history will be equivalent to test-set predictions.
                When True, any plot or export of forecasts into a future horizon will fail 
                and not all methods will raise descriptive errors.
                Will automatically change to True if object was initiated with require_future_dates = False.
            **kwargs: Passed to the Prophet() function from prophet. 
                See https://facebook.github.io/prophet/docs/quick_start.html#python-api.
        """
        from prophet import Prophet
        self._warn_about_dynamic_testing(
            dynamic_testing=dynamic_testing,
            does_not_use_lags=True,
        )
        
        X = pd.DataFrame(
            {k: v for k, v in self.current_xreg.items() if not k.startswith("AR")}
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
        if tune:
            X_train = X.iloc[: -(self.test_length + self.validation_length)]
            X_test = (
                X.iloc[-(self.test_length + self.validation_length) : -self.test_length]
                if self.test_length > 0 else X.iloc[-(self.test_length + self.validation_length) :]
            )
            y_test = (
                X["y"].values[-(self.test_length + self.validation_length) : -self.test_length]
                if self.test_length > 0 else X["y"].values[-(self.test_length + self.validation_length) :]
            )
            model.fit(X_train)
            pred = model.predict(X_test)
            self._metrics(y_test, pred["yhat"].to_list())
            return self._tune()
        elif self.test_length > 0:
            model.fit(X.iloc[: -self.test_length] if self.test_length > 0 else X)
            pred = model.predict(X.iloc[-self.test_length :])
            pred = pred["yhat"].values
            self._metrics(X["y"].values[-self.test_length :], pred)

            if test_only:
                return (
                    pred["yhat"],
                    model.predict(X.iloc[: -self.test_length])["yhat"].to_list()
                    + [np.nan] * self.test_length,
                    model,
                    Xvars,
                )
        elif test_only:
            self._raise_test_only_error()
        else:
            self._metrics([],[])
        regr = Prophet(**kwargs)
        regr.fit(X)
        fcst = regr.predict(p)
        return (fcst["yhat"], regr.predict(X)["yhat"], Xvars, regr)

    def _forecast_silverkite(
        self, tune=False, dynamic_testing=True, Xvars=None, test_only=False, **kwargs
    ):
        """ Forecasts with the silverkite model from LinkedIn greykite library.
        See the example: https://scalecast-examples.readthedocs.io/en/latest/silverkite/silverkite.html.

        Args:
            tune (bool): Default False.
                Whether to tune the forecast.
                It does not need to be specified by the user.
            dynamic_testing (bool): Default True.
                Always True for silverkite. It can use lags but they are always far enough in the past to allow a direct forecast.
            Xvars (list-like, str, or None): The regressors to predict with.
                None means no Xvars used (unlike sklearn models).
            test_only (bool): Default False.
                Whether to stop the model after the testing process and not forecast into future periods.
                The forecast info stored in the object's history will be equivalent to test-set predictions.
                When True, any plot or export of forecasts into a future horizon will fail 
                and not all methods will raise descriptive errors.
                Will automatically change to True if object was initiated with require_future_dates = False.
            **kwargs: Passed to the ModelComponentsParam function from greykite.framework.templates.autogen.forecast_config.
        """
        from greykite.framework.templates.autogen.forecast_config import (
            ForecastConfig,
            MetadataParam,
            ModelComponentsParam,
            EvaluationPeriodParam,
        )
        from greykite.framework.templates.forecaster import Forecaster as SKForecaster
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

        if tune:
            y_test = (
                self.y.values[-(self.test_length + self.validation_length) : -self.test_length]
                if self.test_length > 0 else self.y.values[-(self.test_length + self.validation_length) :]
            )
            result = _forecast_sk(
                df,
                Xvars,
                self.validation_length,
                self.test_length,
                self.validation_length,
            )
            pred = result[0]
            self._metrics(y_test, pred[-self.validation_length :])
            return self._tune()
        elif self.test_length > 0:
            result = _forecast_sk(df, Xvars, self.test_length, 0, self.test_length)
            Xvars = Xvars if Xvars != [] else None
            pred = result[0]
            pred = pred[-self.test_length :]
            self._metrics(self.y.values[-self.test_length :], pred)
            if test_only:
                self.summary_stats = result[1].set_index("Pred_col")
                return (pred, pred[: -self.test_length], Xvars, None)
        elif test_only:
            self._raise_test_only_error()
        else:
            self._metrics([],[])
        result = _forecast_sk(df, Xvars, 0, 0, fcst_length)
        self.summary_stats = result[1].set_index("Pred_col")
        return (result[0][-fcst_length:], result[0][:-fcst_length], Xvars, None)

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
        """ Forecasts with a long-short term memory neural network from TensorFlow.
        Cannot be tuned.
        Only xvar options are the series' own history (specified in the `lags` argument).
        Always uses a minmax scaler on the inputs and outputs. The resulting point forecasts are unscaled.
        Anything this function can do, rnn can also do, 
        but this function is simpler to set up than rnn.
        The model is saved in the tf_model attribute and a summary can be called by calling Forecaster.tf_model.summary().
        See the example: https://scalecast-examples.readthedocs.io/en/latest/lstm/lstm.html.
            
        Args:
            dynamic_testing (bool): Default True.
                Always True for lstm. The model uses a direct forecast.
            lags (int): Must be greater than 0 (otherwise, what are your model inputs?). Default 1.
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
            test_only (bool): Default False.
                Whether to stop the model after the testing process and not forecast into future periods.
                The forecast info stored in the object's history will be equivalent to test-set predictions.
                When True, any plot or export of forecasts into a future horizon will fail 
                and not all methods will raise descriptive errors.
                Will automatically change to True if object was initiated with require_future_dates = False.
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
            test_only=test_only,
            **kwargs,
        )
        return self._forecast_rnn(**new_kwargs)

    def _forecast_rnn(
        self,
        dynamic_testing=True,
        lags=1,
        layers_struct=[("SimpleRNN", {"units": 8, "activation": "tanh"})],
        loss="mean_absolute_error",
        optimizer="Adam",
        learning_rate=0.001,
        random_seed=None,
        plot_loss_test=False,
        plot_loss=False,
        test_only=False,
        **kwargs,
    ):
        """ Forecasts with a recurrent neural network from TensorFlow, such as LSTM or simple recurrent.
        Not all features from tensorflow are available, but many of the most common ones for time series models are.
        Cannot be tuned.
        Only xvar options are the series' own history (specified in the lags argument).
        Always uses a minmax scaler on the inputs and outputs. The resulting point forecasts are unscaled.
        The model is saved in the tf_model attribute and a summary can be called by calling Forecaster.tf_model.summary().
        see example: https://scalecast-examples.readthedocs.io/en/latest/rnn/rnn.html.

        Args:
            dynamic_testing (bool): Default True.
                Always True for rnn. The model uses a direct forecast.
            lags (int): Must be greater than 0 (otherwise, what are your model inputs?). Default 1.
                The number of lags to train the model with.
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
            test_only (bool): Default False.
                Whether to stop the model after the testing process and not forecast into future periods.
                The forecast info stored in the object's history will be equivalent to test-set predictions.
                When True, any plot or export of forecasts into a future horizon will fail 
                and not all methods will raise descriptive errors.
                Will automatically change to True if object was initiated with require_future_dates = False.
            **kwargs: Passed to fit() and can include epochs, verbose, callbacks, validation_split, and more.
        """

        def get_compiled_model(y):
            # build model
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
                        [layer(**kv[1], input_shape=(n_timesteps, n_features),)]
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
                x_line = ylist[idx_start : idx_start + n_past]
                y_line = ylist[idx_start + n_past : idx_start + total_period]

                X_new.append(x_line)
                y_new.append(y_line)

                idx_start = idx_start - 1

            X_new = np.array(X_new)
            y_new = np.array(y_new)

            return X_new, y_new

        def plot_loss_rnn(history, title):
            plt.plot(history.history["loss"], label="train_loss")
            if "val_loss" in history.history.keys():
                plt.plot(history.history["val_loss"], label="val_loss")
            plt.title(title)
            plt.xlabel("epoch")
            plt.legend(loc="upper right")
            plt.show()

        descriptive_assert(
            len(layers_struct) >= 1,
            ValueError,
            f"must pass at least one layer to layers_struct, got {layers_struct}",
        )
        self._warn_about_dynamic_testing(
            dynamic_testing=dynamic_testing,
            does_not_use_lags=False,
            uses_direct=True,
        )

        if random_seed is not None:
            random.seed(random_seed)

        n_features = 1

        if self.test_length > 0:
            y_train = self.y.values[: -self.test_length].copy()
            y_test = self.y.values[-self.test_length :].copy()

            ymin = y_train.min()
            ymax = y_train.max()

            X_train, y_train_new = prepare_rnn(y_train, lags, self.test_length)

            X_test = np.array(
                [
                    [
                        (i - ymin) / (ymax - ymin)
                        for i in self.y.values[
                            -(lags + self.test_length) : -self.test_length
                        ].copy()
                    ]
                ]
            )
            y_test_new = np.array(
                [
                    [
                        (i - ymin) / (ymax - ymin)
                        for i in self.y.values[-self.test_length :].copy()
                    ]
                ]
            )

            n_timesteps = X_train.shape[1]

            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
            X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
            test_model = get_compiled_model(y_train_new)
            hist = test_model.fit(X_train, y_train_new, **kwargs)
            pred = test_model.predict(X_test)
            pred = [p * (ymax - ymin) + ymin for p in pred[0]]  # un-minmax
            self._metrics(y_test, pred)
            if plot_loss_test:
                plot_loss_rnn(hist, "model loss - test")
            if test_only:
                fvs = test_model.predict(X_train)
                fvs = [p[0] * (ymax - ymin) + ymin for p in fvs[1:][::-1]] + [
                    p * (ymax - ymin) + ymin for p in fvs[0]
                ]
                self.tf_model = test_model
                return (
                    pred, 
                    fvs + [np.nan] * self.test_length, 
                    None, 
                    None,
                )
        elif test_only:
            self._raise_test_only_error()
        else:
            self._metrics([],[])
        X, y_new = prepare_rnn(self.y.values.copy(), lags, len(self.future_dates))
        fut = np.array(
            [
                [
                    (i - self.y.min()) / (self.y.max() - self.y.min())
                    for i in self.y.values[-lags:].copy()
                ]
            ]
        )
        n_timesteps = X.shape[1]
        X = X.reshape(X.shape[0], X.shape[1], 1)
        fut = fut.reshape(fut.shape[0], fut.shape[1], 1)
        model = get_compiled_model(y_new)
        hist = model.fit(X, y_new, **kwargs)
        if plot_loss:
            plot_loss_rnn(hist, "model loss - full")
        fcst = model.predict(fut)
        fvs = model.predict(X)
        fvs = [
            p[0] * (self.y.max() - self.y.min()) + self.y.min() for p in fvs[1:][::-1]
        ] + [p * (self.y.max() - self.y.min()) + self.y.min() for p in fvs[0]]
        fcst = [p * (self.y.max() - self.y.min()) + self.y.min() for p in fcst[0]]
        self.tf_model = model
        return (fcst, fvs, None, None)

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
            self.test_only = False

            m = _convert_m(m,self.freq) if seasonal else 1
            obs = self.y.values.copy()
            train_obs = self.y.values[:-self.test_length].copy()
            test_obs = self.y.values[-self.test_length:].copy()
            pred = (pd.Series(train_obs).to_list()[-m:] * int(np.ceil(self.test_length/m)))[:self.test_length]
            fcst = (pd.Series(obs).to_list()[-m:] * int(np.ceil(len(self.future_dates)/m)))[:len(self.future_dates)]
            fvs = pd.Series(obs).shift(m).dropna().values

            self._metrics(test_obs,pred)
            return(fcst, fvs, None, None)

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
        """ Combines at least two previously evaluted forecasts to create a new model.
        This method cannot be run on models that were run test_only = True.
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
            determine_best_by (str): One of _determine_best_by_, default 'ValidationMetricValue'.
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
            test_only (bool): default False:
                Always forced to be False in the combo model.
        """
        self.dynamic_testing = True
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
                    "Must pass as many weights as models.",
                )
                descriptive_assert(
                    not isinstance(weights, str),
                    TypeError,
                    f"weights argument not recognized: {weights}.",
                )
                weights = pd.DataFrame(zip(models, weights)).set_index(0).transpose()
                if weights.sum(axis=1).values[0] == 1:
                    scale = False
                    rebalance_weights = 0
            try:
                descriptive_assert(
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
                descriptive_assert(
                    len(models) == len(splice_points) + 1,
                    ForecastError,
                    "Must have exactly 1 more model passed to models as splice points.",
                )
                splice_points = pd.to_datetime(sorted(splice_points)).to_list()
                future_dates = self.future_dates.to_list()
                descriptive_assert(
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

        self._metrics(actuals, pred)
        self.weights = tuple(weights.values[0]) if weights is not None else None
        self.models = models
        return (fcst, fv, None, None)

    def _parse_models(self, models, determine_best_by):
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
        descriptive_assert(
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

    def fillna_y(self, how="ffill"):
        """ Fills null values in the y attribute.

        Args:
            how (str): One of {'backfill', 'bfill', 'pad', 'ffill', 'midpoint'}.
                Midpoint is unique to this library and only works if there is not more than two missing values sequentially.
                All other possible arguments are from pandas.DataFrame.fillna() method and will do the same.

        Returns:
            None
        """
        if (
            how != "midpoint"
        ):  # only works if there aren't more than 2 na one after another
            self.y = self.y.fillna(method=how)
        else:
            for i, val in enumerate(self.y.values):
                if val is None:
                    self.y.values[i] = (self.y.values[i - 1] + self.y.values[i + 1]) / 2

    def generate_future_dates(self, n):
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

    def set_last_future_date(self, date):
        """ Generates future dates in the same frequency as current_dates that ends on a specified date.

        Args:
            date (datetime.datetime, pd.Timestamp, or str):
                The date to end on. Must be parsable by pandas' Timestamp() function.

        Returns:
            None

        >>> f.set_last_future_date('2021-06-01') # creates future dates up to this one in the expected frequency
        """
        self.future_dates = pd.Series(
            pd.date_range(
                start=self.current_dates.values[-1], end=date, freq=self.freq
            ).values[1:]
        )

    def typ_set(self):
        """ Converts all objects in y, current_dates, future_dates, current_xreg, and future_xreg to appropriate types if possible.
        Automatically gets called when object is initiated to minimize potential errors.

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
                "length of array representing the input '{}' does match the length of the stored observed values: ({} vs. {})."
                " if you do not know how this happened, consider raising an issue: https://github.com/mikekeith52/scalecast/issues/new.".format(
                    k,
                    len(self.current_xreg[k]),
                    len(self.y),
                ),
            )
            self.future_xreg[k] = [float(x) for x in self.future_xreg[k]]

        self.infer_freq()

    def diff(self):
        """ Differences the y attribute, as well as all AR values stored in current_xreg and future_xreg.
        If series is already differenced, calling this function does nothing.
        As of 0.14.8, only one-differencing supported. Two-differencing and other transformations must use the 
        SeriesTransformer object:
        https://scalecast.readthedocs.io/en/latest/Forecaster/SeriesTransformer.html

        Returns:
            None

        >>> f.diff() # differences y once
        """
        if self.integration == 1:
            return

        self.integration = 1
        self.y = self.y.diff()
        for k, v in self.current_xreg.items():
            if k.startswith("AR"):
                ar = int(k[2:])
                self.current_xreg[k] = v.diff()
                self.future_xreg[k] = [self.y.values[-ar]] # just gets one future ar

    def integrate(self, critical_pval=0.05, train_only=False, **kwargs):
        """ Differences the series up to 1 time based on Augmented Dickey-Fuller test results.
        If series is already differenced, calling this function does nothing.

        Args:
            critical_pval (float): Default 0.05.
                The p-value threshold in the statistical test to accept the alternative hypothesis.
            train_only (bool): Default False.
                If True, will exclude the test data set from the Augmented Dickey-Fuller test (to avoid leakage).
            **kwargs: Passed to the `adfuller()` function from statsmodels.
                See https://www.statsmodels.org/dev/generated/statsmodels.tsa.stattools.adfuller.html.

        Returns:
            None

        >>> f.integrate() # differences y once if it is not stationarity
        """
        if self.integration == 1:
            return
        
        res = adfuller(
            self.y.dropna()
            if not train_only
            else self.y.dropna().values[: -self.test_length],
            **kwargs,
        )
        if res[1] >= critical_pval:
            self.diff()

    def add_ar_terms(self, n):
        """ Adds auto-regressive terms.

        Args:
            n (int): The number of lags to add to the object (1 to this number will be added).

        Returns:
            None

        >>> f.add_ar_terms(4) # adds four lags of y (called 'AR1' - 'AR4') to predict with
        """
        self._validate_future_dates_exist()
        n = int(n)

        if n == 0:
            return

        descriptive_assert(
            n >= 0, ValueError, f"n must be greater than or equal to 0, got {n}"
        )
        for i in range(1, n + 1):
            self.current_xreg[f"AR{i}"] = pd.Series(self.y).shift(i)
            self.future_xreg[f"AR{i}"] = [self.y.values[-i]]

    def add_AR_terms(self, N):
        """ Adds seasonal auto-regressive terms.
            
        Args:
            N (tuple): First element is the number of lags to add and the second element is the space between lags.

        Returns:
            None

        >>> f.add_AR_terms((2,12)) # adds 12th and 24th lags called 'AR12', 'AR24'
        """
        self._validate_future_dates_exist()
        descriptive_assert(
            (len(N) == 2) & (not isinstance(N, str)),
            ValueError,
            f"n must be an array-like of length 2 (P,m), got {N}",
        )
        for i in range(N[1], N[1] * N[0] + 1, N[1]):
            self.current_xreg[f"AR{i}"] = pd.Series(self.y).shift(i)
            self.future_xreg[f"AR{i}"] = [self.y.values[-i]]

    def ingest_Xvars_df(
        self, df, date_col="Date", drop_first=False, use_future_dates=False
    ):
        """ Ingests a dataframe of regressors and saves its Xvars to the object.
        The user must specify a date column name in the dataframe being ingested.
        All non-numeric values are dummied.
        Any columns in the dataframe that begin with "AR" will be confused with autoregressive terms and could cause errors.

        Args:
            df (DataFrame): The dataframe that is at least the length of the y array stored in the object plus the forecast horizon,
                except if the object was initiated with require_future_dates = False. Then it just has to be the same length as
                the y array stored in the object.
            date_col (str): Default 'Date'.
                The name of the date column in the dataframe.
                This column must have the same frequency as the dates stored in the Forecaster object.
            drop_first (bool): Default False.
                Whether to drop the first observation of any dummied variables.
                Irrelevant if passing all numeric values.
            use_future_dates (bool): Default False.
                Whether to use the future dates in the dataframe as the resulting future_dates attribute in the Forecaster object.

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
        if self.require_future_dates:
            future_df = df.loc[df[date_col] > self.current_dates.values[-1]]
            descriptive_assert(
                future_df.shape[0] > 0,
                ForecastError,
                'Regressor values must be known into the future unless require_future_dates attr is set to False.'
            )
        else:
            future_df = pd.DataFrame({date_col: self.future_dates.to_list()})
            for c in current_df:
                if c != date_col:
                    future_df[c] = 0 # shortcut to make the rest of everything else work better
        descriptive_assert(
            current_df.shape[0] == len(self.y),
            ForecastError,
            "Could not ingest Xvars dataframe."
            " Make sure the dataframe spans the entire date range as y and is at least one observation to the future."
            " Specify the date column name in the `date_col` argument.",
        )

        if not use_future_dates:
            descriptive_assert(
                future_df.shape[0] >= len(self.future_dates),
                ValueError,
                "The future dates in the dataframe should be at least the same length as the future dates in the Forecaster object." 
                " If you want to use the dataframe to set the future dates for the object, pass True to the use_future_dates argument.",
            )
        else:
            self.future_dates = future_df[date_col]

        for c in [c for c in future_df if c != date_col]:
            self.future_xreg[c] = future_df[c].to_list()[: len(self.future_dates)]
            self.current_xreg[c] = current_df[c]

        for x, v in self.future_xreg.items():
            self.future_xreg[x] = v[: len(self.future_dates)]
            if not len(v) == len(self.future_dates):
                warnings.warning(
                    f"{x} is not the correct length in the future_dates attribute and this can cause errors when forecasting."
                    f" Its length is {len(v)} and future_dates length is {len(self.future_dates)}."
                )

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
        After each variable reduction, the model is re-run and pfi re-evaluated. When using pfi, feature scores
        are adjusted to account for colinearity, which is a known issue with this method, 
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
            estimator (str): One of _sklearn_estimators_. Default 'lasso'.
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
            monitor (str): One of _determine_best_by_. Default 'ValidationSetMetric'.
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
        descriptive_assert(
            method in ("l1", "pfi", "shap"),
            ValueError,
            f'method must be one of "pfi", "l1", "shap", got {method}.',
        )
        f = self.__deepcopy__()
        descriptive_assert(
            estimator in _sklearn_estimators_,
            ValueError,
            f"estimator must be one of {_sklearn_estimators_}, got {estimator}.",
        )
        if self.test_length == 0:
            warnings.warn(
                'Test set length is 0. ' 
                'When calling reduce_Xvars, '
                'a copy of the object is taken and its test set length is set to 1.'
            )
            f.set_test_length(1)
        f.set_estimator(estimator if method in ("pfi", "shap") else "lasso")
        if grid_search:
            if not use_loaded_grid:
                from scalecast import GridGenerator

                GridGenerator.get_example_grids()
                f.ingest_grid(f.estimator)
            else:
                descriptive_assert(
                    hasattr(f, "grid"),
                    ForecastError,
                    "Grid not loaded, try setting use_loaded_grid to False.",
                )
        else:
            f.ingest_grid({k: [v] for k, v in kwargs.items()})

        if not cross_validate:
            f.tune(dynamic_tuning=dynamic_tuning)
        else:
            f.cross_validate(dynamic_tuning=dynamic_tuning, **cvkwargs)
        f.ingest_grid({k: [v] for k, v in f.best_params.items()})
        f.auto_forecast(test_only=True)

        self.reduction_hyperparams = f.best_params.copy()

        if method == "l1":
            coef_fi_lasso = pd.DataFrame(
                {
                    x: [np.abs(co)]
                    for x, co in zip(f.history["lasso"]["Xvars"], f.regr.coef_,)
                },
                index=["feature"],
            ).T
            self.reduced_Xvars = coef_fi_lasso.loc[
                coef_fi_lasso["feature"] != 0
            ].index.to_list()
        else:
            f.save_feature_importance(method, on_error="raise")
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
                f.grid["Xvars"] = [features]
                if not cross_validate:
                    f.tune(dynamic_tuning=dynamic_tuning)
                else:
                    f.cross_validate(dynamic_tuning=dynamic_tuning, **cvkwargs)
                f.auto_forecast(test_only=True)
                new_error = f.history[estimator][monitor]
                new_error = -new_error if using_r2 else new_error
                errors.append(new_error)

                f.save_feature_importance(method, on_error="raise")
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
            f.manual_forecast(**kwargs,Xvars=Xvars)
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
            estimator (str): One of _sklearn_estimators_. Default 'mlr'.
                The estimator to use to determine the best seasonal and lag regressors.
            try_trend (bool): Default True.
                Whether to search for trend representations of the series.
            trend_estimator (str): One of _sklearn_estimators_. Default 'mlr'.
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
                If 'auto', will use the test-set length as the lag order.
                Set to 0 to skip searching for lag terms.
            test_already_added (bool): Default True.
                If there are already regressors added to the series, you can either always keep them in the object
                by setting this to False, or by default, it is possible they will be dropped when looking for the
                optimal combination of regressors in the object.
            must_keep (list-like): Default []. The names of any regressors that must be kept in the object.
                All regressors here must already be added to the Forecaster object before calling the function.
                This is ignored if test_already_added is False since it becomes redundant.
            monitor (str): One of _determine_best_by_. Default 'ValidationMetricValue'.
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

        >>> import pandas as pd
        >>> from scalecast.Forecaster import Forecaster
        >>> import pandas_datareader as pdr
        >>> df = pdr.get_data_fred('HOUSTNSA',start='1900-01-01',end='2021-06-01')
        >>> f = Forecaster(y=df['HOUSTNSA'],current_dates=df.index,future_dates=24)
        >>> f.diff()
        >>> f.add_covid19_regressor()
        >>> f.set_test_length(24)
        >>> f.set_validation_length(12)
        >>> f.auto_Xvar_select()
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

        using_r2 = monitor.endswith("R2") or (
            self.validation_metric == "r2" and monitor == "ValidationMetricValue"
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
                        require_future_dates=False,
                    )
                except Exception as e:
                    warnings.warn(
                        f'Trend decomposition did not work and raised this error: {e} '
                        'Switching to the non-decomp method'
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
                warnings.warn(f'No seasonalities are currently associated with the {f.freq} frequency.')
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
                        raise TypeError(f'seasonality_repr must be list or dict type, got {type(seasonality_repr)}')
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
            max_ar = f.test_length if max_ar == 'auto' else max_ar
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
                        warnings.warn(f'Cannot estimate {estimator} model with {i} AR terms.')
                        ar_metrics.pop(i)
                        break
                except (IndexError,AttributeError,ForecastError):
                    warnings.warn(f'Cannot estimate {estimator} model with {i} AR terms.')
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
        Do not run this when using the `SeriesTransform.DiffRevert()` process. 
        There is no danger in running this with the `Forecaster.diff()` function. 

        Args:
            estimator (str): One of _estimators_. Default 'mlr'.
                The estimator to use to determine the best series length.
            min_obs (int): Default 100.
                The shortest representation of the series to search.
            max_obs (int): Optional.
                The longest representation of the series to search.
                By default, the last estimation will be run on all available observations.
            step (int): Default 25.
                How big a step to take between iterations.
            monitor (str): One of _determine_best_by_. Default 'ValidationSetMetric'.
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

        >>> import pandas as pd
        >>> from scalecast.Forecaster import Forecaster
        >>> import pandas_datareader as pdr
        >>> df = pdr.get_data_fred('HOUSTNSA',start='1900-01-01',end='2021-06-01')
        >>> f = Forecaster(y=df['HOUSTNSA'],current_dates=df.index,future_dates=24)
        >>> f.diff()
        >>> f.add_covid19_regressor()
        >>> f.set_test_length(24)
        >>> f.set_validation_length(12)
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

    def set_test_length(self, n=1):
        """ Sets the length of the test set. As of version 0.16.0, 0-length test sets are supported.

        Args:
            n (int or float): Default 1.
                The length of the resulting test set.
                Pass 0 to skip testing models.
                Fractional splits are supported by passing a float less than 1 and greater than 0.

        Returns:
            None

        >>> f.set_test_length(12) # test set of 12
        >>> f.set_test_length(.2) # 20% test split
        """
        float(n)
        if n == 0:
            self.test_length = 0
        if n >= 1:
            n = int(n)
            descriptive_assert(
                isinstance(n, int),
                ValueError,
                f"n must be an int of at least 0 or float greater than 0 and less than 1, got {n} of type {type(n)}",
            )
            self.test_length = n
        else:
            descriptive_assert(
                n >= 0,
                ValueError,
                f"n must be an int of at least 0 or float greater than 0 and less than 1, got {n} of type {type(n)}",
            )
            self.test_length = int(len(self.y) * n)

    def set_validation_length(self, n=1):
        """ Sets the length of the validation set. This will never matter for models that are not tuned.

        Args:
            n (int): Default 1.
                The length of the resulting validation set.

        Returns:
            None

        >>> f.set_validation_length(6) # validation length of 6
        """
        n = int(n)
        descriptive_assert(n > 0, ValueError, f"n must be greater than 0, got {n}")
        if (self.validation_metric == "r2") & (n == 1):
            raise ValueError(
                "can only set a validation_length of 1 if validation_metric is not r2. try set_validation_metric()"
            )
        self.validation_length = n

    def set_cilevel(self, n):
        """ Sets the level for the resulting confidence intervals (95% default).

        Args:
            n (float): Greater than 0 and less than 1.

        Returns:
            None

        >>> f.set_cilevel(.80) # next forecast will get 80% confidence intervals
        """
        descriptive_assert(
            n < 1 and n > 0, ValueError, "n must be greater than 0 and less than 1"
        )
        self.cilevel = n

    def set_bootstrap_samples(self, n):
        """ Sets the number of bootstrap samples to set confidence intervals for each model (100 default).

        Args:
            n (int): Greater than or equal to 30.
                30 because you need around there to satisfy central limit theorem.
                The lower this number, the faster the performance, but the less confident in the resulting intervals you should be.

        Returns:
            None

        >>> f.set_bootstrap_samples(1000) # next forecast will get confidence intervals with 1,000 bootstrap sample
        """
        descriptive_assert(n >= 30, ValueError, "n must be greater than or equal to 30")
        self.bootstrap_samples = n

    def adf_test(
        self, critical_pval=0.05, full_res=True, train_only=False, **kwargs
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
            **kwargs: Passed to the `adfuller()` function from statsmodels. See
                https://www.statsmodels.org/dev/generated/statsmodels.tsa.stattools.adfuller.html.

        Returns:
            (bool or tuple): If bool (full_res = False), returns whether the test suggests stationarity.
            Otherwise, returns the full results (stat, pval, etc.) of the test.

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
        descriptive_assert(
            isinstance(train_only, bool), ValueError, "train_only must be True or False"
        )
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
        descriptive_assert(
            isinstance(train_only, bool), ValueError, "train_only must be True or False"
        )
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

        descriptive_assert(
            isinstance(train_only, bool), ValueError, "train_only must be True or False"
        )
        y = self._diffy(diffy)
        y = y.values if not train_only else y.values[: -self.test_length]
        return periodogram(y)

    def seasonal_decompose(self, diffy=False, train_only=False, **kwargs):
        """ Plots a signal/seasonal decomposition of the y values.

        Args:
            diffy (bool or int): One of {True,False,0,1}. Default False.
                Whether to difference the data before passing the values to the function.
                If False or 0, does not difference.
                If True or 1, differences 1 time.
                If 2, differences 2 times.
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
        self, 
        *args, 
        raw=True, 
        sincos=False, 
        dummy=False, 
        drop_first=False,
        cycle_lens=None,
    ):
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
                If this is not specified or a selected seasonality is not added to the dictionary as a key, the
                cycle length will be selected automatically as the maximum value observed for the given seasonality.
                Not relevant when sincos = False.

        Returns:
            None

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
                    _raw.max() if cycle_lens is None 
                    else _raw.max()  if s not in cycle_lens 
                    else cycle_lens[s]
                )
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
        """ Adds a time trend from 1 to length of the series + the forecast horizon as a current and future Xvar.

        Args:
            Called (str): Default 't'.
                What to call the resulting variable

        Returns:
            None

        >>> f.add_time_trend() # adds time trend called 't'
        """
        self._validate_future_dates_exist()
        self.current_xreg[called] = pd.Series(range(1, len(self.y) + 1))
        self.future_xreg[called] = list(
            range(len(self.y) + 1, len(self.y) + len(self.future_dates) + 1)
        )

    def add_cycle(self, cycle_length, called=None):
        """ Adds a regressor that acts as a seasonal cycle.
        Use this function to capture non-normal seasonality.

        Args:
            cycle_length (int): How many time steps make one complete cycle.
            called (str): Optional. What to call the resulting variable.
                Two variables will be created--one for a sin transformation and the other for cos
                resulting variable names will have "sin" or "cos" at the end.
                Example, called = 'cycle5' will become 'cycle5sin', 'cycle5cos'.
                If left unspecified, 'cycle{cycle_length}' will be used as the name.

        Returns:
            None

        >>> f.add_cycle(13) # adds a seasonal effect that cycles every 13 observations callec 'cycle13'
        """
        self._validate_future_dates_exist()
        if called is None:
            called = f"cycle{cycle_length}"
        full_sin = pd.Series(range(1, len(self.y) + len(self.future_dates) + 1)).apply(
            lambda x: np.sin(np.pi * x / (cycle_length / 2))
        )
        full_cos = pd.Series(range(1, len(self.y) + len(self.future_dates) + 1)).apply(
            lambda x: np.cos(np.pi * x / (cycle_length / 2))
        )
        self.current_xreg[called + "sin"] = pd.Series(full_sin.values[: len(self.y)])
        self.current_xreg[called + "cos"] = pd.Series(full_cos.values[: len(self.y)])
        self.future_xreg[called + "sin"] = list(full_sin.values[len(self.y) :])
        self.future_xreg[called + "cos"] = list(full_cos.values[len(self.y) :])

    def add_other_regressor(self, called, start, end):
        """ Adds a dummy variable that is 1 during the specified time period, 0 otherwise.

        Args:
            called (str):
                What to call the resulting variable.
            start (str, datetime.datetime, or pd.Timestamp): Start date.
                Must be parsable by pandas' Timestamp function.
            end (str, datetime.datetime, or pd.Timestamp): End date.
                Must be parsable by pandas' Timestamp function.

        Returns:
            None

        >>> f.add_other_regressor('january_2021','2021-01-01','2021-01-31')
        """
        self._validate_future_dates_exist()
        self.current_xreg[called] = pd.Series(
            [1 if (x >= pd.Timstamp(start)) & (x <= pd.Timstamp(end)) else 0 for x in self.current_dates]
        )
        self.future_xreg[called] = [
            1 if (x >= pd.Timstamp(start)) & (x <= pd.Timstamp(end)) else 0 for x in self.future_dates
        ]

    def add_covid19_regressor(
        self,
        called="COVID19",
        start=datetime.datetime(2020, 3, 15),
        end=datetime.datetime(2021, 5, 13),
    ):
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
        self._validate_future_dates_exist()
        self.add_other_regressor(called=called, start=start, end=end)

    def add_combo_regressors(self, *args, sep="_"):
        """ Combines all passed variables by multiplying their values together.

        Args:
            *args (str): Names of Xvars that aleady exist in the object.
            sep (str): Default '_'.
                The separator between each term in arg to create the final variable name.

        Returns:
            None

        >>> f.add_combo_regressors('t','monthsin') # multiplies these two together (called 't_monthsin')
        >>> f.add_combo_regressors('t','monthcos') # multiplies these two together (called 't_monthcos')
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
            *args (str): Names of Xvars that aleady exist in the object
            pwr (int): Default 2.
                The max power to add to each term in args (2 to this number will be added).
            sep (str): default '^'.
                The separator between each term in arg to create the final variable name.

        Returns:
            None

        >>> f.add_poly_terms('t','year',pwr=3) # raises t and year to 2nd and 3rd powers (called 't^2', 't^3', 'year^2', 'year^3')
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
            None

        >>> f.add_exp_terms('t',pwr=.5) # adds square root t called 't^0.5'
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
            None

        >>> f.add_logged_terms('t') # adds natural log t callend 'lnt'
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
            None

        >>> f.add_pt_terms('t') # adds box cox of t called 'box-cox_t'
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
        """ Differences all passed variables (no AR terms) up to 2 times.

        Args:
            *args (str): Names of Xvars that aleady exist in the object.
            diff (int): One of {1,2}. Default 1.
                The number of times to difference each variable passed to args.
            sep (str): Default '_'.
                The separator between each term in arg to create the final variable name.
                Resulting variable names will be like "tdiff_1" or "tdiff_2" by default.
            drop (bool): Default False.
                Whether to drop the regressors passed to *args.

        Returns:
            None

        >>> add_diffed_terms('t') # adds first difference of t as regressor called 't_diff1'
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

        Returns:
            None

        >>> add_lagged_terms('t',lags=3) # adds first, second, and third lag of t called 't_lag1' - 't_lag3'
        >>> add_lagged_terms('t',lags=6,upto=False) # adds 6th lag of t only called 't_lag6'
        """
        self._validate_future_dates_exist()
        lags = int(lags)
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

    def undiff(self):
        """ Undifferences y to original level and drops all regressors (such as AR terms).
        If y was never differenced, calling this does nothing.

        >>> f.diff() # differences the series
        >>> f.undiff() # undifferences the series and drops any added Xvars
        """
        self.typ_set()
        if self.integration == 0:
            return

        self.current_xreg = {}
        self.future_xreg = {}

        self.current_dates = pd.Series(self.init_dates)
        self.y = pd.Series(self.levely)

        self.integration = 0

    def restore_series_length(self):
        """ Restores the series to its original size, undifferences, and drops all Xvars.
        """
        self.current_xreg = {}
        self.future_xreg = {}

        self.current_dates = pd.Series(self.init_dates)
        self.y = pd.Series(self.levely)

        self.integration = 0

    def set_estimator(self, estimator):
        """ Sets the estimator to forecast with.

        Args:
            estimator (str): One of _estimators_.

        Returns:
            None

        >>> f.set_estimator('lasso')
        >>> f.manual_forecast(alpha = .5)
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
                self._clear_the_deck()
                self.estimator = estimator
        else:
            self.estimator = estimator

    def set_grids_file(self,name='Grids'):
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
        descriptive_assert(isinstance(name,str),ValueError,f'name argument expected str type, got {type(name)}')
        self.grids_file = name

    def ingest_grid(self, grid):
        """ Ingests a grid to tune the estimator.

        Args:
            grid (dict or str):
                If dict, must be a user-created grid.
                If str, must match the name of a dict grid stored in a grids file.

        Returns:
            None

        >>> f.set_estimator('mlr')
        >>> f.ingest_grid({'normalizer':['scale','minmax']})
        """
        from itertools import product

        def expand_grid(d):
            return pd.DataFrame([row for row in product(*d.values())], columns=d.keys())

        try:
            if isinstance(grid, str):
                Grids = importlib.import_module(self.grids_file)
                importlib.reload(Grids)
                grid = getattr(Grids, grid)
        except SyntaxError:
            raise
        except:
            raise ForecastError(
                f"tried to load a grid called {self.estimator} from {self.grids_file}.py, "
                "but either the file could not be found in the current directory, "
                "there is no grid with that name, or the dictionary values are not list-like. "
                "try ingest_grid() with a dictionary grid passed manually."
            )
        grid = expand_grid(grid)
        self.grid = grid

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
        """ Sets the metric that will be used to tune all subsequent models.

        Args:
            metric: One of _metrics_. Default 'rmse'.
                The metric to optimize the models with using the validation set.

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

    def tune(self, dynamic_tuning=False, cv=False):
        """ Tunes the specified estimator using an ingested grid (ingests a grid from Grids.py with same name as 
        the estimator by default).
        Any parameters that can be passed as arguments to manual_forecast() can be tuned with this process.
        The chosen parameters are stored in the best_params attribute.
        The full validation grid is stored in grid_evaluated.

        Args:
            dynamic_tuning (bool or int):
                Whether to dynamically/recursively test the forecast during the tuning process 
                (meaning AR terms will be propagated with predicted values).
                If True, evaluates recursively over the entire out-of-sample slice of data.
                If int, window evaluates over that many steps (2 for 2-step recurvie testing, 12 for 12-step, etc.).
                Setting this to False or 1 means faster performance, 
                but gives a less-good indication of how well the forecast will perform more than one period out.
            cv (bool): default False.
                whether the tune is part of a larger cross-validation process.
                this does not need to specified by the user and should be kept False when calling `tune()`.

        Returns:
            None

        >>> f.set_estimator('xgboost')
        >>> f.tune()
        >>> f.auto_forecast()
        """
        if not hasattr(self, "grid"):
            self.ingest_grid(self.estimator)

        if self.estimator in _cannot_be_tuned_:
            raise ForecastError(f"{self.estimator} models cannot be tuned")
            self.best_params = {}
            return

        metrics = []
        iters = self.grid.shape[0]
        for i in range(iters):
            try:
                hp = {k: v[i] for k, v in self.grid.to_dict(orient="list").items()}
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
                        getattr(self, f"_forecast_{self.estimator}")(tune=True, **hp)
                    )
            except TypeError:
                raise
            except Exception as e:
                #raise # good to uncomment when debugging
                self.grid.drop(i, axis=0, inplace=True)
                logging.warning(f"could not evaluate the paramaters: {hp}. error: {e}")

        if len(metrics) > 0:
            self.grid_evaluated = self.grid.copy()
            self.grid_evaluated["validation_length"] = self.validation_length
            self.grid_evaluated["validation_metric"] = self.validation_metric
            self.grid_evaluated["metric_value"] = metrics
            self.dynamic_tuning = dynamic_tuning
        else:
            self.grid_evaluated = pd.DataFrame()
        if not cv:
            self._find_best_params(self.grid_evaluated)

    def cross_validate(self, k=5, rolling=False, dynamic_tuning=False):
        """ Tunes a model's hyperparameters using time-series cross validation. 
        Monitors the metric specified in the valiation_metric attribute. 
        Set an estimator before calling. 
        Reads a grid for the estimator from a grids file unless a grid is ingested manually. 
        Each fold size is equal to one another and is determined such that the last fold's 
        training and validation sizes are the same (or close to the same). with rolling = True, 
        all train sizes will be the same for each fold. 
        The chosen parameters are stored in the best_params attribute.
        The full validation grid is stored in grid_evaluated.
        Normal cv diagram: https://scalecast-examples.readthedocs.io/en/latest/misc/validation/validation.html#5-Fold-Time-Series-Cross-Validation.
        Rolling cv diagram: https://scalecast-examples.readthedocs.io/en/latest/misc/validation/validation.html#5-Fold-Rolling-Time-Series-Cross-Validation. 

        Args:
            k (int): Default 5. 
                The number of folds. 
                Must be at least 2.
            rolling (bool): Default False. Whether to use a rolling method.
            dynamic_tuning (bool or int):
                Whether to dynamically/recursively test the forecast during the tuning process 
                (meaning AR terms will be propagated with predicted values).
                If True, evaluates recursively over the entire out-of-sample slice of data.
                If int, window evaluates over that many steps (2 for 2-step recurvie testing, 12 for 12-step, etc.).
                Setting this to False or 1 means faster performance, 
                but gives a less-good indication of how well the forecast will perform more than one period out.

        Returns:
            None

        >>> f.set_estimator('xgboost')
        >>> f.cross_validate() # tunes hyperparam values
        >>> f.auto_forecast() # forecasts with the best params
        """
        rolling = bool(rolling)
        k = int(k)
        descriptive_assert(k >= 2, ValueError, f"k must be at least 2, got {k}")
        f = self.__deepcopy__()
        usable_obs = len(f.y) - f.test_length
        if f.estimator in _sklearn_estimators_:
            ars = [
                int(x.split("AR")[-1])
                for x in self.current_xreg.keys()
                if x.startswith("AR")
            ]
            if ars:
                usable_obs -= max(ars)
        val_size = usable_obs // (k + 1)
        descriptive_assert(
            val_size > 0,
            ForecastError,
            f'not enough observations in sample to cross validate.'
        )
        f.set_validation_length(val_size)

        grid_evaluated_cv = pd.DataFrame()
        for i in range(k):
            if i > 0:
                f.current_xreg = {
                    k: pd.Series(v.values[:-val_size])
                    for k, v in f.current_xreg.items()
                }
                f.current_dates = pd.Series(f.current_dates.values[:-val_size])
                f.y = pd.Series(f.y.values[:-val_size])
                f.levely = f.levely[:-val_size]

            f2 = f.__deepcopy__()
            if rolling:
                f2.keep_smaller_history(val_size * 2 + f2.test_length)

            f2.tune(dynamic_tuning=dynamic_tuning,cv=True)
            orig_grid = f2.grid.copy()
            if f2.grid_evaluated.shape[0] == 0:
                self.grid, self.grid_evaluated = pd.DataFrame(), pd.DataFrame()
                self._find_best_params(f2.grid_evaluated)
                return

            f2.grid_evaluated["fold"] = i
            f2.grid_evaluated["rolling"] = rolling
            f2.grid_evaluated["train_length"] = len(f2.y) - val_size - f2.test_length
            grid_evaluated_cv = pd.concat([grid_evaluated_cv, f2.grid_evaluated])

        # convert into tuple for the group or it fails
        if "Xvars" in grid_evaluated_cv:
            grid_evaluated_cv["Xvars"] = grid_evaluated_cv["Xvars"].apply(
                lambda x: (
                    "None" if x is None else x if isinstance(x, str) else tuple(x)
                )
            )

        grid_evaluated = grid_evaluated_cv.groupby(
            grid_evaluated_cv.columns.to_list()[:-4],
            dropna=False,
            as_index=False,
            sort=False,
        )["metric_value"].mean()

        # convert back to list or it fails when calling the hyperparam vals
        # contributions welcome for a more elegant solution
        if "Xvars" in grid_evaluated:
            grid_evaluated["Xvars"] = grid_evaluated["Xvars"].apply(
                lambda x: (
                    None if x == "None" else x if isinstance(x, str) else list(x)
                )
            )

        self.grid = grid_evaluated.iloc[:, :-3]
        self.dynamic_tuning = f2.dynamic_tuning
        self._find_best_params(grid_evaluated)
        self.grid_evaluated = grid_evaluated_cv.reset_index(drop=True)
        self.grid = orig_grid

    def _find_best_params(self, grid_evaluated):
        if grid_evaluated.shape[0] > 0:
            if self.validation_metric == "r2":
                best_params_idx = self.grid.loc[
                    grid_evaluated["metric_value"]
                    == grid_evaluated["metric_value"].max()
                ].index.to_list()[0]
            elif self.validation_metric == 'mape' and grid_evaluated['metric_value'].isna().all():
                raise ValueError('validation metric cannot be mape when 0s are in the validation set.')
            else:
                best_params_idx = self.grid.loc[
                    grid_evaluated["metric_value"]
                    == grid_evaluated["metric_value"].min()
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
                best_params_idx, "metric_value"
            ]
        else:
            logging.warning(
                f"none of the keyword/value combos stored in the grid could be evaluated for the {self.estimator} model"
            )
            self.validation_metric_value = np.nan
            self.best_params = {}

    def manual_forecast(
        self, call_me=None, dynamic_testing=True, test_only=False, **kwargs
    ):
        """ Manually forecasts with the hyperparameters, Xvars, and normalizer selection passed as keywords.
        See https://scalecast.readthedocs.io/en/latest/Forecaster/_forecast.html.

        Args:
            call_me (str): Optional.
                What to call the model when storing it in the object's history.
                If not specified, the model's nickname will be assigned the estimator value ('mlp' will be 'mlp', etc.).
                Duplicated names will be overwritten with the most recently called model.
            dynamic_testing (bool or int):
                Whether to dynamically/recursively test the forecast (meaning AR terms will be propagated with predicted values).
                If True, evaluates dynamically over the entire out-of-sample slice of data.
                If int, window evaluates over that many steps (2 for 2-step dynamic forecasting, 12 for 12-step, etc.).
                Setting this to False or 1 means faster performance, 
                but gives a less-good indication of how well the forecast will perform more than one period out.
                The model will skip testing if the test_length attribute is set to 0.
            test_only (bool): Default False.
                Whether to stop the model after the testing process and not forecast into future periods.
                The forecast info stored in the object's history will be equivalent to test-set predictions.
                When True, any plot or export of forecasts into a future horizon will fail 
                and not all methods will raise descriptive errors.
                Will automatically change to True if object was initiated with require_future_dates = False.
            **kwargs: passed to the _forecast_{estimator}() method and can include such parameters as Xvars, 
                normalizer, cap, and floor, in addition to any given model's specific hyperparameters.
                For sklearn models, can inlcude normalizer and Xvars.
                For ARIMA, Prophet and Silverkite models, can include Xvars but not normalizer.
                LSTM and RNN models have their own sets of possible keywords.
                See https://scalecast.readthedocs.io/en/latest/Forecaster/_forecast.html.

        Returns:
            None

        >>> f.set_estimator('lasso')
        >>> f.manual_forecast(alpha=.5)
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

        if "tune" in kwargs.keys():
            kwargs.pop("tune")
            logging.warning("tune argument will be ignored")

        self._clear_the_deck()
        test_only = True if not self.require_future_dates else test_only
        self.test_only = test_only
        self.dynamic_testing = dynamic_testing
        self.call_me = self.estimator if call_me is None else call_me
        result = (
            self._forecast_sklearn(
                fcster=self.estimator,
                dynamic_testing=dynamic_testing,
                test_only=test_only,
                **kwargs,
            )
            if self.estimator in _sklearn_estimators_
            else getattr(self, f"_forecast_{self.estimator}")(
                dynamic_testing=dynamic_testing, test_only=test_only, **kwargs
            )
        )  # 0 - forecast, 1 - fitted vals, 2 - Xvars, 3 - regr
        self.forecast = pd.Series(result[0]).fillna(method='ffill').to_list()
        self.fitted_values = pd.Series(result[1]).fillna(method='ffill').to_list()
        self.Xvars = result[2]
        self.regr = result[3]
        if self.estimator in ("arima", "hwes"):
            self._set_summary_stats()

        self._bank_history(**kwargs)

    def auto_forecast(
        self,
        call_me=None,
        dynamic_testing=True,
        test_only=False,
        probabilistic=False,
        n_iter=20,
    ):
        """ Auto forecasts with the best parameters indicated from the tuning process.

        Args:
            call_me (str): Optional.
                What to call the model when storing it in the object's history dictionary.
                If not specified, the model's nickname will be assigned the estimator value ('mlp' will be 'mlp', etc.).
                Duplicated names will be overwritten with the most recently called model.
            dynamic_testing (bool or int):
                Whether to dynamically/recursively test the forecast (meaning AR terms will be propagated with predicted values).
                If True, evaluates dynamically over the entire out-of-sample slice of data.
                If int, window evaluates over that many steps (2 for 2-step dynamic forecasting, 12 for 12-step, etc.).
                Setting this to False or 1 means faster performance, 
                but gives a less-good indication of how well the forecast will perform more than one period out.
                The model will skip testing if the test_length attribute is set to 0.
            test_only (bool): Default False.
                Whether to stop the model after the testing process and not forecast into future periods.
                The forecast info stored in the object's history will be equivalent to test-set predictions.
                When True, any plot or export of forecasts into a future horizon will fail 
                and not all methods will raise descriptive errors.
                Will automatically change to True if object was initiated with require_future_dates = False.
            probabilistic (bool): Default False.
                Whether to use a probabilistic forecasting process to set confidence intervals.
            n_iter (int): Default 20.
                How many iterations to use in probabilistic forecasting. Ignored if probabilistic = False.

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
        if not probabilistic:
            self.manual_forecast(
                call_me=call_me,
                dynamic_testing=dynamic_testing,
                test_only=test_only,
                **self.best_params,
            )
        else:
            self.proba_forecast(
                call_me=call_me,
                dynamic_testing=dynamic_testing,
                test_only=test_only,
                n_iter=n_iter,
                **self.best_params,
            )

    def proba_forecast(
        self, call_me=None, dynamic_testing=True, test_only=False, n_iter=20, **kwargs
    ):
        """ Forecast with a probabilistic process where the final point estimate is an average of
        several forecast calls. Confidence intervals are overwritten through this process with a probabilistic technique.
        Level and difference confidence intervals are then possible to display. if the model in question is fundamentally
        deterministic, this approach will just waste resources.

        Args:
            call_me (str): Optional.
                What to call the model when storing it in the object's history dictionary.
                If not specified, the model's nickname will be assigned the estimator value ('mlp' will be 'mlp', etc.).
                Duplicated names will be overwritten with the most recently called model.
            dynamic_testing (bool or int):
                Whether to dynamically/recursively test the forecast (meaning AR terms will be propagated with predicted values).
                If True, evaluates dynamically over the entire out-of-sample slice of data.
                If int, window evaluates over that many steps (2 for 2-step dynamic forecasting, 12 for 12-step, etc.).
                Setting this to False or 1 means faster performance, 
                but gives a less-good indication of how well the forecast will perform more than one period out.
                The model will skip testing if the test_length attribute is set to 0.
            test_only (bool): Default False.
                Whether to stop the model after the testing process and not forecast into future periods.
                The forecast info stored in the object's history will be equivalent to test-set predictions.
                When True, any plot or export of forecasts into a future horizon will fail 
                and not all methods will raise descriptive errors.
                Will automatically change to True if object was initiated with require_future_dates = False.
            n_iter (int): Default 20.
                The number of forecast calls to use when creating the final point estimate and confidence intervals.
                Increasing this gives more sound results but costs resources.
            **kwargs: passed to the _forecast_{estimator}() method.
                can include lags and normalizer in addition to any given model's specific hyperparameters.

        Returns:
            None

        >>> f.set_estimator('mlp')
        >>> f.proba_forecast(hidden_layer_sizes=(25,25,25))
        """
        estimator = self.estimator
        call_me = self.estimator if call_me is None else call_me
        f = self.__deepcopy__()
        for i in range(n_iter):
            f.manual_forecast(
                call_me=f"{estimator}{i}",
                dynamic_testing=dynamic_testing,
                test_only=test_only,
                **kwargs,
            )
        f.set_estimator("combo")
        f.manual_forecast(
            how="simple",
            models=[f"{estimator}{i}" for i in range(n_iter)],
            call_me="final",
        )

        # set history attr
        # most everything we need is going to be equal to the last estimator called
        self.history[call_me] = f.history[f"{estimator}{i}"]

        # these ones will come from the combo model
        for attr in (
            "Forecast",
            "TestSetPredictions",
            "TestSetRMSE",
            "TestSetMAPE",
            "TestSetMAE",
            "TestSetR2",
            "LevelForecast",
            "LevelTestSetPreds",
            "LevelTestSetRMSE",
            "LevelTestSetMAPE",
            "LevelTestSetMAE",
            "LevelTestSetR2",
        ):
            self.history[call_me][attr] = f.history["final"][attr]

        # this is how we make the confidence intervals
        attr_set_map = {
            "UpperCI": "Forecast",
            "LowerCI": "Forecast",
            "TestSetUpperCI": "TestSetPredictions",
            "TestSetLowerCI": "TestSetPredictions",
            "LevelUpperCI": "LevelForecast",
            "LevelLowerCI": "LevelForecast",
            "LevelTSUpperCI": "LevelTestSetPreds",
            "LevelTSLowerCI": "LevelTestSetPreds",
        }

        for i, kv in enumerate(attr_set_map.items()):
            if i % 2 == 0:
                fcsts = np.array(
                    [f.history[f"{estimator}{i}"][kv[1]] for i in range(n_iter)]
                )
                self.history[call_me][kv[0]] = [
                    fcst_step + _set_ci_step(f=self,s=fcsts.std(axis=0)[idx],)
                    for idx, fcst_step in enumerate(fcsts.mean(axis=0))
                ]
            else:
                self.history[call_me][kv[0]] = [
                    fcst_step - _set_ci_step(f=self,s=fcsts.std(axis=0)[idx],)
                    for idx, fcst_step in enumerate(fcsts.mean(axis=0))
                ]
        self.history[call_me]['CIPlusMinus'] = None

    def reeval_cis(self, method = 'naive', models='all', n_iter=10, jump_back=1):
        """ Generates a dynamic confidence interval for passed models.
        See https://scalecast-examples.readthedocs.io/en/latest/misc/cis/cis.html.

        Args:
            method (str): One of "naive", "backtest". Default "naive".
                If "naive", calculates confidence intervals by determining the standard deviation of each point of
                every evaluated model passed to the function. 
                Time-steps that all models estimated closer together will receive smaller
                intervals and steps where the models diverge significantly receive larger intervals.
                This is a computationally cheap method but if all models perform similarly poorly, it can generate tight intervals, which is 
                a downside. It is recommended to use at least 3 models with this method.
                If "backtest", calculates confidence intervals by backtesting each model and determining the standard deviation of the out-of-sample
                Residual of each forecast step for both test-set predictions and actuals.
                This is a generalizable way to obtain trustworthy confidence intervals for all models, but it is more computationally expensive,
                It does not work for objects that were transformed and then reverted. it also does not work for combo models.
                It is recommended to set n_iter to at least 3 with this method.
            models (str or list-like): Default 'all'. The models to regenerate cis for. 
                Recommended to have at least three for method = 'naive'.
            n_iter (int): Default 10. The number of iterations to backtest. Recommended to be at least 3 for method = 'backtest'. 
                Models will iteratively train on all data before the fcst_length worth of values. 
                Each iteration takes observations (this number is determined by the value passed to the jump_back arg) 
                off the end to redo the cast until all of n_iter is exhausted. Ignored when method == 'naive'.
            jump_back (int): Default 1. The number of time steps between two consecutive training sets. ignored when method == 'naive'.
        Returns:
            None

        >>> from scalecast.util import pdr_load
        >>> from scalecast import GridGenerator
        >>> f = pdr_load(
        >>>    'UNRATE',
        >>>    start='2000-01-01',
        >>>    end='2022-07-01',
        >>>    src='fred',
        >>>    future_dates = 24,
        >>> )
        >>> GridGenerator.get_example_grids()
        >>> f.add_ar_terms(6)
        >>> models = ('elasticnet','mlp','arima')
        >>> f.tune_test_forecast(models)
        >>> f.reeval_cis(method = 'backtest') # creates cis based on the results from each model
        """
        models = self._parse_models(models,None)

        attr_set_map = {
            "UpperCI": "Forecast",
            "LowerCI": "Forecast",
            "TestSetUpperCI": "TestSetPredictions",
            "TestSetLowerCI": "TestSetPredictions",
            "LevelUpperCI": "LevelForecast",
            "LevelLowerCI": "LevelForecast",
            "LevelTSUpperCI": "LevelTestSetPreds",
            "LevelTSLowerCI": "LevelTestSetPreds",
        }

        if method == 'naive':
            if len(models) < 3:
                warnings.warn('reeval_cis(method="naive") does not work well with fewer than three models.')
            for m in models:
                for i, kv in enumerate(attr_set_map.items()):
                    if i % 2 == 0:
                        fcsts = np.array(
                            [self.history[m][kv[1]] for m in models]
                        )
                        self.history[m][kv[0]] = [
                            self.history[m][kv[1]][idx] + _set_ci_step(f=self,s=fcsts.std(axis=0)[idx],)
                            for idx, fcst_step in enumerate(fcsts.mean(axis=0))
                        ]
                    else:
                        self.history[m][kv[0]] = [
                            self.history[m][kv[1]][idx] - _set_ci_step(f=self,s=fcsts.std(axis=0)[idx],)
                            for idx, fcst_step in enumerate(fcsts.mean(axis=0))
                        ]
        elif method == 'backtest':
            for m in models:
                bt_length = max(len(self.history[m]['Forecast']),self.history[m]['TestSetLength'])
                try:
                    self.backtest(model=m, fcst_length=bt_length, n_iter=n_iter, jump_back=jump_back)
                except ForecastError:
                    raise ForecastError(
                        'backtested confidence intervals do not work when object was transformed and then reverted.'
                        ' if object was not transformed and then reverted, raise an issue: https://github.com/mikekeith52/scalecast/issues/new.'
                    )
                results = self.export_backtest_values(m)
                resids = pd.DataFrame(columns = ['resids','stepnum'])
                for i, col in enumerate(results):
                    if i%3 == 0:
                        resids = pd.concat(
                            [
                                resids,
                                pd.DataFrame(
                                    {
                                        'resids':(results.iloc[:,i+1] - results.iloc[:,i+2]).values,
                                        'stepnum':np.arange(results.shape[0])
                                    }
                                )
                            ]
                        )
                resids_std = resids.groupby('stepnum')['resids'].std()
                for i, kv in enumerate(attr_set_map.items()):
                    vals = self.history[m][kv[1]]
                    resids_tmp = resids_std.iloc[:len(vals)]
                    if i % 2 == 0:
                        self.history[m][kv[0]] = [
                            vals[idx] + _set_ci_step(f=self,s=val)
                            for idx, val in enumerate(resids_tmp)
                        ]
                    else:
                        self.history[m][kv[0]] = [
                            vals[idx] - _set_ci_step(f=self,s=val)
                            for idx, val in enumerate(resids_tmp)
                        ]
            self.history[m]['CIPlusMinus'] = None
        else:
            raise ValueError(f'method expected one of "naive", "backtest", got {method}')

    def add_sklearn_estimator(self, imported_module, called):
        """ Adds a new estimator from scikit-learn not built-in to the forecaster object that can be called using set_estimator().
        Only regression models are accepted.
        
        Args:
            imported_module (sklearn regression model):
                The model from sklearn to add. must have already been imported locally.
                Supports models from sklearn and sklearn APIs.
            called (str):
                The name of the estimator that can be called using set_estimator().

        Returns:
            None

        >>> from sklearn.ensemble import StackingRegressor
        >>> f.add_sklearn_estimator(StackingRegressor,called='stacking')
        >>> f.set_estimator('stacking')
        >>> f.manual_forecast(...)
        """
        globals()[called + "_"] = imported_module
        _sklearn_imports_[called] = globals()[called + "_"]
        _sklearn_estimators_.append(called)
        _sklearn_estimators_.sort()
        _estimators_.append(called)
        _estimators_.sort()
        _can_be_tuned_.append(called)
        _can_be_tuned_.sort()

    def tune_test_forecast(
        self,
        models,
        cross_validate=False,
        dynamic_tuning=False,
        dynamic_testing=True,
        probabilistic=False,
        n_iter=20,
        summary_stats=False,
        feature_importance=False,
        fi_method="pfi",
        limit_grid_size=None,
        suffix=None,
        error='raise',
        **cvkwargs,
    ):
        """ Iterates through a list of models, tunes them using grids in a grids file, forecasts them, and can save feature information.

        Args:
            models (list-like):
                Each element must be in _can_be_tuned_.
            cross_validate (bool): Default False.
                Whether to tune the model with cross validation. 
                If False, uses the validation slice of data to tune.
            dynamic_tuning (bool or int): default False.
                whether to dynamically tune the forecast (meaning AR terms will be propagated with predicted values).
                if True, evaluates dynamically over the entire out-of-sample slice of data.
                if int, window evaluates over that many steps (2 for 2-step dynamic forecasting, 12 for 12-step, etc.).
                setting this to False or 1 means faster performance, 
                but gives a less-good indication of how well the forecast will perform out x amount of periods.
            dynamic_testing (bool or int):
                whether to dynamically test the forecast (meaning AR terms will be propagated with predicted values).
                if True, evaluates dynamically over the entire out-of-sample slice of data.
                if int, window evaluates over that many steps (2 for 2-step dynamic forecasting, 12 for 12-step, etc.).
                setting this to False or 1 means faster performance, 
                but gives a less-good indication of how well the forecast will perform out x amount of periods.
            probabilistic (bool, str, or list-like): Default False.
                If bool, whether to use a probabilistic forecasting process to set confidence intervals for all models.
                If str, the name of a single model to apply a probabilistic process to.
                If list-like, a list of models to apply a probabilistic process to.
            n_iter (int): Default 20.
                How many iterations to use in probabilistic forecasting. Ignored if probabilistic = False.
            summary_stats (bool): Default False.
                Whether to save summary stats for the models that offer those.
            feature_importance (bool): Default False.
                Whether to save permutation feature importance information for the models that offer those.
            fi_method (str): One of {'pfi','shap'}. Default 'pfi'.
                The type of feature importance to save for the models that support it.
                Ignored if feature_importance is False.
            limit_grid_size (int or float): Optional. Pass an argument here to limit each of the grids being read.
                See https://scalecast.readthedocs.io/en/latest/Forecaster/Forecaster.html#src.scalecast.Forecaster.Forecaster.limit_grid_size.
            suffix (str): Optional. A suffix to add to each model as it is evaluate to differentiate them when called
                later. If unspecified, each model can be called by its estimator name.
            error (str): One of 'ignore','raise','warn'; default 'raise'.
                What to do with the error if a given model fails.
                'warn' logs a warning that the model could not be evaluated.
            **cvkwargs: Passed to the cross_validate() method.

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
        for m in models:
            m_prob = (
                probabilistic if isinstance(probabilistic,bool) 
                else m == probabilistic if isinstance(probabilistic,str) 
                else m in probabilistic
            )
            call_me = m if suffix is None else m + suffix
            self.set_estimator(m)
            if limit_grid_size is not None:
                self.ingest_grid(m)
                self.limit_grid_size(limit_grid_size)
            if not cross_validate:
                self.tune(dynamic_tuning=dynamic_tuning)
            else:
                self.cross_validate(dynamic_tuning=dynamic_tuning, **cvkwargs)
            try:

                self.auto_forecast(
                    dynamic_testing=dynamic_testing,
                    call_me=call_me,
                    probabilistic=m_prob,
                    n_iter=n_iter,
                )
            except Exception as e:
                if error == 'raise':
                    raise
                elif error == 'warn':
                    warnings.warn(
                        f"{m} Model could not be evaluated. "
                        f"Here's the error: {e}."
                    )
                    continue
                elif error == 'ignore':
                    continue
                else:
                    raise ValueError(f'Value passed to error arg not recognized: {error}')

            if summary_stats:
                self.save_summary_stats()
            if feature_importance:
                self.save_feature_importance(fi_method)

    def save_feature_importance(self, method="pfi", on_error="warn"):
        """ Saves feature info for models that offer it (sklearn models).
        Call after evaluating the model you want it for and before changing the estimator.
        This method saves a dataframe listing the feature as the index and its score. This dataframe can be recalled using
        the `export_feature_importance()` method. Scores for the pfi method are the average decrease in accuracy
        over 10 permutations for each feature. For shap, it is determined as the average score applied to each
        feature in each observation.

        Args:
            method (str): One of {'pfi','shap'}.
                The type of feature importance to set.
                'pfi' supported for all sklearn model types. 
                'shap' for xgboost, lightgbm and some others.
            on_error (str): One of {'warn','raise'}. Default 'warn'.
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
        descriptive_assert(
            method in ("pfi", "shap"),
            ValueError,
            f'kind must be one of "pfi","shap", got {method}',
        )
        fail = False
        try:
            if method == "pfi":
                import eli5
                from eli5.sklearn import PermutationImportance

                perm = PermutationImportance(self.regr).fit(
                    self.X, self.y.values[: len(self.X)],
                )
                self.feature_importance = eli5.explain_weights_df(
                    perm, feature_names=self.history[self.call_me]["Xvars"]
                ).set_index("feature")
            else:
                import shap

                explainer = shap.TreeExplainer(self.regr)
                shap_values = explainer.shap_values(self.X)
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
                logging.warning(
                    f"cannot set {method} feature importance on the {self.estimator} model"
                )
            elif on_error == "raise":
                raise TypeError(str(error))
            else:
                raise ValueError(f"on_error arg not recognized: {on_error}")
            return

        self._bank_fi_to_history()

    def save_summary_stats(self):
        """ Saves summary stats for models that offer it and will not raise errors if not available.
        Call after evaluating the model you want it for and before changing the estimator.

        >>> f.set_estimator('arima')
        >>> f.manual_forecast(order=(1,1,1))
        >>> f.save_summary_stats()
        """
        if not hasattr(self, "summary_stats"):
            logging.warning(f"{self.estimator} does not have summary stats")
            return
        self._bank_summary_stats_to_history()

    def keep_smaller_history(self, n):
        """ Cuts the amount of y observations in the object.

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
        descriptive_assert(
            isinstance(n, int),
            ValueError,
            "n must be an int, datetime object, or str and there must be more than 2 observations to keep.",
        )
        descriptive_assert(
            n > 2,
            ValueError,
            "n must be an int, datetime object, or str and there must be more than 2 observations to keep.",
        )
        self.y = self.y[-n:]
        self.current_dates = self.current_dates[-n:]
        for k, v in self.current_xreg.items():
            self.current_xreg[k] = v[-n:]

    def order_fcsts(self, models, determine_best_by="TestSetRMSE"):
        """ Gets estimated forecasts ordered from best-to-worst.
        
        Args:
            models (list-like):
                Each element must match an evaluated model's nickname (which is the same as its estimator name by default).
            determine_best_by (str): Default 'TestSetRMSE'. One of _determine_best_by_.

        Returns:
            (list): The ordered models.

        >>> models = ('mlr','mlp','lightgbm')
        >>> f.tune_test_forecast(models,dynamic_testing=False,feature_importance=True)
        >>> ordered_models = f.order_fcsts(models,"LevelTestSetMAPE")
        """
        descriptive_assert(
            determine_best_by in _determine_best_by_,
            ValueError,
            f"determine_best_by must be one of {_determine_best_by_}, got {determine_best_by}.",
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
                f"the following regressors are in current_xreg but not future_xreg: {case1}\nthe following regressors are in future_xreg but not current_xreg {case2}"
            )

    def plot(
        self, 
        models="all", 
        order_by=None, 
        level=False, 
        print_attr=[], 
        ci=False,
        ax = None,
        figsize=(12,6),
    ):
        """ Plots all forecasts with the actuals, or just actuals if no forecasts have been evaluated or are selected.
        If any models passed to models were run test_only=True, will raise an error.

        Args:
            models (list-like, str, or None): Default 'all'.
               The forecasted models to plot.
               Can start with "top_" and the metric specified in order_by will be used to order the models appropriately.
               If None or models/order_by combo invalid, will plot only actual values.
            order_by (str): Optional. One of _determine_best_by_.  
                How to order the display of forecasts on the plots (from best-to-worst according to the selected metric).
            level (bool): Default False.
                If True, will always plot level forecasts.
                If False, will plot the forecasts at whatever level they were called on.
                If False and there are a mix of models passed with different integrations, will default to True.
            print_attr (list-like): Default [].
                Attributes from history to print to console.
                If the attribute doesn't exist for a passed model, will not raise error, will just skip that element.
            ci (bool): Default False.
                Whether to display the confidence intervals.
            ax (Axis): Optional. The existing axis to write the resulting figure to.
            figsize (tuple): Default (12,6). The size of the resulting figure. Ignored when ax is not None.

        Returns:
            (Axis): The figure's axis.

        >>> models = ('mlr','mlp','lightgbm')
        >>> f.tune_test_forecast(models,dynamic_testing=False,feature_importance=True)
        >>> f.plot(order_by='LevelTestSetMAPE') # plots all forecasts
        >>> plt.show()
        """
        try:
            models = self._parse_models(models, order_by)
        except (ValueError, TypeError):
            models = None

        if ax is None:
            _, ax = plt.subplots(figsize=figsize)

        if models is None:
            sns.lineplot(
                x=self.current_dates.values, y=self.y.values, label="actuals", ax=ax
            )
            plt.legend()
            plt.xlabel("Date")
            plt.ylabel("Values")
            return ax

        integration = set(
            [d["Integration"] for m, d in self.history.items() if m in models]
        ) # how many different integrations are we working with?
        if len(integration) > 1:
            level = True

        y = self.y.copy()
        if self.integration == 0 and max(integration) == 1:
            y = y.diff()
        self._validate_no_test_only(models)
        plot = {
            "date": self.current_dates.to_list()[-len(y.dropna()) :]
            if not level
            else self.current_dates.to_list()[
                -len(self.levely) :
            ],
            "actuals": y.dropna().to_list()
            if not level
            else self.levely,
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
                ax=ax,
            )
            if ci:
                plt.fill_between(
                    x=self.future_dates.to_list(),
                    y1=self.history[m]["UpperCI"]
                    if not level
                    else self.history[m]["LevelUpperCI"],
                    y2=self.history[m]["LowerCI"]
                    if not level
                    else self.history[m]["LevelLowerCI"],
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
        ax = None,
        figsize=(12,6),
    ):
        """ Plots all test-set predictions with the actuals.

        Args:
            models (list-like or str): Default 'all'.
               The forecated models to plot.
               Can start with "top_" and the metric specified in order_by will be used to order the models appropriately.
            order_by (str): Optional. One of _determine_best_by_.
                How to order the display of forecasts on the plots (from best-to-worst according to the selected metric).
            include_train (bool or int): Default True.
                Use to zoom into testing results.
                If True, plots the test results with the entire history in y.
                If False, matches y history to test results and only plots this.
                If int, plots that length of y to match to test results.
            level (bool): Default False.
                If True, always plots level forecasts.
                If False, will plot the forecasts at whatever level they were called on.
                If False and there are a mix of models passed with different integrations, will default to True.
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
        >>> f.plot(order_by='LevelTestSetMAPE') # plots all test-set results
        >>> plt.show()
        """
        descriptive_assert(
            self.test_length > 0,
            ForecastError,
            'plot_test_set() does not work when models were not tested (test_length set to 0).',
        )
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
        models = self._parse_models(models, order_by)
        integration = set(
            [d["Integration"] for m, d in self.history.items() if m in models]
        )
        if len(integration) > 1:
            level = True

        y = self.y.copy()
        if self.integration == 0 and max(integration) == 1:
            y = y.diff()

        plot = {
            "date": self.current_dates.to_list()[-len(y.dropna()) :]
            if not level
            else self.current_dates.to_list()[
                -len(self.levely) :
            ],
            "actuals": y.dropna().to_list()
            if not level
            else self.levely,
        }
        plot["actuals_len"] = min(len(plot["date"]), len(plot["actuals"]))

        if not isinstance(include_train,bool):
            include_train = int(include_train)
            descriptive_assert(
                include_train > 1,
                ValueError,
                f"include_train must be a bool type or an int greater than 1, got {include_train}",
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
                ax=ax,
            )
            if ci:
                plt.fill_between(
                    x=test_dates,
                    y1=self.history[m]["TestSetUpperCI"]
                    if not level
                    else self.history[m]["LevelTSUpperCI"],
                    y2=self.history[m]["TestSetLowerCI"]
                    if not level
                    else self.history[m]["LevelTSLowerCI"],
                    alpha=0.2,
                    color=_colors_[i],
                    label="{} {:.0%} CI".format(m, self.history[m]["CILevel"]),
                )

        plt.legend()
        plt.xlabel("Date")
        plt.ylabel("Values")
        return ax

    def plot_backtest_values(self,model,ax=None,figsize=(12,6)):
        """ Plots all backtest values over every iteration. Can only plot one model at a time.

        Args:
            model (str): The model nickname to plot the backtest values for. Must have called 
                Forecaster.backtest(model) previously.
            ax (Axis): Optional. The existing axis to write the resulting figure to.
            figsize (tuple): Default (12,6). Size of the resulting figure. Ignored when ax is not None.
        
        Returns:
            (Axis): The figure's axis.

        >>> f.set_estimator('elasticnet')
        >>> f.manual_forecast(alpha=.2)
        >>> f.backtest('elasticnet')
        >>> f.plot_backtest_values('elasticnet') # plots all backtest values
        >>> plt.show()
        """
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
        values = self.export_backtest_values(model)
        y = self.levely[:]
        dates = self.current_dates.to_list()
        ac_len = min(len(y),len(dates))

        sns.lineplot(
            x = dates[-ac_len:],
            y = y[-ac_len:],
            label="actuals",
            ax=ax,
        )

        for i, col in enumerate(values):
            if i % 3 == 0:
                sns.lineplot(
                    x = values.iloc[:,i], # dates
                    y = values.iloc[:,i+2], # predictions
                    label = f'iter {i//3+1}',
                    ax = ax,
                    color=_colors_[i//3],
                    alpha = 0.7,
                )

        plt.legend()
        plt.xlabel("Date")
        plt.ylabel("Values")
        return ax

    def plot_fitted(
        self, 
        models="all", 
        order_by=None, 
        level=False,
        ax = None,
        figsize=(12,6),
    ):
        """ Plots all fitted values with the actuals. Does not support level fitted values (for now).

        Args:
            models (list-like,str): Default 'all'.
               The forecated models to plot.
               Can start with "top_" and the metric specified in order_by will be used to order the models appropriately.
            order_by (str): Optional. One of _determine_best_by_.
                How to order the display of forecasts on the plots (from best-to-worst according to the selected metric).
            level (bool): Default False.
                If True, always plots level forecasts.
                If False, plots the forecasts at whatever level they were called on.
                If False and there are a mix of models passed with different integrations, defaults to True.
            ax (Axis): Optional. The existing axis to write the resulting figure to.
            figsize (tuple): Default (12,6). Size of the resulting figure. Ignored when ax is not None.

        Returns:
            (Axis): The figure's axis.

        >>> models = ('mlr','mlp','lightgbm')
        >>> f.tune_test_forecast(models,dynamic_testing=False,feature_importance=True)
        >>> f.plot_fitted(order_by='LevelTestSetMAPE') # plots all fitted values
        >>> plt.show()
        """
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
        models = self._parse_models(models, order_by)
        integration = set(
            [d["Integration"] for m, d in self.history.items() if m in models]
        )
        if len(integration) > 1:
            level = True

        dates = self.current_dates.to_list() if not level else self.init_dates
        actuals = self.y.to_list() if not level else self.levely

        plot = {
            "date": dates,
            "actuals": actuals,
        }
        sns.lineplot(x=plot["date"], y=plot["actuals"], label="actuals", ax=ax)

        for i, m in enumerate(models):
            plot[m] = (
                self.history[m]["FittedVals"]
                if not level
                else self.history[m]["LevelFittedVals"]
            )
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

    def drop_regressors(self, *args, error = 'raise'):
        """ Drops regressors.

        Args:
            *args (str): The names of regressors to drop.
            error (str): One of 'ignore','raise'. Default 'raise'.
                What to do with the error if the Xvar is not found in the object.

        Returns:
            None

        >>> f.add_time_trend()
        >>> f.add_exp_terms('t',pwr=.5)
        >>> f.drop_regressors('t','t^0.5')
        """
        for a in args:
            if a not in self.current_xreg:
                if error == 'raise':
                    raise ForecastError(f'cannot find {a} in Forecaster object')
                elif error == 'ignore':
                    continue
                else:
                    raise ValueError(f'arg passed to error not recognized: {error}')
            self.current_xreg.pop(a)
            self.future_xreg.pop(a)

    def drop_Xvars(self, *args, error = 'raise'):
        """ Drops regressors.

        Args:
            *args (str): The names of regressors to drop.
            error (str): One of 'ignore','raise'. Default 'raise'.
                What to do with the error if the Xvar is not found in the object.

        Returns:
            None

        >>> f.add_time_trend()
        >>> f.add_exp_terms('t',pwr=.5)
        >>> f.drop_Xvars('t','t^0.5')
        """
        self.drop_regressors(*args)

    def drop_all_Xvars(self):
        """ drops all regressors.
        """
        self.drop_Xvars(*self.get_regressor_names())

    def pop(self, *args):
        """ Deletes evaluated forecasts from the object's memory.

        Args:
            *args (str): Names of models matching what was passed to call_me when model was evaluated.

        >>> models = ('mlr','mlp','lightgbm')
        >>> f.tune_test_forecast(models,dynamic_testing=False,feature_importance=True)
        >>> f.pop('mlr')
        """
        for a in args:
            self.history.pop(a)

    def pop_using_criterion(self, metric, evaluated_as, threshold, delete_all=True):
        """ Deletes all forecasts from history that meet a given criterion.

        Args:
            metric (str): One of _determine_best_by_ + ['AnyPrediction','AnyLevelPrediction'].
            evaluated_as (str): One of {"<","<=",">",">=","=="}.
            threshold (float): The threshold to compare the metric and operator to.
            delete_all (bool): Default True.
                If the passed criterion deletes all forecasts, whether to actually delete all forecasts.
                If False and all forecasts meet criterion, will keep them all.

        Returns:
            None

        >>> models = ('mlr','mlp','lightgbm')
        >>> f.tune_test_forecast(models,dynamic_testing=False,feature_importance=True)
        >>> f.pop_using_criterion('LevelTestSetMAPE','>',2)
        >>> f.pop_using_criterion('AnyPrediction','<',0,delete_all=False)
        """
        descriptive_assert(
            evaluated_as in ("<", "<=", ">", ">=", "=="),
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
            "model_summaries",
            "all_fcsts",
            "test_set_predictions",
            "lvl_test_set_predictions",
            "lvl_fcsts",
        ],
        models="all",
        best_model="auto",
        determine_best_by="TestSetRMSE",
        cis=False,
        to_excel=False,
        out_path="./",
        excel_name="results.xlsx",
    ) -> Union[Dict[str, pd.DataFrame], pd.DataFrame]:
        """ Exports 1-all of 5 pandas dataframes. Can write to excel with each dataframe on a separate sheet.
        Will return either a dictionary with dataframes as values (df str arguments as keys) or a single dataframe if only one df is specified.

        Args:
            dfs (list-like or str): Default 
                ['all_fcsts','model_summaries','best_fcst','test_set_predictions','lvl_test_set_predictions','lvl_fcsts'].
                A list or name of the specific dataframe(s) you want returned and/or written to excel.
                Must be one of or multiple of the elements in default.
            models (list-like or str): Default 'all'.
                The models to write information for.
                Can start with "top_" and the metric specified in `determine_best_by` will be used to order the models appropriately.
            best_model (str): Default 'auto'.
                The name of the best model, if "auto", will determine this by the metric in determine_best_by.
                If not "auto", must match a model nickname of an already-evaluated model.
            determine_best_by (str): One of _determine_best_by_. Default 'TestSetRMSE'.
            to_excel (bool): Default False.
                Whether to save to excel.
            out_path (str): Default './'.
                The path to save the excel file to (ignored when `to_excel=False`).
            cis (bool): Default False.
                Whether to export confidence intervals for models in 
                "all_fcsts", "test_set_predictions", "lvl_test_set_predictions", "lvl_fcsts"
                dataframes.
            excel_name (str): Default 'results.xlsx'.
                The name to call the excel file (ignored when `to_excel=False`).

        Returns:
            (DataFrame or Dict[str,DataFrame]): either a single pandas dataframe if one element passed to dfs 
            or a dictionary where the keys match what was passed to dfs and the values are dataframes. 

        >>> results = f.export(dfs=['model_summaries','lvl_fcsts'],to_excel=True) # returns a dict
        >>> model_summaries = results['model_summaries'] # returns a dataframe
        >>> lvl_fcsts = results['lvl_fcsts'] # returns a dataframe
        >>> ts_preds = f.export('test_set_predictions') # returns a dataframe
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
        determine_best_by = determine_best_by if best_model == "auto" else None
        models = self._parse_models(models, determine_best_by)
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
        best_fcst_name = (
            self.order_fcsts(models, determine_best_by)[0]
            if best_model == "auto"
            else best_model
        )
        output = {}
        if "model_summaries" in dfs:
            cols = [
                "ModelNickname",
                "Estimator",
                "Xvars",
                "HyperParams",
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
                "LevelInSampleRMSE",
                "LevelInSampleMAPE",
                "LevelInSampleMAE",
                "LevelInSampleR2",
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
                    elif c == "best_model":
                        model_summary_m[c] = m == best_fcst_name
                model_summaries = pd.concat(
                    [model_summaries, model_summary_m], ignore_index=True
                )
            output["model_summaries"] = model_summaries
        if "all_fcsts" in dfs:
            all_fcsts = pd.DataFrame({"DATE": self.future_dates.to_list()})
            for m in self.history.keys():
                all_fcsts[m] = self.history[m]["Forecast"]
                if cis:
                    all_fcsts[m + "_upperci"] = self.history[m]["UpperCI"]
                    all_fcsts[m + "_lowerci"] = self.history[m]["LowerCI"]
            output["all_fcsts"] = all_fcsts
        if "test_set_predictions" in dfs:
            if self.test_length == 0:
                output["test_set_predictions"] = pd.DataFrame()
            else:
                test_set_predictions = pd.DataFrame(
                    {"DATE": self.current_dates[-self.test_length :]}
                )
                test_set_predictions["actual"] = self.y.to_list()[-self.test_length :]
                for m in models:
                    test_set_predictions[m] = self.history[m]["TestSetPredictions"]
                    if cis:
                        test_set_predictions[m + "_upperci"] = self.history[m][
                            "TestSetUpperCI"
                        ]
                        test_set_predictions[m + "_lowerci"] = self.history[m][
                            "TestSetLowerCI"
                        ]
                output["test_set_predictions"] = test_set_predictions
        if "lvl_fcsts" in dfs:
            self._validate_no_test_only(models)
            lvl_fcsts = pd.DataFrame({"DATE": self.future_dates.to_list()})
            for m in models:
                if "LevelForecast" in self.history[m].keys():
                    lvl_fcsts[m] = self.history[m]["LevelForecast"]
                    if cis and "LevelUpperCI" in self.history[m].keys():
                        lvl_fcsts[m + "_upperci"] = self.history[m]["LevelUpperCI"]
                        lvl_fcsts[m + "_lowerci"] = self.history[m]["LevelLowerCI"]
            if lvl_fcsts.shape[1] > 1:
                output["lvl_fcsts"] = lvl_fcsts
        if "lvl_test_set_predictions" in dfs:
            if self.test_length == 0:
                output["lvl_test_set_predictions"] = pd.DataFrame()
            else:
                test_set_predictions = pd.DataFrame(
                    {"DATE": self.current_dates[-self.test_length :]}
                )
                test_set_predictions["actual"] = self.levely[-self.test_length :]
                for m in models:
                    test_set_predictions[m] = self.history[m]["LevelTestSetPreds"]
                    if cis and "LevelTSUpperCI" in self.history[m].keys():
                        test_set_predictions[m + "_upperci"] = self.history[m][
                            "LevelTSUpperCI"
                        ]
                        test_set_predictions[m + "_lowerci"] = self.history[m][
                            "LevelTSLowerCI"
                        ]
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

    def export_validation_grid(self, model) -> pd.DataFrame:
        """ Exports the validation grid from a model.
        Raises an error if the model was not tuned.

        Args:
            model (str):
                The name of them model to export for.
                Matches what was passed to call_me when evaluating the model.
        Returns:
            (DataFrame): The resulting validation grid of the evaluated model passed to model arg.
        """
        return self.history[model]["grid_evaluated"]

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
                "no saved feature importance or summary stats could be found"
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
        """ Returns an object equivalent to the original state when initiated.

        Returns:
            (Forecaster): The original object.

        >>> f = Forecaster()
        >>> f.add_time_trend()
        >>> f1 = f.reset()
        """
        return f_init_

    def export_Xvars_df(self, dropna=False):
        """ Gets all utilized regressors and values.
            
        Args:
            dropna (bool): Default False.
                Whether to drop null values from the resulting dataframe

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
                        pd.DataFrame(self.current_xreg).reset_index(drop=True),
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

        if not self.require_future_dates:
            df = df.loc[df["DATE"].isin(self.current_dates.values)]

        return df.dropna() if dropna else df

    def export_fitted_vals(self, model, level=False):
        """ Exports a single dataframe with dates, fitted values, actuals, and residuals for one model.

        Args:
            model (str):
                The model nickname.
            level (bool): Default False.
                Whether to extract level fitted values

        Returns:
            (DataFrame): A dataframe with dates, fitted values, actuals, and residuals.
        """
        df = pd.DataFrame(
            {
                "DATE": (
                    self.current_dates.to_list()[
                        -len(self.history[model]["FittedVals"]) :
                    ]
                    if not level
                    else self.current_dates.to_list()[
                        -len(self.history[model]["LevelFittedVals"]) :
                    ]
                ),
                "Actuals": (
                    self.y.to_list()[-len(self.history[model]["FittedVals"]) :]
                    if not level
                    else self.levely[-len(self.history[model]["LevelFittedVals"]) :]
                ),
                "FittedVals": (
                    self.history[model]["FittedVals"]
                    if not level
                    else self.history[model]["LevelFittedVals"]
                ),
            }
        )

        df["Residuals"] = df["Actuals"] - df["FittedVals"]
        return df

    def backtest(self, model, fcst_length="auto", n_iter=10, jump_back=1):
        """ Runs a backtest of a selected evaluated model over a certain 
        amount of iterations to test the average error if that model were 
        implemented over the last so-many actual forecast intervals.
        All scoring is dynamic to give a true out-of-sample result.
        All metrics are specific to level data.
        Two results are extracted: a dataframe of actuals and predictions across each iteration and
        a dataframe of test-set metrics across each iteration with a mean total as the last column.
        These results are stored in the Forecaster object's history and can be extracted by calling 
        `f.export_backtest_metrics()` and `f.export_backtest_values()`.
        combo models cannot be backtest and will raise an error if you attempt to do so.
        Do not backtest a model after the series has been transformed/reverted 
        after the model you want to backtest was evaluated.

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

        >>> f.set_estimator('mlr')
        >>> f.manual_forecast()
        >>> f.backtest('mlr')
        >>> backtest_metrics = f.export_backtest_metrics('mlr')
        >>> backetest_values = f.export_backtest_values('mlr')
        """
        if fcst_length == "auto":
            fcst_length = len(self.future_dates)
        fcst_length = int(fcst_length)
        metric_results = pd.DataFrame(
            columns=[f"iter{i}" for i in range(1, n_iter + 1)],
            index=["RMSE", "MAE", "R2", "MAPE"],
        )
        value_results = pd.DataFrame()
        for i in range(n_iter):
            f = self.__deepcopy__()
            if i > 0:
                f.current_xreg = {
                    k: pd.Series(v.values[: -i * jump_back])
                    for k, v in f.current_xreg.items()
                }
                f.current_dates = pd.Series(f.current_dates.values[: -i * jump_back])
                f.y = pd.Series(f.y.values[: -i * jump_back])
                f.levely = f.levely[: -i * jump_back]

            f.set_test_length(fcst_length)
            f.set_estimator(f.history[model]["Estimator"])
            descriptive_assert(
                f.estimator != "combo", ValueError, "combo models cannot be backtest"
            )
            params = f.history[model]["HyperParams"].copy()
            if f.history[model]["Xvars"] is not None:
                params["Xvars"] = f.history[model]["Xvars"][:]
            if f.estimator in _sklearn_estimators_:
                params["normalizer"] = f.history[model]["Scaler"]
            f.history = {}
            f.manual_forecast(**params, test_only=True)
            test_mets = f.export("model_summaries")
            test_preds = f.export("lvl_test_set_predictions")
            metric_results.loc["RMSE", f"iter{i+1}"] = test_mets[
                "LevelTestSetRMSE"
            ].values[0]
            metric_results.loc["MAE", f"iter{i+1}"] = test_mets[
                "LevelTestSetMAE"
            ].values[0]
            metric_results.loc["R2", f"iter{i+1}"] = test_mets["LevelTestSetR2"].values[
                0
            ]
            metric_results.loc["MAPE", f"iter{i+1}"] = test_mets[
                "LevelTestSetMAPE"
            ].values[0]
            value_results[f"iter{i+1}dates"] = test_preds['DATE'].values.copy()
            value_results[f"iter{i+1}actuals"] = test_preds.iloc[:, 1].values.copy()
            value_results[f"iter{i+1}preds"] = test_preds.iloc[:, 2].values.copy()

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
        The DataFrame will return columns for the date (first), actuals (second), and predictions (third)
        across every backtest iteration (10 by default).

        Args:
            model (str): The model nickname to extract values for.

        Returns:
            (DataFrame): A copy of the backtest values.
        """
        return self.history[model]["BacktestValues"].copy()

# Other Globals
def mape(y, pred):
    return (
        np.nan if np.abs(y).min() == 0 
        else mean_absolute_percentage_error(y, pred)
    )

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

def _return_na_if_len_zero(y,pred,func):
    return np.nan if len(pred) == 0 else func(y,pred)

def _set_ci_step(f,s):
    return stats.norm.ppf(1 - (1 - f.cilevel) / 2) * s

def _convert_m(m,freq):
    if m == 'auto':
        if freq is not None:
            if freq.startswith('M'):
                return 12
            elif freq.startswith('Q'):
                return 4
            elif freq.startswith('H'):
                return 24
            else:
                return 1
        else:
            return 1
    return m

# estimators
_sklearn_estimators_ = sorted(_sklearn_imports_.keys())
# to add non-sklearn models, add to the list below
# sklearn estimators go to the _sklearn_imports_ dict at the top
_non_sklearn_estimators_ = [
    "arima",
    "hwes",
    "prophet",
    "silverkite",
    "rnn",
    "lstm",
    'naive',
    "theta",
    "combo",
]
_estimators_ = sorted(_sklearn_estimators_ + _non_sklearn_estimators_)
_cannot_be_tuned_ = ["combo", "rnn", "lstm", "naive"]
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
    "LevelInSampleRMSE",
    "LevelInSampleMAPE",
    "LevelInSampleMAE",
    "LevelInSampleR2",
    "ValidationMetricValue",
    "LevelTestSetRMSE",
    "LevelTestSetMAPE",
    "LevelTestSetMAE",
    "LevelTestSetR2",
]
_normalizer_ = ["minmax", "normalize", "scale", None]
# i do it this way to make mvforecaster work a little better
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
# keywords that are passed to _bank_history() that I don't want to be recognized as hyperparams
_not_hyperparams_ = ["Xvars", "normalizer", "tuned", "plot_loss", "plot_loss_test"]
# descriptive errors
class ForecastError(Exception):
    pass