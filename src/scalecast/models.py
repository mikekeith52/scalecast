from __future__ import annotations
from ._utils import _developer_utils
from .util import find_seasonal_length
from .typing_utils import ScikitLike, NormalizerLike
from .types import (
    DynamicTesting, 
    XvarValues, 
    PositiveInt, 
    ConfInterval, 
    ModelValues, 
    DetermineBestBy, 
)
from .classes import AR
from typing import TYPE_CHECKING, Self, Optional, Any, Literal, Sequence
import pandas as pd
import numpy as np
if TYPE_CHECKING:
    from .Forecaster import Forecaster
    from .MVForecaster import MVForecaster

class SKLearnUni:
    """ Model class that supports any scikit-learn API estimator for univariate forecasting.

    Args:
        f (Forecaster): The Forecaster object storing the actual series and associated dates.
        model (Scikit-learn API Estimator): The imported scikit-learn API regression estimator/class (such as LinearRegressor or XGBRegressor).
        dynamic_testing (bool or int): Whether to dynamically test the model or how many steps. Ignored when test_set_actuals not specified.
        Xvars (list[str]): List of regressors to use from the passed Forecaster object.
        normalizer (NormalizerLike): Default 'minmax'. The label of the normalizer to use.
        test_set_actuals (list[float]): Optional. Test-set actuals to use for testing the model. This enables the dynamic_testing option.
        **kwargs: Passed to the scikit-learn model passed to model.
    """
    def __init__(
        self, 
        f:'Forecaster', 
        model:ScikitLike, 
        dynamic_testing:DynamicTesting = True, 
        Xvars:XvarValues = None, 
        normalizer:NormalizerLike = 'minmax', 
        test_set_actuals:Optional[list[float]]=None,
        **kwargs:Any,
    ):
        self.regr = model(**kwargs)
        self.f = f
        self.normalizer = normalizer
        self.scaler = self.f.lookup_normalizer(normalizer)()
        self.dynamic_testing = self._parse_dynamic_testing(dynamic_testing)
        self.Xvars = self._parse_Xvars(Xvars)
        self.max_lag_order = self._determine_max_lag_order(self.Xvars)
        self.test_set_actuals = test_set_actuals
        self.current_actuals = f.y.to_list()
            
    def _parse_dynamic_testing(self,dynamic_testing) -> int:
        steps = len(self.f.future_dates)
        match dynamic_testing:
            case False|None:
                return 1
            case True:
                return steps+1
            case i if i <= 0:
                raise ValueError(f'Invalid value passed to dynamic_testing: {dynamic_testing}')
            case _:
                return dynamic_testing
            
    def _determine_max_lag_order(self,Xvars):
        lag_orders = [x.lag_order for x in Xvars if isinstance(x,AR)]
        if lag_orders:
            return max(lag_orders)
        return 0
    
    def _parse_Xvars(self, Xvars):
        match Xvars:
            case 'all'|None:
                return list(self.f.current_xreg.keys())
            case _:
                return list(Xvars)

    def generate_current_X(self) -> np.ndarray:
        """ Returns the matrix of the current input dataset.
        """
        obs_to_drop = self.max_lag_order
        X = np.array([self.f.current_xreg[x].values[obs_to_drop:].copy() for x in self.Xvars]).T
        self.scaler = self.scaler.fit(X)
        return X

    def generate_future_X(self) -> np.ndarray:
        """ Returns the matrix of the future input dataset. 
        """
        X = np.array([np.array(self.f.future_xreg[x][:]) for x in self.Xvars]).T
        return X

    @_developer_utils.log_warnings
    def fit(self,X:np.ndarray,y:np.ndarray,**fit_params:Any) -> Self:
        """ Fits the estimator.

        Args:
            X (np.ndarray): The input data.
            y (np.ndarray): The observed actuals.
            **fit_params: Passed to the .fit() method from the scikit-learn model.

        Returns:
            Self
        """
        obs_to_drop = self.max_lag_order
        X = self.scaler.transform(X)
        self.regr.fit(X,np.asarray(y)[obs_to_drop:],**fit_params)
        return self

    def predict(self,X:np.ndarray,in_sample:bool=False,**predict_params:Any) -> list[float]:
        """ Makes predictions.

        Args:
            X (np.ndarray): The input data.
            in_sample (bool): Default False. If True, returns fitted values with a one-step ahead forecast.
            **predict_params: Passed to the estimator's predict() method.
        
        Returns:
            list[float]: The predictions.
        """
        if self.max_lag_order == 0 or in_sample:
            X = self.scaler.transform(X)
            return list(self.regr.predict(X,**predict_params))
        
        if self.test_set_actuals:
            peeks = [a if (i+1)%self.dynamic_testing == 0 else np.nan for i, a in enumerate(self.test_set_actuals)]
        else:
            peeks = []
        
        series = self.current_actuals # this is used to add peeking to the models

        preds = [] # this is used to produce the real predictions
        n_steps, n_feat = X.shape
        for i in range(n_steps):
            p = self.scaler.transform(X[i,:].reshape(1,n_feat))
            pred = self.regr.predict(p,**predict_params)[0]
            preds.append(pred)
            if peeks and not np.isnan(peeks[i]):
                series.append(peeks[i])
            else:
                series.append(pred)

            if i == (n_steps-1):
                break

            for pos, x in enumerate(self.Xvars):
                if isinstance(x,AR):
                    X[i+1,pos] = series[-x.lag_order]

        return list(preds)

    def fit_predict(self,X:np.ndarray,y:np.ndarray) -> list[float]:
        """ Runs fit and predict methods, returning predictions.

        Args:
            X (np.ndarray): The input data.
            y (np.ndarray): The observed actuals.

        Returns:
            list[float]: The predictions.
        """
        self.fit(X,y)
        return self.predict(X)
    
class SKLearnMV:
    """ Model class that supports any scikit-learn API estimator for multivariate forecasting.

    Args: 
        f (Forecaster): The Forecaster object storing the actual series and associated dates.
        model (Scikit-learn API Estimator): The imported scikit-learn API regression estimator/class (such as LinearRegressor or XGBRegressor).
        lags (None or int or list[int] or dict[str,int or list[int]]): The number of lags to add to the model.
            If int, that many lags added to every model.
            If a list of ints, only the lags in the list are added.
            If dict, key is a series name and value is int or list of ints that follows the behavior descrbied above, but only targeting passed series.
        dynamic_testing (bool or int): Whether to dynamically test the model or how many steps. Ignored when test_set_actuals not specified.
        Xvars (list[str]): List of regressors to use from the passed Forecaster object.
        normalizer (NormalizerLike): Default 'minmax'. The label of the normalizer to use.
        test_set_actuals (list[float]): Optional. Test-set actuals to use for testing the model. This enables the dynamic_testing option.
        **kwargs: Passed to the scikit-learn model passed to model.
    """
    def __init__(
        self,
        f:'MVForecaster', 
        model:ScikitLike, 
        lags:None|int|list[int]|dict[str,int|list[int]]=1,
        dynamic_testing:DynamicTesting = True, 
        Xvars:XvarValues = 'all', 
        normalizer:NormalizerLike = 'minmax', 
        test_set_actuals:Optional[dict[str,list[float]]]=None,
        **kwargs:Any,
    ):
        self.regr = {label:model(**kwargs) for label in f.y}
        self.f = f
        self.normalizer = normalizer
        self.lags = self._parse_lags(lags)
        self.scaler = self.f.lookup_normalizer(normalizer)()
        self.dynamic_testing = self._parse_dynamic_testing(dynamic_testing)
        self.Xvars = self._parse_Xvars(Xvars)
        self.test_set_actuals = test_set_actuals
        self.current_actuals = f.y

    def _parse_dynamic_testing(self,dynamic_testing) -> int:
        steps = len(self.f.future_dates)
        match dynamic_testing:
            case False|None:
                return 1
            case True:
                return steps+1
            case i if i <= 0:
                raise ValueError(f'Invalid value passed to dynamic_testing: {dynamic_testing}')
            case _:
                return dynamic_testing

    def _parse_lags(self,lags):
        match lags:
            case None|0:
                return 0
            case str():
                raise ValueError(f'Unrecognized value passed to lags: {lags}')
            case float():
                return int(lags)
            case _:
                return lags

    def _parse_Xvars(self,Xvars):
        match Xvars:
            case 'all':
                return list(self.f.current_xreg.keys())
            case None:
                return []
            case _:
                return list(Xvars)
            
    def _generate_X(self):
        self.predict_with_Xvars = self.Xvars[:]
        observed = np.array([self.f.current_xreg[x].values.copy() for x in self.Xvars]).T
        future = np.array([np.array(self.f.future_xreg[x][:]) for x in self.Xvars]).T

        ylen = len(self.f.y[self.f.names[0]])
        
        no_other_xvars = not observed.shape[0]
        if no_other_xvars:
            observed_future = np.array([0]*(ylen + len(self.f.future_dates))).reshape(-1,1) # column of 0s
            observed = np.array([0]*ylen).reshape(-1,1)
        else:
            observed_future = np.concatenate([observed,future],axis=0)

        if not self.lags: # handle none
            observedy = np.array([v.to_list() for _, v in self.f.y.items()]).T
            futurey = np.zeros((len(self.f.future_dates),self.f.n_series))
            if no_other_xvars:
                observed = observedy
                future = futurey
            else:
                observed = np.concatenate([observedy,observed],axis=1)
                future = np.concatenate([futurey,future],axis=1)
            
            self.current_X = observed
            self.future_X = future
            return observed, future
        elif isinstance(self.lags,int): # handle int case
            max_lag = self.lags
            lag_matrix = np.zeros((observed_future.shape[0],max_lag*self.f.n_series))

            pos = 0
            for i in range(self.f.n_series):
                for j in range(self.lags):
                    self.predict_with_Xvars.append(f'LAG_{self.f.names[i]}_{j+1}') # LAG_UTUR_1 for first lag to keep track of position
                    lag_matrix[:,pos] = (
                        [np.nan] * (j+1)
                        + self.f.y[self.f.names[i]].to_list() 
                        + [np.nan] * (lag_matrix.shape[0] - ylen - (j+1)) # pad with nas
                    )[:lag_matrix.shape[0]] 
                    pos += 1
        elif isinstance(self.lags,dict): # handle dict case
            total_lags = 0
            for _, lag_val in self.lags.items():
                local_lags = self._parse_lags(lag_val)
                if hasattr(local_lags,'__len__'):
                    total_lags += len(local_lags)
                else:
                    total_lags += local_lags
            lag_matrix = np.zeros((observed_future.shape[0],total_lags))
            pos = 0
            max_lag = 1
            
            for label, lag_val in self.lags.items():
                if hasattr(lag_val,'__len__'):
                    for i in lag_val:
                        lag_matrix[:,pos] = (
                            [np.nan] * i
                            + self.f.y[k].to_list()
                            + [np.nan]
                            * (lag_matrix.shape[0] - ylen - i)
                        )[:lag_matrix.shape[0]] 
                        self.predict_with_Xvars.append(f'LAG_{label}_{i}')
                        pos+=1
                    max_lag = max(max_lag,max(lag_val))
                else:
                    for i in range(lag_val):
                        lag_matrix[:,pos] = (
                            [np.nan] * (i+1)
                            + self.f.y[k].to_list()
                            + [np.nan]
                            * (lag_matrix.shape[0] - ylen - (i+1))
                        )[:lag_matrix.shape[0]] 
                        self.predict_with_Xvars.append(f'LAG_{label}_{i+1}')
                        pos+=1
                    max_lag = max(max_lag,lag_val)
        
        else:
            lag_matrix = np.zeros((observed_future.shape[0],len(self.lags)*self.f.n_series))
            pos = 0
            max_lag = max(self.lags)
            for i in range(self.f.n_series):
                for lag in self.lags:
                    self.predict_with_Xvars.append(f'LAG_{self.f.names[i]}_{lag}')
                    lag_matrix[:,pos] = (
                        [np.nan] * lag
                        + self.f.y[self.f.names[i]].to_list()
                        + [np.nan] * (lag_matrix.shape[0] - ylen - lag)
                    )[:lag_matrix.shape[0]]
                    pos+=1

        observed_future = np.concatenate([observed_future,lag_matrix],axis=1)

        if no_other_xvars:
            start_col = 1
        else:
            start_col = 0

        future = observed_future[observed.shape[0]:,start_col:]
        observed = observed_future[max_lag:observed.shape[0],start_col:]

        self.current_X = observed
        self.future_X = future

        return observed, future

    def generate_current_X(self) -> np.ndarray:
        """ Returns the matrix of the current input dataset.
        """
        X = self._generate_X()[0]
        self.scaler = self.scaler.fit(X)
        return X

    def generate_future_X(self) -> np.ndarray:
        """ Returns the matrix of the future input dataset. 
        """       
        return self._generate_X()[1]

    @_developer_utils.log_warnings
    def fit(self,X,y,**fit_params):
        """ Fits the estimator.

        Args:
            X (np.ndarray): The input data.
            y (np.ndarray): The observed actuals.
            **fit_params: Passed to the .fit() method from the scikit-learn model.

        Returns:
            Self
        """
        for label, series in y.items():
            X_scaled = self.scaler.transform(X)
            self.regr[label].fit(X_scaled,series[-X.shape[0]:].copy(),**fit_params)
        return self

    def predict(self,X,in_sample:bool=False,**predict_params) -> dict[str,list[float]]:
        """ Makes predictions.

        Args:
            X (np.ndarray): The input data.
            in_sample (bool): Default False. If True, returns fitted values with a one-step ahead forecast.
            **predict_params: Passed to the estimator's predict() method.
        
        Returns:
            list[float]: The predictions.
        """       
        preds = {}
        if not self.lags or in_sample or (self.dynamic_testing == 1 and self.test_set_actuals):
            for label, regr in self.regr.items():
                X_scaled = self.scaler.transform(X)
                preds[label] = list(regr.predict(X_scaled,**predict_params))
            return preds

        series = {k: v.to_list() for k, v in self.f.y.items()}
        for i in range(X.shape[0]):
            X_scaled = self.scaler.transform(X[i,:].reshape(1,-1))
            for label, regr in self.regr.items():
                series_loc = list(self.f.y.keys()).index(label)
                pred = regr.predict(X_scaled,**predict_params)[0]
                preds.setdefault(label,[]).append(pred)
                if i < X.shape[0] - 1:
                    if (i+1) % self.dynamic_testing == 0 and self.test_set_actuals:
                        series[label].append(self.test_set_actuals[series_loc][i])
                    else:
                        series[label].append(pred)
            if i < X.shape[0] - 1:
                for x in self.predict_with_Xvars:
                    if x.startswith('LAG_'):
                        idx = self.predict_with_Xvars.index(x)
                        label = x.split('_')[1]
                        lag_no = int(x.split('_')[-1])
                        series[label][-lag_no]
                        X[i+1,idx] = series[label][-lag_no]

        return preds

    def fit_predict(self,X,y):
        """ Runs fit and predict methods, returning predictions.

        Args:
            X (np.ndarray): The input data.
            y (np.ndarray): The observed actuals.

        Returns:
            list[float]: The predictions.
        """
        self.fit(X,y)
        return self.predict(X)
    
class VECM:
    """ Forecasts using a vector error-correction model (multivariate forecaster).

    Args: 
        f (Forecaster): The Forecaster object storing the actual series and associated dates.
        model (Scikit-learn API Estimator): The imported scikit-learn API regression estimator/class (such as LinearRegressor or XGBRegressor).
        lags (None or int or list[int] or dict[str,int or list[int]]): The number of lags to add to the model.
            If int, that many lags added to every model.
            If a list of ints, only the lags in the list are added.
            If dict, key is a series name and value is int or list of ints that follows the behavior descrbied above, but only targeting passed series.
        Xvars (list[str]): List of exogenous regressors to use from the passed Forecaster object. If unspecified, no regressors are used.
        test_set_actuals (list[float]): Optional. Test-set actuals to use for testing the model. This enables the dynamic_testing option.
        lags (int): The number of lags from each series to use in the model.
        coint_rank (int): Cointegration rank.
        deterministic (str): One of {"n", "co", "ci", "lo", "li"}. Default "n".
            "n" - no deterministic terms.
            "co" - constant outside the cointegration relation.
            "ci" - constant within the cointegration relation.
            "lo" - linear trend outside the cointegration relation.
            "li" - linear trend within the cointegration relation.
            Combinations of these are possible (e.g. "cili" or "colo" for linear trend with intercept). 
            When using a constant term you have to choose whether you want to restrict it to the cointegration relation 
            (i.e. "ci") or leave it unrestricted (i.e. "co"). Do not use both "ci" and "co". The same applies for "li" 
            and "lo" when using a linear term. 
        seasons (int): Default 0. Number of periods in a seasonal cycle. 0 means no seasons.
        first_season (int): Default 0. Season of the first observation.
        **kwargs: Passed to the scikit-learn model passed to model.
    """
    def __init__(
        self,
        f:'MVForecaster',
        model:Literal['auto']='auto',
        Xvars:XvarValues = None, 
        test_set_actuals:Optional[dict[str,list[float]]]=None,
        lags:int=1,
        coint_rank:int=1,
        deterministic:Literal["n", "co", "ci", "lo", "li"]="n",
        seasons:int=0,
        first_season:int=0,
        **kwargs,
    ):
        if model == 'auto':
            from statsmodels.tsa.vector_ar.vecm import VECM
        else:
            raise ValueError(f'Unrecognized value passed to model: {model}')
        
        self.f = f
        self.Xvars = self._parse_Xvars(Xvars)
        self.test_set_actuals = test_set_actuals
        self.lags = lags
        self.coint_rank = coint_rank
        self.deterministic = deterministic
        self.seasons = seasons
        self.first_season = first_season
        self.model_kwargs = kwargs
        self.init_regr = VECM

    def _parse_Xvars(self,Xvars):
        match Xvars:
            case 'all':
                return list(self.f.current_xreg.keys())
            case None:
                return []
            case _:
                return list(Xvars)
            
    def generate_current_X(self) -> np.ndarray:
        """ Returns the matrix of the current input dataset.
        """
        if self.Xvars:
            return np.array([self.f.current_xreg[x].values.copy() for x in self.Xvars]).T
        return None

    def generate_future_X(self) -> np.ndarray:
        """ Returns the matrix of the future input dataset. 
        """  
        if self.Xvars:
            return np.array([np.array(self.f.future_xreg[x][:]) for x in self.Xvars]).T
        return None

    @_developer_utils.log_warnings
    def fit(self,X:np.ndarray|None,y:np.ndarray,**fit_params):
        """ Fits the estimator.

        Args:
            X (np.ndarray): The exogenours input data. None is an accepted value.
            y (np.ndarray): The observed actuals.
            **fit_params: Passed to the .fit() method from the scikit-learn model.

        Returns:
            Self
        """
        y = np.array([v.values.copy() for v in self.f.y.values()]).T
        self.regr = (
            self.init_regr(
                y, 
                exog = X, 
                k_ar_diff=self.lags,
                coint_rank=self.coint_rank,
                deterministic=self.deterministic,
                seasons=self.seasons,
                first_season=self.first_season,
                dates=self.f.current_dates,
                **self.model_kwargs,
            )
            .fit(**fit_params)
        )
        return self

    def predict(self,X:np.ndarray,in_sample:bool=False,**predict_params):
        """ Makes predictions.

        Args:
            X (np.ndarray): The input data.
            in_sample (bool): Default False. If True, returns fitted values with a one-step ahead forecast.
            **predict_params: Passed to the estimator's predict() method.
        
        Returns:
            list[float]: The predictions.
        """    
        if not in_sample:
            preds = self.regr.predict(steps=len(self.f.future_dates), exog_fc=X, **predict_params)
            return {name:[p[i] for p in preds] for i, name in enumerate(self.f.y)}
        else:
            return {name:[p[i] for p in self.regr.fittedvalues] for i, name in enumerate(self.f.y)}

    def fit_predict(self,X,y):
        """ Runs fit and predict methods, returning predictions.

        Args:
            X (np.ndarray): The input data.
            y (np.ndarray): The observed actuals.

        Returns:
            list[float]: The predictions.
        """
        self.fit(X,y)
        return self.predict(X)
    
class RNN:
    """ Forecasts using a recurrent neural network model from Tensorflow.

    Args: 
        f (Forecaster): The Forecaster object storing the actual series and associated dates.
        model (str): Default 'auto'. 'auto' is the only accepted value.
        lags (None or int or list[int] or dict[str,int or list[int]]): The number of lags to add to the model.
            If int, that many lags added to every model.
            If a list of ints, only the lags in the list are added.
            If dict, key is a series name and value is int or list of ints that follows the behavior descrbied above, but only targeting passed series.
        Xvars (list[str]): List of regressors to use from the passed Forecaster object.
        normalizer (NormalizerLike): Default 'minmax'. The label of the normalizer to use.
        test_set_actuals (list[float]): Optional. Test-set actuals to use for testing the model. This enables the dynamic_testing option.
        layers_struct (list[tuple[str,dict[str,Union[float,str]]]]): Default [('SimpleRNN',{'units':8,'activation':'tanh'})].
            Each element in the list is a tuple with two elements.
            First element of the list is the input layer (input_shape set automatically).
            First element of the tuple in the list is the type of layer ('SimpleRNN','LSTM', or 'Dense').
            Second element is a dict.
            In the dict, key is a str representing hyperparameter name: 'units','activation', etc.
            The value is the hyperparameter value.
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
    @_developer_utils.log_warnings
    def __init__(
        self, 
        f:'Forecaster', 
        model:Literal['auto']='auto', 
        test_set_actuals:Optional[list[float]]=None, 
        Xvars:XvarValues='all',
        normalizer:NormalizerLike = 'minmax', 
        layers_struct:list[tuple[dict[str,Any]]]=[("SimpleRNN", {"units": 8, "activation": "tanh"})],
        loss:str="mean_absolute_error",
        optimizer:str="Adam",
        learning_rate:ConfInterval=0.001,
        random_seed:int=None,
        scale_y:bool = True,
        **kwargs:Any,
    ):
        if model == 'auto':
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Dense, LSTM, SimpleRNN
            import tensorflow.keras.optimizers
        else:
            raise ValueError(f'Unrecognized value passed to model: {model}')
        
        self.f = f
        self.normalizer = normalizer
        self.lag_scaler = self.f.lookup_normalizer(normalizer)()
        self.scaler = self.f.lookup_normalizer(normalizer)()
        self.test_set_actuals = test_set_actuals
        self.Xvars = self._parse_Xvars(Xvars)
        self.max_lag_order = self._determine_max_lag_order(self.Xvars)
        self.layers_struct = layers_struct
        self.loss = loss
        self.learning_rate = learning_rate
        self.random_seed = random_seed
        self.scale_y = scale_y
        self.fit_kwargs = kwargs

        y = self.f.y.to_list()
        self.n_timesteps = len(self.f.future_dates)
        total_periods = self.max_lag_order + self.n_timesteps
        
        idx_end = len(y)
        idx_start = idx_end - total_periods
        y_new = []

        while idx_start > 0:
            y_line = y[idx_start + self.max_lag_order:idx_start + total_periods]
            y_new.append(y_line)
            idx_start -= 1

        self.y = np.array(y_new[::-1])
        self.lag_scaler = self.lag_scaler.fit(self.f.y.values.reshape(-1,1))
        if scale_y:
            y_pieces = []
            for col in self.y.T:
                piece = self.lag_scaler.transform(col.reshape(-1,1))
                y_pieces.append(piece)
            self.y = np.concatenate(y_pieces,axis=1)

        if isinstance(optimizer, str):
            self.optimizer = eval(f"tensorflow.keras.optimizers.{optimizer}")(learning_rate=learning_rate)
        else:
            self.optimizer = optimizer

        for i, kv in enumerate(layers_struct):
            layer_label, params = kv
            match layer_label:
                case 'SimpleRNN':
                    layer = SimpleRNN
                case 'LSTM':
                    layer = LSTM
                case 'Dense':
                    layer = Dense
                case _:
                    raise ValueError(f'Unknown model layer value passed: {layer}')
                
            if i == 0:
                if layer_label in ['LSTM','SimpleRNN']:
                    params['return_sequences'] = len(layers_struct) > 1
                model = Sequential([layer(**params, input_shape = (self.n_timesteps, 1))])
            else:
                if layer_label in ('LSTM','SimpleRNN'):
                    params['return_sequences'] = not i == (len(layers_struct) - 1)
                    if params['return_sequences']:
                        params['return_sequences'] = layers_struct[i+1][0] != 'Dense'
                model.add(layer(**params))
        model.add(Dense(self.y.shape[1]))
        self.regr = model
        self.regr.compile(optimizer=self.optimizer,loss=self.loss)

    def _determine_max_lag_order(self,Xvars):
        lag_orders = [x.lag_order for x in Xvars if isinstance(x,AR)]
        if lag_orders:
            return max(lag_orders)
        return 0
    
    def _parse_Xvars(self, Xvars):
        match Xvars:
            case 'all'|None:
                return list(self.f.current_xreg.keys())
            case _:
                return list(Xvars)
            
    def _generate_X(self):
        X_lags = np.array(
            [v.to_list() + self.f.future_xreg[k][:1] for k,v in self.f.current_xreg.items() if isinstance(k,AR) and k in self.Xvars]
        ).T
        X_other = np.array(
            [v.to_list() + self.f.future_xreg[k][:1] for k,v in self.f.current_xreg.items() if not isinstance(k,AR) and k in self.Xvars]
        ).T

        X_lags_new = X_lags[self.max_lag_order:]
        X_other_new = X_other[self.max_lag_order:]
        
        # scale lags
        if len(X_lags_new) > 0:
            X_lags_new_pieces = []
            for col in X_lags_new.T:
                new_col = self.lag_scaler.transform(col.reshape(-1,1))
                X_lags_new_pieces.append(new_col)
            X_lags_new = np.concatenate(X_lags_new_pieces,axis=1)
        # scale other regressors
        if len(X_other_new) > 0:
            X_other_train = X_other_new[:-1]
            self.scaler = self.scaler.fit(X_other_train)
            X_other_new = self.scaler.transform(X_other_new)
            
        # combine
        if len(X_lags_new) > 0 and len(X_other_new) > 0:
            X = np.concatenate([X_lags_new,X_other_new],axis=1)
        elif len(X_lags_new) > 0:
            X = X_lags_new
        else:
            X = X_other_new

        current_X = X[:-1]
        future_X = X[-1:]

        return current_X, future_X

    def generate_current_X(self) -> np.ndarray:
        """ Returns the matrix of the current input dataset.
        """
        X = self._generate_X()[0]
        if self.n_timesteps > 1:
            X = X[1:-(self.n_timesteps-1)]
        else:
            X = X[1:]

        X = X.reshape(X.shape[0], X.shape[1], 1)
        return X

    def generate_future_X(self) -> np.ndarray:
        """ Returns the matrix of the future input dataset. 
        """
        X = self._generate_X()[1]
        X = X.reshape(X.shape[0], X.shape[1], 1)
        return X

    @_developer_utils.log_warnings
    def fit(self,X:np.ndarray,y:None=None,**fit_params:Any) -> Self:
        """ Fits the estimator.

        Args:
            X (np.ndarray): The input data.
            y (np.ndarray): The observed actuals. Ignored for RNN models since the actuals are already stored in self.y, which is used for training. 
                This is just for API consistency with other models.
            **fit_params: Passed to the .fit() method from the scikit-learn model.

        Returns:
            Self
        """
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
        
        self.hist = self.regr.fit(X, self.y, **self.fit_kwargs, **fit_params)
        return self

    def predict(self,X:np.ndarray,in_sample:bool=False,**predict_params) -> list[float]:
        """ Makes predictions.

        Args:
            X (np.ndarray): The input data.
            in_sample (bool): Default False. If True, returns fitted values with a one-step ahead forecast.
            **predict_params: Passed to the estimator's predict() method.
        
        Returns:
            list[float]: The predictions.
        """ 
        if in_sample:
            preds = self.regr.predict(X,**predict_params)
            preds = [p[0] for p in preds[:-1]] + [p for p in preds[-1]]
        else:
            preds = self.regr.predict(X,**predict_params)
            preds = [p for p in preds[0]]
            
        if self.scale_y:
            preds = [p[0] for p in self.lag_scaler.inverse_transform(np.asarray(preds).reshape(-1,1))]
        return preds

    def fit_predict(self,X,y) -> list[float]:
        self.fit(X,y)
        return self.predict(X)
    
class LSTM(RNN):
    """ Forecasts using an LSTM model from Tensorflow. Inherits from RNN and simply sets the default layer to LSTM and adds lags by adding AR terms to the Forecaster object.

    Args:
        f (Forecaster): The Forecaster object storing the actual series and associated dates.
        model (str): Default 'auto'. 'auto' is the only accepted value.
        lags (int): The number of lags to add to the model. Default
            1. This is added to the Forecaster object as AR terms, so the model will automatically use them as input features.
        normalizer (NormalizerLike): Default 'minmax'. The label of the normalizer to use.
        lstm_layer_sizes (list[int]): Default [8]. The number of units in each LSTM layer. The number of layers is determined by the length of the list.
        dropout (list[float]): Default [0.0]. The dropout rate to use for each LSTM layer. Should be the same length as lstm_layer_sizes. If 0, no dropout is applied.
        loss (str or tf.keras.losses.Loss): Default 'mean_absolute_error'.The loss function to minimize.
            See available options here: https://www.tensorflow.org/api_docs/python/tf/keras/losses.
            Be sure to choose one that is suitable for regression tasks.
        optimizer (str or tf Optimizer): Default "Adam". The optimizer to use when compiling the model. 
            See available values here: https://www.tensorflow.org/api_docs/python/tf/keras/optimizers.  
            If str, it will use the optimizer with default args.
            If type Optimizer, will use the optimizer exactly as specified.
        learning_rate (float): Default 0.001. The learning rate to use when compiling the model. Ignored if you pass your own optimizer with a learning rate.
        random_seed (int): Optional. Set a seed for consistent results. With tensorflow networks, setting seeds does not guarantee consistent results.
        **kwargs: Passed to fit() and can include epochs, verbose, callbacks, validation_split, and more.
    """
    def __init__(
        self,
        f:'Forecaster',
        model:Literal['auto']='auto',
        lags:PositiveInt=1,
        normalizer:NormalizerLike = 'minmax', 
        lstm_layer_sizes:Sequence[int]=[8],
        dropout:Sequence[float]=[0.0],
        loss:str="mean_absolute_error",
        activation:str="tanh",
        optimizer:str="Adam",
        learning_rate:ConfInterval=0.001,
        random_seed:Optional[int]=None,
        **kwargs:Any,
    ):
        f.add_ar_terms(lags)
        lstm_kwargs = {
            'f':f,
            'model':model,
            'Xvars':[AR(i) for i in range(1,lags+1)],
            'normalizer':normalizer,
            'lstm_layer_sizes':lstm_layer_sizes,
            'dropout':dropout,
            'loss':loss,
            'activation':activation,
            'optimizer':optimizer,
            'learning_rate':learning_rate,
            'random_seed':random_seed,
        }
        new_kwargs = {
            k: v
            for k, v in lstm_kwargs.items()
            if k not in ("lstm_layer_sizes", "dropout", "activation")
        }
        new_kwargs["layers_struct"] = [
            ("LSTM",{"units": v,"activation": lstm_kwargs["activation"],"dropout": lstm_kwargs["dropout"][i]})
            for i, v in enumerate(lstm_kwargs["lstm_layer_sizes"])
        ]
        super().__init__(**new_kwargs,**kwargs)
    
class Theta:
    """ Forecasts using a Theta model from Darts.

    Args:
        f (Forecaster): The Forecaster object storing the actual series and associated dates.
        model (str): The Theta model to use. Default 'auto' which selects the FourTheta model from Darts. Currently, 'auto' is the only option, but more Theta variants may be added in the future.
        test_set_actuals (list[float]): Optional. Test-set actuals to use for testing the model. This enables the dynamic_testing option.
        **kwargs: Passed to the Darts Theta model specified in model.
    """
    def __init__(
        self, 
        f:'Forecaster', 
        model:Literal['auto']='auto', 
        test_set_actuals:Optional[list[float]]=None, 
        **kwargs:Any,
    ):
        if model == 'auto':
            from darts import TimeSeries
            from darts.models.forecasting.theta import FourTheta
        else:
            raise ValueError(f'Unrecognized value passed to model: {model}')
        
        self.f = f
        self.regr = FourTheta(**kwargs)
        y = pd.Series(f.y.to_list(),f.current_dates.to_list())
        self.current_actuals = TimeSeries.from_series(y)
        self.test_set_actuals = test_set_actuals

    def generate_current_X(self) -> np.ndarray:
        """ Returns the matrix of the current input dataset.
        """
        self.current_X = self.current_actuals
        return self.current_X

    def generate_future_X(self) -> np.ndarray:
        """ Returns the matrix of the future input dataset. 
        """  
        self.future_X = [0]*len(self.f.future_dates)
        return self.future_X

    @_developer_utils.log_warnings
    def fit(self,X:np.ndarray,y:None=None,**fit_params:Any) -> Self:
        """ Fits the estimator.

        Args:
            X (np.ndarray): The input data.
            y (np.ndarray): The observed actuals. Ignored for Theta models since the actuals are already stored in self.current_actuals, which is used for training. 
                This is just for API consistency with other models.
            **fit_params: Passed to the .fit() method from the scikit-learn model.

        Returns:
            Self
        """
        self.regr.fit(X,**fit_params)
        return self

    def predict(self,X:np.ndarray,in_sample:bool=False,**predict_params) -> list[float]:
        """
        Makes predictions.
        
        Args:
            X (np.ndarray): The input data.
            in_sample (bool): Default False. If True, returns fitted values with a one-step ahead forecast.
            **predict_params: Passed to the estimator's predict() method.
        
        Returns:
            list[float]: The predictions.
        """
        if not in_sample:
            pred = self.regr.predict(len(X),**predict_params)
            return [p[0] for p in pred.values()]
        else:
            resid = [r[0] for r in self.regr.residuals(self.current_actuals).values()]
            actuals = self.current_actuals.values().flatten()[-len(resid) :]
            return [r + a for r, a in zip(resid, actuals)]
        
    def fit_predict(self,X:np.ndarray,y:None=None) -> list[float]:
        """
        Fits and predicts on the same dataset.

        Args:
            X (np.ndarray): The input data.
            y (np.ndarray): The observed actuals. Ignored for Theta models since the actuals are already stored in self.current_actuals, which is used for training. 
                This is just for API consistency with other models.
        Returns:
            list[float]: The predictions.
        """
        self.fit(X,y)
        return self.predict(X)

class HWES:
    """ Forecasts using a Holt-Winters Exponential Smoothing model from Statsmodels.

    Args:
        f (Forecaster): The Forecaster object storing the actual series and associated dates.
        model (str): The HWES model to use. Default 'auto' which selects the ExponentialSmoothing model from Statsmodels. 
            Currently, 'auto' is the only option, but more HWES variants may be added in the future.
        test_set_actuals (list[float]): Optional. Test-set actuals to use for testing the model. This enables the dynamic_testing option.
        **kwargs: Passed to the Statsmodels ExponentialSmoothing model specified in model. 
    """
    def __init__(
        self,
        f:'Forecaster', 
        model:Literal['auto']='auto', 
        test_set_actuals:Optional[list[float]]=None, 
        **kwargs:Any,
    ):
        if model == 'auto':
            from statsmodels.tsa.holtwinters import ExponentialSmoothing as HWES
        else:
            raise ValueError(f'Unrecognized value passed to model: {model}')
        self.test_set_actuals = test_set_actuals
        self.f = f
        self.model_kwargs = kwargs
        self.init_regr = HWES

    def generate_current_X(self):
        """ Placeholder method to remain consistent with other models. HWES does not use an input matrix, so this method does not need to do anything. 
            It is only included for API consistency across models.
        """
        return

    def generate_future_X(self):
        """ Placeholder method to remain consistent with other models. HWES does not use an input matrix, so this method does not need to do anything. 
            It is only included for API consistency across models.
        """
        return
    
    @_developer_utils.log_warnings
    def fit(self,X:None,y:np.ndarray,optimized:bool=True,use_brute:bool=True,**fit_params:Any) -> Self:
        """ Fits the estimator.

        Args:
            X (np.ndarray): The input data. Ignored for HWES models since HWES does not use an input matrix. This is just for API consistency with other models.
            y (np.ndarray): The observed actuals.
            optimized (bool): Default True. Whether to optimize the model's smoothing level parameters. 
                If False, the parameters will be set to the values passed in **kwargs or to the default values from the statsmodels ExponentialSmoothing model if not passed in **kwargs.
            use_brute (bool): Default True. Whether to use the brute-force optimization method when optimizing the model's smoothing level parameters. 
                This is passed to the fit() method from the statsmodels ExponentialSmoothing model and is only relevant if optimized is True. 
                    If False, the model will use the default optimization method from the statsmodels ExponentialSmoothing model when optimizing the smoothing level parameters. 
            **fit_params: Passed to the .fit() method from the scikit-learn model.

        Returns:
            Self
        """
        self.regr = (
            self
            .init_regr(y, dates = self.f.current_dates, freq = self.f.freq, **self.model_kwargs)
            .fit(optimized=optimized, use_brute=use_brute, **fit_params)
        )
        return self
    
    def predict(self,X:None=None,in_sample:bool=False,**predict_params) -> list[float]:
        """ Makes predictions.
        Args:
            X (np.ndarray): The input data. Ignored for HWES models since HWES does not use an input matrix. This is just for API consistency with other models.
            in_sample (bool): Default False. If True, returns fitted values with a one-step ahead forecast.
            **predict_params: Passed to the estimator's predict() method. 

        Returns:
            list[float]: The predictions.
        """
        if in_sample:
            return list(self.regr.fittedvalues)
        else:
            pred = self.regr.forecast(len(self.f.future_dates), **predict_params)
            return list(pred)
        
    def fit_predict(self,X:None,y:np.ndarray) -> list[float]:
        """ Fits and predicts on the same dataset.

        Args:
            X (np.ndarray): The input data. Ignored for HWES models since HWES does not use an input matrix. This is just for API consistency with other models.
            y (np.ndarray): The observed actuals.
        
        Returns:
            list[float]: The predictions.
        """
        self.fit(X,y)
        return self.predict(X)

class TBATS:
    """ Forecasts using a TBATS model from the tbats package.

    Args:
        f (Forecaster): The Forecaster object storing the actual series and associated dates.
        model (str): The TBATS model to use. Default 'auto' which selects the TBATS model from the tbats package. Currently, 'auto' is the only option, but more TBATS variants may be added in the future.
        test_set_actuals (list[float]): Optional. Test-set actuals to use for testing the model. This enables the dynamic_testing option.
        random_seed (int): Optional. Set a seed for consistent results. With TBATS models, setting seeds does not guarantee consistent results.
        **kwargs: Passed to the TBATS model specified in model.
    """
    def __init__(
        self,
        f:'Forecaster',
        model:Literal['auto']='auto',
        test_set_actuals:Optional[list[float]]=None, 
        random_seed:Optional[int]=None,
        **kwargs:Any,
    ):
        if model == 'auto':
            from tbats import TBATS
        else:
            raise ValueError(f'Unrecognized value passed to model: {model}')
        
        self.f = f
        self.regr = TBATS(show_warnings=True,**kwargs)
        self.current_actuals = f.y.to_list()
        self.test_set_actuals = test_set_actuals
        self.random_seed = random_seed

    def generate_current_X(self) -> np.ndarray:
        """ Returns the matrix of the current input dataset.
        """
        return np.asarray(self.current_actuals)

    def generate_future_X(self) -> np.ndarray:
        """ Returns the matrix of the future input dataset. For TBATS, this is just an array of zeros equal to the length of the forecast.
        """
        return np.asarray([0]*len(self.f.future_dates))

    @_developer_utils.log_warnings
    def fit(self,X:np.ndarray,y:None=None,**fit_params:Any) -> Self:
        """ Fits the estimator.

        Args:
            X (np.ndarray): The input data.
            y (np.ndarray): The observed actuals. Ignored for TBATS models since the actuals are already stored in self.current_actuals, which is used for training. 
                This is just for API consistency with other models.
            **fit_params: Passed to the .fit() method from the scikit-learn model.
        
        Returns:
            Self
        """
        if self.random_seed:
            np.random.seed(self.random_seed)
        self.regr.fit(X,**fit_params)
        return self

    def predict(self,X:np.ndarray,in_sample:bool=False,**predict_params) -> list[float]:
        """ Makes predictions.

        Args:
            X (np.ndarray): The input data.
            in_sample (bool): Default False. If True, returns fitted values with a one-step ahead forecast.
            **predict_params: Passed to the estimator's predict() method.
        
        Returns:
            list[float]: The predictions.
        """
        if not in_sample:
            preds = self.regr.predict(steps = len(X),**predict_params)
            return list(preds)
        else:
            return self.regr.y_hat
        
    def fit_predict(self,X:np.ndarray,y:None=None) -> list[float]:
        """ Fits and predicts on the same dataset.

        Args:
            X (np.ndarray): The input data.
            y (np.ndarray): The observed actuals. Ignored for TBATS models since the actuals are already stored in self.current_actuals, which is used for training. 
                This is just for API consistency with other models.
        
        Returns:
            list[float]: The predictions.
        """
        self.fit(X,y)
        return self.predict(X)


class ARIMA:
    """ Forecasts using an ARIMA model from Statsmodels.

    Args:
        f (Forecaster): The Forecaster object storing the actual series and associated dates.
        model (str): The ARIMA model to use. Default 'auto' which selects the
        ARIMA model from Statsmodels. Currently, 'auto' is the only option.
        Xvars (list[str]): List of regressors to use from the passed Forecaster object.
        test_set_actuals (list[float]): Optional. Test-set actuals to use for testing the model. This enables the dynamic_testing option.
        **kwargs: Passed to the Statsmodels ARIMA model specified in model.
    """
    def __init__(
        self,
        f:'Forecaster',
        model:Literal['auto']='auto',
        Xvars:XvarValues = None, 
        test_set_actuals:Optional[list[float]]=None, 
        **kwargs:Any,
    ):
        if model == 'auto':
            from statsmodels.tsa.arima.model import ARIMA
        else:
            raise ValueError(f'Unrecognized value passed to model: {model}')
        
        self.f = f
        self.current_actuals = f.y.to_list()
        self.test_set_actuals = test_set_actuals
        self.Xvars = self._parse_Xvars(Xvars)
        self.model_kwargs = kwargs
        self.init_regr = ARIMA

    def _parse_Xvars(self, Xvars):
        match Xvars:
            case 'all':
                return [x for x in self.f.current_xreg if not isinstance(x,AR)]
            case None:
                return []
            case _:
                return list(Xvars)

    def generate_current_X(self) -> np.ndarray:
        """ Returns the matrix of the current input exogenous variables.
        """
        if self.Xvars:
            return np.array([self.f.current_xreg[x].values.copy() for x in self.Xvars]).T
        return None

    def generate_future_X(self) -> np.ndarray:
        """ Returns the matrix of the future input exogenous variables. 
            If no regressors specified, returns None.
        """
        if self.Xvars:
            return np.array([np.array(self.f.future_xreg[x][:]) for x in self.Xvars]).T
        return None

    @_developer_utils.log_warnings
    def fit(self,X:np.ndarray,y:np.ndarray,**fit_params:Any) -> Self:
        """ Fits the estimator.

        Args:
            X (np.ndarray): The input data.
            y (np.ndarray): The observed actuals.
            **fit_params: Passed to the .fit() method from the scikit-learn model.  

        Returns:
            Self
        """
        self.regr = self.init_regr(
            y.values,
            exog=X,
            dates=self.f.current_dates.to_list(),
            freq=self.f.freq,
            **self.model_kwargs
        ).fit(**fit_params)
        return self

    def predict(
        self,
        X:np.ndarray,
        in_sample:bool=False,
        dynamic:Optional[bool]=True,
        **predict_params:Any,
    ) -> list[float]:
        """ Makes predictions.

        Args:
            X (np.ndarray): The input data.
            in_sample (bool): Default False. If True, returns fitted values with a one-step ahead forecast.
            dynamic (bool): Default True. Only relevant if in_sample is True. 
                Whether to use dynamic predictions when generating in-sample fitted values. 
                If True, uses dynamic predictions, which means that when generating fitted values for the in-sample period, 
                the model uses its own previous predictions as input rather than the actuals. 
                If False, uses one-step ahead predictions, which means that when generating fitted values for the in-sample period, 
                the model always uses the actuals from the previous time step as input rather than its own predictions. 
                Using dynamic predictions can give a better sense of out-of-sample performance since it does not rely on actuals from the in-sample period, 
                but it can also lead to worse performance since any mistakes the model makes are compounded in future predictions. 
                Using one-step ahead predictions can give a better sense of in-sample fit since it always uses the actuals from the in-sample period, 
                but it can also lead to overly optimistic performance since it relies on actuals from the in-sample period that would not be available in an out-of-sample forecasting scenario.
            **predict_params: Passed to the estimator's predict() method.
        """
        if not in_sample:
            preds = self.regr.predict(
                exog=X,
                start=len(self.current_actuals),
                end=len(self.current_actuals) + len(self.f.future_dates) - 1, 
                dynamic=dynamic,
                **predict_params,
            )
            return list(preds)
        else:
            return list(self.regr.fittedvalues)
        
    def fit_predict(self,X:np.ndarray,y:np.ndarray) -> list[float]:
        """ Fits and predicts on the same dataset.

        Args:
            X (np.ndarray): The input data.
            y (np.ndarray): The observed actuals.   

        Returns:
            list[float]: The predictions.
        """
        self.fit(X,y)
        return self.predict(X)

class Prophet:
    """ Forecasts using a Prophet model from the prophet package.

    Args:
        f (Forecaster): The Forecaster object storing the actual series and associated dates.
        model (str): The Prophet model to use. Default 'auto' which selects the Prophet model from the prophet package. Currently, 'auto' is the only option.
        Xvars (list[str]): List of regressors to use from the passed Forecaster object. These are added as extra regressors in the Prophet model. 
            If 'all', will use all available regressors in the Forecaster object. Default None, which uses no regressors.
        test_set_actuals (list[float]): Optional. Test-set actuals to use for testing the model. This enables the dynamic_testing option.
        cap (float): Optional. The capacity parameter to use for the Prophet model if you want to fit a logistic growth model. If not passed, the model will fit a linear growth model.
        floor (float): Optional. The floor parameter to use for the Prophet model if you want to fit a logistic growth model. If not passed, the model will fit a linear growth model.
        callback_func (callable): Optional. A function that takes the initialized but unfitted Prophet model as input and performs some operations on it, 
            such as adding holidays or changing hyperparameters, before it is fitted. 
            This allows you to customize the Prophet model in ways that are not currently supported by the parameters of this class. 
            If not passed, no operations will be performed on the initialized Prophet model before fitting.
        **kwargs: Passed to the Prophet model specified in model. Note that if you want to use the 
            dynamic_testing option with Prophet, you must pass the parameter 'interval_width' in **kwargs with a value less than 1 (e.g. 0.8) 
            to ensure that the prediction intervals are narrow enough to be useful for testing.
    """
    def __init__(
        self,
        f:'Forecaster',
        model:Literal['auto']='auto',
        Xvars:XvarValues = None, 
        test_set_actuals:Optional[list[float]]=None, 
        cap:Optional[float]=None,
        floor:Optional[float]=None,
        callback_func:callable=None,
        **kwargs:Any,
    ):
        if model == 'auto':
            from prophet import Prophet
        else:
            raise ValueError(f'Unrecognized value passed to model: {model}')
        
        self.f = f
        self.current_actuals = f.y.to_list()
        self.test_set_actuals = test_set_actuals
        self.Xvars = self._parse_Xvars(Xvars)
        self.regr = Prophet(**kwargs)
        self.cap = cap
        self.floor = floor
        self.callback_func = callback_func

    def _parse_Xvars(self, Xvars):
        match Xvars:
            case 'all':
                return [x for x in self.f.current_xreg if not isinstance(x,AR)]
            case None:
                return []
            case _:
                return list(Xvars)
            
    def generate_current_X(self) -> pd.DataFrame:
        """ Returns the DataFrame of the current input dataset. For Prophet, this includes a 'ds' column for dates, a 'y' column for the actuals, and columns for any specified regressors. 
            If cap and/or floor are specified, these are also included as columns.
        """
        df = pd.DataFrame({'ds':self.f.current_dates.to_list(),'y':self.current_actuals})
        for x in self.Xvars:
            df[x] = self.f.current_xreg[x][:]
        if self.cap:
            df['cap'] = self.cap
        if self.floor:
            df['floor'] = self.floor
        
        return df

    def generate_future_X(self) -> pd.DataFrame:
        """ Returns the DataFrame of the future input dataset. For Prophet, this includes a 'ds' column for dates and columns for any specified regressors. 
            If cap and/or floor are specified, these are also included as columns.
        """
        df = pd.DataFrame({'ds':self.f.future_dates.to_list()})
        for x in self.Xvars:
            df[x] = self.f.future_xreg[x][:]
        if self.cap:
            df['cap'] = self.cap
        if self.floor:
            df['floor'] = self.floor
        
        return df
    
    @_developer_utils.log_warnings
    def fit(self,X:pd.DataFrame,y:None=None,**fit_params:Any) -> Self:
        """ Fits the estimator.
        
        Args:
            X (pd.DataFrame): The input data.
                y (pd.DataFrame): The observed actuals. Ignored for Prophet models since the actuals are already stored in self.current_actuals, which is used for training.
                This is just for API consistency with other models.
            **fit_params: Passed to the .fit() method from the scikit-learn model.
        
        Returns:
            Self
        """
        if callable(self.callback_func):
            self.callback_func(self.regr)

        self.regr = self.regr.fit(X,**fit_params)
        return self
    
    def predict(self,X:pd.DataFrame,in_sample:None=None,**predict_params:Any) -> list[float]:
        """ Makes predictions.

        Args:
            X (pd.DataFrame): The input data.
            in_sample (bool): Ignored for Prophet models since Prophet does not have a built-in method for generating in-sample fitted values with a one-step ahead forecast. 
                This is just for API consistency with other models.
            **predict_params: Passed to the estimator's predict() method.
        Returns:
            list[float]: The predictions.
        """
        preds = self.regr.predict(X,**predict_params)
        return preds['yhat'].to_list()
    
    def fit_predict(self,X:pd.DataFrame,y:None=None) -> list[float]:
        """ Fits and predicts on the same dataset.
        
        Args:
            X (pd.DataFrame): The input data.
            y (pd.DataFrame): The observed actuals. Ignored for Prophet models since the actuals are already stored in self.current_actuals, which is used for training. 
                This is just for API consistency with other models.    
        
        Returns:
            list[float]: The predictions.
        """
        self.fit(X,y)
        return self.predict(X)

class Naive:
    """ Forecasts using a Naive model. This model simply uses the last observed value as the forecast for all future periods. 
    If seasonal, it uses the value from the same period in the previous season as the forecast.

    Args:
        f (Forecaster): The Forecaster object storing the actual series and associated dates.
        model (str): The Naive model to use. Must be 'auto'.
        test_set_actuals (list[float]): Not used
        seasonal (bool): Whether to use a seasonal naive model. 
            If False, the forecast for all future periods is the last observed value. 
            If True, the forecast for each future period is the value from the same period in the previous season.
        m (int or 'auto'): The seasonal period to use if seasonal is True. 
            If 'auto', the seasonal period is determined based on the frequency of the data. For example, if the frequency is monthly, the seasonal period will be 12. 
            If the frequency is quarterly, the seasonal period will be 4. If the frequency is daily, the seasonal period will be 7. 
            If the frequency is yearly, the seasonal period will be 1 (which means the seasonal naive model will be the same as the non-seasonal naive model). 
            You can also specify an integer value for the seasonal period if you want to use a different seasonal period than the one determined by the frequency.
    """
    def __init__(
        self,
        f:'Forecaster',
        model:Literal['auto']='auto',
        test_set_actuals:Optional[list[float]]=None,
        seasonal:bool=False,
        m:int|Literal['auto']='auto',
    ):
        if model != 'auto':
            raise ValueError(f'Unrecognized value passed to model: {model}')
        
        self.f = f
        self.test_set_actuals = test_set_actuals
        if seasonal:
            self.m = find_seasonal_length(m,f.freq)
        else:
            self.m = 1

    def generate_current_X(self) -> None:
        """ Placeholder method to remain consistent with other models. HWES does not use an input matrix, so this method does not need to do anything. 
            It is only included for API consistency across models.
        """
        return
    
    def generate_future_X(self) -> None:
        """ Placeholder method to remain consistent with other models. HWES does not use an input matrix, so this method does not need to do anything. 
            It is only included for API consistency across models.
        """
        return
    
    def fit(self,X:None=None,y:None=None) -> Self:
        """ Fits the estimator. For the Naive model, there is no fitting process since the model simply uses the last observed value (or the value from the same period in the previous season) as the forecast. 
            This method is included for API consistency with other models, but it does not need to do anything for the Naive model.
        
        Args:
            X (None): Ignored for the Naive model since it does not use an input matrix. This is just for API consistency with other models.
            y (None): Ignored for the Naive model since it does not use an input matrix. This is just for API consistency with other models.
        
        Returns:
            Self
        """
        return self
    
    def predict(self,X:None=None,in_sample:bool=False) -> list[float]:
        if in_sample:
            return self.f.y.shift(self.m).dropna().to_list()
        else:
            return (self.f.y.to_list()[-self.m:] * int(np.ceil(len(self.f.future_dates)/self.m)))[:len(self.f.future_dates)]
        
    def fit_predict(self,X:None=None,y:None=None) -> list[float]:
        self.fit(X,y)
        return self.predict(X)


class Combo:
    """ Forecasts using a combination of the forecasts from multiple models. The forecasts are combined with either simple or weighted averaging.

    
    Args:
        f (Forecaster): The Forecaster object storing the actual series and associated dates.
        model (str): The Combo model to use. Must be 'auto'.
        test_set_actuals (list[float]): Optional. Test-set actuals to use for testing the model. This enables the dynamic_testing option.
        how (str): The method to use when combining forecasts. Default 'simple' which uses simple averaging. 'weighted' uses weighted averaging where the weights are determined by the relative performance of the models on some metric.
        models (str or list[str]): The models to include in the combination. 
            Default 'all' which includes all models in the Forecaster's history except the most recent one (which is assumed to be the Combo model itself). 
            You can also specify a list of model names to include, or use the syntax 'top_n' to specify the top n models based on the metric specified in determine_best_by.
        determine_best_by (str): The metric to use when determining the best models for the 'top_n' syntax in the models argument. 
            Default 'ValidationMetricValue' which uses the validation metric value stored in the Forecaster's history for each model. 
            This is the most common use-case, but you can also specify other metrics that are stored in the history such as 'TestSetRMSE' or 'InSampleRMSE'.
        weights (list[float]): Optional. If how is 'weighted', you can optionally provide your own weights for each model instead of using the relative performance.               
        replace_negative_weights (bool|float): Whether to replace negative-scoring metrics with some positive (or 0) value to avoid situations where predictions might become nonsensical.
            This will be ignored in situations where lower scores are better (R2 is the main use-case).
            Change this to False to turn it off. 0 is an acceptable replacement value.
        exclude_models_with_no_fvs (bool): Whether to exclude models that have no fitted values stored in the history when generating the combined forecast. 
            This is relevant because if a model has no fitted values, it cannot generate in-sample predictions, 
            which means it can only contribute to the future forecast and not the in-sample fitted values. 
            This can lead to situations where the combined forecast is essentially just the forecast from that one model, which may not be desirable. Default True.
    """
    def __init__(
        self,
        f:'Forecaster',
        model:Literal['auto']='auto', 
        test_set_actuals:Optional[list[float]]=None, 
        how:Literal['simple','weighted']="simple",
        models:ModelValues="all",
        determine_best_by:DetermineBestBy="ValidationMetricValue",
        weights:Optional[Sequence[float|int]]=None,
        replace_negative_weights:bool|float=.001,
        exclude_models_with_no_fvs:bool = True,
    ):
        if model != 'auto':
            raise ValueError(f'Unrecognized value passed to model: {model}')
        
        self.f = f
        self.test_set_actuals = test_set_actuals
        self.how = how
        self.models = self._parse_models(models=models, determine_best_by=determine_best_by)
        self.weights = weights
        self.replace_negative_weights = replace_negative_weights
        if how == 'weighted' and not weights:
            self.metrics = [f.history[m][determine_best_by] for m in self.models]
            self.lower_is_better = self.metrics[0].store.lower_is_better
        elif how == 'weighted' and weights:
            if len(self.weights) != len(self.models):
                raise ValueError('When how is weighted and weights are provided, the number of provided weights must match the number of provided models')
        else:
            self.metrics = None
        self.exclude_models_with_no_fvs = exclude_models_with_no_fvs

    def _validate_how(self,how):
        if how not in ('simple','weighted'):
            raise ValueError(f'Argument passed to how not recognized: {how}')
    
    def _parse_models(self, models:ModelValues, determine_best_by:DetermineBestBy):
        all_models = [m for m in self.f.history][:-1] # exclude last model because it's the one just created (combo)
        match models:
            case 'all':
                return all_models
            case str():
                if models.startswith('top_'):
                    top_n = int(models.split('_')[-1])
                    return self.f._parse_models(models=all_models, determine_best_by=determine_best_by)[:top_n]
                else:
                    return [models]

        return models
    
    def generate_current_X(self):
        """ Generates the matrix of the current input dataset by extracting the fitted values for each model in the combination from the Forecaster's history.
        """
        lengths = []
        all_fvs = []
        for m in self.models:
            fvs = self.f.history[m].get('FittedVals',[])
            if not fvs and self.exclude_models_with_no_fvs:
                continue
            else:   
                lengths.append(len(fvs))
            all_fvs.append(fvs)
        
        min_length = min(lengths)
        return np.array([fv[-min_length:] for fv in all_fvs]).T

    def generate_future_X(self):
        """ Generates the matrix of the future input dataset by extracting either the test set predictions (if test_set_actuals were provided) 
        or the forecasts for each model in the combination from the Forecaster's history.
        """
        if self.test_set_actuals:
            return np.array([self.f.history[m]['TestSetPredictions'] for m in self.models]).T
        else:
            return np.array([self.f.history[m]['Forecast'] for m in self.models]).T
    
    @_developer_utils.log_warnings
    def fit(self,X:None=None,y:None=None,**fit_params:None) -> Self:
        """ Fits the estimator. For the Combo model, there is no fitting process since the model simply combines the forecasts from the specified models using either simple or weighted averaging. 
            This method is included for API consistency with other models, but it does not need to do anything for the Combo model.

        Args:
            X (None): Ignored for the Combo model since it does not use an input matrix for fitting. This is just for API consistency with other models.
            y (None): Ignored for the Combo model since it does not use an input matrix for fitting. This is just for API consistency with other models.
            **fit_params: Ignored for the Combo model since there is no fitting process. This is just for API consistency with other models.

        Returns:
            Self
        """
        match self.how:
            case 'simple':
                self.weights = [1/len(self.models) for _ in self.models]
            case _:
                if not self.weights:
                    weights = [m.score/sum([m.score for m in self.metrics]) for m in self.metrics]
                    if self.lower_is_better:
                        weights.reverse()
                    elif self.replace_negative_weights is not False:
                        weights = [self.replace_negative_weights if i < 0 else i for i in weights]
                    self.weights = weights
                else:
                    self.weights = [w/sum(self.weights) for w in self.weights]

        return self
    
    def predict(self,X:None=None,in_sample:bool=False,**predict_params:None) -> list[float]:
        """ Makes predictions by combining the forecasts from the specified models using either simple or weighted averaging.
        
        Args:
            X (np.ndarray): The input data. This is the matrix of forecasts from the specified models for the future periods, or the matrix of fitted values from the specified models for the in-sample period.
            in_sample (bool): Whether the predictions being generated are for the in-sample period (i.e. fitted values) or for the future forecast period. 
                This is just for API consistency with other models since the Combo model generates predictions for both the in-sample period and the 
                future forecast period in the same way by combining the forecasts from the specified models using either simple or weighted averaging.
            **predict_params: Ignored for the Combo model since there is no fitting process. This is just for API consistency with other models.
        
        Returns:
            list[float]: The combined predictions.
        """
        return list(np.sum(X * self.weights,axis=1))

    def fit_predict(self,X:None,y:None) -> list[float]:
        """ Fits and predicts on the same dataset. For the Combo model, there is no fitting process since the model simply combines the forecasts from the specified models using either simple or weighted averaging. 
            This method is included for API consistency with other models, but it does not need to do anything for the Combo model.

        Args:
            X (None): Ignored for the Combo model since it does not use an input matrix for fitting. This is just for API consistency with other models.
            y (None): Ignored for the Combo model since it does not use an input matrix for fitting. This is just for API consistency with other models.
            **fit_params: Ignored for the Combo model since there is no fitting process. This is just for API consistency with other models.

        Returns:
            list[float]: The combined predictions.
        """
        self.fit(X,y)
        return self.predict(X)
