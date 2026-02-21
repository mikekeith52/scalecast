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
    DatetimeLike,
)
from .classes import AR
from typing import TYPE_CHECKING, Self, Optional, Any, Literal, Sequence
import pandas as pd
import numpy as np
import warnings
if TYPE_CHECKING:
    from ._Forecaster_parent import Forecaster_parent

class SKLearnUni:
    """
    Docstring for SKlearnEstimator
    """
    def __init__(
        self, 
        f:'Forecaster_parent', 
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
            case False:
                return 1
            case True:
                return steps+1
            case i if i <= 0:
                raise ValueError(f'Invalid value passed to dynamic_testing: {dynamic_testing}')
            case _:
                dynamic_testing = dynamic_testing
            
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
        obs_to_drop = self.max_lag_order
        X = np.array([self.f.current_xreg[x].values[obs_to_drop:].copy() for x in self.Xvars]).T
        self.scaler = self.scaler.fit(X)
        return X

    def generate_future_X(self) -> np.ndarray:
        X = np.array([np.array(self.f.future_xreg[x][:]) for x in self.Xvars]).T
        return X

    @_developer_utils.log_warnings
    def fit(self,X:np.ndarray,y:np.ndarray,**fit_params) -> Self:
        obs_to_drop = self.max_lag_order
        X = self.scaler.transform(X)
        self.regr.fit(X,np.asarray(y)[obs_to_drop:],**fit_params)
        return self

    def predict(self,X,in_sample:bool=False,**predict_params) -> list[float]:
        if self.max_lag_order == 0 or in_sample or (self.dynamic_testing == 1 and self.test_set_actuals):
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

    def fit_predict(self,X,y) -> list[float]:
        self.fit(X,y)
        return self.predict(X)
    
class SKLearnMV:
    def __init__(
        self,
        f:'Forecaster_parent', 
        model:ScikitLike, 
        lags:None|int|list[int]|dict[str,int|list[int]]=1,
        dynamic_testing:DynamicTesting = True, 
        Xvars:XvarValues = 'all', 
        normalizer:NormalizerLike = 'minmax', 
        test_set_actuals:Optional[list[float]]=None,
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
            case False:
                return 1
            case True:
                return steps+1
            case i if i <= 0:
                raise ValueError(f'Invalid value passed to dynamic_testing: {dynamic_testing}')
            case _:
                dynamic_testing = dynamic_testing

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
        current_X = np.array([self.f.current_xreg[x].values.copy() for x in self.Xvars]).T
        future_X = np.array([np.array(self.f.future_xreg[x][:]) for x in self.Xvars]).T

        ylen = len(self.f.y[self.f.names[0]])
        
        no_other_xvars = current_X.ndim == 1
        if no_other_xvars:
            observed_future = np.array([0]*(ylen + len(self.f.future_dates))).reshape(-1,1) # column of 0s
            observed = np.array([0]*ylen).reshape(-1,1)
        else:
            observed_future = np.concatenate([current_X,future_X],axis=0)

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

    def generate_current_X(self):
        X = self._generate_X()[0]
        self.scaler = self.scaler.fit(X)
        return X

    def generate_future_X(self):
        if hasattr(self,'future_X'):
            return self.future_X
        else:
            return self._generate_X()[1]

    @_developer_utils.log_warnings
    def fit(self,X,y,**fit_params):
        for label, series in y.items():
            X_scaled = self.scaler.transform(X)
            self.regr[label].fit(X_scaled,series[-X.shape[0]:].copy(),**fit_params)
        return self

    def predict(self,X,in_sample:bool=False,**predict_params) -> dict[str,list[float]]:
        preds = {}
        if not self.lags or in_sample or (self.dynamic_testing == 1 and self.test_set_actuals):
            for label, regr in self.regr.items():
                X_scaled = self.scaler.transform(X)
                preds[label] = list(regr.predict(X_scaled,**predict_params))
        else:
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
                            X[i+1,idx] = series[label][-lag_no]

        return preds

    def fit_predict(self,X,y):
        self.fit(X,y)
        return self.predict(X)
    
class RNN:
    """
    Docstring for RNN
    """
    def __init__(
        self, 
        f:'Forecaster_parent', 
        model:Literal['auto']='auto', 
        test_set_actuals:Optional[list[float]]=None, 
        Xvars:XvarValues=None,
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
        self.model = model
        self.model.compile(optimizer=self.optimizer,loss=self.loss)

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

    def generate_current_X(self):
        X = self._generate_X()[0]
        if self.n_timesteps > 1:
            X = X[1:-(self.n_timesteps-1)]
        else:
            X = X[1:]

        X = X.reshape(X.shape[0], X.shape[1], 1)
        return X

    def generate_future_X(self):
        X = self._generate_X()[1]
        X = X.reshape(X.shape[0], X.shape[1], 1)
        return X

    @_developer_utils.log_warnings
    def fit(self,X,y,**fit_params:Any) -> Self:
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
        
        self.hist = self.model.fit(X, self.y, **self.fit_kwargs, **fit_params)
        return self

    def predict(self,X,in_sample:bool=False,**predict_params) -> list[float]:
        if in_sample:
            preds = self.model.predict(X,**predict_params)
            preds = [p[0] for p in preds[:-1]] + [p for p in preds[-1]]
        else:
            preds = self.model.predict(X,**predict_params)
            preds = [p for p in preds[0]]
            
        if self.scale_y:
            preds = [p[0] for p in self.lag_scaler.inverse_transform(np.asarray(preds).reshape(-1,1))]
        return preds

    def fit_predict(self,X,y) -> list[float]:
        self.fit(X,y)
        return self.predict(X)
    
class LSTM(RNN):
    """
    Docstring for LSTM
    """
    def __init__(
        self,
        f:'Forecaster_parent',
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
    """
    Docstring for Theta
    """
    def __init__(
        self, 
        f:'Forecaster_parent', 
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
        self.current_X = self.current_actuals
        return self.current_X

    def generate_future_X(self) -> np.ndarray:
        self.future_X = [0]*len(self.f.future_dates)
        return self.future_X

    @_developer_utils.log_warnings
    def fit(self,X,y,**fit_params:Any) -> Self:
        self.regr.fit(X,**fit_params)
        return self

    def predict(self,X,in_sample:bool=False,**predict_params) -> list[float]:
        if not in_sample:
            pred = self.regr.predict(len(X),**predict_params)
            return [p[0] for p in pred.values()]
        else:
            resid = [r[0] for r in self.regr.residuals(self.current_actuals).values()]
            actuals = self.current_actuals.values().flatten()[-len(resid) :]
            return [r + a for r, a in zip(resid, actuals)]
        
    def fit_predict(self,X,y) -> list[float]:
        self.fit(X,y)
        return self.predict(X)

class HWES:
    """
    Docstring for HWES
    """
    def __init__(
        self,
        f:'Forecaster_parent', 
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
        self.regr = HWES(f.y, dates = f.current_dates, freq = f.freq, **kwargs)

    def generate_current_X(self):
        return

    def generate_future_X(self):
        return
    
    @_developer_utils.log_warnings
    def fit(self,X:None=None,y:None=None,optimized:bool=True,use_brute:bool=True,**fit_params:Any) -> Self:
        self.regr = self.regr.fit(optimized=optimized, use_brute=use_brute, **fit_params)
        return self
    
    def predict(self,X:None=None,in_sample:bool=False,**predict_params) -> list[float]:
        if in_sample:
            return list(self.regr.fittedvalues)
        else:
            pred = self.regr.forecast(len(self.f.future_dates), **predict_params)
            return list(pred)
        
    def fit_predict(self,X,y) -> list[float]:
        self.fit(X,y)
        return self.predict(X)

class TBATS:
    """
    Docstring for TBATS
    """
    def __init__(
        self,
        f:'Forecaster_parent',
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
        return np.asarray(self.current_actuals)

    def generate_future_X(self) -> np.ndarray:
        return np.asarray([0]*len(self.f.future_dates))

    @_developer_utils.log_warnings
    def fit(self,X,y,**fit_params:Any) -> Self:
        if self.random_seed:
            np.random.seed(self.random_seed)
        self.regr.fit(X,**fit_params)
        return self

    def predict(self,X,in_sample:bool=False,**predict_params) -> list[float]:
        if not in_sample:
            preds = self.regr.predict(steps = len(X),**predict_params)
            return list(preds)
        else:
            return self.regr.y_hat
        
    def fit_predict(self,X,y) -> list[float]:
        self.fit(X,y)
        return self.predict(X)


class ARIMA:
    """
    Docstring for ARIMA
    """
    def __init__(
        self,
        f:'Forecaster_parent',
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
        self.regr = ARIMA

    def _parse_Xvars(self, Xvars):
        match Xvars:
            case 'all':
                return [x for x in self.f.current_xreg if not isinstance(x,AR)]
            case None:
                return []
            case _:
                return list(Xvars)

    def generate_current_X(self) -> np.ndarray:
        if len(self.Xvars):
            return np.array([self.f.current_xreg[x].values.copy() for x in self.Xvars]).T
        return None

    def generate_future_X(self) -> np.ndarray:
        if len(self.Xvars):
            return np.array([np.array(self.f.future_xreg[x][:]) for x in self.Xvars]).T
        return None

    @_developer_utils.log_warnings
    def fit(self,X,y,**fit_params:Any) -> Self:
        self.regr = self.regr(
            self.current_actuals,
            exog=X,
            dates=self.f.current_dates.to_list(),
            freq=self.f.freq,
            **self.model_kwargs
        ).fit(**fit_params)
        return self

    def predict(
        self,
        X,
        in_sample:bool=False,
        typ:Literal['levels','linear']='levels',
        dynamic:Optional[bool]=True,
        **predict_params
    ) -> list[float]:
        if not in_sample:
            preds = self.regr.predict(
                exog=X,
                start=len(self.current_actuals),
                end=len(self.current_actuals) + len(self.f.future_dates) - 1, 
                typ=typ,
                dynamic=dynamic,
                **predict_params,
            )
            return list(preds)
        else:
            return list(self.regr.fittedvalues)
        
    def fit_predict(self,X,y) -> list[float]:
        self.fit(X,y)
        return self.predict(X)

class Prophet:
    """
    Docstring for Prophet
    """
    def __init__(
        self,
        f:'Forecaster_parent',
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
            
    def generate_current_X(self) -> np.ndarray:
        df = pd.DataFrame({'ds':self.f.current_dates.to_list(),'y':self.current_actuals})
        for x in self.Xvars:
            df[x] = self.f.current_xreg[x][:]
        if self.cap:
            df['cap'] = self.cap
        if self.floor:
            df['floor'] = self.floor
        
        return df

    def generate_future_X(self) -> np.ndarray:
        df = pd.DataFrame({'ds':self.f.future_dates.to_list()})
        for x in self.Xvars:
            df[x] = self.f.future_xreg[x][:]
        return df
    
    @_developer_utils.log_warnings
    def fit(self,X,y,**fit_params:Any) -> Self:
        if callable(self.callback_func):
            self.callback_func(self.regr)

        self.regr = self.regr.fit(X,**fit_params)
        return self
    
    def predict(self,X,in_sample:None=None,**predict_params) -> list[float]:
        preds = self.regr.predict(X)
        return preds['yhat'].to_list()
    
    def fit_predict(self,X,y) -> list[float]:
        self.fit(X,y)
        return self.predict(X)

class Naive:
    """
    Docstring for Naive
    """
    def __init__(
        self,
        f:'Forecaster_parent',
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

    def generate_current_X(self):
        return
    
    def generate_future_X(self):
        return
    
    def fit(self,X:None=None,y:None=None):
        return self
    
    def predict(self,X:None,in_sample:bool=False) -> list[float]:
        if in_sample:
            return self.f.y.shift(self.m).dropna().to_list()
        else:
            return (self.f.y.to_list()[-self.m:] * int(np.ceil(len(self.f.future_dates)/self.m)))[:len(self.f.future_dates)]
        
    def fit_predict(self,X,y) -> list[float]:
        self.fit(X,y)
        return self.predict(X)


class Combo:
    def __init__(
        self,
        f:'Forecaster_parent',
        model:Literal['auto']='auto', 
        test_set_actuals:Optional[list[float]]=None, 
        how:Literal['simple','weighted','splice']="simple",
        models:ModelValues="all",
        determine_best_by:DetermineBestBy="ValidationMetricValue",
        rebalance_weights:ConfInterval=0.1,
        weights:Optional[Sequence[float|int]]=None,
        splice_points:Optional[Sequence[DatetimeLike]]=None,
    ):
        if model != 'auto':
            raise ValueError(f'Unrecognized value passed to model: {model}')
        
        self.f = f
        self.test_set_actuals = test_set_actuals
        self.how = how
        self.models = self._parse_models(models=models, determine_best_by = determine_best_by)
        self.metrics = [f.history[m][determine_best_by] for m in self.models]
        self.rebalance_weights = rebalance_weights
        self.weights = weights
        self.splice_points = splice_points
    
    def _parse_models(self, models:ModelValues, determine_best_by:DetermineBestBy):
        match models:
            case 'all':
                models = [m for m in self.f.history]
                warnings.warn('Combining all models may lead to previous combination models being overwritten.')
            case i if i.startswith('top_'):
                top_n = int(models.split('_')[-1])
                models = self.f.order_fcsts(determine_best_by=determine_best_by)[:top_n]
            case str():
                models = [models]
            case _:
                models = models

        return models
    
    def fit(self,X,y,**fit_params):
        pass