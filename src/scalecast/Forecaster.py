import typing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import logging
import warnings
import os
from collections import Counter
from scipy import stats
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error, mean_absolute_error
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose

logging.basicConfig(filename='warnings.log',level=logging.WARNING)
logging.captureWarnings(True)
warnings.simplefilter("ignore")

def mape(y,pred):
    return None if 0 in y else mean_absolute_percentage_error(y,pred) # average o(1) worst-case o(n)
def rmse(y,pred):
    return mean_squared_error(y,pred)**.5
def mae(y,pred):
    return mean_absolute_error(y,pred)
def r2(y,pred):
    return r2_score(y,pred)

_estimators_ = {'arima', 'mlr', 'mlp', 'gbt', 'xgboost', 'lightgbm', 'rf', 'prophet', 
                'silverkite', 'hwes', 'elasticnet', 'svr', 'knn', 'combo'}
_metrics_ = {'r2', 'rmse', 'mape', 'mae'}
_determine_best_by_ = {'TestSetRMSE', 'TestSetMAPE', 'TestSetMAE', 'TestSetR2', 'InSampleRMSE', 'InSampleMAPE', 'InSampleMAE',
                        'InSampleR2', 'ValidationMetricValue', 'LevelTestSetRMSE', 'LevelTestSetMAPE', 'LevelTestSetMAE',
                        'LevelTestSetR2', None}
_colors_ = [
    '#FFA500','#DC143C','#00FF7F','#808000','#BC8F8F','#A9A9A9',
    '#8B008B','#FF1493','#FFDAB9','#20B2AA','#7FFFD4','#A52A2A',
    '#DCDCDC','#E6E6FA','#BDB76B','#DEB887'
]*10

class ForecastError(Exception):
    class CannotDiff(Exception):
        pass
    class CannotUndiff(Exception):
        pass
    class NoGrid(Exception):
        pass
    class PlottingError(Exception):
        pass

class Forecaster:
    def __init__(self,
        y,
        current_dates,
        **kwargs):

        self.y = y
        self.current_dates = current_dates
        self.future_dates = pd.Series([])
        self.current_xreg = {} # values should be pandas series (to make differencing work more easily)
        self.future_xreg = {} # values should be lists (to make iterative forecasting work more easily)
        self.history = {}
        self.test_length = 1
        self.validation_length = 1
        self.validation_metric = 'rmse'
        self.integration = 0
        for key, value in kwargs.items():
            setattr(self,key,value)

        self.typ_set() # ensures that the passed values are the right types

    def __str__(self):
        models = self.history.keys()
        if len(models) == 0:
            first_prt = 'Forecaster object with no models evaluated.'
        else:
            first_prt = 'Forecaster object with the following models evaluated: {}.'.format(', '.join(models))
        whole_thing = first_prt + ' Data starts at {}, ends at {}, loaded to forecast out {} periods, has {} regressors.'.format(self.current_dates.min(),self.current_dates.max(),len(self.future_dates),len(self.current_xreg.keys()))
        return whole_thing

    def __repr__(self):
        return self.__str__()

    def _adder(self):
        assert len(self.future_dates) > 0,'before adding regressors, please make sure you have generated future dates by calling generate_future_dates(), set_last_future_date(), or ingest_Xvars_df(use_future_dates=True)'
        
    def _bank_history(self,**kwargs):
        call_me = self.call_me
        self.history[call_me] = {
            'Estimator':self.estimator,
            'Xvars':self.Xvars,
            'HyperParams':{k:v for k,v in kwargs.items() if k not in ('Xvars','normalizer','auto')},
            'Scaler':kwargs['normalizer'] if 'normalizer' in kwargs.keys() else None if self.estimator in ('prophet','combo') else None if hasattr(self,'univariate') else 'minmax',
            'Forecast':self.forecast[:],
            'FittedVals':self.fitted_values[:],
            'Tuned':kwargs['auto'],
            'Integration':self.integration,
            'TestSetLength':self.test_length,
            'TestSetRMSE':self.rmse,
            'TestSetMAPE':self.mape,
            'TestSetMAE':self.mae,
            'TestSetR2':self.r2,
            'TestSetPredictions':self.test_set_pred[:],
            'TestSetActuals':self.test_set_actuals[:],
            'InSampleRMSE':rmse(self.y.values,self.fitted_values),
            'InSampleMAPE':mape(self.y.values,self.fitted_values),
            'InSampleMAE':mae(self.y.values,self.fitted_values),
            'InSampleR2':r2(self.y.values,self.fitted_values),
        }

        if kwargs['auto']:
            self.history[call_me]['ValidationSetLength'] = self.validation_length
            self.history[call_me]['ValidationMetric'] = self.validation_metric
            self.history[call_me]['ValidationMetricValue'] = self.validation_metric_value

        for attr in ('univariate','first_obs','first_dates','grid_evaluated','models','weights'):
            if hasattr(self,attr):
                self.history[call_me][attr] = getattr(self,attr)

        if self.integration > 0:
            first_obs = self.first_obs.copy()
            fcst = self.forecast[::-1]
            integration = self.integration
            y = self.y.to_list()[::-1]
            pred = self.history[call_me]['TestSetPredictions'][::-1]
            if integration == 2:
                first_ = first_obs[1] - first_obs[0]
                y.append(first_)
                y = list(np.cumsum(y[::-1]))[::-1]
            y.append(first_obs[0])
            y = list(np.cumsum(y[::-1]))
            
            if integration == 2:
                fcst.append(self.y.values[-2] + self.y.values[-1])
                pred.append(self.y.values[-(len(pred) + 2)] + self.y.values[-(len(pred) + 1)])
            else:
                fcst.append(y[-1])
                pred.append(y[-(len(pred) + 1)])

            fcst = list(np.cumsum(fcst[::-1]))[1:]
            pred = list(np.cumsum(pred[::-1]))[1:]

            if integration == 2:
                fcst.reverse()
                fcst.append(y[-1])
                fcst = list(np.cumsum(fcst[::-1]))[1:]
                pred.reverse()
                pred.append(y[-(len(pred) + 1)])
                pred = list(np.cumsum(pred[::-1]))[1:]

            self.history[call_me]['LevelForecast'] = fcst[:]
            self.history[call_me]['LevelY'] = y[integration:]
            self.history[call_me]['LevelTestSetPreds'] = pred
            self.history[call_me]['LevelTestSetRMSE'] = rmse(y[-len(pred):],pred)
            self.history[call_me]['LevelTestSetMAPE'] = mape(y[-len(pred):],pred)
            self.history[call_me]['LevelTestSetMAE'] = mae(y[-len(pred):],pred)
            self.history[call_me]['LevelTestSetR2'] = r2(y[-len(pred):],pred)
        else: # better to have these attributes populated for all series
            self.history[call_me]['LevelForecast'] = self.forecast[:]
            self.history[call_me]['LevelY'] = self.y.to_list()
            self.history[call_me]['LevelTestSetPreds'] = self.test_set_pred[:]
            self.history[call_me]['LevelTestSetRMSE'] = self.rmse
            self.history[call_me]['LevelTestSetMAPE'] = self.mape
            self.history[call_me]['LevelTestSetMAE'] = self.mae
            self.history[call_me]['LevelTestSetR2'] = self.r2

    def _set_summary_stats(self):
        results_summary = self.regr.summary()
        results_as_html = results_summary.tables[1].as_html()
        self.summary_stats = pd.read_html(results_as_html, header=0, index_col=0)[0]

    def _bank_fi_to_history(self):
        call_me = self.call_me
        self.history[call_me]['feature_importance'] = self.feature_importance

    def _bank_summary_stats_to_history(self):
        call_me = self.call_me
        self.history[call_me]['summary_stats'] = self.summary_stats

    def _parse_normalizer(self,X_train,normalizer):
        if normalizer == 'minmax':
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
            scaler.fit(X_train)
        elif normalizer == 'scale':
            from sklearn.preprocessing import Normalizer
            scaler = Normalizer()
            scaler.fit(X_train)
        else:
            scaler = None
        return scaler

    def _train_test_split(self,X,y,test_size):
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=test_size,shuffle=False)
        return X_train, X_test, y_train, y_test

    def _metrics(self,y,pred):
        self.test_set_actuals = list(y)
        self.test_set_pred = list(pred)
        self.rmse = rmse(y,pred)
        self.r2 = r2(y,pred)
        self.mae = mae(y,pred)
        self.mape = mape(y,pred)

    def _tune(self):
        metric = getattr(self,getattr(self,'validation_metric'))
        for attr in ('r2','rmse','mape','mae','test_set_pred','test_set_actuals'):
            delattr(self,attr)
        return metric

    def _scale(self,scaler,X):
        if not scaler is None:
            return scaler.transform(X)
        else:
            return X

    def _clear_the_deck(self):
        for attr in ('univariate','fitted_values','regr','X','feature_importance','summary_stats','models','weights'):
            try:
                delattr(self,attr)
            except AttributeError:
                pass

    def _prepare_sklearn(self,tune,Xvars):
        if Xvars is None:
            Xvars = list(self.current_xreg.keys())
        if tune:
            y = self.y.to_list()[:-self.test_length]
            X = pd.DataFrame({k:v.to_list() for k, v in self.current_xreg.items()}).iloc[:-self.test_length,:]
            test_size = self.validation_length
        else:
            y = self.y.to_list()
            X = pd.DataFrame({k:v.to_list() for k, v in self.current_xreg.items()})
            test_size = self.test_length
        X = X[Xvars]
        self.Xvars = Xvars
        return Xvars, y, X, test_size

    def _forecast_sklearn(self,scaler,regr,X,y,Xvars,future_dates,future_xreg,true_forecast=False):
        if true_forecast:
            self._clear_the_deck()
        X = self._scale(scaler,X)
        regr.fit(X,y)
        if true_forecast:
            self.regr = regr
            self.X = X
            self.fitted_values = list(regr.predict(X))
        if len([x for x in self.current_xreg.keys() if x.startswith('AR')]) > 0:
            fcst = []
            for i, _ in enumerate(future_dates):
                p = pd.DataFrame({k:[v[i]] for k,v in future_xreg.items() if k in Xvars})
                p = self._scale(scaler,p)
                fcst.append(regr.predict(p)[0])
                if not i == len(future_dates) - 1:
                    for k, v in future_xreg.items():
                        if k.startswith('AR'):
                            ar = int(k[2:])
                            idx = i + 1 - ar
                            if idx > -1:
                                try:
                                    future_xreg[k][i+1] = fcst[idx]
                                except IndexError:
                                    future_xreg[k].append(fcst[idx])
                            else:
                                try:
                                    future_xreg[k][i+1] = self.y.values[idx]
                                except IndexError:
                                    future_xreg[k].append(self.y.values[idx])
        else:
            p = pd.DataFrame(future_xreg)
            p = self._scale(scaler,p)
            fcst = list(regr.predict(p))
        return fcst

    def _full_sklearn(self,fcster,tune,Xvars,normalizer,**kwargs):
        assert len(self.current_xreg.keys()) > 0,f'need at least 1 Xvar to forecast with the {self.estimator} model'
        Xvars, y, X, test_size = self._prepare_sklearn(tune,Xvars)
        X_train, X_test, y_train, y_test = self._train_test_split(X,y,test_size)
        scaler = self._parse_normalizer(X_train,normalizer)
        X_train = self._scale(scaler,X_train)
        X_test = self._scale(scaler,X_test)
        regr = fcster(**kwargs)
        regr.fit(X_train,y_train)
        pred = self._forecast_sklearn(scaler,regr,X_train,y_train,Xvars,self.current_dates.values[-test_size:], {x:v.values[-test_size:] for x,v in self.current_xreg.items()})
        self._metrics(y_test,pred)
        if tune:
            return self._tune()
        else:
            return self._forecast_sklearn(scaler,regr,X,y,Xvars,self.future_dates,self.future_xreg.copy(),true_forecast=True)

    def _forecast_mlp(self,tune=False,Xvars=None,normalizer='minmax',**kwargs):
        """ normalizer: {'scale','minmax',None}, default 'minmax'
        """
        from sklearn.neural_network import MLPRegressor as fcster
        return self._full_sklearn(fcster,tune,Xvars,normalizer,**kwargs)

    def _forecast_mlr(self,tune=False,Xvars=None,normalizer='minmax',**kwargs):
        from sklearn.linear_model import LinearRegression as fcster
        return self._full_sklearn(fcster,tune,Xvars,normalizer,**kwargs)

    def _forecast_xgboost(self,tune=False,Xvars=None,normalizer='minmax',**kwargs):
        from xgboost import XGBRegressor as fcster
        return self._full_sklearn(fcster,tune,Xvars,normalizer,**kwargs)

    def _forecast_lightgbm(self,tune=False,Xvars=None,normalizer='minmax',**kwargs):
        from lightgbm import LGBMRegressor as fcster
        return self._full_sklearn(fcster,tune,Xvars,normalizer,**kwargs)

    def _forecast_gbt(self,tune=False,Xvars=None,normalizer='minmax',**kwargs):
        from sklearn.ensemble import GradientBoostingRegressor as fcster
        return self._full_sklearn(fcster,tune,Xvars,normalizer,**kwargs)

    def _forecast_rf(self,tune=False,Xvars=None,normalizer='minmax',**kwargs):
        from sklearn.ensemble import RandomForestRegressor as fcster
        return self._full_sklearn(fcster,tune,Xvars,normalizer,**kwargs)

    def _forecast_elasticnet(self,tune=False,Xvars=None,normalizer='minmax',**kwargs):
        from sklearn.linear_model import ElasticNet as fcster
        return self._full_sklearn(fcster,tune,Xvars,normalizer,**kwargs)

    def _forecast_svr(self,tune=False,Xvars=None,normalizer='minmax',**kwargs):
        from sklearn.svm import SVR as fcster
        return self._full_sklearn(fcster,tune,Xvars,normalizer,**kwargs)

    def _forecast_knn(self,tune=False,Xvars=None,normalizer='minmax',**kwargs):
        from sklearn.neighbors import KNeighborsRegressor as fcster
        return self._full_sklearn(fcster,tune,Xvars,normalizer,**kwargs)
    
    def _forecast_hwes(self,tune=False,**kwargs):
        from statsmodels.tsa.holtwinters import ExponentialSmoothing as HWES
        y = self.y.to_list()
        if tune:
            y_train = y[:-(self.validation_length + self.test_length)]
            y_test = y[-(self.test_length + self.validation_length):-self.test_length]
        else:
            y_train = y[:-self.test_length]
            y_test = y[-self.test_length:]
        self.Xvars = None
        hwes_train = HWES(y_train,dates=self.current_dates.values[:-self.test_length],freq=self.freq,**kwargs).fit(optimized=True,use_brute=True)
        pred = hwes_train.predict(start=len(y_train),end=len(y_train) + len(y_test) - 1)
        self._metrics(y_test,pred)
        if tune:
            return self._tune()
        else: # forecast
            self._clear_the_deck()
            self.univariate = True
            self.X = None
            regr = HWES(self.y,dates=self.current_dates,freq=self.freq,**kwargs).fit(optimized=True,use_brute=True)
            self.fitted_values = list(regr.fittedvalues)
            self.regr = regr
            self._set_summary_stats()
            return list(regr.predict(start=len(y),end=len(y) + len(self.future_dates) - 1))

    def _forecast_arima(self,tune=False,Xvars=None,**kwargs):
        """ Xvars = 'all' will use all Xvars except any "AR" terms since they are special and incorporated in the model already anyway
        """
        from statsmodels.tsa.arima.model import ARIMA
        Xvars_orig = Xvars
        Xvars = [x for x in self.current_xreg.keys() if not x.startswith('AR')] if Xvars == 'all' else Xvars
        Xvars, y, X, test_size = self._prepare_sklearn(tune,Xvars)
        if len(self.current_xreg.keys()) > 0:
            X_train, X_test, y_train, y_test = self._train_test_split(X,y,test_size)
        else:
            y_train = self.y.values[:test_size]
            y_test = self.y.values[-test_size:]
        if Xvars_orig is None:
            X, X_train, X_test = None, None, None
            self.Xvars = None
        arima_train = ARIMA(y_train,exog=X_train,dates=self.current_dates.values[:-self.test_length],freq=self.freq,**kwargs).fit()
        pred = arima_train.predict(exog=X_test,start=len(y_train),end=len(y_train) + len(y_test) - 1,typ='levels')
        self._metrics(y_test,pred)
        if tune:
            return self._tune()
        else:
            self._clear_the_deck()
            if Xvars_orig is None: self.univariate = True
            self.X = X
            regr = ARIMA(self.y.values[:],exog=X,dates=self.current_dates,freq=self.freq,**kwargs).fit()
            self.fitted_values = list(regr.fittedvalues)
            self.regr = regr
            self._set_summary_stats()
            p = pd.DataFrame({k:v for k,v in self.future_xreg.items() if k in self.Xvars}) if self.Xvars is not None else None
            fcst = regr.predict(exog=p,start=len(y),end=len(y) + len(self.future_dates) - 1, typ = 'levels', dynamic = True)
            return list(fcst)

    def _forecast_prophet(self,tune=False,Xvars=None,cap=None,floor=None,**kwargs):
        """ Xvars = 'all' will use all Xvars except any "AR" terms since they are special and incorporated in the model already anyway
        """
        from fbprophet import Prophet
        X = pd.DataFrame({k:v for k,v in self.current_xreg.items() if not k.startswith('AR')})
        p = pd.DataFrame({k:v for k,v in self.future_xreg.items() if not k.startswith('AR')})
        Xvars = [x for x in self.current_xreg.keys() if not x.startswith('AR')] if Xvars == 'all' else Xvars if Xvars is not None else []
        if cap is not None: X['cap'] = cap
        if floor is not None: x['floor'] = floor
        X['y'] = self.y.to_list()
        X['ds'] = self.current_dates.to_list()
        p['ds'] = self.future_dates.to_list()

        model = Prophet(**kwargs)
        for x in Xvars:
            model.add_regressor(x)
        if tune:
            X_train = X.iloc[:-(self.test_length + self.validation_length)]
            X_test = X.iloc[-(self.test_length + self.validation_length):-self.test_length]
            y_test = X['y'].values[-(self.test_length + self.validation_length):-self.test_length]
            model.fit(X_train)
            pred = model.predict(X_test)
            self._metrics(y_test,pred['yhat'].to_list())
            return self._tune()
        else:
            model.fit(X.iloc[:-self.test_length])
            pred = model.predict(X.iloc[-self.test_length:])
            self._metrics(X['y'].values[-self.test_length:],pred['yhat'].to_list())
            self._clear_the_deck()
            self.X = X[Xvars]
            if len(Xvars) == 0:
                self.univariate = True
                self.X = None
            self.Xvars = Xvars if Xvars != [] else None

            regr = Prophet(**kwargs)
            regr.fit(X)
            self.fitted_values = regr.predict(X)['yhat'].to_list()
            self.regr = regr
            fcst = regr.predict(p)
            return fcst['yhat'].to_list()

    def _forecast_silverkite(self,tune=False,Xvars=None,**kwargs):
        """ requires:
                pip install greykite
            kwargs: https://linkedin.github.io/greykite/docs/0.1.0/html/pages/model_components/0100_introduction.html
        """
        from greykite.framework.templates.autogen.forecast_config import ForecastConfig, MetadataParam, ModelComponentsParam, EvaluationPeriodParam
        from greykite.framework.templates.forecaster import Forecaster as SKForecaster

        Xvars = [x for x in self.current_xreg.keys() if not x.startswith('AR')] if Xvars == 'all' else Xvars if Xvars is not None else []
        def _forecast_sk(df,Xvars,validation_length,test_length,forecast_length):
            test_length = test_length if test_length > 0 else -(df.shape[0]+1)
            validation_length = validation_length if validation_length > 0 else -(df.shape[0]+1)
            pred_df = df.iloc[:-test_length,:]
            if validation_length > 0:
                pred_df.loc[:-validation_length,'y'] = None
            metadata = MetadataParam(time_col="ts",value_col="y",freq=self.freq)
            components = ModelComponentsParam(regressors={'regressor_cols':Xvars} if Xvars is not None else None,**kwargs)
            forecaster = SKForecaster()
            result = forecaster.run_forecast_config(
                df=pred_df,
                config=ForecastConfig(
                    forecast_horizon=forecast_length,
                    metadata_param=metadata,
                    evaluation_period_param=EvaluationPeriodParam(cv_max_splits=0) # makes it very much faster
                )
            )
            return (result.forecast.df['forecast'].to_list(),result.model[-1].summary().info_dict["coef_summary_df"])

        fcst_length = len(self.future_dates)
        ts_df = pd.DataFrame({'ts':self.current_dates.to_list() + self.future_dates.to_list(),'y':self.y.to_list() + [None]*fcst_length})
        reg_df = pd.DataFrame({x:self.current_xreg[x].to_list() + self.future_xreg[x] for x in Xvars})
        df = pd.concat([ts_df,reg_df],axis=1)
        
        if tune:
            y_test = self.y.values[-(self.test_length + self.validation_length):-self.test_length]
            pred = _forecast_sk(df,Xvars,self.validation_length,self.test_length,self.validation_length)[0]
            self._metrics(y_test,pred[-self.validation_length:])
            return self._tune()
        else:
            pred = _forecast_sk(df,Xvars,self.test_length,0,self.test_length)[0]
            self._metrics(self.y.values[-self.test_length:],pred[-self.test_length:])
            self._clear_the_deck()
            self.X = df[Xvars]
            if len(Xvars) == 0:
                self.univariate = True
                self.X = None
            self.Xvars = Xvars if Xvars != [] else None
            self.regr = None # placeholder to make feature importance work
            result = _forecast_sk(df,Xvars,0,0,fcst_length)
            self.summary_stats = result[1]
            self.fitted_values = result[0][:-fcst_length]
            return result[0][-fcst_length:]

    def _forecast_combo(self,how='simple',models='all',determine_best_by='ValidationMetricValue',rebalance_weights=.1,weights=None,splice_points=None):
        """ how: one of {'simple','weighted','splice'}, default 'simple'
                the type of combination
                all test lengths must be the same for all combined models
            models: 'all', starts with "top_", or list-like, default 'all'
                which models to combine
                must be at least 2 in length
                if using list-like object, elements must match model nicknames specified in call_me when forecasting
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
        determine_best_by = determine_best_by if (weights is None) & ((models[:4] == 'top_') | (how == 'weighted')) else None if how != 'weighted' else determine_best_by
        minmax = (str(determine_best_by).endswith('R2')) | ((determine_best_by == 'ValidationMetricValue') & (self.validation_metric.upper() == 'R2')) | (weights is not None)
        models = self._parse_models(models,determine_best_by)
        assert len(models) > 1,f'need at least two models to average, got {models}'
        fcsts = pd.DataFrame({m:self.history[m]['Forecast'] for m in models})
        preds = pd.DataFrame({m:self.history[m]['TestSetPredictions'] for m in models})
        fvs = pd.DataFrame({m:self.history[m]['FittedVals'] for m in models})
        actuals = self.y.values[-preds.shape[0]:]
        if how == 'weighted':
            scale = True
            if weights is None:
                weights = pd.DataFrame({m:[self.history[m][determine_best_by]] for m in models}) # always use r2 since higher is better (could use maxmin scale for other metrics?)
            else:
                assert len(weights) == len(models),'must pass as many weights as models'
                assert not isinstance(weights,str),f'weights argument not recognized: {weights}'
                weights = pd.DataFrame(zip(models,weights)).set_index(0).transpose()
                if weights.sum(axis=1).values[0] == 1:
                    scale = False
                    rebalance_weights=0
            try:
                assert rebalance_weights >= 0,'when using a weighted average, rebalance_weights must be numeric and at least 0 in value'
                if scale:
                    if minmax:
                        weights = (weights - weights.min(axis=1).values[0])/(weights.max(axis=1).values[0] - weights.min(axis=1).values[0]) # minmax scaler
                    else:
                        weights = (weights - weights.max(axis=1).values[0])/(weights.min(axis=1).values[0] - weights.max(axis=1).values[0]) # maxmin scaler
                weights+=rebalance_weights # by default, add .1 to every value here so that every model gets some weight instead of 0 for the worst one
                weights = weights/weights.sum(axis=1).values[0]
                pred = (preds * weights.values[0]).sum(axis=1).to_list()
                fv = (fvs * weights.values[0]).sum(axis=1).to_list()
                fcst = (fcsts * weights.values[0]).sum(axis=1).to_list()
            except ZeroDivisionError:
                how = 'simple' # all models have the same test set metric value so it's a simple average (never seen this, but jic)
        if how in ('simple','splice'):
            pred = preds.mean(axis=1).to_list()
            fv = fvs.mean(axis=1).to_list()
            if how == 'simple':
                fcst = fcsts.mean(axis=1).to_list()
            elif how == 'splice':
                assert len(models) == len(splice_points) + 1,'must have exactly 1 more model passed to models as splice points passed to splice_points'
                splice_points = pd.to_datetime(sorted(splice_points)).to_list()
                future_dates = self.future_dates.to_list()
                assert np.array([p in future_dates for p in splice_points]).all(), 'all elements in splice_points must be datetime objects or str in yyyy-mm-dd format and must be in future_dates attribute'
                fcst = [None]*len(future_dates)
                start = 0
                for i, _ in enumerate(splice_points):
                    end = [idx for idx,v in enumerate(future_dates) if v == splice_points[i]][0]
                    fcst[start:end] = fcsts[models[i]].values[start:end]
                    start = end
                fcst[start:] = fcsts[models[-1]].values[start:]

        self._metrics(actuals,pred)
        self._clear_the_deck()
        self.weights = tuple(weights.values[0]) if weights is not None else None
        self.models = models
        self.fitted_values = fv
        self.Xvars = None
        self.X = None
        self.regr = None
        return fcst

    def _parse_models(self,models,determine_best_by):
        if determine_best_by is None:
            if models[:4] == 'top_':
                raise ValueError('cannot use models starts with "top_" unless the determine_best_by or order_by argument is specified and not None')
            elif models == 'all':
                models = list(self.history.keys())
            elif isinstance(models,str):
                models = [models]
            else:
                models = list(models)
            if len(models) == 0:
                raise ValueError(f'models argument with determine_best_by={determine_best_by} returns no evaluated forecasts')
        else:
            all_models = [m for m,d in self.history.items() if determine_best_by in d.keys()]
            all_models = self.order_fcsts(all_models,determine_best_by)
            if models == 'all':
                models = all_models[:]
            elif models[:4] == 'top_':
                models = all_models[:int(models.split('_')[1])]
            elif isinstance(models,str):
                models = [models]
            else:
                models = [m for m in all_models if m in models]
        return models

    def _diffy(self,n):
        n = int(n)
        assert (n <= 2) & (n >= 0),'diffy cannot be less than 0 or greater than 2'
        y = self.y.copy()
        for i in range(n):
            y = y.diff().dropna()
        return y
            
    def infer_freq(self):
        if not hasattr(self,'freq'):
            self.freq = pd.infer_freq(self.current_dates)
            self.current_dates.freq = self.freq

    def fillna_y(self,how='ffill'):
        """ how: {'backfill', 'bfill', 'pad', 'ffill', None}
        """
        self.y = pd.Series(self.y)
        if how != 'midpoint': # only works if there aren't more than 2 na one after another
            self.y = self.y.fillna(method=how)
        else:
            for i, val in enumerate(self.y.values):
                if val is None:
                    self.y.values[i] = (self.y.values[i-1] + self.y.values[i+1]) / 2 

    def generate_future_dates(self,n):
        """ way to specify future forecast dates by specifying a forecast period
        """
        self.infer_freq()
        self.future_dates = pd.Series(pd.date_range(start=self.current_dates.values[-1],periods=n+1,freq=self.freq).values[1:])

    def set_last_future_date(self,date):
        """ way to specify future forecast dates by specifying the last desired forecasted date and letting pandas infer the dates in between
        """
        self.infer_freq()
        if isinstance(date,str):
            date = datetime.datetime.strptime(date,'%Y-%m-%d')
        self.future_dates = pd.Series(pd.date_range(start=self.current_dates.values[-1],end=date,freq=self.freq).values[1:])

    def typ_set(self):
        self.y = pd.Series(self.y).dropna().astype(np.float64)
        self.current_dates = pd.to_datetime(pd.Series(list(self.current_dates)[-len(self.y):]),infer_datetime_format=True)
        assert len(self.y) == len(self.current_dates)
        self.future_dates = pd.to_datetime(pd.Series(self.future_dates),infer_datetime_format=True)
        for k,v in self.current_xreg.items():
            self.current_xreg[k] = pd.Series(list(v)[-len(self.y):]).astype(np.float64)
            assert len(self.current_xreg[k]) == len(self.y)
            self.future_xreg[k] = [float(x) for x in self.future_xreg[k]]

    def diff(self,i=1):
        if hasattr(self,'first_obs'):
            raise ForecastError.CannotDiff('series has already been differenced, if you want to difference again, use undiff() first, then diff(2)')

        if i == 0:
            return

        assert i in (1,2),f'only 1st and 2nd order integrations supported for now, got i={i}'
        self.first_obs = self.y.values[:i] # np array
        self.first_dates = self.current_dates.values[:i] # np array
        self.integration = i
        for _ in range(i):
            self.y = self.y.diff()
        for k, v in self.current_xreg.items():
            if k.startswith('AR'):
                ar = int(k[2:])
                for _ in range(i):
                    self.current_xreg[k] = v.diff()
                self.future_xreg[k] = [self.y.values[-ar]]

        if hasattr(self,'adf_stationary'):
            delattr(self,'adf_stationary')

    def add_ar_terms(self,n):
        self._adder()
        assert isinstance(n,int),f'n must be an int, got {n}'
        assert n > 0,f'n must be greater than 0, got {n}'
        assert self.integration == 0,"AR terms must be added before differencing (don't worry, they will be differenced too)"
        for i in range(1,n+1):
            self.current_xreg[f'AR{i}'] = pd.Series(np.roll(self.y,i))
            self.future_xreg[f'AR{i}'] = [self.y.values[-i]]

    def add_AR_terms(self,N):
        """ seasonal AR terms
            N: tuple of len 2 (P,m)
        """
        self._adder()
        assert (len(N) == 2) & (not isinstance(N,str)),f'n must be an array-like of length 2 (P,m), got {N}'
        assert self.integration == 0,"AR terms must be added before differencing (don't worry, they will be differenced too)"
        for i in range(N[1],N[1]*N[0] + 1,N[1]):
            self.current_xreg[f'AR{i}'] = pd.Series(np.roll(self.y,i))
            self.future_xreg[f'AR{i}'] = [self.y.values[-i]]

    def ingest_Xvars_df(self,df,date_col='Date',drop_first=False,use_future_dates=False):
        assert df.shape[0] == len(df[date_col].unique()), 'each date supplied must be unique'
        df[date_col] = pd.to_datetime(df[date_col]).to_list()
        df = df.loc[df[date_col] >= self.current_dates.values[0]]
        df = pd.get_dummies(df,drop_first=drop_first)
        current_df = df.loc[df[date_col].isin(self.current_dates)]
        future_df = df.loc[df[date_col] > self.current_dates.values[-1]]
        assert current_df.shape[0] == len(self.y), 'something went wrong--make sure the dataframe spans the entire daterange as y and is at least one observation to the future and specify a date column in date_col parameter'
        if not use_future_dates:
            assert future_df.shape[0] >= len(self.future_dates),'the future dates in the dataframe should be at least the same length as the future dates in the Forecaster object. if you desire to use the dataframe to set the future dates for the object, use use_future_dates=True'
        else:
            self.infer_freq()
            self.future_dates = future_df[date_col]

        for c in [c for c in future_df if c != date_col]:
            self.future_xreg[c] = future_df[c].to_list()[:len(self.future_dates)]
            self.current_xreg[c] = current_df[c]

        for x,v in self.future_xreg.items():
            self.future_xreg[x] = v[:len(self.future_dates)]
            if not len(v) == len(self.future_dates):
                warnings.warn(f'warning: {x} is not the correct length in the future_dates attribute and this can cause errors when forecasting. its length is {len(v)} and future_dates length is {len(future_dates)}')

    def set_test_length(self,n=1):
        assert isinstance(n,int),f'n must be an int, got {n}'
        self.test_length=n

    def set_validation_length(self,n=1):
        assert isinstance(n,int),f'n must be an int, got {n}'
        assert n > 0,f'n must be greater than 1, got {n}'
        if (self.validation_metric == 'r2') & (n == 1):
            raise ValueError('can only set a validation_length of 1 if validation_metric is not r2. try set_validation_metric()')
        self.validation_length=n

    def adf_test(self,critical_pval=0.05,quiet=True,full_res=False,**kwargs):
        res = adfuller(self.y.dropna(),**kwargs)
        if not full_res:
            if res[1] <= critical_pval:
                if not quiet:
                    print('series appears to be stationary')
                self.adf_stationary = True
                return True
            else:
                if not quiet:
                    print('series might not be stationary')
                self.adf_stationary = False
                return False
        else:
            return res

    def plot_acf(self,diffy=False,**kwargs):
        """ https://www.statsmodels.org/dev/generated/statsmodels.graphics.tsaplots.plot_acf.html
        """
        y = self._diffy(diffy)
        return plot_acf(y.values,**kwargs)

    def plot_pacf(self,diffy=False,**kwargs):
        """ https://www.statsmodels.org/dev/generated/statsmodels.graphics.tsaplots.plot_pacf.html
        """
        y = self._diffy(diffy)
        return plot_pacf(y.values,**kwargs)

    def plot_periodogram(self,diffy=False):
        """ https://www.statsmodels.org/0.8.0/generated/statsmodels.tsa.stattools.periodogram.html
        """
        from scipy.signal import periodogram
        y = self._diffy(diffy)
        return periodogram(y.values)

    def seasonal_decompose(self,diffy=False,**kwargs):
        """ https://www.statsmodels.org/stable/generated/statsmodels.tsa.seasonal.seasonal_decompose.html
        """
        self.infer_freq()
        y = self._diffy(diffy)
        X = pd.DataFrame({'y':y.values},index=self.current_dates.values[-len(y):])
        X.index.freq = self.freq
        return seasonal_decompose(X.dropna(),**kwargs)

    def add_seasonal_regressors(self,*args,raw=True,sincos=False,dummy=False,drop_first=False):
        self._adder()
        if not (raw|sincos|dummy):
            raise ValueError('at least one of raw, sincos, dummy must be True')
        for s in args:
            try:
                if s in ('week','weekofyear'):
                    _raw = getattr(self.current_dates.dt.isocalendar(),s)
                else:
                    _raw = getattr(self.current_dates.dt,s)
            except AttributeError:
                raise ValueError(f'cannot set "{s}". see possible values here: https://pandas.pydata.org/docs/reference/api/pandas.Series.dt.year.html')

            try:
                _raw.astype(int)
            except ValueError:
                f'{s} must return an int; use dummy = True to get dummies'

            if s in ('week','weekofyear'):
                _raw_fut = getattr(self.future_dates.dt.isocalendar(),s)
            else:
                _raw_fut = getattr(self.future_dates.dt,s)
            if raw:
                self.current_xreg[s] = _raw
                self.future_xreg[s] = _raw_fut.to_list()
            if sincos:
                _cycles = _raw.max() # not the best way to do this but usually good enough
                self.current_xreg[f'{s}sin'] = np.sin(np.pi*_raw/(_cycles/2))
                self.current_xreg[f'{s}cos'] = np.cos(np.pi*_raw/(_cycles/2))
                self.future_xreg[f'{s}sin'] = np.sin(np.pi*_raw_fut/(_cycles/2)).to_list()
                self.future_xreg[f'{s}cos'] = np.cos(np.pi*_raw_fut/(_cycles/2)).to_list()
            if dummy:
                all_dummies = []
                stg_df = pd.DataFrame({s:_raw.astype(str)})
                stg_df_fut = pd.DataFrame({s:_raw_fut.astype(str)})
                for c,v in pd.get_dummies(stg_df,drop_first=drop_first).to_dict(orient='series').items():
                    self.current_xreg[c] = v
                    all_dummies.append(c)
                for c,v in pd.get_dummies(stg_df_fut,drop_first=drop_first).to_dict(orient='list').items():
                    if c in all_dummies:
                        self.future_xreg[c] = v
                for c in [d for d in all_dummies if d not in self.future_xreg.keys()]:
                    self.future_xreg[c] = [0]*len(self.future_dates)

    def add_time_trend(self,called='t'):
        self._adder()
        self.current_xreg[called] = pd.Series(range(len(self.y)))
        self.future_xreg[called] = list(range(len(self.y) + 1,len(self.y) + 1 + len(self.future_dates)))
        assert len(self.future_xreg[called]) == len(self.future_dates)

    def add_other_regressor(self,called,start,end):
        self._adder()
        if isinstance(start,str):
            start = datetime.datetime.strptime(start,'%Y-%m-%d')
        if isinstance(end,str):
            end = datetime.datetime.strptime(end,'%Y-%m-%d')
        self.current_xreg[called] = pd.Series([1 if (x >= start) & (x <= end) else 0 for x in self.current_dates])
        self.future_xreg[called] = [1 if (x >= start) & (x <= end) else 0 for x in self.future_dates]

    def add_covid19_regressor(self,called='COVID19',start=datetime.datetime(2020,3,15),end=datetime.datetime(2021,5,13)): # default is from when disney world closed to the end of the national (USA) mask mandate
        self._adder()
        self.add_other_regressor(called=called,start=start,end=end)

    def add_combo_regressors(self,*args,sep='_'):
        self._adder()
        assert len(args) > 1,'need at least two variables to combine regressors'
        for i,a in enumerate(args):
            assert not a.startswith('AR'),'no combining AR terms at this time -- it confuses the forecasting mechanism'
            if i == 0:
                self.current_xreg[sep.join(args)] = self.current_xreg[a]
                self.future_xreg[sep.join(args)] = self.future_xreg[a]
            else:
                self.current_xreg[sep.join(args)] = pd.Series([a*b for a, b in zip(self.current_xreg[sep.join(args)],self.current_xreg[a])])
                self.future_xreg[sep.join(args)] = [a*b for a, b in zip(self.future_xreg[sep.join(args)],self.future_xreg[a])]

    def add_poly_terms(self,*args,pwr=2,sep='^'):
        self._adder()
        for a in args:
            assert not a.startswith('AR'),'no polynomial AR terms at this time -- it confuses the forecasting mechanism'
            for i in range(2,pwr+1):
                self.current_xreg[f'{a}{sep}{i}'] = self.current_xreg[a]**i
                self.future_xreg[f'{a}{sep}{i}'] = [x**i for x in self.future_xreg[a]]

    def undiff(self,suppress_error=False):
        """ always drops all regressors (to make sure lengths are right) -- just re-add them if you still want them
            if the series hasn't been differenced yet, will do nothing else except raise an error -- in this case, use suppress_error to control exceptions
        """
        self.typ_set()
        self.current_xreg = {}
        self.future_xreg = {}
        if self.integration == 0:
            if suppress_error:
                return
            else:
                raise ForecastError.CannotUndiff('cannot undiff a series that was never differenced')

        first_obs = self.first_obs.copy()
        first_dates = list(self.first_dates.copy())
        integration = self.integration
        for attr in ('first_obs','first_dates'):
            delattr(self,attr)

        y = self.y.to_list()[::-1]
        current_dates = self.current_dates.to_list()[::-1]
        if integration == 2:
            first_ = first_obs[1] - first_obs[0]
            y.append(first_)
            y = list(np.cumsum(y[::-1]))[::-1]
        y.append(first_obs[0])
        y = np.cumsum(y[::-1])

        current_dates += first_dates[::-1] # correction 2021-07-14
        self.current_dates = pd.Series(current_dates[::-1])
        self.y = pd.Series(y)
        assert (len(self.current_dates) == len(self.y))
        self.integration = 0
        
    def set_estimator(self,estimator):
        """estimator: one of _estimators_"""
        assert estimator in _estimators_,f'estimator must be one of {_estimators_}, got {estimator}'
        self.typ_set()
        if hasattr(self,'estimator'):
            if estimator != self.estimator:
                for attr in ('grid','grid_evaluated','best_params','validation_metric_value'):
                    if hasattr(self,attr):
                        delattr(self,attr)
                self.estimator = estimator
        else:
            self.estimator = estimator

    def ingest_grid(self,grid):
        from itertools import product
        expand_grid = lambda d: pd.DataFrame([row for row in product(*d.values())],columns=d.keys())
        if isinstance(grid,str):
            import Grids
            grid = getattr(Grids,grid)
        grid = expand_grid(grid)
        self.grid = grid

    def limit_grid_size(self,n):
        if n >= 1:
            self.grid = self.grid.sample(n=min(n,self.grid.shape[0])).reset_index(drop=True)
        elif (n < 1) & (n > 0):
            self.grid = self.grid.sample(frac=n).reset_index(drop=True)
        else:
            raise ValueError(f'argment passed to n not usable: {n}')

    def set_validation_metric(self,metric='rmse'):
        assert metric in _metrics_,f'metric must be one of {_metrics_}, got {metric}'
        if (metric == 'r2') & (self.validation_length < 2):
            raise ValueError('can only validate with r2 if the validation length is at least 2, try set_validation_length()')
        self.validation_metric = metric

    def tune(self):
        if not hasattr(self,'grid'):
            try:
                self.ingest_grid(self.estimator)
            except SyntaxError:
                raise
            except:
                raise ForecastError.NoGrid(f'to tune, a grid must be loaded. we tried to load a grid called {self.estimator}, but either the Grids.py file could not be found in the current directory or there is no grid with that name. try ingest_grid() with a dictionary grid passed manually.')

        if self.estimator == 'combo':
            raise ForecastError('combo models cannot be tuned')
            self.best_params = {}
            return

        metrics = []
        for i, v in self.grid.iterrows():
            try:
                metrics.append(getattr(self,f'_forecast_{self.estimator}')(tune=True,**v))
            except TypeError:
                raise
            except Exception as e:
                self.grid.drop(i,axis=0,inplace=True)
                warnings.warn(f'could not evaluate the paramaters: {dict(v)}. error: {e}')

        if len(metrics) > 0:

            self.grid_evaluated = self.grid.copy()
            self.grid_evaluated['validation_length'] = self.validation_length
            self.grid_evaluated['validation_metric'] = self.validation_metric
            self.grid_evaluated['metric_value'] = metrics
            if self.validation_metric == 'r2':
                best_params_idx = self.grid.loc[self.grid_evaluated['metric_value'] == self.grid_evaluated['metric_value'].max()].index.to_list()[0]
                self.best_params = dict(self.grid.loc[best_params_idx])
            else:
                best_params_idx = self.grid.loc[self.grid_evaluated['metric_value'] == self.grid_evaluated['metric_value'].min()].index.to_list()[0]
                self.best_params = dict(self.grid.loc[best_params_idx])

            self.validation_metric_value = self.grid_evaluated.loc[best_params_idx,'metric_value']

        else:
            warnings.warn(f'none of the keyword/value combos stored in the grid could be evaluated for the {self.estimator} model')
            self.best_params = {}

    def manual_forecast(self,call_me=None,**kwargs):
        call_me = self.estimator if call_me is None else call_me
        assert isinstance(call_me,str),'call_me must be a str type or None'
        self.forecast = getattr(self,f'_forecast_{self.estimator}')(**kwargs)
        self.call_me = call_me
        self._bank_history(auto=False,**kwargs)

    def auto_forecast(self,call_me=None):
        call_me = self.estimator if call_me is None else call_me
        assert isinstance(call_me,str),'call_me must be a str type or None'
        if not hasattr(self,'best_params'):
            warnings.warn(f'since tune() has not been called, {self.estimator} model will be run with default parameters')
            self.best_params = {}
        self.forecast = getattr(self,f'_forecast_{self.estimator}')(**self.best_params)
        self.call_me = call_me
        self._bank_history(auto=len(self.best_params.keys()) > 0,**self.best_params)

    def save_feature_importance(self,quiet=True):
        import eli5
        from eli5.sklearn import PermutationImportance
        try:
            perm = PermutationImportance(self.regr).fit(self.X,self.y)
        except TypeError:
            if not quiet: print(f'cannot set feature importance on the {self.estimator} model')
            return
        self.feature_importance = eli5.explain_weights_df(perm,feature_names=self.history[self.call_me]['Xvars']).set_index('feature')
        self._bank_fi_to_history()

    def save_summary_stats(self,quiet=True):
        if not hasattr(self,'summary_stats'):
            if not quiet: print('last model run does not have summary stats')
            return
        self._bank_summary_stats_to_history()

    def keep_smaller_history(self,n):
        if isinstance(n,str):
            n = datetime.datetime.strptime(n,'%Y-%m-%d')
        if (type(n) is datetime.datetime) or (type(n) is pd.Timestamp):
            n = len([i for i in self.current_dates if i >= n])
        assert (isinstance(n,int)) & (n > 2), 'n must be an int, datetime object, or str in yyyy-mm-dd format and there must be more than 2 observations to keep'
        self.y = self.y[-n:]
        self.current_dates = self.current_dates[-n:]
        for k, v in self.current_xreg.items():
            self.current_xreg[k] = v[-n:]

    def order_fcsts(self,models,determine_best_by='TestSetRMSE'):
        assert determine_best_by in _determine_best_by_,f'determine_best_by must be one of {_determine_best_by_}, got {determine_best_by}'
        models_metrics = {m:self.history[m][determine_best_by] for m in models}
        x = [h[0] for h in Counter(models_metrics).most_common()]
        return x if (determine_best_by.endswith('R2')) | ((determine_best_by == 'ValidationMetricValue') & (self.validation_metric.upper() == 'R2')) else x[::-1]

    def get_regressor_names(self):
        return [k for k in self.current_xreg.keys()]

    def get_freq(self):
        return self.freq

    def validate_regressor_names(self):
        try:
            assert sorted(self.current_xreg.keys()) == sorted(self.future_xreg.keys())
        except AssertionError:
            case1 = [k for k in self.current_xreg.keys() if k not in self.future_xreg.keys()]
            case2 = [k for k in self.future_xreg.keys() if k not in self.current_xreg.keys()]
            raise ValueError(f'the following regressors are in current_xreg but not future_xreg: {case1}\nthe following regressors are in future_xreg but not current_xreg {case2}')

    def plot(self,models='all',order_by=None,level=False,print_attr=[]):
        try:
            models = self._parse_models(models,order_by)
        except ValueError:
            sns.lineplot(x=self.current_dates.values,y=self.y.values,label='actuals')
            plt.legend()
            plt.xlabel('Date')
            plt.ylabel('Values')
            plt.title('Plot of y Vals')
            plt.show()
            return

        integration = set([d['Integration'] for m,d in self.history.items() if m in models])
        if len(integration) > 1:
            level = True

        y = self.y.copy()
        if self.integration == 0:
            for _ in range(max(integration)):
                y = y.diff()

        plot = {
            'date':self.current_dates.to_list()[-len(y.dropna()):] if not level else self.current_dates.to_list()[-len(self.history[models[0]]['LevelY']):],
            'actuals':y.dropna().to_list() if not level else self.history[models[0]]['LevelY'],
        }
        print_attr_map = {}
        sns.lineplot(x=plot['date'],y=plot['actuals'],label='actuals')
        for i, m in enumerate(models):
            plot[m] = self.history[m]['Forecast'] if not level else self.history[m]['LevelForecast'] 
            sns.lineplot(x=self.future_dates.to_list(),y=plot[m],color=_colors_[i],label=m)
            print_attr_map[m] = {a:self.history[m][a] for a in print_attr if a in self.history[m].keys()}

        for m, d in print_attr_map.items():
            for k, v in d.items():
                print(f'{m} {k}: {v}')

        plt.legend()
        plt.xlabel('Date')
        plt.ylabel('Values')
        plt.title('Forecast Results')
        plt.show()

    def plot_test_set(self,models='all',order_by=None,include_train=True,level=False):
        models = self._parse_models(models,order_by)
        integration = set([d['Integration'] for m,d in self.history.items() if m in models])
        if len(integration) > 1:
            level = True

        y = self.y.copy()
        if self.integration == 0:
            for _ in range(max(integration)):
                y = y.diff()

        plot = {
            'date':self.current_dates.to_list()[-len(y.dropna()):] if not level else self.current_dates.to_list()[-len(self.history[models[0]]['LevelY']):],
            'actuals':y.dropna().to_list() if not level else self.history[models[0]]['LevelY'],
        }
        if str(include_train).isnumeric():
            assert (include_train > 1) & isinstance(include_train,int), f'include_train must be a bool type or an int greater than 1, got {include_train}'
            plot['actuals'] = plot['actuals'][-include_train:]
            plot['date'] = plot['date'][-include_train:]
        elif isinstance(include_train,bool):
            if not include_train:
                plot['actuals'] = plot['actuals'][-self.test_length:]
                plot['date'] = plot['date'][-self.test_length:]
        else:
            raise ValueError(f'include_train argument not recognized: ({include_train})')

        sns.lineplot(x=plot['date'],y=plot['actuals'],label='actuals')

        for i, m in enumerate(models):
            plot[m] = self.history[m]['TestSetPredictions'] if not level else self.history[m]['LevelTestSetPreds']
            test_dates = self.current_dates.to_list()[-len(plot[m]):]
            sns.lineplot(x=test_dates,y=plot[m],linestyle='--',color=_colors_[i],alpha=0.7,label=m)

        plt.legend()
        plt.xlabel('Date')
        plt.ylabel('Values')
        plt.title('Test Set Results')
        plt.show()
        
    def plot_fitted(self,models='all',order_by=None):
        models = self._parse_models(models,order_by)
        integration = set([d['Integration'] for m,d in self.history.items() if m in models])
        if len(integration) > 1:
            raise ForecastError.PlottingError('cannot plot fitted values when forecasts run at different levels')

        y = self.y.copy()
        if self.integration == 0:
            for _ in range(max(integration)):
                y = y.diff()

        plot = {
            'date':self.current_dates.to_list()[-len(y.dropna()):],
            'actuals':y.dropna().to_list(),
        }
        sns.lineplot(x=plot['date'],y=plot['actuals'],label='actuals')

        for i, m in enumerate(models):
            plot[m] = self.history[m]['FittedVals']
            sns.lineplot(x=plot['date'][-len(plot[m]):],y=plot[m][-len(plot['date']):],linestyle='--',color=_colors_[i],alpha=0.7,label=m)

        plt.legend()
        plt.xlabel('Date')
        plt.ylabel('Values')
        plt.title('Fitted Values')
        plt.show()

    def drop_regressors(self,*args):
        for a in args:
            self.current_xreg.pop(a)
            self.future_xreg.pop(a)

    def pop(self,*args):
        for a in args:
            self.history.pop(a)

    def export(self,
               dfs=['all_fcsts','model_summaries','best_fcst','test_set_predictions','lvl_fcsts'],
               models='all',
               best_model='auto',
               determine_best_by='TestSetRMSE',
               to_excel=False,
               out_path='./',
               excel_name='results.xlsx'):
        if isinstance(dfs,str):
            dfs = [dfs]
        else:
            dfs = list(dfs)
        if len(dfs) == 0:
            raise ValueError('no dfs passed to dfs')
        determine_best_by = determine_best_by if best_model == 'auto' else None
        models = self._parse_models(models,determine_best_by)
        _dfs_ = ['all_fcsts','model_summaries','best_fcst','test_set_predictions','lvl_fcsts']
        _bad_dfs_ = [i for i in dfs if i not in _dfs_]
        
        if len(_bad_dfs_) > 0:
            raise ValueError(f'the values passed to the dfs list not valid: {_bad_dfs_}') 

        best_fcst_name = self.order_fcsts(models,determine_best_by)[0] if best_model == 'auto' else best_model
        output = {}
        if 'all_fcsts' in dfs:
            all_fcsts = pd.DataFrame({'DATE':self.future_dates.to_list()})
            for m in self.history.keys():
                all_fcsts[m] = self.history[m]['Forecast']
            output['all_fcsts'] = all_fcsts
        if 'model_summaries' in dfs:
            cols = [
                'ModelNickname',
                'Estimator',
                'Xvars',
                'HyperParams',
                'Scaler',
                'Tuned',
                'Integration',
                'TestSetLength',
                'TestSetRMSE',
                'TestSetMAPE',
                'TestSetMAE',
                'TestSetR2',
                'LastTestSetPrediction',
                'LastTestSetActual',
                'InSampleRMSE',
                'InSampleMAPE',
                'InSampleMAE',
                'InSampleR2',
                'ValidationSetLength',
                'ValidationMetric',
                'ValidationMetricValue',
                'univariate',
                'models',
                'weights',
                'integration',
                'LevelTestSetRMSE',
                'LevelTestSetMAPE',
                'LevelTestSetMAE',
                'LevelTestSetR2',
                'best_model'
            ]

            model_summaries = pd.DataFrame()
            for m in models:
                model_summary_m = pd.DataFrame({'ModelNickname':[m]})
                for c in cols:
                    if c not in ('ModelNickname','LastTestSetPrediction','LastTestSetActual','best_model'):
                        model_summary_m[c] = [self.history[m][c] if c in self.history[m].keys() else None]
                    elif c == 'LastTestSetPrediction':
                        model_summary_m[c] = [self.history[m]['TestSetPredictions'][-1]]
                    elif c == 'LastTestSetActual':
                        model_summary_m[c] = [self.history[m]['TestSetActuals'][-1]]
                    elif c == 'best_model':
                        model_summary_m[c] = (m == best_fcst_name)
                model_summaries = pd.concat([model_summaries,model_summary_m],ignore_index=True)
            output['model_summaries'] = model_summaries
        if 'best_fcst' in dfs:
            best_fcst = pd.DataFrame({'DATE':self.current_dates.to_list() + self.future_dates.to_list()})
            best_fcst['VALUES'] = self.y.to_list() + self.history[best_fcst_name]['Forecast']
            best_fcst['MODEL'] = ['actual'] * len(self.current_dates) + [best_fcst_name] * len(self.future_dates)
            output['best_fcst'] = best_fcst
        if 'test_set_predictions' in dfs:
            test_set_predictions = pd.DataFrame({'DATE':self.current_dates[-self.test_length:]})
            test_set_predictions['actual'] = self.y.to_list()[-self.test_length:]
            for m in models:
                test_set_predictions[m] = self.history[m]['TestSetPredictions']
            output['test_set_predictions'] = test_set_predictions
        if 'lvl_fcsts' in dfs:
            lvl_fcsts = pd.DataFrame({'DATE':self.future_dates.to_list()})
            for m in models:
                if 'LevelForecast' in self.history[m].keys():
                    lvl_fcsts[m] = self.history[m]['LevelForecast']
            if lvl_fcsts.shape[1] > 1:
                output['lvl_fcsts'] = lvl_fcsts

        if to_excel:
            with pd.ExcelWriter(os.path.join(out_path,excel_name),engine='openpyxl') as writer:
                for k, df in output.items():
                    df.to_excel(writer,sheet_name=k,index=False)

        if len(output.keys()) == 1:
            return list(output.values())[0]
        else:
            return output

    def export_summary_stats(self,model):
        return self.history[model]['summary_stats']

    def export_feature_importance(self,model):
        return self.history[model]['feature_importance']

    def export_validation_grid(self,model):
        return self.history[model]['grid_evaluated']
