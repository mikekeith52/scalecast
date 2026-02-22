from .classes import NoScaler, Estimator, MetricStore, ValidatedList
from .Metrics import Metrics
from .models import SKLearnUni, SKLearnMV, VECM, ARIMA, Theta, HWES, TBATS, Prophet, LSTM, RNN, Naive, Combo
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import (
    LinearRegression,
    ElasticNet,
    Lasso,
    Ridge,
    SGDRegressor,
)
from sklearn.preprocessing import (
    MinMaxScaler, 
    StandardScaler, 
    Normalizer,
    RobustScaler,
)
from itertools import cycle
from dataclasses import replace

ESTIMATORS:ValidatedList = ValidatedList([
    Estimator(name='catboost',imported_model=CatBoostRegressor,interpreted_model=SKLearnUni),
    Estimator(name='elasticnet',imported_model=ElasticNet,interpreted_model=SKLearnUni),
    Estimator(name='gbt',imported_model=GradientBoostingRegressor,interpreted_model=SKLearnUni),
    Estimator(name='knn',imported_model=KNeighborsRegressor,interpreted_model=SKLearnUni),
    Estimator(name='lasso',imported_model=Lasso,interpreted_model=SKLearnUni),
    Estimator(name='mlp',imported_model=MLPRegressor,interpreted_model=SKLearnUni),
    Estimator(name='mlr',imported_model=LinearRegression,interpreted_model=SKLearnUni),
    Estimator(name='rf',imported_model=RandomForestRegressor,interpreted_model=SKLearnUni),
    Estimator(name='ridge',imported_model=Ridge,interpreted_model=SKLearnUni),
    Estimator(name='sgd',imported_model=SGDRegressor,interpreted_model=SKLearnUni),
    Estimator(name='svr',imported_model=SVR,interpreted_model=SKLearnUni),
    Estimator(name='xgboost',imported_model=XGBRegressor,interpreted_model=SKLearnUni),
    Estimator(name='arima',imported_model='auto',interpreted_model=ARIMA),
    Estimator(name='hwes',imported_model='auto',interpreted_model=HWES),
    Estimator(name='prophet',imported_model='auto',interpreted_model=Prophet),
    Estimator(name='rnn',imported_model='auto',interpreted_model=RNN),
    Estimator(name='lstm',imported_model='auto',interpreted_model=LSTM),
    Estimator(name='naive',imported_model='auto',interpreted_model=Naive),
    Estimator(name='tbats',imported_model='auto',interpreted_model=TBATS),
    Estimator(name='theta',imported_model='auto',interpreted_model=Theta),
    Estimator(name='combo',imported_model='auto',interpreted_model=Combo),
],enforce_type='Estimator')

MV_ESTIMATORS:ValidatedList[Estimator] = ValidatedList(
    [replace(e,interpreted_model=SKLearnMV) for e in ESTIMATORS if e.interpreted_model is SKLearnUni] + 
    [Estimator(name='vecm',imported_model='auto',interpreted_model=VECM)]
    ,enforce_type='Estimator'
)

METRICS:ValidatedList[MetricStore] = ValidatedList([
    MetricStore(name='rmse',eval_func=Metrics.rmse),
    MetricStore(name='r2',eval_func=Metrics.r2,lower_is_better=False, min_obs_required=2),
    MetricStore(name='mae',eval_func=Metrics.mae),
    MetricStore(name='mape',eval_func=Metrics.mape),
    MetricStore(name='smape',eval_func=Metrics.smape),
    MetricStore(name='abias',eval_func=Metrics.smape),
    MetricStore(name='bias',eval_func=Metrics.smape,lower_is_better=False), # be careful ever choosing based off this metric, but leaving it here because it's good to view
    MetricStore(name='mse',eval_func=Metrics.mse),
],enforce_type='MetricStore')

NORMALIZERS:dict[str,callable] = {
    "minmax":MinMaxScaler, 
    "normalize":Normalizer, 
    "scale":StandardScaler, 
    "robust":RobustScaler,
    None:NoScaler,
}

COLORS = cycle([
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
    "#BDB76B",
    "#DEB887",
])

SERIES_COLORS = cycle([
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
])

IGNORE_AS_HYPERPARAMS = ["Xvars", "lags"]
CLEAR_ATTRS_ON_ESTIMATOR_CHANGE = ["grid","grid_evaluated","best_params","validation_metric_value","actuals"]