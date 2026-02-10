from .util import metrics
from .typing_utils import ScikitLike
from ._utils import _none
import inspect
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


SKLEARN_IMPORTS:dict[str,ScikitLike] = {
    "catboost": CatBoostRegressor,
    "elasticnet": ElasticNet,
    "gbt": GradientBoostingRegressor,
    "knn": KNeighborsRegressor,
    "lasso": Lasso,
    "mlp": MLPRegressor,
    "mlr":LinearRegression,
    "rf": RandomForestRegressor,
    "ridge": Ridge,
    "sgd": SGDRegressor,
    "svr": SVR,
    "xgboost": XGBRegressor,	
}

SKLEARN_ESTIMATORS:list[str] = list(SKLEARN_IMPORTS.keys())
OTHER_ESTIMATORS:list[str] = [
    "arima",
    "hwes",
    "prophet",
    "silverkite",
    "rnn",
    "lstm",
    'naive',
    "tbats",
    "theta",
    "combo",
]

ESTIMATORS:list[str] = SKLEARN_ESTIMATORS + OTHER_ESTIMATORS
METRICS:dict[str,callable] = {
    name:method for name, method in inspect.getmembers(metrics, inspect.isroutine) 
    if not name.startswith('_') and len(inspect.signature(method).parameters) == 2
}

NORMALIZERS:dict[str,callable] = {
    "minmax":MinMaxScaler, 
    "normalize":Normalizer, 
    "scale":StandardScaler, 
    "robust":RobustScaler,
    None:_none,
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
    "#DCDCDC",
    "#E6E6FA",
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

IGNORE_AS_HYPERPARAMS = ["Xvars", "tuned", "plot_loss", "plot_loss_test", "lags", "mvf"]