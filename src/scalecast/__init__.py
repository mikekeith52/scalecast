__version__ = '0.18.9'

from .util import metrics
import inspect
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
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

def _none(x):
    return x

__sklearn_imports__ = {
    "catboost": CatBoostRegressor,
    "elasticnet": ElasticNet,
    "gbt": GradientBoostingRegressor,
    "knn": KNeighborsRegressor,
    "lasso": Lasso,
    "lightgbm": LGBMRegressor,
    "mlp": MLPRegressor,
    "mlr":LinearRegression,
    "rf": RandomForestRegressor,
    "ridge": Ridge,
    "sgd": SGDRegressor,
    "svr": SVR,
    "xgboost": XGBRegressor,
}
# estimators
__sklearn_estimators__ = list(__sklearn_imports__.keys())
# to add non-sklearn models, add to the list below
# sklearn estimators go to the _sklearn_imports_ dict at the top
__non_sklearn_estimators__ = [
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
__estimators__ = __sklearn_estimators__ + __non_sklearn_estimators__
__cannot_be_tuned__ = ["combo"]
__can_be_tuned__ = [m for m in __estimators__ if m not in __cannot_be_tuned__]

# only methods with two arguments (a, f) can be used in validation
__metrics__ = {
    name:method for name, method in inspect.getmembers(metrics, inspect.isroutine) 
    if not name.startswith('_') and len(inspect.signature(method).parameters) == 2
}
__normalizer__ = {
    "minmax":MinMaxScaler, 
    "normalize":Normalizer, 
    "scale":StandardScaler, 
    "robust":RobustScaler,
    None:_none,
}
# i do it this way to make mvforecaster work a little better
__colors__ = [
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
__series_colors__ = [
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
] * 10
# keywords that are passed to _bank_history() that I don't want to be recognized as hyperparams
__not_hyperparams__ = ["Xvars", "tuned", "plot_loss", "plot_loss_test", "lags", "mvf"]