from .types import AvailableModel, XvarValues
from typing import TYPE_CHECKING, Any
if TYPE_CHECKING:
    from .Forecaster import Forecaster

def auto_arima(f:'Forecaster',call_me:str='auto_arima',Xvars:XvarValues=None,train_only:bool=False,**kwargs:Any) -> 'Forecaster':
    """ Adds a forecast to a `Forecaster` object using the auto_arima function from pmdarima.
    This function attempts to find the optimal arima order by minimizing information criteria.

    Args:
        f (Forecaster): The object to add the forecast to.
        call_me (str): Default 'auto_arima'. The name of the resulting model.
        Xvars (str or list-like): Optional. Xvars to add to the model.
        train_only (bool): Default False. Whether to minimize the IC over the training set only.
        **kwargs: Passed to the auto_arima function from pmdarima.
            See https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.auto_arima.html.

    Returns:
        (Forecaster): The passed Forecaster object.

    >>> from scalecast.util import pdr_load
    >>> from scalecast.auxmodels import auto_arima
    >>> f = pdr_load('HOUSTNSA',start='1900-01-01',end='2021-06-01',future_dates=24)
    >>> auto_arima(f,m=12) # saves a model called auto_arima
    >>> print(f.auto_arima_params) # access the selected orders
    """
    import pmdarima
    train = f.y.values[:-f.test_length] if train_only else f.y.values
    auto_model = pmdarima.auto_arima(train,**kwargs)
    best_params = auto_model.get_params()
    f.auto_arima_params = best_params
    order = best_params['order']
    seasonal_order = best_params['seasonal_order']
    trend = best_params['trend']

    f.set_estimator('arima')
    f.manual_forecast(
        order = order,
        seasonal_order = seasonal_order,
        trend = trend,
        Xvars = Xvars,
        call_me = call_me,
    )

    return f

def mlp_stack(
    f:'Forecaster',
    model_nicknames:list[AvailableModel],
    max_samples:float=0.9,
    max_features:float=0.5,
    n_estimators:int=10,
    hidden_layer_sizes:list[int]=[100,100,100],
    solver:str='lbfgs',
    call_me:str='mlp_stack',
    **kwargs:Any,
) -> 'Forecaster':
    """ Applies a stacking model using a bagged MLP regressor as the final estimator and adds it to a `Forecaster` or `MVForecaster` object.
    See what it does: https://scalecast-examples.readthedocs.io/en/latest/sklearn/sklearn.html#StackingRegressor.
    Recommended to use at least four models in the stack.

    Args:
        f (Forecaster or MVForecaster): The object to add the model to.
        model_nicknames (list-like): The names of models previously evaluated within the object.
        max_samples (float or int): Default 0.9.
            The number of samples to draw with replacement from training set to train each base estimator.
            If int, then draw max_samples samples.
            If float, then draw that percentage of samples.
        max_features (float or int): Default 0.5
            The number of features to draw from training set to train each base estimator.
            If int, then draw max_features features.
            If float, then draw that percentage of features.
        n_estimators (int): Default 10.
            The number of base estimators in the ensemble.
        hidden_layer_sizes (tuple): Default (100,100,100).
            The layer/hidden layer sizes for the bagged mlp regressor that is the final estimator in the stacked model.
        solver (str): Default 'lbfgs'.
            The mlp solver.
        call_me (str): Default 'mlp_stack'. The name of the resulting model.
        **kwargs: Passed to the `manual_forecast()` method (can include normalizer argument).

    Returns:
        Forecaster: The passed Forecaster object.

    >>> from scalecast.auxmodels import mlp_stack
    >>> from scalecast import GridGenerator
    >>> GridGenerator.get_example_grids()
    >>> models = ('xgboost','lightgbm','knn','elasticnet')
    >>> f.auto_Xvar_select()
    >>> f.tune_test_forecast(models,cross_validate=True)
    >>> mlp_stack(f,model_nicknames=models) # saves a model called mlp_stack
    >>> f.export('model_summaries',models='mlp_stack')
    """
    from sklearn.neural_network import MLPRegressor
    from sklearn.ensemble import BaggingRegressor
    from sklearn.ensemble import StackingRegressor
    
    estimators = [
        (
            m,
            f.estimators.lookup_item(f.history[m]['Estimator']).imported_model(
                **{k:v for k,v in f.history[m]['HyperParams'].items() if k != 'normalizer'}
            )
        ) for m in model_nicknames
    ]

    final_estimator = BaggingRegressor(
        estimator = MLPRegressor(
            hidden_layer_sizes=tuple(hidden_layer_sizes),
            solver=solver,
        ),
        max_samples = max_samples,
        max_features = max_features,
        n_estimators = n_estimators,
    )

    f.add_sklearn_estimator(StackingRegressor,'stacking')
    f.set_estimator('stacking')
    f.manual_forecast(
        estimators=estimators,
        final_estimator=final_estimator,
        call_me=call_me,
        **kwargs,
    )

    return f
