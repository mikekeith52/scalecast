from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import StackingRegressor
from scalecast.Forecaster import _sklearn_imports_

def auto_arima(f,call_me='auto_arima',Xvars=None,**kwargs):
    """ adds a forecast to a `Forecaster` object using the auto_arima function from pmdarima.
    this function attempts to find the optimal arima order by minimizing information criteria
    on the training slice of data only.

    Args:
        f (Forecaster): the object to add the forecast to
        call_me (str): default 'auto_arima'. the name of the resulting model.
        Xvars (str or list-like): optional. Xvars to add to the model.
        **kwargs: passed to the auto_arima function from pmdarima.
            see https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.auto_arima.html

    Returns:
        None

    >>> from scalecast.util import pdr_load
    >>> from scalecast.auxmodels import auto_arima
    >>> f = pdr_load('HOUSTNSA',start='1900-01-01',end='2021-06-01',future_dates=24)
    >>> auto_arima(f,m=12) # saves a model called auto_arima
    >>> print(f.auto_arima_params) # access the selected orders
    """
    import pmdarima
    train = f.y.values[:-f.test_length]
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

def mlp_stack(
    f,
    model_nicknames,
    max_samples=0.9,
    max_features=0.5,
    n_estimators=10,
    hidden_layer_sizes=(100,100,100),
    solver='lbfgs',
    call_me='mlp_stack',
    probabilistic=False,
    **kwargs,
):
    """ applies a stacking model using a bagged MLP regressor as the final estimator and adds it to a `Forecaster` or `MVForecaster` object.
    see what it does: https://scalecast-examples.readthedocs.io/en/latest/sklearn/sklearn.html#StackingRegressor
    this function is not meant to be a model that allows for full customization but full customization is possible when using the 
    `Forecaster` and `MVForecaster` objects.
    default values usually perform pretty well from what we have observed.
    recommended to use at least four models in the stack.

    Args:
        f (Forecaster or MVForecaster): the object to add the model to.
        model_nicknames (list-like): the names of models previously evaluated within the object.
            must be sklearn api models.
        max_samples (float or int): default 0.9.
            the number of samples to draw with replacement from training set to train each base estimator.
            if int, then draw max_samples samples.
            if float, then draw that percentage of samples.
        max_features (float or int): default 0.5
            the number of features to draw from training set to train each base estimator.
            if int, then draw max_features features.
            if float, then draw that percentage of features.
        n_estimators (int): default 10.
            the number of base estimators in the ensemble.
        hidden_layer_sizes (tuple): default (100,100,100).
            the layer/hidden layer sizes for the bagged mlp regressor that is the final estimator in the stacked model.
        solver (str): default 'lbfgs'.
            the mlp solver.
        call_me (str): default 'mlp_stack'. the name of the resulting model.
        probabilistic (bool): default False. whether to use probabilistic modeling.
        **kwargs: passed to the `manual_forecast()` or `proba_forecast()` method.

    Returns:
        None

    >>> from scalecast.util import pdr_load
    >>> from scalecast.auxmodels import mlp_stack
    >>> from scalecast import GridGenerator

    >>> GridGenerator.get_example_grids()
    >>> models = ('xgboost','lightgbm','knn','elasticnet')
    >>> f = pdr_load('HOUSTNSA',start='1900-01-01',end='2021-06-01',future_dates=24)
    >>> f.set_test_length(24)
    >>> f.add_ar_terms(1)
    >>> f.add_AR_terms((1,6))
    >>> f.add_AR_terms((1,12))
    >>> f.add_seasonal_regressors('month',raw=False,sincos=True)
    >>> f.diff()
    >>> f.tune_test_forecast(models,cross_validate=True)
    >>> mlp_stack(f,model_nicknames=models) # saves a model called mlp_stack
    >>> f.export('model_summaries',models='mlp_stack')
    """
    results = f.export('model_summaries')
    
    estimators = [
        (
            m,
             _sklearn_imports_[
                results.loc[
                    results['ModelNickname'] == m,
                    'Estimator'
                ].values[0]
             ](
                 **results.loc[
                    results['ModelNickname'] == m,
                    'HyperParams'
                ].values[0]
             )
        ) for m in model_nicknames
    ]

    final_estimator = BaggingRegressor(
        base_estimator = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            solver=solver,
        ),
        max_samples = max_samples,
        max_features = max_features,
        n_estimators = n_estimators,
    )

    f.add_sklearn_estimator(StackingRegressor,'stacking')
    f.set_estimator('stacking')
    if not probabilistic:
        f.manual_forecast(
            estimators=estimators,
            final_estimator=final_estimator,
            call_me=call_me,
            **kwargs,
        )
    else:
        f.proba_forecast(
            estimators=estimators,
            final_estimator=final_estimator,
            call_me=call_me,
            **kwargs,
        )
