from statsmodels.tsa.vector_ar.vecm import VECM

class vecm:
    def __init__(
        self,
        k_ar_diff=1, # always 0
        coint_rank=1,
        deterministic="n",
        seasons=0,
        first_season=0,
        freq = None,
    ):
        """ Initializes a Vector Error Correction Model.
        Uses the statsmodels implementation: https://www.statsmodels.org/dev/generated/statsmodels.tsa.vector_ar.vecm.VECM.html.
        See it used with scalecast: https://scalecast-examples.readthedocs.io/en/latest/vecm/vecm.html.

        Args:
            k_ar_diff (int): The number of lags from each series to use in the model.
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
            freq (str): Optional. The frequency of the time-series. 
                A pandas offset or 'B', 'D', 'W', 'M', 'A', or 'Q'.
        """
        self.k_ar_diff = k_ar_diff
        self.coint_rank = coint_rank
        self.deterministic = deterministic
        self.seasons = seasons
        self.first_season = first_season
        self.freq = freq
        self._scalecast_set = ['dates','n_series'] # these attrs are set when imported into scalecast

    def fit(self,X,y=None):
        """ Fits the model.

        Args:
            X (ndarray): The known observations (all known series plus exogenous regressors in a matrix).
                MVForecaster will split endog from exog appropriately.
            y: Not required for this model and left None. Kept in to be consistent
                with other scikit-learn models syntax.
        """
        exog = X[:,self.n_series:]
        X = X[:,:self.n_series]

        self.exogdim = exog.shape[1]
        self.Xdim = X.shape[1]

        exog = None if exog.shape[1] == 0 else exog

        self.mod = VECM(
            X,
            exog=exog,
            k_ar_diff=self.k_ar_diff,
            coint_rank=self.coint_rank,
            deterministic=self.deterministic,
            seasons=self.seasons,
            first_season=self.first_season,
            dates=self.dates[:X.shape[0]],
            freq = self.freq,
        ).fit()
        
    def predict(self,X):
        """ Forecasts into an unknown horizon.

        Args:
            X (ndarray): The future sereis values and future exognenous regressors.
                MVForecaster will split this up appropriately without leaking.
        """
        # testing
        if X.shape[1] == self.Xdim + self.exogdim:
            exog = X[:,self.n_series:]
            exog = None if exog.shape[1] == 0 else exog
        # true forecast
        elif X.shape[1] == self.exogdim:
            exog = None if X.shape[1] == 0 else X
        else: # i don't see how this would ever happen but i'd like to know if it does
            raise Exception(
                'something went wrong. ' 
                'please consider raising this issue on the scalecast issues tab: '
                'https://github.com/mikekeith52/scalecast/issues'
            )
        return self.mod.predict(steps=X.shape[0],exog_fc=exog)

def auto_arima(f,call_me='auto_arima',Xvars=None,train_only=False,**kwargs):
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
        None

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

def mlp_stack(
    f,
    model_nicknames,
    max_samples=0.9,
    max_features=0.5,
    n_estimators=10,
    hidden_layer_sizes=(100,100,100),
    solver='lbfgs',
    call_me='mlp_stack',
    **kwargs,
):
    """ Applies a stacking model using a bagged MLP regressor as the final estimator and adds it to a `Forecaster` or `MVForecaster` object.
    See what it does: https://scalecast-examples.readthedocs.io/en/latest/sklearn/sklearn.html#StackingRegressor.
    This function is not meant to be a model that allows for full customization but full customization is possible when using the 
    `Forecaster` and `MVForecaster` objects.
    Default values usually perform pretty well.
    Recommended to use at least four models in the stack.

    Args:
        f (Forecaster or MVForecaster): The object to add the model to.
        model_nicknames (list-like): The names of models previously evaluated within the object.
            Must be sklearn api models.
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
        **kwargs: Passed to the `manual_forecast()` or `proba_forecast()` method.

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
    from sklearn.neural_network import MLPRegressor
    from sklearn.ensemble import BaggingRegressor
    from sklearn.ensemble import StackingRegressor
    results = f.export('model_summaries',models=model_nicknames)

    if len(results['ModelNickname'].unique()) != len(model_nicknames):
        raise ValueError(
            '{} not found in the Forecaster object.'
            ' The available models to pass to mlp_stack are: {}'.format(
                model_nicknames,
                [m for m in f.history if m in f.sklearn_imports],
            )
        )
    
    estimators = [
        (
            m,
            f.sklearn_imports[
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
    f.manual_forecast(
        estimators=estimators,
        final_estimator=final_estimator,
        call_me=call_me,
        **kwargs,
    )
