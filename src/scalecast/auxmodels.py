from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import StackingRegressor
from statsmodels.tsa.vector_ar.vecm import VECM
from scalecast.Forecaster import _sklearn_imports_

class vecm:
    def __init__(
        self,
        exog_coint=None,
        k_ar_diff=1, # always 0
        coint_rank=1,
        deterministic="n",
        seasons=0,
        first_season=0,
    ):
        """ initializes a Vector Error Correction Model.
        uses the statsmodels implementation: https://www.statsmodels.org/dev/generated/statsmodels.tsa.vector_ar.vecm.VECM.html

        Args:
            exog_coint (ndarray): default None. deterministic terms inside the cointegration relation.
            k_ar_diff (int): default 1. number of lagged differences in the model.
            coint_rank (int): cointegration rank.
            deterministic (str): one of {"n", "co", "ci", "lo", "li"}. default "n".
                "n" - no deterministic terms.
                "co" - constant outside the cointegration relation.
                "ci" - constant within the cointegration relation.
                "lo" - linear trend outside the cointegration relation.
                "li" - linear trend within the cointegration relation.
                Combinations of these are possible (e.g. "cili" or "colo" for linear trend with intercept). 
                When using a constant term you have to choose whether you want to restrict it to the cointegration relation 
                (i.e. "ci") or leave it unrestricted (i.e. "co"). Do not use both "ci" and "co". The same applies for "li" 
                and "lo" when using a linear term. 
            seasons (int): default 0. Number of periods in a seasonal cycle. 0 means no seasons.
            first_season (int): default 0. Season of the first observation.

        Returns:
            an initialized model.
        """
        self.k_ar_diff = k_ar_diff
        self.exog_coint = exog_coint
        self.coint_rank = coint_rank
        self.deterministic = deterministic
        self.seasons = seasons
        self.first_season = first_season
        self._scalecast_set = ['dates','freq','n_series'] # these attrs are set when imported into scalecast

    def fit(self,X,y=None):
        """ fits the model.

        Args:
            X (ndarray): the known observations (all known series plus exogenous regressors in a matrix).
                MVForecaster will split endog from exog appropriately.
            y: not required for this model and left None. kept in to be consistent
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
            exog_coint=self.exog_coint,
            coint_rank=self.coint_rank,
            deterministic=self.deterministic,
            seasons=self.seasons,
            first_season=self.first_season,
            dates = self.dates[:X.shape[0]],
            freq = self.freq,
        ).fit()
        
    def predict(self,X):
        """ forecasts into an unknown horizon

        Args:
            X (ndarray): the future sereis values and future exognenous regressors.
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
