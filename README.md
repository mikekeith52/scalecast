
# Scalecast

<p align="center">
  <img src="_static/logo2.png" alt="Scalecast Logo"/>
</p>

## About

Scalecast helps you forecast time series. Here is how to initiate its main object:
```python
from scalecast.Forecaster import Forecaster

f = Forecaster(
    y = array_of_values,
    current_dates = array_of_dates,
    future_dates=fcst_horizon_length,
    test_length = 0, # do you want to test all models? if so, on how many or what percent of observations?
    cis = False, # evaluate conformal confidence intervals for all models?
    metrics = ['rmse','mape','mae','r2'], # what metrics to evaluate over the validation/test sets?
)
```
Uniform ML modeling (with models from a diverse set of libraries, including scikit-learn, statsmodels, and tensorflow), reporting, and data visualizations are offered through the `Forecaster` and `MVForecaster` interfaces. Data storage and processing then becomes easy as all applicable data, predictions, and many derived metrics are contained in a few objects with much customization available through different modules. [Feature requests and issue reporting](https://github.com/mikekeith52/scalecast/issues/new) are welcome!  

## Documentation  
- [Read the Docs](https://scalecast.readthedocs.io/en/latest/)  
- [Introductory Notebook](https://scalecast-examples.readthedocs.io/en/latest/misc/introduction/Introduction2.html)  
- [Change Log](https://scalecast.readthedocs.io/en/latest/change_log.html)  
 
## Popular Features
1. **Easy LSTM Modeling:** setting up an LSTM model for time series using tensorflow is hard. Using scalecast, it's easy. Many tutorials and Kaggle notebooks that are designed for those getting to know the model use scalecast (see the [aritcle](https://medium.com/towards-data-science/exploring-the-lstm-neural-network-model-for-time-series-8b7685aa8cf)).
```python
f.set_estimator('lstm')
f.manual_forecast(
    lags=36,
    batch_size=32,
    epochs=15,
    validation_split=.2,
    activation='tanh',
    optimizer='Adam',
    learning_rate=0.001,
    lstm_layer_sizes=(100,)*3,
    dropout=(0,)*3,
)
```
2. **Auto lag, trend, and seasonality selection:**
```python
f.auto_Xvar_select( # iterate through different combinations of covariates
    estimator = 'lasso', # what estimator?
    alpha = .2, # estimator hyperparams?
    monitor = 'ValidationMetricValue', # what metric to monitor to make decisions?
    cross_validate = True, # cross validate
    cvkwargs = {'k':3}, # 3 folds
)
```
3. **Hyperparameter tuning using grid search and time series cross validation:**
```python
from scalecast import GridGenerator

GridGenerator.get_example_grids()
models = ['ridge','lasso','xgboost','lightgbm','knn']
f.tune_test_forecast(
    models,
    limit_grid_size = .2,
    feature_importance = True, # save pfi feature importance for each model?
    cross_validate = True, # cross validate? if False, using a seperate validation set that the user can specify
    rolling = True, # rolling time series cross validation?
    k = 3, # how many folds?
)
```
4. **Plotting results:** plot test predictions, forecasts, fitted values, and more.
```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(2,1, figsize = (12,6))
f.plot_test_set(models=models,order_by='TestSetRMSE',ax=ax[0])
f.plot(models=models,order_by='TestSetRMSE',ax=ax[1])
plt.show()
```
5. **Pipelines that include transformations, reverting, and backtesting:**
```python
from scalecast import GridGenerator
from scalecast.Pipeline import Transformer, Reverter, Pipeline
from scalecast.util import find_optimal_transformation, backtest_metrics

def forecaster(f):
    models = ['ridge','lasso','xgboost','lightgbm','knn']
    f.tune_test_forecast(
        models,
        limit_grid_size = .2, # randomized grid search on 20% of original grid sizes
        feature_importance = True, # save pfi feature importance for each model?
        cross_validate = True, # cross validate? if False, using a seperate validation set that the user can specify
        rolling = True, # rolling time series cross validation?
        k = 3, # how many folds?
    )

transformer, reverter = find_optimal_transformation(f) # just one of several ways to select transformations for your series

pipeline = Pipeline(
    steps = [
        ('Transform',transformer),
        ('Forecast',forecaster),
        ('Revert',reverter),
    ]
)

f = pipeline.fit_predict(f)
backtest_results = pipeline.backtest(f)
metrics = backtest_metrics(backtest_results)
```
6. **Model stacking:** There are two ways to stack models with scalecast, with the [`StackingRegressor`](https://medium.com/towards-data-science/expand-your-time-series-arsenal-with-these-models-10c807d37558) from scikit-learn or using [its own stacking procedure](https://medium.com/p/7977c6667d29).
```python
from scalecast.auxmodels import auto_arima

f.set_estimator('lstm')
f.manual_forecast(
    lags=36,
    batch_size=32,
    epochs=15,
    validation_split=.2,
    activation='tanh',
    optimizer='Adam',
    learning_rate=0.001,
    lstm_layer_sizes=(100,)*3,
    dropout=(0,)*3,
)

f.set_estimator('prophet')
f.manual_forecast()

auto_arima(f)

# stack previously evaluated models
f.add_signals(['lstm','prophet','arima'])
f.set_estimator('catboost')
f.manual_forecast()
```
7. **Multivariate modeling and multivariate pipelines:**
```python
from scalecast.MVForecaster import MVForecaster
from scalecast.Pipeline import MVPipeline
from scalecast.util import find_optimal_transformation, backtest_metrics
from scalecast import GridGenerator

GridGenerator.get_mv_grids()

def mvforecaster(mvf):
    models = ['ridge','lasso','xgboost','lightgbm','knn']
    mvf.tune_test_forecast(
        models,
        limit_grid_size = .2, # randomized grid search on 20% of original grid sizes
        cross_validate = True, # cross validate? if False, using a seperate validation set that the user can specify
        rolling = True, # rolling time series cross validation?
        k = 3, # how many folds?
    )

mvf = MVForecaster(f1,f2,f3) # can take N Forecaster objects

transformer1, reverter1 = find_optimal_transformation(f1)
transformer2, reverter2 = find_optimal_transformation(f2)
transformer3, reverter3 = find_optimal_transformation(f3)

pipeline = MVPipeline(
    steps = [
        ('Transform',[transformer1,transformer2,transformer3]),
        ('Forecast',mvforecaster),
        ('Revert',[reverter1,reverter2,reverter3])
    ]
)

f1, f2, f3 = pipeline.fit_predict(f1, f2, f3)
backtest_results = pipeline.backtest(f1, f2, f3)
metrics = backtest_metrics(backtest_results)
```

## Installation
- Only the base package is needed to get started:  
  - `pip install --upgrade scalecast`  
- Optional add-ons:  
  - `pip install tensorflow` (for RNN/LSTM on Windows) or `pip install tensorflow-macos` (for MAC/M1)
  - `pip install darts`  
  - `pip install prophet`  
  - `pip install greykite` (for the silverkite model)  
  - `pip install shap` (SHAP feature importance)  
  - `pip install kats` (changepoint detection)  
  - `pip install pmdarima` (auto arima)  
  - `pip install tqdm` (progress bar for notebook)  
  - `pip install ipython` (widgets for notebook)  
  - `pip install ipywidgets` (widgets for notebook)  
  - `jupyter nbextension enable --py widgetsnbextension` (widgets for notebook)  
  - `jupyter labextension install @jupyter-widgets/jupyterlab-manager` (widgets for Lab)  

## Articles and Links

### [Forecasting with Different Model Types](https://scalecast.readthedocs.io/en/latest/Forecaster/_forecast.html)
- Sklearn Univariate
  - [Expand your Time Series Arsenal with These Models](https://towardsdatascience.com/expand-your-time-series-arsenal-with-these-models-10c807d37558)
  - [Notebook](https://scalecast-examples.readthedocs.io/en/latest/sklearn/sklearn.html)
- Sklearn Multivariate
  - [Multiple Series? Forecast Them together with any Sklearn Model](https://towardsdatascience.com/multiple-series-forecast-them-together-with-any-sklearn-model-96319d46269)
  - [Notebook 1](https://scalecast-examples.readthedocs.io/en/latest/multivariate/multivariate.html)
  - [Notebook 2](https://scalecast-examples.readthedocs.io/en/latest/multivariate-beyond/mv.html)  
- RNN 
  - [Exploring the LSTM Neural Network Model for Time Series](https://towardsdatascience.com/exploring-the-lstm-neural-network-model-for-time-series-8b7685aa8cf)
  - [LSTM Notebook](https://scalecast-examples.readthedocs.io/en/latest/lstm/lstm.html)
  - [RNN Notebook](https://scalecast-examples.readthedocs.io/en/latest/rnn/rnn.html)
- ARIMA
  - [Forecast with ARIMA in Python More Easily with Scalecast](https://towardsdatascience.com/forecast-with-arima-in-python-more-easily-with-scalecast-35125fc7dc2e)
  - [Notebook](https://scalecast-examples.readthedocs.io/en/latest/arima/arima.html)
- Theta
  - [Easily Employ A Theta Model For Time Series](https://medium.com/towards-data-science/easily-employ-a-theta-model-for-time-series-b94465099a00)
  - [Notebook](https://scalecast-examples.readthedocs.io/en/latest/theta/theta.html)
- VECM
  - [Employ a VECM to predict FANG Stocks with an ML Framework](https://medium.com/p/52f170ec68e6)
  - [Notebook](https://scalecast-examples.readthedocs.io/en/latest/vecm/vecm.html)
- Stacking
   - [Stacking Time Series Models to Improve Accuracy](https://medium.com/towards-data-science/stacking-time-series-models-to-improve-accuracy-7977c6667d29)
   - [Notebook](https://scalecast-examples.readthedocs.io/en/latest/misc/stacking/custom_stacking.html)
- Other Notebooks
  - [Prophet](https://scalecast-examples.readthedocs.io/en/latest/prophet/prophet.html)
  - [Combo](https://scalecast-examples.readthedocs.io/en/latest/combo/combo.html)
  - [Holt-Winters Exponential Smoothing](https://scalecast-examples.readthedocs.io/en/latest/hwes/hwes.html)
  - [Silverkite](https://scalecast-examples.readthedocs.io/en/latest/silverkite/silverkite.html)

### [Transforming and Reverting](https://scalecast.readthedocs.io/en/latest/Forecaster/SeriesTransformer.html)
- [Time Series Transformations (and Reverting) Made Easy](https://medium.com/towards-data-science/time-series-transformations-and-reverting-made-easy-f4f768c18f63)
- [Notebook](https://scalecast-examples.readthedocs.io/en/latest/transforming/medium_code.html)  
  
### Confidence Intervals
- [Easy Distribution-Free Conformal Intervals for Time Series](https://medium.com/towards-data-science/easy-distribution-free-conformal-intervals-for-time-series-665137e4d907)  
- [Dynamic Conformal Intervals for any Time Series Model](https://towardsdatascience.com/dynamic-conformal-intervals-for-any-time-series-model-d1638aa48527)
- [Notebook 1](https://scalecast-examples.readthedocs.io/en/latest/misc/cis/cis.html)  
- [Notebook 2](https://scalecast-examples.readthedocs.io/en/latest/misc/cis-bt/cis-bt.html)

### Dynamic Validation
- [How Not to be Fooled by Time Series Models](https://towardsdatascience.com/how-not-to-be-fooled-by-time-series-forecasting-8044f5838de3)
- [Model Validation Techniques for Time Series](https://towardsdatascience.com/model-validation-techniques-for-time-series-3518269bd5b3)
- [Notebook](https://scalecast-examples.readthedocs.io/en/latest/misc/validation/validation.html)

### Model Input Selection
- [Variable Reduction Techniques for Time Series](https://medium.com/towards-data-science/variable-reduction-techniques-for-time-series-646743f726d4)
- [Auto Model Specification with ML Techniques for Time Series](https://mikekeith52.medium.com/auto-model-specification-with-ml-techniques-for-time-series-e7b9a90ae9d7)
- [Notebook 1](https://scalecast-examples.readthedocs.io/en/latest/misc/feature-selection/feature_selection.html)
- [Notebook 2](https://scalecast-examples.readthedocs.io/en/latest/misc/auto_Xvar/auto_Xvar.html)

### Scaled Forecasting on Many Series
- [May the Forecasts Be with You](https://towardsdatascience.com/may-the-forecasts-be-with-you-introducing-scalecast-pt-2-692f3f7f0be5)
- [Introductory Notebook Section](https://scalecast-examples.readthedocs.io/en/latest/misc/introduction/Introduction2.html#Scaled-Automated-Forecasting)

### Anomaly Detection
- [Anomaly Detection for Time Series with Monte Carlo Simulations](https://towardsdatascience.com/anomaly-detection-for-time-series-with-monte-carlo-simulations-e43c77ba53c?source=email-85177a9cbd35-1658325190052-activity.collection_post_approved)
- [Notebook1](https://scalecast-examples.readthedocs.io/en/latest/misc/anomalies/anomalies.html)
- [Notebook2](https://github.com/mikekeith52/scalecast-examples/blob/main/misc/anomalies/monte%20carlo/monte%20carlo.ipynb)

## Contributing
- [Contributing.md](https://github.com/mikekeith52/scalecast/blob/main/Contributing.md)
- Want something that's not listed? Open an [issue](https://github.com/mikekeith52/scalecast/issues/new)!  