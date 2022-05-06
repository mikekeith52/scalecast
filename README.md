# 🌄 Scalecast: The practitioner's time series forecasting library

<p align="center">
  <img src="https://github.com/mikekeith52/scalecast/blob/main/assets/logo2.png" />
</p>

## About

Scalecast is a light-weight modeling procedure and wrapper meant for those who are looking for the fastest way possible to apply, tune, and validate many different model classes for forecasting applications. In the Data Science industry, it is often asked of practitioners to deliver predictions and ranges of predictions for several lines of businesses or data slices, 100s or even 1000s. In such situations, it is common to see a simple linear regression or some other quick procedure applied to all lines due to the complexity of the task. This works well enough for people who need to deliver something, but more can be achieved.  

The scalecast package was designed to address this situation and offer advanced machine learning models that can be applied, optimized, and validated quickly. Unlike many libraries, the predictions produced by scalecast are always dynamic by default, not averages of one-step forecasts, so you don't run into the situation where the estimator looks great on the test-set but can't generalize to real data. What you see is what you get, with no attempt to oversell results. If you download a library that looks like it's able to predict the COVID pandemic in your test-set, you probably have a one-step forecast happening under-the-hood. You can't predict the unpredictable, and you won't see such things with scalecast.  

The library provides the Forecaster (for one series) and MVForecaster (for multiple series) wrappers around the following estimators: 

- Any regression model from [Sklearn](https://scikit-learn.org/stable/), including Sklearn APIs (like [Xgboost](https://xgboost.readthedocs.io/en/stable/), and [LightGBM](https://lightgbm.readthedocs.io/en/latest/)).
- Recurrent neural nets from [Keras TensorFlow](https://keras.io/)
- Classic econometric models from [statsmodels](https://www.statsmodels.org/stable/): Holt-Winters Exponential Smoothing and ARIMA
- [Facebook Prophet](https://facebook.github.io/prophet)
- [LinkedIn Silverkite](https://engineering.linkedin.com/blog/2021/greykite--a-flexible--intuitive--and-fast-forecasting-library)
- Average, weighted average, and spliced models

A simple scalecast process to load data, add regressors, and create validated forecasts looks like this:

```python
import pandas as pd
import pandas_datareader as pdr
import matplotlib.pyplot as plt
from scalecast import GridGenerator
from scalecast.Forecaster import Forecaster

# choose models:
models = (
    'mlr',
    'knn',
    'svr',
    'xgboost',
    'elasticnet',
    'mlp',
    'prophet',
)
# read data:
df = pdr.get_data_fred('HOUSTNSA',start='2009-01-01',end='2021-06-01')

# import validation grids to tune models:
GridGenerator.get_example_grids() # saves Grids.py with validation grids for each model that can be used to tune the forecasts

# create the Forecaster object
f = Forecaster(y=df.HOUSTNSA,current_dates=df.index) # to initialize, specify y and current_dates (must be arrays of the same length)

# prepare forecasts:
f.set_test_length(12) # specify a test length for your models - do this before eda
f.generate_future_dates(24) # this will create future dates that are on the same interval as the current dates and it will also set the forecast length
f.add_ar_terms(4) # add lagged y terms before differencing
f.add_AR_terms((2,12)) # seasonal lagged terms
f.integrate() # automatically decides if the y term and all ar terms should be differenced to make the series stationary
f.add_seasonal_regressors('month',raw=False,sincos=True) # uses pandas attributes: raw=True creates integers (default), sincos=True creates fourier transformations, dummy=True also available
f.add_seasonal_regressors('year') # to capture the yearly trend
f.add_covid19_regressor() # dates are flexible, default is from when disney world closed to when US CDC lifted mask recommendations, the dates most likely to have affected the economy
f.add_time_trend()
f.set_validation_length(6) # length, different than test_length, to tune the hyperparameters 

# call forecasts (tunes hyperparameters, tests, forecasts into future)
f.tune_test_forecast(models)

# plot results
f.plot_test_set(models='top_3',order_by='LevelTestSetMAPE',ci=True) # plots the differenced test set with confidence intervals
plt.show()
f.plot(order_by='LevelTestSetMAPE',level=True) # plots the level forecast
plt.show()
```
![](https://github.com/mikekeith52/scalecast/blob/main/assets/main_forecast_test_set.png)
![](https://github.com/mikekeith52/scalecast/blob/main/assets/main_forecast.png)

## Installation
- Base package only: `pip install scalecast`  
- Prophet installs separately due to how big it is: `pip install fbprophet`
  - to resolve a common installation issue for Anaconda, see this [Stack Overflow post](https://stackoverflow.com/questions/49889404/fbprophet-installation-error-failed-building-wheel-for-fbprophet)
- Greykite also installs separately: `pip install greykite`
- Notebook functions require:
  - `pip install tqdm`
  - `pip install ipython`
  - `pip install ipywidgets`
  - `jupyter nbextension enable --py widgetsnbextension`
  - `jupyter labextension install @jupyter-widgets/jupyterlab-manager` (if using Lab)

## Links
|Links||
|----|----|
|[📚 Read the Docs](https://scalecast.readthedocs.io/en/latest/)|Official scalecast docs|
|[📋 Examples](https://scalecast-examples.readthedocs.io/en/latest/)|Official scalecast notebooks|
|[📓 Binder Notebook](https://mybinder.org/v2/gh/mikekeith52/housing_prices/HEAD?filepath=housing_prices.ipynb)|Play with an example in your browser|
|[🛠️ Change Log](https://scalecast.readthedocs.io/en/latest/change_log.html)|See what's changed|

## What can scalecast do?
|||
|---|---|
|Dynamic Univariate Forecasting|✔️|
|Dynamic Univariate Forecasting with Exogenous Regressors|✔️|
|Dynamic Multivariate Vector Forecasting|✔️|
|Dynamic Multivariate Vector Forecasting with Exogenous Regressors|✔️|
|Hyperparameter Tuning|✔️|
|Backcasting|✔️|
|Model Validation|✔️|
|Model Summary Generation|✔️|
|Future Period Forecasting|✔️|
|Plotting|✔️|
|Bootstrapped Confidence Intervals|✔️|
|Seasonality Capturing|✔️|
|Feature Importance|✔️|
|Feature Selection|✔️|
|Linear Models|✔️|
|Non-linear Models|✔️|
|Tree Models|✔️|
|Dense Neural Networks|✔️|
|Recurrent Neural Networks|✔️|
|ARIMA|✔️|
|Exponential Smoothing|✔️|
|Facebook Prophet|✔️|
|LinkedIn Silverkite|✔️|
|Ensemble Modeling|✔️|
|Any Scikit-learn Regressor or API|✔️|
|Differencing|✔️|
|Undifferencing|✔️|
|Level Results|✔️| 
