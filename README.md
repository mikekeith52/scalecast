# 🌄 Scalecast: Easy dynamic time series forecasting in Python

<p align="center">
  <img src="https://github.com/mikekeith52/scalecast/blob/main/assets/logo2.png" />
</p>

## About

Scalecast is a package meant for those who have at least an intermediate understanding of time series forecasting theory and want to cut the tedious part of data processing, applying autoregression to models, differencing and undifferencing series, and visualizing results, usually on small-to-medium sized datasets (less than 1,000 data points). It can certainly be used for larger, more complex datasets, but probably isn't the best option for such a task. It is meant for standardizing and scaling an approach to many smaller series. For a package with more emphasis on deep learning and larger datasets that offers many of the same features as scalecast, [darts](https://unit8co.github.io/darts/) is recommended.

Scalecast has the following estimators available: 
- Any regression model from [Sklearn](https://scikit-learn.org/stable/), including Sklearn APIs (like [Xgboost](https://xgboost.readthedocs.io/en/stable/), and [LightGBM](https://lightgbm.readthedocs.io/en/latest/)).
- Recurrent neural nets from [Keras TensorFlow](https://keras.io/)
- Classic econometric models from [statsmodels](https://www.statsmodels.org/stable/): Holt-Winters Exponential Smoothing and ARIMA
- [Facebook Prophet](https://facebook.github.io/prophet)
- [LinkedIn Silverkite](https://engineering.linkedin.com/blog/2021/greykite--a-flexible--intuitive--and-fast-forecasting-library)
- Native combo/ensemble model

A very simple scalecast process to load data, add regressors, and create validated forecasts looks like this:

```python
import pandas as pd
import pandas_datareader as pdr
import matplotlib.pyplot as plt
from scalecast import GridGenerator
from scalecast.Forecaster import Forecaster

models = ('mlr','knn','svr','xgboost','elasticnet','mlp','prophet')
df = pdr.get_data_fred('HOUSTNSA',start='2009-01-01',end='2021-06-01')
GridGenerator.get_example_grids() # saves Grids.py with validation grids for each model that can be used to tune the forecasts
f = Forecaster(y=df.HOUSTNSA,current_dates=df.index) # to initialize, specify y and current_dates (must be arrays of the same length)
f.set_test_length(12) # specify a test length for your models - do this before eda
f.generate_future_dates(24) # this will create future dates that are on the same interval as the current dates and it will also set the forecast length
f.add_ar_terms(4) # add AR terms before differencing
f.add_AR_terms((2,12)) # seasonal AR terms
f.integrate() # automatically decides if the y term and all ar terms should be differenced to make the series stationary
f.add_seasonal_regressors('month',raw=False,sincos=True) # uses pandas attributes: raw=True creates integers (default), sincos=True creates wave functions
f.add_seasonal_regressors('year')
f.add_covid19_regressor() # dates are flexible, default is from when disney world closed to when US CDC lifted mask recommendations
f.add_time_trend()
f.set_validation_length(6) # length, different than test_length, to tune the hyperparameters 
f.tune_test_forecast(models)
f.plot_test_set(models='top_3',order_by='LevelTestSetMAPE',ci=True) # plots the differenced test set with confidence intervals
plt.show()
f.plot(order_by='LevelTestSetMAPE',level=True) # plots the level forecast
plt.show()
```
![](https://github.com/mikekeith52/scalecast/blob/main/assets/main_forecast_test_set.png)
![](https://github.com/mikekeith52/scalecast/blob/main/assets/main_forecast.png)

## Installation
1. `pip install scalecast`  
    - installs the base package and most dependencies
2. `pip install fbprophet`
    - only necessary if you plan to forecast with Facebook prophet models
    - to resolve a common installation issue, see this [Stack Overflow post](https://stackoverflow.com/questions/49889404/fbprophet-installation-error-failed-building-wheel-for-fbprophet)
3. `pip install greykite`
    - only necessary if you plan to forecast with LinkedIn Silverkite
4. If using notebook:
    - `pip install tqdm`
    - `pip install ipython`
    - `pip install ipywidgets`
    - `jupyter nbextension enable --py widgetsnbextension`
    - if using Jupyter Lab: `jupyter labextension install @jupyter-widgets/jupyterlab-manager`

## Initialization
```python
from scalecast.Forecaster import Forecaster
array_of_dates = ['2021-01-01','2021-01-02','2021-01-03']
array_of_values = [1,2,3]
f = Forecaster(y=array_of_values, current_dates=array_of_dates)
```

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
