# 🌄 Scalecast: Dynamic Forecasting at Scale

<p align="center">
  <img src="https://github.com/mikekeith52/scalecast/blob/main/assets/logo2.png" />
</p>

## About

This package uses a scaleable forecasting approach in Python with common [scikit-learn](https://scikit-learn.org/stable/) and [statsmodels](https://www.statsmodels.org/stable/), as well as [Facebook Prophet](https://facebook.github.io/prophet/), [Microsoft LightGBM](https://lightgbm.readthedocs.io/en/latest/) and [LinkedIn Silverkite](https://engineering.linkedin.com/blog/2021/greykite--a-flexible--intuitive--and-fast-forecasting-library) models, to forecast time series. Use your own regressors or load the object with its own seasonal, auto-regressive, and other regressors, or combine all of the above. All forecasting is dynamic so that auto-regressive terms can be used without leaking data into the test set, setting it apart from other time-series libraries. Differencing to achieve stationarity is built into the library and metrics can be compared across the time series' original level or first or second difference. This library was written to easily apply and compare many forecasts fairly across the same series.

```python
import pandas as pd
import pandas_datareader as pdr
from scalecast import GridGenerator
from scalecast.Forecaster import Forecaster

models = ('mlr','knn','svr','xgboost','elasticnet','mlp','prophet')
df = pdr.get_data_fred('HOUSTNSA',start='2009-01-01',end='2021-06-01')
GridGenerator.get_example_grids()
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
f.plot(order_by='LevelTestSetMAPE',level=True) # plots the forecast
```
![](assets/main_forecast.png)

  
## Installation
1. `pip install scalecast`  
    - installs the base package and most dependencies
2. `pip install fbprophet`
    - only necessary if you plan to forecast with Facebook prophet models
    - to resolve a common installation issue, see this [Stack Overflow post](https://stackoverflow.com/questions/49889404/fbprophet-installation-error-failed-building-wheel-for-fbprophet)
3. `pip install greykite`
    - only necessary if you plan to forecast with LinkedIn Silverkite
4. If using notebook functions:
    - `pip install ipython`
    - `pip install ipywidgets`
    - `jupyter nbextension enable --py widgetsnbextension`
    - if using Jupyter Lab: `jupyter labextension install @jupyter-widgets/jupyterlab-manager`


## Documentation
|Documentation||
|----|----|
|[📋 Examples](/examples)|Get straight to the process|
|[➡ Towards Data Science Series](https://towardsdatascience.com/introducing-scalecast-a-forecasting-library-pt-1-33b556d9b019)|Read the 3-part series|
|[📓 Binder Notebook](https://mybinder.org/v2/gh/mikekeith52/housing_prices/HEAD?filepath=housing_prices.ipynb)|Play with an example in your browser|
|[🛠️ Change Log](docs/change_log.md)|See what's changed|
|[📚 Documentation Markdown Files](/docs)|Review all high-level concepts in the library|

## Contribute
**The following contributions are needed (contact mikekeith52@gmail.com)**
1. Documentation moved to a proper website with better organization
2. Built-in smoother for more accurate results (possibly from [tsmoothie](https://github.com/cerlymarco/tsmoothie))
2. Confidence intervals for all models (need to be consistently derived and able to toggle on/off or at different levels: 95%, 80%, etc.)
3. More built-in feature analysis
4. More examples, blog posts, and getting the word out
5. Error/issue reporting
6. Better source code documentation
7. More efficient source code without sacrficing dynamic results
