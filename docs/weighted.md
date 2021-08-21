## weighted average modeling
- weighted averaging can easily mean data leakage and overfitting
- determining how to weigh each passed regressor should use `'ValidationMetricValue'` only if wishing to set automatically, you can also pass manually set weights
- if your manual weights do not add to one, they will be rebalanced to do so
- if they do add to one, no rebalancing anywhere will be performed
```python
>>> import pandas as pd
>>> import pandas_datareader as pdr
>>> from scalecast.Forecaster import Forecaster

>>> df = pdr.get_data_fred('HOUSTNSA',start='1900-01-01',end='2021-05-01')
>>> f = Forecaster(y=df['HOUSTNSA'],current_dates=df.index)
>>> f.set_test_length(12)
>>> f.generate_future_dates(24) # forecast length
>>> f.set_estimator('arima')
>>> f.manual_forecast(order=(1,1,1),seasonal_order=(2,1,0,12),trend='ct')
>>> f.set_estimator('hwes')
>>> f.manual_forecast(trend='mul',seasonal='add')
>>> f.set_estiamtor('combo')
>>> f.manual_forecast(how='weighted',models=['arima','hwes'],weights=(.75,.25))
>>> print(f.export('model_summaries'))
  ModelNickname Estimator Xvars  ... LevelTestSetMAE LevelTestSetR2  best_model
0         combo     combo  None  ...       29.327446      -7.696709        True
1          hwes      hwes  None  ...       38.758404     -10.243523       False
2         arima     arima  None  ...       48.316691     -21.313984       False
```
### weight rebalancing
- by default, weighted average will use a minmax or maxmin estimator depending on which metric is passed to it to determine the weights
- this means the lowest-performing model is always weighted at 0
- to make sure all models are given some weight, automatic rebalancing is performed so that .1 is added to all standardized values, then all values are rebalanced to add to 1
- this can be adjusted in the `rebalance_weights` parameter
  - the higher you set this value, the closer the weighted average becomes to a simple average, 0 means the worst model gets no weight
```python
import pandas as pd
import pandas_datareader as pdr
import matplotlib.pyplot as plt
import seaborn as sns
from scalecast.Forecaster import Forecaster

df = pdr.get_data_fred('HOUSTNSA',start='1900-01-01')
f = Forecaster(y=df['HOUSTNSA'],current_dates=df.index) # to initialize, specify y and current_dates (must be arrays of the same length)

f.set_test_length(12) # specify a test length for your models--it's a good idea to keep this the same for all forecasts
f.generate_future_dates(25) # this will create future dates that are on the same interval as the current dates and it will also set the forecast length
f.add_ar_terms(4) # add AR terms before differencing
f.add_AR_terms((2,12)) # seasonal AR terms
f.adf_test() # will print out whether it thinks the series is stationary and return a bool representing stationarity based on the augmented dickey fuller test
f.diff() # differences the y term and all ar terms to make a series stationary (also supports 2-level integration)
f.add_seasonal_regressors('month','dayofyear','week',raw=False,sincos=True) # uses pandas attributes: raw=True creates integers (default), sincos=True creates wave functions (not default), dummy=True creates dummy vars (not default)
f.add_seasonal_regressors('year')
f.add_covid19_regressor() # dates are flexible, default is from when disney world closed to when US CDC lifted mask recommendations
f.add_time_trend()
f.add_combo_regressors('t','COVID19') # multiplies regressors together
f.add_poly_terms('t',pwr=3) # by default, creates an order 2 regressor, n-order polynomial terms are allowed
f.set_validation_length(6) # length, different than test_length, to tune the hyperparameters 

# automatically tune and forecast with a series of models
models = ('mlr','knn','svr','xgboost','gbt','elasticnet','mlp','prophet')
for m in models:
  f.set_estimator(m)
  #f.ingest_grid('mlr') # manually pull any grid name that is specified in Grids.py
  f.tune() # by default, will pull the grid with the same name as the estimator (mlr will pull the mlr grid, etc.)
  f.auto_forecast()

f.set_estimator('combo')
f.manual_forecast(how='weighted',rebalance_weights=0)
```