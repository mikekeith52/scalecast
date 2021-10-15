## plotting

[plot](#plot)  
[plot_test_set](#plot_test_set)  
[plot_fitted](#plot_fitted)  
[plot_acf](#plot_acf)  
[plot_pacf](#plot_pacf)  
[plot_periodogram](#plot_periodogram)  
[seasonal_decompose](#seasonal_decompose)  

### plot

- `Forecaster.plot()`
```python
>>> import pandas as pd
>>> import pandas_datareader as pdr
>>> import matplotlib.pyplot as plt
>>> import seaborn as sns

>>> from scalecast.Forecaster import Forecaster

>>> df = pdr.get_data_fred('HOUSTNSA',start='1900-01-01')
>>> f = Forecaster(y=df['HOUSTNSA'],current_dates=df.index) # to initialize, specify y and current_dates (must be arrays of the same length)

>>> f.set_test_length(12) # specify a test length for your models--it's a good idea to keep this the same for all forecasts
>>> f.generate_future_dates(25) # this will create future dates that are on the same interval as the current dates and it will also set the forecast length
>>> f.add_ar_terms(4) # add AR terms before differencing
>>> f.add_AR_terms((2,12)) # seasonal AR terms
>>> f.adf_test() # will print out whether it thinks the series is stationary and return a bool representing stationarity based on the augmented dickey fuller test
>>> f.diff() # differences the y term and all ar terms to make a series stationary (also supports 2-level integration)
>>> f.add_seasonal_regressors('month','dayofyear','week',raw=False,sincos=True) # uses pandas attributes: raw=True creates integers (default), sincos=True creates wave functions (not default), dummy=True creates dummy vars (not default)
>>> f.add_seasonal_regressors('year')
>>> f.add_covid19_regressor() # dates are flexible, default is from when disney world closed to when US CDC lifted mask recommendations
>>> f.add_time_trend()
>>> f.add_combo_regressors('t','COVID19') # multiplies regressors together
>>> f.add_poly_terms('t',pwr=3) # by default, creates an order 2 regressor, n-order polynomial terms are allowed
>>> f.set_validation_length(6) # length, different than test_length, to tune the hyperparameters 

>>> # automatically tune and forecast with a series of models
>>> models = ('mlr','knn','svr','xgboost','gbt','elasticnet','mlp','prophet')
>>> for m in models:
>>>   f.set_estimator(m)
>>>   #f.ingest_grid('mlr') # manually pull any grid name that is specified in Grids.py
>>>   f.tune() # by default, will pull the grid with the same name as the estimator (mlr will pull the mlr grid, etc.)
>>>   f.auto_forecast()

>>> f.plot(models='top_5',order_by='LevelTestSetMAPE',print_attr=['TestSetRMSE','HyperParams','Xvars']) # plots the forecast differences or levels based on the level the forecast was performed on
knn TestSetRMSE: 15.270860125581308
knn HyperParams: {'n_neighbors': 19, 'weights': 'uniform'}
knn Xvars: ['AR1', 'AR2', 'AR3', 'AR4', 'AR12', 'AR24', 'monthsin', 'monthcos', 'dayofyearsin', 'dayofyearcos', 'weeksin', 'weekcos', 'year', 'COVID19', 't', 't_COVID19', 't^2', 't^3']
prophet TestSetRMSE: 15.136371374950649
prophet HyperParams: {'n_changepoints': 2}
prophet Xvars: None
svr TestSetRMSE: 16.67679416471487
svr HyperParams: {'kernel': 'poly', 'degree': 2, 'gamma': 'scale', 'C': 3.0, 'epsilon': 0.1}
svr Xvars: ['AR1', 'AR2', 'AR3', 'AR4', 'AR12', 'AR24', 'monthsin', 'monthcos', 'dayofyearsin', 'dayofyearcos', 'weeksin', 'weekcos', 'year', 'COVID19', 't', 't_COVID19', 't^2', 't^3']
mlp TestSetRMSE: 16.27657072564657
mlp HyperParams: {'activation': 'tanh', 'hidden_layer_sizes': (25, 25), 'solver': 'lbfgs', 'random_state': 20}
mlp Xvars: ['AR1', 'AR2', 'AR3', 'AR4', 'AR12', 'AR24', 'monthsin', 'monthcos', 'dayofyearsin', 'dayofyearcos', 'weeksin', 'weekcos', 'year', 'COVID19', 't', 't_COVID19', 't^2', 't^3']
elasticnet TestSetRMSE: 16.269472253462983
elasticnet HyperParams: {'alpha': 0.1, 'l1_ratio': 0.0}
elasticnet Xvars: ['AR1', 'AR2', 'AR3', 'AR4', 'AR12', 'AR24', 'monthsin', 'monthcos', 'dayofyearsin', 'dayofyearcos', 'weeksin', 'weekcos', 'year', 'COVID19', 't', 't_COVID19', 't^2', 't^3']
```
![](https://github.com/mikekeith52/scalecast/blob/main/assets/plot.png)

### plot_test_set

- `Forecaster.plot_test_set()`
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

# combine models and run manually specified models of other varieties
f.set_estimator('combo')
f.manual_forecast(how='simple',models='top_3',determine_best_by='ValidationMetricValue',call_me='avg') # simple average of top_3 models based on performance in validation
f.manual_forecast(how='weighted',models=models,determine_best_by='ValidationMetricValue',call_me='weighted') # weighted average of all models based on metric specified in determine_best_by (default is the validation metric)

f.plot_test_set(models='top_5',order_by='TestSetR2',include_train=60) # see test-set performance visually of top 5 best models by r2 (last 60 obs only)
```
![](https://github.com/mikekeith52/scalecast/blob/main/assets/plot_test_set.png)

### plot_fitted

- `Forecaster.plot_fitted()`
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

# combine models and run manually specified models of other varieties
f.set_estimator('combo')
f.manual_forecast(how='simple',models='top_3',determine_best_by='ValidationMetricValue',call_me='avg') # simple average of top_3 models based on performance in validation
f.manual_forecast(how='weighted',models=models,determine_best_by='ValidationMetricValue',call_me='weighted') # weighted average of all models based on metric specified in determine_best_by (default is the validation metric)

f.plot_fitted(order_by='TestSetR2') # plot fitted values of all models ordered by r2
```
![](https://github.com/mikekeith52/scalecast/blob/main/assets/plot_fitted.png)

### plot_acf

- `Forecaster.plot_acf(diffy=False,train_only=False,**kwargs)`
  - `plot_acf()` from `statsmodels`
  - **diffy**: `bool` or `int`, default `False `
    - if bool, whether to call the function on the first differenced `y` series
    - if int, will use that many differences in y before passing to plot function
  - **train_only**: `bool`, default `False`
    - whether to plot only training data (new in 0.2.6 to reduce data leakage chances)
  - `**kwargs` passed to the sm function
```python
import pandas as pd
import pandas_datareader as pdr
import matplotlib.pyplot as plt
import seaborn as sns

from scalecast.Forecaster import Forecaster

df = pdr.get_data_fred('HOUSTNSA',start='1900-01-01')
f = Forecaster(y=df['HOUSTNSA'],current_dates=df.index) # to initialize, specify y and current_dates (must be arrays of the same length)
f.set_test_length(12)

# time series exploration
f.plot_acf(train_only=True)
plt.show()
```
![](https://github.com/mikekeith52/scalecast/blob/main/assets/plot_acf.png)

### plot_pacf

- `Forecaster.plot_pacf()`
```python
import pandas as pd
import pandas_datareader as pdr
import matplotlib.pyplot as plt
import seaborn as sns

from scalecast.Forecaster import Forecaster

df = pdr.get_data_fred('HOUSTNSA',start='1900-01-01')
f = Forecaster(y=df['HOUSTNSA'],current_dates=df.index) # to initialize, specify y and current_dates (must be arrays of the same length)
f.set_test_length(12)

# time series exploration
f.plot_pacf(diffy=True,train_only=True)
plt.show()
```
![](https://github.com/mikekeith52/scalecast/blob/main/assets/plot_pacf.png)

### plot_periodogram

- `Forecaster.plot_periodogram()`
```python
import pandas as pd
import pandas_datareader as pdr
import matplotlib.pyplot as plt
import seaborn as sns

from scalecast.Forecaster import Forecaster

df = pdr.get_data_fred('HOUSTNSA',start='1900-01-01')
f = Forecaster(y=df['HOUSTNSA'],current_dates=df.index) # to initialize, specify y and current_dates (must be arrays of the same length)
f.set_test_length(12)

a, b = f.plot_periodogram(diffy=True,train_only=True)
plt.semilogy(a, b)
plt.show()
```
![](https://github.com/mikekeith52/scalecast/blob/main/assets/plot_periodogram.png)

### seasonal_decompose

- `Forecaster.seasonal_decompose()`
```python
import pandas as pd
import pandas_datareader as pdr
import matplotlib.pyplot as plt
import seaborn as sns

from scalecast.Forecaster import Forecaster

df = pdr.get_data_fred('HOUSTNSA',start='1900-01-01')
f = Forecaster(y=df['HOUSTNSA'],current_dates=df.index) # to initialize, specify y and current_dates (must be arrays of the same length)
f.set_test_length(12)

f.seasonal_decompose(train_only=True).plot()
plt.show()
```
![](https://github.com/mikekeith52/scalecast/blob/main/assets/plot_seasonal_decompose.png)