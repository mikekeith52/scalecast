## tuning models
- all except combination and lstm models can be tuned with a straightforward process
- `set_validation_length()` will set n periods aside before the test set chronologically to validate different model hyperparameters
- grids can be fed to the object that are dictionaries with a keyword as the key and array-like object as the value
- for each  model, there are different hyperparameters that can be tuned this way -- see the specific model's source documentation
- all combinations will be tried like any other grid, however, combinations that cannot be estimated together will be passed over to not break loops (this issue comes up frequently with hwes estimators)
- most estimators will also accept an `Xvars` and `normalizer` argument and these can be added to the grid
- you can pass the `dynamic_tuning=True` argument for dynamic tuning. by default, this is `False` to improve speed

[tuning grids](#grids.py)  
[grid generator](#grid-generator)  
[limit grid size](#limit-grid-size)  
[combo modeling](#combo-modeling)
[validation metric](#validation-metric)

```python
import pandas as pd
import pandas_datareader as pdr
from scalecast.Forecaster import Forecaster

elasticnet_grid = {
  'alpha':[i/10 for i in range(1,101)],
  'l1_ratio':[0,0.25,0.5,0.75,1],
  'normalizer':['scale','minmax',None]
}

df = pdr.get_data_fred('HOUSTNSA',start='1900-01-01',end='2021-05-01')
f = Forecaster(y=df['HOUSTNSA'],current_dates=df.index)
f.set_test_length(12)
f.generate_future_dates(24) # forecast length
f.add_ar_terms(4)
f.add_AR_terms((2,12)) # seasonal AR terms
f.add_seasonal_regressors('month','dayofyear','week',raw=False,sincos=True)
f.add_seasonal_regressors('year')
f.add_covid19_regressor() # default is from when disney world closed to when U.S. cdc no longer recommended masks but can be changed
f.add_time_trend()
f.add_combo_regressors('t','COVID19') # multiplies time trend and COVID19 regressor
f.add_poly_terms('t') # t^2
f.diff() # non-stationary data forecasts better differenced with this model
f.set_validation_length(6)
f.set_estimator('elasticnet')
f.ingest_grid(elasticnet_grid)
f.tune()
f.auto_forecast()
```

### Grids.py
- instead of placing grids inside your code, you can create, copy/paste, or use [GridGenerator](#grid-generator) functions to place a file called Grids.py to store your grids in the working directory and they will be read in automatically
- you can pass the name of the grid as str to `ingest_grid()` but you can also skip straight to `tune()` and it will automatically search for a grid in Grids.py with the same name as the estimator
```python
# Grids.py
elasticnet = {
  'alpha':[i/10 for i in range(1,101)],
  'l1_ratio':[0.5],
  'normalizer':['scale','minmax',None]  
}

lasso = {
  'alpha':[i/10 for i in range(1,101)],
  'l1_ratio':[1],
  'normalizer':['scale','minmax',None]
}

ridge = {
  'alpha':[i/10 for i in range(1,101)],
  'l1_ratio':[0],
  'normalizer':['scale','minmax',None]
}

# main.py
import pandas as pd
import pandas_datareader as pdr
from scalecast.Forecaster import Forecaster

df = pdr.get_data_fred('HOUSTNSA',start='1900-01-01',end='2021-05-01')
f = Forecaster(y=df['HOUSTNSA'],current_dates=df.index)
f.set_test_length(12)
f.generate_future_dates(24) # forecast length
f.add_ar_terms(4)
f.add_AR_terms((2,12)) # seasonal AR terms
f.add_seasonal_regressors('month','dayofyear','week',raw=False,sincos=True)
f.add_seasonal_regressors('year')
f.add_covid19_regressor() # default is from when disney world closed to when U.S. cdc no longer recommended masks but can be changed
f.add_time_trend()
f.add_combo_regressors('t','COVID19') # multiplies time trend and COVID19 regressor
f.add_poly_terms('t') # t^2
f.diff() # non-stationary data forecasts better differenced with this model
f.set_validation_length(6)
f.set_estimator('elasticnet')

# GRIDS
f.tune() # automatically ingests the elasticnet grid since it is the same as the estimator
f.auto_forecast()

f.ingest_grid('lasso') # ingests the lasso grid in Grids.py
f.tune()
f.auto_forecast(call_me='lasso')

f.ingest_grid('ridge') # ingests the ridge grid in Grids.py
f.tune()
f.auto_forecast(call_me='ridge')
```
### grid generator
- you can write the Grids.py file in /examples/Grid.py to your working directory using `GridGenerator.get_example_grids()`
- you can write empty grids using `GridGenerator.get_empty_grids()`
```python
from scalecast import GridGenerator
GridGenerator.get_example_grids() # writes the Grids.py file from /examples in the working directory, to overwrite an existing grid use overwrite=True
```
### limit grid size
- `Forecaster.limit_grid_size(n)`
- use to limit big grids to a smaller size of randomly kept rows
  - **n**: `int` or `float`
    - if `int`, that many of random rows will be kept
    - if `float`, must be 0 < n < 1 and that percentage of random rows will be kept
```python
import pandas as pd
import pandas_datareader as pdr
from scalecast.Forecaster import Forecaster

mlp = {
  'activation':['relu','tanh'],
  'hidden_layer_sizes':[(25,),(25,50),(25,50,25),(100,100,100),(100,100,100,100)],
  'solver':['lbfgs','adam'],
  'normalizer':['scale','minmax',None],
  'random_state':[20]
  }

df = pdr.get_data_fred('HOUSTNSA',start='1900-01-01',end='2021-05-01')
f = Forecaster(y=df['HOUSTNSA'],current_dates=df.index)
f.set_test_length(12)
f.generate_future_dates(24) # forecast length
f.add_ar_terms(4)
f.add_AR_terms((2,12)) # seasonal AR terms
f.add_seasonal_regressors('month','dayofyear','week',raw=False,sincos=True)
f.add_seasonal_regressors('year')
f.add_covid19_regressor() # default is from when disney world closed to when U.S. cdc no longer recommended masks but can be changed
f.add_time_trend()
f.add_combo_regressors('t','COVID19') # multiplies time trend and COVID19 regressor
f.add_poly_terms('t') # t^2
f.diff() # non-stationary data forecasts better differenced with this model
f.set_validation_length(6)
f.set_estimator('mlp')
f.ingest_grid(mlp)
f.limit_grid_size(10) # random 10 rows
f.ingest_grid(mlp)
f.limit_grid_size(.1) # random 10 percent
f.tune()
f.auto_forecast()
```

### combo modeling
- the only safe way to autamatically select and weight top models for the "combo" estimator is to use in conjunction with auto forecasating
- use `determine_best_by = 'ValidationMetricValue'` (its default)
```python
# Grids.py
elasticnet = {
  'alpha':[i/10 for i in range(1,101)],
  'l1_ratio':[0.5],
  'normalizer':['scale','minmax',None]  
}

lasso = {
  'alpha':[i/10 for i in range(1,101)],
  'l1_ratio':[1],
  'normalizer':['scale','minmax',None]
}

ridge = {
  'alpha':[i/10 for i in range(1,101)],
  'l1_ratio':[0],
  'normalizer':['scale','minmax',None]
}

# main.py
import pandas as pd
import pandas_datareader as pdr
from scalecast.Forecaster import Forecaster

df = pdr.get_data_fred('HOUSTNSA',start='1900-01-01',end='2021-05-01')
f = Forecaster(y=df['HOUSTNSA'],current_dates=df.index)
f.set_test_length(12)
f.generate_future_dates(24) # forecast length
f.add_ar_terms(4)
f.add_AR_terms((2,12)) # seasonal AR terms
f.add_seasonal_regressors('month','dayofyear','week',raw=False,sincos=True)
f.add_seasonal_regressors('year')
f.add_covid19_regressor() # default is from when disney world closed to when U.S. cdc no longer recommended masks but can be changed
f.add_time_trend()
f.add_combo_regressors('t','COVID19') # multiplies time trend and COVID19 regressor
f.add_poly_terms('t') # t^2
f.diff() # non-stationary data forecasts better differenced with this model
f.set_validation_length(6)
f.set_estimator('elasticnet')

# GRIDS
f.tune() # automatically ingests the elasticnet grid since it is the same as the estimator
f.auto_forecast()

f.ingest_grid('lasso') # ingests the lasso grid in Grids.py
f.tune()
f.auto_forecast(call_me='lasso')

f.ingest_grid('ridge') # ingests the ridge grid in Grids.py
f.tune()
f.auto_forecast(call_me='ridge')

# COMBO
f.set_estimator(cobmo)
f.manual_forecast(how='simple',models='top_2',call_me='avg')
f.manual_forecast(how='weighted',models='top_2',call_me='weighted')
```

### validation metric
- you can change validation metrics to any of `'rmse','mape','mae','r2'`
```python
# Grids.py
elasticnet = {
  'alpha':[i/10 for i in range(1,101)],
  'l1_ratio':[0.5],
  'normalizer':['scale','minmax',None]  
}

lasso = {
  'alpha':[i/10 for i in range(1,101)],
  'l1_ratio':[1],
  'normalizer':['scale','minmax',None]
}

ridge = {
  'alpha':[i/10 for i in range(1,101)],
  'l1_ratio':[0],
  'normalizer':['scale','minmax',None]
}

# main.py
import pandas as pd
import pandas_datareader as pdr
from scalecast.Forecaster import Forecaster

df = pdr.get_data_fred('HOUSTNSA',start='1900-01-01',end='2021-05-01')
f = Forecaster(y=df['HOUSTNSA'],current_dates=df.index)
f.set_test_length(12)
f.generate_future_dates(24) # forecast length
f.add_ar_terms(4)
f.add_AR_terms((2,12)) # seasonal AR terms
f.add_seasonal_regressors('month','dayofyear','week',raw=False,sincos=True)
f.add_seasonal_regressors('year')
f.add_covid19_regressor() # default is from when disney world closed to when U.S. cdc no longer recommended masks but can be changed
f.add_time_trend()
f.add_combo_regressors('t','COVID19') # multiplies time trend and COVID19 regressor
f.add_poly_terms('t') # t^2
f.diff() # non-stationary data forecasts better differenced with this model
f.set_validation_length(6)
f.set_validation_metric('r2')
f.set_estimator('elasticnet')

# GRIDS
f.tune() # automatically ingests the elasticnet grid since it is the same as the estimator
f.auto_forecast()

f.ingest_grid('lasso') # ingests the lasso grid in Grids.py
f.tune()
f.auto_forecast(call_me='lasso')

f.ingest_grid('ridge') # ingests the ridge grid in Grids.py
f.tune()
f.auto_forecast(call_me='ridge')

# COMBO
f.set_estimator(cobmo)
f.manual_forecast(how='simple',models='top_2',call_me='avg')
f.manual_forecast(how='weighted',models='top_2',call_me='weighted')
```
