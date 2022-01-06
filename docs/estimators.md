## estimators

[arima](#arima)  
[combo](#combo)  
[elasticnet](#elasticnet)  
[gbt](#gbt)  
[hwes](#hwes)  
[knn](#knn)  
[lightgbm](#lightgbm)  
[lstm](#lstm)  
[mlr](#mlr)  
[mlp](#mlp)  
[prophet](#prophet)  
[rf](#rf)  
[silverkite](#silverkite)  
[svr](#svr)  
[xgboost](#xgboost)  
```python
_estimators_ = {'arima', 'mlr', 'mlp', 'gbt', 'xgboost', 'lstm', 'lightgbm', 'rf', 'prophet', 'silverkite', 'hwes', 'elasticnet', 'svr', 'knn', 'combo'}
```

### arima
- [Stats Models Documentation](https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html)
- uses no Xvars by default but does accept the Xvars argument
- does not accept the normalizer argument
- can be used effectively on level or differenced data
```python
import pandas as pd
import pandas_datareader as pdr
from scalecast.Forecaster import Forecaster

df = pdr.get_data_fred('HOUSTNSA',start='1900-01-01',end='2021-05-01')
f = Forecaster(y=df['HOUSTNSA'],current_dates=df.index)
f.set_test_length(12)
f.generate_future_dates(24) # forecast length
f.set_estimator('arima')
f.manual_forecast(order=(1,1,1),seasonal_order=(2,1,0,12),trend='ct')
```

### combo
- three combination models are available (`specify in manual_forecast(how=...)`):
  - **simple** average of specified models
  - **weighted** average of specified models
    - weights are based on `determine_best_by` parameter -- see [metrics](#error/accuracy-metrics)
  - **splice** of specified models at specified splice point(s)
    - specify splice points in `splice_points`
      - should be one less in length than models
      - models[0] --> :splice_points[0]
      - models[-1] --> splice_points[-1]:
```python
import pandas as pd
import pandas_datareader as pdr
from scalecast.Forecaster import Forecaster

df = pdr.get_data_fred('HOUSTNSA',start='1900-01-01',end='2021-05-01')
f = Forecaster(y=df['HOUSTNSA'],current_dates=df.index)
f.set_test_length(12)
f.generate_future_dates(24) # forecast length
f.set_estimator('hwes')
f.manual_forecast(trend='add',seasonal='add',call_me='hwes_add')
f.manual_forecast(trend='mul',seasonal='mul',call_me='hwes_mul')

f.set_estimator('combo')
f.manual_forecast(how='simple',models=['hwes_add','hwes_mul'],call_me='avg')
f.manual_forecast(how='weighted',determine_best_by='InSampleRMSE',models=['hwes_add','hwes_mul'],call_me='weighted') # this leaks data -- see auto_forecast for better weighted average modeling
f.manual_forecast(how='splice',models=['hwes_add','hwes_mul'],determine_best_by='InSampleRMSE',splice_points=['2022-01-01'],call_me='splice')
```
- the above weighted average model will probably overfit since `determine_best_by` is specified with an in-sample error metric that is partly calculated with the test-set
- the models argument can be a str type beginning with "top_" and that number of models will be averaged, determined by `determine_best_by`
- when using multiple models of the same estimator, be sure to use the `call_me` paramater to differentiate them--otherwise, you will not be able to access all of their statistics later
```python
import pandas as pd
import pandas_datareader as pdr
from scalecast.Forecaster import Forecaster

df = pdr.get_data_fred('HOUSTNSA',start='1900-01-01',end='2021-05-01')
f = Forecaster(y=df['HOUSTNSA'],current_dates=df.index)
f.set_test_length(12)
f.generate_future_dates(24) # forecast length
f.set_estimator('hwes')
f.manual_forecast(trend='add',seasonal='add',call_me='hwes_add')
f.manual_forecast(trend='mul',seasonal='mul',call_me='hwes_mul')
f.manual_forecast(trend=None,seasonal='add',call_me='hwes_add_no_trend')

f.set_estimator('combo')
f.manual_forecast(how='simple',models='top_2',determine_best_by='InSampleRMSE',call_me='avg') # this leaks data
f.manual_forecast(how='weighted',determine_best_by='InSampleRMSE',models='top_2',call_me='weighted') # this leaks data
```
- again, both combo models in the above example include data leakage
- to avoid data leakage with combo modeling, tune your models or manually pass models to `models` parameter and weights to `weights` parameter

### elasticnet
- [Sklearn Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html)
- combines a lasso and ridge regressor
- uses all Xvars and a MinMax normalizer by default
- better on differenced data for non-stationary series
```python
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
f.set_estimator('elasticnet')
f.manual_forecast(alpha=.5,l1_ratio=.5,normalizer='scale')
```
### gbt
- [Sklearn Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html)
- Gradient Boosted Trees
- robust to overfitting
- takes longer to run than xgboost
- uses all Xvars and a MinMax normalizer by default
- better on differenced data for non-stationary series
```python
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
f.diff() # non-stationary data forecasts better differenced with this model
f.set_estimator('gbt')
f.manual_forecast(max_depth=2,normalizer=None)
```

### hwes
- [Stats Models Documentation](https://www.statsmodels.org/dev/generated/statsmodels.tsa.holtwinters.ExponentialSmoothing.html)
- Holt-Winters Exponential Smoothing
- does not accept the normalizer or Xvars argument
- usually better on level data, whether or not the series is stationary
```python
import pandas as pd
import pandas_datareader as pdr
from scalecast.Forecaster import Forecaster

df = pdr.get_data_fred('HOUSTNSA',start='1900-01-01',end='2021-05-01')
f = Forecaster(y=df['HOUSTNSA'],current_dates=df.index)
f.set_test_length(12)
f.generate_future_dates(24) # forecast length
f.set_estimator('hwes')
f.manual_forecast(trend='add',seasonal='mul')
```

### knn
- [Sklearn Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html)
- K-nearest Neighbors
- uses all Xvars and a MinMax normalizer by default
- better on differenced data for non-stationary series
```python
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
f.set_estimator('knn')
f.manual_forecast(n_neigbors=5,weights='uniform')
```
### lightgbm
- [LightGBM Documentation](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html)
- light gradient boosted tree model
- uses all Xvars and a MinMax normalizer by default
- better on differenced data for non-stationary series  
 
```python
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
f.diff() # non-stationary data forecasts better differenced with this model
f.set_estimator('xgboost')
f.manual_forecast(max_depth=3)
```

### lstm
- [TensorFlow Documentation](https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM)
- Long-Short Term Memory Neural Network
- uses all Xvars and a MinMax normalizer by default
- better on differenced data for non-stationary series
``` python
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
f.add_logged_terms('t') # lnt
f.diff() # non-stationary data forecasts better differenced with this model
f.set_estimator('lstm')
f.manual_forecast(Xvars=['monthsin','monthcos','year','lnt'],epochs=10)
```

### mlp
- [Sklearn Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html)
- Multi-Level Perceptron Neural Network
- uses all Xvars and a MinMax normalizer by default
- better on differenced data for non-stationary series
```python
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
f.set_estimator('mlp')
f.manual_forecast(Xvars=['monthsin','monthcos','year','t'],solver='lbfgs')
```
### mlr
- [Sklearn Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
- Multiple Linear Regression
- uses all Xvars and a MinMax normalizer by default
- better on differenced data for non-stationary series
```python
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
f.set_estimator('mlr')
f.manual_forecast(normalizer=None)
```
### prophet
- [Prophet Documentation](https://facebook.github.io/prophet/)
- uses no Xvars by default but does accept the Xvars argument
- does not accept the normalizer argument
- whether it performs better on differenced or level data depends on the series but it should be okay with either
```
pip install fbprophet
```
```python
import pandas as pd
import pandas_datareader as pdr
from scalecast.Forecaster import Forecaster

df = pdr.get_data_fred('HOUSTNSA',start='1900-01-01',end='2021-05-01')
f = Forecaster(y=df['HOUSTNSA'],current_dates=df.index)
f.set_test_length(12)
f.generate_future_dates(24) # forecast length
f.set_estimator('prophet')
f.manual_forecast(n_changepoints=3)
```

### rf
- [Sklearn Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
- Random Forest
- uses all Xvars and a MinMax normalizer by default
- better on differenced data for non-stationary series
- prone to overfitting
```python
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
f.diff() # non-stationary data forecasts better differenced with this model
f.set_estimator('rf')
f.manual_forecast(n_estimators=1000,max_depth=6)
```

### silverkite
- [GreyKite Documentation](https://linkedin.github.io/greykite/docs/0.1.0/html/index.html)
- uses no Xvars by default but does accept the Xvars argument
- does not accept the normalizer argument
- All `**kwargs` passed to ModelComponentsParam
  - default parameters with no Xvars should lead to good results most of the time as the library does a lot of under-the-hood optimization
- whether it performs better on differenced or level data depends on the series but it should be okay with either
```
pip install greykite
```
```python
import pandas as pd
import pandas_datareader as pdr
from scalecast.Forecaster import Forecaster

df = pdr.get_data_fred('HOUSTNSA',start='1900-01-01',end='2021-05-01')
f = Forecaster(y=df['HOUSTNSA'],current_dates=df.index)
f.set_test_length(12)
f.generate_future_dates(24) # forecast length
f.set_estimator('silverkite')
f.manual_forecast()
```
- when plotting after evaluating a silverkite forecast, you need to use an appropriate matplotlib aggregator
  - for Jupyter Notebooks:
    ```python
    matplotlib.use('nbAgg')
    %matplotlib inline    
    ```
  - for command line interface or Pycharm:
    ```python
    matplotlib.use('QT5Agg') 
    ```
  - add these lines after calling the silverkite forecast but before plotting
  - see other aggregators [here](https://matplotlib.org/stable/tutorials/introductory/usage.html)

### svr
- [Sklearn Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html)
- Support Vector Regressor
- uses all Xvars and a MinMax normalizer by default
- better on differenced data for non-stationary series
```python
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
f.diff() # non-stationary data forecasts better differenced with this model
f.set_estimator('svr')
f.manual_forecast(kernel='linear',gamma='scale',C=2,epsilon=0.01)
```

### xgboost
- [Xgboost Documentation](https://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.sklearn)
- extreme gradient boosted tree model
- uses all Xvars and a MinMax normalizer by default
- better on differenced data for non-stationary series
- runs faster than gbt
```python
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
f.diff() # non-stationary data forecasts better differenced with this model
f.set_estimator('xgboost')
f.manual_forecast(max_depth=2)
```
