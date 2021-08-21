## export
- `Forecaster.export(dfs=['all_fcsts','model_summaries','best_fcst','test_set_predictions','lvl_fcsts'],
               models='all',
               best_model='auto',
               determine_best_by='TestSetRMSE',
               to_excel=False,
               out_path='./',
               excel_name='results.xlsx')`
- exports 1-all of 5 pandas dataframes, can write to excel with each dataframe on a separate sheet, will return either a dictionary with dataframes as values or a single dataframe if only one df is specified
  - **dfs**: list-like or `str`, default `['all_fcsts','model_summaries','best_fcst','test_set_predictions','lvl_fcsts']`
    - a list or name of the specific dataframe(s) you want returned and/or written to excel
    - must be one of default
  - **models**: list-like or `str`, default `'all'`
    - the models to write information for
    - can start with "top_" and the metric specified in `determine_best_by` will be used to order the models appropriately
  - **best_model**: `str`, default `'auto'`
    - the name of the best model, if "auto", will determine this by the metric in determine_best_by
    - if not "auto", must match a model nickname of an already-evaluated model
  - **determine_best_by**: one of `_determine_best_by_`, default `'TestSetRMSE'`
  - **to_excel**: `bool`, default `False`
    - whether to save to excel
  - **out_path**: `str`, default `'./'`
    - the path to save the excel file to (ignored when `to_excel=False`)
  - **excel_name**: `str`, default `'results.xlsx'`
    - the name to call the excel file (ignored when `to_excel=False`)
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

f.export(to_excel=True,excel_name='all_results.xlsx') # will write all five dataframes as separate sheets to excel in the local directory as "all_results.xlsx"
```
- see [examples/housing_results](https://github.com/mikekeith52/scalecast/blob/main/examples/housing_results.xlsx) for an example of all 5 dataframes
- see [examples/avocado_model_summaries.csv](https://github.com/mikekeith52/scalecast/blob/main/examples/avocado_model_summaries.csv) for an example of the `'model_summaries'` dataframe concatenated from the run of many series