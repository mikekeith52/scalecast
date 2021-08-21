## Xvars
- all estimators except hwes and combo accept an `Xvars` argument
- accepted arguments are an array-like of named regressors, a `str` of a single regressor name, `'all'`, or `None`
  - for estimators that require Xvars (sklearn models), `None` and `'all'` will both use all Xvars
- all regressors must be numeric type

[seasonal regressors](#seasonal-regressors)  
[ar terms](#ar-terms)  
[time trend](#time-trend)  
[combination regressors](#combination-regressors)  
[poly terms](#poly-terms)  
[covid19](#covid19)  
[ingesting a dataframe of x variables](#ingesting-a-dataframe-of-x-variables)  
[holidays/other](#other)  

### seasonal regressors
- `Forecaster.add_seasonal_regressors(*args,raw=True,sincos=False,dummy=False,drop_first=False)`
  - **args**: includes all `pandas.Series.dt` attributes ('month','day','dayofyear','week',etc.) that return `pandas.Series.astype(int)`
    - I'm not sure there exists anywhere a complete list of possible attributes, but a good place to start is [here](https://pandas.pydata.org/docs/reference/api/pandas.Series.dt.year.html)
    - only use attributes that return a series of int type
  - **raw**: `bool`, default `True`
    - by default, the output of calling this method results in Xvars added to current_xreg and future_xreg that are int (ordinal) type
    - setting raw to `False` will bypass that
    - at least one of raw, sincos, dummy must be true
  - **sincos**: `bool`, default `False`
    - this creates two wave transformations out of the pandas output and stores them in future_xreg, current_xreg
      - formula: `sin(pi*raw_output/(max(raw_output)/2))`, `cos(pi*raw_output/(max(raw_output)/2))`
    - it uses the max from the pandas output to automatically set the cycle length, so if you think this might cause problems in the analysis, using `dummy=True` is a safer bet to achieve a similar result, but it adds more total variables
  - **dummy**: `bool`, default `False`
    - changes the raw int output into dummy 0/1 variables and stores them in future_xreg, current_xreg
  - **drop_first**: `bool`, default `False`
    - whether to drop one class for dummy variables
    - ignored when `dummy=False`
```python
>>> import pandas as pd
>>> import pandas_datareader as pdr
>>> from scalecast.Forecaster import Forecaster

>>> df = pdr.get_data_fred('HOUSTNSA',start='1900-01-01',end='2021-05-01')
>>> f = Forecaster(y=df['HOUSTNSA'],current_dates=df.index)
>>> f.generate_future_dates(24) # forecast length
>>> f.add_seasonal_regressors('month',dummy=True,sincos=True)
>>> f.add_seasonal_regressors('year')
>>> print(f)
"Forecaster object with no models evaluated. Data starts at 1959-01-01 00:00:00, ends at 2021-05-01 00:00:00, loaded to forecast out 24 periods, has 16 regressors."
>>> print(f.get_regressor_names())
['month', 'monthsin', 'monthcos', 'month_1', 'month_10', 'month_11', 'month_12', 'month_2', 'month_3', 'month_4', 'month_5', 'month_6', 'month_7', 'month_8', 'month_9', 'year']
```

### ar terms
- `Forecaster.add_ar_terms(n)`
  - **n**: `int`
    - the number of ar terms to add, will add 1 to n ar terms

- `Forecast.add_AR_terms(N)`
  - **N**: `tuple([int,int])`
    - tuple shape: `(P,m)`
      - P is the number of terms to add 
      - m is the seasonal length (12 for monthly data, for example)
```python
>>> import pandas as pd
>>> import pandas_datareader as pdr
>>> from scalecast.Forecaster import Forecaster

>>> df = pdr.get_data_fred('HOUSTNSA',start='1900-01-01',end='2021-05-01')
>>> f = Forecaster(y=df['HOUSTNSA'],current_dates=df.index)
>>> f.generate_future_dates(24) # forecast length
>>> f.add_ar_terms(4)
>>> f.add_AR_terms((2,12)) # seasonal AR terms
>>> print(f)
"Forecaster object with no models evaluated. Data starts at 1959-01-01 00:00:00, ends at 2021-05-01 00:00:00, loaded to forecast out 24 periods, has 6 regressors."
>>> print(f.get_regressor_names())
['AR1', 'AR2', 'AR3', 'AR4', 'AR12', 'AR24']
```

- the beautiful part of adding auto-regressive terms in this framework is that all metrics and forecasts use an iterative process that plugs in forecasted values to future terms, making all test set and validation predictions and forecasts dynamic
- however, doing it this way means lots of loops in the evaluation process, which means some models run very slowly
- add ar/AR terms before differencing (don't worry, they will be differenced as well)
- don't begin any other regressor names you add with "AR" as it will confuse the forecasts

### time trend
- `Forecaster.add_time_trend(called='t')`
  - **called**: `str`, default `'t'`
    - what to call the resulting time trend
```python
>>> import pandas as pd
>>> import pandas_datareader as pdr
>>> from scalecast.Forecaster import Forecaster

>>> df = pdr.get_data_fred('HOUSTNSA',start='1900-01-01',end='2021-05-01')
>>> f = Forecaster(y=df['HOUSTNSA'],current_dates=df.index)
>>> f.generate_future_dates(24) # forecast length
>>> f.add_time_trend()
>>> print(f)
"Forecaster object with no models evaluated. Data starts at 1959-01-01 00:00:00, ends at 2021-05-01 00:00:00, loaded to forecast out 24 periods, has 1 regressors."
>>> print(f.get_regressor_names())
['t']
```
### combination regressors
- `Forecaster.add_combo_regressors(*args,sep='_')`
  - **args**: names of Xvars that aleady exist in the object
    - all vars in args will be multiplied together
  - **sep**: `str`, default `"_"`
    - the separator between each term in arg to create the final variable name
```python
>>> import pandas as pd
>>> import pandas_datareader as pdr
>>> from scalecast.Forecaster import Forecaster

>>> df = pdr.get_data_fred('HOUSTNSA',start='1900-01-01',end='2021-05-01')
>>> f = Forecaster(y=df['HOUSTNSA'],current_dates=df.index)
>>> f.generate_future_dates(24) # forecast length
>>> f.add_time_trend()
>>> f.add_covid19_regressor()
>>> f.add_combo_regressors('t','COVID19')
>>> print(f)
"Forecaster object with no models evaluated. Data starts at 1959-01-01 00:00:00, ends at 2021-05-01 00:00:00, loaded to forecast out 24 periods, has 3 regressors."
>>> print(f.get_regressor_names())
['t','COVID19','t_COVID19']
```
### poly terms
- `Forecaster.add_poly_terms(*args,pwr=2,sep='^')`
  - **args**: names of Xvars that aleady exist in the object
  - **pwr**: `int`, default `2`
    - the max power to add to each term in args (2 to this number will be added)
  - **sep**: `str`, default `"^"`
    - the separator between each term in arg and pwr to create the final vairable name
```python
>>> import pandas as pd
>>> import pandas_datareader as pdr
>>> from scalecast.Forecaster import Forecaster

>>> df = pdr.get_data_fred('HOUSTNSA',start='1900-01-01',end='2021-05-01')
>>> f = Forecaster(y=df['HOUSTNSA'],current_dates=df.index)
>>> f.generate_future_dates(24) # forecast length
>>> f.add_time_trend()
>>> f.add_poly_terms('t',pwr=3)
>>> print(f)
"Forecaster object with no models evaluated. Data starts at 1959-01-01 00:00:00, ends at 2021-05-01 00:00:00, loaded to forecast out 24 periods, has 3 regressors."
>>> print(f.get_regressor_names())
['t','t^2','t^3']
```
### covid19
- `Forecaster.add_covid19_regressor(called='COVID19',start=datetime.datetime(2020,3,15),end=datetime.datetime(2021,5,13))`
- adds dummy variable that is 1 during the time period that covid19 effects are present for the series, 0 otherwise
  - **called**: `str`, default `'COVID19'`
    - what to call the resulting variable
  - **start**: `str` or `datetime` object, default `datetime.datetime(2020,3,15)`
    - the start date (default is day Walt Disney World closed in the U.S.)
    - use format yyyy-mm-dd when passing strings
  - **end**: `str` or `datetime` object, default `datetime.datetime(2021,5,13)`
    - the end date (default is day the U.S. CDC dropped mask mandate/recommendation for vaccinated people)
    - use format yyyy-mm-dd when passing strings
```python
>>> import pandas as pd
>>> import pandas_datareader as pdr
>>> from scalecast.Forecaster import Forecaster

>>> df = pdr.get_data_fred('HOUSTNSA',start='1900-01-01',end='2021-05-01')
>>> f = Forecaster(y=df['HOUSTNSA'],current_dates=df.index)
>>> f.generate_future_dates(24) # forecast length
>>> f.add_covid19_regressor()
>>> print(f)
"Forecaster object with no models evaluated. Data starts at 1959-01-01 00:00:00, ends at 2021-05-01 00:00:00, loaded to forecast out 24 periods, has 1 regressors."
>>> print(f.get_regressor_names())
['COVID19']
```
### ingesting a dataframe of x variables
- `Forecaster.ingest_Xvars_df(df,date_col='Date',drop_first=False,use_future_dates=False)`
- reads all variables from a dataframe and stores them in `current_xreg` and `future_xreg`, can use future dates here instead of `generate_future_dates()`, will convert all non-numeric variables to dummies
  - **df**: pandas dataframe
    - must span the same time period as current_dates
    - if `use_future_dates = False`, must span at least the same time period as future_dates
  - **date_col**: `str`
    - the name of the date column in df
  - **drop_first**: `bool`, default `False`
    - whether to drop a class in any columns that have to be dummied
  - **use_future_dates**: `bool`, default `False`
    - whether to set the forecast periods in the object with the future dates in df
```python
>>> import pandas as pd
>>> import pandas_datareader as pdr
>>> from scalecast.Forecaster import Forecaster

>>> df = pdr.get_data_fred('HOUSTNSA',start='1970-01-01',end='2021-03-01')
>>> ur = pdr.get_data_fred('UNRATE',start='1970-01-01',end='2021-05-01').reset_index()
>>> f = Forecaster(y=df['HOUSTNSA'],current_dates=df.index)
>>> f.ingest_Xvars_df(ur,date_col='DATE',use_future_dates=True)
>>> print(f)
"Forecaster object with no models evaluated. Data starts at 1970-01-01 00:00:00, ends at 2021-03-01 00:00:00, loaded to forecast out 2 periods, has 1 regressors."
>>> print(f.get_regressor_names())
['UNRATE']
```
### other
- `Forecaster.add_other_regressor(called,start,end)`
- adds dummy variable that is 1 during the specified time period, 0 otherwise
  - **called**: `str`
    - what to call the resulting variable
  - **start**: `str` or `datetime` object
  - **end**: `str` or `datetime` object
```python
>>> import pandas as pd
>>> import pandas_datareader as pdr
>>> from scalecast.Forecaster import Forecaster

>>> df = pdr.get_data_fred('HOUSTNSA',start='1900-01-01',end='2021-05-01')
>>> f = Forecaster(y=df['HOUSTNSA'],current_dates=df.index)
>>> f.generate_future_dates(24) # forecast length
>>> f.add_other_regressor(called='Sept2001',start='2001-09-01',end='2001-09-01')
>>> print(f)
"Forecaster object with no models evaluated. Data starts at 1959-01-01 00:00:00, ends at 2021-05-01 00:00:00, loaded to forecast out 24 periods, has 1 regressors."
>>> print(f.get_regressor_names())
['Sept2001']
```