Overview
=================================

Introduction
---------------
This package uses a scaleable forecasting approach in Python with scikit-learn, statsmodels, Facebook Prophet, Microsoft LightGBM, LinkedIn Silverkite, and Keras models to forecast time series. Use your own regressors or load the object with its own seasonal, auto-regressive, and other regressors, or combine all of the above. All forecasting is dynamic by default so that auto-regressive terms can be used without leaking data into the test set, setting it apart from other time-series libraries. Dynamic model testing can be disabled to improve model evaluation speed. Differencing to achieve stationarity is built into the library and metrics can be compared across the time series' original level or first or second difference. Bootstrapped confidence intervals consistently applied across all models for comparable results. This library was written to easily apply and compare many forecasts fairly across the same series.

>>> import pandas as pd
>>> import pandas_datareader as pdr
>>> from scalecast import GridGenerator
>>> from scalecast.Forecaster import Forecaster

>>> models = ('mlr','knn','svr','xgboost','elasticnet','mlp','prophet')
>>> df = pdr.get_data_fred('HOUSTNSA',start='2009-01-01',end='2021-06-01')
>>> GridGenerator.get_example_grids() # saves Grids.py with validation grids for each model that can be used to tune the forecasts
>>> f = Forecaster(y=df.HOUSTNSA,current_dates=df.index) # to initialize, specify y and current_dates (must be arrays of the same length)
>>> f.set_test_length(12) # specify a test length for your models - do this before eda
>>> f.generate_future_dates(24) # this will create future dates that are on the same interval as the current dates and it will also set the forecast length
>>> f.add_ar_terms(4) # add AR terms before differencing
>>> f.add_AR_terms((2,12)) # seasonal AR terms
>>> f.integrate() # automatically decides if the y term and all ar terms should be differenced to make the series stationary
>>> f.add_seasonal_regressors('month',raw=False,sincos=True) # uses pandas attributes: raw=True creates integers (default), sincos=True creates wave functions
>>> f.add_seasonal_regressors('year')
>>> f.add_covid19_regressor() # dates are flexible, default is from when disney world closed to when US CDC lifted mask recommendations
>>> f.add_time_trend()
>>> f.set_validation_length(6) # length, different than test_length, to tune the hyperparameters 
>>> f.tune_test_forecast(models)
>>> f.plot_test_set(models='top_3',order_by='LevelTestSetMAPE',ci=True) # plots the differenced test set with confidence intervals
>>> f.plot(order_by='LevelTestSetMAPE',level=True) # plots the level forecast

.. image:: https://github.com/mikekeith52/scalecast/blob/main/assets/main_forecast.png
   :target: https://github.com/mikekeith52/scalecast/blob/main/assets/main_forecast.png
   :alt: Example of a test set plot.

.. image:: https://github.com/mikekeith52/scalecast/blob/main/assets/main_forecast_test_set.png
   :target: https://github.com/mikekeith52/scalecast/blob/main/assets/main_forecast_test_set.png
   :alt: Example of a forecast plot.

Installation
------------------
>>> pip install scalecast 
>>> pip install fbprophet # for prophet models only
>>> pip install greykite # for silverkite models only
>>> pip install tqdm # for notebooks only
>>> pip install ipython  # for notebooks only
>>> pip install ipywidgets  # for notebooks only
>>> jupyter nbextension enable --py widgetsnbextension  # for notebooks only
>>> jupyter labextension install @jupyter-widgets/jupyterlab-manager # jupyter lab only