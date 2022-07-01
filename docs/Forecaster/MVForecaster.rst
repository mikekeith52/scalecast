MVForecaster
=================================================

This object can be used to extend the univariate/exogenous regressor approach from the Forecaster class to make forecasts with multiple series that are all predicted forward dynamically using each other's lags, seasonality, and any other exogenous regressors. This object is initiated by combining several Forecaster objects together. This approach can utilize any sklearn regressor model to make forecasts. All models can be dynamically tuned and tested and level and differenced results are available.

.. code:: python

    from Forecaster import Forecaster
    from MVForecaster import MVForecaster
    import pandas_datareader as pdr # pip install pandas-datareader
    for s in ('UNRATE','UTUR'):
      df = pdr.get_data_fred(s,start='2000-01-01',end='2022-01-01') # fetch data
      f = Forecaster(y=df[s],current_dates=df.index) # load it into a Forecaster object
      f.generate_future_dates(24) # create the forecast horizon
      f.add_seasonal_regressors('month','quarter','daysinmonth') # add regressors
      f.integrate() # take differences to make data stationary
      f_dict[s] = f # store everything in a dictionary
    
    # initiate the MVForecaster object
    mvf = MVForecaster(f_dict['UNRATE'],
      f_dict['UTUR'],
      not_same_len_action='trim',
      merge_Xvars='union',
      merge_future_dates='longest',
      names=f_dict.keys()) 

A description of the options available when initializing the object:

MVForecaster
--------------------
.. automodule:: src.scalecast.MVForecaster.MVForecaster.__init__
    :members:
    :undoc-members:
    :show-inheritance:

A description of the class and associated methods:

.. autoclass:: src.scalecast.MVForecaster.MVForecaster
   :members:
   :undoc-members:
   :show-inheritance: