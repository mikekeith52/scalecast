MVForecaster
=================================================

This object can be used to extend the univariate/exogenous regressor approach from the Forecaster class to make forecasts with multiple series that are all predicted forward dynamically using each other's lags, seasonality, and any other exogenous regressors. This object is initiated by combining several Forecaster objects together. This approach can utilize any sklearn regressor model to make forecasts. All models can be dynamically tuned and tested.

.. code:: python

    from scalecast.Forecaster import Forecaster
    from scalecast.MVForecaster import MVForecaster
    from scalecast.SeriesTransformer import SeriesTransformer
    import pandas_datareader as pdr # pip install pandas-datareader
    for s in ('UNRATE','UTUR'):
      df = pdr.get_data_fred(s,start='2000-01-01',end='2022-01-01') # fetch data
      f = Forecaster(y=df[s],current_dates=df.index) # load it into a Forecaster object
      f.generate_future_dates(24) # create the forecast horizon
      f.auto_Xvar_select()
      f = SeriesTransformer(f).DiffTransform(1) # difference to make stationary
      f_dict[s] = f # store everything in a dictionary
    
    # initiate the MVForecaster object
    mvf = MVForecaster(
      f_dict['UNRATE'], # series 1
      f_dict['UTUR'], # sereis 2
      # add more series here
      # defaults below
      not_same_len_action='trim',
      merge_Xvars='union',
      merge_future_dates='longest',
      test_length = 0,
      cis = False,
      # specify names if you want them
      names=('UNRATE','UTUR'),
    ) 

.. autoclass:: src.scalecast.MVForecaster.MVForecaster
   :members:
   
   .. automethod:: __init__