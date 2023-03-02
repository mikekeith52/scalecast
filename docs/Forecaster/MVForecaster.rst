MVForecaster
=================================================

This object can be used to extend the univariate/exogenous regressor approach from the Forecaster class to make forecasts with multiple series that are all predicted forward dynamically using each other's lags, seasonality, and any other exogenous regressors. This object is initiated by combining several Forecaster objects together. This approach can utilize any sklearn regressor model to make forecasts. All models can be dynamically tuned and tested.

.. code:: python

    from scalecast.Forecaster import Forecaster
    from scalecast.MVForecaster import MVForecaster
    from scalecast.SeriesTransformer import SeriesTransformer
    import pandas_datareader as pdr # pip install pandas-datareader
    data = pd.read_csv('data.csv') # df with 3 cols - Date, Series1, Series2
    f1 = Forecaster(
      y = data['Series1'],
      current_dates = data['Date'],
      future_dates = 24,
    )
    f2 = Forecaster(
      y = data['Series2'],
      current_dates = data['Date'],
      future_dates = 24,
    )
    # before feeding to the MVForecaster object, you may want to add seasonal and other regressors
    # you can add to one Forecaster object and in the MVForecaster object, it will be added to forecast both series
    
    # initiate the MVForecaster object
    mvf = MVForecaster(
      f1,
      f2,
      # add more Forecaster objects here
      # defaults below
      not_same_len_action='trim',
      merge_Xvars='union',
      merge_future_dates='longest',
      test_length = 0,
      cis = False,
      metrics = ['rmse','mape','mae','r2'],
      # specify names if you want them
      names=['My First Series', 'My Second Series'],
    ) 

.. autoclass:: src.scalecast.MVForecaster.MVForecaster
   :members:
   :undoc-members:
   :inherited-members:
   
   .. automethod:: __init__