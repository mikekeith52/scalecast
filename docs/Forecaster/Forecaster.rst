Forecaster
============================================

This is the main object that is utilized for making predictions on the test set, making forecasts, evaluating models, data differencing, adding regressors, and saving, visualizing, and exporting results.

.. code:: python

    from scalecast.Forecaster import Forecaster
    array_of_dates = ['2021-01-01','2021-01-02','2021-01-03']
    array_of_values = [1,2,3]
    f = Forecaster(
      y=array_of_values, 
      current_dates=array_of_dates,
      # defaults below
      require_future_dates=True, 
      future_dates=None,
      test_length = 0,
      cis = False,
      metrics = ['rmse','mape','mae','r2'],
   )

.. autoclass:: src.scalecast.Forecaster.Forecaster
   :members:
   :undoc-members:
   :inherited-members:

   .. automethod:: __init__