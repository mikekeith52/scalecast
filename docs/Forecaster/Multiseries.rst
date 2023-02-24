multiseries
============================================

If forecasting with many series in a loop, these functions and code examples may facilitate setting up the process and getting key information for each model and series.

export_model_summaries()
--------------------------
.. autofunction:: src.scalecast.multiseries.export_model_summaries

.. code:: python

    from scalecast.Forecaster import Forecaster
    from scalecast import GridGenerator
    from scalecast.multiseries import export_model_summaries
    import pandas_datareader as pdr

    f_dict = {}
    models = ('mlr','elasticnet','mlp')
    GridGenerator.get_example_grids() # writes the Grids.py file to your working directory

    for sym in ('UNRATE','GDP'):
      df = pdr.get_data_fred(sym, start = '2000-01-01')
      f = Forecaster(
        y=df[sym],
        current_dates=df.index,
        future_dates = 12,
        test_length = .1,
        validation_length = 12,
      )
      f.add_ar_terms(12)
      f.add_time_trend()
      f.tune_test_forecast(models)
      f_dict[sym] = f

    model_summaries = export_model_summaries(f_dict,determine_best_by='LevelTestSetMAPE')

keep_smallest_first_date()
------------------------------
.. autofunction:: src.scalecast.multiseries.keep_smallest_first_date