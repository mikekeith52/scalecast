notebook
============================================

If you are using a Jupyter notebook, the notebook functions may come in handy to evaluate models and view results more easily.

results_vis()
-----------------------------
.. autofunction:: src.scalecast.notebook.results_vis

.. code:: python

    from scalecast.Forecaster import Forecaster
    from scalecast import GridGenerator
    from scalecast.notebook import tune_test_forecast, results_vis
    import pandas_datareader as pdr # !pip install pandas-datareader
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set(rc={"figure.figsize": (12, 8)})

    f_dict = {}
    models = ('mlr','elasticnet','mlp')
    GridGenerator.get_example_grids() # writes the Grids.py file to your working directory

    for sym in ('UNRATE','GDP'):
      df = pdr.get_data_fred(sym, start = '2000-01-01')
      f = Forecaster(y=df[sym],current_dates=df.index)
      f.generate_future_dates(12) # forecast 12 periods to the future
      f.set_test_length(12) # test models on 12 periods
      f.set_validation_length(4) # validate on the previous 4 periods
      f.add_time_trend()
      f.add_seasonal_regressors('quarter',raw=False,dummy=True)
      tune_test_forecast(f,models) # adds a progress bar that is nice for notebooks
      f_dict[sym] = f

    results_vis(f_dict) # toggle through results with jupyter widgets

results_vis_mv()
-----------------------------
.. autofunction:: src.scalecast.notebook.results_vis_mv

tune_test_forecast()
--------------------------------
.. autofunction:: src.scalecast.notebook.tune_test_forecast

.. code:: python

    from scalecast.notebook import tune_test_forecast
    models = ('arima','mlr','mlp')
    tune_test_forecast(f,models) # displays a progress bar through tqdm