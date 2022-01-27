Getting Started
=================================

You will need to import the Forecaster object and initialize it with an array of actual observed values and an array of corresponding dates. It's generally a good idea to keep the dates sorted from least-to-most recent and ensure there are no missing observations. As long as these arrays are in some kind of collection (whether it be a list, numpy array, pandas series, etc.), the object should process them correctly.

.. code:: python
    
    from scalecast.Forecaster import Forecaster
    array_of_dates = ['2021-01-01','2021-01-02','2021-01-03']
    array_of_values = [1,2,3]
    f = Forecaster(y=array_of_values, current_dates=array_of_dates)

One of the best parts of using scalecast is the ability to automatically tune and test models dynamically. The easiest way to do that is to import a file of model grids and specify a collection of supported models:

.. code:: python
    
    from scalecast.Forecaster import Forecaster
    from scalecast import GridGenerator
    models = ('mlr','elasticnet','mlp') # many others available
    GridGenerator.get_example_grid() # writes the Grids.py file to your working directory
    array_of_dates = ['2021-01-01','2021-01-02','2021-01-03','2021-01-04','2021-01-05']
    array_of_values = [1,2,3,4,5]
    f = Forecaster(y=array_of_values, current_dates=array_of_dates)
    f.generate_future_dates(5) # forecast length of 5
    f.add_time_trend()
    f.tune_test_forecast(models)
    f.plot(ci=True) # see the results visually

If you are working on Jupyter notebook and are forecasting many series, you can try something like this.

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

These are simple procedures that barely scratch the surface of what scalecast can do! Happy reading!