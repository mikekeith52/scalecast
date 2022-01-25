Notebook
============================================

If you are using a Jupyter notebook, the notebook functions may come in handy to evaluate models and view results more easily.

notebook.results_vis
-----------------------------
.. automodule:: src.scalecast.notebook.results_vis
    :members:
    :undoc-members:
    :show-inheritance:

>>> from scalecast.Forecaster import Forecaster
>>> from scalecast.notebook import results_vis
>>> import pandas_datareader as pdr # pip install pandas-datareader
>>> f_dict = {}
>>> models = ('arima','mlr','mlp')
>>> for sym in ('UNRATE','GDP'):
>>>   df = pdr.get_data_fred(sym)
>>>   f = Forecaster(y=df[sym],current_dates=df.index)
>>>   f.tune_test_forecast(models)
>>>   f_dict[sym] = f
>>> results_vis(f)

notebook.tune_test_forecast
--------------------------------
.. automodule:: src.scalecast.notebook.tune_test_forecast
    :members:
    :undoc-members:
    :show-inheritance:

>>> from scalecast.notebook import tune_test_forecast
>>> models = ('arima','mlr','mlp')
>>> tune_test_forecast(f,models) # displays a progress bar through tqdm