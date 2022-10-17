util
============================================

Miscellaneous util functions native to scalecast.

metrics
----------
.. autoclass:: src.scalecast.util.metrics
   :members:

break_mv_forecaster()
-----------------------------
.. automodule:: src.scalecast.util.break_mv_forecaster
    :members:

.. code:: python

    from scalecast.MVForecaster import MVForecaster
    from scalecast.util import break_mv_forecaster, pdr_load

    mvf = pdr_load(
        ['UTUR','UNRATE'],
        start='2000-01-01',
        end='2022-01-01',
        future_dates=12,
    )

    f1, f2 = break_mv_forecaster(mvf)

find_optimal_coint_rank()
----------------------------
.. automodule:: src.scalecast.util.find_optimal_coint_rank
    :members:

.. code:: python

  from scalecast.Forecaster import Forecaster
  from scalecast.MVForecaster import MVForecaster
  from scalecast.util import find_optimal_coint_rank
  import pandas_datareader as pdr

  s1 = pdr.get_data_fred('UTUR',start='2000-01-01',end='2022-01-01')
  s2 = pdr.get_data_fred('UNRATE',start='2000-01-01',end='2022-01-01')

  f1 = Forecaster(y=s1['UTUR'],current_dates=s1.index)
  f2 = Forecaster(y=s2['UNRATE'],current_dates=s2.index)

  mvf = MVForecaster(f1,f2,names=['UTUR','UNRATE'])

  coint_res = find_optimal_coint_rank(mvf,det_order=-1,k_ar_diff=8,train_only=True)
  print(coint_res) # prints a report
  rank = coint_res.rank # best rank

find_optimal_lag_order()
---------------------------
.. automodule:: src.scalecast.util.find_optimal_lag_order
    :members:

.. code:: python

  from scalecast.Forecaster import Forecaster
  from scalecast.MVForecaster import MVForecaster
  from scalecast.util import find_optimal_lag_order
  import pandas_datareader as pdr

  s1 = pdr.get_data_fred('UTUR',start='2000-01-01',end='2022-01-01')
  s2 = pdr.get_data_fred('UNRATE',start='2000-01-01',end='2022-01-01')

  f1 = Forecaster(y=s1['UTUR'],current_dates=s1.index)
  f2 = Forecaster(y=s2['UNRATE'],current_dates=s2.index)

  f1.diff()
  f2.diff()

  mvf = MVForecaster(f1,f2,names=['UTUR','UNRATE'])

  lag_order_res = find_optimal_lag_order(mvf,train_only=True)
  lag_order_aic = lag_order_res.aic # picks the best lag order according to aic

find_series_transformation()
-----------------------------------
.. automodule:: src.scalecast.util.find_series_transformation
    :members:

.. code:: python

  from scalecast.Forecaster import Forecaster
  from scaleast.Pipeline import Pipeline, Transformer, Reverter
  from scalecast.util import find_series_transformation

  t, r = find_series_transformation(
      f,
      goal=['stationary','seasonally_adj'],
      train_only=True,
      critical_pval = .01,
  )

  def forecaster(f):
      f.add_covid19_regressor()
      f.auto_Xvar_select(cross_validate=True)
      f.set_estimator('mlr')
      f.manual_forecast()
  df = pdr.get_data_fred(
      'HOUSTNSA',
      start='1959-01-01',
      end='2022-08-01'
  )
  f = Forecaster(
      y=df['HOUSTNSA'],
      current_dates=df.index,
      future_dates=24,
  )
  f.set_test_length(0.2)
  f.set_validation_length(24)
  transformer, reverter = find_series_transformation(
      f,
      goal=['stationary','seasonally_adj'],
      train_only=True,
      critical_pval = .01,
  )
  pipeline = Pipeline(
      steps = [
          ('Transform',transformer),
          ('Forecast',forecaster),
          ('Revert',reverter),
      ],
  )
  f = pipeline.fit_predict(f)

pdr_load()
----------------
.. automodule:: src.scalecast.util.pdr_load
    :members:

.. code:: python

    from scalecast.util import pdr_load
    f = pdr_load('UNRATE',start='2000-01-01',src='fred')
    mvf = pdr_load(['UNRATE','UTUR'],start='2000-01-01',src='fred')

plot_reduction_errors()
-----------------------------
.. automodule:: src.scalecast.util.plot_reduction_errors
    :members:

.. code:: python

    from scalecast.Forecaster import Forecaster
    from scalecast.util import plot_reduction_errors
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import pandas_datareader as pdr

    df = pdr.get_data_fred('HOUSTNSA',start='1900-01-01',end='2021-06-01')
    f = Forecaster(y=df['HOUSTNSA'],current_dates=df.index)

    f.set_test_length(.2)
    f.generate_future_dates(24)

    f.add_ar_terms(24)
    f.integrate(critical_pval=.01)
    f.add_seasonal_regressors('month',raw=False,sincos=True,dummy=True)
    f.add_seasonal_regressors('year')
    f.add_time_trend()

    f.reduce_Xvars(method='pfi')
    
    plot_reduction_errors(f)
