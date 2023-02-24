auxmodels
============================================

Supplemental models that build on scalecast's functionality.

vecm
----------

This is a vector error correction model adapted from `statsmodels <https://www.statsmodels.org/dev/generated/statsmodels.tsa.vector_ar.vecm.VECM.html>`_. Since it has a similar sklearn API, it can be imported into a multivariate forecasting application using `MVForecaster.add_sklearn_estimator()`.

This framework also offers a basis for adding other non-scikit-learn forecast models to the scalecast interface. The lags argument must always be 0 or None in the `manual_forecast()` function, but lags for the model can be specified through the `k_ar_diff` argument in the vecm model.

.. autoclass:: src.scalecast.auxmodels.vecm

   .. automethod:: __init__

.. code:: python

  from scalecast.Forecaster import Forecaster
  from scalecast.MVForecaster import MVForecaster
  from scalecast.auxmodels import vecm
  import pandas_datareader as pdr
  import matplotlib.pyplot as plt

  df = pdr.get_data_fred(
    [
      'APU000074714', # (monthly) retail gas prices
      'WTISPLC',      # (monthly) crude oil prices
    ],
    start = '1975-01-01',
    end = '2022-08-01',
  )
  
  rgp = Forecaster(
    y = df['APU000074714'],
    current_dates = df.index,
    future_dates = 12,
  )
  cop = Forecaster(
    y = df['WTISPLC'],
    current_dates = df.index,
    future_dates = 12,
  )

  mvf = MVForecaster(rgp,cop,names=['retail gas prices','crude oil prices'])
  mvf.set_test_length(12)

  mvf.add_sklearn_estimator(vecm,called='vecm')

  vecm_grid = {
    'lags':[0],  # lags will be specified from statsmodels function, so this needs to be None or 0
    'normalizer':[None], # data will not be scaled -- use SeriesTransformer for scaling if desired
    'k_ar_diff':range(1,13), # try 1-12 lags
    'deterministic':["n","co","lo","li","cili","colo"], # deterministic part
    'seasons':[0,12], # seasonal part
  }

  mvf.set_estimator('vecm')
  mvf.ingest_grid(vecm_grid)
  mvf.cross_validate()
  mvf.auto_forecast()

  # access results
  mvf.export('lvl_fcsts')
  mvf.export('model_summaries')

  # plot
  mvf.plot()
  plt.show()

auto_arima()
-----------------------------
.. autofunction:: src.scalecast.auxmodels.auto_arima

mlp_stack()
-----------------------------
.. autofunction:: src.scalecast.auxmodels.mlp_stack