util
============================================

Miscellaneous util functions native to scalecast.

metrics
----------
.. autoclass:: src.scalecast.util.metrics
   :members:

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

break_mv_forecaster()
-----------------------------
.. automodule:: src.scalecast.util.break_mv_forecaster
    :members:

.. code:: python

    from scalecast.MVForecaster import MVForecaster
    from scalecast.util import break_mv_forecaster, pdr_load

    f1 = prd_load('UTUR',start='2000-01-01',end='2022-01-01',src='fred')
    f2 = prd_load('UNRATE',start='2000-01-01',end='2022-01-01',src='fred')

    mvf = MVForecaster(f1,f2,names=['UTUR','UNRATE'])

    f1, f2 = break_mv_forecaster(mvf)


pdr_load()
----------------
.. automodule:: src.scalecast.util.pdr_load
    :members:

.. code:: python

    from scalecast.util import pdr_load
    f = prd_load('UNRATE',start='2000-01-01',src='fred')