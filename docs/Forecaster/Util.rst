util
============================================

Miscellaneous util functions native to scalecast.

plot_reduction_errors
-----------------------------
.. automodule:: src.scalecast.util.plot_reduction_errors
    :members:
    :undoc-members:
    :show-inheritance:

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

break_mv_forecaster
-----------------------------
.. automodule:: src.scalecast.util.break_mv_forecaster
    :members:
    :undoc-members:
    :show-inheritance:
