SeriesTransformer
=================================================

This object can be used to perform more complex transformations on your `Forecaster` object. It can be used to transform the dependent variable to adjust for trends, seasonality, and more, and every transformation is revertible. Revert functions must be called in opposite order as the applied transformation functions.

.. code:: python

    import pandas as pd
    import pandas_datareader as pdr
    import matplotlib.pyplot as plt
    from scalecast.Forecaster import Forecaster
    from scalecast.SeriesTransformer import SeriesTransformer
    from scalecast import GridGenerator

    GridGenerator.get_example_grids()

    df = pdr.get_data_fred('HOUSTNSA',start='1900-01-01',end='2021-06-01')
    f = Forecaster(y=df['HOUSTNSA'],current_dates=df.index) # to initialize, specify y and current_dates (must be arrays of the same length)

    transformer = SeriesTransformer(f)

    f = transformer.LogTransform()
    f = transformer.DiffTransform(1)
    f = transformer.DiffTransform(12)
    f = transformer.ScaleTransform()

    f.generate_future_dates(12)
    f.set_test_length(12)
    f.add_time_trend()
    f.add_ar_terms(24)

    f.set_estimator('elasticnet')
    f.cross_validate(rolling=True)
    f.auto_forecast()

    # call in opposite order
    f = transformer.ScaleRevert()
    f = transformer.DiffRevert(12)
    f = transformer.DiffRevert(1)
    f = transformer.LogRevert()

    f.plot()

.. autoclass:: src.scalecast.SeriesTransformer.SeriesTransformer
   :members:

   .. automethod:: __init__