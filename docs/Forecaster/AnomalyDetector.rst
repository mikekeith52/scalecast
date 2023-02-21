AnomalyDetector
=================================================

This object can be used to detect anomalies in a time series using any of three methods. See the example notebook: https://scalecast-examples.readthedocs.io/en/latest/misc/anomalies/anomalies.html

.. code:: python

    import pandas as pd
    import pandas_datareader as pdr
    import matplotlib.pyplot as plt
    from scalecast.Forecaster import Forecaster
    from scalecast.AnomalyDetector import AnomalyDetector

    df = pdr.get_data_fred('HOUSTNSA',start='1900-01-01',end='2021-06-01')
    f = Forecaster(y=df['HOUSTNSA'],current_dates=df.index)
    f.set_test_length(12)

    detector = AnomalyDetector(f)

    detector.NaiveDetect(extrapolate_trend='freq',cilevel=.99,train_only=True)

    detector.MonteCarloDetect('2010-01-01','2020-12-01',cilevel=.99)

    detector.EstimatorDetect(
        estimator='lstm',
        cilevel=.99,
        test_only=False,
        lags=24,
        epochs=25,
        validation_split=.2,
        shuffle=True,
        lstm_layer_sizes=(16,16,16),
        dropout=(0,0,0),
    )

.. autoclass:: src.scalecast.AnomalyDetector.AnomalyDetector
   :members: