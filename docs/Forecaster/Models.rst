Forecasting Different Model Types
===================================
Use the following syntax to initiate and forecast with different model types:

.. code-block:: python

   from scalecast.Forecaster import Forecaster
   from scalecast.models import ARIMA

   f = Forecaster(y=df['y'], current_dates=df['dt'], future_dates=12)
   f.init_estimator('arima',order=(1,1,1), seasonal_order=(1,0,0,12))
   f.fit()
   preds = f.predict()


See also `auxmodels <https://scalecast.readthedocs.io/en/latest/Forecaster/Auxmodels.html>`_.

- :py:class:`scalecast.models.ARIMA`
- :py:class:`scalecast.models.Combo`
- :py:class:`scalecast.models.HWES`
- :py:class:`scalecast.models.LSTM`
- :py:class:`scalecast.models.Naive`
- :py:class:`scalecast.models.Prophet`
- :py:class:`scalecast.models.SKLearnMV`
- :py:class:`scalecast.models.SKLearnUni`
- :py:class:`scalecast.models.RNN`
- :py:class:`scalecast.models.TBATS`
- :py:class:`scalecast.models.Theta`

.. autoclass:: scalecast.models.ARIMA
   :members:
   :undoc-members:
   :inherited-members:

.. autoclass:: scalecast.models.Combo
   :members:
   :undoc-members:
   :inherited-members:

.. autoclass:: scalecast.models.HWES
   :members:
   :undoc-members:
   :inherited-members:

.. autoclass:: scalecast.models.LSTM
   :members:
   :undoc-members:
   :inherited-members:

.. autoclass:: scalecast.models.Naive
   :members:
   :undoc-members:
   :inherited-members:

.. autoclass:: scalecast.models.Prophet
   :members:
   :undoc-members:
   :inherited-members:

.. autoclass:: scalecast.models.SKLearnMV
   :members:
   :undoc-members:
   :inherited-members:

.. autoclass:: scalecast.models.SKLearnUni
   :members:
   :undoc-members:
   :inherited-members:

.. autoclass:: scalecast.models.RNN
   :members:
   :undoc-members:
   :inherited-members:

.. autoclass:: scalecast.models.TBATS
   :members:
   :undoc-members:
   :inherited-members:

.. autoclass:: scalecast.models.Theta
   :members:
   :undoc-members:
   :inherited-members:
