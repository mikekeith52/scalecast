Forecasting Different Model Types
===================================
Any time you set an estimator, different arguments become available to you when calling `manual_forecast`, `proba_forecast`, or tuning the model. This page lists all model types native to scalecast. See also the `auxmodels <https://scalecast.readthedocs.io/en/latest/Forecaster/Auxmodels.html>`_ module.

arima
--------------------------------------------------
See also `auto_arima <https://scalecast.readthedocs.io/en/latest/Forecaster/Auxmodels.html>`_.

.. automodule:: src.scalecast.Forecaster.Forecaster._forecast_arima
    :members:
    :undoc-members:
    :show-inheritance:

>>> f.set_estimator('arima')
>>> f.manual_forecast() # above args are now available in this function

combo
--------------------------------------------------
.. automodule:: src.scalecast.Forecaster.Forecaster._forecast_combo
    :members:
    :undoc-members:
    :show-inheritance:

>>> f.set_estimator('combo')
>>> f.manual_forecast() # above args are now available in this function

hwes
--------------------------------------------------
.. automodule:: src.scalecast.Forecaster.Forecaster._forecast_hwes
    :members:
    :undoc-members:
    :show-inheritance:

>>> f.set_estimator('hwes')
>>> f.manual_forecast() # above args are now available in this function

lstm
--------------------------------------------------
.. automodule:: src.scalecast.Forecaster.Forecaster._forecast_lstm
    :members:
    :undoc-members:
    :show-inheritance:

>>> f.set_estimator('lstm')
>>> f.manual_forecast() # above args are now available in this function
>>> f.tf_model.summary() # view a summary of the model's parameters

multivariate
---------------------------------------------------------------
See also `vecm <https://scalecast.readthedocs.io/en/latest/Forecaster/Auxmodels.html#vecm>`_.

.. automodule:: src.scalecast.MVForecaster.MVForecaster._forecast
    :members:
    :undoc-members:
    :show-inheritance:

>>> mvf.set_estimator('xgboost')
>>> mvf.proba_forecast() # probabilistic forecasting

naive
--------------
.. automodule:: src.scalecast.Forecaster.Forecaster._forecast_naive
    :members:
    :undoc-members:
    :show-inheritance:

>>> f.set_estimator('naive')
>>> f.manual_forecast()
>>> f.manual_forecast(seasonal=True)

prophet
--------------------------------------------------
.. automodule:: src.scalecast.Forecaster.Forecaster._forecast_prophet
    :members:
    :undoc-members:
    :show-inheritance:

>>> f.set_estimator('prophet')
>>> f.manual_forecast() # above args are now available in this function

rnn
--------------------------------------------------
.. automodule:: src.scalecast.Forecaster.Forecaster._forecast_rnn
    :members:
    :undoc-members:
    :show-inheritance:

>>> f.set_estimator('rnn')
>>> f.manual_forecast() # above args are now available in this function
>>> f.tf_model.summary() # view a summary of the model's parameters

silverkite
--------------------------------------------------
.. automodule:: src.scalecast.Forecaster.Forecaster._forecast_silverkite
    :members:
    :undoc-members:
    :show-inheritance:

>>> f.set_estimator('silverkite')
>>> f.manual_forecast() # above args are now available in this function


sklearn
--------------------------------------------------
See also `mlp_stack <https://scalecast.readthedocs.io/en/latest/Forecaster/Auxmodels.html#module-src.scalecast.auxmodels.mlp_stack>`_.

.. automodule:: src.scalecast.Forecaster.Forecaster._forecast_sklearn
    :members:
    :undoc-members:
    :show-inheritance:

>>> f.set_estimator('mlp')
>>> f.proba_forecast() # probabilistic forecasting
>>> f.regr # access the sklearn model properties

theta
-------------------------------------------------
.. automodule:: src.scalecast.Forecaster.Forecaster._forecast_theta
    :members:
    :undoc-members:
    :show-inheritance:

>>> f.set_estimator('theta')
>>> f.manual_forecast() # above args are now available in this function