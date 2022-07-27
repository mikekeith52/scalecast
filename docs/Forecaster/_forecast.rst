Forecasting Different Model Types
===================================
Any time you set an estimator, different arguments become available to you when calling `manual_forecast`, `proba_forecast`, or tuning the model. Here is the documentation for all these models:

arima
--------------------------------------------------
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

multivariate
---------------------------------------------------------------
.. automodule:: src.scalecast.MVForecaster.MVForecaster._forecast
    :members:
    :undoc-members:
    :show-inheritance:

>>> mvf.set_estimator('xgboost')
>>> mvf.proba_forecast() # probabilistic forecasting

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
.. automodule:: src.scalecast.Forecaster.Forecaster._forecast_sklearn
    :members:
    :undoc-members:
    :show-inheritance:

>>> f.set_estimator('mlp')
>>> f.proba_forecast() # probabilistic forecasting

theta
-------------------------------------------------
.. automodule:: src.scalecast.Forecaster.Forecaster._forecast_theta
    :members:
    :undoc-members:
    :show-inheritance:

>>> f.set_estimator('theta')
>>> f.manual_forecast() # above args are now available in this function