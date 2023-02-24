Forecasting Different Model Types
===================================
Any time you set an estimator, different arguments become available to you when calling `manual_forecast` or tuning the model. This page lists all model types native to scalecast. See also the `auxmodels module <https://scalecast.readthedocs.io/en/latest/Forecaster/Auxmodels.html>`_.

arima
--------------------------------------------------
See also `auto_arima <https://scalecast.readthedocs.io/en/latest/Forecaster/Auxmodels.html>`_.

.. autofunction:: src.scalecast.Forecaster.Forecaster._forecast_arima

>>> f.set_estimator('arima')
>>> f.manual_forecast() # above args are now available in this function

combo
--------------------------------------------------
.. autofunction:: src.scalecast.Forecaster.Forecaster._forecast_combo

>>> f.set_estimator('combo')
>>> f.manual_forecast() # above args are now available in this function

hwes
--------------------------------------------------
.. autofunction:: src.scalecast.Forecaster.Forecaster._forecast_hwes

>>> f.set_estimator('hwes')
>>> f.manual_forecast() # above args are now available in this function

lstm
--------------------------------------------------
.. autofunction:: src.scalecast.Forecaster.Forecaster._forecast_lstm

>>> f.set_estimator('lstm')
>>> f.manual_forecast() # above args are now available in this function
>>> f.tf_model.summary() # view a summary of the model's parameters

multivariate
---------------------------------------------------------------
See also the `vecm model <https://scalecast.readthedocs.io/en/latest/Forecaster/Auxmodels.html#vecm>`_.

.. autofunction:: src.scalecast.MVForecaster.MVForecaster._forecast

>>> mvf.set_estimator('xgboost')
>>> mvf.manual_forecast()

naive
--------------
.. autofunction:: src.scalecast.Forecaster.Forecaster._forecast_naive

>>> f.set_estimator('naive')
>>> f.manual_forecast()
>>> f.manual_forecast(seasonal=True)

prophet
--------------------------------------------------
.. autofunction:: src.scalecast.Forecaster.Forecaster._forecast_prophet

>>> f.set_estimator('prophet')
>>> f.manual_forecast() # above args are now available in this function

rnn
--------------------------------------------------
.. autofunction:: src.scalecast.Forecaster.Forecaster._forecast_rnn

>>> f.set_estimator('rnn')
>>> f.manual_forecast() # above args are now available in this function
>>> f.tf_model.summary() # view a summary of the model's parameters

silverkite
--------------------------------------------------
.. autofunction:: src.scalecast.Forecaster.Forecaster._forecast_silverkite

>>> f.set_estimator('silverkite')
>>> f.manual_forecast() # above args are now available in this function


sklearn
--------------------------------------------------
See also `mlp_stack <https://scalecast.readthedocs.io/en/latest/Forecaster/Auxmodels.html#module-src.scalecast.auxmodels.mlp_stack>`_.

.. autofunction:: src.scalecast.Forecaster.Forecaster._forecast_sklearn

>>> f.set_estimator('mlp')
>>> f.manual_forecast()
>>> f.regr # access the sklearn model properties

theta
-------------------------------------------------
.. autofunction:: src.scalecast.Forecaster.Forecaster._forecast_theta

>>> f.set_estimator('theta')
>>> f.manual_forecast() # above args are now available in this function