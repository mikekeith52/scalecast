About
========
Scalecast is a light-weight modeling procedure, wrapper, and results container meant for those who are looking for the fastest way possible to apply, tune, and validate many different model classes for forecasting applications. In the Data Science industry, it is often asked of practitioners to deliver predictions and ranges of predictions for several lines of businesses or data slices, 100s or even 1000s. In such situations, it is common to see a simple linear regression or some other quick procedure applied to all lines due to the complexity of the task. This works well enough for people who need to deliver something, but more can be achieved.  

The scalecast package was designed to address this situation and offer advanced machine learning models that can be applied, optimized, and validated quickly. Unlike many libraries, the predictions produced by scalecast are always dynamic by default, not averages of one-step forecasts, so you don't run into the situation where the estimator looks great on the test-set but can't generalize to real data. What you see is what you get, with no attempt to oversell results. If you download a library that looks like it's able to predict the COVID pandemic in your test-set, you probably have a one-step forecast happening under-the-hood. You can't predict the unpredictable, and you won't see such things with scalecast.  

The library provides the Forecaster (for one series) and MVForecaster (for multiple series) wrappers around the following estimators: 

* Any regression model from `Sklearn <https://scikit-learn.org/stable/>`_, including Sklearn APIs (like `Xgboost <https://xgboost.readthedocs.io/en/stable/>`_ and `LightGBM <https://lightgbm.readthedocs.io/en/latest/>`_)

* Recurrent neural nets from `Keras TensorFlow <https://keras.io/>`_

* Classic econometric models from `statsmodels <https://www.statsmodels.org/stable/>`_: Holt-Winters Exponential Smoothing and ARIMA

* - The `Four Theta model <https://unit8co.github.io/darts/generated_api/darts.models.forecasting.theta.html>`_ from `darts <https://unit8co.github.io/darts/>`_

* `Facebook Prophet <https://facebook.github.io/prophet/>`_

* `LinkedIn Silverkite <https://engineering.linkedin.com/blog/2021/greykite--a-flexible--intuitive--and-fast-forecasting-library>`_

* Average, weighted average, and spliced models

+------------------+--------------+-----------------+-----------------+--------+-------------+------------+--------+-----------+
| Scalecast Model  | ARComponents | ExogComponents  | SeasComponents  | Trends | AutoHoliday | MultSeries | Linear | NonLinear |
+------------------+--------------+-----------------+-----------------+--------+-------------+------------+--------+-----------+
| Any Sklearn Regr | Y            | Y               | Y               | Y      | N           | Y          | Y      | Y         |
+------------------+--------------+-----------------+-----------------+--------+-------------+------------+--------+-----------+
| XGBoost          | Y            | Y               | Y               | Y      | N           | Y          | N      | Y         |
+------------------+--------------+-----------------+-----------------+--------+-------------+------------+--------+-----------+
| LightGBM         | Y            | Y               | Y               | Y      | N           | Y          | N      | Y         |
+------------------+--------------+-----------------+-----------------+--------+-------------+------------+--------+-----------+
| RNN              | Y            | N               | N               | N      | N           | N          | N      | Y         |
+------------------+--------------+-----------------+-----------------+--------+-------------+------------+--------+-----------+
| ARIMA            | Y            | Y               | Y               | Y      | N           | N          | Y      | N         |
+------------------+--------------+-----------------+-----------------+--------+-------------+------------+--------+-----------+
| Holt-Winters ES  | N            | N               | Y               | Y      | N           | N          | Y      | Y         |
+------------------+--------------+-----------------+-----------------+--------+-------------+------------+--------+-----------+
| Prophet          | N            | Y               | Y               | Y      | Y           | N          | Y      | N         |
+------------------+--------------+-----------------+-----------------+--------+-------------+------------+--------+-----------+
| Silverkite       | Y            | Y               | Y               | Y      | Y           | N          | Y      | N         |
+------------------+--------------+-----------------+-----------------+--------+-------------+------------+--------+-----------+
| Fourt Theta      | N            | N               | Y               | Y      | N           | N          | Y      | Y         |
+------------------+--------------+-----------------+-----------------+--------+-------------+------------+--------+-----------+
| Combo            | This model is a simple average, weighted average, or splice of 2+ already evaluated models.               |
+------------------+--------------+-----------------+-----------------+--------+-------------+------------+--------+-----------+