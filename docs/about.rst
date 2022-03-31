About
========

Scalecast is a package meant for those who have at least an intermediate understanding of time series forecasting theory and want to cut the tedious part of data processing, applying autoregression to models, differencing and undifferencing series, and visualizing results, usually on small-to-medium sized datasets (less than 1,000 data points). It can certainly be used for larger, more complex datasets, but probably isn't the best option for such a task. It is meant for standardizing and scaling an approach to many smaller series. For a package with more emphasis on deep learning and larger datasets that offers many of the same features as scalecast, I recommend `darts <https://unit8co.github.io/darts/>`_.

Scalecast has the following estimators available: 

* Any regression model from `Sklearn <https://scikit-learn.org/stable/>`_, including Sklearn APIs (like `Xgboost <https://xgboost.readthedocs.io/en/stable/>`_ and `LightGBM <https://lightgbm.readthedocs.io/en/latest/>`_)
* Recurrent neural nets from `Keras TensorFlow <https://keras.io/>`_
* Classic econometric models from `statsmodels <https://www.statsmodels.org/stable/>`_: Holt-Winters Exponential Smoothing and ARIMA
* `Facebook Prophet <https://facebook.github.io/prophet/>`_
* `LinkedIn Silverkite <https://engineering.linkedin.com/blog/2021/greykite--a-flexible--intuitive--and-fast-forecasting-library>`_
* Native combo/ensemble model

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
| Prophet          | N            | Y               | N               | Y      | Y           | N          | Y      | N         |
+------------------+--------------+-----------------+-----------------+--------+-------------+------------+--------+-----------+
| Silverkite       | Y            | Y               | Y               | Y      | Y           | N          | Y      | N         |
+------------------+--------------+-----------------+-----------------+--------+-------------+------------+--------+-----------+
| Combo            | This model is a simple average, weighted average, or splice of 2+ already evaluated models.               |
+------------------+--------------+-----------------+-----------------+--------+-------------+------------+--------+-----------+

The above table might be overly simplistic, but gives an idea of the various ways different models can be specified using scalecast.  

Scalecast offers the following advantages:  

* All models are validated out-of-sample with dynamic multi-step forecasting and this process extends to the future automatically, making implementation of any model on any series fast!
* Most models can be tuned by using a grid search on a validation slice of data.
* The package relies on new object types as little as possible, with only two native classes that each behave logically and with explicit commands. 
  * When a list-like object is required in an argument, lists, tuples, arrays, series, or other similar objects are all accepted.
  * Calling esitmators or scaling the data requires str arguments, which the objects know how to parse.
* Many different approaches available for mixing and matching time series concepts

Next, check out how to install the library and get started!