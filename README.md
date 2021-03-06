# 🌄 Scalecast: The practitioner's time series forecasting library

<p align="center">
  <img src="https://github.com/mikekeith52/scalecast-examples/blob/main/logo2.png" />
</p>

## About

Scalecast is a light-weight modeling procedure, wrapper, and results container meant for those who are looking for the fastest way possible to apply, tune, and validate many different model classes for forecasting applications. In the Data Science industry, it is often asked of practitioners to deliver predictions and ranges of predictions for several lines of businesses or data slices, 100s or even 1000s. In such situations, it is common to see a simple linear regression or some other quick procedure applied to all lines due to the complexity of the task. This works well enough for people who need to deliver *something*, but more can be achieved.  

The scalecast package was designed to address this situation and offer advanced machine learning models that can be applied, optimized, and validated quickly. Unlike many libraries, the predictions produced by scalecast are always dynamic by default, not averages of one-step forecasts, so you don't run into the situation where the estimator looks great on the test-set but can't generalize to real data. What you see is what you get, with no attempt to oversell results. If you download a library that looks like it's able to predict the COVID pandemic in your test-set, you probably have a one-step forecast happening under-the-hood. You can't predict the unpredictable, and you won't see such things with scalecast.  

The library provides the [`Forecaster`](https://scalecast.readthedocs.io/en/latest/Forecaster/Forecaster.html) (for one series) and [`MVForecaster`](https://scalecast.readthedocs.io/en/latest/Forecaster/MVForecaster.html) (for multiple series) wrappers around the following estimators: 

- [Scikit-Learn](https://scikit-learn.org/stable/)
  - [ElasticNet](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html)
  - [Gradient Boosted Trees](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html)
  - [k-Nearest Neighbors](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html)
  - [Lasso](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html)
  - [Multi-level Perceptron](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html)
  - [Multiple Linear Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
  - [Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
  - [Ridge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)
  - [Stochastic Gradient Descent](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html)
  - [Support Vector Machine](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html)
  - [Any other sklearn regression model or regression model that uses an sklearn interface can be ported](https://scalecast.readthedocs.io/en/latest/Forecaster/Forecaster.html#src.scalecast.Forecaster.Forecaster.add_sklearn_estimator)
- [XGBoost](https://xgboost.readthedocs.io/en/stable/)
- [LightGBM](https://lightgbm.readthedocs.io/en/latest/)

The `Forecaster` object only can also use:

- [StatsModels](https://www.statsmodels.org/stable/)
  - [ARIMA](https://www.statsmodels.org/devel/generated/statsmodels.tsa.arima.model.ARIMA.html)
  - [Holt-Winters Exponential Smoothing](https://www.statsmodels.org/dev/generated/statsmodels.tsa.holtwinters.ExponentialSmoothing.html)
- [Darts](https://unit8co.github.io/darts/)
  - [Four Theta](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.theta.html)
- [Keras TensorFlow Cells](https://keras.io/)
  - [LSTM](https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM)
  - [SimpleRNN](https://www.tensorflow.org/api_docs/python/tf/keras/layers/SimpleRNN)
  - [Dense](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense)
- [Facebook Prophet](https://facebook.github.io/prophet)
- [LinkedIn Greykite](https://engineering.linkedin.com/blog/2021/greykite--a-flexible--intuitive--and-fast-forecasting-library)
  - [silverkite](https://linkedin.github.io/greykite/docs/0.1.0/html/pages/model_components/0100_introduction.html)
- [Native Combo model](https://scalecast.readthedocs.io/en/latest/Forecaster/_forecast.html#module-src.scalecast.Forecaster.Forecaster._forecast_combo)

The library interfaces nicely with interactive notebook applications.

<p align="center">
  <img src="https://media2.giphy.com/media/vV2Mbr9v6pH1D8hiLb/giphy.gif?cid=790b7611eb56b43191020435cbedf6453a74ddc2cebd017d&rid=giphy.gif&ct=g" width="700" height="300"/>
</p>

In addition, scalecast offers:
- Model Validation
  - [Grid search on validation data](https://scalecast.readthedocs.io/en/latest/Forecaster/Forecaster.html#src.scalecast.Forecaster.Forecaster.tune)
  - [Grid search using time series cross validation](https://scalecast.readthedocs.io/en/latest/Forecaster/Forecaster.html#src.scalecast.Forecaster.Forecaster.cross_validate)
  - [Backtest](https://scalecast.readthedocs.io/en/latest/Forecaster/Forecaster.html#src.scalecast.Forecaster.Forecaster.backtest)
- Model input analysis
  - [Feature importance scoring](https://scalecast.readthedocs.io/en/latest/Forecaster/Forecaster.html#src.scalecast.Forecaster.Forecaster.save_feature_importance)
    - [SHAP](https://shap.readthedocs.io/en/latest/index.html)
    - [Permutated feature scoring with ELI5](https://eli5.readthedocs.io/en/latest/index.html)
  - [Summary stats for descriptive models](https://scalecast.readthedocs.io/en/latest/Forecaster/Forecaster.html#src.scalecast.Forecaster.Forecaster.save_summary_stats)
  - [Feature reduction](https://scalecast.readthedocs.io/en/latest/Forecaster/Forecaster.html#src.scalecast.Forecaster.Forecaster.reduce_Xvars)
- [Anomaly detection](https://scalecast.readthedocs.io/en/latest/Forecaster/AnomalyDetector.html)
- [Changepoint detection](https://scalecast.readthedocs.io/en/latest/Forecaster/ChangepointDetector.html)
- [Series transformation/revert functions](https://scalecast.readthedocs.io/en/latest/Forecaster/SeriesTransformer.html)

## Installation
- Only the base package is needed to get started:  
`pip install scalecast`  
- Optional add-ons:  
`pip install darts`  
`pip install fbprophet` (see [here](https://stackoverflow.com/questions/49889404/fbprophet-installation-error-failed-building-wheel-for-fbprophet) to resolve a common installation issue if using Anaconda)  
`pip install greykite`   
`pip install shap` (SHAP feature importance)  
`pip install kats` (for changepoint detection)  
`pip install tqdm` (progress bar with notebook)  
`pip install ipython` (widgets with notebook)  
`pip install ipywidgets` (widgets with notebook)  
`jupyter nbextension enable --py widgetsnbextension` (widgets with notebook)  
`jupyter labextension install @jupyter-widgets/jupyterlab-manager` (widgets with Lab)  

## Links
### Official Docs
  - [Read the Docs](https://scalecast.readthedocs.io/en/latest/)
  - [Introductory Example](https://scalecast-examples.readthedocs.io/en/latest/misc/introduction/introduction.html)
  - [Examples Repository](https://github.com/mikekeith52/scalecast-examples)
  - [Change Log](https://scalecast.readthedocs.io/en/latest/change_log.html)

### [Forecasting with Different Model Types](https://scalecast.readthedocs.io/en/latest/Forecaster/_forecast.html)
- Sklearn Univariate
  - [Expand your Time Series Arsenal with These Models](https://towardsdatascience.com/expand-your-time-series-arsenal-with-these-models-10c807d37558)
  - [Notebook](https://scalecast-examples.readthedocs.io/en/latest/sklearn/sklearn.html)
- Sklearn Multivariate
  - [Multiple Series? Forecast Them together with any Sklearn Model](https://towardsdatascience.com/multiple-series-forecast-them-together-with-any-sklearn-model-96319d46269)
  - [Notebook](https://scalecast-examples.readthedocs.io/en/latest/multivariate/multivariate.html)
- RNN 
  - [Exploring the LSTM Neural Network Model for Time Series](https://towardsdatascience.com/exploring-the-lstm-neural-network-model-for-time-series-8b7685aa8cf)
  - [LSTM Notebook](https://scalecast-examples.readthedocs.io/en/latest/lstm/lstm.html)
  - [RNN Notebook](https://scalecast-examples.readthedocs.io/en/latest/rnn/rnn.html)
- ARIMA
  - [Forecast with ARIMA in Python More Easily with Scalecast](https://towardsdatascience.com/forecast-with-arima-in-python-more-easily-with-scalecast-35125fc7dc2e)
  - [Notebook](https://scalecast-examples.readthedocs.io/en/latest/arima/arima.html)
- Theta
  - [Easily Employ A Theta Model For Time Series](https://medium.com/towards-data-science/easily-employ-a-theta-model-for-time-series-b94465099a00)
  - [Notebook](https://scalecast-examples.readthedocs.io/en/latest/theta/theta.html)
- Other Notebooks
  - [Prophet](https://scalecast-examples.readthedocs.io/en/latest/prophet/prophet.html)
  - [Combo](https://scalecast-examples.readthedocs.io/en/latest/combo/combo.html)
  - [Holt-Winters Exponential Smoothing](https://scalecast-examples.readthedocs.io/en/latest/hwes/hwes.html)
  - [Silverkite](https://scalecast-examples.readthedocs.io/en/latest/silverkite/silverkite.html)
  
### The importance of dynamic validation
- [How Not to be Fooled by Time Series Models](https://towardsdatascience.com/how-not-to-be-fooled-by-time-series-forecasting-8044f5838de3)
- [Model Validation Techniques for Time Series](https://towardsdatascience.com/model-validation-techniques-for-time-series-3518269bd5b3)
- [Notebook](https://scalecast-examples.readthedocs.io/en/latest/misc/validation/validation.html)

### Model Input Selection
- [Variable Reduction Techniques for Time Series](https://medium.com/towards-data-science/variable-reduction-techniques-for-time-series-646743f726d4)
- [Notebook](https://scalecast-examples.readthedocs.io/en/latest/misc/feature-selection/feature_selection.html)

### Scaled Forecasting on Many Series
- An example using the [M4 dataset](https://github.com/Mcompetitions/M4-methods) coming soon!
- [May the Forecasts Be with You](https://towardsdatascience.com/may-the-forecasts-be-with-you-introducing-scalecast-pt-2-692f3f7f0be5)
- [Notebook](https://scalecast-examples.readthedocs.io/en/latest/misc/multi-series/multi-series.html)

### Anomaly Detection
- [Anomaly Detection for Time Series with Monte Carlo Simulations](https://towardsdatascience.com/anomaly-detection-for-time-series-with-monte-carlo-simulations-e43c77ba53c?source=email-85177a9cbd35-1658325190052-activity.collection_post_approved)
- [Notebook1](https://scalecast-examples.readthedocs.io/en/latest/misc/anomalies/anomalies.html)
- [Notebook2](https://github.com/mikekeith52/scalecast-examples/blob/main/misc/anomalies/monte%20carlo/monte%20carlo.ipynb)

## See Contributing
- [Contributing.md](./Contributing.md)
