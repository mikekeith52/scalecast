# 🌄 Scalecast: The practitioner's time series forecasting library

<p align="center">
  <img src="assets/logo2.png" />
</p>

## About

Scalecast is a light-weight modeling procedure, wrapper, and results container meant for those who are looking for the fastest way possible to apply, tune, and validate many different model classes for forecasting applications. In the Data Science industry, it is often asked of practitioners to deliver predictions and ranges of predictions for several lines of businesses or data slices, 100s or even 1000s. In such situations, it is common to see a simple linear regression or some other quick procedure applied to all lines due to the complexity of the task. This works well enough for people who need to deliver *something*, but more can be achieved.  

The scalecast package was designed to address this situation and offer advanced machine learning models that can be applied, optimized, and validated quickly. Unlike many libraries, the predictions produced by scalecast are always dynamic by default, not averages of one-step forecasts, so you don't run into the situation where the estimator looks great on the test-set but can't generalize to real data. What you see is what you get, with no attempt to oversell results. If you download a library that looks like it's able to predict the COVID pandemic in your test-set, you probably have a one-step forecast happening under-the-hood. You can't predict the unpredictable, and you won't see such things with scalecast.  

The library provides the `Forecaster` (for one series) and `MVForecaster` (for multiple series) wrappers around the following estimators: 

- Any regression model from [Sklearn](https://scikit-learn.org/stable/), including Sklearn APIs (like [Xgboost](https://xgboost.readthedocs.io/en/stable/) and [LightGBM](https://lightgbm.readthedocs.io/en/latest/))
- Recurrent neural nets from [Keras TensorFlow](https://keras.io/)
- Classic econometric models from [statsmodels](https://www.statsmodels.org/stable/): Holt-Winters Exponential Smoothing and ARIMA
- [Facebook Prophet](https://facebook.github.io/prophet)
- [LinkedIn Silverkite](https://engineering.linkedin.com/blog/2021/greykite--a-flexible--intuitive--and-fast-forecasting-library)
- Average, weighted average, and spliced models

<p align="center">
  <img src="https://media2.giphy.com/media/vV2Mbr9v6pH1D8hiLb/giphy.gif?cid=790b7611eb56b43191020435cbedf6453a74ddc2cebd017d&rid=giphy.gif&ct=g" width="600" height="400"/>
</p>

## Installation
- Only the base package is needed to get started:  
`pip install scalecast`  
- Optional add-ons:  
`pip install fbprophet` (prophet model--see [here](https://stackoverflow.com/questions/49889404/fbprophet-installation-error-failed-building-wheel-for-fbprophet) to resolve a common installation issue if using Anaconda)  
`pip install greykite` (silverkite model)  
`pip install tqdm` (progress bar with notebook)  
`pip install ipython` (widgets with notebook)  
`pip install ipywidgets` (widgets with notebook)  
`jupyter nbextension enable --py widgetsnbextension` (widgets with notebook)  
`jupyter labextension install @jupyter-widgets/jupyterlab-manager` (widgets with Lab)  

## Links
### Official Docs
  - [Read the Docs](https://scalecast.readthedocs.io/en/latest/)
  - [Introductory Example](https://scalecast-examples.readthedocs.io/en/latest/misc/introduction/introduction.html)
  - [Change Log](https://scalecast.readthedocs.io/en/latest/change_log.html)

### Forecasting with Different Model Types
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
- [May the Forecasts Be with You](https://towardsdatascience.com/may-the-forecasts-be-with-you-introducing-scalecast-pt-2-692f3f7f0be5)
- [Notebook](https://scalecast-examples.readthedocs.io/en/latest/misc/multi-series/multi-series.html)

## See Contributing
- [Contributing.md](./Contributing.md)