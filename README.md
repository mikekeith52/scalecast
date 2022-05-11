# 🌄 Scalecast: The practitioner's time series forecasting library

<p align="center">
  <img src="assets/logo2.png" />
</p>

## About

Scalecast is a light-weight modeling procedure and wrapper meant for those who are looking for the fastest way possible to apply, tune, and validate many different model classes for forecasting applications. In the Data Science industry, it is often asked of practitioners to deliver predictions and ranges of predictions for several lines of businesses or data slices, 100s or even 1000s. In such situations, it is common to see a simple linear regression or some other quick procedure applied to all lines due to the complexity of the task. This works well enough for people who need to deliver *something*, but more can be achieved.  

The scalecast package was designed to address this situation and offer advanced machine learning models that can be applied, optimized, and validated quickly. Unlike many libraries, the predictions produced by scalecast are always dynamic by default, not averages of one-step forecasts, so you don't run into the situation where the estimator looks great on the test-set but can't generalize to real data. What you see is what you get, with no attempt to oversell results. If you download a library that looks like it's able to predict the COVID pandemic in your test-set, you probably have a one-step forecast happening under-the-hood. You can't predict the unpredictable, and you won't see such things with scalecast.  

The library provides the `Forecaster` (for one series) and `MVForecaster` (for multiple series) wrappers around the following estimators: 

- Any regression model from [Sklearn](https://scikit-learn.org/stable/), including Sklearn APIs (like [Xgboost](https://xgboost.readthedocs.io/en/stable/), and [LightGBM](https://lightgbm.readthedocs.io/en/latest/)).
- Recurrent neural nets from [Keras TensorFlow](https://keras.io/)
- Classic econometric models from [statsmodels](https://www.statsmodels.org/stable/): Holt-Winters Exponential Smoothing and ARIMA
- [Facebook Prophet](https://facebook.github.io/prophet)
- [LinkedIn Silverkite](https://engineering.linkedin.com/blog/2021/greykite--a-flexible--intuitive--and-fast-forecasting-library)
- Average, weighted average, and spliced models

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
|Links||
|----|----|
|[📚 Read the Docs](https://scalecast.readthedocs.io/en/latest/)|Official scalecast docs|
|[📋 Examples](https://scalecast-examples.readthedocs.io/en/latest/)|Official scalecast notebooks|
|[📓 TDS Article 1](https://towardsdatascience.com/expand-your-time-series-arsenal-with-these-models-10c807d37558)|Univariate Forecasting|
|[📓 TDS Article 2](https://towardsdatascience.com/multiple-series-forecast-them-together-with-any-sklearn-model-96319d46269)|Multivariate Forecasting|
|[🛠️ Change Log](https://scalecast.readthedocs.io/en/latest/change_log.html)|See what's changed|

## Example

Let's say we wanted to forecast each of the 1-year, 5-year, 10-year, 20-year, and 30-year corporate bond rates through the next 12 months. There are two ways we could do this with scalecast:  
1. Forecast each series individually (univariate)  
2. Forecast all series together (multivariate)  


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scalecast.Forecaster import Forecaster
from scalecast.MVForecaster import MVForecaster
from scalecast import GridGenerator
from scalecast.multiseries import export_model_summaries
import pandas_datareader as pdr

sns.set(rc={'figure.figsize':(14,7)})

df = pdr.get_data_fred(
    ['HQMCB1YR','HQMCB5YR','HQMCB10YR','HQMCB20YR','HQMCB30YR'],
    start='2000-01-01',
    end='2022-03-01'
)

f_dict = {c: Forecaster(y=df[c],current_dates=df.index) for c in df}
```

### Option 1 - Univariate

#### Select Models


```python
models = (
    'arima',       # linear time series model
    'elasticnet',  # linear model with regularization
    'knn',         # nearest neighbor model
    'xgboost',     # boosted tree model
)
```

#### Create Grids
These grids will be used to tune each model. To get example grids, you can use:  

```python
GridGenerator.get_example_grids()
```

This saves a Grids.py file to your working directory by default, which scalecast will know how to read. This example creates its own grids:  


```python
arima_grid = dict(
    order = [(1,1,1),(0,1,1),(0,1,0)],
    seasonal_order = [(1,1,1,12),(0,1,1,12),(0,1,0,12)],
    Xvars = [None,'all']
)

elasticnet_grid = dict(
    l1_ratio = [0.25,0.5,0.75,1],
    alpha = np.linspace(0,1,100),
)

knn_grid = dict(
    n_neighbors = np.arange(2,100,2)
)

xgboost_grid = dict(
     n_estimators=[150,200,250],
     scale_pos_weight=[5,10],
     learning_rate=[0.1,0.2],
     gamma=[0,3,5],
     subsample=[0.8,0.9],
)

grid_list = [arima_grid,elasticnet_grid,knn_grid,xgboost_grid]
grids = dict(
    zip(models,grid_list)
)
```

#### Select test length, validation length, and forecast horizon


```python
def prepare_fcst(f):
    f.set_test_length(0.2)
    f.set_validation_length(12)
    f.generate_future_dates(12)
```

#### Add seaonal regressors
These are regressors like month, quarter, dayofweek, dayofyear, minute, hour, etc. Raw integer values, dummy variables, or fourier transformed variables are avialable to be applied this way.  


```python
def add_seasonal_regressors(f):
    f.add_seasonal_regressors('month',raw=False,sincos=True)
    f.add_seasonal_regressors('year')
    f.add_seasonal_regressors('quarter',raw=False,dummy=True)
```

#### Choose Autoregressive Terms
A better way to do this would be to examine each series individually for autocorrelation, but this example uses three lags for each series and one seasonal seasonal lag (assuming 12-month seasonality).


```python
def add_ar_terms(f):
    f.add_ar_terms(3)       # lags
    f.add_AR_terms((1,12))  # seasonal lags
```

#### Write the forecast procedure


```python
def tune_test_forecast(k,f,models):
    for m in models:
        print(f'forecasting {m} for {k}')
        f.set_estimator(m)
        f.ingest_grid(grids[m])
        f.tune()
        f.auto_forecast()
```

#### Run a forecast loop


```python
for k, f in f_dict.items():
    prepare_fcst(f)
    add_seasonal_regressors(f)
    add_ar_terms(f)
    f.diff() # takes a first-difference in the series
    tune_test_forecast(k,f,models)
```

    forecasting arima for HQMCB1YR
    forecasting elasticnet for HQMCB1YR
    forecasting knn for HQMCB1YR
    forecasting xgboost for HQMCB1YR
    forecasting arima for HQMCB5YR
    forecasting elasticnet for HQMCB5YR
    forecasting knn for HQMCB5YR
    forecasting xgboost for HQMCB5YR
    forecasting arima for HQMCB10YR
    forecasting elasticnet for HQMCB10YR
    forecasting knn for HQMCB10YR
    forecasting xgboost for HQMCB10YR
    forecasting arima for HQMCB20YR
    forecasting elasticnet for HQMCB20YR
    forecasting knn for HQMCB20YR
    forecasting xgboost for HQMCB20YR
    forecasting arima for HQMCB30YR
    forecasting elasticnet for HQMCB30YR
    forecasting knn for HQMCB30YR
    forecasting xgboost for HQMCB30YR
    

#### Visualize results
Since there are 5 series to visualize, it might be undeserible to write a plot function for each one. Instead, scalecast let's you leverage Jupyter widgets by using this function:

```python
from scalecast.notebook import results_vis
results_vis(f_dict)
```

Because we aren't able to show widgets through markdown, this readme shows a visualization for the 30-year rate only:

##### Integrated Results


```python
f.plot_test_set(ci=True,order_by='LevelTestSetMAPE')
plt.title(f'{k} test-set results',size=16)
plt.show()
```


    
![png](README_files/README_17_0.png)
    


##### Level Results


```python
f.plot_test_set(level=True,order_by='LevelTestSetMAPE')
plt.title(f'{k} test-set results',size=16)
plt.show()
```


    
![png](README_files/README_19_0.png)
    


Using level test-set MAPE, the K-nearest Neighbor model performed the best, although, as we can see, predicting bond rates accurately is difficult if not impossible. To make the forecasts look better, we can set `dynamic_testing=False` in the `manual_forecast()` or `auto_forecast()` methods when calling forecasts. This will make test-set predictions an average on one-step forecasts. By default, everything is dynamic with scalecast to give a more realistic sense of how the models perform. To see our future predictions:


```python
f.plot(level=True,models='knn')
plt.title(f'{k} forecast',size=16)
plt.show()
```


    
![png](README_files/README_21_0.png)
    


### View Results
We can print a dataframe that shows how each model did on each series.


```python
results = export_model_summaries(f_dict,determine_best_by='LevelTestSetMAPE')
results.columns
```




    Index(['ModelNickname', 'Estimator', 'Xvars', 'HyperParams', 'Scaler',
           'Observations', 'Tuned', 'DynamicallyTested', 'Integration',
           'TestSetLength', 'TestSetRMSE', 'TestSetMAPE', 'TestSetMAE',
           'TestSetR2', 'LastTestSetPrediction', 'LastTestSetActual', 'CILevel',
           'CIPlusMinus', 'InSampleRMSE', 'InSampleMAPE', 'InSampleMAE',
           'InSampleR2', 'ValidationSetLength', 'ValidationMetric',
           'ValidationMetricValue', 'models', 'weights', 'LevelTestSetRMSE',
           'LevelTestSetMAPE', 'LevelTestSetMAE', 'LevelTestSetR2', 'best_model',
           'Series'],
          dtype='object')




```python
results[['ModelNickname','Series','LevelTestSetMAPE','LevelTestSetR2','HyperParams']]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ModelNickname</th>
      <th>Series</th>
      <th>LevelTestSetMAPE</th>
      <th>LevelTestSetR2</th>
      <th>HyperParams</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>elasticnet</td>
      <td>HQMCB1YR</td>
      <td>3.178337</td>
      <td>-1.156049</td>
      <td>{'l1_ratio': 0.25, 'alpha': 0.0}</td>
    </tr>
    <tr>
      <th>1</th>
      <td>knn</td>
      <td>HQMCB1YR</td>
      <td>3.526606</td>
      <td>-1.609881</td>
      <td>{'n_neighbors': 6}</td>
    </tr>
    <tr>
      <th>2</th>
      <td>arima</td>
      <td>HQMCB1YR</td>
      <td>5.053145</td>
      <td>-4.459315</td>
      <td>{'order': (1, 1, 1), 'seasonal_order': (0, 1, ...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>xgboost</td>
      <td>HQMCB1YR</td>
      <td>6.346458</td>
      <td>-8.099860</td>
      <td>{'n_estimators': 150, 'scale_pos_weight': 5, '...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>xgboost</td>
      <td>HQMCB5YR</td>
      <td>0.372265</td>
      <td>0.328466</td>
      <td>{'n_estimators': 250, 'scale_pos_weight': 5, '...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>knn</td>
      <td>HQMCB5YR</td>
      <td>0.521959</td>
      <td>0.119923</td>
      <td>{'n_neighbors': 26}</td>
    </tr>
    <tr>
      <th>6</th>
      <td>elasticnet</td>
      <td>HQMCB5YR</td>
      <td>0.664711</td>
      <td>-0.263214</td>
      <td>{'l1_ratio': 0.25, 'alpha': 0.0}</td>
    </tr>
    <tr>
      <th>7</th>
      <td>arima</td>
      <td>HQMCB5YR</td>
      <td>1.693368</td>
      <td>-7.332881</td>
      <td>{'order': (1, 1, 1), 'seasonal_order': (0, 1, ...</td>
    </tr>
    <tr>
      <th>8</th>
      <td>elasticnet</td>
      <td>HQMCB10YR</td>
      <td>0.145834</td>
      <td>0.390825</td>
      <td>{'l1_ratio': 0.25, 'alpha': 0.010101010101010102}</td>
    </tr>
    <tr>
      <th>9</th>
      <td>knn</td>
      <td>HQMCB10YR</td>
      <td>0.175513</td>
      <td>0.341443</td>
      <td>{'n_neighbors': 26}</td>
    </tr>
    <tr>
      <th>10</th>
      <td>xgboost</td>
      <td>HQMCB10YR</td>
      <td>0.179618</td>
      <td>0.271955</td>
      <td>{'n_estimators': 150, 'scale_pos_weight': 5, '...</td>
    </tr>
    <tr>
      <th>11</th>
      <td>arima</td>
      <td>HQMCB10YR</td>
      <td>0.569389</td>
      <td>-4.968184</td>
      <td>{'order': (1, 1, 1), 'seasonal_order': (0, 1, ...</td>
    </tr>
    <tr>
      <th>12</th>
      <td>elasticnet</td>
      <td>HQMCB20YR</td>
      <td>0.096262</td>
      <td>0.475912</td>
      <td>{'l1_ratio': 0.25, 'alpha': 0.030303030303030304}</td>
    </tr>
    <tr>
      <th>13</th>
      <td>knn</td>
      <td>HQMCB20YR</td>
      <td>0.103565</td>
      <td>0.486593</td>
      <td>{'n_neighbors': 26}</td>
    </tr>
    <tr>
      <th>14</th>
      <td>xgboost</td>
      <td>HQMCB20YR</td>
      <td>0.105995</td>
      <td>0.441079</td>
      <td>{'n_estimators': 200, 'scale_pos_weight': 5, '...</td>
    </tr>
    <tr>
      <th>15</th>
      <td>arima</td>
      <td>HQMCB20YR</td>
      <td>0.118035</td>
      <td>0.378294</td>
      <td>{'order': (1, 1, 1), 'seasonal_order': (0, 1, ...</td>
    </tr>
    <tr>
      <th>16</th>
      <td>knn</td>
      <td>HQMCB30YR</td>
      <td>0.086472</td>
      <td>0.559102</td>
      <td>{'n_neighbors': 68}</td>
    </tr>
    <tr>
      <th>17</th>
      <td>elasticnet</td>
      <td>HQMCB30YR</td>
      <td>0.089038</td>
      <td>0.538360</td>
      <td>{'l1_ratio': 0.25, 'alpha': 0.04040404040404041}</td>
    </tr>
    <tr>
      <th>18</th>
      <td>xgboost</td>
      <td>HQMCB30YR</td>
      <td>0.098148</td>
      <td>0.495318</td>
      <td>{'n_estimators': 200, 'scale_pos_weight': 5, '...</td>
    </tr>
    <tr>
      <th>19</th>
      <td>arima</td>
      <td>HQMCB30YR</td>
      <td>0.099879</td>
      <td>0.498333</td>
      <td>{'order': (1, 1, 1), 'seasonal_order': (0, 1, ...</td>
    </tr>
  </tbody>
</table>
</div>



### Option 2: Multivariate

#### Select Models
Only sklearn models are available with multivariate forecasting, so we can replace ARIMA with mlr.


```python
mv_models = (
    'mlr',
    'elasticnet',
    'knn',
    'xgboost',
)
```

#### Create Grids
We can use three of the same grids as we did in univariate forecasting and create a new MLR grid, with a modification to also search the optimal lag numbers. The `lags` argument can be an `int`, `list`, or `dict` type and all series will use the other series' lags (as well as theirown lags) in each model that is called. Again, for mv forecasting, you can use:  

```python
GridGenerator.get_mv_grids()
```

To save the MVGrids.py file to your working directory by default, which scalecast will know how to read.  


```python
mlr_grid = dict(lags = np.arange(1,13,1))
elasticnet_grid['lags'] = np.arange(1,13,1)
knn_grid['lags'] = np.arange(1,13,1)
xgboost_grid['lags'] = np.arange(1,13,1)

mv_grid_list = [mlr_grid,elasticnet_grid,knn_grid,xgboost_grid]
mv_grids = dict(zip(mv_models,mv_grid_list))
```

### Create multivariate forecasting object
- Need to change test and validation length
- Regressors are already carried forward from the underlying `Forecaster` objects
- Integration level are also carried forward from the underlying `Forecaster` objects


```python
mvf = MVForecaster(
    *f_dict.values(),
    names = f_dict.keys()
)

mvf.set_test_length(.2)
mvf.set_validation_length(12)
mvf
```




    MVForecaster(
        DateStartActuals=2000-02-01T00:00:00.000000000
        DateEndActuals=2022-03-01T00:00:00.000000000
        Freq=MS
        N_actuals=266
        N_series=5
        SeriesNames=['HQMCB1YR', 'HQMCB5YR', 'HQMCB10YR', 'HQMCB20YR', 'HQMCB30YR']
        ForecastLength=12
        Xvars=['monthsin', 'monthcos', 'year', 'quarter_1', 'quarter_2', 'quarter_3', 'quarter_4']
        TestLength=53
        ValidationLength=12
        ValidationMetric=rmse
        ForecastsEvaluated=[]
        CILevel=0.95
        BootstrapSamples=100
        CurrentEstimator=mlr
        OptimizeOn=mean
    )



### Choose how to optimize the models when tuning hyperparameters
Default behavior is use the mean performance of each model on all series. We don't have to run the line below to keep this behavior, but we also have the option to use this code to optimize performance on one of our series over the others. A future enhancement could include a weighted average.  


```python
mvf.set_optimize_on('mean')
```

#### Write Forecasting Procedure
- Instead of grid search, we will use randomized grid search to speed up evaluation times


```python
for m in mv_models:
    print(f'forecasting {m}')
    mvf.set_estimator(m)
    mvf.ingest_grid(mv_grids[m])
    mvf.limit_grid_size(100,random_seed=20) # do this because now grids are larger and this speeds it up
    mvf.tune()
    mvf.auto_forecast()
```

    forecasting mlr
    forecasting elasticnet
    forecasting knn
    forecasting xgboost
    

### Set best model


```python
mvf.set_best_model(determine_best_by='LevelTestSetMAPE')
mvf.best_model
```




    'elasticnet'



The elasticnet model was chosen based on its average test-set MAPE performance on all series.

#### Visualize results
Multivariate forecasting allows us to view all series and all models together. This could get jumbled, so let's just see the mlr and elasticnet results, knowing we can see the others if we want later.

##### Integrated Results


```python
mvf.plot_test_set(ci=True,models=['mlr','xgboost'])
plt.title(f'test-set results',size=16)
plt.show()
```


    
![png](README_files/README_39_0.png)
    


##### Level Results


```python
mvf.plot_test_set(level=True,models=['mlr','xgboost'])
plt.title(f'test-set results',size=16)
plt.show()
```


    
![png](README_files/README_41_0.png)
    


Once again, in this object, we can also set `dynamic_testing=False` in the `manual_forecast()` or `auto_forecast()` methods when calling forecasts. Let's see model forecasts into the future using the elasticent model only:


```python
mvf.plot(level=True,models='elasticnet')
plt.title(f'forecasts',size=16)
plt.show()
```


    
![png](README_files/README_43_0.png)
    


### View Results
We can print a dataframe that shows how each model did on each series.


```python
mvresults = mvf.export_model_summaries()
mvresults.columns
```




    Index(['Series', 'ModelNickname', 'Estimator', 'Xvars', 'HyperParams', 'Lags',
           'Scaler', 'Observations', 'Tuned', 'DynamicallyTested', 'Integration',
           'TestSetLength', 'TestSetRMSE', 'TestSetMAPE', 'TestSetMAE',
           'TestSetR2', 'LastTestSetPrediction', 'LastTestSetActual', 'CILevel',
           'CIPlusMinus', 'InSampleRMSE', 'InSampleMAPE', 'InSampleMAE',
           'InSampleR2', 'ValidationSetLength', 'ValidationMetric',
           'ValidationMetricValue', 'LevelTestSetRMSE', 'LevelTestSetMAPE',
           'LevelTestSetMAE', 'LevelTestSetR2', 'OptimizedOn', 'MetricOptimized',
           'best_model'],
          dtype='object')




```python
mvresults[['ModelNickname','Series','LevelTestSetMAPE','LevelTestSetR2','HyperParams']]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ModelNickname</th>
      <th>Series</th>
      <th>LevelTestSetMAPE</th>
      <th>LevelTestSetR2</th>
      <th>HyperParams</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>elasticnet</td>
      <td>HQMCB1YR</td>
      <td>1.750073</td>
      <td>0.066960</td>
      <td>{'l1_ratio': 0.25, 'alpha': 0.020202020202020204}</td>
    </tr>
    <tr>
      <th>1</th>
      <td>mlr</td>
      <td>HQMCB1YR</td>
      <td>3.793842</td>
      <td>-2.039209</td>
      <td>{}</td>
    </tr>
    <tr>
      <th>2</th>
      <td>knn</td>
      <td>HQMCB1YR</td>
      <td>2.370124</td>
      <td>-0.305976</td>
      <td>{'n_neighbors': 18}</td>
    </tr>
    <tr>
      <th>3</th>
      <td>xgboost</td>
      <td>HQMCB1YR</td>
      <td>6.137135</td>
      <td>-7.077600</td>
      <td>{'n_estimators': 150, 'scale_pos_weight': 10, ...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>elasticnet</td>
      <td>HQMCB5YR</td>
      <td>0.343014</td>
      <td>0.331734</td>
      <td>{'l1_ratio': 0.25, 'alpha': 0.020202020202020204}</td>
    </tr>
    <tr>
      <th>5</th>
      <td>mlr</td>
      <td>HQMCB5YR</td>
      <td>0.824692</td>
      <td>-0.856614</td>
      <td>{}</td>
    </tr>
    <tr>
      <th>6</th>
      <td>knn</td>
      <td>HQMCB5YR</td>
      <td>0.430460</td>
      <td>0.294105</td>
      <td>{'n_neighbors': 18}</td>
    </tr>
    <tr>
      <th>7</th>
      <td>xgboost</td>
      <td>HQMCB5YR</td>
      <td>1.177128</td>
      <td>-2.750770</td>
      <td>{'n_estimators': 150, 'scale_pos_weight': 10, ...</td>
    </tr>
    <tr>
      <th>8</th>
      <td>elasticnet</td>
      <td>HQMCB10YR</td>
      <td>0.140750</td>
      <td>0.367196</td>
      <td>{'l1_ratio': 0.25, 'alpha': 0.020202020202020204}</td>
    </tr>
    <tr>
      <th>9</th>
      <td>mlr</td>
      <td>HQMCB10YR</td>
      <td>0.243091</td>
      <td>-0.063537</td>
      <td>{}</td>
    </tr>
    <tr>
      <th>10</th>
      <td>knn</td>
      <td>HQMCB10YR</td>
      <td>0.139242</td>
      <td>0.381919</td>
      <td>{'n_neighbors': 18}</td>
    </tr>
    <tr>
      <th>11</th>
      <td>xgboost</td>
      <td>HQMCB10YR</td>
      <td>0.168247</td>
      <td>0.151429</td>
      <td>{'n_estimators': 150, 'scale_pos_weight': 10, ...</td>
    </tr>
    <tr>
      <th>12</th>
      <td>elasticnet</td>
      <td>HQMCB20YR</td>
      <td>0.090550</td>
      <td>0.475224</td>
      <td>{'l1_ratio': 0.25, 'alpha': 0.020202020202020204}</td>
    </tr>
    <tr>
      <th>13</th>
      <td>mlr</td>
      <td>HQMCB20YR</td>
      <td>0.107826</td>
      <td>0.438056</td>
      <td>{}</td>
    </tr>
    <tr>
      <th>14</th>
      <td>knn</td>
      <td>HQMCB20YR</td>
      <td>0.090178</td>
      <td>0.453847</td>
      <td>{'n_neighbors': 18}</td>
    </tr>
    <tr>
      <th>15</th>
      <td>xgboost</td>
      <td>HQMCB20YR</td>
      <td>0.318796</td>
      <td>-3.920447</td>
      <td>{'n_estimators': 150, 'scale_pos_weight': 10, ...</td>
    </tr>
    <tr>
      <th>16</th>
      <td>elasticnet</td>
      <td>HQMCB30YR</td>
      <td>0.083620</td>
      <td>0.545363</td>
      <td>{'l1_ratio': 0.25, 'alpha': 0.020202020202020204}</td>
    </tr>
    <tr>
      <th>17</th>
      <td>mlr</td>
      <td>HQMCB30YR</td>
      <td>0.084045</td>
      <td>0.558262</td>
      <td>{}</td>
    </tr>
    <tr>
      <th>18</th>
      <td>knn</td>
      <td>HQMCB30YR</td>
      <td>0.080525</td>
      <td>0.546820</td>
      <td>{'n_neighbors': 18}</td>
    </tr>
    <tr>
      <th>19</th>
      <td>xgboost</td>
      <td>HQMCB30YR</td>
      <td>0.353802</td>
      <td>-5.572455</td>
      <td>{'n_estimators': 150, 'scale_pos_weight': 10, ...</td>
    </tr>
  </tbody>
</table>
</div>



## Backtest results
To test how well, on average, our models would have done across the last-10 12-month forecast horizons, we can use the `backtest()` method. It works for both the `Forecaster` and `MVForecaster` objects.


```python
mvf.backtest('elasticnet')
mvf.export_backtest_metrics('elasticnet')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>iter1</th>
      <th>iter2</th>
      <th>iter3</th>
      <th>iter4</th>
      <th>iter5</th>
      <th>iter6</th>
      <th>iter7</th>
      <th>iter8</th>
      <th>iter9</th>
      <th>iter10</th>
      <th>mean</th>
    </tr>
    <tr>
      <th>series</th>
      <th>metric</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="4" valign="top">HQMCB1YR</th>
      <th>RMSE</th>
      <td>0.706852</td>
      <td>0.527615</td>
      <td>0.390606</td>
      <td>0.322483</td>
      <td>0.251028</td>
      <td>0.187788</td>
      <td>0.151881</td>
      <td>0.140694</td>
      <td>0.095492</td>
      <td>0.069687</td>
      <td>0.284413</td>
    </tr>
    <tr>
      <th>MAE</th>
      <td>0.459374</td>
      <td>0.386159</td>
      <td>0.305939</td>
      <td>0.257141</td>
      <td>0.19432</td>
      <td>0.141388</td>
      <td>0.110274</td>
      <td>0.104175</td>
      <td>0.064198</td>
      <td>0.064893</td>
      <td>0.208786</td>
    </tr>
    <tr>
      <th>R2</th>
      <td>-1.255519</td>
      <td>-2.283721</td>
      <td>-4.359326</td>
      <td>-7.036131</td>
      <td>-12.338531</td>
      <td>-29.907128</td>
      <td>-57.073258</td>
      <td>-41.417445</td>
      <td>-12.909921</td>
      <td>-1.875425</td>
      <td>-17.045640</td>
    </tr>
    <tr>
      <th>MAPE</th>
      <td>0.627229</td>
      <td>0.78698</td>
      <td>0.796553</td>
      <td>0.783229</td>
      <td>0.663931</td>
      <td>0.534689</td>
      <td>0.448602</td>
      <td>0.42999</td>
      <td>0.270442</td>
      <td>0.247597</td>
      <td>0.558924</td>
    </tr>
    <tr>
      <th rowspan="4" valign="top">HQMCB5YR</th>
      <th>RMSE</th>
      <td>0.733883</td>
      <td>0.81338</td>
      <td>0.757777</td>
      <td>0.709017</td>
      <td>0.621308</td>
      <td>0.536518</td>
      <td>0.515865</td>
      <td>0.504708</td>
      <td>0.416658</td>
      <td>0.304795</td>
      <td>0.591391</td>
    </tr>
    <tr>
      <th>MAE</th>
      <td>0.471666</td>
      <td>0.69315</td>
      <td>0.677834</td>
      <td>0.639691</td>
      <td>0.539043</td>
      <td>0.448541</td>
      <td>0.435568</td>
      <td>0.420885</td>
      <td>0.314717</td>
      <td>0.236134</td>
      <td>0.487723</td>
    </tr>
    <tr>
      <th>R2</th>
      <td>-0.992364</td>
      <td>-4.055345</td>
      <td>-6.994801</td>
      <td>-8.990405</td>
      <td>-7.049609</td>
      <td>-6.095777</td>
      <td>-6.708569</td>
      <td>-5.73555</td>
      <td>-3.185744</td>
      <td>-1.240836</td>
      <td>-5.104900</td>
    </tr>
    <tr>
      <th>MAPE</th>
      <td>0.225336</td>
      <td>0.430285</td>
      <td>0.466327</td>
      <td>0.471569</td>
      <td>0.410762</td>
      <td>0.354883</td>
      <td>0.364207</td>
      <td>0.362355</td>
      <td>0.267427</td>
      <td>0.211556</td>
      <td>0.356471</td>
    </tr>
    <tr>
      <th rowspan="4" valign="top">HQMCB10YR</th>
      <th>RMSE</th>
      <td>0.422801</td>
      <td>0.553046</td>
      <td>0.584547</td>
      <td>0.640398</td>
      <td>0.567735</td>
      <td>0.501976</td>
      <td>0.530126</td>
      <td>0.581573</td>
      <td>0.544647</td>
      <td>0.352862</td>
      <td>0.527971</td>
    </tr>
    <tr>
      <th>MAE</th>
      <td>0.278056</td>
      <td>0.478105</td>
      <td>0.544114</td>
      <td>0.604623</td>
      <td>0.511173</td>
      <td>0.431248</td>
      <td>0.451311</td>
      <td>0.499245</td>
      <td>0.438672</td>
      <td>0.294899</td>
      <td>0.453145</td>
    </tr>
    <tr>
      <th>R2</th>
      <td>-0.415119</td>
      <td>-4.45616</td>
      <td>-10.06333</td>
      <td>-12.765917</td>
      <td>-6.293862</td>
      <td>-3.949755</td>
      <td>-4.386444</td>
      <td>-4.736976</td>
      <td>-3.351877</td>
      <td>-0.685583</td>
      <td>-5.110502</td>
    </tr>
    <tr>
      <th>MAPE</th>
      <td>0.093302</td>
      <td>0.173109</td>
      <td>0.206824</td>
      <td>0.235942</td>
      <td>0.199909</td>
      <td>0.169466</td>
      <td>0.179948</td>
      <td>0.201402</td>
      <td>0.175654</td>
      <td>0.122154</td>
      <td>0.175771</td>
    </tr>
    <tr>
      <th rowspan="4" valign="top">HQMCB20YR</th>
      <th>RMSE</th>
      <td>0.34036</td>
      <td>0.284787</td>
      <td>0.374716</td>
      <td>0.490949</td>
      <td>0.417145</td>
      <td>0.314183</td>
      <td>0.381984</td>
      <td>0.491789</td>
      <td>0.504693</td>
      <td>0.296888</td>
      <td>0.389749</td>
    </tr>
    <tr>
      <th>MAE</th>
      <td>0.290539</td>
      <td>0.209799</td>
      <td>0.342232</td>
      <td>0.465807</td>
      <td>0.375293</td>
      <td>0.266775</td>
      <td>0.315278</td>
      <td>0.426912</td>
      <td>0.42566</td>
      <td>0.258542</td>
      <td>0.337684</td>
    </tr>
    <tr>
      <th>R2</th>
      <td>-0.314721</td>
      <td>-0.662623</td>
      <td>-3.77501</td>
      <td>-7.090738</td>
      <td>-3.639573</td>
      <td>-1.399332</td>
      <td>-2.501422</td>
      <td>-4.672199</td>
      <td>-4.319498</td>
      <td>-0.656571</td>
      <td>-2.903169</td>
    </tr>
    <tr>
      <th>MAPE</th>
      <td>0.088109</td>
      <td>0.061599</td>
      <td>0.10509</td>
      <td>0.145226</td>
      <td>0.116395</td>
      <td>0.082425</td>
      <td>0.097111</td>
      <td>0.132432</td>
      <td>0.131665</td>
      <td>0.08193</td>
      <td>0.104198</td>
    </tr>
    <tr>
      <th rowspan="4" valign="top">HQMCB30YR</th>
      <th>RMSE</th>
      <td>0.341241</td>
      <td>0.219195</td>
      <td>0.316709</td>
      <td>0.439526</td>
      <td>0.376742</td>
      <td>0.254178</td>
      <td>0.345932</td>
      <td>0.461224</td>
      <td>0.487511</td>
      <td>0.288019</td>
      <td>0.353028</td>
    </tr>
    <tr>
      <th>MAE</th>
      <td>0.309357</td>
      <td>0.170457</td>
      <td>0.279142</td>
      <td>0.412227</td>
      <td>0.333939</td>
      <td>0.209202</td>
      <td>0.284627</td>
      <td>0.401713</td>
      <td>0.416847</td>
      <td>0.241643</td>
      <td>0.305915</td>
    </tr>
    <tr>
      <th>R2</th>
      <td>-0.528339</td>
      <td>0.053586</td>
      <td>-1.716496</td>
      <td>-4.152778</td>
      <td>-2.369588</td>
      <td>-0.493216</td>
      <td>-1.793291</td>
      <td>-4.127078</td>
      <td>-4.26394</td>
      <td>-0.692692</td>
      <td>-2.008383</td>
    </tr>
    <tr>
      <th>MAPE</th>
      <td>0.094696</td>
      <td>0.050634</td>
      <td>0.083647</td>
      <td>0.125428</td>
      <td>0.10085</td>
      <td>0.062829</td>
      <td>0.085035</td>
      <td>0.120783</td>
      <td>0.125097</td>
      <td>0.073433</td>
      <td>0.092243</td>
    </tr>
  </tbody>
</table>
</div>



## Correlation Matrix


```python
heatmap_kwargs = dict(
    disp='heatmap',
    vmin=-1,
    vmax=1,
    annot=True,
)
mvf.corr(**heatmap_kwargs)
plt.show()
```


    
![png](README_files/README_50_0.png)
    



```python
mvf.corr_lags(y='HQMCB1YR',x='HQMCB30YR',lags=12,**heatmap_kwargs)
plt.show()
```


    
![png](README_files/README_51_0.png)
    



```python

```
