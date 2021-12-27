## call_me
- in `manual_forecast()` and `auto_forecast()` you can use the call_me parameter to specify the model nickname which is also the key stored in the object's history
- by default, this will be the same as whatever the estimator is called, so if you are using one of each kind of estimator, you don't need to worry about it

## history
structure:  
```
dict(call_me = 
  dict(
    'Estimator' = str: name of estimator in `_estimators_`, always set
    'Xvars' = list: name of utilized Xvars, None when no Xvars used, always set
    'HyperParams' = dict: name/value of hyperparams, empty when all defaults, always set 
    'Scaler' = str: name of normalizer used ('minmax','scale',None), always set
    'Forecast' = list: the forecast at whatever level it was run, always set
    'FittedVals' = list: the fitted values at whatever level the forecast was run, always set
    'Tuned' = str: "Not Tuned" when model was not tuned, "Dynamically" if `dynamic_tuning=True` was passed when tune() function was called, otherwise "Not Dynamically"; always set
    'DynamicallyTested' = bool: whether the models were dynamically tested, always set
    'Integration' = int: the integration of the model run, 0 when no series diff taken, never greater than 2, always set
    'TestSetLength' = int: the number of periods set aside to test the model, always set
    'TestSetRMSE' = float: the RMSE of the model on the test set at whatever level the forecast was run, always set
    'TestSetMAPE' = float: the MAPE of the model on the test set at whatever level the forecast was run, `None` when there is a 0 in the actuals, always set
    'TestSetMAE' = float: the MAE of the model on the test set at whatever level the forecast was run, always set
    'TestSetR2' = float: the MAE of the model on the test set at whatever level the forecast was run, never greater than 1, can be less than 0, always set
    'TestSetPredictions' = list: the predictions on the test set, always set
    'TestSetActuals' = list: the test-set actuals, always set 
    'InSampleRMSE' = float: the RMSE of the model on the entire y series using the fitted values to compare, always set
    'InSampleMAPE' = float: the MAPE of the model on the entire y series using the fitted values to compare, `None` when there is a 0 in the actuals, always set
    'InSampleMAE' = float: the MAE of the model on the entire y series using the fitted values to compare, always set
    'InSampleR2' = float: the R2 of the model on the entire y series using the fitted values to compare, always set
    'ValidationSetLength' = int: the number of periods before the test set periods to validate the model, only set when the model has been tuned
    'ValidationMetric' = str: the name of the metric used to validate the model, only set when the model has been tuned
    'ValidationMetricValue' = float: the value of the metric used to validate the model, only set when the model has been tuned
    'univariate' = bool: True if the model uses univariate features only (hwes, prophet, arima are the only estimators where this could be True), otherwise not set
    'first_obs' = list: the first y values from the undifferenced data, only set when `diff()` has been called
    'first_dates' = list: the first date values from the undifferenced data, only set when `diff()` has been called
    'grid_evaluated' = pandas dataframe: the evaluated grid, only set when the model has been tuned
    'models' = list: the models used in the combination, only set when the model is a 'combo' estimator
    'weights' = tuple: the weights used in the weighted average modeling, only set for weighted average combo models
    'LevelForecast' = list: the forecast in level (undifferenced terms), when data has not been differenced this is the same as 'Forecast', always set
    'LevelY' = list: the y value in level (undifferenced terms), when data has not been differenced this is the same as the y attribute, always set
    'LevelTestSetPreds' = list: the test-set predictions in level (undifferenced terms), when data has not been differenced this is the same as 'TestSetPredictions', always set
    'LevelTestSetRMSE' = float: the RMSE of the level test-set predictions vs. the level actuals, when data has not been differenced this is the same as 'TestSetRMSE', always set
    'LevelTestSetMAPE' = float: the MAPE of the level test-set predictions vs. the level actuals, when data has not been differenced this is the same as 'TestSetRMSE', None when there is a 0 in the level test-set actuals, always set
    'LevelTestSetMAE' = float: the MAE of the level test-set predictions vs. the level actuals, when data has not been differenced this is the same as 'TestSetRMSE', always set
    'LevelTestSetR2' = float: the R2 of the level test-set predictions vs. the level actuals, when data has not been differenced this is the same as 'TestSetRMSE', always set
    'LevelInSampleRMSE' = float: the RMSE of the level fitted values vs. the level actuals, when data has not been differenced this is the same as 'InSampleRMSE', always set
    'LevelInSampleMAPE' = float: the MAPE of the level fitted values vs. the level actuals, when data has not been differenced this is the same as 'InSampleRMSE', None if there is a 0 in the level actuals, always set
    'LevelInSampleMAE' = float: the MAE of the level fitted values vs. the level actuals, when data has not been differenced this is the same as 'InSampleRMSE', always set
    'LevelInSampleR2' = float: the R2 of the level fitted values vs. the level actuals, when data has not been differenced this is the same as 'InSampleRMSE', always set
    'feature_importance' = pandas dataframe: eli5 feature importance information (based on change in accuracy when a certain feature is filled with random data), only set when save_feature_importance() is called
    'summary_stats' = pandas dataframe: statsmodels summary stats information, only set when save_summary_stats() is called
  )
)
```
