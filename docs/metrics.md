## error accuracy metrics
- for model evaluation, combination modeling, and plotting
- both level and integrated metrics are available 
  - if the forecasts were performed on level data, these will be the same
  - if the series were differenced, these can offer interesting contrasts and views of accuracy
```python
_metrics_ = {'r2','rmse','mape','mae'}
_determine_best_by_ = {'TestSetRMSE','TestSetMAPE','TestSetMAE','TestSetR2','InSampleRMSE','InSampleMAPE','InSampleMAE',
                        'InSampleR2','ValidationMetricValue','LevelTestSetRMSE','LevelTestSetMAPE','LevelTestSetMAE',
                        'LevelTestSetR2',None}
```

### in-sample metrics
- `'InSampleRMSE','InSampleMAPE','InSampleMAE','InSampleR2'`
- These can be used to detect overfitting
- Should not be used for determining best models/weights when combination modeling as these also include the test set within them
- Still available for combination modeling in case you want to use them, but it should be understood that the accuracy metrics will be unreliable
- stored in the history attribute

### out-of-sample metrics
- `'TestSetRMSE','TestSetMAPE','TestSetMAE','TestSetR2','LevelTestSetRMSE','LevelTestSetMAPE','LevelTestSetMAE','LevelTestSetR2'`
- These are good for ordering models from best to worst according to how well they predicted out-of-sample values
- Should not be used for  for determining best models/weights when combination modeling as it will lead to data leakage and overfitting
- Compare to in-sample metrics for a good sense of how well-fit the model is
- stored in the history attribute

### validation metrics
- `'ValidationMetricValue'` is stored in the history attribute
  - based on `'ValidationMetric'`, also stored in history
    - one of `{'r2','rmse','mape','mae'}`
- This will only be populated if you first tune the model with a grid and use the tune() method
- This metric can be used for combination modeling without data leakage/overfitting as they are derived from out-of-sample data but not included in the test-set
  - If you change the validation metric during the tuning process, this will no longer be reliable for combination modeling 