import pandas as pd
import pandas_datareader as pdr
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from scalecast.Forecaster import Forecaster

models = ('mlr','knn','svr','xgboost','gbt','elasticnet','mlp','prophet','silverkite')
df = pdr.get_data_fred('HOUSTNSA',start='1900-01-01',end='2021-06-01')
f = Forecaster(y=df['HOUSTNSA'],current_dates=df.index) # to initialize, specify y and current_dates (must be arrays of the same length)
f.set_test_length(12) # specify a test length for your models - do this before eda
f.plot()

# eda
f.plot_acf(train_only=True)
f.plot_pacf(train_only=True)
plt.show()

# eda on differenced data
f.plot_acf(diffy=True,train_only=True)
f.plot_pacf(diffy=True,train_only=True)
plt.show()
f.seasonal_decompose(diffy=True,train_only=True).plot()
plt.show()

# model preprocessing
f.generate_future_dates(24) # this will create future dates that are on the same interval as the current dates and it will also set the forecast length
f.add_ar_terms(4) # add AR terms before differencing
f.add_AR_terms((2,12)) # seasonal AR terms
f.diff() # differences the y term and all ar terms to make a series stationary (also supports 2-level integration)
f.plot()
plt.show()
f.adf_test(quiet=False) # it is now stationary
f.add_seasonal_regressors('month',raw=False,sincos=True) # uses pandas attributes: raw=True creates integers (default), sincos=True creates wave functions (not default), dummy=True creates dummy vars (not default)
f.add_seasonal_regressors('year')
f.add_covid19_regressor() # dates are flexible, default is from when disney world closed to when US CDC lifted mask recommendations
f.add_time_trend()
f.add_combo_regressors('t','COVID19') # multiplies regressors together
f.set_validation_length(6) # length, different than test_length, to tune the hyperparameters 
f.tune_test_forecast(models,feature_importance=True,summary_stats=True) # by default, tuning is not dynamic but testing is

# combine models and run manually specified models of other varieties
f.set_estimator('combo')
f.manual_forecast(how='simple',models='top_3',determine_best_by='ValidationMetricValue',call_me='avg') # simple average of top_3 models based on performance in validation
f.manual_forecast(how='weighted',models=models,determine_best_by='ValidationMetricValue',call_me='weighted') # weighted average of all models based on metric specified in determine_best_by (default is the validation metric)

# plot results
matplotlib.use('QT5Agg')
f.plot(models='top_5',order_by='TestSetRMSE',print_attr=['TestSetRMSE','TestSetR2','TestSetMAPE','HyperParams','Xvars','models']) # plots the forecast differences or levels based on the level the forecast was performed on
plt.show()
f.plot(models='top_5',order_by='LevelTestSetMAPE',level=True,print_attr=['LevelTestSetRMSE','LevelTestSetR2','LevelTestSetMAPE']) # plot the level forecast
plt.show()
f.plot_test_set(models='top_5',order_by='TestSetR2',include_train=60) # see test-set performance visually of top 5 best models by r2 (last 60 obs only)
plt.show()
f.plot_fitted(order_by='TestSetR2') # plot fitted values of all models ordered by r2
plt.show()

# export key results
f.export(to_excel=True,determine_best_by='LevelTestSetMAPE',excel_name='housing_results.xlsx') # export interesting model metrics and forecasts (both level and non-level)
f.all_feature_info_to_excel(excel_name='housing_feature_info.xlsx')