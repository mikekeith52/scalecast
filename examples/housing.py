import pandas as pd
import pandas_datareader as pdr
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scalecastdev.Forecaster import Forecaster

df = pdr.get_data_fred('HOUSTNSA',start='1900-01-01',end='2021-05-01')
f = Forecaster(y=df['HOUSTNSA'],current_dates=df.index) # to initialize, specify y and current_dates (must be arrays of the same length)
f.plot()

# time series exploration
f.plot_acf()
f.plot_pacf()
plt.show()
# time series exploration on differenced data
f.plot_acf(diffy=True)
f.plot_pacf(diffy=True)
plt.show()
f.seasonal_decompose(diffy=True).plot()
plt.show()

f.set_test_length(12) # specify a test length for your models--it's a good idea to keep this the same for all forecasts
f.generate_future_dates(24) # this will create future dates that are on the same interval as the current dates and it will also set the forecast length
f.add_ar_terms(4) # add AR terms before differencing
f.add_AR_terms((2,12)) # seasonal AR terms
f.diff() # differences the y term and all ar terms to make a series stationary (also supports 2-level integration)
f.plot()
f.adf_test(quiet=False) # it is now stationary
f.add_seasonal_regressors('month',raw=False,sincos=True) # uses pandas attributes: raw=True creates integers (default), sincos=True creates wave functions (not default), dummy=True creates dummy vars (not default)
f.add_seasonal_regressors('year')
f.add_covid19_regressor() # dates are flexible, default is from when disney world closed to when US CDC lifted mask recommendations
f.add_time_trend()
f.add_combo_regressors('t','COVID19') # multiplies regressors together
f.add_poly_terms('t',pwr=3) # by default, creates an order 2 regressor, n-order polynomial terms are allowed
f.set_validation_length(6) # length, different than test_length, to tune the hyperparameters 

# automatically tune and forecast with a series of models
models = ('mlr','knn','svr','xgboost','gbt','elasticnet','mlp','prophet','silverkite')
for m in models:
	f.set_estimator(m)
	f.tune() # by default, will pull the grid with the same name as the estimator (mlr will pull the mlr grid, etc.)
	f.auto_forecast()

# combine models and run manually specified models of other varieties
f.set_estimator('combo')
f.manual_forecast(how='simple',models='top_3',determine_best_by='ValidationMetricValue',call_me='avg') # simple average of top_3 models based on performance in validation
f.manual_forecast(how='weighted',models=models,determine_best_by='ValidationMetricValue',call_me='weighted') # weighted average of all models based on metric specified in determine_best_by (default is the validation metric)

matplotlib.use('QT5Agg')
f.plot(models='top_5',order_by='TestSetRMSE',print_attr=['TestSetRMSE','TestSetR2','TestSetMAPE','HyperParams','Xvars','models']) # plots the forecast differences or levels based on the level the forecast was performed on
f.plot(models='top_5',order_by='LevelTestSetMAPE',level=True,print_attr=['LevelTestSetRMSE','LevelTestSetR2','LevelTestSetMAPE']) # plot the level forecast
f.plot_test_set(models='top_5',order_by='TestSetR2',include_train=60) # see test-set performance visually of top 5 best models by r2 (last 60 obs only)
f.plot_fitted(order_by='TestSetR2') # plot fitted values of all models ordered by r2

f.export(to_excel=True,determine_best_by='LevelTestSetMAPE',excel_name='housing_results.xlsx') # export interesting model metrics and forecasts (both level and non-level)