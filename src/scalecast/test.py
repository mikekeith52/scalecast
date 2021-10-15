from Forecaster import Forecaster

f = Forecaster(y=[1,2,3],current_dates=['2021-01-01','2021-02-01','2021-03-01'])
f.generate_future_dates(2)
f.add_time_trend()
f.set_estimator('mlr')
f.manual_forecast()
f.plot()