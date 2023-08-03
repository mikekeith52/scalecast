from scalecast.util import Forecaster_with_missing_vals
import pandas as pd
import numpy as np

def main():
	data = pd.DataFrame({
		'y':[1,2,np.nan,4],
		'Date':['2020-01-01','2020-02-01','2020-03-01','2020-04-01'],
	})
	f = Forecaster_with_missing_vals(
		y = data['y'],
		current_dates = data['Date'],
		fill_strategy = 'linear_interp',
	).round()

	assert f.y.values[2] == 3.0

	data = pd.DataFrame({
		'y':[1,2,4],
		'Date':['2020-01-01','2020-02-01','2020-04-01'],
	})

	f = Forecaster_with_missing_vals(
		y = data['y'],
		current_dates = data['Date'],
		fill_strategy = 'linear_interp',
		desired_frequency = 'MS',
	).round()

	assert f.y.values[2] == 3.0

if __name__ == '__main__':
    main()