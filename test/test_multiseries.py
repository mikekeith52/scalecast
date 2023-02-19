from scalecast.Forecaster import Forecaster
from scalecast.multiseries import export_model_summaries, keep_smallest_first_date
import pandas_datareader as pdr 

def main():
    f_dict = {}
    models = ('elasticnet','xgboost')
    for sym in ('UNRATE','GDP'):
        df = pdr.get_data_fred(sym, start = '1900-01-01')
        f = Forecaster(
            y=df[sym],
            current_dates=df.index,
            future_dates = 12,
            test_length = .2,
            validation_length = 12,
        )
        f_dict[sym] = f
    keep_smallest_first_date(*f_dict.values())

    for k, f in f_dict.items():
        f.add_ar_terms(12)
        f.add_time_trend()
        for m in models:
            f.set_estimator(m)
            f.manual_forecast()
        
    model_summaries = export_model_summaries(f_dict,determine_best_by='TestSetMAE')

if __name__ == '__main__':
    main()