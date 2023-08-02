from test_Forecaster import build_Forecaster
from scalecast.SeriesTransformer import SeriesTransformer
import numpy as np
import matplotlib.pyplot as plt

def forecaster(f):
    for m in ['mlr','elasticnet']:
        f.drop_all_Xvars()
        f.set_estimator(m)
        f.auto_Xvar_select(estimator=m)
        f.determine_best_series_length(estimator=m)
        f.tune()
        f.auto_forecast()
        f.restore_series_length()

    f.set_estimator('prophet') # testing #68
    def add_seasonregr(m):
          m.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    f.manual_forecast(callback_func = add_seasonregr)

def comp_vals(orig_vals,new_vals,transformation):
    assert (orig_vals == new_vals).all(), f'{transformation} revert did not work'

def main():
    for tl in (0,24):
        print(tl)
        f = build_Forecaster(test_length = tl)

        orig_vals = np.round(f.y.to_list()[:2] + f.y.to_list()[-2:],2)
        transformer = SeriesTransformer(f)
        for t in (
            'Deseason',
            'Detrend',
            'Log',
            'MinMax',
            'Scale',
            'RobustScale',
            'Sqrt',
            'Diff',
        ):
            f = getattr(transformer,f'{t}Transform')()
            forecaster(f)
            f = getattr(transformer,f'{t}Revert')()
            new_vals = np.round(f.y.to_list()[:2] + f.y.to_list()[-2:],2)
            comp_vals(orig_vals,new_vals,t)

        # test loess
        f = transformer.DetrendTransform(loess=True,frac=.4,it=4)
        forecaster(f)
        f = transformer.DetrendRevert()
        new_vals = np.round(f.y.to_list()[:2] + f.y.to_list()[-2:],2)
        comp_vals(orig_vals,new_vals,t)


if __name__ == '__main__':
    main()