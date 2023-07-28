from test_Forecaster import build_Forecaster
from scalecast.SeriesTransformer import SeriesTransformer
import numpy as np

def forecaster(f):
    f.add_ar_terms(12)
    f.set_estimator('elasticnet')
    f.manual_forecast(alpha=.2)

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