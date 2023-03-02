from scalecast.Forecaster import Forecaster
from scalecast.MVForecaster import MVForecaster
from scalecast.util import break_mv_forecaster, find_optimal_lag_order, find_optimal_coint_rank
from scalecast.auxmodels import vecm, mlp_stack
import pandas_datareader as pdr
import matplotlib.pyplot as plt
import pickle

def build_MVForecaster(test_length=24):
    s1 = pdr.get_data_fred('UTUR',start='2000-01-01',end='2022-10-01')
    s2 = pdr.get_data_fred('UNRATE',start='2000-01-01',end='2022-10-01')
    s3 = pdr.get_data_fred('SAHMREALTIME',start='2000-01-01',end='2022-10-01')

    f1 = Forecaster(y=s1['UTUR'],current_dates=s1.index,future_dates=24)
    f2 = Forecaster(y=s2['UNRATE'],current_dates=s2.index,future_dates=24)
    f3 = Forecaster(y=s3['SAHMREALTIME'],current_dates=s3.index,future_dates=24)

    return MVForecaster(
        f1,
        f2,
        f3,
        test_length = test_length,
        names=['UTUR','UNRATE','SAHMREALTIME'],
        merge_Xvars='i',
        metrics = [
            'rmse',
            'smape',
            'mse',
        ],
    )

def weighted_series(x):
    return x[0]*.75+x[1]*.25

def test_pickle():
    mvf = build_MVForecaster()
    mvf.add_optimizer_func(weighted_series,'weighted')
    mvf.set_optimize_on('weighted')
    with open('../../mvf.pckl','wb') as pckl:
        pickle.dump(mvf,pckl)

def test_modeling():
    for tl in (0,36):
        mvf = build_MVForecaster(test_length = tl)
        mvf.set_metrics(['rmse','r2'])
        mvf
        if tl > 0:
            mvf.eval_cis(
                cilevel = .9,
            )

        models = ('lasso','xgboost')

        mvf.corr_lags('UNRATE','UTUR',lags=5)
        mvf.corr_lags('UNRATE','UTUR',lags=5,disp='heatmap',annot=True,vmin=-1,vmax=1)
        plt.savefig('../../corr_lags.png')
        plt.close()
        mvf.add_sklearn_estimator(vecm,'vecm')
        find_optimal_lag_order(mvf)
        find_optimal_coint_rank(mvf,det_order=-1,k_ar_diff=8)
        mvf.tune_test_forecast(
            models,
            limit_grid_size=.2,
            cross_validate = True,
            rolling = True,
            k = 2,
            dynamic_tuning = 24,
            error = 'warn',
            suffix = '_cv',
        )
        mvf.set_estimator('vecm')
        mvf.set_grids_file('VECMGrid')
        mvf.cross_validate(k=2)
        mvf.auto_forecast()

        if tl > 0:
            mlp_stack(
                mvf,
                model_nicknames=['xgboost_cv','lasso_cv'],
                lags = 6,
            )

        mvf.set_best_model(
            determine_best_by = (
                'ValidationMetricValue' if tl == 0 
                else 'TestSetRMSE'
            )
        )

        mvf.plot(ci=True)
        plt.savefig(f'../../mvf_plot_{tl}.png')
        plt.close()

        if tl > 0:
            mvf.plot_test_set(ci=True)
            plt.savefig('../../mvf_plot_ts.png')
            plt.close()

        mvf.plot_fitted()
        plt.savefig(f'../../mvf_fvs_{tl}.png')
        plt.close()

        mvf.export(to_excel=True,out_path='../..',excel_name=f'mv_results_{tl}.xlsx',cis=True)
        mvf.export_fitted_vals().to_excel('../../mv_fvs.xlsx',index=False)
        with open('../../mvf.pckl','wb') as pckl:
            pickle.dump(mvf,pckl)

        f1, f2, f3 = break_mv_forecaster(mvf)

def main():
    test_pickle()
    test_modeling()

if __name__ == '__main__':
    main()