import pandas_datareader as pdr
from scalecast.Forecaster import Forecaster
from scalecast.auxmodels import mlp_stack, auto_arima
from scalecast.util import plot_reduction_errors, metrics
import matplotlib.pyplot as plt
import pickle

df = pdr.get_data_fred(
    'HOUSTNSA',
    start = '1959-01-01',
    end = '2022-12-31',
)

def rmse_mae(a, f):
    return (metrics.rmse(a,f) + metrics.mae(a,f)) / 2

def build_Forecaster(
    cis = False, 
    test_length = 48,
    **kwargs,
):
    global df
    return Forecaster(
        y = df['HOUSTNSA'],
        current_dates = df.index,
        future_dates = 24,
        test_length = test_length,
        cis = cis,
        metrics = [
            'abias',
            'rmse',
            'smape',
            'mse',
        ],
        **kwargs,
    )

def test_add_terms():
    f = build_Forecaster()
    f
    f.add_AR_terms((2,12))
    assert 'AR24' in f.get_regressor_names(), 'regressor AR24 not added'

    f.add_ar_terms(12)
    assert 'AR12' in f.get_regressor_names(), 'regressor AR12 not added'

    f.add_time_trend()
    assert 't' in f.get_regressor_names(), 'regressor t not added'

    f.add_covid19_regressor()
    assert 'COVID19' in f.get_regressor_names(), 'regressor COVID19 not added'

    f.add_other_regressor(called='other',start='2021-01-01',end='2021-06-30')
    assert 'other' in f.get_regressor_names(), 'regressor other not added'

    f.add_combo_regressors('t','COVID19')
    assert 't_COVID19' in f.get_regressor_names(), 'regressor t_COVID19 not added'

    f.add_lagged_terms('t')
    assert 'tlag_1' in f.get_regressor_names(), 'regressor t_lag1 not added'

    f.add_logged_terms('t')
    assert 'lnt' in f.get_regressor_names(), 'regressor lnt not added'

    f.add_logged_terms('t',base=10)
    assert 'log10t' in f.get_regressor_names(), 'regressor log10t not added'

    f.add_pt_terms('t')
    assert 'box-cox_t' in f.get_regressor_names(), 'regressor cox_t not added'

    f.add_seasonal_regressors('month',sincos=True,dummy=True,cycle_lens={'month':12})
    assert 'month' in f.get_regressor_names(), 'regressor month not added'
    assert 'monthsin' in f.get_regressor_names(), 'regressor monthsin not added'
    assert 'month_12' in f.get_regressor_names(), 'regressor month_12 not added'

    f.add_poly_terms('t',pwr=3)
    assert 't^3' in f.get_regressor_names(), 'regressor t^3 not added'

    f.add_exp_terms('t',pwr=.509)
    assert 't^0.51' in f.get_regressor_names(), 'regressor t^0.51 not added'

def test_feature_selection_reduction():
    f = build_Forecaster(test_length = 0)
    f.set_grids_file('ExampleGrids')
    f.auto_Xvar_select(estimator='elasticnet')
    f.reduce_Xvars(estimator='elasticnet',method='pfi')

    f.auto_Xvar_select(estimator='xgboost')
    f.reduce_Xvars(estimator='xgboost',method='shap')

    plot_reduction_errors(f)
    plt.savefig('../../reduction_errors.png')
    plt.close()

    f.auto_Xvar_select(estimator='lasso')
    f.reduce_Xvars(method='l1')

def test_pickle():
    f = build_Forecaster()
    with open('../../f.pckl','wb') as pckl:
        pickle.dump(f,pckl)

def test_statistical_tests():
    f = build_Forecaster()
    f.adf_test()
    f.normality_test()
    f.adf_test(diffy=True)
    f.normality_test(diffy=True)

def test_modeling():
    for tl in (0,48): # make sure 0 and non-0 length test sets work
        f = build_Forecaster(test_length=tl)
        f.set_metrics(['rmse','smape',rmse_mae])
        f.set_grids_file('ExampleGrids')
        f.add_metric(rmse_mae)
        f.set_validation_metric('rmse_mae')
        f.set_validation_length(12)
        if tl != 0: 
            f.eval_cis(cilevel=.9)
        f.auto_Xvar_select(
            estimator = 'ridge',
            alpha = 0.2,
            decomp_trend = False,
            monitor = 'ValidationMetricValue' if tl == 0 else 'TestSetRMSE'
        )

        f.set_estimator('lstm')
        f.ingest_grid('lstm')
        f.limit_grid_size(1)
        f.cross_validate() # 0.18.7

        models = (
            'elasticnet',
            'prophet',
            'silverkite',
            'theta',
            'gbt',
            'catboost',
            'arima',
            'hwes',
            'tbats',
            'rnn',
            'lstm',
        )

        f.tune_test_forecast(
            models,
            cross_validate = True,
            rolling = True,
            k = 2,
            test_length = 36,
            train_length = 500,
            space_between_sets = 12,
            dynamic_tuning = 24,
            dynamic_testing = 24,
            feature_importance = True,
            summary_stats = True,
            suffix = '_cv',
            limit_grid_size = .2,
            min_grid_size = 4,
            verbose = True,
            error = 'raise',
        )

        f.set_estimator('mlr')
        f.add_signals(['lstm_cv'],fill_strategy = 'bfill')
        f.manual_forecast()
        f.add_signals(['lstm_cv'],fill_strategy = None)
        f.add_signals(['lstm_cv'],train_only=tl > 0)

        f.set_estimator('combo')
        f.manual_forecast()
        f.manual_forecast(
            how='weighted',
            models='top_3',
            determine_best_by='ValidationMetricValue' if tl == 0 else 'TestSetRMSE',
            call_me = 'weighted',
        )

        f.set_estimator('naive')
        f.manual_forecast()
        f.manual_forecast(seasonal=True,call_me='snaive')

        mlp_stack(f,model_nicknames=['gbt_cv','elasticnet_cv'])
        auto_arima(f,m=12)

        best_model = f.order_fcsts(
            determine_best_by='ValidationMetricValue' if tl == 0 else 'TestSetSMAPE',
        )[0]

        f.plot(ci=True,exclude = ['mlr'])
        plt.savefig(f'../../plot_{tl}.png')
        plt.close()

        if tl != 0:
            f.plot_test_set(ci=True,exclude = ['mlr'],include_train=96)
            plt.savefig(f'../../plot_ts_{tl}.png')
            plt.close()

        f.plot_fitted()
        plt.savefig(f'../../flot_fvs_{tl}')
        plt.close()

        if tl != 0:
            f.export(to_excel=True,out_path='../..',excel_name=f'results_{tl}.xlsx',cis=True)
            f.export_fitted_vals(model=best_model).to_excel('../../fvs.xlsx',index=False)
            f.all_feature_info_to_excel(out_path='../..')
            f.export_Xvars_df().to_excel('../../Xvars.xlsx',index=False)
            f.export_validation_grid(models[0] + '_cv').to_excel('../../validation_grid.xlsx',index=False)
            with open('../../f.pckl','wb') as pckl:
                pickle.dump(f,pckl)

def main():
    test_add_terms()
    test_feature_selection_reduction()
    test_pickle()
    test_modeling()

if __name__ == '__main__':
    main()