from scalecast.Pipeline import Pipeline, MVPipeline, Transformer, Reverter
from scalecast.util import (
    find_statistical_transformation, 
    find_optimal_transformation, 
    break_mv_forecaster,
    backtest_metrics,
) 
from test_SeriesTransformer import forecaster
from test_Forecaster import build_Forecaster
from test_MVForecaster import build_MVForecaster

def mv_forecaster(mvf):
    mvf.set_estimator('elasticnet')
    mvf.manual_forecast(lags=12,alpha=.2)
    mvf.set_estimator('ridge')
    mvf.manual_forecast(lags=12,alpha=.2)

def test_pipeline():
    f = build_Forecaster(cis=True)
    transformer, reverter = find_optimal_transformation(
        f,
        #estimator='elasticnet',
        #alpha = .2,
        num_test_sets = 2,
        space_between_sets = 24,
        train_length = 500,
        test_length = 24,
        verbose = True,
        return_train_only = True,
    )
    print(reverter)
    pipeline = Pipeline(
        steps = [
            ('Transform',transformer),
            ('Forecast',forecaster),
            ('Revert',reverter),
        ],
    )
    f = pipeline.fit_predict(f)
    backtest_results = pipeline.backtest(f)
    backtest_mets = backtest_metrics(backtest_results)
    backtest_mets.to_excel('../../uv_backtest_results.xlsx')

def test_mvpipeline():
    mvf = build_MVForecaster()
    f1, f2, f3 = break_mv_forecaster(mvf)
    transformer1, reverter1 = find_statistical_transformation(
        f1,
        goal=['stationary','seasonally_adj'],
    )
    transformer2, reverter2 = find_statistical_transformation(
        f2,
        goal=['stationary','seasonally_adj'],
    )
    transformer3, reverter3 = find_statistical_transformation(
        f3,
        goal=['stationary','seasonally_adj'],
    )
    pipeline = MVPipeline(
        steps = [
            ('Transform',[transformer1,transformer2,transformer3]),
            ('Forecast',mv_forecaster),
            ('Revert',[reverter1,reverter2,reverter3]),
        ],
        test_length = 20,
        cis = True,
    )
    f1, f2, f3 = pipeline.fit_predict(f1,f2,f3)
    backtest_results = pipeline.backtest(f1,f2,f3,n_iter=2,jump_back=12)
    backtest_mets = backtest_metrics(
        backtest_results,
        mets = ['rmse','smape','mape','r2','mae'],
        names=['UTUR','UNRATE','SAHMREALTIME'],
        mase = True,
        msis = True,
        m = 12,
    )
    backtest_mets.to_excel('../../mv_backtest_results.xlsx')

def main():
    test_pipeline()
    test_mvpipeline()

if __name__ == '__main__':
    main()