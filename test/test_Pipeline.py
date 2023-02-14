import config
from src.scalecast.Pipeline import Pipeline, MVPipeline, Transformer, Reverter
from src.scalecast.util import find_statistical_transformation, find_optimal_transformation
from src.scalecast.util import break_mv_forecaster
from test_SeriesTransformer import forecaster
from test_Forecaster import build_Forecaster
from test_MVForecaster import build_MVForecaster

def mv_forecaster(mvf):
    mvf.set_estimator('elasticnet')
    mvf.manual_forecast(lags=12)

def test_pipeline():
    f = build_Forecaster()
    transformer, reverter = find_optimal_transformation(
        f,
        estimator='elasticnet',
    )
    pipeline = Pipeline(
        steps = [
            ('Transform',transformer),
            ('Forecast',forecaster),
            ('Revert',reverter),
        ],
    )
    f = pipeline.fit_predict(f)

def test_mvpipeline():
    mvf = build_MVForecaster()
    f1, f2, f3 = break_mv_forecaster(mvf)
    transformer1, reverter1 = find_statistical_transformation(
        f1,
        goal=['stationary','seasonally_adj'],
        log = True if min(f1.y) > 0 else False,
    )
    transformer2, reverter2 = find_statistical_transformation(
        f2,
        goal=['stationary','seasonally_adj'],
        log = True if min(f2.y) > 0 else False,
    )
    transformer3, reverter3 = find_statistical_transformation(
        f3,
        goal=['stationary','seasonally_adj'],
        log = True if min(f3.y) > 0 else False,
    )
    pipeline = MVPipeline(
        steps = [
            ('Transform',[transformer1,transformer2,transformer3]),
            ('Forecast',mv_forecaster),
            ('Revert',[reverter1,reverter2,reverter3]),
        ],
        test_length = 12,
    )
    f1, f2, f3 = pipeline.fit_predict(f1,f2,f3)

def main():
    test_pipeline()
    test_mvpipeline()

if __name__ == '__main__':
    main()