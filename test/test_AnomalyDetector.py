from scalecast.AnomalyDetector import AnomalyDetector
from test_Forecaster import build_Forecaster
import matplotlib.pyplot as plt

def main():
    f = build_Forecaster(test_length = 0)
    detector = AnomalyDetector(f)
    detector.MonteCarloDetect('2010-01-01','2021-06-01')
    detector.plot_mc_results()
    plt.savefig('../../mc_anom.png')
    plt.close()
    detector.MonteCarloDetect_sliding(60,30)
    detector.labeled_anom
    detector.plot_anom(label=True)
    plt.savefig('../../mc_sliding_anom.png')
    plt.close()
    f = detector.adjust_anom(f=f,method='q')
    detector.EstimatorDetect(
        estimator='lstm',
        cilevel=.99,
        test_only=False,
        lags=24,
        epochs=25,
        validation_split=.2,
        shuffle=True,
        lstm_layer_sizes=(16,16,16),
        dropout=(0,0,0),
    )
    detector.plot_anom(label=True)
    plt.savefig('../../lstm_anom.png')
    plt.close()
    f = detector.WriteAnomtoXvars(f=f,drop_first=True)

if __name__ == '__main__':
    main()