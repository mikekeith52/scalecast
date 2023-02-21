from scalecast.ChangepointDetector import ChangepointDetector
from test_Forecaster import build_Forecaster
import matplotlib.pyplot as plt

def main():
    for tl in (0,24):
        print(tl)
        f = build_Forecaster(test_length = tl)
        detector = ChangepointDetector(f)
        detector.DetectCPCUSUM(return_all_changepoints=True)
        detector.DetectCPCUSUM_sliding(60,60,60)
        detector.plot()
        plt.savefig(f'../../cp_{tl}.png')
        plt.close()

        # dependency hell error - I think I'll delete this method from the package soon
        detector.DetectCPBOCPD()
        plt.savefig(f'../../cpbocpd_{tl}.png')
        plt.close()

        f = detector.WriteCPtoXvars(f)

if __name__ == '__main__':
    main()