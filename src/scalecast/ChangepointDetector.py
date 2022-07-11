import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

class ChangepointDetector:
    def __init__(self,f):
        from kats.consts import TimeSeriesData
        self.f = f.__deepcopy__()
        df = pd.DataFrame(
            {
                'time':f.current_dates.to_list(),
                'value':f.y.to_list()
            }
        )
        self.df = df
        self.ts = TimeSeriesData(df)

    def DetectCPCUSUM(self,**kwargs):
        """ detects changepoints using the `CUSUMDetector.detector()` function from kats.
        this function assumes there is at most one increase change point and at most one decrease change point in the series.
        use `DetectCPCUSUM_sliding()` or `DetectCPBOCPD()` to find multiple of each kind of changepoint.
        saves output in the changepoints attribute.
        https://facebookresearch.github.io/Kats/api/kats.detectors.cusum_detection.html

        **kwargs passed to the referenced kats function.

        Returns:
            (list) A list of tuple of TimeSeriesChangePoint and CUSUMMetadata.

        >>> from scalecast.ChangepointDetector import ChangepointDetector
        >>> from scalecast.Forecaster import Forecaster
        >>> import pandas_datareader as pdr
        >>> df = pdr.get_data_fred('HOUSTNSA',start='1900-01-01',end='2021-06-01')
        >>> f = Forecaster(y=df['HOUSTNSA'],current_dates=df.index)
        >>> detector = ChangepointDetector(f)
        >>> detector.DetectCPCUSUM()
        """
        # importing here to stop read the docs from failing
        from kats.detectors.cusum_detection import CUSUMDetector
        self.detector = CUSUMDetector(self.ts)
        self.changepoints = self.detector.detector(**kwargs)
        return self.changepoints

    def DetectCPCUSUM_sliding(
            self,
            historical_window,
            scan_window,
            step,
            **kwargs,
        ):
        """ detects multiple changepoints using the `CUSUMDetector.detector()` function from kats over a sliding window.
        this idea is taken from the kats example:
        https://github.com/facebookresearch/Kats/blob/main/tutorials/kats_202_detection.ipynb

        Args:
            historical_window (int): the number of periods to begin the initial search.
            scan_window (int): how far into the future to scan for changepoints after each step.
            step (int): how far to step forward after a scan.
            **kwargs: passed to the `CUSUMDetector.detector()` function. `interest_window` passed
                automatically based on the values passed to the other arguments in this function.

        Returns:
            (list) A list of tuple of TimeSeriesChangePoint and CUSUMMetadata.

        >>> from scalecast.ChangepointDetector import ChangepointDetector
        >>> from scalecast.Forecaster import Forecaster
        >>> import pandas_datareader as pdr
        >>> df = pdr.get_data_fred('HOUSTNSA',start='1900-01-01',end='2021-06-01')
        >>> f = Forecaster(y=df['HOUSTNSA'],current_dates=df.index)
        >>> detector = ChangepointDetector(f)
        >>> detector.DetectCPCUSUM_sliding(20,10,5)
        """
        from kats.detectors.cusum_detection import CUSUMDetector
        from kats.consts import TimeSeriesData
        self.detector = CUSUMDetector(self.ts)
        self.changepoints = []
        n = self.df.shape[0]
        for end_idx in range(historical_window + scan_window, n, step):
            ts = self.df[end_idx - (historical_window + scan_window): end_idx]
            self.changepoints += CUSUMDetector(
                TimeSeriesData(ts),
            ).detector(
                interest_window=[
                    historical_window, 
                    historical_window + scan_window
                ],
                **kwargs,
            )
        return self.changepoints

    def DetectCPBOCPD(self,**kwargs):
        """ detects changepoints using the `BOCDPDetector.detector()` function from kats.
        docs: https://facebookresearch.github.io/Kats/api/kats.detectors.bocpd_model.html
        tutorial: https://github.com/facebookresearch/Kats/blob/main/tutorials/kats_202_detection.ipynb

        Args:
            **kwargs: passed to the function referenced above

        Returns:
            (list) A list of tuple of TimeSeriesChangePoint and CUSUMMetadata.

        >>> from scalecast.ChangepointDetector import ChangepointDetector
        >>> from scalecast.Forecaster import Forecaster
        >>> import pandas_datareader as pdr
        >>> df = pdr.get_data_fred('HOUSTNSA',start='1900-01-01',end='2021-06-01')
        >>> f = Forecaster(y=df['HOUSTNSA'],current_dates=df.index)
        >>> detector = ChangepointDetector(f)
        >>> detector.DetectCPBOCPD()
        """
        from kats.detectors.bocpd_model import BOCPDetector
        self.detector = BOCPDetector(self.ts)
        self.changepoints = self.detector.detector(**kwargs)
        return self.changepoints

    def plot(self):
        """ plots identified changepoints.

        >>> from scalecast.ChangepointDetector import ChangepointDetector
        >>> from scalecast.Forecaster import Forecaster
        >>> import matplotlib.pyplot as plt
        >>> import pandas_datareader as pdr
        >>> df = pdr.get_data_fred('HOUSTNSA',start='1900-01-01',end='2021-06-01')
        >>> f = Forecaster(y=df['HOUSTNSA'],current_dates=df.index)
        >>> detector = ChangepointDetector(f)
        >>> detector.DetectCPCUSUM()
        >>> detector.plot()
        >>> plt.show()
        """
        self.detector.plot(self.changepoints)

    def WriteCPtoXvars(self,f=None,future_dates=None,end=None):
        """ writes identified changepoints as variables to a Forecaster object.

        Args:
            f (Forecaster): optional. if you pass an object here,
                that object will receive the Xvars. otherwise,
                it will pass to the copy of the object stored in
                the AnomalyDetector object when it was initialized.
                this Forecaster object is stored in the f attribute.
            future_dates (int): optional. if you pass a future dates
                length here, it will write that many dates to the
                Forecaster object and future anomaly variables will be
                passed as arrays of 1s so that any algorithm you train
                will be able to use them into a future horizon.
            end (None or 'auto'): default None.
                if None, will use '2999-12-31' as the end date for each identified changepoint.
                if "auto", will use whatever changepoint end date identified by kats, but since
                this is usually the same value as the start date, the default behavior in this
                object is to use an indefinite end date ('2999-12-31').
        Returns:
            (Forecaster) an object with the Xvars written.

        >>> from scalecast.ChangepointDetector import ChangepointDetector
        >>> from scalecast.Forecaster import Forecaster
        >>> import pandas_datareader as pdr
        >>> df = pdr.get_data_fred('HOUSTNSA',start='1900-01-01',end='2021-06-01')
        >>> f = Forecaster(y=df['HOUSTNSA'],current_dates=df.index)
        >>> detector = ChangepointDetector(f)
        >>> detector.DetectCPCUSUM()
        >>> f = detector.WriteCPtoXvars(future_dates=12)
        """
        f = self.f if f is None else f.deepcopy()
        if future_dates is not None:
            f.generate_future_dates(future_dates)
        for i,cp in enumerate(self.changepoints):
            if end is None:
                f.add_other_regressor(start=cp.start_time,end='2999-12-31',called=f'cp{i+1}')
            elif end == 'auto':
                f.add_other_regressor(start=cp.start_time,end=cp.end_time,called=f'cp{i+1}')
            else:
                raise ValueError(f'end arg expected None or "auto", got {end}')
        return f
