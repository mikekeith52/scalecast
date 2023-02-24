import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

class ChangepointDetector:
    def __init__(self, f):
        from kats.consts import TimeSeriesData

        self.f = f.__deepcopy__()
        df = pd.DataFrame({"time": f.current_dates.to_list(), "value": f.y.to_list()})
        self.df = df
        self.ts = TimeSeriesData(df)

    def DetectCPCUSUM(self, **kwargs):
        """ Detects changepoints using the `CUSUMDetector.detector()` function from kats.
        This function assumes there is at most one increase change point and at most one decrease change point in the series.
        Use `DetectCPCUSUM_sliding()` or `DetectCPBOCPD()` to find multiple of each kind of changepoint.
        Saves output in the changepoints attribute.
        See https://facebookresearch.github.io/Kats/api/kats.detectors.cusum_detection.html.

        Args:
            **kwargs: Passed to the referenced kats function.

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
        self, historical_window, scan_window, step, **kwargs,
    ):
        """ Detects multiple changepoints using the `CUSUMDetector.detector()` function from kats over a sliding window.
        This idea is taken from the kats example:
        https://github.com/facebookresearch/Kats/blob/main/tutorials/kats_202_detection.ipynb.

        Args:
            historical_window (int): The number of periods to begin the initial search.
            scan_window (int): How far into the future to scan for changepoints after each step.
            step (int): How far to step forward after a scan.
            **kwargs: Passed to the `CUSUMDetector.detector()` function. `interest_window` passed
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
            ts = self.df[end_idx - (historical_window + scan_window) : end_idx]
            self.changepoints += CUSUMDetector(TimeSeriesData(ts),).detector(
                interest_window=[historical_window, historical_window + scan_window],
                **kwargs,
            )
        return self.changepoints

    def DetectCPBOCPD(self, **kwargs):
        """ Detects changepoints using the `BOCDPDetector.detector()` function from kats.
        Docs: https://facebookresearch.github.io/Kats/api/kats.detectors.bocpd_model.html.
        Tutorial: https://github.com/facebookresearch/Kats/blob/main/tutorials/kats_202_detection.ipynb.

        Args:
            **kwargs: Passed to the function referenced above

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
        """ Plots identified changepoints.

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

    def WriteCPtoXvars(self, f=None, future_dates=None, end=None):
        """ Writes identified changepoints as variables to a Forecaster object.

        Args:
            f (Forecaster): Optional. If you pass an object here,
                that object will receive the Xvars. Otherwise,
                it will pass to the copy of the object stored in
                the AnomalyDetector object when it was initialized.
                This Forecaster object is stored in the f attribute.
            future_dates (int): Optional. If you pass a future dates
                length here, it will write that many dates to the
                Forecaster object and future anomaly variables will be
                passed as arrays of 1s so that any algorithm you train
                will be able to use them into a future horizon.
            end (None or 'auto'): Default None.
                If None, will use '2999-12-31' as the end date for each identified changepoint.
                If "auto", will use whatever changepoint end date identified by kats, but since
                this is usually the same value as the start date, the default behavior in this
                object is to use an indefinite end date ('2999-12-31').
        Returns:
            (Forecaster) An object with the Xvars written.

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
        for i, cp in enumerate(self.changepoints):
            if end is None:
                f.add_other_regressor(
                    start=cp.start_time, end=f.future_dates.values[-1], called=f"cp{i+1}"
                )
            elif end == "auto":
                f.add_other_regressor(
                    start=cp.start_time, end=cp.end_time, called=f"cp{i+1}"
                )
            else:
                raise ValueError(f'end arg expected None or "auto", got {end}')
        return f
