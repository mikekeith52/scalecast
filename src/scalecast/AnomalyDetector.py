from ._utils import _developer_utils
import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

class AnomalyDetector:
    def __init__(self, f):
        self.f = f.__deepcopy__()

    def NaiveDetect(self, cilevel=0.99, **kwargs):
        """ Detects anomalies by breaking a series into its fundamental components:
        trend, seasonality, and residual. anomalies are defined as standard normal residuals
        further than a number of standard deviations away from the mean, determined by the value
        passed to cilevel. This is a simple, computationally cheap anomaly detector. Results
        are saved to the raw_anom and labeled_anom attributes.

        Args:
            cilevel (float): Default 0.99. The confidence interval used to determine how far
                away a given residual must be from the mean to be considered an anomaly. In a normal
                series that is decomposed effectively in this process, a cilevel of 0.95 would still expect
                to label 5% of its points as anomalies.
            **kwargs: Passed to the Forecaster.seasonal_decompose() method. 
                If extrapolate_trend is left unspecified, this will fail to produce results.
                See https://scalecast.readthedocs.io/en/latest/Forecaster/Forecaster.html#src.scalecast.Forecaster.Forecaster.seasonal_decompose.
        
        Returns:
            None

        >>> from scalecast.AnomalyDetector import AnomalyDetector
        >>> from scalecast.Forecaster import Forecaster
        >>> import pandas_datareader as pdr
        >>> df = pdr.get_data_fred('HOUSTNSA',start='1900-01-01',end='2021-06-01')
        >>> f = Forecaster(y=df['HOUSTNSA'],current_dates=df.index)
        >>> detector = AnomalyDetector(f)
        >>> detector.NaiveDetect(extrapolate_trend='freq',train_only=True)
        """
        resid = self.f.seasonal_decompose(**kwargs).resid
        snr = (resid - resid.mean()) / resid.std()
        anom = snr.apply(lambda x: stats.norm.cdf(x))
        labeled_anom = anom.apply(
            lambda x: 1
            if x > cilevel + (1 - cilevel) / 2 or x < (1 - cilevel) / 2
            else 0
        )

        self.y = self.f.y.to_list()
        self.fvs = None
        self.lower = None
        self.upper = None
        self.raw_anom = anom
        self.labeled_anom = labeled_anom

    def EstimatorDetect(
        self,
        estimator,
        future_dates=None,
        cilevel=0.99,
        samples=100,
        return_fitted_vals=False,
        random_seed=None,
        **kwargs,
    ):
        """ Detects anomalies with one of a Forecaster object's estimators.
        An anomaly in this instance is defined as any value that falls
        out of the fitted values' bootstrapped confidence intervals
        determined by the value passed to cilevel. This can be a good method
        to detect anomalies if you want to attempt to break down a series'
        into trends, seasonalities, and autoregressive parts in a more complex
        manner than NaiveDetect would let you. It also gives access to RNN estimators,
        which are shown to be effective anomaly detectors for time series. Results
        are saved to the labeled_anom attribute.

        Args:
            estimator (str): One of `Forecaster.estimators`.
                The estimator to track anomalies with.
            future_dates (int): Optional. If this is specified with an integer, 
                the estimator will use that number of forecast steps. If you want 
                to span an entire series for anomalies, not just the training set, 
                future dates should be created either before initiating the AnomalyDetector 
                object or by passing an int to this arg. Future dates are what signal 
                to the object that we want to train the entire dataset.
            cilevel (float): Default 0.99. The confidence interval to use when
                bootstrapping confidence intervals.
            samples (int): Default 100. How many samples in the bootstrap to find confidence intervals.
            return_fitted_vals (bool): Default False. Whether to return a DataFrame
                of the fitted values and confidence intervals from the fitting process.
            random_seed (bool): Optional. Set a seed for consistent results.
            **kwargs: Passed to the Forecaster.manual_forecast() method.

        Returns:
            (DataFrame or None): A DataFrame of fitted values if return_fitted_vals is True.

        >>> from scalecast.AnomalyDetector import AnomalyDetector
        >>> from scalecast.SeriesTransformer import SeriesTransformer
        >>> from scalecast.Forecaster import Forecaster
        >>> import pandas_datareader as pdr
        >>> df = pdr.get_data_fred('HOUSTNSA',start='1900-01-01',end='2021-06-01')
        >>> f = Forecaster(y=df['HOUSTNSA'],current_dates=df.index)
        >>> transformer = SeriesTransformer(f)
        >>> f = transformer.LogTransform()
        >>> f = transformer.DiffTransform(1)
        >>> f = transformer.DiffTransform(12)
        >>> detector = AnomalyDetector(f)
        >>> detector.EstimatorDetect(
        >>>    estimator='lstm',
        >>>    cilevel=.99,
        >>>    lags=24,
        >>>    epochs=25,
        >>>    validation_split=.2,
        >>>    shuffle=True,
        >>>    lstm_layer_sizes=(16,16,16),
        >>>    dropout=(0,0,0),
        >>> )
        """
        def _find_cis(f, samples):
            """ bootstraps the upper and lower forecast estimates using the info stored in cilevel and bootstrap_samples
            """
            y = f.y.values[-len(f.fitted_values):]
            fvs = f.fitted_values
            resids = [fv - ac for fv, ac in zip(fvs, y[-len(fvs) :])]
            bootstrapped_resids = np.random.choice(resids, size=samples)
            bootstrap_mean = np.mean(bootstrapped_resids)
            bootstrap_std = np.std(bootstrapped_resids)
            return _developer_utils._set_ci_step(
                f = f,
                s = bootstrap_std,
            ) + bootstrap_mean
        
        if random_seed is not None:
            random.seed(random_seed)
        
        f1 = self.f.__deepcopy__()
        call_me = estimator if "call_me" not in kwargs.keys() else kwargs["call_me"]

        if future_dates is None and not f1.future_dates.to_list():
            f1.generate_future_dates(1)  # because we have to have at least 1
        elif future_dates is not None:
            f1.generate_future_dates(future_dates)
        f1.set_estimator(estimator)
        f1.set_cilevel(cilevel)
        f1.manual_forecast(**kwargs)
        fvs = f1.export_fitted_vals(call_me).set_index("DATE")
        #fvs["range"] = f1.history[call_me]["CIPlusMinus"]
        fvs["range"] = _find_cis(
            f = f1,
            samples = samples,
        )
        fvs["labeled_anom"] = fvs[["Actuals", "FittedVals", "range"]].apply(
            lambda x: 1 if (x[1] > (x[0] + x[2])) | (x[1] < (x[0] - x[2])) else 0,
            axis=1,
        )
        self.y = fvs["Actuals"].to_list()
        self.fvs = fvs["FittedVals"]
        self.lower = fvs[["FittedVals", "range"]].apply(lambda x: x[0] - x[1], axis=1)
        self.upper = fvs[["FittedVals", "range"]].apply(lambda x: x[0] + x[1], axis=1)
        self.raw_anom = None
        self.labeled_anom = fvs["labeled_anom"]

        if return_fitted_vals:
            return fvs

    def MonteCarloDetect(self, start_at, stop_at, sims=100, cilevel=0.99):
        """ Detects anomalies by running a series of monte carlo simulations
        over a span of the series, using the observations before the span start 
        to determine the initial assumed distribution. Results are saved to the 
        raw_anom, labeled_anom, and mc_results attributes. It is a good idea to 
        transform the series before running so that it is stationary and not seasonal.
        In other words, the series distribution should be as close to normal as possible.
        
        Args:
            start_at (int, str, Datetime.Datetime, or pandas.Timestamp):
                If int, will start at that number obs in the series.
                Anything else should be a date-like object that can be
                parsed by the pandas.Timestamp() function, representing the
                starting point of the simulation. All observations before this
                point will be used to determine the mean/std of the intial distribution.
            stop_at (int, str, Datetime.Datetime, or pandas.Timestamp):
                If int, will stop at that number obs in the series.
                Anything else should be a date-like object that can be
                parsed by the pandas.Timestamp() function, representing the
                stopping point of the simulation.
            sims (int): The number of simulations.
            cilevel (float): Default .99.
                The percentage of points in the simulation that a given actual
                observation needs to be outside of the simulated series to be considered
                an anomaly. 

        Returns:
            None

        >>> from scalecast.AnomalyDetector import AnomalyDetector
        >>> from scalecast.SeriesTransformer import SeriesTransformer
        >>> from scalecast.Forecaster import Forecaster
        >>> import pandas_datareader as pdr
        >>> df = pdr.get_data_fred('HOUSTNSA',start='1900-01-01',end='2021-06-01')
        >>> f = Forecaster(y=df['HOUSTNSA'],current_dates=df.index)
        >>> transformer = SeriesTransformer(f)
        >>> f = transformer.LogTransform()
        >>> f = transformer.DiffTransform(1)
        >>> f = transformer.DiffTransform(12)
        >>> detector = AnomalyDetector(f)
        >>> detector.MonteCarloDetect('2010-01-01','2020-12-01',cilevel=.99)
        """
        f1 = self.f.__deepcopy__()
        if not isinstance(start_at, int):
            start_at = f1.current_dates.to_list().index(pd.Timestamp(start_at))
        if not isinstance(stop_at, int):
            stop_at = f1.current_dates.to_list().index(pd.Timestamp(stop_at))
        stop_at = stop_at + 1
        results = pd.DataFrame(
            {
                "Date": f1.current_dates.to_list()[start_at:stop_at],
                "Actuals": f1.y.to_list()[start_at:stop_at],
            },
        ).set_index("Date")
        for s in range(sims):
            obs = [i for i in f1.y.to_list()[:start_at]]
            simmed_line = []
            for i in range(stop_at - start_at):
                simmed_line.append(
                    np.random.normal(loc=np.mean(obs), scale=np.std(obs))
                )
                obs.append(simmed_line[-1])
            df_tmp = pd.DataFrame({f"Iter{s}":simmed_line},index=results.index)
            results = pd.concat([results,df_tmp],axis=1)
            #results[f"Iter{s}"] = simmed_line
        results2 = results.copy()
        results2["sims_mean"] = results.drop("Actuals", axis=1).mean(axis=1)
        results2["sims_std"] = results.drop("Actuals", axis=1).std(axis=1)
        anom = results2[["Actuals", "sims_mean", "sims_std"]].apply(
            lambda x: stats.norm.cdf(x[0], loc=x[1], scale=x[2]), axis=1,
        )
        labeled_anom = anom.apply(
            lambda x: 1
            if x > cilevel + (1 - cilevel) / 2 or x < (1 - cilevel) / 2
            else 0
        )

        self.raw_anom = anom
        self.labeled_anom = labeled_anom
        self.y = results["Actuals"].to_list()
        self.fvs = None
        self.lower = None
        self.upper = None
        self.mc_results = results.reset_index()

    def MonteCarloDetect_sliding(
        self, historical_window, step, **kwargs,
    ):
        """ Detects anomalies by running a series of monte carlo simulations
        rolling over a span of the series. It is a good idea to 
        transform the series before running so that it is stationary and not seasonal.
        In other words, the series distribution should be as close to normal as possible.

        Args:
            historical_window (int): The number of periods to begin the initial search.
            step (int): How far to step forward after a scan.
            **kwargs: Passed to the `MonteCarloDetect()` method. `start_at` and `stop_at` passed
                automatically based on the values passed to the other arguments in this function.

        Returns:
            None

        >>> from scalecast.AnomalyDetector import AnomalyDetector
        >>> from scalecast.SeriesTransformer import SeriesTransformer
        >>> from scalecast.Forecaster import Forecaster
        >>> import pandas_datareader as pdr
        >>> df = pdr.get_data_fred('HOUSTNSA',start='1900-01-01',end='2021-06-01')
        >>> f = Forecaster(y=df['HOUSTNSA'],current_dates=df.index)
        >>> transformer = SeriesTransformer(f)
        >>> f = transformer.LogTransform()
        >>> f = transformer.DiffTransform(1)
        >>> f = transformer.DiffTransform(12)
        >>> detector = AnomalyDetector(f)
        >>> detector.MonteCarloDetect_sliding(60,30)
        """
        raw_anom = pd.Series(dtype=float)
        labeled_anom = pd.Series(dtype=float)
        y = []
        n = len(self.f.y)
        y = self.f.y.to_list()[historical_window:]
        for end_idx in range(historical_window * 2, n + historical_window, step):
            print(f"scanning from obs {end_idx - historical_window} to obs {end_idx}")
            self.MonteCarloDetect(
                end_idx - historical_window, min(end_idx, n - 1), **kwargs
            )
            raw_anom = (
                pd.concat([raw_anom, self.raw_anom])
                .reset_index()
                .sort_values(["index", 0])
                .drop_duplicates(subset=["index"], keep="last")
                .set_index("index")[0]
            )
            labeled_anom = (
                pd.concat([labeled_anom, self.labeled_anom])
                .reset_index()
                .sort_values(["index", 0])
                .drop_duplicates(subset=["index"], keep="last")
                .set_index("index")[0]
            )

        self.raw_anom = raw_anom
        self.labeled_anom = labeled_anom
        self.y = y

    def WriteAnomtoXvars(self, f=None, future_dates=None, **kwargs):
        """ Writes the Xvars from the previously called anomaly detector to Xvars in
        a Forecaster object. Each anomaly is its own dummy variable on the date it is 
        found. A future distriution could detect level shifts.

        Args:
            f (Forecaster): optional. if you pass an object here,
                that object will receive the Xvars. otherwise,
                it will apply to the copy of the object stored in the
                the AnomalyDetector object when it was initialized.
                this Forecaster object is stored in the f attribute.
            future_dates (int): optional. if you pass a future dates
                length here, it will write that many dates to the
                Forecaster object and future anomaly variables will be
                passed as arrays of 0s so that any algorithm you train
                will be able to use them into future horizon.
            **kwargs: passed to the Forecaster.ingest_Xvars_df() function.
                see https://scalecast.readthedocs.io/en/latest/Forecaster/Forecaster.html#src.scalecast.Forecaster.Forecaster.ingest_Xvars_df
        Returns:
            (Forecaster) an object with the Xvars written.
        
        >>> from scalecast.AnomalyDetector import AnomalyDetector
        >>> from scalecast.SeriesTransformer import SeriesTransformer
        >>> from scalecast.Forecaster import Forecaster
        >>> import pandas_datareader as pdr
        >>> df = pdr.get_data_fred('HOUSTNSA',start='1900-01-01',end='2021-06-01')
        >>> f = Forecaster(y=df['HOUSTNSA'],current_dates=df.index)
        >>> transformer = SeriesTransformer(f)
        >>> f = transformer.LogTransform()
        >>> f = transformer.DiffTransform(1)
        >>> f = transformer.DiffTransform(12)
        >>> detector = AnomalyDetector(f)
        >>> detector.MonteCarloDetect('2010-01-01','2020-12-01',cilevel=.99)
        >>> f = detector.WriteAnomtoXvars(drop_first=True)
        """
        f = self.f if f is None else f.deepcopy()
        if future_dates is not None:
            f.generate_future_dates(future_dates)
        df = pd.DataFrame(index=f.current_dates.to_list() + f.future_dates.to_list())
        df["Anomaly"] = self.labeled_anom
        df = df.reset_index()
        df.columns = ["Date", "Anomaly"]
        df["Anomaly"] = df[["Date", "Anomaly"]].apply(
            lambda x: x[0].strftime("%Y-%m-%d") if x[1] == 1 else "No", axis=1
        )
        df["Anomaly"] = pd.Categorical(
            df["Anomaly"].to_list(),
            categories=["No"] + [c for c in df["Anomaly"].unique() if c != "No"],
            ordered=True,
        )  # all this to make "No" the first cat
        f.ingest_Xvars_df(df, date_col="Date", **kwargs)
        return f

    def adjust_anom(self, f=None, method="q", q=10):
        """ Changes the values of identified anomalies and returns a Forecaster object.

        Args: 
            f (Forecaster): Optional. If you pass an object here,
                that object will have its y values altered. Otherwise,
                it will apply to the copy of the object stored in the
                the AnomalyDetector object when it was initialized.
                this Forecaster object is stored in the f attribute.
            method (str): The following methods are supported: "q" and "interpolate".
                "q" uses q-cutting from pandas and fills values with second-to-last
                q value in either direction. For example, if q == 10, then high anomaly
                values will be replaced with the 90th percentile of the rest of the series
                data. Low anomaly values will be replaced with the 10th percentile of the 
                rest of the series. This is a good method for when your data is stationary.
                For data with a trend, 'interpolate' is better as it fills in values linearly
                based on the values before and after consecutive anomaly values. Be careful
                when using "q" with differenced data. when undifferencing,
                original values will be reverted back to.
            q (int): Default 10. 
                The q-value to use when method == 'q'.
                Ignored when method != 'q'.

        >>> from scalecast.AnomalyDetector import AnomalyDetector
        >>> from scalecast.Forecaster import Forecaster
        >>> import pandas_datareader as pdr
        >>> df = pdr.get_data_fred('HOUSTNSA',start='1900-01-01',end='2021-06-01')
        >>> f = Forecaster(y=df['HOUSTNSA'],current_dates=df.index)
        >>> detector = AnomalyDetector(f)
        >>> detector.EstimatorDetect(
        >>>    estimator='arima',
        >>>    order=(1,1,1),
        >>>    seasonal_order=(1,1,1),
        >>> )
        >>> f = detector.adjust_anom(method='interpolate')
        """
        f = self.f if f is None else f.deepcopy()
        df = pd.DataFrame({"y": f.y.to_list()}, index=f.current_dates.to_list(),)
        df["a"] = self.labeled_anom

        if method == "q":
            df["r"] = pd.qcut(df["y"], q=q, labels=list(range(q)))

            top = df.loc[df["r"] == (q - 1), "y"].min()
            bottom = df.loc[df["r"] == 0, "y"].max()

            for i, v in df.reset_index().iterrows():
                if v["a"] == 1:
                    f.y.values[i] = top if v["y"] > df["y"].mean() else bottom
        elif method == "interpolate":
            df["anom_num"] = 0
            df["a_1"] = df["a"].shift()
            df["a_-1"] = df["a"].shift(-1)
            df = df.fillna(0)
            df.loc[(df["a_-1"] == 1) & (df["a"] == 0), "anom_num"] = 1
            df["anom_num"] = df["anom_num"].cumsum()
            df.loc[
                ~(
                    ((df["a_-1"] == 1) & (df["a"] == 0))
                    | (df["a"] == 1)
                    | ((df["a_1"] == 1) & (df["a"] == 0))
                ),
                "anom_num",
            ] = 0

            for anom in df["anom_num"].unique():
                if anom == 0:
                    continue

                df_tmp = df.loc[df["anom_num"] == anom]
                idx = df_tmp.iloc[1:-1, :].index.to_list()
                slope = (df_tmp.iloc[-1, 0] - df_tmp.iloc[0, 0]) / (df_tmp.shape[0] - 1)

                for i, d in enumerate(idx):
                    df.loc[d, "y"] = df_tmp.iloc[0, 0] + slope * (i + 1)

            f.y = df["y"].reset_index(drop=True)
        else:
            raise ValueError(f'method arg expected "q" or "interpolate", got {method}')

        return f

    def plot_anom(self, label=True, strftime_fmt="%Y-%m-%d"):
        """ Plots the series used to detect anomalies and red dashes around points that
        were identified as anomalies from the last algorithm run.

        Args:
            label (bool): Default True.
                Whether to add the date label to each plotted point.
            strftime_fmt (str): Default '%Y-%m-%d'.
                The string format to convert dates to when label is True.
                When label is False, this is ignored.

        Returns:
            (Axis): The figure's axis.

        >>> from scalecast.AnomalyDetector import AnomalyDetector
        >>> from scalecast.SeriesTransformer import SeriesTransformer
        >>> from scalecast.Forecaster import Forecaster
        >>> import pandas_datareader as pdr
        >>> import matplotlib.pyplot as plt
        >>> df = pdr.get_data_fred('HOUSTNSA',start='1900-01-01',end='2021-06-01')
        >>> f = Forecaster(y=df['HOUSTNSA'],current_dates=df.index)
        >>> transformer = SeriesTransformer(f)
        >>> f = transformer.LogTransform()
        >>> f = transformer.DiffTransform(1)
        >>> f = transformer.DiffTransform(12)
        >>> detector = AnomalyDetector(f)
        >>> detector.MonteCarloDetect('2010-01-01','2020-12-01',cilevel=.99)
        >>> ax = f.plot_anom()
        >>> plt.show()
        """
        ax = self.f.plot(models=None)

        for y, a, d in zip(self.y, self.labeled_anom, self.labeled_anom.index):
            if a == 1:
                ax.scatter(d, y, marker="_", color="red", linewidths=2, label=None)
                if label:
                    ax.text(
                        d,
                        y * 1.05,
                        d.strftime(strftime_fmt),
                        color="red",
                        size=11,
                        label=None,
                    )
        if self.fvs is not None:
            ax.plot(
                self.labeled_anom.index,
                self.fvs,
                color="#FFA500",
                alpha=0.5,
                label="Fitted Vals",
            )
            ax.fill_between(
                x=self.labeled_anom.index,
                y1=self.upper,
                y2=self.lower,
                alpha=0.2,
                color="#FFA500",
                label="Conf. Interval",
            )

        ax.legend()
        return ax

    def plot_mc_results(self,ax=None,figsize=(12,6)):
        """ Plots the results from a monte-carlo detector: the series' original values
        and the simulated lines.

        Args:
            ax (Axis): Optional. An existing axis to display the figure on.
            figsize (tuple): Default (12,6). The size of the resulting figure.
                Ignored if axis is not None.

        Returns:
            (Axis): The figure's axis.

        >>> from scalecast.AnomalyDetector import AnomalyDetector
        >>> from scalecast.SeriesTransformer import SeriesTransformer
        >>> from scalecast.Forecaster import Forecaster
        >>> import pandas_datareader as pdr
        >>> import matplotlib.pyplot as plt
        >>> df = pdr.get_data_fred('HOUSTNSA',start='1900-01-01',end='2021-06-01')
        >>> f = Forecaster(y=df['HOUSTNSA'],current_dates=df.index)
        >>> transformer = SeriesTransformer(f)
        >>> f = transformer.LogTransform()
        >>> f = transformer.DiffTransform(1)
        >>> f = transformer.DiffTransform(12)
        >>> detector = AnomalyDetector(f)
        >>> detector.MonteCarloDetect('2010-01-01','2020-12-01',cilevel=.99)
        >>> ax = f.plot_mc_results()
        >>> plt.show()
        """
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
        results = self.mc_results
        sns.lineplot(x="Date", y="Actuals", data=results, label="actuals", ax=ax)
        for c in results.iloc[:, 2:]:
            ax.plot(results["Date"], results[c], alpha=0.2)
        return ax
