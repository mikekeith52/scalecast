from scalecast import Forecaster
from scalecast import MVForecaster
from scalecast import SeriesTransformer
from scalecast.auxmodels import auto_arima
import scalecast.Pipeline
import pandas_datareader as pdr
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

class metrics:
    def mape(a,f):
        """ mean absolute percentage error (MAPE).

        Args:
            a (list-like): the actuals over the forecast horizon.
            f (list-like): the predictions over the forecast horizon.

        Returns:
            (float): the derived MAPE.

        >>> from scalecast.util import metrics
        >>> a = [1,2,3,4,5]
        >>> f = [1,2,3,4,6]
        >>> metrics.mape(a,f)
        """
        return Forecaster.mape(a,f)

    def r2(a,f):
        """ r-squared (R2).

        Args:
            a (list-like): the actuals over the forecast horizon.
            f (list-like): the predictions over the forecast horizon.

        Returns:
            (float): the derived R2.

        >>> from scalecast.util import metrics
        >>> a = [1,2,3,4,5]
        >>> f = [1,2,3,4,6]
        >>> metrics.r2(a,f)
        """
        return Forecaster.r2(a,f)

    def mse(a,f):
        """ mean squared error (MSE).

        Args:
            a (list-like): the actuals over the forecast horizon.
            f (list-like): the predictions over the forecast horizon.

        Returns:
            (float): the derived MSE.

        >>> from scalecast.util import metrics
        >>> a = [1,2,3,4,5]
        >>> f = [1,2,3,4,6]
        >>> metrics.mse(a,f)
        """
        return Forecaster.rmse(a,f)**2

    def rmse(a,f):
        """ root mean squared error (RMSE).

        Args:
            a (list-like): the actuals over the forecast horizon.
            f (list-like): the predictions over the forecast horizon.

        Returns:
            (float): the derived RMSE.

        >>> from scalecast.util import metrics
        >>> a = [1,2,3,4,5]
        >>> f = [1,2,3,4,6]
        >>> metrics.rmse(a,f)
        """
        return Forecaster.rmse(a,f)

    def mae(a,f):
        """ mean absolute error (MAE).

        Args:
            a (list-like): the actuals over the forecast horizon.
            f (list-like): the predictions over the forecast horizon.

        Returns:
            (float): the derived MAE.

        >>> from scalecast.util import metrics
        >>> a = [1,2,3,4,5]
        >>> f = [1,2,3,4,6]
        >>> metrics.mae(a,f)
        """
        return Forecaster.mae(a,f)

    def smape(a,f):
        """ symmetric mean absolute percentage error (sMAPE).
        uses the same definition as used in M4 competition.
        does not multiply by 100.
        see https://ideas.repec.org/a/eee/intfor/v36y2020i1p54-74.html.

        Args:
            a (list-like): the actuals over the forecast horizon.
            f (list-like): the predictions over the forecast horizon.

        Returns:
            (float): the derived sMAPE.

        >>> from scalecast.util import metrics
        >>> a = [1,2,3,4,5]
        >>> f = [1,2,3,4,6]
        >>> metrics.smape(a,f)
        """
        a = np.array(a)
        f = np.array(f)
        return (
            1/len(a) *
            np.sum(
                2*np.abs(f-a) / (
                    np.abs(a) + np.abs(f)
                )
            )
        )

    def mase(a,f,obs,m):
        """ mean absolute scaled error (MASE).
        uses the same definition as used in M4 competition.
        see https://ideas.repec.org/a/eee/intfor/v36y2020i1p54-74.html.

        Args:
            a (list-like): the actuals over the forecast horizon.
            f (list-like): the predictions over the forecast horizon.
            obs (list-like): the actual observations used to create the forecast.
            m (int): the seasonal period.

        Returns:
            (float): the derived MASE.

        >>> from scalecast.util import metrics
        >>> a = [1,2,3,4,5]
        >>> f = [1,2,3,4,6]
        >>> obs = [-5,-4,-3,-2,-1,0]
        >>> metrics.mase(a,f,obs,1)
        """
        a = np.array(a)
        f = np.array(f)
        avger = 1/len(a)
        num = np.sum(np.abs(f-a))
        davger = 1 / (len(obs) - m)
        denom = np.sum(
            np.abs(
                pd.Series(obs).diff(m).values[m:]
            )
        )
        return avger * (num / (davger * denom))

    def msis(a,uf,lf,obs,m,alpha=0.05):
        """ mean scaled interval score (MSIS) for evaluating confidence intervals.
        uses the same definition as used in M4 competition.
        lower values are better.
        see https://ideas.repec.org/a/eee/intfor/v36y2020i1p54-74.html.

        Args:
            a (list-like): the actuals over the forecast horizon.
            uf (list-like): the upper-forecast bound according to the confidence interval.
            lf (list-like): the lower-forecast bound according to the confidence interval.
            obs (list-like): the actual observations used to create the forecast.
            m (int): the seasonal period.
            alpha (float): default 0.05. 0.05 for 95% confidence intervals, etc.

        Returns:
            (float): the derived MSIS.

        >>> from scalecast.util import metrics
        >>> a = [1,2,3,4,5]
        >>> f = [1,2,3,4,6]
        >>> uf = [1.5,2.5,3.5,4.5,6.5]
        >>> lf = [.5,1.5,2.5,3.5,5.5]
        >>> obs = [-5,-4,-3,-2,-1,0]
        >>> metrics.msis(a,uf,lf,obs,1) # returns a value of 5.0
        """
        a = np.array(a)
        uf = np.array(uf)
        lf = np.array(lf)
        avger = 1/len(a)
        num1 = uf-lf
        num2 = np.array([(
            (2/alpha*(lfs-acs)) if lfs > acs else
            (2/alpha*(acs-ufs)) if acs > ufs else
            0
        ) for acs, ufs, lfs in zip(a,uf,lf)])
        davger = 1/(len(obs) - m)
        denom = np.sum(np.abs(pd.Series(obs).diff(m).values[m:]))
        return avger * (np.sum(num1 + num2) / (davger * denom))

def pdr_load(
    sym,
    start=None,
    end=None,
    src='fred',
    require_future_dates=True,
    future_dates=None,
    integrate=False,
    diff=False,
    MVForecaster_kwargs={},
    **kwargs
):
    """ gets data using `pandas_datareader.DataReader()` and loads the series into a Forecaster or MVForecaster object.
    this functions works well when the src arg is its default ('fred'), but there are some issues with other sources.

    Args:
        sym (str or list-like): the name of the series to extract.
            if str (one series), returns a Forecaster object.
            if list-like, returns an MVForecaster object. 
            series of higher frequencies will having missing values filled using a forward fill.
        start (str or datetime): the start date to extract data.
        end (str or datetime): the end date to extract data.
        src (str): the source of the API pull.
            supported values: 'fred', 'yahoo', 'alphavantage', 'enigma', 
            'famafrench','moex', 'quandl', 'stooq', 'tiingo'.
        require_future_dates (bool): default True.
            if False, none of the models from the resulting Forecaster object 
            will forecast into future periods by default.
            if True, all models will forecast into future periods, 
            unless run with test_only = True, and when adding regressors, they will automatically
            be added into future periods.
            will be ignored if sym is list-like.
        future_dates (int): optional. the future dates to add to the model upon initialization.
            if not added when object is initialized, can be added later.
        integrate (bool): default False. whether to take first differences in extraced data if it is found to be non-stationary.
        diff (bool): default False. whether to take first differences in extracted data 
            without running the ADF test to check stationarity.
        MVForecaster_kwargs (dict): default {}. if sym is list-like, 
            these arguments will be passed to the `MVForecaster()` init function.
            if 'names' is not found in the dict, names are automatically added so that the
            MVForecaster keeps the names of the extracted symbols.
            to keep no names, pass `MVForecaster_kwargs = {'names':None,...}`.
        **kwargs: passed to pdr.DataReader() function. 
            see https://pandas-datareader.readthedocs.io/en/latest/remote_data.html.

    Returns:
        (Forecaster or MVForecaster): an object with the dates and y-values loaded.
    """
    df = pdr.DataReader(sym,data_source=src,start=start,end=end,**kwargs)
    if isinstance(sym,str):
        f = Forecaster.Forecaster(
            y=df[sym],
            current_dates=df.index,
            require_future_dates=require_future_dates,
            future_dates = future_dates,
        )
        if integrate:
            f.integrate()
        elif diff:
            f.diff()
        return f
    else:
        fs = []
        for s in sym:
            df[s].fillna(method='ffill',inplace=True)
            f = Forecaster.Forecaster(
                y = df[s],
                current_dates=df.index,
                future_dates = future_dates,
            )
            if integrate:
                f.integrate()
            elif diff:
                f.diff()
            fs.append(f)
        if 'names' not in MVForecaster_kwargs:
            MVForecaster_kwargs['names'] = sym
        return MVForecaster.MVForecaster(
            *fs,
            **MVForecaster_kwargs,
        )

def plot_reduction_errors(f):
    """ plots the resulting error/accuracy of a Forecaster object where `reduce_Xvars()` method has been called
    with method = 'pfi'.
    
    Args:
        f (Forecaster): an object that has called the `reduce_Xvars()` method with method = 'pfi'.
        
    Returns:
        (Axis) the figure's axis.
    """
    dropped = f.pfi_dropped_vars
    errors = f.pfi_error_values
    _, ax = plt.subplots()
    sns.lineplot(
        x=np.arange(0, len(dropped) + 1, 1), y=errors,
    )
    plt.xlabel("dropped Xvars")
    plt.ylabel("error")
    return ax


def break_mv_forecaster(mvf):
    """ breaks apart an MVForecaster object and returns as many Foreaster objects as series loaded into the object.

    Args:
        mvf (MVForecaster): the object to break apart.

    Returns:
        (tuple): a sequence of at least two Forecaster objects
    """

    def convert_mv_hist(f, mvhist: dict, series_num: int):
        hist = {}
        for k, v in mvhist.items():
            hist[k] = {}
            for k2, v2 in v.items():
                if k2 in (""):
                    continue
                elif not isinstance(v2, dict) or k2 == "HyperParams":
                    hist[k][k2] = v2
                elif isinstance(v2, dict):
                    hist[k][k2] = list(v2.values())[series_num]
            hist[k]["TestOnly"] = False
        return hist

    mvf1 = mvf.deepcopy()

    set_len = (
        len(mvf1.series1['y']) 
        if not mvf1.current_xreg 
        else len(list(mvf1.current_xreg.values())[0])
    )
    to_return = []
    for s in range(mvf1.n_series):
        f = Forecaster.Forecaster(
            y=getattr(mvf1, f"series{s+1}")["y"].values[-set_len:],
            current_dates=mvf1.current_dates.values[-set_len:],
            integration=getattr(mvf1, f"series{s+1}")["integration"],
            levely=getattr(mvf1, f"series{s+1}")["levely"][-set_len:],
            init_dates=getattr(mvf1, f"series{s+1}")["init_dates"][-set_len:],
            future_dates=len(mvf1.future_dates),
            current_xreg={k:v.copy() for k,v in mvf1.current_xreg.items()},
            future_xreg={k:v.copy() for k,v in mvf1.future_xreg.items()},
            test_length=mvf1.test_length,
            validation_length=mvf1.validation_length,
        )
        f.history = convert_mv_hist(f, mvf1.history, s)
        to_return.append(f)

    return tuple(to_return)

def find_optimal_lag_order(mvf,train_only=False,**kwargs):
    """ returns the otpimal lag order for a mutlivariate process using the statsmodels function:
    https://www.statsmodels.org/dev/generated/statsmodels.tsa.vector_ar.var_model.VAR.select_order.html.
    the exogenous regressors are set based on Xvars loaded in the `MVForecaster` object.

    Args:
        mvf (MVForecaster): the MVForecaster object with series loaded to find the optimal order for
        train_only (bool): default False. whether to use the training data only in the test.
        **kwargs: passed to the referenced statsmodels function

    Returns:
        (LagOrderResults): lag selections.
    """
    from statsmodels.tsa.vector_ar.var_model import VAR
    data = np.array(
        [getattr(mvf,f'series{i+1}')['y'].astype(float).values for i in range(mvf.n_series)],
    ).T

    if mvf.current_xreg:
        exog = pd.DataFrame(mvf.current_xreg).values
    else:
        exog = None

    if train_only:
        data = data[:-mvf.test_length]
        if exog is not None:
            exog = exog[:-mvf.test_length]

    model = VAR(data,exog=exog)

    return model.select_order(
        **kwargs,
    )

def find_optimal_coint_rank(mvf,det_order,k_ar_diff,train_only=False,**kwargs):
    """ returns the optimal cointigration rank for a multivariate process using the function from statsmodels: 
    https://www.statsmodels.org/dev/generated/statsmodels.tsa.vector_ar.vecm.select_coint_rank.html

    Args:
       mvf (MVForecaster): the MVForecaster object with series loaded to find the optimal rank for
       train_only (bool): default False. whether to use the training data only in the test.
        **kwargs: passed to the referenced statsmodels function

    Returns:
        (CointRankResults): object containing the cointegration rank suggested by the test and allowing a summary to be printed.
    """
    from statsmodels.tsa.vector_ar.vecm import select_coint_rank
    data = np.array(
        [getattr(mvf,f'series{i+1}')['y'].values for i in range(mvf.n_series)],
    ).T
    if train_only:
        data = data[:-mvf.test_length]
    
    return select_coint_rank(
        data,
        det_order=det_order,
        k_ar_diff=k_ar_diff,
        **kwargs,
    )

def find_series_transformation(
    f,
    goal=['stationary'],
    train_only=False,
    critical_pval=.05,
    log = True,
    m='auto',
    adf_kwargs = {},
    **kwargs,
):
    """ finds an optimal set of transformations to achieve some passed goal with the end result that forecasts are more accurate.

    Args:
        f (Forecaster): the object that stores the series to test.
        goal (list-like): one or multiple of 'stationary', 'seasonally_adj'. other options may be coming in the future.
            if more than one goal is passed, will try to satisfy all goals in the order they passed.
            for stationary: uses an Augmented Dickey-Fuller test to determine if the series is stationary.
            if not stationary, returns a diff transformation and log transformation if log is True.
            for seasonall_adj: uses seasonal auto_arima to find the optimal seasonal diff.
        train_only (bool): default False. whether to use train set only in all statistical tests.
        log (bool): default True. whether to log and diff the series if it is found to be non-stationary or just diff.
        critical_pval (float): default 0.05. the cutoff p-value to use to determine statistical signficance in the 
            Augmented Dickey-Fuller test and to run the auto_arima selection (substitutes for `alpha` arg).
        m (str or int): default 'auto': the time-steps in the data that count one seasonal step.
            uses the M4 competition values. for Hourly: 24, Monthly: 12, Quarterly: 4. everything else gets 1 (no seasonality assumed)
            so pass your own values for other frequencies.
        adf_kwargs (dict): default {}. keyword args to pass to the Augmented Dickey-Fuller test function. 
        **kwargs: passed to the auto_arima() function when searching for optimal seasonal diff.

    Returns:
        (Transformer, Reverter): a `Transformer` object with the identified transforming functions and
        the `Reverter` object with the `Transformer` counterpart functions.
    """
    def make_stationary(f,train_only,critical_pval,log,adf_kwargs,**kwargs):
        transformers = []
        stationary = f.adf_test(
            train_only=train_only,
            full_res=False,
            critical_pval=critical_pval,
            **adf_kwargs
        )
        if not stationary:
            if log and f.y.min() > 0:
                transformers += [('LogTransform',)]
            transformers += [('DiffTransform',1)]  
        return transformers

    def seasonally_adj(f,train_only,critical_pval,log,adf_kwargs,**kwargs):
        transformers = []
        if m == 1:
            return transformers

        auto_arima(f,m=m,seasonal=True,alpha=critical_pval,**kwargs)
        I = f.auto_arima_params['seasonal_order'][1]
        for i in range(1,I+1):
            transformers += [('DiffTransform',m)]

        return transformers

    f = f.deepcopy()
    transformer = SeriesTransformer.SeriesTransformer(f)

    if m == 'auto':
        if f.freq is not None:
            if f.freq.startswith('M'):
                m = 12
            elif f.freq.startswith('Q'):
                m = 4
            elif f.freq.startswith('H'):
                m = 24
            else:
                m = 1
        else:
            m = 1

    possible_args = {
        'stationary':make_stationary,
        'seasonally_adj':seasonally_adj,
    }
    bad_args = [g for g in goal if g not in possible_args]
    if len(bad_args) > 0:
        raise ValueError(f'values passed to goal arg cannot be used: {bad_args}')

    transformers = []
    reverters = []
    for g in goal:
        t = possible_args[g](
            f,
            train_only=train_only,
            critical_pval=critical_pval,
            log=log,
            adf_kwargs=adf_kwargs,
            **kwargs
        )
        transformers += t
    for t in transformers[::-1]:
        r = (t[0].replace('Transform','Revert'),)
        if len(t) > 1:
            r += t[1:]
        reverters.append(r)

    final_transformer = Pipeline.Transformer(transformers = transformers)
    final_reverter = Pipeline.Reverter(
        reverters = reverters, 
        base_transformer = final_transformer
    )
    return final_transformer, final_reverter
