import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import warnings
import random
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_percentage_error,
    mean_absolute_error,
)
from ._utils import _developer_utils

class metrics:
    @staticmethod
    def bias(a,f):
        """ Returns the total bias over a given forecast horizon. 
        When this is larger than 0, means aggregated predicted points are higher than actuals.
        Divide by the length of the forecast horizon to get average bias.

        Args:
            a (list-like): The actuals over the forecast horizon.
            f (list-like): The predictions over the forecast horizon.

        Returns:
            (float): The derived bias.

        >>> from scalecast.util import metrics
        >>> a = [1,2,3,4,5]
        >>> f = [1,2,3,4,6]
        >>> metrics.bias(a,f) # returns 1
        """
        return np.sum(np.array(f) - np.array(a))

    @staticmethod
    def abias(a,f):
        """ Returns the total bias over a given forecast horizon in terms of absolute values. 
        Divide by the length of the forecast horizon to get average bias.
        This is a good metric to minimize when testing/tuning models.

        Args:
            a (list-like): The actuals over the forecast horizon.
            f (list-like): The predictions over the forecast horizon.

        Returns:
            (float): The derived bias.

        >>> from scalecast.util import metrics
        >>> a = [1,2,3,4,5]
        >>> f = [1,2,3,4,6]
        >>> metrics.abias(a,f) # returns 1
        """
        return np.abs(np.sum(np.array(f) - np.array(a)))

    @staticmethod
    def mape(a,f):
        """ Mean absolute percentage error (MAPE).

        Args:
            a (list-like): The actuals over the forecast horizon.
            f (list-like): The predictions over the forecast horizon.

        Returns:
            (float): The derived MAPE.

        >>> from scalecast.util import metrics
        >>> a = [1,2,3,4,5]
        >>> f = [1,2,3,4,6]
        >>> metrics.mape(a,f)
        """
        return (
            np.nan if np.abs(a).min() == 0 
            else mean_absolute_percentage_error(a, f)
        )


    @staticmethod
    def r2(a,f):
        """ R-squared (R2).

        Args:
            a (list-like): The actuals over the forecast horizon.
            f (list-like): The predictions over the forecast horizon.

        Returns:
            (float): The derived R2.

        >>> from scalecast.util import metrics
        >>> a = [1,2,3,4,5]
        >>> f = [1,2,3,4,6]
        >>> metrics.r2(a,f)
        """
        return r2_score(a, f)

    @staticmethod
    def mse(a,f):
        """ Mean squared error (MSE).

        Args:
            a (list-like): The actuals over the forecast horizon.
            f (list-like): The predictions over the forecast horizon.

        Returns:
            (float): The derived MSE.

        >>> from scalecast.util import metrics
        >>> a = [1,2,3,4,5]
        >>> f = [1,2,3,4,6]
        >>> metrics.mse(a,f)
        """
        return mean_squared_error(a, f)

    @staticmethod
    def rmse(a,f):
        """ Root mean squared error (RMSE).

        Args:
            a (list-like): The actuals over the forecast horizon.
            f (list-like): The predictions over the forecast horizon.

        Returns:
            (float): The derived RMSE.

        >>> from scalecast.util import metrics
        >>> a = [1,2,3,4,5]
        >>> f = [1,2,3,4,6]
        >>> metrics.rmse(a,f)
        """
        return mean_squared_error(a, f) ** 0.5

    @staticmethod
    def mae(a,f):
        """ Mean absolute error (MAE).

        Args:
            a (list-like): The actuals over the forecast horizon.
            f (list-like): The predictions over the forecast horizon.

        Returns:
            (float): The derived MAE.

        >>> from scalecast.util import metrics
        >>> a = [1,2,3,4,5]
        >>> f = [1,2,3,4,6]
        >>> metrics.mae(a,f)
        """
        return mean_absolute_error(a, f)

    @staticmethod
    def smape(a,f):
        """ Symmetric mean absolute percentage error (sMAPE).
        Uses the same definition as used in the M4 competition.
        Does not multiply by 100.
        See https://ideas.repec.org/a/eee/intfor/v36y2020i1p54-74.html.

        Args:
            a (list-like): The actuals over the forecast horizon.
            f (list-like): The predictions over the forecast horizon.

        Returns:
            (float): The derived sMAPE.

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

    @staticmethod
    def mase(a,f,obs,m):
        """ Mean absolute scaled error (MASE).
        Uses the same definition as used in the M4 competition.
        See https://ideas.repec.org/a/eee/intfor/v36y2020i1p54-74.html.

        Args:
            a (list-like): The actuals over the forecast horizon.
            f (list-like): The predictions over the forecast horizon.
            obs (list-like): The actual observations used to create the forecast.
            m (int): The seasonal period.

        Returns:
            (float): The derived MASE.

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

    @staticmethod
    def msis(a,uf,lf,obs,m,alpha=0.05):
        """ Mean scaled interval score (MSIS) for evaluating confidence intervals.
        Uses the same definition as used in the M4 competition.
        Lower values are better.
        See https://ideas.repec.org/a/eee/intfor/v36y2020i1p54-74.html.

        Args:
            a (list-like): The actuals over the forecast horizon.
            uf (list-like): The upper-forecast bound according to the confidence interval.
            lf (list-like): The lower-forecast bound according to the confidence interval.
            obs (list-like): The actual observations used to create the forecast.
            m (int): The seasonal period.
            alpha (float): Default 0.05. 0.05 for 95% confidence intervals, etc.

        Returns:
            (float): The derived MSIS.

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

def plot_reduction_errors(f, ax = None, figsize=(12,6)):
    """ Plots the resulting error/accuracy of a Forecaster object where `reduce_Xvars()` method has been called
    with method = 'pfi' or method = 'shap'.
    
    Args:
        f (Forecaster): An object that has called the `reduce_Xvars()` method with method = 'pfi'.
        ax (Axis): Optional. The existing axis to write the resulting figure to.
        figsize (tuple): Default (12,6). The size of the resulting figure. Ignored when ax is not None.
        
    Returns:
        (Axis) The figure's axis.

    >>> from scalecast.Forecaster import Forecaster
    >>> from scalecast.util import plot_reduction_errors
    >>> import matplotlib.pyplot as plt
    >>> import seaborn as sns
    >>> import pandas as pd
    >>> import pandas_datareader as pdr
    >>> 
    >>> df = pdr.get_data_fred('HOUSTNSA',start='1900-01-01',end='2021-06-01')
    >>> f = Forecaster(y=df['HOUSTNSA'],current_dates=df.index)
    >>> f.set_test_length(.2)
    >>> f.generate_future_dates(24)
    >>> f.add_ar_terms(24)
    >>> f.add_seasonal_regressors('month',raw=False,sincos=True,dummy=True)
    >>> f.add_seasonal_regressors('year')
    >>> f.add_time_trend()
    >>> f.reduce_Xvars(method='pfi')
    >>> plot_reduction_errors(f)
    >>> plt.show()
    """
    dropped = f.pfi_dropped_vars
    errors = f.pfi_error_values
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    sns.lineplot(
        x=np.arange(0, len(dropped) + 1, 1), y=errors,
    )
    plt.xlabel("dropped Xvars")
    plt.ylabel("error")
    return ax

def backtest_metrics(
    backtest_results: list,
    models = None,
    mets = ['rmse'],
    mase = False, 
    msis = False,
    msis_alpha = 0.05,
    m = 1, 
    names = None,
) -> pd.DataFrame:
    """ Ingests the results output from `Pipeline.backtest()` and converts results to metrics.
    
    Args:
        backtest_results (list): The output returned from `Pipeline.backtest()` or `MVPipeline.backtest()`.
        models (collection): The names of the models to display metrics for. Default displays all models.
        mets (list[str or callable]): Default ['rmse']. A list of metrics to calculate.
            If the element is str type, must be taken from the `util.metrics` class 
            where the only two accepted arguments are `a` and `f`.
            If the element in the list is callable, must be a function that only accepts two arguments (first actuals second forecast) 
            and returns a float.
        mase (bool): Default False.
            Whether to also calculate mase. Must specify seasonality in m.
        msis (bool): Default False.
            Whether to also calculate msis. Must specify seasonality in m. This will fail if confidence intervals were not evaluated.
        msis_alpha (float): Default 0.05. The level that confidence intervals were evaluated at. Ignored if msis is False.
        m (int): Default 1. The number of steps that count one seasonal cycle. Ignored if both of msis and mase is False.
        names (list): Optional. The names to assign each passed series. Ignored if there is only one passed series.
    
    Returns:
        (DataFrame): The metrics dataframe that gives info about each backtested series, model, and selected metric.

    >>> f1, f2, f3 = pipeline.fit_predict(f1,f2,f3)
    >>> backtest_results = pipeline.backtest(f1,f2,f3,n_iter=2,jump_back=12)
    >>> backtest_mets = backtest_metrics(
    >>>     backtest_results,
    >>>     mets = ['rmse','smape','r2','mae'],
    >>>     names=['UTUR','UNRATE','SAHMREALTIME'],
    >>>     mase = True,
    >>>     msis = True,
    >>>     m = 12,
    >>> )
    """
    m_ = m
    labels = names if names is not None else [f'Series{i}' for i in range(len(backtest_results))]
    res_dict = {k:None for k in labels}
    mets_str = [m if isinstance(m,str) else m.__name__ for m in mets]
    actuals_obs = ['Actuals','Obs']
    if models is None:
        models = [m for m in backtest_results[0].keys() if m not in actuals_obs]
    elif isinstance(models,str):
        models = [models]
    for h, s in enumerate(backtest_results):
        for i, m in enumerate(actuals_obs + models):
            if i == 0:
                a_df = s[m][[c for c in s[m] if c.endswith('Vals')]]
                results = pd.DataFrame(
                    columns = [f'Iter{i}' for i in range(a_df.shape[1])],
                    index = pd.MultiIndex.from_product(
                            [
                                models[:],
                                mets_str + ([] if not mase else ['mase']) + ([] if not msis else ['msis']),
                            ],
                            names = ['Model','Metric']
                    ),
                )
                a_df.columns = results.columns.to_list()
            elif i == 1:
                ob_df = s[m]
            else:
                f_df = s[m][[c for c in s[m] if c.endswith('Fcst')]]
                f_df.columns = results.columns.to_list()
                for met in mets_str:
                    for c in results:
                        results.loc[(m,met),c] = (
                            getattr(metrics,met)(
                                a = a_df[c],
                                f = f_df[c],
                            ) if isinstance(met,str) 
                            else met(
                                a_df[c],
                                f_df[c],
                            )
                        )
                if mase:
                    for c in results:
                        results.loc[(m,'mase'),c] = getattr(metrics,'mase')(
                            a = a_df[c],
                            f = f_df[c],
                            obs = ob_df[c].dropna(),
                            m = m_,
                        )
                if msis:
                    up_df = s[m][[c for c in s[m] if c.endswith('Upper')]]
                    lo_df = s[m][[c for c in s[m] if c.endswith('Lower')]]
                    up_df.columns = f_df.columns.to_list()
                    lo_df.columns = f_df.columns.to_list()
                    for c in results:
                        results.loc[(m,'msis'),c] = getattr(metrics,'msis')(
                            a = a_df[c],
                            uf = up_df[c],
                            lf = lo_df[c],
                            obs = ob_df[c].dropna(),
                            m = m_,
                            alpha = msis_alpha,
                        )
        res_dict[labels[h]] = results

    if len(res_dict) == 1:
        results_df = res_dict[labels[0]]
    else:
        res_list = []
        for k, df in res_dict.items():
            df['Series'] = k
            df = df.reset_index().set_index(['Series','Model','Metric'])
            res_list.append(df)
        results_df = pd.concat(res_list)

    results_df['Average'] = results_df.mean(axis=1)
    return results_df

def _backtest_plot(
    backtest_results,
    models = None,
    series = None,
    names = None,
    ci = False,
    ax=None,
    figsize=(12,6),
):
    """ (Coming soon). Plots the results from a backtested pipeline. If all default arguments are passed, the resulting plot can
    be messy, so it's recommended to limit some of the output by specifying arguments.

    Args:
        backtest_results (list): The output returned from `Pipeline.backtest()` or `MVPipeline.backtest()`.
        models (list): Which models to plot results for. By default, all models will be plotted.
        series (list): Which series to plot results for. This argument expects a list with an index position.
            Ex: [0,2,3] will plot series in the first, third, and fourth index positions of backtest_results.
        names (list): Optional. The names to give to the corresponding series. Must be the same length as series.
            If not specified, the series will be named 'Series1' - 'SeriesN'.
        ci (bool): Default False. Whether to plot confidence intervals. This will raise a warning if confidence intervals are not found.
        ax (Axis): Optional. A pre-existing axis to write the figure to.
        figsize (tuple): Default (12,6). The size of the resulting figure. Ignored when ax is specified.
    
    Returns:
        (Axis): The figure's axis.

    >>>
    >>>
    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

def break_mv_forecaster(mvf,drop_all_Xvars = True):
    """ Breaks apart an MVForecaster object and returns as many Foreaster objects as series loaded into the object.

    Args:
        mvf (MVForecaster): The object to break apart.
        drop_all_Xvars (bool): Default True. Whether to drop all Xvars during the conversion.
            It's a good idea to leave this True because length mismatches can cause future univariate models
            to error out.

    Returns:
        (tuple[Forecaster]): A sequence of at least two Forecaster objects
    
    >>> from scalecast.MVForecaster import MVForecaster
    >>> from scalecast.util import break_mv_forecaster
    >>> 
    >>> f1, f2 = break_mv_forecaster(mvf)
    """
    from .Forecaster import Forecaster
    def convert_mv_hist(f, mvhist: dict, series_name: str):
        hist = {}
        for k, v in mvhist.items():
            hist[k] = {}
            for k2, v2 in v.items():
                if k2 in (""):
                    continue
                elif not isinstance(v2, dict) or k2 == "HyperParams":
                    hist[k][k2] = v2
                elif isinstance(v2, dict):
                    try:
                        hist[k][k2] = v2[series_name]
                    except IndexError:
                        hist[k][k2] = []
        return hist

    mvf1 = mvf.deepcopy()

    set_len = (
        len(mvf1.y[mvf1.names[0]]) 
        if not mvf1.current_xreg 
        else len(list(mvf1.current_xreg.values())[0])
    )
    to_return = []
    for s in mvf1.names:
        f = Forecaster(
            y=mvf1.y[s].values[-set_len:],
            current_dates=mvf1.current_dates.values[-set_len:],
            future_dates=len(mvf1.future_dates),
            current_xreg={k:v.copy() for k,v in mvf1.current_xreg.items()},
            future_xreg={k:v.copy() for k,v in mvf1.future_xreg.items()},
            test_length=mvf1.test_length,
            validation_length=mvf1.validation_length,
            cis = mvf1.cis,
            cilevel = mvf1.cilevel,
        )
        f.history = convert_mv_hist(
            f=f, 
            mvhist=mvf1.history, 
            series_name=s,
        )
        if drop_all_Xvars:
            f.drop_all_Xvars()

        to_return.append(f)

    return tuple(to_return)

def find_optimal_lag_order(mvf,train_only=False,**kwargs):
    """ Returns the otpimal lag order for a mutlivariate process using the statsmodels function:
    https://www.statsmodels.org/dev/generated/statsmodels.tsa.vector_ar.var_model.VAR.select_order.html.
    The exogenous regressors are set based on Xvars loaded in the `MVForecaster` object.

    Args:
        mvf (MVForecaster): The MVForecaster object with series loaded to find the optimal order for
        train_only (bool): Default False. Whether to use the training data only in the test.
        **kwargs: Passed to the referenced statsmodels function

    Returns:
        (LagOrderResults): Lag selections.
    
    >>> from scalecast.Forecaster import Forecaster
    >>> from scalecast.MVForecaster import MVForecaster
    >>> from scalecast.util import find_optimal_lag_order
    >>> import pandas_datareader as pdr
    >>>
    >>> s1 = pdr.get_data_fred('UTUR',start='2000-01-01',end='2022-01-01')
    >>> s2 = pdr.get_data_fred('UNRATE',start='2000-01-01',end='2022-01-01')
    >>>
    >>> f1 = Forecaster(y=s1['UTUR'],current_dates=s1.index)
    >>> f2 = Forecaster(y=s2['UNRATE'],current_dates=s2.index)
    >>>
    >>> mvf = MVForecaster(f1,f2,names=['UTUR','UNRATE'])
    >>> lag_order_res = find_optimal_lag_order(mvf,train_only=True)
    >>> lag_order_aic = lag_order_res.aic # picks the best lag order according to aic
    """
    from statsmodels.tsa.vector_ar.var_model import VAR
    data = np.array([v.astype(float).values.copy() for k, v in mvf.y.items()]).T

    if mvf.current_xreg:
        exog = np.array(v.values.copy() for k, v in mvf.current_xreg.items()).T
    else:
        exog = None

    if train_only:
        data = data[:-mvf.test_length]
        if exog is not None:
            exog = exog[:-mvf.test_length]

    model = VAR(data,exog=exog)

    return model.select_order(**kwargs)

def infer_apply_Xvar_selection(infer_from,apply_to,return_copy=False):
    """ Attempts to infer what Xvars have been added to one Forecaster object and applies the guess to another Forecaster object.
    If using default fourier seasonal terms, linear or log trend terms, and autoregressive terms only, with default namin, 
    this will guess all variables successfully. Other variables (such as through `Forecaster.add_Xvars_df()`) will not be added. 
    Any variables that cannot be inferred will be raised in a warning.

    Args:
        infer_from (Forecaster): The `Forecaster` object to infer the Xvars from.
        apply_to (Forecaster): The `Forecaster` object to apply the guess to.
        return_copy (bool): Default False. Whether to create a copy of the `Forecaster` object passed to `apply_to`.
            Default will add Xvars to the instance passed to `apply_to`.

    Returns:
        (Forecaster): The Forecaster object with the inferred variables added to it.

    >>> f2 = infer_apply_Xvar_selection(infer_from=f1,apply_to=f2)
    """
    if return_copy:
        apply_to = apply_to.deepcopy()
    
    not_guessed = []
    for k in infer_from.current_xreg.keys():
        if k.startswith('AR'):
            apply_to.add_ar_terms([int(k.split('AR')[-1])])
        elif k.endswith('sin'):
            apply_to.add_seasonal_regressors(k.split('sin')[0],sincos=True,raw=False)
        elif k.endswith('cos'):
            continue
        elif k == 't' or k == 'lnt':
            apply_to.add_time_trend()
            if k == 'lnt':
                apply_to.add_logged_terms('t',drop=True)
        else:
            not_guessed.append(k)

    if len(not_guessed):
        warnings.warn(
            f'The inference was unable to guess the following variables: {not_guessed}.'
            ' All others have been added to the Forecaster object passed to the `apply_to` argument.',
            category=Warning,
        )

    return apply_to

def find_optimal_coint_rank(mvf,det_order,k_ar_diff,train_only=False,**kwargs):
    """ Returns the optimal cointigration rank for a multivariate process using the function from statsmodels: 
    https://www.statsmodels.org/dev/generated/statsmodels.tsa.vector_ar.vecm.select_coint_rank.html

    Args:
       mvf (MVForecaster): The MVForecaster object with series loaded to find the optimal rank for.
       train_only (bool): Default False. Whether to use the training data only in the test.
        **kwargs: Passed to the referenced statsmodels function.

    Returns:
        (CointRankResults): Object containing the cointegration rank suggested by the test and allowing a summary to be printed.

    >>> from scalecast.Forecaster import Forecaster
    >>> from scalecast.MVForecaster import MVForecaster
    >>> from scalecast.util import find_optimal_coint_rank
    >>> import pandas_datareader as pdr
    >>>
    >>> s1 = pdr.get_data_fred('UTUR',start='2000-01-01',end='2022-01-01')
    >>> s2 = pdr.get_data_fred('UNRATE',start='2000-01-01',end='2022-01-01')
    >>>
    >>> f1 = Forecaster(y=s1['UTUR'],current_dates=s1.index)
    >>> f2 = Forecaster(y=s2['UNRATE'],current_dates=s2.index)
    >>>
    >>> mvf = MVForecaster(f1,f2,names=['UTUR','UNRATE'])
    >>> coint_res = find_optimal_coint_rank(mvf,det_order=-1,k_ar_diff=8,train_only=True)
    >>> print(coint_res) # prints a report
    >>> rank = coint_res.rank # best rank
    """
    from statsmodels.tsa.vector_ar.vecm import select_coint_rank
    data = np.array([v.astype(float).values.copy() for k, v in mvf.y.items()]).T
    if train_only:
        data = data[:-mvf.test_length]
    
    return select_coint_rank(
        data,
        det_order=det_order,
        k_ar_diff=k_ar_diff,
        **kwargs,
    )

@_developer_utils.log_warnings
def find_statistical_transformation(
    f,
    goal=['stationary'],
    train_only=False,
    critical_pval=.05,
    log = True,
    m='auto',
    adf_kwargs = {},
    **kwargs,
):
    """ Finds a set of transformations to achieve stationarity or seasonal adjustment, based on results from statistical tests.

    Args:
        f (Forecaster): The object that stores the series to test.
        goal (list-like): Default ['stationary']. One or multiple of 'stationary', 'seasonally_adj'. 
            Other options may be coming in the future.
            If more than one goal is passed, will try to satisfy all goals in the order passed.
            For stationary: uses an Augmented Dickey-Fuller test to determine if the series is stationary.
            If not stationary, returns a diff transformation and log transformation if log is True.
            For seasonall_adj: uses seasonal auto_arima to find the optimal seasonal diff.
        train_only (bool): Default False. Whether to use train set only in all statistical tests.
        log (bool): Default True. Whether to log and difference the series if it is found to be non-stationary or just difference.
            This will set itself to False if the lowest observed series value is 0 or lower.
        critical_pval (float): Default 0.05. The cutoff p-value to use to determine statistical signficance in the 
            Augmented Dickey-Fuller test and to run the auto_arima selection (substitutes for `alpha` arg).
        m (str or int): Default 'auto': The number of observations that counts one seasonal step.
            When 'auto', uses the M4 competition values: 
            for Hourly: 24, Monthly: 12, Quarterly: 4. everything else gets 1 (no seasonality assumed)
            so pass your own values for other frequencies.
        adf_kwargs (dict): Default {}. Keyword args to pass to the Augmented Dickey-Fuller test function.
            See the `maxlag`, `regression`, and `autolag` arguments from
            https://www.statsmodels.org/dev/generated/statsmodels.tsa.stattools.adfuller.html.
        **kwargs: Passed to the auto_arima() function when searching for optimal seasonal diff.

    Returns:
        (Transformer, Reverter): A `Transformer` object with the identified transforming functions and
        the `Reverter` object with the `Transformer` counterpart functions.

    >>> from scalecast.Forecaster import Forecaster
    >>> from scaleast.Pipeline import Pipeline, Transformer, Reverter
    >>> from scalecast.util import find_statistical_transformation
    >>> import pandas_datareader as pdr
    >>> 
    >>> def forecaster(f):
    >>>     f.add_covid19_regressor()
    >>>     f.auto_Xvar_select(cross_validate=True)
    >>>     f.set_estimator('mlr')
    >>>     f.manual_forecast()
    >>> df = pdr.get_data_fred(
    >>>     'HOUSTNSA',
    >>>     start='1959-01-01',
    >>>     end='2022-08-01'
    >>> )
    >>> f = Forecaster(
    >>>     y=df['HOUSTNSA'],
    >>>     current_dates=df.index,
    >>>     future_dates=24,
    >>>     test_length = .2,
    >>> )
    >>> f.set_validation_length(24)
    >>> transformer, reverter = find_statistical_transformation(
    >>>     f,
    >>>     goal=['stationary','seasonally_adj'],
    >>>     train_only=True,
    >>>     critical_pval = .01,
    >>> )
    >>> print(reverter) # see what transformers and reverters were chosen
    >>> pipeline = Pipeline(
    >>>   steps = [
    >>>       ('Transform',transformer),
    >>>       ('Forecast',forecaster),
    >>>       ('Revert',reverter),
    >>>   ],
    >>> )
    >>> f = pipeline.fit_predict(f)
    """
    from . import SeriesTransformer
    from . import Pipeline
    from .auxmodels import auto_arima
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
        if I > 0:
            for i in range(1,I+1):
                transformers += [('DiffTransform',m)]

        return transformers

    f = f.deepcopy()
    transformer = SeriesTransformer.SeriesTransformer(f)

    m = _developer_utils._convert_m(m,f.freq)

    possible_args = {
        'stationary':make_stationary,
        'seasonally_adj':seasonally_adj,
    }
    bad_args = [g for g in goal if g not in possible_args]
    if len(bad_args) > 0:
        raise ValueError(f'Values passed to goal arg cannot be used: {bad_args}.')

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
        base_transformer = final_transformer,
    )
    return final_transformer, final_reverter

@_developer_utils.log_warnings
def find_optimal_transformation(
    f,
    estimator=None,
    monitor='rmse',
    test_length=None,
    train_length=None,
    num_test_sets=1,
    space_between_sets = 1,
    lags='auto',
    try_order = ['detrend','seasonal_adj','boxcox','first_diff','first_seasonal_diff','scale'],
    boxcox_lambdas = [-0.5,0,0.5],
    detrend_kwargs = [{'loess':True},{'poly_order':1},{'poly_order':2}],
    scale_type = ['Scale','MinMax','RobustScale'],
    m='auto',
    model = 'add',
    return_train_only = False,
    verbose = False,
    **kwargs,
):
    """ Finds a set of transformations based on what maximizes forecast accuracy on some out-of-sample metric.
    Works by comparing each transformation individually and stacking the set of transformations that leads to the best
    performance. The estimator only uses series lags as inputs. When an attempted transformation fails, a warning is logged.
    The function uses `Pipeline.backtest()` to assure that the selected set of transformations is truly tested out-of-sample.

    Args:
        f (Forecaster): The Forecaster object that contains the series that will be transformed.
        estimator (str): One of `Forecaster.can_be_tuned`. The estimator to use to choose the best 
            transformations with. The default will read whatever is set to f.estimator.
        monitor (str or callable): Default 'rmse'. The error metric to minimize.
            If str, must exist in `util.metrics` 
            and accept only two arguments. If callable, must accept only two arguments 
            (an array of actuals and an array of forecasts) and return a float.
            If 'r2' is passed, this will monitor a negative r2 value.
        test_length (int): The amount of observations to hold out-of-sample. By default
            reads the number of dates in f.future_dates.
        train_length (int): The number of observations to train the model in each iteration.
            By default, uses all available observations that come before each test set.
        num_test_sets (int): Default 1. The number of test sets to iterate through. The final metric will be 
            an average across all test sets.
        space_between_sets (int): Default 1. The space between consecutive training sets.
            Not applicable when num_test_sets is 1.
        lags (str or int): Default 'auto'. The number of lags that will be used as inputs for the estimator.
            If 'auto', uses the value passed or assigned to m (one seasonal cycle). 
            If multiple values passed to m, uses the first.
        try_order (list-like): Default ['detrend','seasonal_adj','boxcox','first_diff','first_seasonal_diff','scale'].
            The transformations to try and also the order to try them in.
            Changing the order here can change the final transformations derived, since level will 
            be compared to the first transformation and if it is found to be better than level, it will
            carry over to be tried in conjunction with the next transformation and so forth.
            The default list contains all possible transformations for this function.
        boxcox_lambdas (list-like): Default [-0.5,0,0.5].
            The lambda values to try for a boxcox transformation.
            0 means natural log. Only up to one boxcox transformation will be selected.
        detrend_kwargs (list-like[dict]): Default 
            [{'loess':True},{'poly_order':1},{'poly_order':2}].
            The types of detrending to try. Only up to one one detrender will be selected.
        scale_type (list-like): Default ['Scale','MinMax','RobustScale']. The type of scaling to try.
            Only up to one scaler will be selected.
            Must exist a `SeriesTranformer.{scale_type}Transform()` function for this to work.
        m (int or str): Default 'auto'. The number of observations that counts one seasonal step.
            Ignored when seasonal_lags = 0.
            When 'auto', uses the M4 competition values:
            for Hourly: 24, Monthly: 12, Quarterly: 4. Everything else gets inferred if possible.
            If list, multiple adjustments will be tried and up to that many adjustments can be selected.
        model (str): Default 'add'. One of {"additive", "add", "multiplicative", "mul"}.
            The type of seasonal component. Only relevant for the 'seasonal_adj' option in try_order.
        return_train_only (bool): Default False. Whether the returned selections should be set to train_only.
            All tries are completely out-of-sample but the returned transformations will not hold out the
            test-set in the Forecaster object when detrending, deseasoning, and scaling, so setting this to True
            can prevent leakage.
        verbose (bool): Default False. Whether to print info about the transformers/reverters being tried.
        **kwargs: Passed to the `Forecaster.manual_forecast()` function and possible values change based on which
            estimator is used.

    Returns:
        (Transformer, Reverter): A `Transformer` object with the identified transforming functions and
        the `Reverter` object with the `Transformer` counterpart functions.

    >>> from scalecast.Forecaster import Forecaster
    >>> from scaleast.Pipeline import Pipeline, Transformer, Reverter
    >>> from scalecast.util import find_optimal_transformation
    >>> import pandas_datareader as pdr
    >>> 
    >>> def forecaster(f):
    >>>     f.add_covid19_regressor()
    >>>     f.auto_Xvar_select(cross_validate=True)
    >>>     f.set_estimator('mlr')
    >>>     f.manual_forecast()
    >>> df = pdr.get_data_fred(
    >>>     'HOUSTNSA',
    >>>     start='1959-01-01',
    >>>     end='2022-08-01'
    >>> )
    >>> f = Forecaster(
    >>>     y=df['HOUSTNSA'],
    >>>     current_dates=df.index,
    >>>     future_dates=24,
    >>>     test_length = .2, # this will be monitored for performance
    >>> )
    >>> f.set_validation_length(24)
    >>> transformer, reverter = find_optimal_transformation(f)
    >>> print(reverter) # see what transformers and reverters were chosen
    >>> pipeline = Pipeline(
    >>>   steps = [
    >>>       ('Transform',transformer),
    >>>       ('Forecast',forecaster),
    >>>       ('Revert',reverter),
    >>>   ],
    >>> )
    >>> f = pipeline.fit_predict(f)
    """
    from . import Pipeline
    from ._Forecaster_parent import ForecastError
    def forecaster(f):
        f.add_ar_terms(lags)
        f.set_estimator(estimator)
        f.manual_forecast(**kwargs)

    def make_pipeline_fit_predict(f,transformer,reverter):
        tr = Pipeline.Transformer(transformers=transformer)
        re = Pipeline.Reverter(reverters=reverter,base_transformer=tr)
        pipeline = Pipeline.Pipeline(
            steps = [
                ('Transform',tr),
                ('Forecast',forecaster),
                ('Revert',re)
            ],
        )
        res = pipeline.backtest(
            f,
            n_iter = num_test_sets, 
            jump_back = space_between_sets,
            fcst_length = test_length,
            series_length = train_length,
            cis = False,
        )
        mets = backtest_metrics(res,mets=[monitor])
        if verbose:
            print(f'Last transformer tried:\n{tr.transformers}')
            print(f'Score ({monitor if isinstance(monitor,str) else monitor.__name__}): {mets.iloc[0,-1]}')
            print('-'*50)
        return mets.iloc[0,-1]

    def neg_r2(metric):
        return -metric if monitor == 'r2' else metric

    estimator = f.estimator if estimator is None else estimator
    test_length = len(f.future_dates) if test_length is None else int(test_length)
    if test_length<=0:
        raise ValueError(
            'The argument test_length must be above 0 and is the number of future dates in the Forecaster object by default.'
            f' The value received is {test_length}.'
        )

    f = f.deepcopy()
    f.set_metrics([monitor])
    f.drop_all_Xvars()
    f.history = {}

    m = _developer_utils._convert_m(m,f.freq)
    lags = m if lags == 'auto' and not hasattr(m,'__len__') else m[0] if lags == 'auto' else lags

    if verbose:
        print(f'Using {estimator} model to find the best transformation set on {num_test_sets} test sets, each {test_length} in length.')
        if estimator in f.sklearn_estimators:
            print(f'All transformation tries will be evaluated with {lags} lags.')

    level_met = neg_r2(make_pipeline_fit_predict(f,[],[]))
    final_transformer = []
    final_reverter = []

    exception_types = (IndexError,AttributeError,ValueError,ZeroDivisionError,ForecastError) # errors to pass over

    for tr in try_order:
        if tr == 'boxcox':
            def boxcox_tr(x,lmbda):
                return [(i**lmbda - 1) / lmbda for i in x] if lmbda != 0 else [np.log(i) for i in x]
            def boxcox_re(x,lmbda):
                return [(i*lmbda + 1)**(1/lmbda) for i in x] if lmbda != 0 else [np.exp(i) for i in x]
            for i, lmbda in enumerate(boxcox_lambdas):
                transformer = final_transformer[:]
                reverter = final_reverter[:]
                if i == 0:
                    met = level_met
                    best_transformer = transformer[:]
                    best_reverter = reverter[:]
                try:
                    transformer.append(('Transform',boxcox_tr,{'lmbda':lmbda}))
                    reverter.reverse(); reverter.append(('Revert',boxcox_re,{'lmbda':lmbda})); reverter.reverse()
                    comp_met = neg_r2(make_pipeline_fit_predict(f,transformer,reverter))
                    if comp_met < met:
                        met = comp_met
                        best_transformer = transformer[:]
                        best_reverter = reverter[:]
                except exception_types as e:
                    warnings.warn(f'Lambda value of {lmbda} cannot be evaluated. error: {e}')
            final_transformer = best_transformer[:]
            final_reverter = best_reverter[:]
            level_met = met
        elif tr == 'detrend':
            for i, kw in enumerate(detrend_kwargs):
                transformer = final_transformer[:]
                reverter = final_reverter[:]
                if i == 0:
                    met = level_met
                    best_transformer = transformer[:]
                    best_reverter = reverter[:]
                try:
                    transformer.append(('DetrendTransform',kw))
                    reverter.reverse(); reverter.append(('DetrendRevert',)); reverter.reverse()
                    comp_met = neg_r2(make_pipeline_fit_predict(f,transformer,reverter))
                    if comp_met < met:
                        met = comp_met
                        best_transformer = transformer[:]
                        best_reverter = reverter[:]
                except exception_types as e:
                    warnings.warn(f'Detrend_kwargs {kw} cannot be evaluated. error: {e}')
            final_transformer = best_transformer[:]
            final_reverter = best_reverter[:]
            level_met = met
        elif tr == 'first_diff':
            met = level_met
            transformer = final_transformer[:]
            reverter = final_reverter[:]
            best_transformer = transformer[:]
            best_reverter = reverter[:]
            try:
                transformer.append(('DiffTransform',1))
                reverter.reverse(); reverter.append(('DiffRevert',1)); reverter.reverse()
                comp_met = neg_r2(make_pipeline_fit_predict(f,transformer,reverter))
                if comp_met < met:
                    met = comp_met
                    best_transformer = transformer[:]
                    best_reverter = reverter[:]
            except exception_types as e:
                warnings.warn(f'Series first difference could not be evaluated. error: {e}')
            final_transformer = best_transformer[:]
            final_reverter = best_reverter[:]
            level_met = met
        elif tr in ('seasonal_adj', 'first_seasonal_diff'):
            if not hasattr(m,'__len__'):
                m = [m]
            for mi in m:
                if mi > 1:
                    met = level_met
                    transformer = final_transformer[:]
                    reverter = final_reverter[:]
                    best_transformer = transformer[:]
                    best_reverter = reverter[:]
                    try:
                        transformer += (
                            [('DiffTransform',mi)] if tr == 'first_seasonal_diff'
                            else [('DeseasonTransform',{'m':mi,'model':model})]
                        )
                        reverter.reverse()
                        reverter += (
                            [('DiffRevert',mi)] if tr == 'first_seasonal_diff' 
                            else [('DeseasonRevert',{'m':mi})]
                        )
                        reverter.reverse()
                        comp_met = neg_r2(make_pipeline_fit_predict(f,transformer,reverter))
                        if comp_met < met:
                            met = comp_met
                            best_transformer = transformer[:]
                            best_reverter = reverter[:]
                    except exception_types as e:
                        warnings.warn(f'Series seasonal adjustment could not be evaluated. error: {e}')
                    final_transformer = best_transformer[:]
                    final_reverter = best_reverter[:]
                    level_met = met
                else:
                    warnings.warn('Seasonal differences and adjustments cannot be evaluated when m = 1.')
        elif tr == 'scale':
            for i, s in enumerate(scale_type):
                transformer = final_transformer[:]
                reverter = final_reverter[:]
                if i == 0:
                    met = level_met
                    best_transformer = transformer[:]
                    best_reverter = reverter[:]
                try:
                    transformer.append((f'{s}Transform',))
                    reverter.reverse(); reverter.append((f'{s}Revert',)); reverter.reverse()
                    comp_met = neg_r2(make_pipeline_fit_predict(f,transformer,reverter))
                    if comp_met < met:
                        met = comp_met
                        best_transformer = transformer[:]
                        best_reverter = reverter[:]
                except exception_types as e:                 
                    warnings.warn(f'{s} scaler cannot be evaluated. error: {e}')
            final_transformer = best_transformer[:]
            final_reverter = best_reverter[:]
            level_met = met
        else:
            warnings.warn(f'Value: {tr} found in the try_order list cannot be used and will be skipped.')
    
    final_transformer = Pipeline.Transformer(transformers = final_transformer)
    final_reverter = Pipeline.Reverter(reverters = final_reverter,base_transformer = final_transformer)

    if return_train_only:
        has_train_only = (
            'DetrendTransform',
            'DeseasonTransform',
            'ScaleTransform',
            'MinMaxTransform',
            'RobustScaleTransform',
        )
        for i, t in enumerate(final_transformer.transformers):
            if t[0] in has_train_only:
                if len(t) == 1 or not isinstance(t[-1],dict):
                    final_transformer.transformers[i] = t + ({'train_only':True},)
                else:
                    t[-1]['train_only'] = True

    if verbose:
        print(f'Final Selection:\n{final_transformer.transformers}')
    
    return final_transformer, final_reverter 

def Forecaster_with_missing_vals(
    y,
    current_dates,
    desired_frequency = None,
    fill_strategy = 0.0,
    impute_value_pool = None,
    m = None,
    impute_lookback = None,
    add_noise = False,
    noise_value_pool = None,
    noise_std = None,
    noise_lookback = None,
    cannot_be_below = None,
    cannot_be_above = None,
    first_ob_strategy = 'drop',
    random_seed = None,
    **kwargs,
):
    """ Imputes missing values in a given time series such that the result has a user-specified 
    date frequency and/or no remaining null values. If you pass no missing values through this function,
    it will not raise errors.

    Args:
        y (collection): An array of all observed values. Can include NAs for dates in which the values
            are unknown.
        current_dates (collection): An array of all observed dates. 
            Must be same length as y and in the same sequence. 
        desired_frequency (str): The desired frequency of the resulting Forecaster object.
            If this is left unspecified and a frequency cannot be inferred, the resulting
            object will not have a logical frequency. See available values here:
            https://pandas.pydata.org/docs/user_guide/timeseries.html#timeseries-offset-aliases
        fill_strategy (float or str): Default 0.0. 
            If str, must be one of 
            {'linear_interp', 'moving_average', 'moving_seasonal_average', 'impute_pool'}.
            If not one of those values, will be passed to the `df.fillna()` or `df.fillna(method=...)` method from pandas
            (valid values include 'ffill' and 'bfill'). Therefore, the default fills with 0.
        m (int): Optional. The number of steps that count one seasonal cycle if using a seasonal fill strategy.
            If left unspecified, will attempt to be inferred. If it cannot be inferred, will raise an error.
        impute_value_pool (collection): Optional. The pool of values to use when `fill_strategy = 'impute_pool'`.
        impute_lookback (int): Required when `fill_strategy in ('moving_average','moving_seasonal_average')`.
            The lookback to use when imputing a moving average to missing values. If using 'moving_seaosnal_average', 
            make sure to include at least one full seasonal cycle in the lookback. Must be 1 or greater. If there are not enough
            observations to create a seasonal fill, will use all available observations for a normal moving average and raise a warning.
        add_noise (bool): Default False. Whether to add random noise to the imputed values.
        noise_value_pool (collection): Optional. The pool of values to randomly choose from when adding noise.
            The noise will add the imputed value with a random draw from this pool to come up with the final value.
            Specifying this argument overrides any of the subsequent noise-related arguments 
            (noise_std, noise_lookback, etc.).
        noise_std (float): Optional. The standard deviation to use when adding a noise to the values.
            Assumes a normal distribution where the mean is the value imputed.
        noise_lookback (int): Optional. Must be 2 or greater.
            If adding noise, the lookback period before the missing obs
            to use to add the noise, assuming a normal distribution with the standard deviation from the lookback.
            If this is larger than the number of observations before a given missing observation, will use all
            observations before the missing one. If this and all the other noise-related arguments
            are left unspeficied, uses all observations before each missing one to find the 
            standard deviation. If the first observation(s) is missing, no noise is given to it.
        cannot_be_below (float): Optional. A minimum value that the final imputation cannot drop below.
        cannot_be_above (float): Optional. A maximum value that the final imputation cannot be above.
        first_ob_strategy (str): Default 'drop'. What to do if the first observation(s) is null. Default will drop.
            Other options include 'ignore', which could cause unexpected results depending on the employed strategy.
            Can also start with 'fill_', where the next digits will be used to create a static fill 
            ('fill_0' fills with 0, for example).
        random_seed (int): Optional. A random seed to set for reproducible results.
        **kwargs: Passed to the Forecaster object (https://scalecast.readthedocs.io/en/latest/Forecaster/Forecaster.html#src.scalecast.Forecaster.Forecaster.__init__)

    Returns:
         (Forecaster): A Forecaster object with missing dates/values filled in.

    >>> # using the function with null values in y
    >>> f = Forecaster_with_missing_vals(
    >>>    y = [1,2,np.nan,4],
    >>>    current_dates=['2020-01-01','2020-01-02','2020-01-03','2020-01-04'],
    >>>    fill_strategy = 'linear_interp',
    >>> ) # replaces missing val with 3
    >>> # using the function with missing dates
    >>> f = Forecaster_with_missing_vals(
    >>>    y = [1,2,4],
    >>>    current_dates=['2020-01-01','2020-01-02','2020-01-04'], # missing '2020-01-03'
    >>>    desired_frequency = 'D', # tell it to use daily frequency
    >>>    fill_strategy = 'linear_interp',
    >>> ) # adds 3 to the 2nd index position in y and adds '2020-01-03' to 2nd index position in current_dates
    """
    from .Forecaster import Forecaster
    if random_seed is not None:
        random.seed(random_seed)

    valid_strategies = [
        'linear_interp',
        'moving_average', 
        'moving_seasonal_average', 
        'impute_pool',
    ]

    ts_df = pd.DataFrame({
        'Date':pd.to_datetime(current_dates),
        'y':[i for i in y],
    })

    if desired_frequency is not None:
        full_ts_df = pd.DataFrame({
            'Date':pd.date_range(
                start=ts_df['Date'].min(),
                end=ts_df['Date'].max(),
                freq = desired_frequency,
            )
        })
        ts_df = full_ts_df.merge(
            ts_df, on = 'Date', how = 'left',
        )
    ts_df['missing'] = ts_df['y'].isnull().astype(int)
    if ts_df['missing'].sum() > 0:
        if (
            fill_strategy not in valid_strategies 
            and not add_noise 
            and (
                not np.isnan(ts_df['y'].values[0])
                or first_ob_strategy == 'ignore'
            )
        ):
            try:
                ts_df.fillna(float(fill_strategy),inplace=True)
            except ValueError:
                ts_df.fillna(method=fill_strategy,inplace=True)
        else:
            ts_df['missing_number'] = (
                (ts_df['missing'] == 1) 
                & (
                    (ts_df['missing'].shift() == 0)
                    | (ts_df['missing'].shift().isnull())
                )
            ).astype(int).cumsum()
            ts_df['missing_number'] = ts_df[['missing','missing_number']].apply(
                lambda x: x[1] if x[0] == 1 else 0,
                axis = 1,
            )

            if np.isnan(ts_df['y'].values[0]):
                if first_ob_strategy == 'ignore':
                    if fill_strategy in [valid_strategies[0],valid_strategies[2]]: # li, sma
                        raise ValueError(
                            'Cannot perform `fill_strategy="linear_interp" '
                            'or `fill_strategy="moving_seasonal_average"` when the first observation is missing.'
                        )
                else:
                    if first_ob_strategy == 'drop':
                        ts_df = ts_df.loc[ts_df['missing_number'] != 1].reset_index(drop=True)
                    elif first_ob_strategy.startswith('fill_'):
                        ts_df.loc[ts_df['missing_number'] == 1,'y'] = float(first_ob_strategy.split('fill_')[-1])
                    else:
                        raise ValueError(f'Unexpected value passed to argument first_ob_strategy: {first_ob_strategy}.')
                    ts_df['missing_number'] = ts_df['missing_number'].apply(lambda x: max(0,x-1))

            if fill_strategy in [valid_strategies[0],valid_strategies[2]]: # li, sma
                ts_df['m'] = 0
                ts_df['b'] = 0
                ts_df['x'] = 0

                dfs = []
                for cn in ts_df['missing_number'].unique():
                    if cn == 0:
                        dfs.append(ts_df.loc[ts_df['missing_number'] == cn])
                    else:
                        ts_df_tmp = ts_df.loc[ts_df['missing_number'] == cn].copy()
                        ts_df_tmp['x'] = 1
                        ts_df_tmp['x'] = ts_df_tmp['x'].cumsum()
                        y1 = ts_df.loc[ts_df_tmp.index[0] - 1,'y']
                        y2 = ts_df.loc[ts_df_tmp.index[-1] + 1,'y']
                        ts_df_tmp['m'] = (y2-y1) / (ts_df_tmp.shape[0] + 1)
                        ts_df_tmp['b'] = y1
                        dfs.append(ts_df_tmp)
                ts_df = pd.concat(dfs).sort_values('Date').reset_index(drop=True)

            for i, v in ts_df.iterrows():
                if np.isnan(v['y']):
                    if fill_strategy == valid_strategies[0]: # li
                        ts_df.loc[i,'y'] = v['m'] * v['x'] + v['b'] # y = mx + b :)
                    
                    elif fill_strategy in (valid_strategies[1],valid_strategies[2]): # ma, sma
                        if v['missing_number'] > 0:
                            _developer_utils.descriptive_assert(
                                impute_lookback is None or impute_lookback >= 1, 
                                ValueError, 
                                f'impute_lookback must be 1 or greater, got {impute_lookback}.'
                            )
                        impute_lookback_i = impute_lookback
                        fill_strategy_i = fill_strategy
                        ma_pool = ts_df.loc[
                            (ts_df['Date'] < v['Date']) 
                            & (ts_df['missing_number'] < v['missing_number'])
                        ]

                        if fill_strategy == valid_strategies[2]: # sma
                            if v['missing_number'] == 1:
                                if m is None:
                                    m = _developer_utils._convert_m('auto',pd.infer_freq(ts_df['Date']))
                            if impute_lookback_i is None:
                                impute_lookback_i = ma_pool.shape[0]
                            else:
                                ma_pool = ma_pool.iloc[-int(impute_lookback_i):]
                            if m > impute_lookback_i:
                                warnings.warn(
                                    'Not enough observations to impute missing value with '
                                    'a seaonal moving average for date {}. '
                                    'Defaulting to a normal moving average.'.format(v['Date']),
                                    category = Warning,
                                )
                                fill_strategy_i = valid_strategies[1]
                        else:
                            if impute_lookback is not None:
                                ma_pool = ma_pool.iloc[-int(impute_lookback):]

                        if fill_strategy_i == valid_strategies[1]: # ma
                            ts_df.loc[i,'y'] = ma_pool['y'].mean()
                        else: # sma
                            ts_df.loc[i,'y'] = np.mean(
                                [j for i, j in enumerate(ma_pool['y'].values[::-1]) if (i + int(v['x'])) % m == 0]
                            )
                    
                    elif fill_strategy == valid_strategies[3]: # pool
                        ts_df.loc[i,'y'] = np.random.choice(impute_value_pool)
                    else:
                        try:
                            ts_df.loc[i,'y'] = float(fill_strategy)
                        except ValueError:
                            ts_df_cut = ts_df.loc[ts_df['Date'] <= v['Date']].copy()
                            ts_df_cut['y'] = ts_df_cut['y'].fillna(method = fill_strategy)
                            ts_df.loc[i,'y'] = ts_df_cut['y'].values[-1]
                    
                    if add_noise:
                        if noise_value_pool is not None:
                            ts_df.loc[i,'y'] += np.random.choice(noise_value_pool)
                        else:
                            if noise_std is None:
                                noise_std_df = ts_df.loc[
                                    (ts_df['Date'] < v['Date']) 
                                    & (ts_df['missing_number'] < v['missing_number'])
                                ]
                                if noise_lookback is not None:
                                    _developer_utils.descriptive_assert(
                                        noise_lookback >= 2,
                                        ValueError,
                                        f'noise_lookback must be 2 or greater, got {noise_lookback}.'
                                    )
                                    noise_std_df = noise_std_df.iloc[-int(noise_lookback):]
                                noise_std = noise_std_df['y'].std()
                            
                            ts_df.loc[i,'y'] = np.random.normal(ts_df.loc[i,'y'], noise_std)
                    if cannot_be_below is not None:
                        ts_df.loc[i,'y'] = max(ts_df.loc[i,'y'], cannot_be_below)
                    if cannot_be_above is not None:
                        ts_df.loc[i,'y'] = min(ts_df.loc[i,'y'], cannot_be_above)

    return Forecaster(
        y = ts_df['y'].values,
        current_dates = ts_df['Date'].values,
        **kwargs,
    )

def backtest_for_resid_matrix(
    *fs,
    pipeline,
    alpha = 0.05,
    bt_n_iter = None,
    jump_back = 1,
):
    """ Performs a backtest on one or more Forecaster objects using pipelines.
    Specifically, performs a backtest so that a residual matrix to make dynamic intervals
    can easily be obtained. (See `util.get_backtest_resid_matrix()` and 
    `util.overwrite_forecast_intervals()`).

    Args:
        *fs (Forecaster): The objects that contain the evaluated forecasts.
            Send one if univariate forecasting with the `Pipeline` class, 
            more than one if multivariate forecasting with the `MVPipeline` class.
        pipeline (Pipeline or MVPipeline): The pipeline to send `*fs` through.
        alpha (float): Default 0.05. The level that confidence intervals need to be evaluated at. 0.05 = 95%.
        bt_n_iter (int): Optional. The number of iterations to backtest. If left unspecified, chooses 1/alpha, 
            the minimum needed to set reliable conformal intervals.
        jump_back (int): Default 1. The space between consecutive training sets in the backtest.


    Returns:
        (List[Dict[str,pd.DataFrame]]): The results from each model and backtest iteration.
        Each dict element of the resulting list corresponds to the `Forecaster` objects in the order
        they were passed (will be length 1 if univariate forecasting). Each key of each dict is either 'Actuals', 'Obs',
        or the name of a model that got backtested. Each value is a DataFrame with the iteration values.
        The 'Actuals' frame has the date information and are the actuals over each forecast horizon. 
        The 'Obs' frame has the actual historical observations to make each forecast, 
        back padded with NA values to make each array the same length.
    """
    bt_n_iter = int(round(1/alpha)) if bt_n_iter is None else bt_n_iter
    if bt_n_iter < round(1/alpha):
        raise ValueError(
            'bt_n_iter must be at least 1/alpha.'
            f' alpha is {alpha} and bt_n_iter is {bt_n_iter}.'
            f' bt_n_iter must be at least {1/alpha:.0f} to successfully '
            f' backtest {1-alpha:.0%} confidence intervals.'
        )
        
    bt_results = pipeline.backtest(
        *fs,
        n_iter=bt_n_iter,
        jump_back = jump_back,
        test_length = 0,
        cis = False,
    )
    
    return bt_results

def get_backtest_resid_matrix(backtest_results):
    """ Converts results from a backtest pipeline into a matrix of residuals.
    Each row in this residual is for a backtest iteration and the columns are a forecast step.

    Args:
        backtest_results (list): The output returned from `Pipeline.backtest()` or `MVPipeline.backtest()`.
            Recommend to obtain this from running `util.backtest_for_resid_matrix()` and to pass
            the results to `util.overwrite_forecast_intervals()`.

    Returns:
        (list[dict[str,numpy.ndarray]]): A list where each element corresponds to the given Forecaster object
        in a backtest. The elements of the list are dictionaries where each key is an evaluated model name and
        each value is a numpy matrix of the appropriate dimensions that can be used to determine a dynamic prediction interval.
    """
    mats = []
    for btr in backtest_results:
        mat = {
            m:np.zeros((btr['Obs'].shape[1],btr['Actuals'].shape[0]))
            for m in btr if m not in ('Actuals','Obs')
        }
        for m, v in btr.items():
            if m in ('Actuals','Obs'):
                continue
            for i in range(mat[m].shape[0]):
                mat[m][i,:] = btr['Actuals'][f'Iter{i}Vals'].values - v[f'Iter{i}Fcst'].values
        mats.append(mat)
    return mats

def overwrite_forecast_intervals(*fs,backtest_resid_matrix,models=None,alpha = .05):
    """ Overwrites naive forecast intervals stored in passed Forecaster objects with dynamic intervals.
    Overwrites future predictions only; does not overwrite intervals for test-set prediction intervals.

    Args:
        *fs (Forecaster): The objects that contain the evaluated forecasts to overwrite confidence intervals.
        backtest_resid_matrix (list): The output returned from `util.get_backtest_resid_matrix()`.
        models (list): Optional. The models to overwrite intervals for. By default, overwrites all
            models found in backtest_resid_matrix.
        alpha (float): Default 0.05. The level that confidence intervals need to be evaluated at. 0.05 = 95%.
            Use the same or larger value passed to backtest_for_resid_matrix() or else this will fail.
    """
    for f, matrix in zip(fs,backtest_resid_matrix):
        i = 0
        models = matrix.keys() if models is None else models
        for m, mat in matrix.items():
            if m not in models:
                continue
            elif i == 0 and mat.shape[0] < round(1/alpha):
                raise ValueError(
                    f'Not enough backtested observations to evaluate confidence intervals at the {1-alpha:.0%} level.'
                    ' Please set alpha to whatever it was set to when running backtest_for_resid_matrix().'
                )
                i += 1
            percentiles = np.percentile(np.abs(mat),100*(1-alpha),axis=0)
            f.history[m]['UpperCI'] = np.array(f.history[m]['Forecast']) + percentiles
            f.history[m]['LowerCI'] = np.array(f.history[m]['Forecast']) - percentiles
            f.history[m]['CILevel'] = 1-alpha
