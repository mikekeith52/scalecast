from scalecast.Forecaster import descriptive_assert, Forecaster
import pandas_datareader as pdr
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def pdr_load(
    sym,
    start=None,
    end=None,
    src='fred',
    require_future_dates=True,
    future_dates=None,
    **kwargs
):
    """ gets data using `pdr.DataReader()`

    Args:
        sym (str): the name of the series to extract.
            cannot be an array of symbols.
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
        future_dates (int): optional: the future dates to add to the model upon initialization.
            if not added when object is initialized, can be added later.
        **kwargs passed to pdr.DataReader() function. 
            see https://pandas-datareader.readthedocs.io/en/latest/remote_data.html.

    Returns:
        (Forecaster): a Forecaster object with the dates and y-values loaded.
    """
    descriptive_assert(
        isinstance(sym,str),
        ValueError,
        f'sym argument only accepts str types, got {type(sym)}'
    )
    df = pdr.DataReader(sym,data_source=src,start=start,end=end,**kwargs)
    return Forecaster(
        y=df[sym],
        current_dates=df.index,
        require_future_dates=require_future_dates,
        future_dates = future_dates,
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
        f = Forecaster(
            y=getattr(mvf1, f"series{s+1}")["y"].values[-set_len:],
            current_dates=mvf1.current_dates.values[-set_len:],
            integration=getattr(mvf1, f"series{s+1}")["integration"],
            levely=getattr(mvf1, f"series{s+1}")["levely"][-set_len:],
            init_dates=getattr(mvf1, f"series{s+1}")["init_dates"][-set_len:],
            future_dates=mvf1.future_dates.copy(),
            current_xreg={k:v.copy() for k,v in mvf1.current_xreg.items()},
            future_xreg={k:v.copy() for k,v in mvf1.future_xreg.items()},
            test_length=mvf1.test_length,
            validation_length=mvf1.validation_length,
        )
        f.history = convert_mv_hist(f, mvf1.history, s)
        to_return.append(f)

    return tuple(to_return)
