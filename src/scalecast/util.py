from scalecast.Forecaster import Forecaster
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_reduction_errors(f):
    """ plots the resulting error/accuracy of a Forecaster object where reduce_Xvars() method has been called
    with method = 'pfi'.
    
    Args:
        f (Forecaster): an object that has called the reduce_Xvars() method with method = 'pfi'.
        
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
                if k2 in (''):
                    continue
                elif not isinstance(v2,dict) or k2 == "HyperParams":
                    hist[k][k2] = v2
                elif isinstance(v2,dict):
                    hist[k][k2] = list(v2.values())[series_num]
            hist[k]['TestOnly'] = False
        return hist

                    
    to_return = []
    for s in range(mvf.n_series):
        f = Forecaster(
            y = getattr(mvf,f'series{s+1}')['y'],
            current_dates = mvf.current_dates,
            integration = getattr(mvf,f'series{s+1}')['integration'],
            levely = getattr(mvf,f'series{s+1}')['levely'],
            future_dates = mvf.future_dates,
            current_xreg = mvf.current_xreg,
            future_xreg = mvf.future_xreg,
            test_length = mvf.test_length,
            validation_length = mvf.validation_length,
        )
        f.history = convert_mv_hist(f, mvf.history, s)
        to_return.append(f)

    return tuple(to_return)