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