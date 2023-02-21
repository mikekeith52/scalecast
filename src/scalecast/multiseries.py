import pandas as pd


def export_model_summaries(f_dict, **kwargs):
    """ Exports a pandas dataframe with information about each model run on each
    eries when doing forecasting using many different series.

    Args:
        f_dict (dict[str,Forecaster]): Dictionary of forcaster objects.
        **kwargs: Passed to the Forecaster.export() function (do not pass dfs arg as that is set automatically to 'model_summaries').

    Returns:
        (DataFrame) The combined model summaries.
    """
    forecast_info = pd.DataFrame()
    for k, f in f_dict.items():
        df = f.export(dfs="model_summaries", **kwargs)
        df["Series"] = k
        forecast_info = pd.concat([forecast_info, df], ignore_index=True)
    return forecast_info


def keep_smallest_first_date(*fs):
    """ Trims all passed Forecaster objects so they all have the same first date.
    
    Args:
        *fs (Forecaster objects): The objects to check and trim.

    Returns:
        None
    """
    first_date = max([min(f.current_dates) for f in fs])
    for f in fs:
        f.keep_smaller_history(first_date)
