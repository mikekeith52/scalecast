import typing
from typing import Dict, Union

from ipywidgets import widgets
from IPython.display import display, clear_output

from scalecast.Forecaster import Forecaster, _determine_best_by_, _estimators_

import matplotlib.pyplot as plt
import seaborn as sns

from tqdm.notebook import tqdm as log_progress

def results_vis(f_dict: Dict[str,Forecaster],plot_type: str='forecast', print_attr: list = [], include_train: Union[bool,int] = True) -> None:
    """ visualize the forecast results from many different Forecaster objects leveraging Jupyter widgets.

    Args:
        f_dict (dict[str,Forecaster]): dictionary of forcaster objects.
            works best if two or more models have been evaluated in each dictionary value.
        plot_type (str): one of {"forecast","test"}, default "forecast".
            the type of results to visualize.
        print_attr (list): optional.
            the attributes from history to print.
            passed to print_attr parameter when plot_type = 'forecast'.
            ignored when plot_type = 'test'.
        include_train (bool or int): optional.
            whether to include the complete training set in the plot or how many traning-set observations to include.
            passed to include_train parameter when plot_type = 'test'.
            ignored when plot_type = 'forecast'.

    Returns:
        None 
    """
    if plot_type not in {'forecast','test'}:
        raise ValueError(f'plot_type must be "forecast" or "test", got {plot_type}')

    def display_user_selections(ts_selection,mo_selection,lv_selection,ci_selection,me_selection):
        selected_data = f_dict[ts_selection]
        print(ts_selection)
        if plot_type == 'forecast':
            selected_data.plot(models=f'top_{mo_selection}',order_by=me_selection,level=lv_selection,
                               print_attr=print_attr,ci=ci_selection)
        else:
            selected_data.plot_test_set(models=f'top_{mo_selection}',order_by=me_selection,include_train=include_train,level=lv_selection,
                                        ci=ci_selection)
            
    def on_button_clicked(b):
        mo_selection = mo_dd.value
        ts_selection = ts_dd.value
        lv_selection = lv_dd.value
        ci_selection = ci_dd.value
        me_selection = me_dd.value
        with output:
            clear_output()
            display_user_selections(ts_selection,mo_selection,lv_selection,ci_selection,me_selection)
    
    all_models = []
    for k,f in f_dict.items():
        all_models += [fcst for fcst in f.history.keys() if fcst not in all_models]
    ts_dd = widgets.Dropdown(options=f_dict.keys(), description = 'Time Series:')
    mo_dd = widgets.Dropdown(options=range(1,len(all_models)+1), description = 'No. Models')
    lv_dd = widgets.Dropdown(options=[True,False],description='View Level')
    ci_dd = widgets.Dropdown(options=[True,False],description='View Confidence Intervals')
    me_dd = widgets.Dropdown(options=sorted(_determine_best_by_)
        ,description='Order By')

    # never changes
    button = widgets.Button(description="Select Time Series")
    output = widgets.Output()

    display(ts_dd,mo_dd,lv_dd,ci_dd,me_dd)
    display(button, output)
    
    button.on_click(on_button_clicked)

def tune_test_forecast(forecaster,models,dynamic_tuning=False,dynamic_testing=True,summary_stats=False,feature_importance=False):
    """ tunes, tests, and forecasts a series of models with a progress bar through tqdm.

    Args:
        forecaster (Forecaster): the object to visualize.
        models (list-like):
            each element must be in _can_be_tuned_.
        dynamic_tuning (bool): default False.
            whether to dynamically tune the forecast (meaning AR terms will be propogated with predicted values).
            setting this to False means faster performance, but gives a less-good indication of how well the forecast will perform out x amount of periods.
            when False, metrics effectively become an average of one-step forecasts.
        dynamic_testing (bool): default True.
            whether to dynamically test the forecast (meaning AR terms will be propogated with predicted values).
            setting this to False means faster performance, but gives a less-good indication of how well the forecast will perform out x amount of periods.
            when False, test-set metrics effectively become an average of one-step forecasts.
        summary_stats (bool): default False.
            whether to save summary stats for the models that offer those.
        feature_importance (bool): default False.
            whether to save permutation feature importance information for the models that offer those.

    Returns:
        None
    """
    if len([m for m in models if m not in [m for m in _estimators_ if m != 'combo']]) > 0:
        raise ValueError('values passed to models must be list-like and in {}'.format([m for m in _estimators_ if m != 'combo']))
    for m in log_progress(models):
        forecaster.set_estimator(m)
        forecaster.tune(dynamic_tuning=dynamic_tuning)
        forecaster.auto_forecast(dynamic_testing=dynamic_testing)

        if summary_stats:
            forecaster.save_summary_stats()
        if feature_importance:
            forecaster.save_feature_importance()