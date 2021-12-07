import typing
from typing import Dict, Union

from ipywidgets import widgets
from IPython.display import display, clear_output

from scalecast.Forecaster import Forecaster
from scalecast.Forecaster import _determine_best_by_

import matplotlib.pyplot as plt
import seaborn as sns

def results_vis(f_dict: Dict[str,Forecaster],plot_type: str='forecast', print_attr: list = [], include_train: Union[bool,int] = True) -> None:
    """ visualize the forecast results leveraging widgets
        Parameters:
            f_dict: dictionary of forcaster objects
            plot_type: one of {"forecast","test"}, default "forecast"
                the type of results to visualize
            print_attr: list, optional
                the attributes from history to print
                passed to print_attr parameter when plot_type = 'forecast'
                ignored when plot_type = 'test'
            include_train: bool or int, optional
                whether to include the complete training set in the results or how many traning-set observations to include
                passed to include_train parameter when plot_type = 'test'
                ignored when plot_type = 'forecast'
    """
    def display_user_selections(ts_selection,mo_selection,lv_selection,me_selection):
        selected_data = f_dict[ts_selection]
        print(ts_selection)
        if plot_type == 'forecast':
            selected_data.plot(models=f'top_{mo_selection}',order_by=me_selection,level=lv_selection,
                               print_attr=print_attr)
        elif plot_type == 'test':
            selected_data.plot_test_set(models=f'top_{mo_selection}',order_by=me_selection,include_train=include_train,level=lv_selection)

    def on_button_clicked(b):
        mo_selection = mo_dd.value
        ts_selection = ts_dd.value
        lv_selection = lv_dd.value
        me_selection = me_dd.value
        with output:
            clear_output()
            display_user_selections(ts_selection,mo_selection,lv_selection,me_selection)
    
    all_models = []
    for k,f in f_dict.items():
        all_models += [fcst for fcst in f.history.keys() if fcst not in all_models]
    ts_dd = widgets.Dropdown(options=f_dict.keys(), description = 'Time Series:')
    mo_dd = widgets.Dropdown(options=range(1,len(all_models)+1), description = 'No. Models')
    lv_dd = widgets.Dropdown(options=[True,False],description='View Level')
    me_dd = widgets.Dropdown(options=sorted([e for e in _determine_best_by_ if e is not None])
        ,description='Order By')

    # never changes
    button = widgets.Button(description="Select Time Series")
    output = widgets.Output()

    display(ts_dd,mo_dd,lv_dd,me_dd)
    display(button, output)
    
    button.on_click(on_button_clicked)