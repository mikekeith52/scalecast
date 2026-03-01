from ._utils import _tune_test_forecast
from .Forecaster import Forecaster
from .MVForecaster import MVForecaster
from ipywidgets import widgets
from IPython.display import display, clear_output
import matplotlib.pyplot as plt
from typing import Literal, Optional, Any
from .types import PositiveInt, ModelValues, DynamicTesting, FIMethod, ConfInterval

def results_vis(
    f_dict: dict[str,Forecaster],
    plot_type: Literal['forecast','test'] = "forecast",
    include_train: bool|PositiveInt = True,
    figsize:tuple[int,int] = (12,6),
):
    """ Visualize the forecast results from many different Forecaster objects leveraging Jupyter widgets.

    Args:
        f_dict (dict[str,Forecaster]): Dictionary of forcaster objects.
            Works best if two or more models have been evaluated in each dictionary value.
        plot_type (str): One of {"forecast","test"}. Default "forecast".
            The type of results to visualize.
        include_train (bool or int): Optional.
            Whether to include the complete training set in the plot or how many traning-set observations to include.
            Passed to include_train parameter when plot_type = 'test'.
            Ignored when plot_type = 'forecast'.
        figsize (tuple): Default (12,6). Size of the resulting figure.

    Returns:
        None 
    """
    if plot_type not in {"forecast", "test"}:
        raise ValueError(f'plot_type must be "forecast" or "test", got {plot_type}')

    def display_user_selections(
        ts_selection, mo_selection, ex_selection, ci_selection, me_selection
    ):
        selected_data = f_dict[ts_selection]
        if plot_type == "forecast":
            selected_data.plot(
                models=f"top_{mo_selection}",
                exclude=ex_selection,
                order_by=me_selection,
                ci=ci_selection,
                figsize=figsize,
            )
        else:
            selected_data.plot_test_set(
                models=f"top_{mo_selection}",
                exclude=ex_selection,
                order_by=me_selection,
                include_train=include_train,
                ci=ci_selection,
                figsize=figsize,
            )
        plt.title(str(ts_selection) + " Forecast Results", size=16)
        plt.show()

    def on_button_clicked(b):
        mo_selection = mo_dd.value
        ex_selection = ex_se.value
        ts_selection = ts_dd.value
        ci_selection = ci_dd.value
        me_selection = me_dd.value
        with output:
            clear_output()
            display_user_selections(
                ts_selection, 
                mo_selection, 
                ex_selection, 
                ci_selection, 
                me_selection,
            )

    all_models = []
    for k, f in f_dict.items():
        all_models += [fcst for fcst in f.history.keys() if fcst not in all_models]
    ts_dd = widgets.Dropdown(options=f_dict.keys(), description="Time Series:")
    mo_dd = widgets.Dropdown(
        options=range(1, len(all_models) + 1), description="No. Models"
    )
    ex_se = widgets.SelectMultiple(
        options=all_models, description="Exclude"
    )
    ci_dd = widgets.Dropdown(
        options=[False, True], description="View Confidence Intervals"
    )
    me_dd = widgets.Dropdown(
        options=sorted(f.determine_best_by), # f will be last object iterated through above
        description="Order By", 
        value = [m for m in f.determine_best_by if m.startswith('TestSet')][0],
    )

    # never changes
    button = widgets.Button(description="Select Time Series")
    output = widgets.Output()

    display(ts_dd, mo_dd, ex_se, ci_dd, me_dd)
    display(button, output)

    button.on_click(on_button_clicked)


def results_vis_mv(
    f_dict: dict[str,MVForecaster], 
    plot_type:Literal['forecast','test']="forecast", 
    include_train:bool|PositiveInt=True, 
    figsize:tuple[int,int] = (12,6)
):
    """ Visualize the forecast results from many different MVForecaster objects leveraging Jupyter widgets.

    Args:
        f_dict (dict[str,MVForecaster]): Dictionary of forcaster objects.
            Works best if two or more models have been evaluated in each dictionary value.
        plot_type (str): One of {"forecast","test"}. Default "forecast".
            The type of results to visualize.
        include_train (bool or int): Optional.
            Whether to include the complete training set in the plot or how many traning-set observations to include.
            Passed to include_train parameter when plot_type = 'test'.
            Ignored when plot_type = 'forecast'.
        figsize (tuple): Default (12,6). Size of the resulting figure.

    Returns:
        None
    """
    if plot_type not in {"forecast", "test"}:
        raise ValueError(f'plot_type must be "forecast" or "test", got {plot_type}')

    def display_user_selections(
        mo_selection, ts_selection, ci_selection, se_selection
    ):
        selected_data = f_dict[ts_selection]
        if plot_type == "forecast":
            selected_data.plot(
                models=mo_selection,
                series=se_selection,
                ci=ci_selection,
                figsize=figsize,
            )
        else:
            selected_data.plot_test_set(
                models=mo_selection,
                series=se_selection,
                ci=ci_selection,
                include_train=include_train,
                figsize=figsize,
            )
        plt.title(ts_selection + " Forecast Results", size=16)
        plt.show()

    def on_button_clicked(b):
        mo_selection = mo_se.value
        ts_selection = ts_dd.value
        ci_selection = ci_dd.value
        se_selection = se_se.value
        with output:
            clear_output()
            display_user_selections(
                mo_selection, ts_selection, ci_selection, se_selection
            )

    all_models = []
    n_series = 2
    for k, f in f_dict.items():
        all_models += [fcst for fcst in f.history.keys() if fcst not in all_models]
        n_series = max(n_series, f.n_series)
    series = [f"series{i+1}" for i in range(n_series)]
    ts_dd = widgets.Dropdown(options=f_dict.keys(), description="Time Series:")
    mo_se = widgets.SelectMultiple(
        options=all_models, description="Models", selected=all_models
    )
    se_se = widgets.SelectMultiple(
        options=series, description="Series", selected=series
    )
    ci_dd = widgets.Dropdown(
        options=[False, True], description="View Confidence Intervals"
    )

    # never changes
    button = widgets.Button(description="Select Time Series")
    output = widgets.Output()

    display(ts_dd, se_se, mo_se, ci_dd)
    display(button, output)

    button.on_click(on_button_clicked)


def tune_test_forecast(
    f:Forecaster|MVForecaster,
    models:ModelValues,
    cross_validate:bool=False,
    dynamic_tuning:DynamicTesting=False,
    dynamic_testing:DynamicTesting=True,
    feature_importance:bool=False,
    fi_try_order:list[FIMethod]=None,
    limit_grid_size:Optional[PositiveInt|ConfInterval]=None,
    min_grid_size:PositiveInt=1,
    suffix:Optional[str]=None,
    error:Literal['raise','warn','ignore']='raise',
    **cvkwargs:Any,
) -> Forecaster|MVForecaster:
    """ Tunes, tests, and forecasts a series of models with a progress bar through tqdm.

    Args:
        f (Forecaster or MVForecaster): The object to run the models through.
        models (list-like):
            Each element must be in Forecaster.can_be_tuned.
        cross_validate (bool): Default False.
            Whether to tune the model with cross validation. 
            If False, uses the validation slice of data to tune.
        dynamic_tuning (bool or int): Default False.
            Whether to dynamically tune the model or, if int, how many forecast steps to dynamically tune it.
        dynamic_testing (bool or int): Default True.
            Whether to dynamically/recursively test the forecast (meaning AR terms will be propogated with predicted values).
            If True, evaluates recursively over the entire out-of-sample slice of data.
            If int, window evaluates over that many steps (2 for 2-step recurvie testing, 12 for 12-step, etc.).
            Setting this to False or 1 means faster performance, 
            but gives a less-good indication of how well the forecast will perform more than one period out.
        feature_importance (bool): Default False.
            Whether to save feature importance information for the models that offer it.
            Does not work for `MVForecaster` objects.
        fi_try_order (list): Optional.
            If the `feature_importance` argument is `True`, what feature importance methods to try? If using a combination
            of tree-based and linear models, for example, it might be good to pass ['TreeExplainer','LinearExplainer']. The default
            will use whatever is specifiec by default in `Forecaster.save_feature_importance()`, which usually ends up being
            the `PermutationExplainer`.
        limit_grid_size (int or float): Optional. Pass an argument here to limit each of the grids being read.
            See https://scalecast.readthedocs.io/en/latest/Forecaster/Forecaster.html#src.scalecast.Forecaster.Forecaster.limit_grid_size.
        min_grid_size (int): Default 1. The smallest grid size to keep. Ignored if limit_grid_size is None.
        suffix (str): Optional. A suffix to add to each model as it is evaluated to differentiate them when called
            later. If unspecified, each model can be called by its estimator name.
        error (str): One of 'ignore','raise','warn'; default 'raise'.
            What to do with the error if a given model fails.
            'warn' prints a warning that the model could not be evaluated.
        **cvkwargs: Passed to the cross_validate() method.

    Returns:
        Forecater or MVForecaster: The same object as the input, but with the models tuned, tested, and forecasted and the results saved in the history attribute.
    """
    _tune_test_forecast(
        f=f,
        models=models,
        cross_validate=cross_validate,
        dynamic_tuning=dynamic_tuning,
        dynamic_testing=dynamic_testing,
        limit_grid_size=limit_grid_size,
        min_grid_size=min_grid_size,
        suffix=suffix,
        error=error,
        feature_importance=feature_importance,
        fi_try_order = fi_try_order,
        use_progress_bar = True,
        **cvkwargs,
    )
    return f
