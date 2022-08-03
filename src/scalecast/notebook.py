import typing
from typing import Dict, Union

from ipywidgets import widgets
from IPython.display import display, clear_output

from scalecast.Forecaster import Forecaster, _determine_best_by_, _estimators_

import matplotlib.pyplot as plt
import seaborn as sns

from tqdm.notebook import tqdm as log_progress


def results_vis(
    f_dict: Dict[str, Forecaster],
    plot_type: str = "forecast",
    print_attr: list = [],
    include_train: Union[bool, int] = True,
) -> None:
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
    if plot_type not in {"forecast", "test"}:
        raise ValueError(f'plot_type must be "forecast" or "test", got {plot_type}')

    def display_user_selections(
        ts_selection, mo_selection, lv_selection, ci_selection, me_selection
    ):
        selected_data = f_dict[ts_selection]
        if plot_type == "forecast":
            selected_data.plot(
                models=f"top_{mo_selection}",
                order_by=me_selection,
                level=lv_selection,
                print_attr=print_attr,
                ci=ci_selection,
            )
        else:
            selected_data.plot_test_set(
                models=f"top_{mo_selection}",
                order_by=me_selection,
                include_train=include_train,
                level=lv_selection,
                ci=ci_selection,
            )
        plt.title(ts_selection + " Forecast Results", size=16)
        plt.show()

    def on_button_clicked(b):
        mo_selection = mo_dd.value
        ts_selection = ts_dd.value
        lv_selection = lv_dd.value
        ci_selection = ci_dd.value
        me_selection = me_dd.value
        with output:
            clear_output()
            display_user_selections(
                ts_selection, mo_selection, lv_selection, ci_selection, me_selection
            )

    all_models = []
    for k, f in f_dict.items():
        all_models += [fcst for fcst in f.history.keys() if fcst not in all_models]
    ts_dd = widgets.Dropdown(options=f_dict.keys(), description="Time Series:")
    mo_dd = widgets.Dropdown(
        options=range(1, len(all_models) + 1), description="No. Models"
    )
    lv_dd = widgets.Dropdown(options=[True, False], description="View Level")
    ci_dd = widgets.Dropdown(
        options=[True, False], description="View Confidence Intervals"
    )
    me_dd = widgets.Dropdown(
        options=sorted(_determine_best_by_), description="Order By"
    )

    # never changes
    button = widgets.Button(description="Select Time Series")
    output = widgets.Output()

    display(ts_dd, mo_dd, lv_dd, ci_dd, me_dd)
    display(button, output)

    button.on_click(on_button_clicked)


def results_vis_mv(f_dict, plot_type="forecast", include_train=True):
    """ visualize the forecast results from many different MVForecaster objects leveraging Jupyter widgets.

    Args:
        f_dict (dict[str,MVForecaster]): dictionary of forcaster objects.
            works best if two or more models have been evaluated in each dictionary value.
        plot_type (str): one of {"forecast","test"}, default "forecast".
            the type of results to visualize.
        include_train (bool or int): optional.
            whether to include the complete training set in the plot or how many traning-set observations to include.
            passed to include_train parameter when plot_type = 'test'.
            ignored when plot_type = 'forecast'.

    Returns:
        None
    """
    if plot_type not in {"forecast", "test"}:
        raise ValueError(f'plot_type must be "forecast" or "test", got {plot_type}')

    def display_user_selections(
        mo_selection, ts_selection, lv_selection, ci_selection, se_selection
    ):
        selected_data = f_dict[ts_selection]
        if plot_type == "forecast":
            selected_data.plot(
                models=mo_selection,
                series=se_selection,
                level=lv_selection,
                ci=ci_selection,
            )
        else:
            selected_data.plot_test_set(
                models=mo_selection,
                series=se_selection,
                level=lv_selection,
                ci=ci_selection,
                include_train=include_train,
            )
        plt.title(ts_selection + " Forecast Results", size=16)
        plt.show()

    def on_button_clicked(b):
        mo_selection = mo_se.value
        ts_selection = ts_dd.value
        lv_selection = lv_dd.value
        ci_selection = ci_dd.value
        se_selection = se_se.value
        with output:
            clear_output()
            display_user_selections(
                mo_selection, ts_selection, lv_selection, ci_selection, se_selection
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
    lv_dd = widgets.Dropdown(options=[True, False], description="View Level")
    ci_dd = widgets.Dropdown(
        options=[True, False], description="View Confidence Intervals"
    )

    # never changes
    button = widgets.Button(description="Select Time Series")
    output = widgets.Output()

    display(ts_dd, se_se, mo_se, lv_dd, ci_dd)
    display(button, output)

    button.on_click(on_button_clicked)


def tune_test_forecast(
    f,
    models,
    cross_validate=False,
    dynamic_tuning=False,
    dynamic_testing=True,
    probabilistic=False,
    n_iter=20,
    summary_stats=False,
    feature_importance=False,
    fi_method="pfi",
    limit_grid_size=None,
    suffix=None,
    **cvkwargs,
):
    """ tunes, tests, and forecasts a series of models with a progress bar through tqdm.

    Args:
        f (Forecaster or MVForecaster): the object to run the models through.
        models (list-like):
            each element must be in _can_be_tuned_.
        cross_validate (bool): default False
                whether to tune the model with cross validation. 
                if False, uses the validation slice of data to tune.
        dynamic_tuning (bool or int): default False.
            whether to dynamically tune the forecast (meaning AR terms will be propogated with predicted values).
            if True, evaluates dynamically over the entire out-of-sample slice of data.
            if int, window evaluates over that many steps (2 for 2-step dynamic forecasting, 12 for 12-step, etc.).
            setting this to False or 1 means faster performance, 
            but gives a less-good indication of how well the forecast will perform out x amount of periods.
        dynamic_testing (bool or int):
            whether to dynamically test the forecast (meaning AR terms will be propogated with predicted values).
            if True, evaluates dynamically over the entire out-of-sample slice of data.
            if int, window evaluates over that many steps (2 for 2-step dynamic forecasting, 12 for 12-step, etc.).
            setting this to False or 1 means faster performance, 
            but gives a less-good indication of how well the forecast will perform out x amount of periods.
        probabilistic (bool): default False.
            whether to use a probabilistic forecasting process to set confidence intervals.
        n_iter (int): default 20.
            how many iterations to use in probabilistic forecasting. ignored if probabilistic = False.
        summary_stats (bool): default False.
            whether to save summary stats for the models that offer those.
            does not work for `MVForecaster` objects.
        feature_importance (bool): default False.
            whether to save permutation feature importance information for the models that offer those.
            does not work for `MVForecaster` objects.
        fi_method (str): one of {'pfi','shap'}, default 'pfi'.
            the type of feature importance to save for the models that support it.
            ignored if feature_importance is False.
            does not work for `MVForecaster` objects.
        limit_grid_size (int or float): optional. pass an argument here to limit each of the grids being read.
            see https://scalecast.readthedocs.io/en/latest/Forecaster/Forecaster.html#src.scalecast.Forecaster.Forecaster.limit_grid_size
        suffix (str): optional. a suffix to add to each model as it is evaluate to differentiate them when called
            later. if unspecified, each model can be called by its estimator name.
        **cvkwargs: passed to the cross_validate() method.

    Returns:
        None
    """
    if (
        len([m for m in models if m not in [m for m in _estimators_ if m != "combo"]])
        > 0
    ):
        raise ValueError(
            "values passed to models must be list-like and in {}".format(
                [m for m in _estimators_ if m != "combo"]
            )
        )
    for m in log_progress(models):
        call_me = m if suffix is None else m + suffix
        f.set_estimator(m)
        if limit_grid_size is not None:
            f.ingest_grid(m)
            f.limit_grid_size(limit_grid_size)
        if cross_validate:
            f.cross_validate(dynamic_tuning=dynamic_tuning, **cvkwargs)
        else:
            f.tune(dynamic_tuning=dynamic_tuning)
        f.auto_forecast(
            dynamic_testing=dynamic_testing,
            call_me=call_me,
            probabilistic=probabilistic,
            n_iter=n_iter,
        )

        if summary_stats:
            f.save_summary_stats()
        if feature_importance:
            f.save_feature_importance(fi_method)
