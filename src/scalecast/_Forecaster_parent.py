from .__init__ import (
    __sklearn_imports__,
    __sklearn_estimators__,
    __non_sklearn_estimators__,
    __estimators__,
    __cannot_be_tuned__,
    __can_be_tuned__,
    __metrics__,
    __normalizer__,
    __colors__,
    __not_hyperparams__,
)
from ._utils import _developer_utils
from .util import metrics
import copy
import pandas as pd
import numpy as np
import importlib
import warnings
import logging

# descriptive errors

class ForecastError(Exception):
    pass

class Forecaster_parent:
    def __init__(
        self,
        y,
        test_length,
        cis,
        metrics,
        **kwargs,
    ):
        logging.basicConfig(filename="warnings.log", level=logging.WARNING)
        self.y = y
        self.sklearn_imports = __sklearn_imports__
        self.sklearn_estimators = __sklearn_estimators__
        self.estimators = __sklearn_estimators__ # this will be overwritten with Forecaster but maintained in MVForecaster
        self.can_be_tuned = __sklearn_estimators__ # this will be overwritten with Forecaster but maintained in MVForecaster
        self.normalizer = __normalizer__
        self.set_test_length(test_length)
        self.validation_length = 1
        self.validation_metric = metrics[0]
        self.cis = cis
        self.cilevel = 0.95
        self.current_xreg = {} # Series
        self.future_xreg = {} # lists
        self.history = {}
        self.metrics = {k:__metrics__[k] for k in metrics}
        self.set_estimator("mlr")
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __copy__(self):
        if hasattr(self,'tf_model'):
            delattr(self,'tf_model')
        obj = type(self).__new__(self.__class__)
        obj.__dict__.update(self.__dict__)
        return obj

    def __deepcopy__(self):
        if hasattr(self,'tf_model'):
            delattr(self,'tf_model')
        obj = type(self).__new__(self.__class__)
        obj.__dict__ = copy.deepcopy(self.__dict__)
        return obj

    def _check_right_test_length_for_cis(self,cilevel):
        min_test_length = np.ceil(1/(1-cilevel))
        if self.test_length < min_test_length:
            raise ValueError(
                'Cannot evaluate confidence intervals at the '
                '{:.0%} level when test_length is set to less than {} observations. '
                'The test length is currently set to {} observations. '
                'The test length must be at least 1/(1-cilevel) in length for conformal intervals to work.'.format(
                    cilevel,
                    int(min_test_length),
                    self.test_length,
                )
            )

    def _clear_the_deck(self):
        """ deletes the following attributes to prepare a new forecast:
            'univariate','fitted_values','regr','X','feature_importance','summary_stats','models','weights'
        """
        for attr in (
            "fitted_values",
            "regr",
            "X",
            "Xvars",
            "feature_importance",
            "perm",
            "summary_stats",
            "models",
            "weights",
            "tf_model",
            "scaler",
        ):
            try:
                delattr(self, attr)
            except AttributeError:
                pass

    def copy(self):
        """ Creates an object copy.
        """
        return self.__copy__()

    def deepcopy(self):
        """ Creates an object deepcopy.
        """
        return self.__deepcopy__()

    def add_sklearn_estimator(self, imported_module, called):
        """ Adds a new estimator from scikit-learn not built-in to the forecaster object that can be called using set_estimator().
        Only regression models are accepted.
        
        Args:
            imported_module (scikit-learn regression model):
                The model from scikit-learn to add. Must have already been imported locally.
                Supports models from sklearn and sklearn APIs.
            called (str):
                The name of the estimator that can be called using set_estimator().

        Returns:
            None

        >>> from sklearn.ensemble import StackingRegressor
        >>> f.add_sklearn_estimator(StackingRegressor,called='stacking')
        >>> f.set_estimator('stacking')
        >>> f.manual_forecast(...)
        """
        self.sklearn_imports[called] = imported_module
        self.sklearn_estimators.append(called)
        self.estimators.append(called)
        self.can_be_tuned.append(called)

    def auto_forecast(
        self,
        call_me=None,
        dynamic_testing=True,
        test_only=False,
    ):
        """ Auto forecasts with the best parameters indicated from the tuning process.

        Args:
            call_me (str): Optional.
                What to call the model when storing it in the object's history dictionary.
                If not specified, the model's nickname will be assigned the estimator value ('mlp' will be 'mlp', etc.).
                Duplicated names will be overwritten with the most recently called model.
            dynamic_testing (bool or int): Default True.
                Whether to dynamically/recursively test the forecast (meaning AR terms will be propagated with predicted values).
                If True, evaluates dynamically over the entire out-of-sample slice of data.
                If int, window evaluates over that many steps (2 for 2-step dynamic forecasting, 12 for 12-step, etc.).
                Setting this to False or 1 means faster performance, 
                but gives a less-good indication of how well the forecast will perform more than one period out.
                The model will skip testing if the test_length attribute is set to 0.
            test_only (bool): Default False.
                Whether to stop the model after the testing process and not forecast into future periods.
                The forecast info stored in the object's history will be equivalent to test-set predictions.
                When True, any plot or export of forecasts into a future horizon will fail 
                and not all methods will raise descriptive errors.
                Will automatically change to True if object was initiated with require_future_dates = False.
                Must always be False for multivariate forecasting.

        Returns:
            None

        >>> f.set_estimator('xgboost')
        >>> f.tune()
        >>> f.auto_forecast()
        """
        if not hasattr(self, "best_params"):
            warnings.warn(
                f"Since tune() or cross_validate() has not been called, {self.estimator} model will be run with default hyperparameters.",
                category = Warning,
            )
            self.best_params = {}
        self.manual_forecast(
            call_me=call_me,
            dynamic_testing=dynamic_testing,
            test_only=test_only,
            **self.best_params,
        )

    def eval_cis(self,mode=True,cilevel=.95):
        """ Call this function to change whether or not the Forecaster sets confidence intervals on all evaluated models.
        Beginning 0.17.0, only conformal confidence intervals are supported. Conformal intervals need a test set to be configured soundly.
        Confidence intervals cannot be evaluated when there aren't at least 1/(1-cilevel) observations in the test set.

        Args:
            mode (bool): Default True. Whether to set confidence intervals on or off for models.
            cilevel (float): Default .95. Must be greater than 0, less than 1. The confidence level
                to use to set intervals.
        """
        if mode is True:
            self._check_right_test_length_for_cis(cilevel)
        
        self.cis=mode
        self.set_cilevel(cilevel)

    def ingest_grid(self, grid):
        """ Ingests a grid to tune the estimator.

        Args:
            grid (dict or str):
                If dict, must be a user-created grid.
                If str, must match the name of a dict grid stored in a grids file.

        Returns:
            None

        >>> f.set_estimator('mlr')
        >>> f.ingest_grid({'normalizer':['scale','minmax']})
        """
        from itertools import product

        def expand_grid(d):
            return pd.DataFrame([row for row in product(*d.values())], columns=d.keys())

        try:
            if isinstance(grid, str):
                Grids = importlib.import_module(self.grids_file)
                importlib.reload(Grids)
                grid = getattr(Grids, grid)
        except SyntaxError:
            raise
        except:
            raise ForecastError(
                f"Tried to load a grid called {self.estimator} from {self.grids_file}.py, "
                "but either the file could not be found in the current directory, "
                "there is no grid with that name, or the dictionary values are not list-like. "
                "Try the ingest_grid() method with a dictionary grid passed manually."
            )
        grid = expand_grid(grid)
        self.grid = grid

    def limit_grid_size(self, n, random_seed=None):
        """ Makes a grid smaller randomly.

        Args:
            n (int or float):
                If int, randomly selects that many parameter combinations.
                If float, must be less than 1 and greater 0, randomly selects that percentage of parameter combinations.
            random_seed (int): Optional.
                Set a seed to make results consistent.

        Returns:
            None

        >>> from scalecast import GridGenerator
        >>> GridGenerator.get_example_grids()
        >>> f.set_estimator('mlp')
        >>> f.ingest_grid('mlp')
        >>> f.limit_grid_size(10,random_seed=20) # limits grid to 10 iterations
        >>> f.limit_grid_size(.5,random_seed=20) # limits grid to half its original size
        """
        if random_seed is not None:
            random.seed(random_seed)

        if n >= 1:
            self.grid = self.grid.sample(n=min(n, self.grid.shape[0])).reset_index(
                drop=True
            )
        elif (n < 1) & (n > 0):
            self.grid = self.grid.sample(frac=n).reset_index(drop=True)
        else:
            raise ValueError(f"argment passed to n not usable: {n}")

    def set_metrics(self,metrics):
        """ Set or change the evaluated metrics for all model testing and validation.

        Args:
            metrics (list): The metrics to evaluate when validating
                and testing models. Each element must exist in utils.metrics and take only two arguments: a and f.
                See https://scalecast.readthedocs.io/en/latest/Forecaster/Util.html#metrics.
                For each metric and model that is tested, the test-set and in-sample metrics will be evaluated and can be
                exported. Level test-set and in-sample metrics are also currently available, but will be removed in a future version.
        """
        bad_metrics = [met for met in metrics if met not in __metrics__]
        if len(bad_metrics) > 0:
            raise ValueError(
                f'Each element in metrics must be one of {__metrics__}.'
                f' Got the following invalid values: {bad_metrics}.'
            )
        self.metrics = {k:__metrics__[k] for k in metrics}
        self.determine_best_by = _developer_utils._determine_best_by(self.metrics)

    def set_validation_metric(self, metric):
        """ Sets the metric that will be used to tune all subsequent models.

        Args:
            metric: One of Forecaster.metrics.
                The metric to optimize the models with using the validation set.
                Although model testing will evaluate all metrics in Forecaster.metrics,
                model optimization with tuning and cross validation only uses one of these.

        Returns:
            None

        >>> f.set_validation_metric('mae')
        """
        _developer_utils.descriptive_assert(
            metric in self.metrics,
            ValueError,
            f"metric must be one of {list(self.metrics)}, got {metric}.",
        )
        self.validation_metric = metric

    def set_test_length(self, n=1):
        """ Sets the length of the test set. As of version 0.16.0, 0-length test sets are supported.

        Args:
            n (int or float): Default 1.
                The length of the resulting test set.
                Pass 0 to skip testing models.
                Fractional splits are supported by passing a float less than 1 and greater than 0.

        Returns:
            None

        >>> f.set_test_length(12) # test set of 12
        >>> f.set_test_length(.2) # 20% test split
        """
        float(n)
        if n == 0:
            self.test_length = 0
        if n >= 1:
            n = int(n)
            _developer_utils.descriptive_assert(
                isinstance(n, int),
                ValueError,
                f"n must be an int of at least 0 or float greater than 0 and less than 1, got {n} of type {type(n)}.",
            )
            self.test_length = n
        else:
            _developer_utils.descriptive_assert(
                n >= 0,
                ValueError,
                f"n must be an int of at least 0 or float greater than 0 and less than 1, got {n} of type {type(n)}.",
            )
            self.test_length = int(len(self.y) * n)

    def set_validation_length(self, n=1):
        """ Sets the length of the validation set. This will never matter for models that are not tuned.

        Args:
            n (int): Default 1.
                The length of the resulting validation set.

        Returns:
            None

        >>> f.set_validation_length(6) # validation length of 6
        """
        n = int(n)
        _developer_utils.descriptive_assert(n > 0, ValueError, f"n must be greater than 0, got {n}.")
        if (self.validation_metric == "r2") & (n == 1):
            raise ValueError(
                "Can only set a validation_length of 1 if validation_metric is not r2. Try calling set_validation_metric() with a different metric."
                f"Possible values are: {_metrics_}."
            )
        self.validation_length = n

    def set_cilevel(self, n):
        """ Sets the level for the resulting confidence intervals (95% default).

        Args:
            n (float): Greater than 0 and less than 1.

        Returns:
            None

        >>> f.set_cilevel(.80) # next forecast will get 80% confidence intervals
        """
        _developer_utils.descriptive_assert(
            n < 1 and n > 0, ValueError, "n must be a float greater than 0 and less than 1."
        )
        self.cilevel = n

    def set_estimator(self, estimator):
        """ Sets the estimator to forecast with.

        Args:
            estimator (str): One of Forecaster.estimators.

        Returns:
            None

        >>> f.set_estimator('lasso')
        >>> f.manual_forecast(alpha = .5)
        """
        _developer_utils._check_if_correct_estimator(estimator,self.estimators)
        self.determine_best_by = _developer_utils._determine_best_by(self.metrics)
        if hasattr(self, "estimator"):
            if estimator != self.estimator:
                for attr in (
                    "grid",
                    "grid_evaluated",
                    "best_params",
                    "validation_metric_value",
                ):
                    if hasattr(self, attr):
                        delattr(self, attr)
                self._clear_the_deck()
                self.estimator = estimator
        else:
            self.estimator = estimator

    def set_grids_file(self,name='Grids'):
        """ Sets the name of the file where the object will look automatically for grids when calling 
        `tune()`, `cross_validate()`, `tune_test_forecast()`, or similar function.
        If the grids file does not exist in the working directory, the error will only be raised once tuning is called.
        
        Args:
            name (str): Default 'Grids'.
                The name of the file to look for.
                This file must exist in the working directory.
                The default will look for a file called "Grids.py".

        >>> f.set_grids_file('ModGrids') # expects to find a file called ModGrids.py in working directory.
        """
        _developer_utils.descriptive_assert(isinstance(name,str),ValueError,f'name argument expected str type, got {type(name)}')
        self.grids_file = name

def _tune_test_forecast(
    f,
    models,
    cross_validate,
    dynamic_tuning,
    dynamic_testing,
    limit_grid_size,
    suffix,
    error,
    summary_stats = False,
    feature_importance = False,
    fi_method=None,
    tqdm = False,
    **cvkwargs,
):
    if tqdm: # notebooks only get progress bar
        from tqdm.notebook import tqdm
    else:
        tqdm = list

    [_developer_utils._check_if_correct_estimator(m,f.can_be_tuned) for m in models]
    for m in tqdm(models):
        call_me = m if suffix is None else m + suffix
        f.set_estimator(m)
        if limit_grid_size is not None:
            f.ingest_grid(m)
            f.limit_grid_size(limit_grid_size)
        if cross_validate:
            f.cross_validate(dynamic_tuning=dynamic_tuning, **cvkwargs)
        else:
            f.tune(dynamic_tuning=dynamic_tuning)
        try:
            f.auto_forecast(
                dynamic_testing=dynamic_testing,
                call_me=call_me,
            )
        except Exception as e:
            if error == 'raise':
                raise
            elif error == 'warn':
                warnings.warn(
                    f"{m} model could not be evaluated. "
                    f"Here's the error: {e}",
                    category=Warning,
                )
                continue
            elif error == 'ignore':
                continue
            else:
                raise ValueError(f'Value passed to error arg not recognized: {error}')
        if summary_stats:
            f.save_summary_stats()
        if feature_importance:
            f.save_feature_importance(fi_method)