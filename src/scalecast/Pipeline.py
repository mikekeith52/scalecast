from .Forecaster import Forecaster
from .MVForecaster import MVForecaster
from .util import break_mv_forecaster
from .SeriesTransformer import SeriesTransformer
import numpy as np
import pandas as pd
from typing import List, Tuple, Union, Dict
import typing
import copy

class Transformer:
    def __init__(self,transformers: List[Tuple]):
        """ Initiates the transformer pipeline.

        Args:
            transformers (list[tuple]): A list of transformations to apply to the time series stored in a Forecaster object.
                The tuple's first element should match the name of a transform function from the SeriesTransformer object: 
                https://scalecast.readthedocs.io/en/latest/Forecaster/SeriesTransformer.html.
                Positional and keyword arguments can be passed to these functions.
                If a given tuple is more than 1 in length, the `fit_transform()` method will parse
                Elements after index 0 as positional arguments.
                Keywords are passed as a dictionary in the last position of tuples greater than 1 in length. 
                Therefore, if the last argument in the tuple is a dict type,
                This is assumed to contain the keyword arguments. 
                If the last positional argument you wish to pass happens to be dict type,
                you can eaither pass it as a keyword argument or place an additional (empty) dictionary at the end of the tuple.

        >>> from scalecast.Pipeline import Transformer
        >>> transformer = Transformer(
        >>>     transformers = [
        >>>         ('LogTransform',),
        >>>         ('DiffTransform',1),
        >>>         ('DiffTransform',12),
        >>>     ],
        >>> )
        """
        # validate types (transform str types to tuple)
        for i, transformer in enumerate(transformers):
            if isinstance(transformer,str):
                transformers[i] = (transformer,)
            elif not isinstance(transformer,tuple):
                raise TypeError(f'Expected elements of transformer list to be tuple type, got {type(transformer)}.')

        self.transformers = transformers

    def __repr__(self):
        return (
            "Transformer(\n"
            "  transformers = [\n"
            "    {}\n"
            "  ]\n"
            ")".format(",\n    ".join([str(i) for i in self.transformers]))
        )

    def __str__(self):
        return self.__repr__()

    def __copy__(self):
        obj = type(self).__new__(self.__class__)
        obj.__dict__.update(self.__dict__)
        return obj

    def __deepcopy__(self, memo={}):
        obj = type(self).__new__(self.__class__)
        memo[id(self)] = obj
        for k, v in self.__dict__.items():
            setattr(obj, k, copy.deepcopy(v, memo))
        return obj

    def fit_transform(self,f: Forecaster) -> Forecaster:
        """ Applies the transformation to the series stored in the Forecaster object.

        Args:
            f (Forecaster): The Forecaster object that stores the series that will be transformed.

        Returns:
            (Forecaster): A Forecaster object with the transformed series.

        >>> from scalecast.Pipeline import Transformer
        >>> transformer = Transformer(
        >>>     transformers = [
        >>>         ('LogTransform',),
        >>>         ('DiffTransform',1),
        >>>         ('DiffTransform',12),
        >>>     ],
        >>> )
        >>> f = transformer.fit_transform(f)
        """
        for i, transformer in enumerate(self.transformers):
            if len(transformer) > 1:
                args = [i for i in transformer[1:-1]]
                args += transformer[-1:] if not isinstance(transformer[-1],dict) else []
                kwargs = transformer[-1] if isinstance(transformer[-1],dict) else {}
            else:
                args = []
                kwargs = {}
            
            if i == 0:
                self.base_transformer = SeriesTransformer(f)
            
            f = getattr(self.base_transformer,transformer[0])(*args,**kwargs)
        return f

class Reverter:
    def __init__(self,
        reverters: List[Tuple],
        base_transformer: Union[Transformer,SeriesTransformer]
    ):
        """ Initiates the reverter pipeline.

        Args:
            reverters (list[tuple]): A list of revert funcs to apply to the time series stored in a Forecaster object.
                The tuple's first element should match the name of a revert function from the SeriesTransformer object: 
                https://scalecast.readthedocs.io/en/latest/Forecaster/SeriesTransformer.html.
                Positional and keyword arguments can be passed to these functions.
                If a given tuple is more than 1 in length, the `fit_transform()` method will parse
                elements after index 0 as positional arguments.
                Keywords are passed as a dictionary in the last position of tuples greater than 1 in length. 
                Therefore, if the last argument in the tuple is a dict type,
                this is assumed to contain the keyword arguments. 
                If the last positional argument you wish to pass happens to be dict type,
                You can eaither pass it as a keyword argument or place an additional (empty) dictionary at the end of the tuple.
            base_transformer (Transformer|SeriesTransformer): The object that was used to make the original transformations.
                These objects contain the key information to undifference and unscale the stored data 
                and therefore this argument is required.

        >>> from scalecast.Pipeline import Reverter
        >>> reverter = Reverter(
        >>>     reverters = [
        >>>         ('DiffRevert',12),
        >>>         ('DiffRevert',1),
        >>>         ('LogRevert',),
        >>>     ],
        >>>     base_transformer = transformer,
        >>> )
        """
        # validate types (transform str types to tuple)
        for i, reverter in enumerate(reverters):
            if isinstance(reverter,str):
                reverters[i] = (reverter,)
            elif not isinstance(reverter,tuple):
                raise TypeError(f'Expected elements of reverter list to be tuple type, got {type(reverter)}.')
        
        self.reverters = reverters
        self.base_transformer = base_transformer

    def __repr__(self):
        return (
            "Reverter(\n"
            "  reverters = [\n"
            "    {}\n"
            "  ],\n"
            "  base_transformer = {}\n"
            ")".format(
                ",\n    ".join([str(i) for i in self.reverters]),
                self.base_transformer,
            )
        )

    def __str__(self):
        return self.__repr__()

    def __copy__(self):
        obj = type(self).__new__(self.__class__)
        obj.__dict__.update(self.__dict__)
        return obj

    def __deepcopy__(self, memo={}):
        obj = type(self).__new__(self.__class__)
        memo[id(self)] = obj
        for k, v in self.__dict__.items():
            setattr(obj, k, copy.deepcopy(v, memo))
        return obj

    def copy(self):
        return self.__deepcopy__()

    def fit_transform(self,f: Forecaster, exclude_models = []) -> Forecaster:
        """ Applies the revert function to the series stored in the Forecaster object.

        Args:
            f (Forecaster): The Forecaster object that stores the series that will be reverted.
            exclude_models (list-like): Optional. Models to not revert.

        Returns:
            (Forecaster): A Forecaster object with the reverted series.

        >>> from scalecast.Pipeline import Reverter
        >>> reverter = Reverter(
        >>>     reverters = [
        >>>         ('DiffRevert',12),
        >>>         ('DiffRevert',1),
        >>>         ('LogRevert',),
        >>>     ],
        >>>     base_transformer = transformer,
        >>> )
        >>> f = reverter.fit_transform(f)
        """
        base_transformer = (
            self.base_transformer if not hasattr(self.base_transformer,'base_transformer')
            else self.base_transformer.base_transformer
        )
        base_transformer.f = f
        for reverter in self.reverters:
            if len(reverter) > 1:
                args = [i for i in reverter[1:-1]]
                args += reverter[-1:] if not isinstance(reverter[-1],dict) else []
                kwargs = reverter[-1] if isinstance(reverter[-1],dict) else {}
            else:
                args = []
                kwargs = {}

            f = getattr(base_transformer,reverter[0])(*args, **kwargs, exclude_models = exclude_models)
        return f

class Pipeline_parent:
    def __copy__(self):
        obj = type(self).__new__(self.__class__)
        obj.__dict__.update(self.__dict__)
        return obj

    def __deepcopy__(self, memo={}):
        obj = type(self).__new__(self.__class__)
        memo[id(self)] = obj
        for k, v in self.__dict__.items():
            setattr(obj, k, copy.deepcopy(v, memo))
        return obj

    def _prepare_backtest(
        self,
        *fs,
        n_iter,
        jump_back,
        series_length,
        fcst_length,
        test_length,
        cis,
        cilevel,
    ) -> List[List[Tuple[Forecaster,np.array]]]:
        results = []
        for h, f in enumerate(fs):
            fcst_length = len(f.future_dates) if fcst_length is None else fcst_length
            results.append([])
            for i in range(n_iter):
                hold_out_len = fcst_length + i * jump_back
                hold_out = f.y.values[-hold_out_len :][: fcst_length]
                f1 = Forecaster(
                    y = f.y.values[: -hold_out_len],
                    current_dates = f.current_dates[: -hold_out_len],
                    future_dates = fcst_length,
                    test_length = f.test_length if test_length is None else test_length,
                    cis = f.cis if cis is None else cis,
                    cilevel = f.cilevel if cilevel is None else cilevel,
                )
                if series_length is not None:
                    f1.keep_smaller_history(series_length)
                results[h].append((f1,hold_out))
        return results

    def backtest(
        self,        
        *fs,
        n_iter=5,
        jump_back=1,
        series_length=None,
        fcst_length=None,
        test_length=None,
        cis=None,
        cilevel=None,
        verbose=False,
        **kwargs,
    ) -> List[Dict[str,pd.DataFrame]]:
        """ Runs an out-of-sample backtest of the pipeline over a certain amount of iterations.

        Args:
            *fs (Forecaster): Send one if univariate forecasting with the `Pipeline` class, 
                more than one if multivariate forecasting with the `MVPipeline` class.
            n_iter (int): Default 5. How many backtest iterations to perform.
            jump_back (int): Default 1. The space between consecutive training sets.
            series_length (int): Optional. The total length of each traning set. 
                Leave unspecified if you want to use every available training observation for each iteration.
            fcst_length (int): Optional. The forecast horizon length to forecast over for each iteration.
                Leave unspecified if you want to use the forecast horizon already programmed into the `Forecaster` object.
            test_length (int): Optional. The test set to hold out for each model evaluation.
                Leave unspecified if you want to use the test length already programmed into the `Forecaster` object.
            cis (bool): Optional. Whether to backtest confidence intervals. 
                Leave unspecified if you want to use whatever is already programmed into the `Forecaster` object.
            cilevel (float): Optional. What level to evaluate confidence intervals at.
                Leave unspecified if you want to use whatever is already programmed into the `Forecaster` object.
            **kwargs: Passed to the `fit_predict()` method from `Pipeline` or `MVPipeline`.

        Returns:
            (List[Dict[str,pd.DataFrame]]): The results from each model and backtest iteration.
            Each dict element of the resulting list corresponds to the Forecaster objects in the order
            they were passed (will be length 1 if univariate forecasting). Each key of each dict is either 'Actuals', 'Obs',
            or the name of a model that got backtested. Each value is a DataFrame with the iteration values.
            The 'Actuals' frame has the date information and are the actuals over each forecast horizon. 
            The 'Obs' frame has the actual historical observations to make each forecast, back padded with NA values to make each array the same length.

        >>> # univariate forecasting
        >>> pipeline = Pipeline(
        >>>     steps = [
        >>>         ('Transform',transformer),
        >>>         ('Forecast',forecaster),
        >>>         ('Revert',reverter),
        >>>     ],
        >>> )
        >>> backtest_results = pipeline.backtest(f,models=models)
        >>>
        >>> # multivariate forecasting
        >>> pipeline = MVPipeline(
        >>>    steps = [
        >>>        ('Transform',[transformer1,transformer2,transformer3]),
        >>>        ('Select Xvars',[auto_Xvar_select]*3),
        >>>        ('Forecast',forecaster,),
        >>>        ('Revert',[reverter1,reverter2,reverter3]),
        >>>    ],
        >>>    names = ['UTUR','UTPHCI','UNRATE'], # used to combine to the mvf object
        >>>    merge_Xvars = 'i', # used to combine to the mvf object
        >>> )
        >>> backtest_results = pipeline.backtest(f1,f2,f3)
        """
        results = []
        _prepare_backtest_results = self._prepare_backtest(
            *fs,
            n_iter=n_iter,
            jump_back=jump_back,
            series_length=series_length,
            fcst_length=fcst_length,
            test_length=test_length,
            cis=cis,
            cilevel=cilevel,
        )
        for res in _prepare_backtest_results:
            results.append({'Actuals':pd.DataFrame()})
            results[-1]['Obs'] = pd.DataFrame()
            for i, f in enumerate(res):
                results[-1]['Actuals'][f'Iter{i}Dates'] = f[0].future_dates.values
                results[-1]['Actuals'][f'Iter{i}Vals'] = f[1]
                if i == 0:
                    results[-1]['Obs'][f'Iter{i}'] = f[0].y.to_list()
                else:
                    results[-1]['Obs'][f'Iter{i}'] = (
                        [np.nan] * 
                        (
                            results[-1]['Obs'].shape[0] - len(f[0].y)
                        ) + f[0].y.to_list() 
                    )
        for i, fsi in enumerate(_prepare_backtest_results[0]):
            fs = self.fit_predict(*[ft[i][0] for ft in _prepare_backtest_results],**kwargs)
            if isinstance(fs,MVForecaster):
                fs = break_mv_forecaster(fs)
            elif not isinstance(fs,tuple):
                fs = (fs,)
            for k, f in enumerate(fs):
                for m,v in f.history.items():
                    if i == 0:
                        results[k][m] = pd.DataFrame({'Iter0Fcst':v['Forecast']})
                    else:
                        results[k][m][f'Iter{i}Fcst'] = v['Forecast']
                    if f.cis:
                        results[k][m][f'Iter{i}Lower'] = v['LowerCI']
                        results[k][m][f'Iter{i}Upper'] = v['UpperCI']

        return results

class Pipeline(Pipeline_parent):
    def __init__(self,steps: List[Tuple[str,Union[Transformer,Reverter,'function']]]):
        """ Initiates the full pipeline.

        Args:
            steps (list[tuple]): A list of transform, forecast, and revert funcs to apply
                to a Forecaster object. The first element of each tuple names the step.
                The second element should either be a Transformer or Reverter type or a function.
                If it is a function, the first argument in the function should require a Forecaster object.
                Functions are checked for as objects that do not have the `fit_transform()` method,
                so adding more elements to the Pipeline may be possible if they have a `fit_transform()` method.

        >>> from scalecast.Forecaster import Forecaster
        >>> from scalecast.Pipeline import Transformer, Reverter, Pipeline
        >>> import pandas_datareader as pdr
        >>> 
        >>> models = ['mlr','elasticnet']
        >>> def forecaster(f,models):
        >>>     f.add_covid19_regressor()
        >>>     f.auto_Xvar_select(cross_validate=True)
        >>>     f.tune_test_forecast(models)
        >>>
        >>> df = pdr.get_data_fred(
        >>>     'HOUSTNSA',
        >>>     start='1959-01-01',
        >>>     end='2022-08-01'
        >>> )
        >>> f = Forecaster(
        >>>     y=df['HOUSTNSA'],
        >>>     current_dates=df.index,
        >>>     future_dates=24,
        >>> )
        >>> f.set_test_length(0.2)
        >>> f.set_validation_length(24)
        >>> transformer = Transformer(
        >>>     transformers = [
        >>>         ('LogTransform',),
        >>>         ('DiffTransform',1),
        >>>         ('DiffTransform',12),
        >>>     ],
        >>> )
        >>> reverter = Reverter(
        >>>     reverters = [
        >>>         ('DiffRevert',12),
        >>>         ('DiffRevert',1),
        >>>         ('LogRevert',),
        >>>     ],
        >>>     base_transformer = transformer,
        >>> )
        >>> pipeline = Pipeline(
        >>>     steps = [
        >>>         ('Transform',transformer),
        >>>         ('Forecast',forecaster),
        >>>         ('Revert',reverter),
        >>>     ],
        >>> )
        """
        # validate we have tuples
        for step in steps:
            if not isinstance(step,tuple):
                raise TypeError(f'Expected elements of pipeline steps list to be tuple type, got {type(step)}.')
        
        self.steps = steps

    def __repr__(self):
        return (
            "Pipeline(\n"
            "  steps = [\n"
            "    {}\n"
            "  ]\n"
            ")".format(",\n    ".join([str(i) for i in self.steps]))
        )

    def __str__(self):
        return self.__repr__()

    def fit_predict(self,f: Forecaster,**kwargs) -> Forecaster:
        """ Applies the transform, forecast, and revert functions to the series stored in the Forecaster object.

        Args:
            f (Forecaster): The Forecaster object that stores the series that will be sent through the pipeline.
            **kwargs: Passed to any 'function' types passed in the pipeline.

        Returns:
            (Forecaster): A Forecaster object with the stored results from the pipeline run.
        
        >>> pipeline = Pipeline(
        >>>     steps = [
        >>>         ('Transform',transformer),
        >>>         ('Forecast',forecaster),
        >>>         ('Revert',reverter),
        >>>     ],
        >>> )
        >>> f = pipeline.fit_predict(f,models=models)
        """
        for step in self.steps:
            func = step[1]
            if hasattr(func,'fit_transform'):
                f = func.fit_transform(f)
            else:
                func(f,**kwargs)
        return f

class MVPipeline(Pipeline_parent):
    def __init__(
        self,
        steps: List[Tuple[str,Union[List[Transformer],List[Reverter],'function']]],
        **kwargs,
    ):
        """ Initiates the full pipeline for multivariate forecasting applications.

        Args:
            steps: (list[tuple]): A list of transform, forecast, and revert funcs to apply
                to multiple Forecaster objects. The first element of each tuple names the step.
                The second element should be a list of Transformer objects, a list of Reverter objects,
                a list of functions, or a single function. If it is a function or list of functions, 
                the first argument in the should require a Forecaster or MVForecaster object.
                If it is a list of functions, Transformer, or Revereter objects,
                each one of these will be called on the Forecaster objects in the order they are passed
                to the `fit_predict()` method.
                Functions are checked for as objects that do not have the `fit_transform()` method,
                so adding more elements to the Pipeline may be possible if they have a `fit_transform()` method.
            **kwargs: Passed to MVForecaster(). See
                https://scalecast.readthedocs.io/en/latest/Forecaster/MVForecaster.html#src.scalecast.MVForecaster.MVForecaster.__init__.

        >>> from scalecast.Forecaster import Forecaster
        >>> from scalecast.Pipeline import MVPipeline
        >>> from scalecast.util import pdr_load, find_optimal_transformation
        >>> 
        >>> def auto_Xvar_select(f):
        >>>    f.auto_Xvar_select(max_ar=0)
        >>> def forecaster(mvf):
        >>>     mvf.set_test_length(24)
        >>>     mvf.set_estimator('elasticnet')
        >>>     mvf.manual_forecast(alpha=.2,lags=12)
        >>>
        >>> f1 = pdr_load('UTUR',future_dates=24,start='1970-01-01',end='2022-07-01')
        >>> f2 = pdr_load('UTPHCI',future_dates=24,start='1970-01-01',end='2022-07-01')
        >>> f3 = pdr_load('UNRATE',future_dates=24,start='1970-01-01',end='2022-07-01')
        >>> # doing this helps the `DetrendTransform()` function
        >>> fs = [f1,f2,f3]
        >>> for f in fs:
        >>>     f.set_test_length(24)
        >>>
        >>> transformer1, reverter1 = find_optimal_transformation(f1)
        >>> transformer2, reverter2 = find_optimal_transformation(f2)
        >>> transformer3, reverter3 = find_optimal_transformation(f3)
        >>> 
        >>> pipeline = MVPipeline(
        >>>     steps = [
        >>>         ('Transform',[transformer1,transformer2,transformer3]),
        >>>         ('Select Xvars',[auto_Xvar_select]*3), # finds xvars for each object
        >>>         ('Forecast',forecaster,), # combines to an mvf object
        >>>         ('Revert',[reverter1,reverter2,reverter3]), # breaks back to f objects
        >>>     ],
        >>>     names = ['UTUR','UTPHCI','UNRATE'],
        >>>     merge_Xvars = 'i',
        >>> )
        """
        for step in steps:
            if not isinstance(step,tuple):
                raise TypeError(f'Expected elements of pipeline steps list to be tuple type, got {type(step)}.')
        
        self.steps = steps
        self.kwargs = kwargs

    def __repr__(self):
        return (
            "MVPipeline(\n"
            "  steps = [\n"
            "    {}\n"
            "  ]\n"
            ")".format(",\n    ".join([str(i) for i in self.steps]))
        )

    def __str__(self):
        return self.__repr__()

    def fit_predict(self,*fs: Forecaster,**kwargs):
        """ Applies the transform, forecast, and revert functions to the series stored in the Forecaster object.
        The order of Forecaster passed to *fs is the order all functions in lists will be applied.

        Args:
            *fs (Forecaster): The Forecaster objects that stores the series that will be sent through the pipeline.
            **kwargs: Passed to any 'function' types passed in the pipeline.

        Returns:
            (Tuple[Forecaster] | MVForecaster): If the last element in the pipeline is a list of reverter functions
            this function returns the individual Forecaster objects. If not, an MVForecaster object is returned.
        
        >>> pipeline = MVPipeline(
        >>>    steps = [
        >>>        ('Transform',[transformer1,transformer2,transformer3]),
        >>>        ('Select Xvars',[auto_Xvar_select]*3), # applied to Forecaster objects
        >>>        ('Forecast',forecaster,), # combines to an mvf object and calls the function
        >>>        ('Revert',[reverter1,reverter2,reverter3]), # breaks back to f objects
        >>>    ],
        >>>    names = ['UTUR','UTPHCI','UNRATE'], # used to combine to the mvf object
        >>>    merge_Xvars = 'i', # used to combine to the mvf object
        >>> )
        >>> f1, f2, f3 = pipeline.fit_predict(f1,f2,f3)
        """
        from .multiseries import line_up_dates

        if 'not_same_len_action' not in kwargs:
            line_up_dates(*fs)
        elif kwargs['not_same_len_action'] == 'trim':
            line_up_dates(*fs)

        i = 0
        fs = list(fs)
        for step in self.steps:
            func_list = step[1]
            if hasattr(func_list,'__len__'):
                if len(fs) != len(func_list):
                    raise ValueError('Must pass as many functions as there are Forecaster objects.')
                if hasattr(func_list[0],'fit_transform'):
                    if i == 1:
                        fs = list(break_mv_forecaster(mvf))
                        i += 1
                    for idx, func in enumerate(zip(fs,func_list)):
                        if func[1] is not None:
                            fs[idx] = func[1].fit_transform(func[0])
                else:
                    for f, func in zip(fs,func_list):
                        if func is not None:
                            func(f,**kwargs)
            else:
                if i == 0:
                    mvf = MVForecaster(*fs,**self.kwargs)
                    i += 1
                func_list(mvf,**kwargs)
        return tuple(fs) if i == 2 else mvf