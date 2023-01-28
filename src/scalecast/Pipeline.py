from scalecast.Forecaster import Forecaster
from scalecast.SeriesTransformer import SeriesTransformer
from typing import List, Tuple, Union
import typing

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
                raise TypeError(f'expected elements of transformer list to be tuple type, got {type(transformer)}')

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
                raise TypeError(f'expected elements of reverter list to be tuple type, got {type(reverter)}')
        
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

    def fit_transform(self,f: Forecaster) -> Forecaster:
        """ Applies the revert function to the series stored in the Forecaster object.

        Args:
            f (Forecaster): The Forecaster object that stores the series that will be reverted.

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
        for reverter in self.reverters:
            if len(reverter) > 1:
                args = [i for i in reverter[1:-1]]
                args += reverter[-1:] if not isinstance(reverter[-1],dict) else []
                kwargs = reverter[-1] if isinstance(reverter[-1],dict) else {}
            else:
                args = []
                kwargs = {}

            f = getattr(base_transformer,reverter[0])(*args,**kwargs)
        return f

class Pipeline:
    def __init__(self,steps: List[Tuple[str,Union[Transformer,Reverter,'function']]]):
        """ Initiates the full pipeline.

        Args:
            steps (list[tuple]): A list of transform, forecast, and revert funcs to apply
                to a Forecaster object. The first element of each tuple names the step.
                The second element should either be a Transformer or Reverter type or a function.
                If it is a function, the first argument in the function should require a Forecaster object.
                functions are checked for as objects that do not contain the `fit_transform()` method,
                so adding more elements to the Pipeline may be possible if it has a `fit_transform()` method.

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
                raise TypeError(f'expected elements of pipeline steps list to be tuple type, got {type(step)}')
        
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

class MVPipeline:
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
                Functions are checked for as objects that do not contain the `fit_transform()` method,
                so adding more elements to the Pipeline may be possible if it has a `fit_transform()` method.
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
                raise TypeError(f'expected elements of pipeline steps list to be tuple type, got {type(step)}')
        
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
        from scalecast.MVForecaster import MVForecaster
        from scalecast.util import break_mv_forecaster
        from scalecast.multiseries import keep_smallest_first_date

        if 'not_same_len_action' not in kwargs:
            keep_smallest_first_date(*fs)
        elif kwargs['not_same_len_action'] == 'trim':
            keep_smallest_first_date(*fs)

        i = 0
        fs = list(fs)
        for step in self.steps:
            func_list = step[1]
            if hasattr(func_list,'__len__'):
                if len(fs) != len(func_list):
                    raise ValueError('must pass as many functions as there are Forecaster objects.')
                if hasattr(func_list[0],'fit_transform'):
                    if i == 1:
                        fs = list(break_mv_forecaster(mvf))
                        i += 1
                    for idx, func in enumerate(zip(fs,func_list)):
                        if func[1] is not None:
                            if i == 2: # reverting
                                if hasattr(func[1].base_transformer,'base_transformer'):
                                    func[1].base_transformer.base_transformer.f = func[0]
                                else:
                                    func[1].base_transformer.f = func[0]
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


