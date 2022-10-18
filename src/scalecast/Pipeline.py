from scalecast.Forecaster import Forecaster
from scalecast.SeriesTransformer import SeriesTransformer
from typing import List, Tuple, Union
import typing


class Transformer:
    def __init__(self,transformers: List[Tuple]):
        """ initiates the transformer pipeline.

        Args:
            transformers (list[tuple]): a list of transformations to apply to the time series stored in a Forecaster object.
                the tuple's first element should match the name of a transform function from the SeriesTransformer object: 
                https://scalecast.readthedocs.io/en/latest/Forecaster/SeriesTransformer.html.
                positional and keyword arguments can be passed to these functions.
                if a given tuple is more than 1 in length, the `fit_transform()` method will parse
                elements after index 0 as positional arguments.
                keywords are passed as a dictionary in the last position of tuples greater than 1 in length. 
                therefore, if the last argument in the tuple is a dict type,
                this is assumed to contain the keyword arguments. 
                if the last positional argument you wish to pass happens to be dict type,
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
        """ applies the transformation to the series stored in the Forecaster object.

        Args:
            f (Forecaster): the Forecaster object that stores the series that will be transformed.

        Returns:
            (Forecaster): a Forecaster object with the transformed series.

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
        """ initiates the reverter pipeline.

        Args:
            reverters (list[tuple]): a list of revert funcs to apply to the time series stored in a Forecaster object.
                the tuple's first element should match the name of a revert function from the SeriesTransformer object: 
                https://scalecast.readthedocs.io/en/latest/Forecaster/SeriesTransformer.html.
                positional and keyword arguments can be passed to these functions.
                if a given tuple is more than 1 in length, the `fit_transform()` method will parse
                elements after index 0 as positional arguments.
                keywords are passed as a dictionary in the last position of tuples greater than 1 in length. 
                therefore, if the last argument in the tuple is a dict type,
                this is assumed to contain the keyword arguments. 
                if the last positional argument you wish to pass happens to be dict type,
                you can eaither pass it as a keyword argument or place an additional (empty) dictionary at the end of the tuple.
            base_transformer (Transformer|SeriesTransformer): the object that was used to make the original transformations.
                these objects contain the key information to undifference and unscale the stored data 
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
        """ applies the revert function to the series stored in the Forecaster object.

        Args:
            f (Forecaster): the Forecaster object that stores the series that will be reverted.

        Returns:
            (Forecaster): a Forecaster object with the reverted series.

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
        """ initiates the full pipeline.

        Args:
            steps (list[tuple]): a list of transform, forecast, and revert funcs to apply
                to a Forecaster object. the first element of each tuple names the step.
                the second element should either be a Transformer or Reverter type or a function.
                if it is a function, the first argument in the function should require a Forecaster object.
                functions are checked for as objects that do not contain the `fit_transform()` method,
                so adding more elements to the Pipeline may be possible if it has a `fit_transform()` method.

        >>> from scalecast.Forecaster import Forecaster
        >>> from scalecast.Pipeline import Transformer, Reverter, Pipeline
        >>> import pandas_datareader as pdr
        >>> 
        >>> models = ['mlr','elasticnet']
        >>> def forecaster(f,models):
        >>>     f.set_test_length(0.2)
        >>>     f.set_validation_length(24)
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
        """ applies the transform, forecast, and revert functions to the series stored in the Forecaster object.

        Args:
            f (Forecaster): the Forecaster object that stores the series that will be sent through the pipeline.
            **kwargs: passed to any 'function' types passed in the pipeline.

        Returns:
            (Forecaster): a Forecaster object with the stored results from the pipeline run.
        
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
