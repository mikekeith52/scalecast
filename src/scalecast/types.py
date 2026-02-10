from typing import Union, Annotated, Any, Literal
from datetime import datetime, date
import pandas as pd
import numpy as np

# Datetimes
DatetimeLike = Union[date,datetime,pd.Timestamp,np.datetime64,str]
# Float between 0 and 1
ConfInterval = Annotated[float, "must be > 0 and < 1"]
# Integers
PositiveInt = Annotated[int,"must be > 0"]
NonNegativeInt = Annotated[int,"must be >= 0"]
# Dynamic testing
DynamicTesting = Union[int,NonNegativeInt]
# models
AvailableModel = Annotated[str,"must exist in object's estimators attribute"]
SKLearnModel = Annotated[str,"must exist in object's sklearn_estimators attribute"]
EvaluatedModel = Annotated[str,"must exist as a key in object's history attribute"]
TopNModels = Annotated[str,'must begin with top_ followed by a positive integer']
ModelValues = Union[EvaluatedModel,list[EvaluatedModel],TopNModels,Literal['all']]
# Xvars
AutoRegressive = Annotated[str,'must begin with AR followed by a positive integer, denoting the lag order']
AvailableXvar = Annotated[str,"must exist in the object's current_xreg attribute"]
XvarValues = Union[AvailableXvar,list[AvailableXvar],Literal['all']]
# Normalizers
AvailableNormalizer = Annotated[str,"must exist in object's normalizer attribute"]
# Metrics
Metric = Annotated[str,"must exist in object's metrics attribute"]
DetermineBestBy = Annotated[str,"must exist in object's determine_best_by attribute"]
# Feature Importance
FIMethod = Literal['PermutationExplainer','TreeExplainer','LinearExplainer','KernelExplainer','SamplingExplainer']
# Export Options
ExportOptions = Literal["model_summaries","lvl_test_set_predictions","lvl_fcsts"]
# Series Name
SeriesName = Annotated[str,"must exist in object's names attribute"]
SeriesValues = list[SeriesName]|SeriesName|Literal['all']
# Transformer/Reverter
ReadAsTransformer = list[tuple[str]|tuple[str,Any]]
PipelineFunction = Annotated[callable,"first argument is a Forecaster object and does not have the fit_transform method"]
TryTransformations = Literal['detrend','seasonal_adj','boxcox','first_diff','first_seasonal_diff','scale']
# Fill Strategies
FFillOption = Annotated[str,'begins with ffill_ followed by an integer']
# To denote parameter is unused -- TODO: stop using this shit
Unused = Any
