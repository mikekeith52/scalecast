from dataclasses import dataclass, fields
from .types import PositiveInt, MetricCall, DynamicTesting, ConfInterval
from .typing_utils import ScikitLike, ForecasterEstimatorLike
from typing import Literal, Self, Generic, TypeVar, Optional, Any, Iterator
from collections import Counter
import pandas as pd
import numpy as np
from functools import total_ordering

T = TypeVar("T")

@dataclass(frozen=True)
class Estimator:
    """ Object for storing information about estimators used in the scalecast library.

    Attributes:
        name (str): Name of the estimator. This is the str value referenced by a Forecaster or MVForecaster object when setting estimators.
        imported_Model (ScikitLike or 'auto'): The imported model. Every supported model class outside of scikit-learn estimators has its own protocol,
            so this should remain 'auto' except for models that follow a scikit-learn API.
        interpreted_model: The model class from the models module in scalecast that is used to run the model.
    """
    name:str
    imported_model:ScikitLike|Literal['auto']
    interpreted_model:ForecasterEstimatorLike

@dataclass(frozen=True)
class MetricStore:
    """ Object to store information about metrics supported in scalecast. This object allows for automatic sorting of derived metric values and keeps track of how the value was derived.

    Attributes:
        name (str): The name of the metric. This is the str value referenced by a Forecaster or MVForecaster object when setting metrics.
        eval_func (callable): Function that accepts two arguments (a and f) and returns a float.
        lower_is_better (bool): Whether a lower score indicates better performance.
        min_obs_required (int): The minimum required observations (sizes of the a and f arrays) to derive the metric value.
    """
    name:str
    eval_func:MetricCall
    lower_is_better:bool=True
    min_obs_required:PositiveInt=1

    def lookup_determine_best_by_maps(self) -> str:
        """ Returns the values of DetermineBestBy the given metric maps to in a Forecaster or MVForecaster object.

        Returns:
            str: The mapped values.
        """
        return [f'InSample{self.name.upper()}',f'TestSet{self.name.upper()}']

@total_ordering
@dataclass(frozen=True)
class EvaluatedMetric:
    """ Object to store evaluated metric values. Used to keep track of how it maps to determine_best_by values and how to sort the results.
    A list of these will automatically sort worst-to-best (ascending order).

    Attributes:
        store (MetricStore): The MetricStore object associated with the object result.
        score (float): The metric score.
    """
    store:MetricStore
    score:float

    def __eq__(self, other):
        if not isinstance(other, EvaluatedMetric):
            return NotImplemented
        return (self.store, self.score) == (other.store, other.score)

    def __lt__(self, other):
        if not isinstance(other, EvaluatedMetric):
            return NotImplemented
        elif self.store != other.store:
            raise ValueError(f'Invalid comparison of {self.store.name} and {other.store.name}')
        elif self.store.lower_is_better:
            return self.score > other.score
        else:
            return self.score < other.score
    
    def __call__(self):
        return self.score

@dataclass(frozen=True)
class AR:
    """ Class that represents autoregressive terms.

    Attributes:
        lag_order (int): The lag order the term represents.
    """
    lag_order: PositiveInt

    def __post_init__(self):
        if self.lag_order <= 0:
            raise ValueError('lag_order must be strictly positive')
        
class NoScaler:
    """ Does not fit a scaler, but keeps all the expected methods so that it can easily be folded into a scalecast pipeline when `normalizer=None`.
    """
    def __init__(self):
        pass

    def fit(self, X:None) -> Self:
        """ Does not fit a scaler.

        Returns:
            Self
        """
        X = np.asarray(X)
        self.n_features_in_ = X.shape[1] if X.ndim > 1 else 1
        return self
    
    def transform(self, X:np.ndarray) -> np.ndarray:
        """ Returns the object passed to X.

        Args:
            X (np.ndarray): An array of values to not transform.
        
        Returns:
            np.ndarray: The object passed to X.
        """
        return X
    
    def fit_transform(self, X:np.ndarray) -> np.ndarray:
        """ Returns the object passed to X.

        Args:
            X (np.ndarray): An array of values to not transform.
        
        Returns:
            np.ndarray: The object passed to X.
        """
        return X
    
    def inverse_transform(self, X:np.ndarray) -> np.ndarray:
        """ Returns the object passed to X.

        Args:
            X (np.ndarray): An array of values to not transform.
        
        Returns:
            np.ndarray: The object passed to X.
        """
        return X

@dataclass
class ModelHistory:
    """ Object used to store history of each tested/forecasted model.

    Attributes:
        Estimator (str): Name of the estimator matched against estimators attribute in Forecaster object.
        HyperParams (dict[str,Any]): The hyperparameters used to evaluate the model. Empty means default.
        Observations (int): Number of observations the model was trained on.
        Forecast (list[float]|dict[str,list[float]]): The forecasted values that matches the length of the future_dates attribute in the Forecaster or MVForecaster object.
        Lags (int|list[int]|dict[str,int|list[int]]): The lags used to train the model. Used only in MVForecaster objects.
        FittedVals (list[float]|dict[str,list[float]]): In-sample fitted values from the trained model.
        TestSetLength (int): Length of the test set if one was specified.
        TestSetPredictions (list[float]|dict[str,list[float]]): Predictions on the held-out values if model testing was performed.


    """
    Estimator: str
    HyperParams: dict[str,Any]
    Observations: int
    Forecast: Optional[list[float]|dict[str,list[float]]] = None
    Lags: Optional[int|list[int]|dict[str,int|list[int]]] = None
    FittedVals: Optional[list[float]|dict[str,list[float]]] = None
    TestSetLength: Optional[int] = None
    TestSetPredictions: Optional[list[float]|dict[str,list[float]]] = None
    TestSetActuals: Optional[list[float]|dict[str,list[float]]] = None
    DynamicallyTested: Optional[DynamicTesting] = None
    Xvars: Optional[list[str]] = None
    ValidationMetric: Optional[str] = None
    ValidationMetricValue: Optional[float] = None
    CILevel: Optional[ConfInterval] = None
    TestSetUpperCI: Optional[list[float]|dict[str,list[float]]] = None
    TestSetLowerCI: Optional[list[float]|dict[str,list[float]]] = None
    UpperCI: Optional[list[float]|dict[str,list[float]]] = None
    LowerCI: Optional[list[float]|dict[str,list[float]]] = None
    ci_range: Optional[float|dict[str,float]] = None
    call_estimator: Optional[ForecasterEstimatorLike|dict[str,ForecasterEstimatorLike]] = None
    grid: Optional[dict[str,Any]] = None
    grid_evaluated: Optional[np.ndarray] = None
    models: Optional[list[str]] = None
    weights: Optional[list[float]] = None
    feature_importance_explainer: Optional[str] = None
    feature_importance: Optional[pd.DataFrame] = None
    _AllowSetPrefix: tuple[str,...] = ('InSample','TestSet')

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        match key:
            case i if i.startswith(self._AllowSetPrefix):
                pass
            case i if i not in {f.name for f in fields(self)}:
                raise KeyError(f"Cannot store {key!r} in model history")

        setattr(self, key, value)

    def __contains__(self, key):
        return key in {f.name for f in fields(self)}

    def __iter__(self):
        for field in fields(self):
            yield getattr(self, field.name)

    def get(self, key:str, default:Any) -> Any:
        """ Fetches an attribute's value using a key. If key does not exist in the object, returns a default value.

        Args:
            key (str): The attribute to search for in the object.
            default (Any): The object to return if they key is not found.
        
        Returns:
            Any: Either the attribute value or the default argument.
        """
        if key in self:
            return self[key]
        return default

    def items(self) -> Iterator[tuple[str,Any]]:
        """ Returns all attributes and corresponding values.
        """
        return zip([i for i in self],[self[i] for i in self])

    def to_list(self) -> list[str]:
        """Converts object attributes into a list."""
        return [i for i in self]
    
class ValidatedList(Generic[T]):
    """ A list-like object that allows storing specialized classes that have a name attribute. Enforces no duplication and that each element in the list is one of the specialized types.

    Args:
        item_list (list[Estimator|MetricStore]): The list of elements to be represented in the object.
        enforce_type ('Estimator' or 'MetricStore'): The object type to enforce.

    Raises:
        ValueError: If duplication is detected or the elements are the wrong type.

    Attributes:
        item_list (list[Estimator|MetricStore]): The list of elements represented in the object.
    """
    def __init__(self,item_list:list[Estimator|MetricStore],enforce_type:Literal['Estimator','MetricStore']):
        if not isinstance(item_list,list):
            raise ValueError(f'item_list must be a list type, got {item_list}')
        elif [i for i in item_list if not isinstance(i,eval(enforce_type))]:
            raise ValueError(f'Every element in item_list but be an Estimator type, got {item_list}')
        
        all_names = [i.name for i in item_list]
        name_counts = Counter(all_names)
        duplicates = [v for v, c in name_counts.items() if c>1]
        if duplicates:
            raise ValueError(f'None of the names in the list can be reused, got multiple instances of {duplicates}')
        
        self.item_list = item_list

    def __repr__(self):
        return f"ValidatedList({self.item_list})"
    
    def __contains__(self, item):
        return item == self.lookup_item(item).name
    
    def __getitem__(self, index):
        return self.item_list[index]
    
    def lookup_item(self,name:str) -> Any:
        """ Looks up an element from the list in the item_list attribute.
        
        Args:
            name (str): The name attribute of the element being looked up in the item_list attribute.

        Returns:
            The referenced element.
        """
        returned = [i for i in self.item_list if i.name == name]
        if not returned:
            raise KeyError(f'Cannot find {name}.')
        return returned[0]
    
class DetermineBestBy:
    """ Object used to keep track of possible values that can be passed to determine_best_by arguments in Forecaster and MVForecaster objects.

    Args:
        metrics (list[MetricStore]): List of metrics to map.
        validation_metric (MetricStore): The metric in the validation_metric attribute in the object.

    Attributes:
        metrics (list[MetricStore]): List of metrics to map.
        labels (list[str]): Names of the mapped metrics.
        values (list[str]): Values that are passed to determine_best_by arguments.
    """
    def __init__(self,metrics:list[MetricStore],validation_metric:MetricStore):
        self.metrics = [validation_metric] + metrics*2
        self.labels = [met.name for met in self.metrics]
        self.values = ['ValidationMetricValue'] + [f'TestSet{met.name.upper()}' for met in metrics] + [f'InSample{met.name.upper()}' for met in metrics]

    def lookup_label(self,value:str) -> str:
        """ Returns the metric label associated with the determine_best_by value.

        Args:
            value (str): determine_best_by value to lookup.
        
        Returns:
            str: Associated metric label.
        """
        idx = self.values.index(value)
        return self.labels[idx]
    
    def lookup_metric(self,label:str) -> MetricStore:
        """ Returns the MetricStore object associated with the metric label.

        Args:
            label (str): Metric label

        Returns:
            MetricStore: The associated MetricStore object.
        """
        for met in self.metrics:
            if met.name == label:
                return met

    def __repr__(self):
        return self.values.__repr__()
    
    def __iter__(self):
        return iter(self.values)
