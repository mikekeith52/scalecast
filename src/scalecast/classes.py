from dataclasses import dataclass
from .types import PositiveInt, MetricCall
from .typing_utils import ScikitLike, ForecasterEstimatorLike
from typing import Literal, Self, Generic, TypeVar, Optional, Any
from collections import Counter
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
