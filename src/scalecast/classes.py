from dataclasses import dataclass
from .types import PositiveInt, MetricCall
from .typing_utils import ScikitLike, ForecasterEstimatorLike
from typing import Literal, Self, Generic, TypeVar, Optional
from collections import Counter
import numpy as np
from functools import total_ordering

T = TypeVar("T")

@dataclass(frozen=True)
class Estimator:
    name:str
    imported_model:ScikitLike|Literal['auto']
    interpreted_model:ForecasterEstimatorLike

@dataclass(frozen=True)
class MetricStore:
    name:str
    eval_func:MetricCall
    lower_is_better:bool=True
    min_obs_required:PositiveInt=1

    def lookup_determine_best_by_maps(self):
        return [f'InSample{self.name.upper()}',f'TestSet{self.name.upper()}']

@total_ordering
@dataclass(frozen=True)
class EvaluatedMetric:
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
    def __init__(self):
        pass

    def fit(self, X:None) -> Self:
        X = np.asarray(X)
        self.n_features_in_ = X.shape[1] if X.ndim > 1 else 1
        return self
    
    def transform(self, X:np.ndarray) -> np.ndarray:
        return X
    
    def fit_transform(self, X:np.ndarray) -> np.ndarray:
        return X
    
    def inverse_transform(self, X:np.ndarray) -> np.ndarray:
        return X
    
class ValidatedList(Generic[T]):
    """
    Docstring for ValidatedList
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
    
    def lookup_item(self,name):
        """
        Docstring for find_estimator
        
        :param self: Description
        :param name: Description
        """
        returned = [i for i in self.item_list if i.name == name]
        if not returned:
            raise KeyError(f'Cannot find {name}.')
        return returned[0]
    
class DetermineBestBy:
    """
    Docstring for DetermineBestBy
    """
    def __init__(self,metrics:list[MetricStore],validation_metric:MetricStore):
        self.metrics = [validation_metric] + metrics*2
        self.labels = [met.name for met in self.metrics]
        self.values = ['ValidationMetricValue'] + [f'TestSet{met.name.upper()}' for met in metrics] + [f'InSample{met.name.upper()}' for met in metrics]

    def lookup_label(self,value):
        """
        Docstring for lookup_value
        
        :param label: Description
        """
        idx = self.values.index(value)
        return self.labels[idx]
    
    def lookup_metric(self,label):
        """
        Docstring for lookup_metric
        
        :param self: Description
        :param label: Description
        """
        for met in self.metrics:
            if met.name == label:
                return met

    def __repr__(self):
        return self.values.__repr__()
