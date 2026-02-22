from typing import TYPE_CHECKING, Protocol, Any, Self, Literal, Optional, Sequence
from .types import XvarValues
import numpy as np
if TYPE_CHECKING:
    from .Forecaster import Forecaster
    from .MVForecaster import MVForecaster

class ScikitLike(Protocol):
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs: Any) -> Self:
        pass
    
    def predict(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        pass

class NormalizerLike(Protocol):
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs: Any) -> Self:
        pass
    
    def transform(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        pass

    def fit_transform(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        pass

    def inverse_transform(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        pass

class ForecasterEstimatorLike(Protocol):
    def __init__(self,f:'Forecaster',model:ScikitLike|Literal['auto'],test_set_actuals:Optional[list[float]]=None,**kwargs:Any):
        pass

    def generate_current_X(self):
        pass

    def generate_future_X(self):
        pass

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs: Any) -> Self:
        pass
    
    def predict(self, X: np.ndarray, in_sample:bool, **kwargs: Any) -> np.ndarray:
        pass

    def fit_predict(self, X: np.ndarray, y: np.ndarray, **kwargs: Any) -> np.ndarray:
        pass

class MVForecasterEstimatorLike(Protocol):
    def __init__(
        self,
        f:'MVForecaster',
        model:ScikitLike|Literal['auto'],
        lags:None|int|list[int]|dict[str,int|list[int]]=1,
        test_set_actuals:Optional[dict[str,Sequence[float]]]=None,
        **kwargs:Any
    ):
        pass

    def generate_current_X(self):
        pass

    def generate_future_X(self):
        pass

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs: Any) -> Self:
        pass
    
    def predict(self, X: np.ndarray, in_sample:bool, **kwargs: Any) -> np.ndarray:
        pass

    def fit_predict(self, X: np.ndarray, y: np.ndarray, **kwargs: Any) -> np.ndarray:
        pass