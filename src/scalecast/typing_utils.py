from typing import Protocol, Any, Self
import numpy as np

class ScikitLike(Protocol):
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs: Any) -> Self:
        pass
    
    def predict(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        pass

class TFLike(Protocol):
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