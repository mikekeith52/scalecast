from .types import ConfInterval
from typing import Sequence
import numpy as np
import pandas as pd
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_percentage_error,
    mean_absolute_error,
)

class Metrics:
    @staticmethod
    def bias(a:Sequence[float],f:Sequence[float]) -> float:
        """ Returns the total bias over a given forecast horizon. 
        When this is larger than 0, means aggregated predicted points are higher than actuals.
        Divide by the length of the forecast horizon to get average bias.

        Args:
            a (list-like): The actuals over the forecast horizon.
            f (list-like): The predictions over the forecast horizon.

        Returns:
            (float): The derived bias.

        >>> from scalecast.util import metrics
        >>> a = [1,2,3,4,5]
        >>> f = [1,2,3,4,6]
        >>> metrics.bias(a,f) # returns 1
        """
        return np.sum(np.array(f) - np.array(a))

    @staticmethod
    def abias(a:Sequence[float],f:Sequence[float]) -> float:
        """ Returns the total bias over a given forecast horizon in terms of absolute values. 
        Divide by the length of the forecast horizon to get average bias.
        This is a good metric to minimize when testing/tuning models.

        Args:
            a (list-like): The actuals over the forecast horizon.
            f (list-like): The predictions over the forecast horizon.

        Returns:
            (float): The derived bias.

        >>> from scalecast.util import metrics
        >>> a = [1,2,3,4,5]
        >>> f = [1,2,3,4,6]
        >>> metrics.abias(a,f) # returns 1
        """
        return np.abs(np.sum(np.array(f) - np.array(a)))

    @staticmethod
    def mape(a:Sequence[float],f:Sequence[float]) -> float:
        """ Mean absolute percentage error (MAPE).

        Args:
            a (list-like): The actuals over the forecast horizon.
            f (list-like): The predictions over the forecast horizon.

        Returns:
            (float): The derived MAPE.

        >>> from scalecast.util import metrics
        >>> a = [1,2,3,4,5]
        >>> f = [1,2,3,4,6]
        >>> metrics.mape(a,f)
        """
        return (
            np.nan if np.abs(a).min() == 0 
            else mean_absolute_percentage_error(a, f)
        )


    @staticmethod
    def r2(a:Sequence[float],f:Sequence[float]) -> float:
        """ R-squared (R2).

        Args:
            a (list-like): The actuals over the forecast horizon.
            f (list-like): The predictions over the forecast horizon.

        Returns:
            (float): The derived R2.

        >>> from scalecast.util import metrics
        >>> a = [1,2,3,4,5]
        >>> f = [1,2,3,4,6]
        >>> metrics.r2(a,f)
        """
        return r2_score(a, f)

    @staticmethod
    def mse(a:Sequence[float],f:Sequence[float]) -> float:
        """ Mean squared error (MSE).

        Args:
            a (list-like): The actuals over the forecast horizon.
            f (list-like): The predictions over the forecast horizon.

        Returns:
            (float): The derived MSE.

        >>> from scalecast.util import metrics
        >>> a = [1,2,3,4,5]
        >>> f = [1,2,3,4,6]
        >>> metrics.mse(a,f)
        """
        return mean_squared_error(a, f)

    @staticmethod
    def rmse(a:Sequence[float],f:Sequence[float]) -> float:
        """ Root mean squared error (RMSE).

        Args:
            a (list-like): The actuals over the forecast horizon.
            f (list-like): The predictions over the forecast horizon.

        Returns:
            (float): The derived RMSE.

        >>> from scalecast.util import metrics
        >>> a = [1,2,3,4,5]
        >>> f = [1,2,3,4,6]
        >>> metrics.rmse(a,f)
        """
        return mean_squared_error(a, f) ** 0.5

    @staticmethod
    def mae(a:Sequence[float],f:Sequence[float]) -> float:
        """ Mean absolute error (MAE).

        Args:
            a (list-like): The actuals over the forecast horizon.
            f (list-like): The predictions over the forecast horizon.

        Returns:
            (float): The derived MAE.

        >>> from scalecast.util import metrics
        >>> a = [1,2,3,4,5]
        >>> f = [1,2,3,4,6]
        >>> metrics.mae(a,f)
        """
        return mean_absolute_error(a, f)

    @staticmethod
    def smape(a:Sequence[float],f:Sequence[float]) -> float:
        """ Symmetric mean absolute percentage error (sMAPE).
        Uses the same definition as used in the M4 competition.
        Does not multiply by 100.
        See https://ideas.repec.org/a/eee/intfor/v36y2020i1p54-74.html.

        Args:
            a (list-like): The actuals over the forecast horizon.
            f (list-like): The predictions over the forecast horizon.

        Returns:
            (float): The derived sMAPE.

        >>> from scalecast.util import metrics
        >>> a = [1,2,3,4,5]
        >>> f = [1,2,3,4,6]
        >>> metrics.smape(a,f)
        """
        a = np.array(a)
        f = np.array(f)
        return (
            1/len(a) *
            np.sum(
                2*np.abs(f-a) / (
                    np.abs(a) + np.abs(f)
                )
            )
        )

    @staticmethod
    def mase(a:Sequence[float],f:Sequence[float],obs:Sequence[float],m:int) -> float:
        """ Mean absolute scaled error (MASE).
        Uses the same definition as used in the M4 competition.
        See https://ideas.repec.org/a/eee/intfor/v36y2020i1p54-74.html.

        Args:
            a (list-like): The actuals over the forecast horizon.
            f (list-like): The predictions over the forecast horizon.
            obs (list-like): The actual observations used to create the forecast.
            m (int): The seasonal period.

        Returns:
            (float): The derived MASE.

        >>> from scalecast.util import metrics
        >>> a = [1,2,3,4,5]
        >>> f = [1,2,3,4,6]
        >>> obs = [-5,-4,-3,-2,-1,0]
        >>> metrics.mase(a,f,obs,1)
        """
        a = np.array(a)
        f = np.array(f)
        avger = 1/len(a)
        num = np.sum(np.abs(f-a))
        davger = 1 / (len(obs) - m)
        denom = np.sum(
            np.abs(pd.Series(obs).diff(m).values[m:])
        )
        return avger * (num / (davger * denom))

    @staticmethod
    def msis(a:Sequence[float],uf:Sequence[float],lf:Sequence[float],obs:Sequence[float],m:int,alpha:ConfInterval=0.05) -> float:
        """ Mean scaled interval score (MSIS) for evaluating confidence intervals.
        Uses the same definition as used in the M4 competition.
        Lower values are better.
        See https://ideas.repec.org/a/eee/intfor/v36y2020i1p54-74.html.

        Args:
            a (list-like): The actuals over the forecast horizon.
            uf (list-like): The upper-forecast bound according to the confidence interval.
            lf (list-like): The lower-forecast bound according to the confidence interval.
            obs (list-like): The actual observations used to create the forecast.
            m (int): The seasonal period.
            alpha (float): Default 0.05. 0.05 for 95% confidence intervals, etc.

        Returns:
            (float): The derived MSIS.

        >>> from scalecast.util import metrics
        >>> a = [1,2,3,4,5]
        >>> f = [1,2,3,4,6]
        >>> uf = [1.5,2.5,3.5,4.5,6.5]
        >>> lf = [.5,1.5,2.5,3.5,5.5]
        >>> obs = [-5,-4,-3,-2,-1,0]
        >>> metrics.msis(a,uf,lf,obs,1) # returns a value of 5.0
        """
        a = np.array(a)
        uf = np.array(uf)
        lf = np.array(lf)
        avger = 1/len(a)
        num1 = uf-lf
        num2 = np.array([(
            (2/alpha*(lfs-acs)) if lfs > acs else
            (2/alpha*(acs-ufs)) if acs > ufs else
            0
        ) for acs, ufs, lfs in zip(a,uf,lf)])
        davger = 1/(len(obs) - m)
        denom = np.sum(np.abs(pd.Series(obs).diff(m).values[m:]))
        return avger * (np.sum(num1 + num2) / (davger * denom))