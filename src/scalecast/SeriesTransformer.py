import pandas as pd
import numpy as np
from scalecast.Forecaster import (
    Forecaster,
    rmse,
    mape,
    mae,
    r2,
)


class SeriesTransformer:
    def __init__(self, f):
        self.f = f.__deepcopy__()

    def Transform(self, transform_func, **kwargs):
        """ transforms the y attribute in the Forecaster object.
        
        Args:
            transform_func (function): the function that will be used to make the transformation.
                if using a user function, first argument must be the array to transform.
            **kwargs: passed to the function passed to transform_func.

        Returns:
            (Forecaster): a Forecaster object with the transformed attributes.

        >>> import math
        >>> from scalecast.Forecaster import Forecaster
        >>> from scalecast.SeriesTransformer import SeriesTransformer
        >>> def log10(x):
        >>>     return [math.log(i,base=10) for i in x]
        >>> f = Forecaster(...)
        >>> transformer = SeriesTransformer(f)
        >>> f = transformer.Transform(log10)
        """
        self.f.y = pd.Series(transform_func(self.f.y, **kwargs))
        self.f.levely = list(transform_func(self.f.levely, **kwargs))
        return self.f

    def Revert(self, revert_func, full=True, **kwargs):
        """ reverts the y attribute in the Forecaster object, along with all model results.

        Args:
            revert_func (function): the function that will be used to revert the values.
                if using a user function, first argument must be the array to transform.
            full (bool): whether to revert all attributes, or just the level attributes.
                if False, all non-level results will remain unchanged.
            **kwargs: passed to the function passed to revert_func.

        Returns:
            (Forecaster): a Forecaster object with the reverted attributes.

        >>> import math
        >>> from scalecast.Forecaster import Forecaster
        >>> from scalecast.SeriesTransformer import SeriesTransformer
        >>> def log10(x):
        >>>     return [math.log(i,base=10) for i in x]
        >>> def log10_revert(x):
        >>>     return [10**i for i in x]
        >>> f = Forecaster(...)
        >>> transformer = SeriesTransformer(f)
        >>> f = transformer.Transform(log10)
        >>> f = transformer.Revert(log10_revert)
        """
        self.f.levely = list(revert_func(self.f.levely, **kwargs))
        self.f.y = pd.Series(revert_func(self.f.y, **kwargs)) if full else self.f.y

        for m, h in self.f.history.items():
            for k in ("LevelY", "LevelForecast", "LevelTestSetPreds", "LevelFittedVals"):
                h[k] = list(revert_func(h[k], **kwargs))

            for i, preds in enumerate(('LevelTestSetPreds','LevelFittedVals')):
                pred = h[preds]
                act = self.f.levely[-len(pred) :]
                if i == 0:
                    h["LevelTestSetRMSE"] = rmse(act, pred)
                    h["LevelTestSetMAPE"] = mape(act, pred)
                    h["LevelTestSetMAE"] = mae(act, pred)
                    h["LevelTestSetR2"] = r2(act, pred)
                elif i == 1:
                    h["LevelInSampleRMSE"] = rmse(act, pred)
                    h["LevelInSampleMAPE"] = mape(act, pred)
                    h["LevelInSampleMAE"] = mae(act, pred)
                    h["LevelInSampleR2"] = r2(act, pred)

            if full:
                for k in (
                    "Forecast",
                    "TestSetPredictions",
                    "TestSetUpperCI",
                    "TestSetLowerCI",
                    "CIPlusMinus",
                    "UpperCI",
                    "LowerCI",
                    "TestSetActuals",
                    "FittedVals",
                ):
                    try:
                        h[k] = list(revert_func(h[k], **kwargs))
                    except TypeError:
                        h[k] = revert_func([h[k]], **kwargs)[0]

                for i, preds in enumerate(("TestSetPredictions","FittedVals")):
                    pred = h[preds]
                    act = self.f.levely[-len(pred) :]
                    if i == 0:
                        h["TestSetRMSE"] = rmse(act, pred)
                        h["TestSetMAPE"] = mape(act, pred)
                        h["TestSetMAE"] = mae(act, pred)
                        h["TestSetR2"] = r2(act, pred)
                    elif i == 1:
                        h["InSampleRMSE"] = rmse(act, pred)
                        h["InSampleMAPE"] = mape(act, pred)
                        h["InSampleMAE"] = mae(act, pred)
                        h["InSampleR2"] = r2(act, pred)

        return self.f

    def LogTransform(self):
        """ transforms the y attribute in the Forecaster object using a natural log transformation.

        Returns:
            (Forecaster): a Forecaster object with the transformed attributes.

        >>> from scalecast.Forecaster import Forecaster
        >>> from scalecast.SeriesTransformer import SeriesTransformer
        >>> f = Forecaster(...)
        >>> transformer = SeriesTransformer(f)
        >>> f = transformer.LogTransform()
        """
        return self.Transform(np.log)

    def LogRevert(self, full=True):
        """ reverts the y attribute in the Forecaster object, along with all model results.
        assumes a natural log transformation has been called at some point.

        Args:
            full (bool): whether to revert all attributes, or just the level attributes.
                if False, all non-level results will remain unchanged.

        Returns:
            (Forecaster): a Forecaster object with the reverted attributes.

        >>> from scalecast.Forecaster import Forecaster
        >>> from scalecast.SeriesTransformer import SeriesTransformer
        >>> f = Forecaster(...)
        >>> transformer = SeriesTransformer(f)
        >>> f = transformer.LogTransform()
        >>> f = transformer.LogRevert()
        """
        return self.Revert(np.exp, full=full)

    def ScaleTransform(self,train_only=False):
        """ transforms the y attribute in the Forecaster object using a scale transformation.
        scale defined as (array[i] - array.mean()) / array.std().

        Args:
            train_only (bool): default False.
                whether to fit the scale transformer on the training set only.
        Returns:
            (Forecaster): a Forecaster object with the transformed attributes.

        >>> from scalecast.Forecaster import Forecaster
        >>> from scalecast.SeriesTransformer import SeriesTransformer
        >>> f = Forecaster(...)
        >>> transformer = SeriesTransformer(f)
        >>> f = transformer.ScaleTransform()
        """
        if hasattr(self, "orig_mean"):
            return

        stop_at = len(self.f.y) if not train_only else len(self.f.y) - self.f.test_length

        self.orig_mean = self.f.y.values[:stop_at].mean()
        self.orig_std = self.f.y.values[:stop_at].std()

        def func(x, mean, std):
            return [(i - mean) / std for i in x]

        return self.Transform(func, mean=self.orig_mean, std=self.orig_std)

    def ScaleRevert(self, full=True):
        """ reverts the y attribute in the Forecaster object, along with all model results.
        assumes the scale transformation has been called on the object at some point.
        reversion function: array.std()*array[i]+array.mean().

        Args:
            full (bool): whether to revert all attributes, or just the level attributes.
                if False, all non-level results will remain unchanged.

        Returns:
            (Forecaster): a Forecaster object with the reverted attributes.

        >>> from scalecast.Forecaster import Forecaster
        >>> from scalecast.SeriesTransformer import SeriesTransformer
        >>> f = Forecaster(...)
        >>> f.set_test_length(.2) # specify a test set to not leak data with this func
        >>> transformer = SeriesTransformer(f)
        >>> f = transformer.ScaleTransform(train_only=True)
        >>> f = transformer.ScaleRevert()
        """

        def func(x, mean, std):
            return [std * i + mean for i in x]

        try:
            f = self.Revert(func, mean=self.orig_mean, std=self.orig_std, full=full,)
            delattr(self, "orig_mean")
            delattr(self, "orig_std")
            return f
        except AttributeError:
            raise ValueError("cannot revert a series that was never scaled.")

    def MinMaxTransform(self,train_only=False):
        """ transforms the y attribute in the Forecaster object using a min-max scale transformation.
        min-max scale defined as (array[i] - array.min()) / (array.max() - array.min()).

        Args:
            train_only (bool): default False.
                whether to fit the minmax transformer on the training set only.
        Returns:
            (Forecaster): a Forecaster object with the transformed attributes.

        >>> from scalecast.Forecaster import Forecaster
        >>> from scalecast.SeriesTransformer import SeriesTransformer
        >>> f = Forecaster(...)
        >>> transformer = SeriesTransformer(f)
        >>> f = transformer.MinMaxTransform()
        """
        if hasattr(self, "orig_min"):
            return

        stop_at = len(self.f.y) if not train_only else len(self.f.y) - self.f.test_length

        self.orig_min = self.f.y.values[:stop_at].min()
        self.orig_max = self.f.y.values[:stop_at].max()

        def func(x, amin, amax):
            return [(i - amin) / (amax - amin) for i in x]

        return self.Transform(func, amin=self.orig_min, amax=self.orig_max,)

    def MinMaxRevert(self, full=True):
        """ reverts the y attribute in the Forecaster object, along with all model results.
        assumes the min-max scale transformation has been called on the object at some point.
        reversion function: array[i]*(array.max() - array.min()) + array.min().

        Args:
            full (bool): whether to revert all attributes, or just the level attributes.
                if False, all non-level results will remain unchanged.

        Returns:
            (Forecaster): a Forecaster object with the reverted attributes.

        >>> from scalecast.Forecaster import Forecaster
        >>> from scalecast.SeriesTransformer import SeriesTransformer
        >>> f = Forecaster(...)
        >>> f.set_test_length(.2) # specify a test set to not leak data with this func
        >>> transformer = SeriesTransformer(f)
        >>> f = transformer.MinMaxTransform(train_only=True)
        >>> f = transformer.MinMaxRevert()
        """

        def func(x, amin, amax):
            return [i * (amax - amin) + amin for i in x]

        try:
            f = self.Revert(func, amin=self.orig_min, amax=self.orig_max, full=full,)
            delattr(self, "orig_min")
            delattr(self, "orig_max")
            return f
        except AttributeError:
            raise ValueError("cannot revert a series that was never scaled.")

    def DiffTransform(self, m):
        """ takes differences or seasonal differences in the Forecaster object's y attribute.
        if using this transformation, call `Forecaster.add_diffed_terms()` and 
        `Forecaster.add_lagged_terms()` if you want to use those before calling this function.
        call `Forecaster.add_ar_terms()` and `Forecaster.add_AR_terms()` after calling this function.
        call twice with the same value of m to take second differences.
        if using this to take series differences, do not also use the native `Forecaster.diff()` function.

        Args:
            m (int): the seasonal difference to take.
                1 will difference once. 
                12 will take a seasonal difference assuming 12 periods makes a season (monthly data).
                any int available.

        Returns:
            (Forecaster): a Forecaster object with the transformed attributes.

        >>> from scalecast.Forecaster import Forecaster
        >>> from scalecast.SeriesTransformer import SeriesTransformer
        >>> f = Forecaster(...)
        >>> transformer = SeriesTransformer(f)
        >>> f = transformer.DiffTransform(1)  # first difference
        >>> f = transformer.DiffTransform(1)  # second difference
        >>> f = transformer.DiffTransform(12) # first 12-period difference
        """
        n = (
            0
            if not hasattr(self, f"orig_y_{m}_0")
            else max(
                [
                    int(a.split("_")[-1]) + 1
                    for a in dir(self)
                    if a.startswith(f"orig_y_{m}")
                ]
            )
        )

        setattr(self, f"orig_y_{m}_{n}", self.f.y.to_list())
        setattr(self, f"orig_dates_{m}_{n}", self.f.current_dates.to_list())

        func = lambda x, m: pd.Series(x).diff(m)
        f = self.Transform(func, m=m)
        f.keep_smaller_history(len(f.y) - m)
        return f

    def DiffRevert(self, m, revert_fvs=True):
        """ reverts the y attribute in the Forecaster object, along with all model results.
        calling this makes so that AR values become unusable and have to be re-added to the object.
        unlike other revert functions, there is no option for full = False. 
        if using this to revert differences, you should not also use the native Forecaster.diff() function.

        Args:
            m (int): the seasonal difference to take.
                1 will undifference once. 
                12 will undifference seasonally 12 periods (monthly data).
                any int available. use the same values to revert as you used
                to transform the object originally.
            revert_fvs (bool): default True.
                whether to revert fitted values and apply bootstrapped confidence intervals.
                fitted vals can look very distorted after more than one difference transformation has
                been taken.

        Returns:
            (Forecaster): a Forecaster object with the reverted attributes.

        >>> from scalecast.Forecaster import Forecaster
        >>> from scalecast.SeriesTransformer import SeriesTransformer
        >>> f = Forecaster(...)
        >>> transformer = SeriesTransformer(f)
        >>> f = transformer.DiffTransform(1)  # first difference
        >>> f = transformer.DiffTransform(1)  # second difference
        >>> f = transformer.DiffTransform(12) # first 12-period difference
        >>> # call revert funcs in reverse order
        >>> f = transformer.DiffRevert(12)
        >>> f = transformer.DiffRevert(1)
        >>> f = transformer.DiffRevert(1) # back to normal
        """

        def seasrevert(transformed, orig, m):
            orig = list(orig)[:m]
            transformed = list(transformed)
            for i in range(len(orig)):
                transformed.insert(i, orig[i])
            for i in range(m, len(transformed)):
                transformed[i] = transformed[i - m] + transformed[i]
            return transformed

        n = max(
            [int(a.split("_")[-1]) for a in dir(self) if a.startswith(f"orig_y_{m}")]
        )

        y_orig = getattr(self, f"orig_y_{m}_{n}")
        dates_orig = getattr(self, f"orig_dates_{m}_{n}")

        self.f.y = pd.Series(seasrevert(self.f.y, y_orig, m))
        self.f.levely = self.f.y.to_list()
        self.f.current_dates = pd.Series(dates_orig)

        for _, h in self.f.history.items():
            h["Forecast"] = list(seasrevert(h["Forecast"], self.f.y[-m:], m))[m:]
            h["TestSetPredictions"] = list(
                seasrevert(
                    h["TestSetPredictions"],
                    self.f.y.to_list()[-(m + self.f.test_length) : -self.f.test_length],
                    m,
                )
            )[m:]
            h["LevelY"] = self.f.levely[:]
            h["LevelForecast"] = h["Forecast"][:]
            h["LevelTestSetPreds"] = h["TestSetPredictions"][:]
            h["TestSetActuals"] = self.f.y.to_list()[-self.f.test_length :]

            if revert_fvs:
                h['FittedVals'] = list(
                    seasrevert(h['FittedVals'],self.f.y[-len(h['FittedVals'])-m:], m)
                )[m:]
                h['LevelFittedVals'] = h['FittedVals'][:]
                ci_range = self.f._find_cis(self.f.y[-len(h['FittedVals']):],h['FittedVals'])
                for k in ("LevelUpperCI","UpperCI"):
                    h[k] = [i + ci_range for i in h["Forecast"]]
                for k in ("TestSetUpperCI","LevelTSUpperCI"):
                    h[k] = [i + ci_range for i in h["TestSetPredictions"]]
                for k in ("LevelLowerCI","LowerCI"):
                    h[k] = [i - ci_range for i in h["Forecast"]]
                for k in ("TestSetLowerCI","LevelTSLowerCI"):
                    h[k] = [i - ci_range for i in h["TestSetPredictions"]]
            else:
                for k in (
                    "FittedVals",
                    "TestSetUpperCI",
                    "TestSetLowerCI",
                    "UpperCI",
                    "LowerCI",
                    "LevelUpperCI",
                    "LevelLowerCI",
                    "LevelTSUpperCI",
                    "LevelTSLowerCI",
                ):
                    h[k] = [np.nan for _ in h[k]]
                h["CIPlusMinus"] = np.nan

        delattr(self, f"orig_y_{m}_{n}")
        delattr(self, f"orig_dates_{m}_{n}")

        return self.Revert(lambda x: x)  # call here to assign correct test-set metrics
