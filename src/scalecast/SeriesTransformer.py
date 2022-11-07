import pandas as pd
import numpy as np
import warnings
from scalecast.Forecaster import (
    Forecaster,
    ForecastError,
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
            for k in ( 
                "LevelForecast", 
                "LevelTestSetPreds", 
                "LevelFittedVals",
                "LevelLowerCI",
                "LevelTSLowerCI",
                "LevelUpperCI",
                "LevelTSUpperCI",
            ):
                h[k] = list(revert_func(h[k], **kwargs))
            h['LevelY'] = self.f.levely

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

    def DetrendTransform(
            self,
            poly_order=1,
            ln_trend=False,
            seasonal_lags=0,
            m='auto',
            fit_intercept=True,
            train_only=False,
        ):
        """ detrends the series using an OLS estimator.
        only call this once if you want to revert the series later.
        the passed Forecaster object must have future dates or be initiated with `require_future_dates=False`.
        make sure the test length has already been set as well.
        if the test length changes between the time the transformation is called
        and when models are called, you will get this error when reverting: 
        ValueError: All arrays must be of the same length.
        the ols model from statsmodels will be stored in the detrend_model attribute.

        Args:
            poly_order (int): default 1. the polynomial order to use.
            ln_trend (bool): default False. whether to use a natural logarithmic trend.
            seasonal_lags (int): default 0. the number of seasonal lags to use in the estimation.
            m (int or str): default 'auto'. the number of observations that counts one seasonal step.
                ignored when seasonal_lags = 0.
                when 'auto', uses the M4 competition values:
                for Hourly: 24, Monthly: 12, Quarterly: 4. everything else gets 1 (no seasonality assumed)
                so pass your own values for other frequencies.
            fit_intercept (bool): default True. whether to fit an intercept in the model.
            train_only (bool): default False. whether to fit the OLS model on the training set only.

        Returns:
            (Forecaster): a Forecaster object with the transformed attributes.

        >>> from scalecast.Forecaster import Forecaster
        >>> from scalecast.SeriesTransformer import SeriesTransformer
        >>> f = Forecaster(...)
        >>> transformer = SeriesTransformer(f)
        >>> f = transformer.DetrendTransform(ln_trend=True)
        """
        import statsmodels.api as sm 
        from scalecast.util import _convert_m
        
        self.detrend_origy = self.f.y.copy()
        self.detrend_origlevely = self.f.levely.copy()
        self.detrend_origdates = self.f.current_dates.copy()

        fmod = self.f.deepcopy()
        fmod.drop_all_Xvars()
        fmod.add_time_trend()

        if seasonal_lags > 0:
            if m == 'auto':
                m = _convert_m(m,fmod.freq)
                if m == 1:
                    warnings.warn(
                        f'cannot add seasonal lags automatically for the {fmod.freq} frequency. '
                        'set a value for m manually.'
                    )
            if m > 1:
                fmod.add_lagged_terms('t',lags=m*seasonal_lags)
                fmod.drop_Xvars(*[
                    x for x in fmod.get_regressor_names() if (
                        x.startswith(
                            'tlag'
                        ) and int(
                            x.split('tlag_')[-1]
                        ) % m != 0
                    ) and x != 't'
                ])
        if ln_trend:
            fmod.add_logged_terms(*fmod.get_regressor_names(),drop=True)
        fmod.add_poly_terms(*fmod.get_regressor_names(),pwr=poly_order)

        dataset = fmod.export_Xvars_df().set_index('DATE')
        if fit_intercept:
            dataset = sm.add_constant(dataset)

        train_set = dataset.loc[fmod.current_dates] # full dataset minus future dates
        model_inputs = train_set.iloc[:-fmod.test_length,:] if train_only else train_set.copy() # what will be used to fit the model
        test_set = train_set.iloc[-fmod.test_length:,:] # the test set for reverting models later
        future_set = dataset.loc[fmod.future_dates] # the future inputs for reverting models later

        y = fmod.y.values
        y_train = y.copy() if not train_only else y[:-fmod.test_length].copy()

        ols_mod = sm.OLS(y_train,model_inputs).fit()
        fvs = ols_mod.predict(train_set) # reverts fitted values
        fvs_fut = ols_mod.predict(future_set) # reverts forecasts
        fvs_test = ols_mod.predict(test_set) # reverts test-set predictions

        self.f.keep_smaller_history(len(train_set))
        self.f.y = self.f.y.values - fvs.values
        self.f.levely = list(self.f.y)

        # i'm not 100% sure we need this and it does cause one thing to break so i'm doing this for now.
        try: 
            self.f.typ_set(); 
        except: 
            warnings.warn('type seting the Forecaster object did not work in the trend transform, continuing as is.')

        self.detrend_model = ols_mod
        self.detrend_fvs = fvs
        self.detrend_fvs_fut = fvs_fut
        self.detrend_fvs_test = fvs_test
        return self.f

    def DetrendRevert(self):
        """ reverts the y attribute in the Forecaster object, along with all model results.
        assumes a detrend transformation has already been called and uses all model information
        already recorded from that transformation to revert.
        if the test length changes in the Forecaster object between the time the transformation is called
        and when models are called, you will get this error when reverting: 
        ValueError: All arrays must be of the same length.
        
        Returns:
            (Forecaster): a Forecaster object with the reverted attributes.

        >>> from scalecast.Forecaster import Forecaster
        >>> from scalecast.SeriesTransformer import SeriesTransformer
        >>> f = Forecaster(...)
        >>> transformer = SeriesTransformer(f)
        >>> f = transformer.DetrendTransform(ln_trend=True)
        >>> f = transformer.DetrendRevert()
        """
        try:
            self.f.levely = self.detrend_origlevely
            self.f.y = self.detrend_origy
            self.f.current_dates = self.detrend_origdates
        except AttributeError:
            raise ForecastError('before reverting a trend, make sure DetrendTransform has already been called.')

        fvs = {
            "LevelTestSetPreds":self.detrend_fvs_test,
            "TestSetPredictions":self.detrend_fvs_test,
            "LevelFittedVals":self.detrend_fvs,
            "FittedVals":self.detrend_fvs,
            "LevelForecast":self.detrend_fvs_fut,
            "Forecast":self.detrend_fvs_fut,
            "TestSetUpperCI":self.detrend_fvs_test,
            "TestSetLowerCI":self.detrend_fvs_test,
            "UpperCI":self.detrend_fvs_fut,
            "LowerCI":self.detrend_fvs_fut,
            "LevelLowerCI":self.detrend_fvs_fut,
            "LevelTSLowerCI":self.detrend_fvs_test,
            "LevelUpperCI":self.detrend_fvs_fut,
            "LevelTSUpperCI":self.detrend_fvs_test,
        }

        for m, h in self.f.history.items():
            for k,v in fvs.items():
                h[k] = [i + fvs for i, fvs in zip(h[k],v)]

        for a in (
            'detrend_origy',
            'detrend_origlevely',
            'detrend_origdates',
            'detrend_model',
            'detrend_fvs',
            'detrend_fvs_fut',
            'detrend_fvs_test',
        ):
            delattr(self,a)

        return self.Revert(lambda x: x)  # call here to assign correct test-set metrics
        
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
        assumes a natural log transformation has already been called.

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

    def SqrtTransform(self):
        """ transforms the y attribute in the Forecaster object using a square-root transformation.

        Returns:
            (Forecaster): a Forecaster object with the transformed attributes.

        >>> from scalecast.Forecaster import Forecaster
        >>> from scalecast.SeriesTransformer import SeriesTransformer
        >>> f = Forecaster(...)
        >>> transformer = SeriesTransformer(f)
        >>> f = transformer.SqrtTransform()
        """
        return self.Transform(np.sqrt)

    def SqrtRevert(self, full=True):
        """ reverts the y attribute in the Forecaster object, along with all model results.
        assumes a square-root transformation has already been called.

        Args:
            full (bool): whether to revert all attributes, or just the level attributes.
                if False, all non-level results will remain unchanged.

        Returns:
            (Forecaster): a Forecaster object with the reverted attributes.

        >>> from scalecast.Forecaster import Forecaster
        >>> from scalecast.SeriesTransformer import SeriesTransformer
        >>> f = Forecaster(...)
        >>> transformer = SeriesTransformer(f)
        >>> f = transformer.SqrtTransform()
        >>> f = transformer.SqrtRevert()
        """
        return self.Revert(np.square, full=full)

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

    def DiffRevert(self, m):
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

            h['FittedVals'] = list(
                seasrevert(h['FittedVals'],self.f.y.values[-len(h['FittedVals'])-m:], m)
            )[m:]
            h['LevelFittedVals'] = h['FittedVals'][:]
            #ci_range = self.f._find_cis(self.f.y.values[-len(h['FittedVals']):],h['FittedVals'])
            for k in ("LevelUpperCI","UpperCI","LevelLowerCI","LowerCI"):
                h[k] = list(seasrevert(h[k], self.f.y[-m:], m))[m:]
            for k in ("TestSetUpperCI","LevelTSUpperCI","TestSetLowerCI","LevelTSLowerCI"):
                h[k] = list(
                    seasrevert(
                        h[k],
                        self.f.y.to_list()[-(m + self.f.test_length) : -self.f.test_length],
                        m,
                    )
                )[m:]
        delattr(self, f"orig_y_{m}_{n}")
        delattr(self, f"orig_dates_{m}_{n}")

        return self.Revert(lambda x: x)  # call here to assign correct test-set metrics
