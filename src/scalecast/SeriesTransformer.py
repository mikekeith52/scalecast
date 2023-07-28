from ._utils import _developer_utils  
from ._Forecaster_parent import ForecastError
from .Forecaster import Forecaster
import pandas as pd
import numpy as np
import warnings
import logging
import copy
from sklearn.preprocessing import RobustScaler

class SeriesTransformer:
    def __init__(self, f, deepcopy=True):
        """ Initiates the object.

        Args:
            f (Forecaster): The Forecaster object that will receive each transformation/revert.
            deepcopy (bool): Default True. Whether to store a deepcopy of the Forecaster object in the SeriesTransformer object.
        """
        self.f = f.__deepcopy__() if deepcopy else f

    def __repr__(self):
        return "SeriesTransformer(\n{}\n)".format(self.f)

    def __str__(self):
        return self.__repr__()

    def __copy__(self):
        obj = type(self).__new__(self.__class__)
        obj.__dict__.update(self.__dict__)
        return obj

    def __deepcopy__(self, memo={}):
        obj = type(self).__new__(self.__class__)
        memo[id(self)] = obj
        for k, v in self.__dict__.items():
            setattr(obj, k, copy.deepcopy(v, memo))
        return obj

    def Transform(self, transform_func, **kwargs):
        """ Transforms the y attribute in the Forecaster object.
        
        Args:
            transform_func (function): The function that will be used to make the transformation.
                If using a user function, first argument must be the array to transform.
            **kwargs: Passed to the function passed to transform_func.

        Returns:
            (Forecaster): A Forecaster object with the transformed attributes.

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
        return self.f

    def Revert(self, revert_func, exclude_models = [], **kwargs):
        """ Reverts the y attribute in the Forecaster object, along with all model results.

        Args:
            revert_func (function): The function that will be used to revert the values.
                If using a user function, first argument must be the array to transform.
            exclude_models (list-like): Models to not revert. This is useful if you are transforming
                and reverting an object over and over.
            **kwargs: Passed to the function passed to revert_func.

        Returns:
            (Forecaster): A Forecaster object with the reverted attributes.

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
        self.f.y = pd.Series(revert_func(self.f.y, **kwargs))

        for m, h in self.f.history.items():
            if m in exclude_models:
                continue
            for k in ( 
                "Forecast",
                "TestSetPredictions",
                "TestSetUpperCI",
                "TestSetLowerCI",
                "UpperCI",
                "LowerCI",
                "TestSetActuals",
                "FittedVals",
            ):
                if k in h:
                    h[k] = pd.Series(revert_func(h[k], **kwargs),dtype=float).fillna(method='ffill').to_list()
                elif not k.endswith('CI'):
                    h[k] = []
            for met, func in self.f.metrics.items():
                h['TestSet' + met.upper()] = _developer_utils._return_na_if_len_zero(
                    self.f.y.iloc[-len(h["TestSetPredictions"]) :], 
                    h["TestSetPredictions"], 
                    func,
                )
                h['InSample' + met.upper()] = _developer_utils._return_na_if_len_zero(
                    self.f.y.iloc[-len(h['FittedVals']) :], 
                    h['FittedVals'],
                    func,
                )

        return self.f

    def DetrendTransform(
        self,
        loess = False,
        frac = 0.5,
        it = 3,
        poly_order=1,
        ln_trend=False,
        seasonal_lags=0,
        m='auto',
        fit_intercept=True,
        train_only=False,
    ):
        """ Detrends the series using an OLS estimator or using LOESS.
        Only call this once if you want to revert the series later.
        The passed Forecaster object must have future dates or be initiated with `require_future_dates=False`.
        Make sure the test length has already been set as well.
        If the test length changes between the time the transformation is called
        and when models are called, you will get this error when reverting: 
        ValueError: All arrays must be of the same length.
        The ols or lowess model from statsmodels will be stored in the detrend_params attribute with the 'model' key.

        Args:
            loess (bool): Default False. Whether to fit a LOESS curve.
            frac (float): Optional. Default 0.5.
                The fraction of the data used when estimating each y-value.
                A smaller frac value will produce a more rigid trend line, 
                while a larger frac value will produce a smoother trend line.
                Ignored when loess is False. 
            it (int): Optional. Default 3.
                The number of iterations used in the loess algorithm.
                A larger it value will produce a smoother trend line, 
                but may take longer to compute.
                Ignored when loess is False. 
            poly_order (int): Default 1. The polynomial order to use in the fitted trend line.
                Ignored when loess is True. 
            ln_trend (bool): Default False. Whether to use a natural logarithmic trend.
                Ignored when loess is True.
            seasonal_lags (int): Default 0. The number of seasonal lags to use in the estimation.
                Ignored when loess is True.
            m (int or str): Default 'auto'. The number of observations that counts one seasonal step.
                Ignored when seasonal_lags = 0.
                When 'auto', uses the M4 competition values:
                for Hourly: 24, Monthly: 12, Quarterly: 4. everything else gets 1 (no seasonality assumed)
                so pass your own values for other frequencies.
            fit_intercept (bool): Default True. Whether to fit an intercept in the model.
            train_only (bool): Default False. Whether to fit the LOESS or OLS model on the training set only.

        Returns:
            (Forecaster): A Forecaster object with the transformed attributes.

        >>> from scalecast.Forecaster import Forecaster
        >>> from scalecast.SeriesTransformer import SeriesTransformer
        >>> f = Forecaster(...)
        >>> transformer = SeriesTransformer(f)
        >>> f = transformer.DetrendTransform(ln_trend=True)
        """
        self.detrend_params = {
            'origy':self.f.y.copy(),
            'origdates':self.f.current_dates.copy()
        }

        train_only = train_only if self.f.test_length > 0 else False

        fmod = self.f.deepcopy()
        fmod.drop_all_Xvars()
        fmod.add_time_trend()

        if not loess:
            import statsmodels.api as sm 
            if seasonal_lags > 0:
                if m == 'auto':
                    m = _developer_utils._convert_m(m,fmod.freq)
                    if m == 1:
                        warnings.warn(
                            f'Cannot add seasonal lags automatically for the {fmod.freq} frequency. '
                            'Set a value for m manually.'
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
            self.detrend_params['model'] = ols_mod
            self.detrend_params['fvs'] = fvs.values
            self.detrend_params['fvs_fut'] = fvs_fut
            self.detrend_params['fvs_test'] = fvs_test
        else:
            from statsmodels.nonparametric.smoothers_lowess import lowess
            y = fmod.y.values if not train_only else fmod.y.values[-fmod.test_length:]
            loess_fit = lowess(
                y, 
                fmod.current_xreg['t'][:len(y)], 
                frac=frac, 
                it=it,
            )
            self.detrend_params['model'] = loess_fit
            self.detrend_params['fvs'] = np.interp(
                fmod.current_xreg['t'], 
                loess_fit[:, 0], 
                loess_fit[:, 1], 
            )
            self.detrend_params['fvs_test'] = (
                self.detrend_params['fvs'][-self.f.test_length:] 
                if self.f.test_length > 0 
                else []
            )
            self.detrend_params['fvs_fut'] = np.interp(
                fmod.future_xreg['t'], 
                loess_fit[:, 0], 
                loess_fit[:, 1], 
            )

        self.f.y = pd.Series(self.f.y.values - self.detrend_params['fvs'],dtype=float)
        return self.f

    def DetrendRevert(self, exclude_models = []):
        """ Reverts the y attribute in the Forecaster object, along with all model results.
        Assumes a detrend transformation has already been called and uses all model information
        already recorded from that transformation to revert.
        If the test length changes in the Forecaster object between the time the transformation is called
        and when models are called, you will get this error when reverting: 
        ValueError: All arrays must be of the same length.

        Args:
            exclude_models (list-like): Models to not revert. This is useful if you are transforming
                and reverting an object multiple times.
        
        Returns:
            (Forecaster): A Forecaster object with the reverted attributes.

        >>> from scalecast.Forecaster import Forecaster
        >>> from scalecast.SeriesTransformer import SeriesTransformer
        >>> f = Forecaster(...)
        >>> transformer = SeriesTransformer(f)
        >>> f = transformer.DetrendTransform(ln_trend=True)
        >>> f = transformer.DetrendRevert()
        """
        if not hasattr(self,'detrend_params'):
            raise ForecastError('Before reverting a trend, make sure DetrendTransform has already been called.')

        self.f.y = self.detrend_params['origy']
        self.f.current_dates = self.detrend_params['origdates']

        fvs = {
            "TestSetPredictions":self.detrend_params['fvs_test'],
            "TestSetActuals":self.detrend_params['fvs_test'],
            "FittedVals":self.detrend_params['fvs'],
            "Forecast":self.detrend_params['fvs_fut'],
            "TestSetUpperCI":self.detrend_params['fvs_test'],
            "TestSetLowerCI":self.detrend_params['fvs_test'],
            "UpperCI":self.detrend_params['fvs_fut'],
            "LowerCI":self.detrend_params['fvs_fut'],
        }

        for m, h in self.f.history.items():
            if m in exclude_models:
                continue
            for k,v in fvs.items():
                if k in h:
                    h[k] = [i + fvs for i, fvs in zip(h[k],v)]
                elif not k.endswith('CI'):
                    h[k] = []

        #delattr(self,'detrend_params')
        return self.Revert(lambda x: x)  # call here to assign correct test-set metrics
        
    def LogTransform(self):
        """ Transforms the y attribute in the Forecaster object using a natural log transformation.

        Returns:
            (Forecaster): A Forecaster object with the transformed attributes.

        >>> from scalecast.Forecaster import Forecaster
        >>> from scalecast.SeriesTransformer import SeriesTransformer
        >>> f = Forecaster(...)
        >>> transformer = SeriesTransformer(f)
        >>> f = transformer.LogTransform()
        """
        return self.Transform(np.log)

    def LogRevert(self, **kwargs):
        """ Reverts the y attribute in the Forecaster object, along with all model results.
        Assumes a natural log transformation has already been called.

        Args:
            **kwargs: Passed to Transformer.Revert() - 
                arg `exclude_models` accepted here.

        Returns:
            (Forecaster): A Forecaster object with the reverted attributes.

        >>> from scalecast.Forecaster import Forecaster
        >>> from scalecast.SeriesTransformer import SeriesTransformer
        >>> f = Forecaster(...)
        >>> transformer = SeriesTransformer(f)
        >>> f = transformer.LogTransform()
        >>> f = transformer.LogRevert()
        """
        return self.Revert(np.exp, **kwargs)

    def SqrtTransform(self):
        """ Transforms the y attribute in the Forecaster object using a square-root transformation.

        Returns:
            (Forecaster): A Forecaster object with the transformed attributes.

        >>> from scalecast.Forecaster import Forecaster
        >>> from scalecast.SeriesTransformer import SeriesTransformer
        >>> f = Forecaster(...)
        >>> transformer = SeriesTransformer(f)
        >>> f = transformer.SqrtTransform()
        """
        return self.Transform(np.sqrt)

    def SqrtRevert(self, **kwargs):
        """ Reverts the y attribute in the Forecaster object, along with all model results.
        Assumes a square-root transformation has already been called.

        Args:
            **kwargs: Passed to Transformer.Revert() - 
                arg `exclude_models` accepted here.

        Returns:
            (Forecaster): A Forecaster object with the reverted attributes.

        >>> from scalecast.Forecaster import Forecaster
        >>> from scalecast.SeriesTransformer import SeriesTransformer
        >>> f = Forecaster(...)
        >>> transformer = SeriesTransformer(f)
        >>> f = transformer.SqrtTransform()
        >>> f = transformer.SqrtRevert()
        """
        return self.Revert(np.square, **kwargs)

    def ScaleTransform(self,train_only=False):
        """ Transforms the y attribute in the Forecaster object using a scale transformation.
        Scale defined as (array[i] - array.mean()) / array.std().

        Args:
            train_only (bool): Default False.
                Whether to fit the scale transformer on the training set only.
        Returns:
            (Forecaster): A Forecaster object with the transformed attributes.

        >>> from scalecast.Forecaster import Forecaster
        >>> from scalecast.SeriesTransformer import SeriesTransformer
        >>> f = Forecaster(...)
        >>> transformer = SeriesTransformer(f)
        >>> f = transformer.ScaleTransform()
        """
        if hasattr(self, "orig_mean"):
            return

        train_only = train_only if self.f.test_length > 0 else False
        stop_at = len(self.f.y) if not train_only else len(self.f.y) - self.f.test_length

        self.orig_mean = self.f.y.values[:stop_at].mean()
        self.orig_std = self.f.y.values[:stop_at].std()

        def func(x, mean, std):
            return [(i - mean) / std for i in x]

        return self.Transform(func, mean=self.orig_mean, std=self.orig_std)

    def ScaleRevert(self, **kwargs):
        """ Reverts the y attribute in the Forecaster object, along with all model results.
        Assumes the scale transformation has been called on the object at some point.
        Revert function: array.std()*array[i]+array.mean().

        Args:
            **kwargs: Passed to Transformer.Revert() - 
                arg `exclude_models` accepted here.

        Returns:
            (Forecaster): A Forecaster object with the reverted attributes.

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
            f = self.Revert(func, mean=self.orig_mean, std=self.orig_std, **kwargs)
            #delattr(self, "orig_mean")
            #delattr(self, "orig_std")
            return f
        except AttributeError:
            raise ValueError("Cannot revert a series that was never scaled.")

    def RobustScaleTransform(self,train_only=False,**kwargs):
        """ Transforms the y attribute in the Forecaster object using a robust scale transformation.
        See the function from scikit-learn: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html#sklearn.preprocessing.RobustScaler.

        Args:
            train_only (bool): Default False.
                Whether to fit the transformer on the training set only.
            **kwargs: Passed to the scikit-learn function.
        Returns:
            (Forecaster): A Forecaster object with the transformed attributes.

        >>> from scalecast.Forecaster import Forecaster
        >>> from scalecast.SeriesTransformer import SeriesTransformer
        >>> f = Forecaster(...)
        >>> transformer = SeriesTransformer(f)
        >>> f = transformer.RobustScaleTransform()
        """

        train_only = train_only if self.f.test_length > 0 else False
        stop_at = len(self.f.y) if not train_only else len(self.f.y) - self.f.test_length

        y = self.f.y.values[:stop_at].reshape(-1,1)

        transformer = RobustScaler(**kwargs).fit(y)
        self.rs_transformer = transformer

        return self.Transform(_developer_utils._reshape_func_input,func=transformer.transform)

    def RobustScaleRevert(self, **kwargs):
        """ Reverts the y attribute in the Forecaster object, along with all model results.
        Assumes the scale transformation has been called on the object at some point.
        Revert function: array.std()*array[i]+array.mean().

        Args:
            **kwargs: Passed to Transformer.Revert() - 
                arg `exclude_models` accepted here.

        Returns:
            (Forecaster): A Forecaster object with the reverted attributes.

        >>> from scalecast.Forecaster import Forecaster
        >>> from scalecast.SeriesTransformer import SeriesTransformer
        >>> f = Forecaster(...)
        >>> f.set_test_length(.2) # specify a test set to not leak data with this func
        >>> transformer = SeriesTransformer(f)
        >>> f = transformer.ScaleTransform(train_only=True)
        >>> f = transformer.ScaleRevert()
        """

        if not hasattr(self,'rs_transformer'):
            raise ValueError("Cannot revert a series that was never scaled with RobustScaleTransform.")

        func = self.rs_transformer.inverse_transform

        return self.Revert(_developer_utils._reshape_func_input,func=func,**kwargs)

    def MinMaxTransform(self,train_only=False):
        """ Transforms the y attribute in the Forecaster object using a min-max scale transformation.
        Min-max scale defined as (array[i] - array.min()) / (array.max() - array.min()).

        Args:
            train_only (bool): Default False.
                Whether to fit the minmax transformer on the training set only.
        Returns:
            (Forecaster): A Forecaster object with the transformed attributes.

        >>> from scalecast.Forecaster import Forecaster
        >>> from scalecast.SeriesTransformer import SeriesTransformer
        >>> f = Forecaster(...)
        >>> transformer = SeriesTransformer(f)
        >>> f = transformer.MinMaxTransform()
        """
        if hasattr(self, "orig_min"):
            return

        train_only = train_only if self.f.test_length > 0 else False
        stop_at = len(self.f.y) if not train_only else len(self.f.y) - self.f.test_length

        self.orig_min = self.f.y.values[:stop_at].min()
        self.orig_max = self.f.y.values[:stop_at].max()

        def func(x, amin, amax):
            return [(i - amin) / (amax - amin) for i in x]

        return self.Transform(func, amin=self.orig_min, amax=self.orig_max,)

    def MinMaxRevert(self, **kwargs):
        """ Reverts the y attribute in the Forecaster object, along with all model results.
        Assumes the min-max scale transformation has been called on the object at some point.
        Revert function: array[i]*(array.max() - array.min()) + array.min().

        Args:
            **kwargs: {assed to Transformer.Revert() - 
                arg `exclude_models` accepted here.

        Returns:
            (Forecaster): A Forecaster object with the reverted attributes.

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
            f = self.Revert(func, amin=self.orig_min, amax=self.orig_max, **kwargs)
            #delattr(self, "orig_min")
            #delattr(self, "orig_max")
            return f
        except AttributeError:
            raise ValueError("Cannot revert a series that was never scaled.")

    def DiffTransform(self, m=1):
        """ Takes differences or seasonal differences in the Forecaster object's y attribute.
        If using this transformation, call `Forecaster.add_diffed_terms()` and 
        `Forecaster.add_lagged_terms()` if you want to use those before calling this function.
        Call `Forecaster.add_ar_terms()` and `Forecaster.add_AR_terms()` after calling this function.
        Call twice with the same value of m to take second differences.

        Args:
            m (int): Default 1. The seasonal difference to take.
                1 will difference once. 
                12 will take a seasonal difference assuming 12 periods makes a season (monthly data).
                Any int available.

        Returns:
            (Forecaster): A Forecaster object with the transformed attributes.

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

    def DiffRevert(self, m=1, exclude_models = []):
        """ Reverts the y attribute in the Forecaster object, along with all model results.
        Calling this makes so that AR values become unusable and have to be re-added to the object.

        Args:
            m (int): Default 1. The seasonal difference to revert.
                1 will undifference once. 
                12 will undifference seasonally 12 periods (monthly data).
                Any int available. Use the same values to revert as you used
                to transform the object originally.
            exclude_models (list-like): Models to not revert. This is useful if you are transforming
                and reverting an object over and over.

        Returns:
            (Forecaster): A Forecaster object with the reverted attributes.

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
        self.f.current_dates = pd.Series(dates_orig)

        for mod, h in self.f.history.items():
            if mod in exclude_models:
                continue
            if hasattr(h['Forecast'],'__len__'):
                h["Forecast"] = list(seasrevert(h["Forecast"], self.f.y[-m:], m))[m:]
                h['FittedVals'] = list(
                    seasrevert(h['FittedVals'],self.f.y.values[-len(h['FittedVals'])-m:], m)
                )[m:]
            if hasattr(h['TestSetPredictions'],'__len__'):
                h["TestSetPredictions"] = list(
                    seasrevert(
                        h["TestSetPredictions"],
                        self.f.y.to_list()[-(m + self.f.test_length) : -self.f.test_length],
                        m,
                    )
                )[m:]
                h["TestSetActuals"] = self.f.y.to_list()[-self.f.test_length :]
            if np.isnan(self.f.history[mod]['CILevel']): # no cis evaluated
                continue
            # undifference cis
            fcst = h["Forecast"]
            test_preds = h["TestSetPredictions"]
            test_actuals = h["TestSetActuals"]
            test_resids = np.abs([p - a for p, a in zip(test_preds,test_actuals)])
            ci_range = np.percentile(test_resids, 100 * self.f.cilevel)
            self.f._set_cis(
                "UpperCI",
                "LowerCI",
                "TestSetUpperCI",
                "TestSetLowerCI",
                m = mod,
                ci_range = ci_range,
                forecast = fcst,
                tspreds = test_preds,
            )
        #delattr(self, f"orig_y_{m}_{n}")
        #delattr(self, f"orig_dates_{m}_{n}")
        return self.Revert(lambda x: x, exclude_models = exclude_models)  # call here to assign correct test-set metrics

    def DeseasonTransform(
        self,
        m = None,
        model='add',
        extrapolate_trend='freq',
        train_only = False,
        **kwargs,
    ):
        """ Deseasons a series using the moving average method offered by statsmodel through the seasonal_decompose() function.

        Args:
            m (int): The number of observations that counts one seasonal step.
                If not specified, will use the inferred seasonality from statsmodels.
            model (str): Default 'add'. One of {"additive", "add", "multiplicative", "mul"}.
                The type of seasonal component.
            extrapolate_trend (str or int): Default 'freq'. If set to > 0, the trend resulting from the convolution is
                linear least-squares extrapolated on both ends (or the single one
                if two_sided is False) considering this many (+1) closest points.
                If set to 'freq', use `freq` closest points. Setting this parameter
                results in no NaN values in trend or resid components.
            train_only (bool): Default False. Whether to fit the seasonal decomposition model on the training set only.
            **kwargs: Passed to seasonal_decompose() function from statsmodels.
                See https://www.statsmodels.org/dev/generated/statsmodels.tsa.seasonal.seasonal_decompose.html.

        Returns:
            (Forecaster): A Forecaster object with the transformed attributes.

        >>> from scalecast.Forecaster import Forecaster
        >>> from scalecast.SeriesTransformer import SeriesTransformer
        >>> f = Forecaster(...)
        >>> transformer = SeriesTransformer(f)
        >>> f = transformer.DeseasonTransform(model='mul')  # multiplicative deseasoning
        """
        _developer_utils.descriptive_assert(
            model in ("additive", "add", "multiplicative", "mul"),
            ValueError,
            f'The value passed to the model argument must be one of ("additive", "add", "multiplicative", "mul"), got {model}.'
        )
        train_only = train_only if self.f.test_length > 0 else False
        if m is None:
            from statsmodels.tsa.tsatools import freq_to_period
            m = freq_to_period(self.f.freq)
        decomp_res = self.f.seasonal_decompose(
            model=model,
            extrapolate_trend=extrapolate_trend,
            period = m,
            train_only=train_only,
            **kwargs,
        )
        deseasoned = (
            (decomp_res.trend + decomp_res.resid) if model in ('add','additive') 
            else (decomp_res.trend * decomp_res.resid)
        )
        current_seasonality = decomp_res.seasonal # check
        f = Forecaster(
            y = current_seasonality,
            current_dates = current_seasonality.index,
            future_dates = len(self.f.future_dates) + (self.f.test_length if train_only else 0)
        )
        f.set_estimator('naive')
        f.manual_forecast(
            seasonal=True,
            m=m,
            test_again=False,
            bank_history=False,
        )
        params = {
            'model':model,
            'y':self.f.y.copy(),
            'current_seasonality':current_seasonality.to_list() + f.forecast[:-len(self.f.future_dates)],
            'future_seasonality':f.forecast[-len(self.f.future_dates):],
        }
        setattr(self,f'deseason_params_{m}',params)
        self.f.y = pd.Series(
            self.f.y.values - np.array(params['current_seasonality']) if model in ('add','additive') 
            else self.f.y.values / np.array(params['current_seasonality']),
            dtype=float,
        )
        return self.f

    def DeseasonRevert(self, m = None, exclude_models = []):
        """ Reverts a seasonal adjustment already taken on the series. Call DeseasonTransform() before calling this.

        Args:
            m (int): The number of observations that counts one seasonal step.
                If not specified, will use the inferred seasonality from statsmodels.
            exclude_models (list-like): Models to not revert. This is useful if you are transforming
                and reverting an object multiple times.

        Returns:
            (Forecaster): A Forecaster object with the reverted attributes.

        >>> from scalecast.Forecaster import Forecaster
        >>> from scalecast.SeriesTransformer import SeriesTransformer
        >>> f = Forecaster(...)
        >>> transformer = SeriesTransformer(f)
        >>> f = transformer.DeseasonTransform(model='mul')  # multiplicative deseasoning
        >>> f = transformer.DeseasonRevert() # back to normal
        """
        if m is None:
            from statsmodels.tsa.tsatools import freq_to_period
            m = freq_to_period(self.f.freq)
        params = getattr(self,f'deseason_params_{m}')
        try:
            self.f.y = params['y']
        except AttributeError:
            raise ForecastError(
                f'Before reverting a seasonal transformation, make sure DeseasonTransform with m = {m} has already been called.'
            )

        atts = {
            "TestSetPredictions":params['current_seasonality'],
            "TestSetActuals":params['current_seasonality'],
            "FittedVals":params['current_seasonality'],
            "Forecast":params['future_seasonality'],
            "TestSetUpperCI":params['current_seasonality'],
            "TestSetLowerCI":params['current_seasonality'],
            "UpperCI":params['future_seasonality'],
            "LowerCI":params['future_seasonality'],
        }

        for mod, h in self.f.history.items():
            if mod in exclude_models:
                continue
            for a, v in atts.items():
                if a in h:
                    h[a] = [
                        (i + s) if params['model'] in ('add','additive') 
                        else (i * s) for i, s in zip(h[a],v[-len(h[a]):])
                    ]
                elif not a.endswith('CI'):
                    h[a] = []

        #delattr(self, f"deseason_params_{m}")
        return self.Revert(lambda x: x)  # call here to assign correct test-set metrics