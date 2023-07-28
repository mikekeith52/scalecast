# Changelog
All notable changes to this project are documented in this file since v0.1.8. The source code for most releases is available on [GitHub](https://github.com/mikekeith52/scalecast).

## [0.18.9] - 2023-07-27
### Added
- Added the `RobustScaler` transformer and added it to the default optimal transformation search.
- Added `'robust'` as a valid normalizer argument when forecasting with scikit-learn esitmators.
### Changed
### Fixed

## [0.18.8] - 2023-07-03
### Added
### Changed
- Made an explicit error message when the RNN model does not have enough observations to evaluate (#58).
- Changed the title of the loss plot from the RNN model.
### Fixed
- Fixed how the prophet model creates its externals regressors dataframe to avoid a `ValueError` (#55).
- Fixed an error with the RNN model when forecast horizon is 1.

## [0.18.7] - 2023-06-24
### Added
### Changed
- Changed requirements to avoid a dask/lightgbm error when importing the Forecaster module (#46).
### Fixed
- Calling `auto_forecast()` without tuning a model first no longer raises an error when banking the model's history (#52).
- RNN and LSTM models no longer raise errors with default CV parameters.


## [0.18.6] - 2023-06-01
### Added
- Added `exclude` argument to `Forecaster.plot()`, `Forecaster.plot_test_set()`, and `Forecaster.plot_fitted()`.
### Changed
### Fixed

## [0.18.5] - 2023-05-17
### Added
### Changed
- The combo model can now accept one-model arguments to facilitate auto-selecting "best" models.
### Fixed
- Updated requirement version of lightgbm (#46)

## [0.18.4] - 2023-04-28
### Added
### Changed
- Took out `infer_datetime_format` args from pandas functions since they are deprecated.
- Modified example grids.
### Fixed

## [0.18.3] - 2023-04-22
### Added
### Changed
### Fixed
- Fixed the test-set length check when determining min length required for confidence intervals, which was not processing floating points precisely enough.

## [0.18.2] - 2023-04-19
### Added
- Added `MVForecaster.add_signals()` for multivariate stacking.
- Added `Forecaster.add_series()` to make multivariate RNNs possible.
- Added `return_train_only` in the `util.find_optimal_transformation()` function to add additional leaking-prevention measures to this function.
- Added padding options in `Forecaster.ingest_Xvars_df()`.
- Added `exclude_models` argument to `Reverter.fit_transform()`.
### Changed
- `SeriesTransformer` no longer deletes attributes after reverting, making the object more reusable.
- `Reverter` objects always copy the passed `Forecaster` objects to make them more flexible and able to be used in more multivariate contexts.
- Added `__copy__()` and `__deepcopy__()` methods to several objects.
- Removed `Forecaster.add_diffed_terms()`.
- Changed how `Forecaster.add_lagged_terms()` deals with the introduction of N/A values -- just warn instead of chopping observations.
- Changed how `train_only` arguments are read in `SeriesTransformer` when there is no test_length in the `Forecaster` object. If test_length is 0, then `train_only` behaves as if it were False.
### Fixed
- Fixed how exogenous variables are read in `util.find_optimal_lag_order()`.
- `train_only` was being ignored in `SeriesTransformer.DeseasonTransform()`.

## [0.18.1] - 2023-04-14
### Added
- Added functions to create expanding dynamic intervals using the conformal framework with backtesting: `util.backtest_for_resid_matrix()`, `util.get_backtest_resid_matrix()`, and `util.overwrite_forecast_intervals()`.
- Added a check for the correct estimator type with `Forecaster.auto_Xvar_select()`.
### Changed
### Fixed
- Fixed `Forecaster.export_Xvars_df()`, which wasn't working correctly when exogenous regressors were added with `Forecaster.ingest_Xvars_df()`.

## [0.18.0] - 2023-04-13
### Added
- Added tbats model.
- `MVForecaster` can now have Xvars added to it.
- Added `multiseries.line_up_dates()` function.
- New argument `test_again` added to `Forecaster.manual_forecast()` and `MVForecaster.manual_forecast()`. `auto_forecaster()` in the same objects has the same new argument.
- `model` argument added to `Forecaster.save_feature_importance()` and doesn't have to be called right after a model is evaluated.
- Added `verbose` args to `cross_validate()` and `util.find_optimal_transformation()` functions to offer more transparency around these processes.
- Added `min_grid_size` argument to `Forecaster.limit_grid_size()` and `tune_test_forecast()`.
### Changed
- Distinction between level/non-level is no longer tracked within `Forecaster` and `MVForecaster` objects and all methods that facilitated with that have gone away (`diff()`, `integrate()`, etc). Use `SeriesTransformer` to gain the same plus more functionality.
- Cross validation and tuning faster due to using more efficient objects.
- Cross validation has more customization available.
- RNNs and naive models can be optimized using grid search.
- All forecasting functions rewritten to be more efficient by using more efficient objects and fewer loops.
- Added `Forecaster.chop_from_front()` and `Forecaster.chop_from_back()` methods.
- Less information stored in history.
- Can only refer to series with default or user-selected names within `MVForecaster`.
- The structure of `MVForecaster` has changed. `mvf.y` is a dictionary with series in it instead of being located in `mvf.series1['y']`. The old way was to keep better track of level/nonlevel series but that no longer exists.
- test_only arg is gone in `Forecaster.manual_forecast()` in favor of `Forecaster.test()`, `MVForecaster.test()`. All out-of-sample methods facilited with `chop_from_front()` to split data and ensure no data leaks.
- `MVForecaster.add_optimizer_func()` now accepts functions.
- `MVForecaster.manual_forecast()` now accepts `Xvars` as an argument.
- `normalizer` argument considered a hyperparameter where applicable and not given its own entry in history.
- Got rid of several `Forecaster` methods that are never demonstrated in examples.
- Got rid of `Forecaster.backtest()` and `MVForecaster.backtest()`. `Pipeline.backtest()` is a better alternative.
### Fixed
- The first element in m is taken if multiple are passed to `util.find_optimal_transformation()`. The code was taking the second.
- Fixed the diffy arg in `Forecaaster.adf_test()`.
- `cvkwargs` were not being passed to the `cross_validate()` function in `notebook.tune_test_forecast()`.
- `util.find_optimal_transformation()` was only using first value of `m` for seasonal adjustments when multiple were passed.

## [0.17.20] - 2023-04-02
### Added
### Changed
### Fixed
- Fixed the `MVPipeline.backtest()` method to return consistent results when a transformation is not taken on the `Forecaster` objects.

## [0.17.19] - 2023-04-01
### Added
### Changed
- `util.find_optimal_transformation()` now uses pipeline backtesting, making it harder to leak data into the decision and also to make cross validation possible.
### Fixed

## [0.17.17] - 2023-03-31
### Added
- `Forecaster`/`MVForecaster` objects pickled in now re-initiate warning logging for estimated models.
### Changed
### Fixed

## [0.17.16] - 2023-03-29
### Added
- Added `util.metrics.abias()` function.
### Changed
- Revamped the RNN model. Now accepts exogenous regressors and other small changes that I believe will make it faster and more accurate, as well as allow for more customization.
### Fixed
- Fixed the `Pipeline.backtest()` function, which was adding too much space between consecutive training sets.
- Fixed the train_only argument in `SeriesTransformer.DetrendTransform()` when `loess = True`.

## [0.17.15] - 2023-03-27
### Added
- Added a LOESS detrender (from statsmodels) to the `SeriesTransformer` object. Called using `SeriesTransformer.DetrendTransform()`.
### Changed
- `SeriesTransformer` can seasonally adjust the same series multiple times (to capture multiple seasonalities).
- The `util.find_optimal_transformation()` function now tries a LOESS detrender by default.
### Fixed

## [0.17.14] - 2023-03-17
### Added
### Changed
- "Differenced" no longer displayed when calling `Forecaster.__repr__()`.
### Fixed
- Fixed plotting confidence intervals on separate axes in the `Forecaster` and `MVForecaster` objects.

## [0.17.13] - 2023-03-12
### Added
- Added `Forecaster.STL()` method.
- Added `cilevel` argument to `Pipeline.backtest()` method.
- Added `SeriesTransformer.DeseasonTransform()` and corresponding revert function.
- Added `util.metrics.bias()` function.
- Added `seasonal_adj` in default `try_order` argument from `util.find_optimal_transformation()`.
### Changed
### Fixed
- Fixed how metrics are calculated after reverting a detrend transformation.

## [0.17.12] - 2022-03-08
### Added
- Added a `Forecaster.add_metric()` and `MVForecaster.add_metric()` argument that can be used for custom metrics.
- Added a default `called` argument in `MVForecaster.add_optimizer_func()` to be the function's name.
### Changed
- Changed how metrics are set when initializing the objects and how they can be set later.
- Changed how metrics can be accepted in `util.backtest_metrics()`.
### Fixed
- Fixed the error message that is displayed when setting metrics goes wrong.


## [0.17.11] - 2022-03-03
### Added
### Changed
- Removed pandas-datareader as a dependency.
### Fixed

## [0.17.10] - 2023-03-02
### Added
### Changed
- Small changes in documentation.
### Fixed
- Fixed so that `Forecaster` and `MVForecaster` objects can be pickled again (broke after 0.17.7 update). Added better testing of that feature.

## [0.17.9] - 2023-02-26
### Added
- Added `Forecaster.add_signals()` method for custom model stacking. 
### Changed
- Took out the LevelTestSetActuals column from `MVForecaster.export('model_summaries')`.
### Fixed
- Fixed the default `determine_best_by` arg in `Forecaster.export()`.

## [0.17.8] - 2023-02-25
### Added
### Changed
- Took out an error when setting validation metric to R2 and less than 2 obs in validation set. Because of cross validation, this error isn't applicable.
### Fixed
- Fixed `Forecaster.set_metrics()` method.
 
## [0.17.7] - 2023-02-24
### Added
### Changed
- Code refactoring to gain efficiency and make coding base more maintainable. Better practices being followed.
- Users can now choose which metrics to evaluate for any forecasting procedure. The defaults remain what they always were.
- Changed the documentation in a few key spots to (hopefully) be clearer. For example, `Forecaster._estimators_` global is now `Forecaster().estimator` attribute. This also makes pickling more convenient.
### Fixed

## [0.17.6] - 2023-02-23
### Added
### Changed
- Updated requirements in test directory.
### Fixed
- Fixed `auxmodels.mlp_stack()` for multivariate forecasting.

## [0.17.5] - 2023-02-22
### Added
- Added the catboost model to the out-of-the-box models. Made catboost a required dependency.
- Added catboost grids to example grids files.
- Added `backtest()` method for `Pipeline` and `MVPipeline`.
- Added the `util.backtest_metrics()` function.
- Added `FutureWarning`s to `Forecaster.backtest()` and `MVForecaster.backtest()` related methods. All backtesting will be moved to pipelines in the future.
### Changed
### Fixed

## [0.17.4] - 2023-02-14
### Added
- Added an explicit error for when `auxmodels.mlp_stack()` is specified incorrectly.
- Added `ax` argument to more plotting functions.
- Added test scripts to the GitHub repository.
- Added an explicit error in `Forecaster.auto_Xvar_select()` for situations when monitoring a test metric but no test set is specified.
### Changed
- Explicitly specify `dtype = float` for parts of code that raise a warning when this is not the case.
- Changed to relative library imports wherever possible.
- Changed the `Forecaster.__deepcopy__()` to be more explicit.
- Changed how `GridGenerator` gets grids and added more grids files to choose from.
- Got rid of some confidence-interval information when exporting model summary info from `MVForecaster`.
- `MVForecaster.export('lvl_test_set_predictions')` no longer fails when there is no test set.
### Fixed
- Fixed the `AnomalyDetector.EstimatorDetect()` function.
- Fixed `util.break_mv_forecaster()` for situations when `MVForecaster` does not have a test set.
- Fixed how exporting works in `MVForecaster` so that errors are not returned when test set is 0 and default args are maintained.

## [0.17.2] - 2023-02-09
### Added
### Changed
- Explicitly specify `dtype = float` for parts of code that raise a warning when this is not the case.
- Logging a warning from `SeriesTransformer.DetrendTranform()` instead of raising it.
### Fixed

## [0.17.1] - 2023-02-08
### Added
- Added `FutureWarning`s to functions that touch the `Forecaster.diff()` method. This method will be removed soon as the `SeriesTransformer` object is a better alternative that does the same thing.
- Added `diffy` argument to `Forecaster.adf_test()`.
- Added a `correct_residuals()` function that corrects residuals for autocorrelation before building confidence intervals. 
### Changed
- The `train_only` default argument in `auxmodels.auto_arima()` changed to False.
- Changed how default arguments are parsed in `Forecaster.export()`.
- Refactored code to avoid a `SettingWithCopyWarning` from pandas in `Forecaster.ingest_Xvars_df()`.
### Fixed
- Removed the `print_attr` argument from `notebook.results_vis()` that is no longer accepted in `Forecaster.diff()`.
- Removed a warning from `Forecaster.ingest_Xvars_df()` that shouldn't have been raised.

## [0.17.0] - 2023-02-06
### Added
- Added the `cis` argument to `Forecaster__init__()` and `MVForecaster.__init__()`.
### Changed
- Only conformal intervals now supported in `Forecaster` and `MVForecaster`. By default, these will not be generated and can only be generated if there is a test set and it is specified to a sufficient length.
- Removed `Forecaster.proba_forecast()` and `MVForecaster.proba_forecast()` and all probabilistic arguments in functions.
- Default `test_length` argument in `Forecaster.__init__()` and `MVForecaster.__init__()` is now 0.
- Took out the `exog_coint` argument from `vecm.__init__()` since it's not actually usable.
- Only model-specific and hyperparameter optimization warnings will be logged in warnings.log and other warnings will be printed.
- Changed many categories of warnings from `UserWarning` to `Warning`.
### Fixed
- Added `staticmethod` decorators to `util.metrics` methods. This doesn't change functionality, just the documentation.

## [0.16.6] - 2023-02-03
### Added
### Changed
### Fixed
- Fixed the warning in `Forecaster.ingest_Xvars_df()`.

## [0.16.5] - 2023-01-31
### Added
### Changed
### Fixed
- Fixed an error that can arise from the `util.break_mv_forecaster()` function.
- Fixed the `AnomalyDetector.EstimatorDetect()` function.

## [0.16.4] - 2023-01-31
### Added
- Added mapie as a requirement.
### Changed
- `Forecaster.proba_forecast()` now uses conformal prediction from the mapie package to create confidence intervals and is much more efficient. This will be also implemented in `MVForecaster` soon.
- Took out 'CIPlusMinus' as a history key in `Forecaster` and `MVForecaster`.
### Fixed

## [0.16.3] - 2023-01-29
### Added
- Added a naive/seasonal naive estimator to `Forecaster`.
### Changed
- `Forecaster.add_diffed_terms` no longer supports second differencing.
### Fixed
- Fixed the warning that gets passed for the RNN and Silverkite models when `dynamic_testing` is passed as `False`.

## [0.16.0] - 2023-01-28
### Added
### Changed
- It is now possible to skip model testing by setting test_length = 0 in `Forecaster` and `MVForecaster`.
- Got rid of LastTestSetPrediction and LastTestSetActuals columns from model summary df.
- Got rid of `best_fcst` DataFrame.
- Gave documentation a once-over.
- Refactored code to be slightly more efficient.
### Fixed
- The scaler in `MVForecaster` was not just being applied to the test-set inputs and was being called twice per model train.

## [0.15.16] - 2023-01-25
### Added
- added `ax` argument to all forecast plotting functions.
### Changed
### Fixed
- removed `tensorflow` from the list of requirements and added it and `tensorflow-macos` to the optional add-on list (#31)

## [0.15.14] - 2022-01-23
### Added
- added a str representation for the `SeriesTransformer` object
- added `exclude_models` argument to `SeriesTransformer.Revert()` and similar functions
### Changed
### Fixed
- cleaned up some documentation links

## [0.15.12] - 2022-12-12
### Added
### Changed
- fill N/A for all forecasts with a forward fill to prevent some max-value errors, both when forecasting and reverting transformations
### Fixed

## [0.15.11] - 2022-12-06
### Added
### Changed
### Fixed
- fixed exception catching in `Forecaster.auto_Xvar_select()` and `util.find_optimal_transformation()`

## [0.15.10] - 2022-12-05
### Added
- added `dayofyear` to search for in `Forecater.auto_Xvar_select()` when frequency is daily or lower.
- added `cycle_lens` argument to `Forecaster.add_seasonal_regressors()`.
### Changed
### Fixed
- fixed how seasonalities are selected when they are not Fourier transformed in `Forecaster.auto_Xvar_select()`.

## [0.15.9] - 2022-11-03
### Added
### Changed
### Fixed
- fixed getting cis with backtesting. only the first backtest iteration was being used.

## [0.15.8] - 2022-11-02
### Added
- added `method` argument to `Forecaster.reeval_cis()` and `MVForecaster.reeval_cis()` and included an option to get confidence intervals through backtesting.
- added `Forcaster.plot_backtest_values()`.
### Changed
- changed some error messages to be more descriptive and to encourage raising issues on github.
- changed how level confidence intervals are obtained when calling `SeriesTransformer.DiffRevert()` to be more efficient.
- the backtest_values DataFrame now includes dates in `Forecaster` and `MVForecaster`. the order of the dataframe is Date --> Actuals --> Preds for all iterations.
### Fixed

## [0.15.7] - 2022-10-28
### Added
### Changed
- changed `dynamic_testing = <int>` to same way it was in 0.12.8 when it was introduced because it gives better and more efficient results in both `Forecaster` and `MVForecaster`.
### Fixed

## [0.15.6] - 2022-10-25
### Added
### Changed
- the `m` argument in `util.find_optimal_transformation` can be a list and multiple seasonal differences can be tried in that function
- changed how level plotting is peformed in `Forecaster.plot()` and `Forecaster.plot_test_set()`
### Fixed
- fixed how history['LevelY'] attribute is undifferenced in `SeriesTransformer`, which was causing plots to be incorrect.

## [0.15.5] - 2022-10-22
### Added
- added the `Pipeline.MVPipeline` object.
### Changed
### Fixed

## [0.15.4] - 2022-10-21
### Added
### Changed
### Fixed
- changed where import calls occur in the `SeriesTransformer` module to avoid a circular import that can happen
- added `AttributeError` to the list of exceptions to catch in `util.find_optimal_transformations()` function.
- fixed initiating MVForecaster() when `merge_Xvars=='i'`.

## [0.15.3] - 2022-10-20
### Added
- increased documentation around forecasting different model types.
### Changed
- 'LevelY' passed to history in `util.break_mv_forecaster()`
- changed the optional dependency `pip intall fbprophet` to `pip install prophet` (#18)
### Fixed
- added `IndexError` to the list of exceptions to catch in `util.find_optimal_transformations()` function.
- convert values in `**kwargs` from `numpy.bool_` to `bool` type when forecasting with HWES (#19).

## [0.15.2] - 2022-10-19
### Added
- added the `util.find_optimal_transformation()` function.
### Changed
- changed setup.py so that version needs to be added manually to make an update (#16).
- `Forecaster.integrate()` now has `**kwargs` that are passed to `Forecaster.adf_test()`.
- changed some data processing in `SeriesTransform.DiffRevert()` to accomodate a future change from pandas.
- changed the name of `util.find_series_transformation()` to `util.find_statistical_transformation()`. this function is new as of 0.15.0 so hopefully it has not been adopted widely yet and this change doesn't cause issues for users.
- added a try/except wrapper around the `self.f.typ_set()` code in `SeriesTransformer.DetrendTransform()`
### Fixed
- drops null values before running though `Forecaster.normality_test()`.
- changed the import in `util` to `from scalecast import Pipeline` as the `util.find_series_transformation()` was not working because of it.

## [0.15.1] - 2022-10-18
### Added
- added the `SeriesTransformer.DetrendTransform()` and `SeriesTransformer.DetrendRevert()` functions.
### Changed
### Fixed
- all reverter functions in `SeriesTransformer` revert level confidence intervals.

## [0.15.0] - 2022-10-17
### Added
- added the `Pipeline` module.
- added `util.find_optimal_series_transformation()`
- added `Forecaster.normality_test()` function
### Changed
- changed the default `full_res` arg in `Forecaster.adf_test()` to `True`
- changed links to some model documentation in readme
### Fixed

## [0.14.8] - 2022-10-14
### Added
- added `must_keep` arg to `Forecater.auto_Xvar_select()`.
- added `SeriesTransformer.SqrtTransform()` and `SeriesTransformer.SqrtRevert()` functions.
### Changed
- two-level differencing no longer natively supported in `Forecaster` and `MVForecaster`. it's too much work to maintain two differencing and it is also supported more efficiently and dynamically through the `SeriesTransformer` object and that is available for who needs it.
  - this changed the arguments in the following functions:
    - [`Forecaster.diff()`](https://scalecast.readthedocs.io/en/latest/Forecaster/Forecaster.html#src.scalecast.Forecaster.Forecaster.diff)
    - [`Forecaster.integrate()`](https://scalecast.readthedocs.io/en/latest/Forecaster/Forecaster.html#src.scalecast.Forecaster.Forecaster.integrate)
    - [`Forecaster.undiff()`](https://scalecast.readthedocs.io/en/latest/Forecaster/Forecaster.html#src.scalecast.Forecaster.Forecaster.undiff)
    - [`util.pdr_load()`](https://scalecast.readthedocs.io/en/latest/Forecaster/Util.html#module-src.scalecast.util.pdr_load)
- changed accepted values that can be passed to the `probabilistic` arg in the `Forecaster.tune_test_forecast()`, `MVForecaster.tune_test_forecast()`, and `notebook.tune_test_forecast()` functions so that some models can be forecasted probabilistically and others don't have to be (speeds up processing time generally).
### Fixed
- issues with second-differencing have all been resolved since second differencing is no longer supported natively in the `Forecaster` object.
- took out `revert_fvs` from `SeriesTransformer.DiffRevert()` function because it hasn't been working.

## [0.14.7] - 2022-10-05
### Added
### Changed
### Fixed
- added `trend_estimator_kwargs` to the `Forecaster.auto_Xvar_select()` function to avoid an error that occurs when mixing estimators and not using default hyperparameters

## [0.14.6] - 2022-10-03
### Added
### Changed
- changed the default figsize for all plotting functions to (12,6).
### Fixed

## [0.14.5] - 2022-09-30
### Added
### Changed
- added `figsize` argument in `notebook.results_vis()` and `notebook.results_vis_mv()`
### Fixed

## [0.14.4] - 2022-09-23
### Added
- added the tf_model attribute to the `Forecaster` object for users to access and save rnn and lstm models (#12)
- added `figsize` arg to all forecast plotting methods in `Forecaster` and `MVForecaster`
- added a link to the M4 example in the readme
### Changed
### Fixed
- took out the `freq` argument from `auxmodels.vecm` (which was being set automatically) since it is optional when dates are passed and was causing frequencies that statsmodels does not recognize to fail (#13)

## [0.14.3] - 2022-09-16
### Added
- `util.pdr_load()` now accepts multiple series and returns an MVForecaster object of everything loaded together
- added more arguments to the `util.pdr_load()` function
- added `auxmodels.vecm` model, which is a model class that can be imported using the `MVForecaster.add_sklearn_estimator()` function (#11)
- modified the source code in the `MVForecaster` object to accomodate new model classes (e.g. vecm)
- added a vecm grid to the example grids
- added the `util.find_optimal_coint_rank()` and `util.find_optimal_lag_order()` functions
### Changed
- changed scaling syntax in `Forecaster` and `MVForecaster` to circumvent a warning having to do with feature names--only numpy arrays are scaled now (not dataframes)
### Fixed
- added a call of `Forecaster.Forecaster.typ_set()` right after `MVForecater.__init__()`, before chopping dates to fix weird loading errors that occured once in a while
- fixed the util function that wasn't working in 0.14.2 and yanked that release. everything scheduled for 0.14.3 will now be part of 0.14.4.

## [0.14.1] - 2022-09-09
### Added
- if there are not enough observations to use in cross validation (usually because too many AR terms were added), an error is raised when calling the `Forecater.cross_validate()` and `MVForecaster.cross_validation()` functions
### Changed
- no `Forecaster.auto_Xvar_select()` no longer raises errors if more AR terms passed to max_ar argument than the model is able to estimate
### Fixed
- `Forecaster.determine_best_series_length()` will no longer fail if the min_obs arg value is greater than the amount of observations in the series
- found more instances where `TypeError`s should not be raised (such as passing an `int64` type when `int` is required)
- fixed an issue that occurs after selecting Xvars with `Forecaster.auto_Xvar_select()` on an integrated series then loading to `MVForecaster`
- fixed the error raised when 0 or less is passed to the `Forecaster.set_validation_length()` and `MVForecaster.set_validation_length()` functions

## [0.14.0] - 2022-08-31
### Added
- added the `Forecaster.auto_Xvar_select()` method
- added a check for NAs in `Forecaster` and `MVForecaster` when evaluating grids and validation metric is mape. a descriptive error is raised if NAs are found (#10)
- added `Forecaster.drop_all_Xvars()`
- added `Forecaster.determine_best_series_length()`
- added `Forecaster.restore_series_length()`
### Changed
- removed deprecated functions identified and labeled in 0.13.1
- `Forecaster.keep_smaller_history()` can now accept numpy int types as an argument.
### Fixed

## [0.13.11] - 2022-08-19
### Added
- added grids_file attribute to `Forecaster` and `MVForecaster` objects, as well as `set_grids_file()` method to both objects.
### Changed
### Fixed

## [0.13.10] - 2022-08-15
### Added
- added probabilistic argument option to `auxmodels.mlp_stack()` function.
- added Xvars argument to `auxmodels.auto_arima()` function.
### Changed
- made it so that an error is raised earlier when using `Forecaster.ingest_Xvars_df()` incorrectly.
### Fixed
- fixed an issue with `auxmodels.mlp_stack()` where `**kwargs` were not being passed correctly.

## [0.13.9] - 2022-08-11
### Added
- added error arg to `Forecaster.tune_test_forecast()` and `MVForecater.tune_test_forecast()` methods.
### Changed
- took out an error check that was redundant and not even working
### Fixed
- fixed some documentation syntax for new objects added last dist

## [0.13.8] - 2022-08-08
### Added
- added `auxmodels` module with `auto_arima()` and `mlp_stack()` functions
- added pmdarima as an optional requirement to make auto_arima work
- added `util.metrics` class
- added `Forecaster.reeval_cis()` and `MVForecaster.reeval_cis()` methods
### Changed
### Fixed

## [0.13.6] - 2022-08-04
### Added
- added `error` argument to the `Forecaster.diff()` method
### Changed
- took out the error that's raised when trying to add AR terms after data has already been differenced using `Forecaster.add_AR_terms()`
### Fixed
- fixed an issue with `util.break_mv_forecaster` that was caused from adding `future_dates` arg to `Forecaster.__init__()` method

## [0.13.5] - 2022-08-03
### Added
- added optional `future_dates` arg to `Forecaster.__init__()` method
- added `error` argument to the `Forecaster.drop_Xvars()` and `Forecaster.drop_regressors()` methods
### Changed
- changed `SeriesTransformer` scaling functions to use only training data if `train_only=True`.
- took out the error that's raised when trying to add AR terms after data has already been differenced
### Fixed
- fixed an issue with the `notebook.tune_test_forecast()` function
- `MVForecaster` no longer takes AR terms when `merge_Xvars = 'u'`
- fixed an issue where `util.break_mv_forecaster` was not converting xreg dict correctly

## [0.13.4] - 2022-07-29
### Added
- added `util.pdr_load()` function.
- added `limit_grid_size` as an argument to `Forecaster.tune_test_forecast()`, `MVForecaster.tune_test_forecast()`, and `notebook.tune_test_forecast()` to support randomized grid search through this process.
### Changed
- changed dynamic window forecasting loop
- if trying to cross validate with less data than it is possible to create the correct-sized folds for, the program will no longer raise an error but instead pass default parameters to the `best_params` attribute and log a warning.
- made cross validation slightly more efficient
### Fixed
- changed some source code to reduce the amount of `TypeError`s a user is likely to get (such as passing `int.64` type when `int` type is required)

## [0.13.3] - 2022-07-27
### Added
### Changed
- `SeriesTransformer.diffrevert()` now supports an argument `revert_fvs`, which is `True` by default. since adding level cis, this is now possible.
### Fixed
- fixed an issue that caused model evaluation to fail if models were not tuned from the grid successfully. this was an issue since 0.12.3 due to how cross validation changed tuning.

## [0.13.2] - 2022-07-24
### Added
### Changed
### Fixed
- Fixed an issue from 0.13.1 that was caused by running models `test_only = True` on integrated series.

## [0.13.1] - 2022-07-24
### Added
- added level fitted values and default level confidence intervals for all models called through `Forecaster` and `MVForecaster`.
### Changed
- deprecated several export functions and rewrote `Forecaster.export()` and `MVForecaster.export()` to allow confidence intervals when `cis=True`. all deprecated functions should log a FutureWarning and will be removed in 0.14.0. all of these functionalities are now dupliated in `Forecaster.export()` and `MVForecaster.export()`
  - `Forecaster.export_test_set_preds_with_cis()`
  - `Forecaster.export_test_set_preds_with_cis()`
  - `MVForecaster.export_model_summaries()`
  - `MVForecaster.export_forecasts()`
  - `MVForecaster.export_test_set_preds()`
  - `MVForecaster.export_level_forecasts()`
  - `MVForecaster.export_level_test_set_preds()`
- made shap an optional add-on due to some installation issues by some users
### Fixed
- `notebook.tune_test_forecast()` was missing an argument in the function
- fixed an issue with `MVForecaster.backtest()` causing some models to return a key error when backtested
- fixed an issue where `'ValidationMetricValue'` could not be passed to `MVForecaster.set_best_model(determine_best_by)`

## [0.13.0] - 2022-07-19
### Added
- added probabilistic forecasting through `Forecaster.proba_forecast()` and `MVForecaster.proba_forecast()` methods
- added level confidence intervals always for models run at level (so it won't fail to generate cis anymore when passing `Forecaster.plot(ci=True,level=True)`)
- probabilistic forecasting also makes it possible to derive level confidence intervals even when model was run at difference
- added probabilistic as a `bool` argument to `Forecaster.tune_test_forecast()`, `MVForecaster.tune_test_forecast()`, and `notebook.tune_test_forecast()` functions
### Changed
- changed how it was determined that a model was tuned for efficiency gains
- changed so that "Dynamically" does not appear in the history['tuned'] attribute
- changed the error that is raised when reduce_Xvars() doesn't work due to feature importance not being supported by a given model so that it is more explicit
- changed the default mlp grid so that random_state is no longer a value. this makes that model more ammenable to probabilistic forecasting
### Fixed

## [0.12.9] - 2022-07-15
### Added
- added `suffix` argument to `Forecaster.tune_test_forecast()`, `MVForecaster.tune_test_forecast()`, and `notebook.tune_test_forecast()` functions (#5)
- added `fi_method` argument to `notebook.tune_test_forecast()` function
### Changed
### Fixed

## [0.12.8] - 2022-07-11
### Added
- added `AnomalyDetector.MonteCarloDetect_sliding()` method
### Changed
- changed the `dynamic_testing` and `dynamic_tuning` arguments in `Forecaster` and `MVForecaster` so that window forecast evaluation is now supported. now, instead of having the choice between 1-step and arbitrary multi-step forecasting, any integer value is accepted as arguments in those parameters but `True` and `False` are still supported and do the same thing as always.
### Fixed

## [0.12.7] - 2022-07-08
### Added
### Changed
- got rid of printing when calling `ChangepointDetector.WriteCPtoXvars()`
### Fixed
- fixed cross validation when `Xvars = 'all'` is passed as an argument

## [0.12.6] - 2022-07-06
### Added
- added the `ChangepointDetector` object
- added the `AnomalyDetector.adjust_anom()` method
### Changed
### Fixed
- fixed last index span from `AnomalyDetector.MonteCarloDetect()`

## [0.12.5] - 2022-07-01
### Added
- added `jump_back` parameter in `Forecaster.backtest()` and `MVForecaster.backtest()` methods
- added the theta model from darts
### Changed
- changed how dataframes are grouped in the `cross_validate()` method. turned sorting off to prevent some failures, specifically in the theta model
### Fixed

## [0.12.4] - 2022-06-28
### Added
### Changed
- changed how shap feature scores are sorted in reduce Xvars, no adjustment needed like with PFI
### Fixed

## [0.12.3] - 2022-06-27
### Added
- added shap feature importances in addition to pfi by allowing user to select method = 'shap' when calling `Forecaster.save_feature_importance()`
- added shap library to dependencies list
- added `SeriesTransformer` class
- added `AnomalyDetector` class
- added function to util that breaks an `MVForecaster` class into several objects of `Forecaster` class
- can now init `Forecaster` object with `require_future_dates = False`. when using False, the object will not forecast into future dates and will not make you know values into the future for regressors passed through `Forecast.ingest_Xvars_df()`.
### Changed
- took the 'per' key out of the history attribute
- changed the order of some of the source code to be more efficient (very small gains)
- changed the size of the dataset that pfi feature importance is called on to make it include all values previously seen by any given model passed to it. before, it sliced off the last couple observations only -- this was more or less a mistake but I don't expect results will be affected significantly for anyone using `reduce_Xvars()` due to how features are sorted in that function.
- changed the `Forecaster.reset()` function so that it **returns** a true copy of the initiated object.
- in `Forecaster.save_feature_importance()`, added the `on_error` arg to raise errors if the user prefers. The default is still to log errors as warnings so as not to break loops.
### Fixed
- `test_only` was not working with the lstm estimator, so fixed that
- fixed an issue where the function didn't ignore the argument passed to `estimator` with `reduce_Xvars(method='l1')`

## [0.11.2] - 2022-06-20
### Added
### Changed
### Fixed
- fixed an issue where `None` wasn't being accepted in grid with `'Xvars'` as the key and using `cross_validate()` (#2)

## [0.11.1] - 2022-06-15
### Added
### Changed
- changed how the validation set length is calculated in history attributed -- given na value if cross validation used to tune models
### Fixed
- fixed an issue caused by None values in hyperparam grids being changed to np.nan and therefore not accepted in some functions after cross validation has been called

## [0.11.0] - 2022-06-14
### Added
- added `cross_validate()` methods to `Forecaster` and `MVForecaster` objects, which can now be used for the same purposes as `tune()` but with cross validation
- added `cross_validate` as a (bool) argument to the `Forecaster.tune_test_forecast()`, `Forecaster.reduce_Xvars()`, `MVForecaster.tune_test_forecast()`, and `notebook.tune_test_forecast()` functions
- added "CrossValidated" key to history dict in `Forecaster` and `MVForecaster` objects
### Changed
- if np.nan is passed as a normalizer value, it will convert to None so that it can be used
### Fixed


## [0.10.5] - 2022-06-07
### Added
### Changed
### Fixed
- fixed the `set_best_model()` method in `MVForecaster`, which was broken due to not being able to parse new updates from 0.10.1

## [0.10.4] - 2022-06-05
### Added
### Changed
### Fixed
- sorting from pfi variable reduction method was incorrect after initial variable drop, causing the wrong sequence of variables to be dropped when method = 'pfi'

## [0.10.3] - 2022-06-05
### Added
- added lasso and ridge as default estimators and gave them default grids for both `Forecaster` and `MVForecaster`
- added `reduce_Xvars()` method to `Forecaster` object
- added `util` module with one function (for now) to plot error changes from calling the `Forecaster.reduce_Xvars()` method
### Changed
- Changed default hyperparameter grid values for knn, xgboost, and lightgbm estimators
### Fixed

## [0.10.2] - 2022-05-17
### Added
### Changed
- Changed where logs are called back to when importing library due to warnings that don't get logged when unpickling object
### Fixed

## [0.10.1] - 2022-05-16
### Added
- `MVForecaster.set_optimize_on()` now accepts user functions (like weighted averages) by leveraging new `MVForecaster.add_optimizer_func()` function
### Changed
- Suppressed future warnings on import because of a pandas warning caused from importing LightGBM. If LightGBM does not fix this issue with an update, we might need to pull it from the default list of sklearn estimators; users would still be able to use LightGBM by importing it manually.
### Fixed

## [0.10.0] - 2022-05-13
### Added
- added permutated feature dataset to history for sklearn models with feature importance
### Changed
- changed some pandas syntax to avoid some warnings
- permutation feature importance applied on test set only and works better on models run `test_only = True`
### Fixed
- fixed the order in which Xvars are parsed, which was causing issues when Xvar names were added in a function in a different order than they appeared in the object 

## [0.9.9] - 2022-05-12
### Added
### Changed
- warning logs now called after `Forecaster` and `MVForecaster` objects are initiated
- No more `get_funcs()` method, a bad idea from the start
### Fixed


## [0.9.8] - 2022-05-11
### Added
- added the `corr()` and `corr_lags()` methods to the `MVForecaster` object
### Changed
### Fixed

## [0.9.7] - 2022-05-10
### Added
### Changed
### Fixed
- fixed an issue where `MVForecaster` was not generating the correct forecast horizon worth of values for models run with lags only (no seasonality or trends)

## [0.9.6] - 2022-05-06
### Added
### Changed
- got rid of printing when calling silverkite
### Fixed
- fixed an issue with ARIMA where it didn't work if no Xvars had been added first

## [0.9.4] - 2022-04-29
### Added
### Changed
### Fixed
- fixed an issue where data types get changed from int to float when grid searching

## [0.9.2] - 2022-04-29
### Added
- Added `SGDRegressor` from sklearn as a default estimator (sgd)
### Changed
- Changed "backcast" nomenclature to "backtest", which is what it really is (oops!)
### Fixed

## [0.9.1] - 2022-04-21
### Added
### Changed
### Fixed
- fixed an issue in ARIMA model where future x regressors weren't being read correctly
- fixed an issue where `save_feature_importance()` was throwing an error sometimes

## [0.9.0] - 2022-04-15
### Added
- added `backcast()` method to `Forecaster` and `MVForecaster` objects
- added `copy()` and `deepcopy()` methods to the same objects
- added `test_only` options to all models in `Forecaster`. ignored in combo models. a future dist will have this for `MVForecaster`
- added `plot_loss_test` as arg in lstm and rnn models
- added `notebook.results_vis_mv()` function
### Changed
### Fixed

## [0.8.4] - 2022-04-11
### Added
### Changed
- changed how names are looked for in the `MVForecaster` object
### Fixed

## [0.8.3] - 2022-04-05
### Added
### Changed
### Fixed
- fixed an issue when plotting in MVForecaster that occurs when you didn't initially pass names to `__init__()`
- fixed an instance where labels was spelled wrong

## [0.8.1] - 2022-04-04
### Added
### Changed
### Fixed
- Fixed inconsistent indexing in MVForecaster that was causing some models to fail when different Forecaster objects' dates didn't align
- Fixed an issue where `level=True` wasn't working in `plot_test_set()` sometimes

## [0.8.0] - 2022-04-03
### Added
- Added more description to the `MVForecaster.__repr__()` method
### Changed
- Changed arguments in the rnn estimator to make it more flexible and dynamic
### Fixed

## [0.7.6] - 2022-03-30
### Added
### Changed
### Fixed
- Noticed an error when forecasting with Silverkite on differenced data and fixed it.

## [0.7.5] - 2022-03-28
### Added
### Changed
### Fixed
- `MVForecaster.plot()` wasn't working after a recent update, fixed now

## [0.7.4] - 2022-03-28
### Added
### Changed
### Fixed
- `MVForecaster.export()` wasn't working after last update, fixed now


## [0.7.3] - 2022-03-28
### Added
### Changed
- changed "BestModel" key in MVForecaster.export() method to "best_model" to be consistent with Forecaster notation
### Fixed
- was calling the wrong function when the series weren't the same length in the `MVForecaster.__init__()` method
- fixed `MVForecaster.set_optimize_on()` so that it accepts "mean" as an argument without error

## [0.7.2] - 2022-03-25
### Added
### Changed
### Fixed
- MVForecaster wasn't combining xvars with AR terms correctly
- MVForecaster wasn't plotting some series correctly
- MVForecaster wasn't making copies of arrays in dictionaries when copying xvars

## [0.7.1] - 2022-03-24
### Added
- added the MVForecaster object for multivariate vector forecasting
- added mv_grids to GridGenerator, which is written out to working directory as MVGrids (due to the added "lags" arg)
- added `multiseries.keep_smallest_first_date()` function to make the series of multiple Forecaster objects the same lengths
### Changed
### Fixed

## [0.7.0] - 2022-03-18
### Added
- added the `add_cycle()` method for identification of irregular cycles
### Changed
### Fixed

## [0.6.9] - 2022-03-15
### Added
### Changed
### Fixed
- fixed the default hidden_layers_type arg in `_forecast_rnn()`

## [0.6.8] - 2022-03-15
### Added
### Changed
### Fixed
- fixed the rnn estimator which was using a dict to set hidden layers without unique keys, causing all layers except the last to be ignored, so all rnns were returning one layer results. Now the args in that function are different, but tested and it works to add more layers.

## [0.6.7] - 2022-03-07
### Added
- added `N_actuals` to `__repr__()` function
### Changed
- Took pt normalizer out of all example grids since it still errors out sometimes without being obvious why
- `notebook.results_vis()` no longer prints anything and automatically sets the plot title instead
- `Forecaster.set_test_length()` now accepts fractional splits
### Fixed

## [0.6.6] - 2022-02-28
### Added
- added the `add_sklearn_estimator()` function so a user can add any sklearn regression model to forecast with.
### Changed
### Fixed
- some of the examples weren't plotting correctly after recent updates.

## [0.6.4] - 2022-02-25
### Added
- added residuals to the function `export_fitted_vals()`. Now gives dates, actuals, fitted vals, and residuals
- added `multiseries.export_model_summaries()` function
### Changed
### Fixed
- `notebook.results_vis()` broke after last update and has now been fixed

## [0.6.2] - 2022-02-25
### Added
### Changed
- the `plot()`, `plot_test_set()`, and `plot_fitted()` functions now return figures instead of automatically plotting for you. This means more customization is now possible. Took out `to_png` and related args from these functions since that can be done now with matplotlib
### Fixed
- changed the insample metric evaluations for RNN and LSTM to be on the full training set instead of just the last few observations.

## [0.6.1] - 2022-02-23
### Added
### Changed
### Fixed
- fixed an issue when calling RNN or LSTM models after a combo model that saved attributes of the combo model illogically in history
- fixed an issue that caused test-set evaluation of ARIMA models to be inaccurate (causing severe underperformance)

## [0.6.0] - 2022-02-09
### Added
- Added drop argument to several adder functions, giving the user the option to drop the original regressors after making certain transformations on them
### Changed
- `add_ar_terms()` now accepts 0 as an argument but it doesn't do anything
### Fixed
- fixed issue with `pop_using_criterion()` function that wasn't dropping models correctly in some instances
- fixed the default parameters for the rnn model which weren't working due to mislabeling of one of the parameters in TensorFlow. updated the docstring accordingly

## [0.5.9] - 2022-02-01
### Added
- Added CurrentEstimator to `__repr__()` function.
### Changed
### Fixed
- Made it impossible to forecast without adding future dates first

## [0.5.8] - 2022-01-27
### Added
### Changed
- Made it impossible to pass tune argument to `manual_forecast()`
### Fixed
- Fixed results_vis in notebook (wasn't displaying one of the widgets)

## [0.5.7] - 2022-01-26
### Added
### Changed
- Cleaned up a lot of documentation
### Fixed
- Fixed links in readme

## [0.5.6] - 2022-01-25
### Added
- Added documentation on Read the Docs
- Added cilvel information to export functions
### Changed
### Fixed

## [0.5.5] - 2022-01-20
### Added
- Added CILevel info to export model_summaries function
### Changed
### Fixed
- Fixed an issue where plots were diplaying incorrect confidence levels if `cilevel` had been changed since training it

## [0.5.4] - 2022-01-20
### Added
- Added the rnn estimator
### Changed
- plot_loss argument no longer considered a hyperparameter value for LSTM and RNN models
### Fixed
- Fixed an issue where "==" wasn't being accepted in the `evaluated_as` argument in the `pop_using_criterion()` function
- Scaler in history saved as 'minmax' instead of None for LSTM and RNN models

## [0.5.3] - 2022-01-18
### Added
- EvaluatedModels to `__repr__()`
### Changed
- Does not call `infer_freq()` as often, making code more efficient
### Fixed
- sometimes the attribute `ci_bootstrap_samples` was being called `bootstrap_samples`, changed everything to `bootstrap_samples` only

## [0.5.2] - 2022-01-12
### Added
### Changed
### Fixed
- Fixed an error that occured when calling the `__repr__()` method if no models had been evaluated first

## [0.5.1] - 2022-01-12
### Added
- Added `export_fitted_vals()` function
- Added ci option to the `results_vis()` function in notebook
- Added the `get_funcs()` function
### Changed
- No `Xvars` in LSTM model, changed to lags (now model will only look at its own history)
- No `normalizer` in LSTM model (always uses a minmax scaler now)
- LSTM model can no longer be tuned
- Got rid of all lstm model grids
- changed `__str__()` and `__repr__()` so that they now offer better info
### Fixed
- Fixed the LSTM model by scaling the dependent variable and unscaling it (minmax) when it comes out and getting rid of other Xvars

## [0.5.0] - 2022-01-10
### Added
- Added confidence intervals using bootstrapping
  - `set_cilvel()` function (default .95)
  - `set_bootstrap_samples()` function (default 100)
- Added `ci` parameter to `plot()` and `plot_test_set()` function 
- Added UpperCI, LowerCI, TestSetUpperCI, TestSetLowerCI keys to history dict
- Added `export_forecasts_with_cis()` and `export_test_set_preds_with_cis()` functions
- Added source code (commented out) to get level confidence intervals -- when I tested, the intervals were too large to implement but maybe in the future this will be revisited
### Changed
### Fixed

## [0.4.4] - 2022-01-07
### Added
- added lstm grid in example grids
- added EarlyStopping callback functionality for the LSTM model
- added `get_expanded_lstm_grid()` to GridGenerator module which gives an example of a grid with early stopping
### Changed
- changed default paramaters for the lstm model
- added `**kwargs` to the lstm model forecast function that are passed to the `fit()` function in TensorFlow, got rid of `epochs` and `batch_size` args consequently
### Fixed

## [0.4.25] - (quick fix) 2022-01-06
### Added
### Changed
### Fixed
- source code was using `f` instead of `self` when when calling `pop_using_criterion()`

## [0.4.2] - 2022-01-06
### Added
- lstm estimator
- added `pop_using_criterion()` function
### Changed
- Fixed an issue where sklearn models were being fit on the same data twice -- does not change outcomes but the models run faster now
- Output from `_scale()` function is now always a numpy matrix (could have been either that or pandas dataframe before)
- sorted `_estimators_` list
- changed the error message for when importing a grid fails to account for one other possible reason the failure occured
### Fixed

## [0.4.1] - 2021-12-30
### Added
- Can now sort by metric value in `export_all_validation_grids_to_excel()`
### Changed
### Fixed
- Fixed an issue where sometimes the incorrect AR terms were being propogated for test-set evaluation only

## [0.4.0] - 2021-12-30
### Added
### Changed
- deleted the 'scale' normalizer from the mlp grid
### Fixed
- Fixed an issue with the PowerTransformer normalizer that failed because of a DivideByZero error, now defaults to a StandardScaler when this issue is encountered and logs a warning

## [0.3.9] - 2021-12-30
### Added
- Added `init_dates` and `levely` attributes
- Added `'Observations'` info to history and `export()`
- Added `'lvl_test_set_predictions'` to export dataframes
### Changed
- Got rid of `first_obs` and `first_dates` attributes and wrote more efficient code to do what they were there for
- More information available when `__str__()` is called
### Fixed
- Fixed what became an issue with the last update in which when calling `add_diffed_terms()` or `add_lagged_terms()`, the level series wasn't accurate due to how undifferencing was being executed. After examining this issue, it became evident that the previous way to undifference forecasts was less efficient than it should have been, this update fixed the issues from the last update and made the code more efficient
- Fixed an issue where AR terms were manipulating the underlying xreg structures so that each forecast were using its own test-set propogated AR values instead of the correct AR values
- Fixed the `export_Xvars_df()` method which wasn't working correctly if at least one forecast hadn't been called first

## [0.3.8] - 2021-12-29
### Added
- added the following functions that can each add additional Xvars to forecast with:
	- `add_exp_terms()` - for non polynomial exponential transformations
	- `add_logged_terms()` - for log of any base transformations
	- `add_pt_terms()` - for individual variable power transformations (box cox and yeo johnson available)
	- `add_diffed_terms()` - to difference non-y terms
	- `add_lagged_terms()` - to lag non-y terms
- added the 'pt' normalizer for yeo-johnson normalization (in addition to 'minmax', 'normalize', and 'scale')
- added the `drop_Xvars()` function that is identical to the `drop_regressors()` function
### Changed
- imports all sklearn models as soon as scalecast is imported
- src code cleanup with better coding practices when it comes to forecasting sklearn models (no more copying and pasting new functions)
- changed several set data types to lists in src code
- changed the names of some hidden functions
- other src code cleanup for readability and minor efficiency gains
- better in-line comments and docstring documentation
- got rid of quiet paramater in `save_summary_stats()` and `save_feature_importance()` and now these simply log any problems as warnings
- time trends now start at 1 instead of 0 (makes log transformations possible)
- observation dropping for AR terms in sklearn models now based on the number of N/A values in each AR term instead of just the AR number
- changed some example grids to include the pt normalizer
### Fixed
- now logs all warnings

## [0.3.7] - 2021-12-27
### Added
- `dynamic_testing` argument to `manual_forecast()` and `auto_forecast()` functions -- this is `True` by default (makes all testing comparable between sklearn/non-sklearn models)
- `dynamic_tuning` argument to `tune()` function -- this is `False` by default to majorly improve speed in some applications
### Changed
- native Forecaster warnings will be logged
### Fixed

## [0.3.6] - 2021-12-14
### Added
- added `tune_test_forecast()` function to notebook module to create a progress bar when using a notebook
### Changed
### Fixed
- fixed an issue with `Forecaster.ingest_Xvars_df()` when `use_future_dates=False` causing an error to be raised

## [0.3.5] - 2021-12-07
### Added
- added `include_traing` parameter to `notebook.results_vis()` function
### Changed
### Fixed
- fixed `print_attr` parameter default in `notebook.results_vis()`

## [0.3.4] - 2021-12-07
### Added
- added `results_vis()` notebook function (requires ipywidgets)
- added `Forecaster.export_Xvars_df()` function
- added `max_integration` argument to the `Forecaster.integrate()` function
### Changed
### Fixed

## [0.3.3] - 2021-11-26
### Added
### Changed
- Now reloads Grids file each time `ingest_grid()` is called so that notebooks do not have to be rerun when a grid cannot be found
### Fixed
- Fixed an issue with some sklearn estimators that occurs when passing a subset of regressors in a list to the forecast function

## [0.3.2] - 2021-11-01
#### Added
#### Changed
#### Fixed
- Found an issue when using `floor` in Prophet

## [0.3.1] - 2021-10-29
#### Added
- Added the eCommerce example
- In `limit_grid_size()`, users can now set random_seed parameter for consistent results
#### Changed
#### Fixed
- Scikit-learn models were not accepting `Xvars='all'` as an arguments
- Fixed an issue causing models run at different levels to error out sometimes when plotted
- Fixed a plotting error that occured sometimes when setting models parameter to `None`

## [0.3.0] - 2021-10-15
#### Added
- Added an option to save to png in plot(), plot_test_set(), and plot_fitted() methods using plt.savefig() from matplotlib and calling with `to_png = True`
#### Changed
- Made errors more descriptive, stripping out AssertionError types 
#### Fixed
- fixed typos in doc strings

## [0.2.9] - 2021-09-27
#### Added
#### Changed
- In plot() method, `models=None` is now accepted and will plot only actual values
- Example grids are modified to prevent overfitting in some models
#### Fixed
- Fixed the add_time_trend() method to not skip a time step in the first observation

## [0.2.8] - 2021-08-27
#### Added
- Added a descriptive error when all_feature_info_to_excel() or all_validation_grids_to_excel() fails
#### Changed
- Using pd.shift() instead of np.roll() to create AR terms to avoid further issues with AR terms
- Prophet, silverkite, and ARIMA have better Xvar validation mechanisms to ensure that autoregressive terms aren't fed to them, which could cause errors and doesn't add anything to the models that isn't already built into them. Now, even if a user tries to feed AR terms only, it will pass no Xvars to these models
#### Fixed
- AR terms were not dropping the correct first observations before being estimated with SKLEARN models, so we fixed that but it didn't seem to make a noticeable difference in any of the examples 

## [0.2.7] - 2021-08-20
#### Added
- added reset() function that deletes all regressors and resets the object to how it was initiated
- added documentation and hints in the source code
#### Changed
- changed readme documentation to be more concise
#### Fixed
- in the documentation, it was stated that the 'scale' value passed to the 'normalize' parameter when calling manual_forecast() or auto_forecast() would use a StandardScaler from sklearn, but a Normalizer was actually being applied. Now, you can pass 'scale' to get the StandardScaler, 'normalize' to get the Normalizer, or 'minmax' to get the MinMaxScaler (unchanged from previous distributions). 'minmax' is still the default for all estimators that accept this argument

## [0.2.6] - 2021-08-12
#### Added
- added train_only argument to following functions to reduce data leakage in eda/preprocessing steps: integrate, adf_test, plot_acf, plot_pacf, plot_periodogram, seasonal_decompose -- default argument is still False for these. Now it is suggested to set a test length before running any one of these methods and only examine the training set correlations to prevent data leakage
#### Changed
#### Fixed

## [0.2.5] - 2021-08-09
#### Added
- added integrate() method that can be used to automatically find the series appropriate level to achieve stationarity, according to augmented dickey fuller test
#### Changed
#### Fixed

## [0.2.4] - 2021-08-03
#### Added
- added tune_test_forecast() function that allows what used to take four lines to be aggregated into one, also allows more easy saving of feature information by setting feature_importance or summary_stats parameters to True
- added all_feature_info_to_excel() function
- added all_validation_grids_to_excel() function
#### Changed
#### Fixed
- removed a duplicate column from the dataframe created when calling the export() method

## [0.2.3] - 2021-07-19
#### Added
#### Changed
- changed removed pandas-datareader from imports in setup.py since it is not a package dependency (change having to do with installation only and should not affect anything when applying the library)
#### Fixed

## [0.2.2] - 2021-07-16
#### Added
- added GridGenerator module so user can more easily create grids in working directory
#### Changed
- changed all functions with diffy parameter (plot_acf, plot_pacf, seasonal_decompose) now accept True, False, 0, 1, or 2 as possible values
#### Fixed
- fixed issues with two-level undifferencing where it was adding values exponentially because the level was being added to the first and second-level differences
- fixed issues with two-level undifferencing where dates were being mixed up
- fixed issues with one-level test-set evaluation where the incorrect initial value was set to undifference values in the test-set only, causing miscalculation of metrics, although the bias was in both directions so when rerunning avocados.ipynb, for example, the results were virtually the same with different models now outperforming others but metrics remaining more or less the same on average; forecasted values did not change

## [0.1.9] - 2021-07-09
#### Added
- added lightgbm and silverkite as estimators
#### Changed
- changed 'which' parameter in set_valiation_metric() to 'metric' for clarity
- changed 'which' parameter in set_estimator() to 'estimator' for clarity
#### Fixed

## [0.1.8] - 2021-07-05
#### Added
#### Changed
#### Fixed
- fixed an error in combo modeling that was causing incorrect applications of weights in weighted averaging -- the weights were generating correctly but not being applied to the best-performing models in the correct order
