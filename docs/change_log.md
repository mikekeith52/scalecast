# Changelog
All notable changes to this project will be documented in this file. We keep track of changes in this file since v0.1.8. The source code for all releases is available on GitHub.

## [0.4.0] - 2021-12-30
## Added
## Changed
- deleted the 'scale' normalizer from the mlp grid
## Fixed
- Fixed an issue with the PowerTransformer normalizer that failed because of a DivideByZero error, now defaults to a StandardScaler when this issue is encountered and logs a warning

## [0.3.9] - 2021-12-30
## Added
- Added `init_dates` and `levely` attributes
- Added `'Observations'` info to history and `export()`
- Added `'lvl_test_set_predictions'` to export dataframes
## Changed
- Got rid of `first_obs` and `first_dates` attributes and wrote more efficient code to do what they were there for
- More information available when `__str__()` is called
## Fixed
- Fixed what became an issue with the last update in which when calling `add_diffed_terms()` or `add_lagged_terms()`, the level series wasn't accurate due to how undifferencing was being executed. After examining this issue, it became evident that the previous way to undifference forecasts was less efficient than it should have been, this update fixed the issues from the last update and made the code more efficient
- Fixed an issue where AR terms were manipulating the underlying xreg structures so that each forecast were using its own test-set propogated AR values instead of the correct AR values
- Fixed the `export_Xvars_df()` method which wasn't working correctly if at least one forecast hadn't been called first

## [0.3.8] - 2021-12-29
## Added
- added the following functions that can each add additional Xvars to forecast with:
	- `add_exp_terms()` - for non polynomial exponential transformations
	- `add_logged_terms()` - for log of any base transformations
	- `add_pt_terms()` - for individual variable power transformations (box cox and yeo johnson available)
	- `add_diffed_terms()` - to difference non-y terms
	- `add_lagged_terms()` - to lag non-y terms
- added the 'pt' normalizer for yeo-johnson normalization (in addition to 'minmax', 'normalize', and 'scale')
- added the `drop_Xvars()` function that is identical to the `drop_regressors()` function
## Changed
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
## Fixed
- now logs all warnings

## [0.3.7] - 2021-12-27
## Added
- `dynamic_testing` argument to `manual_forecast()` and `auto_forecast()` functions -- this is `True` by default (makes all testing comparable between sklearn/non-sklearn models)
- `dynamic_tuning` argument to `tune()` function -- this is `False` by default to majorly improve speed in some applications
## Changed
- native Forecaster warnings will be logged
## Fixed

## [0.3.6] - 2021-12-14
## Added
- added `tune_test_forecast()` function to notebook module to create a progress bar when using a notebook
## Changed
## Fixed
- fixed an issue with `Forecaster.ingest_Xvars_df()` when `use_future_dates=False` causing an error to be raised

## [0.3.5] - 2021-12-07
## Added
- added `include_traing` parameter to `notebook.results_vis()` function
## Changed
## Fixed
- fixed `print_attr` parameter default in `notebook.results_vis()`

## [0.3.4] - 2021-12-07
## Added
- added `results_vis()` notebook function (requires ipywidgets)
- added `Forecaster.export_Xvars_df()` function
- added `max_integration` argument to the `Forecaster.integrate()` function
## Changed
## Fixed

## [0.3.3] - 2021-11-26
## Added
## Changed
- Now reloads Grids file each time `ingest_grid()` is called so that notebooks do not have to be rerun when a grid cannot be found
## Fixed
- Fixed an issue with some sklearn estimators that occurs when passing a subset of regressors in a list to the forecast function

## [0.3.2] - 2021-11-01
### Added
### Changed
### Fixed
- Found an issue when using `floor` in Prophet

## [0.3.1] - 2021-10-29
### Added
- Added the eCommerce example
- In `limit_grid_size()`, users can now set random_seed parameter for consistent results
### Changed
### Fixed
- Scikit-learn models were not accepting `Xvars='all'` as an arguments
- Fixed an issue causing models run at different levels to error out sometimes when plotted
- Fixed a plotting error that occured sometimes when setting models parameter to `None`

## [0.3.0] - 2021-10-15
### Added
- Added an option to save to png in plot(), plot_test_set(), and plot_fitted() methods using plt.savefig() from matplotlib and calling with `to_png = True`
### Changed
- Made errors more descriptive, stripping out AssertionError types 
### Fixed
- fixed typos in doc strings

## [0.2.9] - 2021-09-27
### Added
### Changed
- In plot() method, `models=None` is now accepted and will plot only actual values
- Example grids are modified to prevent overfitting in some models
### Fixed
- Fixed the add_time_trend() method to not skip a time step in the first observation

## [0.2.8] - 2021-08-27
### Added
- Added a descriptive error when all_feature_info_to_excel() or all_validation_grids_to_excel() fails
### Changed
- Using pd.shift() instead of np.roll() to create AR terms to avoid further issues with AR terms
- Prophet, silverkite, and ARIMA have better Xvar validation mechanisms to ensure that autoregressive terms aren't fed to them, which could cause errors and doesn't add anything to the models that isn't already built into them. Now, even if a user tries to feed AR terms only, it will pass no Xvars to these models
### Fixed
- AR terms were not dropping the correct first observations before being estimated with SKLEARN models, so we fixed that but it didn't seem to make a noticeable difference in any of the examples 

## [0.2.7] - 2021-08-20
### Added
- added reset() function that deletes all regressors and resets the object to how it was initiated
- added documentation and hints in the source code
### Changed
- changed readme documentation to be more concise
### Fixed
- in the documentation, it was stated that the 'scale' value passed to the 'normalize' parameter when calling manual_forecast() or auto_forecast() would use a StandardScaler from sklearn, but a Normalizer was actually being applied. Now, you can pass 'scale' to get the StandardScaler, 'normalize' to get the Normalizer, or 'minmax' to get the MinMaxScaler (unchanged from previous distributions). 'minmax' is still the default for all estimators that accept this argument

## [0.2.6] - 2021-08-12
### Added
- added train_only argument to following functions to reduce data leakage in eda/preprocessing steps: integrate, adf_test, plot_acf, plot_pacf, plot_periodogram, seasonal_decompose -- default argument is still False for these. Now it is suggested to set a test length before running any one of these methods and only examine the training set correlations to prevent data leakage
### Changed
### Fixed

## [0.2.5] - 2021-08-09
### Added
- added integrate() method that can be used to automatically find the series appropriate level to achieve stationarity, according to augmented dickey fuller test
### Changed
### Fixed

## [0.2.4] - 2021-08-03
### Added
- added tune_test_forecast() function that allows what used to take four lines to be aggregated into one, also allows more easy saving of feature information by setting feature_importance or summary_stats parameters to True
- added all_feature_info_to_excel() function
- added all_validation_grids_to_excel() function
### Changed
### Fixed
- removed a duplicate column from the dataframe created when calling the export() method

## [0.2.3] - 2021-07-19
### Added
### Changed
- changed removed pandas-datareader from imports in setup.py since it is not a package dependency (change having to do with installation only and should not affect anything when applying the library)
### Fixed

## [0.2.2] - 2021-07-16
### Added
- added GridGenerator module so user can more easily create grids in working directory
### Changed
- changed all functions with diffy parameter (plot_acf, plot_pacf, seasonal_decompose) now accept True, False, 0, 1, or 2 as possible values
### Fixed
- fixed issues with two-level undifferencing where it was adding values exponentially because the level was being added to the first and second-level differences
- fixed issues with two-level undifferencing where dates were being mixed up
- fixed issues with one-level test-set evaluation where the incorrect initial value was set to undifference values in the test-set only, causing miscalculation of metrics, although the bias was in both directions so when rerunning avocados.ipynb, for example, the results were virtually the same with different models now outperforming others but metrics remaining more or less the same on average; forecasted values did not change

## [0.1.9] - 2021-07-09
### Added
- added lightgbm and silverkite as estimators
### Changed
- changed 'which' parameter in set_valiation_metric() to 'metric' for clarity'
- changed 'which' parameter in set_estimator() to 'estimator' for clarity
### Fixed

## [0.1.8] - 2021-07-05
### Added
### Changed
### Fixed
- fixed an error in combo modeling that was causing incorrect applications of weights in weighted averaging -- the weights were generating correctly but not being applied to the best-performing models in the correct order
