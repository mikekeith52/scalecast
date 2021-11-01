# Changelog
All notable changes to this project will be documented in this file. We keep track of changes in this file since v0.1.8. The source code for all releases is available on GitHub.

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
