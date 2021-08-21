# Changelog
All notable changes to this project will be documented in this file. We keep track of changes in this file since v0.1.8. The source code for all releases is available on GitHub.

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
- fixed issues with one-level test-set evaluation where the incorrect initial value was set to undifference the values, causing miscalculation of metrics, although the bias was in both directions so when rerunning avocados.ipynb, for example, the results were virtually the same with different models now outperforming others but metrics remaining more or less the same on average

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

