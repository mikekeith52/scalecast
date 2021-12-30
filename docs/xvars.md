## Xvars
- all estimators except hwes and combo accept an `Xvars` argument
- accepted arguments are an array-like of named regressors, a `str` of a single regressor name, `'all'`, or `None`
  - for estimators that require Xvars (sklearn models), `None` and `'all'` will both use all Xvars
- all regressors must be numeric type

### Adding Xvar methods
- `add_seasonal_regressors()` - adds any kind of seasonality, including hourly, daily, weekly, monthly, quarterly, and more; dummy and wave transformations are simple to apply
- `add_ar_terms()` - adds 1 to n lagged y regressors to forecast with -- all AR terms dynamically propogated when forecasting and when testing by default (it can be disabled for testing for faster performance)
- `add_AR_terms()` - adds seasonal AR terms, like AR12 and AR24 when modeling monthly data
- `add_time_trend()` - adds a time trend
- `add_combo_regressors()` - multiplies any number of already existing regressors together
- `add_poly_terms()` - adds polynomial terms (quadratic, cubic, etc.)
- `add_covid19_regressor()` - this may be deprecated as the pandemic has lasted longer than anyone thought, but this method adds a dummy variable that is 1 during the time period that covid19 effects are present for the series, 0 otherwise; by default this is from when Disney World initially closed to when the CDC lifted its mask recommendations for vaccinated individuals; it can still be argued that this is when COVID had the largest effect on the economy so it can still be useful
- `ingest_Xvars_df()` - ingests a pandas dataframe; specify the date variable and use its future dates if you wish
- `add_other_regressor()` - adds other regressors that can be 1 during a date range, 0 otherwise
- `add_exp_terms()` - for non polynomial exponential transformations
- `add_logged_terms()` - for log of any base transformations
- `add_pt_terms()` - for individual variable power transformations (box cox and yeo johnson available)
- `add_diffed_terms()` - to difference non-y terms
- `add_lagged_terms()` - to lag non-y terms
