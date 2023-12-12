import numpy as np

def _prepare_data_mv(mvf,Xvars,lags):
    observed = np.array([mvf.current_xreg[x].values.copy() for x in Xvars]).T
    future = np.array([np.array(mvf.future_xreg[x][:]) for x in Xvars]).T

    ylen = len(mvf.y[mvf.names[0]])
    if len(observed.shape) > 1:
        no_other_xvars = False
        observed_future = np.concatenate([observed,future],axis=0)
    else:
        no_other_xvars = True
        observed_future = np.array([0]*(ylen + len(mvf.future_dates))).reshape(-1,1) # column of 0s
        observed = np.array([0]*ylen).reshape(-1,1)

    err_message = f'Cannot accept this lags argument: {lags}.'

    nolags = lags is None or not lags
    if nolags: # vecm
        observedy = np.array(
            [v.to_list() for k, v in mvf.y.items()]
        ).T
        futurey = np.zeros((len(mvf.future_dates),mvf.n_series))
        if no_other_xvars:
            observed = observedy
            future = futurey
        else:
            observed = np.concatenate([observedy,observed],axis=1)
            future = np.concatenate([futurey,future],axis=1)
        return observed, future, None
    elif isinstance(lags, (float,int)):
        lags = int(lags)
        max_lag = lags
        lag_matrix = np.zeros((observed_future.shape[0],max_lag*mvf.n_series))
        pos = 0
        for i in range(mvf.n_series):
            for j in range(lags):
                Xvars.append('LAG_' + mvf.names[i] + "_" + str(j+1)) # UTUR_1 for first lag to keep track of position
                lag_matrix[:,pos] = (
                    [np.nan] * (j+1)
                    + mvf.y[mvf.names[i]].to_list() 
                    + [np.nan] * (lag_matrix.shape[0] - ylen - (j+1)) # pad with nas
                )[:lag_matrix.shape[0]] 
                pos += 1
    elif isinstance(lags, dict):
        total_lags = 0
        for k, v in lags.items():
            if hasattr(v,'__len__') and not isinstance(v,str):
                total_lags += len(v)
            elif isinstance(v,(float,int)):
                total_lags += v
            else:
                raise ValueError(err_message)
        lag_matrix = np.zeros((observed_future.shape[0],total_lags))
        pos = 0
        max_lag = 1
        for k,v in lags.items():
            if hasattr(v,'__len__') and not isinstance(v,str):
                for i in v:
                    lag_matrix[:,pos] = (
                        [np.nan] * i
                        + mvf.y[k].to_list()
                        + [np.nan]
                        * (lag_matrix.shape[0] - ylen - i)
                    )[:lag_matrix.shape[0]] 
                    Xvars.append('LAG_' + k + "_" + str(i))
                    pos+=1
                max_lag = max(max_lag,max(v))
            elif isinstance(v,(float,int)):
                for i in range(v):
                    lag_matrix[:,pos] = (
                        [np.nan] * (i+1)
                        + mvf.y[k].to_list()
                        + [np.nan]
                        * (lag_matrix.shape[0] - ylen - (i+1))
                    )[:lag_matrix.shape[0]] 
                    Xvars.append('LAG_' + k + "_" + str(i+1))
                    pos+=1
                max_lag = max(max_lag,v)
    elif hasattr(lags,'__len__') and not isinstance(lags,str):
        lag_matrix = np.zeros((observed_future.shape[0],len(lags)*mvf.n_series))
        pos = 0
        max_lag = max(lags)
        for i in range(mvf.n_series):
            for v in lags:
                Xvars.append('LAG_' + mvf.names[i] + "_" + str(v))
                lag_matrix[:,pos] = (
                    [np.nan] * v
                    + mvf.y[mvf.names[i]].to_list()
                    + [np.nan] * (lag_matrix.shape[0] - ylen - v)
                )[:lag_matrix.shape[0]]
                pos+=1
    else:
        raise ValueError(err_message)

    observed_future = np.concatenate([observed_future,lag_matrix],axis=1)
    start_col = 1 if no_other_xvars else 0
    future = observed_future[observed.shape[0]:,start_col:]
    observed = observed_future[max_lag:observed.shape[0],start_col:]
    return observed, future, Xvars

def _scale_mv(scaler, X) -> np.ndarray:
	""" uses scaler parsed from _parse_normalizer() function to transform matrix passed to X.

	Args:
	    scaler (MinMaxScaler, Normalizer, StandardScaler, PowerTransformer, or None): 
	        the fitted scaler or None type
	    X (ndarray or DataFrame):
	        the matrix to transform

	Returns:
	    (ndarray): The scaled x values.
	"""
	if scaler is not None:
	    return scaler.transform(X)
	else:
	    return X

def _train_mv(mvf, X, y, fcster,**kwargs):
    X = _scale_mv(mvf.scaler, X)
    regr = mvf.sklearn_imports[fcster](**kwargs)
    # below added for vecm model -- could be expanded for others as well
    extra_kws_map = {
        'dates':mvf.current_dates.values.copy(),
        'n_series':mvf.n_series,
    }
    if hasattr(regr,'_scalecast_set'):
        for att in regr._scalecast_set:
            setattr(regr,att,extra_kws_map[att])
    
    regr.fit(X, y)
    return regr

def _predict_mv(mvf,trained_models,future,dynamic_testing,nolags,Xvars=None):
    if nolags:
        future = _scale_mv(mvf.scaler, future)
        p = trained_models.predict(future)
        preds = {k:list(p[:,i]) for i, k in enumerate(mvf.y)}
    elif dynamic_testing is False:
        preds = {}
        future = _scale_mv(mvf.scaler, future)
        for series, regr in trained_models.items():
            preds[series] = list(regr.predict(future))
    else:
        preds = {series: [] for series in trained_models.keys()}
        series = {
            k:v.to_list()
            for k,v in mvf.y.items()
        }
        for i in range(future.shape[0]):
            fut = _scale_mv(mvf.scaler,future[i,:].reshape(1,-1))
            for s, regr in trained_models.items():
                snum = list(mvf.y.keys()).index(s)
                pred = regr.predict(fut)[0]
                preds[s].append(pred)
                if (i < len(future) - 1):
                    if ((i+1) % dynamic_testing == 0) and (hasattr(mvf,'actuals')):
                        series[s].append(mvf.actuals[snum][i])
                    else:
                        series[s].append(pred)
            if (i < len(future) - 1):
                for x in Xvars:
                    if x.startswith('LAG_'):
                        idx = Xvars.index(x)
                        s = x.split('_')[1]
                        lagno = int(x.split('_')[-1])
                        future[i+1,idx] = series[s][-lagno]
    return preds