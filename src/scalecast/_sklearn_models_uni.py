import numpy as np

def _fit_sklearn(f,regr,current_X,obs_to_drop=0):
    f.X = f._scale(f.scaler, current_X)
    y = f.y.to_list()[obs_to_drop:]
    regr.fit(f.X, y)

def _get_obs_to_drop(Xvars):
    # list of integers, each one representing the n/a values in each AR term
    ars = [int(x[2:]) for x in Xvars if x.startswith("AR")]
    # if using ARs, instead of foregetting those obs, ignore them with sklearn forecasts (leaves them available for other types of forecasts)
    return max(ars) if len(ars) > 0 else 0

def _generate_X_sklearn(f,normalizer,Xvars,obs_to_drop=0):
    current_X = np.array([f.current_xreg[x].values[obs_to_drop:].copy() for x in Xvars]).T
    future_X = np.array([np.array(f.future_xreg[x][:]) for x in Xvars]).T
    f.scaler = f._fit_normalizer(current_X, normalizer)
    f.Xvars = Xvars
    return current_X, future_X

def _predict_sklearn(
    f,
    regr,
    future_X,
    steps = None,
    dynamic_testing = True,
):
    Xvars = f.Xvars
    has_ars = len([x for x in Xvars if x.startswith('AR')]) > 0

    if steps == 0:
        return []

    if (dynamic_testing == 1 and not np.any(np.isnan(future_X))) or not has_ars:
        p = f._scale(f.scaler,future_X)
        preds = regr.predict(p)
    else:
        # handle peeking
        actuals = f.actuals if hasattr(f,'actuals') else []
        peeks = [a if (i+1)%dynamic_testing == 0 else np.nan for i, a in enumerate(actuals)]

        series = f.y.to_list() # this is used to add peeking to the models
        preds = [] # this is used to produce the real predictions
        for i in range(steps):
            p = f._scale(f.scaler,future_X[i,:].reshape(1,len(Xvars)))
            pred = regr.predict(p)[0]
            preds.append(pred)
            if hasattr(f,'actuals') and (i+1) % dynamic_testing == 0:
                series.append(peeks[i])
            else:
                series.append(pred)

            if i == (steps-1):
                break

            for pos, x in enumerate(Xvars):
                if x.startswith('AR'):
                    ar = int(x[2:])
                    future_X[i+1,pos] = series[-ar]

    return list(preds)