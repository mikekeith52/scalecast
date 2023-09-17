import numpy as np
import matplotlib.pyplot as plt

def _process_y_tf(y,lags,total_period,scale_y):
    ymin = y.min()
    ymax = y.max()

    if scale_y:
        ylist = [(yi - ymin) / (ymax - ymin) for yi in y]
    else:
        ylist = [yi for yi in y]

    idx_end = len(y)
    idx_start = idx_end - total_period
    y_new = []

    while idx_start > 0:
        y_line = ylist[idx_start + lags : idx_start + total_period]
        y_new.append(y_line)
        idx_start -= 1

    return (
        np.array(y_new[::-1]),
        ymin,                      # for scaling lags
        ymax,
    )

def _get_compiled_model_tf(
    y,
    optimizer,
    layers_struct,
    learning_rate,
    loss,
    n_timesteps,
):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, LSTM, SimpleRNN
    import tensorflow.keras.optimizers
    if isinstance(optimizer, str):
        local_optimizer = eval(f"tensorflow.keras.optimizers.{optimizer}")(
            learning_rate=learning_rate
        )
    else:
        local_optimizer = optimizer
    for i, kv in enumerate(layers_struct):
        layer = locals()[kv[0]]

        if i == 0:
            if kv[0] in ("LSTM", "SimpleRNN"):
                kv[1]["return_sequences"] = len(layers_struct) > 1
            model = Sequential(
                [layer(**kv[1], input_shape=(n_timesteps, 1),)]
            )
        else:
            if kv[0] in ("LSTM", "SimpleRNN"):
                kv[1]["return_sequences"] = not i == (len(layers_struct) - 1)
                if kv[1]["return_sequences"]:
                    kv[1]["return_sequences"] = (
                        layers_struct[i + 1][0] != "Dense"
                    )

            model.add(layer(**kv[1]))
    model.add(Dense(y.shape[1]))  # output layer

    # compile model
    model.compile(optimizer=local_optimizer, loss=loss)
    return model

def _process_X_tf(
    f,
    Xvars,
    lags,
    forecast_length,
    ymin,
    ymax,
    scale_X,
    scale_y,
):
    def minmax_scale(x):
        return (x - ymin) / (ymax - ymin)

    X_lags = np.array([
        v.to_list() + f.future_xreg[k][:1] 
        for k,v in f.current_xreg.items() 
        if k in Xvars and k.startswith('AR')
    ]).T
    X_other = np.array([
        v.to_list() + f.future_xreg[k][:1] 
        for k,v in f.current_xreg.items() 
        if k in Xvars and not k.startswith('AR')
    ]).T

    X_lags_new = X_lags[lags:]
    X_other_new = X_other[lags:]
    
    # scale lags
    if len(X_lags_new) > 0 and scale_y:
        X_lags_new = np.vectorize(minmax_scale)(X_lags_new)
    # scale other regressors
    if len(X_other_new) > 0 and scale_X:
        X_other_train = X_other_new[:-1]
        scaler = f._fit_normalizer(X_other_train,'minmax')
        X_other_new = f._scale(scaler,X_other_new)
        
    # combine
    if len(X_lags_new) > 0 and len(X_other_new) > 0:
        X = np.concatenate([X_lags_new,X_other_new],axis=1)
    elif len(X_lags_new) > 0:
        X = X_lags_new
    else:
        X = X_other_new

    fut = X[-1:]
    X = X[:-1]

    return X, fut

def _finish_process_X_tf(X,fut,forecast_length):
    X = X[1:-(forecast_length-1)] if forecast_length > 1 else X[1:]
    n_timesteps = X.shape[1]
    X = X.reshape(X.shape[0], X.shape[1], 1)
    fut = fut.reshape(fut.shape[0], fut.shape[1], 1)
    return X, n_timesteps, fut

def _get_preds_tf(tf_model,fut,scale_y,ymax,ymin):
    preds = tf_model.predict(fut)
    preds = [p for p in preds[0]]
    if scale_y:
        preds = [p * (ymax - ymin) + ymin for p in preds]
    return preds

def _get_fvs_tf(tf_model,X,scale_y,ymax,ymin):
    fvs = tf_model.predict(X)
    fvs =  [p[0] for p in fvs[:-1]] + [p for p in fvs[-1]]
    if scale_y:
        fvs = [p * (ymax - ymin) + ymin for p in fvs]
    return fvs

def _plot_loss_rnn(history, title):
    plt.plot(history.history["loss"], label="train_loss")
    if "val_loss" in history.history.keys():
        plt.plot(history.history["val_loss"], label="val_loss")
    plt.title(title)
    plt.xlabel("epoch")
    plt.legend(loc="upper right")
    plt.show()

def _convert_lstm_args_to_rnn(**kwargs):
    new_kwargs = {
        k: v
        for k, v in kwargs.items()
        if k not in ("lstm_layer_sizes", "dropout", "activation")
    }
    new_kwargs["layers_struct"] = [
        (
            "LSTM",
            {
                "units": v,
                "activation": kwargs["activation"],
                "dropout": kwargs["dropout"][i],
            },
        )
        for i, v in enumerate(kwargs["lstm_layer_sizes"])
    ]
    return new_kwargs



