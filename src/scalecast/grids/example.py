arima = {
    'order':[(2,1,0),(0,1,2),(1,1,1)],
    'seasonal_order':[(0,1,1,12),(2,1,0,12)],
}

catboost = {
    'iterations': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'depth': [4, 6, 8],
    'l2_leaf_reg': [1, 3, 5, 7, 9],
    'verbose': [0],
}

elasticnet = {
    'alpha':[i/10 for i in range(1,21)],
    'l1_ratio':[0,0.25,0.5,0.75,1],
    'normalizer':['scale','minmax'],
}

gbt = {
    'max_depth':[2,3,4,5],
    'max_features':['sqrt',None],
}

hwes = {
    'trend':['add','mul',None],
    'seasonal':['add','mul',None],
    'use_boxcox':[True,False],
}

knn = {
    'n_neighbors':range(2,101),
}

lasso = {
    'alpha':[i/100 for i in range(1,101)],
}

lightgbm = {
    'n_estimators':[150,200,250],
    'boosting_type':['gbdt','dart','goss'],
    'max_depth':[1,2,3],
    'learning_rate':[0.001,0.01,0.1],
}

lstm = {
    'lstm_layer_sizes':[(50,50,50)],
    'activation':['relu','tanh'],
    'dropout':[(0,0,0),(.2,.2,.2),],
    'lags':[10,25,50],
    'verbose':[0],
    'epochs':[25],
}

mlp = {
    'activation':['relu','tanh'],
    'hidden_layer_sizes':[(25,),(25,25,)],
    'solver':['lbfgs','adam'],
    'normalizer':['minmax','scale'],
}

mlr = {
    'normalizer':['scale','minmax',None],
}

naive = {
    'seasonal':[True,False],
}

prophet = {
    'n_changepoints':range(5),
}

rf = {
    'max_depth':[2,5],
    'n_estimators':[100,500],
    'max_features':['auto','sqrt'],
    'max_samples':[.75,.9,1],
}

ridge = {
    'alpha':[i/100 for i in range(1,101)],
}

rnn = {
    'layers_struct':[
        [
            ('LSTM',{'units':50,'activation':'relu'}),
            ('LSTM',{'units':50,'activation':'relu'}),
            ('LSTM',{'units':50,'activation':'relu'}),
        ],
        [
            ('LSTM',{'units':50,'activation':'tanh'}),
            ('LSTM',{'units':50,'activation':'tanh'}),
            ('LSTM',{'units':50,'activation':'tanh'}),
        ],
        [
            ('LSTM',{'units':50,'activation':'tanh','dropout':.2}),
            ('LSTM',{'units':50,'activation':'tanh','dropout':.2}),
            ('LSTM',{'units':50,'activation':'tanh','dropout':.2}),
        ],
    ],
    'epochs':[25],
    'verbose':[0],
}

silverkite = {
    'changepoints':range(5),
}

sgd={
    'penalty':['l2','l1','elasticnet'],
    'l1_ratio':[0,0.15,0.5,0.85,1],
    'learning_rate':['invscaling','constant','optimal','adaptive'],
}

svr={
    'kernel':['linear'],
    'C':[.5,1,2,3],
    'epsilon':[0.01,0.1,0.5],
}

tbats = {
    'seasonal_periods':[[12],None],
}

theta = {
    'theta':[0.5,1,1.5,2],
}

vecm = {
    'lags':[0],
    'normalizer':[None],
    'k_ar_diff':[1,2,3,4,5,6,7],
    'deterministic':["n","co","lo","li","cili","colo"],
    'seasons':[0,12],
}

xgboost = {
     'n_estimators':[150,200,250],
     'scale_pos_weight':[5,10],
     'learning_rate':[0.1,0.2],
     'gamma':[0,3,5],
     'subsample':[0.8,0.9],
}