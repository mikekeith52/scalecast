catboost = {
    'iterations': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'depth': [4, 6, 8],
    'l2_leaf_reg': [1, 3, 5, 7, 9],
    'lags':[1,3,6],
}

elasticnet = {
    'alpha':[i/10 for i in range(1,21)],
    'l1_ratio':[0,0.25,0.5,0.75,1],
    'normalizer':['scale','minmax',None],
    'lags':[1,3,6],
}

gbt = {
    'max_depth':[2,3],
    'max_features':['sqrt',None],
    'lags':[1,3,6],
}

knn = {
    'n_neighbors':range(2,101),
    'lags':[1,3,6],
}

lasso = {
    'alpha':[i/100 for i in range(1,101)],
    'lags':[1,3,6],
}

lightgbm = {
    'n_estimators':[150,200,250],
    'boosting_type':['gbdt','dart','goss'],
    'max_depth':[1,2,3],
    'learning_rate':[0.001,0.01,0.1],
    'lags':[1,3,6],
}

mlp = {
    'activation':['relu','tanh'],
    'hidden_layer_sizes':[(25,),(25,25,)],
    'solver':['lbfgs','adam'],
    'normalizer':['minmax','scale'],
    'lags':[1,3,6],
}

mlr = {
    'normalizer':['scale','minmax',None],
    'lags':[1,3,6],
}

rf = {
    'max_depth':[2,5],
    'n_estimators':[100,500],
    'max_features':['auto','sqrt'],
    'max_samples':[.75,.9,1],
    'lags':[1,3,6],
}

ridge = {
    'alpha':[i/100 for i in range(1,101)],
    'lags':[1,3,6],
}

sgd={
    'penalty':['l2','l1','elasticnet'],
    'l1_ratio':[0,0.15,0.5,0.85,1],
    'learning_rate':['invscaling','constant','optimal','adaptive'],
    'lags':[1,3,6],
}

svr={
    'kernel':['linear'],
    'C':[.5,1,2,3],
    'epsilon':[0.01,0.1,0.5],
    'lags':[1,3,6],
}

vecm = {
    'lags':[0],
    'normalizer':[None],
    'k_ar_diff':range(1,13),
    'deterministic':["n","co","lo","li","cili","colo"],
    'seasons':[0,12],
}

xgboost = {
    'n_estimators':[150,200,250],
    'scale_pos_weight':[5,10],
    'learning_rate':[0.1,0.2],
    'gamma':[0,3,5],
    'subsample':[0.8,0.9],
    'lags':[1,3,6],
}