arima = {
	'order':[(2,1,0),(0,1,2),(1,1,1)],
	'seasonal_order':[(0,0,0,0),(0,1,1,12)],
	'trend':['n','c','t','ct']
}

elasticnet = {
	'alpha':[i/10 for i in range(1,101)],
	'l1_ratio':[0,0.25,0.5,0.75,1],
	'normalizer':['scale','minmax',None]
}

gbt = {
	'max_depth':[2,3],
	'n_estimators':[100,500]
}

hwes = {
	'trend':[None,'add','mul'],
	'seasonal':[None,'add','mul'],
	'damped_trend':[True,False]
}

knn = {
	'n_neighbors':range(2,20),
	'weights':['uniform','distance']
}

mlp = {
	'activation':['relu','tanh'],
	'hidden_layer_sizes':[(25,),(25,25,)],
	'solver':['lbfgs','adam'],
	'normalizer':['scale','minmax',None],
	'random_state':[20]
}

mlr = {
	'normalizer':['scale','minmax',None]
}

prophet = {
	'n_changepoints':range(5)
}

rf = {
	'max_depth':[5,10,None],
	'n_estimators':[100,500,1000]
}

silverkite = {
	'changepoints':range(5)
}

svr={
	'kernel':['linear'],
    'C':[.5,1,2,3],
    'epsilon':[0.01,0.1,0.5]
}

xgboost = {
	'max_depth':[2,3,4,5,6]
}