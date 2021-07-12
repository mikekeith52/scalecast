example_grids = """
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

lightgbm = {
	'max_depth':[i for i in range(5)] + [-1]
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
"""

empty_grids = """
arima = {}
elasticnet = {}
gbt = {}
hwes = {}
knn = {}
lightgbm = {}
mlp = {}
mlr = {}
prophet = {}
rf = {}
silverkite = {}
svr={}
xgboost = {}
"""


import os

def get_example_grids(overwrite=False):
	""" overwrite: bool
			whether to overwrite a Grids.py file if one is already in the working directory
	"""
	if 'Grids.py' in os.listdir('./'):
		if not overwrite:
			return
	
	with open('Grids.py','w') as f:
		f.write(example_grids)

def get_empty_grids(overwrite=False):
	""" overwrite: bool
			whether to overwrite a Grids.py file if one is already in the working directory
	"""
	if 'Grids.py' in os.listdir('./'):
		if not overwrite:
			return
	
	with open('Grids.py','w') as f:
		f.write(empty_grids)