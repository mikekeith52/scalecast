Pipeline
==================

Here are some objects that can be placed in a list and executed sequentially, similar to a `pipeline from sklearn <https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html>`_. But because it's scalecast, fitting, testing, and producing forecasts are all done in one step. So in stead of separate `fit()`, `transform()`, and `predict()` methods, we only have `fit_transform()` and `fit_predict()`. The end result are some streamlined, low-code applications with optimal readability.

.. code:: python

	from scalecast.Forecaster import Forecaster
	from scalecast.Pipeline import Pipeline, Transformer, Reverter
	import pandas_datareader as pdr
	import matplotlib.pyplot as plt

	# get and load data into a Forecaster object
	df = pdr.get_data_fred(
	    'HOUSTNSA',
	    start='1959-01-01',
	    end='2022-08-01'
	)
	f = Forecaster(
	    y=df['HOUSTNSA'],
	    current_dates=df.index,
	    future_dates=24,
	)
	# pipeline applications for forecasting should be written into a function(s)
	def forecaster(f):
	    f.set_test_length(0.2)
	    f.set_validation_length(24)
	    f.add_covid19_regressor()
	    f.auto_Xvar_select(cross_validate=True)
	    f.set_estimator('mlr')
	    f.manual_forecast()
	# transformer piece to get stationary and boost results
	transformer = Transformer(
	    transformers = [
	        ('LogTransform',),
	        ('DiffTransform',1),
	        ('DiffTransform',12),
	    ],
	)
	# reverter piece for interpretation
	reverter = Reverter(
	    reverters = [
	        ('DiffRevert',12),
	        ('DiffRevert',1),
	        ('LogRevert',),
	    ],
	    base_transformer = transformer,
	)
	# full pipeline
	pipeline = Pipeline(
	    steps = [
	        ('Transform',transformer),
	        ('Forecast',forecaster),
	        ('Revert',reverter),
	    ],
	)
	f = pipeline.fit_predict(f)

	# plot results
	f.plot(ci=True,order_by='LevelTestSetMAPE')
	plt.show()

	# extract results
	results_dfs = f.export(
	  ['model_summaries','lvl_fcsts']
	)

.. autoclass:: src.scalecast.Pipeline.Transformer
   :members:

   .. automethod:: __init__

.. autoclass:: src.scalecast.Pipeline.Reverter
   :members:

   .. automethod:: __init__

.. autoclass:: src.scalecast.Pipeline.Pipeline
   :members:
   :undoc-members:
   :inherited-members:

   .. automethod:: __init__

.. autoclass:: src.scalecast.Pipeline.MVPipeline
   :members:
   :undoc-members:
   :inherited-members:
   
   .. automethod:: __init__