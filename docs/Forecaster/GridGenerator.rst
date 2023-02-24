GridGenerator
============================================

To automatically generate grids to tune models with.

get_grids()
--------------------------------------------------
.. autofunction:: src.scalecast.GridGenerator.get_grids

>>> from scalecast import GridGenerator
>>> GridGenerator.get_grids() # writes a Grids.py file with all available models that can be tuned and example dictionaries

get_empty_grids()
--------------------------------------------------
.. autofunction:: src.scalecast.GridGenerator.get_empty_grids

>>> from scalecast import GridGenerator
>>> GridGenerator.get_empty_grids() # writes a Grids.py file with all available models that can be tuned and empty dictionaries

get_example_grids()
--------------------------------------------------
.. autofunction:: src.scalecast.GridGenerator.get_example_grids

>>> from scalecast import GridGenerator
>>> GridGenerator.get_example_grids() # writes a Grids.py file with all available models that can be tuned and example dictionaries

get_mv_grids()
-------------------------------------
.. autofunction:: src.scalecast.GridGenerator.get_mv_grids

>>> from scalecast import GridGenerator
>>> GridGenerator.get_mv_grids() # writes a Grids.py file with models available for multivariate forecasting and some extra args for such models