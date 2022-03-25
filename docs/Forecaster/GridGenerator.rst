GridGenerator Functions
============================================

If you want to automatically generate grids to tune models with, here's how.

get_empty_grids
--------------------------------------------------
.. automodule:: src.scalecast.GridGenerator.get_empty_grids
    :members:
    :undoc-members:
    :show-inheritance:

>>> from scalecast import GridGenerator
>>> GridGenerator.get_empty_grids() # writes a Grids.py file with all available models that can be tuned and empty dictionaries

get_example_grids
--------------------------------------------------
.. automodule:: src.scalecast.GridGenerator.get_example_grids
    :members:
    :undoc-members:
    :show-inheritance:

>>> from scalecast import GridGenerator
>>> GridGenerator.get_example_grids() # writes a Grids.py file with all available models that can be tuned and example dictionaries

get_mv_grids
-------------------------------------
.. automodule:: src.scalecast.GridGenerator.get_mv_grids
    :members:
    :undoc-members:
    :show-inheritance:

>>> from scalecast import GridGenerator
>>> GridGenerator.get_mv_grids() # writes a Grids.py file with models available for multivariate forecasting and some extra args for such models