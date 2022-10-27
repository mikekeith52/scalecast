.. scalecast documentation master file, created by
   sphinx-quickstart on Fri Jan 21 15:40:54 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Scalecast Official Docs
==========================

**The pratictioner's forecasting library.** Including automated model selection, model optimization, pipelines, visualization, and reporting.

.. code:: console

   $ pip install --upgrade scalecast

.. image:: https://media2.giphy.com/media/vV2Mbr9v6pH1D8hiLb/giphy.gif?cid=790b7611eb56b43191020435cbedf6453a74ddc2cebd017d&rid=giphy.gif&ct=g
 :target: https://scalecast-examples.readthedocs.io/en/latest/misc/introduction/Introduction2.html#Scaled-Automated-Forecasting

Forecasting with Python has never been easier.

.. code:: python
   
   import pandas as pd
   from scalecast.Forecaster import Forecaster
   from scalecast import GridGenerator

   GridGenerator.get_example_grids()

   data = pd.read_csv('data.csv')
   f = Forecaster(
      y = data['values'],
      current_dates = data['date'],
      future_dates = 24, # forecast horizon
   )
   f.set_estimator('xgboost')

   f.auto_Xvar_select()
   f.cross_validate(k=3)
   f.auto_forecast()

   results = f.export(['lvl_fcsts','model_summaries'])

:doc:`about` Readme
   Overview, starter code, and installation.

:doc:`Forecaster/ForecasterGlobals`
   Key terms to know that will make your life easier.

:doc:`Forecaster/_forecast`
   What models are available?

:doc:`change_log`
   Recent additions and bug fixes.

.. image:: https://img.shields.io/badge/python-3.7+-blue.svg
 :target: https://pepy.tech/project/scalecast
.. image:: https://static.pepy.tech/personalized-badge/scalecast?period=total&units=international_system&left_color=black&right_color=brightgreen&left_text=Downloads
 :target: https://pepy.tech/project/scalecast
.. image:: https://static.pepy.tech/personalized-badge/scalecast?period=month&units=international_system&left_color=black&right_color=brightgreen&left_text=Downloads/Month
 :target: https://pepy.tech/project/scalecast
.. image:: https://img.shields.io/github/workflow/status/unit8co/darts/darts%20release%20workflow/master
 :target: https://img.shields.io/github/workflow/status/unit8co/darts/darts%20release%20workflow/master
.. image:: https://img.shields.io/badge/code%20style-black-black
 :target: https://github.com/psf/black

Index
------
* :ref:`genindex`

.. Hidden TOCs

.. toctree::
   :maxdepth: 4
   :caption: ReadMe:
   :hidden:  

   about

.. toctree::
   :maxdepth: 4
   :caption: High Level:
   :hidden:

   Forecaster/ForecasterGlobals
   Forecaster/_forecast

.. toctree::
   :maxdepth: 4
   :caption: Classes:
   :hidden:

   Forecaster/Forecaster
   Forecaster/MVForecaster
   Forecaster/Pipeline
   Forecaster/SeriesTransformer
   Forecaster/AnomalyDetector
   Forecaster/ChangepointDetector

.. toctree::
   :maxdepth: 4
   :caption: Modules:
   :hidden:

   Forecaster/GridGenerator
   Forecaster/Notebook
   Forecaster/Multiseries
   Forecaster/Auxmodels
   Forecaster/Util

.. toctree::
   :maxdepth: 4
   :caption: Examples:
   :hidden: 

   https://scalecast-examples.readthedocs.io/en/latest/

.. toctree::
   :maxdepth: 1
   :caption: ChangeLog:  
   :hidden:

   change_log
