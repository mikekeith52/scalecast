.. scalecast documentation master file, created by
   sphinx-quickstart on Fri Jan 21 15:40:54 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Official Docs
==========================

Welcome to the scalecast official docs!

.. image:: https://img.shields.io/badge/python-3.7+-blue.svg
 :target: https://pepy.tech/project/scalecast
.. image:: https://static.pepy.tech/personalized-badge/scalecast?period=total&units=international_system&left_color=black&right_color=brightgreen&left_text=Downloads
 :target: https://pepy.tech/project/scalecast
.. image:: https://static.pepy.tech/personalized-badge/scalecast?period=month&units=international_system&left_color=black&right_color=brightgreen&left_text=Downloads/Month
 :target: https://pepy.tech/project/scalecast
.. image:: https://img.shields.io/github/workflow/status/unit8co/darts/darts%20release%20workflow/master
 :target: https://img.shields.io/github/workflow/status/unit8co/darts/darts%20release%20workflow/master

Overview
--------------

:doc:`about`
   Who is this meant for and what sets it apart?

:doc:`installation`
   Base package + dependencies.

:doc:`initialization`
   How to call the object and make forecasts.

:doc:`change_log`
   See what's changed.

Index
------
* :ref:`genindex`

.. Hidden TOCs

.. toctree::
   :maxdepth: 4
   :caption: Overview:
   :hidden:  

   about
   installation
   initialization

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
