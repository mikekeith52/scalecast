Installation
=================================

Install the package:

.. code:: console

   $ pip install scalecast

If you want to apply the theta (from darts), prophet (from Facebook), and/or silverkite (from LinkedIn) models:

.. code:: console

   $ pip install darts
   $ pip install prophet
   $ pip install greykite

These packages can cause issues when installing so they are not included by default in the scalecast dependencies list.

For auto ARIMA, changepoint detection, and shap feature importance:

   $ pip install pmdarima
   $ pip install kats
   $ pip install shap

If you are using Jupyter, there are notebook functions available that require the following installation:

.. code:: console

   $ pip install tqdm
   $ pip install ipython
   $ pip install ipywidgets
   $ jupyter nbextension enable --py widgetsnbextension
   $ jupyter labextension install @jupyter-widgets/jupyterlab-manager

The last command is only necessary for Jupyter lab.