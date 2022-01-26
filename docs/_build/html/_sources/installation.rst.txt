Installation
=================================

Install the package:

.. code:: console

   $ pip install scalecast

If you want to apply prophet (from Facebook) and silverkite (from LinkedIn) models:

.. code:: console

   $ pip install prophet
   $ pip install greykite

The two packages above can cause issues when installing so they are not included by default in the scalecast dependencies list.

If you are using Jupyter, there are notebook functions available that require the following installation:

.. code:: console

   $ pip install tqdm
   $ pip install ipython
   $ pip install ipywidgets
   $ jupyter nbextension enable --py widgetsnbextension
   $ jupyter labextension install @jupyter-widgets/jupyterlab-manager

The last command is only necessary for Jupyter lab.