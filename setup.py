
from setuptools import setup, find_packages
import shutil
import re
from src.scalecast.__init__ import __version__ as version


shutil.copy('./examples/eCommerce.ipynb','./docs/Forecaster/examples/eCommerce.ipynb')
shutil.copy('./examples/LSTM.ipynb','./docs/Forecaster/examples/LSTM.ipynb')

long_description = open('README.md', 'r', encoding="UTF-8").read()

setup(
  name = 'SCALECAST',
  version = version,
  license='MIT',
  long_description=long_description,
  long_description_content_type='text/markdown',
  author = 'Michael Keith',
  author_email = 'mikekeith52@gmail.com',
  packages=find_packages('src'),
  package_dir={'': 'src'},
  numpydoc_show_class_members = False,
  url = 'https://github.com/mikekeith52/scalecast',
  keywords = ['FORECAST', 'SCALE', 'DYNAMIC'],
  install_requires = [
    'scikit-learn',
    'tensorflow',
    'statsmodels',
    'scipy',
    'eli5',
    'numpy',
    'pandas',
    'scipy',
    'matplotlib',
    'seaborn',
    'xgboost',
    'lightgbm',
    'openpyxl'
  ],
)
