
from setuptools import setup, find_packages
import shutil
import os
import sys

sys.path.insert(0, os.path.abspath("src"))
from scalecast.__init__ import __version__

long_description = open('README.md', 'r', encoding="UTF-8").read()

setup(
  name = 'SCALECAST',
  version = __version__,
  license='MIT',
  description="The practitioner's time series forecasting library",
  long_description=long_description,
  long_description_content_type='text/markdown',
  author = 'Michael Keith',
  author_email = 'mikekeith52@gmail.com',
  packages=find_packages('src'),
  package_dir={'': 'src'},
  package_data = {
    'scalecast':['grids/*'],
  },
  project_urls = {
    'GitHub': 'https://github.com/mikekeith52/scalecast',
    'Read the Docs': 'https://scalecast.readthedocs.io/en/latest/',
    'Examples': 'https://scalecast-examples.readthedocs.io/en/latest/',
  },
  url = 'https://github.com/mikekeith52/scalecast',
  keywords = ['FORECAST', 'SCALE', 'DYNAMIC', 'MACHINE LEARNING', 'APPLIED'],
  install_requires = [
    'scikit-learn',
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
    'catboost',
    'openpyxl',
    'pandas-datareader',
  ],
)
