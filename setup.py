
from setuptools import setup, find_packages
import shutil
#from src.scalecast.__init__ import __version__ as version
import os

long_description = open('README.md', 'r', encoding="UTF-8").read()

setup(
  name = 'SCALECAST',
  version = '0.15.9',
  license='MIT',
  description="The practitioner's time series forecasting library",
  long_description=long_description,
  long_description_content_type='text/markdown',
  author = 'Michael Keith',
  author_email = 'mikekeith52@gmail.com',
  packages=find_packages('src'),
  package_dir={'': 'src'},
  project_urls = {
    'GitHub': 'https://github.com/mikekeith52/scalecast',
    'Read the Docs': 'https://scalecast.readthedocs.io/en/latest/',
    'Examples': 'https://scalecast-examples.readthedocs.io/en/latest/',
  },
  url = 'https://github.com/mikekeith52/scalecast',
  keywords = ['FORECAST', 'SCALE', 'DYNAMIC', 'MACHINE LEARNING', 'APPLIED'],
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
    'openpyxl',
    'pandas-datareader',
  ],
)
