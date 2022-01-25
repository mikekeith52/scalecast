
from setuptools import setup, find_packages
import shutil

shutil.copy('./examples/eCommerce.ipynb','./docs/source/Forecaster/examples/eCommerce.ipynb')
shutil.copy('./examples/LSTM.ipynb','./docs/source/Forecaster/examples/LSTM.ipynb')

long_description = open('README.md', 'r', encoding="UTF-8").read()

setup(
  name = 'SCALECAST',
  version = '0.5.6',
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
