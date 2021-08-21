
from setuptools import setup, find_packages

setup(
  name = 'SCALECAST',
  version = '0.2.7',
  license='MIT',
  long_description='See the documentation on [GitHub](https://github.com/mikekeith52/scalecast).',
  long_description_content_type='text/markdown',
  author = 'Michael Keith',
  author_email = 'mikekeith52@gmail.com',
  packages=find_packages('src'),
  package_dir={'': 'src'},
  url = 'https://github.com/mikekeith52/scalecast',
  keywords = ['FORECAST', 'SCALE', 'FLEXIBLE'],
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
    'openpyxl'
  ],
)
