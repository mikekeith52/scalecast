
from setuptools import setup, find_packages

setup(
  name = 'SCALECAST',
  version = '0.4.0',
  license='MIT',
  long_description='Dynamic forecasting at scale. See the documentation on [GitHub](https://github.com/mikekeith52/scalecast).',
  long_description_content_type='text/markdown',
  author = 'Michael Keith',
  author_email = 'mikekeith52@gmail.com',
  packages=find_packages('src'),
  package_dir={'': 'src'},
  url = 'https://github.com/mikekeith52/scalecast',
  keywords = ['FORECAST', 'SCALE', 'DYNAMIC'],
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
