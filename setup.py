
from setuptools import setup, find_packages
import shutil
from src.scalecast.__init__ import __version__ as version

logo = """<p align="center">
  <img src="https://github.com/mikekeith52/scalecast/blob/main/assets/logo2.png" />
</p>"""

long_description = open('README.md', 'r', encoding="UTF-8").read().replace(logo,"")

setup(
  name = 'SCALECAST',
  version = version,
  license='MIT',
  description='Easy dynamic time series forecasting in Python',
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
