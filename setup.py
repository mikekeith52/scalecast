from distutils.core import setup
setup(
  name = 'SCALECAST',
  packages = ['Scalecast'],
  version = '0.1.2',
  license='MIT',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
  author = 'Michael Keith',
  author_email = 'mikekeith52@gmail.com',
  url = 'https://github.com/mikekeith52/scalecast',
  keywords = ['FORECAST', 'SCALE', 'FLEXIBLE'],
  install_requires = [
    'scikit-learn',
    'statsmodels',
    'eli5',
    'numpy',
    'pandas',
    'scipy',
    'matplotlib',
    'seaborn',
  ],
)
