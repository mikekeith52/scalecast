from distutils.core import setup
setup(
  name = 'SCALECAST',
  packages = ['Scalecast'],
  version = '0.1.1',
  license='MIT',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
  description = 'A flexible, minimal-code forecasting object meant to be used with loops to forecast many series or to focus on one series for maximum accuracy.',   # Give a short description about your library
  author = 'Michael Keith',
  author_email = 'mikekeith52@gmail.com',
  url = 'https://github.com/mikekeith52/scalecast',
  download_url = 'https://github.com/mikekeith52/scalecast/archive/refs/tags/0.1.1.tar.gz',
  keywords = ['FORECAST', 'SCALE', 'FLEXIBLE'],
  install_requires=[ 
          'sklearn',
          'fbprophet',
          'pandas',
          'numpy',
          'seaborn',
          'matplotlib',
          'pandas_datareader',
          'scipy',
          'statsmodels',
          'eli5'
      ],
  classifiers=[
    'Development Status :: 4 - Beta',      # "3 - Alpha", "4 - Beta" or "5 - Production/Stable"
    'Intended Audience :: Developers',      
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
  ],
)
