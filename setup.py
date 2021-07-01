from distutils.core import setup
setup(
  name = 'SCALECAST',         # How you named your package folder (MyLib)
  packages = ['Forecaster'],   # Chose the same as "name"
  version = '0.1',      # Start with a small number and increase it with every change you make
  license='MIT',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
  description = 'A flexible, minimal-code forecasting object meant to be used with loops to forecast many series or to focus on one series for maximum accuracy.',   # Give a short description about your library
  author = 'Michael Keith',                   # Type in your name
  author_email = 'mikekeith52@gmail.com',      # Type in your E-Mail
  url = 'https://github.com/mikekeith52/scalecast',   # Provide either the link to your github or to your website
  download_url = 'https://github.com/mikekeith52/scalecast/archive/v_01.tar.gz',    # I explain this later on
  keywords = ['FORECAST', 'SCALE', 'FLEXIBLE'],   # Keywords that define your package best
  install_requires=[            # I get to this in a second
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
    'Development Status :: 4 - Beta',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Developers',      # Define that your audience are developers
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',   # Again, pick a license
    'Programming Language :: Python :: 3.6',      #Specify which pyhton versions that you want to support
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
  ],
)