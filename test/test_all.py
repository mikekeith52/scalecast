import logging
import traceback
import os
import datetime
import importlib

script_files = [ # comment out scripts to skip testing
    'test_AnomalyDetector',
    'test_ChangepointDetector',
    'test_Forecaster',
    'test_GridGenerator',
    'test_multiseries',
    'test_MVForecaster',
    'test_Pipeline',
    'test_SeriesTransformer',
]

timestamp = datetime.datetime.today().strftime('%Y%m%d%H%m%S')

logging.basicConfig(filename=f'error_{timestamp}.log', level=logging.ERROR)

for script in script_files:
    print('='*50,script'='*50,sep='\n')
    try:
        mod = importlib.import_module(script)
        mod.main()
    except Exception as e:
        logging.error(traceback.format_exc())
        print(f'Error in {script}: {e}')