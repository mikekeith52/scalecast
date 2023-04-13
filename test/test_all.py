import logging
import traceback
import os
import datetime
import importlib

script_files = [ # comment out scripts to skip testing
    'test_GridGenerator',
    'test_Forecaster',
    'test_MVForecaster',
    'test_AnomalyDetector',
    'test_multiseries',
    'test_Pipeline',
    'test_SeriesTransformer',
    #'test_ChangepointDetector', # kats is giving me problems and I don't want to deal with it right now
]

timestamp = datetime.datetime.today().strftime('%Y%m%d%H%m%S')

if __name__ == '__main__':
    logging.basicConfig(filename=f'error_{timestamp}.log', level=logging.ERROR)
    for script in script_files:
        print('='*50,script,'='*50,sep='\n')
        try:
            mod = importlib.import_module(script)
            mod.main()
            print(f'No errors in {script}')
        except Exception as e:
            logging.error(traceback.format_exc())
            print(f'Error in {script}: {e}')