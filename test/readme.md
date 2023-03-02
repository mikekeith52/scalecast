# Testing Updates

Version incremeting occurs by editing `__version__` in:

`./src/scalecast/__init__.py`  

Generally, increment the place after the second dot (.):

`0.17.0 --> 0.17.1`

Big updates will get an increment after the first dot (.):

`0.17.xx --> 0.18.0`

## Testing Steps

### 1. Create a virtual environment (skip if scalecast-env already exists)
`python3 -m venv scalecast-env`

### 2. Activate the environment
`source scalecast-env/bin/activate`

### 3. Change to the root of the directory:
`cd /path/to/scalecast/`

### 4. Install the package in editable mode with pip
`pip install -e .`

### 5. Check that it worked
`python -c "import scalecast; print(scalecast.__version__)"`

### 6. Run the test
`cd path/to/scalecast/test`  
`pip install -r requirements.txt`  
`python test_all.py`  

### 7. Deactivate environment
`source deactivate`

## Testing Steps (Anaconda)

### 1. Update conda
`conda update conda`

### 2. Create a virtual environment (skip if scalecast-env already exists)
`conda create -n scalecast-env python=3.x anaconda`

### 3. Activate the environment
`conda activate scalecast-env`

### 4. Change to the root of the directory:
`cd /path/to/scalecast/`

### 5. Install the package in editable mode with pip
`pip install -e .`

### 6. Check that it worked
`python -c "import scalecast; print(scalecast.__version__)"`

### 7. Run the test
`cd path/to/scalecast/test`  
`pip install -r requirements.txt`  
`conda install -c conda-forge cmdstanpy`  
*On MAC, may have to reinstall some packages using conda install*  
`python test_all.py`  

### 8. Deactivate environment
`conda deactivate`

If the resulting `error.log` file is free of errors, the test was successful and a new version can be committed. Only keep the most recently created error log.

## Committing Steps
`cd path/to/scalecast`  
`rm dist/*`  
`python setup.py sdist`  
`twine upload dist/*`  