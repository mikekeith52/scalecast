# Testing Updates

Version incremeting occurs in two places:

`./setup.py`
`./src/scalecast/__init__.py`

Generally, increment the place after the second '.':

`0.17.0 --> 0.17.1`

Big updates will get an increment after the first '.':

`0.17.xx --> 0.18.0`

## Testing Steps

### Create a virtual environment (skip if scalecast-env already exists)
`python3 -m venv scalecast-env`

### Activate the environment
`source scalecast-env/bin/activate`

### Change to the root of the directory:
`cd /path/to/scalecast/`

### Install the package in editable mode with pip
`pip install -e --upgrade .`

### Check that it worked
`python -c "import scalecast; print(scalecast.__version___)`

### Run the test
`cd path/to/scalecast/test`
`pip install -r requirements.txt`
`python test_all.py`

### Deactivate environment
`source deactivate`

## Testing Steps (Anaconda)

### Update conda
`conda update conda`

### Create a virtual environment (skip if scalecast-env already exists)
`conda create -n scalecast-env python=3.x anaconda`

### Activate the environment
`ssource activate scalecast-env`

### Change to the root of the directory:
`cd /path/to/scalecast/`

### Install the package in editable mode with pip
`pip install -e --upgrade .`

### Check that it worked
`python -c "import scalecast; print(scalecast.__version___)`

### Run the test
`cd path/to/scalecast/test`
`pip install -r requirements.txt`
`python test_all.py`

### Deactivate environment
`source deactivate`

If the resulting `error.log` file is free of errors, the test was successful and a new version can be committed. Only keep the most recently created error log.

## Committing Steps
`cd path/to/scalecast`
`python setup.py sdist`
`twine upload sdist`