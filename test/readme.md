# Testing Updates

Version incremeting occurs by editing `__version__` in:

`./src/scalecast/__init__.py`  

Generally, increment the place after the second dot (.):

`0.17.0 --> 0.17.1`

Big updates will get an increment after the first dot (.):

`0.17.xx --> 0.18.0`

## Testing Steps - UV Only

### 1. Create a virtual environment

From root directory:

`uv venv` 

### 2. Activate the environment
`source .venv/bin/activate`

### 3. Install the package in editable mode with pip
`uv pip install -e ".[test]"`

### 4. Run the tests
```bash
cd test
uv run test_all.py
```

If the resulting `error.log` file is free of errors, the test was successful.

## Committing Steps
From root:

```bash
rm -rf dist build *.egg-info
uv run python -m build
twine check dist/*
twine upload dist/*
```