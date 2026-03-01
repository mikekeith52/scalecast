from pathlib import Path

def get_grids(grid:str='example',out_name:str='Grids.py',overwrite:bool=False) -> None:
    """ Saves a grids file to the working directory.
    See all available grids files here: https://github.com/mikekeith52/scalecast/tree/main/src/scalecast/grids.
    Make your own grids file and open a pull request on GitHub to add it to the library.

    Args:
        grid (str): Default 'example'. The name of the grids file within scalecast.
            Do not add the '.py' extension.
        out_name (str): Default 'Grids.py'. The name of the grids file that will be
            saved to the user's working directory.
        overwrite (bool): Default False.
            Whether to overwrite a file (with the out_name name) if one is already in the working directory.

    Returns:
        None
    """
    output_file = Path.cwd()/out_name
    if not overwrite and output_file.exists():
        return

    grids_dir = Path(__file__).parent/"grids"
    input_file = grids_dir/f'{grid}.py'

    with open(input_file, "r") as fl:
        contents = fl.read()

    # Write the contents to a file in the user's working directory
    with open(out_name, "w") as fl:
        fl.write(contents)

def get_example_grids(out_name:str='Grids.py',overwrite:bool=False):
    """ Saves example grids to working directory as Grids.py (does not overwrite by default).

    Args:
        out_name (str): Default 'Grids.py'. The name of the file to write the grids to.
        overwrite (bool): Default False.
            Whether to overwrite a file (with the out_name name) if one is already in the working directory.

    Returns:
        None
    """
    get_grids(
        out_name = out_name,
        overwrite = overwrite,
    )


def get_mv_grids(out_name:str='MVGrids.py',overwrite:bool=False):
    """ Saves example grids to working directory as MVGrids.py (does not overwrite by default).

    Args:
        out_name (str): Default 'MVGrids.py'. The name of the file to write the grids to.
        overwrite (bool): Default False.
            Whether to overwrite a file (with the out_name name) if one is already in the working directory.

    Returns:
        None
    """
    get_grids(
        grid = 'mv',
        out_name = out_name,
        overwrite = overwrite,
    )


def get_empty_grids(out_name:str='Grids.py',overwrite:bool=False):
    """ Saves empty grids to working directory as Grids.py (does not overwrite by default).

    Args:
        out_name (str): Default 'Grids.py'. The name of the file to write the grids to.
        overwrite (bool): Default False.
            Whether to overwrite a file (with the out_name name) if one is already in the working directory.

    Returns:
        None
    """
    get_grids(
        grid = 'empty',
        out_name = out_name,
        overwrite = overwrite,
    )
