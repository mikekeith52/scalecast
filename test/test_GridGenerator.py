from scalecast import GridGenerator

def main():
    GridGenerator.get_example_grids(out_name='ExampleGrids.py',overwrite=True)
    GridGenerator.get_grids('vecm',out_name='VECMGrid.py',overwrite=True)
    GridGenerator.get_mv_grids(overwrite=True)
    GridGenerator.get_empty_grids(overwrite=True)

if __name__ == '__main__':
    main()