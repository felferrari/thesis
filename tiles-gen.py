import argparse
import pathlib
from conf import paths, default
from utils.ops import load_opt_image, save_geotiff
import numpy as np
import os
from osgeo import gdal, gdalconst

parser = argparse.ArgumentParser(
    description='Generate .tif with training (0) and validation (1) areas.'
)

parser.add_argument( # base image 
    '-b', '--base-image-path',
    type = pathlib.Path,
    default = paths.OPT_PATH,
    help = 'Path to the folder of the geotiff files to generate aligned labels of the same region'
)

parser.add_argument( # number of tiles lines
    '-l', '--lines',
    type = int,
    default = default.TILES_LIN,
    help = 'Number of lines to split the image'
)

parser.add_argument( # number of tiles columns
    '-c', '--columns',
    type = int,
    default = default.TILES_COL,
    help = 'Number of columns to split the image'
)

parser.add_argument( # List of train index tiles
    '-v', '--train-tiles',
    type = int,
    default = default.TILES_TRAIN,
    nargs='+',
    help = 'List of train index tiles'
)

parser.add_argument( # output file
    '-o', '--output',
    type = pathlib.Path,
    default = paths.TILES_PATH,
    help = 'Path to the output tiles file (.tif)'
)

args = parser.parse_args()

base_image = os.listdir(str(args.base_image_path))[0]

shape = load_opt_image(os.path.join(str(args.base_image_path), base_image)).shape[0:2]

tiles = np.zeros(shape, dtype=np.uint8).reshape((-1,1))
idx_matrix = np.arange(shape[0]*shape[1], dtype=np.uint32).reshape(shape)

tiles_idx = []
for hor in np.array_split(idx_matrix, args.lines, axis=0):
    for tile in np.array_split(hor, args.columns, axis=1):
        tiles_idx.append(tile)

   
for i, tile in enumerate(tiles_idx):
    if i in args.train_tiles:
        tiles[tile] = 1

tiles = tiles.reshape(shape)

save_geotiff(
    os.path.join(str(args.base_image_path), base_image), 
    str(args.output), 
    tiles, 
    'byte'
    )