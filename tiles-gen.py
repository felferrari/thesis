import argparse
from utils.ops import load_opt_image, save_geotiff
import numpy as np
from  pathlib import Path
import yaml

parser = argparse.ArgumentParser(
    description='Generate .tif with training (0) and validation (1) areas.'
)

parser.add_argument( # The path to the config file (.yaml)
    '-c', '--cfg',
    type = Path,
    default = 'cfg.yaml',
    help = 'Path to the config file (.yaml).'
)

args = parser.parse_args()

with open(args.cfg, 'r') as file:
    cfg = yaml.load(file, Loader=yaml.Loader)

original_data = cfg['original_data']
tiles_params = cfg['tiles_params']

base_image = Path(original_data['opt']['folder']) / original_data['opt']['imgs']['train'][0][0]

shape = load_opt_image(base_image).shape[0:2]

tiles = np.zeros(shape, dtype=np.uint8).reshape((-1,1))
idx_matrix = np.arange(shape[0]*shape[1], dtype=np.uint32).reshape(shape)

tiles_idx = []
for hor in np.array_split(idx_matrix, tiles_params['lines'], axis=0):
    for tile in np.array_split(hor, tiles_params['columns'], axis=1):
        tiles_idx.append(tile)

   
for i, tile in enumerate(tiles_idx):
    if i in tiles_params['train_tiles']:
        tiles[tile] = 1

tiles = tiles.reshape(shape)

save_geotiff(
    base_image, 
    tiles_params['path'], 
    tiles, 
    'byte'
    )