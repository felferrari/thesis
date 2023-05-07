import argparse
from  pathlib import Path
import importlib
#from conf import default, general, paths
import os
import time
import sys
from utils.dataloader import PredDataSet
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import numpy as np
from osgeo import ogr, gdal, gdalconst
from utils.ops import save_geotiff, load_sb_image
from multiprocessing import Process
from skimage.morphology import area_opening
import pandas as pd
import yaml
from multiprocessing import Pool
from itertools import combinations, product, repeat

parser = argparse.ArgumentParser(
    description='Train NUMBER_MODELS models based in the same parameters'
)

parser.add_argument( # The path to the config file (.yaml)
    '-c', '--cfg',
    type = Path,
    default = 'cfg.yaml',
    help = 'Path to the config file (.yaml).'
)

parser.add_argument( # Experiment number
    '-e', '--experiment',
    type = int,
    default = 1,
    help = 'The number of the experiment'
)

parser.add_argument( # Generate the Max Cloud Map Geotiff
    '-s', '--cloud-max-map',
    #type = bool,
    action='store_true',
    help = 'Generate the Max Cloud Map Geotiff'
)


args = parser.parse_args()

with open(args.cfg, 'r') as file:
    cfg = yaml.load(file, Loader=yaml.Loader)

prediction_params = cfg['prediction_params']
preparation_params = cfg['preparation_params']
experiment_params = cfg['experiments'][f'exp_{args.experiment}']
label_params = cfg['label_params']
previous_def_params = cfg['previous_def_params']
original_data_params = cfg['original_data']

experiments_paths = prediction_params['experiments_paths']
prepared_folder = Path(preparation_params['folder'])

exp_path = Path(experiments_paths['folder']) / f'exp_{args.experiment}'

models_path = exp_path / experiments_paths['models']
logs_path = exp_path / experiments_paths['logs']
visual_path = exp_path / experiments_paths['visual']
predicted_path = exp_path / experiments_paths['predicted']
results_path = exp_path / experiments_paths['results']

patch_size = prediction_params['patch_size']
n_classes = prediction_params['n_classes']
prediction_prefix = prediction_params['prediction_prefix']
n_models = prediction_params['n_models']

n_opt_imgs_groups = len(original_data_params['opt']['imgs']['test'])
n_sar_imgs_groups = len(original_data_params['sar']['imgs']['test'])
imgs_groups_idxs = []
for opt_imgs_group_idx in range(n_opt_imgs_groups):
     for sar_imgs_group_idx in range(n_sar_imgs_groups):
          imgs_groups_idxs.append([opt_imgs_group_idx, sar_imgs_group_idx])

#mean prediction

def mean_prediction(opt_imgs_groups_idx, sar_imgs_groups_idx):
     label = load_sb_image(Path(label_params['test_path'])).astype(np.uint8)
     pred_prob = np.zeros_like(label, dtype=np.float16)
     for model_idx in range(n_models):
          pred_prob_file = predicted_path / f'{prediction_prefix}_prob_{img_idx_0}_{img_idx_1}_{model_i}.npy'
          
    

if __name__=="__main__":
    

    with Pool(5) as pool2:
            models_pool_mean_results=pool2.starmap(mean_prediction, imgs_groups_idxs)