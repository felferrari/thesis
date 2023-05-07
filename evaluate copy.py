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
    cfg = yaml.safe_load(file)

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

prediction_prefix = general_params['prediction_prefix']
n_models = general_params['n_models']
min_cloud_cover = general_params['min_cloud_cover']
base_data = paths['opt']['y0'][0]

outfile = os.path.join(path_logs, f'eval_{args.experiment}.txt')

def eval_data(img_idx_0, img_idx_1, label_file, model_idx = None):

    #reading files
    cmap_0_file = paths['cmaps']['y1'][img_idx_0]
    cmap_1_file = paths['cmaps']['y2'][img_idx_1]
    label = load_sb_image(label_file)
    cmap_0 = load_sb_image(cmap_0_file)
    cmap_1 = load_sb_image(cmap_1_file)

    if model_idx is None:
        pred_prob = np.zeros_like(label, dtype=np.float16)
        for model_i in range(n_models):
            pred_prob_file = path_predicted / f'{prediction_prefix}_prob_{img_idx_0}_{img_idx_1}_{model_i}.npy'
            pred_prob += np.load(pred_prob_file).astype(np.float16)
        pred_prob = pred_prob / n_models
    else:
        pred_prob_file = path_predicted / f'{prediction_prefix}_prob_{img_idx_0}_{img_idx_1}_{model_idx}.npy'
        pred_prob = np.load(pred_prob_file).astype(np.float16)

    

    #generating maximum map
    max_cmap = np.maximum(cmap_0, cmap_1)
    del cmap_0, cmap_1


    pred_b = np.zeros_like(label, dtype=np.uint8)
    pred_b[pred_prob > 0.5] = 1
    del pred_prob

    pred_b[label==2] = 0
    pred_red = area_opening(pred_b, 625)
    #pred_red = np.zeros_like(label, dtype=np.uint8)
    pred_clean = pred_b
    pred_clean[label==2] = 2
    pred_clean[(pred_b - pred_red) == 1] = 2 
    label_clean = label.copy()
    label_clean[(pred_b - pred_red) == 1] = 2
    del label, pred_b, pred_red

    error_map = np.zeros_like(label_clean, dtype=np.uint8)
    error_map[np.logical_and(pred_clean == 0, label_clean == 0)] = 0 #tn
    error_map[np.logical_and(pred_clean == 1, label_clean == 1)] = 1 #tp
    error_map[np.logical_and(pred_clean == 0, label_clean == 1)] = 2 #fn
    error_map[np.logical_and(pred_clean == 1, label_clean == 0)] = 3 #fp

    if model_idx is None:
        error_map_file = path_visual / f'{prediction_prefix}_error_map_{args.experiment}_{img_idx_0}_{img_idx_1}.tif'
        save_geotiff(label_file, error_map_file, error_map, dtype = 'byte')
    else:
        error_map_file = path_visual / f'{prediction_prefix}_error_map_{args.experiment}_{img_idx_0}_{img_idx_1}_{model_idx}.tif'
        save_geotiff(label_file, error_map_file, error_map, dtype = 'byte')

    cloud_pixels = np.zeros_like(label_clean, dtype=np.uint8)
    cloud_pixels[max_cmap>0.5] = 1

    no_cloud_tns = np.logical_and(error_map == 0, cloud_pixels == 0).sum()
    no_cloud_tps = np.logical_and(error_map == 1, cloud_pixels == 0).sum()
    no_cloud_fns = np.logical_and(error_map == 2, cloud_pixels == 0).sum()
    no_cloud_fps = np.logical_and(error_map == 3, cloud_pixels == 0).sum()

    cloud_tns = np.logical_and(error_map == 0, cloud_pixels == 1).sum()
    cloud_tps = np.logical_and(error_map == 1, cloud_pixels == 1).sum()
    cloud_fns = np.logical_and(error_map == 2, cloud_pixels == 1).sum()
    cloud_fps = np.logical_and(error_map == 3, cloud_pixels == 1).sum()

    no_cloud_precision = no_cloud_tps / (no_cloud_tps + no_cloud_fps)
    no_cloud_recall = no_cloud_tps / (no_cloud_tps + no_cloud_fns)
    no_cloud_f1 = 2 * no_cloud_precision * no_cloud_recall / (no_cloud_precision + no_cloud_recall)

    cloud_precision = cloud_tps / (cloud_tps + cloud_fps)
    cloud_recall = cloud_tps / (cloud_tps + cloud_fns)
    cloud_f1 = 2 * cloud_precision * cloud_recall / (cloud_precision + cloud_recall)
  
    return [
        img_idx_0, 
        img_idx_1,
        model_idx,
        no_cloud_f1,
        no_cloud_precision,
        no_cloud_recall,
        cloud_f1,
        cloud_precision,
        cloud_recall,
        no_cloud_tns,
        no_cloud_tps,
        no_cloud_fns,
        no_cloud_fps,
        cloud_tns, 
        cloud_tps, 
        cloud_fns, 
        cloud_fps
    ]



if __name__ == '__main__':
    with open(outfile, 'w') as sys.stdout:
        label_file = paths['labels']['y2']
        args_list = []
        for im_0 in range(n_images):
            for im_1 in range(n_images):
                for model_idx in range(n_models):
                    args_list.append((im_0, im_1, label_file, model_idx))
        
        with Pool(10) as pool:
            models_pool_results=pool.starmap(eval_data, args_list)

        args_list = []
        for im_0 in range(n_images):
            for im_1 in range(n_images):
                args_list.append((im_0, im_1, label_file, None))
        
        with Pool(10) as pool2:
            models_pool_mean_results=pool2.starmap(eval_data, args_list)

        
        
        headers =  [
        'idx_0', 
        'idx_1',
        'model_idx',
        'no_cloud_f1',
        'no_cloud_precision',
        'no_cloud_recall',
        'cloud_f1',
        'cloud_precision',
        'cloud_recall',
        'no_cloud_tns',
        'no_cloud_tps',
        'no_cloud_fns',
        'no_cloud_fps',
        'cloud_tns', 
        'cloud_tps', 
        'cloud_fns', 
        'cloud_fps'
        ]
        
        results_df = pd.DataFrame(models_pool_results, columns=headers)
        mean_results_df = pd.DataFrame(models_pool_mean_results, columns=headers)

        for met in ['tns', 'tps', 'fns', 'fps']:
            results_df[f'global_{met}'] = results_df[f'no_cloud_{met}'] + results_df[f'cloud_{met}']
            mean_results_df[f'global_{met}'] = mean_results_df[f'no_cloud_{met}'] + mean_results_df[f'cloud_{met}']

        results_df['global_precision'] = results_df['global_tps'] / (results_df['global_tps'] + results_df['global_fps'])
        results_df['global_recall'] = results_df['global_tps'] / (results_df['global_tps'] + results_df['global_fns'])
        results_df['global_f1'] = 2 * results_df['global_precision'] * results_df['global_recall'] / (results_df['global_precision'] + results_df['global_recall'])

        mean_results_df['global_precision'] = mean_results_df['global_tps'] / (mean_results_df['global_tps'] + mean_results_df['global_fps'])
        mean_results_df['global_recall'] = mean_results_df['global_tps'] / (mean_results_df['global_tps'] + mean_results_df['global_fns'])
        mean_results_df['global_f1'] = 2 * mean_results_df['global_precision'] * mean_results_df['global_recall'] / (mean_results_df['global_precision'] + mean_results_df['global_recall'])

        mean_results_df = results_df.groupby(['model_idx'])[[
            'no_cloud_tns',
            'no_cloud_tps',
            'no_cloud_fns',
            'no_cloud_fps',
            'cloud_tns', 
            'cloud_tps', 
            'cloud_fns', 
            'cloud_fps',
            'global_tns',
            'global_tps',
            'global_fns',
            'global_fps',
        ]].apply(sum)

        overall_results_df = mean_results_df.apply(sum)
        
        for subset in ['no_cloud', 'cloud', 'global']:
            mean_results_df[f'{subset}_precision'] = mean_results_df[f'{subset}_tps'] / (mean_results_df[f'{subset}_tps'] + mean_results_df[f'{subset}_fps'])
            mean_results_df[f'{subset}_recall'] = mean_results_df[f'{subset}_tps'] / (mean_results_df[f'{subset}_tps'] + mean_results_df[f'{subset}_fns'])
            mean_results_df[f'{subset}_f1'] = 2 * mean_results_df[f'{subset}_precision'] * mean_results_df[f'{subset}_recall'] / (mean_results_df[f'{subset}_precision'] + mean_results_df[f'{subset}_recall'])

            overall_results_df[f'{subset}_precision'] = overall_results_df[f'{subset}_tps'] / (overall_results_df[f'{subset}_tps'] + overall_results_df[f'{subset}_fps'])
            overall_results_df[f'{subset}_recall'] = overall_results_df[f'{subset}_tps'] / (overall_results_df[f'{subset}_tps'] + overall_results_df[f'{subset}_fns'])
            overall_results_df[f'{subset}_f1'] = 2 * overall_results_df[f'{subset}_precision'] * overall_results_df[f'{subset}_recall'] / (overall_results_df[f'{subset}_precision'] + overall_results_df[f'{subset}_recall'])

        
        results_file = path_results / f'results_{args.experiment}.xlsx'
        with pd.ExcelWriter(results_file) as writer:     
            results_df.to_excel(writer, sheet_name='general results')
            mean_results_df.to_excel(writer, sheet_name='models results')
            overall_results_df.to_excel(writer, sheet_name='overall results')

        