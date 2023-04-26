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

params = cfg['params']
paths = cfg['paths']

path_exp = Path(paths['experiments']) / f'exp_{args.experiment}'
path_exp.mkdir(exist_ok=True)

path_logs = path_exp / paths['exp_subpaths']['logs']
path_models = path_exp / paths['exp_subpaths']['models']
path_visual = path_exp / paths['exp_subpaths']['visual']
path_predicted = path_exp / paths['exp_subpaths']['predicted']
path_results = path_exp / paths['exp_subpaths']['results']

n_images = params['n_images']
prediction_prefix = params['prediction_prefix']
n_models = params['n_models']
min_cloud_cover = params['min_cloud_cover']
base_data = paths['opt']['y0'][0]

outfile = os.path.join(path_logs, f'eval_{args.experiment}.txt')
with open(outfile, 'w') as sys.stdout:

    #evaluate average predictions
    #label = np.load(os.path.join(paths.PREPARED_PATH, f'{general.LABEL_PREFIX}_{args.year}.npy'))
    label = load_sb_image(paths['labels']['y2'])
    tps, tns, fps, fns = [], [], [], []
    c_tps, c_tns, c_fps, c_fns = [], [], [], []
    nc_tps, nc_tns, nc_fps, nc_fns = [], [], [], []
    for im_0 in tqdm(range(n_images), desc = 'Img 0'):
        for im_1 in tqdm(range(n_images), desc = 'Img 1', leave = False):
            pred_sum = np.zeros_like(label, dtype=np.float16)
            for model_idx in range(n_models):
                pred_prob_file = path_predicted / f'{prediction_prefix}_prob_{im_0}_{im_1}_{model_idx}.npy'
                pred_prob = np.load(pred_prob_file)
                pred_sum += pred_prob
            average_pred_prob = pred_sum / n_models
            avg_pred_file = path_visual /  f'{prediction_prefix}_avg_prob_{args.experiment}_{im_0}_{im_1}.tif'
            save_geotiff(base_data, avg_pred_file, average_pred_prob, dtype = 'float')
            pred_prob_file = path_predicted / f'{prediction_prefix}_prob_{im_0}_{im_1}.npy'
            np.save(pred_prob_file, average_pred_prob)

            pred_b = np.zeros_like(label, dtype=np.uint8)
            pred_b[average_pred_prob > 0.5] = 1
            pred_b[label==2] = 0
            pred_red = area_opening(pred_b, 625)

            pred_clean = pred_b
            pred_clean[label==2] = 2
            pred_clean[(pred_b - pred_red) == 1] = 2 
            label_clean = label.copy()
            label_clean[(pred_b - pred_red) == 1] = 2
            avg_bin_file = path_visual /  f'{prediction_prefix}_avg_bin_{args.experiment}_{im_0}_{im_1}.tif'
            save_geotiff(base_data, avg_bin_file, pred_clean, dtype = 'byte')

            error_map = np.zeros_like(label, dtype=np.uint8)
            error_map[np.logical_and(pred_clean == 0, label_clean == 0)] = 0
            error_map[np.logical_and(pred_clean == 1, label_clean == 1)] = 1
            error_map[np.logical_and(pred_clean == 0, label_clean == 1)] = 2
            error_map[np.logical_and(pred_clean == 1, label_clean == 0)] = 3

            error_map_file = path_visual / f'{prediction_prefix}_error_map_{args.experiment}_{im_0}_{im_1}.tif'
            save_geotiff(base_data, error_map_file, error_map, dtype = 'byte')

            keep_samples = (label.flatten() != 2) #remove samples of class 2
            keep_samples[(pred_b - pred_red).flatten() == 1] = False #remove samples of deforestation smaller than 6.25ha

            predicted_samples = pred_b.flatten()[keep_samples]
            label_samples = label.flatten()[keep_samples]

            tp = np.logical_and(predicted_samples == 1, label_samples == 1).sum()
            tn = np.logical_and(predicted_samples == 0, label_samples == 0).sum()
            fp = np.logical_and(predicted_samples == 1, label_samples == 0).sum()
            fn = np.logical_and(predicted_samples == 0, label_samples == 1).sum()

            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1 = 2 * precision * recall / (precision + recall)

            print(f'Image Year 0: {im_0}, Image Year 1: {im_1}, Pecision: {precision:.6f}, Recall: {recall:.6f}, F1-Score: {f1:.6f}')
            print(f'Image Year 0: {im_0}, Image Year 1: {im_1}, {tp:,}|{tn:,}|{fp:,}|{fn:,}')

            tps.append(tp)
            tns.append(tn)
            fps.append(fp)
            fns.append(fn)

            '''year_0 = str(args.year-1)[2:]
            year_1 = str(args.year)[2:]

            opt_files = os.listdir(paths.PREPARED_OPT_PATH)
            sar_files = os.listdir(paths.PREPARED_SAR_PATH)

            opt_files_0 = [fi for fi in opt_files if fi.startswith(year_0)]
            opt_file_0 = opt_files_0[im_0]
            opt_files_1 = [fi for fi in opt_files if fi.startswith(year_1)]
            opt_file_1 = opt_files_1[im_1]

            cmap_0 = load_sb_image(os.path.join(paths.GENERAL_PATH, f'{general.CMAP_PREFIX}_{opt_file_0[:-4]}.tif'))
            cmap_1 = load_sb_image(os.path.join(paths.GENERAL_PATH, f'{general.CMAP_PREFIX}_{opt_file_1[:-4]}.tif'))'''

            cmap_0 = load_sb_image(paths['cmaps']['y1'][im_0])
            cmap_1 = load_sb_image(paths['cmaps']['y2'][im_1])

            max_cmap = np.maximum(cmap_0, cmap_1)
            mcp_map = max_cmap>=min_cloud_cover
            mcp_map = mcp_map.astype(np.uint8)
            if args.cloud_max_map:
                mcp_map_file = path_visual / f'mcp_map_{args.experiment}_{im_0}_{im_1}.tif'
                save_geotiff(base_data, mcp_map_file, mcp_map, dtype = 'byte')
            max_cmap = max_cmap.flatten()[keep_samples]

            cloud_args = max_cmap >= min_cloud_cover

            cloud_predicted_samples = predicted_samples[cloud_args]
            cloud_label_samples = label_samples[cloud_args]

            no_cloud_predicted_samples = predicted_samples[np.logical_not(cloud_args)]
            no_cloud_label_samples = label_samples[np.logical_not(cloud_args)]

            cloud_tp = np.logical_and(cloud_predicted_samples == 1, cloud_label_samples == 1).sum()
            cloud_tn = np.logical_and(cloud_predicted_samples == 0, cloud_label_samples == 0).sum()
            cloud_fp = np.logical_and(cloud_predicted_samples == 1, cloud_label_samples == 0).sum()
            cloud_fn = np.logical_and(cloud_predicted_samples == 0, cloud_label_samples == 1).sum()

            no_cloud_tp = np.logical_and(no_cloud_predicted_samples == 1, no_cloud_label_samples == 1).sum()
            no_cloud_tn = np.logical_and(no_cloud_predicted_samples == 0, no_cloud_label_samples == 0).sum()
            no_cloud_fp = np.logical_and(no_cloud_predicted_samples == 1, no_cloud_label_samples == 0).sum()
            no_cloud_fn = np.logical_and(no_cloud_predicted_samples == 0, no_cloud_label_samples == 1).sum()

            c_tps.append(cloud_tp)
            c_tns.append(cloud_tn)
            c_fps.append(cloud_fp)
            c_fns.append(cloud_fn)

            nc_tps.append(no_cloud_tp)
            nc_tns.append(no_cloud_tn)
            nc_fps.append(no_cloud_fp)
            nc_fns.append(no_cloud_fn)

    print('end')
    tps = np.array(tps).sum()
    tns = np.array(tns).sum()
    fps = np.array(fps).sum()
    fns = np.array(fns).sum()

    c_tps = np.array(c_tps).sum()
    c_tns = np.array(c_tns).sum()
    c_fps = np.array(c_fps).sum()
    c_fns = np.array(c_fns).sum()

    nc_tps = np.array(nc_tps).sum()
    nc_tns = np.array(nc_tns).sum()
    nc_fps = np.array(nc_fps).sum()
    nc_fns = np.array(nc_fns).sum()

    precision = tps / (tps + fps)
    recall = tps / (tps + fns)
    f1 = 2 * precision * recall / (precision + recall)

    cloud_precision = c_tps / (c_tps + c_fps)
    cloud_recall = c_tps / (c_tps + c_fns)
    cloud_f1 = 2 * cloud_precision * cloud_recall / (cloud_precision + cloud_recall)

    no_cloud_precision = nc_tps / (nc_tps + nc_fps)
    no_cloud_recall = nc_tps / (nc_tps + nc_fns)
    no_cloud_f1 = 2 * no_cloud_precision * no_cloud_recall / (no_cloud_precision + no_cloud_recall)

    print(f'Global Results, Pecision: {precision:.6f}, Recall: {recall:.6f}, F1-Score: {f1:.6f}')
    print(f'Global Results, {tps:,}|{tns:,}|{fps:,}|{fns:,}')

    print(f'Cloud Results, Pecision: {cloud_precision:.6f}, Recall: {cloud_recall:.6f}, F1-Score: {cloud_f1:.6f}')
    print(f'Cloud Results, {c_tps:,}|{c_tns:,}|{c_fps:,}|{c_fns:,}')

    print(f'No Cloud Results, Pecision: {no_cloud_precision:.6f}, Recall: {no_cloud_recall:.6f}, F1-Score: {no_cloud_f1:.6f}')
    print(f'No Cloud Results, {nc_tps:,}|{nc_tns:,}|{nc_fps:,}|{nc_fns:,}')

    results_file = path_logs / f'results_{args.experiment}.xlsx'
    with pd.ExcelWriter(results_file) as writer:  
        df = pd.DataFrame({
            'nc prec': [no_cloud_precision],
            'nc recall': [no_cloud_recall],
            'nc f1': [no_cloud_f1],
            'nc tps': [nc_tps],
            'nc tns': [nc_tns],
            'nc fps': [nc_fps],
            'nc fns': [nc_fns],

            'c prec': [cloud_precision],
            'c recall': [cloud_recall],
            'c f1': [cloud_f1],
            'c tps': [c_tps],
            'c tns': [c_tns],
            'c fps': [c_fps],
            'c fns': [c_fns],

            'global prec': [precision],
            'global recall': [recall],
            'global f1': [f1],
            'global tps': [tps],
            'global tns': [tns],
            'global fps': [fps],
            'global fns': [fns]
        })

        df.to_excel(writer, sheet_name='results')



