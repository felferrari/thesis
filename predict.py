import argparse
from pathlib import Path
import importlib
#from conf import general, paths
import os
import time
from utils.dataloader import PredDataset
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm, trange
import numpy as np
from utils.ops import save_geotiff, load_json, load_sb_image
import logging
import yaml
from multiprocessing import Process, freeze_support
from pydoc import locate

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
    default = 12,
    help = 'The number of the experiment'
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
prediction_remove_border = prediction_params['prediction_remove_border']
n_models = prediction_params['n_models']
batch_size = prediction_params['batch_size']
overlaps = prediction_params['prediction_overlaps']

prepared_folder = Path(preparation_params['folder'])
test_folder = prepared_folder / preparation_params['test_folder']
prediction_prefix = experiment_params['prefixs']['prediction']

device = "cuda" if torch.cuda.is_available() else "cpu"
    
def run(model_idx):

    outfile = logs_path / f'pred_{args.experiment}_{model_idx}.txt'
    logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s:%(levelname)s:%(name)s:%(message)s',
            filename=outfile,
            filemode='w'
            )
    log = logging.getLogger('predict')

    model = locate(experiment_params['model'])(experiment_params)
    model.to(device)

    model_file = models_path / f'model_{model_idx}.pth'
    model.load_state_dict(torch.load(model_file))
    model.eval()

    one_window = np.ones((patch_size, patch_size, n_classes))
    total_time = 0
    n_processed_images = 0

    #label = load_sb_image(Path(label_params['test_path'])).astype(np.uint8)
    #original_shape = label.shape
    #pad_shape = ((patch_size, patch_size),(patch_size, patch_size))
    #padded_shape = np.pad(label, pad_shape, mode='reflect').shape

    pred_ds = PredDataset(patch_size, device, experiment_params, test_folder)

    for opt_group_i, opt_group in enumerate(tqdm(experiment_params['test_opt_imgs'], leave = False, desc = 'OPT group')):
        pred_ds.load_opt_data(opt_group)
        for sar_group_i, sar_group in enumerate(tqdm(experiment_params['test_sar_imgs'], leave = False, desc = 'SAR group')):
            pred_ds.load_sar_data(sar_group)

            pred_global_sum = np.zeros(pred_ds.original_shape+(n_classes,))
            t0 = time.perf_counter()

            for overlap in tqdm(overlaps, leave=False, desc='Overlap'):
                pred_ds.generate_overlap_patches(overlap)
                pred_ds[0]
                #dataloader = DataLoader(pred_ds, batch_size=batch_size, num_workers=2, shuffle=False)
                dataloader = DataLoader(pred_ds, batch_size=batch_size, shuffle=False)
                
                pbar = tqdm(dataloader, desc='Prediction', leave = False)
                #preds = None
                preds = torch.zeros((len(pred_ds), n_classes, patch_size, patch_size))
                for i, X in enumerate(pbar):
                    with torch.no_grad():
                        preds[batch_size*i: batch_size*(i+1)] =  model(X[0]).to('cpu')
                preds = np.moveaxis(preds.numpy().astype(np.float16), 1, -1)
                pred_sum = np.zeros(pred_ds.padded_shape+(n_classes,)).reshape((-1, n_classes))
                pred_count = np.zeros(pred_ds.padded_shape+(n_classes,)).reshape((-1, n_classes))
                for idx, idx_patch in enumerate(tqdm(pred_ds.idx_patches, desc = 'Rebuild', leave = False)):
                    crop_val = prediction_remove_border
                    idx_patch_crop = idx_patch[crop_val:-crop_val, crop_val:-crop_val]
                    pred_sum[idx_patch_crop] += preds[idx][crop_val:-crop_val, crop_val:-crop_val]
                    pred_count[idx_patch_crop] += one_window[crop_val:-crop_val, crop_val:-crop_val]

                pred_sum = pred_sum.reshape(pred_ds.padded_shape+(n_classes,))
                pred_count = pred_count.reshape(pred_ds.padded_shape+(n_classes,))

                pred_sum = pred_sum[patch_size:-patch_size, patch_size:-patch_size,:]
                pred_count = pred_count[patch_size:-patch_size, patch_size:-patch_size,:]

                pred_global_sum += pred_sum / pred_count

            p_time = (time.perf_counter() - t0)/60
            total_time += p_time
            n_processed_images += 1
            log.info(f'Prediction time: {p_time} mins')
            pred_global = pred_global_sum / len(overlaps)

            pred_global_file = predicted_path / f'{prediction_prefix}_prob_{opt_group_i}_{sar_group_i}_{model_idx}.npy'
            np.save(pred_global_file, pred_global[:,:,1].astype(np.float16))

            pred_b2 = (pred_global[:,:,1] > 0.5).astype(np.uint8)
            pred_b2[pred_ds.original_label == 2] = 2

            base_data = Path(original_data_params['opt']['folder']) / original_data_params['opt']['imgs']['test'][0]
            prediction_tif_file = visual_path / f'{prediction_prefix}_{args.experiment}_{opt_group_i}_{sar_group_i}_{model_idx}.tif'
            save_geotiff(base_data, prediction_tif_file, pred_b2, dtype = 'byte')
    m_time = total_time / n_processed_images
    log.info(f'Mean Prediction time: {m_time} mins')

if __name__=="__main__":
    freeze_support()
    
    for model_idx in trange(prediction_params['n_models'], desc = 'Model'):
        p = Process(target=run, args=(model_idx,))
        p.start()
        p.join()