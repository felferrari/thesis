import argparse
from pathlib import Path
import time
from utils.datasets import PredDataset
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import numpy as np
from utils.ops import save_geotiff, load_yaml, save_yaml, load_sb_image
import logging
import yaml
from multiprocessing import Process, freeze_support, Pool
from pydoc import locate
import lightning.pytorch as pl

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
    default = 8,
    help = 'The number of the experiment'
)

parser.add_argument( # Model number
    '-m', '--model',
    type = int,
    default = -1,
    help = 'Number of the model to be retrained'
)

parser.add_argument( # Model number
    '-i', '--start-model',
    type = int,
    default = -1,
    help = 'Number of the model to be retrained'
)

parser.add_argument( # Model number
    '-d', '--device',
    type = int,
    default = 0,
    help = 'Number of the model to be retrained'
)

parser.add_argument( # specific site location number
    '-s', '--site',
    type = int,
    default=1,
    help = 'Site location number'
)

args = parser.parse_args()

cfg = load_yaml(args.cfg)
site_cfg = load_yaml(f'site_{args.site}.yaml')

prediction_params = cfg['prediction_params']
preparation_params = cfg['preparation_params']
experiment_params = cfg['experiments'][f'exp_{args.experiment}']
paths_params = cfg['paths']
general_params = cfg['general_params']

experiments_folders = general_params['experiments_folders']

original_opt_imgs = site_cfg['original_data']['opt_imgs']
original_sar_imgs = site_cfg['original_data']['sar_imgs']

experiments_paths = paths_params['experiments']

exp_path = Path(experiments_paths) / f'exp_{args.experiment}'

models_path = exp_path / experiments_folders['models']
logs_path = exp_path / experiments_folders['logs']
visual_path = exp_path / experiments_folders['visual']
predicted_path = exp_path / experiments_folders['predicted']
results_path = exp_path / experiments_folders['results']
visual_logs_path = exp_path / experiments_folders['visual_logs']

patch_size = general_params['patch_size']
n_classes = general_params['n_classes']
prediction_remove_border = prediction_params['prediction_remove_border']
n_models = general_params['n_models']
batch_size = prediction_params['batch_size']
overlaps = prediction_params['prediction_overlaps']

prepared_folder = Path(paths_params['prepared_data'])
test_folder = prepared_folder / preparation_params['test_folder']
prediction_prefix = general_params['prefixs']['prediction']

opt_folder = Path(paths_params['opt_data'])
sar_folder = Path(paths_params['sar_data'])

statistics_file = prepared_folder / preparation_params['statistics_data']

device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"

def run_prediction(models_pred_idx, test_opt_img, test_sar_img, opt_i, sar_i):

    print(f'loading files... Opt gp {opt_i} SAR Gp {sar_i}')

    statistics = load_yaml(statistics_file)
    
    label = load_sb_image(paths_params['label_test'])
    pred_ds = PredDataset(patch_size, device, experiment_params, test_opt_img, test_sar_img, paths_params['previous_test'], statistics)

    pred_global_sum = np.zeros(pred_ds.original_shape+(n_classes,))
    one_window = np.ones((patch_size, patch_size, n_classes))

    for model_idx in tqdm(models_pred_idx, desc = 'Models\' prediction', mininterval = 2):
        pred_results = load_yaml(logs_path / f'model_{model_idx}' / 'train_results.yaml')
        model_class = locate(experiment_params['model'])#(experiment_params, training_params)
        model = model_class.load_from_checkpoint(pred_results['model_path'])
        model.to(device)

        for overlap in overlaps:
            pred_ds.generate_overlap_patches(overlap)
            dataloader = DataLoader(pred_ds, batch_size=batch_size, shuffle=False)

            pbar = tqdm(dataloader, desc='Prediction', leave = False, mininterval = 2)
            #preds = None
            preds = torch.zeros((len(pred_ds), n_classes, patch_size, patch_size))

            for i, X in enumerate(pbar):
                with torch.no_grad():
                    preds[batch_size*i: batch_size*(i+1)] =  model(X).to('cpu')

            preds = np.moveaxis(preds.numpy().astype(np.float16), 1, -1)
            pred_sum = np.zeros(pred_ds.padded_shape+(n_classes,)).reshape((-1, n_classes))
            pred_count = np.zeros(pred_ds.padded_shape+(n_classes,)).reshape((-1, n_classes))
            for idx, idx_patch in enumerate(tqdm(pred_ds.idx_patches, desc = 'Rebuild', leave = False, mininterval = 2)):
                crop_val = prediction_remove_border
                idx_patch_crop = idx_patch[crop_val:-crop_val, crop_val:-crop_val]
                pred_sum[idx_patch_crop] += preds[idx][crop_val:-crop_val, crop_val:-crop_val]
                pred_count[idx_patch_crop] += one_window[crop_val:-crop_val, crop_val:-crop_val]

            pred_sum = pred_sum.reshape(pred_ds.padded_shape+(n_classes,))
            pred_count = pred_count.reshape(pred_ds.padded_shape+(n_classes,))

            pred_sum = pred_sum[patch_size:-patch_size, patch_size:-patch_size,:]
            pred_count = pred_count[patch_size:-patch_size, patch_size:-patch_size,:]

            pred_global_sum += pred_sum / pred_count

        pred_global = pred_global_sum / len(overlaps)
        pred_global_file = predicted_path / f'{prediction_prefix}_prob_{opt_i}_{sar_i}_{model_idx}.npy'
        np.save(pred_global_file, pred_global[:,:,1].astype(np.float16))

        #pred_b2 = (pred_global[:,:,1] > 0.5).astype(np.uint8)
        pred_b2 = (np.argmax(pred_global, -1)==1).astype(np.uint8)
        pred_b2[label == 2] = 2

        base_data = Path(paths_params['opt_data']) / original_opt_imgs['test'][0]
        prediction_tif_file = visual_path / f'{prediction_prefix}_{args.experiment}_{opt_i}_{sar_i}_{model_idx}.tif'
        save_geotiff(base_data, prediction_tif_file, pred_b2, dtype = 'byte')

        pred_results = {
            #'models_predicted': models_pred_idx,
            'opt_files': str(test_opt_img),
            'sar_files': str(test_sar_img)
        }
        save_yaml(pred_results, logs_path / f'pred_{opt_i}_{sar_i}.yaml')



if __name__=="__main__":
    freeze_support()
    

    if args.model == -1:
        if args.start_model == -1:
            models_pred_idx = np.arange(n_models)
        else:
            models_pred_idx = np.arange(args.start_model, n_models)
    else:
        models_pred_idx = [args.model]

    test_opt_imgs = original_opt_imgs['test']
    test_sar_imgs = original_sar_imgs['test']

    test_opt_imgs_idx = experiment_params['test_opt_imgs']
    test_sar_imgs_idx = experiment_params['test_sar_imgs']
    
    run_params = []

    for opt_i, opt_idx in enumerate(test_opt_imgs_idx):
        for sar_i, sar_idx in enumerate(test_sar_imgs_idx):
            opt_images_path = [opt_folder /  test_opt_imgs[i] for i in opt_idx]
            sar_images_path = [sar_folder /  test_sar_imgs[i] for i in sar_idx]
            run_params.append((models_pred_idx, opt_images_path, sar_images_path, opt_i, sar_i))

    t0 = time.perf_counter()
    for run_param in run_params:
        p = Process(target=run_prediction, args=run_param)
        p.start()
        p.join()
    total_time = time.perf_counter() - t0
    m_time = total_time / (len(run_params) * len(models_pred_idx))
    pred_results = {
        'models_predicted': models_pred_idx,
        'total_pred_time': total_time,
        'train_time_per_epoch': m_time
    }
    save_yaml(pred_results, logs_path / 'pred_results.yaml')




    

    