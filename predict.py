import argparse
from pathlib import Path
import importlib
#from conf import general, paths
import os
import time
from utils.dataloader import PredDataSet
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import numpy as np
from utils.ops import save_geotiff, load_json, get_model
import logging
import yaml
from multiprocessing import Process

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

args = parser.parse_args()

with open(args.cfg, 'r') as file:
    cfg = yaml.safe_load(file)

cfg_exp_file = Path('conf') / f'exp_{args.experiment}.yaml'
with open(cfg_exp_file, 'r') as file:
    model_cfg = yaml.safe_load(file)

params = cfg['params']
paths = cfg['paths']

path_exp = Path(paths['experiments']) / f'exp_{args.experiment}'
path_exp.mkdir(exist_ok=True)

path_logs = path_exp / paths['exp_subpaths']['logs']
path_models = path_exp / paths['exp_subpaths']['models']
path_visual = path_exp / paths['exp_subpaths']['visual']
path_predicted = path_exp / paths['exp_subpaths']['predicted']
path_results = path_exp / paths['exp_subpaths']['results']
path_prepared = Path(paths['prepared']['main'])

patch_size = params['patch_size']
n_classes = params['n_classes']
opt_images = paths['opt']
sar_images = paths['sar']
n_images = params['n_images']
prediction_remove_border = params['prediction_remove_border']
prediction_prefix = params['prediction_prefix']
n_models = params['n_models']
batch_size = params['batch_size']


device = "cuda" if torch.cuda.is_available() else "cpu"
#print(f"Using {device} device")

#def run(model_idx):
#if __name__ == '__main__':
#    freeze_support()
#    for model_idx in tqdm(range(args.number_models), desc='Model Idx'):
    
def run(model_idx):

    outfile = path_logs / f'pred_{args.experiment}_{model_idx}.txt'
    logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s:%(levelname)s:%(name)s:%(message)s',
            filename=outfile,
            filemode='w'
            )
    log = logging.getLogger('predict')

    statistics_file = path_prepared / cfg['files']['statistics']
    stats = load_json(statistics_file)

    model = get_model(cfg, model_cfg)
    model.to(device)
    log.info(model)

    model_path = os.path.join(path_models, f'model_{model_idx}.pth')
    model.load_state_dict(torch.load(model_path))

    torch.set_num_threads(8)

    overlaps = params['prediction_overlaps']
    log.info(f'Overlaps pred: {overlaps}')
    one_window = np.ones((patch_size, patch_size, n_classes))
    total_time = 0
    n_processed_images = 0

    for im_0 in tqdm(range(n_images), leave=False, desc='Img 1'):
        for im_1 in tqdm(range(n_images), leave = False, desc='Img 2'):
            images = {
                'opt_0': opt_images['y1'][im_0],
                'sar_0': sar_images['y1'][im_0],
                'opt_1': opt_images['y2'][im_1],
                'sar_1': sar_images['y2'][im_1],
                'previous': paths['previous']['y2'],
                'label': paths['labels']['y2']
            }
            dataset = PredDataSet(images = images, device = device, patch_size= patch_size, stats = stats)
            label = dataset.label
            log.info(f'Optical Image Year 1:{images["opt_0"]}')
            log.info(f'Optical Image Year 2:{images["opt_1"]}')
            log.info(f'SAR Image Year 1:{images["sar_0"]}')
            log.info(f'SAR Image Year 2:{images["sar_1"]}')
            #print(f'CMAP Image Year 0:{dataset.cmap_file_0}')
            #print(f'CMAP Image Year 1:{dataset.cmap_file_1}')
            log.info(f'Prev Def Image Year 1:{images["previous"]}')
            pred_global_sum = np.zeros(dataset.original_shape+(n_classes,))
            t0 = time.perf_counter()
            for overlap in tqdm(overlaps, leave=False, desc='Overlap'):
                log.info(f'Predicting overlap {overlap}')
                dataset.gen_patches(overlap = overlap)
                dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
                
                pbar = tqdm(dataloader, desc='Prediction', leave = False)
                #preds = None
                preds = torch.zeros((len(dataset), n_classes, patch_size, patch_size))
                for i, X in enumerate(pbar):
                    with torch.no_grad():
                        preds[batch_size*i: batch_size*(i+1)] =  model(X).to('cpu')
                preds = np.moveaxis(preds.numpy().astype(np.float16), 1, -1)
                pred_sum = np.zeros(dataset.padded_shape+(n_classes,)).reshape((-1, n_classes))
                pred_count = np.zeros(dataset.padded_shape+(n_classes,)).reshape((-1, n_classes))
                for idx, idx_patch in enumerate(tqdm(dataset.idx_patches, desc = 'Rebuild', leave = False)):
                    crop_val = prediction_remove_border
                    idx_patch_crop = idx_patch[crop_val:-crop_val, crop_val:-crop_val]
                    pred_sum[idx_patch_crop] += preds[idx][crop_val:-crop_val, crop_val:-crop_val]
                    pred_count[idx_patch_crop] += one_window[crop_val:-crop_val, crop_val:-crop_val]

                    #pred_sum[idx_patch] += preds[idx]
                    #pred_count[idx_patch] += one_window
                pred_sum = pred_sum.reshape(dataset.padded_shape+(n_classes,))
                pred_count = pred_count.reshape(dataset.padded_shape+(n_classes,))

                pred_sum = pred_sum[patch_size:-patch_size, patch_size:-patch_size,:]
                pred_count = pred_count[patch_size:-patch_size, patch_size:-patch_size,:]

                pred_global_sum += pred_sum / pred_count

            p_time = (time.perf_counter() - t0)/60
            total_time += p_time
            n_processed_images += 1
            log.info(f'Prediction time: {p_time} mins')
            pred_global = pred_global_sum / len(overlaps)
            #pred_b = pred_global.argmax(axis=-1).astype(np.uint8)

            #pred_b[label == 2] = 2

            #np.save(os.path.join(predicted_path, f'{general.PREDICTION_PREFIX}_{img_pair[0]}_{img_pair[1]}_{model_idx}.npy'), pred_b)
            np.save(os.path.join(path_predicted, f'{prediction_prefix}_prob_{im_0}_{im_1}_{model_idx}.npy'), pred_global[:,:,1].astype(np.float16))

            #save_geotiff(str(args.base_image), os.path.join(visual_path, f'{general.PREDICTION_PREFIX}_{args.experiment}_{img_pair[0]}_{img_pair[1]}_{model_idx}.tif'), pred_b, dtype = 'byte')

            pred_b2 = (pred_global[:,:,1] > 0.5).astype(np.uint8)
            pred_b2[label == 2] = 2

            base_data = opt_images['y0'][0]
            save_geotiff(base_data, os.path.join(path_visual, f'{prediction_prefix}_{args.experiment}_{im_0}_{im_1}_{model_idx}.tif'), pred_b2, dtype = 'byte')
            #save_geotiff(str(args.base_image), os.path.join(visual_path, f'{general.PREDICTION_PREFIX}_probs_{args.experiment}_{img_pair[0]}_{img_pair[1]}_{model_idx}.tif'), pred_global, dtype = 'float')
    m_time = total_time / n_processed_images
    log.info(f'Mean Prediction time: {m_time} mins')


if __name__=="__main__":
    
    for model_idx in range(n_models):
        print(f'Predicting model {model_idx}')
        p = Process(target=run, args=(model_idx,))
        p.start()
        p.join()
  