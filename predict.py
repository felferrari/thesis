import argparse
from pathlib import Path
import time
from utils.datasets import PredDataset, ImageWriter
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import numpy as np
from utils.ops import save_geotiff, load_yaml
import yaml
from multiprocessing import Process, freeze_support
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
    default = 1,
    help = 'The number of the experiment'
)

parser.add_argument( # Model number
    '-m', '--model',
    type = int,
    default = -1,
    help = 'Number of the model to be retrained'
)

parser.add_argument( # Model number
    '-s', '--start-model',
    type = int,
    default = -1,
    help = 'Number of the model to be retrained'
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
training_params = cfg['training_params']


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
opt_prefix = preparation_params['prefixs']['opt']
sar_prefix = preparation_params['prefixs']['sar']
label_prefix = preparation_params['prefixs']['label']
previous_prefix = preparation_params['prefixs']['previous']



device = "cuda" if torch.cuda.is_available() else "cpu"

def run(model_idx):

    print(f'\nPredicting Model {model_idx}...')

    run_results = load_yaml(logs_path / f'model_{model_idx}' / 'train_results.yaml')
    model_class = locate(experiment_params['model'])#(experiment_params, training_params)
    model = model_class.load_from_checkpoint(run_results['model_path'])

    images_combinations = []
    for opt_group in experiment_params['test_opt_imgs']:
        for sar_group in experiment_params['test_sar_imgs']:
            images_combinations.append([opt_group, sar_group])

    prev_img_file = test_folder / f'{previous_prefix}.h5'
    #label_img_file = test_folder / f'{label_prefix}.h5'
    for img_group in images_combinations:
        opt_img_idxs, sar_img_idxs =  img_group
        opt_img_files = [test_folder / f'{opt_prefix}_{img_idx}.h5' for img_idx in opt_img_idxs]
        sar_img_files = [test_folder / f'{sar_prefix}_{img_idx}.h5' for img_idx in sar_img_idxs]

        pred_ds = PredDataset(
            opt_img_files,
            sar_img_files,
            prev_img_file,
            patch_size)
        
        pred_dl = DataLoader(
            dataset=pred_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=5
        )

        img_writer_callback = ImageWriter(
            pred_ds.get_original_size(),
            patch_size,
            n_classes
        )

        trainer = pl.Trainer(
            accelerator  = 'gpu',
            callbacks = [img_writer_callback], 
            )
        
        trainer.predict(
            model = model,
            dataloaders = pred_dl
        )

                

if __name__=="__main__":
    freeze_support()
    

    if args.model == -1:
        if args.start_model == -1:
            for model_idx in range(prediction_params['n_models']):
                p = Process(target=run, args=(model_idx,))
                p.start()
                p.join()
        else:
            for model_idx in range(args.start_model, prediction_params['n_models']):
                p = Process(target=run, args=(model_idx,))
                p.start()
                p.join()
    
    else:
        run(args.model)    