import argparse
import pathlib
#from conf import default, general, paths
#import os
from utils.ops import count_parameters, save_yaml, load_yaml
#from utils.dataloader import TrainDataSet
import torch
import logging, sys
from torch.multiprocessing import Process, freeze_support
from torch.utils.data import DataLoader, RandomSampler
from utils.trainer import EarlyStop, train_model, ModelTrainer
import time
from  pathlib import Path
import yaml
from utils.datasets import TrainDataset, ValDataset#, to_gpu
from pydoc import locate
import lightning.pytorch as pl
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
import comet_ml
from pytorch_lightning.loggers import CometLogger
from lightning.pytorch.core import LightningDataModule

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
    default = 3,
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

torch.set_float32_matmul_precision('highest')

args = parser.parse_args()

with open(args.cfg, 'r') as file:
    cfg = yaml.load(file, Loader=yaml.Loader)

training_params = cfg['training_params']
preparation_params = cfg['preparation_params']
experiment_params = cfg['experiments'][f'exp_{args.experiment}']
original_data_params = cfg['original_data']
comet_params = cfg['comet_params']

experiments_paths = training_params['experiments_paths']

#create experiment folder structure
exp_path = Path(experiments_paths['folder']) / f'exp_{args.experiment}'

models_path = exp_path / experiments_paths['models']
logs_path = exp_path / experiments_paths['logs']
visual_path = exp_path / experiments_paths['visual']
predicted_path = exp_path / experiments_paths['predicted']
results_path = exp_path / experiments_paths['results']
visual_logs_path = exp_path / experiments_paths['visual_logs']

exp_path.mkdir(exist_ok=True)
models_path.mkdir(exist_ok=True)
logs_path.mkdir(exist_ok=True)
visual_path.mkdir(exist_ok=True)
predicted_path.mkdir(exist_ok=True)
results_path.mkdir(exist_ok=True)
visual_logs_path.mkdir(exist_ok=True)

#setting up prepared data source
prepared_folder = Path(preparation_params['folder'])

train_folder = prepared_folder / preparation_params['train_folder']
val_folder = prepared_folder / preparation_params['validation_folder']
prepared_patches_file = prepared_folder / preparation_params['prepared_data']
prepared_patches = load_yaml(prepared_patches_file)


patch_size = training_params['patch_size']
batch_size = training_params['batch_size']
min_val_loss = training_params['min_val_loss']

workspace_name = comet_params['workspace_name']
project_name = comet_params['project_name']

def run(model_idx):
    last_val_loss = float('inf')
    
    while True:

        model = locate(experiment_params['model'])(experiment_params, training_params)
        comet_api = locate('keys.comet_ml')

        train_ds = TrainDataset(experiment_params, train_folder, prepared_patches['train'])
        val_ds = ValDataset(experiment_params, val_folder, prepared_patches['val'])
        
        data_module = LightningDataModule.from_datasets(
            train_dataset=train_ds,
            val_dataset=val_ds,
            batch_size=batch_size,
            num_workers=4
        )

        comet_api = locate('keys.comet_ml')
        api = comet_ml.API(api_key=comet_api)
        exp = api.get(workspace=workspace_name, project_name=project_name, experiment=f'exp_{args.experiment}_run_{model_idx}')
        if exp is not None:
            exp_key = exp.get_metadata()['experimentKey']
            api.archive_experiment(exp_key)

        comet_logger = CometLogger(
            api_key= comet_api,
            project_name = project_name,
            experiment_name = f'exp_{args.experiment}_run_{model_idx}'
        )
        comet_logger.experiment.log_parameter('model_name', experiment_params['model'])
        early_stop_callback = EarlyStopping(monitor="val_loss", verbose = True, mode="min", **training_params['early_stop'])
        monitor_checkpoint_callback = ModelCheckpoint(
            str(models_path), 
            monitor = 'val_loss', 
            verbose = True, 
            filename = f'model_{model_idx}'
            )
        trainer = pl.Trainer(
            accelerator  = 'gpu',
            limit_train_batches = training_params['max_train_batches'], 
            limit_val_batches = training_params['max_val_batches'], 
            max_epochs = 2, #training_params['max_epochs'], 
            callbacks = [early_stop_callback, monitor_checkpoint_callback], 
            logger = comet_logger,
            log_every_n_steps = 1,
            )
        
        t0 = time.time()
        trainer.fit(model = model, datamodule=data_module)
        train_time = time.time() - t0
        
        last_val_loss = monitor_checkpoint_callback.best_model_score.item()

        if last_val_loss <= min_val_loss:
            comet_logger.experiment.log_parameter(f'model_file_path', monitor_checkpoint_callback.best_model_path)
            comet_logger.experiment.log_parameter(f'train_time', train_time)
            comet_logger.experiment.add_tag(f'trained')
            #comet_logger.experiment.log_model(f'exp_{args.experiment}-{model_idx}', monitor_checkpoint_callback.best_model_path)
            break
        else:
            print('Model didn\'t converged. Repeating the training...')
            model_file = Path(monitor_checkpoint_callback.best_model_path)
            model_file.unlink()

if __name__=="__main__":
    freeze_support()
    
    if args.model == -1:
        if args.start_model == -1:
            for model_idx in range(training_params['n_models']):
                p = Process(target=run, args=(model_idx,))
                p.start()
                p.join()
        else:
            for model_idx in range(args.start_model, training_params['n_models']):
                p = Process(target=run, args=(model_idx,))
                p.start()
                p.join()
    
    else:
        run(args.model)


    