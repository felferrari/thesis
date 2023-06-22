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
from utils.dataloader import TrainDataset, ValDataset#, to_gpu
from pydoc import locate
import lightning.pytorch as pl
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger
import mlflow

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

training_params = cfg['training_params']
preparation_params = cfg['preparation_params']
experiment_params = cfg['experiments'][f'exp_{args.experiment}']
original_data_params = cfg['original_data']
neptune_params = cfg['neptune_params']

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

#train_csv = prepared_folder / preparation_params['train_data']
#test_csv = prepared_folder / preparation_params['test_data']
#validation_csv = prepared_folder / preparation_params['validation_data']
train_folder = prepared_folder / preparation_params['train_folder']
val_folder = prepared_folder / preparation_params['validation_folder']
prepared_patches_file = prepared_folder / preparation_params['prepared_data']
prepared_patches = load_yaml(prepared_patches_file)


patch_size = training_params['patch_size']
batch_size = training_params['batch_size']
min_val_loss = training_params['min_val_loss']

neptune_project_name = neptune_params['name']

#runs_file = Path(experiments_paths['folder']) / 'runs.yaml'
#if runs_file.exists():
#    runs_dict = load_yaml(runs_file)
#else:
#    runs_dict = {}

def run(model_idx):
    last_val_loss = float('inf')
    
    while last_val_loss >= min_val_loss:

        device = "cuda" if torch.cuda.is_available() else "cpu"

        model_train = locate(experiment_params['model'])(experiment_params)

        model = ModelTrainer(
            model_train, 
            torch.tensor(training_params['loss_fn']['weights'])#.to(device) 
            )#.to(device)

        train_ds = TrainDataset(device, experiment_params, train_folder, prepared_patches['train'])
        val_ds = ValDataset(device, experiment_params, val_folder, prepared_patches['val'])
        
        train_sampler = RandomSampler(train_ds)
        val_sampler = RandomSampler(val_ds)

        train_dl = DataLoader(
            dataset=train_ds, 
            batch_size=batch_size, 
            num_workers=5, 
            persistent_workers = True, 
            drop_last = True, 
            sampler = train_sampler
            )
        val_dl = DataLoader(
            dataset=val_ds, 
            batch_size=batch_size, 
            num_workers=5, 
            persistent_workers = True, 
            drop_last = True, 
            #sampler = val_sampler
            )
        
        
        mlf_logger = MLFlowLogger(experiment_name=f'exp_{args.experiment}')
        early_stop_callback = EarlyStopping(monitor="val_loss", verbose = True, mode="min", **training_params['early_stop'])
        monitor_checkpoint_callback = ModelCheckpoint(str(models_path), monitor = 'val_loss', verbose = True, filename = f'model_{model_idx}')
        trainer = pl.Trainer(
            accelerator  = 'gpu',
            limit_train_batches=training_params['max_train_batches'], 
            limit_val_batches=training_params['max_val_batches'], 
            max_epochs=training_params['max_epochs'], 
            callbacks=[early_stop_callback, monitor_checkpoint_callback], 
            logger=mlf_logger,
            log_every_n_steps = 1,
            )
        
        with mlflow.start_run(run_name  = f'run_{model_idx}'):
            trainer.fit(model=model, train_dataloaders=train_dl, val_dataloaders=val_dl)


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


    