import argparse
from utils.ops import count_parameters, save_yaml, load_yaml
import torch
from torch.multiprocessing import Process, freeze_support
from torch.utils.data import DataLoader, RandomSampler
import time
from  pathlib import Path
import yaml
from utils.datasets import TrainDataset, ValDataset#, to_gpu
from pydoc import locate
import lightning.pytorch as pl
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers.csv_logs import CSVLogger
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger

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
    default = 11,
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

parser.add_argument( # Accelerator
    '-d', '--devices',
    type = int,
    nargs='+',
    default = [0],
    help = 'Accelerator devices to be used'
)

parser.add_argument( # Log in neptune
    '-n', '--neptune-log',
    action='store_true',
    help = 'Log in neptune'
)


args = parser.parse_args()

with open(args.cfg, 'r') as file:
    cfg = yaml.load(file, Loader=yaml.Loader)

training_params = cfg['training_params']
preparation_params = cfg['preparation_params']
experiment_params = cfg['experiments'][f'exp_{args.experiment}']
original_data_params = cfg['original_data']

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
n_classes = training_params['n_classes']

def run(model_idx):
    last_val_loss = float('inf')
    
    while True:
        torch_seed = int(1000*time.time())
        torch.manual_seed(torch_seed)
        torch.set_float32_matmul_precision('high')

        model = locate(experiment_params['model'])(experiment_params, training_params)

        tb_logger = TensorBoardLogger(
            save_dir = logs_path,
            name = f'model_{model_idx}',
            version = ''
        )

        loggers = [tb_logger]

        log_cfg = Path('loggers.yaml')
        if log_cfg.exists() and args.neptune_log:
            log_cfg = load_yaml(log_cfg)
            if 'neptune' in log_cfg.keys():
                neptune_cfg = log_cfg['neptune']
                neptune_logger = locate(neptune_cfg['module'])(
                    project = neptune_cfg['project'],
                    api_key = neptune_cfg['api_key']
                )
                params = {
                    'experiment': args.experiment,
                    'model_idx': model_idx,
                    'model': experiment_params['model']
                }
                neptune_logger.log_hyperparams(params=params)
                loggers.append(neptune_logger)


        train_ds = TrainDataset(experiment_params, train_folder, prepared_patches['train'])
        val_ds = ValDataset(experiment_params, val_folder, prepared_patches['val'])
        
        train_dl = DataLoader(
            dataset=train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=8,
            persistent_workers=True
        )

        val_dl = DataLoader(
            dataset=val_ds,
            batch_size=batch_size,
            num_workers=8,
            persistent_workers=True
        )

        early_stop_callback = EarlyStopping(monitor="val_loss", verbose = True, mode="min", **training_params['early_stop'])
        monitor_checkpoint_callback = ModelCheckpoint(
            str(models_path), 
            monitor = 'val_loss', 
            verbose = True, 
            filename = f'model_{model_idx}'
            )
        trainer = pl.Trainer(
            accelerator  = 'gpu',
            devices = args.devices,
            limit_train_batches = training_params['max_train_batches'], 
            limit_val_batches = training_params['max_val_batches'], 
            max_epochs = training_params['max_epochs'], 
            callbacks = [early_stop_callback, monitor_checkpoint_callback], 
            logger = loggers,
            log_every_n_steps = 1,
            #num_sanity_val_steps = 0
            )
        
        t0 = time.time()
        trainer.fit(model = model, train_dataloaders=train_dl, val_dataloaders=val_dl) #, datamodule=data_module)
        train_time = time.time() - t0
        
        last_val_loss = monitor_checkpoint_callback.best_model_score.item()

        if last_val_loss <= min_val_loss:
            run_results = {
                'model_path': monitor_checkpoint_callback.best_model_path,
                'total_train_time': train_time,
                'train_per_epoch': train_time / trainer.current_epoch,
                'n_paramters': count_parameters(model), 
                'converged': True
            }
            save_yaml(run_results, logs_path / f'model_{model_idx}' / 'train_results.yaml')
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


    