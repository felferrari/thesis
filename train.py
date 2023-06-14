import argparse
import pathlib
#from conf import default, general, paths
#import os
from utils.ops import count_parameters, save_yaml, load_yaml
#from utils.dataloader import TrainDataSet
import torch
import logging, sys
from torch.multiprocessing import Process, freeze_support
import importlib
from torch.utils.data import DataLoader, RandomSampler
from torch import nn
from utils.trainer import train_loop, val_loop, EarlyStop, sample_figures_loop
import time
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.tensorboard import SummaryWriter
from  pathlib import Path
import yaml
from utils.dataloader import TrainDataset, ValDataset#, to_gpu
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
    default = 2,
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
max_train_patches = training_params['max_train_patches']
max_val_patches = training_params['max_val_patches']
min_val_loss = training_params['min_val_loss']

def run(model_idx):
    last_val_loss = float('inf')
    while last_val_loss >= min_val_loss:
        outfile = logs_path /  f'train_{args.experiment}_{model_idx}.txt'
        logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s:%(levelname)s:%(name)s:%(message)s',
                filename=outfile,
                filemode='w'
                )
        log = logging.getLogger('training')

        cfg_log_file = logs_path / f'cfg_{args.experiment}_{model_idx}.yaml'
        save_yaml(cfg, cfg_log_file)

        device = "cuda" if torch.cuda.is_available() else "cpu"

        model = locate(experiment_params['model'])(experiment_params)
        model.to(device)
        #a = [[torch.rand((32, 13, 224, 224)).to(device), torch.rand((32, 13, 224, 224)).to(device)], [torch.rand((32, 2, 224, 224)).to(device), torch.rand((32, 2, 224, 224)).to(device), torch.rand((32, 2, 224, 224)).to(device), torch.rand((32, 2, 224, 224)).to(device), torch.rand((32, 2, 224, 224)).to(device), torch.rand((32, 2, 224, 224)).to(device), torch.rand((32, 2, 224, 224)).to(device), torch.rand((32, 2, 224, 224)).to(device), torch.rand((32, 2, 224, 224)).to(device), torch.rand((32, 2, 224, 224)).to(device), torch.rand((32, 2, 224, 224)).to(device), torch.rand((32, 2, 224, 224)).to(device)], torch.rand((32, 1, 224, 224)).to(device) ]
        #model(a)
        log.info(f'Model trainable parameters: {count_parameters(model)}')


        train_ds = TrainDataset(patch_size, device, experiment_params, train_folder, prepared_patches['train'])
        val_ds = ValDataset(patch_size, device, experiment_params, val_folder, prepared_patches['val'])
        samples_ds = ValDataset(patch_size, device, experiment_params, val_folder, prepared_patches['val'])
        train_ds[1000]
        #val_ds[10]
        
        train_sampler = RandomSampler(train_ds, num_samples=max_train_patches)
        val_sampler = RandomSampler(val_ds, num_samples=max_val_patches)
        samples_sampler = RandomSampler(val_ds, num_samples=max_val_patches)

        train_dl = DataLoader(
            dataset=train_ds, 
            batch_size=batch_size, 
            num_workers=4, 
            persistent_workers = True, 
            drop_last = True, 
            sampler = train_sampler#,
            #collate_fn = to_gpu
            )
        val_dl = DataLoader(
            dataset=val_ds, 
            batch_size=batch_size, 
            num_workers=4, 
            persistent_workers = True, 
            drop_last = True, 
            sampler = val_sampler#,
            #collate_fn = to_gpu
            )
        sampler_dl = DataLoader(
            dataset=samples_ds, 
            batch_size=batch_size
            )#, sampler = samples_sampler)

        #train_dl = DataLoader(dataset=train_ds, batch_size=batch_size, sampler = train_sampler, drop_last = True)
        #val_dl = DataLoader(dataset=val_ds, batch_size=batch_size)

        loss_params = training_params['loss_fn']
        loss_fn = locate(loss_params['module'])(weight=torch.tensor(loss_params['weights']).to(device),  ignore_index = loss_params['ignore_index'])

        optimizer_params = training_params['optimizer']
        optimizer = locate(optimizer_params['module'])(model.parameters(),  **optimizer_params['params'])

        optimizer_params = training_params['scheduler']
        scheduler = locate(optimizer_params['module'])(optimizer, **optimizer_params['params'])

        model_path = models_path / f'model_{model_idx}.pth'

        early_stop_params = training_params['early_stop']
        early_stop = EarlyStop(
            train_patience = early_stop_params['patience'],
            path_to_save = str(model_path),
            min_delta = early_stop_params['min_delta'],
            min_epochs = early_stop_params['min_epochs']
            )
        
        total_t0 = time.perf_counter()

        path_tb_train_logdir = logs_path / f'train_model_{model_idx}'
        path_tb_val_logdir = logs_path / f'val_model_{model_idx}'

        train_writer = SummaryWriter(log_dir=str(path_tb_train_logdir))
        val_writer = SummaryWriter(log_dir=str(path_tb_val_logdir))

        for t in range(training_params['max_epochs']):
            epoch = t+1
            print(f"-------------------------------\nEpoch {epoch}")
            model.train()
            loss, f1_0, f1_1 = train_loop(train_dl, model, loss_fn, optimizer, training_params)
            train_writer.add_scalar('Loss', loss, t)
            train_writer.add_scalar('Class0_F1', f1_0, t)
            train_writer.add_scalar('Class1_F1', f1_1, t)
            model.eval()
            val_loss, f1_0, f1_1 = val_loop(val_dl, model, loss_fn, training_params)
            val_writer.add_scalar('Loss', val_loss, t)
            val_writer.add_scalar('Class0_F1', f1_0, t)
            val_writer.add_scalar('Class1_F1', f1_1, t)

            #sample_figures_loop(sampler_dl, model, 16, epoch, visual_logs_path, model_idx)

            if early_stop.testEpoch(model = model, val_value = val_loss):
                min_val = early_stop.better_value
                log.info(f'Min Validation Value:{min_val}')
                break

            scheduler.step()
        t_time = (time.perf_counter() - total_t0)/60
        log.info(f'Total Training time: {t_time} mins, for {t} epochs, Avg Training Time per epoch:{t_time/t}')

        last_val_loss = min_val

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


    