import argparse
import pathlib
#from conf import default, general, paths
#import os
from utils.ops import count_parameters, get_model
#from utils.dataloader import TrainDataSet
import torch
import logging, sys
from torch.multiprocessing import Process, freeze_support
import importlib
from torch.utils.data import DataLoader, RandomSampler
from torch import nn
from utils.trainer import train_loop, val_loop, EarlyStop
import time
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.tensorboard import SummaryWriter
from  pathlib import Path
import yaml
from utils.dataloader import TrainDataset, ValDataset

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
    help = 'The number of the model to be retrained'
)


args = parser.parse_args()

with open(args.cfg, 'r') as file:
    cfg = yaml.load(file, Loader=yaml.Loader)

training_params = cfg['training_params']
experiments_cfg_path = training_params['experiments_cfg_path']


cfg_exp_file = Path(experiments_cfg_path) / f'exp_{args.experiment}.yaml'
with open(cfg_exp_file, 'r') as file:
    exp_cfg = yaml.load(file, Loader=yaml.Loader)

cfg.update(exp_cfg)

path_exp = Path(paths['experiments']) / f'exp_{args.experiment}'
path_exp.mkdir(exist_ok=True)

path_logs = path_exp / paths['exp_subpaths']['logs']
path_models = path_exp / paths['exp_subpaths']['models']
path_visual = path_exp / paths['exp_subpaths']['visual']
path_predicted = path_exp / paths['exp_subpaths']['predicted']
path_results = path_exp / paths['exp_subpaths']['results']

path_logs.mkdir(exist_ok=True)
path_models.mkdir(exist_ok=True)
path_visual.mkdir(exist_ok=True)
path_predicted.mkdir(exist_ok=True)
path_results.mkdir(exist_ok=True)

path_prepared = Path(cfg['paths']['prepared']['main'])
path_train = path_prepared / cfg['paths']['prepared']['train']
path_val = path_prepared / cfg['paths']['prepared']['val']
path_test = path_prepared / cfg['paths']['prepared']['test']
path_general = path_prepared / cfg['paths']['prepared']['general']

patch_size = general_params['patch_size']


def run(model_idx):
    outfile = path_logs /  f'train_{args.experiment}_{model_idx}.txt'
    logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s:%(levelname)s:%(name)s:%(message)s',
            filename=outfile,
            filemode='w'
            )
    log = logging.getLogger('training')

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = get_model(cfg, model_cfg)

    log.info(cfg)

    log.info('Loading data...')

    #train_dp = get_datapipe(path_train, subset='train', patch_size=patch_size, device = device)
    #val_dp = get_datapipe(path_train, subset='validation', patch_size=patch_size, device = device)
    train_ds = TrainDataset(path_train/paths['prepared']['train_csv'], patch_size=patch_size,  device = device, params=general_params, n_patches_tile=train_params['n_patches_tile'])
    val_ds = ValDataset(path_val/paths['prepared']['val_csv'], patch_size=patch_size, device = device, params=general_params, n_patches_tile=train_params['n_patches_tile'])
    
    train_sampler = RandomSampler(train_ds)

    train_dl = DataLoader(dataset=train_ds, batch_size=train_params['batch_size'], num_workers=5, persistent_workers = True, drop_last = True, sampler = train_sampler)
    val_dl = DataLoader(dataset=val_ds, batch_size=train_params['batch_size'], num_workers=5, persistent_workers = True)
    
    log.info('Data loaded.')

    model.to(device)
    log.info(f'Model trainable parameters: {count_parameters(model)}')

    loss_fn = nn.CrossEntropyLoss(ignore_index=2, weight=torch.tensor(train_params['classes_weights']).to(device))

    optimizer = torch.optim.Adam(model.parameters(), lr=train_params['learning_rate'])
    path_model = path_models / f'model_{model_idx}.pth'
    early_stop = EarlyStop(
        train_patience=train_params['early_stop_patience'],
        path_to_save = str(path_model),
        min_delta = train_params['early_stop_min_delta'],
        min_epochs = train_params['early_stop_min_epochs']
        )

    total_t0 = time.perf_counter()

    path_tb_train_logdir = path_logs / f'train_model_{model_idx}'
    path_tb_val_logdir = path_logs / f'val_model_{model_idx}'

    train_writer = SummaryWriter(log_dir=str(path_tb_train_logdir))
    val_writer = SummaryWriter(log_dir=str(path_tb_val_logdir))
    for t in range(train_params['max_training_epochs']):
        epoch = t+1
        print(f"-------------------------------\nEpoch {epoch}")
        model.train()
        loss, f1 = train_loop(train_dl, model, loss_fn, optimizer)
        train_writer.add_scalar('Loss', loss, t)
        train_writer.add_scalar('F1Score', f1, t)
        model.eval()
        val_loss, val_f1 = val_loop(val_dl, model, loss_fn)
        val_writer.add_scalar('Loss', val_loss, t)
        val_writer.add_scalar('F1Score', val_f1, t)

        if early_stop.testEpoch(model = model, val_value = val_loss):
            min_val = early_stop.better_value
            log.info(f'Min Validation Value:{min_val}')
            break
    t_time = (time.perf_counter() - total_t0)/60
    log.info(f'Total Training time: {t_time} mins, for {t} epochs, Avg Training Time per epoch:{t_time/t}')

if __name__=="__main__":
    freeze_support()
    
    if args.model == -1:
        for model_idx in range(general_params['n_models']):
            p = Process(target=run, args=(model_idx,))
            p.start()
            p.join()
    else:
        run(args.model)

