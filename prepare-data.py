import argparse
from  pathlib import Path
import shutil
import yaml
import logging
from utils.ops import load_opt_image, load_SAR_image, load_sb_image, load_json, save_json, save_geotiff
import numpy as np
from skimage.util import view_as_windows
from tqdm import tqdm
import pandas as pd
import csv

parser = argparse.ArgumentParser(
    description='prepare the original files, generating .npy files to be used in the training/testing steps'
)

parser.add_argument( # The path to the config file (.yaml)
    '-c', '--cfg',
    type = Path,
    default = 'cfg.yaml',
    help = 'Path to the config file (.yaml).'
)

parser.add_argument( # Clear the prepared folder before prepare new data
    '-x', '--clear-prepare',
    type = bool,
    default = False,
    help = 'Clear the prepared folder before prepare new data.'
)

parser.add_argument( # Create test data.
    '-t', '--test-data',
    type = bool,
    default = False,
    help = 'Create test data.'
)

parser.add_argument( # Generate statistics.
    '-s', '--gen-statistics',
    type = bool,
    default = False,
    help = 'Generate statistics.'
)


args = parser.parse_args()

with open(args.cfg, 'r') as file:
    cfg = yaml.safe_load(file)

paths = cfg['paths']

path_prepared = Path(paths['prepared']['main'])
path_train = path_prepared / paths['prepared']['train']
path_val = path_prepared / paths['prepared']['val']
path_test = path_prepared / paths['prepared']['test']
path_general = path_prepared / paths['prepared']['general']
path_train_data = path_train / paths['prepared']['data']
path_train_label = path_train / paths['prepared']['label']
path_val_data = path_val / paths['prepared']['data']
path_val_label = path_val / paths['prepared']['label']

if args.clear_prepare:
    shutil.rmtree(path_prepared)

path_prepared.mkdir(exist_ok=True)
path_train.mkdir(exist_ok=True)
path_val.mkdir(exist_ok=True)
path_test.mkdir(exist_ok=True)
path_general.mkdir(exist_ok=True)
path_train_data.mkdir(exist_ok=True)
path_train_label.mkdir(exist_ok=True)
path_val_data.mkdir(exist_ok=True)
path_val_label.mkdir(exist_ok=True)

opt_files = {}
opt_files['y0'] = paths['opt']['y0']
opt_files['y1'] = paths['opt']['y1']
opt_files['y2'] = paths['opt']['y2']

sar_files = {}
sar_files['y0'] = paths['sar']['y0']
sar_files['y1'] = paths['sar']['y1']
sar_files['y2'] = paths['sar']['y2']

label_train_file = Path(paths['labels']['y1'])
label_test_file = Path(paths['labels']['y2'])

previous_train_file = Path(paths['previous']['y1'])
previous_test_file = Path(paths['previous']['y2'])

for _, o_y in opt_files.items():
    for f in o_y:
        assert Path(f).exists(), f'{f} not found. Fix config file paths.'

for _, s_y in sar_files.items():
    for f in s_y:
        assert Path(f).exists(), f'{f} not found. Fix config file paths.'

assert label_train_file.exists(), f'{label_train_file} not found. Fix config file paths.'
assert label_test_file.exists(), f'{label_test_file} not found. Fix config file paths.'
assert previous_train_file.exists(), f'{previous_train_file} not found. Fix config file paths.'
assert previous_test_file.exists(), f'{previous_test_file} not found. Fix config file paths.'

outfile = path_prepared / 'prepared.txt'
logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s:%(levelname)s:%(name)s:%(message)s',
        filename=outfile,
        filemode='w'
        )
log = logging.getLogger('prepare')

statistics_file = path_prepared / cfg['files']['statistics']

if args.gen_statistics:
    log.info('Generating statistics...')

    opt_means, opt_stds = [], []
    sar_means, sar_stds = [], []
    stats = {}

    for _, o_y in opt_files.items():
        for f in o_y:
            opt_file = Path(f)
            img = load_opt_image(opt_file)
            img[np.isnan(img)] = 0
            opt_means.append(img.mean(axis=(0,1)))
            opt_stds.append(img.std(axis=(0,1)))

    opt_mean = np.array(opt_means).mean(axis=0)
    opt_std = np.array(opt_stds).mean(axis=0)

    log.info(f'Optical means: {opt_mean}')
    log.info(f'Optical stds: {opt_std}')

    stats['opt_mean'] = opt_mean.tolist()
    stats['opt_std'] = opt_std.tolist()

    for _, s_y in sar_files.items():
        for f in s_y:
            sar_file = Path(f)
            img = load_SAR_image(sar_file)
            img[np.isnan(img)] = 0
            sar_means.append(img.mean(axis=(0,1)))
            sar_stds.append(img.std(axis=(0,1)))

    sar_mean = np.array(sar_means).mean(axis=0)
    sar_std = np.array(sar_stds).mean(axis=0)

    log.info(f'SAR means: {sar_mean}')
    log.info(f'SAR stds: {sar_std}')

    stats['sar_mean'] = sar_mean.tolist()
    stats['sar_std'] = sar_std.tolist()

    save_json(stats, statistics_file)

    log.info('Statistics generated.')

else:
    assert statistics_file.exists(), f'Statistics file {statistics_file} not found.'
    stats = load_json(statistics_file)
    opt_mean = np.array(stats['opt_mean'])
    opt_std = np.array(stats['opt_std'])
    sar_mean = np.array(stats['sar_mean'])
    sar_std = np.array(stats['sar_std'])

log.info('Generating training tiles...')
previous = load_sb_image(previous_train_file)
label = load_sb_image(label_train_file)
tile_size = cfg['params']['tile_size']
shape = label.shape
pad_0 = (tile_size - (shape[0] % tile_size))% tile_size
pad_1 = (tile_size - (shape[1] % tile_size))% tile_size
pad_width = ((0, pad_0), (0, pad_1))
pad_img_width = ((0, pad_0), (0, pad_1), (0,0))
pad_mode = 'reflect'
label_padded = np.pad(label, pad_width, pad_mode)
padded_shape = label_padded.shape
label_padded = label_padded.flatten()
previous = np.pad(previous, pad_width, pad_mode).reshape((-1,1))
idx = np.arange(padded_shape[0] * padded_shape[1]).reshape(padded_shape)
idx_tiles = view_as_windows(idx, tile_size, tile_size).reshape((-1, tile_size, tile_size))
tile_min_def_prop = cfg['params']['tile_min_def_prop']
keep = ((label_padded[idx_tiles] == 1).sum(axis=(1,2)) / tile_size**2) >= tile_min_def_prop

keep_def_args = np.argwhere(keep).flatten()
n_def_keep_tiles = len(keep_def_args)
keep_no_def_args = np.argwhere(np.logical_not(keep)).flatten()
keep_no_def_args = np.random.choice(keep_no_def_args, n_def_keep_tiles)
keep_def_args = np.concatenate((keep_def_args, keep_no_def_args))
np.random.shuffle(keep_def_args)

val_prop = cfg['params']['tiles_val_prop']
n_val_tiles = int(val_prop*len(keep_def_args))

keep_val = keep_def_args[:n_val_tiles]
keep_train = keep_def_args[n_val_tiles:]

val_tiles = idx_tiles[keep_val]
train_tiles = idx_tiles[keep_train]

tiles_map = np.zeros_like(label_padded)
tiles_map[val_tiles] = 1
tiles_map[train_tiles] = 2
tiles_map = tiles_map.reshape(padded_shape)
tiles_map = tiles_map[:shape[0], :shape[1]]
save_geotiff(opt_files['y0'][0], path_general/'tiles.tif', tiles_map, 'byte')


label_padded = label_padded.reshape((-1,1))

n_images = cfg['params']['n_images']

train_dataset = {
    'data': [],
    'label': []
}
val_dataset = {
    'data': [],
    'label': []
}

for y0i in tqdm(range(n_images), desc = 'Year 0 data'):
    opt_file_0 = opt_files['y0'][y0i]
    sar_file_0 = sar_files['y0'][y0i]

    opt_file_0 = Path(opt_file_0)
    opt_0 = load_opt_image(opt_file_0)
    opt_0 = (opt_0 - opt_mean)/opt_std
    opt_0 = np.pad(opt_0, pad_img_width, pad_mode)
    opt_0 = opt_0.reshape((-1, opt_0.shape[-1]))

    sar_file_0 = Path(sar_file_0)
    sar_0 = load_SAR_image(sar_file_0)
    sar_0 = (sar_0 - sar_mean)/sar_std
    sar_0 = np.pad(sar_0, pad_img_width, pad_mode)
    sar_0 = sar_0.reshape((-1, sar_0.shape[-1]))

    for y1i in tqdm(range(n_images), desc = 'Year 1 data', leave=False):
        opt_file_1 = opt_files['y1'][y1i]
        sar_file_1 = sar_files['y1'][y1i]

        opt_file_1 = Path(opt_file_1)
        opt_1 = load_opt_image(opt_file_1)
        opt_1 = (opt_1 - opt_mean)/opt_std
        opt_1 = np.pad(opt_1, pad_img_width, pad_mode)
        opt_1 = opt_1.reshape((-1, opt_1.shape[-1]))

        sar_file_1 = Path(sar_file_1)
        sar_1 = load_SAR_image(sar_file_1)
        sar_1 = (sar_1 - sar_mean)/sar_std
        sar_1 = np.pad(sar_1, pad_img_width, pad_mode)
        sar_1 = sar_1.reshape((-1, sar_1.shape[-1]))

        for idx_i, idx_tile in enumerate(tqdm(train_tiles, desc='Training tiles', leave = False)):
            tile_opt_0 = opt_0[idx_tile]
            tile_opt_1 = opt_1[idx_tile]
            tile_sar_0 = sar_0[idx_tile]
            tile_sar_1 = sar_1[idx_tile]
            tile_prev = previous[idx_tile]
            tile_label = label_padded[idx_tile]

            data = np.concatenate((tile_opt_0, tile_opt_1, tile_sar_0, tile_sar_1, tile_prev), axis=-1)

            tile_data_file = path_train_data / f'{idx_i}-{y0i}-{y1i}.npy'
            tile_label_file = path_train_label / f'{idx_i}-{y0i}-{y1i}.npy'
            #assert not tile_file.exists(), 'File {tile_file} already exists.'
            np.save(tile_data_file, data.astype(np.float16))
            np.save(tile_label_file, tile_label.squeeze().astype(np.uint8))
            train_dataset['data'].append(tile_data_file)
            train_dataset['label'].append(tile_label_file)
            


        for idx_i, idx_tile in enumerate(tqdm(val_tiles, desc='Validation tiles', leave = False)):
            tile_opt_0 = opt_0[idx_tile]
            tile_opt_1 = opt_1[idx_tile]
            tile_sar_0 = sar_0[idx_tile]
            tile_sar_1 = sar_1[idx_tile]
            tile_prev = previous[idx_tile]
            tile_label = label_padded[idx_tile]
            data = np.concatenate((tile_opt_0, tile_opt_1, tile_sar_0, tile_sar_1, tile_prev), axis=-1)

            tile_data_file = path_val_data / f'{idx_i}-{y0i}-{y1i}.npy'
            tile_label_file = path_val_label / f'{idx_i}-{y0i}-{y1i}.npy'
            #assert not tile_file.exists(), 'File {tile_file} already exists.'
            np.save(tile_data_file, data.astype(np.float16))
            np.save(tile_label_file, tile_label.squeeze().astype(np.uint8))
            val_dataset['data'].append(tile_data_file)
            val_dataset['label'].append(tile_label_file)

        
log.info(f'{train_tiles.sum()} train tiles extracted.')
log.info(f'{val_tiles.sum()} validation tiles extracted.')

train_dataset = pd.DataFrame(train_dataset)
val_dataset = pd.DataFrame(val_dataset)

train_dataset.to_csv(
    path_train / paths['prepared']['train_csv'],
    index = False
)

val_dataset.to_csv(
    path_val / paths['prepared']['val_csv'],
    index = False
)