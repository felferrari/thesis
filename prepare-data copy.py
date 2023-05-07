import argparse
from  pathlib import Path
import yaml
import logging
from utils.ops import load_opt_image, load_SAR_image, load_sb_image, load_json, save_json
import numpy as np
from skimage.util import view_as_windows
from tqdm import tqdm
import pandas as pd
from shutil import rmtree

parser = argparse.ArgumentParser(
    description='Prepare the original files, generating .npy files to be used in the training/testing steps'
)

parser.add_argument( # The path to the config file (.yaml)
    '-c', '--cfg',
    type = Path,
    default = 'cfg.yaml',
    help = 'Path to the config file (.yaml).'
)

parser.add_argument( # Create training/validation data.
    '-r', '--train-data',
    default=True,
    action=argparse.BooleanOptionalAction,
    help = 'Create training/validation data.'
)

parser.add_argument( # Create test data.
    '-t', '--test-data',
    default=True,
    action=argparse.BooleanOptionalAction,
    help = 'Create test data.'
)

parser.add_argument( # Generate statistics.
    '-s', '--gen-statistics',
    action=argparse.BooleanOptionalAction,
    default=True,
    help = 'Generate statistics.'
)

parser.add_argument( # Clear prepared folder before prepare new data.
    '-x', '--clear-prepared-folder',
    action=argparse.BooleanOptionalAction,
    default=True,
    help = 'Clear prepared folder before prepare new data.'
)

parser.add_argument( # Create prediction data.
    '-p', '--prediction-data',
    default=True,
    action=argparse.BooleanOptionalAction,
    help = 'Create prediction data.'
)


args = parser.parse_args()

with open(args.cfg, 'r') as file:
    cfg = yaml.safe_load(file)

preparation_params = cfg['preparation_params']
tiles_params = cfg['tiles_params']
label_params = cfg['label_params']
previous_def_params = cfg['previous_def_params']
original_data_params = cfg['original_data']
prediction_params = cfg['prediction_params']

patch_size = preparation_params['patch_size']
patch_overlap = preparation_params['patch_overlap']
min_def_proportion = preparation_params['min_def_proportion']
data_prefix = preparation_params['data_prefix']
label_prefix = preparation_params['label_prefix']

prepared_folder = Path(preparation_params['folder'])
train_folder = prepared_folder / preparation_params['train_folder']
validation_folder = prepared_folder / preparation_params['validation_folder']
test_folder = prepared_folder / preparation_params['test_folder']
prediction_folder = prepared_folder / preparation_params['prediction_folder']

if args.clear_prepared_folder:
    rmtree(prepared_folder)
prepared_folder.mkdir(exist_ok=True)
train_folder.mkdir(exist_ok=True)
test_folder.mkdir(exist_ok=True)
validation_folder.mkdir(exist_ok=True)
prediction_folder.mkdir(exist_ok=True)

train_csv = prepared_folder / preparation_params['train_data']
test_csv = prepared_folder / preparation_params['test_data']
validation_csv = prepared_folder / preparation_params['validation_data']

outfile = prepared_folder / 'data-prep.txt'
logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s:%(levelname)s:%(name)s:%(message)s',
        filename=outfile,
        filemode='w'
        )
log = logging.getLogger('preparing')

#statistics estimation
statistics_file = prepared_folder / preparation_params['statistics_file']
if args.gen_statistics:
    log.info('Generating statistics.')
    opt_means, opt_stds, sar_means, sar_stds = [], [], [], []

    #optical
    #train files
    print('train OPT files')
    opt_path = Path(original_data_params['opt']['folder'])
    for opt_file_group in original_data_params['opt']['imgs']['train']:
        for opt_file in opt_file_group:
            file_path = opt_path / opt_file
            data = load_opt_image(file_path)
            opt_means.append(data.mean(axis=(0,1)))
            opt_stds.append(data.std(axis=(0,1)))

    #test files
    print('test SAR files')
    for opt_file_group in original_data_params['opt']['imgs']['test']:
        for opt_file in opt_file_group:
            file_path = opt_path / opt_file
            data = load_opt_image(file_path)
            opt_means.append(data.mean(axis=(0,1)))
            opt_stds.append(data.std(axis=(0,1)))

    opt_mean = np.array(opt_means).mean(axis=0)
    opt_std = np.array(opt_stds).mean(axis=0)

    #SAR
    #train files
    print('train OPT files')
    sar_path = Path(original_data_params['sar']['folder'])
    for sar_file_group in original_data_params['sar']['imgs']['train']:
        for sar_file in sar_file_group:
            file_path = sar_path / sar_file
            data = load_SAR_image(file_path)
            sar_means.append(data.mean(axis=(0,1)))
            sar_stds.append(data.std(axis=(0,1)))

    #test files
    print('test SAR files')
    for sar_file_group in original_data_params['sar']['imgs']['test']:
        for sar_file in sar_file_group:
            file_path = sar_path / sar_file
            data = load_SAR_image(file_path)
            sar_means.append(data.mean(axis=(0,1)))
            sar_stds.append(data.std(axis=(0,1)))

    sar_mean = np.array(sar_means).mean(axis=0)
    sar_std = np.array(sar_stds).mean(axis=0)

    statistics = {
        'opt_mean': opt_mean.tolist(),
        'opt_std': opt_std.tolist(),
        'sar_mean': sar_mean.tolist(),
        'sar_std': sar_std.tolist()
    }

    save_json(statistics, statistics_file)
else:
    log.info(f'Loading statistics from {statistics_file}.')
    statistics = load_json(statistics_file)
    opt_mean = statistics['opt_mean']
    opt_std = statistics['opt_std']
    sar_mean = statistics['sar_mean']
    sar_std = statistics['sar_std']


for st in statistics:
    log.info(f'{st}: {statistics[st]}')

#Generating patches idx
tiles = load_sb_image(Path(tiles_params['path']))
shape = tiles.shape
tiles = tiles.flatten()

#def_prev = def_prev.flatten()
idx = np.arange(shape[0] * shape[1]).reshape(shape)
window_shape = (patch_size, patch_size)
slide_step = int((1-patch_overlap)*patch_size)
idx_patches = view_as_windows(idx, window_shape, slide_step).reshape((-1, patch_size, patch_size))

#training patches
if args.train_data:
    train_dataset = {
        'data': [],
        'label': []
    }
    val_dataset = {
        'data': [],
        'label': []
    }
    train_label = load_sb_image(Path(label_params['train_path'])).astype(np.uint8).flatten()
    def_prev = load_sb_image(Path(previous_def_params['train_path'])).astype(np.float32)

    keep = ((train_label[idx_patches] == 1).sum(axis=(1,2)) / patch_size**2) >= min_def_proportion
    keep_args = np.argwhere(keep == True).flatten() #args with at least min_prop deforestation
    no_keep_args = np.argwhere(keep == False).flatten() #args with less than min_prop of deforestation
    no_keep_args = np.random.choice(no_keep_args, (keep==True).sum())

    keep_final = np.concatenate((keep_args, no_keep_args))

    all_idx_patches = idx_patches[keep_final]
    #all_idx_patches = idx_patches[keep_args]

    keep_val = (tiles[all_idx_patches] == 0).sum(axis=(1,2)) == patch_size**2
    keep_train = (tiles[all_idx_patches] == 1).sum(axis=(1,2)) == patch_size**2

    log.info(f'Train patches: {keep_train.sum()}')
    log.info(f'Validation patches: {keep_val.sum()}')

    val_idx_patches = all_idx_patches[keep_val]
    train_idx_patches = all_idx_patches[keep_train]

    opt_path = Path(original_data_params['opt']['folder'])
    sar_path = Path(original_data_params['sar']['folder'])

    for opt_group_i, opt_group in enumerate(original_data_params['opt']['imgs']['train']):
        for sar_group_i, sar_group in enumerate(original_data_params['sar']['imgs']['train']):
            group_idx = f'{opt_group_i}-{opt_group_i}'
            imgs = []
            for opt_img_file in opt_group:
                data_file = opt_path / opt_img_file
                data = load_opt_image(data_file)
                data = (data - opt_mean) / opt_std
                imgs.append(data.astype(np.float32))

            for sar_img_file in sar_group:
                data_file = sar_path / sar_img_file
                data = load_SAR_image(data_file)
                data = (data - sar_mean) / sar_std
                imgs.append(data.astype(np.float32))

            imgs.append(np.expand_dims(def_prev, axis=-1))
            data = np.concatenate(imgs, axis=-1)
            data = data.reshape(-1, data.shape[-1])
            del imgs
            for patch_i, train_idx_patch in enumerate(tqdm(train_idx_patches, desc='Training Patches')):
                data_patch = data[train_idx_patch]
                label_patch = train_label[train_idx_patch]
                data_patch_file = train_folder / f'{data_prefix}_{patch_i:06d}-{group_idx}.npy'
                label_patch_file = train_folder / f'{label_prefix}_{patch_i:06d}-{group_idx}.npy'
                train_dataset['data'].append(data_patch_file)
                train_dataset['label'].append(label_patch_file)
                np.save(data_patch_file, data_patch)
                np.save(label_patch_file, label_patch)

            for patch_i, val_idx_patch in enumerate(tqdm(val_idx_patches, desc='Validation Patches')):
                data_patch = data[val_idx_patch]
                label_patch = train_label[val_idx_patch]
                data_patch_file = validation_folder / f'{data_prefix}_{patch_i:06d}-{group_idx}.npy'
                label_patch_file = validation_folder / f'{label_prefix}_{patch_i:06d}-{group_idx}.npy'
                val_dataset['data'].append(data_patch_file)
                val_dataset['label'].append(label_patch_file)
                np.save(data_patch_file, data_patch)
                np.save(label_patch_file, label_patch)

            del data, train_label

    train_dataset = pd.DataFrame(train_dataset)
    val_dataset = pd.DataFrame(val_dataset)

    train_dataset.to_csv(
        prepared_folder / preparation_params['train_data'],
        index = False
    )

    val_dataset.to_csv(
        prepared_folder / preparation_params['validation_data'],
        index = False
    )


#test patches
if args.test_data:
    test_dataset = {
        'data': [],
        'label': []
    }
    test_label = load_sb_image(Path(label_params['test_path'])).astype(np.uint8).flatten()
    def_prev = load_sb_image(Path(previous_def_params['test_path'])).astype(np.float32)

    keep = ((test_label[idx_patches] == 1).sum(axis=(1,2)) / patch_size**2) >= min_def_proportion
    keep_args = np.argwhere(keep == True).flatten() #args with at least min_prop deforestation
    no_keep_args = np.argwhere(keep == False).flatten() #args with less than min_prop of deforestation
    no_keep_args = np.random.choice(no_keep_args, (keep==True).sum())

    keep_final = np.concatenate((keep_args, no_keep_args))

    all_idx_patches = idx_patches[keep_final]

    log.info(f'Test patches: {all_idx_patches.sum()}')


    opt_path = Path(original_data_params['opt']['folder'])
    sar_path = Path(original_data_params['sar']['folder'])

    for opt_group_i, opt_group in enumerate(original_data_params['opt']['imgs']['test']):
        for sar_group_i, sar_group in enumerate(original_data_params['sar']['imgs']['test']):
            group_idx = f'{opt_group_i}-{opt_group_i}'
            imgs = []
            for opt_img_file in opt_group:
                data_file = opt_path / opt_img_file
                data = load_opt_image(data_file)
                data = (data - opt_mean) / opt_std
                imgs.append(data.astype(np.float32))

            for sar_img_file in sar_group:
                data_file = sar_path / sar_img_file
                data = load_SAR_image(data_file)
                data = (data - sar_mean) / sar_std
                imgs.append(data.astype(np.float32))

            imgs.append(np.expand_dims(def_prev, axis=-1))
            data = np.concatenate(imgs, axis=-1)
            data = data.reshape(-1, data.shape[-1])
            del imgs
            for patch_i, test_idx_patch in enumerate(tqdm(all_idx_patches, desc='Test Patches')):
                data_patch = data[test_idx_patch]
                label_patch = test_label[test_idx_patch]
                data_patch_file = test_folder / f'{data_prefix}_{patch_i:06d}-{group_idx}.npy'
                label_patch_file = test_folder / f'{label_prefix}_{patch_i:06d}-{group_idx}.npy'
                test_dataset['data'].append(data_patch_file)
                test_dataset['label'].append(label_patch_file)
                np.save(data_patch_file, data_patch)
                np.save(label_patch_file, label_patch)


    test_dataset = pd.DataFrame(test_dataset)

    test_dataset.to_csv(
        prepared_folder / preparation_params['test_data'],
        index = False
    )
#prediction patches
if args.prediction_data:
    test_label = load_sb_image(Path(label_params['test_path'])).astype(np.uint8)
    def_prev = load_sb_image(Path(previous_def_params['test_path'])).astype(np.float32)
    pad_shape = ((patch_size, patch_size),(patch_size, patch_size))

    test_label = np.pad(test_label, pad_shape, mode = 'reflect')
    def_prev = np.pad(def_prev, pad_shape, mode = 'reflect')

    shape = test_label.shape

    test_label = test_label.flatten()

    idx = np.arange(shape[0] * shape[1]).reshape(shape)
    window_shape = (patch_size, patch_size)

    pad_shape = ((patch_size, patch_size),(patch_size, patch_size),(0,0))

    for overlap_i, overlap in enumerate(prediction_params['prediction_overlaps']):

        slide_step = int((1-overlap)*patch_size)
        pred_idx_patches = view_as_windows(idx, window_shape, slide_step).reshape((-1, patch_size, patch_size))
        pred_idx_file = prepared_folder / f'pred_idx_{overlap_i}.npy'
        np.save(pred_idx_file, pred_idx_patches)

        log.info(f'Prediction patches: {pred_idx_patches.sum()}')

        opt_path = Path(original_data_params['opt']['folder'])
        sar_path = Path(original_data_params['sar']['folder'])

        for opt_group_i, opt_group in enumerate(original_data_params['opt']['imgs']['test']):
            for sar_group_i, sar_group in enumerate(original_data_params['sar']['imgs']['test']):
                group_idx = f'{overlap_i}-{opt_group_i}-{sar_group_i}'
                prediction_dataset = {
                    'data': [],
                    'label': []
                }
                imgs = []
                for opt_img_file in opt_group:
                    data_file = opt_path / opt_img_file
                    data = load_opt_image(data_file)
                    data = (data - opt_mean) / opt_std
                    data = np.pad(data, pad_shape, mode = 'reflect')
                    imgs.append(data.astype(np.float32))

                for sar_img_file in sar_group:
                    data_file = sar_path / sar_img_file
                    data = load_SAR_image(data_file)
                    data = (data - sar_mean) / sar_std
                    data = np.pad(data, pad_shape, mode = 'reflect')
                    imgs.append(data.astype(np.float32))

                imgs.append(np.expand_dims(def_prev, axis=-1))
                data = np.concatenate(imgs, axis=-1)
                data = data.reshape(-1, data.shape[-1])
                del imgs
                for patch_i, pred_idx_patch in enumerate(tqdm(pred_idx_patches, desc='Prediction Patches')):
                    data_patch = data[pred_idx_patch]
                    label_patch = test_label[pred_idx_patch]
                    data_patch_file = prediction_folder / f'{data_prefix}_{patch_i:08d}-{group_idx}.npy'
                    label_patch_file = prediction_folder / f'{label_prefix}_{patch_i:08d}-{group_idx}.npy'
                    prediction_dataset['data'].append(data_patch_file)
                    prediction_dataset['label'].append(label_patch_file)
                    np.save(data_patch_file, data_patch)
                    np.save(label_patch_file, label_patch)


                prediction_dataset = pd.DataFrame(prediction_dataset)

                prepared_file_path = prepared_folder / f'{preparation_params["prediction_data"]}_{group_idx}.csv'
                prediction_dataset.to_csv(
                    prepared_file_path,
                    index = False
                )