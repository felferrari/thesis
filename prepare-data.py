import argparse
from  pathlib import Path
import yaml
import logging
from utils.ops import load_opt_image, load_SAR_image, load_sb_image, save_yaml, load_yaml
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
    default=False,
    action=argparse.BooleanOptionalAction,
    help = 'Create training/validation data.'
)

parser.add_argument( # Create test data.
    '-t', '--test-data',
    default=False,
    action=argparse.BooleanOptionalAction,
    help = 'Create test data.'
)

parser.add_argument( # (Re)Generate statistics file.
    '-s', '--statistics',
    default=False,
    action=argparse.BooleanOptionalAction,
    help = '(Re)Generate statistics file.'
)


parser.add_argument( # Clear training prepared folder before prepare new data.
    '-x', '--clear-train-folder',
    action=argparse.BooleanOptionalAction,
    default=False,
    help = 'Clear training prepared folder before prepare new data.'
)

parser.add_argument( # Clear test prepared folder before prepare new data.
    '-z', '--clear-test-folder',
    action=argparse.BooleanOptionalAction,
    default=False,
    help = 'Clear test prepared folder before prepare new data.'
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
opt_prefix = preparation_params['prefixs']['opt']
sar_prefix = preparation_params['prefixs']['sar']
label_prefix = preparation_params['prefixs']['label']
previous_prefix = preparation_params['prefixs']['previous']
opt_bands = preparation_params['opt_bands']
sar_bands = preparation_params['sar_bands']


prepared_folder = Path(preparation_params['folder'])
train_folder = prepared_folder / preparation_params['train_folder']
validation_folder = prepared_folder / preparation_params['validation_folder']
test_folder = prepared_folder / preparation_params['test_folder']
#prediction_folder = prepared_folder / preparation_params['prediction_folder']

prepared_folder.mkdir(exist_ok=True)
if (args.clear_train_folder or args.train_data) and train_folder.exists():
    rmtree(train_folder)
    rmtree(validation_folder)
    train_folder.mkdir()
    validation_folder.mkdir()
if (args.clear_test_folder  or args.test_data) and test_folder.exists():
    rmtree(test_folder)
    test_folder.mkdir()

train_folder.mkdir(exist_ok=True)
validation_folder.mkdir(exist_ok=True)
test_folder.mkdir(exist_ok=True)


prepared_patches_file = prepared_folder / preparation_params['prepared_data']
statistics_file = prepared_folder / preparation_params['statistics_data']

#Generating patches idx
tiles = load_sb_image(Path(tiles_params['path']))
shape = tiles.shape
tiles = tiles.flatten()

#def_prev = def_prev.flatten()
idx = np.arange(shape[0] * shape[1]).reshape(shape)
window_shape = (patch_size, patch_size)
slide_step = int((1-patch_overlap)*patch_size)
idx_patches = view_as_windows(idx, window_shape, slide_step).reshape((-1, patch_size, patch_size))

np.random.seed(123)


if args.statistics:

    opt_path = Path(original_data_params['opt']['folder'])
    sar_path = Path(original_data_params['sar']['folder'])

    opt_means, opt_stds = [], []
    sar_means, sar_stds = [], []
    for opt_img_i, opt_img_file in enumerate(tqdm(original_data_params['opt']['imgs']['train'], desc = 'Generating OPT statistics')):
        data_file = opt_path / opt_img_file
        data = load_opt_image(data_file)
        opt_means.append(data.mean(axis=(0,1)))
        opt_stds.append(data.std(axis=(0,1)))
    for sar_img_i, sar_img_file in enumerate(tqdm(original_data_params['sar']['imgs']['train'], desc = 'Generating SAR statistics')):
        data_file = sar_path / sar_img_file
        data = load_SAR_image(data_file)
        sar_means.append(data.mean(axis=(0,1)))
        sar_stds.append(data.std(axis=(0,1)))
    opt_means = np.array(opt_means).mean(axis=0)
    opt_stds = np.array(opt_stds).mean(axis=0)

    sar_means = np.array(sar_means).mean(axis=0)
    sar_stds = np.array(sar_stds).mean(axis=0)

    statistics = {
        'opt_means': opt_means.tolist(),
        'opt_stds': opt_stds.tolist(),
        'sar_means': sar_means.tolist(),
        'sar_stds': sar_stds.tolist()
    }

    save_yaml(statistics, statistics_file)

data = None
#training patches
if args.train_data:
    statistics = load_yaml(statistics_file)

    opt_means = statistics['opt_means']
    opt_stds = statistics['opt_stds']
    sar_means = statistics['sar_means']
    sar_stds = statistics['sar_stds']

    outfile = prepared_folder / 'train-data-prep.txt'
    logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s:%(levelname)s:%(name)s:%(message)s',
            filename=outfile,
            filemode='w'
            )
    log = logging.getLogger('preparing')

    train_label = load_sb_image(Path(label_params['train_path'])).astype(np.uint8).flatten()
    previous_map = load_sb_image(Path(previous_def_params['train_path'])).astype(np.float16)

    keep = ((train_label[idx_patches] == 1).sum(axis=(1,2)) / patch_size**2) >= min_def_proportion
    keep_args = np.argwhere(keep == True).flatten() #args with at least min_prop deforestation
    no_keep_args = np.argwhere(keep == False).flatten() #args with less than min_prop of deforestation

    no_keep_args = np.random.choice(no_keep_args, (keep==True).sum() // 5)

    keep_final = np.concatenate((keep_args, no_keep_args))

    all_idx_patches = idx_patches[keep_final]

    keep_train = np.squeeze(np.argwhere(np.all(tiles[all_idx_patches] == 1, axis=(1,2))))
    keep_val = np.squeeze(np.argwhere(np.all(tiles[all_idx_patches] == 0, axis=(1,2))))

    log.info(f'Train patches: {keep_train.sum()}')
    log.info(f'Validation patches: {keep_val.sum()}')
    
    train_idx_patches = all_idx_patches[keep_train]
    val_idx_patches = all_idx_patches[keep_val]

    prepared_patches = {
        'train': len(train_idx_patches),
        'val': len(val_idx_patches)
    }
    
    save_yaml(prepared_patches, prepared_patches_file)

    np.random.shuffle(val_idx_patches)
    np.random.shuffle(train_idx_patches)

    opt_path = Path(original_data_params['opt']['folder'])
    sar_path = Path(original_data_params['sar']['folder'])

    opt_imgs = []
    sar_imgs = []
    for opt_img_i, opt_img_file in enumerate(tqdm(original_data_params['opt']['imgs']['train'], desc = 'Reading OPT Training files')):
        data_file = opt_path / opt_img_file
        data = load_opt_image(data_file)
        data = (data - opt_means) / opt_stds
        opt_imgs.append(data.astype(np.float16).reshape((-1, opt_bands)))
    for sar_img_i, sar_img_file in enumerate(tqdm(original_data_params['sar']['imgs']['train'], desc = 'Reading SAR Training files')):
        data_file = sar_path / sar_img_file
        data = load_SAR_image(data_file)
        data = (data - sar_means) / sar_stds
        sar_imgs.append(data.astype(np.float16).reshape((-1, sar_bands)))

    previous_map = previous_map.astype(np.float16).reshape(-1, 1)
    del data
    
    for patch_i, idx_patch in enumerate(tqdm(train_idx_patches, desc='Training Patches')):
        for opt_img_i, opt_img in enumerate(opt_imgs):
            data_i = opt_img[idx_patch]
            data_patch_file = train_folder / f'{opt_prefix}_{patch_i:d}-{opt_img_i}.npy'
            np.save(data_patch_file, data_i)
        for sar_img_i, sar_img in enumerate(sar_imgs):
            data_i = sar_img[idx_patch]
            data_patch_file = train_folder / f'{sar_prefix}_{patch_i:d}-{sar_img_i}.npy'
            np.save(data_patch_file, data_i)

        label_patch = train_label[idx_patch]
        label_patch_file = train_folder / f'{label_prefix}_{patch_i:d}.npy'
        np.save(label_patch_file, label_patch)

        previous_patch = previous_map[idx_patch]
        previous_patch_file = train_folder / f'{previous_prefix}_{patch_i:d}.npy'
        np.save(previous_patch_file, previous_patch)
        
        #train_dataset['patch_idx'].append(patch_i)

    for patch_i, idx_patch in enumerate(tqdm(val_idx_patches, desc='Validation Patches')):
        for opt_img_i, opt_img in enumerate(opt_imgs):
            data_i = opt_img[idx_patch]
            data_patch_file = validation_folder / f'{opt_prefix}_{patch_i:d}-{opt_img_i}.npy'
            np.save(data_patch_file, data_i)
        for sar_img_i, sar_img in enumerate(sar_imgs):
            data_i = sar_img[idx_patch]
            data_patch_file = validation_folder / f'{sar_prefix}_{patch_i:d}-{sar_img_i}.npy'
            np.save(data_patch_file, data_i)

        label_patch = train_label[idx_patch]
        label_patch_file = validation_folder / f'{label_prefix}_{patch_i:d}.npy'
        np.save(label_patch_file, label_patch)

        previous_patch = previous_map[idx_patch]
        previous_patch_file = validation_folder / f'{previous_prefix}_{patch_i:d}.npy'
        np.save(previous_patch_file, previous_patch)
        
        #val_dataset['patch_idx'].append(patch_i)
        
    del train_label
    data = None

#prediction/test image preparation
if args.test_data:

    statistics = load_yaml(statistics_file)

    opt_means = statistics['opt_means']
    opt_stds = statistics['opt_stds']
    sar_means = statistics['sar_means']
    sar_stds = statistics['sar_stds']

    opt_path = Path(original_data_params['opt']['folder'])
    sar_path = Path(original_data_params['sar']['folder'])

    for opt_img_i, opt_img_file in enumerate(tqdm(original_data_params['opt']['imgs']['test'], desc = 'Converting OPT Testing files')):
        data_file = opt_path / opt_img_file
        data = load_opt_image(data_file)
        data = (data - opt_means) / opt_stds
        data_patch_file = test_folder / f'{opt_prefix}_{opt_img_i}.npy'
        np.save(data_patch_file, data.astype(np.float16))
    for sar_img_i, sar_img_file in enumerate(tqdm(original_data_params['sar']['imgs']['test'], desc = 'Converting SAR Testing files')):
        data_file = sar_path / sar_img_file
        data = load_SAR_image(data_file)
        data = (data - sar_means) / sar_stds
        data_patch_file = test_folder / f'{sar_prefix}_{sar_img_i}.npy'
        np.save(data_patch_file, data.astype(np.float16))

    previous_map = load_sb_image(Path(previous_def_params['test_path'])).astype(np.float16)
    data_patch_file = test_folder / f'{previous_prefix}.npy'
    np.save(data_patch_file, previous_map.astype(np.float16))

    test_label = load_sb_image(Path(label_params['test_path'])).astype(np.uint8)
    data_patch_file = test_folder / f'{label_prefix}.npy'
    np.save(data_patch_file, test_label)