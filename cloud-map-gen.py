import argparse
from utils.ops import load_opt_image, save_geotiff
import numpy as np
from  pathlib import Path
import yaml
from s2cloudless import S2PixelCloudDetector
from tqdm import tqdm

parser = argparse.ArgumentParser(
    description='Generate .tif with training (0) and validation (1) areas.'
)

parser.add_argument( # The path to the config file (.yaml)
    '-c', '--cfg',
    type = Path,
    default = 'cfg.yaml',
    help = 'Path to the config file (.yaml).'
)

args = parser.parse_args()

with open(args.cfg, 'r') as file:
    cfg = yaml.load(file, Loader=yaml.Loader)

original_data = cfg['original_data']
preparation_params = cfg['preparation_params']

cloud_prefix = preparation_params['prefixs']['cloud']

opt_folder = Path(original_data['opt']['folder'])

base_image = opt_folder/ original_data['opt']['imgs']['train'][0]

cloud_detector = S2PixelCloudDetector(threshold=0.4, average_over=4, dilation_size=2, all_bands=True)
for opt_img_file in tqdm(original_data['opt']['imgs']['train'], desc = 'Generating Clouds for Trainig files'):
    opt_img = load_opt_image(opt_folder / opt_img_file) / 10000
    cloud_map = cloud_detector.get_cloud_probability_maps(opt_img)

    cloud_tif_file = opt_folder / f'{cloud_prefix}_{opt_img_file}'
    save_geotiff(base_image, cloud_tif_file, cloud_map, dtype = 'float')

for opt_img_file in tqdm(original_data['opt']['imgs']['test'], desc = 'Generating Clouds for Testing files'):
    opt_img = load_opt_image(opt_folder / opt_img_file) / 10000
    cloud_map = cloud_detector.get_cloud_probability_maps(opt_img)

    cloud_tif_file = opt_folder / f'{cloud_prefix}_{opt_img_file}'
    save_geotiff(base_image, cloud_tif_file, cloud_map, dtype = 'float')
