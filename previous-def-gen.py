import argparse
from pathlib import Path
import numpy as np
from osgeo import gdal, gdalconst, ogr
import yaml
from utils.ops import load_yaml

parser = argparse.ArgumentParser(
    description='Generate .tif previous deforestation temporal distance map. As older is the deforestation, the value is close to 0. As recent is the deforestation, the value is close to 1'
)

parser.add_argument( # The path to the config file (.yaml)
    '-c', '--cfg',
    type = Path,
    default = 'cfg.yaml',
    help = 'Path to the config file (.yaml).'
)

parser.add_argument( # specific site location number
    '-s', '--site',
    type = int,
    default=1,
    help = 'Site location number'
)

args = parser.parse_args()

#with open(args.cfg, 'r') as file:
#    cfg = yaml.load(file, Loader=yaml.Loader)
cfg = load_yaml(args.cfg)
site_cfg = load_yaml(f'site_{args.site}.yaml')

paths_params = cfg['paths']
general_params = cfg['general_params']
prodes_params = cfg['prodes_params']

original_data = site_cfg['original_data']

prodes_folder = Path(paths_params['prodes_data'])

base_image = Path(paths_params['opt_data']) / original_data['opt_imgs']['train'][0]

#previous_def_params = cfg['previous_def_params']

f_yearly_def = prodes_folder / prodes_params['yearly_deforestation']
v_yearly_def = ogr.Open(str(f_yearly_def))
l_yearly_def = v_yearly_def.GetLayer()

f_previous_def = prodes_folder / prodes_params['previous_def']
v_previous_def = ogr.Open(str(f_previous_def))
l_previous_def = v_previous_def.GetLayer()

base_data = gdal.Open(str(base_image), gdalconst.GA_ReadOnly)

geo_transform = base_data.GetGeoTransform()
x_res = base_data.RasterXSize
y_res = base_data.RasterYSize
crs = base_data.GetSpatialRef()
proj = base_data.GetProjection()

#train previous deforestation


base_year = 2007 #! CHANGE 2001 2007
delta_years = 10
vals = np.linspace(0.1,1, delta_years)

train_output = paths_params['previous_train']

target_train = gdal.GetDriverByName('GTiff').Create(train_output, x_res, y_res, 1, gdal.GDT_Float32)
target_train.SetGeoTransform(geo_transform)
target_train.SetSpatialRef(crs)
target_train.SetProjection(proj)

train_year = general_params['train_year']
prev_train_year = train_year - 1

years = np.linspace(prev_train_year-delta_years+1 ,prev_train_year, delta_years).astype(np.int32)


gdal.RasterizeLayer(target_train, [1], l_previous_def, burn_values=[vals[0]])
print('prev', vals[0])

where = f'"year"<{years[0]}'
l_yearly_def.SetAttributeFilter(where)
gdal.RasterizeLayer(target_train, [1], l_yearly_def, burn_values=[vals[0]])
print(where, vals[0])

for i, year in enumerate(years):
    v = vals[i]
    print(year, v)
    where = f'"year"={year}'
    l_yearly_def.SetAttributeFilter(where)
    gdal.RasterizeLayer(target_train, [1], l_yearly_def, burn_values=[v])

target_train = None
    

test_output = paths_params['previous_test']

target_test = gdal.GetDriverByName('GTiff').Create(test_output, x_res, y_res, 1, gdal.GDT_Float32)
target_test.SetGeoTransform(geo_transform)
target_test.SetSpatialRef(crs)
target_test.SetProjection(proj)

test_year = general_params['test_year']
prev_test_year = test_year - 1

years = np.linspace(prev_test_year-delta_years+1 ,prev_test_year, delta_years).astype(np.int32)


gdal.RasterizeLayer(target_test, [1], l_previous_def, burn_values=[vals[0]])
print('prev', vals[0])

where = f'"year"<{years[0]}'
l_yearly_def.SetAttributeFilter(where)
gdal.RasterizeLayer(target_test, [1], l_yearly_def, burn_values=[vals[0]])
print(where, vals[0])

for i, year in enumerate(years):
    v = vals[i]
    print(year, v)
    where = f'"year"={year}'
    l_yearly_def.SetAttributeFilter(where)
    gdal.RasterizeLayer(target_test, [1], l_yearly_def, burn_values=[v])

target_test = None