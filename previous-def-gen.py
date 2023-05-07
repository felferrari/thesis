import argparse
from pathlib import Path
import numpy as np
from osgeo import gdal, gdalconst, ogr
import yaml

parser = argparse.ArgumentParser(
    description='Generate .tif previous deforestation temporal distance map. As older is the deforestation, the value is close to 0. As recent is the deforestation, the value is close to 1'
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

prodes_folder = Path(original_data['prodes']['folder'])

base_image = Path(original_data['opt']['folder']) / original_data['opt']['imgs']['train'][0][0]

previous_def_params = cfg['previous_def_params']

f_yearly_def = prodes_folder / original_data['prodes']['yearly_deforestation']
v_yearly_def = ogr.Open(str(f_yearly_def))
l_yearly_def = v_yearly_def.GetLayer()

f_previous_def = prodes_folder / original_data['prodes']['defor_2007']
v_previous_def = ogr.Open(str(f_previous_def))
l_previous_def = v_previous_def.GetLayer()

base_data = gdal.Open(str(base_image), gdalconst.GA_ReadOnly)

geo_transform = base_data.GetGeoTransform()
x_res = base_data.RasterXSize
y_res = base_data.RasterYSize
crs = base_data.GetSpatialRef()
proj = base_data.GetProjection()

#train previous deforestation
test_output = previous_def_params['train_path']

target_train = gdal.GetDriverByName('GTiff').Create(test_output, x_res, y_res, 1, gdal.GDT_Float32)
target_train.SetGeoTransform(geo_transform)
target_train.SetSpatialRef(crs)
target_train.SetProjection(proj)

train_year = previous_def_params['train_year']
last_year = train_year - 1
b_year = 2007
years = np.arange(b_year, train_year)
vals = np.linspace(0,1, len(years)+1)


gdal.RasterizeLayer(target_train, [1], l_previous_def, burn_values=[vals[1]])
print('prev', vals[1])

for i, t_year in enumerate(years[1:]):
    v = vals[i+2]
    print(t_year, v)
    where = f'"year"={t_year}'
    l_yearly_def.SetAttributeFilter(where)
    gdal.RasterizeLayer(target_train, [1], l_yearly_def, burn_values=[v])

target_train = None

#test previous deforestation
test_output = previous_def_params['test_path']

target_test = gdal.GetDriverByName('GTiff').Create(test_output, x_res, y_res, 1, gdal.GDT_Float32)
target_test.SetGeoTransform(geo_transform)
target_test.SetSpatialRef(crs)
target_test.SetProjection(proj)

test_year = previous_def_params['test_year']
last_year = test_year - 1
b_year = 2007
years = np.arange(b_year, test_year)
vals = np.linspace(0,1, len(years)+1)


gdal.RasterizeLayer(target_test, [1], l_previous_def, burn_values=[vals[1]])
print('prev', vals[1])

for i, t_year in enumerate(years[1:]):
    v = vals[i+2]
    print(t_year, v)
    where = f'"year"={t_year}'
    l_yearly_def.SetAttributeFilter(where)
    gdal.RasterizeLayer(target_test, [1], l_yearly_def, burn_values=[v])

target_test = None