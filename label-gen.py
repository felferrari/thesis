import argparse
from pathlib import Path
from osgeo import ogr, gdal, gdalconst
from skimage.morphology import disk, dilation, erosion, area_opening
import numpy as np
import yaml

parser = argparse.ArgumentParser(
    description='Generate .tif label file from PRODES deforestation shapefile.'
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
label_params = cfg['label_params']

prodes_folder = Path(original_data['prodes']['folder'])

base_image = Path(original_data['opt']['folder']) / original_data['opt']['imgs']['train'][0]
f_yearly_def = prodes_folder / original_data['prodes']['yearly_deforestation']
v_yearly_def = ogr.Open(str(f_yearly_def))
l_yearly_def = v_yearly_def.GetLayer()

f_previous_def = prodes_folder / original_data['prodes']['defor_2007']
v_previous_def = ogr.Open(str(f_previous_def))
l_previous_def = v_previous_def.GetLayer()

f_no_forest = prodes_folder / original_data['prodes']['no_forest']
v_no_forest = ogr.Open(str(f_no_forest))
l_no_forest = v_no_forest.GetLayer()

f_residual = prodes_folder / original_data['prodes']['residual']
v_residual = ogr.Open(str(f_residual))
l_residual = v_residual.GetLayer()

f_hydrography = prodes_folder / original_data['prodes']['hydrography']
v_hydrography = ogr.Open(str(f_hydrography))
l_hydrography = v_hydrography.GetLayer()


base_data = gdal.Open(str(base_image), gdalconst.GA_ReadOnly)

geo_transform = base_data.GetGeoTransform()
x_res = base_data.RasterXSize
y_res = base_data.RasterYSize
crs = base_data.GetSpatialRef()
proj = base_data.GetProjection()

#train label
train_output = label_params['train_path']

target_train = gdal.GetDriverByName('GTiff').Create(train_output, x_res, y_res, 1, gdal.GDT_Byte)
target_train.SetGeoTransform(geo_transform)
target_train.SetSpatialRef(crs)
target_train.SetProjection(proj)

band = target_train.GetRasterBand(1)
band.FlushCache()
train_year = label_params['train_year']
where_past = f'"year"<={train_year -1}'
where_ref = f'"year"={train_year}'

gdal.RasterizeLayer(target_train, [1], l_previous_def, burn_values=[2])
gdal.RasterizeLayer(target_train, [1], l_no_forest, burn_values=[2])
gdal.RasterizeLayer(target_train, [1], l_hydrography, burn_values=[2])
gdal.RasterizeLayer(target_train, [1], l_residual, burn_values=[2])

l_yearly_def.SetAttributeFilter(where_past)
gdal.RasterizeLayer(target_train, [1], l_yearly_def, burn_values=[2])

l_yearly_def.SetAttributeFilter(where_ref)
gdal.RasterizeLayer(target_train, [1], l_yearly_def, burn_values=[1])

rasterized_data = target_train.ReadAsArray() 

defor_data = rasterized_data == 1
defor_data = defor_data.astype(np.uint8)

def_inner_buffer = label_params['def_inner_buffer']
def_outer_buffer = label_params['def_outer_buffer']
border_data = dilation(defor_data, disk(def_inner_buffer)) - erosion(defor_data, disk(def_outer_buffer))

rasterized_data[border_data==1] = 2

target_train.GetRasterBand(1).WriteArray(rasterized_data)
target_train = None

#test label
test_output = label_params['test_path']

target_test = gdal.GetDriverByName('GTiff').Create(test_output, x_res, y_res, 1, gdal.GDT_Byte)
target_test.SetGeoTransform(geo_transform)
target_test.SetSpatialRef(crs)
target_test.SetProjection(proj)

band = target_test.GetRasterBand(1)
band.FlushCache()
test_year = label_params['test_year']
where_past = f'"year"<={test_year -1}'
where_ref = f'"year"={test_year}'

gdal.RasterizeLayer(target_test, [1], l_previous_def, burn_values=[2])
gdal.RasterizeLayer(target_test, [1], l_no_forest, burn_values=[2])
gdal.RasterizeLayer(target_test, [1], l_hydrography, burn_values=[2])
gdal.RasterizeLayer(target_test, [1], l_residual, burn_values=[2])

l_yearly_def.SetAttributeFilter(where_past)
gdal.RasterizeLayer(target_test, [1], l_yearly_def, burn_values=[2])

l_yearly_def.SetAttributeFilter(where_ref)
gdal.RasterizeLayer(target_test, [1], l_yearly_def, burn_values=[1])

rasterized_data = target_test.ReadAsArray() 

defor_data = rasterized_data == 1
defor_data = defor_data.astype(np.uint8)

def_inner_buffer = label_params['def_inner_buffer']
def_outer_buffer = label_params['def_outer_buffer']
border_data = dilation(defor_data, disk(def_inner_buffer)) - erosion(defor_data, disk(def_outer_buffer))

del defor_data

rasterized_data[border_data==1] = 2

defor_label = (rasterized_data==1).astype(np.uint8)
defor_remove = defor_label - area_opening(defor_label, 625)
rasterized_data[defor_remove == 1] = 2

target_test.GetRasterBand(1).WriteArray(rasterized_data)
target_test = None