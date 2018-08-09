from pyresample import geometry
from pyresample import utils
from xarray import DataArray
import numpy as np
area_id = 'ease_sh'
description = 'Antarctic EASE grid'
proj_id = 'ease_sh'
shape = [425, 425]
center = (0, 0)
radius = (5326849.0625, 5326849.0625)
area_extent = [-5326849.0625, -5326849.0625, 5326849.0625, 5326849.0625]
pixel_size = [25067.525, 25067.525]
top_left_extent = (-5326849.0625, 5326849.0625)
proj_dict = {'a': '6371228.0', 'units': 'm', 'lon_0': '0', 'proj': 'laea', 'lat_0': '-90'}
area_def = geometry.AreaDefinition(area_id, description, proj_id, proj_dict, shape[1], shape[0], area_extent)
print(area_def)
lons = area_def.get_lonlats()[0]
lats = area_def.get_lonlats()[1]
print('---')
# area = geometry.AreaDefinition.from_geotiff(description, '+lat_0=-90 +a=6371228.0 +units=m +lon_0=0 +proj=laea',
#                                                      top_left_extent, pixel_size, shape)
area = geometry.AreaDefinition.from_params(description, proj4=proj_dict, lons=lons, lats=lats)
# shape, top_left_extent, pixel_size
print(area)
# print(*utils.load_area('/Users/wroberts/Desktop/pyresample_extent/pyresample/test/test_files/areas.yaml'))
# print(area_def.create_areas_def_legacy())
# proj_str='+lat_0=-90 +a=6371228.0 +units=m +lon_0=0 +proj=laea'
