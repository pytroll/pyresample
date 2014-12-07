from __future__ import absolute_import

import numpy as np

from . import _spatial_mp
from . import geometry

class GridFilter(object):
    """Geographic filter from a grid
    
    :Parameters:
    grid_ll_x : float
        Projection x coordinate of lower left corner of lower left pixel
    grid_ll_y : float
        Projection y coordinate of lower left corner of lower left pixel
    grid_ur_x : float
        Projection x coordinate of upper right corner of upper right pixel
    grid_ur_y : float 
        Projection y coordinate of upper right corner of upper right pixel 
    proj4_string : string 
    mask : numpy array
        Mask as boolean numpy array
        
    """
    
    def __init__(self, area_def, filter, nprocs=1):
        self.area_def = area_def
        self._filter = filter.astype(np.bool)
        self.nprocs = nprocs
        
    def get_valid_index(self, geometry_def):
        """Calculates valid_index array  based on lons and lats
        
        :Parameters:
        lons : numpy array
        lats : numpy array
        
        :Returns:
            Boolean numpy array of same shape as lons and lats
             
        """
        
        lons = geometry_def.lons[:]
        lats = geometry_def.lats[:]
        
        #Get projection coords
        if self.nprocs > 1:
            proj = _spatial_mp.Proj_MP(**self.area_def.proj_dict)
        else:
            proj = _spatial_mp.Proj(**self.area_def.proj_dict)
            
        x_coord, y_coord = proj(lons, lats, nprocs=self.nprocs)
                        
        #Find array indices of coordinates   
        target_x = ((x_coord / self.area_def.pixel_size_x) + 
                    self.area_def.pixel_offset_x).astype(np.int32)
        target_y = (self.area_def.pixel_offset_y - 
                    (y_coord / self.area_def.pixel_size_y)).astype(np.int32)        
        
        #Create mask for pixels outside array (invalid pixels)
        target_x_valid = (target_x >= 0) & (target_x < self.area_def.x_size)
        target_y_valid = (target_y >= 0) & (target_y < self.area_def.y_size)
        
        #Set index of invalid pixels to 0
        target_x[np.invert(target_x_valid)] = 0 
        target_y[np.invert(target_y_valid)] = 0
        
        #Find mask
        filter = self._filter[target_y, target_x]
        
        #Remove invalid pixels
        filter = (filter & target_x_valid & target_y_valid).astype(np.bool)
    
        return filter
    
    def filter(self, geometry_def, data):
        lons = geometry_def.lons[:]
        lats = geometry_def.lats[:]
        valid_index = self.get_valid_index(geometry_def)
        lons_f = lons[valid_index]
        lats_f = lats[valid_index]
        data_f = data[valid_index]
        geometry_def_f = \
            geometry.CoordinateDefinition(lons_f, lats_f, 
                                          nprocs=geometry_def.nprocs)
        return geometry_def_f, data_f
        
        
