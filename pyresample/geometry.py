#pyresample, Resampling of remote sensing image data in python
# 
#Copyright (C) 2010  Esben S. Nielsen
#
#This program is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, either version 3 of the License, or
#(at your option) any later version.
#
#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#GNU General Public License for more details.
#
#You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""Classes for geometry operations"""

import numpy as np

import _spatial_mp


class BaseDefinition(object):
    
    def __init__(self, lons=None, lats=None, 
                 cartesian_coords=None, nprocs=1):
        self.nprocs = nprocs
        self._lons = lons
        self._lats = lats
        self._cartesian_coords = cartesian_coords
    
    def get_lonlats(self, *args, **kwargs):
        if self._lons is None or self._lats is None:
            raise ValueError('lon/lat values are not defined')
        return self._lons, self._lats   
#    def get_lonlats(self, *args, **kwargs):
#        raise NotImplementedError('Method "get_lonlats" not implemented ' 
#                                  'in base class')
    
    def get_cartesian_coords(self, nprocs=None):
        if self._cartesian_coords is None:
            if nprocs is None:
                nprocs = self.nprocs
                
            if nprocs > 1:
                cartesian = _spatial_mp.Cartesian_MP(nprocs)
            else:
                cartesian = _spatial_mp.Cartesian()
            
            lons, lats = self.get_lonlats(nprocs)
            cartesian_coords = cartesian.transform_lonlats(lons.ravel(), 
                                                           lats.ravel())
            if lons.ndim > 1:
                cartesian_coords = cartesian_coords.reshape(lons.shape[0], 
                                                            lons.shape[1], 3)
                
        else:
            cartesian_coords = self._cartesian_coords
            
        return cartesian_coords
    
    @property
    def lons(self):
        if self._lons is None:
            self._lons, self._lats = self.get_lonlats()
        return self._lons
    
    @property
    def lats(self):
        if self._lats is None:
            self._lons, self._lats = self.get_lonlats()
        return self._lats
    
    @property
    def cartesian_coords(self):
        if self._cartesian_coords is None:
            self._cartesian_coords = self.get_cartesian_coords()
        return self._cartesian_coords
 
 
class CoordinateDefinition(BaseDefinition):
 
    def __init__(self, lons=None, lats=None, cartesian_coords=None, 
                 nprocs=1):
        if (lons is not None or lats is not None):
            self.shape = lons.shape
            self.size = lons.size
        elif cartesian_coords is not None:
            self.shape = cartesian_coords.shape[:-1]
            self.size = cartesian_coords.size // 3
        else:
            raise ValueError(('%s must be created with either '
                             'lon/lats or cartesian coordinates') % 
                             self.__class__.__name__)
        super(CoordinateDefinition, self).__init__(lons, lats, 
                                                   cartesian_coords, nprocs)


class GridDefinition(CoordinateDefinition):
 
    def __init__(self, lons=None, lats=None, cartesian_coords=None, nprocs=1):
        super(GridDefinition, self).__init__(lons, lats, 
                                             cartesian_coords, nprocs)


class AreaDefinition(BaseDefinition):    
    """Holds definition of an area.

    :Parameters:
    area_id : str 
        ID of area
    proj_id : str 
        ID of projection
    proj_dict : dict 
        Dictionary with Proj.4 parameters
    x_size : int 
        x dimension in number of pixels
    y_size : int     
        y dimension in number of pixels    
    area_extent : list 
        Area extent as a list (LL_x, LL_y, UR_x, UR_y)
    
    :Attributes:
    area_id : str         
        ID of area
    proj_id : str         
        ID of projection
    proj_dict : dict        
        Dictionary with Proj.4 parameters
    x_size : int          
        x dimension in number of pixels
    y_size : int          
        y dimension in number of pixels
    shape : tuple
        Corresponding array shape as (rows, cols)
    size : int
        Number of points in grid
    area_extent : tuple     
        Area extent as a list (LL_x, LL_y, UR_x, UR_y)
    pixel_size_x : float    
        Pixel width in projection units
    pixel_size_y : float    
        Pixel height in projection units
    pixel_upper_left : list 
        Coordinates (x, y) of center of upper left pixel in projection units
    pixel_offset_x : float 
        x offset between projection center and upper left corner of upper 
        left pixel in units of pixels.
    pixel_offset_y : float 
        y offset between projection center and upper left corner of upper 
        left pixel in units of pixels..
    proj4_string : str
        Projection defined as Proj.4 string
    """

    def __init__(self, area_id, name, proj_id, proj_dict, x_size, y_size,
                 area_extent, nprocs=1, lons=None, lats=None, 
                 cartesian_coords=None):
        if not isinstance(proj_dict, dict):
            raise TypeError('Wrong type for proj_dict: %s. Expected dict.'
                            % type(proj_dict))

        super(AreaDefinition, self).__init__(lons, lats, cartesian_coords,
                                             nprocs)
        self.area_id = area_id
        self.name = name
        self.proj_id = proj_id
        self.x_size = x_size
        self.y_size = y_size
        self.shape = (y_size, x_size)
        self.size = y_size * x_size
        self.pixel_size_x = (area_extent[2] - area_extent[0]) / float(x_size)
        self.pixel_size_y = (area_extent[3] - area_extent[1]) / float(y_size)
        self.proj_dict = proj_dict
        self.area_extent = tuple(area_extent)
                
        #Calculate projection coordinates of center of upper left pixel
        self.pixel_upper_left = \
                              (float(area_extent[0]) + 
                               float(self.pixel_size_x) / 2,
                               float(area_extent[3]) - 
                               float(self.pixel_size_y) / 2)
        
        #Pixel_offset defines the distance to projection center from origen (UL)
        #of image in units of pixels. 
        self.pixel_offset_x = -self.area_extent[0] / self.pixel_size_x
        self.pixel_offset_y = self.area_extent[3] / self.pixel_size_y
        
    def __str__(self):
        return ('Area ID: %s\nName: %s\nProjection ID: %s\n'
               'Projection: %s\nNumber of columns: %s\nNumber of rows: %s\n'
               'Area extent: %s') % (self.area_id, self.name, self.proj_id, 
                                   self.proj_dict, self.x_size, self.y_size, 
                                   self.area_extent)
               
    __repr__ = __str__
               
    def get_lonlat(self, row, col):
        """Retrieves lon and lat values of single point in area grid
        
        :Parameters:
        row : int
        col : int
        
        :Returns:
        (lon, lat) : list of floats
        """
        
        if self._lons is None or self._lats is None:
            #Negative indices wrap-around
            if row < 0:
                row = self.x_size + row
            if col < 0:
                col = self.y_size + col
            
            #Get projection coordinates of point
            x_coord = self.pixel_upper_left[0] + col * self.pixel_size_x
            y_coord = self.pixel_upper_left[1] - row * self.pixel_size_y
            
            #Reproject
            proj = _spatial_mp.Proj(**self.proj_dict)
            lon, lat = proj(x_coord, y_coord, inverse=True)
        else:
            lon = self._lons[row, col]
            lat = self._lats[row, col]
            
        return lon, lat
    
    def get_proj_coords(self):
        """Get projection coordinates of grid        
    
        :Returns: 
        (target_x, target_y) : tuple of numpy arrays
            Grids of area x- and y-coordinates in projection units
        """        
        
        #create coordinates of local area as ndarrays
        target_x = np.fromfunction(lambda i, j: j * self.pixel_size_x + 
                                   self.pixel_upper_left[0],
                                   (self.y_size, 
                                    self.x_size))
    
        target_y = np.fromfunction(lambda i, j: 
                                   self.pixel_upper_left[1] - 
                                   i * self.pixel_size_y,
                                   (self.y_size, 
                                    self.x_size))
        return target_x, target_y
    
    def get_lonlats(self, nprocs=None):
        """Returns lon and lat arrays of area.
    
        :Parameters:        
        nprocs : int, optional 
            Number of processor cores to be used
        
        :Returns: 
        (lons, lats) : list of numpy arrays
            Grids of area lons and and lats
        """        
        
        if self._lons is None or self._lats is None:
            if nprocs is None:
                nprocs = self.nprocs
                
            #Proj.4 definition of target area projection
            if nprocs > 1:
                target_proj = _spatial_mp.Proj_MP(**self.proj_dict)
            else:
                target_proj = _spatial_mp.Proj(**self.proj_dict)
        
            #Get coordinates of local area as ndarrays
            target_x, target_y = self.get_proj_coords()
            
            #Get corresponding longitude and latitude values
            lons, lats = target_proj(target_x, target_y, inverse=True,
                                     nprocs=nprocs)        
            
            #Free memory
            del(target_x)
            del(target_y)
        else:
            lons = self._lons
            lats = self._lats
            
        return lons, lats

    @property
    def proj4_string(self):
        """Returns projection definition as Proj.4 string"""
        
        items = self.proj_dict.items()
        return '+' + ' +'.join([ t[0] + '=' + t[1] for t in items])         

    
class SwathDefinition(CoordinateDefinition):
 
    def __init__(self, lons, lats, cartesian_coords=None, nprocs=1):
        super(SwathDefinition, self).__init__(lons, lats, cartesian_coords, 
                                              nprocs)
        
    