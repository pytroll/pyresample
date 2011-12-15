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
import weakref

import numpy as np

import _spatial_mp


class DimensionError(Exception):
    pass


class Boundary(object):
    """Container for geometry boundary.
    Labelling starts in upper left corner and proceeds clockwise"""
      
    def __init__(self, side1, side2, side3, side4):
        self.side1 = side1
        self.side2 = side2
        self.side3 = side3
        self.side4 = side4


class _GeoCoords(object):
        """Container for geographic coordinates"""        
        
        def __init__(self, data=None):
            self.data = data
            
        def __getitem__(self, key):
            """Slicing and selecting"""
            
            if self.data is not None:
                return self.data[key]
            else:
                return self._get_coords(key)
        
        def _set_data(self, data):
            self.data = data
        
        def _get_coords(self, *args):
            raise NotImplementedError('Slice calculation not implemented '
                                      'in base class')
            
        @property
        def boundary(self):
            """Returns Boundary object"""
#            if self.data.ndim != 2:
#                raise DimensionError(('Can only retrieve bondary for 2D '
#                                      'geometry not %D') % self.data.ndim)
            
            side1 = self[0, :]
            side2 = self[:, -1]
            side3 = self[-1, :][::-1]
            side4 = self[:, 0][::-1]
            return Boundary(side1, side2, side3, side4)
    
    
class _GeoCoordsCached(_GeoCoords):
    """Container for geographic coordinates with caching"""
    
    def __init__(self, holder, index=None, data=None):
        super(_GeoCoordsCached, self).__init__(data)
        self._holder = holder
        self._index = index
    
    def _set_data(self, data):
        self.data = data
        self._holder._reset()
        
    def _get_coords(self, key):
        """Method for delegating caching"""
        
        if self._index is None:
            value = self._holder._get_coords(key)
        else:
            value = self._holder._get_coords(key)[self._index]
            
        if key == slice(None):
            self.data = value        
        return value
    

class _Lons(_GeoCoordsCached):
    """Container for lons"""
    
    def __init__(self, holder, data=None):
        super(_Lons, self).__init__(holder, index=0, data=data)
        
        
class _Lats(_GeoCoordsCached):
    """Container for lats"""
    
    def __init__(self, holder, data=None):
        super(_Lats, self).__init__(holder, index=1, data=data)
        

class _CartesianCoords(_GeoCoordsCached):
    """Container for cartesian coordinates"""
    
    def __init__(self, holder, data=None):
        super(_CartesianCoords, self).__init__(holder, index=None, data=data)
    
            
class _Holder(object):
    """Caching manager"""
    
    def __init__(self, get_function):
        self._get_function = get_function
        self.last_slice = None
        self.last_data = None
    
    def _reset(self):
        self.last_slice = None
        self.last_data = None
    
    def _get_coords(self, key):
        """Retrieve coordinates with caching"""
        
        if self.last_slice == key:
            data = self.last_data
        else:
            try:
                #Test if key is iterable
                key.__iter__
                data_slice = key                    
            except AttributeError:
                #Try to create row select key
                if isinstance(key, slice):
                    data_slice = (key, slice(None))                        
                else:
                    raise ValueError('slice could not be interpreted')
            data = self._get_function(data_slice=data_slice)
            self.last_slice = key
            self.last_data = data
            
        return data        
   
    
class BaseDefinition(object):
    """Base class for geometry definitions"""
           
    def __init__(self, lons=None, lats=None, nprocs=1):
        if type(lons) != type(lats):
            raise TypeError('lons and lats must be of same type')
        elif lons is not None:
            if lons.shape != lats.shape:
                raise ValueError('lons and lats must have same shape')
        self.nprocs = nprocs

        lonlat_holder = _Holder(weakref.proxy(self)._get_lonlats)
        self.lons = _Lons(lonlat_holder, data=lons)
        self.lats = _Lats(lonlat_holder, data=lats)
        
        cartesian_holder = _Holder(weakref.proxy(self)._get_cartesian_coords)
        self.cartesian_coords = _CartesianCoords(cartesian_holder)
    
    def __eq__(self, other):
        """Test for approximate equality"""

        if other.lons.data is None or other.lats.data is None:
            other.lons.data, other.lats.data = other.get_lonlats()
        if self.lons.data is None or self.lats.data is None:
            self.lons.data, self.lats.data = self.get_lonlats()
        try:
            return (np.allclose(self.lons.data, other.lons.data, atol=1e-6, 
                                rtol=5e-9) and
                    np.allclose(self.lats.data, other.lats.data, atol=1e-6, 
                                rtol=5e-9))
        except AttributeError:
            return False
    
    def __ne__(self, other):
        """Test for approximate equality"""
        
        return not self.__eq__(other)
    
    def get_lonlats(self, *args, **kwargs):
        """Retrieve lons and lats of geometry definition"""
        
        if self.lons.data is None or self.lats.data is None:
            raise ValueError('lon/lat values are not defined')
        return self.lons.data, self.lats.data
    
    def get_lonlat(self, row, col):
        """Retrieve lon and lat of single pixel
        
        :Parameters:
        row : int
        col : int
        
        :Returns:
        (lon, lat) : tuple of floats
        """
        
        if self.ndim != 2:
            raise DimensionError(('operation undefined '
                                  'for %sD geometry ') % self.ndim)
        elif self.lons.data is None or self.lats.data is None:
            raise ValueError('lon/lat values are not defined')
        return self.lons.data[row, col], self.lats.data[row, col]
    
    def _get_lonlats(self, nprocs=None, data_slice=None):
        """Base method for lon lat retrieval with slicing"""
        
        if self.lons.data is None or self.lats.data is None:
            raise ValueError('lon/lat values are not defined')
        return self.lons.data[data_slice], self.lats.data[data_slice]
             
    def get_cartesian_coords(self, nprocs=None):
        """Retrieve cartesian coordinates of geometry definition
        
        :Parameters:
        nprocs : int, optional
            Number of processor cores to be used.
            Defaults to the nprocs set when instantiating object
            
        :Returns:
        cartesian_coords : numpy array
        """
        
        return self._get_cartesian_coords(nprocs=nprocs)
    
    def _get_cartesian_coords(self, nprocs=None, data_slice=None):
        """Base method for cartesian coordinate retrieval with slicing"""
        
        if self.cartesian_coords.data is None:
            #Coordinates are not cached
            if nprocs is None:
                nprocs = self.nprocs
            
            if data_slice is None:
                #Use full slice
                data_slice = slice(None)
                
            lons, lats = self._get_lonlats(nprocs=nprocs, data_slice=data_slice)
                    
            if nprocs > 1:
                cartesian = _spatial_mp.Cartesian_MP(nprocs)
            else:
                cartesian = _spatial_mp.Cartesian()
                            
            cartesian_coords = cartesian.transform_lonlats(np.ravel(lons), 
                                                           np.ravel(lats))
            
            if isinstance(lons, np.ndarray) and lons.ndim > 1:
                #Reshape to correct shape
                cartesian_coords = cartesian_coords.reshape(lons.shape[0], 
                                                            lons.shape[1], 3)
                
        else:
            #Coordinates are cached
            if data_slice is None:
                cartesian_coords = self.cartesian_coords.data
            else:
                cartesian_coords = self.cartesian_coords.data[data_slice]
                
        return cartesian_coords    

    @property
    def corners(self):
        """Returns the corners of the current area.
        """
        from pyresample.spherical_geometry import Coordinate
        return [Coordinate(*self.get_lonlat(0, 0)),
                Coordinate(*self.get_lonlat(0, -1)),
                Coordinate(*self.get_lonlat(-1, -1)),
                Coordinate(*self.get_lonlat(-1, 0))]
        
    def __contains__(self, point):
        """Is a point inside the 4 corners of the current area? This uses
        great circle arcs as area boundaries.
        """
        from pyresample.spherical_geometry import point_inside, Coordinate
        corners = self.corners

        if isinstance(point, tuple):
            return point_inside(Coordinate(*point), corners)
        else:
            return point_inside(point, corners)

    def overlaps(self, other):
        """Tests if the current area overlaps the *other* area. This is based
        solely on the corners of areas, assuming the boundaries to be great
        circles.
        
        :Parameters:
        other : object
            Instance of subclass of BaseDefinition
            
        :Returns:
        overlaps : bool
        """

        from pyresample.spherical_geometry import Arc
        
        self_corners = self.corners

        other_corners = other.corners
        
        for i in self_corners:
            if i in other:
                return True
        for i in other_corners:
            if i in self:
                return True
    
        self_arc1 = Arc(self_corners[0], self_corners[1])
        self_arc2 = Arc(self_corners[1], self_corners[2])
        self_arc3 = Arc(self_corners[2], self_corners[3])
        self_arc4 = Arc(self_corners[3], self_corners[0])

        other_arc1 = Arc(other_corners[0], other_corners[1])
        other_arc2 = Arc(other_corners[1], other_corners[2])
        other_arc3 = Arc(other_corners[2], other_corners[3])
        other_arc4 = Arc(other_corners[3], other_corners[0])

        for i in (self_arc1, self_arc2, self_arc3, self_arc4):
            for j in (other_arc1, other_arc2, other_arc3, other_arc4):
                if i.intersects(j):
                    return True
        return False

    def get_area(self):
        """Get the area of the convex area defined by the corners of the current
        area.
        """
        from pyresample.spherical_geometry import get_polygon_area

        return get_polygon_area(self.corners)

    def intersection(self, other):
        """Returns the corners of the intersection polygon of the current area
        with *other*.
        
        :Parameters:
        other : object
            Instance of subclass of BaseDefinition
            
        :Returns:
        (corner1, corner2, corner3, corner4) : tuple of points
        """
        from pyresample.spherical_geometry import intersection_polygon
        return intersection_polygon(self.corners, other.corners)

    def overlap_rate(self, other):
        """Get how much the current area overlaps an *other* area.
        
        :Parameters:
        other : object
            Instance of subclass of BaseDefinition
            
        :Returns:
        overlap_rate : float
        """
        
        from pyresample.spherical_geometry import get_polygon_area
        other_area = other.get_area()
        inter_area = get_polygon_area(self.intersection(other))
        return inter_area / other_area


 
class CoordinateDefinition(BaseDefinition):
    """Base class for geometry definitions defined by lons and lats only"""
     
    def __init__(self, lons, lats, nprocs=1):
        if lons.shape == lats.shape:
            self.shape = lons.shape
            self.size = lons.size
            self.ndim = lons.ndim        
        else:
            raise ValueError(('%s must be created with either '
                             'lon/lats of the same shape') % 
                             self.__class__.__name__)
        super(CoordinateDefinition, self).__init__(lons, lats, nprocs)
        
    def concatenate(self, other):
        if self.ndim != other.ndim:
            raise DimensionError(('Unable to concatenate %sD and %sD '
                                  'geometries') % (self.ndim, other.ndim))
        klass = _get_highest_level_class(self, other)        
        lons = np.concatenate((self.lons.data, other.lons.data))
        lats = np.concatenate((self.lats.data, other.lats.data))
        nprocs = min(self.nprocs, other.nprocs)
        return klass(lons, lats, nprocs=nprocs)
        
    def append(self, other):    
        if self.ndim != other.ndim:
            raise DimensionError(('Unable to append %sD and %sD '
                                  'geometries') % (self.ndim, other.ndim))
        lons = np.concatenate((self.lons.data, other.lons.data))
        lats = np.concatenate((self.lats.data, other.lats.data))
        self.lons._set_data(lons)
        self.lats._set_data(lats)
        self.shape = lons.shape
        self.size = lons.size

    def __str__(self):
        #Rely on numpy's object printing
        return ('Shape: %s\nLons: %s\nLats: %s') % (str(self.shape), 
                                                    str(self.lons.data),
                                                    str(self.lats.data))
        

class GridDefinition(CoordinateDefinition):
    """Grid defined by lons and lats
    
    :Parameters:
    lons : numpy array
    lats : numpy array
    nprocs : int, optional
        Number of processor cores to be used for calculations.
        
    :Attributes:
    shape : tuple
        Grid shape as (rows, cols)
    size : int
        Number of elements in grid
        
    Properties:
    lons : object
        Grid lons
    lats : object
        Grid lats
    cartesian_coords : object
        Grid cartesian coordinates
    """
    
    def __init__(self, lons, lats, nprocs=1):
        if lons.shape != lats.shape:
            raise ValueError('lon and lat grid must have same shape')
        elif lons.ndim != 2:
            raise ValueError('2 dimensional lon lat grid expected')
        
        super(GridDefinition, self).__init__(lons, lats, nprocs)


class SwathDefinition(CoordinateDefinition):
    """Swath defined by lons and lats
    
    :Parameters:
    lons : numpy array
    lats : numpy array
    nprocs : int, optional
        Number of processor cores to be used for calculations.
        
    :Attributes:
    shape : tuple
        Swath shape
    size : int
        Number of elements in swath
    ndims : int
        Swath dimensions
        
    Properties:
    lons : object
        Swath lons
    lats : object
        Swath lats
    cartesian_coords : object
        Swath cartesian coordinates
    """
    
    def __init__(self, lons, lats, nprocs=1):
        if lons.shape != lats.shape:
            raise ValueError('lon and lat arrays must have same shape')
        elif lons.ndim > 2:
            raise ValueError('Only 1 and 2 dimensional swaths are allowed')
        super(SwathDefinition, self).__init__(lons, lats, nprocs)


class _ProjectionXCoords(_GeoCoordsCached):
    """Container for projection x coordinates"""
        
    def __init__(self, holder):
        super(_ProjectionXCoords, self).__init__(holder, index=0)


class _ProjectionYCoords(_GeoCoordsCached):
    """Container for projection y coordinates"""
    
    def __init__(self, holder):
        super(_ProjectionYCoords, self).__init__(holder, index=1)


class AreaDefinition(BaseDefinition):    
    """Holds definition of an area.

    :Parameters:
    area_id : str 
        ID of area
    name : str
        Name of area
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
    nprocs : int, optional 
        Number of processor cores to be used
    lons : numpy array, optional
        Grid lons
    lats : numpy array, optional
        Grid lats
    
    :Attributes:
    area_id : str         
        ID of area
    name : str
        Name of area
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
        Area extent as a tuple (LL_x, LL_y, UR_x, UR_y)
    area_extent_ll : tuple     
        Area extent in lons lats as a tuple (LL_lon, LL_lat, UR_lon, UR_lat)
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
    
    Properties:
    proj4_string : str
        Projection defined as Proj.4 string
    lons : object
        Grid lons
    lats : object
        Grid lats
    cartesian_coords : object
        Grid cartesian coordinates
    projection_x_coords : object
        Grid projection x coordinate
    projection_y_coords : object
        Grid projection y coordinate
    """
                  
            
    def __init__(self, area_id, name, proj_id, proj_dict, x_size, y_size,
                 area_extent, nprocs=1, lons=None, lats=None):
        if not isinstance(proj_dict, dict):
            raise TypeError('Wrong type for proj_dict: %s. Expected dict.'
                            % type(proj_dict))

        super(AreaDefinition, self).__init__(lons, lats, nprocs)
        self.area_id = area_id
        self.name = name
        self.proj_id = proj_id
        self.x_size = x_size
        self.y_size = y_size
        self.shape = (y_size, x_size)
        if lons is not None:
            if lons.shape != self.shape:
                raise ValueError('Shape of lon lat grid must match '
                                 'area definition')
        self.size = y_size * x_size
        self.ndim = 2
        self.pixel_size_x = (area_extent[2] - area_extent[0]) / float(x_size)
        self.pixel_size_y = (area_extent[3] - area_extent[1]) / float(y_size)
        self.proj_dict = proj_dict
        self.area_extent = tuple(area_extent)
        
        # Calculate area_extent in lon lat
        proj = _spatial_mp.Proj(**proj_dict)
        corner_lons, corner_lats = proj((area_extent[0], area_extent[2]), 
                                        (area_extent[1], area_extent[3]), 
                                        inverse=True)
        self.area_extent_ll = (corner_lons[0], corner_lats[0], 
                               corner_lons[1], corner_lats[1])
                
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
        
        proj_coords_holder = _Holder(weakref.proxy(self)._get_proj_coords)
        self.projection_x_coords = _ProjectionXCoords(proj_coords_holder)
        self.projection_y_coords = _ProjectionYCoords(proj_coords_holder)
        
    def __str__(self):
        #We need a sorted dictionary for a unique hash of str(self)
        proj_dict = self.proj_dict
        proj_str = ('{' + 
                    ', '.join(["'%s': '%s'"%(str(k), str(proj_dict[k]))
                               for k in sorted(proj_dict.keys())]) +
                    '}')
        return ('Area ID: %s\nName: %s\nProjection ID: %s\n'
                'Projection: %s\nNumber of columns: %s\nNumber of rows: %s\n'
                'Area extent: %s') % (self.area_id, self.name, self.proj_id, 
                                      proj_str, self.x_size, self.y_size, 
                                      self.area_extent)
               
    __repr__ = __str__
    
    def __eq__(self, other):
        """Test for equality"""
        
        try:
            return ((self.proj_dict == other.proj_dict) and
                    (self.shape == other.shape) and
                    (self.area_extent == other.area_extent))
        except AttributeError:
            return super(AreaDefinition, self).__eq__(other)
        
    def __ne__(self, other):
        """Test for equality"""
        
        return not self.__eq__(other)
               
    def get_lonlat(self, row, col):
        """Retrieves lon and lat values of single point in area grid
        
        :Parameters:
        row : int
        col : int
        
        :Returns:
        (lon, lat) : tuple of floats
        """
        
        return self._get_lonlats(nprocs=None, data_slice=(row, col))
    
    def get_proj_coords(self):
        """Get projection coordinates of grid        
    
        :Returns: 
        (target_x, target_y) : tuple of numpy arrays
            Grids of area x- and y-coordinates in projection units
        """        
    
        return self._get_proj_coords()
    
    def _get_proj_coords(self, data_slice=None):
        """Method for projection coordinate retrieval with slicing"""
        
        def get_val(val, sub_val, max):
            #Get value with substitution and wrapping
            if val is None:
                return sub_val
            else:
                if val < 0:
                    #Wrap index
                    return max + val
                else:
                    return val
        
        is_single_value = False
        is_1d_select = False    

        #create coordinates of local area as ndarrays
        if data_slice is None or data_slice == slice(None):
            #Full slice
            rows = self.y_size
            cols = self.x_size
            row_start = 0
            col_start = 0
        else:            
            if isinstance(data_slice, slice):
                #Row slice
                row_start = get_val(data_slice.start, 0, self.y_size)
                col_start = 0
                rows = get_val(data_slice.stop, self.y_size, self.y_size) - row_start                                 
                cols = self.x_size
            elif isinstance(data_slice[0], slice) and isinstance(data_slice[1], slice):
                #Block slice
                row_start = get_val(data_slice[0].start, 0, self.y_size)
                col_start = get_val(data_slice[1].start, 0, self.x_size)
                rows = get_val(data_slice[0].stop, self.y_size, self.y_size) - row_start
                cols = get_val(data_slice[1].stop, self.x_size, self.x_size) - col_start
            elif isinstance(data_slice[0], slice):
                #Select from col
                is_1d_select = True
                row_start = get_val(data_slice[0].start, 0, self.y_size)
                col_start = get_val(data_slice[1], 0, self.x_size)
                rows = get_val(data_slice[0].stop, self.y_size, self.y_size) - row_start
                cols = 1
            elif isinstance(data_slice[1], slice):
                #Select from row
                is_1d_select = True
                row_start = get_val(data_slice[0], 0, self.y_size)
                col_start = get_val(data_slice[1].start, 0, self.x_size)
                rows = 1
                cols = get_val(data_slice[1].stop, self.x_size, self.x_size) - col_start
            else:
                #Single element select
                is_single_value = True
                
                row_start = get_val(data_slice[0], 0, self.y_size)                
                col_start = get_val(data_slice[1], 0, self.x_size)
                    
                rows = 1
                cols = 1    
        
        #Calculate coordinates
        target_x = np.fromfunction(lambda i, j: (j + col_start) * 
                                   self.pixel_size_x + 
                                   self.pixel_upper_left[0],
                                   (rows, 
                                    cols))
    
        target_y = np.fromfunction(lambda i, j: 
                                   self.pixel_upper_left[1] - 
                                   (i + row_start) * self.pixel_size_y,
                                   (rows, 
                                    cols))
        
        if is_single_value:
            #Return single values
            target_x = float(target_x)
            target_y = float(target_y)
        elif is_1d_select:
            #Reshape to 1D array
            target_x = target_x.reshape((target_x.size,))
            target_y = target_y.reshape((target_y.size,))
        
        return target_x, target_y
    
    def get_lonlats(self, nprocs=None):
        """Returns lon and lat arrays of area.
    
        :Parameters:        
        nprocs : int, optional 
            Number of processor cores to be used.
            Defaults to the nprocs set when instantiating object
        
        :Returns: 
        (lons, lats) : tuple of numpy arrays
            Grids of area lons and and lats
        """        
        return self._get_lonlats(nprocs=nprocs)
    
    def _get_lonlats(self, nprocs=None, data_slice=None):
        """Method for lon lat coordinate retrieval with slicing"""
            
        if self.lons.data is None or self.lats.data is None:
            #Data is not cached
            if nprocs is None:
                nprocs = self.nprocs
                
            #Proj.4 definition of target area projection
            if nprocs > 1:
                target_proj = _spatial_mp.Proj_MP(**self.proj_dict)
            else:
                target_proj = _spatial_mp.Proj(**self.proj_dict)
        
            #Get coordinates of local area as ndarrays
            target_x, target_y = self._get_proj_coords(data_slice=data_slice)
            
            #Get corresponding longitude and latitude values
            lons, lats = target_proj(target_x, target_y, inverse=True,
                                     nprocs=nprocs)        
                
            #Free memory
            del(target_x)
            del(target_y)
        else:
            #Data is cached
            if data_slice is None:
                #Full slice
                lons = self.lons.data
                lats = self.lats.data
            else:
                lons = self.lons.data[data_slice]
                lats = self.lats.data[data_slice]
            
        return lons, lats

    @property
    def proj4_string(self):
        """Returns projection definition as Proj.4 string"""
        
        items = self.proj_dict.items()
        return '+' + ' +'.join([ t[0] + '=' + t[1] for t in items])         
    

def _get_slice(segments, shape):
    """Generator for segmenting a 1D or 2D array"""
    
    if not (1 <= len(shape) <= 2):
        raise ValueError('Cannot segment array of shape: %s' % str(shape))
    else:
        size = shape[0]
        slice_length = np.ceil(float(size) / segments)
        start_idx = 0
        end_idx = slice_length
        while start_idx < size:
            if len(shape) == 1:
                yield slice(start_idx, end_idx)
            else:
                yield (slice(start_idx, end_idx), slice(None))
            start_idx = end_idx
            end_idx = min(start_idx + slice_length, size)

def _flatten_cartesian_coords(cartesian_coords):
    """Flatten array to (n, 3) shape"""
    
    shape = cartesian_coords.shape 
    if len(shape) > 2:
        cartesian_coords = cartesian_coords.reshape(shape[0] * 
                                                    shape[1], 3)
    return cartesian_coords

def _get_highest_level_class(obj1, obj2):
    if (not issubclass(obj1.__class__, obj2.__class__) or 
        not issubclass(obj2.__class__, obj1.__class__)):
        raise TypeError('No common superclass for %s and %s' % 
                        (obj1.__class__, obj2.__class__))

    if obj1.__class__ == obj2.__class__:
        klass = obj1.__class__
    elif issubclass(obj1.__class__, obj2.__class__):
        klass = obj2.__class__
    else:
        klass = obj1.__class__
    return klass    
           
        
