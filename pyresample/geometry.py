# pyresample, Resampling of remote sensing image data in python
#
# Copyright (C) 2010-2016
#
# Authors:
#    Esben S. Nielsen
#    Thomas Lavergne
#    Adam Dybbroe
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License along
# with this program.  If not, see <http://www.gnu.org/licenses/>.

"""Classes for geometry operations"""

import warnings
import numpy as np
from pyresample import utils, _spatial_mp


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


class BaseDefinition(object):

    """Base class for geometry definitions"""

    def __init__(self, lons=None, lats=None, nprocs=1):
        if type(lons) != type(lats):
            raise TypeError('lons and lats must be of same type')
        elif lons is not None:
            if lons.shape != lats.shape:
                raise ValueError('lons and lats must have same shape')

        self.nprocs = nprocs

        # check the latitutes
        if lats is not None and ((lats.min() < -90. or lats.max() > +90.)):
            # throw exception
            raise ValueError(
                'Some latitudes are outside the [-90.;+90] validity range')
        else:
            self.lats = lats

        # check the longitudes
        if lons is not None and ((lons.min() < -180. or lons.max() >= +180.)):
            # issue warning
            warnings.warn('All geometry objects expect longitudes in the [-180:+180[ range. ' +
                          'We will now automatically wrap your longitudes into [-180:+180[, and continue. ' +
                          'To avoid this warning next time, use routine utils.wrap_longitudes().')
            # wrap longitudes to [-180;+180[
            self.lons = utils.wrap_longitudes(lons)
        else:
            self.lons = lons

        self.ndim = None
        self.cartesian_coords = None

    def __eq__(self, other):
        """Test for approximate equality"""

        if other.lons is None or other.lats is None:
            other_lons, other_lats = other.get_lonlats()
        else:
            other_lons = other.lons
            other_lats = other.lats

        if self.lons is None or self.lats is None:
            self_lons, self_lats = self.get_lonlats()
        else:
            self_lons = self.lons
            self_lats = self.lats

        try:
            return (np.allclose(self_lons, other_lons, atol=1e-6,
                                rtol=5e-9) and
                    np.allclose(self_lats, other_lats, atol=1e-6,
                                rtol=5e-9))
        except (AttributeError, ValueError):
            return False

    def __ne__(self, other):
        """Test for approximate equality"""

        return not self.__eq__(other)

    def get_area_extent_for_subset(self, row_LR, col_LR, row_UL, col_UL):
        """Retrieves area_extent for a subdomain 
        rows    are counted from upper left to lower left
        columns are counted from upper left to upper right 

        :Parameters:
        row_LR : int
            row of the lower right pixel 
        col_LR : int
            col of the lower right pixel
        row_UL : int
            row of the upper left pixel 
        col_UL : int
            col of the upper left pixel

        :Returns:
        area_extent : list 
            Area extent as a list (LL_x, LL_y, UR_x, UR_y) of the subset

        :Author:
        Ulrich Hamann
        """
        
        (a,b) = self.get_proj_coords( data_slice=(row_LR,col_LR) )
        a = a - 0.5*self.pixel_size_x
        b = b - 0.5*self.pixel_size_y
        (c,d) = self.get_proj_coords( data_slice=(row_UL,col_UL) )
        c = c + 0.5*self.pixel_size_x
        d = d + 0.5*self.pixel_size_y

        return (a,b,c,d)

    def get_lonlat(self, row, col):
        """Retrieve lon and lat of single pixel

        Parameters
        ----------
        row : int
        col : int

        Returns
        -------
        (lon, lat) : tuple of floats
        """

        if self.ndim != 2:
            raise DimensionError(('operation undefined '
                                  'for %sD geometry ') % self.ndim)
        elif self.lons is None or self.lats is None:
            raise ValueError('lon/lat values are not defined')
        return self.lons[row, col], self.lats[row, col]

    def get_lonlats(self, data_slice=None, **kwargs):
        """Base method for lon lat retrieval with slicing"""

        if self.lons is None or self.lats is None:
            raise ValueError('lon/lat values are not defined')
        elif data_slice is None:
            return self.lons, self.lats
        else:
            return self.lons[data_slice], self.lats[data_slice]

    def get_boundary_lonlats(self):
        """Returns Boundary objects"""

        side1 = self.get_lonlats(data_slice=(0, slice(None)))
        side2 = self.get_lonlats(data_slice=(slice(None), -1))
        side3 = self.get_lonlats(data_slice=(-1, slice(None)))
        side4 = self.get_lonlats(data_slice=(slice(None), 0))
        return Boundary(side1[0], side2[0], side3[0][::-1], side4[0][::-1]), Boundary(side1[1], side2[1], side3[1][::-1], side4[1][::-1])

    def get_cartesian_coords(self, nprocs=None, data_slice=None, cache=False):
        """Retrieve cartesian coordinates of geometry definition

        Parameters
        ----------
        nprocs : int, optional
            Number of processor cores to be used.
            Defaults to the nprocs set when instantiating object
        data_slice : slice object, optional
            Calculate only cartesian coordnates for the defined slice
        cache : bool, optional
            Store result the result. Requires data_slice to be None

        Returns
        -------
        cartesian_coords : numpy array
        """

        if self.cartesian_coords is None:
            # Coordinates are not cached
            if nprocs is None:
                nprocs = self.nprocs

            if data_slice is None:
                # Use full slice
                data_slice = slice(None)

            lons, lats = self.get_lonlats(nprocs=nprocs, data_slice=data_slice)

            if nprocs > 1:
                cartesian = _spatial_mp.Cartesian_MP(nprocs)
            else:
                cartesian = _spatial_mp.Cartesian()

            cartesian_coords = cartesian.transform_lonlats(np.ravel(lons),
                                                           np.ravel(lats))

            if isinstance(lons, np.ndarray) and lons.ndim > 1:
                # Reshape to correct shape
                cartesian_coords = cartesian_coords.reshape(lons.shape[0],
                                                            lons.shape[1], 3)

            if cache and data_slice is None:
                self.cartesian_coords = cartesian_coords
        else:
            # Coordinates are cached
            if data_slice is None:
                cartesian_coords = self.cartesian_coords
            else:
                cartesian_coords = self.cartesian_coords[data_slice]

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

        Parameters
        ----------
        other : object
            Instance of subclass of BaseDefinition

        Returns
        -------
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

        Parameters
        ----------
        other : object
            Instance of subclass of BaseDefinition

        Returns
        -------
        (corner1, corner2, corner3, corner4) : tuple of points
        """
        from pyresample.spherical_geometry import intersection_polygon
        return intersection_polygon(self.corners, other.corners)

    def overlap_rate(self, other):
        """Get how much the current area overlaps an *other* area.

        Parameters
        ----------
        other : object
            Instance of subclass of BaseDefinition

        Returns
        -------
        overlap_rate : float
        """

        from pyresample.spherical_geometry import get_polygon_area
        other_area = other.get_area()
        inter_area = get_polygon_area(self.intersection(other))
        return inter_area / other_area


class CoordinateDefinition(BaseDefinition):
    """Base class for geometry definitions defined by lons and lats only"""

    def __init__(self, lons, lats, nprocs=1):
        super(CoordinateDefinition, self).__init__(lons, lats, nprocs)
        if lons.shape == lats.shape and lons.dtype == lats.dtype:
            self.shape = lons.shape
            self.size = lons.size
            self.ndim = lons.ndim
            self.dtype = lons.dtype
        else:
            raise ValueError(('%s must be created with either '
                              'lon/lats of the same shape with same dtype') %
                             self.__class__.__name__)

    def concatenate(self, other):
        if self.ndim != other.ndim:
            raise DimensionError(('Unable to concatenate %sD and %sD '
                                  'geometries') % (self.ndim, other.ndim))
        klass = _get_highest_level_class(self, other)
        lons = np.concatenate((self.lons, other.lons))
        lats = np.concatenate((self.lats, other.lats))
        nprocs = min(self.nprocs, other.nprocs)
        return klass(lons, lats, nprocs=nprocs)

    def append(self, other):
        if self.ndim != other.ndim:
            raise DimensionError(('Unable to append %sD and %sD '
                                  'geometries') % (self.ndim, other.ndim))
        self.lons = np.concatenate((self.lons, other.lons))
        self.lats = np.concatenate((self.lats, other.lats))
        self.shape = self.lons.shape
        self.size = self.lons.size

    def __str__(self):
        # Rely on numpy's object printing
        return ('Shape: %s\nLons: %s\nLats: %s') % (str(self.shape),
                                                    str(self.lons),
                                                    str(self.lats))


class GridDefinition(CoordinateDefinition):

    """Grid defined by lons and lats

    Parameters
    ----------
    lons : numpy array
    lats : numpy array
    nprocs : int, optional
        Number of processor cores to be used for calculations.

    Attributes
    ----------
    shape : tuple
        Grid shape as (rows, cols)
    size : int
        Number of elements in grid
    lons : object
        Grid lons
    lats : object
        Grid lats
    cartesian_coords : object
        Grid cartesian coordinates
    """

    def __init__(self, lons, lats, nprocs=1):
        super(GridDefinition, self).__init__(lons, lats, nprocs)
        if lons.shape != lats.shape:
            raise ValueError('lon and lat grid must have same shape')
        elif lons.ndim != 2:
            raise ValueError('2 dimensional lon lat grid expected')


class SwathDefinition(CoordinateDefinition):

    """Swath defined by lons and lats

    Parameters
    ----------
    lons : numpy array
    lats : numpy array
    nprocs : int, optional
        Number of processor cores to be used for calculations.

    Attributes
    ----------
    shape : tuple
        Swath shape
    size : int
        Number of elements in swath
    ndims : int
        Swath dimensions
    lons : object
        Swath lons
    lats : object
        Swath lats
    cartesian_coords : object
        Swath cartesian coordinates
    """

    def __init__(self, lons, lats, nprocs=1):
        super(SwathDefinition, self).__init__(lons, lats, nprocs)
        if lons.shape != lats.shape:
            raise ValueError('lon and lat arrays must have same shape')
        elif lons.ndim > 2:
            raise ValueError('Only 1 and 2 dimensional swaths are allowed')


class AreaDefinition(BaseDefinition):

    """Holds definition of an area.

    Parameters
    ----------
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

    Attributes
    ----------
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
                 area_extent, nprocs=1, lons=None, lats=None, dtype=np.float64):
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

        # Calculate projection coordinates of center of upper left pixel
        self.pixel_upper_left = \
            (float(area_extent[0]) +
             float(self.pixel_size_x) / 2,
             float(area_extent[3]) -
             float(self.pixel_size_y) / 2)

        # Pixel_offset defines the distance to projection center from origen (UL)
        # of image in units of pixels.
        self.pixel_offset_x = -self.area_extent[0] / self.pixel_size_x
        self.pixel_offset_y = self.area_extent[3] / self.pixel_size_y

        self.projection_x_coords = None
        self.projection_y_coords = None

        self.dtype = dtype

    def __str__(self):
        # We need a sorted dictionary for a unique hash of str(self)
        proj_dict = self.proj_dict
        proj_str = ('{' +
                    ', '.join(["'%s': '%s'" % (str(k), str(proj_dict[k]))
                               for k in sorted(proj_dict.keys())]) +
                    '}')
        return ('Area ID: %s\nName: %s\nProjection ID: %s\n'
                'Projection: %s\nNumber of columns: %s\nNumber of rows: %s\n'
                'Area extent: %s') % (self.area_id, self.name, self.proj_id,
                                      proj_str, self.x_size, self.y_size,
                                      self.area_extent)

    def create_areas_def(self):
        proj_dict = self.proj_dict
        proj_str = ','.join(["%s=%s" % (str(k), str(proj_dict[k])) for k in sorted(proj_dict.keys())])

        fmt = "REGION: {name} {{\n"
        fmt += "\tNAME:\t{name}\n"
        fmt += "\tPCS_ID:\t{area_id}\n"
        fmt += "\tPCS_DEF:\t{proj_str}\n"
        fmt += "\tXSIZE:\t{x_size}\n"
        fmt += "\tYSIZE:\t{y_size}\n"
        fmt += "\tAREA_EXTENT: {area_extent}\n}};\n"
        area_def_str = fmt.format(name=self.name, area_id=self.area_id, proj_str=proj_str,
                                x_size=self.x_size, y_size=self.y_size, area_extent=self.area_extent)
        return area_def_str

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


    def colrow2lonlat(self,cols,rows):
        """
        Return longitudes and latitudes for the given image columns
        and rows. Both scalars and arrays are supported.
        To be used with scarse data points instead of slices
        (see get_lonlats).
        """
        p = _spatial_mp.Proj(self.proj4_string)
        x = self.proj_x_coords
        y = self.proj_y_coords
        return p(y[y.size-cols], x[x.size-rows], inverse=True)


    def lonlat2colrow(self, lons, lats):
        """
        Return image columns and rows for the given longitudes
        and latitudes. Both scalars and arrays are supported.
        Same as get_xy_from_lonlat, renamed for convenience.
        """
        return self.get_xy_from_lonlat(lons, lats)


    def get_xy_from_lonlat(self, lon, lat):
        """Retrieve closest x and y coordinates (column, row indices) for the
        specified geolocation (lon,lat) if inside area. If lon,lat is a point a
        ValueError is raised if the return point is outside the area domain. If
        lon,lat is a tuple of sequences of longitudes and latitudes, a tuple of
        masked arrays are returned.

        :Input:

        lon : point or sequence (list or array) of longitudes
        lat : point or sequence (list or array) of latitudes

        :Returns:

        (x, y) : tuple of integer points/arrays
        """

        if isinstance(lon, list):
            lon = np.array(lon)
        if isinstance(lat, list):
            lat = np.array(lat)

        if ((isinstance(lon, np.ndarray) and
             not isinstance(lat, np.ndarray)) or
            (not isinstance(lon, np.ndarray) and
             isinstance(lat, np.ndarray))):
            raise ValueError("Both lon and lat needs to be of " +
                             "the same type and have the same dimensions!")

        if isinstance(lon, np.ndarray) and isinstance(lat, np.ndarray):
            if lon.shape != lat.shape:
                raise ValueError("lon and lat is not of the same shape!")

        pobj = _spatial_mp.Proj(self.proj4_string)
        upl_x = self.area_extent[0]
        upl_y = self.area_extent[3]
        xscale = abs(self.area_extent[2] -
                     self.area_extent[0]) / float(self.x_size)
        yscale = abs(self.area_extent[1] -
                     self.area_extent[3]) / float(self.y_size)

        xm_, ym_ = pobj(lon, lat)
        x__ = (xm_ - upl_x) / xscale
        y__ = (upl_y - ym_) / yscale

        if isinstance(x__, np.ndarray) and isinstance(y__, np.ndarray):
            mask = (((x__ < 0) | (x__ > self.x_size)) |
                    ((y__ < 0) | (y__ > self.y_size)))
            return (np.ma.masked_array(x__.astype('int'), mask=mask,
                                       fill_value=-1),
                    np.ma.masked_array(y__.astype('int'), mask=mask,
                                       fill_value=-1))
        else:
            if ((x__ < 0 or x__ > self.x_size) or
                    (y__ < 0 or y__ > self.y_size)):
                raise ValueError('Point outside area:( %f %f)' % (x__, y__))
            return int(x__), int(y__)


    def get_xy_from_proj_coords(self, xm_, ym_):
        """Retrieve closest x and y coordinates (column, row indices) for a 
        location specified with projection coordinates (xm_,ym_) in meters. 
        A ValueError is raised, if the return point is outside the area domain. If
        xm_,ym_ is a tuple of sequences of projection coordinates, a tuple of
        masked arrays are returned.

        :Input:
        xm_ : point or sequence (list or array) of x-coordinates in m (map projection)
        ym_ : point or sequence (list or array) of y-coordinates in m (map projection)

        :Returns:
        (x, y) : tuple of integer points/arrays
        """

        if isinstance(xm_, list):
            xm_ = np.array(xm_)
        if isinstance(ym_, list):
            ym_ = np.array(ym_)

        if ((isinstance(xm_, np.ndarray) and
             not isinstance(ym_, np.ndarray)) or
            (not isinstance(xm_, np.ndarray) and
             isinstance(ym_, np.ndarray))):
            raise ValueError("Both projection coordinates xm_ and ym_ needs to be of " +
                             "the same type and have the same dimensions!")

        if isinstance(xm_, np.ndarray) and isinstance(ym_, np.ndarray):
            if xm_.shape != ym_.shape:
                raise ValueError("projection coordinates xm_ and ym_ is not of the same shape!")

        upl_x = self.area_extent[0]
        upl_y = self.area_extent[3]
        xscale = abs(self.area_extent[2] -
                     self.area_extent[0]) / float(self.x_size)
        yscale = abs(self.area_extent[1] -
                     self.area_extent[3]) / float(self.y_size)

        x__ = (xm_ - upl_x) / xscale
        y__ = (upl_y - ym_) / yscale

        if isinstance(x__, np.ndarray) and isinstance(y__, np.ndarray):
            mask = (((x__ < 0) | (x__ > self.x_size)) |
                    ((y__ < 0) | (y__ > self.y_size)))
            return (np.ma.masked_array(x__.astype('int'), mask=mask,
                                       fill_value=-1),
                    np.ma.masked_array(y__.astype('int'), mask=mask,
                                       fill_value=-1))
        else:
            if ((x__ < 0 or x__ > self.x_size) or
                    (y__ < 0 or y__ > self.y_size)):
                raise ValueError('Point outside area:( %f %f)' % (x__, y__))
            return int(x__), int(y__)


    def get_lonlat(self, row, col):
        """Retrieves lon and lat values of single point in area grid

        Parameters
        ----------
        row : int
        col : int

        Returns
        -------
        (lon, lat) : tuple of floats
        """

        return self.get_lonlats(nprocs=None, data_slice=(row, col))

    def get_proj_coords(self, data_slice=None, cache=False, dtype=None):
        """Get projection coordinates of grid

        Parameters
        ----------
        data_slice : slice object, optional
            Calculate only coordinates for specified slice
        cache : bool, optional
            Store result the result. Requires data_slice to be None

        Returns
        -------
        (target_x, target_y) : tuple of numpy arrays
            Grids of area x- and y-coordinates in projection units
        """

        def get_val(val, sub_val, max):
            # Get value with substitution and wrapping
            if val is None:
                return sub_val
            else:
                if val < 0:
                    # Wrap index
                    return max + val
                else:
                    return val

        if self.projection_x_coords is not None and self.projection_y_coords is not None:
            # Projection coords are cached
            if data_slice is None:
                return self.projection_x_coords, self.projection_y_coords
            else:
                return self.projection_x_coords[data_slice], self.projection_y_coords[data_slice]

        is_single_value = False
        is_1d_select = False

        if dtype is None:
            dtype = self.dtype

        # create coordinates of local area as ndarrays
        if data_slice is None or data_slice == slice(None):
            # Full slice
            rows = self.y_size
            cols = self.x_size
            row_start = 0
            col_start = 0
        else:
            if isinstance(data_slice, slice):
                # Row slice
                row_start = get_val(data_slice.start, 0, self.y_size)
                col_start = 0
                rows = get_val(
                    data_slice.stop, self.y_size, self.y_size) - row_start
                cols = self.x_size
            elif isinstance(data_slice[0], slice) and isinstance(data_slice[1], slice):
                # Block slice
                row_start = get_val(data_slice[0].start, 0, self.y_size)
                col_start = get_val(data_slice[1].start, 0, self.x_size)
                rows = get_val(
                    data_slice[0].stop, self.y_size, self.y_size) - row_start
                cols = get_val(
                    data_slice[1].stop, self.x_size, self.x_size) - col_start
            elif isinstance(data_slice[0], slice):
                # Select from col
                is_1d_select = True
                row_start = get_val(data_slice[0].start, 0, self.y_size)
                col_start = get_val(data_slice[1], 0, self.x_size)
                rows = get_val(
                    data_slice[0].stop, self.y_size, self.y_size) - row_start
                cols = 1
            elif isinstance(data_slice[1], slice):
                # Select from row
                is_1d_select = True
                row_start = get_val(data_slice[0], 0, self.y_size)
                col_start = get_val(data_slice[1].start, 0, self.x_size)
                rows = 1
                cols = get_val(
                    data_slice[1].stop, self.x_size, self.x_size) - col_start
            else:
                # Single element select
                is_single_value = True

                row_start = get_val(data_slice[0], 0, self.y_size)
                col_start = get_val(data_slice[1], 0, self.x_size)

                rows = 1
                cols = 1

        # Calculate coordinates
        target_x = np.fromfunction(lambda i, j: (j + col_start) *
                                   self.pixel_size_x +
                                   self.pixel_upper_left[0],
                                   (rows,
                                    cols), dtype=dtype)

        target_y = np.fromfunction(lambda i, j:
                                   self.pixel_upper_left[1] -
                                   (i + row_start) * self.pixel_size_y,
                                   (rows,
                                    cols), dtype=dtype)

        if is_single_value:
            # Return single values
            target_x = float(target_x)
            target_y = float(target_y)
        elif is_1d_select:
            # Reshape to 1D array
            target_x = target_x.reshape((target_x.size,))
            target_y = target_y.reshape((target_y.size,))

        if cache and data_slice is None:
            # Cache the result if requested
            self.projection_x_coords = target_x
            self.projection_y_coords = target_y

        return target_x, target_y

    @property
    def proj_x_coords(self):
        return self.get_proj_coords(data_slice=(0, slice(None)))[0]

    @property
    def proj_y_coords(self):
        return self.get_proj_coords(data_slice=(slice(None), 0))[1]

    @property
    def outer_boundary_corners(self):
        """Returns the lon,lat of the outer edges of the corner points
        """
        from pyresample.spherical_geometry import Coordinate
        proj = _spatial_mp.Proj(**self.proj_dict)

        corner_lons, corner_lats = proj((self.area_extent[0], self.area_extent[2],
                                         self.area_extent[2], self.area_extent[0]),
                                        (self.area_extent[3], self.area_extent[3],
                                         self.area_extent[1], self.area_extent[1]),
                                        inverse=True)
        return [Coordinate(corner_lons[0], corner_lats[0]),
                Coordinate(corner_lons[1], corner_lats[1]),
                Coordinate(corner_lons[2], corner_lats[2]),
                Coordinate(corner_lons[3], corner_lats[3])]

    def get_lonlats(self, nprocs=None, data_slice=None, cache=False, dtype=None):
        """Returns lon and lat arrays of area.

        Parameters
        ----------
        nprocs : int, optional
            Number of processor cores to be used.
            Defaults to the nprocs set when instantiating object
        data_slice : slice object, optional
            Calculate only coordinates for specified slice
        cache : bool, optional
            Store result the result. Requires data_slice to be None

        Returns
        -------
        (lons, lats) : tuple of numpy arrays
            Grids of area lons and and lats
        """

        if dtype is None:
            dtype = self.dtype

        if self.lons is None or self.lats is None:
            #Data is not cached
            if nprocs is None:
                nprocs = self.nprocs

            # Proj.4 definition of target area projection
            if nprocs > 1:
                target_proj = _spatial_mp.Proj_MP(**self.proj_dict)
            else:
                target_proj = _spatial_mp.Proj(**self.proj_dict)

            # Get coordinates of local area as ndarrays
            target_x, target_y = self.get_proj_coords(
                data_slice=data_slice, dtype=dtype)

            # Get corresponding longitude and latitude values
            lons, lats = target_proj(target_x, target_y, inverse=True,
                                     nprocs=nprocs)
            lons = np.asanyarray(lons, dtype=dtype)
            lats = np.asanyarray(lats, dtype=dtype)

            if cache and data_slice is None:
                # Cache the result if requested
                self.lons = lons
                self.lats = lats

            # Free memory
            del(target_x)
            del(target_y)
        else:
            #Data is cached
            if data_slice is None:
                # Full slice
                lons = self.lons
                lats = self.lats
            else:
                lons = self.lons[data_slice]
                lats = self.lats[data_slice]

        return lons, lats

    @property
    def proj4_string(self):
        """Returns projection definition as Proj.4 string"""

        items = self.proj_dict.items()
        return '+' + ' +'.join([t[0] + '=' + t[1] for t in items])


def _get_slice(segments, shape):
    """Generator for segmenting a 1D or 2D array"""

    if not (1 <= len(shape) <= 2):
        raise ValueError('Cannot segment array of shape: %s' % str(shape))
    else:
        size = shape[0]
        slice_length = int(np.ceil(float(size) / segments))
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
