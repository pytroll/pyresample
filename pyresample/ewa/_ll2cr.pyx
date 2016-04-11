# Copyright (C) 2014 Space Science and Engineering Center (SSEC),
#  University of Wisconsin-Madison.
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# This file is part of the polar2grid software package. Polar2grid takes
# satellite observation data, remaps it, and writes it to a file format for
# input into another program.
# Documentation: http://www.ssec.wisc.edu/software/polar2grid/
#
#     Written by David Hoese    December 2014
#     University of Wisconsin-Madison
#     Space Science and Engineering Center
#     1225 West Dayton Street
#     Madison, WI  53706
#     david.hoese@ssec.wisc.edu
"""Map longitude/latitude points to column/rows of a grid.

:author:       David Hoese (davidh)
:contact:      david.hoese@ssec.wisc.edu
:organization: Space Science and Engineering Center (SSEC)
:copyright:    Copyright (c) 2014 University of Wisconsin SSEC. All rights reserved.
:date:         Jan 2014
:license:      GNU GPLv3
"""
__docformat__ = "restructuredtext en"

# cython _ll2cr.pyx
# Too compile: gcc -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing -L /opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/ -L /opt/local/lib/ -I /opt/local/Library/Frameworks/Python.framework/Versions/2.7/include/python2.7/ -I /opt/local/include/ -o _ll2cr.so _ll2cr.c -lpython2.7
# Second Try: gcc -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing -L /opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/ -L /opt/local/lib/ -I /opt/local/Library/Frameworks/Python.framework/Versions/2.7/include/python2.7/ -I /opt/local/include/ -I /opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/numpy/core/include/ -o _ll2cr.so _ll2cr.c -lpython2.7


# from polar2grid.proj import Proj
from pyproj import _proj, Proj
import numpy
cimport cython
from cpython cimport bool
cimport numpy
from libc.math cimport isnan

# DTYPE = numpy.float32
# DTYPE = numpy.float64
# ctypedef numpy.float32_t DTYPE_t

# column and rows can only be doubles for now until the PROJ.4 is linked directly so float->double casting can be done
# inside the loop
ctypedef fused cr_dtype:
    # numpy.float32_t
    numpy.float64_t

# @cython.boundscheck(False)
# @cython.wraparound(False)
# def minmax_float32(numpy.ndarray[DTYPE_t, ndim=2] arr):
#     cdef DTYPE_t fmin = arr[0, 0]
#     cdef DTYPE_t fmax = arr[0, 0]
#     cdef DTYPE_t val
#     cdef unsigned int row_max = arr.shape[0]
#     cdef unsigned int col_max = arr.shape[1]
#     cdef unsigned int x, y
#     for y in range(row_max):
#         for x in range(col_max):
#             val = arr[y, x]
#             # min = arr[y, x] if arr[y, x] < min else min
#             # max = arr[y, x] if arr[y, x] > max else max
#             if val < fmin:
#                 fmin = val
#             elif val > fmax:
#                 fmax = val
#
#     return fmin, fmax


class MyProj(Proj):
    """Custom class to make ll2cr projection work faster without compiling against the PROJ.4 library itself.

    THIS SHOULD NOT BE USED OUTSIDE OF LL2CR! It makes assumptions and has requirements that may not make sense outside
    of the ll2cr modules.
    """
    def __call__(self, lons, lats, **kwargs):
        if self.is_latlong():
            return lons, lats
        elif isinstance(lons, numpy.ndarray):
            # Because we are doing this we know that we are getting a double array
            inverse = kwargs.get('inverse', False)
            radians = kwargs.get('radians', False)
            errcheck = kwargs.get('errcheck', False)
            # call proj4 functions. inx and iny modified in place.
            if inverse:
                _proj.Proj._inv(self, lons, lats, radians=radians, errcheck=errcheck)
            else:
                _proj.Proj._fwd(self, lons, lats, radians=radians, errcheck=errcheck)
            # if inputs were lists, tuples or floats, convert back.
            return lons, lats
        else:
            return super(MyProj, self).__call__(lons, lats, **kwargs)


def projection_circumference(p):
    """Return the projection circumference if the projection is cylindrical. None is returned otherwise.

    Projections that are not cylindrical and centered on the globes axis can not easily have data cross the antimeridian
    of the projection.
    """
    lon0, lat0 = p(0, 0, inverse=True)
    lon1 = lon0 + 180.0
    lat1 = lat0 + 5.0
    x0, y0 = p(lon0, lat0)  # should result in zero or near zero
    x1, y1 = p(lon1, lat0)
    x2, y2 = p(lon1, lat1)
    if y0 != y1 or x1 != x2:
        return 0.0
    return abs(x1 - x0) * 2

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def ll2cr_dynamic(numpy.ndarray[cr_dtype, ndim=2] lon_arr, numpy.ndarray[cr_dtype, ndim=2] lat_arr,
                  cr_dtype fill_in, str proj4_definition,
                  double cell_width, double cell_height,
                  width=None, height=None,
                  origin_x=None, origin_y=None):
    """Project longitude and latitude points to column rows in the specified grid in place

    :param lon_arr: Numpy array of longitude floats
    :param lat_arr: Numpy array of latitude floats
    :param grid_info: dictionary of grid information (see below)
    :param fill_in: Fill value for input longitude and latitude arrays and used for output
    :returns: tuple(points_in_grid, cols_out, rows_out)

    The provided grid info must have the following parameters (optional grids mean dynamic):

        - proj4_definition
        - cell_width
        - cell_height
        - width (optional/None)
        - height (optional/None)
        - origin_x (optional/None)
        - origin_y (optional/None)

    Steps taken in this function:

        1. Convert (lon, lat) points to (X, Y) points in the projection space
        2. If grid is missing some parameters (dynamic grid), then fill them in
        3. Convert (X, Y) points to (column, row) points in the grid space
    """
    # pure python stuff for now
    p = MyProj(proj4_definition)
    # when we update this to not make copies we can probably just make this a view
    # rows_arr = numpy.empty_like(lat_arr)
    # cols_arr = numpy.empty_like(lon_arr)

    # Pyproj currently makes a copy so we don't have to do anything special here
    cdef tuple projected_tuple = p(lon_arr, lat_arr)
    cdef cr_dtype [:, ::1] rows_out = projected_tuple[1]
    cdef cr_dtype [:, ::1] cols_out = projected_tuple[0]
    cdef double proj_circum = projection_circumference(p)
    cdef unsigned int w
    cdef unsigned int h
    cdef double ox
    cdef double oy

    # indexes
    cdef unsigned int row
    cdef unsigned int col
    # index bounds
    cdef unsigned int num_rows = lon_arr.shape[0]
    cdef unsigned int num_cols = lon_arr.shape[1]
    cdef cr_dtype xmin = cols_out[0, 0]
    cdef cr_dtype xmax = cols_out[0, 0]
    cdef cr_dtype ymin = rows_out[0, 0]
    cdef cr_dtype ymax = rows_out[0, 0]
    cdef cr_dtype x_tmp
    cdef cr_dtype y_tmp
    cdef unsigned int points_in_grid = 0
    for row in range(num_rows):
        for col in range(num_cols):
            x_tmp = cols_out[row, col]
            y_tmp = rows_out[row, col]

            if x_tmp >= 1e30:
                # pyproj library should have set both x and y to the fill value
                # we technically don't ever check for the fill value, but if fill values are valid lon/lats then WTF
                continue
            elif x_tmp < xmin or isnan(xmin):
                xmin = x_tmp
            elif x_tmp > xmax or isnan(xmax) or xmax == 1e30:
                # Note: technically 2 valid points are required to get here if there are a lot of NaNs
                xmax = x_tmp

            if y_tmp < ymin or isnan(ymin):
                ymin = y_tmp
            elif y_tmp > ymax or isnan(ymax) or ymax == 1e30:
                # Note: technically 2 valid points are required to get here if there are a lot of NaNs
                ymax = y_tmp

    # Check if we cross the antimeridian
    if proj_circum != 0 and xmax - xmin >= proj_circum * .75:
        # xmax will increase, but we need to reset xmin so that it gets properly detected
        if xmin < 0:
            xmin = xmax
        for row in range(num_rows):
            for col in range(num_cols):
                x_tmp = cols_out[row, col]
                if x_tmp < 0:
                    x_tmp += proj_circum
                    cols_out[row, col] = x_tmp
                    # xmax won't increase unless we've added the circumference
                    if x_tmp > xmax:
                        xmax = x_tmp
                elif x_tmp >= 1e30:
                    continue
                elif x_tmp < xmin:
                    # xmin could change with any of the remaining entries
                    xmin = x_tmp

    if origin_x is None:
        # upper-left corner
        ox = xmin
        oy = ymax
    else:
        ox = origin_x
        oy = origin_y

    if width is None:
        w = int(abs((xmax - ox) / cell_width))
        h = int(abs((oy - ymin) / cell_height))
    else:
        w = width
        h = height

    for row in range(num_rows):
        for col in range(num_cols):
            x_tmp = cols_out[row, col]
            y_tmp = rows_out[row, col]
            if x_tmp >= 1e30:
                cols_out[row, col] = fill_in
                rows_out[row, col] = fill_in
                continue

            x_tmp = (x_tmp - ox) / cell_width
            y_tmp = (y_tmp - oy) / cell_height
            if x_tmp >= -1 and x_tmp <= w + 1 and y_tmp >= -1 and y_tmp <= h + 1:
                points_in_grid += 1
            cols_out[row, col] = x_tmp
            rows_out[row, col] = y_tmp

    # return points_in_grid, x_arr, y_arr
    return points_in_grid, lon_arr, lat_arr, ox, oy, w, h


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def ll2cr_static(numpy.ndarray[cr_dtype, ndim=2] lon_arr, numpy.ndarray[cr_dtype, ndim=2] lat_arr,
                      cr_dtype fill_in, str proj4_definition,
                      double cell_width, double cell_height,
                      unsigned int width, unsigned int height,
                      double origin_x, double origin_y):
    """Project longitude and latitude points to column rows in the specified grid in place

    :param lon_arr: Numpy array of longitude floats
    :param lat_arr: Numpy array of latitude floats
    :param fill_in: Fill value for input longitude and latitude arrays and used for output
    :param cell_width: Pixel resolution in the X direction in projection space
    :param cell_height: Pixel resolution in the Y direction in projection space
    :param width: Number of pixels in the X direction in the final output grid
    :param height: Number of pixels in the Y direction in the final output grid
    :param origin_x:
    :returns: tuple(points_in_grid, cols_out, rows_out)

    Steps taken in this function:

        1. Convert (lon, lat) points to (X, Y) points in the projection space
        2. Convert (X, Y) points to (column, row) points in the grid space

    """
    # pure python stuff for now
    p = MyProj(proj4_definition)

    # Pyproj currently makes a copy so we don't have to do anything special here
    cdef tuple projected_tuple = p(lon_arr, lat_arr)
    cdef cr_dtype [:, ::1] rows_out = projected_tuple[1]
    cdef cr_dtype [:, ::1] cols_out = projected_tuple[0]
    cdef double proj_circum = projection_circumference(p)

    # indexes
    cdef unsigned int row
    cdef unsigned int col
    # index bounds
    cdef unsigned int num_rows = lon_arr.shape[0]
    cdef unsigned int num_cols = lon_arr.shape[1]
    cdef cr_dtype x_tmp
    cdef cr_dtype y_tmp
    cdef unsigned int points_in_grid = 0

    for row in range(num_rows):
        for col in range(num_cols):
            x_tmp = cols_out[row, col]
            y_tmp = rows_out[row, col]
            if x_tmp >= 1e30:
                cols_out[row, col] = fill_in
                rows_out[row, col] = fill_in
                continue
            elif proj_circum != 0 and abs(x_tmp - origin_x) >= (0.75 * proj_circum):
                # if x is more than 75% around the projection space, it is probably crossing the anti-meridian
                x_tmp += proj_circum

            x_tmp = (x_tmp - origin_x) / cell_width
            y_tmp = (y_tmp - origin_y) / cell_height
            if x_tmp >= -1 and x_tmp <= width + 1 and y_tmp >= -1 and y_tmp <= height + 1:
                points_in_grid += 1
            cols_out[row, col] = x_tmp
            rows_out[row, col] = y_tmp

    # return points_in_grid, x_arr, y_arr
    return points_in_grid
