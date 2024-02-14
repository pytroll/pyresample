#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2016

# Author(s):

#   David Hoese <david.hoese@ssec.wisc.edu>

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""Map longitude/latitude points to column/rows of a grid."""
__docformat__ = "restructuredtext en"

import numpy
from pyproj import Proj

cimport cython
cimport numpy

numpy.import_array()

# column and rows can only be doubles for now until the PROJ.4 is linked directly so float->double casting can be done
# inside the loop
ctypedef fused cr_dtype:
    # numpy.float32_t
    numpy.float64_t

cdef extern from "numpy/npy_math.h":
    bint npy_isnan(double x)


def projection_circumference(p):
    """Return the projection circumference if the projection is cylindrical, None otherwise.

    Projections that are not cylindrical and centered on the globes axis
    can not easily have data cross the antimeridian of the projection.
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
    """Project longitude and latitude points to column rows in the specified grid in place.

    This function is meant to operate on dynamic grids and is theoretically
    slower than the `ll2cr_static` function. Dynamic grids are those that
    have one or more set of parameters unspecified like the number of pixels
    in the grid (width, height) or the origin coordinates of the grid
    (origin_x, origin_y). This function will analyze the provided lon/lat data
    and determine what the missing parameters must be so all input swath data
    falls in the resulting grid.

    :param lon_arr: Numpy array of longitude floats
    :param lat_arr: Numpy array of latitude floats
    :param grid_info: dictionary of grid information (see below)
    :param fill_in: Fill value for input longitude and latitude arrays and used for output
    :param proj4_definition: PROJ.4 string projection definition
    :param cell_width: Pixel resolution in the X direction in projection space
    :param cell_height: Pixel resolution in the Y direction in projection space
    :param width: (optional) Number of pixels in the X direction in the final output grid
    :param height: (optional) Number of pixels in the Y direction in the final output grid
    :param origin_x: (optional) Grid X coordinate for the upper-left pixel of the output grid
    :param origin_y: (optional) Grid Y coordinate for the upper-left pixel of the output grid
    :returns: tuple(points_in_grid, cols_out, rows_out, origin_x, origin_y, width, height)

    Steps taken in this function:

        1. Convert (lon, lat) points to (X, Y) points in the projection space
        2. If grid is missing some parameters (dynamic grid), then fill them in
        3. Convert (X, Y) points to (column, row) points in the grid space

    Note longitude and latitude arrays are limited to 64-bit floats because
    of limitations in pyproj.
    """
    # pure python stuff for now
    p = Proj(proj4_definition)

    # Pyproj currently makes a copy so we don't have to do anything special here
    cdef tuple projected_tuple = p(lon_arr, lat_arr)
    cdef cr_dtype[:, ::1] rows_out = projected_tuple[1]
    cdef cr_dtype[:, ::1] cols_out = projected_tuple[0]
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
            elif x_tmp < xmin or npy_isnan(xmin):
                xmin = x_tmp
            elif x_tmp > xmax or npy_isnan(xmax) or xmax >= 1e30:
                # Note: technically 2 valid points are required to get here if there are a lot of NaNs
                xmax = x_tmp

            if y_tmp < ymin or npy_isnan(ymin):
                ymin = y_tmp
            elif y_tmp > ymax or npy_isnan(ymax) or ymax >= 1e30:
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
                lon_arr[row, col] = fill_in
                lat_arr[row, col] = fill_in
                continue

            x_tmp = (x_tmp - ox) / cell_width
            y_tmp = (y_tmp - oy) / cell_height
            if x_tmp >= -1 and x_tmp <= w + 1 and y_tmp >= -1 and y_tmp <= h + 1:
                points_in_grid += 1
            lon_arr[row, col] = x_tmp
            lat_arr[row, col] = y_tmp

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
    """Project longitude and latitude points to column rows in the specified grid in place.

    :param lon_arr: Numpy array of longitude floats
    :param lat_arr: Numpy array of latitude floats
    :param fill_in: Fill value for input longitude and latitude arrays and used for output
    :param cell_width: Pixel resolution in the X direction in projection space
    :param cell_height: Pixel resolution in the Y direction in projection space
    :param width: Number of pixels in the X direction in the final output grid
    :param height: Number of pixels in the Y direction in the final output grid
    :param origin_x: Grid X coordinate for the upper-left pixel of the output grid
    :param origin_y: Grid Y coordinate for the upper-left pixel of the output grid
    :returns: tuple(points_in_grid, cols_out, rows_out)

    Steps taken in this function:

        1. Convert (lon, lat) points to (X, Y) points in the projection space
        2. Convert (X, Y) points to (column, row) points in the grid space

    Note longitude and latitude arrays are limited to 64-bit floats because
    of limitations in pyproj.
    """
    # TODO: Rewrite so it is no GIL
    # pure python stuff for now
    p = Proj(proj4_definition)

    # Pyproj currently makes a copy so we don't have to do anything special here
    cdef tuple projected_tuple = p(lon_arr, lat_arr)
    cdef cr_dtype[:, ::1] rows_out = projected_tuple[1]
    cdef cr_dtype[:, ::1] cols_out = projected_tuple[0]
    cdef cr_dtype[:, ::1] lons_view = lon_arr
    cdef cr_dtype[:, ::1] lats_view = lat_arr

    # indexes
    cdef unsigned int row
    cdef unsigned int col
    # index bounds
    cdef unsigned int num_rows = lons_view.shape[0]
    cdef unsigned int num_cols = lons_view.shape[1]
    cdef cr_dtype x_tmp
    cdef cr_dtype y_tmp
    cdef unsigned int points_in_grid = 0

    with nogil:
        for row in range(num_rows):
            for col in range(num_cols):
                x_tmp = cols_out[row, col]
                y_tmp = rows_out[row, col]
                if x_tmp >= 1e30:
                    lons_view[row, col] = fill_in
                    lats_view[row, col] = fill_in
                    continue

                x_tmp = (x_tmp - origin_x) / cell_width
                y_tmp = (y_tmp - origin_y) / cell_height
                if x_tmp >= -1 and x_tmp <= width + 1 and y_tmp >= -1 and y_tmp <= height + 1:
                    points_in_grid += 1
                lons_view[row, col] = x_tmp
                lats_view[row, col] = y_tmp

    return points_in_grid
