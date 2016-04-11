#!/usr/bin/env python
# encoding: utf-8
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

import logging

import numpy

from polar2grid.core.proj import Proj
import _ll2cr


LOG = logging.getLogger(__name__)


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
        return None
    return abs(x1 - x0) * 2


def mask_helper(arr, fill):
    if numpy.isnan(fill):
        return numpy.isnan(arr)
    else:
        return arr == fill


def ll2cr(lon_arr, lat_arr, grid_info, fill_in=numpy.nan, inplace=True):
    """Project longitude and latitude points to columns and rows in the specified grid in place

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
    p = grid_info["proj4_definition"]
    cw = grid_info["cell_width"]
    ch = grid_info["cell_height"]
    w = grid_info.get("width", None)
    h = grid_info.get("height", None)
    ox = grid_info.get("origin_x", None)
    oy = grid_info.get("origin_y", None)
    is_static = None not in [w, h, ox, oy]

    # C code requires 64-bit floats and to do in place it must be writeable, otherwise we need to make a copy
    is_double = lon_arr.dtype == numpy.float64 and lon_arr.dtype == numpy.float64
    is_writeable = lon_arr.flags.writeable and lat_arr.flags.writeable
    lon_orig = lon_arr
    lat_orig = lat_arr
    copy_result = False
    # XXX: Check that automatic casting can't be done on the C side
    if not is_double or not is_writeable:
        LOG.debug("Copying longitude and latitude arrays because inplace processing could not be done")
        copy_result = True
        lon_arr = lon_arr.astype(numpy.float64)
        lat_arr = lat_arr.astype(numpy.float64)

    if is_static:
        LOG.debug("Running static version of ll2cr...")
        points_in_grid = _ll2cr.ll2cr_static(lon_arr, lat_arr, fill_in, p, cw, ch, w, h, ox, oy)
    else:
        LOG.debug("Running dynamic version of ll2cr...")
        results = _ll2cr.ll2cr_dynamic(lon_arr, lat_arr, fill_in, p, cw, ch,
                                    width=w, height=h, origin_x=ox, origin_y=oy)
        points_in_grid, lon_arr, lat_arr, origin_x, origin_y, width, height = results
        # edit the grid info dictionary in place
        grid_info["origin_x"] = origin_x
        grid_info["origin_y"] = origin_y
        grid_info["width"] = width
        grid_info["height"] = height

    if copy_result and inplace:
        LOG.debug("Copying result arrays back to provided inplace array")
        lon_orig[:] = lon_arr[:]
        lat_orig[:] = lat_arr[:]

    return points_in_grid, lon_orig, lat_orig


def python_ll2cr(lon_arr, lat_arr, grid_info, fill_in=numpy.nan, fill_out=None, cols_out=None, rows_out=None):
    """Project longitude and latitude points to column rows in the specified grid.

    :param lon_arr: Numpy array of longitude floats
    :param lat_arr: Numpy array of latitude floats
    :param grid_info: dictionary of grid information (see below)
    :param fill_in: (optional) Fill value for input longitude and latitude arrays (default: NaN)
    :param fill_out: (optional) Fill value for output column and row array (default: `fill_in`)
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
    p = Proj(grid_info["proj4_definition"])
    cw = grid_info["cell_width"]
    ch = grid_info["cell_height"]
    w = grid_info.get("width", None)
    h = grid_info.get("height", None)
    ox = grid_info.get("origin_x", None)
    oy = grid_info.get("origin_y", None)
    is_static = None not in [w, h, ox, oy]
    proj_circum = projection_circumference(p)

    if rows_out is None:
        rows_out = numpy.empty_like(lat_arr)
    if cols_out is None:
        cols_out = numpy.empty_like(lon_arr)
    if fill_out is None:
        fill_out = fill_in

    mask = ~(mask_helper(lon_arr, fill_in) | mask_helper(lat_arr, fill_in))
    x, y = p(lon_arr, lat_arr)
    mask = mask & (x < 1e30) & (y < 1e30)
    # need temporary storage because x and y are might NOT be copies (latlong projections)
    cols_out[:] = numpy.where(mask, x, fill_out)
    rows_out[:] = numpy.where(mask, y, fill_out)
    # we only need the good Xs and Ys from here on out
    x = cols_out[mask]
    y = rows_out[mask]

    if not is_static:
        # fill in grid parameters
        xmin = numpy.nanmin(x)
        xmax = numpy.nanmax(x)
        ymin = numpy.nanmin(y)
        ymax = numpy.nanmax(y)
        # if the data seems to be covering more than 75% of the projection space then the antimeridian is being crossed
        # if proj_circum is None then we can't simply wrap the data around projection, the grid will probably be large
        LOG.debug("Projection circumference: %f", proj_circum)
        if proj_circum is not None and xmax - xmin >= proj_circum * .75:
            old_xmin = xmin
            old_xmax = xmax
            x[x < 0] += proj_circum
            xmin = numpy.nanmin(x)
            xmax = numpy.nanmax(x)
            LOG.debug("Data seems to cross the antimeridian: old_xmin=%f; old_xmax=%f; xmin=%f; xmax=%f", old_xmin, old_xmax, xmin, xmax)
        LOG.debug("Xmin=%f; Xmax=%f; Ymin=%f; Ymax=%f", xmin, xmax, ymin, ymax)

        if ox is None:
            # upper-left corner
            ox = grid_info["origin_x"] = float(xmin)
            oy = grid_info["origin_y"] = float(ymax)
            LOG.debug("Dynamic grid origin (%f, %f)", xmin, ymax)
        if w is None:
            w = grid_info["width"] = int(abs((xmax - xmin) / cw))
            h = grid_info["height"] = int(abs((ymax - ymin) / ch))
            LOG.debug("Dynamic grid width and height (%d x %d) with cell width and height (%f x %f)", w, h, cw, ch)

    good_cols = (x - ox) / cw
    good_rows = (y - oy) / ch
    cols_out[mask] = good_cols
    rows_out[mask] = good_rows

    points_in_grid = numpy.count_nonzero((good_cols >= -1) & (good_cols <= w + 1) & (good_rows >= -1) & (good_rows <= h + 1))

    return points_in_grid, cols_out, rows_out
