#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2020
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""EWA algorithms operating on numpy arrays."""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

# Pyresample does not have type definition for all of it's cython modules
from pyresample.ewa import _fornav, _ll2cr  # type: ignore

LOG = logging.getLogger(__name__)


def ll2cr(swath_def, area_def, fill=np.nan, copy=True):
    """Map input swath pixels to output grid column and rows.

    Parameters
    ----------
    swath_def : pyresample.geometry.SwathDefinition
        Navigation definition for swath data to remap
    area_def : pyresample.geometry.AreaDefinition
        Grid definition to be mapped to
    fill : float, optional
        Fill value used in longitude and latitude arrays
    copy : bool, optional
        Create a copy of the longitude and latitude arrays (default: True)

    Returns
    -------
    (swath_points_in_grid, cols, rows) : tuple of integer, numpy array, numpy array
        Number of points from the input swath overlapping the destination
        area and the column and row arrays to pass to `fornav`.

    .. note::

        ll2cr uses the pyproj library which is limited to 64-bit float
        navigation arrays in order to not do additional copying or casting
        of data types.

    """
    lons, lats = swath_def.get_lonlats()
    # ll2cr requires 64-bit floats due to pyproj limitations
    # also need a copy of lons, lats since they are written to in-place
    try:
        lons = lons.astype(np.float64, copy=copy)
        lats = lats.astype(np.float64, copy=copy)
    except TypeError:
        lons = lons.astype(np.float64)
        lats = lats.astype(np.float64)

    # Break the input area up in to the expected parameters for ll2cr
    p = area_def.crs_wkt if hasattr(area_def, 'crs_wkt') else area_def.proj_str
    cw = area_def.pixel_size_x
    # cell height must be negative for this to work as expected
    ch = -abs(area_def.pixel_size_y)
    w = area_def.width
    h = area_def.height
    ox = area_def.area_extent[0] + cw / 2.
    oy = area_def.area_extent[3] + ch / 2.
    swath_points_in_grid = _ll2cr.ll2cr_static(lons, lats, fill,
                                               p, cw, ch, w, h, ox, oy)
    return swath_points_in_grid, lons, lats


def fornav(cols, rows, area_def, data_in,
           rows_per_scan=None, fill=None, out=None,
           weight_count=10000, weight_min=0.01, weight_distance_max=1.0,
           weight_delta_max=10.0, weight_sum_min=-1.0,
           maximum_weight_mode=False):
    """Remap data in to output grid using elliptical weighted averaging.

    This algorithm works under the assumption that the data is observed
    one scan line at a time. However, good results can still be achieved
    for non-scan based data is provided if `rows_per_scan` is set to the
    number of rows in the entire swath or by setting it to `None`.

    Parameters
    ----------
    cols : numpy array
        Column location for each input swath pixel (from `ll2cr`)
    rows : numpy array
        Row location for each input swath pixel (from `ll2cr`)
    area_def : pyresample.geometry.AreaDefinition
        Grid definition to be mapped to
    data_in : numpy array or tuple of numpy arrays
        Swath data to be remapped to output grid
    rows_per_scan : int or None, optional
        Number of data rows for every observed scanline. If None then the
        entire swath is treated as one large scanline.
    fill : float/int or None, optional
        If `data_in` is made of numpy arrays then this represents the fill
        value used to mark invalid data pixels. This value will also be
        used in the output array(s). If None, then np.nan will be used
        for float arrays and -999 will be used for integer arrays.
    out : numpy array or tuple of numpy arrays, optional
        Specify a numpy array to be written to for each input array. This can
        be used as an optimization by providing `np.memmap` arrays or other
        array-like objects.
    weight_count : int, optional
        number of elements to create in the gaussian weight table.
        Default is 10000. Must be at least 2
    weight_min : float, optional
        the minimum value to store in the last position of the
        weight table. Default is 0.01, which, with a
        `weight_distance_max` of 1.0 produces a weight of 0.01
        at a grid cell distance of 1.0. Must be greater than 0.
    weight_distance_max : float, optional
        distance in grid cell units at which to
        apply a weight of `weight_min`. Default is
        1.0. Must be greater than 0.
    weight_delta_max : float, optional
        maximum distance in grid cells in each grid
        dimension over which to distribute a single swath cell.
        Default is 10.0.
    weight_sum_min : float, optional
        minimum weight sum value. Cells whose weight sums
        are less than `weight_sum_min` are set to the grid fill value.
        Default is EPSILON.
    maximum_weight_mode : bool, optional
        If False (default), a weighted average of
        all swath cells that map to a particular grid cell is used.
        If True, the swath cell having the maximum weight of all
        swath cells that map to a particular grid cell is used. This
        option should be used for coded/category data, i.e. snow cover.

    Returns
    -------
    (valid grid points, output arrays): tuple of integer tuples and numpy array tuples
        The valid_grid_points tuple holds the number of output grid pixels that
        were written with valid data. The second element in the tuple is a tuple of
        output grid numpy arrays for each input array. If there was only one input
        array provided then the returned tuple is simply the singe points integer
        and single output grid array.
    """
    data_in, convert_to_masked, fill = _data_in_as_masked_arrays(data_in, fill)

    if out is not None:
        # the user may have provided memmapped arrays or other array-like objects
        if isinstance(out, (tuple, list)):
            out = tuple(out)
        else:
            out = (out,)
    else:
        # create a place for output data to be written
        out = tuple(np.empty(area_def.shape, dtype=in_arr.dtype)
                    for in_arr in data_in)

    # see if the user specified rows per scan
    # otherwise, use the entire swath as one "scanline"
    rows_per_scan = rows_per_scan or data_in[0].shape[0]

    results = _fornav.fornav_wrapper(cols, rows, data_in, out,
                                     np.nan, np.nan, rows_per_scan,
                                     weight_count=weight_count,
                                     weight_min=weight_min,
                                     weight_distance_max=weight_distance_max,
                                     weight_delta_max=weight_delta_max,
                                     weight_sum_min=weight_sum_min,
                                     maximum_weight_mode=maximum_weight_mode)

    if convert_to_masked:
        # they gave us masked arrays so give them masked arrays back
        out = [np.ma.masked_where(_mask_helper(out_arr, fill), out_arr)
               for out_arr in out]
    if len(out) == 1:
        # they only gave us one data array as input, so give them one back
        out = out[0]
        results = results[0]

    return results, out


def _data_in_as_masked_arrays(
        data_in: Any,
        fill: float | int | None
) -> tuple[tuple[np.ma.MaskedArray, ...], bool, float | int]:
    if isinstance(data_in, (tuple, list)):
        # we can only support one data type per call at this time
        for in_arr in data_in[1:]:
            if in_arr.dtype != data_in[0].dtype:
                raise ValueError("All input arrays must be the same dtype")
    else:
        # assume they gave us a single numpy array-like object
        data_in = [data_in]

    # need a list for replacing these arrays later
    data_in = [np.ascontiguousarray(d) for d in data_in]
    # determine a fill value if they didn't tell us what they have as a
    # fill value in the numpy arrays
    if fill is None:
        if np.issubdtype(data_in[0].dtype, np.floating):
            fill = np.nan
        elif np.issubdtype(data_in[0].dtype, np.integer):
            fill = -999
        else:
            raise ValueError(
                "Unsupported input data type for EWA Resampling: {}".format(data_in[0].dtype))

    convert_to_masked = False
    for idx, in_arr in enumerate(data_in):
        if isinstance(in_arr, np.ma.MaskedArray):
            convert_to_masked = True
            # convert masked arrays to single numpy arrays
            data_in[idx] = in_arr.filled(fill)
    return tuple(data_in), convert_to_masked, fill


def _mask_helper(data, fill):
    if np.isnan(fill):
        return np.isnan(data)
    return data == fill
