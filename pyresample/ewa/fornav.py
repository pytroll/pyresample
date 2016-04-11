#!/usr/bin/env python
# encoding: utf-8
# Copyright (C) 2014 Space Science and Engineering Center (SSEC),
# University of Wisconsin-Madison.
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
"""Elliptical weighted averaging (EWA) resampling algorithm

:author:       David Hoese (davidh)
:contact:      david.hoese@ssec.wisc.edu
:organization: Space Science and Engineering Center (SSEC)
:copyright:    Copyright (c) 2014 University of Wisconsin SSEC. All rights reserved.
:date:         Dec 2014
:license:      GNU GPLv3

"""
__docformat__ = "restructuredtext en"

import os
import sys
import logging

import numpy

from polar2grid.remap import ms2gt
import _fornav

try:
    import psutil
    get_free_memory = lambda: psutil.phymem_usage().free
except ImportError:
    psutil = None
    get_free_memory = lambda: None

DEFAULT_GROUP_SIZE = os.getenv("P2G_EWA_DEF_GROUP_SIZE", None)
GROUP_SIZE = os.getenv("P2G_EWA_GROUP_SIZE", None)


LOG = logging.getLogger(__name__)


def calculate_group_size(swath_cols, swath_rows, grid_cols, grid_rows, default_group_size=DEFAULT_GROUP_SIZE,
                         grid_multiplier=3, swath_multiplier=1, geo_multiplier=2, additional_used=0):
    """Split a swath scene in to reasonably sized groups based on shared geolocation.

    If `use_memory` is True and available memory is known then the groups are split based on number of bytes.

    Default items in a group is 5.
    """
    free_memory_bytes = get_free_memory()
    if free_memory_bytes is None:
        if default_group_size is None:
            return None
        return int(default_group_size)

    # Assumes input and output of 32-bit float type
    item_size = 4
    swath_size = swath_cols * swath_rows * item_size
    grid_size = grid_cols * grid_rows * item_size
    grid_effect = grid_size * grid_multiplier
    geo_effect = swath_size * geo_multiplier
    swath_effect = swath_size * swath_multiplier
    max_group_size = max(int((free_memory_bytes - geo_effect - additional_used) / (grid_effect + swath_effect)), 1)
    LOG.debug("Max group size calculated to be %d (free: %d, grid: %d, geo: %d, swath: %d)",
              max_group_size, free_memory_bytes, grid_effect, geo_effect, swath_effect)
    return max_group_size


def group_iter(input_arrays, swath_cols, swath_rows, input_dtype, output_arrays, grid_cols, grid_rows, group_size):
    ret_input_arrays = []
    ret_output_arrays = []

    for idx, (ia, oa) in enumerate(zip(input_arrays, output_arrays)):
        if isinstance(ia, (str, unicode)):
            ret_input_arrays.append(numpy.memmap(ia, shape=(swath_rows, swath_cols), dtype=input_dtype, mode='r'))
        else:
            ret_input_arrays.append(ia)

        # We iterate over this so that we only create output arrays when they are used
        if oa is None:
            ret_output_arrays.append(numpy.empty((grid_rows, grid_cols), dtype=ia.dtype))
            # we should return the numpy arrays in the main function since the user didn't provide any
            output_arrays[idx] = ret_output_arrays[-1]
        elif isinstance(oa, (str, unicode)):
            ret_output_arrays.append(numpy.memmap(oa, shape=(grid_rows, grid_cols), dtype=input_dtype, mode='w+'))
        else:
            ret_output_arrays.append(oa)

        if group_size is None or len(ret_input_arrays) >= group_size:
            LOG.debug("Yielding group of size %d because group size is %d", len(ret_input_arrays), group_size)
            yield tuple(ret_input_arrays), tuple(ret_output_arrays)
            ret_input_arrays = []
            ret_output_arrays = []

    if len(ret_input_arrays):
        LOG.debug("Yielding remaining group items to process for EWA resampling")
        yield tuple(ret_input_arrays), tuple(ret_output_arrays)


def fornav(cols_array, rows_array, rows_per_scan, input_arrays, input_dtype=None, input_fill=numpy.nan,
           output_arrays=None, output_fill=None, grid_cols=None, grid_rows=None,
           weight_count=10000, weight_min=0.01, weight_distance_max=1.0, weight_delta_max=10.0,
           weight_sum_min=-1.0, maximum_weight_mode=False, use_group_size=False):
    include_output = False

    if input_dtype is None:
        if isinstance(input_arrays[0], (str, unicode)):
            raise ValueError("Must provide `input_dtype` when using input filenames")
        input_dtype = [ia.dtype for ia in input_arrays if hasattr(ia, "dtype")]

    if isinstance(input_dtype, list):
        if not all(in_dtype == input_dtype[0] for in_dtype in input_dtype):
            raise ValueError("EWA remapping does not support multiple data types yet")
        input_dtype = input_dtype[0]
    if isinstance(input_fill, list):
        if not all(in_fill == input_fill[0] or (numpy.isnan(in_fill) and numpy.isnan(input_fill[0])) for in_fill in input_fill):
            raise ValueError("EWA remapping does not support multiple fill values yet")
        input_fill = input_fill[0]

    if output_arrays is None:
        include_output = True
        output_arrays = [None] * len(input_arrays)
        if grid_cols is None or grid_rows is None:
            raise ValueError("Must specify grid_cols and grid_rows when output_arrays is not specified")

    if grid_cols is None or grid_rows is None:
        shapes = [x.shape for x in output_arrays if x is not None and hasattr(x, "shape")]
        if not len(shapes):
            raise ValueError("Must specify grid_cols and grid_rows when output_arrays are filenames")
        grid_rows, grid_cols = shapes[0]

    if output_fill is None:
        output_fill = input_fill

    if use_group_size:
        if GROUP_SIZE is not None:
            group_size = GROUP_SIZE
        else:
            group_size = None
            # It seems like this could be a smart way of handling this is we were using multiprocessing, but
            # a lot of testing is required to verify that assumption. The proper way to handle this or make it faster
            # in general is to have parrallel operations inside fornav (OpenMP or OpenCL/GPU).
            # group_size = calculate_group_size(cols_array.shape[1], cols_array.shape[0], grid_cols, grid_rows)
    if group_size is None:
        group_size = len(input_arrays)

    valid_list = []
    for in_arrays, out_arrays in group_iter(input_arrays, cols_array.shape[1], cols_array.shape[0], input_dtype,
                                            output_arrays, grid_cols, grid_rows, group_size):
        LOG.debug("Processing %d of %d input arrays", len(in_arrays), len(input_arrays))
        tmp_valid_list = _fornav.fornav_wrapper(cols_array, rows_array, in_arrays, out_arrays,
                              input_fill, output_fill, rows_per_scan,
                              weight_count=weight_count, weight_min=weight_min, weight_distance_max=weight_distance_max,
                              weight_delta_max=weight_delta_max, weight_sum_min=weight_sum_min,
                              maximum_weight_mode=maximum_weight_mode)

        valid_list.extend(tmp_valid_list)

    if include_output:
        return valid_list, output_arrays
    else:
        return valid_list


def ms2gt_fornav(*args, **kwargs):
    """Run the ms2gt wrapper for fornav.

    This is how we use to run it from remap.py

    run_fornav_c(
        len(product_filepaths),
        swath_def["swath_columns"],
        swath_def["swath_rows"]/rows_per_scan,
        rows_per_scan,
        cols_fn,
        rows_fn,
        product_filepaths,
        grid_def["width"],
        grid_def["height"],
        fornav_filepaths,
        swath_data_type_1="f4",
        swath_fill_1=swath_scene.get_fill_value(product_names),
        grid_fill_1=numpy.nan,
        weight_delta_max=fornav_D,
        weight_distance_max=kwargs.get("fornav_d", None),
        maximum_weight_mode=kwargs.get("maximum_weight_mode", None),
        # We only specify start_scan for the 'image'/channel
        # data because ll2cr is not 'forced' so it only writes
        # useful data to the output cols/rows files
        start_scan=(0, 0),
        )
    """
    return ms2gt.fornav(*args, **kwargs)



def main():
    pass


if __name__ == "__main__":
    sys.exit(main())
