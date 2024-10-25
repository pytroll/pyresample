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
"""Resampling swath data to uniform grid using an Elliptical Weighted Averaging
(EWA) algorithm."""
from libc.stdlib cimport calloc, free, malloc

import numpy

__docformat__ = "restructuredtext en"

import cython

cimport cpython
cimport cython
cimport numpy

numpy.import_array()

cdef extern from "_fornav_templates.h":
    ctypedef float weight_type
    ctypedef float ewa_param_type
    ctypedef float accum_type
    cdef float EPSILON

    ctypedef struct ewa_parameters:
        ewa_param_type a
        ewa_param_type b
        ewa_param_type c
        ewa_param_type f
        ewa_param_type u_del
        ewa_param_type v_del

    ctypedef struct ewa_weight:
        int count
        weight_type min
        weight_type distance_max
        weight_type delta_max
        weight_type sum_min
        weight_type alpha
        weight_type qmax
        weight_type qfactor
        weight_type * wtab

    cdef int initialize_weight(size_t chan_count, unsigned int weight_count, weight_type weight_min, weight_type weight_distance_max,
                               weight_type weight_delta_max, weight_type weight_sum_min, ewa_weight * ewaw) noexcept nogil
    cdef void deinitialize_weight(ewa_weight * ewaw) nogil
    cdef accum_type ** initialize_grid_accums(size_t chan_count, size_t grid_cols, size_t grid_rows) noexcept nogil
    cdef weight_type ** initialize_grid_weights(size_t chan_count, size_t grid_cols, size_t grid_rows) noexcept nogil
    cdef void deinitialize_grids(size_t chan_count, void ** grids) noexcept nogil

    cdef int compute_ewa_parameters[CR_TYPE](size_t swath_cols, size_t swath_rows,
                                             CR_TYPE * uimg, CR_TYPE * vimg, ewa_weight * ewaw, ewa_parameters * ewap) noexcept nogil

    cdef int compute_ewa[CR_TYPE, IMAGE_TYPE](
        size_t chan_count, bint maximum_weight_mode,
        size_t swath_cols, size_t swath_rows, size_t grid_cols, size_t grid_rows,
        CR_TYPE * uimg, CR_TYPE * vimg,
        IMAGE_TYPE ** images, IMAGE_TYPE img_fill, accum_type ** grid_accums, weight_type ** grid_weights,
        ewa_weight * ewaw, ewa_parameters * ewap) noexcept nogil

    cdef int compute_ewa_single[CR_TYPE, IMAGE_TYPE](
        bint maximum_weight_mode,
        size_t swath_cols, size_t swath_rows, size_t grid_cols, size_t grid_rows,
        CR_TYPE * uimg, CR_TYPE * vimg,
        IMAGE_TYPE * image, IMAGE_TYPE img_fill, accum_type * grid_accum, weight_type * grid_weight,
        ewa_weight * ewaw, ewa_parameters * ewap) noexcept nogil

    # For some reason cython can't deduce the type when using the template
    # cdef int write_grid_image[GRID_TYPE](GRID_TYPE *output_image, GRID_TYPE fill, size_t grid_cols, size_t grid_rows,
    #    accum_type *grid_accum, weight_type *grid_weights,
    #    int maximum_weight_mode, weight_type weight_sum_min)
    cdef unsigned int write_grid_image(numpy.float32_t * output_image, numpy.float32_t fill, size_t grid_cols, size_t grid_rows,
                                       accum_type * grid_accum, weight_type * grid_weights,
                                       int maximum_weight_mode, weight_type weight_sum_min) noexcept nogil
    cdef unsigned int write_grid_image(numpy.float64_t * output_image, numpy.float64_t fill, size_t grid_cols, size_t grid_rows,
                                       accum_type * grid_accum, weight_type * grid_weights,
                                       int maximum_weight_mode, weight_type weight_sum_min) noexcept nogil
    cdef unsigned int write_grid_image(numpy.int8_t * output_image, numpy.int8_t fill, size_t grid_cols, size_t grid_rows,
                                       accum_type * grid_accum, weight_type * grid_weights,
                                       int maximum_weight_mode, weight_type weight_sum_min) noexcept nogil

ctypedef fused cr_dtype:
    numpy.float32_t
    numpy.float64_t

# FUTURE: Add other types, but for now these are the basics
ctypedef fused image_dtype:
    numpy.float32_t
    numpy.float64_t
    numpy.int8_t

ctypedef fused grid_dtype:
    numpy.float32_t
    numpy.float64_t
    numpy.int8_t


@cython.boundscheck(False)
@cython.wraparound(False)
cdef int fornav(unsigned int * valid_list, size_t chan_count, size_t swath_cols, size_t swath_rows, size_t grid_cols, size_t grid_rows,
                cr_dtype * cols_pointer, cr_dtype * rows_pointer,
                image_dtype ** input_arrays, grid_dtype ** output_arrays,
                image_dtype input_fill, grid_dtype output_fill, size_t rows_per_scan,
                unsigned int weight_count, weight_type weight_min, weight_type weight_distance_max, weight_type weight_delta_max,
                weight_type weight_sum_min, bint maximum_weight_mode) nogil except -1:
    cdef unsigned int row_idx
    cdef unsigned int idx
    cdef bint got_point = 0
    cdef bint tmp_got_point
    cdef int func_result
    cdef cr_dtype * tmp_cols_pointer
    cdef cr_dtype * tmp_rows_pointer
    cdef image_dtype ** input_images
    cdef ewa_weight ewaw
    cdef ewa_parameters * ewap

    # other defaults
    if weight_sum_min == -1.0:
        weight_sum_min = weight_min

    func_result = initialize_weight(chan_count, weight_count, weight_min, weight_distance_max, weight_delta_max,
                                    weight_sum_min, & ewaw)
    if func_result < 0:
        raise RuntimeError("Could not initialize weight structure for EWA resampling")

    # Allocate location for storing the sum of all of the pixels involved in each grid cell
    # XXX: Do these need to be initialized to a fill value?
    cdef accum_type ** grid_accums = initialize_grid_accums(chan_count, grid_cols, grid_rows)
    if grid_accums is NULL:
        raise MemoryError()
    cdef weight_type ** grid_weights = initialize_grid_weights(chan_count, grid_cols, grid_rows)
    if grid_weights is NULL:
        raise MemoryError()
    # Allocate memory for the parameters specific to each column
    ewap = <ewa_parameters * >malloc(swath_cols * sizeof(ewa_parameters))
    if ewap is NULL:
        raise MemoryError()
    # Allocate pointers to the correct portion of the data arrays that we will use
    input_images = <image_dtype ** >malloc(chan_count * sizeof(image_dtype *))
    if input_images is NULL:
        raise MemoryError()

    # NOTE: Have to use old school pyrex for loop because cython only supports compile-time known steps
    for row_idx from 0 <= row_idx < swath_rows by rows_per_scan:
        tmp_cols_pointer = &cols_pointer[row_idx * swath_cols]
        tmp_rows_pointer = &rows_pointer[row_idx * swath_cols]
        # print "Current cols pointer: %d" % (<int>tmp_cols_pointer,)

        # Assign the python/numpy array objects to a pointer location for the rest of the functions
        for idx in range(chan_count):
            input_images[idx] = &input_arrays[idx][row_idx * swath_cols]
        # print "Current input 0 pointer: %d" % (<int>input_images[idx],)

        # Calculate EWA parameters for each column index
        func_result = compute_ewa_parameters(swath_cols, rows_per_scan, tmp_cols_pointer, tmp_rows_pointer, & ewaw, ewap)
        if func_result < 0:
            got_point = got_point or 0
            # raise RuntimeError("Could compute EWA parameters for EWA resampling")
            continue

        # NOTE: In the C version this is where the image array data is loaded
        tmp_got_point = compute_ewa(chan_count, maximum_weight_mode,
                                    swath_cols, rows_per_scan, grid_cols, grid_rows,
                                    tmp_cols_pointer, tmp_rows_pointer,
                                    input_images, input_fill, grid_accums, grid_weights, & ewaw, ewap)

        got_point = got_point or tmp_got_point

    free(input_images)
    free(ewap)

    if not got_point:
        raise RuntimeError("EWA Resampling: No swath pixels found inside grid to be resampled")

    for idx in range(chan_count):
        valid_list[idx] = write_grid_image(output_arrays[idx], output_fill, grid_cols, grid_rows,
                                           grid_accums[idx], grid_weights[idx], maximum_weight_mode, weight_sum_min)

    # free(grid_accums)
    deinitialize_weight(& ewaw)
    deinitialize_grids(chan_count, < void ** >grid_accums)
    deinitialize_grids(chan_count, < void ** >grid_weights)

    return 0


@cython.boundscheck(False)
@cython.wraparound(False)
def fornav_wrapper(numpy.ndarray[cr_dtype, ndim=2, mode='c'] cols_array,
                   numpy.ndarray[cr_dtype, ndim=2, mode='c'] rows_array,
                   tuple input_arrays, tuple output_arrays, input_fill, output_fill,
                   size_t rows_per_scan,
                   unsigned int weight_count=10000, weight_type weight_min=0.01, weight_type weight_distance_max=1.0, weight_type weight_delta_max=10.0, weight_type weight_sum_min=-1.0,
                   cpython.bool maximum_weight_mode=False):
    """Python wrapper around the C interface to fornav.

    The main difficulty is that the C code can operate on multiple input
    arrays, but the python->C cython interface doesn't automatically know
    the data type of the numpy arrays inside a tuple. This function casts
    the array pointers to the corresponding C type and then calls the
    templated cython and C++ functions.

    This algorithm works under the assumption that the data is observed
    one scan line at a time. However, good results can still be achieved
    for non-scan based data is provided if `rows_per_scan` is set to the
    number of rows in the entire swath.

    :param cols_array: numpy array of grid column coordinates for each input swath pixel
    :param rows_array: numpy array of grid row coordinates for each input swath pixel
    :param input_arrays: tuple of numpy arrays with the same geolocation and same dtype
    :param output_arrays: tuple of empty numpy arrays to be filled with gridded data (must be writeable)
    :param input_fill: fill value in input data arrays representing invalid/bad values
    :param output_fill: fill value written to output data arrays representing invalid/bad or out of swath values
    :param rows_per_scan: number of input swath rows making up one scan line or all of the rows in the swath
    :param weight_count: number of elements to create in the gaussian weight
                         table. Default is 10000. Must be at least 2
    :param weight_min: the minimum value to store in the last position of the
                       weight table. Default is 0.01, which, with a
                       `weight_distance_max` of 1.0 produces a weight of 0.01
                       at a grid cell distance of 1.0. Must be greater than 0.
    :param weight_distance_max: distance in grid cell units at which to
                                apply a weight of `weight_min`. Default is
                                1.0. Must be greater than 0.
    :param weight_delta_max: maximum distance in grid cells in each grid
             dimension over which to distribute a single swath cell.
             Default is 10.0.
    :param weight_sum_min: minimum weight sum value. Cells whose weight sums
             are less than `weight_sum_min` are set to the grid fill value.
             Default is EPSILON.
    :param maximum_weight_mode: If -m is not present, a weighted average of
             all swath cells that map to a particular grid cell is used.
             If -m is present, the swath cell having the maximum weight of all
             swath cells that map to a particular grid cell is used. The -m
             option should be used for coded data, i.e. snow cover.
    :return: tuple of valid grid points written for each output array
    """
    cdef size_t num_items = len(input_arrays)
    cdef size_t num_outputs = len(output_arrays)
    cdef size_t swath_cols = cols_array.shape[1]
    cdef size_t swath_rows = cols_array.shape[0]
    cdef size_t grid_cols = output_arrays[0].shape[1]
    cdef size_t grid_rows = output_arrays[0].shape[0]
    cdef unsigned int i
    if num_items != num_outputs:
        raise ValueError("Must have same number of inputs and outputs")
    if num_items <= 0:
        raise ValueError("No input arrays given")
    if rows_per_scan < 2 or swath_rows % rows_per_scan != 0:
        raise ValueError("EWA requires 2 or more rows_per_scan and must be a factor of the total number of input rows")

    cdef numpy.dtype in_type = input_arrays[0].dtype
    cdef numpy.dtype out_type = output_arrays[0].dtype
    if in_type != out_type:
        raise ValueError("Input and Output must be of the same type")
    if not all(input_array.dtype == in_type for input_array in input_arrays):
        raise ValueError("Input arrays must all be of the same data type")
    if not all(output_array.dtype == out_type for output_array in output_arrays):
        raise ValueError("Input arrays must all be of the same data type")

    cdef void** input_pointer = <void ** >malloc(num_items * sizeof(void * ))
    if not input_pointer:
        raise MemoryError()
    cdef void** output_pointer = <void ** >malloc(num_items * sizeof(void * ))
    if not output_pointer:
        raise MemoryError()
    cdef unsigned int * valid_arr = <unsigned int * >malloc(num_items * sizeof(unsigned int))
    valid_list = []
    cdef cr_dtype * cols_pointer = &cols_array[0, 0]
    cdef cr_dtype * rows_pointer = &rows_array[0, 0]
    cdef bint mwm = maximum_weight_mode
    cdef int func_result
    cdef numpy.float32_t input_fill_f32
    cdef numpy.float64_t input_fill_f64
    cdef numpy.int8_t input_fill_i8
    cdef numpy.float32_t output_fill_f32
    cdef numpy.float64_t output_fill_f64
    cdef numpy.int8_t output_fill_i8
    cdef numpy.ndarray[numpy.float32_t, ndim= 2] tmp_arr_f32
    cdef numpy.ndarray[numpy.float64_t, ndim= 2] tmp_arr_f64
    cdef numpy.ndarray[numpy.int8_t, ndim= 2] tmp_arr_i8
    cdef cr_dtype[:, ::1] tmp_arr

    if in_type == numpy.float32:
        input_fill_f32 = <numpy.float32_t>input_fill
        output_fill_f32 = <numpy.float32_t>output_fill
        for i in range(num_items):
            tmp_arr_f32 = input_arrays[i]
            input_pointer[i] = &tmp_arr_f32[0, 0]
            tmp_arr_f32 = output_arrays[i]
            output_pointer[i] = &tmp_arr_f32[0, 0]
        with nogil:
            func_result = fornav(valid_arr, num_items, swath_cols, swath_rows, grid_cols, grid_rows,
                                 cols_pointer, rows_pointer,
                                 < numpy.float32_t ** >input_pointer, < numpy.float32_t ** >output_pointer,
                                 output_fill_f32, output_fill_f32, rows_per_scan,
                                 weight_count, weight_min, weight_distance_max, weight_delta_max, weight_sum_min,
                                 mwm)
    elif in_type == numpy.float64:
        input_fill_f64 = <numpy.float64_t>input_fill
        output_fill_f64 = <numpy.float64_t>output_fill
        for i in range(num_items):
            tmp_arr_f64 = input_arrays[i]
            input_pointer[i] = &tmp_arr_f64[0, 0]
            tmp_arr_f64 = output_arrays[i]
            output_pointer[i] = &tmp_arr_f64[0, 0]
        with nogil:
            func_result = fornav(valid_arr, num_items, swath_cols, swath_rows, grid_cols, grid_rows,
                                 cols_pointer, rows_pointer,
                                 < numpy.float64_t ** >input_pointer, < numpy.float64_t ** >output_pointer,
                                 input_fill_f64, output_fill_f64, rows_per_scan,
                                 weight_count, weight_min, weight_distance_max, weight_delta_max, weight_sum_min,
                                 mwm)
    elif in_type == numpy.int8:
        input_fill_i8 = <numpy.int8_t>input_fill
        output_fill_i8 = <numpy.int8_t>output_fill
        for i in range(num_items):
            tmp_arr_i8 = input_arrays[i]
            input_pointer[i] = &tmp_arr_i8[0, 0]
            tmp_arr_i8 = output_arrays[i]
            output_pointer[i] = &tmp_arr_i8[0, 0]
        with nogil:
            func_result = fornav(valid_arr, num_items, swath_cols, swath_rows, grid_cols, grid_rows,
                                 cols_pointer, rows_pointer,
                                 < numpy.int8_t ** >input_pointer, < numpy.int8_t ** >output_pointer,
                                 input_fill_i8, output_fill_i8, rows_per_scan,
                                 weight_count, weight_min, weight_distance_max, weight_delta_max, weight_sum_min,
                                 mwm)
    else:
        raise ValueError("Unknown input and output data type")

    for i in range(num_items):
        valid_list.append(valid_arr[i])

    free(input_pointer)
    free(output_pointer)

    return valid_list


@cython.boundscheck(False)
@cython.wraparound(False)
cdef int fornav_weights_and_sums(
        size_t swath_cols, size_t swath_rows, size_t grid_cols, size_t grid_rows,
        cr_dtype * cols_pointer, cr_dtype * rows_pointer,
        image_dtype * input_array, weight_type * grid_weights, accum_type * grid_accums,
        image_dtype input_fill, grid_dtype output_fill, size_t rows_per_scan,
        unsigned int weight_count, weight_type weight_min, weight_type weight_distance_max, weight_type weight_delta_max,
        weight_type weight_sum_min, bint maximum_weight_mode) nogil except -1:
    """Get the weights and sums arrays from the fornav algorithm.

    Typically fornav performs the entire operation of computing the weights
    and sums and then using those to compute the final average. This function
    only performs the first couple stages of this process and returns the
    intermediate weights and sums. This is primarily structured this way to
    allow higher-level dask functions to split data in to chunks and perform
    the averaging at a later stage.

    """
    cdef unsigned int row_idx
    cdef unsigned int idx
    cdef bint got_point = 0
    cdef bint tmp_got_point
    cdef int func_result
    cdef cr_dtype * tmp_cols_pointer
    cdef cr_dtype * tmp_rows_pointer
    cdef image_dtype * tmp_img_pointer
    cdef ewa_weight ewaw
    cdef ewa_parameters * ewap

    # other defaults
    if weight_sum_min == -1.0:
        weight_sum_min = weight_min

    func_result = initialize_weight(1, weight_count, weight_min, weight_distance_max, weight_delta_max,
                                    weight_sum_min, & ewaw)
    if func_result < 0:
        raise RuntimeError("Could not initialize weight structure for EWA resampling")

    # Allocate memory for the parameters specific to each column
    ewap = <ewa_parameters * >malloc(swath_cols * sizeof(ewa_parameters))
    if ewap is NULL:
        raise MemoryError()

    # NOTE: Have to use old school pyrex for loop because cython only supports compile-time known steps
    for row_idx from 0 <= row_idx < swath_rows by rows_per_scan:
        tmp_cols_pointer = &cols_pointer[row_idx * swath_cols]
        tmp_rows_pointer = &rows_pointer[row_idx * swath_cols]
        tmp_img_pointer = &input_array[row_idx * swath_cols]
        # print "Current cols pointer: %d" % (<int>tmp_cols_pointer,)

        # Calculate EWA parameters for each column index
        func_result = compute_ewa_parameters(swath_cols, rows_per_scan, tmp_cols_pointer, tmp_rows_pointer, & ewaw, ewap)
        if func_result < 0:
            got_point = got_point or 0
            # raise RuntimeError("Could compute EWA parameters for EWA resampling")
            continue

        # NOTE: In the C version this is where the image array data is loaded
        tmp_got_point = compute_ewa_single(
            maximum_weight_mode,
            swath_cols, rows_per_scan, grid_cols, grid_rows,
            tmp_cols_pointer, tmp_rows_pointer,
            tmp_img_pointer, input_fill, grid_accums, grid_weights, & ewaw, ewap)

        got_point = got_point or tmp_got_point

    free(ewap)
    deinitialize_weight(& ewaw)
    if not got_point:
        raise RuntimeError("EWA Resampling: No swath pixels found inside grid to be resampled")
    # -1 is raised on exception, 0 otherwise
    return 0


@cython.boundscheck(False)
@cython.wraparound(False)
def fornav_weights_and_sums_wrapper(numpy.ndarray[cr_dtype, ndim=2, mode='c'] cols_array,
                                    numpy.ndarray[cr_dtype, ndim=2, mode='c'] rows_array,
                                    numpy.ndarray[image_dtype, ndim=2, mode='c'] input_array,
                                    numpy.ndarray[weight_type, ndim=2, mode='c'] grid_weights,
                                    numpy.ndarray[accum_type, ndim=2, mode='c'] grid_accums,
                                    image_dtype input_fill, grid_dtype output_fill,
                                    size_t rows_per_scan,
                                    unsigned int weight_count=10000, weight_type weight_min=0.01, weight_type weight_distance_max=1.0, weight_type weight_delta_max=10.0, weight_type weight_sum_min=-1.0,
                                    cpython.bool maximum_weight_mode=False):
    """Python wrapper around the C interface to fornav weights and sums steps.

    The main difficulty is that the C code can operate on multiple input
    arrays, but the python->C cython interface doesn't automatically know
    the data type of the numpy arrays inside a tuple. This function casts
    the array pointers to the corresponding C type and then calls the
    templated cython and C++ functions.

    This algorithm works under the assumption that the data is observed
    one scan line at a time. However, good results can still be achieved
    for non-scan based data is provided if `rows_per_scan` is set to the
    number of rows in the entire swath.

    :param cols_array: numpy array of grid column coordinates for each input swath pixel
    :param rows_array: numpy array of grid row coordinates for each input swath pixel
    :param input_array: numpy array with the same geolocation and same dtype
    :param grid_weights: numpy array to be filled with the sum of weights data (must be writeable)
    :param grid_accums: numpy array to be filled with sum of weight * image pixel data (must be writeable)
    :param input_fill: fill value in input data arrays representing invalid/bad values
    :param output_fill: fill value written to output data arrays representing invalid/bad or out of swath values
    :param rows_per_scan: number of input swath rows making up one scan line or all of the rows in the swath
    :param weight_count: number of elements to create in the gaussian weight
                         table. Default is 10000. Must be at least 2
    :param weight_min: the minimum value to store in the last position of the
                       weight table. Default is 0.01, which, with a
                       `weight_distance_max` of 1.0 produces a weight of 0.01
                       at a grid cell distance of 1.0. Must be greater than 0.
    :param weight_distance_max: distance in grid cell units at which to
                                apply a weight of `weight_min`. Default is
                                1.0. Must be greater than 0.
    :param weight_delta_max: maximum distance in grid cells in each grid
             dimension over which to distribute a single swath cell.
             Default is 10.0.
    :param weight_sum_min: minimum weight sum value. Cells whose weight sums
             are less than `weight_sum_min` are set to the grid fill value.
             Default is EPSILON.
    :param maximum_weight_mode: If -m is not present, a weighted average of
             all swath cells that map to a particular grid cell is used.
             If -m is present, the swath cell having the maximum weight of all
             swath cells that map to a particular grid cell is used. The -m
             option should be used for coded data, i.e. snow cover.
    :return: boolean if any input data was used on a any output grid cell
    """
    cdef size_t swath_cols = cols_array.shape[1]
    cdef size_t swath_rows = cols_array.shape[0]
    cdef size_t grid_cols = grid_weights.shape[1]
    cdef size_t grid_rows = grid_weights.shape[0]
    cdef unsigned int i
    if rows_per_scan < 2 or swath_rows % rows_per_scan != 0:
        raise ValueError("EWA requires 2 or more rows_per_scan and must be a factor of the total number of input rows")

    cdef cr_dtype * cols_pointer = &cols_array[0, 0]
    cdef cr_dtype * rows_pointer = &rows_array[0, 0]
    cdef image_dtype * input_pointer = &input_array[0, 0]
    cdef weight_type * weights_pointer = &grid_weights[0, 0]
    cdef accum_type * accums_pointer = &grid_accums[0, 0]
    cdef int got_point
    cdef bint mwm = maximum_weight_mode

    with nogil:
        ret_val = fornav_weights_and_sums(swath_cols, swath_rows, grid_cols, grid_rows, cols_pointer, rows_pointer,
                                          input_pointer, weights_pointer, accums_pointer,
                                          input_fill, output_fill, rows_per_scan,
                                          weight_count, weight_min, weight_distance_max, weight_delta_max, weight_sum_min,
                                          mwm)

    succeeded = ret_val == 0
    return succeeded


@cython.boundscheck(False)
@cython.wraparound(False)
def write_grid_image_single(numpy.ndarray[grid_dtype, ndim=2, mode='c'] output_array,
                            numpy.ndarray[weight_type, ndim=2, mode='c'] grid_weights,
                            numpy.ndarray[accum_type, ndim=2, mode='c'] grid_accums,
                            grid_dtype output_fill,
                            weight_type weight_sum_min=-1.0,
                            cpython.bool maximum_weight_mode=False):
    cdef int mwm = <int>maximum_weight_mode
    cdef size_t grid_cols = <size_t>output_array.shape[1]
    cdef size_t grid_rows = <size_t>output_array.shape[0]
    cdef unsigned int result
    with nogil:
        result = write_grid_image(& output_array[0, 0], output_fill, grid_cols, grid_rows,
                                  & grid_accums[0, 0], & grid_weights[0, 0], mwm, weight_sum_min)
    return result
