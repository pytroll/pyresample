import cython
cimport cython
import numpy
cimport numpy
cimport cpython
from libc.stdlib cimport calloc, malloc, free
from libc.math cimport log, exp, sqrt, isnan, NAN

cdef extern from "_fornav_templates.h":
    ctypedef float weight_type
    ctypedef float ewa_param_type
    ctypedef float accum_type
    cdef float EPSILON
    #ctypedef double weight_type
    #ctypedef double ewa_param_type
    #ctypedef double accum_type
    #cdef double EPSILON

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
        weight_type *wtab

    cdef int initialize_weight(size_t chan_count, unsigned int weight_count, weight_type weight_min, weight_type weight_distance_max,
        weight_type weight_delta_max, weight_type weight_sum_min, ewa_weight *ewaw)
    cdef void deinitialize_weight(ewa_weight *ewaw)
    cdef accum_type **initialize_grid_accums(size_t chan_count, size_t grid_cols, size_t grid_rows)
    cdef weight_type **initialize_grid_weights(size_t chan_count, size_t grid_cols, size_t grid_rows)
    cdef void deinitialize_grids(size_t chan_count, void **grids)

    cdef int compute_ewa_parameters[CR_TYPE](size_t swath_cols, size_t swath_rows,
        CR_TYPE *uimg, CR_TYPE *vimg, ewa_weight *ewaw, ewa_parameters *ewap)

    cdef int compute_ewa[CR_TYPE, IMAGE_TYPE](size_t chan_count, bint maximum_weight_mode,
        size_t swath_cols, size_t swath_rows, size_t grid_cols, size_t grid_rows,
        CR_TYPE *uimg, CR_TYPE *vimg,
        IMAGE_TYPE **images, IMAGE_TYPE img_fill, accum_type **grid_accums, weight_type **grid_weights,
        ewa_weight *ewaw, ewa_parameters *ewap)

    # For some reason cython can't deduce the type when using the template
    #cdef int write_grid_image[GRID_TYPE](GRID_TYPE *output_image, GRID_TYPE fill, size_t grid_cols, size_t grid_rows,
    #    accum_type *grid_accum, weight_type *grid_weights,
    #    int maximum_weight_mode, weight_type weight_sum_min)
    cdef unsigned int write_grid_image(numpy.float32_t *output_image, numpy.float32_t fill, size_t grid_cols, size_t grid_rows,
        accum_type *grid_accum, weight_type *grid_weights,
        int maximum_weight_mode, weight_type weight_sum_min)
    cdef unsigned int write_grid_image(numpy.float64_t *output_image, numpy.float64_t fill, size_t grid_cols, size_t grid_rows,
        accum_type *grid_accum, weight_type *grid_weights,
        int maximum_weight_mode, weight_type weight_sum_min)
    cdef unsigned int write_grid_image(numpy.int8_t *output_image, numpy.int8_t fill, size_t grid_cols, size_t grid_rows,
        accum_type *grid_accum, weight_type *grid_weights,
        int maximum_weight_mode, weight_type weight_sum_min)

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
cdef int fornav(unsigned int *valid_list, size_t chan_count, size_t swath_cols, size_t swath_rows, size_t grid_cols, size_t grid_rows,
            cr_dtype *cols_pointer, cr_dtype *rows_pointer,
           image_dtype **input_arrays, grid_dtype **output_arrays,
           image_dtype input_fill, grid_dtype output_fill, size_t rows_per_scan,
           unsigned int weight_count, weight_type weight_min, weight_type weight_distance_max, weight_type weight_delta_max,
           weight_type weight_sum_min, bint maximum_weight_mode) except -1:
    cdef unsigned int row_idx
    cdef unsigned int idx
    cdef bint got_point = 0
    cdef bint tmp_got_point
    cdef int func_result
    cdef cr_dtype *tmp_cols_pointer
    cdef cr_dtype *tmp_rows_pointer
    cdef image_dtype **input_images
    cdef ewa_weight ewaw
    cdef ewa_parameters *ewap

    # other defaults
    if weight_sum_min == -1.0:
        weight_sum_min = weight_min

    func_result = initialize_weight(chan_count, weight_count, weight_min, weight_distance_max, weight_delta_max,
                      weight_sum_min, &ewaw)
    if func_result < 0:
        raise RuntimeError("Could not initialize weight structure for EWA resampling")

    # Allocate location for storing the sum of all of the pixels involved in each grid cell
    # XXX: Do these need to be initialized to a fill value?
    cdef accum_type **grid_accums = initialize_grid_accums(chan_count, grid_cols, grid_rows)
    if grid_accums is NULL:
        raise MemoryError()
    cdef weight_type **grid_weights = initialize_grid_weights(chan_count, grid_cols, grid_rows)
    if grid_weights is NULL:
        raise MemoryError()
    # Allocate memory for the parameters specific to each column
    ewap = <ewa_parameters *>malloc(swath_cols * sizeof(ewa_parameters))
    if ewap is NULL:
        raise MemoryError()
    # Allocate pointers to the correct portion of the data arrays that we will use
    input_images = <image_dtype **>malloc(chan_count * sizeof(image_dtype *))
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
        func_result = compute_ewa_parameters(swath_cols, rows_per_scan, tmp_cols_pointer, tmp_rows_pointer, &ewaw, ewap)
        if func_result < 0:
            got_point = got_point or 0
            # raise RuntimeError("Could compute EWA parameters for EWA resampling")
            continue

        # NOTE: In the C version this is where the image array data is loaded
        tmp_got_point = compute_ewa(chan_count, maximum_weight_mode,
                    swath_cols, rows_per_scan, grid_cols, grid_rows,
                    tmp_cols_pointer, tmp_rows_pointer,
                    input_images, input_fill, grid_accums, grid_weights, &ewaw, ewap)

        got_point = got_point or tmp_got_point

    free(input_images)
    free(ewap)

    if not got_point:
        raise RuntimeError("EWA Resampling: No swath pixels found inside grid to be resampled")

    for idx in range(chan_count):
        valid_list[idx] = write_grid_image(output_arrays[idx], output_fill, grid_cols, grid_rows,
                                          grid_accums[idx], grid_weights[idx], maximum_weight_mode, weight_sum_min)

    # free(grid_accums)
    deinitialize_weight(&ewaw)
    deinitialize_grids(chan_count, <void **>grid_accums)
    deinitialize_grids(chan_count, <void **>grid_weights)

    return 0

@cython.boundscheck(False)
@cython.wraparound(False)
def fornav_wrapper(numpy.ndarray[cr_dtype, ndim=2, mode='c'] cols_array,
           numpy.ndarray[cr_dtype, ndim=2, mode='c'] rows_array,
           tuple input_arrays, tuple output_arrays, input_fill, output_fill,
           size_t rows_per_scan,
           unsigned int weight_count=10000, weight_type weight_min=0.01, weight_type weight_distance_max=1.0, weight_type weight_delta_max=10.0, weight_type weight_sum_min=-1.0,
           cpython.bool maximum_weight_mode=False):
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

    cdef void **input_pointer = <void **>malloc(num_items * sizeof(void *))
    if not input_pointer:
        raise MemoryError()
    cdef void **output_pointer = <void **>malloc(num_items * sizeof(void *))
    if not output_pointer:
        raise MemoryError()
    cdef unsigned int *valid_arr = <unsigned int *>malloc(num_items * sizeof(unsigned int))
    valid_list = []
    cdef numpy.ndarray[numpy.float32_t, ndim=2] tmp_arr_f32
    cdef numpy.ndarray[numpy.float64_t, ndim=2] tmp_arr_f64
    cdef numpy.ndarray[numpy.int8_t, ndim=2] tmp_arr_i8
    cdef cr_dtype *cols_pointer = &cols_array[0, 0]
    cdef cr_dtype *rows_pointer = &rows_array[0, 0]
    cdef int func_result

    if in_type == numpy.float32:
        for i in range(num_items):
            tmp_arr_f32 = input_arrays[i]
            input_pointer[i] = &tmp_arr_f32[0, 0]
            tmp_arr_f32 = output_arrays[i]
            output_pointer[i] = &tmp_arr_f32[0, 0]
        func_result = fornav(valid_arr, num_items, swath_cols, swath_rows, grid_cols, grid_rows, cols_pointer, rows_pointer,
                     <numpy.float32_t **>input_pointer, <numpy.float32_t **>output_pointer,
                     <numpy.float32_t>input_fill, <numpy.float32_t>output_fill, rows_per_scan,
                     weight_count, weight_min, weight_distance_max, weight_delta_max, weight_sum_min,
                     <bint>maximum_weight_mode)
    elif in_type == numpy.float64:
        for i in range(num_items):
            tmp_arr_f64 = input_arrays[i]
            input_pointer[i] = &tmp_arr_f64[0, 0]
            tmp_arr_f64 = output_arrays[i]
            output_pointer[i] = &tmp_arr_f64[0, 0]
        func_result = fornav(valid_arr, num_items, swath_cols, swath_rows, grid_cols, grid_rows, cols_pointer, rows_pointer,
                     <numpy.float64_t **>input_pointer, <numpy.float64_t **>output_pointer,
                     <numpy.float64_t>input_fill, <numpy.float64_t>output_fill, rows_per_scan,
                     weight_count, weight_min, weight_distance_max, weight_delta_max, weight_sum_min,
                     <bint>maximum_weight_mode)
    elif in_type == numpy.int8:
        for i in range(num_items):
            tmp_arr_i8 = input_arrays[i]
            input_pointer[i] = &tmp_arr_i8[0, 0]
            tmp_arr_i8 = output_arrays[i]
            output_pointer[i] = &tmp_arr_i8[0, 0]
        func_result = fornav(valid_arr, num_items, swath_cols, swath_rows, grid_cols, grid_rows, cols_pointer, rows_pointer,
                     <numpy.int8_t **>input_pointer, <numpy.int8_t **>output_pointer,
                     <numpy.int8_t>input_fill, <numpy.int8_t>output_fill, rows_per_scan,
                     weight_count, weight_min, weight_distance_max, weight_delta_max, weight_sum_min,
                     <bint>maximum_weight_mode)
    else:
        raise ValueError("Unknown input and output data type")

    for i in range(num_items):
        valid_list.append(valid_arr[i])

    free(input_pointer)
    free(output_pointer)

    return valid_list
