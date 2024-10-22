#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2013-2022

# Author(s):

#   Martin Raspaud <martin.raspaud@smhi.se>
#   Leon Majewski  <leon.majewski@bom.gov.au>

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

import numpy as np

cimport cython
cimport numpy as np
from libc.math cimport fabs, isinf

ctypedef fused data_type:
    np.float64_t
    np.float32_t

ctypedef np.float64_t float_index
float_index_dtype = np.float64

np.import_array()

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void nn(const data_type[:, :, :] data, int l0, int p0, float_index dl, float_index dp, int lmax, int pmax, data_type[:] res) noexcept nogil:
    cdef int nnl, nnp
    cdef size_t z_size = res.shape[0]
    cdef size_t i
    nnl = l0
    if dl < -0.5 and nnl > 0:
        nnl -= 1
    elif dl > 0.5 and nnl < lmax:
        nnl += 1
    nnp = p0
    if dp < -0.5 and nnp > 0:
        nnp -= 1
    elif dp > 0.5 and nnp < pmax:
        nnp += 1
    for i in range(z_size):
        res[i] = data[i, nnl, nnp]


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void bil(const data_type[:, :, :] data, int l0, int p0, float_index dl, float_index dp, int lmax, int pmax, data_type[:] res) noexcept nogil:
    cdef int l_a, l_b, p_a, p_b
    cdef float_index w_l, w_p
    cdef size_t z_size = res.shape[0]
    cdef size_t i
    if dl < 0:
        l_a = max(0, l0 - 1)
        l_b = l0
        w_l = 1 + dl
    else:
        l_a = l0
        l_b = min(l0 + 1, lmax)
        w_l = dl
    if dp < 0:
        p_a = max(0, p0 - 1)
        p_b = p0
        w_p = 1 + dp
    else:
        p_a = p0
        p_b = min(p0 + 1, pmax)
        w_p = dp
    for i in range(z_size):
        res[i] = <data_type>((1 - w_l) * (1 - w_p) * data[i, l_a, p_a] +
                             (1 - w_l) * w_p * data[i, l_a, p_b] +
                             w_l * (1 - w_p) * data[i, l_b, p_a] +
                             w_l * w_p * data[i, l_b, p_b])


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void indices_xy(const data_type[:, :, :] data, int l0, int p0, float_index dl, float_index dp, int lmax, int pmax, data_type[:] res) noexcept nogil:
    cdef int nnl, nnp
    cdef size_t z_size = res.shape[0]
    cdef size_t i
    res[1] = dl + l0
    res[0] = dp + p0


ctypedef void (*FN)(const data_type[:, :, :] data, int l0, int p0, float_index dl, float_index dp, int lmax, int pmax, data_type[:] res) noexcept nogil


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef one_step_gradient_search(const data_type[:, :, :] data,
                               float_index [:, :] src_x,
                               float_index [:, :] src_y,
                               float_index [:, :] xl,
                               float_index [:, :] xp,
                               float_index [:, :] yl,
                               float_index [:, :] yp,
                               float_index [:, :] dst_x,
                               float_index [:, :] dst_y,
                               str method='bilinear'):
    """Gradient search, simple case variant."""
    cdef FN fun
    if method == 'bilinear':
        fun = bil
    else:
        fun = nn

    # change the output size (x_size, y_size) to match area_def.shape:
    # (lines,pixels)
    cdef size_t z_size = data.shape[0]
    cdef size_t y_size = dst_y.shape[0]
    cdef size_t x_size = dst_x.shape[1]

    if data_type is double:
        dtype = np.float64
    else:
        dtype = np.float32

    # output image array --> needs to be (lines, pixels) --> y,x
    image = np.full([z_size, y_size, x_size], np.nan, dtype=dtype)
    cdef data_type[:, :, :] image_view = image
    with nogil:
        one_step_gradient_search_no_gil(data,
                                        src_x, src_y,
                                        xl, xp, yl, yp,
                                        dst_x, dst_y,
                                        x_size, y_size,
                                        fun, image_view)
    # return the output image
    return image


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void one_step_gradient_search_no_gil(const data_type[:, :, :] data,
                                          const float_index[:, :] src_x,
                                          const float_index[:, :] src_y,
                                          const float_index[:, :] xl,
                                          const float_index[:, :] xp,
                                          const float_index[:, :] yl,
                                          const float_index[:, :] yp,
                                          const float_index[:, :] dst_x,
                                          const float_index[:, :] dst_y,
                                          const size_t x_size,
                                          const size_t y_size,
                                          FN fun,
                                          data_type[:, :, :] result_array) noexcept nogil:

    # pixel max ---> data is expected in [lines, pixels]
    cdef int pmax = src_x.shape[1] - 1
    cdef int lmax = src_x.shape[0] - 1
    # centre of input image - starting point
    cdef int p0 = pmax // 2
    cdef int l0 = lmax // 2
    cdef int last_p0 = p0
    cdef int last_l0 = l0

    # intermediate variables:
    cdef int l_a, l_b, p_a, p_b
    cdef size_t i, j, elt
    cdef float_index dx, dy, d, dl, dp
    cdef int col_step = -1
    # number of iterations
    cdef int cnt = 0
    for i in range(y_size):
        # swap column iteration direction for every row
        if col_step == -1:
            j = 0
            col_step = 1
        else:
            j = x_size - 1
            col_step = -1

        for _ in range(x_size):
            if isinf(dst_x[i, j]):
                continue
            cnt = 0
            while True:
                cnt += 1
                # algorithm does not converge.
                if cnt > 5:
                    p0 = last_p0
                    l0 = last_l0
                    break
                # check we are within the input image bounds
                if lmax >= l0 >= 0 and pmax >= p0 >= 0:
                    # step size
                    dx = dst_x[i, j] - src_x[l0, p0]
                    dy = dst_y[i, j] - src_y[l0, p0]
                else:
                    # reset such that we are back in the input image bounds
                    l0 = max(0, min(lmax, l0))
                    p0 = max(0, min(pmax, p0))
                    continue

                # distance from pixel/line to output location
                d = yl[l0, p0] * xp[l0, p0] - yp[l0, p0] * xl[l0, p0]
                if d == 0.0:
                    # There's no gradient, try again
                    continue
                dl = (xp[l0, p0] * dy - yp[l0, p0] * dx) / d
                dp = (yl[l0, p0] * dx - xl[l0, p0] * dy) / d
                # check that our distance to an output location is less than 1
                # pixel/line
                if fabs(dp) < 1 and fabs(dl) < 1:
                    last_p0 = p0
                    last_l0 = l0
                    if 0 <= dl + l0 <= lmax and 0 <= dp + p0 <= pmax:
                        fun(data, l0, p0, dl, dp, lmax, pmax, result_array[:, i, j])
                    # found our solution, next
                    break
                else:
                    # increment...
                    l0 = int(l0 + dl)
                    p0 = int(p0 + dp)
            j += col_step


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef one_step_gradient_indices(float_index [:, :] src_x,
                                float_index [:, :] src_y,
                                float_index [:, :] xl,
                                float_index [:, :] xp,
                                float_index [:, :] yl,
                                float_index [:, :] yp,
                                float_index [:, :] dst_x,
                                float_index [:, :] dst_y):
    """Gradient search, simple case variant, returning float indices.

    This is appropriate for monotonous gradients only, i.e. not modis or viirs in satellite projection.
    """

    # change the output size (x_size, y_size) to match area_def.shape:
    # (lines,pixels)
    cdef size_t y_size = dst_y.shape[0]
    cdef size_t x_size = dst_x.shape[1]


    # output indices arrays --> needs to be (lines, pixels) --> y,x
    indices = np.full([2, y_size, x_size], np.nan, dtype=float_index_dtype)
    cdef float_index [:, :, :] indices_view_result = indices

    # fake_data is not going to be used anyway as we just fill in the indices
    cdef float_index [:, :, :] fake_data = np.full([1, 1, 1], np.nan, dtype=float_index_dtype)

    with nogil:
        one_step_gradient_search_no_gil[float_index](fake_data,
                                                     src_x, src_y,
                                                     xl, xp, yl, yp,
                                                     dst_x, dst_y,
                                                     x_size, y_size,
                                                     indices_xy, indices_view_result)
    return indices
