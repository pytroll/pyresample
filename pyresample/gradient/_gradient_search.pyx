#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2013-2019

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

from libc.math cimport fabs, isinf, isnan
from libc.stdio cimport printf

import numpy as np

cimport numpy as np

DTYPE = np.double
ctypedef np.double_t DTYPE_t
cimport cython


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void nn(const DTYPE_t[:, :, :] data, int l0, int p0, double dl, double dp, int lmax, int pmax, DTYPE_t[:] res) nogil:
    cdef int nnl, nnp
    cdef size_t z_size = res.shape[0]
    cdef size_t i
    nnl = l0
    if dl < -0.5 and nnl > 0:
        nnl -= 1
    elif dl > 0.5 and nnl < lmax - 1:
        nnl += 1
    nnp = p0
    if dp < -0.5 and nnp > 0:
        nnp -= 1
    elif dp > 0.5 and nnp < pmax - 1:
        nnp += 1
    for i in range(z_size):
        res[i] = data[i, nnl, nnp]


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void bil(const DTYPE_t[:, :, :] data, int l0, int p0, double dl, double dp, int lmax, int pmax, DTYPE_t[:] res) nogil:
    cdef int l_a, l_b, p_a, p_b
    cdef double w_l, w_p
    cdef size_t z_size = res.shape[0]
    cdef size_t i
    if dl < 0:
        l_a = max(0, l0 - 1)
        l_b = l0
        w_l = 1 + dl
    else:
        l_a = l0
        l_b = min(l0 + 1, lmax - 1)
        w_l = dl
    if dp < 0:
        p_a = max(0, p0 - 1)
        p_b = p0
        w_p = 1 + dp
    else:
        p_a = p0
        p_b = min(p0 + 1, pmax - 1)
        w_p = dp
    for i in range(z_size):
        res[i] = ((1 - w_l) * (1 - w_p) * data[i, l_a, p_a] +
                  (1 - w_l) * w_p * data[i, l_a, p_b] +
                  w_l * (1 - w_p) * data[i, l_b, p_a] +
                  w_l * w_p * data[i, l_b, p_b])

ctypedef void(*FN)(const DTYPE_t[:, :, :] data, int l0, int p0, double dl, double dp, int lmax, int pmax, DTYPE_t[:] res) nogil


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef one_step_gradient_search(np.ndarray[DTYPE_t, ndim=3] data,
                               np.ndarray[DTYPE_t, ndim=2] src_x,
                               np.ndarray[DTYPE_t, ndim=2] src_y,
                               np.ndarray[DTYPE_t, ndim=2] xl,
                               np.ndarray[DTYPE_t, ndim=2] xp,
                               np.ndarray[DTYPE_t, ndim=2] yl,
                               np.ndarray[DTYPE_t, ndim=2] yp,
                               np.ndarray[DTYPE_t, ndim=2] dst_x,
                               np.ndarray[DTYPE_t, ndim=2] dst_y,
                               method='bilinear'):
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

    # output image array --> needs to be (lines, pixels) --> y,x
    cdef np.ndarray[DTYPE_t, ndim= 3] image = np.full([z_size, y_size, x_size], np.nan, dtype=DTYPE)
    cdef np.ndarray[size_t, ndim= 1] elements = np.arange(x_size, dtype=np.uintp)

    one_step_gradient_search_no_gil(data,
                                    src_x, src_y,
                                    xl, xp, yl, yp,
                                    dst_x, dst_y,
                                    x_size, y_size,
                                    fun, image,
                                    elements)
    # return the output image
    return image


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void one_step_gradient_search_no_gil(const DTYPE_t[:, :, :] data,
                                          const DTYPE_t[:, :] src_x,
                                          const DTYPE_t[:, :] src_y,
                                          const DTYPE_t[:, :] xl,
                                          const DTYPE_t[:, :] xp,
                                          const DTYPE_t[:, :] yl,
                                          const DTYPE_t[:, :] yp,
                                          const DTYPE_t[:, :] dst_x,
                                          const DTYPE_t[:, :] dst_y,
                                          const size_t x_size,
                                          const size_t y_size,
                                          FN fun,
                                          DTYPE_t[:, :, :] image,
                                          size_t[:] elements) nogil:

    # pixel max ---> data is expected in [lines, pixels]
    cdef int pmax = data.shape[2] - 1
    cdef int lmax = data.shape[1] - 1
    # centre of input image - starting point
    cdef int p0 = pmax // 2
    cdef int l0 = lmax // 2
    cdef int last_p0 = p0
    cdef int last_l0 = l0
    # cdef int l0 = 0
    # intermediate variables:
    cdef int l_a, l_b, p_a, p_b
    cdef size_t i, j, elt
    cdef double dx, dy, d, dl, dp
    # number of iterations
    cdef int cnt = 0
    for i in range(y_size):
        # lines.reverse() --> swapped to elements - provide a reverse view of
        # the array
        elements = elements[::-1]
        for elt in range(x_size):
            j = elements[elt]
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
                if l0 < lmax and l0 >= 0 and p0 < pmax and p0 >= 0:
                    # step size
                    dx = dst_x[i, j] - src_x[l0, p0]
                    dy = dst_y[i, j] - src_y[l0, p0]
                else:
                    # reset such that we are back in the input image bounds
                    if l0 >= lmax or l0 < 0 or p0 >= pmax or p0 < 0:
                        l0 = max(0, min(lmax - 1, l0))
                        p0 = max(0, min(pmax - 1, p0))
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
                    #image[:, i, j] = fun(data, l0, p0, dl, dp, lmax, pmax)
                    fun(data, l0, p0, dl, dp, lmax, pmax, image[:, i, j])
                    # found our solution, next
                    break
                else:
                    # increment...
                    l0 = int(l0 + dl)
                    p0 = int(p0 + dp)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef fast_gradient_search_pg(np.ndarray[DTYPE_t, ndim=2] data,
                              np.ndarray[DTYPE_t, ndim=2] src_x,
                              np.ndarray[DTYPE_t, ndim=2] src_y,
                              np.ndarray[DTYPE_t, ndim=2] xl,
                              np.ndarray[DTYPE_t, ndim=2] xp,
                              np.ndarray[DTYPE_t, ndim=2] yl,
                              np.ndarray[DTYPE_t, ndim=2] yp,
                              np.ndarray[DTYPE_t, ndim=1] dst_x,
                              np.ndarray[DTYPE_t, ndim=1] dst_y,
                              method='bilinear'):
    """Gradient search, simple case variant."""
    #method = 'bilinear'
    cdef FN fun
    if method == 'bilinear':
        fun = bil
    else:
        fun = nn
    #print('method %s' % method)
    first_val = np.min((np.nanargmin(src_x), np.nanargmax(src_x),
                        np.nanargmin(src_y), np.nanargmax(src_y)))
    line, col = np.unravel_index(first_val, (src_x.shape[0], src_x.shape[1]))
    cdef size_t first_line, first_col
    first_line = line
    first_col = col
    #print(first_line, first_col)
    cdef size_t start_line = np.searchsorted(dst_y, src_y[first_line, first_col])
    cdef DTYPE_t[:, :] data_view = data
    cdef DTYPE_t[:, :] source_x_view = src_x
    cdef DTYPE_t[:, :] source_y_view = src_y
    cdef DTYPE_t[:, :] xl_view = xl
    cdef DTYPE_t[:, :] xp_view = xp
    cdef DTYPE_t[:, :] yl_view = yl
    cdef DTYPE_t[:, :] yp_view = yp
    # change the output size (x_size, y_size) to match area_def.shape:
    # (lines,pixels)
    cdef size_t y_size = len(dst_y)
    cdef size_t x_size = len(dst_x)
    cdef DTYPE_t[:] dst_x_view = dst_x
    cdef DTYPE_t[:] dst_y_view = dst_y

    # output image array --> needs to be (lines, pixels) --> y,x
    cdef np.ndarray[DTYPE_t, ndim= 2] image = np.full([y_size, x_size], np.nan, dtype=DTYPE)
    cdef DTYPE_t[:, :] image_view = image
    # this was a bit confusing -- "lines" was based on x_size; change this
    # variable to elements - make it a numpy array (long==numpy int dtype)
    cdef size_t[:] elements = np.arange(x_size, dtype=np.uintp)
    # Make a view to allow running without gil
    fast_gradient_search_ng(data_view,
                            source_x_view, source_y_view,
                            xl_view, xp_view, yl_view, yp_view,
                            x_size, y_size,
                            fun, image,
                            dst_x_view, dst_y_view,
                            elements, first_line,
                            first_col, start_line)
    # return the output image
    return image


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void fast_gradient_search_ng(DTYPE_t[:, :] data,
                                  DTYPE_t[:, :] src_x,
                                  DTYPE_t[:, :] src_y,
                                  DTYPE_t[:, :] xl,
                                  DTYPE_t[:, :] xp,
                                  DTYPE_t[:, :] yl,
                                  DTYPE_t[:, :] yp,
                                  size_t x_size, size_t y_size,
                                  FN fun,
                                  DTYPE_t[:, :] image,
                                  DTYPE_t[:] dst_x,
                                  DTYPE_t[:] dst_y,
                                  size_t[:] elements,
                                  size_t first_line,
                                  size_t first_col,
                                  size_t start_line) nogil:

    # pixel max ---> data is expected in [lines, pixels]
    cdef int pmax = data.shape[1] - 1
    cdef int lmax = data.shape[0] - 1
    # centre of input image - starting point
    cdef int p0 = pmax // 2
    cdef int l0 = lmax // 2
    # cdef int p0 = first_col
    # cdef int l0 = first_line
    cdef int last_p0 = p0
    cdef int last_l0 = l0
    # cdef int l0 = 0
    # intermediate variables:
    cdef int l_a, l_b, p_a, p_b
    cdef size_t i, j, elt
    cdef double dx, dy, d, dl, dp
    # number of iterations
    cdef int cnt = 0
    with nogil:
        for i in range(y_size):
            # lines.reverse() --> swapped to elements - provide a reverse view of
            # the array
            elements = elements[::-1]
            for elt in range(x_size):
                j = elements[elt]
                cnt = 0
                while True:
                    cnt += 1
                    # algorithm does not converge.
                    if cnt > 5:
                        p0 = last_p0
                        l0 = last_l0
                        break
                    # check we are within the input image bounds
                    if l0 < lmax and l0 >= 0 and p0 < pmax and p0 >= 0:
                        # while isnan(src_x[l0, p0]) and (dl > 1 or dp > 1):
                        #     dl /= 2
                        #     dp /= 2
                        #     l0 = int(l0 - dl)
                        #     p0 = int(p0 - dp)
                        # step size
                        dx = dst_x[j] - src_x[l0, p0]
                        dy = dst_y[i] - src_y[l0, p0]
                    else:
                        # while (l0 >= lmax or l0 < 0 or p0 >= pmax or p0 < 0 or isnan(src_x[l0, p0])) and (dl > 1 or dp > 1):
                        #     dl /= 2
                        #     dp /= 2
                        #     l0 = int(l0 - dl)
                        #     p0 = int(p0 - dp)
                        # reset such that we are back in the input image bounds
                        if l0 >= lmax or l0 < 0 or p0 >= pmax or p0 < 0:
                            l0 = max(0, min(lmax - 1, l0))
                            p0 = max(0, min(pmax - 1, p0))
                            continue
                        # if dl <= 1 and dp <= 1:
                        #     p0 = last_p0
                        #     l0 = last_l0
                        #     break
                            # continue
                        dx = dst_x[j] - src_x[l0, p0]
                        dy = dst_y[i] - src_y[l0, p0]

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
                        #image[y_size - 1 - i, j] = fun(data, l0, p0, dl, dp, lmax, pmax)
                        # found our solution, next
                        break
                    else:
                        # increment...
                        l0 = int(l0 + dl)
                        p0 = int(p0 + dp)


# old stuff

# @cython.boundscheck(False)
# @cython.wraparound(False)
# cpdef fast_gradient_search(np.ndarray[DTYPE_t, ndim=2] data, np.ndarray[DTYPE_t, ndim=2] src_x, np.ndarray[DTYPE_t, ndim=2] src_y, area_extent, size):
#       """Gradient search, simple case variant.
#       """
#       # area extent bounds --> area_def.area_extent
#       cdef double x_min, y_min, x_max, y_max
#       x_min, y_min, x_max, y_max = area_extent
#       # change the output size (x_size, y_size) to match area_def.shape:
#       # (lines,pixels)
#       cdef int x_size, y_size
#       y_size, x_size = size
#       # step in x direction (meters); x-output locations (centre of pixel);
#       # repeat for y
#       cdef double x_inc = (x_max - x_min) / x_size
#       cdef np.ndarray[DTYPE_t, ndim = 1] dst_x = np.arange(x_min + x_inc / 2, x_max, x_inc)
#       cdef double y_inc = (y_max - y_min) / y_size
#       cdef np.ndarray[DTYPE_t, ndim = 1] dst_y = np.arange(y_min + y_inc / 2, y_max, y_inc)
#       # output image array --> needs to be (lines, pixels) --> y,x
#       cdef np.ndarray[DTYPE_t, ndim = 2] image = np.zeros([y_size, x_size], dtype=DTYPE)
#       # gradient of output y/x grids (in pixel and line directions)
#       cdef np.ndarray[DTYPE_t, ndim = 2] yp, yl
#       yl, yp = np.gradient(src_y)
#       cdef np.ndarray[DTYPE_t, ndim = 2] xp, xl
#       xl, xp = np.gradient(src_x)
#       # pixel max ---> data is expected in [lines, pixels]
#       cdef int pmax = data.shape[1] - 1
#       cdef int lmax = data.shape[0] - 1
#       # centre of input image - starting point
#       cdef int p0 = pmax / 2
#       cdef int l0 = lmax / 2
#       # intermediate variables:
#       cdef int l_a, l_b, p_a, p_b, nnl, nnp
#       cdef size_t i, j, elt
#       cdef double dx, dy, d, dl, dp, w_l, w_p
#       # number of iterations
#       cdef int cnt = 0
#       # this was a bit confusing -- "lines" was based on x_size; change this
#       # variable to elements - make it a numpy array (long==numpy int dtype)
#       cdef np.ndarray[size_t, ndim = 1] npelements = np.arange(x_size, dtype=np.uintp)
#       # Make a view to allow running without gil
#       cdef size_t [:] elements = npelements
#       with nogil:
#         for i in range(y_size):
#             # lines.reverse() --> swapped to elements - provide a reverse view of
#             # the array
#             elements = elements[::-1]
#             for elt in range(elements.shape[0]):
#                 j = elements[elt]
#                 cnt = 0
#                 while True:
#                     cnt += 1
#                     # algorithm does not converge.
#                     if cnt > 5:
#                         break
#                     # check we are within the input image bounds
#                     if l0 < lmax and l0 >= 0 and p0 < pmax and p0 >= 0:
#                         # step size
#                         dx = dst_x[j] - src_x[l0, p0]
#                         dy = dst_y[i] - src_y[l0, p0]
#                     else:
#                         # reset such that we are back in the input image bounds
#                         l0 = max(0, min(lmax - 1, l0))
#                         p0 = max(0, min(pmax - 1, p0))
#                         continue
#                     # distance from pixel/line to output location
#                     d = yl[l0, p0] * xp[l0, p0] - yp[l0, p0] * xl[l0, p0]
#                     if d == 0.0:
#                         # There's no gradient, try again
#                         continue
#                     dl = (xp[l0, p0] * dy - yp[l0, p0] * dx) / d
#                     dp = (yl[l0, p0] * dx - xl[l0, p0] * dy) / d
#                     # check that our distance to an output location is less than 1
#                     # pixel/line
#                     if fabs(dp) < 1 and fabs(dl) < 1:
#                         # nearest neighbour
#                         # nnl = l0
#                         # if dl < -0.5 and nnl > 0:
#                         #     nnl -= 1
#                         # elif dl > 0.5 and nnl < lmax - 1:
#                         #     nnl += 1
#                         # nnp = p0
#                         # if dp < -0.5 and nnp > 0:
#                         #     nnp -= 1
#                         # elif dp > 0.5 and nnp < pmax - 1:
#                         #     nnp += 1
#                         # image[y_size - 1 - i, j] = data[nnl, nnp]
#                         # bilinear interpolation
#                         if dl < 0:
#                             l_a = max(0, l0 - 1)
#                             l_b = l0
#                             w_l = 1 + dl
#                         else:
#                             l_a = l0
#                             l_b = min(l0 + 1, lmax - 1)
#                             w_l = dl
#                         if dp < 0:
#                             p_a = max(0, p0 - 1)
#                             p_b = p0
#                             w_p = 1 + dp
#                         else:
#                             p_a = p0
#                             p_b = min(p0 + 1, pmax - 1)
#                             w_p = dp
#                         # assign image output... (was x_size in the first dimension
#                         # --> needs to be y_size)
#                         image[y_size - 1 - i, j] = ((1 - w_l) * (1 - w_p) * data[l_a, p_a] +
#                                                     (1 - w_l) * w_p * data[l_a, p_b] +
#                                                     w_l * (1 - w_p) * data[l_b, p_a] +
#                                                     w_l * w_p * data[l_b, p_b])
#                         # found our solution, next
#                         break
#                     else:
#                         # increment...
#                         l0 = int(l0 + dl)
#                         p0 = int(p0 + dp)
#       # return the output image
#       return image
#
#
#
#
# @cython.boundscheck(False)
# def fast_gradient_indices(np.ndarray[DTYPE_t, ndim=2] src_x, np.ndarray[DTYPE_t, ndim=2] src_y,
#                           np.ndarray[DTYPE_t, ndim=1] dst_x, np.ndarray[DTYPE_t, ndim=1] dst_y):
#     """Gradient search, simple case variant.
#     """
#     # area extent bounds --> area_def.area_extent
#     cdef double x_min, y_min, x_max, y_max
#     #x_min, y_min, x_max, y_max = area_extent
#     # change the output size (x_size, y_size) to match area_def.shape:
#     # (lines,pixels)
#     cdef int x_size, y_size
#     #y_size, x_size = size
#     x_size = dst_x.shape[0]
#     y_size = dst_y.shape[0]
#     # step in x direction (meters); x-output locations (centre of pixel);
#     # repeat for y
#     #cdef double x_inc = (x_max - x_min) / x_size
#     cdef double x_inc = (dst_x[-1] - dst_x[0]) / (x_size - 1)
#     #cdef np.ndarray[DTYPE_t, ndim = 1] dst_x = np.arange(x_min + x_inc / 2, x_max, x_inc)
#     #cdef double y_inc = (y_max - y_min) / y_size
#     cdef double y_inc = (dst_y[-1] - dst_y[0]) / (y_size - 1)
#     x_min = dst_x[0] - x_inc / 2
#     x_max = dst_x[-1] + x_inc / 2
#     y_min = dst_y[0] - y_inc / 2
#     y_max = dst_y[-1] + y_inc / 2
#
#     #cdef np.ndarray[DTYPE_t, ndim = 1] dst_y = np.arange(y_min + y_inc / 2, y_max, y_inc)
#     # output image array --> needs to be (lines, pixels) --> y,x
#     cdef np.ndarray[np.double_t, ndim = 3] indices = np.full([2, y_size, x_size], -2.0, dtype=np.double)
#     # gradient of output y/x grids (in pixel and line directions)
#     cdef np.ndarray[DTYPE_t, ndim = 2] yp, yl
#     yl, yp = np.gradient(src_y)
#     cdef np.ndarray[DTYPE_t, ndim = 2] xp, xl
#     xl, xp = np.gradient(src_x)
#     # pixel max ---> data is expected in [lines, pixels]
#     cdef int pmax = src_x.shape[1] - 1
#     cdef int lmax = src_x.shape[0] - 1
#     # centre of input image - starting point
#     cdef int p0 = pmax / 2
#     cdef int l0 = lmax / 2
#     # intermediate variables:
#     cdef int l_a, l_b, p_a, p_b, nnl, nnp
#     cdef size_t i, j
#     cdef double dx, dy, d, dl, dp, w_l, w_p
#     # number of iterations
#     cdef int cnt = 0
#     # this was a bit confusing -- "lines" was based on x_size; change this
#     # variable to elements - make it a numpy array (long==numpy int dtype)
#     cdef np.ndarray[size_t, ndim = 1] elements = np.arange(x_size, dtype=np.uintp)
#     for i in range(y_size):
#         # lines.reverse() --> swapped to elements - provide a reverse view of
#         # the array
#         elements = elements[::-1]
#         for j in elements:
#             cnt = 0
#             while True:
#                 cnt += 1
#                 # algorithm does not converge.
#                 if cnt > 5:
#                     break
#                 # check we are within the input image bounds
#                 if l0 < lmax and l0 >= 0 and p0 < pmax and p0 >= 0:
#                     # step size
#                     dx = dst_x[j] - src_x[l0, p0]
#                     dy = dst_y[i] - src_y[l0, p0]
#                 else:
#                     # reset such that we are back in the input image bounds
#                     l0 = max(0, min(lmax - 1, l0))
#                     p0 = max(0, min(pmax - 1, p0))
#                     continue
#                 # distance from pixel/line to output location
#                 d = yl[l0, p0] * xp[l0, p0] - yp[l0, p0] * xl[l0, p0]
#                 if d == 0.0:
#                     # There's no gradient, try again
#                     continue
#                 dl = (xp[l0, p0] * dy - yp[l0, p0] * dx) / d
#                 dp = (yl[l0, p0] * dx - xl[l0, p0] * dy) / d
#                 # check that our distance to an output location is less than 1
#                 # pixel/line
#                 if abs(dp) < 1 and abs(dl) < 1:
#                     indices[0, i, j] = max(min(l0 + dl, lmax - 1), 0)
#                     indices[1, i, j] = max(min(p0 + dp, pmax - 1), 0)
#                     # found our solution, next
#                     break
#                 else:
#                     # increment...
#                     l0 = int(l0 + dl)
#                     p0 = int(p0 + dp)
#     # return the output image
#     return indices
#
#
# @cython.boundscheck(False)
# def two_step_fast_gradient_search(np.ndarray[DTYPE_t, ndim=2] data, np.ndarray[DTYPE_t, ndim=2] src_x, np.ndarray[DTYPE_t, ndim=2] src_y, int chunk_size, area_extent, size):
#     """Gradient search, discontinuity handling variant (Modis)
#     """
#     # area extent bounds --> area_def.area_extent
#     cdef double x_min, y_min, x_max, y_max
#     x_min, y_min, x_max, y_max = area_extent
#     # change the output size (x_size, y_size) to match area_def.shape:
#     # (lines,pixels)
#     cdef int x_size, y_size
#     y_size, x_size = size
#     # step in x direction (meters); x-output locations (centre of pixel);
#     # repeat for y
#     cdef double x_inc = (x_max - x_min) / x_size
#     cdef np.ndarray[DTYPE_t, ndim = 1] dst_x = np.arange(x_min + x_inc / 2, x_max, x_inc)
#     cdef double y_inc = (y_max - y_min) / y_size
#     cdef np.ndarray[DTYPE_t, ndim = 1] dst_y = np.arange(y_min + y_inc / 2, y_max, y_inc)
#     # output image array --> needs to be (lines, pixels) --> y,x
#     cdef np.ndarray[DTYPE_t, ndim = 2] image = np.zeros([y_size, x_size], dtype=DTYPE)
#
#     cdef size_t i, j
#
#     # reduce the src_x and src_y arrays for inter-segment search
#     cdef np.ndarray[DTYPE_t, ndim = 2] reduced_x, reduced_y
#     # reduced_x = (src_x[4::chunk_size, :] + src_x[5::chunk_size]) / 2.0
#     # reduced_y = (src_y[4::chunk_size, :] + src_y[5::chunk_size]) / 2.0
#     reduced_x = (src_x[::chunk_size, :] + src_x[1::chunk_size] + src_x[2::chunk_size, :] + src_x[3::chunk_size] + src_x[4::chunk_size,
#                                                                                                                                        :] + src_x[5::chunk_size] + src_x[6::chunk_size, :] + src_x[7::chunk_size] + src_x[8::chunk_size, :] + src_x[9::chunk_size]) / chunk_size
#     reduced_y = (src_y[::chunk_size, :] + src_y[1::chunk_size] + src_y[2::chunk_size, :] + src_y[3::chunk_size] + src_y[4::chunk_size,
#                                                                                                                                        :] + src_y[5::chunk_size] + src_y[6::chunk_size, :] + src_y[7::chunk_size] + src_y[8::chunk_size, :] + src_y[9::chunk_size]) / chunk_size
#     cdef np.ndarray[DTYPE_t, ndim = 2] ryp, ryl
#     ryp, ryl = np.gradient(reduced_y)
#     cdef np.ndarray[DTYPE_t, ndim = 2] rxp, rxl
#     rxp, rxl = np.gradient(reduced_x)
#     rxp = -rxp
#     rxl = -rxl
#     ryp = -ryp
#     ryl = -ryl
#
#     # gradient of full output y/x grids (in pixel and line directions)
#     cdef np.ndarray[DTYPE_t, ndim = 2] fyp, fyl
#     cdef np.ndarray[DTYPE_t, ndim = 2] fxp, fxl
#     fyp = np.zeros_like(src_y)
#     fyl = np.zeros_like(src_y)
#     fxp = np.zeros_like(src_x)
#     fxl = np.zeros_like(src_x)
#     for i in range(data.shape[0] / chunk_size):
#         fyp[i * chunk_size:(i + 1) * chunk_size, :], fyl[i * chunk_size:(i + 1) *
#                                                          chunk_size, :] = np.gradient(src_y[i * chunk_size:(i + 1) * chunk_size, :])
#         fxp[i * chunk_size:(i + 1) * chunk_size, :], fxl[i * chunk_size:(i + 1) *
#                                                          chunk_size, :] = np.gradient(src_x[i * chunk_size:(i + 1) * chunk_size, :])
#
#     # fyp, fyl = np.gradient(src_y)
#     # fxp, fxl = np.gradient(src_x)
#     fxp = -fxp
#     fxl = -fxl
#     fyp = -fyp
#     fyl = -fyl
#
#     cdef np.ndarray[DTYPE_t, ndim = 2] yp, yl
#     cdef np.ndarray[DTYPE_t, ndim = 2] xp, xl
#     cdef np.ndarray[DTYPE_t, ndim = 2] array_x, array_y
#     # pixel max ---> data is expected in [lines, pixels]
#     cdef int pmax = data.shape[1] - 1
#     cdef int flmax = data.shape[0] - 1
#     cdef int rlmax = reduced_x.shape[0] - 1
#     cdef int lmax
#     # centre of input image - starting point
#     cdef int p0 = pmax / 2
#     cdef int l0 = flmax / 2
#     # intermediate variables:
#     cdef int l_a, l_b, p_a, p_b, nnl, nnp
#     cdef double dx, dy, d, dl, dp, w_l, w_p
#     # number of iterations
#     cdef int cnt = 0
#     cdef int adj = False
#     # this was a bit confusing -- "lines" was based on x_size; change this
#     # variable to elements - make it a numpy array (long==numpy int dtype)
#     cdef np.ndarray[size_t, ndim = 1] elements = np.arange(x_size, dtype=np.uintp)
#     lmax = rlmax
#     l0 /= chunk_size
#     xp, xl, yp, yl = rxp, rxl, ryp, ryl
#     array_x, array_y = reduced_x, reduced_y
#     for i in range(y_size):
#         # lines.reverse() --> swapped to elements - provide a reverse view of
#         # the array
#         elements = elements[::-1]
#         for j in elements:
#             cnt = 0
#             adj = False
#             while True:
#                 cnt += 1
#                 # algorithm does not converge, try jumping to the next chunk
#                 if cnt > 5:
#                     if not adj and l0 % chunk_size >= chunk_size / 2.0:
#                         l0 += chunk_size
#                         adj = True
#                         cnt = 0
#                     elif not adj and l0 % chunk_size < chunk_size / 2.0:
#                         l0 -= chunk_size
#                         adj = True
#                         cnt = 0
#                     else:
#                         break
#                 # check we are within the input image bounds
#                 if l0 < lmax and l0 >= 0 and p0 < pmax and p0 >= 0:
#                     # step size
#                     dx = dst_x[j] - array_x[l0, p0]
#                     dy = dst_y[i] - array_y[l0, p0]
#                 else:
#                     # reset such that we are back in the input image bounds
#                     l0 = max(0, min(lmax - 1, l0))
#                     p0 = max(0, min(pmax - 1, p0))
#                     break
#                 # distance from pixel/line to output location
#                 d = yl[l0, p0] * xp[l0, p0] - yp[l0, p0] * xl[l0, p0]
#                 dl = -(yl[l0, p0] * dx - xl[l0, p0] * dy) / d
#                 dp = (yp[l0, p0] * dx - xp[l0, p0] * dy) / d
#                 # check that our distance to an output location is less than 1
#                 # pixel/line
#                 if abs(dp) < 1 and abs(dl) < 1:
#                     if lmax == rlmax:  # switch to full resolution
#                         # nearest neighbour
#                         nnl = l0
#                         if dl < -0.5 and nnl > 0:
#                             nnl -= 1
#                         elif dl > 0.5 and nnl < lmax - 1:
#                             nnl += 1
#                         nnp = p0
#                         if dp < -0.5 and nnp > 0:
#                             nnp -= 1
#                         elif dp > 0.5 and nnp < pmax - 1:
#                             nnp += 1
#                         l0 = l0 * chunk_size + chunk_size / 2
#                         lmax = flmax
#                         xp, xl, yp, yl = fxp, fxl, fyp, fyl
#                         array_x, array_y = src_x, src_y
#                         continue
#                     # switch to next chunk if too close from bow-tie borders
#                     if not adj and l0 % chunk_size == chunk_size - 1:
#                         l0 += chunk_size / 2
#                         adj = True
#                         cnt = 0
#                         continue
#                     elif not adj and l0 % chunk_size == 0:
#                         l0 -= chunk_size / 2
#                         adj = True
#                         cnt = 0
#                         continue
#                     # bilinear interpolation
#                     if dl < 0:
#                         l_a = max(0, l0 - 1)
#                         l_b = l0
#                         w_l = 1 + dl
#                     else:
#                         l_a = l0
#                         l_b = min(l0 + 1, lmax - 1)
#                         w_l = dl
#                     if dp < 0:
#                         p_a = max(0, p0 - 1)
#                         p_b = p0
#                         w_p = 1 + dp
#                     else:
#                         p_a = p0
#                         p_b = min(p0 + 1, pmax - 1)
#                         w_p = dp
#                     # assign image output... (was x_size in the first dimension
#                     # --> needs to be y_size)
#                     image[y_size - 1 - i, j] = ((1 - w_l) * (1 - w_p) * data[l_a, p_a] +
#                                                 (1 - w_l) * w_p * data[l_a, p_b] +
#                                                 w_l * (1 - w_p) * data[l_b, p_a] +
#                                                 w_l * w_p * data[l_b, p_b])
#                     # image[y_size - 1 - i, j] = data[l0, p0]
#                     # found our solution, next
#                     break
#                 else:
#                     # increment...
#                     l0 = int(l0 + dl)
#                     p0 = int(p0 + dp)
#     # return the output image
#     return image
#
#
# @cython.boundscheck(False)
# def fast_gradient_search_with_mask(np.ndarray[DTYPE_t, ndim=2] data, np.ndarray[DTYPE_t, ndim=2] src_x, np.ndarray[DTYPE_t, ndim=2] src_y, area_extent, size, np.ndarray[np.uint8_t, ndim=2] mask):
#     """Gradient search, simple case variant.
#     """
#     # area extent bounds --> area_def.area_extent
#     cdef double x_min, y_min, x_max, y_max
#     x_min, y_min, x_max, y_max = area_extent
#     # change the output size (x_size, y_size) to match area_def.shape:
#     # (lines,pixels)
#     cdef int x_size, y_size
#     y_size, x_size = size
#     # step in x direction (meters); x-output locations (centre of pixel);
#     # repeat for y
#     cdef double x_inc = (x_max - x_min) / x_size
#     cdef np.ndarray[DTYPE_t, ndim = 1] dst_x = np.arange(x_min + x_inc / 2, x_max, x_inc)
#     cdef double y_inc = (y_max - y_min) / y_size
#     cdef np.ndarray[DTYPE_t, ndim = 1] dst_y = np.arange(y_min + y_inc / 2, y_max, y_inc)
#     # output image array --> needs to be (lines, pixels) --> y,x
#     cdef np.ndarray[DTYPE_t, ndim = 2] image = np.zeros([y_size, x_size], dtype=DTYPE)
#     # gradient of output y/x grids (in pixel and line directions)
#     cdef np.ndarray[DTYPE_t, ndim = 2] yp, yl
#     yl, yp = np.gradient(src_y)
#     cdef np.ndarray[DTYPE_t, ndim = 2] xp, xl
#     xl, xp = np.gradient(src_x)
#     # pixel max ---> data is expected in [lines, pixels]
#     cdef int pmax = data.shape[1] - 1
#     cdef int lmax = data.shape[0] - 1
#     # centre of input image - starting point
#     cdef int p0 = pmax / 2
#     cdef int l0 = lmax / 2
#     cdef int oldp, oldl
#     # intermediate variables:
#     cdef int l_a, l_b, p_a, p_b, nnl, nnp
#     cdef size_t i, j
#     cdef double dx, dy, d, dl, dp, w_l, w_p
#     # number of iterations
#     cdef int cnt = 0
#     cdef int inc = 0
#     # this was a bit confusing -- "lines" was based on x_size; change this
#     # variable to elements - make it a numpy array (long==numpy int dtype)
#     cdef np.ndarray[size_t, ndim = 1] elements = np.arange(x_size, dtype=np.uintp)
#     for i in range(y_size):
#         # lines.reverse() --> swapped to elements - provide a reverse view of
#         # the array
#         elements = elements[::-1]
#         for j in elements:
#             cnt = 0
#             while True:
#                 cnt += 1
#                 # algorithm does not converge.
#                 if cnt > 15:
#                     break
#                 # step size
#                 dx = dst_x[j] - src_x[l0, p0]
#                 dy = dst_y[i] - src_y[l0, p0]
#
#                 # distance from pixel/line to output location
#                 d = yl[l0, p0] * xp[l0, p0] - yp[l0, p0] * xl[l0, p0]
#                 dl = (xp[l0, p0] * dy - yp[l0, p0] * dx) / d
#                 dp = (yl[l0, p0] * dx - xl[l0, p0] * dy) / d
#                 # check that our distance to an output location is less than 1
#                 # pixel/line
#                 if abs(dp) < 1 and abs(dl) < 1:
#                     image[y_size - 1 - i, j] = data[l0, p0]
#                     # found our solution, next
#                     break
#                 else:
#                     # increment...
#                     l0 = int(l0 + dl)
#                     p0 = int(p0 + dp)
#                     # if dl > 0 and l0 % chunk_size == chunk_size - 1:
#                     #    l0 += 2
#                     # if dl < 0 and l0 % chunk_size == 0:
#                     #    l0 -= 2
#
#                     # oldp = p0
#                     # oldl = l0
#                     # p0 = max(0, min(pmax - 1, int(p0 + dp)))
#                     # l0 = int(l0 + dl)
#                     # inc = int(dl)
#                     # if inc > 0:
#                     #     while (inc > 0 or mask[l0, p0] != 0) and l0 < lmax:
#                     #         if mask[l0, p0] == 0:
#                     #             inc -= 1
#                     #         l0 += 1
#                     #     if l0 >= lmax:
#                     #         l0 = oldl
#                     # else:
#                     #     while (inc < 0 or mask[l0, p0] != 0) and l0 >= 0:
#                     #         if mask[l0, p0] == 0:
#                     #             inc += 1
#                     #         l0 -= 1
#                     #     if l0 < 0:
#                     #         l0 = oldl
#                     # if oldp == p0 and oldl == l0:
#                     #     break
#     # return the output image
#     return image
#
#
# @cython.boundscheck(False)
# def two_step_fast_gradient_search_with_mask(np.ndarray[DTYPE_t, ndim=2] data, np.ndarray[DTYPE_t, ndim=2] src_x, np.ndarray[DTYPE_t, ndim=2] src_y, int chunk_size, area_extent, size, np.ndarray[np.uint8_t, ndim=2] mask):
#     """Gradient search, discontinuity handling variant (Modis)
#     """
#     # area extent bounds --> area_def.area_extent
#     cdef double x_min, y_min, x_max, y_max
#     x_min, y_min, x_max, y_max = area_extent
#     # change the output size (x_size, y_size) to match area_def.shape:
#     # (lines,pixels)
#     cdef int x_size, y_size
#     y_size, x_size = size
#     # step in x direction (meters); x-output locations (centre of pixel);
#     # repeat for y
#     cdef double x_inc = (x_max - x_min) / x_size
#     cdef np.ndarray[DTYPE_t, ndim = 1] dst_x = np.arange(x_min + x_inc / 2, x_max, x_inc)
#     cdef double y_inc = (y_max - y_min) / y_size
#     cdef np.ndarray[DTYPE_t, ndim = 1] dst_y = np.arange(y_min + y_inc / 2, y_max, y_inc)
#     # output image array --> needs to be (lines, pixels) --> y,x
#     cdef np.ndarray[DTYPE_t, ndim = 2] image = np.zeros([y_size, x_size], dtype=DTYPE)
#
#     cdef size_t i, j
#
#     # reduce the src_x and src_y arrays for inter-segment search
#     cdef np.ndarray[DTYPE_t, ndim = 2] reduced_x, reduced_y
#     reduced_x = np.zeros([data.shape[0] / chunk_size, data.shape[1]],
#                          dtype=DTYPE)
#     reduced_y = np.zeros([data.shape[0] / chunk_size, data.shape[1]],
#                          dtype=DTYPE)
#     for i in range(chunk_size):
#         reduced_x += src_x[i::chunk_size, :]
#         reduced_y += src_y[i::chunk_size, :]
#     reduced_x /= chunk_size
#     reduced_y /= chunk_size
#     cdef np.ndarray[DTYPE_t, ndim = 2] ryp, ryl
#     ryp, ryl = np.gradient(reduced_y)
#     cdef np.ndarray[DTYPE_t, ndim = 2] rxp, rxl
#     rxp, rxl = np.gradient(reduced_x)
#     rxp = -rxp
#     rxl = -rxl
#     ryp = -ryp
#     ryl = -ryl
#
#     # gradient of full output y/x grids (in pixel and line directions)
#     cdef np.ndarray[DTYPE_t, ndim = 2] fyp, fyl
#     cdef np.ndarray[DTYPE_t, ndim = 2] fxp, fxl
#     fyp = np.zeros_like(src_y)
#     fyl = np.zeros_like(src_y)
#     fxp = np.zeros_like(src_x)
#     fxl = np.zeros_like(src_x)
#     for i in range(data.shape[0] / chunk_size):
#         fyp[i * chunk_size:(i + 1) * chunk_size, :], fyl[i * chunk_size:(i + 1) *
#                                                          chunk_size, :] = np.gradient(src_y[i * chunk_size:(i + 1) * chunk_size, :])
#         fxp[i * chunk_size:(i + 1) * chunk_size, :], fxl[i * chunk_size:(i + 1) *
#                                                          chunk_size, :] = np.gradient(src_x[i * chunk_size:(i + 1) * chunk_size, :])
#
#     # fyp, fyl = np.gradient(src_y)
#     # fxp, fxl = np.gradient(src_x)
#     fxp = -fxp
#     fxl = -fxl
#     fyp = -fyp
#     fyl = -fyl
#
#     cdef np.ndarray[DTYPE_t, ndim = 2] yp, yl
#     cdef np.ndarray[DTYPE_t, ndim = 2] xp, xl
#     cdef np.ndarray[DTYPE_t, ndim = 2] array_x, array_y
#     # pixel max ---> data is expected in [lines, pixels]
#     cdef int pmax = data.shape[1] - 1
#     cdef int flmax = data.shape[0] - 1
#     cdef int rlmax = reduced_x.shape[0] - 1
#     cdef int lmax
#     # centre of input image - starting point
#     cdef int p0 = pmax / 2
#     cdef int l0 = flmax / 2
#     # intermediate variables:
#     cdef int nnl, nnp
#     cdef int l_a, l_b, l_c, l_d, p_a, p_b, p_c, p_d
#     cdef double dx, dy, d, dl, dp, w_l, w_p
#     # number of iterations
#     cdef int cnt = 0
#     cdef int adj = False
#     cdef int undefined = 0
#     cdef int sdp = 1
#     cdef int sdl = 1
#     cdef int n_l, n_p
#     cdef double distance
#     cdef int masked = False
#     # this was a bit confusing -- "lines" was based on x_size; change this
#     # variable to elements - make it a numpy array (long==numpy int dtype)
#     cdef np.ndarray[size_t, ndim = 1] elements = np.arange(x_size, dtype=np.uintp)
#     lmax = rlmax
#     l0 /= chunk_size
#     xp, xl, yp, yl = rxp, rxl, ryp, ryl
#     array_x, array_y = reduced_x, reduced_y
#
#     for i in range(y_size):
#         # lines.reverse() --> swapped to elements - provide a reverse view of
#         # the array
#         elements = elements[::-1]
#         for j in elements:
#             cnt = 0
#             adj = False
#             while True:
#                 cnt += 1
#                 # algorithm does not converge
#                 if cnt > 5:
#                     break
#
#                 # check we are within the input image bounds
#                 if not (l0 < lmax and l0 >= 0 and p0 < pmax and p0 >= 0):
#                     # reset such that we are back in the input image bounds
#                     l0 = max(0, min(lmax - 1, l0))
#                     p0 = max(0, min(pmax - 1, p0))
#                 # step size
#                 dx = dst_x[j] - array_x[l0, p0]
#                 dy = dst_y[i] - array_y[l0, p0]
#
#                 # distance from pixel/line to output location
#                 d = yl[l0, p0] * xp[l0, p0] - yp[l0, p0] * xl[l0, p0]
#                 dl = -(yl[l0, p0] * dx - xl[l0, p0] * dy) / d
#                 dp = (yp[l0, p0] * dx - xp[l0, p0] * dy) / d
#
#                 # check that our distance to an output location is less than 1
#                 # pixel/line
#                 if ((abs(dp) < 1) and (abs(dl) < 1) and
#                         mask[l0, p0] == 0):
#                     # not (mask[l0, p0] != 0 or
#                     #     (l0 < lmax - 1 and mask[l0 + 1, p0] != 0) or
#                     #     (l0 > 0 and mask[l0 - 1, p0] != 0))):
#                     if lmax == rlmax:  # switch to full resolution
#                         print "switching to full res", i, j
#                         # nearest neighbour
#                         l0 = l0 * chunk_size + chunk_size / 2
#                         lmax = flmax
#                         xp, xl, yp, yl = fxp, fxl, fyp, fyl
#                         array_x, array_y = src_x, src_y
#                         cnt = 0
#                         continue
#
#                     # crude
#                     #image[y_size - 1 - i, j] = data[l0, p0]
#
#                     # nearest neighbour
#                     n_l = max(0, min(lmax - 1, int(l0 + dl)))
#                     n_p = max(0, min(pmax - 1, int(p0 + dp)))
#
#                     if mask[n_l, n_p] != 0:
#                         image[y_size - 1 - i, j] = data[l0, p0]
#                     else:
#                         image[y_size - 1 - i, j] = data[n_l, n_p]
#
#                     # bilinear interpolation
#
#                     # if dl < 0:
#                     #     sdl = -1
#                     # else:
#                     #     sdl = 1
#                     # if dp < 0:
#                     #     sdp = -1
#                     # else:
#                     #     sdp = 1
#
#                     # l_a, p_a = l0, p0
#                     # masked = False
#
#                     # l_b, p_b = l0, max(0, min(p0 + sdp, pmax - 1))
#                     # while (l_b >= 0 and l_b < lmax
#                     #        and mask[l_b, p_b] != 0):
#                     #     l_b -= sdl
#                     #     masked = True
#                     # l_b = max(0, min(l_b, lmax - 1))
#                     # if mask[l_b, p_b] != 0:
#                     #     l_b, p_b = l_a, p_a
#
#                     # l_c, p_c = (max(0, min(l0 + sdl, lmax - 1)),
#                     #             max(0, min(p0 + sdp, pmax - 1)))
#                     # while (l_c >= 0 and l_c < lmax and
#                     #        mask[l_c, p_c] != 0):
#                     #     l_c += sdl
#                     #     masked = True
#                     # l_c = max(0, min(l_c, lmax - 1))
#                     # if mask[l_c, p_c] != 0:
#                     #     l_c, p_c = l_a, p_a
#
#                     # l_d, p_d = max(0, min(l0 + sdl, lmax - 1)), p0
#                     # while (l_d >= 0 and l_d < lmax and
#                     #        mask[l_d, p_d] != 0):
#                     #     l_d += sdl
#                     #     masked = True
#                     # l_d = max(0, min(l_d, lmax - 1))
#                     # if mask[l_d, p_d] != 0:
#                     #     l_d, p_d = l_a, p_a
#
#                     # if masked and abs(l_a - l_d) > 9:
#                     #     print "masked: lines", l_a, l_b, l_c, l_d,
#                     #     print "cols:", p_a, p_b, p_c, p_d
#
#                     # recompute dl and dp
#                     # if masked and l_a != l_d and p_a != p_b:
#                     #     print "before", dl, dp
#                     #     d = ((src_y[l_a, p_a] - src_y[l_d, p_d]) *
#                     #          (src_x[l_a, p_a] - src_x[l_b, p_b]) -
#                     #          (src_y[l_a, p_a] - src_y[l_b, p_b]) *
#                     #          (src_x[l_a, p_a] - src_x[l_d, p_d]))
#                     #     dp = -((src_y[l_a, p_a] - src_y[l_d, p_d]) *
#                     #            dx - (src_x[l_a, p_a] - src_x[l_d, p_d]) * dy) / d
#                     #     dl = ((src_y[l_a, p_a] - src_y[l_b, p_b]) *
#                     #           dx - (src_x[l_a, p_a] - src_x[l_b, p_b]) * dy) / d
#                     #     print "after", dl, dp
#                     #     dp = min(1, max(dp, -1))
#                     #     dl = min(1, max(dl, -1))
#
#                     # w_l = 1 - abs(dl)
#                     # w_p = 1 - abs(dp)
#
#                     # image[y_size - 1 - i, j] = ((1 - w_l) * (1 - w_p) * data[l_c, p_c] +
#                     #                             (1 - w_l) * w_p * data[l_d, p_d] +
#                     #                             w_l * (1 - w_p) * data[l_b, p_b] +
#                     #                             w_l * w_p * data[l_a, p_a])
#
#                     # we found our solution, next
#                     break
#                 else:
#                     # increment...
#                     l0 = int(l0 + dl)
#                     p0 = int(p0 + dp)
#                     # if dl > 0 and l0 % chunk_size >= chunk_size - 1:
#                     #    l0 += 2
#                     # if dl < 0 and l0 % chunk_size < 1:
#                     #    l0 -= 2
#                     if l0 >= lmax or l0 < 0 or p0 >= pmax or p0 < 0:
#                         continue
#
#                     # if dl > 0 and (mask[l0, p0] != 0 or
#                     #                (l0 < lmax - 1 and
#                     #                 mask[l0 + 1, p0] != 0)):
#                     #     while (l0 < lmax - 1 and
#                     #            (mask[l0, p0] != 0 or
#                     #             mask[l0 + 1, p0] != 0)):
#                     #         l0 += 1
#                     #     l0 = min(lmax - 1, l0)
#
#                     # if dl < 0 and (mask[l0, p0] != 0 or
#                     #                (l0 > 0 and
#                     #                 mask[l0 - 1, p0] != 0)):
#
#                     #     while (l0 > 0 and
#                     #            (mask[l0, p0] != 0 or
#                     #             mask[l0 - 1, p0] != 0)):
#                     #         l0 -= 1
#                     #     l0 = max(0, l0)
#                     if dl > 0 and mask[l0, p0] != 0:
#                         while (l0 < lmax - 1 and
#                                mask[l0, p0] != 0):
#                             l0 += 1
#                         l0 = min(lmax - 1, l0)
#
#                     if dl < 0 and mask[l0, p0] != 0:
#
#                         while (l0 > 0 and
#                                mask[l0, p0] != 0):
#                             l0 -= 1
#                         l0 = max(0, l0)
#
#     # return the output image
#     print "undefined", undefined
#     return image
#
#
# #############################
# ########################
# ######################
# #############
# #
# #  Old stuff
# #
# ######
# ##
#
# @cython.boundscheck(False)
# def two_step_fast_gradient_search_with_mask_old(np.ndarray[DTYPE_t, ndim=2] data, np.ndarray[DTYPE_t, ndim=2] src_x, np.ndarray[DTYPE_t, ndim=2] src_y, int chunk_size, area_extent, size, np.ndarray[np.uint8_t, ndim=2] mask):
#     """Gradient search, discontinuity handling variant (Modis)
#     """
#     # area extent bounds --> area_def.area_extent
#     cdef double x_min, y_min, x_max, y_max
#     x_min, y_min, x_max, y_max = area_extent
#     # change the output size (x_size, y_size) to match area_def.shape:
#     # (lines,pixels)
#     cdef int x_size, y_size
#     y_size, x_size = size
#     # step in x direction (meters); x-output locations (centre of pixel);
#     # repeat for y
#     cdef double x_inc = (x_max - x_min) / x_size
#     cdef np.ndarray[DTYPE_t, ndim = 1] dst_x = np.arange(x_min + x_inc / 2, x_max, x_inc)
#     cdef double y_inc = (y_max - y_min) / y_size
#     cdef np.ndarray[DTYPE_t, ndim = 1] dst_y = np.arange(y_min + y_inc / 2, y_max, y_inc)
#     # output image array --> needs to be (lines, pixels) --> y,x
#     cdef np.ndarray[DTYPE_t, ndim = 2] image = np.zeros([y_size, x_size], dtype=DTYPE)
#
#     cdef size_t i, j
#
#     # reduce the src_x and src_y arrays for inter-segment search
#     cdef np.ndarray[DTYPE_t, ndim = 2] reduced_x, reduced_y
#     # reduced_x = (src_x[4::chunk_size, :] + src_x[5::chunk_size]) / 2.0
#     # reduced_y = (src_y[4::chunk_size, :] + src_y[5::chunk_size]) / 2.0
#     # reduced_x = (src_x[::chunk_size, :] + src_x[1::chunk_size] + src_x[2::chunk_size, :] + src_x[3::chunk_size] + src_x[4::chunk_size,:] + src_x[5::chunk_size] + src_x[6::chunk_size, :] + src_x[7::chunk_size] + src_x[8::chunk_size, :] + src_x[9::chunk_size]) / chunk_size
#     # reduced_y = (src_y[::chunk_size, :] + src_y[1::chunk_size] +
#     # src_y[2::chunk_size, :] + src_y[3::chunk_size] +
#     # src_y[4::chunk_size, :] + src_y[5::chunk_size] +
#     # src_y[6::chunk_size, :] + src_y[7::chunk_size] +
#     # src_y[8::chunk_size, :] + src_y[9::chunk_size]) / chunk_size
#     reduced_x = np.zeros([data.shape[0] / chunk_size, data.shape[1]],
#                          dtype=DTYPE)
#     reduced_y = np.zeros([data.shape[0] / chunk_size, data.shape[1]],
#                          dtype=DTYPE)
#     for i in range(chunk_size):
#         reduced_x += src_x[i::chunk_size, :]
#         reduced_y += src_y[i::chunk_size, :]
#     reduced_x /= chunk_size
#     reduced_y /= chunk_size
#     cdef np.ndarray[DTYPE_t, ndim = 2] ryp, ryl
#     ryp, ryl = np.gradient(reduced_y)
#     cdef np.ndarray[DTYPE_t, ndim = 2] rxp, rxl
#     rxp, rxl = np.gradient(reduced_x)
#     rxp = -rxp
#     rxl = -rxl
#     ryp = -ryp
#     ryl = -ryl
#
#     # gradient of full output y/x grids (in pixel and line directions)
#     cdef np.ndarray[DTYPE_t, ndim = 2] fyp, fyl
#     cdef np.ndarray[DTYPE_t, ndim = 2] fxp, fxl
#     fyp = np.zeros_like(src_y)
#     fyl = np.zeros_like(src_y)
#     fxp = np.zeros_like(src_x)
#     fxl = np.zeros_like(src_x)
#     for i in range(data.shape[0] / chunk_size):
#         fyp[i * chunk_size:(i + 1) * chunk_size, :], fyl[i * chunk_size:(i + 1) *
#                                                          chunk_size, :] = np.gradient(src_y[i * chunk_size:(i + 1) * chunk_size, :])
#         fxp[i * chunk_size:(i + 1) * chunk_size, :], fxl[i * chunk_size:(i + 1) *
#                                                          chunk_size, :] = np.gradient(src_x[i * chunk_size:(i + 1) * chunk_size, :])
#
#     # fyp, fyl = np.gradient(src_y)
#     # fxp, fxl = np.gradient(src_x)
#     fxp = -fxp
#     fxl = -fxl
#     fyp = -fyp
#     fyl = -fyl
#
#     cdef np.ndarray[DTYPE_t, ndim = 2] yp, yl
#     cdef np.ndarray[DTYPE_t, ndim = 2] xp, xl
#     cdef np.ndarray[DTYPE_t, ndim = 2] array_x, array_y
#     # pixel max ---> data is expected in [lines, pixels]
#     cdef int pmax = data.shape[1] - 1
#     cdef int flmax = data.shape[0] - 1
#     cdef int rlmax = reduced_x.shape[0] - 1
#     cdef int lmax
#     # centre of input image - starting point
#     cdef int p0 = pmax / 2
#     cdef int l0 = flmax / 2
#     # intermediate variables:
#     cdef int nnl, nnp
#     cdef int l_a00, l_a01, l_a10, l_a11, p_a00, p_a01, p_a10, p_a11
#     cdef double dx, dy, d, dl, dp, w_l, w_p
#     # number of iterations
#     cdef int cnt = 0
#     cdef int adj = False
#     cdef int undefined = 0
#     cdef double distance
#     # this was a bit confusing -- "lines" was based on x_size; change this
#     # variable to elements - make it a numpy array (long==numpy int dtype)
#     cdef np.ndarray[size_t, ndim = 1] elements = np.arange(x_size, dtype=np.uintp)
#     lmax = rlmax
#     l0 /= chunk_size
#     xp, xl, yp, yl = rxp, rxl, ryp, ryl
#     array_x, array_y = reduced_x, reduced_y
#     for i in range(y_size):
#         # lines.reverse() --> swapped to elements - provide a reverse view of
#         # the array
#         elements = elements[::-1]
#         for j in elements:
#             cnt = 0
#             adj = False
#             while True:
#                 cnt += 1
#                 # algorithm does not converge, try jumping to the next chunk
#                 if cnt > 5:
#                     break
#                     # if adj and l0 % chunk_size >= chunk_size / 2.0:
#                     #     l0 += chunk_size
#                     #     adj = True
#                     #     cnt = 0
#                     # elif not adj and l0 % chunk_size < chunk_size / 2.0:
#                     #     l0 -= chunk_size
#                     #     adj = True
#                     #     cnt = 0
#                     # else:
#                     # distance = np.sqrt((src_y[l0, p0] - dst_y[i]) ** 2 +
#                     # (src_x[l0, p0] - dst_x[j]) ** 2)
#                     # TODO: this should be done dynamically or from arg
#                     # if distance < 1:
#                     # image[y_size - 1 - i, j] = data[l0, p0]
#                     # else:
#                     # undefined += 1
#                     #     break
#
#                 # check we are within the input image bounds
#                 if l0 < lmax and l0 >= 0 and p0 < pmax and p0 >= 0:
#                     if mask[l0, p0] != 0:
#                         if dl > 0:
#                             while l0 < lmax and mask[l0, p0] != 0:
#                                 l0 += 1
#                             if l0 >= lmax:
#                                 l0 = lmax - 1
#                                 break
#                         else:
#                             while l0 >= 0 and mask[l0, p0] != 0:
#                                 l0 -= 1
#                             if l0 < 0:
#                                 l0 = 0
#                                 break
#
#                     # step size
#                     dx = dst_x[j] - array_x[l0, p0]
#                     dy = dst_y[i] - array_y[l0, p0]
#                 else:
#                     # reset such that we are back in the input image bounds
#                     l0 = max(0, min(lmax - 1, l0))
#                     p0 = max(0, min(pmax - 1, p0))
#                     dx = dst_x[j] - array_x[l0, p0]
#                     dy = dst_y[i] - array_y[l0, p0]
#
#                 # distance from pixel/line to output location
#                 d = yl[l0, p0] * xp[l0, p0] - yp[l0, p0] * xl[l0, p0]
#                 dl = -(yl[l0, p0] * dx - xl[l0, p0] * dy) / d
#                 dp = (yp[l0, p0] * dx - xp[l0, p0] * dy) / d
#
#                 # take care of pixels outside
#                 if ((l0 == lmax - 1 and dl > 0.5) or
#                         (l0 == 0 and dl < -0.5) or
#                         (p0 == pmax - 1 and dp > 0.5) or
#                         (p0 == 0 and dp < -0.5)):
#                     break
#
#                 # check that our distance to an output location is less than 1
#                 # pixel/line
#                 if (abs(dp) < 1) and (abs(dl) < 1):
#                     if lmax == rlmax:  # switch to full resolution
#                         print "switching to full res", i, j
#                         # nearest neighbour
#                         l0 = l0 * chunk_size + chunk_size / 2
#                         lmax = flmax
#                         xp, xl, yp, yl = fxp, fxl, fyp, fyl
#                         array_x, array_y = src_x, src_y
#                         cnt = 0
#                         continue
#                     # switch to next chunk if too close from bow-tie borders
#                     # if not adj and l0 % chunk_size == chunk_size - 1:
#                     #     l0 += chunk_size / 2
#                     #     adj = True
#                     #     cnt = 0
#                     #     continue
#                     # elif not adj and l0 % chunk_size == 0:
#                     #     l0 -= chunk_size / 2
#                     #     adj = True
#                     #     cnt = 0
#                     #     continue
#                     # check if the data is masked
#                     if mask[l0, p0] != 0:
#                         # if dl > 0:
#                         #     while mask[l0, p0] != 0:
#                         #         l0 += 1
#                         # else:
#                         #     while mask[l0, p0] != 0:
#                         #         l0 -= 1
#                         if l0 % chunk_size > chunk_size / 2:
#                             while l0 < lmax and mask[l0, p0] != 0:
#                                 l0 += 1
#                             if l0 >= lmax:
#                                 l0 = lmax - 1
#                                 break
#                         else:
#                             while l0 >= 0 and mask[l0, p0] != 0:
#                                 l0 -= 1
#                             if l0 < 0:
#                                 l0 = 0
#                                 break
#                         cnt -= 1
#                         continue
#
#                     # image[y_size - 1 - i, j] = data[l0, p0]
#
#                     # bilinear interpolation
#                     # if dl < 0:
#                     #     l_a = max(0, l0 - 1)
#                     #     l_b = l0
#                     #     w_l = 1 + dl
#                     # else:
#                     #     l_a = l0
#                     #     l_b = min(l0 + 1, lmax - 1)
#                     #     w_l = dl
#                     # if dp < 0:
#                     #     p_a = max(0, p0 - 1)
#                     #     p_b = p0
#                     #     w_p = 1 + dp
#                     # else:
#                     #     p_a = p0
#                     #     p_b = min(p0 + 1, pmax - 1)
#                     #     w_p = dp
#
#                     if dl < 0 and dp < 0:
#                         w_l = 1 + dl
#                         w_p = 1 + dp
#                         l_a00 = l0 - 1
#                         p_a00 = p0 - 1
#                         l_a01 = l0 - 1
#                         p_a01 = p0
#                         l_a10 = l0
#                         p_a10 = p0 - 1
#                         l_a11 = l0
#                         p_a11 = p0
#                         while l_a01 > 0 and mask[l_a01, p_a01] != 0:
#                             l_a01 -= 1
#                         if mask[l_a00, p_a00] != 0:
#                             l_a00 = l_a01
#                             p_a00 += 1
#                         if mask[l_a10, p_a10] != 0:
#                             p_a10 += 1
#
#                     if dl > 0 and dp < 0:
#                         w_l = dl
#                         w_p = 1 + dp
#                         l_a00 = l0
#                         p_a00 = p0 - 1
#                         l_a01 = l0
#                         p_a01 = p0
#                         l_a10 = l0 + 1
#                         p_a10 = p0 - 1
#                         l_a11 = l0 + 1
#                         p_a11 = p0
#                         if mask[l_a00, p_a00] != 0:
#                             p_a00 += 1
#                         while l_a11 < lmax - 1 and mask[l_a11, p_a11] != 0:
#                             l_a11 += 1
#                         if mask[l_a10, p_a10] != 0:
#                             l_a10 = l_a11
#                             p_a10 += 1
#
#                     if dl > 0 and dp > 0:
#                         w_l = dl
#                         w_p = dp
#                         l_a00 = l0
#                         p_a00 = p0
#                         l_a01 = l0
#                         p_a01 = p0 + 1
#                         l_a10 = l0 + 1
#                         p_a10 = p0
#                         l_a11 = l0 + 1
#                         p_a11 = p0 + 1
#                         if mask[l_a01, p_a01] != 0:
#                             p_a01 -= 1
#                         while l_a10 < lmax - 1 and mask[l_a10, p_a10] != 0:
#                             l_a10 += 1
#                         if mask[l_a11, p_a11] != 0:
#                             l_a11 = l_a10
#                             p_a11 -= 1
#
#                     if dl < 0 and dp > 0:
#                         w_l = 1 + dl
#                         w_p = dp
#                         l_a00 = l0 - 1
#                         p_a00 = p0
#                         l_a01 = l0 - 1
#                         p_a01 = p0 + 1
#                         l_a10 = l0
#                         p_a10 = p0
#                         l_a11 = l0
#                         p_a11 = p0 + 1
#
#                         while l_a00 > 0 and mask[l_a00, p_a00] != 0:
#                             l_a00 -= 1
#                         if mask[l_a01, p_a01] != 0:
#                             l_a01 = l_a00
#                             p_a01 -= 1
#                         if mask[l_a11, p_a11] != 0:
#                             p_a11 -= 1
#
#                     # assign image output... (was x_size in the first dimension
#                     # --> needs to be y_size)
#                     image[y_size - 1 - i, j] = ((1 - w_l) * (1 - w_p) * data[l_a00, p_a00] +
#                                                 (1 - w_l) * w_p * data[l_a01, p_a01] +
#                                                 w_l * (1 - w_p) * data[l_a10, p_a10] +
#                                                 w_l * w_p * data[l_a11, p_a11])
#
#                     # nearest neighbour
#                     # nnl = l0
#                     # nnp = p0
#                     # if dp < -0.5 and nnp > 0 and mask[nnl, nnp - 1] == 0:
#                     #     nnp -= 1
#                     # elif dp > 0.5 and nnp < pmax - 1 and mask[nnl, nnp + 1] == 0:
#                     #     nnp += 1
#                     # if dl < -0.5 and nnl > 0 and mask[nnl - 1, nnp] == 0:
#                     #     nnl -= 1
#                     # elif dl > 0.5 and nnl < lmax - 1 and mask[nnl + 1, nnp] == 0:
#                     #     nnl += 1
#                     # image[y_size - 1 - i, j] = data[nnl, nnp]
#
#                     # we found our solution, next
#                     break
#                 else:
#                     # increment...
#                     l0 = int(l0 + dl)
#                     p0 = int(p0 + dp)
#                     if dl > 0 and l0 % chunk_size == chunk_size - 1:
#                         l0 += 2
#                     if dl < 0 and l0 % chunk_size == 0:
#                         l0 -= 2
#     # return the output image
#     print "undefined", undefined
#     return image
#
#
# @cython.boundscheck(False)
# def fast_gradient_search_with_mask_old(np.ndarray[DTYPE_t, ndim=2] data, np.ndarray[DTYPE_t, ndim=2] src_x, np.ndarray[DTYPE_t, ndim=2] src_y, area_extent, size, np.ndarray[np.uint8_t, ndim=2] mask):
#     """Gradient search, simple case variant, with masked data.
#     """
#     # area extent bounds --> area_def.area_extent
#     cdef double x_min, y_min, x_max, y_max
#     x_min, y_min, x_max, y_max = area_extent
#     # change the output size (x_size, y_size) to match area_def.shape:
#     # (lines,pixels)
#     cdef int x_size, y_size
#     y_size, x_size = size
#     # step in x direction (meters); x-output locations (centre of pixel);
#     # repeat for y
#     cdef double x_inc = (x_max - x_min) / x_size
#     cdef np.ndarray[DTYPE_t, ndim = 1] dst_x = np.arange(x_min + x_inc / 2, x_max, x_inc)
#     cdef double y_inc = (y_max - y_min) / y_size
#     cdef np.ndarray[DTYPE_t, ndim = 1] dst_y = np.arange(y_min + y_inc / 2, y_max, y_inc)
#     # output image array --> needs to be (lines, pixels) --> y,x
#     cdef np.ndarray[DTYPE_t, ndim = 2] image = np.zeros([y_size, x_size], dtype=DTYPE)
#     # gradient of output y/x grids (in pixel and line directions)
#     cdef np.ndarray[DTYPE_t, ndim = 2] yp, yl
#     yl, yp = np.gradient(src_y)
#     cdef np.ndarray[DTYPE_t, ndim = 2] xp, xl
#     xl, xp = np.gradient(src_x)
#     # pixel max ---> data is expected in [lines, pixels]
#     cdef int pmax = data.shape[1] - 1
#     cdef int lmax = data.shape[0] - 1
#     # centre of input image - starting point
#     cdef int p0 = pmax / 2
#     cdef int l0 = lmax / 2
#     # intermediate variables:
#     cdef int l_a, l_b, p_a, p_b, nnl, nnp
#     cdef size_t i, j
#     cdef double dx, dy, d, dl, dp, w_l, w_p
#     dl = 0
#     dp = 0
#     # number of iterations
#     cdef int cnt = 0
#     # this was a bit confusing -- "lines" was based on x_size; change this
#     # variable to elements - make it a numpy array (long==numpy int dtype)
#     cdef np.ndarray[size_t, ndim = 1] elements = np.arange(x_size, dtype=np.uintp)
#     for i in range(y_size):
#         # lines.reverse() --> swapped to elements - provide a reverse view of
#         # the array
#         elements = elements[::-1]
#         for j in elements:
#             cnt = 0
#             while True:
#                 cnt += 1
#
#                 # algorithm does not converge.
#                 if cnt > 5:
#                     # TODO: this should be done dynamically or from arg
#                     # distance = np.sqrt((src_y[l0, p0] - dst_y[i]) ** 2 +
#                     #                   (src_x[l0, p0] - dst_x[j]) ** 2)
#                     # if distance < 5000 and mask[l0, p0] != 0:
#                     #    image[y_size - 1 - i, j] = data[l0, p0]
#                     break
#                 # check we are within the input image bounds
#                 if l0 < lmax and l0 >= 0 and p0 < pmax and p0 >= 0:
#                     if mask[l0, p0] != 0:
#                         if dl >= 0:
#                             while l0 < lmax and mask[l0, p0] != 0:
#                                 l0 += 1
#                             if l0 >= lmax:
#                                 l0 = lmax - 1
#                                 break
#                         else:
#                             while l0 >= 0 and mask[l0, p0] != 0:
#                                 l0 -= 1
#                             if l0 < 0:
#                                 l0 = 0
#                                 break
#                 else:
#                     # reset such that we are back in the input image bounds
#                     l0 = max(0, min(lmax - 1, l0))
#                     p0 = max(0, min(pmax - 1, p0))
#                 # step size
#                 dx = dst_x[j] - src_x[l0, p0]
#                 dy = dst_y[i] - src_y[l0, p0]
#
#                 # distance from pixel/line to output location
#                 d = yl[l0, p0] * xp[l0, p0] - yp[l0, p0] * xl[l0, p0]
#                 if d == 0:
#                     image[y_size - 1 - i, j] = data[l0, p0]
#                     break
#                 dp = (yl[l0, p0] * dx - xl[l0, p0] * dy) / d
#                 dl = (xp[l0, p0] * dy - yp[l0, p0] * dx) / d
#                 # check that our distance to an output location is less than 1
#                 # pixel/line
#                 if abs(dp) < 1 and abs(dl) < 1:
#                     # nearest neighbour
#                     # nnl = l0
#                     # if dl < -0.5 and nnl > 0:
#                     #     nnl -= 1
#                     # elif dl > 0.5 and nnl < lmax - 1:
#                     #     nnl += 1
#                     # nnp = p0
#                     # if dp < -0.5 and nnp > 0:
#                     #     nnp -= 1
#                     # elif dp > 0.5 and nnp < pmax - 1:
#                     #     nnp += 1
#                     # image[y_size - 1 - i, j] = data[nnl, nnp]
#                     # bilinear interpolation
#                     if dl < 0:
#                         l_a = max(0, l0 - 1)
#                         l_b = l0
#                         w_l = 1 + dl
#                     else:
#                         l_a = l0
#                         l_b = min(l0 + 1, lmax - 1)
#                         w_l = dl
#                     if dp < 0:
#                         p_a = max(0, p0 - 1)
#                         p_b = p0
#                         w_p = 1 + dp
#                     else:
#                         p_a = p0
#                         p_b = min(p0 + 1, pmax - 1)
#                         w_p = dp
#                     # assign image output... (was x_size in the first dimension
#                     # --> needs to be y_size)
#                     image[y_size - 1 - i, j] = ((1 - w_l) * (1 - w_p) * data[l_a, p_a] +
#                                                 (1 - w_l) * w_p * data[l_a, p_b] +
#                                                 w_l * (1 - w_p) * data[l_b, p_a] +
#                                                 w_l * w_p * data[l_b, p_b])
#                     if mask[l0, p0] != 0:
#                         if dl >= 0:
#                             while l0 < lmax and mask[l0, p0] != 0:
#                                 l0 += 1
#                             if l0 >= lmax:
#                                 l0 = lmax - 1
#                                 break
#                         else:
#                             while l0 >= 0 and mask[l0, p0] != 0:
#                                 l0 -= 1
#                             if l0 < 0:
#                                 l0 = 0
#                                 break
#
#                     image[y_size - 1 - i, j] = data[l0, p0]
#                     # found our solution, next
#                     break
#                 else:
#                     # increment...
#                     l0 = int(l0 + dl)
#                     p0 = int(p0 + dp)
#     # return the output image
#     return image
