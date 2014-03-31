#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2013, 2014 Martin Raspaud

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
cimport numpy as np
DTYPE = np.double
ctypedef np.double_t DTYPE_t
cimport cython
@cython.boundscheck(False)
def fast_gradient_search(np.ndarray[DTYPE_t, ndim=2] data, np.ndarray[DTYPE_t, ndim=2] source_x, np.ndarray[DTYPE_t, ndim=2] source_y, area_extent, size):
    """Trishchenko stuff.
    """
    #area extent bounds --> area_def.area_extent
    cdef double x_min, y_min, x_max, y_max
    x_min, y_min, x_max, y_max = area_extent
    #change the output size (x_size, y_size) to match area_def.shape: (lines,pixels)
    cdef int x_size, y_size
    y_size, x_size = size
    #step in x direction (meters); x-output locations (centre of pixel); repeat for y
    cdef double x_inc = (x_max - x_min) / x_size
    cdef np.ndarray[DTYPE_t, ndim=1] x_1d = np.arange(x_min + x_inc/2, x_max, x_inc)
    cdef double y_inc = (y_max - y_min) / y_size
    cdef np.ndarray[DTYPE_t, ndim=1] y_1d = np.arange(y_min + y_inc/2, y_max, y_inc)
    #output image array --> needs to be (lines, pixels) --> y,x
    cdef np.ndarray[DTYPE_t, ndim=2] image = np.zeros([y_size, x_size], dtype=DTYPE)
    #gradient of output y/x grids (in pixel and line directions)
    cdef np.ndarray[DTYPE_t, ndim=2] yp, yl
    yp, yl = np.gradient(source_y)
    cdef np.ndarray[DTYPE_t, ndim=2] xp, xl
    xp, xl = np.gradient(source_x)
    xp = -xp
    xl = -xl
    yp = -yp
    yl = -yl
    #pixel max ---> data is expected in [lines, pixels]
    cdef int pmax = data.shape[1] - 1
    cdef int lmax = data.shape[0] - 1
    #centre of input image - starting point
    cdef int p0 = pmax / 2
    cdef int l0 = lmax / 2
    #intermediate variables:
    cdef int l_a, l_b, p_a, p_b, nnl, nnp
    cdef size_t i, j
    cdef double dx, dy, d, dl, dp, w_l, w_p
    #number of iterations
    cdef int cnt = 0
    #this was a bit confusing -- "lines" was based on x_size; change this variable to elements - make it a numpy array (long==numpy int dtype)
    cdef np.ndarray[size_t, ndim=1] elements = np.arange(x_size, dtype=np.uintp)
    for i in range(y_size):
        #lines.reverse() --> swapped to elements - provide a reverse view of the array
        elements = elements[::-1]
        for j in elements:
            cnt = 0
            while True:
                cnt += 1
                # algorithm does not converge.
                if cnt > 5:
                    break
                #check we are within the input image bounds
                if l0 < lmax and l0 >= 0 and p0 < pmax and p0 >= 0:
                    #step size
                    dx = x_1d[j] - source_x[l0, p0]
                    dy = y_1d[i] - source_y[l0, p0]
                else:
                    #reset such that we are back in the input image bounds
                    l0 = max(0, min(lmax - 1, l0))
                    p0 = max(0, min(pmax - 1, p0))
                    break
                #distance from pixel/line to output location
                d = yl[l0, p0]*xp[l0, p0] - yp[l0, p0]*xl[l0, p0]
                dl = -(yl[l0, p0]*dx - xl[l0, p0]*dy) / d
                dp = (yp[l0, p0]*dx - xp[l0, p0]*dy) / d
                #check that our distance to an output location is less than 1 pixel/line
                if abs(dp) < 1 and abs(dl) < 1:
                    # # nearest neighbour
                    # nnl = l0
                    # if dl < -0.5 and nnl > 0:
                    #     nnl -= 1
                    # elif dl > 0.5 and nnp < lmax - 1:
                    #     nnl += 1
                    # nnp = p0
                    # if dp < -0.5 and nnp > 0:
                    #     nnp -= 1
                    # elif dp > 0.5 and nnp < pmax - 1:
                    #     nnp += 1
                    # image[y_size - 1 - i, j] = data[nnl, nnp]
                    # bilinear interpolation
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
                    #assign image output... (was x_size in the first dimension --> needs to be y_size)
                    image[y_size - 1 - i, j] = ((1 - w_l) * (1 - w_p) * data[l_a, p_a] + 
                                                (1 - w_l) * w_p * data[l_a, p_b] + 
                                                w_l * (1 - w_p) * data[l_b, p_a] + 
                                                w_l * w_p * data[l_b, p_b])
                    #found our solution, next
                    break
                else:
                    #increment...
                    l0 = int(l0 + dl)
                    p0 = int(p0 + dp)
    #return the output image
    return image
