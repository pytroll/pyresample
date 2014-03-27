#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2013, 2014 Martin Raspaud

# Author(s):

#   Martin Raspaud <martin.raspaud@smhi.se>

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
    cdef double x_min, y_min, x_max, y_max
    x_min, y_min, x_max, y_max = area_extent
    cdef int x_size, y_size
    x_size, y_size = size

    cdef double x_inc = (x_max - x_min) / x_size
    cdef np.ndarray[DTYPE_t, ndim=1] x_1d = np.arange(x_min + x_inc/2, x_max, x_inc)
    cdef double y_inc = (y_max - y_min) / y_size

    cdef np.ndarray[DTYPE_t, ndim=1] y_1d = np.arange(y_min + y_inc/2, y_max, y_inc)

    cdef np.ndarray[DTYPE_t, ndim=2] image = np.zeros([x_size, y_size], dtype=DTYPE)

    cdef np.ndarray[DTYPE_t, ndim=2] yp, yl
    yp, yl = np.gradient(source_y)
    cdef np.ndarray[DTYPE_t, ndim=2] xp, xl
    xp, xl = np.gradient(source_x)

    xp = -xp
    xl = -xl
    yp = -yp
    yl = -yl

    cdef int pmax = data.shape[1] - 1
    cdef int lmax = data.shape[0] - 1

    cdef int p0 = pmax / 2
    cdef int l0 = lmax / 2

    cdef int l_a, l_b, p_a, p_b

    #cols = range(len(y_1d))
    #cols.reverse()
    lines = range(x_size)

    cdef int maxcol = 2048
    cdef int maxline = 5680

    cdef size_t i, j
    cdef double dx, dy, d, dl, dp, w_l, w_p

    cdef int cnt = 0



    for i in range(y_size):
        lines.reverse()
        for j in lines:
            cnt = 0
            while True:
                cnt += 1
                # algorithm does not converge.
                if cnt > 5:
                    break

                if l0 < lmax and l0 >= 0 and p0 < pmax and p0 >= 0:
                    dx = x_1d[j] - source_x[l0, p0]
                    dy = y_1d[i] - source_y[l0, p0]
                else:
                    l0 = max(0, min(lmax - 1, l0))
                    p0 = max(0, min(pmax - 1, p0))
                    break

                d = yl[l0, p0]*xp[l0, p0] - yp[l0, p0]*xl[l0, p0]
                dl = -(yl[l0, p0]*dx - xl[l0, p0]*dy) / d
                dp = (yp[l0, p0]*dx - xp[l0, p0]*dy) / d

                if abs(dp) < 1 and abs(dl) < 1:
                    if l0 < lmax and l0 >= 0 and p0 < pmax and p0 >= 0:
                        # bilinear interpolation
                        if dl < 0:
                            l_a = l0 - 1
                            l_b = l0
                            w_l = 1 + dl
                        else:
                            l_a = l0
                            l_b = l0 + 1
                            w_l = dl

                        if dp < 0:
                            p_a = p0 - 1
                            p_b = p0
                            w_p = 1 + dp
                        else:
                            p_a = p0
                            p_b = p0 + 1
                            w_p = dp
                        image[x_size - 1 - i, j] = ((1 - w_l) * (1 - w_p) * data[l_a, p_a] + 
                                                    (1 - w_l) * w_p * data[l_a, p_b] + 
                                                    w_l * (1 - w_p) * data[l_b, p_a] + 
                                                    w_l * w_p * data[l_b, p_b])

                        # simple mode
                        #image[x_size - 1 - i, j] = data[l0, p0]
                    else:
                        l0 = max(0, min(lmax - 1, l0))
                        p0 = max(0, min(pmax - 1, p0))
                    break
                else:
                    l0 = int(l0 + dl)
                    p0 = int(p0 + dp)



    return image
