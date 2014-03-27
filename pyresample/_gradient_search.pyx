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

    cdef int p0 = 1024
    cdef int l0 = 2840

    cols = range(len(y_1d))
    cols.reverse()
    lines = range(len(x_1d))

    cdef int maxcol = 2048
    cdef int maxline = 5680

    cdef int i, j
    cdef double dx, dy, d

    cdef int cnt = 0

    for i in cols:
        lines.reverse()
        for j in lines:
            cnt = 0
            while True:
                cnt += 1
                # algorithm does not converge.
                if cnt > 5:
                    image[x_size - 1 - i, j] = 0
                    break

                try:
                    dx = x_1d[j] - source_x[l0, p0]
                    dy = y_1d[i] - source_y[l0, p0]
                except IndexError:
                    image[x_size - 1 - i, j] = 0
                    l0 = max(0, min(maxline - 1, l0))
                    p0 = max(0, min(maxcol - 1, p0))
                    break

                d = yl[l0, p0]*xp[l0, p0] - yp[l0, p0]*xl[l0, p0]
                dl = -(yl[l0, p0]*dx - xl[l0, p0]*dy) / d
                dp = (yp[l0, p0]*dx - xp[l0, p0]*dy) / d

                l0 += int(dl)
                p0 += int(dp)

                if abs(dp) < 1 and abs(dl) < 1:
                    try:
                        image[x_size - 1 - i, j] = data[l0, p0]

                    except IndexError:
                        l0 = min(maxline - 1, l0)
                        l0 = max(0, l0)
                        p0 = min(maxcol - 1, p0)
                        p0 = max(0, p0)
                    break


    return image
