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

"""Testing the Trishchenko algorithm.
"""

from mpop.satellites import PolarFactory
from datetime import datetime
import mpop.scene
from mpop.utils import debug_on
debug_on()

import mpop.utils
mpop.utils.debug_on()
import pyximport; pyximport.install()
from _gradient_search import fast_gradient_search



def gradient_search(data, source_x, source_y, area_extent, size):
    """Trishchenko stuff.
    """
    x_min, y_min, x_max, y_max = area_extent
    x_size, y_size = size
    gx = source_x
    gy = source_y

    x_inc = (x_max - x_min) / x_size
    x_1d = np.arange(x_min + x_inc/2, x_max, x_inc)
    y_inc = (y_max - y_min) / y_size

    y_1d = np.arange(y_min + y_inc/2, y_max, y_inc)

    x, y = np.meshgrid(x_1d, y_1d)
    image = np.zeros(x.shape)

    yp, yl = np.gradient(gy)
    xp, xl = np.gradient(gx)

    xp = -xp
    xl = -xl
    yp = -yp
    yl = -yl

    p0, l0 = 1024, 2840

    cols = range(len(y_1d))
    cols.reverse()
    lines = range(len(x_1d))

    maxcol = 2048
    maxline = 5680

    prev_l0, prev_p0 = None, None

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
                    dx = x[i, j] - gx[l0, p0]
                    dy = y[i, j] - gy[l0, p0]
                except IndexError:
                    image[x_size - 1 - i, j] = 0
                    l0 = min(maxline - 1, l0)
                    l0 = max(0, l0)
                    p0 = min(maxcol - 1, p0)
                    p0 = max(0, p0)
                    break

                d = yl[l0, p0]*xp[l0, p0] - yp[l0, p0]*xl[l0, p0]
                dl = -(yl[l0, p0]*dx - xl[l0, p0]*dy) / d
                dp = (yp[l0, p0]*dx - xp[l0, p0]*dy) / d

                l0 += dl
                p0 += dp

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

def show(data):
    """Show the stetched data.
    """
    from PIL import Image as pil
    img = pil.fromarray(np.array((data - data.min()) * 255.0 /
                                 (data.max() - data.min()), np.uint8))
    img.show()
    img.save("/tmp/gradient2.png")


if __name__ == '__main__':
    t = datetime(2013, 3, 18, 8, 15, 22, 352000)
    g = PolarFactory.create_scene("noaa", "16", "avhrr", t, orbit="64374")
    g.load()

    # for comparison

    # tic = datetime.now()
    # l = g.project("scan2")
    # toc = datetime.now()
    # print "pyresample took", toc - tic




    import numpy as np
    import pyproj

    tic = datetime.now()
    p = pyproj.Proj(proj="stere", ellps="bessel", lat_0=90, lon_0=14, lat_ts=60)
    g[0.6].area = g.area
    gx, gy = p(g[0.6].area.lons, g[0.6].area.lats)

    # map grid for area "scan2"
    tic2 = datetime.now()
    image = fast_gradient_search(g[0.6].data, gx, gy,
                            (-1268854.1266382949, -4150234.8425892727,
                             779145.8733617051, -2102234.8425892727),
                            (1024, 1024))

    toc = datetime.now()

    print "resampling took", toc - tic
    print "fast_gradient_search took", toc - tic2
    show(image)
