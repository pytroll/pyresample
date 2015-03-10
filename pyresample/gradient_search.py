#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2013, 2014, 2015 Martin Raspaud

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
import numpy as np
import pyproj
from mpop.utils import debug_on
debug_on()

import mpop.utils
mpop.utils.debug_on()
import h5py
import pyximport
pyximport.install()
from _gradient_search import (fast_gradient_search,
                              two_step_fast_gradient_search,
                              two_step_fast_gradient_search_with_mask,
                              fast_gradient_search_with_mask)
from pyresample import data_reduce

import logging

logger = logging.getLogger(__name__)


def gradient_search(data, lons, lats, area, chunk_size=0, mask=None):

    tic = datetime.now()
    p = pyproj.Proj(**area.proj_dict)

    lon_bound, lat_bound = area.get_boundary_lonlats()
    idx = data_reduce.get_valid_index_from_lonlat_boundaries(lon_bound,
                                                             lat_bound,
                                                             lons, lats,
                                                             5000)

    colsmin, colsmax = np.arange(lons.shape[1])[np.sum(idx, 0, bool)][[0, -1]]
    linesmin, linesmax = np.arange(
        lons.shape[0])[np.sum(idx, 1, bool)][[0, -1]]

    if chunk_size != 0:
        linesmin -= linesmin % chunk_size
        linesmax += chunk_size
        linesmax -= linesmax % chunk_size

    projection_x_coords, projection_y_coords = p(lons[linesmin:linesmax,
                                                      colsmin:colsmax],
                                                 lats[linesmin:linesmax,
                                                      colsmin:colsmax])

    toc2 = datetime.now()
    logger.debug("pyproj took %s", str(toc2 - tic))
    tic2 = datetime.now()

    if chunk_size == 0:
        if mask is None:
            image = fast_gradient_search(data[linesmin:linesmax,
                                              colsmin:colsmax],
                                         projection_x_coords, projection_y_coords,
                                         area.area_extent,
                                         area.shape)
        else:
            image = fast_gradient_search_with_mask(data[linesmin:linesmax,
                                                        colsmin:colsmax],
                                                   projection_x_coords, projection_y_coords,
                                                   area.area_extent,
                                                   area.shape,
                                                   mask[linesmin:linesmax,
                                                        colsmin:colsmax])

    elif mask is None:
        image = two_step_fast_gradient_search(data[linesmin:linesmax,
                                                   colsmin:colsmax],
                                              projection_x_coords, projection_y_coords,
                                              chunk_size,
                                              area.area_extent,
                                              area.shape)
    else:
        image = two_step_fast_gradient_search_with_mask(data[linesmin:linesmax,
                                                             colsmin:colsmax],
                                                        projection_x_coords, projection_y_coords,
                                                        chunk_size,
                                                        area.area_extent,
                                                        area.shape,
                                                        mask[linesmin:linesmax,
                                                             colsmin:colsmax])

    logger.debug("min %f max  %f", image.min(), image.max())

    toc = datetime.now()

    logger.debug("resampling took %s", str(toc - tic))
    logger.debug("from which gradient search took %s", str(toc - tic2))
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
    logger = logging.getLogger("gradient_search")
    # avhrr example
    # t = datetime(2013, 3, 18, 8, 15, 22, 352000)
    # g = PolarFactory.create_scene("noaa", "16", "avhrr", t, orbit="64374")
    # g.load()

    # from mpop.projector import get_area_def

    # area = get_area_def("scan2")

    # for comparison

    # tic = datetime.now()
    # l = g.project(area)
    # toc = datetime.now()
    # print "pyresample took", toc - tic

    # res = gradient_search(g[0.6].data, g.area.lons, g.area.lats, area)

    # show(res)

    # modis example

    # t = datetime(2012, 12, 10, 10, 29, 35)
    # g = PolarFactory.create_scene("terra", "", "modis", t)
    # g.load([0.635, 0.85], resolution=1000)
    # g.load([10.8])

    # from mpop.projector import get_area_def

    # area = get_area_def("euron1")

    # for comparison

    # tic = datetime.now()
    # l = g.project(area)
    # toc = datetime.now()
    # print "pyresample took", toc - tic

    # res = gradient_search(g[0.635].data.astype(np.float64),
    # g[0.635].area.lons, g[0.635].area.lats, area, 10)

    # for wl in [0.635, 0.85]:

    #     res = gradient_search(g[wl].data.astype(
    #         np.float64), g[wl].area.lons, g[wl].area.lats, area, 10)
    #     g[wl] = np.ma.masked_values(res, 0)

    # for wl in [10.8]:

    #     res = gradient_search(g[wl].data.astype(
    #         np.float64), g[wl].area.lons, g[wl].area.lats, area, 10)
    #     g[wl] = np.ma.masked_values(res, 0)

    # show(res)
    # g.image.overview().show()

    # npp example

    t = datetime(2013, 6, 11, 2, 17)
    t1 = datetime(2013, 6, 11, 2, 20)
    t2 = datetime(2013, 6, 11, 2, 27)
    g = PolarFactory.create_scene("npp", "", "viirs", t, orbit="08395")
    wl = 10.8
    chunk = 16
    wl = "I05"
    chunk = 32
    #chunk = 0
    g.load([wl], time_interval=(t1, t2))
    from mpop.projector import get_area_def

    print g[wl].area.lons[26:38, (640 + 368) * 2 - 5:(640 + 368) * 2 + 5]
    print g[wl].area.lats[26:38, (640 + 368) * 2 - 5:(640 + 368) * 2 + 5]

    area = get_area_def("test250")

    mask = g[wl].data.mask.astype(np.uint8)
    lons = g[wl].area.lons.data
    lats = g[wl].area.lats.data
    data = g[wl].data.data.astype(np.double)
    res = gradient_search(data, lons, lats, area, chunk, mask)
    print "max valid", np.max(g[wl].data.compressed())
    print "min valid", np.min(g[wl].data.compressed())
    print "max res", np.max(res)
    print "mask", np.min(mask), np.max(mask)
    print data.shape, mask.shape
    show(np.ma.masked_equal(res, 0.0))

    # for comparison

    # tic = datetime.now()
    # l = g.project(area)
    # toc = datetime.now()
    # print "pyresample took", toc - tic
    # l.image.hr_overview().show()
