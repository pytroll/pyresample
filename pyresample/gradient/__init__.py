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

"""Test the Trishchenko algorithm."""

import dask.array as da
import logging
from pyresample import data_reduce
import pyximport
from datetime import datetime
import numpy as np
import pyproj
from satpy.utils import debug_on
from satpy import CHUNK_SIZE
pyximport.install(setup_args={'include_dirs': np.get_include()})
debug_on()

from _gradient_search import (two_step_fast_gradient_search,
                              two_step_fast_gradient_search_with_mask,
                              fast_gradient_search_with_mask,
                              fast_gradient_indices)  # noqa: E402

logger = logging.getLogger(__name__)


def _get_proj_coordinates(lons, lats, prj):
    """Calculate projection coordinates."""
    proj_x, proj_y = prj(lons, lats)

    return np.stack((proj_x, proj_y))


def blockwise_gradient_indices(x_1d, y_1d, px, py):
    """Find indices with blockwise operation."""
    return fast_gradient_indices(px, py, x_1d, y_1d)


def gradient_search(data, lons, lats, area, chunk_size=0, mask=None):
    """Run gradient search."""
    tic = datetime.now()
    prj = pyproj.Proj(**area.proj_dict)

    lon_bound, lat_bound = area.get_boundary_lonlats()
    idx = data_reduce.get_valid_index_from_lonlat_boundaries(lon_bound,
                                                             lat_bound,
                                                             lons, lats,
                                                             5000).compute()
    logger.debug('data reduction takes: %s', str(datetime.now() - tic))
    colsmin, colsmax = np.arange(lons.shape[1])[np.sum(idx, 0, bool)][[0, -1]]
    linesmin, linesmax = np.arange(
        lons.shape[0])[np.sum(idx, 1, bool)][[0, -1]]

    if chunk_size != 0:
        linesmin -= linesmin % chunk_size
        linesmax += chunk_size
        linesmax -= linesmax % chunk_size

    red_lons = lons[linesmin:linesmax, colsmin:colsmax]
    red_lats = lats[linesmin:linesmax, colsmin:colsmax]
    result = da.map_blocks(_get_proj_coordinates,
                           red_lons.data,
                           red_lats.data, prj, new_axis=0,
                           chunks=(2,) + red_lons.chunks)
    projection_x_coords = result[0, :]
    projection_y_coords = result[1, :]
    toc3 = datetime.now()

    toc2 = datetime.now()
    logger.debug("pyproj took %s", str(toc2 - toc3))

    tic2 = datetime.now()

    x_min, y_min, x_max, y_max = area.area_extent
    y_size, x_size = area.shape
    x_inc = (x_max - x_min) / x_size
    x_1d = da.arange(x_min + x_inc / 2, x_max, x_inc, chunks=CHUNK_SIZE)
    # Y dimension is upside-down
    y_inc = (y_min - y_max) / y_size
    y_1d = da.arange(y_max - y_inc / 2, y_min, y_inc, chunks=CHUNK_SIZE)

    if chunk_size == 0:
        if mask is None:

            indices = da.blockwise(blockwise_gradient_indices, 'kij', x_1d, 'j', y_1d, 'i',
                                   px=projection_x_coords, py=projection_y_coords,
                                   new_axes={'k': 2}, dtype=np.float)
        else:
            _ = fast_gradient_search_with_mask(data[linesmin:linesmax,
                                                    colsmin:colsmax],
                                               projection_x_coords,
                                               projection_y_coords,
                                               area.area_extent,
                                               area.shape,
                                               mask[linesmin:linesmax,
                                                    colsmin:colsmax])

    elif mask is None:
        _ = two_step_fast_gradient_search(data[linesmin:linesmax,
                                               colsmin:colsmax],
                                          projection_x_coords,
                                          projection_y_coords,
                                          chunk_size,
                                          area.area_extent,
                                          area.shape)
    else:
        _ = two_step_fast_gradient_search_with_mask(data[linesmin:linesmax,
                                                         colsmin:colsmax],
                                                    projection_x_coords,
                                                    projection_y_coords,
                                                    chunk_size,
                                                    area.area_extent,
                                                    area.shape,
                                                    mask[linesmin:linesmax,
                                                         colsmin:colsmax])

    # logger.debug("min %f max  %f", image.min(), image.max())

    toc = datetime.now()

    logger.debug("resampling took %s", str(toc - tic))
    logger.debug("from which gradient search took %s", str(toc - tic2))
    return indices


def show(data):
    """Show the stretched data."""
    from PIL import Image as pil
    img = pil.fromarray(np.array((data - data.min()) * 255.0 /
                                 (data.max() - data.min()), np.uint8))
    img.show()
    img.save("/tmp/gradient2.png")


def main():
    """Run some tests."""
    from satpy import Scene
    from glob import glob
    # avhrr example
    filenames = sorted(glob('/home/a001673/data/satellite/metop/*'))
    glbl = Scene(
        sensor='avhrr-3',
        filenames=filenames,
        reader="avhrr_l1b_eps",
        filter_parameters={'area': 'euron1'}
    )
    glbl.load(['4'])

    from satpy.resample import get_area_def

    area = get_area_def("euron1")

    tic = datetime.now()
    idx = gradient_search(glbl['4'].values,
                          glbl['4'].attrs['area'].lons,
                          glbl['4'].attrs['area'].lats,
                          area)

    idx = idx.astype(np.int)
    idx_x = idx[0, :, :]
    idx_y = idx[1, :, :]
    res = glbl['4'].values
    cidx, cidy = da.compute(idx_x, idx_y)
    image = res[cidx, cidy]
    toc = datetime.now()
    print("gradient search took", toc - tic)
    show(image)

    # for comparison

    tic = datetime.now()
    lcl = glbl.resample(area)
    lcl['4'].values
    toc = datetime.now()
    print("kd-tree took", toc - tic)

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

    # t = datetime(2013, 6, 11, 2, 17)
    # t1 = datetime(2013, 6, 11, 2, 20)
    # t2 = datetime(2013, 6, 11, 2, 27)
    # g = PolarFactory.create_scene("npp", "", "viirs", t, orbit="08395")
    # wl = 10.8
    # chunk = 16
    # wl = "I05"
    # chunk = 32
    # #chunk = 0
    # g.load([wl], time_interval=(t1, t2))
    # from mpop.projector import get_area_def
    #
    # print g[wl].area.lons[26:38, (640 + 368) * 2 - 5:(640 + 368) * 2 + 5]
    # print g[wl].area.lats[26:38, (640 + 368) * 2 - 5:(640 + 368) * 2 + 5]
    #
    # area = get_area_def("test250")
    #
    # mask = g[wl].data.mask.astype(np.uint8)
    # lons = g[wl].area.lons.data
    # lats = g[wl].area.lats.data
    # data = g[wl].data.data.astype(np.double)
    # res = gradient_search(data, lons, lats, area, chunk, mask)
    # print "max valid", np.max(g[wl].data.compressed())
    # print "min valid", np.min(g[wl].data.compressed())
    # print "max res", np.max(res)
    # print "mask", np.min(mask), np.max(mask)
    # print data.shape, mask.shape
    # show(np.ma.masked_equal(res, 0.0))
    #
    # # for comparison
    #
    # tic = datetime.now()
    # l = g.project("bsea250")
    # #l = g.project(area)
    # toc = datetime.now()
    # print "pyresample took", toc - tic
    # # l.image.hr_overview().show()


if __name__ == '__main__':
    main()
