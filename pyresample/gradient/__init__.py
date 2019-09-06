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
                              fast_gradient_search,
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

    reduce_data = True

    if reduce_data:
        lon_bound, lat_bound = area.get_boundary_lonlats()
        idx = data_reduce.get_valid_index_from_lonlat_boundaries(lon_bound,
                                                                 lat_bound,
                                                                 lons, lats,
                                                                 5000).compute()
        logger.debug('data reduction takes: %s', str(datetime.now() - tic))
        colsmin, colsmax = np.arange(
            lons.shape[1])[np.sum(idx, 0, bool)][[0, -1]]
        linesmin, linesmax = np.arange(
            lons.shape[0])[np.sum(idx, 1, bool)][[0, -1]]

        if chunk_size != 0:
            linesmin -= linesmin % chunk_size
            linesmax += chunk_size
            linesmax -= linesmax % chunk_size

        red_lons = lons[linesmin:linesmax, colsmin:colsmax]
        red_lats = lats[linesmin:linesmax, colsmin:colsmax]
        result = da.map_blocks(_get_proj_coordinates,
                               red_lons,
                               red_lats, prj, new_axis=0,
                               chunks=(2,) + red_lons.chunks)
        idxs = ((red_lons > 180.0) | (red_lons < -180.0) |
                (red_lats > 90.0) | (red_lats < -90.0))
        projection_x_coords = da.where(idxs, np.nan, result[0, :])
        projection_y_coords = da.where(idxs, np.nan, result[1, :])

    else:
        result = da.map_blocks(_get_proj_coordinates,
                               lons,
                               lats, prj, new_axis=0,
                               chunks=(2,) + lons.chunks)
        idxs = (lons > 180.0) | (lons < -180.0) | (lats > 90.0) | (lats < -90.0)
        projection_x_coords = da.where(idxs, np.nan, result[0, :])
        projection_y_coords = da.where(idxs, np.nan, result[1, :])

    toc3 = datetime.now()

    toc2 = datetime.now()
    logger.debug("pyproj took %s", str(toc2 - toc3))

    tic2 = datetime.now()

    # x_min, y_min, x_max, y_max = area.area_extent
    # y_size, x_size = area.shape
    # x_inc = (x_max - x_min) / x_size
    # x_1d = da.arange(x_min + x_inc / 2, x_max, x_inc, chunks=CHUNK_SIZE)
    # Y dimension is upside-down
    # y_inc = (y_min - y_max) / y_size
    # y_1d = da.arange(y_max - y_inc / 2, y_min, y_inc, chunks=CHUNK_SIZE)

    indices = None
    image = None
    if chunk_size == 0:
        if mask is None:
            # indices = da.blockwise(blockwise_gradient_indices,
            #                        'kij', x_1d, 'j', y_1d, 'i',
            #                        px=projection_x_coords,
            #                        py=projection_y_coords,
            #                        new_axes={'k': 2}, dtype=np.float)
            image = fast_gradient_search(
                data[linesmin:linesmax,
                     colsmin:colsmax].astype(np.float),
                projection_x_coords.compute(),
                projection_y_coords.compute(),
                area.area_extent,
                area.shape)
        else:
            image = fast_gradient_search_with_mask(
                data[linesmin:linesmax,
                     colsmin:colsmax].astype(np.float),
                projection_x_coords.compute(),
                projection_y_coords.compute(),
                area.area_extent,
                area.shape,
                mask[linesmin:linesmax,
                     colsmin:colsmax].astype(np.uint8))

    elif mask is None:
        image = two_step_fast_gradient_search(data[linesmin:linesmax,
                                                   colsmin:colsmax],
                                              projection_x_coords,
                                              projection_y_coords,
                                              chunk_size,
                                              area.area_extent,
                                              area.shape)
    else:
        image = two_step_fast_gradient_search_with_mask(
            data[linesmin:linesmax,
                 colsmin:colsmax],
            projection_x_coords,
            projection_y_coords,
            chunk_size,
            area.area_extent,
            area.shape,
            mask[linesmin:linesmax,
                 colsmin:colsmax])

    toc = datetime.now()

    logger.debug("resampling took %s", str(toc - tic))
    logger.debug("from which gradient search took %s", str(toc - tic2))

    if indices is None:
        return image
    if reduce_data:
        return indices, data[linesmin:linesmax, colsmin:colsmax]
    else:
        return indices, data


def show(data):
    """Show the stretched data."""
    from PIL import Image as pil
    img = pil.fromarray(np.array((data - np.nanmin(data)) * 255.0 /
                                 (np.nanmax(data) - np.nanmin(data)), np.uint8))
    img.show()


def main():
    """Run some tests."""
    import sys

    from satpy import Scene
    from satpy.resample import get_area_def

    use_mask = False

    filenames = sorted(sys.argv[1:])
    glbl = Scene(
        filenames=filenames,
        filter_parameters={'area': 'euron1'}
    )
    glbl.load([10.8])

    area = get_area_def("euron1")

    tic = datetime.now()
    lons, lats = glbl[10.8].area.get_lonlats(chunks=CHUNK_SIZE)
    data = glbl[10.8].values

    if use_mask:
        mask = (lons > 180.0) | (lons < -180.0) | (lats > 90.0) | (lats < -90.0)
        image = gradient_search(data,
                                lons,
                                lats,
                                area,
                                mask=mask.compute())
    else:
        image = gradient_search(data,
                                lons,
                                lats,
                                area)

        # tic2 = datetime.now()
        # idx = idx.astype(np.int)
        # idx_x = idx[0, :, :]
        # idx_y = idx[1, :, :]
        # cidx, cidy = da.compute(idx_x, idx_y)
        # image = data[cidx, cidy]
        # The fill value for indices is -2, replace those with NaN
        # image = da.where(cidx == -2, np.nan, image)
        import xarray as xr
        image = da.where(image == 0, np.nan, image)

    glbl['image'] = xr.DataArray(image)
    glbl.save_dataset('image', '/tmp/gradient.png')
    del glbl['image']

    toc = datetime.now()
    # logger.debug("slicing and saving took %s", str(toc - tic2))
    logger.debug("gradient search took %s", str(toc - tic))
    # show(image)

    # for comparison
    tic = datetime.now()
    lcl = glbl.resample(area)
    lcl.save_dataset(10.8, '/tmp/kdtree_nearest.png')
    toc = datetime.now()
    logger.debug("kd-tree took %s", str(toc - tic))


if __name__ == '__main__':
    main()
