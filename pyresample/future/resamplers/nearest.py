#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Pyresample developers
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License along
# with this program.  If not, see <http://www.gnu.org/licenses/>.
"""Nearest neighbor resampler."""

from __future__ import annotations

from .resampler import Resampler

import warnings
from copy import deepcopy
from logging import getLogger

import numpy as np

from pykdtree.kdtree import KDTree
from pyresample import CHUNK_SIZE, geometry

logger = getLogger(__name__)

try:
    from xarray import DataArray
    import dask.array as da
    import dask
except ImportError:
    DataArray = None
    da = None
    dask = None


def lonlat2xyz(lons, lats):
    """Convert lon/lat degrees to geocentric x/y/z coordinates."""
    R = 6370997.0
    x_coords = R * np.cos(np.deg2rad(lats)) * np.cos(np.deg2rad(lons))
    y_coords = R * np.cos(np.deg2rad(lats)) * np.sin(np.deg2rad(lons))
    z_coords = R * np.sin(np.deg2rad(lats))

    stack = np.stack if isinstance(lons, np.ndarray) else da.stack
    return stack(
        (x_coords.ravel(), y_coords.ravel(), z_coords.ravel()), axis=-1)


def query_no_distance(target_lons, target_lats, valid_output_index,
                      mask=None, valid_input_index=None,
                      neighbours=None, epsilon=None, radius=None,
                      kdtree=None):
    """Query the kdtree. No distances are returned.

    NOTE: Dask array arguments must always come before other keyword arguments
          for `da.blockwise` arguments to work.
    """
    voi = valid_output_index
    shape = voi.shape + (neighbours,)
    voir = voi.ravel()
    if mask is not None:
        mask = mask.ravel()[valid_input_index.ravel()]
    target_lons_valid = target_lons.ravel()[voir]
    target_lats_valid = target_lats.ravel()[voir]

    coords = lonlat2xyz(target_lons_valid, target_lats_valid)
    distance_array, index_array = kdtree.query(
        coords,
        k=neighbours,
        eps=epsilon,
        distance_upper_bound=radius,
        mask=mask)

    if index_array.ndim == 1:
        index_array = index_array[:, None]

    # KDTree query returns out-of-bounds neighbors as `len(arr)`
    # which is an invalid index, we mask those out so -1 represents
    # invalid values
    # voi is 2D (trows, tcols)
    # index_array is 2D (valid output pixels, neighbors)
    # there are as many Trues in voi as rows in index_array
    good_pixels = index_array < kdtree.n
    res_ia = np.empty(shape, dtype=int)
    mask = np.zeros(shape, dtype=bool)
    mask[voi, :] = good_pixels
    res_ia[mask] = index_array[good_pixels]
    res_ia[~mask] = -1
    return res_ia


def _my_index(index_arr, vii, data_arr, vii_slices=None, ia_slices=None,
              fill_value=np.nan):
    """Wrap index logic for 'get_sample_from_neighbour_info' to be used inside dask map_blocks."""
    vii_slices = tuple(
        x if x is not None else vii.ravel() for x in vii_slices)
    mask_slices = tuple(
        x if x is not None else (index_arr == -1) for x in ia_slices)
    ia_slices = tuple(
        x if x is not None else index_arr for x in ia_slices)
    res = data_arr[vii_slices][ia_slices]
    res[mask_slices] = fill_value
    return res


class NearestNeighborResampler(Resampler):
    """Resampler using the basic nearest neighbor algorithm."""

    def __init__(self,
                 source_geo_def,
                 target_geo_def,
                 radius_of_influence=None,
                 neighbours=1,
                 epsilon=0):
        """Resampler for xarray DataArrays using a nearest neighbor algorithm.

        Parameters
        ----------
        source_geo_def : object
            Geometry definition of source
        target_geo_def : object
            Geometry definition of target
        radius_of_influence : float, optional
            Cut off distance in geocentric meters.
            If not provided this will be estimated based on the source
            and target geometry definition.
        neighbours : int, optional
            The number of neigbours to consider for each grid point.
            Default 1. Currently 1 is the only supported number.
        epsilon : float, optional
            Allowed uncertainty in meters. Increasing uncertainty
            reduces execution time

        """
        if DataArray is None:
            raise ImportError("Missing 'xarray' and 'dask' dependencies")

        self.valid_input_index = None
        self.valid_output_index = None
        self.index_array = None
        self.distance_array = None
        self.delayed_kdtree = None
        self.neighbours = neighbours
        self.epsilon = epsilon
        self.source_geo_def = source_geo_def
        self.target_geo_def = target_geo_def
        if radius_of_influence is None:
            radius_of_influence = self._compute_radius_of_influence()
        self.radius_of_influence = radius_of_influence
        assert (self.target_geo_def.ndim == 2), \
            "Target area definition must be 2 dimensions"

    def _compute_radius_of_influence(self):
        """Estimate a good default radius_of_influence."""
        try:
            src_res = self.source_geo_def.geocentric_resolution()
        except RuntimeError:
            logger.warning("Could not calculate source definition resolution")
            src_res = np.nan
        try:
            dst_res = self.target_geo_def.geocentric_resolution()
        except RuntimeError:
            logger.warning("Could not calculate destination definition "
                           "resolution")
            dst_res = np.nan
        radius_of_influence = np.nanmax([src_res, dst_res])
        if np.isnan(radius_of_influence):
            logger.warning("Could not calculate radius_of_influence, falling "
                           "back to 10000 meters. This may produce lower "
                           "quality results than expected.")
            radius_of_influence = 10000
        return radius_of_influence

    def _create_resample_kdtree(self, chunks=CHUNK_SIZE):
        """Set up kd tree on input."""
        source_lons, source_lats = self.source_geo_def.get_lonlats(
            chunks=chunks)
        valid_input_idx = ((source_lons >= -180) & (source_lons <= 180) & (source_lats <= 90) & (source_lats >= -90))
        input_coords = lonlat2xyz(source_lons, source_lats)
        input_coords = input_coords[valid_input_idx.ravel(), :]

        # Build kd-tree on input
        input_coords = input_coords.astype(np.float)
        delayed_kdtree = dask.delayed(KDTree, pure=True)(input_coords)
        return valid_input_idx, delayed_kdtree

    def query_resample_kdtree(self,
                              resample_kdtree,
                              tlons,
                              tlats,
                              valid_oi,
                              mask):
        """Query kd-tree on slice of target coordinates."""
        if mask is None:
            args = tuple()
        else:
            ndims = self.source_geo_def.ndim
            dims = 'mn'[:ndims]
            args = (mask, dims, self.valid_input_index, dims)
        # res.shape = rows, cols, neighbors
        # j=rows, i=cols, k=neighbors, m=source rows, n=source cols
        res = da.blockwise(
            query_no_distance, 'jik', tlons, 'ji', tlats, 'ji',
            valid_oi, 'ji', *args, kdtree=resample_kdtree,
            neighbours=self.neighbours, epsilon=self.epsilon,
            radius=self.radius_of_influence, dtype=np.int,
            new_axes={'k': self.neighbours}, concatenate=True)
        return res, None

    def get_neighbour_info(self, mask=None):
        """Return neighbour info.

        Returns
        -------
        (valid_input_index, valid_output_index,
        index_array, distance_array) : tuple of numpy arrays
            Neighbour resampling info
        """
        if self.source_geo_def.size < self.neighbours:
            warnings.warn('Searching for %s neighbours in %s data points' %
                          (self.neighbours, self.source_geo_def.size))

        # Create kd-tree
        chunks = mask.chunks if mask is not None else CHUNK_SIZE
        valid_input_idx, resample_kdtree = self._create_resample_kdtree(
            chunks=chunks)
        self.valid_input_index = valid_input_idx
        self.delayed_kdtree = resample_kdtree

        # TODO: Add 'chunks' keyword argument to this method and use it
        target_lons, target_lats = self.target_geo_def.get_lonlats(chunks=CHUNK_SIZE)
        valid_output_idx = ((target_lons >= -180) & (target_lons <= 180) & (target_lats <= 90) & (target_lats >= -90))

        if mask is not None:
            assert (mask.shape == self.source_geo_def.shape), \
                "'mask' must be the same shape as the source geo definition"
            mask = mask.data
        index_arr, distance_arr = self.query_resample_kdtree(
            resample_kdtree, target_lons, target_lats, valid_output_idx, mask)

        self.valid_output_index, self.index_array = valid_output_idx, index_arr
        self.distance_array = distance_arr

        return (self.valid_input_index,
                self.valid_output_index,
                self.index_array,
                self.distance_array)

    def get_sample_from_neighbour_info(self, data, fill_value=np.nan):
        """Get the pixels matching the target area.

        This method should work for any dimensionality of the provided data
        array as long as the geolocation dimensions match in size and name in
        ``data.dims``. Where source area definition are `AreaDefinition`
        objects the corresponding dimensions in the data should be
        ``('y', 'x')``.

        This method also attempts to preserve chunk sizes of dask arrays,
        but does require loading/sharing the fully computed source data before
        it can actually compute the values to write to the destination array.
        This can result in large memory usage for large source data arrays,
        but is a necessary evil until fancier indexing is supported by dask
        and/or pykdtree.

        Args:
            data (xarray.DataArray): Source data pixels to sample
            fill_value (float): Output fill value when no source data is
                near the target pixel. When omitted, if the input data is an
                integer array then the maximum value for that integer type is
                used, but otherwise, NaN is used and can be detected in the
                result with ``res.isnull()``.

        Returns:
            dask.array.Array: The resampled array. The dtype of the array will
                be the same as the input data. Pixels with no matching data from
                the input array will be filled (see the `fill_value` parameter
                description above).
        """
        if fill_value is not None and np.isnan(fill_value) and \
                np.issubdtype(data.dtype, np.integer):
            fill_value = _get_fill_mask_value(data.dtype)
            logger.warning("Fill value incompatible with integer data "
                           "using {:d} instead.".format(fill_value))

        # Convert back to 1 neighbor
        if self.neighbours > 1:
            raise NotImplementedError("Nearest neighbor resampling can not "
                                      "handle more than 1 neighbor yet.")
        # Convert from multiple neighbor shape to 1 neighbor
        ia = self.index_array[:, :, 0]
        vii = self.valid_input_index

        if isinstance(self.source_geo_def, geometry.SwathDefinition):
            # could be 1D or 2D
            src_geo_dims = self.source_geo_def.lons.dims
        else:
            # assume AreaDefinitions and everything else are 2D with 'y', 'x'
            src_geo_dims = ('y', 'x')
        dst_geo_dims = ('y', 'x')
        # verify that source dims are the same between geo and data
        data_geo_dims = tuple(d for d in data.dims if d in src_geo_dims)
        assert (data_geo_dims == src_geo_dims), \
            "Data dimensions do not match source area dimensions"
        # verify that the dims are next to each other
        first_dim_idx = data.dims.index(src_geo_dims[0])
        num_dims = len(src_geo_dims)
        assert (data.dims[first_dim_idx:first_dim_idx + num_dims] == data_geo_dims), \
            "Data's geolocation dimensions are not consecutive."

        # FIXME: Can't include coordinates whose dimensions depend on the geo
        #        dims either
        def contain_coords(var, coord_list):
            return bool(set(coord_list).intersection(set(var.dims)))

        coords = {c: c_var for c, c_var in data.coords.items()
                  if not contain_coords(c_var, src_geo_dims + dst_geo_dims)}
        try:
            # get these as numpy arrays because xarray is going to compute them anyway
            coord_x, coord_y = self.target_geo_def.get_proj_vectors()
            coords['y'] = coord_y
            coords['x'] = coord_x
        except AttributeError:
            logger.debug("No geo coordinates created")

        # shape of the source data after we flatten the geo dimensions
        flat_src_shape = []
        # slice objects to index in to the source data
        vii_slices = []
        ia_slices = []
        # whether we have seen the geo dims in our analysis
        geo_handled = False
        # dimension indexes for da.blockwise
        src_adims = []
        flat_adim = []
        # map source dimension name to dimension number for da.blockwise
        src_dim_to_ind = {}
        # destination array dimension indexes for da.blockwise
        dst_dims = []
        for i, dim in enumerate(data.dims):
            src_dim_to_ind[dim] = i
            if dim in src_geo_dims and not geo_handled:
                flat_src_shape.append(-1)
                vii_slices.append(None)  # mark for replacement
                ia_slices.append(None)  # mark for replacement
                flat_adim.append(i)
                src_adims.append(i)
                dst_dims.extend(dst_geo_dims)
                geo_handled = True
            elif dim not in src_geo_dims:
                flat_src_shape.append(data.sizes[dim])
                vii_slices.append(slice(None))
                ia_slices.append(slice(None))
                src_adims.append(i)
                dst_dims.append(dim)
        # map destination dimension names to blockwise dimension indexes
        dst_dim_to_ind = src_dim_to_ind.copy()
        dst_dim_to_ind['y'] = i + 1
        dst_dim_to_ind['x'] = i + 2
        # FUTURE: when we allow more than one neighbor
        # neighbors_dim = i + 3

        new_data = data.data.reshape(flat_src_shape)
        vii = vii.ravel()
        dst_adims = [dst_dim_to_ind[dim] for dim in dst_dims]
        ia_adims = [dst_dim_to_ind[dim] for dim in dst_geo_dims]
        # FUTURE: when we allow more than one neighbor add neighbors dimension
        # dst_adims.append(neighbors_dim)
        # ia_adims.append(neighbors_dim)
        # FUTURE: when we allow more than one neighbor we need to add
        #         the new axis to blockwise:
        #         `new_axes={neighbor_dim: self.neighbors}`
        # FUTURE: if/when dask can handle index arrays that are dask arrays
        #         then we can avoid all of this complicated blockwise stuff
        res = da.blockwise(
            _my_index, dst_adims,
            ia, ia_adims,
            vii, flat_adim,
            new_data, src_adims,
            vii_slices=vii_slices, ia_slices=ia_slices,
            fill_value=fill_value,
            dtype=new_data.dtype, concatenate=True)
        res = DataArray(res, dims=dst_dims, coords=coords,
                        attrs=deepcopy(data.attrs))

        return res


def _get_fill_mask_value(data_dtype):
    """Return the maximum value of dtype."""
    if issubclass(data_dtype.type, np.floating):
        fill_value = np.finfo(data_dtype.type).max
    elif issubclass(data_dtype.type, np.integer):
        fill_value = np.iinfo(data_dtype.type).max
    else:
        raise TypeError('Type %s is unsupported for masked fill values' %
                        data_dtype.type)
    return fill_value


def _remask_data(data, is_to_be_masked=True):
    """Interprets half the array as mask for the other half."""
    channels = data.shape[-1]
    if is_to_be_masked:
        mask = data[..., (channels // 2):]
        # All pixels affected by masked pixels are masked out
        mask = (mask != 0)
        data = np.ma.array(data[..., :(channels // 2)], mask=mask)
    else:
        data = data[..., :(channels // 2)]

    if data.shape[-1] == 1:
        data = data.reshape(data.shape[:-1])
    return data
