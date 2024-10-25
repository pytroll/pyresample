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

import warnings
from copy import deepcopy
from logging import getLogger

import numpy as np
from pykdtree.kdtree import KDTree

from pyresample import CHUNK_SIZE, geometry
from pyresample.utils.errors import PerformanceWarning

from ..geometry import StaticGeometry, SwathDefinition
from ._transform_utils import lonlat2xyz
from .resampler import Resampler, update_resampled_coords

logger = getLogger(__name__)

try:
    import dask
    import dask.array as da
    from xarray import DataArray
except ImportError:
    DataArray = None
    da = None
    dask = None


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


# TODO: Add decorator for geom<->geom support
# TODO: Add decorator for object type support
# Must be decorators so that we can both add class attributes with this information
# and add decorators to __init__ and precompute/resample calls for validity checking
class KDTreeNearestXarrayResampler(Resampler):
    """Resampler using the basic nearest neighbor algorithm."""

    def __init__(self,
                 source_geo_def: StaticGeometry,
                 target_geo_def: StaticGeometry,
                 cache=None):
        """Resampler for xarray DataArrays using a nearest neighbor algorithm.

        Parameters
        ----------
        source_geo_def : object
            Geometry definition of source
        target_geo_def : object
            Geometry definition of target

        """
        if DataArray is None:
            raise ImportError("Missing 'xarray' and 'dask' dependencies")
        super().__init__(source_geo_def, target_geo_def, cache=cache)
        self._internal_cache: dict[tuple, dict] = {}
        if self.target_geo_def.ndim != 2:
            raise ValueError("Target area definition must be 2 dimensions")

    @property
    def version(self) -> str:
        """Get the current version of this class used for hashing and caching."""
        return "0.1"

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
        input_coords = input_coords.astype(np.float64)
        delayed_kdtree = dask.delayed(KDTree, pure=True)(input_coords)
        return valid_input_idx, delayed_kdtree

    def _query_resample_kdtree(self,
                               resample_kdtree,
                               tlons,
                               tlats,
                               valid_input_index,
                               valid_output_index,
                               mask,
                               neighbors,
                               radius_of_influence,
                               epsilon):
        """Query kd-tree on slice of target coordinates."""
        if mask is None:
            args = tuple()
        else:
            ndims = self.source_geo_def.ndim
            dims = 'mn'[:ndims]
            args = (mask, dims, valid_input_index, dims)
        # res.shape = rows, cols, neighbors
        # j=rows, i=cols, k=neighbors, m=source rows, n=source cols
        res = da.blockwise(
            query_no_distance, 'jik', tlons, 'ji', tlats, 'ji',
            valid_output_index, 'ji', *args, kdtree=resample_kdtree,
            neighbours=neighbors, epsilon=epsilon,
            radius=radius_of_influence, dtype=np.int64,
            meta=np.array((), dtype=np.int64),
            new_axes={'k': neighbors}, concatenate=True)
        return res

    def _get_neighbor_info(self, mask, neighbors, radius_of_influence, epsilon):
        """Return neighbour info.

        Returns
        -------
        (valid_input_index, valid_output_index,
        index_array, distance_array) : tuple of numpy arrays
            Neighbour resampling info
        """
        if self.source_geo_def.size < neighbors:
            warnings.warn('Searching for %s neighbors in %s data points' %
                          (neighbors, self.source_geo_def.size), stacklevel=3)

        # Create kd-tree
        chunks = mask.chunks if mask is not None else CHUNK_SIZE
        valid_input_idx, resample_kdtree = self._create_resample_kdtree(chunks=chunks)

        # TODO: Add 'chunks' keyword argument to this method and use it
        target_lons, target_lats = self.target_geo_def.get_lonlats(chunks=CHUNK_SIZE)
        valid_output_idx = ((target_lons >= -180) & (target_lons <= 180) & (target_lats <= 90) & (target_lats >= -90))

        if mask is not None:
            if mask.shape != self.source_geo_def.shape:
                raise ValueError("'mask' must be the same shape as the source geo definition")
            mask = mask.data
        index_arr = self._query_resample_kdtree(
            resample_kdtree, target_lons, target_lats, valid_input_idx,
            valid_output_idx, mask,
            neighbors, radius_of_influence, epsilon)

        return valid_input_idx, index_arr

    def get_sample_from_neighbor_info(
            self,
            data,
            valid_input_index,
            index_array,
            neighbors=1,
            fill_value=np.nan):
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
            valid_input_index (ArrayLike): Index array of valid pixels in
                the input geolocation data.
            index_array (ArrayLike): Index array of nearest neighbors.
            neighbors (int): Number of neighbors to return for each
                data pixel. Currently only 1 (the default) is supported.
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
        if neighbors > 1:
            raise NotImplementedError("Nearest neighbor resampling can not "
                                      "handle more than 1 neighbor yet.")
        # Convert from multiple neighbor shape to 1 neighbor
        ia = index_array[:, :, 0]
        vii = valid_input_index

        src_geo_dims = self._get_src_geo_dims()
        dst_geo_dims = ('y', 'x')
        self._verify_data_geo_dims(data, src_geo_dims)

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
        #         `new_axes={neighbor_dim: neighbors}`
        # FUTURE: if/when dask can handle index arrays that are dask arrays
        #         then we can avoid all of this complicated blockwise stuff
        res = da.blockwise(
            _my_index, dst_adims,
            ia, ia_adims,
            vii, flat_adim,
            new_data, src_adims,
            vii_slices=vii_slices, ia_slices=ia_slices,
            fill_value=fill_value,
            meta=np.array((), dtype=new_data.dtype),
            dtype=new_data.dtype, concatenate=True)
        res = DataArray(res, dims=dst_dims,
                        attrs=deepcopy(data.attrs))
        res = update_resampled_coords(data, res, self.target_geo_def)
        return res

    def _verify_data_geo_dims(self, data, src_geo_dims):
        # verify that source dims are the same between geo and data
        data_geo_dims = tuple(d for d in data.dims if d in src_geo_dims)
        if data_geo_dims != src_geo_dims:
            raise ValueError("Data dimensions do not match source area dimensions.")

        # verify that the dims are next to each other
        first_dim_idx = data.dims.index(src_geo_dims[0])
        num_dims = len(src_geo_dims)
        if data.dims[first_dim_idx:first_dim_idx + num_dims] != data_geo_dims:
            raise ValueError("Data's geolocation dimensions are not consecutive.")

        # verify that the dims are the same between src geom and data
        for dim_offset, dim_name in enumerate(data_geo_dims):
            geom_size = self.source_geo_def.shape[dim_offset]
            data_size = data.shape[first_dim_idx + dim_offset]
            if geom_size != data_size:
                raise ValueError("Input data shape is not equal to the shape of "
                                 "the source geometry: "
                                 f"{dim_name}={data_size} versus {geom_size}")

    def _get_src_geo_dims(self):
        if isinstance(self.source_geo_def, geometry.SwathDefinition):
            # could be 1D or 2D
            src_geo_dims = self.source_geo_def.lons.dims
        else:
            # assume AreaDefinitions and everything else are 2D with 'y', 'x'
            src_geo_dims = ('y', 'x')
        return src_geo_dims

    def precompute(self, mask=None, radius_of_influence=None, epsilon=0):
        """Generate neighbor indexes using geolocation information and optional data mask.

        Args:
            mask (ArrayLike, optional):
                Boolean array where True represents invalid pixels in the data
                array to be resampled in the future. This allows the indexes
                computed by this method and used during resampling to filter
                out invalid values and produce a result with more overall valid
                pixels. If provided then pre-computed results will not be
                cached as it is assumed that the mask will likely change for
                every input array.
            radius_of_influence (float, optional):
                Cut off distance in geocentric meters.
                If not provided this will be estimated based on the source
                and target geometry definition.
            epsilon (float, optional):
                Allowed uncertainty in meters. Increasing uncertainty
                reduces execution time

        """
        neighbors = 1
        if mask is not None and mask.shape != self.source_geo_def.shape:
            raise ValueError("'mask' provided to 'precompute' is not the same "
                             "shape as the source geometry.")
        if radius_of_influence is None:
            radius_of_influence = self._compute_radius_of_influence()

        # use dask task name
        mask_hash = None if mask is None else mask.data.name
        internal_cache_key = (mask_hash, neighbors, radius_of_influence, epsilon)
        in_int_cache = internal_cache_key in self._internal_cache
        if not in_int_cache:
            valid_input_index, index_arr = self._get_neighbor_info(
                mask, neighbors, radius_of_influence, epsilon)
            item_to_cache = {
                "valid_input_index": valid_input_index,
                "index_array": index_arr,
            }
            self._internal_cache[internal_cache_key] = item_to_cache

    def resample(self, data, mask_area=None, fill_value=np.nan,
                 radius_of_influence=None, epsilon=0):
        """Resample input ``data`` from the source geometry to the target geometry.

        Args:
            data (ArrayLike): Data to be resampled
            mask_area (bool or ArrayLike): Mask geolocation data where data
                values are invalid. This should be used when data values may
                affect what neighbors are considered valid. For complex masking
                behavior this can also be the mask array to use instead of the
                resampler computing it on the fly. This is useful
                for non-xarray DataArrays that don't have related
                metadata like "_FillValue". By default this is None which means
                a mask will be created automatically for SwathDefinition input.
                Set to ``False`` to disable any masking and ``True`` to ensure
                a mask is created. See :meth:`precompute` for more information.
            fill_value (int or float): Output fill value when no source data is
                near the target pixel. When omitted, if the input data is an
                integer array then the maximum value for that integer type is
                used, but otherwise, NaN is used and can be detected in the
                result with ``res.isnull()`` for DataArray output and
                ``np.isnan(res)`` for dask and numpy arrays.
            radius_of_influence (float, optional):
                Passed directly to :meth:`precompute`.
            epsilon (float, optional):
                Passed directly to :meth:`precompute`.

        Returns:
            Array-like object of the same type as ``data``, resampled to the
            target geographic geometry.

        """
        new_data = self._verify_input_object_type(data)
        src_geo_dims = self._get_src_geo_dims()
        self._verify_data_geo_dims(new_data, src_geo_dims)

        mask = self._get_area_mask(mask_area, new_data)
        if radius_of_influence is None:
            radius_of_influence = self._compute_radius_of_influence()
        self.precompute(mask=mask, radius_of_influence=radius_of_influence, epsilon=epsilon)

        # Get precomputed arrays - use dask array task name
        mask_hash = None if mask is None else mask.data.name
        cache_key = (mask_hash, 1, radius_of_influence, epsilon)
        precompute_dict = self._internal_cache[cache_key]
        result = self.get_sample_from_neighbor_info(
            new_data,
            precompute_dict["valid_input_index"],
            precompute_dict["index_array"],
            fill_value=fill_value)
        return self._verify_result_object_type(result, data)

    def _verify_input_object_type(self, data):
        if isinstance(data, DataArray) and isinstance(data.data, da.Array):
            return data
        if not isinstance(data, DataArray):
            if data.ndim != 2:
                raise ValueError(
                    f"{self.__class__.__name__} requires DataArrays with a dask "
                    "array. Input array is not 2D and can't be automatically "
                    "converted.")
            data = DataArray(data, dims=self._get_src_geo_dims())
        if not isinstance(data.data, da.Array):
            warnings.warn(
                f"{self.__class__.__name__} uses a dask-based implementation, "
                "but a pure numpy array was provided. Data will be converted "
                "to dask arrays for computation and then converted back. To "
                "avoid this warning convert your numpy array before providing "
                "it to the resampler.", PerformanceWarning, stacklevel=3)
            data = data.copy()
            data.data = da.from_array(data.data, chunks="auto")
        return data

    def _verify_result_object_type(self, data, orig_data):
        if isinstance(orig_data, DataArray) and isinstance(orig_data.data, da.Array):
            return data
        if not isinstance(orig_data, DataArray):
            data = data.data
        if isinstance(orig_data, da.Array):
            return data
        return data.compute()

    def _get_area_mask(self, mask_area, data):
        if isinstance(mask_area, (np.ndarray, da.Array, DataArray)):
            return mask_area

        # default is to mask areas for SwathDefinitions
        if mask_area is None and isinstance(self.source_geo_def, SwathDefinition):
            mask_area = True

        if mask_area:
            return self.compute_data_mask(data)

    def compute_data_mask(self, data):
        """Generate a mask array where data is invalid.

        This is used by :meth:`resample` to determine what ``mask`` to pass
        to :meth:`precompute`. It may be useful for users to use this manually
        for special cases of wanting to call `precompute` manually.

        """
        geo_dims = self._get_src_geo_dims()
        flat_dims = [dim for dim in data.dims if dim not in geo_dims]
        if np.issubdtype(data.dtype, np.integer):
            mask = data == data.attrs.get('_FillValue', np.iinfo(data.dtype.type).max)
        else:
            mask = data.isnull()
        mask = mask.all(dim=flat_dims)
        return mask


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
