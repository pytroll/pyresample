#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019-2021 Pyresample developers
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
"""Base resampler class made for subclassing."""
from __future__ import annotations

import logging
import os
from functools import lru_cache, partial
from numbers import Number
from typing import Union

import numpy as np

from .slicer import _enumerate_chunk_slices, create_slicer

try:
    import dask
    import dask.array as da
    from dask.highlevelgraph import HighLevelGraph
except ImportError:
    da = None

try:
    import xarray as xr
except ImportError:
    xr = None

from pyresample.geometry import AreaDefinition, CoordinateDefinition, IncompatibleAreas, SwathDefinition

from .future.resamplers.resampler import hash_dict

logger = logging.getLogger(__name__)


class BaseResampler:
    """Base abstract resampler class."""

    def __init__(self,
                 source_geo_def: Union[SwathDefinition, AreaDefinition],
                 target_geo_def: Union[CoordinateDefinition, AreaDefinition],
                 ):
        """Initialize resampler with geolocation information.

        Args:
            source_geo_def:
                Geolocation definition for the data to be resampled
            target_geo_def:
                Geolocation definition for the area to resample data to.

        """
        self.source_geo_def = source_geo_def
        self.target_geo_def = target_geo_def

    def get_hash(self, source_geo_def=None, target_geo_def=None, **kwargs):
        """Get hash for the current resample with the given *kwargs*."""
        if source_geo_def is None:
            source_geo_def = self.source_geo_def
        if target_geo_def is None:
            target_geo_def = self.target_geo_def
        the_hash = source_geo_def.update_hash()
        target_geo_def.update_hash(the_hash)
        hash_dict(kwargs, the_hash)
        return the_hash.hexdigest()

    def precompute(self, **kwargs):
        """Do the precomputation.

        This is an optional step if the subclass wants to implement more
        complex features like caching or can share some calculations
        between multiple datasets to be processed.

        """
        return None

    def compute(self, data, **kwargs):
        """Do the actual resampling.

        This must be implemented by subclasses.

        """
        raise NotImplementedError

    def resample(self, data, cache_dir=None, mask_area=None, **kwargs):
        """Resample `data` by calling `precompute` and `compute` methods.

        Only certain resampling classes may use `cache_dir` and the `mask`
        provided when `mask_area` is True. The return value of calling the
        `precompute` method is passed as the `cache_id` keyword argument
        of the `compute` method, but may not be used directly for caching. It
        is up to the individual resampler subclasses to determine how this
        is used.

        Args:
            data (xarray.DataArray): Data to be resampled
            cache_dir (str): directory to cache precomputed results
                             (default False, optional)
            mask_area (bool): Mask geolocation data where data values are
                              invalid. This should be used when data values
                              may affect what neighbors are considered valid.
            kwargs: Keyword arguments to pass to both the ``precompute`` and
                ``compute`` stages of the resampler.

        Returns (xarray.DataArray): Data resampled to the target area

        """
        if self._geometries_are_the_same():
            return data
        # default is to mask areas for SwathDefinitions
        if mask_area is None and isinstance(
                self.source_geo_def, SwathDefinition):
            mask_area = True

        if mask_area:
            if isinstance(self.source_geo_def, SwathDefinition):
                geo_dims = self.source_geo_def.lons.dims
            else:
                geo_dims = ('y', 'x')
            flat_dims = [dim for dim in data.dims if dim not in geo_dims]
            if np.issubdtype(data.dtype, np.integer):
                kwargs['mask'] = data == data.attrs.get('_FillValue', np.iinfo(data.dtype.type).max)
            else:
                kwargs['mask'] = data.isnull()
            kwargs['mask'] = kwargs['mask'].all(dim=flat_dims)

        cache_id = self.precompute(cache_dir=cache_dir, **kwargs)
        return self.compute(data, cache_id=cache_id, **kwargs)

    def _geometries_are_the_same(self):
        """Check if two geometries are the same object and resampling isn't needed.

        For area definitions this is a simple comparison using the ``==``.
        When swaths are involved care is taken to not check coordinate equality
        to avoid the expensive computation. A swath and an area are never
        considered equal in this case even if they describe the same geographic
        region.

        Two swaths are only considered equal if the underlying arrays are the
        exact same objects. Otherwise, they are considered not equal and
        coordinate values are never checked. This has
        the downside that if two SwathDefinitions have equal coordinates but
        are loaded or created separately they will be considered not equal.

        """
        if self.source_geo_def is self.target_geo_def:
            return True
        if type(self.source_geo_def) is not type(self.target_geo_def):  # noqa
            # these aren't the exact same class
            return False
        if isinstance(self.source_geo_def, AreaDefinition):
            return self.source_geo_def == self.target_geo_def
        # swath or coordinate definitions
        src_lons, src_lats = self.source_geo_def.get_lonlats()
        dst_lons, dst_lats = self.target_geo_def.get_lonlats()
        if (src_lons is dst_lons) and (src_lats is dst_lats):
            return True

        if not all(isinstance(arr, da.Array) for arr in (src_lons, src_lats, dst_lons, dst_lats)):
            # they aren't the same object and they aren't dask arrays so not equal
            return False
        # if dask task names are the same then they are the same even if the
        # dask Array instance itself is different
        return src_lons.name == dst_lons.name and src_lats.name == dst_lats.name

    def _create_cache_filename(self, cache_dir=None, prefix='',
                               fmt='.zarr', **kwargs):
        """Create filename for the cached resampling parameters."""
        cache_dir = cache_dir or '.'
        hash_str = self.get_hash(**kwargs)

        return os.path.join(cache_dir, prefix + hash_str + fmt)


def resample_blocks(func, src_area, src_arrays, dst_area,
                    dst_arrays=(), chunk_size=None, dtype=None, name=None, fill_value=None, **kwargs):
    """Resample dask arrays blockwise.

    Resample_blocks applies a function blockwise to transform data from a source
    area domain to a destination area domain.

    Args:
        func: A callable to apply on the input data. This function is passed a block of src_arrays,
            dst_arrays in that order, followed by the kwargs, which include the fill_value. If the callable accepts a
            `block_info` keyword argument, block information is passed to it. Block information provides the source
            area, destination area, position of source and destination blocks relative to respectively `src_area` and
            `dst_area`.
        src_area: a source geo definition.
        dst_area: a destination geo definition. If the same as the source definition, a ValueError is raised.
        src_arrays: data to use. When split into smaller bit to pass to func, they are split across the x and y
            dimensions, but not across the other dimensions, so all the dimensions of the smaller arrays will be using
            only one chunk!
        dst_arrays: arrays to use that are already in dst_area space. If the array has more than 2 dimensions,
            the last two are expected to be y, x.
        chunk_size: the chunks size(s) to use in the dst_area space. This has to be provided since it is not guaranteed
            that we can get this information from the other arguments. Moreover, this needs to be an iterable of k
            elements if the resulting array of func is to have a different number of dimensions (k) than the input
            array.
        dtype: the dtype the resulting array is going to have. Has to be provided.
        name: Name prefix of the dask tasks to be generated
        fill_value: Desired value for any invalid values in the output array
        kwargs: any other keyword arguments that will be passed on to func.


    Principle of operations:
        Resample_blocks works by iterating over chunks on the dst_area domain. For each chunk, the corresponding slice
        of the src_area domain is computed and the input src_arrays are cut accordingly to pass to func. To know more
        about how the slicing is performed, refer to the :class:Slicer class and subclasses.

    Examples:
        To generate indices from the gradient resampler, you can apply the corresponding function with no input. Note
        how we provide the chunk sizes knowing that the result array with have 2 elements along a third dimension.

        >>> indices_xy = resample_blocks(gradient_resampler_indices, source_geo_def, [], target_geo_def,
        ...                              chunk_size=(2, "auto", "auto"), dtype=float)

        From these indices, to resample an array using bilinear interpolation:

        >>>  resampled = resample_blocks(block_bilinear_interpolator, source_geo_def, [src_array], target_geo_def,
        ...                              dst_arrays=[indices_xy],
        ...                              chunk_size=("auto", "auto"), dtype=src_array.dtype)


    """
    if dst_area == src_area:
        raise ValueError("Source and destination areas are identical."
                         " Should you be running `map_blocks` instead of `resample_blocks`?")

    name = _create_dask_name(name, func,
                             src_area, src_arrays,
                             dst_area, dst_arrays,
                             fill_value, dtype, chunk_size, kwargs)
    dask_graph = dict()
    dependencies = []

    fill_value = _make_fill_value(fill_value, dtype)

    dst_chunks, output_shape = _normalize_chunks_for_area(dst_area, chunk_size, dtype)

    for dst_block_info, dst_area_chunk in _enumerate_dst_area_chunks(dst_area, dst_chunks):
        position = dst_block_info["chunk-location"]
        dst_block_info["shape"] = output_shape
        try:
            cropped_src_arrays, cropped_src_area, src_block_info = crop_data_around_area(src_area, src_arrays,
                                                                                         dst_area_chunk)
            _check_resolution_mismatch(cropped_src_area, dtype)
        except IncompatibleAreas:  # no relevant data matching
            task = (np.full, dst_block_info["chunk-shape"], fill_value)
            src_dependencies = []
        else:
            task, src_dependencies = _create_task(func,
                                                  cropped_src_arrays, src_block_info,
                                                  dst_arrays, dst_block_info,
                                                  position,
                                                  fill_value, kwargs)
        dask_graph[(name, *position)] = task
        dependencies.extend(src_dependencies)

    dependencies.extend(dst_arrays)

    dask_graph = HighLevelGraph.from_collections(name, dask_graph, dependencies=dependencies)
    return da.Array(dask_graph, name, chunks=dst_chunks, dtype=dtype, shape=output_shape)


def _create_dask_name(name, func, src_area, src_arrays, dst_area, dst_arrays, fill_value, dtype, chunks, kwargs):
    if name is not None:
        name = f"{name}"
    else:
        from dask.base import tokenize
        from dask.utils import funcname
        token = tokenize(func, hash(src_area), *src_arrays, hash(dst_area), *dst_arrays,
                         fill_value, dtype, chunks, **kwargs)
        name = f"{funcname(func)}-{token}"
    return name


def _make_fill_value(fill_value, dtype):
    if fill_value is None:
        if np.issubdtype(dtype, np.integer):
            fill_value = np.iinfo(dtype).min
        else:
            fill_value = np.nan
    return fill_value


def _check_resolution_mismatch(src_area_crop, dtype):
    res_chunks, _ = _normalize_chunks_for_area(src_area_crop, dask.config.get('array.chunk-size', '128MiB'),
                                               dtype)
    if len(res_chunks[0]) * len(res_chunks[1]) >= 4:
        logger.warning("The input area chunks are large. "
                       "This usually means that the input area is of much higher resolution than the output "
                       "area. You can reduce the chunks passed, and ponder whether you are using the right "
                       "resampler for the job.")


def _create_task(func, smaller_src_arrays, src_block_info, dst_arrays, dst_block_info, position, fill_value,
                 kwargs):
    """Create a task for resample_blocks."""
    from dask.utils import has_keyword
    dependencies = []
    args = []
    for smaller_data in smaller_src_arrays:
        args.append((smaller_data.name, *([0] * smaller_data.ndim)))
        dependencies.append(smaller_data)
    for dst_array in dst_arrays:
        dst_position = [0] * (dst_array.ndim - 2) + list(position[-2:])
        args.append((dst_array.name, *dst_position))
    func_kwargs = kwargs.copy()
    func_kwargs['fill_value'] = fill_value
    if has_keyword(func, "block_info"):
        func_kwargs["block_info"] = {0: src_block_info,
                                     None: dst_block_info}
    pfunc = partial(func, **func_kwargs)
    task = (pfunc, *args)
    return task, dependencies


def crop_data_around_area(source_geo_def, src_arrays, target_geo_def):
    """Crop the data around the provided area."""
    small_source_geo_def, x_slice, y_slice = crop_source_area(source_geo_def, target_geo_def)
    smaller_src_arrays = []
    for data in src_arrays:
        smaller_src_arrays.append(data[..., y_slice, x_slice].rechunk([-1] * data.ndim))

    block_info = {"shape": source_geo_def.shape,
                  "array-location": (y_slice, x_slice),
                  "area": small_source_geo_def}
    return smaller_src_arrays, small_source_geo_def, block_info


@lru_cache
def crop_source_area(source_geo_def, target_geo_def):
    """Crop a source area around the provided target area."""
    slicer = create_slicer(source_geo_def, target_geo_def)
    x_slice, y_slice = slicer.get_slices()
    small_source_geo_def = source_geo_def[y_slice, x_slice]
    if isinstance(small_source_geo_def, SwathDefinition):
        small_source_geo_def.lons.data = small_source_geo_def.lons.data.rechunk((-1, -1))
        small_source_geo_def.lats.data = small_source_geo_def.lats.data.rechunk((-1, -1))
    return small_source_geo_def, x_slice, y_slice


def _enumerate_dst_area_chunks(dst_area, dst_chunks):
    """Enumerate the chunks in function of the dst_area."""
    for position, slices in _enumerate_chunk_slices(dst_chunks):
        chunk_shape = tuple(chunk[pos] for pos, chunk in zip(position, dst_chunks))
        target_geo_def = dst_area[slices[-2:]]
        block_info = {"num-chunks": [len(chunk) for chunk in dst_chunks],
                      "chunk-location": position,
                      "array-location": slices,
                      "chunk-shape": chunk_shape,
                      "area": target_geo_def,
                      }
        yield block_info, target_geo_def


def _normalize_chunks_for_area(area, chunk_size, dtype):
    rest_shape = []
    if not isinstance(chunk_size, (Number, str)) and len(chunk_size) > len(area.shape):
        rest_chunks = chunk_size[:-len(area.shape)]
        for elt in rest_chunks:
            try:
                rest_shape.append(sum(elt))
            except TypeError:
                rest_shape.append(elt)
    output_shape = tuple(rest_shape) + area.shape

    dst_chunks = da.core.normalize_chunks(chunk_size, output_shape, dtype=dtype)
    return dst_chunks, output_shape
